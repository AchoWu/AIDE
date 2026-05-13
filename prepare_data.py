"""
将 GenImage 原始数据（ai/nature 目录结构）转换为 AIDE 训练格式（0_real/1_fake）

用法:
    python prepare_data.py \
        --src /group/40092/howu/ai_or_real_img/dataset/genimage_data/Midjourney/imagenet_midjourney/train \
        --dst /group/40143/howu/AIDE/dataset/GenImage/train/Midjourney \
        --mode move

目录映射:
    src/nature  →  dst/0_real
    src/ai      →  dst/1_fake

支持多个数据源（如多个生成器）:
    python prepare_data.py \
        --src /path/to/Midjourney/train /path/to/StableDiffusion/train \
        --dst /path/to/GenImage/train/Midjourney /path/to/GenImage/train/StableDiffusion \
        --mode move
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def prepare_dataset(src_dir, dst_dir, mode="move"):
    """
    将 src_dir 下的 ai/nature 文件夹转换为 AIDE 训练格式

    Args:
        src_dir: 源目录，包含 ai/ 和 nature/ 子目录
        dst_dir: 目标目录，将生成 0_real/ 和 1_fake/ 子目录
        mode: "move" 移动, "symlink" 软链接, "copy" 复制
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    ai_dir = src_dir / "ai"
    nature_dir = src_dir / "nature"

    # 检查源目录
    if not ai_dir.exists():
        print(f"[ERROR] AI 目录不存在: {ai_dir}")
        return False
    if not nature_dir.exists():
        print(f"[ERROR] Nature 目录不存在: {nature_dir}")
        return False

    # 创建目标目录
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_real = dst_dir / "0_real"
    dst_fake = dst_dir / "1_fake"

    # 检查目标是否已存在
    if dst_real.exists() or dst_fake.exists():
        print(f"[WARN] 目标目录已存在:")
        if dst_real.exists():
            print(f"       {dst_real}")
        if dst_fake.exists():
            print(f"       {dst_fake}")
        response = input("是否继续？(y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return False

    print(f"[INFO] 源目录: {src_dir}")
    print(f"[INFO] 目标目录: {dst_dir}")
    print(f"[INFO] 模式: {mode}")
    print(f"[INFO] nature -> 0_real (真实图片)")
    print(f"[INFO] ai     -> 1_fake (AI生成图片)")
    print()

    if mode == "move":
        # 移动目录（同一文件系统下等同于重命名，速度极快）
        print(f"[MOVE] {nature_dir} -> {dst_real}")
        shutil.move(str(nature_dir), str(dst_real))
        print(f"[MOVE] {ai_dir} -> {dst_fake}")
        shutil.move(str(ai_dir), str(dst_fake))

    elif mode == "symlink":
        # 创建软链接（不占用额外空间，速度最快）
        print(f"[SYMLINK] {dst_real} -> {nature_dir}")
        os.symlink(str(nature_dir), str(dst_real))
        print(f"[SYMLINK] {dst_fake} -> {ai_dir}")
        os.symlink(str(ai_dir), str(dst_fake))

    elif mode == "copy":
        # 复制（耗时最长，占用双倍空间）
        print(f"[COPY] {nature_dir} -> {dst_real} (可能需要较长时间...)")
        shutil.copytree(str(nature_dir), str(dst_real))
        print(f"[COPY] {ai_dir} -> {dst_fake} (可能需要较长时间...)")
        shutil.copytree(str(ai_dir), str(dst_fake))

    else:
        print(f"[ERROR] 不支持的模式: {mode}")
        return False

    # 统计文件数量
    real_count = sum(1 for _ in dst_real.rglob("*") if _.is_file())
    fake_count = sum(1 for _ in dst_fake.rglob("*") if _.is_file())

    print()
    print(f"[DONE] 数据准备完成!")
    print(f"       0_real (真实图片): {real_count} 张")
    print(f"       1_fake (AI生成):   {fake_count} 张")
    print(f"       总计: {real_count + fake_count} 张")

    return True


if __name__ == "__main__":
    src = "/group/40092/howu/ai_or_real_img/dataset/genimage_data/Midjourney/imagenet_midjourney/train"
    dst = "/group/40143/howu/AIDE/dataset/GenImage/train/Midjourney"
    mode = "symlink"
    prepare_dataset(src, dst, mode)
