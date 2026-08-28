#!/usr/bin/env python3
# ===-- repack-zstd.py ------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===#

import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ZSTD_COMPRESSION_LEVEL = 21
ZSTD_WINDOW_LOG = 30
ZSTD_JOB_SIZE = "1024M"


def normalize_metadata(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    return info


def repack(input_path: Path, output_path: Path, work_dir: Path, zstd: str) -> None:
    if not input_path.name.endswith(".tar.xz"):
        raise ValueError(f"input filename does not end in .tar.xz: {input_path}")

    package_name = input_path.name.removesuffix(".tar.xz")
    # Six workers kept the ARM64 release benchmark under 20 minutes. macOS
    # uses fewer workers because its release runners have less memory.
    thread_count = 4 if sys.platform == "darwin" else 6

    with tempfile.TemporaryDirectory(prefix="zstd-repack-", dir=work_dir) as temp:
        temp_dir = Path(temp)
        extract_dir = temp_dir / "contents"
        extract_dir.mkdir()

        with tarfile.open(input_path, "r:xz") as archive:
            archive.extractall(extract_dir, filter="data")

        package_dir = extract_dir / package_name
        if not package_dir.is_dir():
            raise ValueError(f"archive does not contain {package_name}/")

        uncompressed_tar = temp_dir / f"{package_name}.tar"
        with tarfile.open(
            uncompressed_tar, "w", format=tarfile.GNU_FORMAT, dereference=False
        ) as archive:
            # TarFile.add() visits directory contents in sorted order.
            archive.add(
                package_dir,
                arcname=package_name,
                filter=normalize_metadata,
            )

        subprocess.run(
            [
                zstd,
                "--ultra",
                f"-{ZSTD_COMPRESSION_LEVEL}",
                # A 1 GiB window captures duplicate code across the large,
                # statically linked tools. Match the job size to the window.
                f"--long={ZSTD_WINDOW_LOG}",
                f"-T{thread_count}",
                f"-B{ZSTD_JOB_SIZE}",
                str(uncompressed_tar),
                "-o",
                str(output_path),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repack a release tar.xz as a name-sorted tar.zst"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--zstd", default="zstd")
    args = parser.parse_args()

    repack(
        args.input.resolve(),
        args.output.resolve(),
        args.work_dir.resolve(),
        args.zstd,
    )


if __name__ == "__main__":
    main()
