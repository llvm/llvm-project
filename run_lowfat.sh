#!/usr/bin/env bash

# Toggle configuration based on argument
if [[ "$1" == "custom" ]]; then
    echo "[+] Configuring custom LowFat sizes..."
    cmake -DLOWFAT_SIZES_CFG="$PWD/compiler-rt/lib/lowfat/tools/sizes.cfg" build
elif [[ "$1" == "pow2" ]]; then
    echo "[+] Configuring default pow2 LowFat sizes..."
    cmake -ULOWFAT_SIZES_CFG build
else
    echo "Usage: $0 [custom|pow2]"
    exit 1
fi

# Symlink compile_commands.json for LSP
echo "[+] Updating compile_commands.json symlink..."
ln -sf build/compile_commands.json .

# Build and run tests
echo "[+] Building..."
ninja -C build

echo "[+] Running LowFat tests..."
ninja -C build/runtimes/runtimes-bins check-lowfat
