# riscv64.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# 用 Clang 當前端，指定目標三元組
set(CMAKE_C_COMPILER   $ENV{HOME}/work/llvm-project/build/bin/clang)
set(CMAKE_CXX_COMPILER $ENV{HOME}/work/llvm-project/build/bin/clang++)

set(CMAKE_C_COMPILER_TARGET   riscv64-linux-gnu)
set(CMAKE_CXX_COMPILER_TARGET riscv64-linux-gnu)

# 指定 sysroot（安裝的 cross glibc）
set(CMAKE_SYSROOT /usr/riscv64-linux-gnu)

# 連結器與 toolchain（用 lld；讓 Clang 找到 /usr 的 cross 工具）
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld --gcc-toolchain=/usr")

# 避免 try_compile 去執行 target 程式
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
