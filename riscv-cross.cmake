set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# 明確設定 host 編譯器
set(CMAKE_HOST_SYSTEM_NAME Linux)
set(CMAKE_HOST_SYSTEM_PROCESSOR aarch64)

# Target 編譯器
set(CMAKE_C_COMPILER "/home/youzhewei.linux/work/llvm-project/build/bin/clang")
set(CMAKE_CXX_COMPILER "/home/youzhewei.linux/work/llvm-project/build/bin/clang++")
set(CMAKE_C_COMPILER_TARGET "riscv64-linux-gnu")
set(CMAKE_CXX_COMPILER_TARGET "riscv64-linux-gnu")

# Host 編譯器（用於構建工具）
set(CMAKE_C_COMPILER_FOR_BUILD "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER_FOR_BUILD "/usr/bin/g++")

set(CMAKE_SYSROOT "/usr/riscv64-linux-gnu")
set(CMAKE_C_FLAGS "-target riscv64-linux-gnu -march=rv64gc -mabi=lp64d")
set(CMAKE_CXX_FLAGS "-target riscv64-linux-gnu -march=rv64gc -mabi=lp64d -stdlib=libstdc++ --gcc-toolchain=/usr")
set(CMAKE_EXE_LINKER_FLAGS "--gcc-toolchain=/usr -fuse-ld=lld -Wl,--sysroot=/ -stdlib=libstdc++")

set(CMAKE_CROSSCOMPILING ON)
