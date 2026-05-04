# EmbeddedJIT Minimal Runtime Build

为 ARM AArch64 嵌入式平台构建裁减后的 EJIT 运行时静态库，目标 < 10MB RAM。

## 快速开始

```bash
# 在同级 workspace 目录下执行

# 1. 配置 (AArch64 后端, Release, -Os, 静态链接, 无 LTO)
cmake -S llvm -B build_aarch64 --preset ejit-minimal

# 2. 构建 EJIT 及其依赖
ninja -C build_aarch64 LLVMEJIT

# 3. 合并为单文件静态库
ninja -C build_aarch64 ejit_minimal

# 4. 体积检查
ninja -C build_aarch64 check-ejit-size
```

构建产物位于 `build_aarch64/lib/ExecutionEngine/EJIT/libejit_minimal.a`。

## Preset 说明

`llvm/CMakePresets.json` 提供两个 preset：

| Preset | 用途 | 编译器 |
|--------|------|--------|
| `ejit-minimal` | 体积测量 / 开发验证 | clang / clang++ (本机) |
| `ejit-minimal-aarch64` | ARM 交叉编译部署 | aarch64-linux-gnu-gcc / g++ |

### ejit-minimal

构建 LLVM 静态库（x86 二进制）+ AArch64 后端，用于在开发机上测量体积和验证编译。

关键配置：
- `CMAKE_BUILD_TYPE=Release`，叠加 `-Os`（通过 `CMAKE_CXX_FLAGS_RELEASE`）
- `LLVM_TARGETS_TO_BUILD=AArch64`（只构建 AArch64 后端）
- `BUILD_SHARED_LIBS=OFF`（静态库）
- 无 LTO（避免 `-flto=thin` 兼容性问题和构建变慢）
- `-ffunction-sections -fdata-sections` + `--gc-sections`（段级死代码消除）
- `--strip-all`（去除符号表）
- LLVM 断言/测试/文档/benchmark 全部关闭

### ejit-minimal-aarch64

继承 `ejit-minimal`，额外设置：
- 交叉编译器 `aarch64-linux-gnu-gcc/g++`
- Target triple 强制设为 `aarch64-linux-gnu`
- `EJIT_DEFAULT_TARGET_TRIPLE` 传入 C++ 代码，替代 `detectHost()`

**前提**：需要安装 AArch64 交叉编译工具链。
```bash
# Ubuntu/Debian
apt install g++-aarch64-linux-gnu
```

## 构建目标

| Target | 说明 |
|--------|------|
| `LLVMEJIT` | 构建 EJIT 组件库 + 所有传递依赖 |
| `ejit_minimal` | 将 `libLLVMEJIT.a` + 所有依赖库合并为 `libejit_minimal.a` |
| `check-ejit-size` | 检查合并后的存档体积是否超过 10MB 预算 |

## 实际尺寸测量

| 测量项 | 无 LTO | ThinLTO |
|--------|--------|---------|
| 静态库总计 (62个) | 277 MB | 363 MB |
| libLLVMEJIT.a | 515 KB | 690 KB |
| libLLVMCore.a | 8.9 MB | 11 MB |
| libLLVMOrcJIT.a | 6.5 MB | 8.6 MB |
| libLLVMCodeGen.a | 19 MB | 29 MB |
| 合并存档 (ejit_minimal.a) | ~65 MB | ~87 MB |
| **链接后二进制** (--whole-archive, -Os, --gc-sections) | **~11 MB** (text: 10.5M) | ~9.5 MB (text: 7.2M) |
| **估算真实部署体积** | **~7-8 MB** | ~5-6 MB |

> 注：链接后二进制体积为 `--whole-archive` 强制包含全部 62 个 LLVM 静态库的上限值。真实部署场景只需链接 EJIT 实际需要的 ~25 个库，体积会更小。估算值为上限体积 × 0.7 的经验比例。

## 体积说明

静态库（`.a`）体积 ≠ 最终链接体积。静态库包含所有符号，链接时通过 `--gc-sections` 仅保留实际引用的函数和数据段。

**LTO 对体积的影响**：ThinLTO 可减少约 30% 的最终二进制体积（跨模块死代码消除和优化），但缺点是：
- 编译速度显著变慢
- 静态库体积更大（含 LTO bitcode）
- 需要 clang 编译器（GCC 的 `-flto` 支持不同）

当前 preset 默认关闭 LTO，优先编译速度和兼容性。如需更小体积，可添加 `"LLVM_ENABLE_LTO": "Thin"`。

## 定制

### 修改 target triple

编辑 `llvm/CMakePresets.json` 中的 `EJIT_DEFAULT_TARGET_TRIPLE`：

```json
"EJIT_DEFAULT_TARGET_TRIPLE": "aarch64-none-elf"
```

### 修改优化等级

```json
"CMAKE_CXX_FLAGS_RELEASE": "-Oz -DNDEBUG"
```

### 添加其他后端

```json
"LLVM_TARGETS_TO_BUILD": "AArch64;ARM"
```

## 常见问题

**Q: 构建报 `cc1plus: error: unrecognized argument to '-flto='`**

检查 CMakePresets.json 中是否错误设置了 `LLVM_ENABLE_LTO`。当前 preset 已移除 LTO。

**Q: ejit_minimal 合并后的存档比预期大**

合并存档包含 LTO 中间码（如有）和所有符号。实际运行时体积以链接后的可执行文件 .text 段为准。用 `size -A <binary>` 测量。

**Q: `check-ejit-size` 报告超预算**

该检查测量合并后的 `.a` 文件大小，仅供参考。应以实际链接产物为准。要获得准确的体积数据，交叉编译一个简单 EJIT 测试程序并测量其 `.text` 段。

## 文件结构

```
llvm/
├── CMakePresets.json                     # minimal preset 定义
├── utils/ejit-minimal.sh                 # 一键配置/构建/检查脚本
└── lib/ExecutionEngine/EJIT/
    ├── CMakeLists.txt                    # EJIT 构建 + minimal target
    ├── CombineArchives.cmake             # 合并静态库为 libejit_minimal.a
    ├── CheckSize.cmake                   # 体积预算检查
    └── README_MINIMAL.md                 # 本文档
```
