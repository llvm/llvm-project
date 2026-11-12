### Apply clang-tidy fixes on the repo

This script runs clang-tidy on every C++ source file in MLIR and commit
the results of the checks individually. Be aware that it'll take over
10h to process the entire codebase.

The advised way to use this is to build clang-tidy (in release mode) and
have another build directory for MLIR. Here is a sample invocation from
the root of the repo:

```bash
{ time \
  CLANG_TIDY=build-clang/bin/clang-tidy \
  TIMING_TIDY=time \
  ./mlir/utils/clang-tidy/apply-clang-tidy.sh build mlir ~/clang-tidy-fails/
} 2>&1 | tee ~/clang-tidy.log
```

- `build-clang/` contains the result of a build of clang-tidy, configured
  and built somehow with:
```bash
$ cmake ../llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;clang-tools-extra" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=Native \
  -G Ninja
$ ninja clang-tidy
```
- `build/` must be a directory with MLIR onfigured. It is highly advised to
  use `ccache` as well, as this directory will be used to rerun
  `ninja check-mlir` after every single clang-tidy fix.
```bash
$ cmake ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DLLVM_CCACHE_BUILD=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -G Ninja
```
- `mlir/` is the directory where to find the files, it can be replaced by a
  subfolder or the path to a single file.
- `mkdir -p ~/clang-tidy-fails/` will be a directory containing the patches
  that clang-tidy produces but also fail the build.

