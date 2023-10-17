# Minimal MLIR binaries

This folder contains example of minimal MLIR setups that can showcase the
intended binary footprint of the framework.

- mlir-cat: This includes the Core IR, the builtin dialect, the textual
  parser/printer, the support for bytecode serialization.
- mlir-minimal-opt: This adds all the tooling for an mlir-opt tool: the pass
  infrastructure and all the instrumentation associated with it.
- mlir-miminal-opt-canonicalize: This add the canonicalizer pass, which pulls in
  all the pattern/rewrite machinery, including the PDL compiler and intepreter.

Below are some example measurements taken at the time of the LLVM 17 release,
using clang-14 on a X86 Ubuntu and [bloaty](https://github.com/google/bloaty).

|                                  | Base   | Os     | Oz     | Os LTO | Oz LTO |
| :------------------------------: | ------ | ------ | ------ | ------ | ------ |
| `mlir-cat`                       | 1018kB | 836KB  | 879KB  | 697KB  | 649KB  |
| `mlir-minimal-opt`               | 1.54MB | 1.25MB | 1.29MB | 1.10MB | 1.00MB |
| `mlir-minimal-opt-canonicalizer` | 2.24MB | 1.81MB | 1.86MB | 1.62MB | 1.48MB |

Base configuration:

```
cmake ../llvm/ -G Ninja \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_CCACHE_BUILD=ON \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_ENABLE_LLD=ON \
   -DLLVM_ENABLE_BACKTRACES=OFF \
   -DCMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO=-Wl,-icf=all
```

Note: to measure the on-disk size, you need to run `strip bin/mlir-cat` first to
remove all the debug info (which are useful for `bloaty` though).

The optimization level can be tuned with `-Os` or `-Oz`:

- `-DCMAKE_C_FLAGS_RELWITHDEBINFO="-Os -g -DNDEBUG" -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-Os -g -DNDEBUG"`
- `-DCMAKE_C_FLAGS_RELWITHDEBINFO="-Oz -g -DNDEBUG" -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-Oz -g -DNDEBUG"`

Finally LTO can also be enabled with `-DLLVM_ENABLE_LTO=FULL`.

Bloaty can provide measurements using:
`bloaty bin/mlir-cat -d compileunits --domain=vm` or
`bloaty bin/mlir-cat -d symbols --demangle=full --domain=vm`
