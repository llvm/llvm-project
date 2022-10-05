// RUN: not mlir-cpu-runner --no-implicit-module %s |& FileCheck %s

// CHECK: Error: top-level op must be a symbol table.
llvm.func @main()
