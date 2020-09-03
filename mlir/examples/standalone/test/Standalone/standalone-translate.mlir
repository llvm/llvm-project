// RUN: standalone-translate --help | FileCheck %s
// CHECK: --avx512-mlir-to-llvmir
// CHECK: --deserialize-spirv
// CHECK: --import-llvm
// CHECK: --mlir-to-llvmir
// CHECK: --mlir-to-nvvmir
// CHECK: --mlir-to-rocdlir
// CHECK: --serialize-spirv
