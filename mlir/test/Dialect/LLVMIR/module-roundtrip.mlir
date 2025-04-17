// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4>,
                     #llvm.mlir.module_flag<min, "PIC Level", 2>,
                     #llvm.mlir.module_flag<max, "PIE Level", 2>,
                     #llvm.mlir.module_flag<max, "uwtable", 2>,
                     #llvm.mlir.module_flag<max, "frame-pointer", 1>]
}

// CHECK: llvm.module_flags [
// CHECK-SAME: #llvm.mlir.module_flag<error, "wchar_size", 4>,
// CHECK-SAME: #llvm.mlir.module_flag<min, "PIC Level", 2>,
// CHECK-SAME: #llvm.mlir.module_flag<max, "PIE Level", 2>,
// CHECK-SAME: #llvm.mlir.module_flag<max, "uwtable", 2>,
// CHECK-SAME: #llvm.mlir.module_flag<max, "frame-pointer", 1>]
