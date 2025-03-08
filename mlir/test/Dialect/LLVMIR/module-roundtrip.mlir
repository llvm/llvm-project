// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
  llvm.module_flags [#llvm.mlir.module_flag<1 : i64, "wchar_size", 4 : i32>,
                     #llvm.mlir.module_flag<8 : i64, "PIC Level", 2 : i32>,
                     #llvm.mlir.module_flag<7 : i64, "PIE Level", 2 : i32>,
                     #llvm.mlir.module_flag<7 : i64, "uwtable", 2 : i32>,
                     #llvm.mlir.module_flag<7 : i64, "frame-pointer", 1 : i32>]
}

// CHECK: llvm.module_flags [
// CHECK-SAME: #llvm.mlir.module_flag<1 : i64, "wchar_size", 4 : i32>,
// CHECK-SAME: #llvm.mlir.module_flag<8 : i64, "PIC Level", 2 : i32>,
// CHECK-SAME: #llvm.mlir.module_flag<7 : i64, "PIE Level", 2 : i32>,
// CHECK-SAME: #llvm.mlir.module_flag<7 : i64, "uwtable", 2 : i32>,
// CHECK-SAME: #llvm.mlir.module_flag<7 : i64, "frame-pointer", 1 : i32>]
