// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>,
                     #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>,
                     #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>,
                     #llvm.mlir.module_flag<max, "uwtable", 2 : i32>,
                     #llvm.mlir.module_flag<max, "frame-pointer", 1 : i32>,
                     #llvm.mlir.module_flag<override, "probe-stack", "inline-asm">,
                     #llvm.mlir.module_flag<append, "CG Profile", [
                       #llvm.cgprofile_entry<from = @from, to = @to, count = 222>,
                       #llvm.cgprofile_entry<from = @from, count = 222>,
                       #llvm.cgprofile_entry<from = @to, to = @from, count = 222>
                    ]>]
}

// CHECK: llvm.module_flags [
// CHECK-SAME: #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>,
// CHECK-SAME: #llvm.mlir.module_flag<min, "PIC Level", 2 : i32>,
// CHECK-SAME: #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>,
// CHECK-SAME: #llvm.mlir.module_flag<max, "uwtable", 2 : i32>,
// CHECK-SAME: #llvm.mlir.module_flag<max, "frame-pointer", 1 : i32>,
// CHECK-SAME: #llvm.mlir.module_flag<override, "probe-stack", "inline-asm">,
// CHECK-SAME: #llvm.mlir.module_flag<append, "CG Profile", [
// CHECK-SAME: #llvm.cgprofile_entry<from = @from, to = @to, count = 222>,
// CHECK-SAME: #llvm.cgprofile_entry<from = @from, count = 222>,
// CHECK-SAME: #llvm.cgprofile_entry<from = @to, to = @from, count = 222>
// CHECK-SAME: ]>]
