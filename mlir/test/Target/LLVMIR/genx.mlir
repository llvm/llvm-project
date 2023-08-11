// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @genx_special_regs() -> i64 {
  // CHECK-LABEL: genx_special_regs
  // CHECK: call i64 @get_local_id(i32 0)
  %1 = genx.workitem.id.x : i64
  // CHECK: call i64 @get_local_id(i32 1)
  %2 = genx.workitem.id.y : i64
  // CHECK: call i64 @get_local_id(i32 2)
  %3 = genx.workitem.id.z : i64
  // CHECK: call i64 @get_group_id(i32 0)
  %4 = genx.workgroup.id.x : i64
  // CHECK: call i64 @get_group_id(i32 1)
  %5 = genx.workgroup.id.y : i64
  // CHECK: call i64 @get_group_id(i32 2)
  %6 = genx.workgroup.id.z : i64
  // CHECK: call i64 @get_local_size(i32 0)
  %7 = genx.workgroup.dim.x : i64
  // CHECK: call i64 @get_local_size(i32 1)
  %8 = genx.workgroup.dim.y : i64
  // CHECK: call i64 @get_local_size(i32 2)
  %9 = genx.workgroup.dim.z : i64
  // CHECK: call i64 @get_global_size(i32 0)
  %10 = genx.grid.dim.x : i64
  // CHECK: call i64 @get_global_size(i32 1)
  %11 = genx.grid.dim.y : i64
  // CHECK: call i64 @get_global_size(i32 2)
  %12 = genx.grid.dim.z : i64

  llvm.return %1 : i64
}
