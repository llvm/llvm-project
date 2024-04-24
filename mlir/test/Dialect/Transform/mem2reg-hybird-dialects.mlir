// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(mem2reg{region-simplify=false}))" --split-input-file | FileCheck %s

// -----
// CHECK-LABEL: @single_define_multiple_regions_with_for
// CHECK-NOT: llvm.alloca
func.func @single_define_multiple_regions_with_for(%arg0 : index, %arg1 : index, %arg2 : index) {
  %0 = llvm.mlir.constant(4 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    scf.for %i1 = %arg0 to %arg1 step %arg2 {
      llvm.store %0, %1 {alignment = 8 : i64} : i32, !llvm.ptr
      %min_cmp = arith.cmpi slt, %i0, %i1 : index
      %min = arith.select %min_cmp, %i0, %i1 : index
      %max_cmp = arith.cmpi sge, %i0, %i1 : index
      %max = arith.select %max_cmp, %i0, %i1 : index
      %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
      %3 = arith.index_cast %2 : i32 to index
      scf.for %i2 = %min to %max step %i1 {
        %val = arith.addi %3, %3 : index
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: @single_define_multiple_regions_with_if
// CHECK-NOT: llvm.alloca
func.func @single_define_multiple_regions_with_if(%arg0 : i1, %arg1 : i32) {
  %0 = llvm.mlir.constant(4 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %0, %1 {alignment = 8 : i64} : i32, !llvm.ptr
  scf.if %arg0 {
    %3 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %4 = arith.addi %arg1, %3 : i32
  } else {
    %5 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %6 = arith.subi %arg1, %5 : i32
  }

  return
}

// -----
// The definition doesn't dominate all uses, mem2reg fails.
// CHECK-LABEL: @single_define_multiple_regions_with_if
// CHECK: llvm.alloca
func.func @single_define_multiple_regions_with_if_fail(%arg0 : i1, %arg1 : i32) {
  %0 = llvm.mlir.constant(4 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  scf.if %arg0 {
    llvm.store %0, %1 {alignment = 8 : i64} : i32, !llvm.ptr
    %3 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %4 = arith.addi %arg1, %3 : i32
  } else {
    %5 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %6 = arith.subi %arg1, %5 : i32
  }

  return
}

// -----

// CHECK-LABEL: @single_define_multiple_regions_with_while
// CHECK-NOT: llvm.alloca
func.func @single_define_multiple_regions_with_while(%arg0 : i32) {
  %0 = llvm.mlir.constant(4 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %0, %1 {alignment = 8 : i64} : i32, !llvm.ptr
  scf.while : () -> () {
    %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %3 = arith.cmpi sge, %2, %arg0 : i32
    scf.condition(%3)
  } do {
    scf.yield
  }
  return
}
