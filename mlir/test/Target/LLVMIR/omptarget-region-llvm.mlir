// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false} {
  llvm.func @omp_target_region_() {
    %0 = llvm.mlir.constant(20 : i32) : i32
    %1 = llvm.mlir.constant(10 : i32) : i32
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {bindc_name = "a", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_regionEa"} : (i64) -> !llvm.ptr<i32>
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x i32 {bindc_name = "b", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_regionEb"} : (i64) -> !llvm.ptr<i32>
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x i32 {bindc_name = "c", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_regionEc"} : (i64) -> !llvm.ptr<i32>
    llvm.store %1, %3 : !llvm.ptr<i32>
    llvm.store %0, %5 : !llvm.ptr<i32>
    omp.target   {
      %8 = llvm.load %3 : !llvm.ptr<i32>
      %9 = llvm.load %5 : !llvm.ptr<i32>
      %10 = llvm.add %8, %9  : i32
      llvm.store %10, %7 : !llvm.ptr<i32>
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: call void @__omp_offloading_[[DEV:.*]]_[[FIL:.*]]_omp_target_region__l[[LINE:.*]](ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
// CHECK: define internal void @__omp_offloading_[[DEV]]_[[FIL]]_omp_target_region__l[[LINE]](ptr %[[ADDR_A:.*]], ptr %[[ADDR_B:.*]], ptr %[[ADDR_C:.*]])
// CHECK: %[[VAL_A:.*]] = load i32, ptr %[[ADDR_A]], align 4
// CHECK: %[[VAL_B:.*]] = load i32, ptr %[[ADDR_B]], align 4
// CHECK: %[[SUM:.*]] = add i32 %[[VAL_A]], %[[VAL_B]]
// CHECK: store i32 %[[SUM]], ptr %[[ADDR_C]], align 4

