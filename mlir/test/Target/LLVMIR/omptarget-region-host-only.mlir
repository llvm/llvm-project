// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false} {
  llvm.func @omp_target_region_() {
    %0 = llvm.mlir.constant(20 : i32) : i32
    %1 = llvm.mlir.constant(10 : i32) : i32
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {bindc_name = "a", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_regionEa"} : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x i32 {bindc_name = "b", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_regionEb"} : (i64) -> !llvm.ptr
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x i32 {bindc_name = "c", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_regionEc"} : (i64) -> !llvm.ptr
    llvm.store %1, %3 : i32, !llvm.ptr
    llvm.store %0, %5 : i32, !llvm.ptr
    %map1 = omp.map.info var_ptr(%3 : !llvm.ptr, i32)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %map2 = omp.map.info var_ptr(%5 : !llvm.ptr, i32)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    %map3 = omp.map.info var_ptr(%7 : !llvm.ptr, i32)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target map_entries(%map1 -> %arg0, %map2 -> %arg1, %map3 -> %arg2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      %8 = llvm.load %arg0 : !llvm.ptr -> i32
      %9 = llvm.load %arg1 : !llvm.ptr -> i32
      %10 = llvm.add %8, %9  : i32
      llvm.store %10, %arg2 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }

  llvm.func @omp_target_no_map() {
    omp.target {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define void @omp_target_region_()
// CHECK-NOT: call i32 @__tgt_target_kernel({{.*}})
// CHECK: call void @__omp_offloading_[[DEV:.*]]_[[FIL:.*]]_omp_target_region__l[[LINE1:.*]](ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
// CHECK-NEXT: ret void

// CHECK: define void @omp_target_no_map()
// CHECK-NOT: call i32 @__tgt_target_kernel({{.*}})
// CHECK: call void @__omp_offloading_[[DEV]]_[[FIL]]_omp_target_no_map_l[[LINE2:.*]]()
// CHECK-NEXT: ret void

// CHECK: define internal void @__omp_offloading_[[DEV]]_[[FIL]]_omp_target_region__l[[LINE1]](ptr %[[ADDR_A:.*]], ptr %[[ADDR_B:.*]], ptr %[[ADDR_C:.*]])
// CHECK: %[[VAL_A:.*]] = load i32, ptr %[[ADDR_A]], align 4
// CHECK: %[[VAL_B:.*]] = load i32, ptr %[[ADDR_B]], align 4
// CHECK: %[[SUM:.*]] = add i32 %[[VAL_A]], %[[VAL_B]]
// CHECK: store i32 %[[SUM]], ptr %[[ADDR_C]], align 4

// CHECK: define internal void @__omp_offloading_[[DEV]]_[[FIL]]_omp_target_no_map_l[[LINE2]]()
// CHECK: ret void
