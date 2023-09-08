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
    omp.target map((to -> %3 : !llvm.ptr<i32>), (to -> %5 : !llvm.ptr<i32>), (from -> %7 : !llvm.ptr<i32>)) {
      omp.parallel {
        %8 = llvm.load %3 : !llvm.ptr<i32>
        %9 = llvm.load %5 : !llvm.ptr<i32>
        %10 = llvm.add %8, %9  : i32
        llvm.store %10, %7 : !llvm.ptr<i32>
        omp.terminator
        }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: call void @__omp_offloading_[[DEV:.*]]_[[FIL:.*]]_omp_target_region__l[[LINE:.*]](ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
// CHECK: define internal void @__omp_offloading_[[DEV]]_[[FIL]]_omp_target_region__l[[LINE]](ptr %[[ADDR_A:.*]], ptr %[[ADDR_B:.*]], ptr %[[ADDR_C:.*]])
// CHECK:  %[[STRUCTARG:.*]] = alloca { ptr, ptr, ptr }, align 8
// CHECK: %[[GEP1:.*]] = getelementptr { ptr, ptr, ptr }, ptr %[[STRUCTARG]], i32 0, i32 0
// CHECK: store ptr %[[ADDR_A]], ptr %[[GEP1]], align 8
// CHECK: %[[GEP2:.*]] = getelementptr { ptr, ptr, ptr }, ptr %[[STRUCTARG]], i32 0, i32 1
// CHECK: store ptr %[[ADDR_B]], ptr %[[GEP2]], align 8
// CHECK: %[[GEP3:.*]] = getelementptr { ptr, ptr, ptr }, ptr %[[STRUCTARG]], i32 0, i32 2
// CHECK: store ptr %[[ADDR_C]], ptr %[[GEP3]], align 8
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @__omp_offloading_[[DEV]]_[[FIL]]_omp_target_region__l[[LINE]]..omp_par, ptr %[[STRUCTARG]])


// CHECK: define internal void @__omp_offloading_[[DEV]]_[[FIL]]_omp_target_region__l[[LINE]]..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %[[STRUCTARG2:.*]]) #0 {
// CHECK: %[[GEP4:.*]] = getelementptr { ptr, ptr, ptr }, ptr %[[STRUCTARG2]], i32 0, i32 0
// CHECK: %[[LOADGEP1:.*]] = load ptr, ptr %[[GEP4]], align 8
// CHECK: %[[GEP5:.*]] = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 1
// CHECK: %[[LOADGEP2:.*]] = load ptr, ptr %[[GEP5]], align 8
// CHECK: %[[GEP6:.*]] = getelementptr { ptr, ptr, ptr }, ptr %0, i32 0, i32 2
// CHECK: %[[LOADGEP3:.*]] = load ptr, ptr %[[GEP6]], align 8

// CHECK: %[[VAL_A:.*]] = load i32, ptr %[[LOADGEP1]], align 4
// CHECK: %[[VAL_B:.*]] = load i32, ptr %[[LOADGEP2]], align 4
// CHECK: %[[SUM:.*]] = add i32 %[[VAL_A]], %[[VAL_B]]
// CHECK: store i32 %[[SUM]], ptr %[[LOADGEP3]], align 4
