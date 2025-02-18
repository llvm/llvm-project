// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
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
}

// CHECK:      @[[SRC_LOC:.*]] = private unnamed_addr constant [23 x i8] c"{{[^"]*}}", align 1
// CHECK:      @[[IDENT:.*]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @[[SRC_LOC]] }, align 8
// CHECK:      @[[DYNA_ENV:.*]] = weak_odr protected global %struct.DynamicEnvironmentTy zeroinitializer
// CHECK:      @[[KERNEL_ENV:.*]] = weak_odr protected constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1, i32 1, i32 256, i32 -1, i32 -1, i32 0, i32 0 }, ptr @[[IDENT]], ptr @[[DYNA_ENV]] }
// CHECK:      define weak_odr protected amdgpu_kernel void @__omp_offloading_{{[^_]+}}_{{[^_]+}}_omp_target_region__l{{[0-9]+}}(ptr %[[DYN_PTR:.*]], ptr %[[ADDR_A:.*]], ptr %[[ADDR_B:.*]], ptr %[[ADDR_C:.*]])
// CHECK:        %[[TMP_A:.*]] = alloca ptr, align 8
// CHECK:        store ptr %[[ADDR_A]], ptr %[[TMP_A]], align 8
// CHECK:        %[[TMP_B:.*]] = alloca ptr, align 8
// CHECK:        store ptr %[[ADDR_B]], ptr %[[TMP_B]], align 8
// CHECK:        %[[TMP_C:.*]] = alloca ptr, align 8
// CHECK:        store ptr %[[ADDR_C]], ptr %[[TMP_C]], align 8
// CHECK:        %[[INIT:.*]] = call i32 @__kmpc_target_init(ptr @[[KERNEL_ENV]], ptr %[[DYN_PTR]])
// CHECK-NEXT:   %[[CMP:.*]] = icmp eq i32 %[[INIT]], -1
// CHECK-NEXT:   br i1 %[[CMP]], label %[[LABEL_ENTRY:.*]], label %[[LABEL_EXIT:.*]]
// CHECK:        [[LABEL_ENTRY]]:
// CHECK:        %[[PTR_A:.*]] = load ptr, ptr %[[TMP_A]], align 8
// CHECK:        %[[PTR_B:.*]] = load ptr, ptr %[[TMP_B]], align 8
// CHECK:        %[[PTR_C:.*]] = load ptr, ptr %[[TMP_C]], align 8
// CHECK-NEXT:   br label %[[LABEL_TARGET:.*]]
// CHECK:        [[LABEL_TARGET]]:
// CHECK:        %[[A:.*]] = load i32, ptr %[[PTR_A]], align 4
// CHECK:        %[[B:.*]] = load i32, ptr %[[PTR_B]], align 4
// CHECK:        %[[C:.*]] = add i32 %[[A]], %[[B]]
// CHECK:        store i32 %[[C]], ptr %[[PTR_C]], align 4
// CHECK:        br label %[[LABEL_DEINIT:.*]]
// CHECK:        [[LABEL_DEINIT]]:
// CHECK-NEXT:   call void @__kmpc_target_deinit()
// CHECK-NEXT:   ret void
// CHECK:        [[LABEL_EXIT]]:
// CHECK-NEXT:   ret void
