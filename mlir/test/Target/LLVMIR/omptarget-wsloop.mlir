// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The aim of the test is to check the GPU LLVM IR codegen
// for nested omp do loop inside omp target region

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true, omp.target = #omp.target<target_cpu = "gfx90a", target_features = "">} {
  llvm.func @writeindex_(%arg0: !llvm.ptr {fir.bindc_name = "int_array"}) attributes {fir.internal_name = "_QPwriteindex", omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>} {
    %0 = llvm.mlir.constant(9 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(10 : index) : i64
    %6 = omp.bounds lower_bound(%1 : i64) upper_bound(%0 : i64) extent(%3 : i64) stride(%2 : i64) start_idx(%2 : i64)
    %7 = omp.map_info var_ptr(%arg0 : !llvm.ptr, !llvm.array<10 x i32>) map_clauses(from) capture(ByRef) bounds(%6) -> !llvm.ptr {name = "int_array"}
    omp.target map_entries(%7 -> %arg1 : !llvm.ptr) {
    ^bb0(%arg1: !llvm.ptr):
      %9 = llvm.mlir.constant(1 : i64) : i64
      %10 = llvm.mlir.constant(10 : i32) : i32
      %11 = llvm.mlir.constant(1 : i32) : i32
      %12 = llvm.mlir.constant(1 : i64) : i64
      %13 = llvm.alloca %12 x i32 {pinned} : (i64) -> !llvm.ptr
      omp.wsloop for  (%arg4) : i32 = (%11) to (%10) inclusive step (%11) {
        llvm.store %arg4, %13 : i32, !llvm.ptr
        %14 = llvm.load %13 : !llvm.ptr -> i32
        %15 = llvm.sext %14 : i32 to i64
        %16 = llvm.sub %15, %9  : i64
        %17 = llvm.getelementptr %arg1[0, %16] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<10 x i32>
        llvm.store %14, %17 : i32, !llvm.ptr
        omp.yield
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define weak_odr protected amdgpu_kernel void @[[FUNC0:.*]](
// CHECK-SAME: ptr %[[ARG0:.*]], ptr %[[ARG1:.*]])
// CHECK:   %[[STRUCTARG:.*]] = alloca { ptr, ptr }, align 8, addrspace(5)
// CHECK:   %[[STRUCTARG_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[STRUCTARG]] to ptr
// CHECK:   store ptr %[[ARG1]], ptr %[[TMP0:.*]], align 8
// CHECK:   %[[TMP1:.*]] = load ptr, ptr %[[TMP0]], align 8
// CHECK:   %[[GEP:.*]] = getelementptr { ptr, ptr }, ptr addrspace(5) %[[STRUCTARG]], i32 0, i32 1
// CHECK:   store ptr %[[TMP1]], ptr addrspace(5) %[[GEP]], align 8
// CHECK:   %[[NUM_THREADS:.*]] = call i32 @omp_get_num_threads()
// CHECK:   call void @__kmpc_for_static_loop_4u(ptr addrspacecast (ptr addrspace(1) @[[GLOB1:[0-9]+]] to ptr), ptr @[[LOOP_BODY_FN:.*]], ptr %[[STRUCTARG_ASCAST]], i32 10, i32 %[[NUM_THREADS]], i32 0)

// CHECK: define internal void @[[LOOP_BODY_FN]](i32 %[[LOOP_CNT:.*]], ptr %[[LOOP_BODY_ARG:.*]])
// CHECK:   %[[GEP2:.*]] = getelementptr { ptr, ptr }, ptr %[[LOOP_BODY_ARG]], i32 0, i32 1
// CHECK:   %[[LOADGEP:.*]] = load ptr, ptr %[[GEP2]], align 8
// CHECK:   %[[GEP3:.*]] = getelementptr [10 x i32], ptr %[[LOADGEP]], i32 0, i64 %[[TMP2:.*]]
// CHECK:   store i32 %[[VAL0:.*]], ptr %[[GEP3]], align 4

