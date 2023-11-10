// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The aim of the test is to check the LLVM IR codegen for the device
// for omp target parallel construct

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true, omp.target = #omp.target<target_cpu = "gfx90a", target_features = "">} {
  llvm.func @_QQmain_omp_outline_1(%arg0: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QQmain"} {
    %0 = omp.map_info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(from) capture(ByRef) -> !llvm.ptr {name = "d"}
    omp.target map_entries(%0 -> %arg2 : !llvm.ptr) {
    ^bb0(%arg2: !llvm.ptr):
      omp.parallel {
        %1 = llvm.mlir.constant(1 : i32) : i32
        llvm.store %1, %arg2 : i32, !llvm.ptr
        omp.terminator
      }
    omp.terminator
    }
  llvm.return
  }

  llvm.func @_test_num_threads(%arg0: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QQmain"} {
    %0 = omp.map_info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(from) capture(ByRef) -> !llvm.ptr {name = "d"}
    omp.target map_entries(%0 -> %arg2 : !llvm.ptr) {
    ^bb0(%arg2: !llvm.ptr):
      %1 = llvm.mlir.constant(156 : i32) : i32
      omp.parallel num_threads(%1 : i32) {
        %2 = llvm.mlir.constant(1 : i32) : i32
        llvm.store %2, %arg2 : i32, !llvm.ptr
        omp.terminator
      }
    omp.terminator
    }
  llvm.return
  }
}

// CHECK: define weak_odr protected amdgpu_kernel void [[FUNC0:@.*]](
// CHECK-SAME: ptr [[TMP:%.*]], ptr [[TMP0:.*]]) {
// CHECK:         [[TMP1:%.*]] = alloca [1 x ptr], align 8, addrspace(5)
// CHECK:         [[TMP2:%.*]] = addrspacecast ptr addrspace(5) [[TMP1]] to ptr
// CHECK:         [[STRUCTARG:%.*]] = alloca { ptr }, align 8, addrspace(5)
// CHECK:         [[STRUCTARG_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[STRUCTARG]] to ptr
// CHECK:         [[TMP3:%.*]] = alloca ptr, align 8, addrspace(5)
// CHECK:         [[TMP4:%.*]] = addrspacecast ptr addrspace(5) [[TMP3]] to ptr
// CHECK:         store ptr [[TMP0]], ptr [[TMP4]], align 8
// CHECK:         [[TMP5:%.*]] = call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) [[KERNEL_ENV:@.*]] to ptr), ptr [[TMP]])
// CHECK:         [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP5]], -1
// CHECK:         br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK:         [[TMP6:%.*]] = load ptr, ptr [[TMP4]], align 8
// CHECK:         [[OMP_GLOBAL_THREAD_NUM:%.*]] = call i32 @__kmpc_global_thread_num(ptr addrspacecast (ptr addrspace(1) @[[GLOB1:[0-9]+]] to ptr))
// CHECK:         [[GEP_:%.*]] = getelementptr { ptr }, ptr addrspace(5) [[STRUCTARG]], i32 0, i32 0
// CHECK:         store ptr [[TMP6]], ptr addrspace(5) [[GEP_]], align 8
// CHECK:         [[TMP7:%.*]] = getelementptr inbounds [1 x ptr], ptr [[TMP2]], i64 0, i64 0
// CHECK:         store ptr [[STRUCTARG_ASCAST]], ptr [[TMP7]], align 8
// CHECK:         call void @__kmpc_parallel_51(ptr addrspacecast (ptr addrspace(1) @[[GLOB1]] to ptr), i32 [[OMP_GLOBAL_THREAD_NUM]], i32 1, i32 -1, i32 -1, ptr [[FUNC1:@.*]], ptr null, ptr [[TMP2]], i64 1)
// CHECK:         call void @__kmpc_target_deinit()

// CHECK: define internal void [[FUNC1]](
// CHECK-SAME: ptr noalias noundef [[TID_ADDR_ASCAST:%.*]], ptr noalias noundef [[ZERO_ADDR_ASCAST:%.*]], ptr [[TMP0:%.*]]) #[[ATTR0:[0-9]+]] {

// CHECK: define weak_odr protected amdgpu_kernel void [[FUNC_NUM_THREADS0:@.*]](
// CHECK-NOT:     call void @__kmpc_push_num_threads(
// CHECK:         call void @__kmpc_parallel_51(ptr addrspacecast (
// CHECK-SAME:  ptr addrspace(1) @[[NUM_THREADS_GLOB:[0-9]+]] to ptr),
// CHECK-SAME:  i32 [[NUM_THREADS_TMP0:%.*]], i32 1, i32 156,
// CHECK-SAME:  i32 -1,  ptr [[FUNC_NUM_THREADS1:@.*]], ptr null, ptr [[NUM_THREADS_TMP1:%.*]], i64 1)
