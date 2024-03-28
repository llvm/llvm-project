// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The aim of the test is to check the GPU LLVM IR codegen
// for nested omp do loop inside omp target region

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true } {
  llvm.func @target_parallel_wsloop(%arg0: !llvm.ptr) attributes {
    target_cpu = "gfx90a",
    target_features = #llvm.target_features<["+gfx9-insts", "+wavefrontsize64"]>
  } {
    omp.parallel {
      %loop_ub = llvm.mlir.constant(9 : i32) : i32
      %loop_lb = llvm.mlir.constant(0 : i32) : i32
      %loop_step = llvm.mlir.constant(1 : i32) : i32
      omp.wsloop for  (%loop_cnt) : i32 = (%loop_lb) to (%loop_ub) inclusive step (%loop_step) {
        %gep = llvm.getelementptr %arg0[0, %loop_cnt] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<10 x i32>
        ptr.store %loop_cnt, %gep : i32, !llvm.ptr
        omp.yield
      }
     omp.terminator
    }

    llvm.return
  }

}
// CHECK:      call void @__kmpc_parallel_51(ptr addrspacecast
// CHECK-SAME:  (ptr addrspace(1) @[[GLOB:[0-9]+]] to ptr),
// CHECK-SAME:  i32 %[[THREAD_NUM:.*]], i32 1, i32 -1, i32 -1,
// CHECK-SAME:  ptr @[[PARALLEL_FUNC:.*]], ptr null, ptr %[[PARALLEL_ARGS:.*]], i64 1)

// CHECK:      define internal void @[[PARALLEL_FUNC]]
// CHECK-SAME:  (ptr noalias noundef %[[TID_ADDR:.*]], ptr noalias noundef %[[ZERO_ADDR:.*]],
// CHECK-SAME:  ptr %[[ARG_PTR:.*]])
// CHECK-SAME:  #[[ATTRS1:[0-9]+]]
// CHECK: call void @__kmpc_for_static_loop_4u(ptr addrspacecast (ptr addrspace(1) @[[GLOB]] to ptr),
// CHECK-SAME:   ptr @[[LOOP_BODY_FUNC:.*]], ptr %[[LOO_BODY_FUNC_ARG:.*]], i32 10,
// CHECK-SAME:   i32 %[[THREAD_NUM:.*]], i32 0)

// CHECK:      define internal void @[[LOOP_BODY_FUNC]](i32 %[[CNT:.*]], ptr %[[LOOP_BODY_ARG_PTR:.*]]) #[[ATTRS2:[0-9]+]] {

// CHECK:      attributes #[[ATTRS2]] = {
// CHECK-SAME:  "target-cpu"="gfx90a"
// CHECK-SAME:  "target-features"="+gfx9-insts,+wavefrontsize64"
// CHECK:      attributes #[[ATTRS1]] = {
// CHECK-SAME:  "target-cpu"="gfx90a"
// CHECK-SAME:  "target-features"="+gfx9-insts,+wavefrontsize64"
