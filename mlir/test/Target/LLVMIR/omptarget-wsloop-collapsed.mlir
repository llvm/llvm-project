// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The aim of the test is to check the GPU LLVM IR codegen
// for nested omp do loop with collapse clause inside omp target region

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true } {
  llvm.func @target_collapsed_wsloop(%arg0: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
    %loop_ub = llvm.mlir.constant(99 : i32) : i32
    %loop_lb = llvm.mlir.constant(0 : i32) : i32
    %loop_step = llvm.mlir.constant(1 : index) : i32
    omp.wsloop for  (%arg1, %arg2) : i32 = (%loop_lb, %loop_lb) to (%loop_ub, %loop_ub) inclusive step (%loop_step, %loop_step) {
      %1 = llvm.add %arg1, %arg2  : i32
      %2 = llvm.mul %arg2, %loop_ub overflow<nsw>  : i32
      %3 = llvm.add %arg1, %2 :i32
      %4 = llvm.getelementptr %arg0[%3] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      llvm.store %1, %4 : i32, !llvm.ptr
      omp.yield
    }
    llvm.return
  }
}

// CHECK: define void @[[FUNC_COLLAPSED_WSLOOP:.*]](ptr %[[ARG0:.*]])
// CHECK:   call void @__kmpc_for_static_loop_4u(ptr addrspacecast (ptr addrspace(1) @[[GLOB2:[0-9]+]] to ptr),
// CHECK-SAME: ptr @[[COLLAPSED_WSLOOP_BODY_FN:.*]], ptr %[[STRUCT_ARG:.*]], i32 10000,
// CHECK-SAME: i32 %[[NUM_THREADS:.*]], i32 0)

// CHECK: define internal void @[[COLLAPSED_WSLOOP_BODY_FN]](i32 %[[LOOP_CNT:.*]], ptr %[[LOOP_BODY_ARG:.*]])
// CHECK:   %[[TMP0:.*]] = urem i32 %[[LOOP_CNT]], 100
// CHECK:   %[[TMP1:.*]] = udiv i32 %[[LOOP_CNT]], 100
// CHECK:   %[[TMP2:.*]] = mul i32 %[[TMP1]], 1
// CHECK:   %[[TMP3:.*]] = add i32 %[[TMP2]], 0
// CHECK:   %[[TMP4:.*]] = mul i32 %[[TMP0]], 1
// CHECK:   %[[TMP5:.*]] = add i32 %[[TMP4]], 0
// CHECK:   %[[TMP6:.*]] = add i32 %[[TMP3]], %[[TMP5]]
// CHECK:   %[[TMP7:.*]] = mul nsw i32 %[[TMP5]], 99
// CHECK:   %[[TMP8:.*]] = add i32 %[[TMP3]], %[[TMP7]]
// CHECK:   %[[TMP9:.*]] = getelementptr i32, ptr %[[ARRAY:.*]], i32 %[[TMP8]]
// CHECK:   store i32 %[[TMP6]], ptr %[[TMP9]], align 4
