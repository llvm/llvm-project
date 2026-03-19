// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// The aim of the test is to check the GPU LLVM IR codegen
// for nested omp do loop inside omp target region

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true } {
  llvm.func @target_wsloop(%arg0: !llvm.ptr ) attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
      %loop_ub = llvm.mlir.constant(9 : i32) : i32
      %loop_lb = llvm.mlir.constant(0 : i32) : i32
      %loop_step = llvm.mlir.constant(1 : i32) : i32
      omp.wsloop {
        omp.loop_nest (%loop_cnt) : i32 = (%loop_lb) to (%loop_ub) inclusive step (%loop_step) {
          %gep = llvm.getelementptr %arg0[0, %loop_cnt] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<10 x i32>
          llvm.store %loop_cnt, %gep : i32, !llvm.ptr
          omp.yield
        }
      }
    llvm.return
  }

  llvm.func @target_empty_wsloop() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
      %loop_ub = llvm.mlir.constant(9 : i32) : i32
      %loop_lb = llvm.mlir.constant(0 : i32) : i32
      %loop_step = llvm.mlir.constant(1 : i32) : i32
      omp.wsloop {
        omp.loop_nest (%loop_cnt) : i32 = (%loop_lb) to (%loop_ub) inclusive step (%loop_step) {
          omp.yield
        }
      }
    llvm.return
  }
}

// CHECK: define void @[[FUNC0:.*]](ptr %[[ARG0:.*]])
// CHECK:   %[[STRUCTARG:.*]] = alloca { ptr }, align 8, addrspace(5)
// CHECK:   %[[STRUCTARG_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[STRUCTARG]] to ptr
// CHECK:   %[[GEP:.*]] = getelementptr { ptr }, ptr addrspace(5) %[[STRUCTARG]], i32 0, i32 0
// CHECK:   store ptr %[[ARG0]], ptr addrspace(5) %[[GEP]], align 8
// CHECK:   %[[NUM_THREADS:.*]] = call i32 @omp_get_num_threads()
// CHECK:   call void @__kmpc_for_static_loop_4u(ptr addrspacecast (ptr addrspace(1) @[[GLOB1:[0-9]+]] to ptr), ptr @[[LOOP_BODY_FN:.*]], ptr %[[STRUCTARG_ASCAST]], i32 10, i32 %[[NUM_THREADS]], i32 0, i8 0)

// CHECK: define internal void @[[LOOP_BODY_FN]](i32 %[[LOOP_CNT:.*]], ptr %[[LOOP_BODY_ARG:.*]])
// CHECK:   %[[GEP2:.*]] = getelementptr { ptr }, ptr %[[LOOP_BODY_ARG]], i32 0, i32 0
// CHECK:   %[[LOADGEP:.*]] = load ptr, ptr %[[GEP2]], align 8
// CHECK:   %[[GEP3:.*]] = getelementptr [10 x i32], ptr %[[LOADGEP]], i32 0, i32 %[[TMP2:.*]]
// CHECK:   store i32 %[[VAL0:.*]], ptr %[[GEP3]], align 4

// CHECK: define void @[[FUNC_EMPTY_WSLOOP:.*]]()
// CHECK:   call void @__kmpc_for_static_loop_4u(ptr addrspacecast (ptr addrspace(1) @[[GLOB2:[0-9]+]] to ptr), ptr @[[LOOP_EMPTY_BODY_FN:.*]], ptr null, i32 10, i32 %[[NUM_THREADS:.*]], i32 0, i8 0)

// CHECK: define internal void @[[LOOP_EMPTY_BODY_FN]](i32 %[[LOOP_CNT:.*]])
