// Tests single-team by-ref GPU reductions.

// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<"dlti.alloca_memory_space" = 5 : ui64, "dlti.global_memory_space" = 1 : ui64>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  omp.private {type = private} @_QFfooEi_private_i32 : i32
  omp.declare_reduction @add_reduction_byref_box_heap_f32 : !llvm.ptr attributes {byref_element_type = f32} alloc {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    omp.yield(%2 : !llvm.ptr)
  } init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg1 : !llvm.ptr)
  } combiner {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.alloca %3 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {alignment = 8 : i64} : (i32) -> !llvm.ptr<5>
    %5 = llvm.addrspacecast %4 : !llvm.ptr<5> to !llvm.ptr
    %6 = llvm.mlir.constant(24 : i32) : i32
    "llvm.intr.memcpy"(%5, %arg0, %6) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    %7 = llvm.mlir.constant(24 : i32) : i32
    "llvm.intr.memcpy"(%2, %arg1, %7) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    %8 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %9 = llvm.load %8 : !llvm.ptr -> !llvm.ptr
    %10 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %11 = llvm.load %10 : !llvm.ptr -> !llvm.ptr
    %12 = llvm.load %9 : !llvm.ptr -> f32
    %13 = llvm.load %11 : !llvm.ptr -> f32
    %14 = llvm.fadd %12, %13 {fastmathFlags = #llvm.fastmath<contract>} : f32
    llvm.store %14, %9 : f32, !llvm.ptr
    omp.yield(%arg0 : !llvm.ptr)
  } data_ptr_ptr {
  ^bb0(%arg0: !llvm.ptr):
    %0 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    omp.yield(%0 : !llvm.ptr)
  }

  llvm.func @foo_() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %0 x i1 : (i64) -> !llvm.ptr<5>
    %5 = llvm.addrspacecast %4 : !llvm.ptr<5> to !llvm.ptr
    %8 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>
    %9 = omp.map.info var_ptr(%5 : !llvm.ptr, f32) map_clauses(implicit, tofrom) capture(ByRef) var_ptr_ptr(%8 : !llvm.ptr) -> !llvm.ptr {name = ""}
    %10 = omp.map.info var_ptr(%5 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(always, implicit, descriptor, to) capture(ByRef) members(%9 : [0] : !llvm.ptr) -> !llvm.ptr {name = "scalar_alloc"}
    omp.target map_entries(%10 -> %arg0 : !llvm.ptr) {
      %13 = llvm.mlir.constant(1000 : i32) : i32
      %14 = llvm.mlir.constant(1 : i32) : i32
      omp.parallel {
        omp.wsloop reduction(byref @add_reduction_byref_box_heap_f32 %arg0 -> %arg4 : !llvm.ptr) {
          omp.loop_nest (%arg5) : i32 = (%14) to (%13) inclusive step (%14) {
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define {{.*}} @_omp_reduction_shuffle_and_reduce_func({{.*}}) {{.*}} {
// CHECK:   %[[REMOTE_RED_LIST:.omp.reduction.remote_reduce_list]] = alloca [1 x ptr], align 8, addrspace(5)
// CHECK:   %[[RED_ELEM:.omp.reduction.element]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, align 8, addrspace(5)
// CHECK:   %[[RED_ELEM_1:.*]] = addrspacecast ptr addrspace(5) %[[RED_ELEM]] to ptr

// CHECK:   %[[SHUFFLE_ELEM:.*]] = alloca float, align 4, addrspace(5)
// CHECK:   %[[REMOTE_RED_LIST_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[REMOTE_RED_LIST]] to ptr

// CHECK:   %[[REMOTE_RED_LIST_ELEM0:.*]] = getelementptr inbounds [1 x ptr], ptr %[[REMOTE_RED_LIST_ASCAST]], i64 0, i64 0

// CHECK:   %[[SHUFFLE_ELEM_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[SHUFFLE_ELEM]] to ptr
// CHECK:   %[[SHUFFLE_RES:.*]] = call i32 @__kmpc_shuffle_int32({{.*}})
// CHECK:   store i32 %[[SHUFFLE_RES]], ptr %[[SHUFFLE_ELEM_ASCAST]], align 4

// CHECK:   %[[RED_ELEM_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[RED_ELEM]] to ptr
// CHECK:   %[[RED_ALLOC_PTR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[RED_ELEM_ASCAST]], i32 0, i32 0
// CHECK:   %[[SHUFFLE_ELEM_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[SHUFFLE_ELEM]] to ptr
// CHECK:   store ptr %[[SHUFFLE_ELEM_ASCAST]], ptr %[[RED_ALLOC_PTR]], align 8
// CHECK:   store ptr %[[RED_ELEM_1]], ptr %[[REMOTE_RED_LIST_ELEM0]], align 8
// CHECK: }

// CHECK: define {{.*}} @_omp_reduction_inter_warp_copy_func({{.*}}) {{.*}} {
// CHECK:   %[[WARP_MASTER_CMP:.*]] = icmp eq i32 %nvptx_lane_id, 0
// CHECK:   br i1 %[[WARP_MASTER_CMP]], label %[[WARP_MASTER_BB:.*]], label %{{.*}}

// CHECK: [[WARP_MASTER_BB]]:
// CHECK:   %[[WARP_RESULT_PTR:.*]] = getelementptr inbounds [1 x ptr], ptr %{{.*}}, i64 0, i64 0
// CHECK:   %[[WARP_RESULT:.*]] = load ptr, ptr %[[WARP_RESULT_PTR]], align 8
// CHECK:   %[[ALLOC_MEM_PTR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[WARP_RESULT]], i32 0, i32 0
// CHECK:   %[[ALLOC_MEM:.*]] = load ptr, ptr %[[ALLOC_MEM_PTR]], align 8
// CHECK:   %[[WARP_TRANSFER_SLOT:.*]] = getelementptr inbounds [32 x i32], ptr addrspace(3) @__openmp_nvptx_data_transfer_temporary_storage, i64 0, i32 %nvptx_warp_id
// CHECK:   %[[WARP_RED_RES:.*]] = load i32, ptr %[[ALLOC_MEM]], align 4
// CHECK:   store volatile i32 %[[WARP_RED_RES]], ptr addrspace(3) %[[WARP_TRANSFER_SLOT]], align 4
// CHECK: }
