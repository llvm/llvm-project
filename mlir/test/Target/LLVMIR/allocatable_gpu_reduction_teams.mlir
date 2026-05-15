// Tests cross-teams by-ref GPU reductions.

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
    %9 = omp.map.info var_ptr(%5 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%8 : !llvm.ptr, f32) -> !llvm.ptr {name = ""}
    %10 = omp.map.info var_ptr(%5 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(always, descriptor, to) capture(ByRef) members(%9 : [0] : !llvm.ptr) -> !llvm.ptr {name = "scalar_alloc"}
    %attach = omp.map.info var_ptr(%5 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(ref_ptr_ptee, attach) capture(ByRef) var_ptr_ptr(%8 : !llvm.ptr, f32) -> !llvm.ptr {name = "scalar_alloc"}
    omp.target map_entries(%10 -> %arg0 : !llvm.ptr) {
      %14 = llvm.mlir.constant(1000000 : i32) : i32
      %15 = llvm.mlir.constant(1 : i32) : i32
      omp.teams reduction(byref @add_reduction_byref_box_heap_f32 %arg0 -> %arg3 : !llvm.ptr) {
        omp.parallel {
          omp.distribute {
            omp.wsloop reduction(byref @add_reduction_byref_box_heap_f32 %arg3 -> %arg5 : !llvm.ptr) {
              omp.loop_nest (%arg6) : i32 = (%15) to (%14) inclusive step (%15) {
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: %[[GLOBALIZED_LOCALS:.*]] = type { float }

// CHECK: define internal void @_omp_reduction_list_to_global_copy_func({{.*}}) {{.*}} {
// CHECK:   %[[RED_ARR_LIST:.*]] = getelementptr inbounds [1 x ptr], ptr %{{.*}}, i64 0, i64 0
// CHECK:   %[[RED_ELEM_PTR:.*]] = load ptr, ptr %[[RED_ARR_LIST]], align 8
// CHECK:   %[[GLOB_ELEM_PTR:.*]] = getelementptr inbounds %[[GLOBALIZED_LOCALS]], ptr %{{.*}}, i32 0, i32 0
// CHECK:   %[[ALLOC_PTR_PTR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[RED_ELEM_PTR]], i32 0, i32 0
// CHECK:   %[[ALLOC_PTR:.*]] = load ptr, ptr %[[ALLOC_PTR_PTR]], align 8
// CHECK:   %[[ALLOC_VAL:.*]] = load float, ptr %[[ALLOC_PTR]], align 4
// Verify that the actual value managed by the descriptor is stored in the globalized 
// locals arrays; rather than a pointer to the descriptor or a pointer to the value.
// CHECK:   store float %[[ALLOC_VAL]], ptr %[[GLOB_ELEM_PTR]], align 4
// CHECK: }

// CHECK: define internal void @_omp_reduction_list_to_global_reduce_func({{.*}}) {{.*}} {
// Allocate a descriptor to manage the element retrieved from the globalized local array.
// CHECK:   %[[ALLOC_DESC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, align 8, addrspace(5)
// CHECK:   %[[ALLOC_DESC_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[ALLOC_DESC]] to ptr

// CHECK:   %[[RED_ARR_LIST:.*]] = getelementptr inbounds [1 x ptr], ptr %{{.*}}, i64 0, i64 0
// CHECK:   %[[GLOB_ELEM_PTR:.*]] = getelementptr inbounds %[[GLOBALIZED_LOCALS]], ptr %{{.*}}, i32 0, i32 0
// CHECK:   %[[ALLOC_PTR_PTR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ALLOC_DESC_ASCAST]], i32 0, i32 0
// Store the pointer to the gloalized local element into the locally allocated descriptor.
// CHECK:   store ptr %[[GLOB_ELEM_PTR]], ptr %[[ALLOC_PTR_PTR]], align 8
// CHECK:   store ptr %[[ALLOC_DESC_ASCAST]], ptr %[[RED_ARR_LIST]], align 8
// CHECK: }

// CHECK: define internal void @_omp_reduction_global_to_list_copy_func({{.*}}) {{.*}} {
// CHECK:   %[[RED_ARR_LIST:.*]] = getelementptr inbounds [1 x ptr], ptr %{{.*}}, i64 0, i64 0
// CHECK:   %[[RED_ELEM_PTR:.*]] = load ptr, ptr %[[RED_ARR_LIST]], align 8
// CHECK:   %[[GLOB_ELEM_PTR:.*]] = getelementptr inbounds %[[GLOBALIZED_LOCALS]], ptr %{{.*}}, i32 0, i32 0
// CHECK:   %[[ALLOC_PTR_PTR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[RED_ELEM_PTR]], i32 0, i32 0
// Similar to _omp_reduction_list_to_global_copy_func(...) but in the reverse direction; i.e.
// the globalized local array is copied from rather than copied to.
// CHECK:   %[[ALLOC_PTR:.*]] = load ptr, ptr %[[ALLOC_PTR_PTR]], align 8
// CHECK:   %[[ALLOC_VAL:.*]] = load float, ptr %[[GLOB_ELEM_PTR]], align 4
// CHECK:   store float %[[ALLOC_VAL]], ptr %[[ALLOC_PTR]], align 4
// CHECK: }

// CHECK: define internal void @_omp_reduction_global_to_list_reduce_func({{.*}}) {{.*}} {
// Allocate a descriptor to manage the element retrieved from the globalized local array.
// CHECK:   %[[ALLOC_DESC:.*]] = alloca { ptr, i64, i32, i8, i8, i8, i8 }, align 8, addrspace(5)
// CHECK:   %[[ALLOC_DESC_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[ALLOC_DESC]] to ptr

// CHECK:   %[[RED_ARR_LIST:.*]] = getelementptr inbounds [1 x ptr], ptr %{{.*}}, i64 0, i64 0
// CHECK:   %[[GLOB_ELEM_PTR:.*]] = getelementptr inbounds %[[GLOBALIZED_LOCALS]], ptr %{{.*}}, i32 0, i32 0
// CHECK:   %[[ALLOC_PTR_PTR:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8 }, ptr %[[ALLOC_DESC_ASCAST]], i32 0, i32 0
// Store the pointer to the gloalized local element into the locally allocated descriptor.
// CHECK:   store ptr %[[GLOB_ELEM_PTR]], ptr %[[ALLOC_PTR_PTR]], align 8
// CHECK:   store ptr %[[ALLOC_DESC_ASCAST]], ptr %[[RED_ARR_LIST]], align 8
// CHECK: }
