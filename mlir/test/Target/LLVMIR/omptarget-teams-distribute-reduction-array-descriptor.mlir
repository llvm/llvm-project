// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck --check-prefixes=AMDGCN,NVPTX %s

// Minimal MLIR to exercise array byref reduction descriptor handling in
// target teams distribute parallel do.

module attributes {dlti.dl_spec = #dlti.dl_spec<"dlti.alloca_memory_space" = 5 : ui64, "dlti.global_memory_space" = 1 : ui64>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  omp.declare_reduction @add_reduction_byref_box_4xi32 : !llvm.ptr attributes {byref_element_type = !llvm.array<4 x i32>} alloc {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    omp.yield(%2 : !llvm.ptr)
  } init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg1 : !llvm.ptr)
  } combiner {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg0 : !llvm.ptr)
  } data_ptr_ptr {
  ^bb0(%arg0: !llvm.ptr):
    %0 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    omp.yield(%0 : !llvm.ptr)
  }

  llvm.func @test_array_reduction_() attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.array<4 x i32> : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %3 = omp.map.info var_ptr(%2 : !llvm.ptr, !llvm.array<4 x i32>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "red_array"}
    omp.target map_entries(%3 -> %arg0 : !llvm.ptr) {
      %4 = llvm.mlir.constant(1 : i32) : i32
      %5 = llvm.mlir.constant(1000 : i32) : i32
      omp.teams reduction(byref @add_reduction_byref_box_4xi32 %arg0 -> %arg1 : !llvm.ptr) {
        omp.parallel {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%iv) : i32 = (%4) to (%5) inclusive step (%4) {
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

// Verify descriptor is copied via memcpy and base_ptr is updated in all helpers
// AMDGCN-LABEL: define internal void @_omp_reduction_shuffle_and_reduce_func
// AMDGCN: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 {{[0-9]+}}, i1 false)
// AMDGCN: getelementptr {{.*}} ptr {{%.*}}, i32 0, i32 0
// AMDGCN: store ptr {{%.*}}, ptr

// AMDGCN-LABEL: define internal void @_omp_reduction_list_to_global_reduce_func
// AMDGCN: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 {{[0-9]+}}, i1 false)
// AMDGCN: getelementptr {{.*}} ptr {{%.*}}, i32 0, i32 0
// AMDGCN: store ptr {{%.*}}, ptr

// AMDGCN-LABEL: define internal void @_omp_reduction_global_to_list_copy_func
// AMDGCN: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 {{[0-9]+}}, i1 false)
// AMDGCN: getelementptr {{.*}} ptr {{%.*}}, i32 0, i32 0
// AMDGCN: store ptr {{%.*}}, ptr

// -----

module attributes {llvm.target_triple = "nvptx64-nvidia-cuda", omp.is_gpu = true, omp.is_target_device = true} {
  omp.declare_reduction @add_reduction_byref_box_4xi32 : !llvm.ptr attributes {byref_element_type = !llvm.array<4 x i32>} alloc {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    omp.yield(%2 : !llvm.ptr)
  } init {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg1 : !llvm.ptr)
  } combiner {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    omp.yield(%arg0 : !llvm.ptr)
  } data_ptr_ptr {
  ^bb0(%arg0: !llvm.ptr):
    %0 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    omp.yield(%0 : !llvm.ptr)
  }

  llvm.func @test_array_reduction_() attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.array<4 x i32> : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %3 = omp.map.info var_ptr(%2 : !llvm.ptr, !llvm.array<4 x i32>) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "red_array"}
    omp.target map_entries(%3 -> %arg0 : !llvm.ptr) {
      %4 = llvm.mlir.constant(1 : i32) : i32
      %5 = llvm.mlir.constant(1000 : i32) : i32
      omp.teams reduction(byref @add_reduction_byref_box_4xi32 %arg0 -> %arg1 : !llvm.ptr) {
        omp.parallel {
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%iv) : i32 = (%4) to (%5) inclusive step (%4) {
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

// Verify descriptor is copied via memcpy and base_ptr is updated in all helpers
// NVPTX-LABEL: define internal void @_omp_reduction_shuffle_and_reduce_func
// NVPTX: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 {{[0-9]+}}, i1 false)
// NVPTX: getelementptr {{.*}} ptr {{%.*}}, i32 0, i32 0
// NVPTX: store ptr {{%.*}}, ptr

// NVPTX-LABEL: define internal void @_omp_reduction_list_to_global_reduce_func
// NVPTX: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 {{[0-9]+}}, i1 false)
// NVPTX: getelementptr {{.*}} ptr {{%.*}}, i32 0, i32 0
// NVPTX: store ptr {{%.*}}, ptr

// NVPTX-LABEL: define internal void @_omp_reduction_global_to_list_copy_func
// NVPTX: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 {{[0-9]+}}, i1 false)
// NVPTX: getelementptr {{.*}} ptr {{%.*}}, i32 0, i32 0
// NVPTX: store ptr {{%.*}}, ptr

