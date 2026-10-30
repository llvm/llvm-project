// RUN: mlir-translate --mlir-to-llvmir --split-input-file %s | FileCheck %s

// Keep private variables and loop callback arguments in the entry block when
// nested SIMD regions split the allocation block.

module attributes {
  dlti.dl_spec = #dlti.dl_spec<"dlti.alloca_memory_space" = 5 : ui64>,
  llvm.target_triple = "amdgcn-amd-amdhsa",
  omp.is_gpu = true,
  omp.is_target_device = true
} {
  omp.private {type = private} @private_i : i32

  llvm.func @distribute_simd(%array: !llvm.ptr, %n: i32) attributes {
    omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>
  } {
    %one = llvm.mlir.constant(1 : i32) : i32
    %i = llvm.alloca %one x i32 : (i32) -> !llvm.ptr<5>
    %i_cast = llvm.addrspacecast %i : !llvm.ptr<5> to !llvm.ptr
    omp.teams {
      omp.distribute {
        omp.simd private(@private_i %i_cast -> %private_i : !llvm.ptr) {
          omp.loop_nest (%iv) : i32 = (%one) to (%n) inclusive step (%one) {
            llvm.store %iv, %private_i : i32, !llvm.ptr
            %offset = llvm.load %private_i : !llvm.ptr -> i32
            %element = llvm.getelementptr %array[%offset] : (!llvm.ptr, i32) -> !llvm.ptr, i32
            llvm.store %iv, %element : i32, !llvm.ptr
            omp.yield
          }
        } {omp.composite}
      } {omp.composite}
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @distribute_simd..omp_par.1(
// CHECK: distribute.alloca:
// CHECK-NOT: br
// CHECK: %[[ARGS:.*]] = alloca { ptr, ptr }, align 8, addrspace(5)
// CHECK: %[[ARGS_CAST:.*]] = addrspacecast ptr addrspace(5) %[[ARGS]] to ptr
// CHECK: br label %omp.region.after_alloca
// CHECK-NOT: = alloca {
// CHECK: call void @__kmpc_distribute_static_loop_4u({{.*}}, ptr %[[ARGS_CAST]],
// CHECK: ret void

// -----

module attributes {
  dlti.dl_spec = #dlti.dl_spec<"dlti.alloca_memory_space" = 5 : ui64>,
  llvm.target_triple = "amdgcn-amd-amdhsa",
  omp.is_gpu = true,
  omp.is_target_device = true
} {
  omp.private {type = private} @private_i : i32

  llvm.func @distribute_parallel_simd(%array: !llvm.ptr, %n: i32) attributes {
    omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>
  } {
    %one = llvm.mlir.constant(1 : i32) : i32
    %i = llvm.alloca %one x i32 : (i32) -> !llvm.ptr<5>
    %i_cast = llvm.addrspacecast %i : !llvm.ptr<5> to !llvm.ptr
    omp.teams {
      omp.parallel {
        omp.distribute {
          omp.wsloop {
            omp.simd private(@private_i %i_cast -> %private_i : !llvm.ptr) {
              omp.loop_nest (%iv) : i32 = (%one) to (%n) inclusive step (%one) {
                llvm.store %iv, %private_i : i32, !llvm.ptr
                %offset = llvm.load %private_i : !llvm.ptr -> i32
                %element = llvm.getelementptr %array[%offset] : (!llvm.ptr, i32) -> !llvm.ptr, i32
                llvm.store %iv, %element : i32, !llvm.ptr
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
        } {omp.composite}
        omp.terminator
      } {omp.composite}
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define internal void @distribute_parallel_simd..omp_par.1(
// CHECK: distribute.alloca:
// CHECK-NOT: br
// CHECK: %[[ARGS:.*]] = alloca { ptr, ptr }, align 8, addrspace(5)
// CHECK: %[[ARGS_CAST:.*]] = addrspacecast ptr addrspace(5) %[[ARGS]] to ptr
// CHECK-NOT: br
// CHECK: %[[PRIVATE:.*]] = alloca i32, align 4, addrspace(5)
// CHECK: %[[PRIVATE_CAST:.*]] = addrspacecast ptr addrspace(5) %[[PRIVATE]] to ptr
// CHECK: br label %omp.region.after_alloca
// CHECK-NOT: = alloca
// CHECK: store ptr %[[PRIVATE_CAST]], ptr addrspace(5)
// CHECK: call void @__kmpc_distribute_for_static_loop_4u({{.*}}, ptr %[[ARGS_CAST]],
// CHECK: ret void
