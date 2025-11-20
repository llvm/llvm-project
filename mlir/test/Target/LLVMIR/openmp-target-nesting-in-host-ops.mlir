// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {

  omp.private {type = private} @i32_privatizer : i32

  llvm.func @test_nested_target_in_parallel(%arg0: !llvm.ptr) {
    omp.parallel {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %4 = omp.map.bounds   lower_bound(%1 : i64) upper_bound(%0 : i64) stride(%1 : i64) start_idx(%1 : i64)
    %mapv1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%4) -> !llvm.ptr {name = ""}
    omp.target map_entries(%mapv1 -> %map_arg : !llvm.ptr) {
      omp.terminator
    }
      omp.terminator
    }
    llvm.return
  }

// CHECK-LABEL: define void @test_nested_target_in_parallel({{.*}}) {
// CHECK-NEXT:    br label %omp.parallel.fake.region
// CHECK:       omp.parallel.fake.region:
// CHECK-NEXT:    br label %omp.region.cont
// CHECK:       omp.region.cont:
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }

  llvm.func @test_nested_target_in_wsloop(%arg0: !llvm.ptr) {
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.alloca %8 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr<5>
    %ascast = llvm.addrspacecast %9 : !llvm.ptr<5> to !llvm.ptr
    %16 = llvm.mlir.constant(10 : i32) : i32
    %17 = llvm.mlir.constant(1 : i32) : i32
    omp.wsloop private(@i32_privatizer %ascast -> %loop_arg : !llvm.ptr) {
      omp.loop_nest (%arg1) : i32 = (%17) to (%16) inclusive step (%17) {
        llvm.store %arg1, %loop_arg : i32, !llvm.ptr
        %0 = llvm.mlir.constant(4 : index) : i64
        %1 = llvm.mlir.constant(1 : index) : i64
        %4 = omp.map.bounds   lower_bound(%1 : i64) upper_bound(%0 : i64) stride(%1 : i64) start_idx(%1 : i64)
        %mapv1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%4) -> !llvm.ptr {name = ""}
        omp.target map_entries(%mapv1 -> %map_arg : !llvm.ptr) {
          omp.terminator
        }
        omp.yield
      }
    }
    llvm.return
  }

// CHECK-LABEL: define void @test_nested_target_in_wsloop(ptr %0) {
// CHECK-NEXT:    %{{.*}} = alloca i32, i64 1, align 4, addrspace(5)
// CHECK-NEXT:    %{{.*}} = addrspacecast ptr addrspace(5) %{{.*}} to ptr
// CHECK-NEXT:    br label %omp.wsloop.fake.region
// CHECK:       omp.wsloop.fake.region:
// CHECK-NEXT:    br label %omp.loop_nest.fake.region
// CHECK:       omp.loop_nest.fake.region:
// CHECK-NEXT:    store i32 poison, ptr %{{.*}}
// CHECK-NEXT:    br label %omp.region.cont1
// CHECK:       omp.region.cont1:
// CHECK-NEXT:    br label %omp.region.cont
// CHECK:       omp.region.cont:
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }

  llvm.func @test_nested_target_in_parallel_with_private(%arg0: !llvm.ptr) {
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.alloca %8 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr<5>
    %ascast = llvm.addrspacecast %9 : !llvm.ptr<5> to !llvm.ptr
    omp.parallel private(@i32_privatizer %ascast -> %i_priv_arg : !llvm.ptr) {
        %1 = llvm.mlir.constant(1 : index) : i64
        // Use the private clause from omp.parallel to make sure block arguments
        // are handled.
        %i_val = llvm.load %i_priv_arg : !llvm.ptr -> i64
        %4 = omp.map.bounds   lower_bound(%1 : i64) upper_bound(%i_val : i64) stride(%1 : i64) start_idx(%1 : i64)
        %mapv1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%4) -> !llvm.ptr {name = ""}
        omp.target map_entries(%mapv1 -> %map_arg : !llvm.ptr) {
          omp.terminator
        }
        omp.terminator
    }
    llvm.return
  }

  llvm.func @test_nested_target_in_task_with_private(%arg0: !llvm.ptr) {
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.alloca %8 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr<5>
    %ascast = llvm.addrspacecast %9 : !llvm.ptr<5> to !llvm.ptr
    omp.task private(@i32_privatizer %ascast -> %i_priv_arg : !llvm.ptr) {
        %1 = llvm.mlir.constant(1 : index) : i64
        // Use the private clause from omp.task to make sure block arguments
        // are handled.
        %i_val = llvm.load %i_priv_arg : !llvm.ptr -> i64
        %4 = omp.map.bounds   lower_bound(%1 : i64) upper_bound(%i_val : i64) stride(%1 : i64) start_idx(%1 : i64)
        %mapv1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.array<10 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%4) -> !llvm.ptr {name = ""}
        omp.target map_entries(%mapv1 -> %map_arg : !llvm.ptr) {
          omp.terminator
        }
        omp.terminator
    }
    llvm.return
  }

  llvm.func @test_target_and_atomic_update(%x: !llvm.ptr, %expr : i32) {
    omp.target {
      omp.terminator
    }

    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%xval: i32):
      %newval = llvm.add %xval, %expr : i32
      omp.yield(%newval : i32)
    }

    llvm.return
  }

// CHECK-LABEL: define void @test_nested_target_in_parallel_with_private({{.*}}) {
// CHECK:        br label %omp.parallel.fake.region
// CHECK:       omp.parallel.fake.region:
// CHECK:         br label %omp.region.cont
// CHECK:       omp.region.cont:
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }

// CHECK-LABEL: define {{.*}} amdgpu_kernel void @__omp_offloading_{{.*}}_nested_target_in_parallel_{{.*}} {
// CHECK:         call i32 @__kmpc_target_init
// CHECK:       user_code.entry:
// CHECK:         call void @__kmpc_target_deinit()
// CHECK:         ret void
// CHECK:       }

// CHECK-LABEL: define {{.*}} amdgpu_kernel void @__omp_offloading_{{.*}}_test_nested_target_in_wsloop_{{.*}} {
// CHECK:         call i32 @__kmpc_target_init
// CHECK:       user_code.entry:
// CHECK:         call void @__kmpc_target_deinit()
// CHECK:         ret void
// CHECK:       }

// CHECK-LABEL: define {{.*}} amdgpu_kernel void @__omp_offloading_{{.*}}_test_nested_target_in_parallel_with_private_{{.*}} {
// CHECK:         call i32 @__kmpc_target_init
// CHECK:       user_code.entry:
// CHECK:         call void @__kmpc_target_deinit()
// CHECK:         ret void
// CHECK:       }

// CHECK-LABEL: define {{.*}} amdgpu_kernel void @__omp_offloading_{{.*}}_test_nested_target_in_task_with_private_{{.*}} {
// CHECK:         call i32 @__kmpc_target_init
// CHECK:       user_code.entry:
// CHECK:         call void @__kmpc_target_deinit()
// CHECK:         ret void
// CHECK:       }

// CHECK-LABEL: define {{.*}} amdgpu_kernel void @__omp_offloading_{{.*}}_test_target_and_atomic_update_{{.*}} {
// CHECK:         call i32 @__kmpc_target_init
// CHECK:       user_code.entry:
// CHECK:         call void @__kmpc_target_deinit()
// CHECK:         ret void
// CHECK:       }
}
