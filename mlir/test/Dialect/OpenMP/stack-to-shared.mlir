// RUN: mlir-opt --omp-stack-to-shared %s | FileCheck %s

module attributes {omp.is_target_device = true} {

omp.declare_reduction @add_f32 : f32
init {
^bb0(%arg: f32):
  %0 = llvm.mlir.constant(0.0 : f32) : f32
  omp.yield (%0 : f32)
}
combiner {
^bb1(%arg0: f32, %arg1: f32):
  %1 = llvm.fadd %arg0, %arg1 : f32
  omp.yield (%1 : f32)
}
atomic {
^bb2(%arg2: !llvm.ptr, %arg3: !llvm.ptr):
  %2 = llvm.load %arg3 : !llvm.ptr -> f32
  llvm.atomicrmw fadd %arg2, %2 monotonic : !llvm.ptr, f32
  omp.yield
}
omp.private {type = private} @privatizer_i32 : i32
omp.private {type = firstprivate} @firstprivatizer_f32 : f32 copy {
^bb0(%arg0: f32, %arg1: f32):
  omp.yield(%arg0 : f32)
}

llvm.func @foo(%arg0: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>}

// CHECK-LABEL: llvm.func @device_func(
// CHECK-SAME:  %[[N:.*]]: i64, %[[COND:.*]]: i1)
llvm.func @device_func(%arg0: i64, %cond: i1) attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>} {
  // CHECK: %[[ALLOC0:.*]] = omp.alloc_shared_mem %[[N]] x i64 : (i64) -> !llvm.ptr
  %0 = llvm.alloca %arg0 x i64 : (i64) -> !llvm.ptr
  // CHECK: %[[ALLOC1:.*]] = omp.alloc_shared_mem %[[N]] x f32 {alignment = 128 : i64} : (i64) -> !llvm.ptr
  %1 = llvm.alloca %arg0 x f32 {alignment = 128} : (i64) -> !llvm.ptr
  // CHECK: %[[ALLOC2:.*]] = omp.alloc_shared_mem %[[N]] x vector<16xf32> : (i64) -> !llvm.ptr
  %2 = llvm.alloca %arg0 x vector<16xf32> : (i64) -> !llvm.ptr
  // CHECK: %[[ALLOC3:.*]] = omp.alloc_shared_mem %[[N]] x i32 : (i64) -> !llvm.ptr
  %3 = llvm.alloca %arg0 x i32 : (i64) -> !llvm.ptr<5>
  %4 = llvm.addrspacecast %3 : !llvm.ptr<5> to !llvm.ptr

  // CHECK: %[[ALLOC4:.*]] = llvm.alloca %[[N]] x i32 : (i64) -> !llvm.ptr
  %5 = llvm.alloca %arg0 x i32 : (i64) -> !llvm.ptr
  // CHECK: %[[ALLOC5:.*]] = llvm.alloca %[[N]] x i32 : (i64) -> !llvm.ptr
  %6 = llvm.alloca %arg0 x i32 : (i64) -> !llvm.ptr
  // CHECK: llvm.cond_br %[[COND]], ^[[IF:.*]], ^[[ELSE:.*]]
  llvm.cond_br %cond, ^if, ^else

// CHECK: ^[[IF]]:
^if:
  // CHECK: omp.parallel reduction(@add_f32 %[[ALLOC0]] -> %{{.*}} : !llvm.ptr)
  omp.parallel reduction(@add_f32 %0 -> %arg1 : !llvm.ptr) {
    // CHECK: %{{.*}} = llvm.load %[[ALLOC2]]
    %7 = llvm.load %2 : !llvm.ptr -> vector<16xf32>
    // CHECK: %{{.*}} = llvm.alloca
    %8 = llvm.alloca %arg0 x i32 : (i64) -> !llvm.ptr
    // CHECK: omp.wsloop private(@privatizer_i32 %[[ALLOC4]] -> %{{.*}}, @firstprivatizer_f32 %[[ALLOC1]] -> %{{.*}} : !llvm.ptr, !llvm.ptr)
    omp.wsloop private(@privatizer_i32 %5 -> %arg2, @firstprivatizer_f32 %1 -> %arg3 : !llvm.ptr, !llvm.ptr) {
      omp.loop_nest (%arg4) : i64 = (%arg0) to (%arg0) inclusive step (%arg0) {
        llvm.call @foo(%arg1) : (!llvm.ptr) -> ()
        llvm.call @foo(%8) : (!llvm.ptr) -> ()
        llvm.call @foo(%arg2) : (!llvm.ptr) -> ()
        llvm.call @foo(%arg3) : (!llvm.ptr) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  // CHECK: llvm.br ^[[EXIT:.*]]
  llvm.br ^exit

// CHECK: ^[[ELSE]]:
^else:
  // CHECK: llvm.call @foo(%[[ALLOC3]]) : (!llvm.ptr) -> ()
  llvm.call @foo(%4) : (!llvm.ptr) -> ()
  // CHECK: %{{.*}} = llvm.load %[[ALLOC5]]
  %8 = llvm.load %6 : !llvm.ptr -> i32
  // CHECK: llvm.br ^[[EXIT]]
  llvm.br ^exit

// CHECK: ^[[EXIT]]:
^exit:
  // CHECK: omp.free_shared_mem %[[ALLOC0]] : !llvm.ptr
  // CHECK: omp.free_shared_mem %[[ALLOC1]] : !llvm.ptr
  // CHECK: omp.free_shared_mem %[[ALLOC2]] : !llvm.ptr
  // CHECK: omp.free_shared_mem %[[ALLOC3]] : !llvm.ptr
  // CHECK-NOT: omp.free_shared_mem
  // CHECK: llvm.return
  llvm.return
}

// CHECK-LABEL: llvm.func @host_func(
// CHECK-SAME:  %[[N:.*]]: i64)
llvm.func @host_func(%arg0: i64) {
  // CHECK: %[[ALLOC0:.*]] = llvm.alloca %[[N]] x i32 : (i64) -> !llvm.ptr
  %0 = llvm.alloca %arg0 x i32 : (i64) -> !llvm.ptr
  // CHECK: omp.parallel
  omp.parallel {
    // CHECK: llvm.call @foo(%[[ALLOC0]]) : (!llvm.ptr) -> ()
    llvm.call @foo(%0) : (!llvm.ptr) -> ()
    // CHECK: omp.target
    omp.target {
      %c0 = llvm.mlir.constant(1 : i64) : i64
      // CHECK: %[[ALLOC1:.*]] = omp.alloc_shared_mem %{{.*}}
      %1 = llvm.alloca %c0 x i32 : (i64) -> !llvm.ptr
      // CHECK-NEXT: llvm.call @foo(%[[ALLOC1]]) : (!llvm.ptr) -> ()
      llvm.call @foo(%1) : (!llvm.ptr) -> ()
      // CHECK-NEXT: omp.free_shared_mem %[[ALLOC1]] : !llvm.ptr
      // CHECK-NEXT: omp.terminator
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: llvm.func @target_spmd(
llvm.func @target_spmd() {
  // CHECK-NOT: omp.alloc_shared_mem
  // CHECK-NOT: omp.free_shared_mem
  omp.target {
    %c = llvm.mlir.constant(1 : i64) : i64
    %0 = llvm.alloca %c x i32 : (i64) -> !llvm.ptr
    omp.teams {
      %1 = llvm.alloca %c x i32 : (i64) -> !llvm.ptr
      omp.parallel {
        %2 = llvm.alloca %c x i32 : (i64) -> !llvm.ptr
        %3 = llvm.load %0 : !llvm.ptr -> i32
        %4 = llvm.load %1 : !llvm.ptr -> i32
        omp.distribute {
          omp.wsloop {
            omp.loop_nest (%arg0) : i64 = (%c) to (%c) inclusive step (%c) {
              %5 = llvm.load %2 : !llvm.ptr -> i32
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
  // CHECK: return
  llvm.return
}

}
