// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// A privatization barrier needs every thread in its team to reach it. These
// constructs select or serialize threads, even when more than one thread can
// execute the region over time.

omp.private {type = private} @private_i32 : i32
omp.private {type = firstprivate} @firstprivate_i32 : i32 copy {
^bb0(%from: !llvm.ptr, %to: !llvm.ptr):
  %value = llvm.load %from : !llvm.ptr -> i32
  llvm.store %value, %to : i32, !llvm.ptr
  omp.yield(%to : !llvm.ptr)
}

// CHECK-LABEL: define void @masked_private(
llvm.func @masked_private(%ptr: !llvm.ptr, %filter: i32) {
  %lo = llvm.mlir.constant(1 : i32) : i32
  %hi = llvm.mlir.constant(2 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.masked filter(%filter : i32) {
      omp.taskloop.context private(@private_i32 %ptr -> %arg0 : !llvm.ptr) private_barrier {
        omp.taskloop.wrapper {
          omp.loop_nest (%i) : i32 = (%lo) to (%hi) inclusive step (%step) {
            llvm.store %i, %arg0 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      } {omp.combined}
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK: omp.private.copy:
// CHECK-NOT: call void @__kmpc_barrier
// CHECK: br label %omp.taskloop.wrapper.start

// The old firstprivate path needs the same protection.
// CHECK-LABEL: define void @masked_firstprivate(
llvm.func @masked_firstprivate(%ptr: !llvm.ptr, %filter: i32) {
  %lo = llvm.mlir.constant(1 : i32) : i32
  %hi = llvm.mlir.constant(2 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.masked filter(%filter : i32) {
      omp.taskloop.context private(@firstprivate_i32 %ptr -> %arg0 : !llvm.ptr) private_barrier {
        omp.taskloop.wrapper {
          omp.loop_nest (%i) : i32 = (%lo) to (%hi) inclusive step (%step) {
            llvm.store %i, %arg0 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      } {omp.combined}
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK: omp.private.copy{{[0-9]*}}:
// CHECK-NOT: call void @__kmpc_barrier
// CHECK: br label %omp.taskloop.wrapper.start

// CHECK-LABEL: define void @master_private(
llvm.func @master_private(%ptr: !llvm.ptr) {
  %lo = llvm.mlir.constant(1 : i32) : i32
  %hi = llvm.mlir.constant(2 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.master {
      omp.taskloop.context private(@private_i32 %ptr -> %arg0 : !llvm.ptr) private_barrier {
        omp.taskloop.wrapper {
          omp.loop_nest (%i) : i32 = (%lo) to (%hi) inclusive step (%step) {
            llvm.store %i, %arg0 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      } {omp.combined}
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK: omp.private.copy:
// CHECK-NOT: call void @__kmpc_barrier
// CHECK: br label %omp.taskloop.wrapper.start

// CHECK-LABEL: define void @critical_private(
llvm.func @critical_private(%ptr: !llvm.ptr) {
  %lo = llvm.mlir.constant(1 : i32) : i32
  %hi = llvm.mlir.constant(2 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.critical {
      omp.taskloop.context private(@private_i32 %ptr -> %arg0 : !llvm.ptr) private_barrier {
        omp.taskloop.wrapper {
          omp.loop_nest (%i) : i32 = (%lo) to (%hi) inclusive step (%step) {
            llvm.store %i, %arg0 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      } {omp.combined}
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK: omp.private.copy:
// CHECK-NOT: call void @__kmpc_barrier
// CHECK: br label %omp.taskloop.wrapper.start

// CHECK-LABEL: define void @single_private(
llvm.func @single_private(%ptr: !llvm.ptr) {
  %lo = llvm.mlir.constant(1 : i32) : i32
  %hi = llvm.mlir.constant(2 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.single {
      omp.taskloop.context private(@private_i32 %ptr -> %arg0 : !llvm.ptr) private_barrier {
        omp.taskloop.wrapper {
          omp.loop_nest (%i) : i32 = (%lo) to (%hi) inclusive step (%step) {
            llvm.store %i, %arg0 : i32, !llvm.ptr
            omp.yield
          }
        }
        omp.terminator
      } {omp.combined}
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK: omp.private.copy:
// CHECK-NOT: call void @__kmpc_barrier
// CHECK: br label %omp.taskloop.wrapper.start

// CHECK-LABEL: define void @section_private(
llvm.func @section_private(%ptr: !llvm.ptr) {
  %lo = llvm.mlir.constant(1 : i32) : i32
  %hi = llvm.mlir.constant(2 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.sections {
      omp.section {
        omp.taskloop.context private(@private_i32 %ptr -> %arg0 : !llvm.ptr) private_barrier {
          omp.taskloop.wrapper {
            omp.loop_nest (%i) : i32 = (%lo) to (%hi) inclusive step (%step) {
              llvm.store %i, %arg0 : i32, !llvm.ptr
              omp.yield
            }
          }
          omp.terminator
        } {omp.combined}
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK: omp.private.copy:
// CHECK-NOT: call void @__kmpc_barrier
// CHECK: br label %omp.taskloop.wrapper.start

// A nested parallel region creates a new team whose members all enter its
// privatization code, even when only one outer-team thread launches it.
// CHECK-LABEL: define void @nested_parallel_private(
llvm.func @nested_parallel_private(%ptr: !llvm.ptr, %filter: i32) {
  omp.parallel {
    omp.masked filter(%filter : i32) {
      omp.parallel private(@private_i32 %ptr -> %arg0 : !llvm.ptr) private_barrier {
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
// CHECK: call void @__kmpc_barrier
