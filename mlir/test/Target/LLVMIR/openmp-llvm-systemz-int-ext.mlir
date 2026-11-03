// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Check SystemZ signext on i32 parameters created by OpenMP lowering.

module attributes {llvm.target_triple = "s390x-unknown-linux-gnu"} {
  llvm.func external @foo() -> ()

  llvm.func @omp_task_if(%boolexpr: i1) {
    omp.task if(%boolexpr) {
      llvm.call @foo() : () -> ()
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define void @omp_task_if(
// CHECK: call void @[[OUTLINED:.+]](i32 signext %{{.+}})
// CHECK: define internal void @[[OUTLINED]](i32 signext %{{.+}})

// -----

module attributes {llvm.target_triple = "s390x-unknown-linux-gnu"} {
  llvm.func @omp_atomic_complex() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(f32, f32)> : (i64) -> !llvm.ptr
    %2 = llvm.alloca %0 x !llvm.struct<(f32, f32)> : (i64) -> !llvm.ptr
    omp.atomic.read %1 = %2 : !llvm.ptr, !llvm.ptr, !llvm.struct<(f32, f32)>
    %3 = llvm.mlir.undef : !llvm.struct<(f32, f32)>
    omp.atomic.write %1 = %3 : !llvm.ptr, !llvm.struct<(f32, f32)>
    llvm.return
  }
}

// CHECK-LABEL: define void @omp_atomic_complex()
// CHECK: call void @__atomic_load(i64 8, ptr %{{.+}}, ptr %{{.+}}, i32 signext 0)
// CHECK: call void @__atomic_store(i64 8, ptr %{{.+}}, ptr %{{.+}}, i32 signext 0)
// CHECK: declare void @__atomic_load(i64, ptr, ptr, i32 signext)
// CHECK: declare void @__atomic_store(i64, ptr, ptr, i32 signext)

// -----

module attributes {llvm.target_triple = "s390x-unknown-linux-gnu"} {
  llvm.func @omp_atomic_update_complex(%x: !llvm.ptr) {
    %c = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %u = llvm.mlir.undef : !llvm.struct<(f32, f32)>
    %v0 = llvm.insertvalue %c, %u[0] : !llvm.struct<(f32, f32)>
    %v1 = llvm.insertvalue %c, %v0[1] : !llvm.struct<(f32, f32)>
    omp.atomic.update %x : !llvm.ptr {
    ^bb0(%a: !llvm.struct<(f32, f32)>):
      omp.yield(%v1 : !llvm.struct<(f32, f32)>)
    }
    llvm.return
  }
}

// CHECK-LABEL: define void @omp_atomic_update_complex(
// CHECK: call i1 @__atomic_compare_exchange(i64 8, ptr %{{.+}}, ptr %{{.+}}, ptr %{{.+}}, i32 signext 0, i32 signext 0)
// CHECK: declare i1 @__atomic_compare_exchange(i64, ptr, ptr, ptr, i32 signext, i32 signext)

// -----

module attributes {llvm.target_triple = "s390x-unknown-linux-gnu"} {
  omp.private {type = private} @i_private : i32

  omp.private {type = firstprivate} @a_firstprivate : i32 copy {
  ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
    %0 = llvm.load %arg0 : !llvm.ptr -> i32
    llvm.store %0, %arg1 : i32, !llvm.ptr
    omp.yield(%arg1 : !llvm.ptr)
  }

  llvm.func @omp_taskloop() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %3 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(5 : i32) : i32
    omp.taskloop.context private(@a_firstprivate %3 -> %arg0, @i_private %1 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      omp.taskloop.wrapper {
        omp.loop_nest (%arg2) : i32 = (%7) to (%8) inclusive step (%7) {
          llvm.store %arg2, %arg1 : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    } {omp.combined}
    llvm.return
  }
}

// CHECK-LABEL: define void @omp_taskloop(
// CHECK: define internal void @omp_taskloop..omp_par(i32 signext %{{.+}}
// CHECK: define internal void @omp_taskloop_dup(ptr %{{.+}}, ptr %{{.+}}, i32 signext %{{.+}})

// -----

module attributes {llvm.target_triple = "s390x-unknown-linux-gnu", omp.is_target_device = false} {
  llvm.func @omp_target_nowait_() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x f32 {bindc_name = "x"} : (i64) -> !llvm.ptr
    %3 = omp.map.info var_ptr(%1 : !llvm.ptr, f32) map_clauses(tofrom) capture(ByRef) name("x") -> !llvm.ptr
    omp.target kernel_type(generic) nowait map_entries(%3 -> %arg0 : !llvm.ptr) {
      %4 = llvm.mlir.constant(5.000000e+00 : f32) : f32
      llvm.store %4, %arg0 : f32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// CHECK-LABEL: define void @omp_target_nowait_()
// CHECK: call void @.omp_target_task_proxy_func(i32 signext %{{.+}}, ptr %{{.+}})
// CHECK: define internal void @.omp_target_task_proxy_func(i32 signext %{{.+}}, ptr %{{.+}})
