// Test code-gen for `omp.distribute` ops with delayed privatizers (i.e. using
// `omp.private` ops).

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

omp.private {type = private} @_QFEi_private_i32 : i32
omp.private {type = private} @_QFEpriv_val_dist_private_f32 : f32

llvm.func @_QQmain() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x f32 {bindc_name = "priv_val_dist"} : (i64) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {bindc_name = "i"} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(3.140000e+00 : f32) : f32
  %5 = llvm.mlir.constant(1000 : i32) : i32
  %6 = llvm.mlir.constant(1 : i32) : i32

  omp.teams {
    omp.distribute private(@_QFEpriv_val_dist_private_f32 %1 -> %arg0, @_QFEi_private_i32 %3 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      omp.loop_nest (%arg2) : i32 = (%6) to (%5) inclusive step (%6) {
        llvm.store %arg2, %arg1 : i32, !llvm.ptr
        llvm.store %4, %arg0 : f32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  }

  llvm.return
}

// CHECK-LABEL: define void @_QQmain() {
// CHECK:         call void {{.*}} @__kmpc_fork_teams(ptr @{{.*}}, i32 0, ptr @[[TEAMS_FUNC:.*]])
// CHECK-NEXT:    br label %teams.exit
// CHECK:       }

// CHECK:       define internal void @[[TEAMS_FUNC]]({{.*}}) {
// CHECK:         call void @[[DIST_FUNC:.*]]()
// CHECK-NEXT:    br label %distribute.exit
// CHECK:       }

// CHECK:       define internal void @[[DIST_FUNC]]() {
// CHECK:         %[[PRIV_VAR_ALLOC:.*]] = alloca float, align 4
// CHECK:         %[[IV_ALLOC:.*]] = alloca i32, align 4

// CHECK:       omp.loop_nest.region:
// CHECK-NEXT:    store i32 %{{.*}}, ptr %[[IV_ALLOC]], align 4
// CHECK-NEXT:    store float 0x40091EB860000000, ptr %[[PRIV_VAR_ALLOC]], align 4
// CHECK:       }

// -----

llvm.func @foo_free(!llvm.ptr)

omp.private {type = firstprivate} @_QFEpriv_val_dist_firstprivate_f32 : f32 copy {
^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
  %0 = llvm.load %arg0 : !llvm.ptr -> f32
  llvm.store %0, %arg1 : f32, !llvm.ptr
  omp.yield(%arg1 : !llvm.ptr)
} dealloc {
  ^bb0(%arg0: !llvm.ptr):
    llvm.call @foo_free(%arg0) : (!llvm.ptr) -> ()
    omp.yield
}

llvm.func @_QQmain() {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x f32 {bindc_name = "priv_val_dist"} : (i64) -> !llvm.ptr
  %4 = llvm.mlir.constant(3.140000e+00 : f32) : f32
  %6 = llvm.mlir.constant(1 : i32) : i32
  omp.distribute private(@_QFEpriv_val_dist_firstprivate_f32 %1 -> %arg0 : !llvm.ptr) {
    omp.loop_nest (%arg2) : i32 = (%6) to (%6) inclusive step (%6) {
      llvm.store %4, %arg0 : f32, !llvm.ptr
      omp.yield
    }
  }
  llvm.return
}

// CHECK-LABEL: define void @_QQmain() {
// CHECK:         %[[SHARED_VAR_ALLOC:.*]] = alloca float, i64 1, align 4
// CHECK:         %[[SHARED_VAR_PTR:.*]] = getelementptr { ptr }, ptr %[[DIST_PARAM:.*]], i32 0, i32 0
// CHECK:         store ptr %[[SHARED_VAR_ALLOC]], ptr %[[SHARED_VAR_PTR]], align 8
// CHECK:         call void @[[DIST_FUNC:.*]](ptr %[[DIST_PARAM]])
// CHECK-NEXT:    br label %distribute.exit
// CHECK:       }

// CHECK:       define internal void @[[DIST_FUNC]](ptr %[[DIST_ARG:.*]]) {
// CHECK:         %[[SHARED_VAR_GEP:.*]] = getelementptr { ptr }, ptr %[[DIST_ARG]], i32 0, i32 0
// CHECK:         %[[SHARED_VAR_PTR2:.*]] = load ptr, ptr %[[SHARED_VAR_GEP]], align 8
// CHECK:         %[[PRIV_VAR_ALLOC:.*]] = alloca float, align 4

// CHECK:       omp.private.copy:
// CHECK-NEXT:    %[[SHARED_VAR_VAL:.*]] = load float, ptr %[[SHARED_VAR_PTR2]], align 4
// CHECK-NEXT:    store float %[[SHARED_VAR_VAL]], ptr %[[PRIV_VAR_ALLOC]], align 4

// CHECK:       omp_loop.after:
// CHECK-NEXT:    br label %omp.region.cont

// CHECK:       omp.region.cont:
// CHECK-NEXT:   call void @foo_free(ptr %[[PRIV_VAR_ALLOC]])

// CHECK:       omp.loop_nest.region:
// CHECK-NEXT:    store float 0x40091EB860000000, ptr %[[PRIV_VAR_ALLOC]], align 4
// CHECK:       }


