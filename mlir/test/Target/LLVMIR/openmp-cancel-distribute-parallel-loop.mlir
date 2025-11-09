// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @cancel_distribute_parallel_do(%lb : i32, %ub : i32, %step : i32) {
  omp.teams {
    omp.parallel {
      omp.distribute {
        omp.wsloop {
          omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
            omp.cancel cancellation_construct_type(loop)
            omp.yield
          }
        } {omp.composite}
      } {omp.composite}
      omp.terminator
    } {omp.composite}
    omp.terminator
  }
  llvm.return
}
// CHECK-LABEL: define internal void @cancel_distribute_parallel_do..omp_par
// [...]
// CHECK:       omp_loop.cond:
// CHECK:         %[[VAL_102:.*]] = icmp ult i32 %{{.*}}, %{{.*}}
// CHECK:         br i1 %[[VAL_102]], label %omp_loop.body, label %omp_loop.exit
// CHECK:       omp_loop.exit:
// CHECK:         call void @__kmpc_for_static_fini(
// CHECK:         %[[VAL_106:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         call void @__kmpc_barrier(ptr @2, i32 %[[VAL_106]])
// CHECK:         br label %omp_loop.after
// CHECK:       omp_loop.after:
// CHECK:         br label %omp.region.cont6
// CHECK:       omp.region.cont6:
// CHECK:         br label %omp.region.cont4
// CHECK:       omp.region.cont4:
// CHECK:         br label %omp.par.exit.exitStub
// CHECK:       omp_loop.body:
// CHECK:         %[[VAL_111:.*]] = add i32 %{{.*}}, %{{.*}}
// CHECK:         %[[VAL_112:.*]] = mul i32 %[[VAL_111]], %{{.*}}
// CHECK:         %[[VAL_113:.*]] = add i32 %[[VAL_112]], %{{.*}}
// CHECK:         br label %omp.loop_nest.region
// CHECK:       omp.loop_nest.region:
// CHECK:         %[[VAL_115:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CHECK:         %[[VAL_116:.*]] = call i32 @__kmpc_cancel(ptr @1, i32 %[[VAL_115]], i32 2)
// CHECK:         %[[VAL_117:.*]] = icmp eq i32 %[[VAL_116]], 0
// CHECK:         br i1 %[[VAL_117]], label %omp.loop_nest.region.split, label %omp.loop_nest.region.cncl
// CHECK:       omp.loop_nest.region.cncl:
// CHECK:         br label %omp_loop.exit
// CHECK:       omp.loop_nest.region.split:
// CHECK:         br label %omp.region.cont7
// CHECK:       omp.region.cont7:
// CHECK:         br label %omp_loop.inc
// CHECK:       omp_loop.inc:
// CHECK:         %[[VAL_100:.*]] = add nuw i32 %{{.*}}, 1
// CHECK:         br label %omp_loop.header
// CHECK:       omp.par.exit.exitStub:
// CHECK:         ret void

