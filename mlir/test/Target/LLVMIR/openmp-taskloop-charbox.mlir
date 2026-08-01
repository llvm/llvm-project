// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

module {
  llvm.func @use_body_box(!llvm.ptr, i64)
  llvm.func @copy_box(!llvm.ptr, i64)
  llvm.func @dealloc_box(!llvm.ptr, i64)

  omp.private {type = firstprivate} @box_firstprivate : !llvm.struct<(ptr, i64)> init {
  ^bb0(%arg0: !llvm.struct<(ptr, i64)>, %arg1: !llvm.struct<(ptr, i64)>):
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, i64)>
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(ptr, i64)>
    %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(ptr, i64)>
    omp.yield(%4 : !llvm.struct<(ptr, i64)>)
  } copy {
  ^bb0(%arg0: !llvm.struct<(ptr, i64)>, %arg1: !llvm.struct<(ptr, i64)>):
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, i64)>
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
    llvm.call @copy_box(%0, %1) : (!llvm.ptr, i64) -> ()
    omp.yield(%arg0 : !llvm.struct<(ptr, i64)>)
  } dealloc {
  ^bb0(%arg0: !llvm.struct<(ptr, i64)>):
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, i64)>
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64)>
    llvm.call @dealloc_box(%0, %1) : (!llvm.ptr, i64) -> ()
    omp.yield
  }

  llvm.func @test(%arg0: !llvm.ptr, %arg1: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, i64)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, i64)>
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    omp.taskloop.context private(@box_firstprivate %2 -> %arg2 : !llvm.struct<(ptr, i64)>) {
      omp.taskloop.wrapper {
        omp.loop_nest (%arg3) : i32 = (%c1) to (%c2) inclusive step (%c1) {
          %3 = llvm.extractvalue %arg2[0] : !llvm.struct<(ptr, i64)>
          %4 = llvm.extractvalue %arg2[1] : !llvm.struct<(ptr, i64)>
          llvm.call @use_body_box(%3, %4) : (!llvm.ptr, i64) -> ()
          omp.yield
        }
      }
      omp.terminator
    } {omp.combined}
    llvm.return
  }
}

// CHECK-LABEL: define void @test(
// CHECK-SAME:     ptr %[[ARG0:.*]], i64 %[[ARG1:.*]]) {
// CHECK-NEXT:   %[[STRUCT_ARG:.*]] = alloca { i64, i64, i64, ptr }, align 8
// CHECK-NEXT:   %[[BOX_WITH_PTR:.*]] = insertvalue { ptr, i64 } undef, ptr %[[ARG0]], 0
// CHECK-NEXT:   %[[ORIG_BOX:.*]] = insertvalue { ptr, i64 } %[[BOX_WITH_PTR]], i64 %[[ARG1]], 1
// CHECK-NEXT:   br label %entry
// CHECK-EMPTY:
// CHECK-NEXT: entry:{{.*}}; preds = %[[ENTRY_PRED:.*]]
// CHECK-NEXT:   br label %omp.private.init
// CHECK-EMPTY:
// CHECK-NEXT: omp.private.init:{{.*}}; preds = %entry
// CHECK-NEXT:   %[[TASK_CONTEXT:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ { ptr, i64 } }, ptr null, i32 1) to i64))
// CHECK-NEXT:   %[[TASK_CONTEXT_BOX:.*]] = getelementptr { { ptr, i64 } }, ptr %[[TASK_CONTEXT]], i32 0, i32 0
// CHECK-NEXT:   %[[INIT_PTR:.*]] = extractvalue { ptr, i64 } %[[ORIG_BOX]], 0
// CHECK-NEXT:   %[[INIT_LEN:.*]] = extractvalue { ptr, i64 } %[[ORIG_BOX]], 1
// CHECK-NEXT:   %[[INIT_BOX_WITH_PTR:.*]] = insertvalue { ptr, i64 } undef, ptr %[[INIT_PTR]], 0
// CHECK-NEXT:   %[[INIT_BOX:.*]] = insertvalue { ptr, i64 } %[[INIT_BOX_WITH_PTR]], i64 %[[INIT_LEN]], 1
// CHECK-NEXT:   store { ptr, i64 } %[[INIT_BOX]], ptr %[[TASK_CONTEXT_BOX]], align 8
// CHECK-NEXT:   %[[INIT_LOAD:.*]] = load { ptr, i64 }, ptr %[[TASK_CONTEXT_BOX]], align 8
// CHECK-NEXT:   br label %omp.private.copy
// CHECK-EMPTY:
// CHECK-NEXT: omp.private.copy:{{.*}}; preds = %omp.private.init
// CHECK-NEXT:   br label %omp.private.copy1
// CHECK-EMPTY:
// CHECK-NEXT: omp.private.copy1:{{.*}}; preds = %omp.private.copy
// CHECK-NEXT:   %[[COPY_LOAD:.*]] = load { ptr, i64 }, ptr %[[TASK_CONTEXT_BOX]], align 8
// CHECK-NEXT:   %[[COPY_PTR:.*]] = extractvalue { ptr, i64 } %[[ORIG_BOX]], 0
// CHECK-NEXT:   %[[COPY_LEN:.*]] = extractvalue { ptr, i64 } %[[ORIG_BOX]], 1
// CHECK-NEXT:   call void @copy_box(ptr %[[COPY_PTR]], i64 %[[COPY_LEN]])
// CHECK-NEXT:   br label %omp.taskloop.wrapper.start
// CHECK-EMPTY:
// CHECK-NEXT: omp.taskloop.wrapper.start:{{.*}}; preds = %omp.private.copy1
// CHECK-NEXT:   br label %codeRepl
// CHECK-EMPTY:
// CHECK-NEXT: codeRepl:{{.*}}; preds = %omp.taskloop.wrapper.start
// CHECK-NEXT:   %[[LB_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCT_ARG]], i32 0, i32 0
// CHECK-NEXT:   store i64 1, ptr %[[LB_ADDR]], align 4
// CHECK-NEXT:   %[[UB_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCT_ARG]], i32 0, i32 1
// CHECK-NEXT:   store i64 2, ptr %[[UB_ADDR]], align 4
// CHECK-NEXT:   %[[STEP_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCT_ARG]], i32 0, i32 2
// CHECK-NEXT:   store i64 1, ptr %[[STEP_ADDR]], align 4
// CHECK-NEXT:   %[[TASK_CONTEXT_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[STRUCT_ARG]], i32 0, i32 3
// CHECK-NEXT:   store ptr %[[TASK_CONTEXT]], ptr %[[TASK_CONTEXT_ADDR]], align 8
// CHECK-NEXT:   %[[THREAD_NUM:.*]] = call i32 @__kmpc_global_thread_num(ptr @[[IDENT:[0-9]+]])
// CHECK-NEXT:   call void @__kmpc_taskgroup(ptr @[[IDENT]], i32 %[[THREAD_NUM]])
// CHECK-NEXT:   %[[TASK:.*]] = call ptr @__kmpc_omp_task_alloc(ptr @[[IDENT]], i32 %[[THREAD_NUM]], i32 1, i64 40, i64 32, ptr @test..omp_par)
// CHECK-NEXT:   %[[TASK_DATA:.*]] = load ptr, ptr %[[TASK]], align 8
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[TASK_DATA]], ptr align 1 %[[STRUCT_ARG]], i64 32, i1 false)
// CHECK-NEXT:   %[[TASK_LB_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[TASK_DATA]], i32 0, i32 0
// CHECK-NEXT:   %[[TASK_UB_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[TASK_DATA]], i32 0, i32 1
// CHECK-NEXT:   %[[TASK_STEP_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[TASK_DATA]], i32 0, i32 2
// CHECK-NEXT:   %[[TASK_STEP:.*]] = load i64, ptr %[[TASK_STEP_ADDR]], align 4
// CHECK-NEXT:   call void @__kmpc_taskloop(ptr @[[IDENT]], i32 %[[THREAD_NUM]], ptr %[[TASK]], i32 1, ptr %[[TASK_LB_ADDR]], ptr %[[TASK_UB_ADDR]], i64 %[[TASK_STEP]], i32 1, i32 0, i64 0, ptr @omp_taskloop_dup)
// CHECK-NEXT:   call void @__kmpc_end_taskgroup(ptr @[[IDENT]], i32 %[[THREAD_NUM]])
// CHECK-NEXT:   br label %taskloop.exit
// CHECK-EMPTY:
// CHECK-NEXT: taskloop.exit:{{.*}}; preds = %codeRepl
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: define internal void @test..omp_par(i32 %[[GLOBAL_TID:.*]], ptr %[[TASK_ARG:.*]]) {
// CHECK-NEXT: taskloop.alloca:
// CHECK-NEXT:   %[[TASK_ARGS:.*]] = load ptr, ptr %[[TASK_ARG]], align 8
// CHECK-NEXT:   %[[PAR_LB_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[TASK_ARGS]], i32 0, i32 0
// CHECK-NEXT:   %[[PAR_LB:.*]] = load i64, ptr %[[PAR_LB_ADDR]], align 4
// CHECK-NEXT:   %[[PAR_UB_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[TASK_ARGS]], i32 0, i32 1
// CHECK-NEXT:   %[[PAR_UB:.*]] = load i64, ptr %[[PAR_UB_ADDR]], align 4
// CHECK-NEXT:   %[[PAR_STEP_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[TASK_ARGS]], i32 0, i32 2
// CHECK-NEXT:   %[[PAR_STEP:.*]] = load i64, ptr %[[PAR_STEP_ADDR]], align 4
// CHECK-NEXT:   %[[PAR_CONTEXT_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[TASK_ARGS]], i32 0, i32 3
// CHECK-NEXT:   %[[PAR_CONTEXT:.*]] = load ptr, ptr %[[PAR_CONTEXT_ADDR]], align 8, !align ![[ALIGN:[0-9]+]]
// CHECK-NEXT:   br label %taskloop.body
// CHECK-EMPTY:
// CHECK-NEXT: taskloop.body:{{.*}}; preds = %taskloop.alloca
// CHECK-NEXT:   %[[PAR_BOX_ADDR:.*]] = getelementptr { { ptr, i64 } }, ptr %[[PAR_CONTEXT]], i32 0, i32 0
// CHECK-NEXT:   %[[PAR_BOX:.*]] = load { ptr, i64 }, ptr %[[PAR_BOX_ADDR]], align 8
// CHECK-NEXT:   br label %omp.taskloop.context.region
// CHECK-EMPTY:
// CHECK-NEXT: omp.taskloop.context.region:{{.*}}; preds = %taskloop.body
// CHECK-NEXT:   br label %omp.taskloop.wrapper.region
// CHECK-EMPTY:
// CHECK-NEXT: omp.taskloop.wrapper.region:{{.*}}; preds = %omp.taskloop.context.region
// CHECK-NEXT:   br label %omp_loop.preheader
// CHECK-EMPTY:
// CHECK-NEXT: omp_loop.preheader:{{.*}}; preds = %omp.taskloop.wrapper.region
// CHECK-NEXT:   %[[RANGE:.*]] = sub i64 %[[PAR_UB]], %[[PAR_LB]]
// CHECK-NEXT:   %[[STEP_COUNT:.*]] = sdiv i64 %[[RANGE]], %[[PAR_STEP]]
// CHECK-NEXT:   %[[TRIP_COUNT:.*]] = add i64 %[[STEP_COUNT]], 1
// CHECK-NEXT:   %[[TRIP_COUNT_I32:.*]] = trunc i64 %[[TRIP_COUNT]] to i32
// CHECK-NEXT:   %[[LB_I32:.*]] = trunc i64 %[[PAR_LB]] to i32
// CHECK-NEXT:   br label %omp_loop.header
// CHECK-EMPTY:
// CHECK-NEXT: omp_loop.header:{{.*}}; preds = %omp_loop.inc, %omp_loop.preheader
// CHECK-NEXT:   %[[LOOP_IV:.*]] = phi i32 [ 0, %omp_loop.preheader ], [ %[[LOOP_NEXT:.*]], %omp_loop.inc ]
// CHECK-NEXT:   br label %omp_loop.cond
// CHECK-EMPTY:
// CHECK-NEXT: omp_loop.cond:{{.*}}; preds = %omp_loop.header
// CHECK-NEXT:   %[[LOOP_CMP:.*]] = icmp ult i32 %[[LOOP_IV]], %[[TRIP_COUNT_I32]]
// CHECK-NEXT:   br i1 %[[LOOP_CMP]], label %omp_loop.body, label %omp_loop.exit
// CHECK-EMPTY:
// CHECK-NEXT: omp_loop.exit:{{.*}}; preds = %omp_loop.cond
// CHECK-NEXT:   br label %omp_loop.after
// CHECK-EMPTY:
// CHECK-NEXT: omp_loop.after:{{.*}}; preds = %omp_loop.exit
// CHECK-NEXT:   br label %omp.region.cont2
// CHECK-EMPTY:
// CHECK-NEXT: omp.region.cont2:{{.*}}; preds = %omp_loop.after
// CHECK-NEXT:   br label %omp.region.cont
// CHECK-EMPTY:
// CHECK-NEXT: omp.region.cont:{{.*}}; preds = %omp.region.cont2
// CHECK-NEXT:   %[[DEALLOC_PTR:.*]] = extractvalue { ptr, i64 } %[[PAR_BOX]], 0
// CHECK-NEXT:   %[[DEALLOC_LEN:.*]] = extractvalue { ptr, i64 } %[[PAR_BOX]], 1
// CHECK-NEXT:   call void @dealloc_box(ptr %[[DEALLOC_PTR]], i64 %[[DEALLOC_LEN]])
// CHECK-NEXT:   tail call void @free(ptr %[[PAR_CONTEXT]])
// CHECK-NEXT:   br label %taskloop.exit.exitStub
// CHECK-EMPTY:
// CHECK-NEXT: omp_loop.body:{{.*}}; preds = %omp_loop.cond
// CHECK-NEXT:   %[[LOOP_OFFSET:.*]] = mul i32 %[[LOOP_IV]], 1
// CHECK-NEXT:   %[[LOGICAL_IV:.*]] = add i32 %[[LOOP_OFFSET]], %[[LB_I32]]
// CHECK-NEXT:   br label %omp.loop_nest.region
// CHECK-EMPTY:
// CHECK-NEXT: omp.loop_nest.region:{{.*}}; preds = %omp_loop.body
// CHECK-NEXT:   %[[BODY_PTR:.*]] = extractvalue { ptr, i64 } %[[PAR_BOX]], 0
// CHECK-NEXT:   %[[BODY_LEN:.*]] = extractvalue { ptr, i64 } %[[PAR_BOX]], 1
// CHECK-NEXT:   call void @use_body_box(ptr %[[BODY_PTR]], i64 %[[BODY_LEN]])
// CHECK-NEXT:   br label %omp.region.cont3
// CHECK-EMPTY:
// CHECK-NEXT: omp.region.cont3:{{.*}}; preds = %omp.loop_nest.region
// CHECK-NEXT:   br label %omp_loop.inc
// CHECK-EMPTY:
// CHECK-NEXT: omp_loop.inc:{{.*}}; preds = %omp.region.cont3
// CHECK-NEXT:   %[[LOOP_NEXT]] = add nuw i32 %[[LOOP_IV]], 1
// CHECK-NEXT:   br label %omp_loop.header
// CHECK-EMPTY:
// CHECK-NEXT: taskloop.exit.exitStub:{{.*}}; preds = %omp.region.cont
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

// CHECK-LABEL: define internal void @omp_taskloop_dup(
// CHECK-SAME:     ptr %[[DEST_TASK:.*]], ptr %[[SRC_TASK:.*]], i32 %[[LASTPRIVATE_FLAG:.*]]) {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %[[DEST_TASK_DATA:.*]] = getelementptr { %struct.kmp_task_ompbuilder_t, { i64, i64, i64, ptr } }, ptr %[[DEST_TASK]], i32 0, i32 1
// CHECK-NEXT:   %[[DEST_CONTEXT_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[DEST_TASK_DATA]], i32 0, i32 3
// CHECK-NEXT:   %[[SRC_TASK_DATA:.*]] = getelementptr { %struct.kmp_task_ompbuilder_t, { i64, i64, i64, ptr } }, ptr %[[SRC_TASK]], i32 0, i32 1
// CHECK-NEXT:   %[[SRC_CONTEXT_ADDR:.*]] = getelementptr { i64, i64, i64, ptr }, ptr %[[SRC_TASK_DATA]], i32 0, i32 3
// CHECK-NEXT:   %[[SRC_CONTEXT:.*]] = load ptr, ptr %[[SRC_CONTEXT_ADDR]], align 8
// CHECK-NEXT:   %[[DEST_CONTEXT:.*]] = tail call ptr @malloc(i64 ptrtoint (ptr getelementptr ({ { ptr, i64 } }, ptr null, i32 1) to i64))
// CHECK-NEXT:   store ptr %[[DEST_CONTEXT]], ptr %[[DEST_CONTEXT_ADDR]], align 8
// CHECK-NEXT:   %[[SRC_BOX_ADDR:.*]] = getelementptr { { ptr, i64 } }, ptr %[[SRC_CONTEXT]], i32 0, i32 0
// CHECK-NEXT:   %[[DEST_BOX_ADDR:.*]] = getelementptr { { ptr, i64 } }, ptr %[[DEST_CONTEXT]], i32 0, i32 0
// CHECK-NEXT:   %[[DUP_MOLD:.*]] = load { ptr, i64 }, ptr %[[SRC_BOX_ADDR]], align 8
// CHECK-NEXT:   %[[DUP_PTR:.*]] = extractvalue { ptr, i64 } %[[DUP_MOLD]], 0
// CHECK-NEXT:   %[[DUP_LEN:.*]] = extractvalue { ptr, i64 } %[[DUP_MOLD]], 1
// CHECK-NEXT:   %[[DUP_BOX_WITH_PTR:.*]] = insertvalue { ptr, i64 } undef, ptr %[[DUP_PTR]], 0
// CHECK-NEXT:   %[[DUP_BOX:.*]] = insertvalue { ptr, i64 } %[[DUP_BOX_WITH_PTR]], i64 %[[DUP_LEN]], 1
// CHECK-NEXT:   store { ptr, i64 } %[[DUP_BOX]], ptr %[[DEST_BOX_ADDR]], align 8
// CHECK-NEXT:   %[[DUP_PRIVATE:.*]] = load { ptr, i64 }, ptr %[[DEST_BOX_ADDR]], align 8
// CHECK-NEXT:   br label %omp.private.copy
// CHECK-EMPTY:
// CHECK-NEXT: omp.private.copy:{{.*}}; preds = %entry
// CHECK-NEXT:   %[[COPY_MOLD:.*]] = load { ptr, i64 }, ptr %[[SRC_BOX_ADDR]], align 8
// CHECK-NEXT:   %[[COPY_PRIVATE:.*]] = load { ptr, i64 }, ptr %[[DEST_BOX_ADDR]], align 8
// CHECK-NEXT:   %[[COPY_MOLD_PTR:.*]] = extractvalue { ptr, i64 } %[[COPY_MOLD]], 0
// CHECK-NEXT:   %[[COPY_MOLD_LEN:.*]] = extractvalue { ptr, i64 } %[[COPY_MOLD]], 1
// CHECK-NEXT:   call void @copy_box(ptr %[[COPY_MOLD_PTR]], i64 %[[COPY_MOLD_LEN]])
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
