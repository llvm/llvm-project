// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Single scalar reduction on omp.taskloop.context. The lowering must:
//   1. Emit an implicit __kmpc_taskgroup in the encountering function (since
//      the user did not write nogroup);
//   2. Build a kmp_taskred_input_t descriptor array and call
//      __kmpc_taskred_init, capturing the returned descriptor handle;
//   3. Pass NoGroup=true to OpenMPIRBuilder::createTaskloop so that it does not
//      emit a second (implicit) taskgroup around the taskloop;
//   4. Inside the outlined task body, call __kmpc_global_thread_num to obtain
//      the executing thread's gtid, then look up the per-task private storage
//      via __kmpc_task_reduction_get_th_data(gtid, redDesc, orig);
//   5. Close the implicit taskgroup with __kmpc_end_taskgroup.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @taskloop_reduction_single(%x : !llvm.ptr, %lb : i32, %ub : i32, %step : i32) {
  omp.taskloop.context reduction(@add_i32 %x -> %prv : !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        %v = llvm.load %prv : !llvm.ptr -> i32
        %s = llvm.add %v, %iv : i32
        llvm.store %s, %prv : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  } {omp.combined}
  llvm.return
}

// CHECK: %kmp_taskred_input_t = type { ptr, ptr, i64, ptr, ptr, ptr, i32 }

// Encountering function emits taskgroup + descriptor + taskred_init.
// CHECK-LABEL: define void @taskloop_reduction_single(
// CHECK-SAME:    ptr %[[X:[^,]+]],
// CHECK:         %[[ARR:.+]] = alloca [1 x %kmp_taskred_input_t]
// CHECK:         call void @__kmpc_taskgroup(
// CHECK:         %[[ELEM:.+]] = getelementptr inbounds [1 x %kmp_taskred_input_t], ptr %[[ARR]], i32 0, i32 0
// CHECK:         %[[SHAR:.+]] = getelementptr {{.+}} %kmp_taskred_input_t, ptr %[[ELEM]], i32 0, i32 0
// CHECK:         store ptr %[[X]], ptr %[[SHAR]]
// CHECK:         store ptr @__omp_taskloop_taskred_add_i32.red.init
// CHECK:         store ptr @__omp_taskloop_taskred_add_i32.red.comb
// CHECK:         %[[DESC:.+]] = call ptr @__kmpc_taskred_init(i32 %{{.+}}, i32 1, ptr %[[ARR]])
// The returned descriptor is stored into the structArg captured by
// __kmpc_omp_task_alloc so the outlined task body can load it back.
// CHECK:         store ptr %[[DESC]], ptr %{{.+}}
// We manually opened a taskgroup above, so the reduction lowering passes
// NoGroup=true to createTaskloop. The __kmpc_taskloop "nogroup" argument is not
// a reliable witness for this (OpenMPIRBuilder hardcodes that operand to 1);
// instead assert that no SECOND implicit taskgroup pair wraps the taskloop.
// CHECK-NOT:     call void @__kmpc_taskgroup(
// CHECK-NOT:     call void @__kmpc_end_taskgroup(
// CHECK:         call void @__kmpc_taskloop(
// CHECK-NOT:     call void @__kmpc_taskgroup(
// CHECK:         call void @__kmpc_end_taskgroup(

// Outlined task body looks up per-task storage via the runtime, passing the
// reloaded descriptor (not null) as the second argument.
// CHECK-LABEL: define internal void @taskloop_reduction_single..omp_par(
// CHECK:         %[[BODY_DESC:.+]] = load ptr, ptr %gep_.taskred.desc
// CHECK:         %[[BODY_ORIG:.+]] = load ptr, ptr %gep_,
// CHECK:         %[[BODY_GTID:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[PRIV:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[BODY_GTID]], ptr %[[BODY_DESC]], ptr %[[BODY_ORIG]])
// CHECK:         load i32, ptr %[[PRIV]]
// CHECK:         store i32 %{{.+}}, ptr %[[PRIV]]

// -----

// Multiple reductions: each entry in the descriptor array gets distinct
// init / combiner helpers and the body issues one
// __kmpc_task_reduction_get_th_data per reduction.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

omp.declare_reduction @mul_i64 : i64
init {
^bb0(%arg0: i64):
  %c1 = llvm.mlir.constant(1 : i64) : i64
  omp.yield(%c1 : i64)
}
combiner {
^bb0(%arg0: i64, %arg1: i64):
  %p = llvm.mul %arg0, %arg1 : i64
  omp.yield(%p : i64)
}

llvm.func @taskloop_reduction_multi(%x : !llvm.ptr, %y : !llvm.ptr, %lb : i32, %ub : i32, %step : i32) {
  omp.taskloop.context reduction(@add_i32 %x -> %a, @mul_i64 %y -> %b : !llvm.ptr, !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        %va = llvm.load %a : !llvm.ptr -> i32
        %vai = llvm.add %va, %iv : i32
        llvm.store %vai, %a : i32, !llvm.ptr
        %vb = llvm.load %b : !llvm.ptr -> i64
        %iv64 = llvm.sext %iv : i32 to i64
        %vbi = llvm.mul %vb, %iv64 : i64
        llvm.store %vbi, %b : i64, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  } {omp.combined}
  llvm.return
}

// CHECK-LABEL: define void @taskloop_reduction_multi(
// CHECK:         %[[ARR2:.+]] = alloca [2 x %kmp_taskred_input_t]
// CHECK:         call void @__kmpc_taskgroup(
// CHECK:         store i64 4
// CHECK:         store ptr @__omp_taskloop_taskred_add_i32.red.init
// CHECK:         store ptr @__omp_taskloop_taskred_add_i32.red.comb
// CHECK:         store i64 8
// CHECK:         store ptr @__omp_taskloop_taskred_mul_i64.red.init
// CHECK:         store ptr @__omp_taskloop_taskred_mul_i64.red.comb
// CHECK:         %[[DESC2:.+]] = call ptr @__kmpc_taskred_init(i32 %{{.+}}, i32 2, ptr %[[ARR2]])
// The descriptor is captured into structArg so the outlined task can reload it.
// CHECK:         store ptr %[[DESC2]], ptr %{{.+}}
// As in the single-reduction case, NoGroup is asserted indirectly rather than
// via the hardcoded __kmpc_taskloop "nogroup" argument: no second implicit
// taskgroup pair may wrap the taskloop.
// CHECK-NOT:     call void @__kmpc_taskgroup(
// CHECK-NOT:     call void @__kmpc_end_taskgroup(
// CHECK:         call void @__kmpc_taskloop(
// CHECK-NOT:     call void @__kmpc_taskgroup(
// CHECK:         call void @__kmpc_end_taskgroup(

// CHECK-LABEL: define internal void @taskloop_reduction_multi..omp_par(
// CHECK:         %[[BODY_GTID2:.+]] = call i32 @__kmpc_global_thread_num(
// Both get_th_data calls share the same body gtid; the descriptor argument
// must be a reloaded SSA value (not null).
// CHECK:         call ptr @__kmpc_task_reduction_get_th_data(i32 %[[BODY_GTID2]], ptr %{{[^,]+}}, ptr %{{.+}})
// CHECK:         call ptr @__kmpc_task_reduction_get_th_data(i32 %[[BODY_GTID2]], ptr %{{[^,]+}}, ptr %{{.+}})

// -----

// in_reduction on omp.taskloop.context nested inside an outer taskgroup
// task_reduction. The taskloop has no reduction clause of its own, so it must
// NOT build a taskred descriptor: the single __kmpc_taskred_init belongs to the
// outer task_reduction. The user did not write nogroup, so the taskloop runs
// with NoGroup=false and createTaskloop opens the taskloop's own implicit
// taskgroup in addition to the outer one. The outlined body's get_th_data call
// passes a NULL descriptor so the runtime walks up to the enclosing taskgroup.

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @taskloop_inreduction(%x : !llvm.ptr, %lb : i32, %ub : i32, %step : i32) {
  omp.taskgroup task_reduction(@add_i32 %x -> %tg : !llvm.ptr) {
    omp.taskloop.context in_reduction(@add_i32 %x -> %prv : !llvm.ptr) {
      omp.taskloop.wrapper {
        omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
          %v = llvm.load %prv : !llvm.ptr -> i32
          %s = llvm.add %v, %iv : i32
          llvm.store %s, %prv : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.terminator
    } {omp.combined}
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @taskloop_inreduction(
// The outer taskgroup task_reduction opens and builds the one descriptor.
// CHECK:         call void @__kmpc_taskgroup(
// CHECK:         call ptr @__kmpc_taskred_init(
// The taskloop carries only an in_reduction clause, so it must NOT build its
// own reduction descriptor.
// CHECK-NOT:     call ptr @__kmpc_taskred_init(
// No nogroup was written, so createTaskloop runs with NoGroup=false and opens
// the taskloop's own implicit taskgroup (separate from the outer one above).
// CHECK:         call void @__kmpc_taskgroup(
// CHECK:         call void @__kmpc_taskloop(
// CHECK:         call void @__kmpc_end_taskgroup(

// In the outlined taskloop task body, the in_reduction lookup passes NULL
// as the descriptor argument so the runtime walks up enclosing taskgroups.
// CHECK-LABEL: define internal void @taskloop_inreduction..omp_par(
// CHECK:         call i32 @__kmpc_global_thread_num(
// CHECK:         call ptr @__kmpc_task_reduction_get_th_data(i32 %{{.+}}, ptr null, ptr %{{.+}})

// -----

// nogroup + in_reduction: the user wrote `nogroup` on the taskloop and only an
// in_reduction clause, so the translator must NOT open an implicit taskgroup
// and must NOT build a taskred descriptor for the taskloop itself, and the
// outlined body must look up the participant with a NULL descriptor so the
// runtime walks up. NoGroup is witnessed below by the absence of any taskgroup
// pair around the taskloop, not by the __kmpc_taskloop "nogroup" operand
// (OpenMPIRBuilder hardcodes that operand to 1 regardless of NoGroup).

omp.declare_reduction @add_i32 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @taskloop_nogroup_inreduction(%x : !llvm.ptr, %lb : i32, %ub : i32, %step : i32) {
  omp.taskloop.context nogroup in_reduction(@add_i32 %x -> %prv : !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        %v = llvm.load %prv : !llvm.ptr -> i32
        %s = llvm.add %v, %iv : i32
        llvm.store %s, %prv : i32, !llvm.ptr
        omp.yield
      }
    }
    omp.terminator
  } {omp.combined}
  llvm.return
}

// Outer caller: NoGroup is proven by the absence of any taskgroup pair (and the
// absence of taskred_init) around the taskloop. The __kmpc_taskloop "nogroup"
// argument is not a reliable witness (OpenMPIRBuilder hardcodes that operand to
// 1), so __kmpc_taskloop is matched only as a plain call anchor here.
// CHECK-LABEL: define void @taskloop_nogroup_inreduction(
// CHECK-NOT:     call void @__kmpc_taskgroup(
// CHECK-NOT:     call ptr @__kmpc_taskred_init(
// CHECK-NOT:     call void @__kmpc_end_taskgroup(
// CHECK:         call void @__kmpc_taskloop(
// CHECK-NOT:     call void @__kmpc_taskgroup(
// CHECK-NOT:     call void @__kmpc_end_taskgroup(

// In the outlined task body, the in_reduction lookup uses a NULL descriptor.
// CHECK-LABEL: define internal void @taskloop_nogroup_inreduction..omp_par(
// CHECK:         call i32 @__kmpc_global_thread_num(
// CHECK:         call ptr @__kmpc_task_reduction_get_th_data(i32 %{{.+}}, ptr null, ptr %{{.+}})

// -----

// Non-default address-space taskloop reduction. The reduction item pointer is a
// `!llvm.ptr<1>`, but __kmpc_task_reduction_get_th_data is declared with a
// generic (default-address-space) `ptr` for the original item pointer and
// returns a generic `ptr`. The body-remapping path must therefore:
//   1. addrspacecast the reloaded original pointer to the generic address space
//      before the lookup (matching the descriptor normalization done for the
//      taskgroup setup), and pass that generic pointer to the runtime call;
//   2. addrspacecast the returned generic private pointer back to addrspace(1)
//      before mapping it to the block argument, so the body's loads/stores act
//      on the per-task private storage in the expected address space.

omp.declare_reduction @add_i32_as1 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @taskloop_reduction_as1(%x : !llvm.ptr<1>, %lb : i32, %ub : i32, %step : i32) {
  omp.taskloop.context reduction(@add_i32_as1 %x -> %prv : !llvm.ptr<1>) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
        %v = llvm.load %prv : !llvm.ptr<1> -> i32
        %s = llvm.add %v, %iv : i32
        llvm.store %s, %prv : i32, !llvm.ptr<1>
        omp.yield
      }
    }
    omp.terminator
  } {omp.combined}
  llvm.return
}

// The encountering function normalizes the addrspace(1) item pointer to generic
// for the taskred descriptor (taskgroup setup), then builds the descriptor.
// CHECK-LABEL: define void @taskloop_reduction_as1(
// CHECK-SAME:    ptr addrspace(1) %[[X:[^,]+]],
// CHECK:         %[[X_GEN:.+]] = addrspacecast ptr addrspace(1) %[[X]] to ptr
// CHECK:         call ptr @__kmpc_taskred_init(i32 %{{.+}}, i32 1, ptr %{{.+}})

// In the outlined task body the addrspace(1) original is normalized to generic
// before the (non-null descriptor) lookup, and the returned generic private
// pointer is cast back to addrspace(1) for the body's loads/stores.
// CHECK-LABEL: define internal void @taskloop_reduction_as1..omp_par(
// CHECK:         %[[BODY_GTID:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[ORIG_GEN:.+]] = addrspacecast ptr addrspace(1) %{{.+}} to ptr
// CHECK:         %[[PRIV:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[BODY_GTID]], ptr %{{.+}}, ptr %[[ORIG_GEN]])
// CHECK:         %[[PRIV_AS1:.+]] = addrspacecast ptr %[[PRIV]] to ptr addrspace(1)
// CHECK:         load i32, ptr addrspace(1) %[[PRIV_AS1]]
// CHECK:         store i32 %{{.+}}, ptr addrspace(1) %[[PRIV_AS1]]

// -----

// Non-default address-space taskloop in_reduction nested inside an outer
// taskgroup task_reduction. The taskloop carries only an in_reduction clause, so
// its body looks up the participant with a NULL descriptor; the addrspace(1)
// original pointer is still normalized to generic before the lookup and the
// returned generic private pointer is cast back to addrspace(1) for the body.

omp.declare_reduction @add_i32_as1 : i32
init {
^bb0(%arg0: i32):
  %c0 = llvm.mlir.constant(0 : i32) : i32
  omp.yield(%c0 : i32)
}
combiner {
^bb0(%arg0: i32, %arg1: i32):
  %s = llvm.add %arg0, %arg1 : i32
  omp.yield(%s : i32)
}

llvm.func @taskloop_inreduction_as1(%x : !llvm.ptr<1>, %lb : i32, %ub : i32, %step : i32) {
  omp.taskgroup task_reduction(@add_i32_as1 %x -> %tg : !llvm.ptr<1>) {
    omp.taskloop.context in_reduction(@add_i32_as1 %x -> %prv : !llvm.ptr<1>) {
      omp.taskloop.wrapper {
        omp.loop_nest (%iv) : i32 = (%lb) to (%ub) step (%step) {
          %v = llvm.load %prv : !llvm.ptr<1> -> i32
          %s = llvm.add %v, %iv : i32
          llvm.store %s, %prv : i32, !llvm.ptr<1>
          omp.yield
        }
      }
      omp.terminator
    } {omp.combined}
    omp.terminator
  }
  llvm.return
}

// In the outlined task body the addrspace(1) original is normalized to generic
// before the NULL-descriptor in_reduction lookup, and the returned generic
// private pointer is cast back to addrspace(1) for the body's loads/stores.
// CHECK-LABEL: define internal void @taskloop_inreduction_as1..omp_par(
// CHECK:         %[[BODY_GTID:.+]] = call i32 @__kmpc_global_thread_num(
// CHECK:         %[[ORIG_GEN:.+]] = addrspacecast ptr addrspace(1) %{{.+}} to ptr
// CHECK:         %[[PRIV:.+]] = call ptr @__kmpc_task_reduction_get_th_data(i32 %[[BODY_GTID]], ptr null, ptr %[[ORIG_GEN]])
// CHECK:         %[[PRIV_AS1:.+]] = addrspacecast ptr %[[PRIV]] to ptr addrspace(1)
// CHECK:         load i32, ptr addrspace(1) %[[PRIV_AS1]]
// CHECK:         store i32 %{{.+}}, ptr addrspace(1) %[[PRIV_AS1]]
