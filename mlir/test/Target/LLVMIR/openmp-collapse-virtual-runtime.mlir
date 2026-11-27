// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s --check-prefix=VIRTUAL
// RUN: mlir-translate -mlir-to-llvmir -split-input-file -openmp-disable-virtual-collapse %s | FileCheck %s --check-prefix=OFF --implicit-check-not=omp_collapsed.init

// Reuse the frozen counts in the scheduled total and loop bounds;
// decoding also uses the frozen inner count.
// VIRTUAL-LABEL: define void @runtime_static(
// VIRTUAL: [[RAW_OUTER:%omp_loop.tripcount]] = select i1
// VIRTUAL: [[RAW_INNER:%omp_loop.tripcount[0-9]+]] = select i1
// VIRTUAL: [[OUTER_COUNT:%[^ ,]+]] = freeze i32 [[RAW_OUTER]]{{$}}
// VIRTUAL-NEXT: [[INNER_COUNT:%[^ ,]+]] = freeze i32 [[RAW_INNER]]{{$}}
// VIRTUAL-NEXT: [[TOTAL:%[^ ,]+]] = mul nuw i32 [[OUTER_COUNT]], [[INNER_COUNT]]{{$}}
// VIRTUAL: [[UB:%[^ ,]+]] = sub i32 [[TOTAL]], 1{{$}}
// VIRTUAL: store i32 [[UB]], ptr %p.upperbound,
// VIRTUAL: call void @__kmpc_for_static_init_4u
// VIRTUAL: [[LB:%[^ ,]+]] = load i32, ptr %p.lowerbound,
// VIRTUAL: [[RUNTIME_UB:%[^ ,]+]] = load i32, ptr %p.upperbound,
// VIRTUAL: [[DIFF:%[^ ,]+]] = sub i32 [[RUNTIME_UB]], [[LB]]{{$}}
// VIRTUAL-NEXT: [[COUNT:%[^ ,]+]] = add i32 [[DIFF]], 1{{$}}
// VIRTUAL: omp_collapsed.init:
// VIRTUAL-NEXT: [[BEGIN:%[^ ,]+]] = add i32 [[LB]], 0{{$}}
// VIRTUAL-NEXT: [[RANGE_COUNT:%[^ ,]+]] = add i32 [[COUNT]], 0{{$}}
// VIRTUAL-NEXT: [[NONEMPTY:%[^ ,]+]] = icmp ne i32 [[RANGE_COUNT]], 0{{$}}
// VIRTUAL-NEXT: br i1 [[NONEMPTY]], label %omp_collapsed.decode, label %[[EXIT:[^ ,]+]]{{$}}
// VIRTUAL: omp_collapsed.decode:
// VIRTUAL-NEXT: [[START:%[^ ,]+]] = urem i32 [[BEGIN]], [[INNER_COUNT]]{{$}}
// VIRTUAL-NEXT: [[ROW:%[^ ,]+]] = udiv i32 [[BEGIN]], [[INNER_COUNT]]{{$}}
// VIRTUAL-NEXT: [[INITIAL:%[^ ,]+]] = add i32 [[RANGE_COUNT]], [[START]]{{$}}
// VIRTUAL: [[OUTER:%[^ ,]+]] = phi i32 [ [[ROW]],
// VIRTUAL-NEXT: [[REMAINING:%[^ ,]+]] = phi i32 [ [[INITIAL]],
// VIRTUAL-NEXT: [[ROW_START:%[^ ,]+]] = phi i32 [ [[START]],
// VIRTUAL: icmp ult i32 [[OUTER]], [[OUTER_COUNT]]{{$}}
// VIRTUAL: [[I_MUL:%[^ ,]+]] = mul i32 [[OUTER]], 1{{$}}
// VIRTUAL-NEXT: [[I:%[^ ,]+]] = add i32 [[I_MUL]], 0{{$}}
// VIRTUAL: [[FINAL:%[^ ,]+]] = icmp ule i32 [[REMAINING]], [[INNER_COUNT]]{{$}}
// VIRTUAL-NEXT: [[STOP:%[^ ,]+]] = select i1 [[FINAL]], i32 [[REMAINING]], i32 [[INNER_COUNT]]{{$}}
// VIRTUAL: [[INNER:%[^ ,]+]] = phi i32 [ [[ROW_START]], %omp_collapsed.row.preheader ]
// VIRTUAL: icmp ult i32 [[INNER]], [[STOP]]{{$}}
// VIRTUAL: [[J_MUL:%[^ ,]+]] = mul i32 [[INNER]], 1{{$}}
// VIRTUAL-NEXT: [[J:%[^ ,]+]] = add i32 [[J_MUL]], 0{{$}}
// VIRTUAL: call void @consume(i32 [[I]], i32 [[J]])
// VIRTUAL: omp_collapsed.row.done:
// VIRTUAL-NEXT: br i1 [[FINAL]], label %[[EXIT]], label %{{[^ ,]+}}{{$}}
// VIRTUAL: sub nuw i32 [[REMAINING]], [[INNER_COUNT]]{{$}}
// VIRTUAL: [[EXIT]]:
// VIRTUAL: call void @__kmpc_for_static_fini
// VIRTUAL: ret void
// OFF-LABEL: define void @runtime_static(
// OFF: call void @__kmpc_for_static_init_4u
// OFF: omp_collapsed.body:
// OFF-NEXT: [[FLAT:%[^ ,]+]] = add i32 %omp_collapsed.iv, %{{[^ ,]+}}{{$}}
// OFF-NEXT: {{.*}} = urem i32 [[FLAT]], [[RADIX:%[^ ,]+]]{{$}}
// OFF-NEXT: {{.*}} = udiv i32 [[FLAT]], [[RADIX]]{{$}}
// OFF: call void @consume
// OFF: call void @__kmpc_for_static_fini
// OFF: ret void
llvm.func @runtime_static(%n: i32, %m: i32) {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  omp.wsloop schedule(static) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%n, %m)
        step (%one, %one) collapse(2) {
      llvm.call @consume(%i, %j) : (i32, i32) -> ()
      omp.yield
    }
  }
  llvm.return
}
llvm.func @consume(i32, i32)

// Schedule floor iterations while each tile keeps its partial source extent.
// VIRTUAL-LABEL: define void @runtime_tiled(
// VIRTUAL: [[REM_I:%[^ ,]+]] = urem i32 %{{[^ ,]+}}, 2{{$}}
// VIRTUAL: [[REM_J:%[^ ,]+]] = urem i32 %{{[^ ,]+}}, 2{{$}}
// VIRTUAL: [[OUTER:%[^ ,]+]] = freeze i32 %omp_floor0.tripcount{{$}}
// VIRTUAL-NEXT: [[INNER:%[^ ,]+]] = freeze i32 %omp_floor1.tripcount{{$}}
// VIRTUAL-NEXT: [[TOTAL:%[^ ,]+]] = mul nuw i32 [[OUTER]], [[INNER]]{{$}}
// VIRTUAL: sub i32 [[TOTAL]], 1{{$}}
// VIRTUAL: call void @__kmpc_for_static_init_4u
// VIRTUAL: omp_collapsed.decode:
// VIRTUAL-NEXT: [[START:%[^ ,]+]] = urem i32 %omp_collapsed.range.begin, [[INNER]]{{$}}
// VIRTUAL-NEXT: [[ROW:%[^ ,]+]] = udiv i32 %omp_collapsed.range.begin, [[INNER]]{{$}}
// VIRTUAL: [[FLOOR_I:%[^ ,]+]] = phi i32 [ [[ROW]], %omp_collapsed.decode ]
// VIRTUAL: [[ROW_START:%[^ ,]+]] = phi i32 [ [[START]], %omp_collapsed.decode ]
// VIRTUAL: icmp ult i32 [[FLOOR_I]], [[OUTER]]{{$}}
// VIRTUAL: phi i32 [ [[ROW_START]], %omp_collapsed.row.preheader ]
// VIRTUAL: [[STOP_I:%[^ ,]+]] = select i1 {{.*}}, i32 [[REM_I]], i32 2{{$}}
// VIRTUAL: [[STOP_J:%[^ ,]+]] = select i1 {{.*}}, i32 [[REM_J]], i32 2{{$}}
// VIRTUAL: icmp ult i32 %omp_tile0.iv, [[STOP_I]]{{$}}
// VIRTUAL: icmp ult i32 %omp_tile1.iv, [[STOP_J]]{{$}}
// VIRTUAL: call void @consume(i32 {{.*}}, i32 {{.*}})
// VIRTUAL: call void @__kmpc_for_static_fini
// OFF-LABEL: define void @runtime_tiled(
// OFF: omp_collapsed.body:
// OFF: urem i32
// OFF: udiv i32
// OFF: call void @consume
llvm.func @runtime_tiled(%n: i32, %m: i32) {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  omp.wsloop schedule(static) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%n, %m)
        step (%one, %one) collapse(2) tiles(2, 2) {
      llvm.call @consume(%i, %j) : (i32, i32) -> ()
      omp.yield
    }
  }
  llvm.return
}

// Keep runtime counts and their freezes in the outlined parallel body.
// VIRTUAL-LABEL: define void @runtime_parallel(
// VIRTUAL: call void {{.*}}@__kmpc_fork_call(
// VIRTUAL-LABEL: define internal void @runtime_parallel
// VIRTUAL: [[RAW_OUTER:%omp_loop.tripcount]] = select i1
// VIRTUAL: [[RAW_INNER:%omp_loop.tripcount[0-9]+]] = select i1
// VIRTUAL: [[OUTER:%[^ ,]+]] = freeze i32 [[RAW_OUTER]]{{$}}
// VIRTUAL-NEXT: [[INNER:%[^ ,]+]] = freeze i32 [[RAW_INNER]]{{$}}
// VIRTUAL-NEXT: [[TOTAL:%[^ ,]+]] = mul nuw i32 [[OUTER]], [[INNER]]{{$}}
// VIRTUAL: sub i32 [[TOTAL]], 1{{$}}
// VIRTUAL: call void @__kmpc_for_static_init_4u
// VIRTUAL: call void @__kmpc_for_static_fini
// VIRTUAL: omp_collapsed.decode:
// VIRTUAL-NEXT: urem i32 %omp_collapsed.range.begin, [[INNER]]{{$}}
// VIRTUAL-NEXT: [[ROW:%[^ ,]+]] = udiv i32 %omp_collapsed.range.begin, [[INNER]]{{$}}
// VIRTUAL: [[IV:%[^ ,]+]] = phi i32 [ [[ROW]], %omp_collapsed.decode ]
// VIRTUAL: icmp ult i32 [[IV]], [[OUTER]]{{$}}
// VIRTUAL: call void @consume
// OFF-LABEL: define void @runtime_parallel(
// OFF: call void {{.*}}@__kmpc_fork_call(
// OFF-LABEL: define internal void @runtime_parallel
// OFF: call void @__kmpc_for_static_init_4u
// OFF: omp_collapsed.body:
// OFF: urem i32
// OFF: udiv i32
// OFF: call void @consume
llvm.func @runtime_parallel(%n: i32, %m: i32) {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.wsloop schedule(static) {
      omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%n, %m)
          step (%one, %one) collapse(2) {
        llvm.call @consume(%i, %j) : (i32, i32) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// -----

// Block-argument PHIs get their incoming values after the loop is translated.
// Reuse one frozen inclusive count in scheduling and the outer comparison.
// Keep nsw only for the constant inner count.
// VIRTUAL-LABEL: define void @pending_bound(
// VIRTUAL: phi i32
// VIRTUAL: [[RAW_COUNT:%[^ ,]+]] = select i1 {{.*}}, i32 0, i32 {{.*}}
// VIRTUAL: [[STABLE_COUNT:%[^ ,]+]] = freeze i32 [[RAW_COUNT]]{{$}}
// VIRTUAL-NEXT: [[TOTAL:%[^ ,]+]] = mul nuw i32 [[STABLE_COUNT]], 5{{$}}
// VIRTUAL: sub i32 [[TOTAL]], 1{{$}}
// VIRTUAL: call void @__kmpc_for_static_init_4u
// VIRTUAL: [[OUTER:%[^ ,]+]] = phi i32
// VIRTUAL: icmp ult i32 [[OUTER]], [[STABLE_COUNT]]{{$}}
// VIRTUAL: [[INNER_IV:%[^ ,]+]] = phi i32
// VIRTUAL: call void @consume
// VIRTUAL: add nuw nsw i32 [[INNER_IV]], 1{{$}}
// VIRTUAL: add nuw i32 [[OUTER]], 1{{$}}
// OFF-LABEL: define void @pending_bound(
// OFF: call void @__kmpc_for_static_init_4u
// OFF: call void @consume
module attributes {omp.integer_wrap_around = #omp.integer_wrap_around<integer_wrap_around = false>} {
  llvm.func @pending_bound(%which: i1, %n: i32, %m: i32) {
    %one = llvm.mlir.constant(1 : i32) : i32
    %three = llvm.mlir.constant(3 : i32) : i32
    %five = llvm.mlir.constant(5 : i32) : i32
    llvm.cond_br %which, ^left, ^right
  ^left:
    %masked_n = llvm.and %n, %three : i32
    llvm.br ^merge(%masked_n : i32)
  ^right:
    %masked_m = llvm.and %m, %three : i32
    llvm.br ^merge(%masked_m : i32)
  ^merge(%bound: i32):
    omp.wsloop schedule(static) {
      omp.loop_nest (%i, %j) : i32 = (%one, %one) to (%bound, %five)
          inclusive step (%one, %one) collapse(2) {
        llvm.call @consume(%i, %j) : (i32, i32) -> ()
        omp.yield
      }
    }
    llvm.return
  }
  llvm.func @consume(i32, i32)
}
