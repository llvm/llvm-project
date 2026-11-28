// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s > %t.ll
// RUN: FileCheck %s --check-prefixes=CHECK,VIRTUAL < %t.ll
// RUN: FileCheck %s --check-prefix=GATE < %t.ll
// RUN: mlir-translate -mlir-to-llvmir -openmp-disable-virtual-collapse=false -split-input-file %s | FileCheck %s --check-prefixes=CHECK,VIRTUAL
// RUN: mlir-translate -mlir-to-llvmir -openmp-disable-virtual-collapse -split-input-file %s | FileCheck %s --implicit-check-not=omp_collapsed.init --check-prefixes=CHECK,DIVMOD

// -----

// The named clauses/bounds independently exclude Virtual.
// GATE-LABEL: define void @fallbacks(
// GATE-NOT: omp_collapsed.init
// GATE: {{^}}}
// CHECK-LABEL: define void @fallbacks(
// CHECK-DAG: call void @sink3(i32 0,
// CHECK-DAG: call void @sink(i32 1,
// CHECK-DAG: call void @sink(i32 2,
// CHECK-DAG: call void @sink(i32 3,
// CHECK-DAG: call void @sink(i32 4,
// CHECK-DAG: call void @sink(i32 5,
// CHECK-DAG: call void @sink(i32 6,
// CHECK-DAG: call void @sink(i32 7,
// CHECK-DAG: call void @sink(i32 8,
// CHECK: {{^}}}
omp.private {type = linear} @lin.i32 : i32

llvm.func @fallbacks(%n: i32, %linear: !llvm.ptr) {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %three = llvm.mlir.constant(3 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %five = llvm.mlir.constant(5 : i32) : i32
  %six = llvm.mlir.constant(6 : i32) : i32
  %seven = llvm.mlir.constant(7 : i32) : i32
  %eight = llvm.mlir.constant(8 : i32) : i32
  omp.wsloop {
    omp.loop_nest (%i, %j, %k) : i32 = (%zero, %zero, %zero) to (%two, %three, %four) step (%one, %one, %one) collapse(3) {
      llvm.call @sink3(%zero, %i, %j, %k) : (i32, i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop schedule(static = %one : i32) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %five) step (%one, %one) collapse(2) {
      llvm.call @sink(%one, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop schedule(dynamic) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %five) step (%one, %one) collapse(2) {
      llvm.call @sink(%two, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop schedule(guided) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %five) step (%one, %one) collapse(2) {
      llvm.call @sink(%three, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop schedule(runtime) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %five) step (%one, %one) collapse(2) {
      llvm.call @sink(%four, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop ordered(2) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %five) step (%one, %one) collapse(2) {
      llvm.call @sink(%five, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%n, %five) step (%one, %one) collapse(2) {
      llvm.call @sink(%six, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %n) step (%one, %one) collapse(2) {
      llvm.call @sink(%seven, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop linear(@lin.i32 %linear -> %linear_private : !llvm.ptr = %one : i32) linear_var_types([i32]) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %five)
        step (%one, %one) collapse(2) {
      llvm.call @sink(%eight, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  llvm.return
}
llvm.func @sink(i32, i32, i32)
llvm.func @sink3(i32, i32, i32, i32)

// -----

// The runtime third level and its collapsed SIMD nest stay inside each
// assigned pair. SIMD metadata belongs to the SIMD loop, not the row loops.
// CHECK-LABEL: define void @partial_simd(
// VIRTUAL: [[KCOUNT:%omp_loop.tripcount[^ ]*]] = select i1 {{.*}}, i32 0, i32 {{.*}}
// VIRTUAL: store i32 14, ptr %p.upperbound,
// VIRTUAL: omp_collapsed.decode:
// VIRTUAL: [[OUTER:omp_loop.header[0-9]*]]:
// VIRTUAL: [[INNER:omp_loop.header[0-9]+]]:
// VIRTUAL: [[KHEADER:omp_loop.header[0-9]+]]:
// VIRTUAL: [[K:%[^ ]+]] = phi i32 [ 0, {{.*}} ], [ [[KNEXT:%[^ ]+]], %[[KLATCH:omp_loop.inc[0-9]+]] ]
// VIRTUAL: icmp ult i32 [[K]], [[KCOUNT]]{{(,|$)}}
// VIRTUAL: [[KSTEP:%[^ ]+]] = mul i32 [[K]], 1{{(,|$)}}
// VIRTUAL: [[KSOURCE:%[^ ]+]] = add i32 [[KSTEP]], 0{{(,|$)}}
// VIRTUAL: call void @consume_prefix(i32 {{.*}}, i32 {{.*}}, i32 [[KSOURCE]])
// DIVMOD: omp_collapsed.header{{[0-9]*}}:
// CHECK: store i32 {{.*}}, ptr {{.*}}, {{.*}}!llvm.access.group ![[GROUP:[0-9]+]]
// CHECK: br label %{{[^ ,]+}}, !llvm.loop ![[LOOP:[0-9]+]]
// VIRTUAL: [[KLATCH]]:
// VIRTUAL: [[KNEXT]] = add nuw i32 [[K]], 1{{(,|$)}}
// VIRTUAL-NEXT: br label %[[KHEADER]]{{$}}
// VIRTUAL: br label %[[INNER]]{{$}}
// VIRTUAL: omp_collapsed.row.done:
// VIRTUAL: br label %[[OUTER]]{{$}}
// CHECK: call void @__kmpc_for_static_fini
// CHECK: ret void
// CHECK-DAG: ![[LOOP]] = distinct !{![[LOOP]], ![[PAR:[0-9]+]], ![[ENABLE:[0-9]+]]}
// CHECK-DAG: ![[PAR]] = !{!"llvm.loop.parallel_accesses", ![[GROUP]]}
// CHECK-DAG: ![[ENABLE]] = !{!"llvm.loop.vectorize.enable"}
llvm.func @partial_simd(%a: !llvm.ptr, %n: i32) {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %three = llvm.mlir.constant(3 : i32) : i32
  %five = llvm.mlir.constant(5 : i32) : i32
  omp.wsloop {
    omp.loop_nest (%i, %j, %k) : i32 = (%zero, %zero, %zero)
        to (%three, %five, %n) step (%one, %one, %one) collapse(2) {
      llvm.call @consume_prefix(%i, %j, %k) : (i32, i32, i32) -> ()
      omp.simd {
        omp.loop_nest (%l, %m) : i32 = (%zero, %zero) to (%two, %three)
            step (%one, %one) collapse(2) {
          llvm.store %l, %a : i32, !llvm.ptr
          omp.yield
        }
      }
      omp.yield
    }
  }
  llvm.return
}
llvm.func @consume_prefix(i32, i32, i32)

// -----

// The no-loop execution mode affects device scheduling, not host fallback.
// CHECK-LABEL: define {{.*}} @__omp_offloading_{{.*}}_target_no_loop_static_{{.*}}(
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_teams({{.*}}, ptr @[[TEAMS:[^ )]+]])
// CHECK: define internal void @[[TEAMS]](
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call({{.*}}, ptr @[[PARALLEL:[^ )]+]])
// CHECK: define internal void @[[PARALLEL]](
// CHECK: call void @__kmpc_for_static_init_4u
// VIRTUAL: omp_collapsed.init:
// VIRTUAL-DAG: omp_collapsed.decode:
// CHECK-DAG: call void @consume_no_loop(i32 {{.*}}, i32 {{.*}})
// CHECK-DAG: call void @__kmpc_for_static_fini
// CHECK: ret void
module attributes {omp.is_target_device = false,
                   omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @consume_no_loop(i32, i32)
  llvm.func @target_no_loop_static() {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %three = llvm.mlir.constant(3 : i32) : i32
    %five = llvm.mlir.constant(5 : i32) : i32
    omp.target kernel_type(spmd_no_loop)
        host_eval(%zero -> %lb, %one -> %step, %three -> %rows,
                  %five -> %cols : i32, i32, i32, i32) {
      omp.teams {
        omp.parallel {
          omp.wsloop {
            omp.loop_nest (%i, %j) : i32 = (%lb, %lb) to (%rows, %cols)
                step (%step, %step) collapse(2) {
              llvm.call @consume_no_loop(%i, %j) : (i32, i32) -> ()
              omp.yield
            }
          }
          omp.terminator
        } {omp.combined}
        omp.terminator
      } {omp.combined}
      omp.terminator
    } {omp.combined}
    llvm.return
  }
}

// -----

// Both cancellation exits reach static cleanup; normal rows retain their exit.
// CHECK-LABEL: define void @cancel_static(
// VIRTUAL: br i1 {{.*}}, label %omp_collapsed.decode, label %[[EXIT:[^ ]+]]{{(,|$)}}
// VIRTUAL: [[CANCEL:%[^ ]+]] = call i32 @__kmpc_cancel({{.*}}i32 2)
// VIRTUAL-NEXT: [[CANCEL_OFF:%[^ ]+]] = icmp eq i32 [[CANCEL]], 0{{(,|$)}}
// VIRTUAL-NEXT: br i1 [[CANCEL_OFF]], label %{{[^,]+}}, label %[[CANCEL_EXIT:[^ ]+]]{{(,|$)}}
// VIRTUAL: [[POINT:%[^ ]+]] = call i32 @__kmpc_cancellationpoint({{.*}}i32 2)
// VIRTUAL-NEXT: [[POINT_OFF:%[^ ]+]] = icmp eq i32 [[POINT]], 0{{(,|$)}}
// VIRTUAL-NEXT: br i1 [[POINT_OFF]], label %{{[^,]+}}, label %[[POINT_EXIT:[^ ]+]]{{(,|$)}}
// VIRTUAL: call void @consume_cancel(
// VIRTUAL: omp_collapsed.row.done:
// VIRTUAL-NEXT: br i1 %omp_collapsed.row.is_final, label %[[EXIT]], label %{{[^ ]+}}
// VIRTUAL: [[EXIT]]:
// VIRTUAL-NEXT: call void @__kmpc_for_static_fini
// VIRTUAL: call void @__kmpc_barrier
// VIRTUAL: [[CANCEL_EXIT]]:
// VIRTUAL-NEXT: br label %[[FINI:[^ ]+]]{{(,|$)}}
// VIRTUAL: [[FINI]]:
// VIRTUAL-NEXT: br label %[[EXIT]]{{$}}
// VIRTUAL: [[POINT_EXIT]]:
// VIRTUAL-NEXT: br label %[[FINI]]{{$}}
llvm.func @cancel_static() {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %eight = llvm.mlir.constant(8 : i32) : i32
  omp.wsloop {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%eight, %eight)
        step (%one, %one) collapse(2) {
      omp.cancel cancellation_construct_type(loop)
      omp.cancellation_point cancellation_construct_type(loop)
      llvm.call @consume_cancel(%i, %j) : (i32, i32) -> ()
      omp.yield
    }
  }
  llvm.return
}
llvm.func @consume_cancel(i32, i32)

// -----

// Tiling precedes collapse: decode into the 2x3 floor nest and retain
// the physical tile IVs used to reconstruct the original 3x5 source indices.
// CHECK-LABEL: define void @tiled_static(
// VIRTUAL: [[I_STEP:%[^ ]+]] = mul i32 [[I:%[^, ]+]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[I_SOURCE:%[^ ]+]] = add i32 [[I_STEP]], 0{{(,|$)}}
// VIRTUAL: store i32 5, ptr %p.upperbound{{(,|$)}}
// VIRTUAL: [[BEGIN:%[^ ]+]] = add i32 {{[^,]+}}, 0{{(,|$)}}
// VIRTUAL-NEXT: [[COUNT:%[^ ]+]] = add i32 {{[^,]+}}, 0{{(,|$)}}
// VIRTUAL: [[BEGIN_J:%[^ ]+]] = urem i32 [[BEGIN]], 3{{(,|$)}}
// VIRTUAL-NEXT: [[BEGIN_I:%[^ ]+]] = udiv i32 [[BEGIN]], 3{{(,|$)}}
// VIRTUAL-NEXT: [[REMAINING:%[^ ]+]] = add i32 [[COUNT]], [[BEGIN_J]]{{(,|$)}}
// VIRTUAL: [[FLOOR_I:%[^ ]+]] = phi i32 [ [[BEGIN_I]], %omp_collapsed.decode ],
// VIRTUAL-NEXT: [[ROW_REMAINING:%[^ ]+]] = phi i32 [ [[REMAINING]], %omp_collapsed.decode ],
// VIRTUAL-SAME: [ [[ROW_NEXT:%[^ ,]+]], %[[ROW_LATCH:[^ ]+]] ]
// VIRTUAL-NEXT: [[ROW_START:%[^ ]+]] = phi i32 [ [[BEGIN_J]], %omp_collapsed.decode ], [ 0,
// VIRTUAL: [[FINAL:%[^ ]+]] = icmp ule i32 [[ROW_REMAINING]], 3{{(,|$)}}
// VIRTUAL-NEXT: [[STOP:%[^ ]+]] = select i1 [[FINAL]], i32 [[ROW_REMAINING]], i32 3{{(,|$)}}
// VIRTUAL: [[FLOOR_J:%[^ ]+]] = phi i32 [ [[ROW_START]], %omp_collapsed.row.preheader ],
// VIRTUAL: icmp ult i32 [[FLOOR_J]], [[STOP]]{{(,|$)}}
// VIRTUAL: [[TILE_I:%[^ ]+]] = phi i32
// VIRTUAL: [[TILE_J:%[^ ]+]] = phi i32
// VIRTUAL: [[I_BASE:%[^ ]+]] = mul nuw i32 2, [[FLOOR_I]]{{(,|$)}}
// VIRTUAL-NEXT: [[I]] = add nuw i32 [[I_BASE]], [[TILE_I]]{{(,|$)}}
// VIRTUAL-NEXT: [[J_BASE:%[^ ]+]] = mul nuw i32 2, [[FLOOR_J]]{{(,|$)}}
// VIRTUAL-NEXT: [[J:%[^ ]+]] = add nuw i32 [[J_BASE]], [[TILE_J]]{{(,|$)}}
// VIRTUAL: [[J_STEP:%[^ ]+]] = mul i32 [[J]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[J_SOURCE:%[^ ]+]] = add i32 [[J_STEP]], 0{{(,|$)}}
// VIRTUAL: call void @consume_tile(i32 [[I_SOURCE]], i32 [[J_SOURCE]])
// VIRTUAL: omp_collapsed.row.done:
// VIRTUAL-NEXT: br i1 [[FINAL]], label %omp_floor0.exit, label %[[ROW_LATCH]]{{(,|$)}}
// VIRTUAL: [[ROW_LATCH]]:
// VIRTUAL-NEXT: {{%[^ ]+}} = add nuw i32 [[FLOOR_I]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[ROW_NEXT]] = sub nuw i32 [[ROW_REMAINING]], 3{{(,|$)}}
// CHECK: call void @__kmpc_for_static_fini
// CHECK: {{^}}}
llvm.func @tiled_static() {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %three = llvm.mlir.constant(3 : i32) : i32
  %five = llvm.mlir.constant(5 : i32) : i32
  omp.wsloop {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %five)
        step (%one, %one) collapse(2) tiles(2, 2) {
      llvm.call @consume_tile(%i, %j) : (i32, i32) -> ()
      omp.yield
    }
  }
  llvm.return
}
llvm.func @consume_tile(i32, i32)

// -----

// CHECK-LABEL: define internal void @wrapped_static..omp_par(
// CHECK: call void @__kmpc_dist_for_static_init_4u
// CHECK: {{^}}}
// GATE-LABEL: define internal void @wrapped_static..omp_par(
// GATE-NOT: omp_collapsed.init
// GATE: {{^}}}
llvm.func @wrapped_static() {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %three = llvm.mlir.constant(3 : i32) : i32
  %five = llvm.mlir.constant(5 : i32) : i32
  omp.teams {
    omp.parallel {
      omp.distribute {
        omp.wsloop {
          omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %five)
              step (%one, %one) collapse(2) {
            llvm.call @consume(%i, %j) : (i32, i32) -> ()
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
llvm.func @consume(i32, i32)

// -----

// Independent ranges: default nowait (0), static SIMD (1), monotonic with
// unit inner extent (2), nonmonotonic with zero outer (3), zero inner (4).
// CHECK-LABEL: define void @sequential_static(
// VIRTUAL: store i32 14, ptr [[UPPER0:%[^ ,]+]],
// CHECK: call void @__kmpc_for_static_init_4u({{.*}}i32 34,
// DIVMOD: urem i32
// DIVMOD: udiv i32
// VIRTUAL: [[LOWER0:%[^ ,]+]] = load i32, ptr {{[^,]+}},
// VIRTUAL-NEXT: [[UPPER_VALUE0:%[^ ,]+]] = load i32, ptr [[UPPER0]],
// VIRTUAL-NEXT: [[DELTA0:%[^ ,]+]] = sub i32 [[UPPER_VALUE0]], [[LOWER0]]{{(,|$)}}
// VIRTUAL-NEXT: [[COUNT0:%[^ ,]+]] = add i32 [[DELTA0]], 1{{(,|$)}}
// VIRTUAL: omp_collapsed.init{{[0-9]*}}:
// VIRTUAL: [[BEGIN0:%[^ ,]+]] = add i32 [[LOWER0]], 0{{(,|$)}}
// VIRTUAL-NEXT: [[RANGE_COUNT0:%[^ ,]+]] = add i32 [[COUNT0]], 0{{(,|$)}}
// VIRTUAL-NEXT: [[NONEMPTY0:%[^ ,]+]] = icmp ne i32 [[RANGE_COUNT0]], 0{{(,|$)}}
// VIRTUAL-NEXT: br i1 [[NONEMPTY0]], label %[[DECODE0:[^ ,]+]], label %[[EXIT0:[^ ,]+]]{{(,|$)}}
// VIRTUAL: [[DECODE0]]:
// VIRTUAL: [[REM0:%[^ ,]+]] = urem i32 [[BEGIN0]], 5{{(,|$)}}
// VIRTUAL-NEXT: [[QUOT0:%[^ ,]+]] = udiv i32 [[BEGIN0]], 5{{(,|$)}}
// VIRTUAL-NEXT: [[INITIAL0:%[^ ,]+]] = add i32 [[RANGE_COUNT0]], [[REM0]]{{(,|$)}}
// VIRTUAL: [[OUTER0:%[^ ,]+]] = phi i32 [ [[QUOT0]],
// VIRTUAL-NEXT: [[REMAIN0:%[^ ,]+]] = phi i32 [ [[INITIAL0]],
// VIRTUAL-SAME: %{{[^ ]+}} ], [ [[NEXT0:%[^ ,]+]], %[[LATCH0:[^ ]+]] ]
// VIRTUAL-NEXT: [[START0:%[^ ,]+]] = phi i32 [ [[REM0]],
// VIRTUAL: [[BOUND0:%[^ ,]+]] = icmp ult i32 [[OUTER0]], 3{{(,|$)}}
// VIRTUAL-NEXT: br i1 [[BOUND0]], label %omp_loop.body, label %[[EXIT0]]{{(,|$)}}
// VIRTUAL: [[IMUL0:%[^ ,]+]] = mul i32 [[OUTER0]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[I0:%[^ ,]+]] = add i32 [[IMUL0]], 0{{(,|$)}}
// VIRTUAL: [[FINAL0:%[^ ,]+]] = icmp ule i32 [[REMAIN0]], 5{{(,|$)}}
// VIRTUAL-NEXT: [[STOP0:%[^ ,]+]] = select i1 [[FINAL0]], i32 [[REMAIN0]], i32 5{{(,|$)}}
// VIRTUAL: [[INNER0:%[^ ,]+]] = phi i32 [ [[START0]],
// VIRTUAL: icmp ult i32 [[INNER0]], [[STOP0]]{{(,|$)}}
// VIRTUAL: [[JMUL0:%[^ ,]+]] = mul i32 [[INNER0]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[J0:%[^ ,]+]] = add i32 [[JMUL0]], 0{{(,|$)}}
// VIRTUAL: call void @consume_sequence(i32 0, i32 [[I0]], i32 [[J0]])
// VIRTUAL: [[LATCH0]]:
// VIRTUAL-NEXT: {{%[^ ]+}} = add nuw i32 [[OUTER0]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[NEXT0]] = sub nuw i32 [[REMAIN0]], 5{{(,|$)}}
// VIRTUAL: [[EXIT0]]:
// VIRTUAL-NEXT: call void @__kmpc_for_static_fini
// VIRTUAL-NOT: call void @__kmpc_barrier
// VIRTUAL: store i32 13, ptr [[UPPER1:%[^ ,]+]],
// VIRTUAL-NOT: call void @__kmpc_barrier
// CHECK: call void @__kmpc_for_static_init_4u({{.*}}i32 34,
// VIRTUAL: [[LOWER1:%[^ ,]+]] = load i32, ptr {{[^,]+}},
// VIRTUAL-NEXT: [[UPPER_VALUE1:%[^ ,]+]] = load i32, ptr [[UPPER1]],
// VIRTUAL-NEXT: [[DELTA1:%[^ ,]+]] = sub i32 [[UPPER_VALUE1]], [[LOWER1]]{{(,|$)}}
// VIRTUAL-NEXT: [[RANGE_COUNT1:%[^ ,]+]] = add i32 [[DELTA1]], 1{{(,|$)}}
// VIRTUAL: omp_collapsed.init{{[0-9]*}}:
// VIRTUAL: [[BEGIN1:%[^ ,]+]] = add i32 [[LOWER1]], 0{{(,|$)}}
// VIRTUAL-NEXT: [[COUNT1:%[^ ,]+]] = add i32 [[RANGE_COUNT1]], 0{{(,|$)}}
// VIRTUAL: [[REM1:%[^ ,]+]] = urem i32 [[BEGIN1]], 7{{(,|$)}}
// VIRTUAL-NEXT: [[QUOT1:%[^ ,]+]] = udiv i32 [[BEGIN1]], 7{{(,|$)}}
// VIRTUAL-NEXT: add i32 [[COUNT1]], [[REM1]]{{(,|$)}}
// VIRTUAL: [[OUTER1:%[^ ,]+]] = phi i32 [ [[QUOT1]],
// VIRTUAL: [[START1:%[^ ,]+]] = phi i32 [ [[REM1]],
// VIRTUAL: [[IMUL1:%[^ ,]+]] = mul i32 [[OUTER1]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[I1:%[^ ,]+]] = add i32 [[IMUL1]], 0{{(,|$)}}
// VIRTUAL: [[INNER1:%[^ ,]+]] = phi i32 [ [[START1]],
// VIRTUAL: [[JMUL1:%[^ ,]+]] = mul i32 [[INNER1]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[J1:%[^ ,]+]] = add i32 [[JMUL1]], 0{{(,|$)}}
// VIRTUAL: call void @consume_sequence(i32 1, i32 [[I1]], i32 [[J1]])
// VIRTUAL: call void @__kmpc_for_static_fini
// VIRTUAL: call void @__kmpc_barrier
// VIRTUAL: store i32 3, ptr [[UPPER2:%[^ ,]+]],
// CHECK: call void @__kmpc_for_static_init_4u({{.*}}i32 34,
// VIRTUAL: [[LOWER2:%[^ ,]+]] = load i32, ptr {{[^,]+}},
// VIRTUAL-NEXT: [[UPPER_VALUE2:%[^ ,]+]] = load i32, ptr [[UPPER2]],
// VIRTUAL-NEXT: [[DELTA2:%[^ ,]+]] = sub i32 [[UPPER_VALUE2]], [[LOWER2]]{{(,|$)}}
// VIRTUAL-NEXT: [[COUNT2:%[^ ,]+]] = add i32 [[DELTA2]], 1{{(,|$)}}
// VIRTUAL: omp_collapsed.init{{[0-9]*}}:
// VIRTUAL: [[BEGIN2:%[^ ,]+]] = add i32 [[LOWER2]], 0{{(,|$)}}
// VIRTUAL-NEXT: [[RANGE_COUNT2:%[^ ,]+]] = add i32 [[COUNT2]], 0{{(,|$)}}
// VIRTUAL: [[REM2:%[^ ,]+]] = urem i32 [[BEGIN2]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[QUOT2:%[^ ,]+]] = udiv i32 [[BEGIN2]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[INITIAL2:%[^ ,]+]] = add i32 [[RANGE_COUNT2]], [[REM2]]{{(,|$)}}
// VIRTUAL: [[OUTER2:%[^ ,]+]] = phi i32 [ [[QUOT2]],
// VIRTUAL-NEXT: [[REMAIN2:%[^ ,]+]] = phi i32 [ [[INITIAL2]],
// VIRTUAL-NEXT: br label %{{[^ ,]+}}{{(,|$)}}
// VIRTUAL: [[FINAL2:%[^ ,]+]] = icmp ule i32 [[REMAIN2]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[STOP2:%[^ ,]+]] = select i1 [[FINAL2]], i32 [[REMAIN2]], i32 1{{(,|$)}}
// VIRTUAL: [[INNER2:%[^ ,]+]] = phi i32 [ 0,
// VIRTUAL: icmp ult i32 [[INNER2]], [[STOP2]]{{(,|$)}}
// CHECK: call void @consume_sequence(i32 2,
// VIRTUAL: br i1 [[FINAL2]], label %[[EXIT2:[^ ,]+]], label %{{[^ ,]+}}{{(,|$)}}
// VIRTUAL: [[EXIT2]]:
// CHECK: call void @__kmpc_for_static_fini
// VIRTUAL: store i32 -1, ptr %{{[^,]+}},
// CHECK: call void @__kmpc_for_static_init_4u({{.*}}i32 34,
// VIRTUAL: omp_collapsed.init{{[0-9]*}}:
// CHECK: call void @consume_sequence(i32 3,
// CHECK: call void @__kmpc_for_static_fini
// VIRTUAL: store i32 -1, ptr %{{[^,]+}},
// CHECK: call void @__kmpc_for_static_init_4u({{.*}}i32 34,
// VIRTUAL: omp_collapsed.init{{[0-9]*}}:
// CHECK: call void @consume_sequence(i32 4,
// CHECK: call void @__kmpc_for_static_fini
llvm.func @sequential_static() {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %three = llvm.mlir.constant(3 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %five = llvm.mlir.constant(5 : i32) : i32
  %seven = llvm.mlir.constant(7 : i32) : i32
  omp.wsloop nowait {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %five) step (%one, %one) collapse(2) {
      llvm.call @consume_sequence(%zero, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop schedule(static, simd) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%two, %seven) step (%one, %one) collapse(2) {
      llvm.call @consume_sequence(%one, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop schedule(static, monotonic) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%four, %one)
        step (%one, %one) collapse(2) {
      llvm.call @consume_sequence(%two, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop schedule(static, nonmonotonic) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%zero, %five)
        step (%one, %one) collapse(2) {
      llvm.call @consume_sequence(%three, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  omp.wsloop schedule(static) {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %zero)
        step (%one, %one) collapse(2) {
      llvm.call @consume_sequence(%four, %i, %j) : (i32, i32, i32) -> ()
      omp.yield
    }
  }
  llvm.return
}
llvm.func @consume_sequence(i32, i32, i32)

// -----

// Outlining preserves an outer source IV and the inner Virtual range.
// CHECK-LABEL: define void @nested_static(
// VIRTUAL: [[NEST_I_IV:%[^ ,]+]] = phi i32 [ %omp_collapsed.begin.0{{[0-9]*}},
// VIRTUAL: [[NEST_I_MUL:%[^ ,]+]] = mul i32 [[NEST_I_IV]], 1{{(,|$)}}
// VIRTUAL-NEXT: [[NEST_I:%[^ ,]+]] = add i32 [[NEST_I_MUL]], 0{{(,|$)}}
// VIRTUAL: store i32 [[NEST_I]], ptr [[NEST_SLOT:%[^ ,]+]],
// VIRTUAL: [[NEST_FIELD:%[^ ,]+]] = getelementptr { ptr }, ptr [[NEST_STRUCT:%[^ ,]+]], i32 0, i32 0{{(,|$)}}
// VIRTUAL-NEXT: store ptr [[NEST_SLOT]], ptr [[NEST_FIELD]],
// VIRTUAL: call void (ptr, i32, ptr, ...) @__kmpc_fork_call({{.*}}, ptr @[[NEST_INNER:[^ ,)]+]], ptr [[NEST_STRUCT]])
// CHECK: ret void
// VIRTUAL: define internal void @[[NEST_INNER]]({{.*}}ptr{{[^%]*}}[[NEST_ARGS:%[^ ,)]+]])
// VIRTUAL: [[NEST_ADDR:%[^ ,]+]] = getelementptr { ptr }, ptr [[NEST_ARGS]], i32 0, i32 0{{(,|$)}}
// VIRTUAL-NEXT: [[NEST_PTR:%[^ ,]+]] = load ptr, ptr [[NEST_ADDR]],
// VIRTUAL: [[NEST_I_VALUE:%[^ ,]+]] = load i32, ptr [[NEST_PTR]],
// VIRTUAL-DAG: {{^}}omp_collapsed.init{{[0-9]*}}:
// VIRTUAL-DAG: [[NEST_K_IV:%[^ ,]+]] = phi i32 [ %omp_collapsed.begin.0,
// VIRTUAL-DAG: [[NEST_K_MUL:%[^ ,]+]] = mul i32 [[NEST_K_IV]], 1{{(,|$)}}
// VIRTUAL-DAG: [[NEST_K:%[^ ,]+]] = add i32 [[NEST_K_MUL]], 0{{(,|$)}}
// VIRTUAL-DAG: [[NEST_L_IV:%[^ ,]+]] = phi i32 [ %omp_collapsed.row.start.1.0,
// VIRTUAL-DAG: [[NEST_L_MUL:%[^ ,]+]] = mul i32 [[NEST_L_IV]], 1{{(,|$)}}
// VIRTUAL-DAG: [[NEST_L:%[^ ,]+]] = add i32 [[NEST_L_MUL]], 0{{(,|$)}}
// VIRTUAL-DAG: call void @consume_nested(i32 [[NEST_I_VALUE]], i32 [[NEST_K]], i32 [[NEST_L]])
// CHECK: ret void
llvm.func @nested_static() {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %three = llvm.mlir.constant(3 : i32) : i32
  %five = llvm.mlir.constant(5 : i32) : i32
  omp.wsloop {
    omp.loop_nest (%i, %j) : i32 = (%zero, %zero) to (%three, %five) step (%one, %one) collapse(2) {
      omp.parallel {
        omp.wsloop {
          omp.loop_nest (%k, %l) : i32 = (%zero, %zero) to (%two, %three) step (%one, %one) collapse(2) {
            llvm.call @consume_nested(%i, %k, %l) : (i32, i32, i32) -> ()
            omp.yield
          }
        }
        omp.terminator
      }
      omp.yield
    }
  }
  llvm.return
}
llvm.func @consume_nested(i32, i32, i32)
