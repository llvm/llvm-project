// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

func.func @testenterdataop(%a: memref<f32>) -> () {
  %ifCond = arith.constant true
  %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  acc.enter_data if(%ifCond) dataOperands(%0 : memref<f32>)
  return
}

// CHECK: acc.enter_data dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testenterdataop(%a: memref<f32>) -> () {
  %ifCond = arith.constant false
  %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  acc.enter_data if(%ifCond) dataOperands(%0 : memref<f32>)
  return
}

// CHECK: func @testenterdataop
// CHECK-NOT: acc.enter_data

// -----

func.func @testexitdataop(%a: memref<f32>) -> () {
  %ifCond = arith.constant true
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.exit_data if(%ifCond) dataOperands(%0 : memref<f32>)
  acc.delete accPtr(%0 : memref<f32>)
  return
}

// CHECK: acc.exit_data dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testexitdataop(%a: memref<f32>) -> () {
  %ifCond = arith.constant false
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.exit_data if(%ifCond) dataOperands(%0 : memref<f32>)
  acc.delete accPtr(%0 : memref<f32>)
  return
}

// CHECK: func @testexitdataop
// CHECK-NOT: acc.exit_data

// -----

func.func @testupdateop(%a: memref<f32>) -> () {
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.update_host accPtr(%0 : memref<f32>) to varPtr(%a : memref<f32>)
  %ifCond = arith.constant true
  acc.update if(%ifCond) dataOperands(%0: memref<f32>)
  return
}

// CHECK: acc.update dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testupdateop(%a: memref<f32>) -> () {
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.update_host accPtr(%0 : memref<f32>) to varPtr(%a : memref<f32>)
  %ifCond = arith.constant false
  acc.update if(%ifCond) dataOperands(%0: memref<f32>)
  return
}

// CHECK: func @testupdateop
// CHECK-NOT: acc.update{{.$}}

// -----

func.func @testenterdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  acc.enter_data if(%ifCond) dataOperands(%0 : memref<f32>)
  return
}

// CHECK:  func @testenterdataop(%{{.*}}: memref<f32>, [[IFCOND:%.*]]: i1)
// CHECK:    acc.enter_data if(%{{.*}}) dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testexitdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.exit_data if(%ifCond) dataOperands(%0 : memref<f32>)
  acc.delete accPtr(%0 : memref<f32>)
  return
}

// CHECK: func @testexitdataop(%{{.*}}: memref<f32>, [[IFCOND:%.*]]: i1)
// CHECK:   acc.exit_data if(%{{.*}}) dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testupdateop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.update_host accPtr(%0 : memref<f32>) to varPtr(%a : memref<f32>)
  acc.update if(%ifCond) dataOperands(%0: memref<f32>)
  return
}

// CHECK:  func @testupdateop(%{{.*}}: memref<f32>, [[IFCOND:%.*]]: i1)
// CHECK:    acc.update if(%{{.*}}) dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testhostdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.use_device varPtr(%a : memref<f32>) -> memref<f32>
  %1 = arith.constant 1 : i32
  %2 = arith.constant 10 : i32
  %false = arith.constant false
  acc.host_data dataOperands(%0 : memref<f32>) if(%false) {
    acc.loop control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
    acc.loop control(%iv : i32) = (%1 : i32) to (%2 : i32) step (%1 : i32) {
      acc.yield
    } inclusiveUpperbound(array<i1: true>) independent
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @testhostdataop
// CHECK-NOT: acc.host_data
// CHECK: acc.loop
// CHECK: acc.yield
// CHECK: acc.loop
// CHECK: acc.yield

// -----

func.func @testhostdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.use_device varPtr(%a : memref<f32>) -> memref<f32>
  %true = arith.constant true
  acc.host_data dataOperands(%0 : memref<f32>) if(%true) {
  }
  return
}

// CHECK-LABEL: func.func @testhostdataop
// CHECK: acc.host_data dataOperands(%{{.*}} : memref<f32>) {

// -----

// A constant false condition removes the construct even when its region is
// empty.

func.func @testhostdataop_empty_region(%a: memref<f32>) -> () {
  %0 = acc.use_device varPtr(%a : memref<f32>) -> memref<f32>
  %false = arith.constant false
  acc.host_data dataOperands(%0 : memref<f32>) if(%false) {
  }
  return
}

// CHECK-LABEL: func.func @testhostdataop_empty_region
// CHECK-NOT: acc.host_data
// CHECK: return

// -----

func.func @update_no_op(%x : memref<i32>) {
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval : i32):
    acc.yield %xval : i32
  }
  return
}

// CHECK-LABEL: func.func @update_no_op
// CHECK-NOT: acc.atomic.update

// -----

func.func @update_write_op(%x : memref<i32>, %value: i32) {
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval : i32):
    acc.yield %value : i32
  }
  return
}

// CHECK-LABEL: func.func @update_write_op
// CHECK-SAME:            (%[[X:.+]]: memref<i32>, %[[VALUE:.+]]: i32)
// CHECK: acc.atomic.write %[[X]] = %[[VALUE]] : memref<i32>, i32
// CHECK-NOT: acc.atomic.update

// -----

func.func @update_normal(%x : memref<i32>, %value: i32) {
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval : i32):
    %newval = arith.addi %xval, %value : i32
    acc.yield %newval : i32
  }
  return
}

// CHECK-LABEL: func.func @update_normal
// CHECK: acc.atomic.update
// CHECK: arith.addi
// CHECK: acc.yield

// -----

func.func @update_unnecessary_computations(%x: memref<i32>) {
  %c0 = arith.constant 0 : i32
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = arith.addi %xval, %c0 : i32
    acc.yield %newval: i32
  }
  return
}

// CHECK-LABEL: func.func @update_unnecessary_computations
// CHECK-NOT: acc.atomic.update

// -----

func.func @update_unnecessary_computations(%x: memref<i32>) {
  %c0 = arith.constant 0 : i32
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = arith.muli %xval, %c0 : i32
    acc.yield %newval: i32
  }
  return
}

// CHECK-LABEL: func.func @update_unnecessary_computations
// CHECK-NOT: acc.atomic.update
// CHECK: acc.atomic.write

// -----

func.func @kernel_environment_canonicalization(%q1: i32, %q2: i32, %q3: i32) {
  // Empty kernel_environment (no wait) - should be removed
  acc.kernel_environment {
  }

  acc.kernel_environment wait(%q1, %q2 : i32, i32) {
  }

  acc.kernel_environment wait {
  }

  acc.kernel_environment wait(%q3 : i32) {
  }

  return
}

// CHECK-LABEL: func.func @kernel_environment_canonicalization
// CHECK-SAME: ([[Q1:%.*]]: i32, [[Q2:%.*]]: i32, [[Q3:%.*]]: i32)
// CHECK-NOT: acc.kernel_environment
// CHECK: acc.wait([[Q1]], [[Q2]] : i32, i32)
// CHECK: acc.wait{{ *$}}
// CHECK: acc.wait([[Q3]] : i32)
// CHECK: return

// -----

// The copy clause stays. The reduction operand takes its place in the operand
// list, ahead of the other variable. Uses of the copy operand inside the
// region are rewritten to the reduction operand.
// CHECK-LABEL: func.func @kernel_environment_reduction_operand
// CHECK-SAME: %[[VAR:.*]]: memref<i32>, %[[OTHER:.*]]: memref<i32>, %[[ASYNC:.*]]: i32
// CHECK: %[[COPY:.*]] = acc.copyin varPtr(%[[VAR]] : memref<i32>) -> memref<i32>
// CHECK: %[[OTHER_MAP:.*]] = acc.copyin varPtr(%[[OTHER]] : memref<i32>) -> memref<i32>
// CHECK: %[[RED:.*]] = acc.copyin varPtr(%[[VAR]] : memref<i32>) dataClause(acc_reduction) -> memref<i32>
// CHECK: acc.kernel_environment dataOperands(%[[RED]], %[[OTHER_MAP]] : memref<i32>, memref<i32>) async(%[[ASYNC]] : i32) {
// CHECK: memref.load %[[RED]]
// CHECK: memref.load %[[RED]]
// CHECK: memref.store %{{.*}}, %[[OTHER_MAP]]
// CHECK: acc.copyout accPtr(%[[COPY]] : memref<i32>) to varPtr(%[[VAR]] : memref<i32>)
// CHECK: acc.copyout accPtr(%[[RED]] : memref<i32>) to varPtr(%[[VAR]] : memref<i32>) dataClause(acc_reduction)
func.func @kernel_environment_reduction_operand(%var: memref<i32>, %other: memref<i32>, %async: i32) {
  %copy = acc.copyin varPtr(%var : memref<i32>) -> memref<i32>
  %other_map = acc.copyin varPtr(%other : memref<i32>) -> memref<i32>
  %red = acc.copyin varPtr(%var : memref<i32>) dataClause(acc_reduction) -> memref<i32>
  acc.kernel_environment dataOperands(%copy, %other_map, %red : memref<i32>, memref<i32>, memref<i32>) async(%async : i32) {
    %v = memref.load %copy[] : memref<i32>
    %w = memref.load %red[] : memref<i32>
    %sum = arith.addi %v, %w : i32
    memref.store %sum, %other_map[] : memref<i32>
  }
  acc.copyout accPtr(%copy : memref<i32>) to varPtr(%var : memref<i32>)
  acc.copyout accPtr(%red : memref<i32>) to varPtr(%var : memref<i32>) dataClause(acc_reduction)
  acc.delete accPtr(%other_map : memref<i32>)
  return
}

// -----

// Two clauses for one variable leave the earliest. A later clause for another
// variable keeps its place behind it.
// CHECK-LABEL: func.func @kernel_environment_duplicate_operand
// CHECK-SAME: %[[VAR:.*]]: memref<i32>, %[[OTHER:.*]]: memref<i32>
// CHECK: %[[FIRST:.*]] = acc.copyin varPtr(%[[VAR]] : memref<i32>) -> memref<i32>
// CHECK: %[[OTHER_MAP:.*]] = acc.copyin varPtr(%[[OTHER]] : memref<i32>) -> memref<i32>
// CHECK: %[[SECOND:.*]] = acc.copyin varPtr(%[[VAR]] : memref<i32>) -> memref<i32>
// CHECK: acc.kernel_environment dataOperands(%[[FIRST]], %[[OTHER_MAP]] : memref<i32>, memref<i32>) {
// CHECK: memref.load %[[FIRST]]
// CHECK: memref.store %{{.*}}, %[[OTHER_MAP]]
// CHECK: acc.delete accPtr(%[[SECOND]] : memref<i32>)
func.func @kernel_environment_duplicate_operand(%var: memref<i32>, %other: memref<i32>) {
  %first = acc.copyin varPtr(%var : memref<i32>) -> memref<i32>
  %other_map = acc.copyin varPtr(%other : memref<i32>) -> memref<i32>
  %second = acc.copyin varPtr(%var : memref<i32>) -> memref<i32>
  acc.kernel_environment dataOperands(%first, %other_map, %second : memref<i32>, memref<i32>, memref<i32>) {
    %v = memref.load %second[] : memref<i32>
    memref.store %v, %other_map[] : memref<i32>
  }
  acc.delete accPtr(%first : memref<i32>)
  acc.delete accPtr(%second : memref<i32>)
  acc.delete accPtr(%other_map : memref<i32>)
  return
}

// -----

// A different section of the same variable is a different operand.
// CHECK-LABEL: func.func @kernel_environment_different_bounds
// CHECK: %[[WHOLE_MAP:.*]] = acc.copyin varPtr(%{{.*}} : memref<10xi32>) bounds(%{{.*}}) -> memref<10xi32>
// CHECK: %[[SLICE_MAP:.*]] = acc.copyin varPtr(%{{.*}} : memref<10xi32>) bounds(%{{.*}}) -> memref<10xi32>
// CHECK: acc.kernel_environment dataOperands(%[[WHOLE_MAP]], %[[SLICE_MAP]] : memref<10xi32>, memref<10xi32>) {
// CHECK: memref.store %{{.*}}, %[[WHOLE_MAP]]
// CHECK: memref.store %{{.*}}, %[[SLICE_MAP]]
func.func @kernel_environment_different_bounds(%var: memref<10xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %whole = acc.bounds lowerbound(%c0 : index) upperbound(%c4 : index) extent(%c5 : index) stride(%c1 : index) startIdx(%c1 : index)
  %slice = acc.bounds lowerbound(%c1 : index) upperbound(%c4 : index) extent(%c4 : index) stride(%c1 : index) startIdx(%c1 : index)
  %whole_map = acc.copyin varPtr(%var : memref<10xi32>) bounds(%whole) -> memref<10xi32>
  %slice_map = acc.copyin varPtr(%var : memref<10xi32>) bounds(%slice) -> memref<10xi32>
  acc.kernel_environment dataOperands(%whole_map, %slice_map : memref<10xi32>, memref<10xi32>) {
    %c0_i32 = arith.constant 0 : i32
    memref.store %c0_i32, %whole_map[%c0] : memref<10xi32>
    memref.store %c0_i32, %slice_map[%c1] : memref<10xi32>
  }
  acc.delete accPtr(%whole_map : memref<10xi32>) bounds(%whole)
  acc.delete accPtr(%slice_map : memref<10xi32>) bounds(%slice)
  return
}

// -----

// The reduction acc.copyin takes the create's place. It is a copyin and also
// records the reduction.
// CHECK-LABEL: func.func @kernel_environment_create_and_reduction
// CHECK-SAME: %[[VAR:.*]]: memref<i32>
// CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[VAR]] : memref<i32>) -> memref<i32>
// CHECK: %[[RED:.*]] = acc.copyin varPtr(%[[VAR]] : memref<i32>) dataClause(acc_reduction) -> memref<i32>
// CHECK: acc.kernel_environment dataOperands(%[[RED]] : memref<i32>) {
// CHECK: memref.load %[[RED]]
// CHECK: memref.store %{{.*}}, %[[RED]]
// CHECK: acc.delete accPtr(%[[CREATE]] : memref<i32>) dataClause(acc_create)
func.func @kernel_environment_create_and_reduction(%var: memref<i32>) {
  %create = acc.create varPtr(%var : memref<i32>) -> memref<i32>
  %red = acc.copyin varPtr(%var : memref<i32>) dataClause(acc_reduction) -> memref<i32>
  acc.kernel_environment dataOperands(%create, %red : memref<i32>, memref<i32>) {
    %v = memref.load %red[] : memref<i32>
    memref.store %v, %create[] : memref<i32>
  }
  acc.delete accPtr(%create : memref<i32>) dataClause(acc_create)
  acc.copyout accPtr(%red : memref<i32>) to varPtr(%var : memref<i32>) dataClause(acc_reduction)
  return
}

// -----

// acc.copyin followed by acc.create names one variable. The copyin stays, and
// the use of the create inside the region is rewritten to it.
// CHECK-LABEL: func.func @kernel_environment_copyin_then_create
// CHECK-SAME: %[[VAR:.*]]: memref<i32>, %[[OTHER:.*]]: memref<i32>
// CHECK: %[[COPY:.*]] = acc.copyin varPtr(%[[VAR]] : memref<i32>) -> memref<i32>
// CHECK: %[[OTHER_MAP:.*]] = acc.copyin varPtr(%[[OTHER]] : memref<i32>) -> memref<i32>
// CHECK: %[[CREATED:.*]] = acc.create varPtr(%[[VAR]] : memref<i32>) -> memref<i32>
// CHECK: acc.kernel_environment dataOperands(%[[COPY]], %[[OTHER_MAP]] : memref<i32>, memref<i32>) {
// CHECK: memref.load %[[COPY]]
// CHECK: memref.store %{{.*}}, %[[OTHER_MAP]]
// CHECK: acc.delete accPtr(%[[CREATED]] : memref<i32>)
func.func @kernel_environment_copyin_then_create(%var: memref<i32>, %other: memref<i32>) {
  %copy = acc.copyin varPtr(%var : memref<i32>) -> memref<i32>
  %other_map = acc.copyin varPtr(%other : memref<i32>) -> memref<i32>
  %created = acc.create varPtr(%var : memref<i32>) -> memref<i32>
  acc.kernel_environment dataOperands(%copy, %other_map, %created : memref<i32>, memref<i32>, memref<i32>) {
    %v = memref.load %created[] : memref<i32>
    memref.store %v, %other_map[] : memref<i32>
  }
  acc.delete accPtr(%copy : memref<i32>)
  acc.delete accPtr(%created : memref<i32>)
  acc.delete accPtr(%other_map : memref<i32>)
  return
}

// -----

// acc.map_info operands are finalized mappings, so they stay in order beside a
// clause for the same variable.
// CHECK-LABEL: func.func @kernel_environment_map_info
// CHECK-SAME: %[[VAR:.*]]: memref<i32>
// CHECK: %[[COPY:.*]] = acc.copyin varPtr(%[[VAR]] : memref<i32>) -> memref<i32>
// CHECK: %[[TO:.*]] = acc.map_info varPtr(%[[VAR]] : memref<i32>) descKind(none) mapFlags(to) -> memref<i32>
// CHECK: %[[FROM:.*]] = acc.map_info varPtr(%[[VAR]] : memref<i32>) descKind(none) mapFlags(from) -> memref<i32>
// CHECK: acc.kernel_environment dataOperands(%[[COPY]], %[[TO]], %[[FROM]] : memref<i32>, memref<i32>, memref<i32>) {
// CHECK: memref.load %[[TO]]
// CHECK: memref.load %[[FROM]]
// CHECK: memref.store %{{.*}}, %[[COPY]]
func.func @kernel_environment_map_info(%var: memref<i32>) {
  %copy = acc.copyin varPtr(%var : memref<i32>) -> memref<i32>
  %to = acc.map_info varPtr(%var : memref<i32>) varType(i32) descKind(none) mapFlags(to) -> memref<i32>
  %from = acc.map_info varPtr(%var : memref<i32>) varType(i32) descKind(none) mapFlags(from) -> memref<i32>
  acc.kernel_environment dataOperands(%copy, %to, %from : memref<i32>, memref<i32>, memref<i32>) {
    %v = memref.load %to[] : memref<i32>
    %w = memref.load %from[] : memref<i32>
    %sum = arith.addi %v, %w : i32
    memref.store %sum, %copy[] : memref<i32>
  }
  acc.delete accPtr(%copy : memref<i32>)
  return
}
