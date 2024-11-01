// RUN: mlir-opt -test-dead-code-analysis 2>&1 %s | FileCheck %s

// CHECK: test_cfg:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK:   ^bb1 = live
// CHECK:    from ^bb1 = live
// CHECK:    from ^bb0 = live
// CHECK:   ^bb2 = live
// CHECK:    from ^bb1 = live
func.func @test_cfg(%cond: i1) -> ()
    attributes {tag = "test_cfg"} {
  cf.br ^bb1

^bb1:
  cf.cond_br %cond, ^bb1, ^bb2

^bb2:
  return
}

func.func @test_region_control_flow(%cond: i1, %arg0: i64, %arg1: i64) -> () {
  // CHECK: test_if:
  // CHECK:  region #0
  // CHECK: region_preds: (all) predecessors:
  // CHECK:   scf.if
  // CHECK:  region #1
  // CHECK: region_preds: (all) predecessors:
  // CHECK:   scf.if
  // CHECK: op_preds: (all) predecessors:
  // CHECK:   scf.yield {then}
  // CHECK:   scf.yield {else}
  scf.if %cond {
    scf.yield {then}
  } else {
    scf.yield {else}
  } {tag = "test_if"}

  // test_while:
  //  region #0
  // region_preds: (all) predecessors:
  //   scf.while
  //   scf.yield
  //  region #1
  // region_preds: (all) predecessors:
  //   scf.condition
  // op_preds: (all) predecessors:
  //   scf.condition
  %c2_i64 = arith.constant 2 : i64
  %0:2 = scf.while (%arg2 = %arg0) : (i64) -> (i64, i64) {
    %1 = arith.cmpi slt, %arg2, %arg1 : i64
    scf.condition(%1) %arg2, %arg2 : i64, i64
  } do {
  ^bb0(%arg2: i64, %arg3: i64):
    %1 = arith.muli %arg3, %c2_i64 : i64
    scf.yield %1 : i64
  } attributes {tag = "test_while"}

  return
}

// CHECK: foo:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK: op_preds: (all) predecessors:
// CHECK:   func.call @foo(%{{.*}}) {tag = "a"}
// CHECK:   func.call @foo(%{{.*}}) {tag = "b"}
func.func private @foo(%arg0: i32) -> i32
    attributes {tag = "foo"} {
  return {a} %arg0 : i32
}

// CHECK: bar:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK: op_preds: predecessors:
// CHECK:   func.call @bar(%{{.*}}) {tag = "c"}
func.func @bar(%cond: i1) -> i32
    attributes {tag = "bar"} {
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  %c0 = arith.constant 0 : i32
  return {b} %c0 : i32

^bb2:
  %c1 = arith.constant 1 : i32
  return {c} %c1 : i32
}

// CHECK: baz
// CHECK: op_preds: (all) predecessors:
func.func private @baz(i32) -> i32 attributes {tag = "baz"}

func.func @test_callgraph(%cond: i1, %arg0: i32) -> i32 {
  // CHECK: a:
  // CHECK: op_preds: (all) predecessors:
  // CHECK:   func.return {a}
  %0 = func.call @foo(%arg0) {tag = "a"} : (i32) -> i32
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // CHECK: b:
  // CHECK: op_preds: (all) predecessors:
  // CHECK:   func.return {a}
  %1 = func.call @foo(%arg0) {tag = "b"} : (i32) -> i32
  return %1 : i32

^bb2:
  // CHECK: c:
  // CHECK: op_preds: (all) predecessors:
  // CHECK:   func.return {b}
  // CHECK:   func.return {c}
  %2 = func.call @bar(%cond) {tag = "c"} : (i1) -> i32
  // CHECK: d:
  // CHECK: op_preds: predecessors:
  %3 = func.call @baz(%arg0) {tag = "d"} : (i32) -> i32
  return %2 : i32
}

// CHECK: test_unknown_branch:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK:   ^bb1 = live
// CHECK:    from ^bb0 = live
// CHECK:   ^bb2 = live
// CHECK:    from ^bb0 = live
func.func @test_unknown_branch() -> ()
    attributes {tag = "test_unknown_branch"} {
  "test.unknown_br"() [^bb1, ^bb2] : () -> ()

^bb1:
  return

^bb2:
  return
}

// CHECK: test_unknown_region:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK:  region #1
// CHECK:   ^bb0 = live
func.func @test_unknown_region() -> () {
  "test.unknown_region_br"() ({
  ^bb0:
    "test.unknown_region_end"() : () -> ()
  }, {
  ^bb0:
    "test.unknown_region_end"() : () -> ()
  }) {tag = "test_unknown_region"} : () -> ()
  return
}

// CHECK: test_known_dead_block:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK:   ^bb1 = live
// CHECK:   ^bb2 = dead
func.func @test_known_dead_block() -> ()
    attributes {tag = "test_known_dead_block"} {
  %true = arith.constant true
  cf.cond_br %true, ^bb1, ^bb2

^bb1:
  return

^bb2:
  return
}

// CHECK: test_known_dead_edge:
// CHECK:   ^bb2 = live
// CHECK:    from ^bb1 = dead
// CHECK:    from ^bb0 = live
func.func @test_known_dead_edge(%arg0: i1) -> ()
    attributes {tag = "test_known_dead_edge"} {
  cf.cond_br %arg0, ^bb1, ^bb2

^bb1:
  %true = arith.constant true
  cf.cond_br %true, ^bb3, ^bb2

^bb2:
  return

^bb3:
  return
}

func.func @test_known_region_predecessors() -> () {
  %false = arith.constant false
  // CHECK: test_known_if:
  // CHECK:  region #0
  // CHECK:   ^bb0 = dead
  // CHECK:  region #1
  // CHECK:   ^bb0 = live
  // CHECK: region_preds: (all) predecessors:
  // CHECK:   scf.if
  // CHECK: op_preds: (all) predecessors:
  // CHECK:   scf.yield {else}
  scf.if %false {
    scf.yield {then}
  } else {
    scf.yield {else}
  } {tag = "test_known_if"}
  return
}

// CHECK: callable:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK: op_preds: predecessors:
// CHECK:   func.call @callable() {then}
func.func @callable() attributes {tag = "callable"} {
  return
}

func.func @test_dead_callsite() -> () {
  %true = arith.constant true
  scf.if %true {
    func.call @callable() {then} : () -> ()
    scf.yield
  } else {
    func.call @callable() {else} : () -> ()
    scf.yield
  }
  return
}

func.func private @test_dead_return(%arg0: i32) -> i32 {
  %true = arith.constant true
  cf.cond_br %true, ^bb1, ^bb1

^bb1:
  return {true} %arg0 : i32

^bb2:
  return {false} %arg0 : i32
}

func.func @test_call_dead_return(%arg0: i32) -> () {
  // CHECK: test_dead_return:
  // CHECK: op_preds: (all) predecessors:
  // CHECK:   func.return {true}
  %0 = func.call @test_dead_return(%arg0) {tag = "test_dead_return"} : (i32) -> i32
  return
}
