// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=aggressive}))' | FileCheck %s

// Test case: Simple case of deleting a dead pure op.

// CHECK:      func @f(%arg0: f32) {
// CHECK-NEXT:   return

func.func @f(%arg0: f32) {
  %0 = "arith.addf"(%arg0, %arg0) : (f32, f32) -> f32
  return
}

// -----

// Test case: Simple case of deleting a block argument.

// CHECK:      func @f(%arg0: f32)
// CHECK-NEXT:   "test.br"()[^bb1]
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   return

func.func @f(%arg0: f32) {
  "test.br"(%arg0)[^succ] : (f32) -> ()
^succ(%0: f32):
  return
}

// -----

// Test case: Deleting recursively dead block arguments.

// CHECK:      func @f(%arg0: f32)
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cf.br ^bb1


func.func @f(%arg0: f32) {
  cf.br ^loop(%arg0: f32)
^loop(%loop: f32):
  cf.br ^loop(%loop: f32)
}

// -----

// Test case: Deleting recursively dead block arguments with pure ops in between.

// CHECK:      func @f(%arg0: f32)
// CHECK-NEXT:   cf.br ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   cf.br ^bb1

func.func @f(%arg0: f32) {
  cf.br ^loop(%arg0: f32)
^loop(%0: f32):
  %1 = "math.exp"(%0) : (f32) -> f32
  cf.br ^loop(%1: f32)
}

// -----

// Test case: Delete block arguments for cf.cond_br.

// CHECK:      func @f(%arg0: f32, %arg1: i1)
// CHECK-NEXT:   return

func.func @f(%arg0: f32, %pred: i1) {
  %exp = "math.exp"(%arg0) : (f32) -> f32
  cf.cond_br %pred, ^true(%exp: f32), ^false(%exp: f32)
^true(%0: f32):
  return
^false(%1: f32):
  return
}

// -----

// Test case: Recursively DCE into enclosed regions.

// CHECK:      func.func @f(%arg0: f32)
// CHECK-NOT:     arith.addf

func.func @f(%arg0: f32) {
  "test.region"() (
    {
      %0 = "arith.addf"(%arg0, %arg0) : (f32, f32) -> f32
    }
  ) : () -> ()
  return
}

// -----

// Test case: Don't delete pure ops that feed into returns.

// CHECK:      func @f(%arg0: f32) -> f32
// CHECK-NEXT:   [[VAL0:%.+]] = arith.addf %arg0, %arg0 : f32
// CHECK-NEXT:   return [[VAL0]] : f32

func.func @f(%arg0: f32) -> f32 {
  %0 = "arith.addf"(%arg0, %arg0) : (f32, f32) -> f32
  return %0 : f32
}

// -----

// Test case: Don't delete potentially side-effecting ops.

// CHECK:      func @f(%arg0: f32)
// CHECK-NEXT:   "foo.print"(%arg0) : (f32) -> ()
// CHECK-NEXT:   return

func.func @f(%arg0: f32) {
  "foo.print"(%arg0) : (f32) -> ()
  return
}

// -----

// Test case: Uses in nested regions are deleted correctly.

// CHECK:      func @f(%arg0: f32)
// CHECK-NEXT:   "foo.has_region"
// CHECK-NEXT:     "foo.return"

func.func @f(%arg0: f32) {
  %0 = "math.exp"(%arg0) : (f32) -> f32
  "foo.has_region"() ({
    %1 = "math.exp"(%0) : (f32) -> f32
    "foo.return"() : () -> ()
  }) : () -> ()
  return
}

// -----

// Test case: Test the mechanics of deleting multiple block arguments.

// CHECK:      func @f(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>, %arg2: tensor<3xf32>, %arg3: tensor<4xf32>, %arg4: tensor<5xf32>)
// CHECK-NEXT:   "test.br"()[^bb1]
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   "foo.print"(%arg1)
// CHECK-NEXT:   "foo.print"(%arg3)
// CHECK-NEXT:   return


func.func @f(
  %arg0: tensor<1xf32>,
  %arg1: tensor<2xf32>,
  %arg2: tensor<3xf32>,
  %arg3: tensor<4xf32>,
  %arg4: tensor<5xf32>) {
  "test.br"(%arg0, %arg1, %arg2, %arg3, %arg4)[^succ] : (tensor<1xf32>, tensor<2xf32>, tensor<3xf32>, tensor<4xf32>, tensor<5xf32>) -> ()
^succ(%t1: tensor<1xf32>, %t2: tensor<2xf32>, %t3: tensor<3xf32>, %t4: tensor<4xf32>, %t5: tensor<5xf32>):
  "foo.print"(%t2) : (tensor<2xf32>) -> ()
  "foo.print"(%t4) : (tensor<4xf32>) -> ()
  return
}

// -----

// Test case: Test values with use-def cycles are deleted properly.

// CHECK:      func @f()
// CHECK-NEXT:   test.graph_region
// CHECK-NEXT:     "test.terminator"() : () -> ()

func.func @f() {
  test.graph_region {
    %0 = "math.exp"(%1) : (f32) -> f32
    %1 = "math.exp"(%0) : (f32) -> f32
    "test.terminator"() : ()->()
  }
  return
}

// -----


// Test case: Delete ops that only have side-effects on an allocated result.

// CHECK:      func @f()
// CHECK-NOT:    test_effects_result
// CHECK-NEXT:   return

func.func @f() {
  %0 = "test.test_effects_result"() : () -> i32
  return
}

// -----

// Test case: Terminate when a block becomes unreachable while patterns run.

// CHECK-LABEL: func @unreachable_loop_terminates
// CHECK-NEXT:   return

func.func @unreachable_loop_terminates(%cond: i1, %init: i32) {
  %true = arith.constant true
  %c1 = arith.constant 1 : i32
  cf.cond_br %true, ^exit, ^header(%init : i32)
^header(%iv: i32):
  cf.cond_br %cond, ^exit, ^latch
^latch:
  %next = arith.addi %iv, %c1 : i32
  cf.br ^header(%next : i32)
^exit:
  return
}

// -----

// The lost edge into ^dead cannot be decided incrementally: ^dead keeps ^pass
// as a predecessor, which is only reachable from ^dead itself. The blocks
// behind ^dead hold an SSA cycle through block arguments.
// CHECK-LABEL: func.func @uncertain_chain
// CHECK-NOT: arith.addi
// CHECK: return
func.func @uncertain_chain(%c: i1, %x: i32) -> i32 {
  %true = arith.constant true
  %zero = arith.constant 0 : i32
  cf.br ^header(%x : i32)
^header(%h: i32):
  cf.cond_br %true, ^next(%h : i32), ^dead(%h : i32)
^pass:
  cf.br ^dead(%x : i32)
^dead(%d: i32):
  %s = arith.addi %d, %d : i32
  cf.cond_br %c, ^dead2(%s : i32), ^pass
^dead2(%e: i32):
  %t = arith.addi %e, %s : i32
  cf.br ^dead(%t : i32)
^next(%a: i32):
  %r = arith.addi %a, %zero : i32
  cf.cond_br %c, ^header(%r : i32), ^exit
^exit:
  return %r : i32
}

// -----

// Nested regions: a constant scf.if whose region holds a CFG is inlined, so
// its cached state must be dropped, and a block that becomes unreachable holds
// nested CFG and scf.if regions whose operations must be skipped.
// CHECK-LABEL: func.func @nested_two_level
// CHECK-NEXT: return %arg1
func.func @nested_two_level(%c: i1, %x: i32) -> i32 {
  %true = arith.constant true
  %zero = arith.constant 0 : i32
  cf.br ^header(%x : i32)
^header(%h: i32):
  %v = scf.if %true -> i32 {
    %w = scf.execute_region -> i32 {
      cf.br ^a
    ^a:
      cf.cond_br %true, ^b(%h : i32), ^c(%h : i32)
    ^b(%bb: i32):
      scf.yield %bb : i32
    ^c(%cc: i32):
      %q = arith.addi %cc, %cc : i32
      cf.br ^c(%q : i32)
    }
    scf.yield %w : i32
  } else {
    scf.yield %zero : i32
  }
  cf.cond_br %true, ^exit(%v : i32), ^dead(%v : i32)
^dead(%d: i32):
  %r = scf.execute_region -> i32 {
    cf.br ^l(%d : i32)
  ^l(%iv: i32):
    %n = arith.addi %iv, %d : i32
    %m = scf.if %c -> i32 {
      %mm = arith.addi %n, %iv : i32
      scf.yield %mm : i32
    } else {
      scf.yield %n : i32
    }
    cf.cond_br %c, ^l(%m : i32), ^done(%m : i32)
  ^done(%z: i32):
    scf.yield %z : i32
  }
  cf.br ^dead(%r : i32)
^exit(%o: i32):
  return %o : i32
}

// -----

// The same CFG change can occur in a region nested below the driver root.
// CHECK-LABEL: func.func @nested_unreachable
// CHECK-NOT: arith.addi
// CHECK: return
func.func @nested_unreachable(%cond: i1, %init: i32) {
  scf.execute_region {
    %true = arith.constant true
    cf.cond_br %true, ^exit, ^header(%init : i32)
  ^header(%iv: i32):
    cf.cond_br %cond, ^exit, ^latch
  ^latch:
    %next = arith.addi %iv, %iv : i32
    cf.br ^header(%next : i32)
  ^exit:
    scf.yield
  }
  return
}
