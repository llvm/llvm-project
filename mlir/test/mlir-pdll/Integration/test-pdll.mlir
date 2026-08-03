// RUN: mlir-opt %s -test-pdll-pass -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @simpleTest
func.func @simpleTest() {
  // CHECK: test.success
  "test.simple"() : () -> ()
  return
}

// CHECK-LABEL: func @testImportedInterface
func.func @testImportedInterface() -> i1 {
  // CHECK: test.non_cast
  // CHECK: test.success
  "test.non_cast"() : () -> ()
  %value = "builtin.unrealized_conversion_cast"() : () -> (i1)
  return %value : i1
}

// CHECK-LABEL: func @testWithConstraint
func.func @testWithConstraint(%a: i32) {
    // CHECK: test.success
    %b = "test.op_a"(%a) { attr = 0 : i32} : (i32) -> (i32)
    return
}

// CHECK-LABEL: func @testMatchInnerOp
func.func @testMatchInnerOp() {
  // CHECK-NOT: test.outer
  "test.outer"() ({
    "test.inner"() : () -> ()
  }) : () -> ()
  // CHECK-NOT: test.outer_explicit
  "test.outer_explicit"() ({
    "test.inner_explicit"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @testMatchIfElse
func.func @testMatchIfElse() {
  // CHECK-NOT: test.if
  "test.if"() ({
    "test.then_op"() : () -> ()
  }, {
    "test.else_op"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @testMatchMultiBlock
func.func @testMatchMultiBlock() {
  // CHECK-NOT: test.branch
  "test.branch"() ({
  ^bb0:
    "test.entry"() : () -> ()
    "test.br"()[^bb1] : () -> ()
  ^bb1:
    "test.exit"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @testMatchAndMoveBlock
func.func @testMatchAndMoveBlock(%arg0: i32) {
  // CHECK: "test.dest_move"() ({
  // CHECK-NEXT: ^bb0(%{{.*}}: i32):
  // CHECK-NEXT: "test.body"
  "test.wrapper"() ({
    "test.source_move"() ({
    ^bb0(%arg1: i32):
      "test.body"() : () -> ()
    }) : () -> ()
    "test.dest_move"() ({
      "test.placeholder"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @testMatchInductionVar
func.func @testMatchInductionVar() {
  // CHECK-NOT: test.for
  "test.for"() ({
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = "test.addi"(%arg0, %arg1) : (i32, i32) -> i32
  }) : () -> ()
  return
}

// CHECK-LABEL: func @testMatchForWithVarArgs
func.func @testMatchForWithVarArgs() {
  // CHECK-NOT: test.for_var_args
  "test.for_var_args"() ({
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    %0 = "test.custom_op"(%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
  }) : () -> ()
  return
}

// CHECK-LABEL: func @testMatchNested
func.func @testMatchNested() {
  // CHECK-NOT: test.pipeline
  "test.pipeline"() ({
    "test.stage"() ({
      "test.compute"() : () -> ()
    }) : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @testInlineWithTerminatorRewrite
func.func @testInlineWithTerminatorRewrite() {
  // CHECK: "test.new_terminator"
  // CHECK: "test.dest_inline"() ({
  // CHECK-NEXT: ^bb0(%{{.*}}: i32):
  // CHECK-NEXT: "test.work"
  "test.wrapper"() ({
    "test.source_inline"() ({
    ^bb0(%arg0: i32):
      "test.work"() : () -> ()
      "test.old_terminator"() : () -> ()
    }) : () -> ()
    "test.dest_inline"() ({
    ^bb0:
    }) : () -> ()
  }) : () -> ()
  return
}

