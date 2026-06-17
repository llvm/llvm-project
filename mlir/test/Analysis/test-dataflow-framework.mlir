// RUN: mlir-opt -test-staged-analyses -split-input-file %s | FileCheck %s

// Test that FooAnalysis does not crash when a func.func carries integer
// attributes with signless i64 type (e.g. llvm.link_call_count).

// CHECK-LABEL: func.func private @callee
module {
  func.func private @callee(%arg0: i32) -> i32 attributes {llvm.link_call_count = 10 : i64, sym.link_call_count = 11 : i64} {
    %0:2 = "test.foo"() {foo = 5 : i32} : () -> (i32, i32)
    %1 = "test.foo"() {foo = 7 : i32} : () -> i32
    %2 = "test.hello"(%0#0, %1) : (i32, i32) -> i32
    return %2 : i32
  }
}
