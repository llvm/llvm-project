// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(test-clone))" --split-input-file | FileCheck %s

module {
  func.func @fixpoint(%arg1 : i32) -> i32 {
    %r = "test.use"(%arg1) ({
      %r2 = "test.use2"(%arg1) ({
         "test.yield2"(%arg1) : (i32) -> ()
      }) : (i32) -> i32
      "test.yield"(%r2) : (i32) -> ()
    }) : (i32) -> i32
    return %r : i32
  }
}

// CHECK: notifyOperationInserted: test.use
// CHECK-NEXT: notifyOperationInserted: test.use2
// CHECK-NEXT: notifyOperationInserted: test.yield2
// CHECK-NEXT: notifyOperationInserted: test.yield
// CHECK-NEXT: notifyOperationInserted: func.return

// CHECK-LABEL: func @fixpoint
// CHECK-SAME:       (%[[arg0:.+]]: i32) -> i32 {
// CHECK-NEXT:     %[[i0:.+]] = "test.use"(%[[arg0]]) ({
// CHECK-NEXT:       %[[r2:.+]] = "test.use2"(%[[arg0]]) ({
// CHECK-NEXT:         "test.yield2"(%[[arg0]]) : (i32) -> ()
// CHECK-NEXT:       }) : (i32) -> i32
// CHECK-NEXT:       "test.yield"(%[[r2]]) : (i32) -> ()
// CHECK-NEXT:     }) : (i32) -> i32
// CHECK-NEXT:     %[[i1:.+]] = "test.use"(%[[i0]]) ({
// CHECK-NEXT:       %[[r2:.+]] = "test.use2"(%[[i0]]) ({
// CHECK-NEXT:         "test.yield2"(%[[i0]]) : (i32) -> ()
// CHECK-NEXT:       }) : (i32) -> i32
// CHECK-NEXT:       "test.yield"(%[[r2]]) : (i32) -> ()
// CHECK-NEXT:     }) : (i32) -> i32
// CHECK-NEXT:     return %[[i1]] : i32
// CHECK-NEXT:   }

// -----

func.func @clone_unregistered_with_attrs() {
  "unregistered.foo"() <{bar = 1 : i64, flag = true, name = "test", value = 3.14 : f32}> : () -> ()
  "unregistered.bar"() : () -> ()
  "unregistered.empty_dict"() <{}> : () -> ()
  "unregistered.complex"() <{
    array = [1, 2, 3],
    dict = {key1 = 42 : i32, key2 = "value"},
    nested = {inner = {deep = 100 : i64}}
  }> : () -> ()
  return
}

// CHECK: notifyOperationInserted: unregistered.foo
// CHECK-NEXT: notifyOperationInserted: unregistered.bar
// CHECK-NEXT: notifyOperationInserted: unregistered.empty_dict
// CHECK-NEXT: notifyOperationInserted: unregistered.complex
// CHECK-NEXT: notifyOperationInserted: func.return

// CHECK-LABEL:  func @clone_unregistered_with_attrs() {
// CHECK-NEXT:     "unregistered.foo"() <{bar = 1 : i64, flag = true, name = "test", value = [[PI:.+]] : f32}> : () -> ()
// CHECK-NEXT:     "unregistered.bar"() : () -> ()
// CHECK-NEXT:     "unregistered.empty_dict"() <{}> : () -> ()
// CHECK-NEXT:     "unregistered.complex"() <{array = [1, 2, 3], dict = {key1 = 42 : i32, key2 = "value"}, nested = {inner = {deep = 100 : i64}}}> : () -> ()
// CHECK-NEXT:     "unregistered.foo"() <{bar = 1 : i64, flag = true, name = "test", value = [[PI]] : f32}> : () -> ()
// CHECK-NEXT:     "unregistered.bar"() : () -> ()
// CHECK-NEXT:     "unregistered.empty_dict"() <{}> : () -> ()
// CHECK-NEXT:     "unregistered.complex"() <{array = [1, 2, 3], dict = {key1 = 42 : i32, key2 = "value"}, nested = {inner = {deep = 100 : i64}}}> : () -> ()
