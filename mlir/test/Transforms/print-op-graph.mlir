// RUN: mlir-opt -allow-unregistered-dialect -print-op-graph %s -o %t 2>&1 | FileCheck %s

// CHECK-LABEL: digraph "merge_blocks"
func @merge_blocks(%arg0: i32, %arg1 : i32) -> () {
  %0:2 = "test.merge_blocks"() ({
  ^bb0:
     "test.br"(%arg0, %arg1)[^bb1] : (i32, i32) -> ()
  ^bb1(%arg3 : i32, %arg4 : i32):
     "test.return"(%arg3, %arg4) : (i32, i32) -> ()
  }) : () -> (i32, i32)
  "test.return"(%0#0, %0#1) : (i32, i32) -> ()
}
