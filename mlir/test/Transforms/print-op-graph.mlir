// RUN: mlir-opt -allow-unregistered-dialect -mlir-elide-elementsattrs-if-larger=2 -print-op-graph %s -o %t 2>&1 | FileCheck %s

// CHECK-LABEL: digraph "merge_blocks"
// CHECK{LITERAL}: value: [[...]] : tensor\<2x2xi32\>}
// CHECK{LITERAL}: value: dense\<1\> : tensor\<5xi32\>}
// CHECK{LITERAL}: value: dense\<[[0, 1]]\> : tensor\<1x2xi32\>}
func @merge_blocks(%arg0: i32, %arg1 : i32) -> () {
  %0 = constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %1 = constant dense<1> : tensor<5xi32>
  %2 = constant dense<[[0, 1]]> : tensor<1x2xi32>

  %3:2 = "test.merge_blocks"() ({
  ^bb0:
     "test.br"(%arg0, %arg1)[^bb1] : (i32, i32) -> ()
  ^bb1(%arg3 : i32, %arg4 : i32):
     "test.return"(%arg3, %arg4) : (i32, i32) -> ()
  }) : () -> (i32, i32)
  "test.return"(%3#0, %3#1) : (i32, i32) -> ()
}
