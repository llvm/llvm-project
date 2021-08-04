// RUN: mlir-opt -allow-unregistered-dialect -mlir-elide-elementsattrs-if-larger=2 -view-op-graph %s -o %t 2>&1 | FileCheck %s

// CHECK-LABEL: digraph G {
//       CHECK:   subgraph {{.*}} {
//       CHECK:     subgraph {{.*}}
//       CHECK:       label = "builtin.func{{.*}}merge_blocks
//       CHECK:       subgraph {{.*}} {
//       CHECK:         v[[ARG0:.*]] [label = "arg0"
//       CHECK:         v[[CONST10:.*]] [label ={{.*}}10 : i32
//       CHECK:         subgraph [[CLUSTER_MERGE_BLOCKS:.*]] {
//       CHECK:           v[[ANCHOR:.*]] [label = " ", shape = plain]
//       CHECK:           label = "test.merge_blocks
//       CHECK:           subgraph {{.*}} {
//       CHECK:             v[[TEST_BR:.*]] [label = "test.br
//       CHECK:           }
//       CHECK:           subgraph {{.*}} {
//       CHECK:           }
//       CHECK:         }
//       CHECK:         v[[TEST_RET:.*]] [label = "test.return
//       CHECK:   v[[ARG0]] -> v[[TEST_BR]]
//       CHECK:   v[[CONST10]] -> v[[TEST_BR]]
//       CHECK:   v[[ANCHOR]] -> v[[TEST_RET]] [{{.*}}, ltail = [[CLUSTER_MERGE_BLOCKS]]]
//       CHECK:   v[[ANCHOR]] -> v[[TEST_RET]] [{{.*}}, ltail = [[CLUSTER_MERGE_BLOCKS]]]
func @merge_blocks(%arg0: i32, %arg1 : i32) -> () {
  %0 = constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %1 = constant dense<1> : tensor<5xi32>
  %2 = constant dense<[[0, 1]]> : tensor<1x2xi32>
  %a = constant 10 : i32
  %b = "test.func"() : () -> i32
  %3:2 = "test.merge_blocks"() ({
  ^bb0:
     "test.br"(%arg0, %b, %a)[^bb1] : (i32, i32, i32) -> ()
  ^bb1(%arg3 : i32, %arg4 : i32, %arg5: i32):
     "test.return"(%arg3, %arg4) : (i32, i32) -> ()
  }) : () -> (i32, i32)
  "test.return"(%3#0, %3#1) : (i32, i32) -> ()
}
