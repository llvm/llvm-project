// RUN: mlir-opt -allow-unregistered-dialect -mlir-elide-elementsattrs-if-larger=2 -view-op-graph %s -o %t 2>&1 | FileCheck -check-prefix=DFG %s
// RUN: mlir-opt -allow-unregistered-dialect -mlir-elide-elementsattrs-if-larger=2 -view-op-graph='print-data-flow-edges=false print-control-flow-edges=true' %s -o %t 2>&1 | FileCheck -check-prefix=CFG %s

// DFG-LABEL: digraph G {
//       DFG:   subgraph {{.*}} {
//       DFG:     subgraph {{.*}}
//       DFG:       label = "builtin.func{{.*}}merge_blocks
//       DFG:       subgraph {{.*}} {
//       DFG:         v[[ARG0:.*]] [label = "arg0"
//       DFG:         v[[CONST10:.*]] [label ={{.*}}10 : i32
//       DFG:         subgraph [[CLUSTER_MERGE_BLOCKS:.*]] {
//       DFG:           v[[ANCHOR:.*]] [label = " ", shape = plain]
//       DFG:           label = "test.merge_blocks
//       DFG:           subgraph {{.*}} {
//       DFG:             v[[TEST_BR:.*]] [label = "test.br
//       DFG:           }
//       DFG:           subgraph {{.*}} {
//       DFG:           }
//       DFG:         }
//       DFG:         v[[TEST_RET:.*]] [label = "test.return
//       DFG:   v[[ARG0]] -> v[[TEST_BR]]
//       DFG:   v[[CONST10]] -> v[[TEST_BR]]
//       DFG:   v[[ANCHOR]] -> v[[TEST_RET]] [{{.*}}, ltail = [[CLUSTER_MERGE_BLOCKS]]]
//       DFG:   v[[ANCHOR]] -> v[[TEST_RET]] [{{.*}}, ltail = [[CLUSTER_MERGE_BLOCKS]]]

// CFG-LABEL: digraph G {
//       CFG:   subgraph {{.*}} {
//       CFG:     subgraph {{.*}}
//       CFG:       label = "builtin.func{{.*}}merge_blocks
//       CFG:       subgraph {{.*}} {
//       CFG:         v[[C1:.*]] [label = "arith.constant
//       CFG:         v[[C2:.*]] [label = "arith.constant
//       CFG:         v[[C3:.*]] [label = "arith.constant
//       CFG:         v[[C4:.*]] [label = "arith.constant
//       CFG:         v[[TEST_FUNC:.*]] [label = "test.func
//       CFG:         subgraph [[CLUSTER_MERGE_BLOCKS:.*]] {
//       CFG:           v[[ANCHOR:.*]] [label = " ", shape = plain]
//       CFG:           label = "test.merge_blocks
//       CFG:           subgraph {{.*}} {
//       CFG:             v[[TEST_BR:.*]] [label = "test.br
//       CFG:           }
//       CFG:           subgraph {{.*}} {
//       CFG:           }
//       CFG:         }
//       CFG:         v[[TEST_RET:.*]] [label = "test.return
//       CFG:   v[[C1]] -> v[[C2]]
//       CFG:   v[[C2]] -> v[[C3]]
//       CFG:   v[[C3]] -> v[[C4]]
//       CFG:   v[[C4]] -> v[[TEST_FUNC]]
//       CFG:   v[[TEST_FUNC]] -> v[[ANCHOR]] [{{.*}}, lhead = [[CLUSTER_MERGE_BLOCKS]]]
//       CFG:   v[[ANCHOR]] -> v[[TEST_RET]] [{{.*}}, ltail = [[CLUSTER_MERGE_BLOCKS]]]

func @merge_blocks(%arg0: i32, %arg1 : i32) -> () {
  %0 = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %1 = arith.constant dense<1> : tensor<5xi32>
  %2 = arith.constant dense<[[0, 1]]> : tensor<1x2xi32>
  %a = arith.constant 10 : i32
  %b = "test.func"() : () -> i32
  %3:2 = "test.merge_blocks"() ({
  ^bb0:
     "test.br"(%arg0, %b, %a)[^bb1] : (i32, i32, i32) -> ()
  ^bb1(%arg3 : i32, %arg4 : i32, %arg5: i32):
     "test.return"(%arg3, %arg4) : (i32, i32) -> ()
  }) : () -> (i32, i32)
  "test.return"(%3#0, %3#1) : (i32, i32) -> ()
}
