// RUN: mlir-opt -view-op-graph -allow-unregistered-dialect %s -o %t 2>&1 | FileCheck -check-prefix=DFG %s

// DFG-LABEL: digraph G {
//       DFG:   compound = true;
//       DFG:   subgraph cluster_1 {
//       DFG:     v2 [label = " ", shape = plain];
//       DFG:     label = "builtin.module : ()\n";
//       DFG:     subgraph cluster_3 {
//       DFG:       v4 [label = " ", shape = plain];
//       DFG:       label = "";
//       DFG:       subgraph cluster_5 {
//       DFG:         v6 [label = " ", shape = plain];
//       DFG:         label = "test.graph_region : ()\n";
//       DFG:         subgraph cluster_7 {
//       DFG:           v8 [label = " ", shape = plain];
//       DFG:           label = "";
//       DFG:           v9 [fillcolor = "0.000000 1.0 1.0", label = "op1 : (i32)\n", shape = ellipse, style = filled];
//       DFG:           subgraph cluster_10 {
//       DFG:             v11 [label = " ", shape = plain];
//       DFG:             label = "test.ssacfg_region : (i32)\n";
//       DFG:             subgraph cluster_12 {
//       DFG:               v13 [label = " ", shape = plain];
//       DFG:               label = "";
//       DFG:               v14 [fillcolor = "0.166667 1.0 1.0", label = "op2 : (i32)\n", shape = ellipse, style = filled];
//       DFG:             }
//       DFG:           }
//       DFG:           v15 [fillcolor = "0.166667 1.0 1.0", label = "op2 : (i32)\n", shape = ellipse, style = filled];
//       DFG:           v16 [fillcolor = "0.500000 1.0 1.0", label = "op3 : (i32)\n", shape = ellipse, style = filled];
//       DFG:         }
//       DFG:       }
//       DFG:     }
//       DFG:   }
//       DFG:   v9 -> v9 [label = "0", style = solid];
//       DFG:   v15 -> v9 [label = "1", style = solid];
//       DFG:   v9 -> v14 [label = "0", style = solid];
//       DFG:   v11 -> v14 [ltail = cluster_10, style = solid];
//       DFG:   v15 -> v14 [label = "2", style = solid];
//       DFG:   v16 -> v14 [label = "3", style = solid];
//       DFG:   v9 -> v15 [label = "0", style = solid];
//       DFG:   v16 -> v15 [label = "1", style = solid];
//       DFG:   v9 -> v16 [label = "", style = solid];
//       DFG: }

"test.graph_region"() ({ // A Graph region
  %1 = "op1"(%1, %3) : (i32, i32) -> (i32)  // OK: %1, %3 allowed here
  %2 = "test.ssacfg_region"() ({
     %5 = "op2"(%1, %2, %3, %4) : (i32, i32, i32, i32) -> (i32) // OK: %1, %2, %3, %4 all defined in the containing region
  }) : () -> (i32)
  %3 = "op2"(%1, %4) : (i32, i32) -> (i32)  // OK: %4 allowed here
  %4 = "op3"(%1) : (i32) -> (i32)
}) : () -> ()
