// RUN: mlir-opt -view-op-graph -allow-unregistered-dialect %s -o %t 2>&1 | FileCheck -check-prefix=DFG %s

// DFG-LABEL: digraph G {
//       DFG:   compound = true;
//       DFG:   subgraph cluster_1 {
//       DFG:     v2 [label = " ", shape = plain];
//       DFG:     label = "builtin.module : ()\l";
//       DFG:     subgraph cluster_3 {
//       DFG:       v4 [label = " ", shape = plain];
//       DFG:       label = "";
//       DFG:       subgraph cluster_5 {
//       DFG:         v6 [label = " ", shape = plain];
//       DFG:         label = "test.graph_region : ()\l";
//       DFG:         subgraph cluster_7 {
//       DFG:           v8 [label = " ", shape = plain];
//       DFG:           label = "";
//       DFG:           v9 [fillcolor = "0.000000 0.3 0.95", label = "{{\{\{}}<arg_0> %0|<arg_2> %2}|op1\l|{<res_0> %0 i32}}", shape = Mrecord, style = filled];
//       DFG:           subgraph cluster_10 {
//       DFG:             v11 [label = " ", shape = plain];
//       DFG:             label = "test.ssacfg_region : (i32)\l";
//       DFG:             subgraph cluster_12 {
//       DFG:               v13 [label = " ", shape = plain];
//       DFG:               label = "";
//       DFG:               v14 [fillcolor = "0.166667 0.3 0.95", label = "{{\{\{}}<arg_0> %0|<arg_1> %1|<arg_2> %2|<arg_3> %3}|op2\l|{<res_4> %4 i32}}", shape = Mrecord, style = filled];
//       DFG:             }
//       DFG:           }
//       DFG:           v15 [fillcolor = "0.166667 0.3 0.95", label = "{{\{\{}}<arg_0> %0|<arg_3> %3}|op2\l|{<res_2> %2 i32}}", shape = Mrecord, style = filled];
//       DFG:           v16 [fillcolor = "0.500000 0.3 0.95", label = "{{\{\{}}<arg_0> %0}|op3\l|{<res_3> %3 i32}}", shape = Mrecord, style = filled];
//       DFG:         }
//       DFG:       }
//       DFG:     }
//       DFG:   }
//       DFG:   v9:res_0:s -> v9:arg_0:n[style = solid];
//       DFG:   v15:res_2:s -> v9:arg_2:n[style = solid];
//       DFG:   v9:res_0:s -> v14:arg_0:n[style = solid];
//       DFG:   v11 -> v14:arg_1:n[ltail = cluster_10, style = solid];
//       DFG:   v15:res_2:s -> v14:arg_2:n[style = solid];
//       DFG:   v16:res_3:s -> v14:arg_3:n[style = solid];
//       DFG:   v9:res_0:s -> v15:arg_0:n[style = solid];
//       DFG:   v16:res_3:s -> v15:arg_3:n[style = solid];
//       DFG:   v9:res_0:s -> v16:arg_0:n[style = solid];
//       DFG: }

"test.graph_region"() ({ // A Graph region
  %1 = "op1"(%1, %3) : (i32, i32) -> (i32)  // OK: %1, %3 allowed here
  %2 = "test.ssacfg_region"() ({
     %5 = "op2"(%1, %2, %3, %4) : (i32, i32, i32, i32) -> (i32) // OK: %1, %2, %3, %4 all defined in the containing region
  }) : () -> (i32)
  %3 = "op2"(%1, %4) : (i32, i32) -> (i32)  // OK: %4 allowed here
  %4 = "op3"(%1) : (i32) -> (i32)
}) : () -> ()
