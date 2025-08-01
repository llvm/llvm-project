// RUN: mlir-opt -view-op-graph -allow-unregistered-dialect %s -o %t 2>&1 | FileCheck -check-prefix=DFG %s

// DFG-LABEL: digraph G {
//  DFG-NEXT:   compound = true;
//  DFG-NEXT:   subgraph cluster_1 {
//  DFG-NEXT:     v2 [label = " ", shape = plain];
//  DFG-NEXT:     label = "builtin.module : ()\l";
//  DFG-NEXT:     subgraph cluster_3 {
//  DFG-NEXT:       v4 [label = " ", shape = plain];
//  DFG-NEXT:       label = "";
//  DFG-NEXT:       subgraph cluster_5 {
//  DFG-NEXT:         v6 [label = " ", shape = plain];
//  DFG-NEXT:         label = "test.graph_region : ()\l";
//  DFG-NEXT:         subgraph cluster_7 {
//  DFG-NEXT:           v8 [label = " ", shape = plain];
//  DFG-NEXT:           label = "";
//  DFG-NEXT:           v9 [fillcolor = "0.000000 0.3 0.95", label = "{{\{\{}}<arg_0> %0|<arg_2> %2}|op1\l|{<res_0> %0 i32}}", shape = Mrecord, style = filled];
//  DFG-NEXT:           subgraph cluster_10 {
//  DFG-NEXT:             v11 [label = " ", shape = plain];
//  DFG-NEXT:             label = "test.ssacfg_region : (i32)\l";
//  DFG-NEXT:             subgraph cluster_12 {
//  DFG-NEXT:               v13 [label = " ", shape = plain];
//  DFG-NEXT:               label = "";
//  DFG-NEXT:               v14 [fillcolor = "0.166667 0.3 0.95", label = "{{\{\{}}<arg_0> %0|<arg_1> %1|<arg_2> %2|<arg_3> %3}|op2\l|{<res_4> %4 i32}}", shape = Mrecord, style = filled];
//  DFG-NEXT:             }
//  DFG-NEXT:           }
//  DFG-NEXT:           v15 [fillcolor = "0.166667 0.3 0.95", label = "{{\{\{}}<arg_0> %0|<arg_3> %3}|op2\l|{<res_2> %2 i32}}", shape = Mrecord, style = filled];
//  DFG-NEXT:           v16 [fillcolor = "0.500000 0.3 0.95", label = "{{\{\{}}<arg_0> %0}|op3\l|{<res_3> %3 i32}}", shape = Mrecord, style = filled];
//  DFG-NEXT:         }
//  DFG-NEXT:       }
//  DFG-NEXT:     }
//  DFG-NEXT:   }
//  DFG-NEXT:   v9:res_0:s -> v9:arg_0:n[style = solid];
//  DFG-NEXT:   v15:res_2:s -> v9:arg_2:n[style = solid];
//  DFG-NEXT:   v9:res_0:s -> v14:arg_0:n[style = solid];
//  DFG-NEXT:   v11 -> v14:arg_1:n[ltail = cluster_10, style = solid];
//  DFG-NEXT:   v15:res_2:s -> v14:arg_2:n[style = solid];
//  DFG-NEXT:   v16:res_3:s -> v14:arg_3:n[style = solid];
//  DFG-NEXT:   v9:res_0:s -> v15:arg_0:n[style = solid];
//  DFG-NEXT:   v16:res_3:s -> v15:arg_3:n[style = solid];
//  DFG-NEXT:   v9:res_0:s -> v16:arg_0:n[style = solid];
//  DFG-NEXT: }

"test.graph_region"() ({ // A Graph region
  %1 = "op1"(%1, %3) : (i32, i32) -> (i32)  // OK: %1, %3 allowed here
  %2 = "test.ssacfg_region"() ({
     %5 = "op2"(%1, %2, %3, %4) : (i32, i32, i32, i32) -> (i32) // OK: %1, %2, %3, %4 all defined in the containing region
  }) : () -> (i32)
  %3 = "op2"(%1, %4) : (i32, i32) -> (i32)  // OK: %4 allowed here
  %4 = "op3"(%1) : (i32) -> (i32)
}) : () -> ()
