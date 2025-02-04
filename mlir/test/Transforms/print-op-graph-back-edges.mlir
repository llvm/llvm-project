// RUN: mlir-opt -view-op-graph %s -o %t 2>&1 | FileCheck -check-prefix=DFG %s

// DFG-LABEL: digraph G {
//       DFG:   compound = true;
//       DFG:   subgraph cluster_1 {
//       DFG:     v2 [label = " ", shape = plain];
//       DFG:     label = "builtin.module : ()\l";
//       DFG:     subgraph cluster_3 {
//       DFG:       v4 [label = " ", shape = plain];
//       DFG:       label = "";
//       DFG:       v5 [fillcolor = "0.000000 0.3 0.95", label = "{{\{\{}}<arg_c0> %c0|<arg_c1> %c1}|arith.addi\l\loverflowFlags: #arith.overflow\<none...\l|{<res_0> %0 index}}", shape = Mrecord, style = filled];
//       DFG:       v6 [fillcolor = "0.333333 0.3 0.95", label = "{arith.constant\l\lvalue: 0 : index\l|{<res_c0> %c0 index}}", shape = Mrecord, style = filled];
//       DFG:       v7 [fillcolor = "0.333333 0.3 0.95", label = "{arith.constant\l\lvalue: 1 : index\l|{<res_c1> %c1 index}}", shape = Mrecord, style = filled];
//       DFG:     }
//       DFG:   }
//       DFG:   v6:res_c0:s -> v5:arg_c0:n[style = solid];
//       DFG:   v7:res_c1:s -> v5:arg_c1:n[style = solid];
//       DFG: }

module {
  %add = arith.addi %c0, %c1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
}
