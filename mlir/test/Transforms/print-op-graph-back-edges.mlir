// RUN: mlir-opt -view-op-graph %s -o %t 2>&1 | FileCheck -check-prefix=DFG %s

// DFG-LABEL: digraph G {
//  DFG-NEXT:   compound = true;
//  DFG-NEXT:   subgraph cluster_1 {
//  DFG-NEXT:     v2 [label = " ", shape = plain];
//  DFG-NEXT:     label = "builtin.module : ()\l";
//  DFG-NEXT:     subgraph cluster_3 {
//  DFG-NEXT:       v4 [label = " ", shape = plain];
//  DFG-NEXT:       label = "";
//  DFG-NEXT:       v5 [fillcolor = "0.000000 0.3 0.95", label = "{{\{\{}}<arg_c0> %c0|<arg_c1> %c1}|arith.addi\l\loverflowFlags: #arith.overflow\<none...\l|{<res_0> %0 index}}", shape = Mrecord, style = filled];
//  DFG-NEXT:       v6 [fillcolor = "0.333333 0.3 0.95", label = "{arith.constant\l\lvalue: 0 : index\l|{<res_c0> %c0 index}}", shape = Mrecord, style = filled];
//  DFG-NEXT:       v7 [fillcolor = "0.333333 0.3 0.95", label = "{arith.constant\l\lvalue: 1 : index\l|{<res_c1> %c1 index}}", shape = Mrecord, style = filled];
//  DFG-NEXT:     }
//  DFG-NEXT:   }
//  DFG-NEXT:   v6:res_c0:s -> v5:arg_c0:n[style = solid];
//  DFG-NEXT:   v7:res_c1:s -> v5:arg_c1:n[style = solid];
//  DFG-NEXT: }

module {
  %add = arith.addi %c0, %c1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
}
