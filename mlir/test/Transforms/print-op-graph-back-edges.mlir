// RUN: mlir-opt -view-op-graph %s -o %t 2>&1 | FileCheck -check-prefix=DFG %s

// DFG-LABEL: digraph G {
//       DFG:   compound = true;
//       DFG:   subgraph cluster_1 {
//       DFG:     v2 [label = " ", shape = plain];
//       DFG:     label = "builtin.module : ()\n";
//       DFG:     subgraph cluster_3 {
//       DFG:       v4 [label = " ", shape = plain];
//       DFG:       label = "";
//       DFG:       v5 [fillcolor = "0.000000 1.0 1.0", label = "arith.addi : (index)\n\noverflowFlags: #arith.overflow<none...", shape = ellipse, style = filled];
//       DFG:       v6 [fillcolor = "0.333333 1.0 1.0", label = "arith.constant : (index)\n\nvalue: 0 : index", shape = ellipse, style = filled];
//       DFG:       v7 [fillcolor = "0.333333 1.0 1.0", label = "arith.constant : (index)\n\nvalue: 1 : index", shape = ellipse, style = filled];
//       DFG:     }
//       DFG:   }
//       DFG:   v6 -> v5 [label = "0", style = solid];
//       DFG:   v7 -> v5 [label = "1", style = solid];
//       DFG: }

module {
  %add = arith.addi %c0, %c1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
}
