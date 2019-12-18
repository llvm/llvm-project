; RUN: opt -module-summary %s -o %t1.bc -module-summary-dot-file=%t1.dot
; RUN: opt -module-summary %p/Inputs/dot-dumper.ll -o %t2.bc -module-summary-dot-file=%t2.dot
; RUN: llvm-lto2 run -save-temps %t1.bc %t2.bc -o %t3 \
; RUN:  -r=%t1.bc,main,px \
; RUN:  -r=%t1.bc,main_alias,p \
; RUN:  -r=%t1.bc,foo, \
; RUN:  -r=%t1.bc,A, \
; RUN:  -r=%t2.bc,foo,p \
; RUN:  -r=%t2.bc,bar,p \
; RUN:  -r=%t2.bc,A,p \
; RUN:  -r=%t2.bc,B,p

; RUN: cat %t1.dot | FileCheck --check-prefix=PERMODULE %s
; RUN: cat %t3.index.dot | FileCheck --check-prefix=COMBINED %s

; PERMODULE: digraph Summary {
; PERMODULE-NEXT:  // Module: 
; PERMODULE-NEXT:  subgraph cluster_0 {
; PERMODULE-NEXT:    style = filled;
; PERMODULE-NEXT:    color = lightgrey;
; PERMODULE-NEXT:    label = "";
; PERMODULE-NEXT:    node [style=filled,fillcolor=lightblue];
; PERMODULE-NEXT:    M0_[[MAIN_ALIAS:[0-9]+]] [style="dotted,filled",shape="box",label="main_alias",fillcolor="red"]; // alias, dead
; PERMODULE-NEXT:    M0_[[MAIN:[0-9]+]] [shape="record",label="main|extern (inst: 4, ffl: 000000)}",fillcolor="red"]; // function, dead
; PERMODULE-NEXT:    // Edges:
; PERMODULE-NEXT:    M0_[[MAIN_ALIAS]] -> M0_[[MAIN]] [style=dotted]; // alias
; PERMODULE-NEXT:  }
; PERMODULE-NEXT:  // Cross-module edges:
; PERMODULE-NEXT:  [[A:[0-9]+]] [label="A"]; // defined externally
; PERMODULE-NEXT:  M0_[[MAIN]] -> [[A]] [style=dashed,color=forestgreen]; // const-ref
; PERMODULE-NEXT:  [[FOO:[0-9]+]] [label="foo"]; // defined externally
; PERMODULE-NEXT:  M0_[[MAIN]] -> [[FOO]] // call (hotness : Unknown)
; PERMODULE-NEXT: }

; COMBINED:     digraph Summary {
; COMBINED-NEXT:  // Module: {{.*}}dot-dumper{{.*}}1.bc
; COMBINED-NEXT:  subgraph cluster_0 {
; COMBINED-NEXT:    style = filled;
; COMBINED-NEXT:    color = lightgrey;
; COMBINED-NEXT:    label = "dot-dumper{{.*}}1.bc";
; COMBINED-NEXT:    node [style=filled,fillcolor=lightblue];
; COMBINED-NEXT:    M0_[[MAIN_ALIAS:[0-9]+]] [style="dotted,filled",shape="box",label="main_alias",fillcolor="red"]; // alias, dead
; COMBINED-NEXT:    M0_[[MAIN:[0-9]+]] [shape="record",label="main|extern (inst: 4, ffl: 000000)}"]; // function, preserved
; COMBINED-NEXT:    // Edges:
; COMBINED-NEXT:    M0_[[MAIN_ALIAS]] -> M0_[[MAIN]] [style=dotted]; // alias
; COMBINED-NEXT:  }
; COMBINED-NEXT:  // Module: {{.*}}dot-dumper{{.*}}2.bc
; COMBINED-NEXT:  subgraph cluster_1 {
; COMBINED-NEXT:    style = filled;
; COMBINED-NEXT:    color = lightgrey;
; COMBINED-NEXT:    label = "dot-dumper{{.*}}2.bc";
; COMBINED-NEXT:    node [style=filled,fillcolor=lightblue];
; COMBINED-NEXT:    M1_[[FOO:[0-9]+]] [shape="record",label="foo|extern (inst: 4, ffl: 000010)}"]; // function
; COMBINED-NEXT:    M1_[[A:[0-9]+]] [shape="Mrecord",label="A|extern}"]; // variable, immutable
; COMBINED-NEXT:    M1_[[B:[0-9]+]] [shape="Mrecord",label="B|extern}"]; // variable, immutable
; COMBINED-NEXT:    M1_{{[0-9]+}} [shape="record",label="bar|extern (inst: 1, ffl: 000000)}",fillcolor="red"]; // function, dead
; COMBINED-NEXT:    // Edges:
; COMBINED-NEXT:    M1_[[FOO]] -> M1_[[B]] [style=dashed,color=forestgreen]; // const-ref
; COMBINED-NEXT:    M1_[[FOO]] -> M1_[[A]] [style=dashed,color=forestgreen]; // const-ref
; COMBINED-NEXT:  }
; COMBINED-NEXT:  // Cross-module edges:
; COMBINED-NEXT:  M0_[[MAIN]] -> M1_[[A]] [style=dashed,color=forestgreen]; // const-ref
; COMBINED-NEXT:  M0_[[MAIN]] -> M1_[[FOO]] // call (hotness : Unknown)
; COMBINED-NEXT: }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind uwtable
define i32 @main() local_unnamed_addr {
  %1 = tail call i32 (...) @foo()
  %2 = load i32, i32* @A, align 4
  %3 = add nsw i32 %2, %1
  ret i32 %3
}
@main_alias = weak_odr alias i32 (), i32 ()* @main
declare i32 @foo(...) local_unnamed_addr
