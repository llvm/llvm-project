; Test writeOnly attribute
; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/dot-dumper2.ll -o %t2.bc
; RUN: llvm-lto2 run -save-temps %t1.bc %t2.bc -o %t3 \
; RUN:  -r=%t1.bc,main,px \
; RUN:  -r=%t1.bc,A, \
; RUN:  -r=%t2.bc,A,p

; RUN: cat %t3.index.dot | FileCheck --check-prefix=COMBINED %s

; COMBINED: digraph Summary {
; COMBINED-NEXT:  // Module:
; COMBINED-NEXT:  subgraph cluster_0 {
; COMBINED-NEXT:    style = filled;
; COMBINED-NEXT:    color = lightgrey;
; COMBINED-NEXT:    label =
; COMBINED-NEXT:    node [style=filled,fillcolor=lightblue];
; COMBINED-NEXT:    M0_[[MAIN:[0-9]+]] [shape="record",label="main|extern (inst: 2, ffl: 0000000000)}"]; // function
; COMBINED-NEXT:    // Edges:
; COMBINED-NEXT:  }
; COMBINED-NEXT:  // Module:
; COMBINED-NEXT:  subgraph cluster_1 {
; COMBINED-NEXT:    style = filled;
; COMBINED-NEXT:    color = lightgrey;
; COMBINED-NEXT:    label =
; COMBINED-NEXT:    node [style=filled,fillcolor=lightblue];
; COMBINED-NEXT:    M1_[[A:[0-9]+]] [shape="Mrecord",label="A|extern}"]; // variable, writeOnly
; COMBINED-NEXT:    // Edges:
; COMBINED-NEXT:  }
; COMBINED-NEXT:  // Cross-module edges:
; COMBINED-NEXT:  M0_[[MAIN]] -> M1_[[A]] [style=dashed,color=violetred]; // writeOnly-ref
; COMBINED-NEXT: }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind uwtable
define i32 @main() local_unnamed_addr {
  store i32 42, i32* @A, align 4
  ret i32 0
}
