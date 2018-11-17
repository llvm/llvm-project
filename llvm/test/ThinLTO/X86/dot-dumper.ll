; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/dot-dumper.ll -o %t2.bc
; RUN: llvm-lto2 run -save-temps %t1.bc %t2.bc -o %t3 \
; RUN:  -r=%t1.bc,main,px \
; RUN:  -r=%t1.bc,main_alias,p \
; RUN:  -r=%t1.bc,foo, \
; RUN:  -r=%t1.bc,A, \
; RUN:  -r=%t2.bc,foo,p \
; RUN:  -r=%t2.bc,bar,p \
; RUN:  -r=%t2.bc,A,p \
; RUN:  -r=%t2.bc,B,p

; Never assume specific order of clusters, nodes or edges
; RUN: cat %t3.index.dot | FileCheck --check-prefix=STRUCTURE %s
; RUN: cat %t3.index.dot | FileCheck --check-prefix=CLUSTER0 %s
; RUN: cat %t3.index.dot | FileCheck --check-prefix=CLUSTER1 %s

; STRUCTURE:        digraph Summary {
; STRUCTURE-DAG:      subgraph cluster_0
; STRUCTURE-DAG:      subgraph cluster_1
; STRUCTURE:          // Cross-module edges:
; STRUCTURE-DAG:      M0_{{[0-9]+}} -> M1_{{[0-9]+}} // call
; STRUCTURE-DAG:      M0_{{[0-9]+}} -> M1_{{[0-9]+}} [{{.*}}]; // const-ref
; STRUCTURE-NEXT:   }

; CLUSTER0:         // Module: {{.*}}1.bc
; CLUSTER0-NEXT:    subgraph cluster_0 {
; CLUSTER0-DAG:       M0_[[MAIN_ALIAS:[0-9]+]] [{{.*}}main_alias{{.*}}]; // alias, dead
; CLUSTER0-DAG:       M0_[[MAIN:[0-9]+]] [{{.*}}main|extern{{.*}}]; // function
; CLUSTER0-NEXT:      // Edges:
; CLUSTER0-NEXT:      M0_[[MAIN_ALIAS]] -> M0_[[MAIN]] [{{.*}}]; // alias
; CLUSTER0-NEXT:    }

; CLUSTER1:         // Module: {{.*}}2.bc
; CLUSTER1-NEXT:    subgraph cluster_1 {
; CLUSTER1-DAG:       M1_[[A:[0-9]+]] [{{.*}}A|extern{{.*}}]; // variable, immutable
; CLUSTER1-DAG:       M1_[[FOO:[0-9]+]] [{{.*}}foo|extern{{.*}} ffl: 00001{{.*}}]; // function
; CLUSTER1-DAG:       M1_[[B:[0-9]+]] [{{.*}}B|extern{{.*}}]; // variable, immutable
; CLUSTER1-DAG:       M1_[[BAR:[0-9]+]] [{{.*}}bar|extern{{.*}}]; // function, dead
; CLUSTER1-NEXT:      // Edges:
; CLUSTER1-DAG:       M1_[[FOO]] -> M1_[[B]] [{{.*}}]; // const-ref
; CLUSTER1-DAG:       M1_[[FOO]] -> M1_[[A]] [{{.*}}]; // const-ref
; CLUSTER1-DAG:     }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
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
