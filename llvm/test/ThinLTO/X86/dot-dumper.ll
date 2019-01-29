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

; Never assume specific order of clusters, nodes or edges
; RUN: cat %t1.dot | FileCheck --check-prefix=STRUCTURE1 %s
; RUN: cat %t1.dot | FileCheck --check-prefix=CLUSTER0 --check-prefix=PERMODULE0 %s
; RUN: cat %t2.dot | FileCheck --check-prefix=STRUCTURE2 %s
; RUN: cat %t2.dot | FileCheck --check-prefix=CLUSTER1 --check-prefix=PERMODULE1 %s
; RUN: cat %t3.index.dot | FileCheck --check-prefix=STRUCTURE %s
; RUN: cat %t3.index.dot | FileCheck --check-prefix=CLUSTER0 --check-prefix=COMBINED0 %s
; RUN: cat %t3.index.dot | FileCheck --check-prefix=CLUSTER1 --check-prefix=COMBINED1 %s

; %t1 index
; STRUCTURE1:        digraph Summary {
; STRUCTURE1:          subgraph cluster_0
; STRUCTURE1:          // Cross-module edges:
; STRUCTURE1:          0 [label="@0"]; // defined externally
; STRUCTURE1:          M0_{{[0-9]+}} -> 0 [style=dotted]; // alias
; STRUCTURE1-DAG:      [[A:[0-9]+]] [label="A"]; // defined externally
; STRUCTURE1-DAG:      [[FOO:[0-9]+]] [label="foo"]; // defined externally
; STRUCTURE1-DAG:      M0_{{[0-9]+}} -> [[FOO]] // call
; STRUCTURE1-DAG:      M0_{{[0-9]+}} -> [[A]] [{{.*}}]; // const-ref
; STRUCTURE1-NEXT:   }

; %t2 index
; STRUCTURE2:        digraph Summary {
; STRUCTURE2:          subgraph cluster_0
; STRUCTURE2:          // Cross-module edges:
; STRUCTURE2-NEXT:   }

; Combined index
; STRUCTURE:        digraph Summary {
; STRUCTURE-DAG:      subgraph cluster_0
; STRUCTURE-DAG:      subgraph cluster_1
; STRUCTURE:          // Cross-module edges:
; STRUCTURE-DAG:      M0_{{[0-9]+}} -> M1_{{[0-9]+}} // call
; STRUCTURE-DAG:      M0_{{[0-9]+}} -> M1_{{[0-9]+}} [{{.*}}]; // const-ref
; STRUCTURE-NEXT:   }

; PERMODULE0:       // Module:
; COMBINED0:	    // Module: {{.*}}1.bc
; CLUSTER0-NEXT:    subgraph cluster_[[ID0:[0-1]]] {
; CLUSTER0-DAG:       M[[ID0]]_[[MAIN_ALIAS:[0-9]+]] [{{.*}}main_alias{{.*}}]; // alias, dead
; CLUSTER0-DAG:       M[[ID0]]_[[MAIN:[0-9]+]] [{{.*}}main|extern{{.*}}]; // function
; CLUSTER0-NEXT:      // Edges:
; COMBINED0-NEXT:     M[[ID0]]_[[MAIN_ALIAS]] -> M[[ID0]]_[[MAIN]] [{{.*}}]; // alias
; CLUSTER0-NEXT:    }

; PERMODULE1:       // Module:
; COMBINED1:	    // Module: {{.*}}2.bc
; CLUSTER1-NEXT:    subgraph cluster_[[ID1:[0-1]]] {
; CLUSTER1-DAG:       M[[ID1]]_[[A:[0-9]+]] [{{.*}}A|extern{{.*}}]; // variable
; COMBINED1-SAME:	, immutable
; CLUSTER1-DAG:       M[[ID1]]_[[FOO:[0-9]+]] [{{.*}}foo|extern{{.*}} ffl: 00001{{.*}}]; // function
; CLUSTER1-DAG:       M[[ID1]]_[[B:[0-9]+]] [{{.*}}B|extern{{.*}}]; // variable
; COMBINED1-SAME:	, immutable
; CLUSTER1-DAG:       M[[ID1]]_[[BAR:[0-9]+]] [{{.*}}bar|extern{{.*}}]; // function, dead
; CLUSTER1-NEXT:      // Edges:
; CLUSTER1-DAG:       M[[ID1]]_[[FOO]] -> M[[ID1]]_[[B]] [{{.*}}]; // const-ref
; CLUSTER1-DAG:       M[[ID1]]_[[FOO]] -> M[[ID1]]_[[A]] [{{.*}}]; // const-ref
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
