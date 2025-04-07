

; This RUN command sets `-data-sections=true -unique-section-names=true` so data
; sections are uniqufied by numbers.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions \
; RUN:     -partition-static-data-sections=true -data-sections=true \
; RUN:     -unique-section-names=true -relocation-model=pic \
; RUN:     -global-var-ref-graph-dot-file=%t1.dot \
; RUN:     %s -o - 2>&1 

;; COM: Test section prefixes after giving the data a hotness.

; RUN: cat %t1.dot | FileCheck --check-prefix=DOT %s

; DOT:      digraph {
; DOT-NEXT:     1 [label="scc1_var1", style=filled, fillcolor="lightgrey", shape="ellipse"]
; DOT-NEXT:     2 [label="scc1_var2", style=filled, fillcolor="lightgrey", shape="ellipse"]
; DOT-NEXT:     3 [label="scc2_var1", style=filled, fillcolor="lightgrey", shape="ellipse"]
; DOT-NEXT:     4 [label="scc2_var3", style=filled, fillcolor="lightgrey", shape="ellipse"]
; DOT-NEXT:     5 [label="scc2_var2", style=filled, fillcolor="lightgrey", shape="ellipse"]
; DOT-NEXT:     6 [label="scc3_var", style=filled, fillcolor="lightgrey", shape="ellipse"]
; DOT-NEXT:     7 [label="scc4_var", style=filled, fillcolor="lightgrey", shape="ellipse"]
; DOT-NEXT:     8 [label="scc5_var1", style=filled, fillcolor="lightgrey", shape="ellipse"]
; DOT-NEXT:     9 [label="scc5_var2", style=filled, fillcolor="lightgrey", shape="ellipse"]
; DOT-NEXT:     1 -> 2
; DOT-NEXT:     2 -> 1
; DOT-NEXT:     3 -> 2
; DOT-NEXT:     3 -> 4
; DOT-NEXT:     4 -> 5
; DOT-NEXT:     5 -> 3
; DOT-NEXT:     6 -> 5
; DOT-NEXT:     7 -> 5
; DOT-NEXT:     8 -> 9
; DOT-NEXT:     9 -> 8
; DOT-NEXT: }


; TODO: Make this cold scc
@scc1_var1 = internal constant [1 x ptr][ptr @scc1_var2]
@scc1_var2 = internal constant [2 x ptr][ptr @scc1_var1, ptr @scc2_var1]

; TODO: In this scc, one hot, one cold, one unknown
@scc2_var1 = internal constant [1 x ptr][ptr @scc2_var2]
@scc2_var2 = internal constant [3 x ptr][ptr @scc2_var3, ptr @scc3_var, ptr @scc4_var]
@scc2_var3 = internal constant [1 x ptr][ptr @scc2_var1]

@scc3_var = internal constant i32 12345

@scc4_var = private constant [5 x i8] c "abcde"

; Have a un-named constant in the middle in this scc graph 
@scc5_var1 = internal constant [1 x ptr][ptr @scc5_var2]
@scc5_var2 = internal constant [1 x ptr][ptr @scc5_var1]


!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 1460183}
!5 = !{!"MaxCount", i64 849024}
!6 = !{!"MaxInternalCount", i64 32769}
!7 = !{!"MaxFunctionCount", i64 849024}
!8 = !{!"NumCounts", i64 23627}
!9 = !{!"NumFunctions", i64 3271}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13}
!12 = !{i32 990000, i64 166, i32 73}
!13 = !{i32 999999, i64 3, i32 1443}
!14 = !{!"function_entry_count", i64 100000}
!15 = !{!"function_entry_count", i64 1}
!16 = !{!"branch_weights", i32 1, i32 99999}
