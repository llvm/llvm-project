; RUN: llc -mtriple=x86_64-linux-gnu %s -o - -jump-table-density=25 | FileCheck %s --check-prefix=DENSE --check-prefix=CHECK
; RUN: llc -mtriple=x86_64-linux-gnu %s -o - -jump-table-density=10 | FileCheck %s --check-prefix=SPARSE --check-prefix=CHECK

declare void @g(i32)

define void @sparse(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 300, label %bb0
    i32 100, label %bb1
    i32 400, label %bb1
    i32 500, label %bb2
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 1) br label %return
return: ret void

; Should pivot around 400 for two subtrees with two jump tables each.
; CHECK-LABEL: sparse
; CHECK-NOT: cmpl
; CHECK: cmpl $399
; CHECK: cmpl $100
; CHECK: cmpl $300
; CHECK: cmpl $400
; CHECK: cmpl $500
}

define void @med(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 30, label %bb0
    i32 10, label %bb1
    i32 40, label %bb1
    i32 50, label %bb2
    i32 20, label %bb3
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 1) br label %return
bb3: tail call void @g(i32 2) br label %return
return: ret void

; Lowered as a jump table when sparse, and branches when dense.
; CHECK-LABEL: med
; SPARSE: addl $-10
; SPARSE: cmpl $40
; SPARSE: ja
; SPARSE: jmpq *.LJTI
; DENSE-NOT: cmpl
; DENSE: cmpl $29
; DENSE-DAG: cmpl $10
; DENSE-DAG: cmpl $20
; DENSE-DAG: cmpl $30
; DENSE-DAG: cmpl $40
; DENSE-DAG: cmpl $50
; DENSE: retq
}

define void @dense(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 12, label %bb0
    i32 4,  label %bb1
    i32 16, label %bb1
    i32 20, label %bb2
    i32 8,  label %bb3
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 1) br label %return
bb3: tail call void @g(i32 2) br label %return
return: ret void

; Lowered as a jump table when sparse, and branches when dense.
; CHECK-LABEL: dense
; CHECK: addl $-4
; CHECK: cmpl $16
; CHECK: ja
; CHECK: jmpq *.LJTI
}

define void @dense_optsize(i32 %x) optsize {
entry:
  switch i32 %x, label %return [
    i32 12, label %bb0
    i32 4,  label %bb1
    i32 16, label %bb1
    i32 20, label %bb2
    i32 8,  label %bb3
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 1) br label %return
bb3: tail call void @g(i32 2) br label %return
return: ret void

; Lowered as branches.
; CHECK-LABEL: dense_optsize
; CHECK: cmpl $11
; CHECK: cmpl $20
; CHECK: cmpl $16
; CHECK: cmpl $12
; CHECK: cmpl $4
; CHECK: cmpl $8
; CHECK: retq
}

define void @dense_pgso(i32 %x) !prof !14 {
entry:
  switch i32 %x, label %return [
    i32 12, label %bb0
    i32 4,  label %bb1
    i32 16, label %bb1
    i32 20, label %bb2
    i32 8,  label %bb3
  ]
bb0: tail call void @g(i32 0) br label %return
bb1: tail call void @g(i32 1) br label %return
bb2: tail call void @g(i32 1) br label %return
bb3: tail call void @g(i32 2) br label %return
return: ret void

; Lowered as branches.
; CHECK-LABEL: dense_pgso
; CHECK: cmpl $11
; CHECK: cmpl $20
; CHECK: cmpl $16
; CHECK: cmpl $12
; CHECK: cmpl $4
; CHECK: cmpl $8
; CHECK: retq
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
