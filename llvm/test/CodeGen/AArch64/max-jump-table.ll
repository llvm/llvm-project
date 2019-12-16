; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40                         -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK0  < %t
; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40 -max-jump-table-size=4  -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK4  < %t
; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40 -max-jump-table-size=8  -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK8  < %t
; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40 -max-jump-table-size=16 -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK16 < %t
; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40 -mcpu=exynos-m3         -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECKM3 < %t

declare void @ext(i32, i32)

define i32 @jt1(i32 %a, i32 %b) {
entry:
  switch i32 %a, label %return [
    i32 1,  label %bb1
    i32 2,  label %bb2
    i32 3,  label %bb3
    i32 4,  label %bb4
    i32 5,  label %bb5
    i32 6,  label %bb6
    i32 7,  label %bb7
    i32 8,  label %bb8
    i32 9,  label %bb9
    i32 10, label %bb10
    i32 11, label %bb11
    i32 12, label %bb12
    i32 13, label %bb13
    i32 14, label %bb14
    i32 15, label %bb15
    i32 16, label %bb16
    i32 17, label %bb17
  ]
; CHECK-LABEL: function jt1:
; CHECK-NEXT: Jump Tables:
; CHECK0-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK0-NOT:   %jump-table.1:
; CHECK4-NEXT:  %jump-table.0: %bb.2 %bb.3 %bb.4 %bb.5
; CHECK4-NEXT:  %jump-table.1: %bb.6 %bb.7 %bb.8 %bb.9
; CHECK4-NEXT:  %jump-table.2: %bb.10 %bb.11 %bb.12 %bb.13
; CHECK4-NEXT:  %jump-table.3: %bb.14 %bb.15 %bb.16 %bb.17
; CHECK4-NOT:   %jump-table.4:
; CHECK8-NEXT:  %jump-table.0: %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9
; CHECK8-NEXT:  %jump-table.1: %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK8-NOT:   %jump-table.2:
; CHECK16-NEXT: %jump-table.0: %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK16-NOT:  %jump-table.1:
; CHECKM3-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECKM3-NOT:  %jump-table.1:

bb1:  tail call void @ext(i32 1, i32 0)  br label %return
bb2:  tail call void @ext(i32 2, i32 2)  br label %return
bb3:  tail call void @ext(i32 3, i32 4)  br label %return
bb4:  tail call void @ext(i32 4, i32 6)  br label %return
bb5:  tail call void @ext(i32 5, i32 8)  br label %return
bb6:  tail call void @ext(i32 6, i32 10) br label %return
bb7:  tail call void @ext(i32 7, i32 12) br label %return
bb8:  tail call void @ext(i32 8, i32 14) br label %return
bb9:  tail call void @ext(i32 9, i32 16) br label %return
bb10: tail call void @ext(i32 1, i32 18) br label %return
bb11: tail call void @ext(i32 2, i32 20) br label %return
bb12: tail call void @ext(i32 3, i32 22) br label %return
bb13: tail call void @ext(i32 4, i32 24) br label %return
bb14: tail call void @ext(i32 5, i32 26) br label %return
bb15: tail call void @ext(i32 6, i32 28) br label %return
bb16: tail call void @ext(i32 7, i32 30) br label %return
bb17: tail call void @ext(i32 8, i32 32) br label %return

return: ret i32 %b
}

define void @jt2(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 1,  label %bb1
    i32 2,  label %bb2
    i32 3,  label %bb3
    i32 4,  label %bb4

    i32 14, label %bb5
    i32 15, label %bb6
  ]
; CHECK-LABEL: function jt2:
; CHECK-NEXT: Jump Tables:
; CHECK0-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.5 %bb.6{{$}}
; CHECK0-NOT:   %jump-table.1:
; CHECK4-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4{{$}}
; CHECK4-NOT:   %jump-table.1:
; CHECK8-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4{{$}}
; CHECK8-NOT:   %jump-table.1:
; CHECK16-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.5 %bb.6{{$}}
; CHECK16-NOT:  %jump-table.1:
; CHECKM3-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.7 %bb.5 %bb.6{{$}}
; CHECKM3-NOT:  %jump-table.1:
; CHECK-DAG: End machine code for function jt2.

bb1: tail call void @ext(i32 6, i32 1) br label %return
bb2: tail call void @ext(i32 5, i32 2) br label %return
bb3: tail call void @ext(i32 4, i32 3) br label %return
bb4: tail call void @ext(i32 3, i32 4) br label %return
bb5: tail call void @ext(i32 2, i32 5) br label %return
bb6: tail call void @ext(i32 1, i32 6) br label %return
return: ret void
}

define void @jt3(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 1,  label %bb1
    i32 2,  label %bb2
    i32 3,  label %bb3
    i32 4,  label %bb4

    i32 14, label %bb5
    i32 15, label %bb6
    i32 16, label %bb7
    i32 17, label %bb8

    i32 19, label %bb9
    i32 20, label %bb10

    i32 22, label %bb11
    i32 23, label %bb12
  ]
; CHECK-LABEL: function jt3:
; CHECK-NEXT: Jump Tables:
; CHECK0-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.5 %bb.6 %bb.7 %bb.8 %bb.13 %bb.9 %bb.10 %bb.13 %bb.11 %bb.12
; CHECK0-NOT:   %jump-table.1:
; CHECK4-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4
; CHECK4-NEXT:  %jump-table.1: %bb.5 %bb.6 %bb.7 %bb.8
; CHECK4-NOT:   %jump-table.2:
; CHECK8-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4
; CHECK8-NEXT:  %jump-table.1: %bb.5 %bb.6 %bb.7 %bb.8 %bb.13 %bb.9 %bb.10
; CHECK8-NOT:   %jump-table.2:
; CHECK16-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.5 %bb.6 %bb.7
; CHECK16-NEXT: %jump-table.1: %bb.8 %bb.13 %bb.9 %bb.10 %bb.13 %bb.11 %bb.12
; CHECK16-NOT:  %jump-table.2:
; CHECKM3-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.5 %bb.6 %bb.7 %bb.8 %bb.13 %bb.9 %bb.10
; CHECKM3-NOT:  %jump-table.1:
; CHECK-DAG: End machine code for function jt3.

bb1:  tail call void @ext(i32 1,  i32 12) br label %return
bb2:  tail call void @ext(i32 2,  i32 11) br label %return
bb3:  tail call void @ext(i32 3,  i32 10) br label %return
bb4:  tail call void @ext(i32 4,  i32 9)  br label %return
bb5:  tail call void @ext(i32 5,  i32 8)  br label %return
bb6:  tail call void @ext(i32 6,  i32 7)  br label %return
bb7:  tail call void @ext(i32 7,  i32 6)  br label %return
bb8:  tail call void @ext(i32 8,  i32 5)  br label %return
bb9:  tail call void @ext(i32 9,  i32 4)  br label %return
bb10: tail call void @ext(i32 10, i32 3)  br label %return
bb11: tail call void @ext(i32 11, i32 2)  br label %return
bb12: tail call void @ext(i32 12, i32 1)  br label %return

return: ret void
}

define void @jt4(i32 %x) {
entry:
  switch i32 %x, label %default [
    i32 1,  label %bb1
    i32 2,  label %bb2
    i32 3,  label %bb3
    i32 4,  label %bb4

    i32 14, label %bb5
    i32 15, label %bb6
    i32 16, label %bb7
    i32 17, label %bb8

    i32 19, label %bb9
    i32 20, label %bb10

    i32 22, label %bb11
    i32 23, label %bb12
  ]
; CHECK-LABEL: function jt4:
; CHECK-NEXT: Jump Tables:
; CHECK0-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.5 %bb.6 %bb.7 %bb.8 %bb.13 %bb.9 %bb.10 %bb.13 %bb.11 %bb.12
; CHECK0-NOT:   %jump-table.1:
; CHECK4-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4
; CHECK4-NEXT:  %jump-table.1: %bb.5 %bb.6 %bb.7 %bb.8
; CHECK4-NOT:   %jump-table.2:
; CHECK8-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4
; CHECK8-NEXT:  %jump-table.1: %bb.5 %bb.6 %bb.7 %bb.8 %bb.13 %bb.9 %bb.10
; CHECK8-NOT:   %jump-table.2:
; CHECK16-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.5 %bb.6 %bb.7
; CHECK16-NEXT: %jump-table.1: %bb.8 %bb.13 %bb.9 %bb.10 %bb.13 %bb.11 %bb.12
; CHECK16-NOT:  %jump-table.2:
; CHECKM3-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.13 %bb.5 %bb.6 %bb.7 %bb.8 %bb.13 %bb.9 %bb.10
; CHECKM3-NOT:  %jump-table.1:
; CHECK-DAG: End machine code for function jt4.

bb1:  tail call void @ext(i32 1,  i32 12) br label %return
bb2:  tail call void @ext(i32 2,  i32 11) br label %return
bb3:  tail call void @ext(i32 3,  i32 10) br label %return
bb4:  tail call void @ext(i32 4,  i32 9)  br label %return
bb5:  tail call void @ext(i32 5,  i32 8)  br label %return
bb6:  tail call void @ext(i32 6,  i32 7)  br label %return
bb7:  tail call void @ext(i32 7,  i32 6)  br label %return
bb8:  tail call void @ext(i32 8,  i32 5)  br label %return
bb9:  tail call void @ext(i32 9,  i32 4)  br label %return
bb10: tail call void @ext(i32 10, i32 3)  br label %return
bb11: tail call void @ext(i32 11, i32 2)  br label %return
bb12: tail call void @ext(i32 12, i32 1)  br label %return
default: unreachable

return: ret void
}

define i32 @jt1_optsize(i32 %a, i32 %b) optsize {
entry:
  switch i32 %a, label %return [
    i32 1,  label %bb1
    i32 2,  label %bb2
    i32 3,  label %bb3
    i32 4,  label %bb4
    i32 5,  label %bb5
    i32 6,  label %bb6
    i32 7,  label %bb7
    i32 8,  label %bb8
    i32 9,  label %bb9
    i32 10, label %bb10
    i32 11, label %bb11
    i32 12, label %bb12
    i32 13, label %bb13
    i32 14, label %bb14
    i32 15, label %bb15
    i32 16, label %bb16
    i32 17, label %bb17
  ]
; CHECK-LABEL: function jt1_optsize:
; CHECK-NEXT: Jump Tables:
; CHECK0-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK0-NOT:   %jump-table.1:
; CHECK4-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK4-NOT:   %jump-table.1:
; CHECK8-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK8-NOT:   %jump-table.1:
; CHECK16-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK16-NOT:  %jump-table.1:
; CHECKM1-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECKM1-NOT:  %jump-table.1:
; CHECKM3-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECKM3-NOT:  %jump-table.1:
; CHECK-DAG: End machine code for function jt1_optsize.

bb1:  tail call void @ext(i32 1, i32 0)  br label %return
bb2:  tail call void @ext(i32 2, i32 2)  br label %return
bb3:  tail call void @ext(i32 3, i32 4)  br label %return
bb4:  tail call void @ext(i32 4, i32 6)  br label %return
bb5:  tail call void @ext(i32 5, i32 8)  br label %return
bb6:  tail call void @ext(i32 6, i32 10) br label %return
bb7:  tail call void @ext(i32 7, i32 12) br label %return
bb8:  tail call void @ext(i32 8, i32 14) br label %return
bb9:  tail call void @ext(i32 9, i32 16) br label %return
bb10: tail call void @ext(i32 1, i32 18) br label %return
bb11: tail call void @ext(i32 2, i32 20) br label %return
bb12: tail call void @ext(i32 3, i32 22) br label %return
bb13: tail call void @ext(i32 4, i32 24) br label %return
bb14: tail call void @ext(i32 5, i32 26) br label %return
bb15: tail call void @ext(i32 6, i32 28) br label %return
bb16: tail call void @ext(i32 7, i32 30) br label %return
bb17: tail call void @ext(i32 8, i32 32) br label %return

return: ret i32 %b
}

define i32 @jt1_pgso(i32 %a, i32 %b) !prof !14 {
entry:
  switch i32 %a, label %return [
    i32 1,  label %bb1
    i32 2,  label %bb2
    i32 3,  label %bb3
    i32 4,  label %bb4
    i32 5,  label %bb5
    i32 6,  label %bb6
    i32 7,  label %bb7
    i32 8,  label %bb8
    i32 9,  label %bb9
    i32 10, label %bb10
    i32 11, label %bb11
    i32 12, label %bb12
    i32 13, label %bb13
    i32 14, label %bb14
    i32 15, label %bb15
    i32 16, label %bb16
    i32 17, label %bb17
  ]
; CHECK-LABEL: function jt1_pgso:
; CHECK-NEXT: Jump Tables:
; CHECK0-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK0-NOT:   %jump-table.1:
; CHECK4-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK4-NOT:   %jump-table.1:
; CHECK8-NEXT:  %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK8-NOT:   %jump-table.1:
; CHECK16-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECK16-NOT:  %jump-table.1:
; CHECKM1-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECKM1-NOT:  %jump-table.1:
; CHECKM3-NEXT: %jump-table.0: %bb.1 %bb.2 %bb.3 %bb.4 %bb.5 %bb.6 %bb.7 %bb.8 %bb.9 %bb.10 %bb.11 %bb.12 %bb.13 %bb.14 %bb.15 %bb.16 %bb.17
; CHECKM3-NOT:  %jump-table.1:
; CHECK-DAG: End machine code for function jt1_pgso.

bb1:  tail call void @ext(i32 1, i32 0)  br label %return
bb2:  tail call void @ext(i32 2, i32 2)  br label %return
bb3:  tail call void @ext(i32 3, i32 4)  br label %return
bb4:  tail call void @ext(i32 4, i32 6)  br label %return
bb5:  tail call void @ext(i32 5, i32 8)  br label %return
bb6:  tail call void @ext(i32 6, i32 10) br label %return
bb7:  tail call void @ext(i32 7, i32 12) br label %return
bb8:  tail call void @ext(i32 8, i32 14) br label %return
bb9:  tail call void @ext(i32 9, i32 16) br label %return
bb10: tail call void @ext(i32 1, i32 18) br label %return
bb11: tail call void @ext(i32 2, i32 20) br label %return
bb12: tail call void @ext(i32 3, i32 22) br label %return
bb13: tail call void @ext(i32 4, i32 24) br label %return
bb14: tail call void @ext(i32 5, i32 26) br label %return
bb15: tail call void @ext(i32 6, i32 28) br label %return
bb16: tail call void @ext(i32 7, i32 30) br label %return
bb17: tail call void @ext(i32 8, i32 32) br label %return

return: ret i32 %b
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
