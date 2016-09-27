; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40                   -o - 2>%t; FileCheck %s --check-prefixes=CHECK,CHECK0  <%t
; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40 -max-jump-table=4 -o - 2>%t; FileCheck %s --check-prefixes=CHECK,CHECK4  <%t
; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40 -max-jump-table=8 -o - 2>%t; FileCheck %s --check-prefixes=CHECK,CHECK8  <%t
; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40 -mcpu=exynos-m1   -o - 2>%t; FileCheck %s --check-prefixes=CHECK,CHECKM1 <%t

declare void @ext(i32)

define i32 @jt1(i32 %a, i32 %b) {
entry:
  switch i32 %a, label %return [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb5
    i32 6, label %bb6
    i32 7, label %bb7
    i32 8, label %bb8
    i32 9, label %bb9
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
; CHECK: Jump Tables:
; CHECK0-NEXT: jt#0:
; CHECK0-NOT: jt#1:
; CHECK4-NEXT: jt#0:
; CHECK4-SAME: jt#1:
; CHECK4-SAME: jt#2:
; CHECK4-SAME: jt#3:
; CHECK4-NOT: jt#4:
; CHECK8-NEXT: jt#0:
; CHECK8-SAME: jt#1:
; CHECK8-SAME: jt#2: BB#14 BB#15 BB#16 BB#17{{$}}
; CHECK8-NOT: jt#3:
; CHECKM1-NEXT: jt#0:
; CHECKM1-SAME: jt#1: BB#13 BB#14 BB#15 BB#16 BB#17{{$}}
; CHECKM1-NOT: jt#2:
; CHEC-NEXT: Function Live Ins:

bb1: tail call void @ext(i32 0) br label %return
bb2: tail call void @ext(i32 2) br label %return
bb3: tail call void @ext(i32 4) br label %return
bb4: tail call void @ext(i32 6) br label %return
bb5: tail call void @ext(i32 8) br label %return
bb6: tail call void @ext(i32 10) br label %return
bb7: tail call void @ext(i32 12) br label %return
bb8: tail call void @ext(i32 14) br label %return
bb9: tail call void @ext(i32 16) br label %return
bb10: tail call void @ext(i32 18) br label %return
bb11: tail call void @ext(i32 20) br label %return
bb12: tail call void @ext(i32 22) br label %return
bb13: tail call void @ext(i32 24) br label %return
bb14: tail call void @ext(i32 26) br label %return
bb15: tail call void @ext(i32 28) br label %return
bb16: tail call void @ext(i32 30) br label %return
bb17: tail call void @ext(i32 32) br label %return

return: ret i32 %b
}

define void @jt2(i32 %x) {
entry:
  switch i32 %x, label %return [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4

    i32 14, label %bb5
    i32 15, label %bb6
  ]
; CHECK-LABEL: function jt2:
; CHECK: Jump Tables:
; CHECK0-NEXT: jt#0:  BB#1 BB#2 BB#3 BB#4 BB#7 BB#7 BB#7 BB#7 BB#7 BB#7 BB#7 BB#7 BB#7 BB#5 BB#6{{$}}
; CHECK4-NEXT: jt#0:  BB#1 BB#2 BB#3 BB#4{{$}}
; CHECK8-NEXT: jt#0:  BB#1 BB#2 BB#3 BB#4{{$}}
; CHECKM1-NEXT: jt#0:  BB#1 BB#2 BB#3 BB#4{{$}}
; CHEC-NEXT: Function Live Ins:

bb1: tail call void @ext(i32 1) br label %return
bb2: tail call void @ext(i32 2) br label %return
bb3: tail call void @ext(i32 3) br label %return
bb4: tail call void @ext(i32 4) br label %return
bb5: tail call void @ext(i32 5) br label %return
bb6: tail call void @ext(i32 6) br label %return
return: ret void
}
