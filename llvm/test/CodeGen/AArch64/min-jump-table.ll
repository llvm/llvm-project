; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40 -min-jump-table-entries=0 -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK0  < %t
; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40 -min-jump-table-entries=4 -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK4  < %t
; RUN: llc %s -O2 -print-machineinstrs -mtriple=aarch64-linux-gnu -jump-table-density=40 -min-jump-table-entries=8 -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK8  < %t

declare void @ext(i32)

define i32 @jt2(i32 %a, i32 %b) {
entry:
  switch i32 %a, label %return [
    i32 1, label %bb1
    i32 2, label %bb2
  ]
; CHECK-LABEL: function jt2:
; CHECK0-NEXT: Jump Tables:
; CHECK0-NEXT: jt#0:
; CHECK0-NOT: jt#1:
; CHECK4-NOT: Jump Tables:
; CHECK8-NOT: Jump Tables:

bb1: tail call void @ext(i32 0) br label %return
bb2: tail call void @ext(i32 2) br label %return

return: ret i32 %b
}

define i32 @jt4(i32 %a, i32 %b) {
entry:
  switch i32 %a, label %return [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
  ]
; CHECK-LABEL: function jt4:
; CHECK0-NEXT: Jump Tables:
; CHECK0-NEXT: jt#0:
; CHECK0-NOT: jt#1:
; CHECK4-NEXT: Jump Tables:
; CHECK4-NEXT: jt#0:
; CHECK4-NOT: jt#1:
; CHECK8-NOT: Jump Tables:

bb1: tail call void @ext(i32 0) br label %return
bb2: tail call void @ext(i32 2) br label %return
bb3: tail call void @ext(i32 4) br label %return
bb4: tail call void @ext(i32 6) br label %return

return: ret i32 %b
}

define i32 @jt8(i32 %a, i32 %b) {
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
  ]
; CHECK-LABEL: function jt8:
; CHECK-NEXT: Jump Tables:
; CHECK-NEXT: jt#0:
; CHECK-NOT: jt#1:

bb1: tail call void @ext(i32 0) br label %return
bb2: tail call void @ext(i32 2) br label %return
bb3: tail call void @ext(i32 4) br label %return
bb4: tail call void @ext(i32 6) br label %return
bb5: tail call void @ext(i32 8) br label %return
bb6: tail call void @ext(i32 10) br label %return
bb7: tail call void @ext(i32 12) br label %return
bb8: tail call void @ext(i32 14) br label %return

return: ret i32 %b
}

