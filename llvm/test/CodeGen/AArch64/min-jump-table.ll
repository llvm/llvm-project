; RUN: llc %s -O2 -print-after-isel -mtriple=aarch64-linux-gnu -jump-table-density=40 -aarch64-min-jump-table-entries=0 -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK0  < %t
; RUN: llc %s -O2 -print-after-isel -mtriple=aarch64-linux-gnu -jump-table-density=40 -aarch64-min-jump-table-entries=2 -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK2  < %t
; RUN: llc %s -O2 -print-after-isel -mtriple=aarch64-linux-gnu -jump-table-density=40 -aarch64-min-jump-table-entries=4 -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK4  < %t
; RUN: llc %s -O2 -print-after-isel -mtriple=aarch64-linux-gnu -jump-table-density=40 -aarch64-min-jump-table-entries=8 -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK8  < %t
; RUN: llc %s -O2 -print-after-isel -mtriple=aarch64-linux-gnu -jump-table-density=40 -aarch64-min-jump-table-entries=12 -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK12  < %t
; RUN: llc %s -O2 -print-after-isel -mtriple=aarch64-linux-gnu -jump-table-density=40 -o /dev/null 2> %t; FileCheck %s --check-prefixes=CHECK,CHECK-DEFAULT  < %t

declare void @ext(i32, i32)

define i32 @jt2(i32 %a, i32 %b) {
entry:
  switch i32 %a, label %return [
    i32 1, label %bb1
    i32 3, label %bb2
  ]
; CHECK-LABEL: function jt2:
; CHECK0-NEXT: Jump Tables:
; CHECK2-NEXT: Jump Tables:
; CHECK4-NOT: {{^}}Jump Tables:
; CHECK8-NOT: {{^}}Jump Tables:
; CHECK12-NOT: {{^}}Jump Tables:
; CHECK-DEFAULT-NOT: {{^}}Jump Tables:

bb1: tail call void @ext(i32 1, i32 0) br label %return
bb2: tail call void @ext(i32 2, i32 2) br label %return

return: ret i32 %b
}

define i32 @jt4(i32 %a, i32 %b) {
entry:
  switch i32 %a, label %return [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 5, label %bb4
  ]
; CHECK-LABEL: function jt4:
; CHECK0-NEXT: Jump Tables:
; CHECK2-NEXT: Jump Tables:
; CHECK4-NEXT: Jump Tables:
; CHECK8-NOT: {{^}}Jump Tables:
; CHECK12-NOT: {{^}}Jump Tables:
; CHECK-DEFAULT-NOT: {{^}}Jump Tables:

bb1: tail call void @ext(i32 1, i32 0) br label %return
bb2: tail call void @ext(i32 3, i32 2) br label %return
bb3: tail call void @ext(i32 4, i32 4) br label %return
bb4: tail call void @ext(i32 5, i32 6) br label %return

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
    i32 9, label %bb8
  ]
; CHECK-LABEL: function jt8:
; CHECK0-NEXT: Jump Tables:
; CHECK2-NEXT: Jump Tables:
; CHECK4-NEXT: Jump Tables:
; CHECK8-NEXT: Jump Tables:
; CHECK12-NOT: Jump Tables:
; CHECK-DEFAULT-NOT: {{^}}Jump Tables:

bb1: tail call void @ext(i32 1, i32 0) br label %return
bb2: tail call void @ext(i32 2, i32 2) br label %return
bb3: tail call void @ext(i32 3, i32 4) br label %return
bb4: tail call void @ext(i32 4, i32 6) br label %return
bb5: tail call void @ext(i32 5, i32 8) br label %return
bb6: tail call void @ext(i32 6, i32 10) br label %return
bb7: tail call void @ext(i32 7, i32 12) br label %return
bb8: tail call void @ext(i32 8, i32 14) br label %return

return: ret i32 %b
}

define i32 @jt12(i32 %a, i32 %b) {
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
  ]
; CHECK-LABEL: function jt12:
; CHECK0-NEXT: Jump Tables:
; CHECK2-NEXT: Jump Tables:
; CHECK4-NEXT: Jump Tables:
; CHECK8-NEXT: Jump Tables:
; CHECK12-NEXT: Jump Tables:
; CHECK-DEFAULT-NOT: {{^}}Jump Tables:

bb1: tail call void @ext(i32 1, i32 0) br label %return
bb2: tail call void @ext(i32 2, i32 2) br label %return
bb3: tail call void @ext(i32 3, i32 4) br label %return
bb4: tail call void @ext(i32 4, i32 6) br label %return
bb5: tail call void @ext(i32 5, i32 8) br label %return
bb6: tail call void @ext(i32 6, i32 10) br label %return
bb7: tail call void @ext(i32 7, i32 12) br label %return
bb8: tail call void @ext(i32 8, i32 14) br label %return
bb9: tail call void @ext(i32 9, i32 16) br label %return
bb10: tail call void @ext(i32 10, i32 18) br label %return
bb11: tail call void @ext(i32 11, i32 20) br label %return
bb12: tail call void @ext(i32 12, i32 22) br label %return

return: ret i32 %b
}

define i32 @jt12_min_size(i32 %a, i32 %b) minsize {
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
  ]
; CHECK-LABEL: function jt12_min_size:
; CHECK0-NEXT: Jump Tables:
; CHECK2-NEXT: Jump Tables:
; CHECK4-NEXT: Jump Tables:
; CHECK8-NEXT: Jump Tables:
; CHECK12-NEXT: Jump Tables:
; CHECK-DEFAULT: Jump Tables:

bb1: tail call void @ext(i32 1, i32 0) br label %return
bb2: tail call void @ext(i32 2, i32 2) br label %return
bb3: tail call void @ext(i32 3, i32 4) br label %return
bb4: tail call void @ext(i32 4, i32 6) br label %return
bb5: tail call void @ext(i32 5, i32 8) br label %return
bb6: tail call void @ext(i32 6, i32 10) br label %return
bb7: tail call void @ext(i32 7, i32 12) br label %return
bb8: tail call void @ext(i32 8, i32 14) br label %return
bb9: tail call void @ext(i32 9, i32 16) br label %return
bb10: tail call void @ext(i32 10, i32 18) br label %return
bb11: tail call void @ext(i32 11, i32 20) br label %return
bb12: tail call void @ext(i32 12, i32 22) br label %return

return: ret i32 %b
}

define i32 @jt13(i32 %a, i32 %b) {
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
  ]
; CHECK-LABEL: function jt13:
; CHECK-NEXT: Jump Tables:

bb1: tail call void @ext(i32 1, i32 0) br label %return
bb2: tail call void @ext(i32 2, i32 2) br label %return
bb3: tail call void @ext(i32 3, i32 4) br label %return
bb4: tail call void @ext(i32 4, i32 6) br label %return
bb5: tail call void @ext(i32 5, i32 8) br label %return
bb6: tail call void @ext(i32 6, i32 10) br label %return
bb7: tail call void @ext(i32 7, i32 12) br label %return
bb8: tail call void @ext(i32 8, i32 14) br label %return
bb9: tail call void @ext(i32 9, i32 16) br label %return
bb10: tail call void @ext(i32 10, i32 18) br label %return
bb11: tail call void @ext(i32 11, i32 20) br label %return
bb12: tail call void @ext(i32 12, i32 22) br label %return
bb13: tail call void @ext(i32 13, i32 24) br label %return

return: ret i32 %b
}
