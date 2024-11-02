; RUN: opt -verify-loop-info -irce-print-changed-loops -irce < %s 2>&1 | FileCheck %s
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes='require<branch-prob>,irce' < %s 2>&1 | FileCheck %s

; This test checks if we update the LoopInfo correctly in the presence
; of parents, uncles and cousins.

; Function Attrs: alwaysinline
define void @inner_loop(ptr %arr, ptr %a_len_ptr, i32 %n) #0 {
; CHECK: irce: in function inner_loop: constrained Loop at depth 1 containing: %loop<header><exiting>,%in.bounds<latch><exiting>

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

loop:                                             ; preds = %in.bounds, %entry
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

in.bounds:                                        ; preds = %loop
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

out.of.bounds:                                    ; preds = %loop
  ret void

exit:                                             ; preds = %in.bounds, %entry
  ret void
}

; Function Attrs: alwaysinline
define void @with_parent(ptr %arr, ptr %a_len_ptr, i32 %n, i32 %parent.count) #0 {
; CHECK: irce: in function with_parent: constrained Loop at depth 2 containing: %loop.i<header><exiting>,%in.bounds.i<latch><exiting>

entry:
  br label %loop

loop:                                             ; preds = %inner_loop.exit, %entry
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %inner_loop.exit ]
  %idx.next = add i32 %idx, 1
  %next = icmp ult i32 %idx.next, %parent.count
  %len.i = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check.i = icmp sgt i32 %n, 0
  br i1 %first.itr.check.i, label %loop.i, label %exit.i

loop.i:                                           ; preds = %in.bounds.i, %loop
  %idx.i = phi i32 [ 0, %loop ], [ %idx.next.i, %in.bounds.i ]
  %idx.next.i = add i32 %idx.i, 1
  %abc.i = icmp slt i32 %idx.i, %len.i
  br i1 %abc.i, label %in.bounds.i, label %out.of.bounds.i, !prof !1

in.bounds.i:                                      ; preds = %loop.i
  %addr.i = getelementptr i32, ptr %arr, i32 %idx.i
  store i32 0, ptr %addr.i
  %next.i = icmp slt i32 %idx.next.i, %n
  br i1 %next.i, label %loop.i, label %exit.i

out.of.bounds.i:                                  ; preds = %loop.i
  br label %inner_loop.exit

exit.i:                                           ; preds = %in.bounds.i, %loop
  br label %inner_loop.exit

inner_loop.exit:                                  ; preds = %exit.i, %out.of.bounds.i
  br i1 %next, label %loop, label %exit

exit:                                             ; preds = %inner_loop.exit
  ret void
}

; Function Attrs: alwaysinline
define void @with_grandparent(ptr %arr, ptr %a_len_ptr, i32 %n, i32 %parent.count, i32 %grandparent.count) #0 {
; CHECK: irce: in function with_grandparent: constrained Loop at depth 3 containing: %loop.i.i<header><exiting>,%in.bounds.i.i<latch><exiting>

entry:
  br label %loop

loop:                                             ; preds = %with_parent.exit, %entry
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %with_parent.exit ]
  %idx.next = add i32 %idx, 1
  %next = icmp ult i32 %idx.next, %grandparent.count
  br label %loop.i

loop.i:                                           ; preds = %inner_loop.exit.i, %loop
  %idx.i = phi i32 [ 0, %loop ], [ %idx.next.i, %inner_loop.exit.i ]
  %idx.next.i = add i32 %idx.i, 1
  %next.i = icmp ult i32 %idx.next.i, %parent.count
  %len.i.i = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check.i.i = icmp sgt i32 %n, 0
  br i1 %first.itr.check.i.i, label %loop.i.i, label %exit.i.i

loop.i.i:                                         ; preds = %in.bounds.i.i, %loop.i
  %idx.i.i = phi i32 [ 0, %loop.i ], [ %idx.next.i.i, %in.bounds.i.i ]
  %idx.next.i.i = add i32 %idx.i.i, 1
  %abc.i.i = icmp slt i32 %idx.i.i, %len.i.i
  br i1 %abc.i.i, label %in.bounds.i.i, label %out.of.bounds.i.i, !prof !1

in.bounds.i.i:                                    ; preds = %loop.i.i
  %addr.i.i = getelementptr i32, ptr %arr, i32 %idx.i.i
  store i32 0, ptr %addr.i.i
  %next.i.i = icmp slt i32 %idx.next.i.i, %n
  br i1 %next.i.i, label %loop.i.i, label %exit.i.i

out.of.bounds.i.i:                                ; preds = %loop.i.i
  br label %inner_loop.exit.i

exit.i.i:                                         ; preds = %in.bounds.i.i, %loop.i
  br label %inner_loop.exit.i

inner_loop.exit.i:                                ; preds = %exit.i.i, %out.of.bounds.i.i
  br i1 %next.i, label %loop.i, label %with_parent.exit

with_parent.exit:                                 ; preds = %inner_loop.exit.i
  br i1 %next, label %loop, label %exit

exit:                                             ; preds = %with_parent.exit
  ret void
}

; Function Attrs: alwaysinline
define void @with_sibling(ptr %arr, ptr %a_len_ptr, i32 %n, i32 %parent.count) #0 {
; CHECK: irce: in function with_sibling: constrained Loop at depth 2 containing: %loop.i<header><exiting>,%in.bounds.i<latch><exiting>
; CHECK: irce: in function with_sibling: constrained Loop at depth 2 containing: %loop.i6<header><exiting>,%in.bounds.i9<latch><exiting>

entry:
  br label %loop

loop:                                             ; preds = %inner_loop.exit12, %entry
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %inner_loop.exit12 ]
  %idx.next = add i32 %idx, 1
  %next = icmp ult i32 %idx.next, %parent.count
  %len.i = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check.i = icmp sgt i32 %n, 0
  br i1 %first.itr.check.i, label %loop.i, label %exit.i

loop.i:                                           ; preds = %in.bounds.i, %loop
  %idx.i = phi i32 [ 0, %loop ], [ %idx.next.i, %in.bounds.i ]
  %idx.next.i = add i32 %idx.i, 1
  %abc.i = icmp slt i32 %idx.i, %len.i
  br i1 %abc.i, label %in.bounds.i, label %out.of.bounds.i, !prof !1

in.bounds.i:                                      ; preds = %loop.i
  %addr.i = getelementptr i32, ptr %arr, i32 %idx.i
  store i32 0, ptr %addr.i
  %next.i = icmp slt i32 %idx.next.i, %n
  br i1 %next.i, label %loop.i, label %exit.i

out.of.bounds.i:                                  ; preds = %loop.i
  br label %inner_loop.exit

exit.i:                                           ; preds = %in.bounds.i, %loop
  br label %inner_loop.exit

inner_loop.exit:                                  ; preds = %exit.i, %out.of.bounds.i
  %len.i1 = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check.i2 = icmp sgt i32 %n, 0
  br i1 %first.itr.check.i2, label %loop.i6, label %exit.i11

loop.i6:                                          ; preds = %in.bounds.i9, %inner_loop.exit
  %idx.i3 = phi i32 [ 0, %inner_loop.exit ], [ %idx.next.i4, %in.bounds.i9 ]
  %idx.next.i4 = add i32 %idx.i3, 1
  %abc.i5 = icmp slt i32 %idx.i3, %len.i1
  br i1 %abc.i5, label %in.bounds.i9, label %out.of.bounds.i10, !prof !1

in.bounds.i9:                                     ; preds = %loop.i6
  %addr.i7 = getelementptr i32, ptr %arr, i32 %idx.i3
  store i32 0, ptr %addr.i7
  %next.i8 = icmp slt i32 %idx.next.i4, %n
  br i1 %next.i8, label %loop.i6, label %exit.i11

out.of.bounds.i10:                                ; preds = %loop.i6
  br label %inner_loop.exit12

exit.i11:                                         ; preds = %in.bounds.i9, %inner_loop.exit
  br label %inner_loop.exit12

inner_loop.exit12:                                ; preds = %exit.i11, %out.of.bounds.i10
  br i1 %next, label %loop, label %exit

exit:                                             ; preds = %inner_loop.exit12
  ret void
}

; Function Attrs: alwaysinline
define void @with_cousin(ptr %arr, ptr %a_len_ptr, i32 %n, i32 %parent.count, i32 %grandparent.count) #0 {
; CHECK: irce: in function with_cousin: constrained Loop at depth 3 containing: %loop.i.i<header><exiting>,%in.bounds.i.i<latch><exiting>
; CHECK: irce: in function with_cousin: constrained Loop at depth 3 containing: %loop.i.i10<header><exiting>,%in.bounds.i.i13<latch><exiting>

entry:
  br label %loop

loop:                                             ; preds = %with_parent.exit17, %entry
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %with_parent.exit17 ]
  %idx.next = add i32 %idx, 1
  %next = icmp ult i32 %idx.next, %grandparent.count
  br label %loop.i

loop.i:                                           ; preds = %inner_loop.exit.i, %loop
  %idx.i = phi i32 [ 0, %loop ], [ %idx.next.i, %inner_loop.exit.i ]
  %idx.next.i = add i32 %idx.i, 1
  %next.i = icmp ult i32 %idx.next.i, %parent.count
  %len.i.i = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check.i.i = icmp sgt i32 %n, 0
  br i1 %first.itr.check.i.i, label %loop.i.i, label %exit.i.i

loop.i.i:                                         ; preds = %in.bounds.i.i, %loop.i
  %idx.i.i = phi i32 [ 0, %loop.i ], [ %idx.next.i.i, %in.bounds.i.i ]
  %idx.next.i.i = add i32 %idx.i.i, 1
  %abc.i.i = icmp slt i32 %idx.i.i, %len.i.i
  br i1 %abc.i.i, label %in.bounds.i.i, label %out.of.bounds.i.i, !prof !1

in.bounds.i.i:                                    ; preds = %loop.i.i
  %addr.i.i = getelementptr i32, ptr %arr, i32 %idx.i.i
  store i32 0, ptr %addr.i.i
  %next.i.i = icmp slt i32 %idx.next.i.i, %n
  br i1 %next.i.i, label %loop.i.i, label %exit.i.i

out.of.bounds.i.i:                                ; preds = %loop.i.i
  br label %inner_loop.exit.i

exit.i.i:                                         ; preds = %in.bounds.i.i, %loop.i
  br label %inner_loop.exit.i

inner_loop.exit.i:                                ; preds = %exit.i.i, %out.of.bounds.i.i
  br i1 %next.i, label %loop.i, label %with_parent.exit

with_parent.exit:                                 ; preds = %inner_loop.exit.i
  br label %loop.i6

loop.i6:                                          ; preds = %inner_loop.exit.i16, %with_parent.exit
  %idx.i1 = phi i32 [ 0, %with_parent.exit ], [ %idx.next.i2, %inner_loop.exit.i16 ]
  %idx.next.i2 = add i32 %idx.i1, 1
  %next.i3 = icmp ult i32 %idx.next.i2, %parent.count
  %len.i.i4 = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check.i.i5 = icmp sgt i32 %n, 0
  br i1 %first.itr.check.i.i5, label %loop.i.i10, label %exit.i.i15

loop.i.i10:                                       ; preds = %in.bounds.i.i13, %loop.i6
  %idx.i.i7 = phi i32 [ 0, %loop.i6 ], [ %idx.next.i.i8, %in.bounds.i.i13 ]
  %idx.next.i.i8 = add i32 %idx.i.i7, 1
  %abc.i.i9 = icmp slt i32 %idx.i.i7, %len.i.i4
  br i1 %abc.i.i9, label %in.bounds.i.i13, label %out.of.bounds.i.i14, !prof !1

in.bounds.i.i13:                                  ; preds = %loop.i.i10
  %addr.i.i11 = getelementptr i32, ptr %arr, i32 %idx.i.i7
  store i32 0, ptr %addr.i.i11
  %next.i.i12 = icmp slt i32 %idx.next.i.i8, %n
  br i1 %next.i.i12, label %loop.i.i10, label %exit.i.i15

out.of.bounds.i.i14:                              ; preds = %loop.i.i10
  br label %inner_loop.exit.i16

exit.i.i15:                                       ; preds = %in.bounds.i.i13, %loop.i6
  br label %inner_loop.exit.i16

inner_loop.exit.i16:                              ; preds = %exit.i.i15, %out.of.bounds.i.i14
  br i1 %next.i3, label %loop.i6, label %with_parent.exit17

with_parent.exit17:                               ; preds = %inner_loop.exit.i16
  br i1 %next, label %loop, label %exit

exit:                                             ; preds = %with_parent.exit17
  ret void
}

; Function Attrs: alwaysinline
define void @with_uncle(ptr %arr, ptr %a_len_ptr, i32 %n, i32 %parent.count, i32 %grandparent.count) #0 {
; CHECK: irce: in function with_uncle: constrained Loop at depth 2 containing: %loop.i<header><exiting>,%in.bounds.i<latch><exiting>
; CHECK: irce: in function with_uncle: constrained Loop at depth 3 containing: %loop.i.i<header><exiting>,%in.bounds.i.i<latch><exiting>

entry:
  br label %loop

loop:                                             ; preds = %with_parent.exit, %entry
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %with_parent.exit ]
  %idx.next = add i32 %idx, 1
  %next = icmp ult i32 %idx.next, %grandparent.count
  %len.i = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check.i = icmp sgt i32 %n, 0
  br i1 %first.itr.check.i, label %loop.i, label %exit.i

loop.i:                                           ; preds = %in.bounds.i, %loop
  %idx.i = phi i32 [ 0, %loop ], [ %idx.next.i, %in.bounds.i ]
  %idx.next.i = add i32 %idx.i, 1
  %abc.i = icmp slt i32 %idx.i, %len.i
  br i1 %abc.i, label %in.bounds.i, label %out.of.bounds.i, !prof !1

in.bounds.i:                                      ; preds = %loop.i
  %addr.i = getelementptr i32, ptr %arr, i32 %idx.i
  store i32 0, ptr %addr.i
  %next.i = icmp slt i32 %idx.next.i, %n
  br i1 %next.i, label %loop.i, label %exit.i

out.of.bounds.i:                                  ; preds = %loop.i
  br label %inner_loop.exit

exit.i:                                           ; preds = %in.bounds.i, %loop
  br label %inner_loop.exit

inner_loop.exit:                                  ; preds = %exit.i, %out.of.bounds.i
  br label %loop.i4

loop.i4:                                          ; preds = %inner_loop.exit.i, %inner_loop.exit
  %idx.i1 = phi i32 [ 0, %inner_loop.exit ], [ %idx.next.i2, %inner_loop.exit.i ]
  %idx.next.i2 = add i32 %idx.i1, 1
  %next.i3 = icmp ult i32 %idx.next.i2, %parent.count
  %len.i.i = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check.i.i = icmp sgt i32 %n, 0
  br i1 %first.itr.check.i.i, label %loop.i.i, label %exit.i.i

loop.i.i:                                         ; preds = %in.bounds.i.i, %loop.i4
  %idx.i.i = phi i32 [ 0, %loop.i4 ], [ %idx.next.i.i, %in.bounds.i.i ]
  %idx.next.i.i = add i32 %idx.i.i, 1
  %abc.i.i = icmp slt i32 %idx.i.i, %len.i.i
  br i1 %abc.i.i, label %in.bounds.i.i, label %out.of.bounds.i.i, !prof !1

in.bounds.i.i:                                    ; preds = %loop.i.i
  %addr.i.i = getelementptr i32, ptr %arr, i32 %idx.i.i
  store i32 0, ptr %addr.i.i
  %next.i.i = icmp slt i32 %idx.next.i.i, %n
  br i1 %next.i.i, label %loop.i.i, label %exit.i.i

out.of.bounds.i.i:                                ; preds = %loop.i.i
  br label %inner_loop.exit.i

exit.i.i:                                         ; preds = %in.bounds.i.i, %loop.i4
  br label %inner_loop.exit.i

inner_loop.exit.i:                                ; preds = %exit.i.i, %out.of.bounds.i.i
  br i1 %next.i3, label %loop.i4, label %with_parent.exit

with_parent.exit:                                 ; preds = %inner_loop.exit.i
  br i1 %next, label %loop, label %exit

exit:                                             ; preds = %with_parent.exit
  ret void
}

attributes #0 = { alwaysinline }

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 64, i32 4}
