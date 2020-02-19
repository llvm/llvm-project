; RUN: opt -irce-print-changed-loops -verify-loop-info -irce -S < %s 2>&1 | FileCheck %s
; RUN: opt -irce-print-changed-loops -verify-loop-info -passes='require<branch-prob>,irce' -S < %s 2>&1 | FileCheck %s

; CHECK-NOT: constrained Loop at depth

; Demonstrates that we don't currently handle the general expression
; `A * I + B'.

define void @general_affine_expressions(i32 *%arr, i32 *%a_len_ptr, i32 %n,
                                        i32 %scale, i32 %offset) {
 entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.mul = mul i32 %idx, %scale
  %array.idx = add i32 %idx.mul, %offset
  %abc.high = icmp slt i32 %array.idx, %len
  %abc.low = icmp sge i32 %array.idx, 0
  %abc = and i1 %abc.low, %abc.high
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %array.idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

; Check that we do the right thing for a loop that could not be
; simplified due to an indirectbr.

define void @multiple_latches(i32 *%arr, i32 *%a_len_ptr, i32 %n) {
 entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  %n.add.1 = add i32 %n, 1
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ], [ %idx.next, %continue ]
  %idx.next = add i32 %idx, 1
  %idx.next2 = add i32 %idx, 2
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %continue

 continue:
  %next2 = icmp slt i32 %idx.next, %n.add.1
  %dest = select i1 %next2, i8* blockaddress(@multiple_latches, %loop), i8* blockaddress(@multiple_latches, %exit)
  indirectbr i8* %dest, [ label %loop, label %exit]

 out.of.bounds:
  ret void

 exit:
  ret void
}

define void @already_cloned(i32 *%arr, i32 *%a_len_ptr, i32 %n) {
 entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit, !irce.loop.clone !{}

 out.of.bounds:
  ret void

 exit:
  ret void
}

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 64, i32 4}
