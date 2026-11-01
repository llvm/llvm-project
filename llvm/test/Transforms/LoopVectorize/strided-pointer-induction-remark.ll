; RUN: opt -passes=loop-vectorize -pass-remarks-output=%t.yaml -disable-output %s
; RUN: FileCheck %s --input-file=%t.yaml

define void @strided_pointer_induction(ptr noalias %d, ptr noalias %s, i64 %n, i64 %st) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = phi ptr [ %s, %entry ], [ %p.next, %loop ]
  %x = load i8, ptr %p, align 1
  %q = getelementptr inbounds i8, ptr %d, i64 %i
  store i8 %x, ptr %q, align 1
  %p.next = getelementptr inbounds i8, ptr %p, i64 %st
  %i.next = add i64 %i, 1
  %ec = icmp eq i64 %i.next, %n
  br i1 %ec, label %exit, label %loop
exit:
  ret void
}

define i32 @unidentified_phi(ptr %p, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %sel, %loop ]
  %gep = getelementptr inbounds i32, ptr %p, i64 %i
  %v = load i32, ptr %gep, align 4
  %cmp = icmp sgt i32 %v, 0
  %sel = select i1 %cmp, i32 %v, i32 %acc
  %div = sdiv i32 %sel, %acc
  store i32 %div, ptr %gep, align 4
  %i.next = add i64 %i, 1
  %ec = icmp eq i64 %i.next, %n
  br i1 %ec, label %exit, label %loop
exit:
  ret i32 %sel
}

; CHECK-LABEL: Function: strided_pointer_induction{{$}}
; CHECK:         - String: value that could not be identified as reduction is used outside the loop
; CHECK-LABEL: Function: unidentified_phi{{$}}
; CHECK:         - String: value that could not be identified as reduction is used outside the loop
