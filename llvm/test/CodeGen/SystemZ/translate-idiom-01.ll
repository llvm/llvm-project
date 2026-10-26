; Test that a byte-translate loop is replaced with llvm.s390.translate.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 -O2 \
; RUN:   -stop-after=systemz-translate-idiom | FileCheck %s

declare void @llvm.s390.translate(ptr, i64, ptr)

; CHECK-LABEL: @positive
; CHECK:       call void @llvm.s390.translate(ptr %buf, i64 100, ptr %table)
; CHECK-NOT:   phi
define void @positive(ptr noalias %buf, ptr noalias readonly %table) {
entry: 
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load i8, ptr %p
  %z = zext i8 %c to i64
  %t = getelementptr inbounds i8, ptr %table, i64 %z
  %v = load i8, ptr %t
  store i8 %v, ptr %p
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, 100
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Negative: sext index must not match.
; CHECK-LABEL: @negative_sext
; CHECK-NOT:   llvm.s390.translate
define void @negative_sext(ptr noalias %buf, ptr noalias readonly %table) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load i8, ptr %p
  %z = sext i8 %c to i64
  %t = getelementptr inbounds i8, ptr %table, i64 %z
  %v = load i8, ptr %t
  store i8 %v, ptr %p
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, 100
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Negative: no noalias on the pointers, so buffer and table may overlap.
; CHECK-LABEL: @negative_noalias
; CHECK-NOT:   llvm.s390.translate
define void @negative_noalias(ptr %buf, ptr readonly %table) {
entry: 
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load i8, ptr %p
  %z = zext i8 %c to i64
  %t = getelementptr inbounds i8, ptr %table, i64 %z
  %v = load i8, ptr %t
  store i8 %v, ptr %p
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, 100
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Negative: trip count=300 (> 256).
; CHECK-LABEL: @negative_overflow
; CHECK-NOT:   llvm.s390.translate
define void @negative_overflow(ptr noalias %buf, ptr noalias readonly %table) {
entry: 
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load i8, ptr %p
  %z = zext i8 %c to i64
  %t = getelementptr inbounds i8, ptr %table, i64 %z
  %v = load i8, ptr %t
  store i8 %v, ptr %p
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, 300
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; Negative: loop value used after the loop.
; CHECK-LABEL: @negative_dangling
; CHECK-NOT:   llvm.s390.translate
define void @negative_dangling(ptr noalias %buf, ptr noalias readonly %table) {
entry: 
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %n = phi i64 [ 0, %entry ], [ %n.next, %loop ]
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load i8, ptr %p
  %z = zext i8 %c to i64
  %t = getelementptr inbounds i8, ptr %table, i64 %z
  %v = load i8, ptr %t
  store i8 %v, ptr %p
  %n.next = add nuw i64 %n, 1
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, 100
  br i1 %done, label %exit, label %loop

exit:
  %numtr = add nuw i64 %n.next, 0
  ret void
}

; Negative: volatile load/store.
; CHECK-LABEL: @negative_volatile
; CHECK-NOT:   llvm.s390.translate
define void @negative_volatile(ptr noalias %buf, ptr noalias readonly %table) {
entry: 
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i8, ptr %buf, i64 %i
  %c = load volatile i8, ptr %p
  %z = zext i8 %c to i64
  %t = getelementptr inbounds i8, ptr %table, i64 %z
  %v = load volatile i8, ptr %t
  store volatile i8 %v, ptr %p
  %i.next = add nuw i64 %i, 1
  %done = icmp eq i64 %i.next, 100
  br i1 %done, label %exit, label %loop

exit:
  ret void
}