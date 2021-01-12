; RUN: opt -loop-unswitch -verify-dom-info -verify-memoryssa -S -enable-new-pm=0 %s | FileCheck %s

declare void @clobber()

define i32 @partial_unswitch_true_successor(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_successor
; CHECK-LABEL: entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @partial_unswitch_false_successor(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_false_successor
; CHECK-LABEL: entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %clobber, label %noclobber

clobber:
  call void @clobber()
  br label %loop.latch

noclobber:
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @partial_unswtich_gep_load_icmp(i32** %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswtich_gep_load_icmp
; CHECK-LABEL: entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep = getelementptr i32*, i32** %ptr, i32 1
  %lv.1 = load i32*, i32** %gep
  %lv = load i32, i32* %lv.1
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @partial_unswitch_reduction_phi(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_reduction_phi
; CHECK-LABEL: entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %red = phi i32 [ 20, %entry ], [ %red.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %clobber, label %noclobber

clobber:
  call void @clobber()
  %add.5 = add i32 %red, 5
  br label %loop.latch

noclobber:
  %add.10 = add i32 %red, 10
  br label %loop.latch

loop.latch:
  %red.next = phi i32 [ %add.5, %clobber ], [ %add.10, %noclobber ]
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  %red.next.lcssa = phi i32 [ %red.next, %loop.latch ]
  ret i32 %red.next.lcssa
}

; Partial unswitching is possible, because the store in %noclobber does not
; alias the load of the condition.
define i32 @partial_unswitch_true_successor_noclobber(i32* noalias %ptr.1, i32* noalias %ptr.2, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_successor
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr.1
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  %gep.1 = getelementptr i32, i32* %ptr.2, i32 %iv
  store i32 %lv, i32* %gep.1
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define void @no_partial_unswitch_phi_cond(i1 %lc, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_phi_cond
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %sc = phi i1 [ %lc, %entry ], [ true, %loop.latch ]
  br i1 %sc, label %clobber, label %noclobber

clobber:
  call void @clobber()
  br label %loop.latch

noclobber:
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret void
}

define void @no_partial_unswitch_clobber_latch(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_clobber_latch
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  call void @clobber()
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret void
}

define void @no_partial_unswitch_clobber_header(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_clobber_header
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  call void @clobber()
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret void
}

define void @no_partial_unswitch_clobber_both(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_clobber_both
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  call void @clobber()
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret void
}

define i32 @no_partial_unswitch_true_successor_storeclobber(i32* %ptr.1, i32* %ptr.2, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_true_successor_storeclobber
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr.1
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  %gep.1 = getelementptr i32, i32* %ptr.2, i32 %iv
  store i32 %lv, i32* %gep.1
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

; Check that MemorySSA updating can deal with a clobbering access of a
; duplicated load being a MemoryPHI outside the loop.
define void @partial_unswitch_memssa_update(i32* noalias %ptr, i1 %c) {
; CHECK-LABEL: @partial_unswitch_memssa_update(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %c, label %loop.ph, label %outside.clobber
;
entry:
  br i1 %c, label %loop.ph, label %outside.clobber

outside.clobber:
  call void @clobber()
  br label %loop.ph

loop.ph:
  br label %loop.header

loop.header:
  %lv = load i32, i32* %ptr, align 4
  %hc = icmp eq i32 %lv, 0
  br i1 %hc, label %if, label %then

if:
  br label %loop.latch

then:
  br label %loop.latch

loop.latch:
  br i1 true, label %loop.header, label %exit

exit:
  ret void
}

; Make sure the duplicated instructions are moved to a preheader that always
; executes when the loop body also executes. Do not check the unswitched code,
; because it is already checked in the @partial_unswitch_true_successor test
; case.
define i32 @partial_unswitch_true_successor_preheader_insertion(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_successor_preheader_insertion(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %ec = icmp ne i32* %ptr, null
; CHECK-NEXT:    br i1 %ec, label %loop.ph, label %exit
;
entry:
  %ec = icmp ne i32* %ptr, null
  br i1 %ec, label %loop.ph, label %exit

loop.ph:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %loop.ph ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

; Make sure the duplicated instructions are hoisted just before the branch of
; the preheader. Do not check the unswitched code, because it is already checked
; in the @partial_unswitch_true_successor test case
define i32 @partial_unswitch_true_successor_insert_point(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_successor_insert_point(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @clobber()
; CHECK-NEXT:    br label %loop.header
;
entry:
  call void @clobber()
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

; Make sure invariant instructions in the loop are also hoisted to the preheader.
; Do not check the unswitched code, because it is already checked in the
; @partial_unswitch_true_successor test case
define i32 @partial_unswitch_true_successor_hoist_invariant(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_successor_hoist_invariant(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep = getelementptr i32, i32* %ptr, i64 1
  %lv = load i32, i32* %gep
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}
