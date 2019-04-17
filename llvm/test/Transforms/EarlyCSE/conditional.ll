; RUN: opt -early-cse -S < %s | FileCheck %s
; RUN: opt -basicaa -early-cse-memssa -S < %s | FileCheck %s

; Can we CSE a known condition to a constant?
define i1 @test(i8* %p) {
; CHECK-LABEL: @test
entry:
  %cnd1 = icmp eq i8* %p, null
  br i1 %cnd1, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: ret i1 true
  %cnd2 = icmp eq i8* %p, null
  ret i1 %cnd2

untaken:
; CHECK-LABEL: untaken:
; CHECK-NEXT: ret i1 false
  %cnd3 = icmp eq i8* %p, null
  ret i1 %cnd3
}

; We can CSE the condition, but we *don't* know it's value after the merge
define i1 @test_neg1(i8* %p) {
; CHECK-LABEL: @test_neg1
entry:
  %cnd1 = icmp eq i8* %p, null
  br i1 %cnd1, label %taken, label %untaken

taken:
  br label %merge

untaken:
  br label %merge

merge:
; CHECK-LABEL: merge:
; CHECK-NEXT: ret i1 %cnd1
  %cnd3 = icmp eq i8* %p, null
  ret i1 %cnd3
}

; Check specifically for a case where we have a unique predecessor, but
; not a single predecessor.  We can not know the value of the condition here.
define i1 @test_neg2(i8* %p) {
; CHECK-LABEL: @test_neg2
entry:
  %cnd1 = icmp eq i8* %p, null
  br i1 %cnd1, label %merge, label %merge

merge:
; CHECK-LABEL: merge:
; CHECK-NEXT: ret i1 %cnd1
  %cnd3 = icmp eq i8* %p, null
  ret i1 %cnd3
}

; Replace a use rather than CSE
define i1 @test2(i8* %p) {
; CHECK-LABEL: @test2
entry:
  %cnd = icmp eq i8* %p, null
  br i1 %cnd, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: ret i1 true
  ret i1 %cnd

untaken:
; CHECK-LABEL: untaken:
; CHECK-NEXT: ret i1 false
  ret i1 %cnd
}

; Not legal to replace use given it's not dominated by edge
define i1 @test2_neg1(i8* %p) {
; CHECK-LABEL: @test2_neg1
entry:
  %cnd1 = icmp eq i8* %p, null
  br i1 %cnd1, label %taken, label %untaken

taken:
  br label %merge

untaken:
  br label %merge

merge:
; CHECK-LABEL: merge:
; CHECK-NEXT: ret i1 %cnd1
  ret i1 %cnd1
}

; Another single predecessor test, but for dominated use
define i1 @test2_neg2(i8* %p) {
; CHECK-LABEL: @test2_neg2
entry:
  %cnd1 = icmp eq i8* %p, null
  br i1 %cnd1, label %merge, label %merge

merge:
; CHECK-LABEL: merge:
; CHECK-NEXT: ret i1 %cnd1
  ret i1 %cnd1
}

