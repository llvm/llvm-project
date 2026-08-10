; RUN: llc < %s -O1 -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -ppc-use-branch-hint=false | FileCheck %s
; RUN: llc < %s -O1 -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -ppc-use-branch-hint=true | FileCheck %s -check-prefix=CHECK-HINT
define void @branch_hint_1(i32 %src) {
entry:
  %cmp = icmp eq i32 %src, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo() #0
  unreachable

if.end:
  call void @goo()
  ret void

; CHECK-LABEL: branch_hint_1:
; CHECK: beq

; CHECK-HINT-LABEL: branch_hint_1:
; CHECK-HINT: beq-
}

define void @branch_hint_2(i32 %src) {
entry:
  %cmp = icmp eq i32 %src, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @goo()
  ret void

if.end:
  tail call void @foo() #0
  unreachable

; CHECK-LABEL: @branch_hint_2
; CHECK: bne

; CHECK-HINT-LABEL: @branch_hint_2
; CHECK-HINT: bne-
}

declare void @foo()
attributes #0 = { noreturn }

define void @branch_hint_3(i32 %src) {
entry:
  %cmp = icmp eq i32 %src, 0
  br i1 %cmp, label %if.then, label %if.end, !prof !0

if.then:
  call void @foo()
  ret void

if.end:
  call void @goo()
  ret void

; CHECK-LABEL: @branch_hint_3
; CHECK: bne

; CHECK-HINT-LABEL: @branch_hint_3
; CHECK-HINT: bne
}

!0 = !{!"branch_weights", i32 64, i32 4}

define void @branch_hint_4(i32 %src) {
entry:
  %cmp = icmp eq i32 %src, 0
  br i1 %cmp, label %if.then, label %if.end, !prof !1

if.then:
  call void @foo()
  ret void

if.end:
  call void @goo()
  ret void

; CHECK-HINT-LABEL: branch_hint_4
; CHECK-HINT: bne
}

!1 = !{!"branch_weights", i32 64, i32 8}

define void @branch_hint_5(i32 %src) {
entry:
  %cmp = icmp eq i32 %src, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  ret void

if.end:
  call void @goo()
  ret void

; CHECK-HINT-LABEL: branch_hint_5:
; CHECK-HINT: beq
}

declare void @goo()

define void @branch_hint_6(i32 %src1, i32 %src2, i32 %src3) {
entry:
  %cmp = icmp eq i32 %src1, 0
  br i1 %cmp, label %if.end.6, label %if.end, !prof !3

if.end:
  %cmp1 = icmp eq i32 %src2, 0
  br i1 %cmp1, label %if.end.3, label %if.then.2

if.then.2:
  tail call void @foo() #0
  unreachable

if.end.3:
  %cmp4 = icmp eq i32 %src3, 1
  br i1 %cmp4, label %if.then.5, label %if.end.6

if.then.5:
  tail call void @foo() #0
  unreachable

if.end.6:
  ret void

; CHECK-HINT-LABEL: branch_hint_6:
; CHECK-HINT: bne
; CHECK-HINT: bne-
; CHECK-HINT: bne+
}

!3 = !{!"branch_weights", i32 64, i32 4}
