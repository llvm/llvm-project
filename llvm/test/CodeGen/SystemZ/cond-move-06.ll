; Test SELR and SELGR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 -verify-machineinstrs | FileCheck %s

; Test SELR.
define i32 @f1(i32 %limit, i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: clfi %r2, 42
; CHECK: selrl %r2, %r3, %r4
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  ret i32 %res
}

; Test SELGR.
define i64 @f2(i64 %limit, i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: clgfi %r2, 42
; CHECK: selgrl %r2, %r3, %r4
; CHECK: br %r14
  %cond = icmp ult i64 %limit, 42
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Test SELR in a case that could use COMPARE AND BRANCH.  We prefer using
; SELR if possible.
define i32 @f3(i32 %limit, i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: chi %r2, 42
; CHECK: selre %r2, %r3, %r4
; CHECK: br %r14
  %cond = icmp eq i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  ret i32 %res
}

; ...and again for SELGR.
define i64 @f4(i64 %limit, i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: cghi %r2, 42
; CHECK: selgre %r2, %r3, %r4
; CHECK: br %r14
  %cond = icmp eq i64 %limit, 42
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check that we also get SELR as a result of early if-conversion.
define i32 @f5(i32 %limit, i32 %a, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: clfi %r2, 41
; CHECK: selrh %r2, %r4, %r3
; CHECK: br %r14
entry:
  %cond = icmp ult i32 %limit, 42
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i32 [ %a, %if.then ], [ %b, %entry ]
  ret i32 %res
}

; ... and likewise for SELGR.
define i64 @f6(i64 %limit, i64 %a, i64 %b) {
; CHECK-LABEL: f6:
; CHECK: clgfi %r2, 41
; CHECK: selgrh %r2, %r4, %r3
; CHECK: br %r14
entry:
  %cond = icmp ult i64 %limit, 42
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i64 [ %a, %if.then ], [ %b, %entry ]
  ret i64 %res
}

; Check that inverting the condition works as well.
define i32 @f7(i32 %limit, i32 %a, i32 %b) {
; CHECK-LABEL: f7:
; CHECK: clfi %r2, 41
; CHECK: selrh %r2, %r3, %r4
; CHECK: br %r14
entry:
  %cond = icmp ult i32 %limit, 42
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i32 [ %b, %if.then ], [ %a, %entry ]
  ret i32 %res
}

; ... and likewise for SELGR.
define i64 @f8(i64 %limit, i64 %a, i64 %b) {
; CHECK-LABEL: f8:
; CHECK: clgfi %r2, 41
; CHECK: selgrh %r2, %r3, %r4
; CHECK: br %r14
entry:
  %cond = icmp ult i64 %limit, 42
  br i1 %cond, label %if.then, label %return

if.then:
  br label %return

return:
  %res = phi i64 [ %b, %if.then ], [ %a, %entry ]
  ret i64 %res
}

