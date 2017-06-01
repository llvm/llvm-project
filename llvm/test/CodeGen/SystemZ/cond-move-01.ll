; Test LOCR and LOCGR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 -verify-machineinstrs | FileCheck %s
;
; Run the test again to make sure it still works the same even
; in the presence of the load-store-on-condition-2 facility.
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs | FileCheck %s

; Test LOCR.
define i32 @f1(i32 %a, i32 %b, i32 %limit) {
; CHECK-LABEL: f1:
; CHECK: clfi %r4, 42
; CHECK: locrhe %r2, %r3
; CHECK: br %r14
  %cond = icmp ult i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  ret i32 %res
}

; Test LOCGR.
define i64 @f2(i64 %a, i64 %b, i64 %limit) {
; CHECK-LABEL: f2:
; CHECK: clgfi %r4, 42
; CHECK: locgrhe %r2, %r3
; CHECK: br %r14
  %cond = icmp ult i64 %limit, 42
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Test LOCR in a case that could use COMPARE AND BRANCH.  We prefer using
; LOCR if possible.
define i32 @f3(i32 %a, i32 %b, i32 %limit) {
; CHECK-LABEL: f3:
; CHECK: chi %r4, 42
; CHECK: locrlh %r2, %r3
; CHECK: br %r14
  %cond = icmp eq i32 %limit, 42
  %res = select i1 %cond, i32 %a, i32 %b
  ret i32 %res
}

; ...and again for LOCGR.
define i64 @f4(i64 %a, i64 %b, i64 %limit) {
; CHECK-LABEL: f4:
; CHECK: cghi %r4, 42
; CHECK: locgrlh %r2, %r3
; CHECK: br %r14
  %cond = icmp eq i64 %limit, 42
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check that we also get LOCR as a result of early if-conversion.
define i32 @f5(i32 %a, i32 %b, i32 %limit) {
; CHECK-LABEL: f5:
; CHECK: clfi %r4, 41
; CHECK: locrh %r2, %r3
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

; ... and likewise for LOCGR.
define i64 @f6(i64 %a, i64 %b, i64 %limit) {
; CHECK-LABEL: f6:
; CHECK: clgfi %r4, 41
; CHECK: locgrh %r2, %r3
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
define i32 @f7(i32 %a, i32 %b, i32 %limit) {
; CHECK-LABEL: f7:
; CHECK: clfi %r4, 41
; CHECK: locrle %r2, %r3
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

; ... and likewise for LOCGR.
define i64 @f8(i64 %a, i64 %b, i64 %limit) {
; CHECK-LABEL: f8:
; CHECK: clgfi %r4, 41
; CHECK: locgrle %r2, %r3
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

