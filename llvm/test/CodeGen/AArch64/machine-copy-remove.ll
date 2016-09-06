; RUN: llc -mtriple=aarch64-linux-gnu -mcpu=cortex-a57 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: f_XX:
; CHECK: cbz x[[REG:[0-9]+]], [[BB:.LBB.*]]
; CHECK: [[BB]]:
; CHECK-NOT: mov x[[REG]], xzr
define i64 @f_XX(i64 %n, i64* nocapture readonly %P) {
entry:
  %tobool = icmp eq i64 %n, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i64, i64* %P
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi i64 [ %0, %if.then ], [ 0, %entry ]
  ret i64 %a.0
}

; CHECK-LABEL: f_WW:
; CHECK: cbz w[[REG:[0-9]+]], [[BB:.LBB.*]]
; CHECK: [[BB]]:
; CHECK-NOT: mov w[[REG]], wzr
define i32 @f_WW(i32 %n, i32* nocapture readonly %P) {
entry:
  %tobool = icmp eq i32 %n, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32, i32* %P
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi i32 [ %0, %if.then ], [ 0, %entry ]
  ret i32 %a.0
}

; CHECK-LABEL: f_XW:
; CHECK: cbz x[[REG:[0-9]+]], [[BB:.LBB.*]]
; CHECK: [[BB]]:
; CHECK-NOT: mov w[[REG]], wzr
define i32 @f_XW(i64 %n, i32* nocapture readonly %P) {
entry:
  %tobool = icmp eq i64 %n, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32, i32* %P
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi i32 [ %0, %if.then ], [ 0, %entry ]
  ret i32 %a.0
}

; CHECK-LABEL: f_WX:
; CHECK: cbz w[[REG:[0-9]+]], [[BB:.LBB.*]]
; CHECK: [[BB]]:
; CHECK: mov x[[REG]], xzr
; Do not remove the mov in this case because we do not know if the upper bits
; of the X register are zero.
define i64 @f_WX(i32 %n, i64* nocapture readonly %P) {
entry:
  %tobool = icmp eq i32 %n, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i64, i64* %P
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi i64 [ %0, %if.then ], [ 0, %entry ]
  ret i64 %a.0
}

; CHECK-LABEL: test_superreg:
; CHECK:     cbz x[[REG:[0-9]+]], [[BB:.LBB.*]]
; CHECK: [[BB]]:
; CHECK:     str x[[REG]], [x1]
; CHECK-NOT: mov w[[REG]], wzr
; Because we returned w0 but x0 was marked live-in to the block, we didn't
; remove the <kill> on the str leading to a verification failure.
define i32 @test_superreg(i64 %in, i64* %dest) {
  %tst = icmp eq i64 %in, 0
  br i1 %tst, label %true, label %false

false:
  ret i32 42

true:
  store volatile i64 %in, i64* %dest
  ret i32 0
}
