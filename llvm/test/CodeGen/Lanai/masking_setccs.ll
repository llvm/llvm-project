; RUN: llc < %s | FileCheck %s

; Test that unnecessary masking with 0x1 is not inserted.

target datalayout = "E-m:e-p:32:32-i64:64-a:0:32-n32-S64"
target triple = "lanai"

; CHECK-LABEL: masking:
; CHECK-NOT: mov 1
define i32 @masking(i32 inreg %a, i32 inreg %b, i32 inreg %c, i32 inreg %d) {
entry:
  %cmp = icmp ne i32 %a, 0
  %cmp1 = icmp ult i32 %a, %b
  %or.cond = and i1 %cmp, %cmp1
  br i1 %or.cond, label %return, label %if.end

if.end:                                           ; preds = %entry
  %cmp2 = icmp ne i32 %b, 0
  %cmp4 = icmp ult i32 %b, %c
  %or.cond29 = and i1 %cmp2, %cmp4
  br i1 %or.cond29, label %return, label %if.end6

if.end6:                                          ; preds = %if.end
  %cmp7 = icmp ne i32 %c, 0
  %cmp9 = icmp ult i32 %c, %d
  %or.cond30 = and i1 %cmp7, %cmp9
  br i1 %or.cond30, label %return, label %if.end11

if.end11:                                         ; preds = %if.end6
  %cmp12 = icmp ne i32 %d, 0
  %cmp14 = icmp ult i32 %d, %a
  %or.cond31 = and i1 %cmp12, %cmp14
  %b. = select i1 %or.cond31, i32 %b, i32 21
  ret i32 %b.

return:                                           ; preds = %if.end6, %if.end, %entry
  %retval.0 = phi i32 [ %c, %entry ], [ %d, %if.end ], [ %a, %if.end6 ]
  ret i32 %retval.0
}

; CHECK-LABEL: notnot:
; CHECK-NOT: mov 1
define i32 @notnot(i32 %x) {
entry:
  %tobool = icmp ne i32 %x, 0
  %lnot.ext = zext i1 %tobool to i32
  ret i32 %lnot.ext
}
