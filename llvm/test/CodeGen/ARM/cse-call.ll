; RUN: llc < %s -mcpu=arm1136jf-s -verify-machineinstrs | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "armv6-apple-ios0.0.0"

; Don't CSE a cmp across a call that clobbers CPSR.
;
; CHECK: cmp
; CHECK: S_trimzeros
; CHECK: cmp
; CHECK: strlen

@F_floatmul.man1 = external global [200 x i8], align 1
@F_floatmul.man2 = external global [200 x i8], align 1

declare i32 @strlen(i8* nocapture) nounwind readonly
declare void @S_trimzeros(...)

define i8* @F_floatmul(i8* %f1, i8* %f2) nounwind ssp {
entry:
  br i1 undef, label %while.end42, label %while.body37

while.body37:                                     ; preds = %while.body37, %entry
  br i1 false, label %while.end42, label %while.body37

while.end42:                                      ; preds = %while.body37, %entry
  %. = select i1 undef, i8* getelementptr inbounds ([200 x i8]* @F_floatmul.man1, i32 0, i32 0), i8* getelementptr inbounds ([200 x i8]* @F_floatmul.man2, i32 0, i32 0)
  %.92 = select i1 undef, i8* getelementptr inbounds ([200 x i8]* @F_floatmul.man2, i32 0, i32 0), i8* getelementptr inbounds ([200 x i8]* @F_floatmul.man1, i32 0, i32 0)
  tail call void bitcast (void (...)* @S_trimzeros to void (i8*)*)(i8* %.92) nounwind
  %call47 = tail call i32 @strlen(i8* %.) nounwind
  unreachable
}
