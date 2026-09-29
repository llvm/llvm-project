; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -pre-RA-sched=list-ilp | FileCheck %s
;
; Check that list-ilp scheduler does not crash due to SystemZ's current use
; of MVT::Untyped.

define void @pr32723(i8) {
; CHECK: .text
BB:
  br label %CF245

CF245:                                            ; preds = %CF245, %BB
  %Shuff57 = shufflevector <4 x i8> zeroinitializer, <4 x i8> zeroinitializer, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Cmp84 = icmp uge i8 %0, undef
  br i1 %Cmp84, label %CF245, label %CF260

CF260:                                            ; preds = %CF245
  %B156 = sdiv <4 x i8> %Shuff57, %Shuff57
  br label %CF255

CF255:                                            ; preds = %CF255, %CF260
  %I186 = insertelement <4 x i8> %B156, i8 %0, i32 2
  br label %CF255
}
