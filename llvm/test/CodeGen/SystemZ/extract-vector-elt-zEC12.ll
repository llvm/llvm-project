; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 | FileCheck %s
;
; Test that <1 x i8> is legalized properly without vector support.

define void @autogen_SD18500(ptr) {
; CHECK: .text
BB:
  %L5 = load i8, ptr %0
  %I22 = insertelement <1 x i8> undef, i8 %L5, i32 0
  %Cmp53 = icmp ule i1 undef, undef
  br label %CF244

CF244:                                            ; preds = %CF244, %BB
  %Sl119 = select i1 %Cmp53, <1 x i8> %I22, <1 x i8> undef
  %Cmp148 = fcmp une float 0x3E03A81780000000, 0x42D92DCD00000000
  br i1 %Cmp148, label %CF244, label %CF241

CF241:                                            ; preds = %CF241, %CF244
  %Sl199 = select i1 true, <1 x i8> %Sl119, <1 x i8> zeroinitializer
  br label %CF241
}
