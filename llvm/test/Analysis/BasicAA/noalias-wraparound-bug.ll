; RUN: opt -S -basic-aa -gvn < %s | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.6.0"

; We incorrectly returned noalias in the example below for "tmp5" and
; "tmp12" returning i32 32, since basicaa converted the offsets to 64b
; and missed the wrap-around

define i32 @foo(i8* %buffer) {
entry:
  %tmp2 = getelementptr i8, i8* %buffer, i32 -2071408432
  %tmp3 = bitcast i8* %tmp2 to i32*
  %tmp4 = getelementptr i8, i8* %buffer, i32 128
  %tmp5 = bitcast i8* %tmp4 to i32*
  store i32 32, i32* %tmp5, align 4
  %tmp12 = getelementptr i32, i32* %tmp3, i32 -1629631508
  store i32 28, i32* %tmp12, align 4
  %tmp13 = getelementptr i8, i8* %buffer, i32 128
  %tmp14 = bitcast i8* %tmp13 to i32*
  %tmp2083 = load i32, i32* %tmp14, align 4
; CHECK: ret i32 28
  ret i32 %tmp2083
}
