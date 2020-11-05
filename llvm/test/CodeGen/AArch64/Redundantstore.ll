; RUN: llc < %s -O3 -mtriple=aarch64-eabi | FileCheck %s 

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
@end_of_array = common global i8* null, align 8

; CHECK-LABEL: @test
; CHECK: stur
; CHECK-NOT: stur
define i8* @test(i32 %size) {
entry:
  %0 = load i8*, i8** @end_of_array, align 8
  %conv = sext i32 %size to i64
  %and = and i64 %conv, -8
  %conv2 = trunc i64 %and to i32
  %add.ptr.sum = add nsw i64 %and, -4
  %add.ptr3 = getelementptr inbounds i8, i8* %0, i64 %add.ptr.sum
  %size4 = bitcast i8* %add.ptr3 to i32*
  store i32 %conv2, i32* %size4, align 4
  %add.ptr.sum9 = add nsw i64 %and, -4
  %add.ptr5 = getelementptr inbounds i8, i8* %0, i64 %add.ptr.sum9
  %size6 = bitcast i8* %add.ptr5 to i32*
  store i32 %conv2, i32* %size6, align 4
  ret i8* %0
}

