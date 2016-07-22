; RUN: llc < %s -mtriple=arm64-apple-ios7.0.0 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Function Attrs: nounwind ssp uwtable
define i32 @test1() #0 {
  %tmp1 = alloca i8
  %tmp2 = alloca i32, i32 4096
  %tmp3 = icmp eq i8* %tmp1, null
  %tmp4 = zext i1 %tmp3 to i32

  ret i32 %tmp4

  ; CHECK-LABEL: test1
  ; CHECK:   adds [[TEMP:[a-z0-9]+]], sp, #4, lsl #12
  ; CHECK:   adds [[TEMP]], [[TEMP]], #15
}
