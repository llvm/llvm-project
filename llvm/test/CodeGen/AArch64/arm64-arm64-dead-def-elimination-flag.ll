; RUN: llc -mtriple=arm64-apple-ios7.0.0 -aarch64-enable-dead-defs=false < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Function Attrs: nounwind ssp uwtable
define i32 @test1() #0 {
  %tmp1 = alloca i8
  %tmp2 = icmp eq i8* %tmp1, null
  %tmp3 = zext i1 %tmp2 to i32

  ret i32 %tmp3

  ; CHECK-LABEL: test1
  ; CHECK: adds {{x[0-9]+}}, sp, #15
}
