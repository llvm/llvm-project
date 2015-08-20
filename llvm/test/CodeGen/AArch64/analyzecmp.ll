; RUN: llc -O3 -mcpu=cortex-a57 < %s | FileCheck %s 

; CHECK-LABLE: @test
; CHECK: tst [[CMP:x[0-9]+]], #0x8000000000000000
; CHECK: csel [[R0:x[0-9]+]], [[S0:x[0-9]+]], [[S1:x[0-9]+]], eq
; CHECK: csel [[R1:x[0-9]+]], [[S2:x[0-9]+]], [[S3:x[0-9]+]], eq
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "arm64--linux-gnueabi"

define void @test(i64 %a, i64* %ptr1, i64* %ptr2) #0 align 2 {
entry:
  %conv = and i64 %a, 4294967295
  %add = add nsw i64 %conv, -1
  %div = sdiv i64 %add, 64
  %rem = srem i64 %add, 64
  %cmp = icmp slt i64 %rem, 0
  br i1 %cmp, label %if.then, label %exit

if.then:                                
  %add2 = add nsw i64 %rem, 64
  %add3 = add i64 %div, -1
  br label %exit

exit:                 
  %__n = phi i64 [ %add3, %if.then ], [ %div, %entry ]
  %__n.0 = phi i64 [ %add2, %if.then ], [ %rem, %entry ]
  store i64 %__n, i64* %ptr1
  store i64 %__n.0, i64* %ptr2
  ret void 
}


