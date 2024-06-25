; RUN: llc < %s -O3 -mtriple=aarch64 | FileCheck %s 

define i16 @test_1cmp_signed_1(ptr %ptr1) {
; CHECK-LABEL: @test_1cmp_signed_1
; CHECK: ldrsh
; CHECK-NEXT: cmn
entry:
  %val = load i16, ptr %ptr1, align 2
  %cmp = icmp eq i16 %val, -1
  br i1 %cmp, label %if, label %if.then
if:
  ret i16 1
if.then:
  ret i16 0
}

define i16 @test_1cmp_signed_2(ptr %ptr1) {
; CHECK-LABEL: @test_1cmp_signed_2
; CHECK: ldrsh
; CHECK-NEXT: cmn
entry:
  %val = load i16, ptr %ptr1, align 2
  %cmp = icmp sge i16 %val, -1
  br i1 %cmp, label %if, label %if.then
if:
  ret i16 1
if.then:
  ret i16 0
}

define i16 @test_1cmp_unsigned_1(ptr %ptr1) {
; CHECK-LABEL: @test_1cmp_unsigned_1
; CHECK: ldrsh
; CHECK-NEXT: cmn
entry:
  %val = load i16, ptr %ptr1, align 2
  %cmp = icmp uge i16 %val, -1
  br i1 %cmp, label %if, label %if.then
if:
  ret i16 1
if.then:
  ret i16 0
}                                           
