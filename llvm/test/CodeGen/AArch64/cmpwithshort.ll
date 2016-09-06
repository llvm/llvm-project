; RUN: llc < %s -O3 -mtriple=aarch64-eabi | FileCheck %s 

define i16 @test_1cmp_signed_1(i16* %ptr1) {
; CHECK-LABLE: @test_1cmp_signed_1
; CHECK: ldrsh
; CHECK-NEXT: cmn
entry:
  %addr = getelementptr inbounds i16, i16* %ptr1, i16 0
  %val = load i16, i16* %addr, align 2
  %cmp = icmp eq i16 %val, -1
  br i1 %cmp, label %if, label %if.then
if:
  ret i16 1
if.then:
  ret i16 0
}

define i16 @test_1cmp_signed_2(i16* %ptr1) {
; CHECK-LABLE: @test_1cmp_signed_2
; CHECK: ldrsh
; CHECK-NEXT: cmn
entry:
  %addr = getelementptr inbounds i16, i16* %ptr1, i16 0
  %val = load i16, i16* %addr, align 2
  %cmp = icmp sge i16 %val, -1
  br i1 %cmp, label %if, label %if.then
if:
  ret i16 1
if.then:
  ret i16 0
}

define i16 @test_1cmp_unsigned_1(i16* %ptr1) {
; CHECK-LABLE: @test_1cmp_unsigned_1
; CHECK: ldrsh
; CHECK-NEXT: cmn
entry:
  %addr = getelementptr inbounds i16, i16* %ptr1, i16 0
  %val = load i16, i16* %addr, align 2
  %cmp = icmp uge i16 %val, -1
  br i1 %cmp, label %if, label %if.then
if:
  ret i16 1
if.then:
  ret i16 0
}                                           
