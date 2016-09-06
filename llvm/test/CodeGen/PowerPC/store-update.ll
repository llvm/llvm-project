; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i8* @test_stbu(i8* %base, i8 zeroext %val) nounwind {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %base, i64 16
  store i8 %val, i8* %arrayidx, align 1
  ret i8* %arrayidx
}
; CHECK: @test_stbu
; CHECK: %entry
; CHECK-NEXT: stbu
; CHECK-NEXT: blr

define i8* @test_stbux(i8* %base, i8 zeroext %val, i64 %offset) nounwind {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %base, i64 %offset
  store i8 %val, i8* %arrayidx, align 1
  ret i8* %arrayidx
}
; CHECK: @test_stbux
; CHECK: %entry
; CHECK-NEXT: stbux
; CHECK-NEXT: blr

define i16* @test_sthu(i16* %base, i16 zeroext %val) nounwind {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %base, i64 16
  store i16 %val, i16* %arrayidx, align 2
  ret i16* %arrayidx
}
; CHECK: @test_sthu
; CHECK: %entry
; CHECK-NEXT: sthu
; CHECK-NEXT: blr

define i16* @test_sthux(i16* %base, i16 zeroext %val, i64 %offset) nounwind {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %base, i64 %offset
  store i16 %val, i16* %arrayidx, align 2
  ret i16* %arrayidx
}
; CHECK: @test_sthux
; CHECK: %entry
; CHECK-NEXT: sldi
; CHECK-NEXT: sthux
; CHECK-NEXT: blr

define i32* @test_stwu(i32* %base, i32 zeroext %val) nounwind {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 16
  store i32 %val, i32* %arrayidx, align 4
  ret i32* %arrayidx
}
; CHECK: @test_stwu
; CHECK: %entry
; CHECK-NEXT: stwu
; CHECK-NEXT: blr

define i32* @test_stwux(i32* %base, i32 zeroext %val, i64 %offset) nounwind {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %offset
  store i32 %val, i32* %arrayidx, align 4
  ret i32* %arrayidx
}
; CHECK: @test_stwux
; CHECK: %entry
; CHECK-NEXT: sldi
; CHECK-NEXT: stwux
; CHECK-NEXT: blr

define i8* @test_stbu8(i8* %base, i64 %val) nounwind {
entry:
  %conv = trunc i64 %val to i8
  %arrayidx = getelementptr inbounds i8, i8* %base, i64 16
  store i8 %conv, i8* %arrayidx, align 1
  ret i8* %arrayidx
}
; CHECK: @test_stbu8
; CHECK: %entry
; CHECK-NEXT: stbu
; CHECK-NEXT: blr

define i8* @test_stbux8(i8* %base, i64 %val, i64 %offset) nounwind {
entry:
  %conv = trunc i64 %val to i8
  %arrayidx = getelementptr inbounds i8, i8* %base, i64 %offset
  store i8 %conv, i8* %arrayidx, align 1
  ret i8* %arrayidx
}
; CHECK: @test_stbux8
; CHECK: %entry
; CHECK-NEXT: stbux
; CHECK-NEXT: blr

define i16* @test_sthu8(i16* %base, i64 %val) nounwind {
entry:
  %conv = trunc i64 %val to i16
  %arrayidx = getelementptr inbounds i16, i16* %base, i64 16
  store i16 %conv, i16* %arrayidx, align 2
  ret i16* %arrayidx
}
; CHECK: @test_sthu
; CHECK: %entry
; CHECK-NEXT: sthu
; CHECK-NEXT: blr

define i16* @test_sthux8(i16* %base, i64 %val, i64 %offset) nounwind {
entry:
  %conv = trunc i64 %val to i16
  %arrayidx = getelementptr inbounds i16, i16* %base, i64 %offset
  store i16 %conv, i16* %arrayidx, align 2
  ret i16* %arrayidx
}
; CHECK: @test_sthux
; CHECK: %entry
; CHECK-NEXT: sldi
; CHECK-NEXT: sthux
; CHECK-NEXT: blr

define i32* @test_stwu8(i32* %base, i64 %val) nounwind {
entry:
  %conv = trunc i64 %val to i32
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 16
  store i32 %conv, i32* %arrayidx, align 4
  ret i32* %arrayidx
}
; CHECK: @test_stwu
; CHECK: %entry
; CHECK-NEXT: stwu
; CHECK-NEXT: blr

define i32* @test_stwux8(i32* %base, i64 %val, i64 %offset) nounwind {
entry:
  %conv = trunc i64 %val to i32
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %offset
  store i32 %conv, i32* %arrayidx, align 4
  ret i32* %arrayidx
}
; CHECK: @test_stwux
; CHECK: %entry
; CHECK-NEXT: sldi
; CHECK-NEXT: stwux
; CHECK-NEXT: blr

define i64* @test_stdu(i64* %base, i64 %val) nounwind {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %base, i64 16
  store i64 %val, i64* %arrayidx, align 8
  ret i64* %arrayidx
}
; CHECK: @test_stdu
; CHECK: %entry
; CHECK-NEXT: stdu
; CHECK-NEXT: blr

define i64* @test_stdux(i64* %base, i64 %val, i64 %offset) nounwind {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %base, i64 %offset
  store i64 %val, i64* %arrayidx, align 8
  ret i64* %arrayidx
}
; CHECK: @test_stdux
; CHECK: %entry
; CHECK-NEXT: sldi
; CHECK-NEXT: stdux
; CHECK-NEXT: blr

