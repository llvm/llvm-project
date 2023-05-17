; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define ptr @test_stbu(ptr %base, i8 zeroext %val) nounwind {
entry:
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 16
  store i8 %val, ptr %arrayidx, align 1
  ret ptr %arrayidx
}
; CHECK: @test_stbu
; CHECK: %entry
; CHECK-NEXT: stbu
; CHECK-NEXT: blr

define ptr @test_stbux(ptr %base, i8 zeroext %val, i64 %offset) nounwind {
entry:
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %offset
  store i8 %val, ptr %arrayidx, align 1
  ret ptr %arrayidx
}
; CHECK: @test_stbux
; CHECK: %entry
; CHECK-NEXT: stbux
; CHECK-NEXT: blr

define ptr @test_sthu(ptr %base, i16 zeroext %val) nounwind {
entry:
  %arrayidx = getelementptr inbounds i16, ptr %base, i64 16
  store i16 %val, ptr %arrayidx, align 2
  ret ptr %arrayidx
}
; CHECK: @test_sthu
; CHECK: %entry
; CHECK-NEXT: sthu
; CHECK-NEXT: blr

define ptr @test_sthux(ptr %base, i16 zeroext %val, i64 %offset) nounwind {
entry:
  %arrayidx = getelementptr inbounds i16, ptr %base, i64 %offset
  store i16 %val, ptr %arrayidx, align 2
  ret ptr %arrayidx
}
; CHECK: @test_sthux
; CHECK: %entry
; CHECK-NEXT: sldi
; CHECK-NEXT: sthux
; CHECK-NEXT: blr

define ptr @test_stwu(ptr %base, i32 zeroext %val) nounwind {
entry:
  %arrayidx = getelementptr inbounds i32, ptr %base, i64 16
  store i32 %val, ptr %arrayidx, align 4
  ret ptr %arrayidx
}
; CHECK: @test_stwu
; CHECK: %entry
; CHECK-NEXT: stwu
; CHECK-NEXT: blr

define ptr @test_stwux(ptr %base, i32 zeroext %val, i64 %offset) nounwind {
entry:
  %arrayidx = getelementptr inbounds i32, ptr %base, i64 %offset
  store i32 %val, ptr %arrayidx, align 4
  ret ptr %arrayidx
}
; CHECK: @test_stwux
; CHECK: %entry
; CHECK-NEXT: sldi
; CHECK-NEXT: stwux
; CHECK-NEXT: blr

define ptr @test_stbu8(ptr %base, i64 %val) nounwind {
entry:
  %conv = trunc i64 %val to i8
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 16
  store i8 %conv, ptr %arrayidx, align 1
  ret ptr %arrayidx
}
; CHECK: @test_stbu8
; CHECK: %entry
; CHECK-NEXT: stbu
; CHECK-NEXT: blr

define ptr @test_stbux8(ptr %base, i64 %val, i64 %offset) nounwind {
entry:
  %conv = trunc i64 %val to i8
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %offset
  store i8 %conv, ptr %arrayidx, align 1
  ret ptr %arrayidx
}
; CHECK: @test_stbux8
; CHECK: %entry
; CHECK-NEXT: stbux
; CHECK-NEXT: blr

define ptr @test_sthu8(ptr %base, i64 %val) nounwind {
entry:
  %conv = trunc i64 %val to i16
  %arrayidx = getelementptr inbounds i16, ptr %base, i64 16
  store i16 %conv, ptr %arrayidx, align 2
  ret ptr %arrayidx
}
; CHECK: @test_sthu
; CHECK: %entry
; CHECK-NEXT: sthu
; CHECK-NEXT: blr

define ptr @test_sthux8(ptr %base, i64 %val, i64 %offset) nounwind {
entry:
  %conv = trunc i64 %val to i16
  %arrayidx = getelementptr inbounds i16, ptr %base, i64 %offset
  store i16 %conv, ptr %arrayidx, align 2
  ret ptr %arrayidx
}
; CHECK: @test_sthux
; CHECK: %entry
; CHECK-NEXT: sldi
; CHECK-NEXT: sthux
; CHECK-NEXT: blr

define ptr @test_stwu8(ptr %base, i64 %val) nounwind {
entry:
  %conv = trunc i64 %val to i32
  %arrayidx = getelementptr inbounds i32, ptr %base, i64 16
  store i32 %conv, ptr %arrayidx, align 4
  ret ptr %arrayidx
}
; CHECK: @test_stwu
; CHECK: %entry
; CHECK-NEXT: stwu
; CHECK-NEXT: blr

define ptr @test_stwux8(ptr %base, i64 %val, i64 %offset) nounwind {
entry:
  %conv = trunc i64 %val to i32
  %arrayidx = getelementptr inbounds i32, ptr %base, i64 %offset
  store i32 %conv, ptr %arrayidx, align 4
  ret ptr %arrayidx
}
; CHECK: @test_stwux
; CHECK: %entry
; CHECK-NEXT: sldi
; CHECK-NEXT: stwux
; CHECK-NEXT: blr

define ptr @test_stdu(ptr %base, i64 %val) nounwind {
entry:
  %arrayidx = getelementptr inbounds i64, ptr %base, i64 16
  store i64 %val, ptr %arrayidx, align 8
  ret ptr %arrayidx
}
; CHECK: @test_stdu
; CHECK: %entry
; CHECK-NEXT: stdu
; CHECK-NEXT: blr

define ptr @test_stdux(ptr %base, i64 %val, i64 %offset) nounwind {
entry:
  %arrayidx = getelementptr inbounds i64, ptr %base, i64 %offset
  store i64 %val, ptr %arrayidx, align 8
  ret ptr %arrayidx
}
; CHECK: @test_stdux
; CHECK: %entry
; CHECK-NEXT: sldi
; CHECK-NEXT: stdux
; CHECK-NEXT: blr

