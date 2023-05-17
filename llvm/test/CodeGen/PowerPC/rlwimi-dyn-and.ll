; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i32 @test1() #0 {
entry:
  %conv67.reload = load i32, ptr undef
  %const = bitcast i32 65535 to i32
  br label %next

next:
  %shl161 = shl nuw nsw i32 %conv67.reload, 15
  %0 = load i8, ptr undef, align 1
  %conv169 = zext i8 %0 to i32
  %shl170 = shl nuw nsw i32 %conv169, 7
  %const_mat = add i32 %const, -32767
  %shl161.masked = and i32 %shl161, %const_mat
  %conv174 = or i32 %shl170, %shl161.masked
  ret i32 %conv174

; CHECK-LABEL: @test1
; CHECK-NOT: rlwimi 3, {{[0-9]+}}, 15, 0, 16
; CHECK: blr
}

define i32 @test2() #0 {
entry:
  %conv67.reload = load i32, ptr undef
  %const = bitcast i32 65535 to i32
  br label %next

next:
  %shl161 = shl nuw nsw i32 %conv67.reload, 15
  %0 = load i8, ptr undef, align 1
  %conv169 = zext i8 %0 to i32
  %shl170 = shl nuw nsw i32 %conv169, 7
  %shl161.masked = and i32 %shl161, 32768
  %conv174 = or i32 %shl170, %shl161.masked
  ret i32 %conv174

; CHECK-LABEL: @test2
; CHECK: rlwinm 3, {{[0-9]+}}, 7, 17, 24
; CHECK: rlwimi 3, {{[0-9]+}}, 15, 16, 16
; CHECK: blr
}

attributes #0 = { nounwind }

