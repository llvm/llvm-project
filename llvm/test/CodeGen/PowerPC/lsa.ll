; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=ppc64 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define signext i32 @foo() #0 {
entry:
  %v = alloca [8200 x i32], align 4
  %w = alloca [8200 x i32], align 4
  %q = alloca [8200 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 32800, ptr %v) #0
  call void @llvm.lifetime.start.p0(i64 32800, ptr %w) #0
  call void @llvm.lifetime.start.p0(i64 32800, ptr %q) #0
  call void @bar(ptr %q, ptr %v, ptr %w) #0
  %0 = load i32, ptr %w, align 4
  %arrayidx3 = getelementptr inbounds [8200 x i32], ptr %w, i64 0, i64 1
  %1 = load i32, ptr %arrayidx3, align 4

; CHECK: @foo
; CHECK-NOT: lwzx
; CHECK: lwz {{[0-9]+}}, 0([[REG:[0-9]+]])
; CHECK: lwz {{[0-9]+}}, 4([[REG]])
; CHECK: blr

  %add = add nsw i32 %1, %0
  call void @llvm.lifetime.end.p0(i64 32800, ptr %q) #0
  call void @llvm.lifetime.end.p0(i64 32800, ptr %w) #0
  call void @llvm.lifetime.end.p0(i64 32800, ptr %v) #0
  ret i32 %add
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #0

declare void @bar(ptr, ptr, ptr)

declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #0

attributes #0 = { nounwind }
