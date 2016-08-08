; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=ppc64 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define signext i32 @foo() #0 {
entry:
  %v = alloca [8200 x i32], align 4
  %w = alloca [8200 x i32], align 4
  %q = alloca [8200 x i32], align 4
  %0 = bitcast [8200 x i32]* %v to i8*
  call void @llvm.lifetime.start(i64 32800, i8* %0) #0
  %1 = bitcast [8200 x i32]* %w to i8*
  call void @llvm.lifetime.start(i64 32800, i8* %1) #0
  %2 = bitcast [8200 x i32]* %q to i8*
  call void @llvm.lifetime.start(i64 32800, i8* %2) #0
  %arraydecay = getelementptr inbounds [8200 x i32], [8200 x i32]* %q, i64 0, i64 0
  %arraydecay1 = getelementptr inbounds [8200 x i32], [8200 x i32]* %v, i64 0, i64 0
  %arraydecay2 = getelementptr inbounds [8200 x i32], [8200 x i32]* %w, i64 0, i64 0
  call void @bar(i32* %arraydecay, i32* %arraydecay1, i32* %arraydecay2) #0
  %3 = load i32, i32* %arraydecay2, align 4
  %arrayidx3 = getelementptr inbounds [8200 x i32], [8200 x i32]* %w, i64 0, i64 1
  %4 = load i32, i32* %arrayidx3, align 4

; CHECK: @foo
; CHECK-NOT: lwzx
; CHECK: lwz {{[0-9]+}}, 4([[REG:[0-9]+]])
; CHECK: lwz {{[0-9]+}}, 0([[REG]])
; CHECK: blr

  %add = add nsw i32 %4, %3
  call void @llvm.lifetime.end(i64 32800, i8* %2) #0
  call void @llvm.lifetime.end(i64 32800, i8* %1) #0
  call void @llvm.lifetime.end(i64 32800, i8* %0) #0
  ret i32 %add
}

declare void @llvm.lifetime.start(i64, i8* nocapture) #0

declare void @bar(i32*, i32*, i32*)

declare void @llvm.lifetime.end(i64, i8* nocapture) #0

attributes #0 = { nounwind }
