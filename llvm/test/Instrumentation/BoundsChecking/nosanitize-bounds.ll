; RUN: opt < %s -passes=bounds-checking -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; CHECK: @foo
define i32 @foo(i32 %i) nosanitize_bounds {
entry:
  %i.addr = alloca i32, align 4
  %b = alloca [64 x i32], align 16
  store i32 %i, ptr %i.addr, align 4
  %0 = load i32, ptr %i.addr, align 4
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds [64 x i32], ptr %b, i64 0, i64 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  ret i32 %1
; CHECK-NOT: call void @llvm.trap()
}

