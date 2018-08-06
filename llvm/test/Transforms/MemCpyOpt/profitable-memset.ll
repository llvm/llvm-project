; RUN: opt < %s -memcpyopt -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: @foo(
; CHECK-NOT: store
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 2 %2, i8 0, i64 8, i1 false)

define void @foo(i64* nocapture %P) {
entry:
  %0 = bitcast i64* %P to i16*
  %arrayidx = getelementptr inbounds i16, i16* %0, i64 1
  %1 = bitcast i16* %arrayidx to i32*
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 3
  store i16 0, i16* %0, align 2
  store i32 0, i32* %1, align 4
  store i16 0, i16* %arrayidx1, align 2
  ret void
}

