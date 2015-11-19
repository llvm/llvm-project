; RUN: opt < %s -scalarrepl -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%struct.S = type { [12 x i32] }

; CHECK-LABEL: @bar4(
define void @bar4(%struct.S* byval %s) nounwind ssp {
entry:
; CHECK: alloca
; CHECK-NOT: load
; CHECK: memcpy
  %t = alloca %struct.S, align 4
  %agg.tmp = alloca %struct.S, align 4
  %tmp = bitcast %struct.S* %t to i8*
  %tmp1 = bitcast %struct.S* %s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp, i8* %tmp1, i64 48, i32 4, i1 false)
  %tmp2 = bitcast %struct.S* %agg.tmp to i8*
  %tmp3 = bitcast %struct.S* %t to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp2, i8* %tmp3, i64 48, i32 4, i1 false)
  %call = call i32 (...) @bazz(%struct.S* byval %agg.tmp)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

declare i32 @bazz(...)
