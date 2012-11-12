; RUN: opt < %s -scalarrepl -S | FileCheck %s
; Radar 7552893

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"

%struct.test = type { [3 x double] }

define void @test_memcpy_self() nounwind {
; CHECK: @test_memcpy_self
; CHECK-NOT: alloca
; CHECK: ret void
  %1 = alloca %struct.test
  %2 = bitcast %struct.test* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %2, i8* %2, i32 24, i32 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
