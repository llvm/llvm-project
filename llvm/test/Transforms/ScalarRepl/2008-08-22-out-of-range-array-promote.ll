; RUN: opt < %s -scalarrepl -S | grep "s = alloca .struct.x"
; PR2423
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

%struct.x = type { [1 x i32], i32, i32 }

define i32 @b() nounwind {
entry:
  %s = alloca %struct.x
  %r = alloca %struct.x
  %0 = call i32 @a(%struct.x* %s) nounwind
  %r1 = bitcast %struct.x* %r to i8*
  %s2 = bitcast %struct.x* %s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %r1, i8* %s2, i32 12, i1 false)
  %1 = getelementptr %struct.x, %struct.x* %r, i32 0, i32 0, i32 1
  %2 = load i32, i32* %1, align 4
  ret i32 %2
}

declare i32 @a(%struct.x*)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
