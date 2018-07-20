; RUN: llc -march=hexagon < %s | FileCheck %s
; Check for the immediate offset.  It must be a multiple of 8.
; CHECK: dcfetch({{.*}}+#8)
; In 6.2 (which supports v4+ only), we generate indexed dcfetch in all cases
; (unlike in 6.1, which supported v2, where dcfetch did not allow an immediate
; offset).
; For expression %2, where the offset is +9, the offset on dcfetch should be
; a multiple of 8, and the offset of 0 is most likely (although not the only
; possible one).  Check for #0 anyways, if the test fails with a false
; positive, the second check can be eliminated, or rewritten, and in the
; meantime it can help catch real problems.
; CHECK: dcfetch({{.*}}+#0)
target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon"

define void @foo(i8* %addr) nounwind {
entry:
  %addr.addr = alloca i8*, align 4
  store i8* %addr, i8** %addr.addr, align 4
  %0 = load i8*, i8** %addr.addr, align 4
  %1 = getelementptr i8, i8* %0, i32 8
  call void @llvm.prefetch(i8* %1, i32 0, i32 3, i32 1)
  %2 = getelementptr i8, i8* %0, i32 9
  call void @llvm.prefetch(i8* %2, i32 0, i32 3, i32 1)
  ret void
}

declare void @llvm.prefetch(i8* nocapture, i32, i32, i32) nounwind
