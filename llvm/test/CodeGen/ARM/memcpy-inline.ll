; RUN: llc < %s -mtriple=thumbv7-apple-darwin -disable-post-ra | FileCheck %s

; CHECK: ldrd
; CHECK: strd
; CHECK: ldrb

%struct.x = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }

@src = external global %struct.x
@dst = external global %struct.x

define i32 @t() {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds (%struct.x* @dst, i32 0, i32 0), i8* getelementptr inbounds (%struct.x* @src, i32 0, i32 0), i32 11, i32 8, i1 false)
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
