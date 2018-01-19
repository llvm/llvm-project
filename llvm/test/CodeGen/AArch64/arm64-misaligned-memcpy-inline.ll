; RUN: llc -mtriple=arm64-apple-ios -mattr=+strict-align < %s | FileCheck %s

; Small (16-bytes here) unaligned memcpys should stay memcpy calls if
; strict-alignment is turned on.
define void @t0(i8* %out, i8* %in) {
; CHECK-LABEL: t0:
; CHECK:         orr w2, wzr, #0x10
; CHECK-NEXT:    bl _memcpy
entry:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %out, i8* %in, i64 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)
