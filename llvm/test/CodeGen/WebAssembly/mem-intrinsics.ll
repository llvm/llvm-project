; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test memcpy, memmove, and memset intrinsics.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1)
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1)
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1)

; Test that return values are optimized.

; CHECK-LABEL: copy_yes:
; CHECK:      i32.call $push0=, memcpy@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
define i8* @copy_yes(i8* %dst, i8* %src, i32 %len) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %len, i32 1, i1 false)
  ret i8* %dst
}

; CHECK-LABEL: copy_no:
; CHECK:      i32.call $discard=, memcpy@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return{{$}}
define void @copy_no(i8* %dst, i8* %src, i32 %len) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %len, i32 1, i1 false)
  ret void
}

; CHECK-LABEL: move_yes:
; CHECK:      i32.call $push0=, memmove@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
define i8* @move_yes(i8* %dst, i8* %src, i32 %len) {
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %len, i32 1, i1 false)
  ret i8* %dst
}

; CHECK-LABEL: move_no:
; CHECK:      i32.call $discard=, memmove@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return{{$}}
define void @move_no(i8* %dst, i8* %src, i32 %len) {
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %len, i32 1, i1 false)
  ret void
}

; CHECK-LABEL: set_yes:
; CHECK:      i32.call $push0=, memset@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
define i8* @set_yes(i8* %dst, i8 %src, i32 %len) {
  call void @llvm.memset.p0i8.i32(i8* %dst, i8 %src, i32 %len, i32 1, i1 false)
  ret i8* %dst
}

; CHECK-LABEL: set_no:
; CHECK:      i32.call $discard=, memset@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return{{$}}
define void @set_no(i8* %dst, i8 %src, i32 %len) {
  call void @llvm.memset.p0i8.i32(i8* %dst, i8 %src, i32 %len, i32 1, i1 false)
  ret void
}
