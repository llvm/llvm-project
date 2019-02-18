; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -tail-dup-placement=0 | FileCheck %s

; Test memcpy, memmove, and memset intrinsics.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1)
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1)
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1)

; Test that return values are optimized.

; CHECK-LABEL: copy_yes:
; CHECK:      i32.call $push0=, memcpy@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
define i8* @copy_yes(i8* %dst, i8* %src, i32 %len) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %len, i1 false)
  ret i8* %dst
}

; CHECK-LABEL: copy_no:
; CHECK:      i32.call $drop=, memcpy@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return{{$}}
define void @copy_no(i8* %dst, i8* %src, i32 %len) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %len, i1 false)
  ret void
}

; CHECK-LABEL: move_yes:
; CHECK:      i32.call $push0=, memmove@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
define i8* @move_yes(i8* %dst, i8* %src, i32 %len) {
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %len, i1 false)
  ret i8* %dst
}

; CHECK-LABEL: move_no:
; CHECK:      i32.call $drop=, memmove@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return{{$}}
define void @move_no(i8* %dst, i8* %src, i32 %len) {
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %len, i1 false)
  ret void
}

; CHECK-LABEL: set_yes:
; CHECK:      i32.call $push0=, memset@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
define i8* @set_yes(i8* %dst, i8 %src, i32 %len) {
  call void @llvm.memset.p0i8.i32(i8* %dst, i8 %src, i32 %len, i1 false)
  ret i8* %dst
}

; CHECK-LABEL: set_no:
; CHECK:      i32.call $drop=, memset@FUNCTION, $0, $1, $2{{$}}
; CHECK-NEXT: return{{$}}
define void @set_no(i8* %dst, i8 %src, i32 %len) {
  call void @llvm.memset.p0i8.i32(i8* %dst, i8 %src, i32 %len, i1 false)
  ret void
}


; CHECK-LABEL: frame_index:
; CHECK: i32.call $drop=, memset@FUNCTION, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK: i32.call $push{{[0-9]+}}=, memset@FUNCTION, ${{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK: return{{$}}
define void @frame_index() {
entry:
  %a = alloca [2048 x i8], align 16
  %b = alloca [2048 x i8], align 16
  %0 = getelementptr inbounds [2048 x i8], [2048 x i8]* %a, i32 0, i32 0
  %1 = getelementptr inbounds [2048 x i8], [2048 x i8]* %b, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* align 16 %0, i8 256, i32 1024, i1 false)
  call void @llvm.memset.p0i8.i32(i8* align 16 %1, i8 256, i32 1024, i1 false)
  ret void
}

; If the result value of memset doesn't get stackified, it should be marked
; $drop. Note that we use a call to prevent tail dup so that we can test
; this specific functionality.

; CHECK-LABEL: drop_result:
; CHECK: i32.call $drop=, memset@FUNCTION, $0, $1, $2
declare i8* @def()
declare void @block_tail_dup()
define i8* @drop_result(i8* %arg, i8 %arg1, i32 %arg2, i32 %arg3, i32 %arg4) {
bb:
  %tmp = icmp eq i32 %arg3, 0
  br i1 %tmp, label %bb5, label %bb9

bb5:
  %tmp6 = icmp eq i32 %arg4, 0
  br i1 %tmp6, label %bb7, label %bb8

bb7:
  call void @llvm.memset.p0i8.i32(i8* %arg, i8 %arg1, i32 %arg2, i1 false)
  br label %bb11

bb8:
  br label %bb11

bb9:
  %tmp10 = call i8* @def()
  br label %bb11

bb11:
  %tmp12 = phi i8* [ %arg, %bb7 ], [ %arg, %bb8 ], [ %tmp10, %bb9 ]
  call void @block_tail_dup()
  ret i8* %tmp12
}

; This is the same as drop_result, except we let tail dup happen, so the
; result of the memset *is* stackified.

; CHECK-LABEL: tail_dup_to_reuse_result:
; CHECK: i32.call $push{{[0-9]+}}=, memset@FUNCTION, $0, $1, $2
define i8* @tail_dup_to_reuse_result(i8* %arg, i8 %arg1, i32 %arg2, i32 %arg3, i32 %arg4) {
bb:
  %tmp = icmp eq i32 %arg3, 0
  br i1 %tmp, label %bb5, label %bb9

bb5:
  %tmp6 = icmp eq i32 %arg4, 0
  br i1 %tmp6, label %bb7, label %bb8

bb7:
  call void @llvm.memset.p0i8.i32(i8* %arg, i8 %arg1, i32 %arg2, i1 false)
  br label %bb11

bb8:
  br label %bb11

bb9:
  %tmp10 = call i8* @def()
  br label %bb11

bb11:
  %tmp12 = phi i8* [ %arg, %bb7 ], [ %arg, %bb8 ], [ %tmp10, %bb9 ]
  ret i8* %tmp12
}
