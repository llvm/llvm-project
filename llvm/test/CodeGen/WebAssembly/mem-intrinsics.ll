; RUN: llc < %s -asm-verbose=false -mcpu=mvp -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -tail-dup-placement=0 | FileCheck %s

; Test memcpy, memmove, and memset intrinsics.

target triple = "wasm32-unknown-unknown"

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1)
declare void @llvm.memmove.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1)
declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1)

; Test that return values are optimized.

; CHECK-LABEL: copy_yes:
; CHECK:      call     $push0=, memcpy, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
define ptr @copy_yes(ptr %dst, ptr %src, i32 %len) {
  call void @llvm.memcpy.p0.p0.i32(ptr %dst, ptr %src, i32 %len, i1 false)
  ret ptr %dst
}

; CHECK-LABEL: copy_no:
; CHECK:      call     $drop=, memcpy, $0, $1, $2{{$}}
; CHECK-NEXT: return{{$}}
define void @copy_no(ptr %dst, ptr %src, i32 %len) {
  call void @llvm.memcpy.p0.p0.i32(ptr %dst, ptr %src, i32 %len, i1 false)
  ret void
}

; CHECK-LABEL: move_yes:
; CHECK:      call     $push0=, memmove, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
define ptr @move_yes(ptr %dst, ptr %src, i32 %len) {
  call void @llvm.memmove.p0.p0.i32(ptr %dst, ptr %src, i32 %len, i1 false)
  ret ptr %dst
}

; CHECK-LABEL: move_no:
; CHECK:      call     $drop=, memmove, $0, $1, $2{{$}}
; CHECK-NEXT: return{{$}}
define void @move_no(ptr %dst, ptr %src, i32 %len) {
  call void @llvm.memmove.p0.p0.i32(ptr %dst, ptr %src, i32 %len, i1 false)
  ret void
}

; CHECK-LABEL: set_yes:
; CHECK:      call     $push0=, memset, $0, $1, $2{{$}}
; CHECK-NEXT: return   $pop0{{$}}
define ptr @set_yes(ptr %dst, i8 %src, i32 %len) {
  call void @llvm.memset.p0.i32(ptr %dst, i8 %src, i32 %len, i1 false)
  ret ptr %dst
}

; CHECK-LABEL: set_no:
; CHECK:      call     $drop=, memset, $0, $1, $2{{$}}
; CHECK-NEXT: return{{$}}
define void @set_no(ptr %dst, i8 %src, i32 %len) {
  call void @llvm.memset.p0.i32(ptr %dst, i8 %src, i32 %len, i1 false)
  ret void
}


; CHECK-LABEL: frame_index:
; CHECK: call $drop=, memset, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK: call $push{{[0-9]+}}=, memset, ${{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK: return{{$}}
define void @frame_index() {
entry:
  %a = alloca [2048 x i8], align 16
  %b = alloca [2048 x i8], align 16
  call void @llvm.memset.p0.i32(ptr align 16 %a, i8 256, i32 1024, i1 false)
  call void @llvm.memset.p0.i32(ptr align 16 %b, i8 256, i32 1024, i1 false)
  ret void
}

; If the result value of memset doesn't get stackified, it should be marked
; $drop. Note that we use a call to prevent tail dup so that we can test
; this specific functionality.

; CHECK-LABEL: drop_result:
; CHECK: call $drop=, memset, $0, $1, $2
declare ptr @def()
declare void @block_tail_dup()
define ptr @drop_result(ptr %arg, i8 %arg1, i32 %arg2, i32 %arg3, i32 %arg4) {
bb:
  %tmp = icmp eq i32 %arg3, 0
  br i1 %tmp, label %bb5, label %bb9

bb5:
  %tmp6 = icmp eq i32 %arg4, 0
  br i1 %tmp6, label %bb7, label %bb8

bb7:
  call void @llvm.memset.p0.i32(ptr %arg, i8 %arg1, i32 %arg2, i1 false)
  br label %bb11

bb8:
  br label %bb11

bb9:
  %tmp10 = call ptr @def()
  br label %bb11

bb11:
  %tmp12 = phi ptr [ %arg, %bb7 ], [ %arg, %bb8 ], [ %tmp10, %bb9 ]
  call void @block_tail_dup()
  ret ptr %tmp12
}

; This is the same as drop_result, except we let tail dup happen, so the
; result of the memset *is* stackified.

; CHECK-LABEL: tail_dup_to_reuse_result:
; CHECK: call $push{{[0-9]+}}=, memset, $0, $1, $2
define ptr @tail_dup_to_reuse_result(ptr %arg, i8 %arg1, i32 %arg2, i32 %arg3, i32 %arg4) {
bb:
  %tmp = icmp eq i32 %arg3, 0
  br i1 %tmp, label %bb5, label %bb9

bb5:
  %tmp6 = icmp eq i32 %arg4, 0
  br i1 %tmp6, label %bb7, label %bb8

bb7:
  call void @llvm.memset.p0.i32(ptr %arg, i8 %arg1, i32 %arg2, i1 false)
  br label %bb11

bb8:
  br label %bb11

bb9:
  %tmp10 = call ptr @def()
  br label %bb11

bb11:
  %tmp12 = phi ptr [ %arg, %bb7 ], [ %arg, %bb8 ], [ %tmp10, %bb9 ]
  ret ptr %tmp12
}
