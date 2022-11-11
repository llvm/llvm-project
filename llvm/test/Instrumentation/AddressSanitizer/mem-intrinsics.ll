; Test memory intrinsics instrumentation

; RUN: opt < %s -passes=asan -S | FileCheck --check-prefixes=CHECK,CHECK-PREFIX %s
; RUN: opt < %s -passes=asan -asan-kernel -S | FileCheck --check-prefixes=CHECK,CHECK-NOPREFIX %s
; RUN: opt < %s -passes=asan -asan-kernel -asan-kernel-mem-intrinsic-prefix -S | FileCheck --check-prefixes=CHECK,CHECK-PREFIX %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
declare void @llvm.memset.inline.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) nounwind
declare void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) nounwind

define void @memintr_test(i8* %a, i8* %b) nounwind uwtable sanitize_address {
  entry:
  tail call void @llvm.memset.p0i8.i64(i8* %a, i8 0, i64 100, i1 false)
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* %a, i8* %b, i64 100, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 100, i1 false)
  ret void
}
; CHECK-LABEL: memintr_test
; CHECK-PREFIX: @__asan_memset
; CHECK-PREFIX: @__asan_memmove
; CHECK-PREFIX: @__asan_memcpy
; CHECK-NOPREFIX: @memset
; CHECK-NOPREFIX: @memmove
; CHECK-NOPREFIX: @memcpy
; CHECK: ret void

define void @memintr_inline_test(i8* %a, i8* %b) nounwind uwtable sanitize_address {
  entry:
  tail call void @llvm.memset.inline.p0i8.i64(i8* %a, i8 0, i64 100, i1 false)
  tail call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* %a, i8* %b, i64 100, i1 false)
  ret void
}
; CHECK-LABEL: memintr_inline_test
; CHECK-PREFIX: @__asan_memset
; CHECK-PREFIX: @__asan_memcpy
; CHECK-NOPREFIX: @memset
; CHECK-NOPREFIX: @memcpy
; CHECK: ret void

define void @memintr_test_nosanitize(i8* %a, i8* %b) nounwind uwtable {
  entry:
  tail call void @llvm.memset.p0i8.i64(i8* %a, i8 0, i64 100, i1 false)
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* %a, i8* %b, i64 100, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 100, i1 false)
  ret void
}
; CHECK-LABEL: memintr_test_nosanitize
; CHECK: @llvm.memset
; CHECK: @llvm.memmove
; CHECK: @llvm.memcpy
; CHECK: ret void

declare void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* nocapture writeonly, i8, i64, i32) nounwind
declare void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32) nounwind
declare void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32) nounwind

define void @memintr_element_atomic_test(i8* %a, i8* %b) nounwind uwtable sanitize_address {
  ; This is a canary test to make sure that these don't get lowered into calls that don't
  ; have the element-atomic property. Eventually, asan will have to be enhanced to lower
  ; these properly.
  ; CHECK-LABEL: memintr_element_atomic_test
  ; CHECK-NEXT: tail call void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* align 1 %a, i8 0, i64 100, i32 1)
  ; CHECK-NEXT: tail call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %a, i8* align 1 %b, i64 100, i32 1)
  ; CHECK-NEXT: tail call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %a, i8* align 1 %b, i64 100, i32 1)
  ; CHECK-NEXT: ret void
  tail call void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* align 1 %a, i8 0, i64 100, i32 1)
  tail call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %a, i8* align 1 %b, i64 100, i32 1)
  tail call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %a, i8* align 1 %b, i64 100, i32 1)
  ret void
}
