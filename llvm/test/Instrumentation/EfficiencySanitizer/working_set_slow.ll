; Test basic EfficiencySanitizer slowpath instrumentation.
;
; RUN: opt < %s -esan -esan-working-set -esan-instrument-fastpath=false -S | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Aligned loads:

define i8 @loadAligned1(i8* %a) {
entry:
  %tmp1 = load i8, i8* %a, align 1
  ret i8 %tmp1
; CHECK: @llvm.global_ctors = {{.*}}@esan.module_ctor
; CHECK:        call void @__esan_aligned_load1(i8* %a)
; CHECK-NEXT:   %tmp1 = load i8, i8* %a, align 1
; CHECK-NEXT:   ret i8 %tmp1
}

define i16 @loadAligned2(i16* %a) {
entry:
  %tmp1 = load i16, i16* %a, align 2
  ret i16 %tmp1
; CHECK:        %0 = bitcast i16* %a to i8*
; CHECK-NEXT:   call void @__esan_aligned_load2(i8* %0)
; CHECK-NEXT:   %tmp1 = load i16, i16* %a, align 2
; CHECK-NEXT:   ret i16 %tmp1
}

define i32 @loadAligned4(i32* %a) {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
; CHECK:        %0 = bitcast i32* %a to i8*
; CHECK-NEXT:   call void @__esan_aligned_load4(i8* %0)
; CHECK-NEXT:   %tmp1 = load i32, i32* %a, align 4
; CHECK-NEXT:   ret i32 %tmp1
}

define i64 @loadAligned8(i64* %a) {
entry:
  %tmp1 = load i64, i64* %a, align 8
  ret i64 %tmp1
; CHECK:        %0 = bitcast i64* %a to i8*
; CHECK-NEXT:   call void @__esan_aligned_load8(i8* %0)
; CHECK-NEXT:   %tmp1 = load i64, i64* %a, align 8
; CHECK-NEXT:   ret i64 %tmp1
}

define i128 @loadAligned16(i128* %a) {
entry:
  %tmp1 = load i128, i128* %a, align 16
  ret i128 %tmp1
; CHECK:        %0 = bitcast i128* %a to i8*
; CHECK-NEXT:   call void @__esan_aligned_load16(i8* %0)
; CHECK-NEXT:   %tmp1 = load i128, i128* %a, align 16
; CHECK-NEXT:   ret i128 %tmp1
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Aligned stores:

define void @storeAligned1(i8* %a) {
entry:
  store i8 1, i8* %a, align 1
  ret void
; CHECK:        call void @__esan_aligned_store1(i8* %a)
; CHECK-NEXT:   store i8 1, i8* %a, align 1
; CHECK-NEXT:   ret void
}

define void @storeAligned2(i16* %a) {
entry:
  store i16 1, i16* %a, align 2
  ret void
; CHECK:        %0 = bitcast i16* %a to i8*
; CHECK-NEXT:   call void @__esan_aligned_store2(i8* %0)
; CHECK-NEXT:   store i16 1, i16* %a, align 2
; CHECK-NEXT:   ret void
}

define void @storeAligned4(i32* %a) {
entry:
  store i32 1, i32* %a, align 4
  ret void
; CHECK:        %0 = bitcast i32* %a to i8*
; CHECK-NEXT:   call void @__esan_aligned_store4(i8* %0)
; CHECK-NEXT:   store i32 1, i32* %a, align 4
; CHECK-NEXT:   ret void
}

define void @storeAligned8(i64* %a) {
entry:
  store i64 1, i64* %a, align 8
  ret void
; CHECK:        %0 = bitcast i64* %a to i8*
; CHECK-NEXT:   call void @__esan_aligned_store8(i8* %0)
; CHECK-NEXT:   store i64 1, i64* %a, align 8
; CHECK-NEXT:   ret void
}

define void @storeAligned16(i128* %a) {
entry:
  store i128 1, i128* %a, align 16
  ret void
; CHECK:        %0 = bitcast i128* %a to i8*
; CHECK-NEXT:   call void @__esan_aligned_store16(i8* %0)
; CHECK-NEXT:   store i128 1, i128* %a, align 16
; CHECK-NEXT:   ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Unaligned loads:

define i16 @loadUnaligned2(i16* %a) {
entry:
  %tmp1 = load i16, i16* %a, align 1
  ret i16 %tmp1
; CHECK:        %0 = bitcast i16* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_load2(i8* %0)
; CHECK-NEXT:   %tmp1 = load i16, i16* %a, align 1
; CHECK-NEXT:   ret i16 %tmp1
}

define i32 @loadUnaligned4(i32* %a) {
entry:
  %tmp1 = load i32, i32* %a, align 1
  ret i32 %tmp1
; CHECK:        %0 = bitcast i32* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_load4(i8* %0)
; CHECK-NEXT:   %tmp1 = load i32, i32* %a, align 1
; CHECK-NEXT:   ret i32 %tmp1
}

define i64 @loadUnaligned8(i64* %a) {
entry:
  %tmp1 = load i64, i64* %a, align 1
  ret i64 %tmp1
; CHECK:        %0 = bitcast i64* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_load8(i8* %0)
; CHECK-NEXT:   %tmp1 = load i64, i64* %a, align 1
; CHECK-NEXT:   ret i64 %tmp1
}

define i128 @loadUnaligned16(i128* %a) {
entry:
  %tmp1 = load i128, i128* %a, align 1
  ret i128 %tmp1
; CHECK:        %0 = bitcast i128* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_load16(i8* %0)
; CHECK-NEXT:   %tmp1 = load i128, i128* %a, align 1
; CHECK-NEXT:   ret i128 %tmp1
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Unaligned stores:

define void @storeUnaligned2(i16* %a) {
entry:
  store i16 1, i16* %a, align 1
  ret void
; CHECK:        %0 = bitcast i16* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_store2(i8* %0)
; CHECK-NEXT:   store i16 1, i16* %a, align 1
; CHECK-NEXT:   ret void
}

define void @storeUnaligned4(i32* %a) {
entry:
  store i32 1, i32* %a, align 1
  ret void
; CHECK:        %0 = bitcast i32* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_store4(i8* %0)
; CHECK-NEXT:   store i32 1, i32* %a, align 1
; CHECK-NEXT:   ret void
}

define void @storeUnaligned8(i64* %a) {
entry:
  store i64 1, i64* %a, align 1
  ret void
; CHECK:        %0 = bitcast i64* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_store8(i8* %0)
; CHECK-NEXT:   store i64 1, i64* %a, align 1
; CHECK-NEXT:   ret void
}

define void @storeUnaligned16(i128* %a) {
entry:
  store i128 1, i128* %a, align 1
  ret void
; CHECK:        %0 = bitcast i128* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_store16(i8* %0)
; CHECK-NEXT:   store i128 1, i128* %a, align 1
; CHECK-NEXT:   ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Unusual loads and stores:

define x86_fp80 @loadUnalignedFP(x86_fp80* %a) {
entry:
  %tmp1 = load x86_fp80, x86_fp80* %a, align 1
  ret x86_fp80 %tmp1
; CHECK:        %0 = bitcast x86_fp80* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_loadN(i8* %0, i64 10)
; CHECK-NEXT:   %tmp1 = load x86_fp80, x86_fp80* %a, align 1
; CHECK-NEXT:   ret x86_fp80 %tmp1
}

define void @storeUnalignedFP(x86_fp80* %a) {
entry:
  store x86_fp80 0xK00000000000000000000, x86_fp80* %a, align 1
  ret void
; CHECK:        %0 = bitcast x86_fp80* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_storeN(i8* %0, i64 10)
; CHECK-NEXT:   store x86_fp80 0xK00000000000000000000, x86_fp80* %a, align 1
; CHECK-NEXT:   ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Ensure that esan converts memcpy intrinsics to calls:

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1)
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1)
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)

define void @memCpyTest(i8* nocapture %x, i8* nocapture %y) {
entry:
    tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %x, i8* %y, i64 16, i32 4, i1 false)
    ret void
; CHECK: define void @memCpyTest
; CHECK: call i8* @memcpy
; CHECK: ret void
}

define void @memMoveTest(i8* nocapture %x, i8* nocapture %y) {
entry:
    tail call void @llvm.memmove.p0i8.p0i8.i64(i8* %x, i8* %y, i64 16, i32 4, i1 false)
    ret void
; CHECK: define void @memMoveTest
; CHECK: call i8* @memmove
; CHECK: ret void
}

define void @memSetTest(i8* nocapture %x) {
entry:
    tail call void @llvm.memset.p0i8.i64(i8* %x, i8 77, i64 16, i32 4, i1 false)
    ret void
; CHECK: define void @memSetTest
; CHECK: call i8* @memset
; CHECK: ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Top-level:

; CHECK: define internal void @esan.module_ctor()
; CHECK: call void @__esan_init(i32 2, i8* null)
; CHECK: define internal void @esan.module_dtor()
; CHECK: call void @__esan_exit(i8* null)
