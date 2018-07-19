; Test basic EfficiencySanitizer working set instrumentation.
;
; RUN: opt < %s -esan -esan-working-set -S | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Intra-cache-line

define i8 @aligned1(i8* %a) {
entry:
  %tmp1 = load i8, i8* %a, align 1
  ret i8 %tmp1
; CHECK: @llvm.global_ctors = {{.*}}@esan.module_ctor
; CHECK:        %0 = ptrtoint i8* %a to i64
; CHECK-NEXT:   %1 = and i64 %0, 17592186044415
; CHECK-NEXT:   %2 = add i64 %1, 1337006139375616
; CHECK-NEXT:   %3 = lshr i64 %2, 6
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   %5 = load i8, i8* %4
; CHECK-NEXT:   %6 = and i8 %5, -127
; CHECK-NEXT:   %7 = icmp ne i8 %6, -127
; CHECK-NEXT:   br i1 %7, label %8, label %11
; CHECK:        %9 = or i8 %5, -127
; CHECK-NEXT:   %10 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   store i8 %9, i8* %10
; CHECK-NEXT:   br label %11
; CHECK:        %tmp1 = load i8, i8* %a, align 1
; CHECK-NEXT:   ret i8 %tmp1
}

define i16 @aligned2(i16* %a) {
entry:
  %tmp1 = load i16, i16* %a, align 2
  ret i16 %tmp1
; CHECK:        %0 = ptrtoint i16* %a to i64
; CHECK-NEXT:   %1 = and i64 %0, 17592186044415
; CHECK-NEXT:   %2 = add i64 %1, 1337006139375616
; CHECK-NEXT:   %3 = lshr i64 %2, 6
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   %5 = load i8, i8* %4
; CHECK-NEXT:   %6 = and i8 %5, -127
; CHECK-NEXT:   %7 = icmp ne i8 %6, -127
; CHECK-NEXT:   br i1 %7, label %8, label %11
; CHECK:        %9 = or i8 %5, -127
; CHECK-NEXT:   %10 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   store i8 %9, i8* %10
; CHECK-NEXT:   br label %11
; CHECK:        %tmp1 = load i16, i16* %a, align 2
; CHECK-NEXT:   ret i16 %tmp1
}

define i32 @aligned4(i32* %a) {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
; CHECK:        %0 = ptrtoint i32* %a to i64
; CHECK-NEXT:   %1 = and i64 %0, 17592186044415
; CHECK-NEXT:   %2 = add i64 %1, 1337006139375616
; CHECK-NEXT:   %3 = lshr i64 %2, 6
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   %5 = load i8, i8* %4
; CHECK-NEXT:   %6 = and i8 %5, -127
; CHECK-NEXT:   %7 = icmp ne i8 %6, -127
; CHECK-NEXT:   br i1 %7, label %8, label %11
; CHECK:        %9 = or i8 %5, -127
; CHECK-NEXT:   %10 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   store i8 %9, i8* %10
; CHECK-NEXT:   br label %11
; CHECK:        %tmp1 = load i32, i32* %a, align 4
; CHECK-NEXT:   ret i32 %tmp1
}

define i64 @aligned8(i64* %a) {
entry:
  %tmp1 = load i64, i64* %a, align 8
  ret i64 %tmp1
; CHECK:        %0 = ptrtoint i64* %a to i64
; CHECK-NEXT:   %1 = and i64 %0, 17592186044415
; CHECK-NEXT:   %2 = add i64 %1, 1337006139375616
; CHECK-NEXT:   %3 = lshr i64 %2, 6
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   %5 = load i8, i8* %4
; CHECK-NEXT:   %6 = and i8 %5, -127
; CHECK-NEXT:   %7 = icmp ne i8 %6, -127
; CHECK-NEXT:   br i1 %7, label %8, label %11
; CHECK:        %9 = or i8 %5, -127
; CHECK-NEXT:   %10 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   store i8 %9, i8* %10
; CHECK-NEXT:   br label %11
; CHECK:        %tmp1 = load i64, i64* %a, align 8
; CHECK-NEXT:   ret i64 %tmp1
}

define i128 @aligned16(i128* %a) {
entry:
  %tmp1 = load i128, i128* %a, align 16
  ret i128 %tmp1
; CHECK:        %0 = ptrtoint i128* %a to i64
; CHECK-NEXT:   %1 = and i64 %0, 17592186044415
; CHECK-NEXT:   %2 = add i64 %1, 1337006139375616
; CHECK-NEXT:   %3 = lshr i64 %2, 6
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   %5 = load i8, i8* %4
; CHECK-NEXT:   %6 = and i8 %5, -127
; CHECK-NEXT:   %7 = icmp ne i8 %6, -127
; CHECK-NEXT:   br i1 %7, label %8, label %11
; CHECK:        %9 = or i8 %5, -127
; CHECK-NEXT:   %10 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   store i8 %9, i8* %10
; CHECK-NEXT:   br label %11
; CHECK:        %tmp1 = load i128, i128* %a, align 16
; CHECK-NEXT:   ret i128 %tmp1
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Not guaranteed to be intra-cache-line, but our defaults are to
; assume they are:

define i16 @unaligned2(i16* %a) {
entry:
  %tmp1 = load i16, i16* %a, align 1
  ret i16 %tmp1
; CHECK:        %0 = ptrtoint i16* %a to i64
; CHECK-NEXT:   %1 = and i64 %0, 17592186044415
; CHECK-NEXT:   %2 = add i64 %1, 1337006139375616
; CHECK-NEXT:   %3 = lshr i64 %2, 6
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   %5 = load i8, i8* %4
; CHECK-NEXT:   %6 = and i8 %5, -127
; CHECK-NEXT:   %7 = icmp ne i8 %6, -127
; CHECK-NEXT:   br i1 %7, label %8, label %11
; CHECK:        %9 = or i8 %5, -127
; CHECK-NEXT:   %10 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   store i8 %9, i8* %10
; CHECK-NEXT:   br label %11
; CHECK:        %tmp1 = load i16, i16* %a, align 1
; CHECK-NEXT:   ret i16 %tmp1
}

define i32 @unaligned4(i32* %a) {
entry:
  %tmp1 = load i32, i32* %a, align 2
  ret i32 %tmp1
; CHECK:        %0 = ptrtoint i32* %a to i64
; CHECK-NEXT:   %1 = and i64 %0, 17592186044415
; CHECK-NEXT:   %2 = add i64 %1, 1337006139375616
; CHECK-NEXT:   %3 = lshr i64 %2, 6
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   %5 = load i8, i8* %4
; CHECK-NEXT:   %6 = and i8 %5, -127
; CHECK-NEXT:   %7 = icmp ne i8 %6, -127
; CHECK-NEXT:   br i1 %7, label %8, label %11
; CHECK:        %9 = or i8 %5, -127
; CHECK-NEXT:   %10 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   store i8 %9, i8* %10
; CHECK-NEXT:   br label %11
; CHECK:        %tmp1 = load i32, i32* %a, align 2
; CHECK-NEXT:   ret i32 %tmp1
}

define i64 @unaligned8(i64* %a) {
entry:
  %tmp1 = load i64, i64* %a, align 4
  ret i64 %tmp1
; CHECK:        %0 = ptrtoint i64* %a to i64
; CHECK-NEXT:   %1 = and i64 %0, 17592186044415
; CHECK-NEXT:   %2 = add i64 %1, 1337006139375616
; CHECK-NEXT:   %3 = lshr i64 %2, 6
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   %5 = load i8, i8* %4
; CHECK-NEXT:   %6 = and i8 %5, -127
; CHECK-NEXT:   %7 = icmp ne i8 %6, -127
; CHECK-NEXT:   br i1 %7, label %8, label %11
; CHECK:        %9 = or i8 %5, -127
; CHECK-NEXT:   %10 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   store i8 %9, i8* %10
; CHECK-NEXT:   br label %11
; CHECK:        %tmp1 = load i64, i64* %a, align 4
; CHECK-NEXT:   ret i64 %tmp1
}

define i128 @unaligned16(i128* %a) {
entry:
  %tmp1 = load i128, i128* %a, align 8
  ret i128 %tmp1
; CHECK:        %0 = ptrtoint i128* %a to i64
; CHECK-NEXT:   %1 = and i64 %0, 17592186044415
; CHECK-NEXT:   %2 = add i64 %1, 1337006139375616
; CHECK-NEXT:   %3 = lshr i64 %2, 6
; CHECK-NEXT:   %4 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   %5 = load i8, i8* %4
; CHECK-NEXT:   %6 = and i8 %5, -127
; CHECK-NEXT:   %7 = icmp ne i8 %6, -127
; CHECK-NEXT:   br i1 %7, label %8, label %11
; CHECK:        %9 = or i8 %5, -127
; CHECK-NEXT:   %10 = inttoptr i64 %3 to i8*
; CHECK-NEXT:   store i8 %9, i8* %10
; CHECK-NEXT:   br label %11
; CHECK:        %tmp1 = load i128, i128* %a, align 8
; CHECK-NEXT:   ret i128 %tmp1
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Ensure that esan converts intrinsics to calls:

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
; Ensure that esan doesn't convert element atomic memory intrinsics to
; calls.

declare void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* nocapture writeonly, i8, i64, i32) nounwind
declare void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32) nounwind
declare void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32) nounwind

define void @elementAtomic_memCpyTest(i8* nocapture %x, i8* nocapture %y) {
  ; CHECK-LABEL: elementAtomic_memCpyTest
  ; CHECK-NEXT: tail call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %x, i8* align 1 %y, i64 16, i32 1)
  ; CHECK-NEXT: ret void
  tail call void @llvm.memcpy.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %x, i8* align 1 %y, i64 16, i32 1)
  ret void
}

define void @elementAtomic_memMoveTest(i8* nocapture %x, i8* nocapture %y) {
  ; CHECK-LABEL: elementAtomic_memMoveTest
  ; CHECK-NEXT:  tail call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %x, i8* align 1 %y, i64 16, i32 1)
  ; CHECK-NEXT: ret void
  tail call void @llvm.memmove.element.unordered.atomic.p0i8.p0i8.i64(i8* align 1 %x, i8* align 1 %y, i64 16, i32 1)
  ret void
}

define void @elementAtomic_memSetTest(i8* nocapture %x) {
  ; CHECK-LABEL: elementAtomic_memSetTest
  ; CHECK-NEXT: tail call void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* align 1 %x, i8 77, i64 16, i32 1)
  ; CHECK-NEXT: ret void
  tail call void @llvm.memset.element.unordered.atomic.p0i8.i64(i8* align 1 %x, i8 77, i64 16, i32 1)
  ret void
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Top-level:

; CHECK: define internal void @esan.module_ctor()
; CHECK: call void @__esan_init(i32 2, i8* null)
; CHECK: define internal void @esan.module_dtor()
; CHECK: call void @__esan_exit(i8* null)
