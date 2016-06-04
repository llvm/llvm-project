; Test EfficiencySanitizer working set instrumentation without aggressive
; optimization flags.
;
; RUN: opt < %s -esan -esan-working-set -esan-assume-intra-cache-line=0 -S | FileCheck %s

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Not guaranteed to be intra-cache-line

define i16 @unaligned2(i16* %a) {
entry:
  %tmp1 = load i16, i16* %a, align 1
  ret i16 %tmp1
; CHECK:        %0 = bitcast i16* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_load2(i8* %0)
; CHECK-NEXT:   %tmp1 = load i16, i16* %a, align 1
; CHECK-NEXT:   ret i16 %tmp1
}

define i32 @unaligned4(i32* %a) {
entry:
  %tmp1 = load i32, i32* %a, align 2
  ret i32 %tmp1
; CHECK:        %0 = bitcast i32* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_load4(i8* %0)
; CHECK-NEXT:   %tmp1 = load i32, i32* %a, align 2
; CHECK-NEXT:   ret i32 %tmp1
}

define i64 @unaligned8(i64* %a) {
entry:
  %tmp1 = load i64, i64* %a, align 4
  ret i64 %tmp1
; CHECK:        %0 = bitcast i64* %a to i8*
; CHECK-NEXT:   call void @__esan_unaligned_load8(i8* %0)
; CHECK-NEXT:   %tmp1 = load i64, i64* %a, align 4
; CHECK-NEXT:   ret i64 %tmp1
}
