; With Darwin PCS, non-virtual thunks generated are generated with musttail
; and are expected to build
; In general Darwin PCS should be tail optimized
; RUN: llc -mtriple=arm64-apple-ios5.0.0 < %s | FileCheck %s

; CHECK-LABEL: __ZThn16_N1C3addEPKcz:
; CHECK:       b __ZN1C3addEPKcz
; CHECK-LABEL: _tailTest:
; CHECK:       b __ZN1C3addEPKcz
; CHECK-LABEL: __ZThn8_N1C1fEiiiiiiiiiz:
; CHECK:       ldr     w9, [sp, #4]
; CHECK:       str     w9, [sp, #4]
; CHECK:       b __ZN1C1fEiiiiiiiiiz

%class.C = type { %class.A.base, [4 x i8], %class.B.base, [4 x i8] }
%class.A.base = type <{ ptr, i32 }>
%class.B.base = type <{ ptr, i32 }>

declare void @_ZN1C3addEPKcz(ptr, ptr, ...) unnamed_addr #0 align 2

define void @_ZThn16_N1C3addEPKcz(ptr %0, ptr %1, ...) unnamed_addr #0 align 2 {
  musttail call void (ptr, ptr, ...) @_ZN1C3addEPKcz(ptr noundef nonnull align 8 dereferenceable(28) undef, ptr noundef %1, ...)
  ret void
}

define void @tailTest(ptr %0, ptr %1, ...) unnamed_addr #0 align 2 {
  tail call void (ptr, ptr, ...) @_ZN1C3addEPKcz(ptr noundef nonnull align 8 dereferenceable(28) undef, ptr noundef %1)
  ret void
}

declare void @_ZN1C1fEiiiiiiiiiz(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 noundef %9, ...) unnamed_addr #1 align 2

define void @_ZThn8_N1C1fEiiiiiiiiiz(ptr %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 noundef %9, ...) unnamed_addr #1 align 2 {
  musttail call void (ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, ...) @_ZN1C1fEiiiiiiiiiz(ptr nonnull align 8 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 noundef %9, ...)
  ret void
}
