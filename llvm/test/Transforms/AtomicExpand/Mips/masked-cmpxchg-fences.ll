; RUN: opt -S -mtriple=mipsel -passes='require<libcall-lowering-info>,atomic-expand' %s | FileCheck %s --implicit-check-not='{{^[ \t]+fence[ \t]}}'
; RUN: opt -S -mtriple=mips -passes='require<libcall-lowering-info>,atomic-expand' %s | FileCheck %s --implicit-check-not='{{^[ \t]+fence[ \t]}}'
; RUN: opt -S -mtriple=mips64el -passes='require<libcall-lowering-info>,atomic-expand' %s | FileCheck %s --implicit-check-not='{{^[ \t]+fence[ \t]}}'
; RUN: opt -S -mtriple=mips64 -passes='require<libcall-lowering-info>,atomic-expand' %s | FileCheck %s --implicit-check-not='{{^[ \t]+fence[ \t]}}'

; Acquire needs only a trailing fence; release needs only a leading fence.

define { i8, i1 } @cmpxchg_acquire_i8(ptr %ptr, i8 %cmp, i8 %val) {
; CHECK-LABEL: @cmpxchg_acquire_i8(
; CHECK:       call i32 @llvm.mips.masked.cmpxchg.p0(
; CHECK-NEXT:  fence acquire
; CHECK:       ret { i8, i1 }
  %pair = cmpxchg ptr %ptr, i8 %cmp, i8 %val acquire monotonic
  ret { i8, i1 } %pair
}

define { i16, i1 } @cmpxchg_acquire_i16(ptr %ptr, i16 %cmp, i16 %val) {
; CHECK-LABEL: @cmpxchg_acquire_i16(
; CHECK:       call i32 @llvm.mips.masked.cmpxchg.p0(
; CHECK-NEXT:  fence acquire
; CHECK:       ret { i16, i1 }
  %pair = cmpxchg ptr %ptr, i16 %cmp, i16 %val acquire monotonic
  ret { i16, i1 } %pair
}

define { i8, i1 } @cmpxchg_release_i8(ptr %ptr, i8 %cmp, i8 %val) {
; CHECK-LABEL: @cmpxchg_release_i8(
; CHECK:       fence release
; CHECK-NEXT:  call i32 @llvm.mips.masked.cmpxchg.p0(
; CHECK:       ret { i8, i1 }
  %pair = cmpxchg ptr %ptr, i8 %cmp, i8 %val release monotonic
  ret { i8, i1 } %pair
}

define { i16, i1 } @cmpxchg_release_i16(ptr %ptr, i16 %cmp, i16 %val) {
; CHECK-LABEL: @cmpxchg_release_i16(
; CHECK:       fence release
; CHECK-NEXT:  call i32 @llvm.mips.masked.cmpxchg.p0(
; CHECK:       ret { i16, i1 }
  %pair = cmpxchg ptr %ptr, i16 %cmp, i16 %val release monotonic
  ret { i16, i1 } %pair
}
