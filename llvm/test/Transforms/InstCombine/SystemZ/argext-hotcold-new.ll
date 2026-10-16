; RUN: opt -passes=instcombine -mtriple=s390x-linux-gnu \
; RUN:     -optimize-hot-cold-new -optimize-nobuiltin-hot-cold-new-new \
; RUN:     -S %s -o - | FileCheck %s
;
; Check that instcombine's hot-cold-new optimisation emits zeroext on the i8
; __hot_cold_t argument.
;
; The zeroext attribute appears on the function declarations emitted by opt,
; not at the call sites.

; CHECK: declare ptr @_Znwm12__hot_cold_t(i64, i8 zeroext)
; CHECK: declare ptr @_Znam12__hot_cold_t(i64, i8 zeroext)
; CHECK: declare ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64, ptr, i8 zeroext)
; CHECK: declare ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64, i64, i8 zeroext)
; CHECK: declare ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64, i64, ptr, i8 zeroext)
; CHECK: declare ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64, ptr, i8 zeroext)
; CHECK: declare ptr @_ZnamSt11align_val_t12__hot_cold_t(i64, i64, i8 zeroext)
; CHECK: declare ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64, i64, ptr, i8 zeroext)
; CHECK: declare { ptr, i64 } @__size_returning_new_hot_cold(i64, i8 zeroext)
; CHECK: declare { ptr, i64 } @__size_returning_new_aligned_hot_cold(i64, i64, i8 zeroext)

; ------------------------------------------------------------------
; emitHotColdNew  (_Znwm -> _Znwm12__hot_cold_t)
define ptr @new_scalar(i64 %n) {
  %p = call ptr @_Znwm(i64 %n) "memprof"="cold"
  ret ptr %p
}

; emitHotColdNew  (_Znam -> _Znam12__hot_cold_t)
define ptr @new_array(i64 %n) {
  %p = call ptr @_Znam(i64 %n) "memprof"="cold"
  ret ptr %p
}

; emitHotColdNewNoThrow  (_ZnwmRKSt9nothrow_t -> ...12__hot_cold_t)
define ptr @new_scalar_nothrow(i64 %n, ptr %nt) {
  %p = call ptr @_ZnwmRKSt9nothrow_t(i64 %n, ptr %nt) "memprof"="cold"
  ret ptr %p
}

; emitHotColdNewAligned  (_ZnwmSt11align_val_t -> ...12__hot_cold_t)
define ptr @new_scalar_aligned(i64 %n, i64 %al) {
  %p = call ptr @_ZnwmSt11align_val_t(i64 %n, i64 %al) "memprof"="cold"
  ret ptr %p
}

; emitHotColdNewAlignedNoThrow  (_ZnwmSt11align_val_tRKSt9nothrow_t -> ...)
define ptr @new_scalar_aligned_nothrow(i64 %n, i64 %al, ptr %nt) {
  %p = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 %n, i64 %al, ptr %nt) "memprof"="cold"
  ret ptr %p
}

; emitHotColdNewNoThrow  (_ZnamRKSt9nothrow_t -> ...12__hot_cold_t)
define ptr @new_array_nothrow(i64 %n, ptr %nt) {
  %p = call ptr @_ZnamRKSt9nothrow_t(i64 %n, ptr %nt) "memprof"="cold"
  ret ptr %p
}

; emitHotColdNewAligned  (_ZnamSt11align_val_t -> ...12__hot_cold_t)
define ptr @new_array_aligned(i64 %n, i64 %al) {
  %p = call ptr @_ZnamSt11align_val_t(i64 %n, i64 %al) "memprof"="cold"
  ret ptr %p
}

; emitHotColdNewAlignedNoThrow  (_ZnamSt11align_val_tRKSt9nothrow_t -> ...)
define ptr @new_array_aligned_nothrow(i64 %n, i64 %al, ptr %nt) {
  %p = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 %n, i64 %al, ptr %nt) "memprof"="cold"
  ret ptr %p
}

; emitHotColdSizeReturningNew  (__size_returning_new -> __size_returning_new_hot_cold)
; Use a literal (anonymous) struct so isValidProtoForSizeReturningNew matches.
define { ptr, i64 } @new_size_returning(i64 %n) {
  %p = call { ptr, i64 } @__size_returning_new(i64 %n) "memprof"="cold"
  ret { ptr, i64 } %p
}

; emitHotColdSizeReturningNewAligned
;   (__size_returning_new_aligned -> __size_returning_new_aligned_hot_cold)
define { ptr, i64 } @new_size_returning_aligned(i64 %n, i64 %al) {
  %p = call { ptr, i64 } @__size_returning_new_aligned(i64 %n, i64 %al) "memprof"="cold"
  ret { ptr, i64 } %p
}

; ------------------------------------------------------------------
declare ptr @_Znwm(i64)
declare ptr @_Znam(i64)
declare ptr @_ZnwmRKSt9nothrow_t(i64, ptr)
declare ptr @_ZnwmSt11align_val_t(i64, i64)
declare ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64, i64, ptr)
declare ptr @_ZnamRKSt9nothrow_t(i64, ptr)
declare ptr @_ZnamSt11align_val_t(i64, i64)
declare ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64, i64, ptr)
declare { ptr, i64 } @__size_returning_new(i64)
declare { ptr, i64 } @__size_returning_new_aligned(i64, i64)
