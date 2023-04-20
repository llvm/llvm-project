; RUN: opt < %s -passes=instcombine -S | FileCheck %s --implicit-check-not=hot_cold_t
; RUN: opt < %s -passes=instcombine -optimize-hot-cold-new -S | FileCheck %s --check-prefix=HOTCOLD

;; Check that operator new(unsigned long) converted to
;; operator new(unsigned long, hot_cold_t) with a hot or cold attribute.
; HOTCOLD-LABEL: @new()
define void @new() {
  ;; Attribute cold converted to hot_cold_t value 0 (coldest).
  ; HOTCOLD: @_Znwm10hot_cold_t(i64 10, i8 0)
  %call = call ptr @_Znwm(i64 10) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_Znwm(i64 10)
  %call1 = call ptr @_Znwm(i64 10) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to hot_cold_t value 255 (-1) (hottest).
  ; HOTCOLD: @_Znwm10hot_cold_t(i64 10, i8 -1)
  %call2 = call ptr @_Znwm(i64 10) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new(unsigned long, std::align_val_t) converted to
;; operator new(unsigned long, std::align_val_t, hot_cold_t) with a hot or
;; cold attribute.
; HOTCOLD-LABEL: @new_align()
define void @new_align() {
  ;; Attribute cold converted to hot_cold_t value 0 (coldest).
  ; HOTCOLD: @_ZnwmSt11align_val_t10hot_cold_t(i64 10, i64 8, i8 0)
  %call = call ptr @_ZnwmSt11align_val_t(i64 10, i64 8) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnwmSt11align_val_t(i64 10, i64 8)
  %call1 = call ptr @_ZnwmSt11align_val_t(i64 10, i64 8) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to hot_cold_t value 255 (-1) (hottest).
  ; HOTCOLD: @_ZnwmSt11align_val_t10hot_cold_t(i64 10, i64 8, i8 -1)
  %call2 = call ptr @_ZnwmSt11align_val_t(i64 10, i64 8) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new(unsigned long, const std::nothrow_t&) converted to
;; operator new(unsigned long, const std::nothrow_t&, hot_cold_t) with a hot or
;; cold attribute.
; HOTCOLD-LABEL: @new_nothrow()
define void @new_nothrow() {
  %nt = alloca i8
  ;; Attribute cold converted to hot_cold_t value 0 (coldest).
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t10hot_cold_t(i64 10, ptr nonnull %nt, i8 0)
  %call = call ptr @_ZnwmRKSt9nothrow_t(i64 10, ptr %nt) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t(i64 10, ptr nonnull %nt)
  %call1 = call ptr @_ZnwmRKSt9nothrow_t(i64 10, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to hot_cold_t value 255 (-1) (hottest).
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t10hot_cold_t(i64 10, ptr nonnull %nt, i8 -1)
  %call2 = call ptr @_ZnwmRKSt9nothrow_t(i64 10, ptr %nt) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new(unsigned long, std::align_val_t, const std::nothrow_t&)
;; converted to
;; operator new(unsigned long, std::align_val_t, const std::nothrow_t&, hot_cold_t)
;; with a hot or cold attribute.
; HOTCOLD-LABEL: @new_align_nothrow()
define void @new_align_nothrow() {
  %nt = alloca i8
  ;; Attribute cold converted to hot_cold_t value 0 (coldest).
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t10hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 0)
  %call = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr nonnull %nt)
  %call1 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to hot_cold_t value 255 (-1) (hottest).
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t10hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 -1)
  %call2 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new[](unsigned long) converted to
;; operator new[](unsigned long, hot_cold_t) with a hot or cold attribute.
; HOTCOLD-LABEL: @array_new()
define void @array_new() {
  ;; Attribute cold converted to hot_cold_t value 0 (coldest).
  ; HOTCOLD: @_Znam10hot_cold_t(i64 10, i8 0)
  %call = call ptr @_Znam(i64 10) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_Znam(i64 10)
  %call1 = call ptr @_Znam(i64 10) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to hot_cold_t value 255 (-1) (hottest).
  ; HOTCOLD: @_Znam10hot_cold_t(i64 10, i8 -1)
  %call2 = call ptr @_Znam(i64 10) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new[](unsigned long, std::align_val_t) converted to
;; operator new[](unsigned long, std::align_val_t, hot_cold_t) with a hot or
;; cold attribute.
; HOTCOLD-LABEL: @array_new_align()
define void @array_new_align() {
  ;; Attribute cold converted to hot_cold_t value 0 (coldest).
  ; HOTCOLD: @_ZnamSt11align_val_t10hot_cold_t(i64 10, i64 8, i8 0)
  %call = call ptr @_ZnamSt11align_val_t(i64 10, i64 8) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnamSt11align_val_t(i64 10, i64 8)
  %call1 = call ptr @_ZnamSt11align_val_t(i64 10, i64 8) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to hot_cold_t value 255 (-1) (hottest).
  ; HOTCOLD: @_ZnamSt11align_val_t10hot_cold_t(i64 10, i64 8, i8 -1)
  %call2 = call ptr @_ZnamSt11align_val_t(i64 10, i64 8) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new[](unsigned long, const std::nothrow_t&) converted to
;; operator new[](unsigned long, const std::nothrow_t&, hot_cold_t) with a hot or
;; cold attribute.
; HOTCOLD-LABEL: @array_new_nothrow()
define void @array_new_nothrow() {
  %nt = alloca i8
  ;; Attribute cold converted to hot_cold_t value 0 (coldest).
  ; HOTCOLD: @_ZnamRKSt9nothrow_t10hot_cold_t(i64 10, ptr nonnull %nt, i8 0)
  %call = call ptr @_ZnamRKSt9nothrow_t(i64 10, ptr %nt) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t(i64 10, ptr nonnull %nt)
  %call1 = call ptr @_ZnamRKSt9nothrow_t(i64 10, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to hot_cold_t value 255 (-1) (hottest).
  ; HOTCOLD: @_ZnamRKSt9nothrow_t10hot_cold_t(i64 10, ptr nonnull %nt, i8 -1)
  %call2 = call ptr @_ZnamRKSt9nothrow_t(i64 10, ptr %nt) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new[](unsigned long, std::align_val_t, const std::nothrow_t&)
;; converted to
;; operator new[](unsigned long, std::align_val_t, const std::nothrow_t&, hot_cold_t)
;; with a hot or cold attribute.
; HOTCOLD-LABEL: @array_new_align_nothrow()
define void @array_new_align_nothrow() {
  %nt = alloca i8
  ;; Attribute cold converted to hot_cold_t value 0 (coldest).
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t10hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 0)
  %call = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr nonnull %nt)
  %call1 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to hot_cold_t value 255 (-1) (hottest).
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t10hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 -1)
  %call2 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #2
  call void @dummy(ptr %call2)
  ret void
}

;; So that instcombine doesn't optimize out the call.
declare void @dummy(ptr)

declare ptr @_Znwm(i64)
declare ptr @_ZnwmSt11align_val_t(i64, i64)
declare ptr @_ZnwmRKSt9nothrow_t(i64, ptr)
declare ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64, i64, ptr)
declare ptr @_Znam(i64)
declare ptr @_ZnamSt11align_val_t(i64, i64)
declare ptr @_ZnamRKSt9nothrow_t(i64, ptr)
declare ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64, i64, ptr)

attributes #0 = { builtin allocsize(0) "memprof"="cold" }
attributes #1 = { builtin allocsize(0) "memprof"="notcold" }
attributes #2 = { builtin allocsize(0) "memprof"="hot" }
