;; Test behavior of -optimize-hot-cold-new and related options.

;; Check that we don't get hot/cold new calls without enabling it explicitly.
; RUN: opt < %s -passes=instcombine -S | FileCheck %s --implicit-check-not=hot_cold_t

;; First check with the default cold and hot hint values (255 = -2).
; RUN: opt < %s -passes=instcombine -optimize-hot-cold-new -S | FileCheck %s --check-prefix=HOTCOLD -DCOLD=1 -DHOT=-2

;; Next check with the non-default cold and hot hint values (200 =-56).
; RUN: opt < %s -passes=instcombine -optimize-hot-cold-new -cold-new-hint-value=5 -hot-new-hint-value=200 -S | FileCheck %s --check-prefix=HOTCOLD -DCOLD=5 -DHOT=-56

;; Make sure that values not in 0..255 are flagged with an error
; RUN: not opt < %s -passes=instcombine -optimize-hot-cold-new -cold-new-hint-value=256 -S 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not opt < %s -passes=instcombine -optimize-hot-cold-new -hot-new-hint-value=5000 -S 2>&1 | FileCheck %s --check-prefix=ERROR
; ERROR: value must be in the range [0, 255]!

;; Check that operator new(unsigned long) converted to
;; operator new(unsigned long, __hot_cold_t) with a hot or cold attribute.
; HOTCOLD-LABEL: @new()
define void @new() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_Znwm12__hot_cold_t(i64 10, i8 [[COLD]])
  %call = call ptr @_Znwm(i64 10) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_Znwm(i64 10)
  %call1 = call ptr @_Znwm(i64 10) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_Znwm12__hot_cold_t(i64 10, i8 [[HOT]])
  %call2 = call ptr @_Znwm(i64 10) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new(unsigned long, std::align_val_t) converted to
;; operator new(unsigned long, std::align_val_t, __hot_cold_t) with a hot or
;; cold attribute.
; HOTCOLD-LABEL: @new_align()
define void @new_align() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[COLD]])
  %call = call ptr @_ZnwmSt11align_val_t(i64 10, i64 8) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnwmSt11align_val_t(i64 10, i64 8)
  %call1 = call ptr @_ZnwmSt11align_val_t(i64 10, i64 8) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[HOT]])
  %call2 = call ptr @_ZnwmSt11align_val_t(i64 10, i64 8) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new(unsigned long, const std::nothrow_t&) converted to
;; operator new(unsigned long, const std::nothrow_t&, __hot_cold_t) with a hot or
;; cold attribute.
; HOTCOLD-LABEL: @new_nothrow()
define void @new_nothrow() {
  %nt = alloca i8
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[COLD]])
  %call = call ptr @_ZnwmRKSt9nothrow_t(i64 10, ptr %nt) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t(i64 10, ptr nonnull %nt)
  %call1 = call ptr @_ZnwmRKSt9nothrow_t(i64 10, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[HOT]])
  %call2 = call ptr @_ZnwmRKSt9nothrow_t(i64 10, ptr %nt) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new(unsigned long, std::align_val_t, const std::nothrow_t&)
;; converted to
;; operator new(unsigned long, std::align_val_t, const std::nothrow_t&, __hot_cold_t)
;; with a hot or cold attribute.
; HOTCOLD-LABEL: @new_align_nothrow()
define void @new_align_nothrow() {
  %nt = alloca i8
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[COLD]])
  %call = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr nonnull %nt)
  %call1 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[HOT]])
  %call2 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new[](unsigned long) converted to
;; operator new[](unsigned long, __hot_cold_t) with a hot or cold attribute.
; HOTCOLD-LABEL: @array_new()
define void @array_new() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_Znam12__hot_cold_t(i64 10, i8 [[COLD]])
  %call = call ptr @_Znam(i64 10) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_Znam(i64 10)
  %call1 = call ptr @_Znam(i64 10) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_Znam12__hot_cold_t(i64 10, i8 [[HOT]])
  %call2 = call ptr @_Znam(i64 10) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new[](unsigned long, std::align_val_t) converted to
;; operator new[](unsigned long, std::align_val_t, __hot_cold_t) with a hot or
;; cold attribute.
; HOTCOLD-LABEL: @array_new_align()
define void @array_new_align() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[COLD]])
  %call = call ptr @_ZnamSt11align_val_t(i64 10, i64 8) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnamSt11align_val_t(i64 10, i64 8)
  %call1 = call ptr @_ZnamSt11align_val_t(i64 10, i64 8) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[HOT]])
  %call2 = call ptr @_ZnamSt11align_val_t(i64 10, i64 8) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new[](unsigned long, const std::nothrow_t&) converted to
;; operator new[](unsigned long, const std::nothrow_t&, __hot_cold_t) with a hot or
;; cold attribute.
; HOTCOLD-LABEL: @array_new_nothrow()
define void @array_new_nothrow() {
  %nt = alloca i8
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[COLD]])
  %call = call ptr @_ZnamRKSt9nothrow_t(i64 10, ptr %nt) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t(i64 10, ptr nonnull %nt)
  %call1 = call ptr @_ZnamRKSt9nothrow_t(i64 10, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[HOT]])
  %call2 = call ptr @_ZnamRKSt9nothrow_t(i64 10, ptr %nt) #2
  call void @dummy(ptr %call2)
  ret void
}

;; Check that operator new[](unsigned long, std::align_val_t, const std::nothrow_t&)
;; converted to
;; operator new[](unsigned long, std::align_val_t, const std::nothrow_t&, __hot_cold_t)
;; with a hot or cold attribute.
; HOTCOLD-LABEL: @array_new_align_nothrow()
define void @array_new_align_nothrow() {
  %nt = alloca i8
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[COLD]])
  %call = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold has no effect.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr nonnull %nt)
  %call1 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[HOT]])
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
