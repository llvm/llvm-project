;; Test behavior of -optimize-hot-cold-new and related options.

;; Check that we don't get hot/cold new calls without enabling it explicitly.
; RUN: opt < %s -passes=instcombine -S | FileCheck %s --check-prefix=OFF
; OFF-NOT: hot_cold_t
; OFF-LABEL: @new_hot_cold()

;; First check with the default hint values (254 = -2, 128 = -128, 222 = -34).
; RUN: opt < %s -passes=instcombine -optimize-hot-cold-new -S | FileCheck %s --check-prefix=HOTCOLD -DCOLD=1 -DHOT=-2 -DNOTCOLD=-128 -DAMBIG=-34 -DPREVHINTCOLD=7 -DPREVHINTNOTCOLD=7 -DPREVHINTHOT=7 -DPREVHINTAMBIG=7

;; Next check with the non-default cold and hot hint values (200 =-56).
; RUN: opt < %s -passes=instcombine -optimize-hot-cold-new -cold-new-hint-value=5 -hot-new-hint-value=200 -notcold-new-hint-value=99 -ambiguous-new-hint-value=44 -S | FileCheck %s --check-prefix=HOTCOLD -DCOLD=5 -DHOT=-56 -DAMBIG=44 -DNOTCOLD=99 -DPREVHINTCOLD=7 -DPREVHINTNOTCOLD=7 -DPREVHINTHOT=7 -DPREVHINTAMBIG=7

;; Try again with the non-default cold and hot hint values (200 =-56), and this
;; time specify that existing hints should be updated.
; RUN: opt < %s -passes=instcombine -optimize-hot-cold-new -cold-new-hint-value=5 -notcold-new-hint-value=100 -hot-new-hint-value=200 -ambiguous-new-hint-value=44 -optimize-existing-hot-cold-new -S | FileCheck %s --check-prefix=HOTCOLD -DCOLD=5 -DHOT=-56 -DNOTCOLD=100 -DAMBIG=44 -DPREVHINTCOLD=5 -DPREVHINTNOTCOLD=100 -DPREVHINTHOT=-56 -DPREVHINTAMBIG=44

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
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_Znwm12__hot_cold_t(i64 10, i8 [[NOTCOLD]])
  %call1 = call ptr @_Znwm(i64 10) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_Znwm12__hot_cold_t(i64 10, i8 [[HOT]])
  %call2 = call ptr @_Znwm(i64 10) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_Znwm12__hot_cold_t(i64 10, i8 [[AMBIG]])
  %call4 = call ptr @_Znwm(i64 10) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin call has no effect.
  ; HOTCOLD: @_Znwm(i64 10)
  %call3 = call ptr @_Znwm(i64 10) #6
  call void @dummy(ptr %call3)
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
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[NOTCOLD]])
  %call1 = call ptr @_ZnwmSt11align_val_t(i64 10, i64 8) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[HOT]])
  %call2 = call ptr @_ZnwmSt11align_val_t(i64 10, i64 8) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[AMBIG]])
  %call4 = call ptr @_ZnwmSt11align_val_t(i64 10, i64 8) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin call has no effect.
  ; HOTCOLD: @_ZnwmSt11align_val_t(i64 10, i64 8)
  %call3 = call ptr @_ZnwmSt11align_val_t(i64 10, i64 8) #6
  call void @dummy(ptr %call3)
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
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[NOTCOLD]])
  %call1 = call ptr @_ZnwmRKSt9nothrow_t(i64 10, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[HOT]])
  %call2 = call ptr @_ZnwmRKSt9nothrow_t(i64 10, ptr %nt) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[AMBIG]])
  %call4 = call ptr @_ZnwmRKSt9nothrow_t(i64 10, ptr %nt) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin call has no effect.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t(i64 10, ptr nonnull %nt)
  %call3 = call ptr @_ZnwmRKSt9nothrow_t(i64 10, ptr %nt) #6
  call void @dummy(ptr %call3)
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
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[NOTCOLD]])
  %call1 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[HOT]])
  %call2 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[AMBIG]])
  %call4 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin call has no effect.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr nonnull %nt)
  %call3 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #6
  call void @dummy(ptr %call3)
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
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_Znam12__hot_cold_t(i64 10, i8 [[NOTCOLD]])
  %call1 = call ptr @_Znam(i64 10) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_Znam12__hot_cold_t(i64 10, i8 [[HOT]])
  %call2 = call ptr @_Znam(i64 10) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_Znam12__hot_cold_t(i64 10, i8 [[AMBIG]])
  %call4 = call ptr @_Znam(i64 10) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin call has no effect.
  ; HOTCOLD: @_Znam(i64 10)
  %call3 = call ptr @_Znam(i64 10) #6
  call void @dummy(ptr %call3)
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
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[NOTCOLD]])
  %call1 = call ptr @_ZnamSt11align_val_t(i64 10, i64 8) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[HOT]])
  %call2 = call ptr @_ZnamSt11align_val_t(i64 10, i64 8) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[AMBIG]])
  %call4 = call ptr @_ZnamSt11align_val_t(i64 10, i64 8) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin call has no effect.
  ; HOTCOLD: @_ZnamSt11align_val_t(i64 10, i64 8)
  %call3 = call ptr @_ZnamSt11align_val_t(i64 10, i64 8) #6
  call void @dummy(ptr %call3)
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
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[NOTCOLD]])
  %call1 = call ptr @_ZnamRKSt9nothrow_t(i64 10, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[HOT]])
  %call2 = call ptr @_ZnamRKSt9nothrow_t(i64 10, ptr %nt) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[AMBIG]])
  %call4 = call ptr @_ZnamRKSt9nothrow_t(i64 10, ptr %nt) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin call has no effect.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t(i64 10, ptr nonnull %nt)
  %call3 = call ptr @_ZnamRKSt9nothrow_t(i64 10, ptr %nt) #6
  call void @dummy(ptr %call3)
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
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[NOTCOLD]])
  %call1 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[HOT]])
  %call2 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[AMBIG]])
  %call4 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin call has no effect.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr nonnull %nt)
  %call3 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64 10, i64 8, ptr %nt) #6
  call void @dummy(ptr %call3)
  ret void
}

;; Check that operator new(unsigned long, __hot_cold_t)
;; optionally has its hint updated.
; HOTCOLD-LABEL: @new_hot_cold()
define void @new_hot_cold() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_Znwm12__hot_cold_t(i64 10, i8 [[PREVHINTCOLD]])
  %call = call ptr @_Znwm12__hot_cold_t(i64 10, i8 7) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_Znwm12__hot_cold_t(i64 10, i8 [[PREVHINTNOTCOLD]])
  %call1 = call ptr @_Znwm12__hot_cold_t(i64 10, i8 7) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_Znwm12__hot_cold_t(i64 10, i8 [[PREVHINTHOT]])
  %call2 = call ptr @_Znwm12__hot_cold_t(i64 10, i8 7) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_Znwm12__hot_cold_t(i64 10, i8 [[PREVHINTAMBIG]])
  %call4 = call ptr @_Znwm12__hot_cold_t(i64 10, i8 7) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin existing hot/cold call updates the hint.
  ; HOTCOLD: @_Znwm12__hot_cold_t(i64 10, i8 [[PREVHINTCOLD]])
  %call3 = call ptr @_Znwm12__hot_cold_t(i64 10, i8 7) #6
  call void @dummy(ptr %call3)
  ret void
}

;; Check that operator new(unsigned long, std::align_val_t, __hot_cold_t)
;; optionally has its hint updated.
; HOTCOLD-LABEL: @new_align_hot_cold()
define void @new_align_hot_cold() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[PREVHINTCOLD]])
  %call = call ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 7) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[PREVHINTNOTCOLD]])
  %call1 = call ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 7) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[PREVHINTHOT]])
  %call2 = call ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 7) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[PREVHINTAMBIG]])
  %call4 = call ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 7) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin existing hot/cold call updates the hint.
  ; HOTCOLD: @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[PREVHINTCOLD]])
  %call3 = call ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 7) #6
  call void @dummy(ptr %call3)
  ret void
}

;; Check that operator new(unsigned long, const std::nothrow_t&, __hot_cold_t)
;; optionally has its hint updated.
; HOTCOLD-LABEL: @new_nothrow_hot_cold()
define void @new_nothrow_hot_cold() {
  %nt = alloca i8
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[PREVHINTCOLD]])
  %call = call ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr %nt, i8 7) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[PREVHINTNOTCOLD]])
  %call1 = call ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr %nt, i8 7) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[PREVHINTHOT]])
  %call2 = call ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr %nt, i8 7) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[PREVHINTAMBIG]])
  %call4 = call ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr %nt, i8 7) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin existing hot/cold call updates the hint.
  ; HOTCOLD: @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[PREVHINTCOLD]])
  %call3 = call ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 10, ptr %nt, i8 7) #6
  call void @dummy(ptr %call3)
  ret void
}

;; Check that operator new(unsigned long, std::align_val_t, const std::nothrow_t&, __hot_cold_t)
;; optionally has its hint updated.
; HOTCOLD-LABEL: @new_align_nothrow_hot_cold()
define void @new_align_nothrow_hot_cold() {
  %nt = alloca i8
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[PREVHINTCOLD]])
  %call = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr %nt, i8 7) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[PREVHINTNOTCOLD]])
  %call1 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr %nt, i8 7) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[PREVHINTHOT]])
  %call2 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr %nt, i8 7) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[PREVHINTAMBIG]])
  %call4 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr %nt, i8 7) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin existing hot/cold call updates the hint.
  ; HOTCOLD: @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[PREVHINTCOLD]])
  %call3 = call ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr %nt, i8 7) #6
  call void @dummy(ptr %call3)
  ret void
}

;; Check that operator new[](unsigned long, __hot_cold_t)
;; optionally has its hint updated.
; HOTCOLD-LABEL: @array_new_hot_cold()
define void @array_new_hot_cold() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_Znam12__hot_cold_t(i64 10, i8 [[PREVHINTCOLD]])
  %call = call ptr @_Znam12__hot_cold_t(i64 10, i8 7) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_Znam12__hot_cold_t(i64 10, i8 [[PREVHINTNOTCOLD]])
  %call1 = call ptr @_Znam12__hot_cold_t(i64 10, i8 7) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_Znam12__hot_cold_t(i64 10, i8 [[PREVHINTHOT]])
  %call2 = call ptr @_Znam12__hot_cold_t(i64 10, i8 7) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_Znam12__hot_cold_t(i64 10, i8 [[PREVHINTAMBIG]])
  %call4 = call ptr @_Znam12__hot_cold_t(i64 10, i8 7) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin existing hot/cold call updates the hint.
  ; HOTCOLD: @_Znam12__hot_cold_t(i64 10, i8 [[PREVHINTCOLD]])
  %call3 = call ptr @_Znam12__hot_cold_t(i64 10, i8 7) #6
  call void @dummy(ptr %call3)
  ret void
}

;; Check that operator new[](unsigned long, std::align_val_t, __hot_cold_t)
;; optionally has its hint updated.
; HOTCOLD-LABEL: @array_new_align_hot_cold()
define void @array_new_align_hot_cold() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[PREVHINTCOLD]])
  %call = call ptr @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 7) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[PREVHINTNOTCOLD]])
  %call1 = call ptr @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 7) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[PREVHINTHOT]])
  %call2 = call ptr @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 7) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[PREVHINTAMBIG]])
  %call4 = call ptr @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 7) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin existing hot/cold call updates the hint.
  ; HOTCOLD: @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 [[PREVHINTCOLD]])
  %call3 = call ptr @_ZnamSt11align_val_t12__hot_cold_t(i64 10, i64 8, i8 7) #6
  call void @dummy(ptr %call3)
  ret void
}

;; Check that operator new[](unsigned long, const std::nothrow_t&, __hot_cold_t)
;; optionally has its hint updated.
; HOTCOLD-LABEL: @array_new_nothrow_hot_cold()
define void @array_new_nothrow_hot_cold() {
  %nt = alloca i8
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[PREVHINTCOLD]])
  %call = call ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr %nt, i8 7) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[PREVHINTNOTCOLD]])
  %call1 = call ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr %nt, i8 7) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[PREVHINTHOT]])
  %call2 = call ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr %nt, i8 7) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[PREVHINTAMBIG]])
  %call4 = call ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr %nt, i8 7) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin existing hot/cold call updates the hint.
  ; HOTCOLD: @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr nonnull %nt, i8 [[PREVHINTCOLD]])
  %call3 = call ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 10, ptr %nt, i8 7) #6
  call void @dummy(ptr %call3)
  ret void
}

;; Check that operator new[](unsigned long, std::align_val_t, const std::nothrow_t&, __hot_cold_t)
;; optionally has its hint updated.
; HOTCOLD-LABEL: @array_new_align_nothrow_hot_cold()
define void @array_new_align_nothrow_hot_cold() {
  %nt = alloca i8
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[PREVHINTCOLD]])
  %call = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr %nt, i8 7) #0
  call void @dummy(ptr %call)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[PREVHINTNOTCOLD]])
  %call1 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr %nt, i8 7) #1
  call void @dummy(ptr %call1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[PREVHINTHOT]])
  %call2 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr %nt, i8 7) #2
  call void @dummy(ptr %call2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[PREVHINTAMBIG]])
  %call4 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr %nt, i8 7) #7
  call void @dummy(ptr %call4)
  ;; Attribute cold on a nobuiltin existing hot/cold call updates the hint.
  ; HOTCOLD: @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr nonnull %nt, i8 [[PREVHINTCOLD]])
  %call3 = call ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 10, i64 8, ptr %nt, i8 7) #6
  call void @dummy(ptr %call3)
  ret void
}

;; Check that operator __size_returning_new(unsigned long) converted to
;; __size_returning_new(unsigned long, __hot_cold_t) with a hot or cold attribute.
; HOTCOLD-LABEL: @size_returning_test()
define void @size_returning_test() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @__size_returning_new_hot_cold(i64 10, i8 [[COLD]])
  %call = call {ptr, i64} @__size_returning_new(i64 10) #3
  %p  = extractvalue {ptr, i64} %call, 0
  call void @dummy(ptr %p)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @__size_returning_new_hot_cold(i64 10, i8 [[NOTCOLD]])
  %call1 = call {ptr, i64} @__size_returning_new(i64 10) #4
  %p1  = extractvalue {ptr, i64} %call1, 0
  call void @dummy(ptr %p1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @__size_returning_new_hot_cold(i64 10, i8 [[HOT]])
  %call2 = call {ptr, i64} @__size_returning_new(i64 10) #5
  %p2  = extractvalue {ptr, i64} %call2, 0
  call void @dummy(ptr %p2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @__size_returning_new_hot_cold(i64 10, i8 [[AMBIG]])
  %call4 = call {ptr, i64} @__size_returning_new(i64 10) #8
  %p4  = extractvalue {ptr, i64} %call4, 0
  call void @dummy(ptr %p4)
  ;; Attribute cold on a nobuiltin call has no effect.
  ; HOTCOLD: @__size_returning_new(i64 10)
  %call3 = call {ptr, i64} @__size_returning_new(i64 10) #6
  %p3 = extractvalue {ptr, i64} %call3, 0
  call void @dummy(ptr %p3)
  ret void
}

;; Check that operator __size_returning_new_aligned(unsigned long, std::align_val_t) converted to
;; __size_returning_new_aligned(unsigned long, std::align_val_t, __hot_cold_t) with a hot or cold attribute.
; HOTCOLD-LABEL: @size_returning_aligned_test()
define void @size_returning_aligned_test() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 [[COLD]])
  %call = call {ptr, i64} @__size_returning_new_aligned(i64 10, i64 8) #3
  %p  = extractvalue {ptr, i64} %call, 0
  call void @dummy(ptr %p)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 [[NOTCOLD]])
  %call1 = call {ptr, i64} @__size_returning_new_aligned(i64 10, i64 8) #4
  %p1  = extractvalue {ptr, i64} %call1, 0
  call void @dummy(ptr %p1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 [[HOT]])
  %call2 = call {ptr, i64} @__size_returning_new_aligned(i64 10, i64 8) #5
  %p2  = extractvalue {ptr, i64} %call2, 0
  call void @dummy(ptr %p2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 [[AMBIG]])
  %call4 = call {ptr, i64} @__size_returning_new_aligned(i64 10, i64 8) #8
  %p4  = extractvalue {ptr, i64} %call4, 0
  call void @dummy(ptr %p4)
  ;; Attribute cold on a nobuiltin call has no effect.
  ; HOTCOLD: @__size_returning_new_aligned(i64 10, i64 8)
  %call3 = call {ptr, i64} @__size_returning_new_aligned(i64 10, i64 8) #6
  %p3 = extractvalue {ptr, i64} %call3, 0
  call void @dummy(ptr %p3)
  ret void
}

;; Check that __size_returning_new_hot_cold(unsigned long, __hot_cold_t)
;; optionally has its hint updated.
; HOTCOLD-LABEL: @size_returning_update_test()
define void @size_returning_update_test() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @__size_returning_new_hot_cold(i64 10, i8 [[PREVHINTCOLD]])
  %call = call {ptr, i64} @__size_returning_new_hot_cold(i64 10, i8 7) #3
  %p  = extractvalue {ptr, i64} %call, 0
  call void @dummy(ptr %p)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @__size_returning_new_hot_cold(i64 10, i8 [[PREVHINTNOTCOLD]])
  %call1 = call {ptr, i64} @__size_returning_new_hot_cold(i64 10, i8 7) #4
  %p1 = extractvalue {ptr, i64} %call1, 0
  call void @dummy(ptr %p1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @__size_returning_new_hot_cold(i64 10, i8 [[PREVHINTHOT]])
  %call2 = call {ptr, i64} @__size_returning_new_hot_cold(i64 10, i8 7) #5
  %p2 = extractvalue {ptr, i64} %call2, 0
  call void @dummy(ptr %p2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @__size_returning_new_hot_cold(i64 10, i8 [[PREVHINTAMBIG]])
  %call4 = call {ptr, i64} @__size_returning_new_hot_cold(i64 10, i8 7) #8
  %p4 = extractvalue {ptr, i64} %call4, 0
  call void @dummy(ptr %p4)
  ;; Attribute cold on a nobuiltin existing hot/cold call updates the hint.
  ; HOTCOLD: @__size_returning_new_hot_cold(i64 10, i8 [[PREVHINTCOLD]])
  %call3 = call {ptr, i64} @__size_returning_new_hot_cold(i64 10, i8 7) #6
  %p3 = extractvalue {ptr, i64} %call3, 0
  call void @dummy(ptr %p3)
  ret void
}

;; Check that __size_returning_new_aligned_hot_cold(unsigned long, __hot_cold_t)
;; optionally has its hint updated.
; HOTCOLD-LABEL: @size_returning_aligned_update_test()
define void @size_returning_aligned_update_test() {
  ;; Attribute cold converted to __hot_cold_t cold value.
  ; HOTCOLD: @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 [[PREVHINTCOLD]])
  %call = call {ptr, i64} @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 7) #3
  %p  = extractvalue {ptr, i64} %call, 0
  call void @dummy(ptr %p)
  ;; Attribute notcold converted to __hot_cold_t notcold value.
  ; HOTCOLD: @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 [[PREVHINTNOTCOLD]])
  %call1 = call {ptr, i64} @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 7) #4
  %p1 = extractvalue {ptr, i64} %call1, 0
  call void @dummy(ptr %p1)
  ;; Attribute hot converted to __hot_cold_t hot value.
  ; HOTCOLD: @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 [[PREVHINTHOT]])
  %call2 = call {ptr, i64} @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 7) #5
  %p2 = extractvalue {ptr, i64} %call2, 0
  call void @dummy(ptr %p2)
  ;; Attribute ambiguous converted to __hot_cold_t ambiguous value.
  ; HOTCOLD: @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 [[PREVHINTAMBIG]])
  %call4 = call {ptr, i64} @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 7) #8
  %p4 = extractvalue {ptr, i64} %call4, 0
  call void @dummy(ptr %p4)
  ;; Attribute cold on a nobuiltin existing hot/cold call updates the hint.
  ; HOTCOLD: @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 [[PREVHINTCOLD]])
  %call3 = call {ptr, i64} @__size_returning_new_aligned_hot_cold(i64 10, i64 8, i8 7) #6
  %p3 = extractvalue {ptr, i64} %call3, 0
  call void @dummy(ptr %p3)
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
declare ptr @_Znwm12__hot_cold_t(i64, i8)
declare ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64, i64, i8)
declare ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64, ptr, i8)
declare ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64, i64, ptr, i8)
declare ptr @_Znam12__hot_cold_t(i64, i8)
declare ptr @_ZnamSt11align_val_t12__hot_cold_t(i64, i64, i8)
declare ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64, ptr, i8)
declare ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64, i64, ptr, i8)


declare {ptr, i64} @__size_returning_new(i64)
declare {ptr, i64} @__size_returning_new_hot_cold(i64, i8)
declare {ptr, i64} @__size_returning_new_aligned(i64, i64)
declare {ptr, i64} @__size_returning_new_aligned_hot_cold(i64, i64, i8)

attributes #0 = { builtin allocsize(0) "memprof"="cold" }
attributes #1 = { builtin allocsize(0) "memprof"="notcold" }
attributes #2 = { builtin allocsize(0) "memprof"="hot" }
attributes #7 = { builtin allocsize(0) "memprof"="ambiguous" }

;; Use separate attributes for __size_returning_new variants since they are not
;; treated as builtins.
attributes #3 = { "memprof" = "cold" }
attributes #4 = { "memprof" = "notcold" }
attributes #5 = { "memprof" = "hot" }
attributes #8 = { "memprof" = "ambiguous" }

attributes #6 = { nobuiltin allocsize(0) "memprof"="cold" }
