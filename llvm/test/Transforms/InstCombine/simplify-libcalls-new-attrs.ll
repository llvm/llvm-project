;; Test attribute and call-flag handling when -optimize-hot-cold-new rewrites
;; operator new calls.

; RUN: opt < %s -passes=instcombine -optimize-hot-cold-new -cold-new-hint-value=5 -optimize-nobuiltin-hot-cold-new-new -optimize-existing-hot-cold-new -S | FileCheck %s --check-prefixes=CHECK,NOMIN
; RUN: opt < %s -passes=instcombine -optimize-hot-cold-new -cold-new-hint-value=5 -optimize-nobuiltin-hot-cold-new-new -optimize-existing-hot-cold-new -min-existing-hot-cold-new-hint -S | FileCheck %s --check-prefixes=CHECK,MIN

;; Call-site return, parameter, and function attributes plus the tail flag are
;; preserved, while "memprof" is removed.
; CHECK-LABEL: define ptr @preserve_callsite_attrs_and_tail(
; CHECK-NEXT:   %[[CALL:.*]] = tail call noalias noundef nonnull align 32 dereferenceable(64) ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64 noundef 64, i64 allocalign noundef 32, i8 5) #[[ATTR_PRESERVE:[0-9]+]]
; CHECK-NEXT:   ret ptr %[[CALL]]
define ptr @preserve_callsite_attrs_and_tail() {
  %call = tail call noalias noundef nonnull align 32 ptr @_ZnwmSt11align_val_t(i64 noundef 64, i64 allocalign noundef 32) #0
  ret ptr %call
}

;; nobuiltin is preserved when -optimize-nobuiltin-hot-cold-new-new rewrites a
;; nobuiltin call.
; CHECK-LABEL: define ptr @preserve_nobuiltin(
; CHECK-NEXT:   %[[CALL:.*]] = call noalias dereferenceable_or_null(16) ptr @_Znwm12__hot_cold_t(i64 16, i8 5) #[[ATTR_NOBUILTIN:[0-9]+]]
; CHECK-NEXT:   ret ptr %[[CALL]]
define ptr @preserve_nobuiltin() {
  %call = call noalias ptr @_Znwm(i64 16) #1
  ret ptr %call
}

;; Call-site speculatable is stripped because the synthesized __hot_cold_t
;; declaration is not speculatable.
; CHECK-LABEL: define ptr @strip_speculatable(
; CHECK-NEXT:   %[[CALL:.*]] = call ptr @_Znam12__hot_cold_t(i64 16, i8 5) #[[ATTR_BUILTIN:[0-9]+]]
; CHECK-NEXT:   ret ptr %[[CALL]]
define ptr @strip_speculatable() {
  %call = call ptr @_Znam(i64 16) #2
  ret ptr %call
}

;; "alloc-variant-zeroed" is stripped when adding the __hot_cold_t parameter
;; changes the function signature.
; CHECK-LABEL: define ptr @strip_alloc_variant_zeroed(
; CHECK-NEXT:   %[[CALL:.*]] = call dereferenceable_or_null(16) ptr @_Znwm12__hot_cold_t(i64 16, i8 5) #[[ATTR_NO_ZEROED:[0-9]+]]
; CHECK-NEXT:   ret ptr %[[CALL]]
define ptr @strip_alloc_variant_zeroed() {
  %call = call ptr @_Znwm(i64 16) #3
  ret ptr %call
}

;; "alloc-variant-zeroed" is preserved when updating an existing __hot_cold_t
;; call because the function signature is unchanged.
; CHECK-LABEL: define ptr @preserve_alloc_variant_zeroed_existing(
; CHECK-NEXT:   %[[CALL:.*]] = call dereferenceable_or_null(16) ptr @_Znwm12__hot_cold_t(i64 16, i8 5) #[[ATTR_ZEROED:[0-9]+]]
; CHECK-NEXT:   ret ptr %[[CALL]]
define ptr @preserve_alloc_variant_zeroed_existing() {
  %call = call ptr @_Znwm12__hot_cold_t(i64 16, i8 7) #4
  ret ptr %call
}

;; When updating an existing __hot_cold_t call, range is stripped from the
;; replaced hint parameter, while noundef on the hint parameter and range on
;; unchanged parameters are preserved.
; CHECK-LABEL: define ptr @strip_range_on_replaced_hint(
; NOMIN-NEXT:   %[[CALL:.*]] = call ptr @_Znwm12__hot_cold_t(i64 range(i64 1, 100) %size, i8 noundef 5) #[[ATTR_ALLOCSIZE:[0-9]+]]
; NOMIN-NEXT:   ret ptr %[[CALL]]
; MIN-NEXT:     %[[UMIN:.*]] = call i8 @llvm.umin.i8(i8 %hint, i8 5)
; MIN-NEXT:     %[[CALL:.*]] = call ptr @_Znwm12__hot_cold_t(i64 range(i64 1, 100) %size, i8 noundef %[[UMIN]]) #[[ATTR_ALLOCSIZE:[0-9]+]]
; MIN-NEXT:     ret ptr %[[CALL]]
define ptr @strip_range_on_replaced_hint(i64 %size, i8 %hint) {
  %call = call ptr @_Znwm12__hot_cold_t(i64 range(i64 1, 100) %size, i8 noundef range(i8 10, 20) %hint) #5
  ret ptr %call
}

declare ptr @_Znwm(i64)
declare ptr @_ZnwmSt11align_val_t(i64, i64)
declare ptr @_Znam(i64) speculatable
declare ptr @_Znwm12__hot_cold_t(i64, i8)
declare ptr @zeroed_new(i64) allockind("alloc,zeroed") "alloc-family"="new"
declare ptr @zeroed_new_hc(i64, i8) allockind("alloc,zeroed") "alloc-family"="new"

attributes #0 = { builtin allockind("alloc,uninitialized,aligned") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "memprof"="cold" }
attributes #1 = { nobuiltin allocsize(0) "memprof"="cold" }
attributes #2 = { builtin speculatable "memprof"="cold" }
attributes #3 = { builtin allockind("alloc,uninitialized") allocsize(0) "alloc-family"="new" "alloc-variant-zeroed"="zeroed_new" "memprof"="cold" }
attributes #4 = { builtin allockind("alloc,uninitialized") allocsize(0) "alloc-family"="new" "alloc-variant-zeroed"="zeroed_new_hc" "memprof"="cold" }
attributes #5 = { builtin allocsize(0) "memprof"="cold" }

; CHECK-DAG: attributes #[[ATTR_PRESERVE]] = { builtin allockind("alloc,uninitialized,aligned") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
; CHECK-DAG: attributes #[[ATTR_NOBUILTIN]] = { nobuiltin allocsize(0) }
; CHECK-DAG: attributes #[[ATTR_BUILTIN]] = { builtin }
; CHECK-DAG: attributes #[[ATTR_NO_ZEROED]] = { builtin allockind("alloc,uninitialized") allocsize(0) "alloc-family"="new" }
; CHECK-DAG: attributes #[[ATTR_ZEROED]] = { builtin allockind("alloc,uninitialized") allocsize(0) "alloc-family"="new" "alloc-variant-zeroed"="zeroed_new_hc" }
; CHECK-DAG: attributes #[[ATTR_ALLOCSIZE]] = { builtin allocsize(0) }
