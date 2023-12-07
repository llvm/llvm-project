; RUN: opt < %s -passes=inferattrs -S | FileCheck %s



; Determine dereference-ability before unused loads get deleted:
; https://bugs.llvm.org/show_bug.cgi?id=21780

define <4 x double> @PR21780(ptr %ptr) {
; CHECK-LABEL: @PR21780(ptr %ptr)

  ; GEP of index 0 is simplified away.
  %arrayidx1 = getelementptr inbounds double, ptr %ptr, i64 1
  %arrayidx2 = getelementptr inbounds double, ptr %ptr, i64 2
  %arrayidx3 = getelementptr inbounds double, ptr %ptr, i64 3

  %t0 = load double, ptr %ptr, align 8
  %t1 = load double, ptr %arrayidx1, align 8
  %t2 = load double, ptr %arrayidx2, align 8
  %t3 = load double, ptr %arrayidx3, align 8

  %vecinit0 = insertelement <4 x double> poison, double %t0, i32 0
  %vecinit1 = insertelement <4 x double> %vecinit0, double %t1, i32 1
  %vecinit2 = insertelement <4 x double> %vecinit1, double %t2, i32 2
  %vecinit3 = insertelement <4 x double> %vecinit2, double %t3, i32 3
  %shuffle = shufflevector <4 x double> %vecinit3, <4 x double> %vecinit3, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x double> %shuffle
}


define double @PR21780_only_access3_with_inbounds(ptr %ptr) {
; CHECK-LABEL: @PR21780_only_access3_with_inbounds(ptr %ptr)

  %arrayidx3 = getelementptr inbounds double, ptr %ptr, i64 3
  %t3 = load double, ptr %arrayidx3, align 8
  ret double %t3
}

define double @PR21780_only_access3_without_inbounds(ptr %ptr) {
; CHECK-LABEL: @PR21780_only_access3_without_inbounds(ptr %ptr)
  %arrayidx3 = getelementptr double, ptr %ptr, i64 3
  %t3 = load double, ptr %arrayidx3, align 8
  ret double %t3
}

define double @PR21780_without_inbounds(ptr %ptr) {
; CHECK-LABEL: @PR21780_without_inbounds(ptr %ptr)

  %arrayidx1 = getelementptr double, ptr %ptr, i64 1
  %arrayidx2 = getelementptr double, ptr %ptr, i64 2
  %arrayidx3 = getelementptr double, ptr %ptr, i64 3

  %t0 = load double, ptr %ptr, align 8
  %t1 = load double, ptr %arrayidx1, align 8
  %t2 = load double, ptr %arrayidx2, align 8
  %t3 = load double, ptr %arrayidx3, align 8

  ret double %t3
}

; Unsimplified, but still valid. Also, throw in some bogus arguments.

define void @gep0(ptr %unused, ptr %other, ptr %ptr) {
; CHECK-LABEL: @gep0(ptr %unused, ptr %other, ptr %ptr)
  %arrayidx1 = getelementptr i8, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i8, ptr %ptr, i64 2
  %t0 = load i8, ptr %ptr
  %t1 = load i8, ptr %arrayidx1
  %t2 = load i8, ptr %arrayidx2
  store i8 %t2, ptr %other
  ret void
}

; Order of accesses does not change computation.
; Multiple arguments may be dereferenceable.

define void @ordering(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: @ordering(ptr %ptr1, ptr %ptr2)
  %a12 = getelementptr i8, ptr %ptr1, i64 2
  %t12 = load i8, ptr %a12
  %a11 = getelementptr i8, ptr %ptr1, i64 1
  %t20 = load i32, ptr %ptr2
  %t10 = load i8, ptr %ptr1
  %t11 = load i8, ptr %a11
  %a21 = getelementptr i32, ptr %ptr2, i64 1
  %t21 = load i32, ptr %a21
  ret void
}

; Not in entry block.

define void @not_entry_but_guaranteed_to_execute(ptr %ptr) {
; CHECK-LABEL: @not_entry_but_guaranteed_to_execute(ptr %ptr)
entry:
  br label %exit
exit:
  %arrayidx1 = getelementptr i8, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i8, ptr %ptr, i64 2
  %t0 = load i8, ptr %ptr
  %t1 = load i8, ptr %arrayidx1
  %t2 = load i8, ptr %arrayidx2
  ret void
}

; Not in entry block and not guaranteed to execute.

define void @not_entry_not_guaranteed_to_execute(ptr %ptr, i1 %cond) {
; CHECK-LABEL: @not_entry_not_guaranteed_to_execute(ptr %ptr, i1 %cond)
entry:
  br i1 %cond, label %loads, label %exit
loads:
  %arrayidx1 = getelementptr i8, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i8, ptr %ptr, i64 2
  %t0 = load i8, ptr %ptr
  %t1 = load i8, ptr %arrayidx1
  %t2 = load i8, ptr %arrayidx2
  ret void
exit:
  ret void
}

; The last load may not execute, so derefenceable bytes only covers the 1st two loads.

define void @partial_in_entry(ptr %ptr, i1 %cond) {
; CHECK-LABEL: @partial_in_entry(ptr %ptr, i1 %cond)
entry:
  %arrayidx1 = getelementptr i16, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i16, ptr %ptr, i64 2
  %t0 = load i16, ptr %ptr
  %t1 = load i16, ptr %arrayidx1
  br i1 %cond, label %loads, label %exit
loads:
  %t2 = load i16, ptr %arrayidx2
  ret void
exit:
  ret void
}

; The volatile load can't be used to prove a non-volatile access is allowed.
; The 2nd and 3rd loads may never execute.

define void @volatile_is_not_dereferenceable(ptr %ptr) {
; CHECK-LABEL: @volatile_is_not_dereferenceable(ptr %ptr)
  %arrayidx1 = getelementptr i16, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i16, ptr %ptr, i64 2
  %t0 = load volatile i16, ptr %ptr
  %t1 = load i16, ptr %arrayidx1
  %t2 = load i16, ptr %arrayidx2
  ret void
}

; TODO: We should allow inference for atomic (but not volatile) ops.

define void @atomic_is_alright(ptr %ptr) {
; CHECK-LABEL: @atomic_is_alright(ptr %ptr)
  %arrayidx1 = getelementptr i16, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i16, ptr %ptr, i64 2
  %t0 = load atomic i16, ptr %ptr unordered, align 2
  %t1 = load i16, ptr %arrayidx1
  %t2 = load i16, ptr %arrayidx2
  ret void
}

declare void @may_not_return()

define void @not_guaranteed_to_transfer_execution(ptr %ptr) {
; CHECK-LABEL: @not_guaranteed_to_transfer_execution(ptr %ptr)
  %arrayidx1 = getelementptr i16, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i16, ptr %ptr, i64 2
  %t0 = load i16, ptr %ptr
  call void @may_not_return()
  %t1 = load i16, ptr %arrayidx1
  %t2 = load i16, ptr %arrayidx2
  ret void
}

; We must have consecutive accesses.

define void @variable_gep_index(ptr %unused, ptr %ptr, i64 %variable_index) {
; CHECK-LABEL: @variable_gep_index(ptr %unused, ptr %ptr, i64 %variable_index)
  %arrayidx1 = getelementptr i8, ptr %ptr, i64 %variable_index
  %arrayidx2 = getelementptr i8, ptr %ptr, i64 2
  %t0 = load i8, ptr %ptr
  %t1 = load i8, ptr %arrayidx1
  %t2 = load i8, ptr %arrayidx2
  ret void
}

; Deal with >1 GEP index.

define void @multi_index_gep(ptr %ptr) {
; CHECK-LABEL: @multi_index_gep(ptr %ptr)
; FIXME: %ptr should be dereferenceable(4)
  %t0 = load i8, ptr %ptr
  ret void
}

; Could round weird bitwidths down?

define void @not_byte_multiple(ptr %ptr) {
; CHECK-LABEL: @not_byte_multiple(ptr %ptr)
  %t0 = load i9, ptr %ptr
  ret void
}

; Missing direct access from the pointer.

define void @no_pointer_deref(ptr %ptr) {
; CHECK-LABEL: @no_pointer_deref(ptr %ptr)
  %arrayidx1 = getelementptr i16, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i16, ptr %ptr, i64 2
  %t1 = load i16, ptr %arrayidx1
  %t2 = load i16, ptr %arrayidx2
  ret void
}

; Out-of-order is ok, but missing access concludes dereferenceable range.

define void @non_consecutive(ptr %ptr) {
; CHECK-LABEL: @non_consecutive(ptr %ptr)
  %arrayidx1 = getelementptr i32, ptr %ptr, i64 1
  %arrayidx3 = getelementptr i32, ptr %ptr, i64 3
  %t1 = load i32, ptr %arrayidx1
  %t0 = load i32, ptr %ptr
  %t3 = load i32, ptr %arrayidx3
  ret void
}

; Improve on existing dereferenceable attribute.

define void @more_bytes(ptr dereferenceable(8) %ptr) {
; CHECK-LABEL: @more_bytes(ptr dereferenceable(8) %ptr)
  %arrayidx3 = getelementptr i32, ptr %ptr, i64 3
  %arrayidx1 = getelementptr i32, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i32, ptr %ptr, i64 2
  %t3 = load i32, ptr %arrayidx3
  %t1 = load i32, ptr %arrayidx1
  %t2 = load i32, ptr %arrayidx2
  %t0 = load i32, ptr %ptr
  ret void
}

; Improve on existing dereferenceable_or_null attribute.

define void @more_bytes_and_not_null(ptr dereferenceable_or_null(8) %ptr) {
; CHECK-LABEL: @more_bytes_and_not_null(ptr dereferenceable_or_null(8) %ptr)
  %arrayidx3 = getelementptr i32, ptr %ptr, i64 3
  %arrayidx1 = getelementptr i32, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i32, ptr %ptr, i64 2
  %t3 = load i32, ptr %arrayidx3
  %t1 = load i32, ptr %arrayidx1
  %t2 = load i32, ptr %arrayidx2
  %t0 = load i32, ptr %ptr
  ret void
}

; But don't pessimize existing dereferenceable attribute.

define void @better_bytes(ptr dereferenceable(100) %ptr) {
; CHECK-LABEL: @better_bytes(ptr dereferenceable(100) %ptr)
  %arrayidx3 = getelementptr i32, ptr %ptr, i64 3
  %arrayidx1 = getelementptr i32, ptr %ptr, i64 1
  %arrayidx2 = getelementptr i32, ptr %ptr, i64 2
  %t3 = load i32, ptr %arrayidx3
  %t1 = load i32, ptr %arrayidx1
  %t2 = load i32, ptr %arrayidx2
  %t0 = load i32, ptr %ptr
  ret void
}

define void @bitcast(ptr %arg) {
; CHECK-LABEL: @bitcast(ptr %arg)
  %arrayidx1 = getelementptr float, ptr %arg, i64 1
  %t0 = load float, ptr %arg
  %t1 = load float, ptr %arrayidx1
  ret void
}

define void @bitcast_different_sizes(ptr %arg1, ptr %arg2) {
; CHECK-LABEL: @bitcast_different_sizes(ptr %arg1, ptr %arg2)
  %a11 = getelementptr float, ptr %arg1, i64 1
  %a12 = getelementptr float, ptr %arg1, i64 2
  %ld10 = load float, ptr %arg1
  %ld11 = load float, ptr %a11
  %ld12 = load float, ptr %a12

  %a21 = getelementptr i64, ptr %arg2, i64 1
  %ld20 = load i64, ptr %arg2
  %ld21 = load i64, ptr %a21
  ret void
}

define void @negative_offset(ptr %arg) {
; CHECK-LABEL: @negative_offset(ptr %arg)
  %arrayidx1 = getelementptr float, ptr %arg, i64 -1
  %t0 = load float, ptr %arg
  %t1 = load float, ptr %arrayidx1
  ret void
}

define void @stores(ptr %arg) {
; CHECK-LABEL: @stores(ptr %arg)
  %arrayidx1 = getelementptr float, ptr %arg, i64 1
  store float 1.0, ptr %arg
  store float 2.0, ptr %arrayidx1
  ret void
}

define void @load_store(ptr %arg) {
; CHECK-LABEL: @load_store(ptr %arg)
  %arrayidx1 = getelementptr float, ptr %arg, i64 1
  %t1 = load float, ptr %arg
  store float 2.0, ptr %arrayidx1
  ret void
}

define void @different_size1(ptr %arg) {
; CHECK-LABEL: @different_size1(ptr %arg)
  store double 0.000000e+00, ptr %arg
  store i32 0, ptr %arg
  ret void
}

define void @different_size2(ptr %arg) {
; CHECK-LABEL: @different_size2(ptr %arg)
  store i32 0, ptr %arg
  store double 0.000000e+00, ptr %arg
  ret void
}
