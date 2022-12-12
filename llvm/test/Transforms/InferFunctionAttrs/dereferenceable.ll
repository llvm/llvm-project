; RUN: opt < %s -passes=inferattrs -S | FileCheck %s



; Determine dereference-ability before unused loads get deleted:
; https://bugs.llvm.org/show_bug.cgi?id=21780

define <4 x double> @PR21780(double* %ptr) {
; CHECK-LABEL: @PR21780(double* %ptr)

  ; GEP of index 0 is simplified away.
  %arrayidx1 = getelementptr inbounds double, double* %ptr, i64 1
  %arrayidx2 = getelementptr inbounds double, double* %ptr, i64 2
  %arrayidx3 = getelementptr inbounds double, double* %ptr, i64 3

  %t0 = load double, double* %ptr, align 8
  %t1 = load double, double* %arrayidx1, align 8
  %t2 = load double, double* %arrayidx2, align 8
  %t3 = load double, double* %arrayidx3, align 8

  %vecinit0 = insertelement <4 x double> undef, double %t0, i32 0
  %vecinit1 = insertelement <4 x double> %vecinit0, double %t1, i32 1
  %vecinit2 = insertelement <4 x double> %vecinit1, double %t2, i32 2
  %vecinit3 = insertelement <4 x double> %vecinit2, double %t3, i32 3
  %shuffle = shufflevector <4 x double> %vecinit3, <4 x double> %vecinit3, <4 x i32> <i32 0, i32 0, i32 2, i32 2>
  ret <4 x double> %shuffle
}


define double @PR21780_only_access3_with_inbounds(double* %ptr) {
; CHECK-LABEL: @PR21780_only_access3_with_inbounds(double* %ptr)

  %arrayidx3 = getelementptr inbounds double, double* %ptr, i64 3
  %t3 = load double, double* %arrayidx3, align 8
  ret double %t3
}

define double @PR21780_only_access3_without_inbounds(double* %ptr) {
; CHECK-LABEL: @PR21780_only_access3_without_inbounds(double* %ptr)
  %arrayidx3 = getelementptr double, double* %ptr, i64 3
  %t3 = load double, double* %arrayidx3, align 8
  ret double %t3
}

define double @PR21780_without_inbounds(double* %ptr) {
; CHECK-LABEL: @PR21780_without_inbounds(double* %ptr)

  %arrayidx1 = getelementptr double, double* %ptr, i64 1
  %arrayidx2 = getelementptr double, double* %ptr, i64 2
  %arrayidx3 = getelementptr double, double* %ptr, i64 3

  %t0 = load double, double* %ptr, align 8
  %t1 = load double, double* %arrayidx1, align 8
  %t2 = load double, double* %arrayidx2, align 8
  %t3 = load double, double* %arrayidx3, align 8

  ret double %t3
}

; Unsimplified, but still valid. Also, throw in some bogus arguments.

define void @gep0(i8* %unused, i8* %other, i8* %ptr) {
; CHECK-LABEL: @gep0(i8* %unused, i8* %other, i8* %ptr)
  %arrayidx0 = getelementptr i8, i8* %ptr, i64 0
  %arrayidx1 = getelementptr i8, i8* %ptr, i64 1
  %arrayidx2 = getelementptr i8, i8* %ptr, i64 2
  %t0 = load i8, i8* %arrayidx0
  %t1 = load i8, i8* %arrayidx1
  %t2 = load i8, i8* %arrayidx2
  store i8 %t2, i8* %other
  ret void
}

; Order of accesses does not change computation.
; Multiple arguments may be dereferenceable.

define void @ordering(i8* %ptr1, i32* %ptr2) {
; CHECK-LABEL: @ordering(i8* %ptr1, i32* %ptr2)
  %a20 = getelementptr i32, i32* %ptr2, i64 0
  %a12 = getelementptr i8, i8* %ptr1, i64 2
  %t12 = load i8, i8* %a12
  %a11 = getelementptr i8, i8* %ptr1, i64 1
  %t20 = load i32, i32* %a20
  %a10 = getelementptr i8, i8* %ptr1, i64 0
  %t10 = load i8, i8* %a10
  %t11 = load i8, i8* %a11
  %a21 = getelementptr i32, i32* %ptr2, i64 1
  %t21 = load i32, i32* %a21
  ret void
}

; Not in entry block.

define void @not_entry_but_guaranteed_to_execute(i8* %ptr) {
; CHECK-LABEL: @not_entry_but_guaranteed_to_execute(i8* %ptr)
entry:
  br label %exit
exit:
  %arrayidx0 = getelementptr i8, i8* %ptr, i64 0
  %arrayidx1 = getelementptr i8, i8* %ptr, i64 1
  %arrayidx2 = getelementptr i8, i8* %ptr, i64 2
  %t0 = load i8, i8* %arrayidx0
  %t1 = load i8, i8* %arrayidx1
  %t2 = load i8, i8* %arrayidx2
  ret void
}

; Not in entry block and not guaranteed to execute.

define void @not_entry_not_guaranteed_to_execute(i8* %ptr, i1 %cond) {
; CHECK-LABEL: @not_entry_not_guaranteed_to_execute(i8* %ptr, i1 %cond)
entry:
  br i1 %cond, label %loads, label %exit
loads:
  %arrayidx0 = getelementptr i8, i8* %ptr, i64 0
  %arrayidx1 = getelementptr i8, i8* %ptr, i64 1
  %arrayidx2 = getelementptr i8, i8* %ptr, i64 2
  %t0 = load i8, i8* %arrayidx0
  %t1 = load i8, i8* %arrayidx1
  %t2 = load i8, i8* %arrayidx2
  ret void
exit:
  ret void
}

; The last load may not execute, so derefenceable bytes only covers the 1st two loads.

define void @partial_in_entry(i16* %ptr, i1 %cond) {
; CHECK-LABEL: @partial_in_entry(i16* %ptr, i1 %cond)
entry:
  %arrayidx0 = getelementptr i16, i16* %ptr, i64 0
  %arrayidx1 = getelementptr i16, i16* %ptr, i64 1
  %arrayidx2 = getelementptr i16, i16* %ptr, i64 2
  %t0 = load i16, i16* %arrayidx0
  %t1 = load i16, i16* %arrayidx1
  br i1 %cond, label %loads, label %exit
loads:
  %t2 = load i16, i16* %arrayidx2
  ret void
exit:
  ret void
}

; The volatile load can't be used to prove a non-volatile access is allowed.
; The 2nd and 3rd loads may never execute.

define void @volatile_is_not_dereferenceable(i16* %ptr) {
; CHECK-LABEL: @volatile_is_not_dereferenceable(i16* %ptr)
  %arrayidx0 = getelementptr i16, i16* %ptr, i64 0
  %arrayidx1 = getelementptr i16, i16* %ptr, i64 1
  %arrayidx2 = getelementptr i16, i16* %ptr, i64 2
  %t0 = load volatile i16, i16* %arrayidx0
  %t1 = load i16, i16* %arrayidx1
  %t2 = load i16, i16* %arrayidx2
  ret void
}

; TODO: We should allow inference for atomic (but not volatile) ops.

define void @atomic_is_alright(i16* %ptr) {
; CHECK-LABEL: @atomic_is_alright(i16* %ptr)
  %arrayidx0 = getelementptr i16, i16* %ptr, i64 0
  %arrayidx1 = getelementptr i16, i16* %ptr, i64 1
  %arrayidx2 = getelementptr i16, i16* %ptr, i64 2
  %t0 = load atomic i16, i16* %arrayidx0 unordered, align 2
  %t1 = load i16, i16* %arrayidx1
  %t2 = load i16, i16* %arrayidx2
  ret void
}

declare void @may_not_return()

define void @not_guaranteed_to_transfer_execution(i16* %ptr) {
; CHECK-LABEL: @not_guaranteed_to_transfer_execution(i16* %ptr)
  %arrayidx0 = getelementptr i16, i16* %ptr, i64 0
  %arrayidx1 = getelementptr i16, i16* %ptr, i64 1
  %arrayidx2 = getelementptr i16, i16* %ptr, i64 2
  %t0 = load i16, i16* %arrayidx0
  call void @may_not_return()
  %t1 = load i16, i16* %arrayidx1
  %t2 = load i16, i16* %arrayidx2
  ret void
}

; We must have consecutive accesses.

define void @variable_gep_index(i8* %unused, i8* %ptr, i64 %variable_index) {
; CHECK-LABEL: @variable_gep_index(i8* %unused, i8* %ptr, i64 %variable_index)
  %arrayidx1 = getelementptr i8, i8* %ptr, i64 %variable_index
  %arrayidx2 = getelementptr i8, i8* %ptr, i64 2
  %t0 = load i8, i8* %ptr
  %t1 = load i8, i8* %arrayidx1
  %t2 = load i8, i8* %arrayidx2
  ret void
}

; Deal with >1 GEP index.

define void @multi_index_gep(<4 x i8>* %ptr) {
; CHECK-LABEL: @multi_index_gep(<4 x i8>* %ptr)
; FIXME: %ptr should be dereferenceable(4)
  %arrayidx00 = getelementptr <4 x i8>, <4 x i8>* %ptr, i64 0, i64 0
  %t0 = load i8, i8* %arrayidx00
  ret void
}

; Could round weird bitwidths down?

define void @not_byte_multiple(i9* %ptr) {
; CHECK-LABEL: @not_byte_multiple(i9* %ptr)
  %arrayidx0 = getelementptr i9, i9* %ptr, i64 0
  %t0 = load i9, i9* %arrayidx0
  ret void
}

; Missing direct access from the pointer.

define void @no_pointer_deref(i16* %ptr) {
; CHECK-LABEL: @no_pointer_deref(i16* %ptr)
  %arrayidx1 = getelementptr i16, i16* %ptr, i64 1
  %arrayidx2 = getelementptr i16, i16* %ptr, i64 2
  %t1 = load i16, i16* %arrayidx1
  %t2 = load i16, i16* %arrayidx2
  ret void
}

; Out-of-order is ok, but missing access concludes dereferenceable range.

define void @non_consecutive(i32* %ptr) {
; CHECK-LABEL: @non_consecutive(i32* %ptr)
  %arrayidx1 = getelementptr i32, i32* %ptr, i64 1
  %arrayidx0 = getelementptr i32, i32* %ptr, i64 0
  %arrayidx3 = getelementptr i32, i32* %ptr, i64 3
  %t1 = load i32, i32* %arrayidx1
  %t0 = load i32, i32* %arrayidx0
  %t3 = load i32, i32* %arrayidx3
  ret void
}

; Improve on existing dereferenceable attribute.

define void @more_bytes(i32* dereferenceable(8) %ptr) {
; CHECK-LABEL: @more_bytes(i32* dereferenceable(8) %ptr)
  %arrayidx3 = getelementptr i32, i32* %ptr, i64 3
  %arrayidx1 = getelementptr i32, i32* %ptr, i64 1
  %arrayidx0 = getelementptr i32, i32* %ptr, i64 0
  %arrayidx2 = getelementptr i32, i32* %ptr, i64 2
  %t3 = load i32, i32* %arrayidx3
  %t1 = load i32, i32* %arrayidx1
  %t2 = load i32, i32* %arrayidx2
  %t0 = load i32, i32* %arrayidx0
  ret void
}

; Improve on existing dereferenceable_or_null attribute.

define void @more_bytes_and_not_null(i32* dereferenceable_or_null(8) %ptr) {
; CHECK-LABEL: @more_bytes_and_not_null(i32* dereferenceable_or_null(8) %ptr)
  %arrayidx3 = getelementptr i32, i32* %ptr, i64 3
  %arrayidx1 = getelementptr i32, i32* %ptr, i64 1
  %arrayidx0 = getelementptr i32, i32* %ptr, i64 0
  %arrayidx2 = getelementptr i32, i32* %ptr, i64 2
  %t3 = load i32, i32* %arrayidx3
  %t1 = load i32, i32* %arrayidx1
  %t2 = load i32, i32* %arrayidx2
  %t0 = load i32, i32* %arrayidx0
  ret void
}

; But don't pessimize existing dereferenceable attribute.

define void @better_bytes(i32* dereferenceable(100) %ptr) {
; CHECK-LABEL: @better_bytes(i32* dereferenceable(100) %ptr)
  %arrayidx3 = getelementptr i32, i32* %ptr, i64 3
  %arrayidx1 = getelementptr i32, i32* %ptr, i64 1
  %arrayidx0 = getelementptr i32, i32* %ptr, i64 0
  %arrayidx2 = getelementptr i32, i32* %ptr, i64 2
  %t3 = load i32, i32* %arrayidx3
  %t1 = load i32, i32* %arrayidx1
  %t2 = load i32, i32* %arrayidx2
  %t0 = load i32, i32* %arrayidx0
  ret void
}

define void @bitcast(i32* %arg) {
; CHECK-LABEL: @bitcast(i32* %arg)
  %ptr = bitcast i32* %arg to float*
  %arrayidx0 = getelementptr float, float* %ptr, i64 0
  %arrayidx1 = getelementptr float, float* %ptr, i64 1
  %t0 = load float, float* %arrayidx0
  %t1 = load float, float* %arrayidx1
  ret void
}

define void @bitcast_different_sizes(double* %arg1, i8* %arg2) {
; CHECK-LABEL: @bitcast_different_sizes(double* %arg1, i8* %arg2)
  %ptr1 = bitcast double* %arg1 to float*
  %a10 = getelementptr float, float* %ptr1, i64 0
  %a11 = getelementptr float, float* %ptr1, i64 1
  %a12 = getelementptr float, float* %ptr1, i64 2
  %ld10 = load float, float* %a10
  %ld11 = load float, float* %a11
  %ld12 = load float, float* %a12

  %ptr2 = bitcast i8* %arg2 to i64*
  %a20 = getelementptr i64, i64* %ptr2, i64 0
  %a21 = getelementptr i64, i64* %ptr2, i64 1
  %ld20 = load i64, i64* %a20
  %ld21 = load i64, i64* %a21
  ret void
}

define void @negative_offset(i32* %arg) {
; CHECK-LABEL: @negative_offset(i32* %arg)
  %ptr = bitcast i32* %arg to float*
  %arrayidx0 = getelementptr float, float* %ptr, i64 0
  %arrayidx1 = getelementptr float, float* %ptr, i64 -1
  %t0 = load float, float* %arrayidx0
  %t1 = load float, float* %arrayidx1
  ret void
}

define void @stores(i32* %arg) {
; CHECK-LABEL: @stores(i32* %arg)
  %ptr = bitcast i32* %arg to float*
  %arrayidx0 = getelementptr float, float* %ptr, i64 0
  %arrayidx1 = getelementptr float, float* %ptr, i64 1
  store float 1.0, float* %arrayidx0
  store float 2.0, float* %arrayidx1
  ret void
}

define void @load_store(i32* %arg) {
; CHECK-LABEL: @load_store(i32* %arg)
  %ptr = bitcast i32* %arg to float*
  %arrayidx0 = getelementptr float, float* %ptr, i64 0
  %arrayidx1 = getelementptr float, float* %ptr, i64 1
  %t1 = load float, float* %arrayidx0
  store float 2.0, float* %arrayidx1
  ret void
}

define void @different_size1(i32* %arg) {
; CHECK-LABEL: @different_size1(i32* %arg)
  %arg-cast = bitcast i32* %arg to double*
  store double 0.000000e+00, double* %arg-cast
  store i32 0, i32* %arg
  ret void
}

define void @different_size2(i32* %arg) {
; CHECK-LABEL: @different_size2(i32* %arg)
  store i32 0, i32* %arg
  %arg-cast = bitcast i32* %arg to double*
  store double 0.000000e+00, double* %arg-cast
  ret void
}
