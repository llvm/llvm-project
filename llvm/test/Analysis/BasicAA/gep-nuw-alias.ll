; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: test_no_lower_bound
;
; CHECK-DAG: MayAlias: i32* %a, i32* %b
define void @test_no_lower_bound(ptr %p, i64 %i) {
  %a = getelementptr i8, ptr %p, i64 4
  %b = getelementptr nuw i8, ptr %p, i64 %i

  load i32, ptr %a
  load i32, ptr %b

  ret void
}

; CHECK-LABEL: test_lower_bound_lt_size
;
; CHECK-DAG: MayAlias: i32* %a, i32* %b
define void @test_lower_bound_lt_size(ptr %p, i64 %i) {
  %a = getelementptr i8, ptr %p
  %add = getelementptr nuw i8, ptr %p, i64 2
  %b = getelementptr nuw i8, ptr %add, i64 %i

  load i32, ptr %a
  load i32, ptr %b

  ret void
}

; CHECK-LABEL: test_lower_bound_ge_size
;
; CHECK-DAG: NoAlias: i32* %a, i32* %b
define void @test_lower_bound_ge_size(ptr %p, i64 %i) {
  %a = getelementptr i8, ptr %p
  %add = getelementptr nuw i8, ptr %p, i64 4
  %b = getelementptr nuw i8, ptr %add, i64 %i

  load i32, ptr %a
  load i32, ptr %b

  ret void
}

; CHECK-LABEL: test_not_all_nuw
;
; If part of the addressing is done with non-nuw GEPs, we can't use properties
; implied by the last GEP with the whole offset. In this case, the calculation
; of %add (%p + 4) could wrap the pointer index type, such that %add +<nuw> %i
; could still alias with %p.
;
; CHECK-DAG: MayAlias: i32* %a, i32* %b
define void @test_not_all_nuw(ptr %p, i64 %i) {
  %a = getelementptr i8, ptr %p
  %add = getelementptr i8, ptr %p, i64 4
  %b = getelementptr nuw i8, ptr %add, i64 %i

  load i32, ptr %a
  load i32, ptr %b

  ret void
}

; CHECK-LABEL: test_multi_step_not_all_nuw
;
; CHECK-DAG: MayAlias: i32* %a, i32* %b
define void @test_multi_step_not_all_nuw(ptr %p, i64 %i, i64 %j, i64 %k) {
  %a = getelementptr i8, ptr %p
  %add = getelementptr i8, ptr %p, i64 4
  %step1 = getelementptr i8, ptr %add, i64 %i
  %step2 = getelementptr i8, ptr %step1, i64 %j
  %b = getelementptr nuw i8, ptr %step2, i64 %k

  load i32, ptr %a
  load i32, ptr %b

  ret void
}

; CHECK-LABEL: test_multi_step_all_nuw
;
; CHECK-DAG: NoAlias: i32* %a, i32* %b
define void @test_multi_step_all_nuw(ptr %p, i64 %i, i64 %j, i64 %k) {
  %a = getelementptr i8, ptr %p
  %add = getelementptr nuw i8, ptr %p, i64 4
  %step1 = getelementptr nuw i8, ptr %add, i64 %i
  %step2 = getelementptr nuw i8, ptr %step1, i64 %j
  %b = getelementptr nuw i8, ptr %step2, i64 %k

  load i32, ptr %a
  load i32, ptr %b

  ret void
}

%struct = type { i64, [2 x i32], i64 }

; CHECK-LABEL: test_struct_no_nuw
;
; The array access may alias with the struct elements before and after, because
; we cannot prove that (%arr + %i) does not alias with the base pointer %p.
;
; CHECK-DAG: MayAlias: i32* %arrayidx, i64* %st
; CHECK-DAG: NoAlias: i64* %after, i64* %st
; CHECK-DAG: MayAlias: i64* %after, i32* %arrayidx

define void @test_struct_no_nuw(ptr %st, i64 %i) {
  %arr = getelementptr i8, ptr %st, i64 8
  %arrayidx = getelementptr [2 x i32], ptr %arr, i64 0, i64 %i
  %after = getelementptr i8, ptr %st, i64 16

  load i64, ptr %st
  load i32, ptr %arrayidx
  load i64, ptr %after

  ret void
}

; CHECK-LABEL: test_struct_nuw
;
; We can prove that the array access does not alias with struct element before,
; because we can prove that (%arr +<nuw> %i) does not wrap the pointer index
; type (add nuw). The array access may still alias with the struct element
; after, as the add nuw property does not preclude this.
;
; CHECK-DAG: NoAlias: i32* %arrayidx, i64* %st
; CHECK-DAG: NoAlias: i64* %after, i64* %st
; CHECK-DAG: MayAlias: i64* %after, i32* %arrayidx

define void @test_struct_nuw(ptr %st, i64 %i) {
  %arr = getelementptr nuw i8, ptr %st, i64 8
  %arrayidx = getelementptr nuw [2 x i32], ptr %arr, i64 0, i64 %i
  %after = getelementptr nuw i8, ptr %st, i64 16

  load i64, ptr %st
  load i32, ptr %arrayidx
  load i64, ptr %after

  ret void
}

; CHECK-LABEL: constant_offset_overflow
;
; If subtraction of constant offsets could overflow in an unsigned sense, we
; cannot prove the lower bound between the GEPs and so they may still alias.
;
; CHECK-DAG: MayAlias: i32* %a, i32* %b

define void @constant_offset_overflow(ptr %p, i64 %i) {
  %a = getelementptr i8, ptr %p, i64 -8
  %add = getelementptr nuw i8, ptr %p, i64 4
  %b = getelementptr nuw i8, ptr %add, i64 %i

  load i32, ptr %a
  load i32, ptr %b

  ret void
}

; CHECK-LABEL: equal_var_idx_noalias
;
; If GEPs have equal variable indices, we can prove NoAlias when the Scale of
; the RHS GEP is greater, as in this scenario the constant lower bound holds.
;
; CHECK-DAG: NoAlias: i32* %a, i32* %b

define void @equal_var_idx_noalias(ptr %p, i64 %i) {
  %a = getelementptr i8, ptr %p, i64 %i

  %add = getelementptr nuw i8, ptr %p, i64 4
  %b = getelementptr nuw i16, ptr %add, i64 %i

  load i32, ptr %a
  load i32, ptr %b

  ret void
}

; CHECK-LABEL: equal_var_idx_alias
;
; If GEPs have equal variable indices, we cannot prove NoAlias when the Scale of
; the RHS GEP is ult Scale of the LHS GEP.
;
; CHECK-DAG: MayAlias: i32* %a, i32* %b

define void @equal_var_idx_alias(ptr %p, i64 %i) {
  %a = getelementptr i32, ptr %p, i64 %i

  %add = getelementptr nuw i8, ptr %p, i64 4
  %b = getelementptr nuw i16, ptr %add, i64 %i

  load i32, ptr %b
  load i32, ptr %a

  ret void
}

; CHECK-LABEL: both_var_idx
;
; If the RHS GEP has unmatched variable indices, we cannot prove a constant
; lower bound between GEPs.
;
; CHECK-DAG: MayAlias: i32* %a, i32* %b

define void @both_var_idx(ptr %p, i64 %i, i64 %j) {
  %a = getelementptr i8, ptr %p, i64 %i

  %add = getelementptr nuw i8, ptr %p, i64 4
  %b = getelementptr nuw i8, ptr %add, i64 %j

  load i32, ptr %a
  load i32, ptr %b

  ret void
}

; CHECK-LABEL: add_no_nuw
; CHECK: MayAlias: i8* %gep, i8* %p
define i8 @add_no_nuw(ptr %p, i64 %n) {
  store i8 3, ptr %p

  %add = add i64 %n, 1
  %gep = getelementptr nuw i8, ptr %p, i64 %add
  %val = load i8, ptr %gep
  ret i8 %val
}

; CHECK-LABEL: add_nuw
; CHECK: NoAlias: i8* %gep, i8* %p
define i8 @add_nuw(ptr %p, i64 %n) {
  store i8 3, ptr %p

  %add = add nuw i64 %n, 1
  %gep = getelementptr nuw i8, ptr %p, i64 %add
  %val = load i8, ptr %gep
  ret i8 %val
}

; CHECK-LABEL: add_no_nuw
; CHECK: MayAlias: i8* %gep, i16* %p
define i8 @add_no_nuw_scale(ptr %p, i64 %n) {
  store i16 3, ptr %p

  %add = add i64 %n, 1
  %gep = getelementptr nuw i16, ptr %p, i64 %add
  %val = load i8, ptr %gep
  ret i8 %val
}

; CHECK-LABEL: add_nuw
; CHECK: NoAlias: i8* %gep, i16* %p
define i8 @add_nuw_scale(ptr %p, i64 %n) {
  store i16 3, ptr %p

  %add = add nuw i64 %n, 1
  %gep = getelementptr nuw i16, ptr %p, i64 %add
  %val = load i8, ptr %gep
  ret i8 %val
}

; CHECK-LABEL: sub_nuw
; CHECK: MayAlias: i8* %gep, i8* %p
define i8 @sub_nuw(ptr %p, i64 %n) {
  store i8 3, ptr %p

  %add = sub nuw i64 %n, 1
  %gep = getelementptr nuw i8, ptr %p, i64 %add
  %val = load i8, ptr %gep
  ret i8 %val
}

; CHECK-LABEL: mul_no_nuw
; CHECK: MayAlias: i8* %gep, i16* %p
define i8 @mul_no_nuw(ptr %p, i64 %n) {
  store i16 3, ptr %p

  %add = add nuw i64 %n, 1
  %mul = mul i64 %add, 2
  %gep = getelementptr nuw i8, ptr %p, i64 %mul
  %val = load i8, ptr %gep
  ret i8 %val
}

; CHECK-LABEL: mul_nuw
; CHECK: NoAlias: i8* %gep, i16* %p
define i8 @mul_nuw(ptr %p, i64 %n) {
  store i16 3, ptr %p

  %add = add nuw i64 %n, 1
  %mul = mul nuw i64 %add, 2
  %gep = getelementptr nuw i8, ptr %p, i64 %mul
  %val = load i8, ptr %gep
  ret i8 %val
}

; CHECK-LABEL: shl_no_nuw
; CHECK: MayAlias: i8* %gep, i16* %p
define i8 @shl_no_nuw(ptr %p, i64 %n) {
  store i16 3, ptr %p

  %add = add nuw i64 %n, 1
  %shl = shl i64 %add, 1
  %gep = getelementptr nuw i8, ptr %p, i64 %shl
  %val = load i8, ptr %gep
  ret i8 %val
}

; CHECK-LABEL: shl_nuw
; CHECK: NoAlias: i8* %gep, i16* %p
define i8 @shl_nuw(ptr %p, i64 %n) {
  store i16 3, ptr %p

  %add = add nuw i64 %n, 1
  %shl = shl nuw i64 %add, 1
  %gep = getelementptr nuw i8, ptr %p, i64 %shl
  %val = load i8, ptr %gep
  ret i8 %val
}
