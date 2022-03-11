; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output %s 2>&1 | FileCheck %s

target datalayout = "p:64:64:64"

; %gep.idx and %gep.6 must-alias if %mul overflows (e.g. %idx == 52).
define void @may_overflow_mul_add_i8([16 x i8]* %ptr, i8 %idx) {
; CHECK-LABEL: Function: may_overflow_mul_add_i8: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 6): [16 x i8]* %ptr, i8* %gep.6
; CHECK-NEXT:    MayAlias:  i8* %gep.6, i8* %gep.idx
;
  %mul = mul i8 %idx, 5
  %add = add i8 %mul, 2
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i8 %add
  store i8 0, i8* %gep.idx, align 1
  %gep.6 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i32 6
  store i8 1, i8* %gep.6, align 1
  ret void
}

define void @nuw_nsw_mul_add_i8([16 x i8]* %ptr, i8 %idx) {
; CHECK-LABEL: Function: nuw_nsw_mul_add_i8: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias: [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 6): [16 x i8]* %ptr, i8* %gep.6
; CHECK-NEXT:    NoAlias:  i8* %gep.6, i8* %gep.idx
;
  %mul = mul nuw nsw i8 %idx, 5
  %add = add nuw nsw i8 %mul, 2
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i8 %add
  store i8 0, i8* %gep.idx, align 1
  %gep.6 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i32 6
  store i8 1, i8* %gep.6, align 1
  ret void
}

; %gep.idx and %gep.3 must-alias if %mul overflows (e.g. %idx == 52).
define void @may_overflow_mul_sub_i8([16 x i8]* %ptr, i8 %idx) {
; CHECK-LABEL: Function: may_overflow_mul_sub_i8: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3): [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    MayAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul i8 %idx, 5
  %sub = sub i8 %mul, 1
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i8 %sub
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i32 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

define void @nuw_nsw_mul_sub_i8([16 x i8]* %ptr, i8 %idx) {
; CHECK-LABEL: Function: nuw_nsw_mul_sub_i8: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3): [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul nuw nsw i8 %idx, 5
  %sub = sub nuw nsw i8 %mul, 1
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i8 %sub
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i32 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

; %gep.idx and %gep.3 must-alias if %mul overflows
; (e.g. %idx == 3689348814741910323).
define void @may_overflow_mul_sub_i64([16 x i8]* %ptr, i64 %idx) {
; CHECK-LABEL: Function: may_overflow_mul_sub_i64: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3): [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    MayAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul i64 %idx, 5
  %sub = sub i64 %mul, 1
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

define void @nuw_nsw_mul_sub_i64([16 x i8]* %ptr, i64 %idx) {
; CHECK-LABEL: Function: nuw_nsw_mul_sub_i64: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3): [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul nuw nsw i64 %idx, 5
  %sub = sub nuw nsw i64 %mul, 1
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

define void @only_nsw_mul_sub_i64([16 x i8]* %ptr, i64 %idx) {
; CHECK-LABEL: Function: only_nsw_mul_sub_i64: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3): [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul nsw i64 %idx, 5
  %sub = sub nsw i64 %mul, 1
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

define void @only_nuw_mul_sub_i64([16 x i8]* %ptr, i64 %idx) {
; CHECK-LABEL: Function: only_nuw_mul_sub_i64: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3): [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    MayAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul nuw i64 %idx, 5
  %sub = sub nuw i64 %mul, 1
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

; Even though the mul and sub may overflow %gep.idx and %gep.3 cannot alias
; because we multiply by a power-of-2.
define void @may_overflow_mul_pow2_sub_i64([16 x i8]* %ptr, i64 %idx) {
; CHECK-LABEL: Function: may_overflow_mul_pow2_sub_i64: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3): [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul i64 %idx, 8
  %sub = sub i64 %mul, 1
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

; Multiplies by power-of-2 preserves modulo and the sub does not wrap.
define void @mul_pow2_sub_nsw_nuw_i64([16 x i8]* %ptr, i64 %idx) {
; CHECK-LABEL: Function: mul_pow2_sub_nsw_nuw_i64: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3): [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul i64 %idx, 8
  %sub = sub nuw nsw i64 %mul, 1
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

define void @may_overflow_shl_sub_i64([16 x i8]* %ptr, i64 %idx) {
; CHECK-LABEL: Function: may_overflow_shl_sub_i64: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3): [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    MayAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = shl i64 %idx, 2
  %sub = sub i64 %mul, 1
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

define void @shl_sub_nsw_nuw_i64([16 x i8]* %ptr, i64 %idx) {
; CHECK-LABEL: Function: shl_sub_nsw_nuw_i64: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3): [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = shl i64 %idx, 3
  %sub = sub nsw nuw i64 %mul, 1
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

; %gep.idx and %gep.3 must-alias if %mul overflows (e.g. %idx == 110).
define void @may_overflow_i32_sext([16 x i8]* %ptr, i32 %idx) {
; CHECK-LABEL: Function: may_overflow_i32_sext: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3):  [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    MayAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul i32 %idx, 678152731
  %sub = sub i32 %mul, 1582356375
  %sub.ext = sext i32 %sub to i64
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub.ext
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i32 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

define void @nuw_nsw_i32_sext([16 x i8]* %ptr, i32 %idx) {
; CHECK-LABEL: Function: nuw_nsw_i32_sext: 3 pointers, 0 call sites
; CHECK-NEXT:    NoAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3):  [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:   i8* %gep.3, i8* %gep.idx
;
  %mul = mul nuw nsw i32 %idx, 678152731
  %sub = sub nuw nsw i32 %mul, 1582356375
  %sub.ext = sext i32 %sub to i64
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub.ext
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i32 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

; %gep.idx and %gep.3 must-alias if %mul overflows (e.g. %idx == 110).
define void @may_overflow_i32_zext([16 x i8]* %ptr, i32 %idx) {
; CHECK-LABEL: Function: may_overflow_i32_zext: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3):  [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    MayAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul i32 %idx, 678152731
  %sub = sub i32 %mul, 1582356375
  %sub.ext = zext i32 %sub to i64
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub.ext
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i32 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

define void @nuw_nsw_i32_zext([16 x i8]* %ptr, i32 %idx) {
; CHECK-LABEL: Function: nuw_nsw_i32_zext: 3 pointers, 0 call sites
; CHECK-NEXT:    NoAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias (off 3):  [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:   i8* %gep.3, i8* %gep.idx
;
  %mul = mul nuw nsw i32 %idx, 678152731
  %sub = sub nuw nsw i32 %mul, 1582356375
  %sub.ext = zext i32 %sub to i64
  %gep.idx = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub.ext
  store i8 0, i8* %gep.idx, align 1
  %gep.3 = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i32 3
  store i8 1, i8* %gep.3, align 1
  ret void
}

; %mul.1 and %sub.2 are equal, if %idx = 9, because %mul.1 overflows. Hence
; %gep.mul.1 and %gep.sub.2 may alias.
define void @may_overflow_pointer_diff([16 x i8]* %ptr, i64 %idx) {
; CHECK-LABEL: Function: may_overflow_pointer_diff: 3 pointers, 0 call sites
; CHECK-NEXT:  MayAlias: [16 x i8]* %ptr, i8* %gep.mul.1
; CHECK-NEXT:  MayAlias: [16 x i8]* %ptr, i8* %gep.sub.2
; CHECK-NEXT:  MayAlias:  i8* %gep.mul.1, i8* %gep.sub.2
;
  %mul.1 = mul i64 %idx, 6148914691236517207
  %gep.mul.1  = getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %mul.1
  store i8 1, i8* %gep.mul.1, align 1
  %mul.2 = mul nsw i64 %idx, 3
  %sub.2 = sub nsw i64 %mul.2, 12
  %gep.sub.2= getelementptr [16 x i8], [16 x i8]* %ptr, i32 0, i64 %sub.2
  store i8 0, i8* %gep.sub.2, align 1

  ret void
}

; %gep.1 and %gep.idx may alias, e.g. if %idx.1 = 8 and %idx.2 == 2. %gep.idx is then
;  (((18446744073709551614 * 8) % 2^64 + 6 * 2) % 2^64 + 10) % 2^64 == 6.
define void @may_overflow_mul_scale_neg([200 x [ 6 x i8]]* %ptr, i64 %idx.1,i64 %idx.2) {
; CHECK-LABEL: Function: may_overflow_mul_scale_neg: 4 pointers, 2 call sites
; CHECK-NEXT:  MustAlias:   [200 x [6 x i8]]* %ptr, i8* %bc
; CHECK-NEXT:  PartialAlias (off 6):    [200 x [6 x i8]]* %ptr, i8* %gep.1
; CHECK-NEXT:  NoAlias: i8* %bc, i8* %gep.1
; CHECK-NEXT:  MayAlias:    [200 x [6 x i8]]* %ptr, i8* %gep.idx
; CHECK-NEXT:  MayAlias: i8* %bc, i8* %gep.idx
; CHECK-NEXT:  MayAlias: i8* %gep.1, i8* %gep.idx
;
  %idx.1.pos = icmp sge i64 %idx.1, 0
  call void @llvm.assume(i1 %idx.1.pos)
  %idx.2.pos = icmp sge i64 %idx.2, 0
  call void @llvm.assume(i1 %idx.2.pos)

  %bc = bitcast [ 200 x [ 6 x i8 ] ]* %ptr to i8*
  %gep.1 = getelementptr i8, i8* %bc, i64 6
  store i8 1, i8* %gep.1, align 1

  %mul.0 = mul i64 %idx.1, -2
  %add = add i64 %mul.0, 10
  %gep.idx = getelementptr [ 200 x [ 6 x i8 ] ], [ 200 x [ 6 x i8 ] ]* %ptr, i64 0, i64 %idx.2, i64 %add
  store i8 0, i8* %gep.idx, align 1
  ret void
}

declare void @llvm.assume(i1)
