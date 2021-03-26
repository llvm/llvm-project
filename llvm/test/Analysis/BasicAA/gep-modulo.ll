; RUN: opt -basic-aa -aa-eval -print-all-alias-modref-info %s 2>&1 | FileCheck %s

target datalayout = "p:64:64:64"

; %gep.idx and %gep.6 must-alias if %mul overflows (e.g. %idx == 52).
define void @may_overflow_mul_add_i8([16 x i8]* %ptr, i8 %idx) {
; CHECK-LABEL: Function: may_overflow_mul_add_i8: 3 pointers, 0 call sites
; CHECK-NEXT:    MayAlias:  [16 x i8]* %ptr, i8* %gep.idx
; CHECK-NEXT:    PartialAlias: [16 x i8]* %ptr, i8* %gep.6
; CHECK-NEXT:    NoAlias:  i8* %gep.6, i8* %gep.idx
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
; CHECK-NEXT:    PartialAlias: [16 x i8]* %ptr, i8* %gep.6
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
; CHECK-NEXT:    PartialAlias: [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:  i8* %gep.3, i8* %gep.idx
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
; CHECK-NEXT:    PartialAlias: [16 x i8]* %ptr, i8* %gep.3
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
; CHECK-NEXT:    PartialAlias: [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:  i8* %gep.3, i8* %gep.idx
;
  %mul = mul i64 %idx, 5
  %sub = sub i64 %mul, 1
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
; CHECK-NEXT:    PartialAlias:  [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    MayAlias:	i8* %gep.3, i8* %gep.idx
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
; CHECK-NEXT:    PartialAlias:  [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:	i8* %gep.3, i8* %gep.idx
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
; CHECK-NEXT:    PartialAlias:  [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    MayAlias:	i8* %gep.3, i8* %gep.idx
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
; CHECK-NEXT:    PartialAlias:  [16 x i8]* %ptr, i8* %gep.3
; CHECK-NEXT:    NoAlias:	i8* %gep.3, i8* %gep.idx
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
