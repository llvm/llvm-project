; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output %s 2>&1 | FileCheck %s

; After GEPs decomposition we have: {i, Scale=12} and {j, Scale=18}. After subtraction:
; VarIdx = 12*i + (-18)*j = 6*(2*i − 3*j), where GCD(12,18)=6, C0=2, C1=3 (C0 even, C1 odd).
; Thus, the min abs offset equals to 6. As both size accesses are one byte, 6 >= 1 holds.
define void @noalias_mul_different_scale(ptr %base, i32 %a, i32 %b) {
; CHECK:       Function: noalias_mul_different_scale
; CHECK-NEXT:    NoAlias:	i8* %gep1, i8* %gep2
  %i = or i32 %a, 1
  %j = or i32 %b, 1
  %mul.idx.i = mul nsw i32 %i, 12
  %mul.idx.j = mul nsw i32 %j, 18
  %gep1 = getelementptr inbounds i8, ptr %base, i32 %mul.idx.i
  %gep2 = getelementptr inbounds i8, ptr %base, i32 %mul.idx.j
  load i8, ptr %gep1
  load i8, ptr %gep2
  ret void
}

; A shl on one index doubles its scale. After GEPs decomposition and scale by type size,
; we have: {i, Scale=4} and {j, Scale=8}.
; Pointer difference: 4*i − 8*j, where GCD(4,8)=4, C0=1, C1=2 (C0 odd, C1 even).
; Thus, the min abs offset equals to 4. As both size accesses are one byte, 4 >= 1 holds.
define void @noalias_shl_different_scale(ptr %base, i32 %a, i32 %b) {
; CHECK:       Function: noalias_shl_different_scale
; CHECK-NEXT:    NoAlias:	i8* %gep1, i8* %gep2
  %i = or i32 %a, 1
  %j = or i32 %b, 1
  %shl.idx = shl nsw i32 %j, 1
  %gep1 = getelementptr inbounds float, ptr %base, i32 %i
  %gep2 = getelementptr inbounds float, ptr %base, i32 %shl.idx
  load i8, ptr %gep1
  load i8, ptr %gep2
  ret void
}

; After GEPs decomposition and scale by type size we have: {i, Scale=4} and {j, Scale=8}.
; Thus, the min abs offset equals to 4. As both size accesses are one byte, 4 >= 1 holds.
define void @noalias_type_size_different_scale(ptr %base, i32 %a, i32 %b) {
; CHECK:       Function: noalias_type_size_different_scale
; CHECK-NEXT:    NoAlias:	i8* %gep1, i8* %gep2
  %i = or i32 %a, 1
  %j = or i32 %b, 1
  %gep1 = getelementptr inbounds float, ptr %base, i32 %i
  %gep2 = getelementptr inbounds double, ptr %base, i32 %j
  load i8, ptr %gep1
  load i8, ptr %gep2
  ret void
}

; Same as above, though after decomposition we have: {i, Scale=4, ZExtBits=32} and
; {j, Scale=8, ZExtBits=32}, thus KnownBits are evaluated accordingly.
define void @noalias_zext_chain(ptr %base, i32 %a, i32 %b) {
; CHECK:       Function: noalias_zext_chain
; CHECK-NEXT:    NoAlias:	i8* %gep1, i8* %gep2
  %i = or i32 %a, 1
  %j = or i32 %b, 1
  %zext.i = zext i32 %i to i64
  %zext.j = zext i32 %j to i64
  %gep1 = getelementptr inbounds float, ptr %base, i64 %zext.i
  %gep2 = getelementptr inbounds double, ptr %base, i64 %zext.j
  load i8, ptr %gep1
  load i8, ptr %gep2
  ret void
}

; Negative tests.

; Variable offset not a difference, thus C0*V0 + C1*V1 != 0 not implied.
define void @mayalias_same_effective_sign(ptr %base, i32 %a, i32 %b) {
; CHECK:       Function: mayalias_same_effective_sign
; CHECK-NEXT:    MayAlias:	i8* %gep1, i8* %gep2
  %i = mul nsw i32 %a, -12
  %j = mul nsw i32 %b, 18
  %gep1 = getelementptr inbounds i8, ptr %base, i32 %i
  %gep2 = getelementptr inbounds i8, ptr %base, i32 %j
  load i8, ptr %gep1
  load i8, ptr %gep2
  ret void
}

; When computing GCD(6,10)=2, we get C0=3 and C1=5. Products are identical,
; no conflicting bits can be found.
define void @mayalias_cofactors_not_different(ptr %base, i32 %a, i32 %b) {
; CHECK:       Function: mayalias_cofactors_not_different
; CHECK-NEXT:    MayAlias:	i8* %gep1, i8* %gep2
  %i = or i32 %a, 1
  %j = or i32 %b, 1
  %mul.idx.i = mul nsw i32 %i, 6
  %mul.idx.j = mul nsw i32 %j, 10
  %gep1 = getelementptr inbounds i8, ptr %base, i32 %mul.idx.i
  %gep2 = getelementptr inbounds i8, ptr %base, i32 %mul.idx.j
  load i8, ptr %gep1
  load i8, ptr %gep2
  ret void
}

; Access sizes exceed GCD, may overlap by 2 bytes.
define void @mayalias_access_wider_than_gcd(ptr %base, i32 %a, i32 %b) {
; CHECK:       Function: mayalias_access_wider_than_gcd
; CHECK-NEXT:    MayAlias:	i64* %gep1, i64* %gep2
  %i = or i32 %a, 1
  %j = or i32 %b, 1
  %mul.idx.i = mul nsw i32 %i, 12
  %mul.idx.j = mul nsw i32 %j, 18
  %gep1 = getelementptr inbounds i8, ptr %base, i32 %mul.idx.i
  %gep2 = getelementptr inbounds i8, ptr %base, i32 %mul.idx.j
  load i64, ptr %gep1
  load i64, ptr %gep2
  ret void
}
