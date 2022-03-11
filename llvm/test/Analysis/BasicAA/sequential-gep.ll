; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK: Function: t1
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @t1([8 x i32]* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], [8 x i32]* %p, i32 2, i32 %addend
  %gep2 = getelementptr [8 x i32], [8 x i32]* %p, i32 2, i32 %add
  ret void
}

; CHECK: Function: t2
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t2([8 x i32]* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], [8 x i32]* %p, i32 1, i32 %addend
  %gep2 = getelementptr [8 x i32], [8 x i32]* %p, i32 0, i32 %add
  ret void
}

; CHECK: Function: t3
; CHECK: MustAlias: i32* %gep1, i32* %gep2
define void @t3([8 x i32]* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], [8 x i32]* %p, i32 0, i32 %add
  %gep2 = getelementptr [8 x i32], [8 x i32]* %p, i32 0, i32 %add
  ret void
}

; CHECK: Function: t4
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t4([8 x i32]* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], [8 x i32]* %p, i32 1, i32 %addend
  %gep2 = getelementptr [8 x i32], [8 x i32]* %p, i32 %add, i32 %add
  ret void
}

; CHECK: Function: t5
; CHECK: MayAlias: i32* %gep2, i64* %bc
define void @t5([8 x i32]* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], [8 x i32]* %p, i32 2, i32 %addend
  %gep2 = getelementptr [8 x i32], [8 x i32]* %p, i32 2, i32 %add
  %bc = bitcast i32* %gep1 to i64*
  ret void
}

; CHECK-LABEL: Function: add_non_zero_simple
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @add_non_zero_simple(i32* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add i32 %addend, %knownnonzero
  %gep1 = getelementptr i32, i32* %p, i32 %addend
  %gep2 = getelementptr i32, i32* %p, i32 %add
  ret void
}

; CHECK-LABEL: Function: add_non_zero_different_scales
; CHECK: MayAlias: i16* %gep2, i32* %gep1
define void @add_non_zero_different_scales(i32* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add i32 %addend, %knownnonzero
  %p16 = bitcast i32* %p to i16*
  %gep1 = getelementptr i32, i32* %p, i32 %addend
  %gep2 = getelementptr i16, i16* %p16, i32 %add
  ret void
}

; CHECK-LABEL: Function: add_non_zero_different_sizes
; CHECK: NoAlias: i16* %gep1.16, i32* %gep2
; CHECK: NoAlias: i16* %gep2.16, i32* %gep1
; CHECK: NoAlias: i16* %gep1.16, i16* %gep2.16
; CHECK: MayAlias: i32* %gep2, i64* %gep1.64
; CHECK: MayAlias: i16* %gep2.16, i64* %gep1.64
; CHECK: MayAlias: i32* %gep1, i64* %gep2.64
; CHECK: MayAlias: i16* %gep1.16, i64* %gep2.64
; CHECK: MayAlias: i64* %gep1.64, i64* %gep2.64
define void @add_non_zero_different_sizes(i32* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add i32 %addend, %knownnonzero
  %gep1 = getelementptr i32, i32* %p, i32 %addend
  %gep2 = getelementptr i32, i32* %p, i32 %add
  %gep1.16 = bitcast i32* %gep1 to i16*
  %gep2.16 = bitcast i32* %gep2 to i16*
  %gep1.64 = bitcast i32* %gep1 to i64*
  %gep2.64 = bitcast i32* %gep2 to i64*
  ret void
}


; CHECK-LABEL: add_non_zero_with_offset
; MayAlias: i32* %gep1, i32* %gep2
; NoAlias: i16* %gep1.16, i16* %gep2.16
define void @add_non_zero_with_offset(i32* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add i32 %addend, %knownnonzero
  %p.8 = bitcast i32* %p to i8*
  %p.off.8 = getelementptr i8, i8* %p.8, i32 2
  %p.off = bitcast i8* %p.off.8 to i32*
  %gep1 = getelementptr i32, i32* %p.off, i32 %addend
  %gep2 = getelementptr i32, i32* %p, i32 %add
  %gep1.16 = bitcast i32* %gep1 to i16*
  %gep2.16 = bitcast i32* %gep2 to i16*
  ret void
}

; CHECK-LABEL: Function: add_non_zero_assume
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @add_non_zero_assume(i32* %p, i32 %addend, i32 %knownnonzero) {
  %cmp = icmp ne i32 %knownnonzero, 0
  call void @llvm.assume(i1 %cmp)
  %add = add i32 %addend, %knownnonzero
  %gep1 = getelementptr i32, i32* %p, i32 %addend
  %gep2 = getelementptr i32, i32* %p, i32 %add
  ret void
}

; CHECK-LABEL: non_zero_index_simple
; CHECK: NoAlias: i32* %gep, i32* %p
; CHECK: NoAlias: i16* %gep.16, i32* %p
; CHECK: MayAlias: i32* %p, i64* %gep.64
define void @non_zero_index_simple(i32* %p, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %gep = getelementptr i32, i32* %p, i32 %knownnonzero
  %gep.16 = bitcast i32* %gep to i16*
  %gep.64 = bitcast i32* %gep to i64*
  ret void
}

; CHECK-LABEL: non_zero_index_with_offset
; CHECK: NoAlias: i32* %gep, i32* %p
; CHECK: NoAlias: i16* %gep.16, i32* %p
define void @non_zero_index_with_offset(i32* %p, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %p.8 = bitcast i32* %p to i8*
  %p.off.8 = getelementptr i8, i8* %p.8, i32 2
  %p.off = bitcast i8* %p.off.8 to i32*
  %gep = getelementptr i32, i32* %p.off, i32 %knownnonzero
  %gep.16 = bitcast i32* %gep to i16*
  ret void
}

; CHECK-LABEL: non_zero_index_assume
; CHECK: NoAlias: i32* %gep, i32* %p
define void @non_zero_index_assume(i32* %p, i32 %knownnonzero) {
  %cmp = icmp ne i32 %knownnonzero, 0
  call void @llvm.assume(i1 %cmp)
  %gep = getelementptr i32, i32* %p, i32 %knownnonzero
  ret void
}

declare void @llvm.assume(i1)

!0 = !{ i32 1, i32 0 }
