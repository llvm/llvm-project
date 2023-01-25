; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK: Function: t1
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @t1(ptr %p, i32 %addend, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], ptr %p, i32 2, i32 %addend
  %gep2 = getelementptr [8 x i32], ptr %p, i32 2, i32 %add
  load i32, ptr %gep1
  load i32, ptr %gep2
  ret void
}

; CHECK: Function: t2
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t2(ptr %p, i32 %addend, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], ptr %p, i32 1, i32 %addend
  %gep2 = getelementptr [8 x i32], ptr %p, i32 0, i32 %add
  load i32, ptr %gep1
  load i32, ptr %gep2
  ret void
}

; CHECK: Function: t3
; CHECK: MustAlias: i32* %gep1, i32* %gep2
define void @t3(ptr %p, i32 %addend, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], ptr %p, i32 0, i32 %add
  %gep2 = getelementptr [8 x i32], ptr %p, i32 0, i32 %add
  load i32, ptr %gep1
  load i32, ptr %gep2
  ret void
}

; CHECK: Function: t4
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t4(ptr %p, i32 %addend, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], ptr %p, i32 1, i32 %addend
  %gep2 = getelementptr [8 x i32], ptr %p, i32 %add, i32 %add
  load i32, ptr %gep1
  load i32, ptr %gep2
  ret void
}

; CHECK: Function: t5
; CHECK: MayAlias: i64* %gep1, i32* %gep2
define void @t5(ptr %p, i32 %addend, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], ptr %p, i32 2, i32 %addend
  %gep2 = getelementptr [8 x i32], ptr %p, i32 2, i32 %add
  load i32, ptr %gep2
  load i64, ptr %gep1
  ret void
}

; CHECK-LABEL: Function: add_non_zero_simple
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @add_non_zero_simple(ptr %p, i32 %addend, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %add = add i32 %addend, %knownnonzero
  %gep1 = getelementptr i32, ptr %p, i32 %addend
  %gep2 = getelementptr i32, ptr %p, i32 %add
  load i32, ptr %gep1
  load i32, ptr %gep2
  ret void
}

; CHECK-LABEL: Function: add_non_zero_different_scales
; CHECK: MayAlias: i32* %gep1, i16* %gep2
define void @add_non_zero_different_scales(ptr %p, i32 %addend, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %add = add i32 %addend, %knownnonzero
  %gep1 = getelementptr i32, ptr %p, i32 %addend
  %gep2 = getelementptr i16, ptr %p, i32 %add
  load i32, ptr %gep1
  load i16, ptr %gep2
  ret void
}

; CHECK-LABEL: Function: add_non_zero_different_sizes
; CHECK: NoAlias: i16* %gep1, i32* %gep2
; CHECK: NoAlias: i32* %gep1, i16* %gep2
; CHECK: NoAlias: i16* %gep1, i16* %gep2
; CHECK: MayAlias: i64* %gep1, i32* %gep2
; CHECK: MayAlias: i64* %gep1, i16* %gep2
; CHECK: MayAlias: i32* %gep1, i64* %gep2
; CHECK: MayAlias: i16* %gep1, i64* %gep2
; CHECK: MayAlias: i64* %gep1, i64* %gep2
define void @add_non_zero_different_sizes(ptr %p, i32 %addend, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %add = add i32 %addend, %knownnonzero
  %gep1 = getelementptr i32, ptr %p, i32 %addend
  %gep2 = getelementptr i32, ptr %p, i32 %add
  load i32, ptr %gep1
  load i32, ptr %gep2
  load i16, ptr %gep1
  load i16, ptr %gep2
  load i64, ptr %gep1
  load i64, ptr %gep2
  ret void
}


; CHECK-LABEL: add_non_zero_with_offset
; MayAlias: ptr %gep1, ptr %gep2
; NoAlias: ptr %gep1, ptr %gep2
define void @add_non_zero_with_offset(ptr %p, i32 %addend, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %add = add i32 %addend, %knownnonzero
  %p.off.8 = getelementptr i8, ptr %p, i32 2
  %gep1 = getelementptr i32, ptr %p.off.8, i32 %addend
  %gep2 = getelementptr i32, ptr %p, i32 %add
  load i32, ptr %gep1
  load i32, ptr %gep2
  load i16, ptr %gep1
  load i16, ptr %gep2
  ret void
}

; CHECK-LABEL: Function: add_non_zero_assume
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @add_non_zero_assume(ptr %p, i32 %addend, i32 %knownnonzero) {
  %cmp = icmp ne i32 %knownnonzero, 0
  call void @llvm.assume(i1 %cmp)
  %add = add i32 %addend, %knownnonzero
  %gep1 = getelementptr i32, ptr %p, i32 %addend
  %gep2 = getelementptr i32, ptr %p, i32 %add
  load i32, ptr %gep1
  load i32, ptr %gep2
  ret void
}

; CHECK-LABEL: non_zero_index_simple
; CHECK: NoAlias: i32* %gep, i32* %p
; CHECK: NoAlias: i16* %gep, i32* %p
; CHECK: MayAlias: i64* %gep, i32* %p
define void @non_zero_index_simple(ptr %p, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %gep = getelementptr i32, ptr %p, i32 %knownnonzero
  load i32, ptr %p
  load i32, ptr %gep
  load i16, ptr %gep
  load i64, ptr %gep
  ret void
}

; CHECK-LABEL: non_zero_index_with_offset
; CHECK: MayAlias: i32* %gep, i32* %p
; CHECK: NoAlias: i16* %gep, i32* %p
define void @non_zero_index_with_offset(ptr %p, ptr %q) {
  %knownnonzero = load i32, ptr %q, !range !0
  %p.off.8 = getelementptr i8, ptr %p, i32 2
  %gep = getelementptr i32, ptr %p.off.8, i32 %knownnonzero
  load i32, ptr %p
  load i32, ptr %gep
  load i16, ptr %gep
  ret void
}

; CHECK-LABEL: non_zero_index_assume
; CHECK: NoAlias: i32* %gep, i32* %p
define void @non_zero_index_assume(ptr %p, i32 %knownnonzero) {
  %cmp = icmp ne i32 %knownnonzero, 0
  call void @llvm.assume(i1 %cmp)
  %gep = getelementptr i32, ptr %p, i32 %knownnonzero
  load i32, ptr %p
  load i32, ptr %gep
  ret void
}

declare void @llvm.assume(i1)

!0 = !{ i32 1, i32 0 }
