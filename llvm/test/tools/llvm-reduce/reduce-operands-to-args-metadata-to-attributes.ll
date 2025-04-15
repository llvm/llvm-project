; Check that equivalent parameter attributes are introduced when
; moving instructions with metadata to arguments.

; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck --input-file %t --check-prefix=REDUCED %s

; INTERESTING-LABEL: define ptr @use_nonnull(
; REDUCED-LABEL: define ptr @use_nonnull(ptr nonnull %nonnull) {
define ptr @use_nonnull() {
  %nonnull = load ptr, ptr null, !nonnull !0
  ret ptr %nonnull
}

; INTERESTING-LABEL: define void @use_noundef(
; REDUCED-LABEL: define void @use_noundef(ptr noundef %noundef, <2 x ptr> noundef %noundef_vec) {
define void @use_noundef() {
  %noundef = load ptr, ptr null, !noundef !0
  %noundef_vec = load <2 x ptr>, ptr null, !noundef !0
  store ptr %noundef, ptr null
  store <2 x ptr> %noundef_vec, ptr null
  ret void
}

; INTERESTING-LABEL: define ptr @use_align(
; REDUCED-LABEL: define ptr @use_align(ptr align 16 %align) {
define ptr @use_align() {
  %align = load ptr, ptr null, !align !1
  ret ptr %align
}

; INTERESTING-LABEL: define ptr @use_dereferenceable(
; REDUCED-LABEL: define ptr @use_dereferenceable(ptr dereferenceable(12345) %deref) {
define ptr @use_dereferenceable() {
  %deref = load ptr, ptr null, !dereferenceable !2
  ret ptr %deref
}

; INTERESTING-LABEL: define ptr @use_dereferenceable_or_null(
; REDUCED-LABEL: define ptr @use_dereferenceable_or_null(ptr dereferenceable(77777) %deref) {
define ptr @use_dereferenceable_or_null() {
  %deref = load ptr, ptr null, !dereferenceable_or_null !3
  ret ptr %deref
}

; INTERESTING-LABEL: define void @use_range(
; REDUCED-LABEL: define void @use_range(i32 range(i32 8, 25) %simple_range, i32 range(i32 8, 420) %disjoint_range, i32 range(i32 42, 0) %wrapping_range, <2 x i32> range(i32 8, 25) %vector_range) {
define void @use_range() {
  %simple_range = load i32, ptr null, !range !4
  %disjoint_range = load i32, ptr null, !range !5
  %wrapping_range = load i32, ptr null, !range !6
  %vector_range = load <2 x i32>, ptr null, !range !4
  store i32 %simple_range, ptr null
  store i32 %disjoint_range, ptr null
  store i32 %wrapping_range, ptr null
  store <2 x i32> %vector_range, ptr null
  ret void
}

; INTERESTING-LABEL: define void @use_noundef_range(
; REDUCED-LABEL: define void @use_noundef_range(i32 noundef range(i32 8, 25) %load, <2 x i32> noundef range(i32 8, 25) %load_vec) {
define void @use_noundef_range() {
  %load = load i32, ptr null, !range !4, !noundef !0
  %load_vec = load <2 x i32>, ptr null, !range !4, !noundef !0
  store i32 %load, ptr null
  store <2 x i32> %load_vec, ptr null
  ret void
}



!0 = !{}
!1 = !{i64 16}
!2 = !{i64 12345}
!3 = !{i64 77777}
!4 = !{i32 8, i32 25}
!5 = !{i32 8, i32 25, i32 69, i32 420}
!6 = !{i32 42, i32 0}
