; RUN: not opt -verify -S < %s 2>&1 >/dev/null | FileCheck %s

;
; Test that extractions/insertion indices are validated.
;

; CHECK: vector_extract index must be a constant multiple of the result type's known minimum vector length.
define <4 x i32> @extract_idx_not_constant_multiple(<8 x i32> %vec) {
  %1 = call <4 x i32> @llvm.vector.extract.v4i32.v8i32(<8 x i32> %vec, i64 1)
  ret <4 x i32> %1
}

; CHECK: vector_insert index must be a constant multiple of the subvector's known minimum vector length.
define <8 x i32> @insert_idx_not_constant_multiple(<8 x i32> %vec, <4 x i32> %subvec) {
  %1 = call <8 x i32> @llvm.vector.insert.v8i32.v4i32(<8 x i32> %vec, <4 x i32> %subvec, i64 2)
  ret <8 x i32> %1
}

;
; Test that extractions/insertion element types are validated.
;

; CHECK: vector_extract result must have the same element type as the input vector.
define <16 x i16> @extract_invalid_mismatched_element_types(<vscale x 16 x i8> %vec) nounwind {
  %retval = call <16 x i16> @llvm.vector.extract.v16i16.nxv16i8(<vscale x 16 x i8> %vec, i64 0)
  ret <16 x i16> %retval
}

; CHECK: vector_insert parameters must have the same element type.
define <vscale x 16 x i8> @insert_invalid_mismatched_element_types(<vscale x 16 x i8> %vec, <4 x i16> %subvec) nounwind {
  %retval = call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v4i16(<vscale x 16 x i8> %vec, <4 x i16> %subvec, i64 0)
  ret <vscale x 16 x i8> %retval
}

;
; Test that extractions/insertions which 'overrun' are captured.
;

; CHECK: vector_extract would overrun.
define <3 x i32> @extract_overrun_fixed_fixed(<8 x i32> %vec) {
  %1 = call <3 x i32> @llvm.vector.extract.v8i32.v3i32(<8 x i32> %vec, i64 6)
  ret <3 x i32> %1
}

; CHECK: vector_extract would overrun.
define <vscale x 3 x i32> @extract_overrun_scalable_scalable(<vscale x 8 x i32> %vec) {
  %1 = call <vscale x 3 x i32> @llvm.vector.extract.nxv8i32.nxv3i32(<vscale x 8 x i32> %vec, i64 6)
  ret <vscale x 3 x i32> %1
}

; We cannot statically check whether or not an extraction of a fixed vector
; from a scalable vector would overrun, because we can't compare the sizes of
; the two. Therefore, this function should not raise verifier errors.
; CHECK-NOT: vector_extract
define <3 x i32> @extract_overrun_scalable_fixed(<vscale x 8 x i32> %vec) {
  %1 = call <3 x i32> @llvm.vector.extract.nxv8i32.v3i32(<vscale x 8 x i32> %vec, i64 6)
  ret <3 x i32> %1
}

; CHECK: subvector operand of vector_insert would overrun the vector being inserted into.
define <8 x i32> @insert_overrun_fixed_fixed(<8 x i32> %vec, <3 x i32> %subvec) {
  %1 = call <8 x i32> @llvm.vector.insert.v8i32.v3i32(<8 x i32> %vec, <3 x i32> %subvec, i64 6)
  ret <8 x i32> %1
}

; CHECK: subvector operand of vector_insert would overrun the vector being inserted into.
define <vscale x 8 x i32> @insert_overrun_scalable_scalable(<vscale x 8 x i32> %vec, <vscale x 3 x i32> %subvec) {
  %1 = call <vscale x 8 x i32> @llvm.vector.insert.nxv8i32.nxv3i32(<vscale x 8 x i32> %vec, <vscale x 3 x i32> %subvec, i64 6)
  ret <vscale x 8 x i32> %1
}

; We cannot statically check whether or not an insertion of a fixed vector into
; a scalable vector would overrun, because we can't compare the sizes of the
; two. Therefore, this function should not raise verifier errors.
; CHECK-NOT: vector_insert
define <vscale x 8 x i32> @insert_overrun_scalable_fixed(<vscale x 8 x i32> %vec, <3 x i32> %subvec) {
  %1 = call <vscale x 8 x i32> @llvm.vector.insert.nxv8i32.v3i32(<vscale x 8 x i32> %vec, <3 x i32> %subvec, i64 6)
  ret <vscale x 8 x i32> %1
}

declare <vscale x 3 x i32> @llvm.vector.extract.nxv8i32.nxv3i32(<vscale x 8 x i32>, i64)
declare <vscale x 8 x i32> @llvm.vector.insert.nxv8i32.nxv3i32(<vscale x 8 x i32>, <vscale x 3 x i32>, i64)
declare <vscale x 8 x i32> @llvm.vector.insert.nxv8i32.v3i32(<vscale x 8 x i32>, <3 x i32>, i64)
declare <3 x i32> @llvm.vector.extract.nxv8i32.v3i32(<vscale x 8 x i32>, i64)
declare <3 x i32> @llvm.vector.extract.v8i32.v3i32(<8 x i32>, i64)
declare <4 x i32> @llvm.vector.extract.v4i32.v8i32(<8 x i32>, i64)
declare <8 x i32> @llvm.vector.insert.v8i32.v3i32(<8 x i32>, <3 x i32>, i64)
declare <8 x i32> @llvm.vector.insert.v8i32.v4i32(<8 x i32>, <4 x i32>, i64)
declare <16 x i16> @llvm.vector.extract.v16i16.nxv16i8(<vscale x 16 x i8>, i64)
declare <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v4i16(<vscale x 16 x i8>, <4 x i16>, i64)
