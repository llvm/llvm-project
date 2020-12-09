; RUN: not opt -verify -S < %s 2>&1 >/dev/null | FileCheck %s

; CHECK: experimental_vector_extract result must have the same element type as the input vector.
define <16 x i16> @invalid_mismatched_element_types(<vscale x 16 x i8> %vec) nounwind {
  %retval = call <16 x i16> @llvm.experimental.vector.extract.v16i16.nxv16i8(<vscale x 16 x i8> %vec, i64 0)
  ret <16 x i16> %retval
}

declare <16 x i16> @llvm.experimental.vector.extract.v16i16.nxv16i8(<vscale x 16 x i8>, i64)
