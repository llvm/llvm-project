; RUN: not opt -verify -S < %s 2>&1 >/dev/null | FileCheck %s

; CHECK: experimental_vector_insert parameters must have the same element type.
define <vscale x 16 x i8> @invalid_mismatched_element_types(<vscale x 16 x i8> %vec, <4 x i16> %subvec) nounwind {
  %retval = call <vscale x 16 x i8> @llvm.experimental.vector.insert.nxv16i8.v4i16(<vscale x 16 x i8> %vec, <4 x i16> %subvec, i64 0)
  ret <vscale x 16 x i8> %retval
}

declare <vscale x 16 x i8> @llvm.experimental.vector.insert.nxv16i8.v4i16(<vscale x 16 x i8>, <4 x i16>, i64)
