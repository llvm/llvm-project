; RUN: not --crash opt -passes='lower-matrix-intrinsics' -verify-matrix-shapes=true -S %s 2>&1 | FileCheck --check-prefix=VERIFY %s
; RUN: opt -passes='lower-matrix-intrinsics' -verify-matrix-shapes=false -S %s 2>&1 | FileCheck --check-prefix=NOVERIFY %s

; VERIFY: Conflicting shapes (6x1 vs 1x6)
; NOVERIFY-NOT: Conflicting shapes

define <1 x float> @intrinsic_column_major_load_dot_product_float_v6(ptr %lhs_address, ptr %rhs_address) {
entry:
  %lhs = tail call fast <6 x float> @llvm.matrix.column.major.load.v6f32.i64(ptr nonnull align 4 %lhs_address, i64 6, i1 false, i32 6, i32 1)
  %rhs = tail call fast <6 x float> @llvm.matrix.column.major.load.v6f32.i64(ptr nonnull align 4 %rhs_address, i64 1, i1 false, i32 1, i32 6)
  %result = tail call fast <1 x float> @llvm.matrix.multiply.v1f32.v6f32.v6f32(<6 x float> %lhs, <6 x float> %rhs, i32 1, i32 6, i32 1)
  ret <1 x float> %result
}

declare <6 x float> @llvm.matrix.column.major.load.v6f32.i64(ptr nonnull align 4, i64, i1, i32, i32)
declare <1 x float> @llvm.matrix.multiply.v1f32.v6f32.v6f32(<6 x float>, <6 x float>, i32, i32, i32)
