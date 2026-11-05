// RUN: llvm-tblgen -gen-intrinsic-feature-checks \
// RUN:   -intrinsic-feature-prefix=test -intrinsic-feature-namespace=TestTarget \
// RUN:   -I %p/../../include %s | FileCheck %s

// Check the instantiated result and input types for each guarded overload.
// CHECK-DAG: MVT::f32, MVT::v4i32, MVT::v4i1, MVT::v2f32, MVT::v8f16, MVT::v4f32, // int_test_dependent_float
// CHECK-DAG: MVT::i32, MVT::i32, MVT::i16, MVT::i64, MVT::i16, // int_test_dependent_scalar
// CHECK-DAG: MVT::nxv4i32, MVT::nxv2f32, MVT::nxv8f16, MVT::nxv4f32, // int_test_dependent_scalable
// CHECK-DAG: MVT::v2i64, MVT::i32, MVT::v4i32, MVT::v4i64, // int_test_dependent_second_overload
// CHECK-DAG: MVT::i32, MVT::v2i32, MVT::v8i16, MVT::v16i8, MVT::v4i32, MVT::v4i64, MVT::v4i16, // int_test_dependent_vector

include "llvm/IR/Intrinsics.td"
include "llvm/Target/TargetSubtarget.td"

def FeatureExample : SubtargetFeature<"example", "Example", "true", "">;

def llvm_test_nxv4f32_ty : LLVMType<nxv4f32>;

let TargetPrefix = "test" in {
  def int_test_dependent_scalar
      : GuardedDefaultIntrinsic<
            [llvm_anyint_ty, LLVMMatchType<0>,
             LLVMScalarOrSameVectorWidth<0, llvm_i16_ty>],
            [LLVMExtendedType<0>, LLVMTruncatedType<0>]> {
    let TargetGuards = [GuardedVariant<["example"], [[llvm_i32_ty]]>];
  }

  def int_test_dependent_vector
      : GuardedDefaultIntrinsic<
            [LLVMVectorElementType<0>, LLVMOneNthElementsVectorType<0, 2>,
             LLVMSubdivide2VectorType<0>, LLVMSubdivide4VectorType<0>],
            [llvm_anyvector_ty, LLVMExtendedType<0>, LLVMTruncatedType<0>]> {
    let TargetGuards = [GuardedVariant<["example"], [[llvm_v4i32_ty]]>];
  }

  def int_test_dependent_float
      : GuardedDefaultIntrinsic<
            [LLVMVectorElementType<0>, LLVMVectorOfBitcastsToInt<0>,
             LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
             LLVMOneNthElementsVectorType<0, 2>,
             LLVMSubdivide2VectorType<0>],
            [llvm_anyvector_ty]> {
    let TargetGuards = [GuardedVariant<["example"], [[llvm_v4f32_ty]]>];
  }

  def int_test_dependent_scalable
      : GuardedDefaultIntrinsic<
            [LLVMVectorOfBitcastsToInt<0>,
             LLVMOneNthElementsVectorType<0, 2>,
             LLVMSubdivide2VectorType<0>],
            [llvm_anyvector_ty]> {
    let TargetGuards =
        [GuardedVariant<["example"], [[llvm_test_nxv4f32_ty]]>];
  }

  def int_test_dependent_second_overload
      : GuardedDefaultIntrinsic<
            [llvm_anyvector_ty, LLVMVectorElementType<1>],
            [llvm_anyvector_ty, LLVMExtendedType<1>]> {
    let TargetGuards = [GuardedVariant<["example"],
                                       [[llvm_v2i64_ty, llvm_v4i32_ty]]>];
  }
}
