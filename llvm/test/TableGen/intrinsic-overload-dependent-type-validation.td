// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST0 2>&1 | FileCheck %s --check-prefix=CHECK-TEST0
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST1 2>&1 | FileCheck %s --check-prefix=CHECK-TEST1
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST2 2>&1 | FileCheck %s --check-prefix=CHECK-TEST2
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST3 2>&1 | FileCheck %s --check-prefix=CHECK-TEST3

// This unit test tests various overloaded intrinsics that use dependent
// types to validate that if they reference an overload type is not one of the
// llvm_any*, it results in an error.

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

#ifdef TEST0

// CHECK-TEST0: error: assertion failed: dependent types must reference an overload index of an 'llvm_any' type
def int_test: DefaultAttrsIntrinsic<
    [ llvm_anyvector_ty],               // overload index 0.
    [ LLVMVectorOfAnyPointersToElt<0>,  // overload index 1.
      llvm_anyvector_ty,                // overload index 2.
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
      // Error: Fully dependent type references overload index 1,
      // which is not one of the llvm_any-types.
      LLVMMatchType<1>,
      LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>,
      LLVMVectorOfAnyPointersToElt<1>,  // overload index 3.
      llvm_i32_ty]>;
#endif // TEST0

#ifdef TEST1

// CHECK-TEST1: error: assertion failed: dependent types must reference an overload index of an 'llvm_any' type
def int_test: DefaultAttrsIntrinsic<
    [ llvm_anyvector_ty],               // overload index 0.
    [ LLVMVectorOfAnyPointersToElt<0>,  // overload index 1.
      llvm_anyvector_ty,                // overload index 2.
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, 
      LLVMMatchType<0>,
      // Error: Fully dependent type references overload index 1,
      // which is not one of the llvm_any-types.
      LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>,
      LLVMVectorOfAnyPointersToElt<1>,  // overload index 3.
      llvm_i32_ty]>;
#endif // TEST1

#ifdef TEST2

// CHECK-TEST2: error: assertion failed: dependent types must reference an overload index of an 'llvm_any' type
def int_test: DefaultAttrsIntrinsic<
    [ llvm_anyvector_ty],               // overload index 0.
    [ LLVMVectorOfAnyPointersToElt<0>,  // overload index 1.
      llvm_anyvector_ty,                // overload index 2.
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, 
      LLVMMatchType<0>,
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
      // Error: Partially dependent type references overload index 1,
      // which is not one of the llvm_any-types.
      LLVMVectorOfAnyPointersToElt<1>,  // overload index 3.
      llvm_i32_ty]>;
#endif // TEST2

#ifdef TEST3

// CHECK-TEST3: error: assertion failed: dependent types must reference an overload index of an 'llvm_any' type
def int_test: DefaultAttrsIntrinsic<
    [ llvm_anyvector_ty],               // overload index 0.
    [ LLVMVectorOfAnyPointersToElt<0>,  // overload index 1.
      llvm_anyvector_ty,                // overload index 2.
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, 
      LLVMMatchType<0>,
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
      // Error: Partially dependent type references overload index 3 (itself).
      LLVMVectorOfAnyPointersToElt<3>,  // overload index 3.
      llvm_i32_ty]>;
#endif // TEST3
