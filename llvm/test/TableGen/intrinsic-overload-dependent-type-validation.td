// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST0 2>&1 | FileCheck %s -DFILE=%s --check-prefix=CHECK-TEST0
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST1 2>&1 | FileCheck %s -DFILE=%s --check-prefix=CHECK-TEST1
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST2 2>&1 | FileCheck %s -DFILE=%s --check-prefix=CHECK-TEST2
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST3 2>&1 | FileCheck %s -DFILE=%s --check-prefix=CHECK-TEST3
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s -DTEST4 2>&1 | FileCheck %s -DFILE=%s --check-prefix=CHECK-TEST4

// This unit test tests various overloaded intrinsics that use dependent
// types to validate that if they reference an overload type is not one of the
// llvm_any*, it results in an error.

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

#ifdef TEST0

def int_test: DefaultAttrsIntrinsic<
    [ llvm_anyvector_ty],               // overload index 0.
    [ LLVMVectorOfAnyPointersToElt<0>,  // overload index 1.
      llvm_anyvector_ty,                // overload index 2.
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
// CHECK-TEST0: [[FILE]]:[[@LINE+3]]:7: error: for intrinsic int_test overload index 1 is invalid, dependent types must reference an overload index of an 'llvm_any' type
      // Error: Fully dependent type references overload index 1,
      // which is not one of the llvm_any-types.
      LLVMMatchType<1>,
      LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>,
      LLVMVectorOfAnyPointersToElt<1>,  // overload index 3.
      llvm_i32_ty]>;
#endif // TEST0

#ifdef TEST1

def int_test: DefaultAttrsIntrinsic<
    [ llvm_anyvector_ty],               // overload index 0.
    [ LLVMVectorOfAnyPointersToElt<0>,  // overload index 1.
      llvm_anyvector_ty,                // overload index 2.
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, 
      LLVMMatchType<0>,
// CHECK-TEST1: [[FILE]]:[[@LINE+3]]:7: error: for intrinsic int_test overload index 1 is invalid, dependent types must reference an overload index of an 'llvm_any' type
      // Error: Fully dependent type references overload index 1,
      // which is not one of the llvm_any-types.
      LLVMScalarOrSameVectorWidth<1, llvm_i1_ty>,
      LLVMVectorOfAnyPointersToElt<1>,  // overload index 3.
      llvm_i32_ty]>;
#endif // TEST1

#ifdef TEST2

def int_test: DefaultAttrsIntrinsic<
    [ llvm_anyvector_ty],               // overload index 0.
    [ LLVMVectorOfAnyPointersToElt<0>,  // overload index 1.
      llvm_anyvector_ty,                // overload index 2.
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, 
      LLVMMatchType<0>,
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
// CHECK-TEST2: [[FILE]]:[[@LINE+3]]:7: error: for intrinsic int_test overload index 1 is invalid, dependent types must reference an overload index of an 'llvm_any' type
      // Error: Partially dependent type references overload index 1,
      // which is not one of the llvm_any-types.
      LLVMVectorOfAnyPointersToElt<1>,  // overload index 3.
      llvm_i32_ty]>;
#endif // TEST2

#ifdef TEST3

def int_test: DefaultAttrsIntrinsic<
    [ llvm_anyvector_ty],               // overload index 0.
    [ LLVMVectorOfAnyPointersToElt<0>,  // overload index 1.
      llvm_anyvector_ty,                // overload index 2.
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>, 
      LLVMMatchType<0>,
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
// CHECK-TEST3: [[FILE]]:[[@LINE+2]]:7: error: for intrinsic int_test overload index 3 is invalid, dependent types must reference an overload index of an 'llvm_any' type
      // Error: Partially dependent type references overload index 3 (itself).
      LLVMVectorOfAnyPointersToElt<3>,  // overload index 3.
      llvm_i32_ty]>;
#endif // TEST3

#ifdef TEST4

def int_test: DefaultAttrsIntrinsic<
    [ llvm_anyvector_ty],               // overload index 0.
    [ LLVMVectorOfAnyPointersToElt<0>,  // overload index 1.
      llvm_anyvector_ty,                // overload index 2.
// CHECK-TEST4: [[FILE]]:[[@LINE+2]]:7: error: for intrinsic int_test overload index 10 is invalid, intrinsic only has 3 overloaded types
      // Error: Overload index out of bounds.
      LLVMScalarOrSameVectorWidth<10, llvm_i1_ty>,
      LLVMMatchType<0>,
      LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
      llvm_i32_ty]>;
#endif // TEST4
