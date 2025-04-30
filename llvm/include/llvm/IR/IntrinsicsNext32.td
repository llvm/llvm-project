//===- IntrinsicsNext32.td - Defines Next32 intrinsics -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Next32-specific intrinsics.
//
//===----------------------------------------------------------------------===//

// Specialized multiplication instructions
let TargetPrefix = "next32" in { // All intrinsics start with "llvm.next32."
  def int_next32_umul8 : ClangBuiltin<"__builtin_next32_umul8">,
                         Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                                   [Commutative, IntrNoMem]>;
  def int_next32_umul16 : ClangBuiltin<"__builtin_next32_umul16">,
                          Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                                    [Commutative, IntrNoMem]>;
  def int_next32_smul8 : ClangBuiltin<"__builtin_next32_smul8">,
                         Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                                   [Commutative, IntrNoMem]>;
  def int_next32_smul16 : ClangBuiltin<"__builtin_next32_smul16">,
                          Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                                    [Commutative, IntrNoMem]>;
}
