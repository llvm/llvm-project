//===- IntrinsicsAlpha.td - Defines Alpha intrinsics -------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines all of the Alpha-specific intrinsics.
//
//===----------------------------------------------------------------------===//


let TargetPrefix = "alpha" in {  // All intrinsics start with "llvm.alpha.".
  def int_alpha_umulh : GCCBuiltin<"__builtin_alpha_umulh">,
              Intrinsic<[llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty], [IntrNoMem]>;
}
