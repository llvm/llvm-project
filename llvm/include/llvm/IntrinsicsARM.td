//===- IntrinsicsARM.td - Defines ARM intrinsics -----------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines all of the ARM-specific intrinsics.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// TLS

let TargetPrefix = "arm" in {  // All intrinsics start with "llvm.arm.".
  def int_arm_thread_pointer : GCCBuiltin<"__builtin_thread_pointer">,
              Intrinsic<[llvm_ptr_ty], [], [IntrNoMem]>;
}
