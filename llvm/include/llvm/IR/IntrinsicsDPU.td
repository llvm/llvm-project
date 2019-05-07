//===- IntrinsicsDPU.td - Defines DPU intrinsics ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the DPU-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "dpu" in {

def int_dpu_tid : Intrinsic<[llvm_anyint_ty], [], [IntrNoMem]>;

}
