//==- IntrinsicsXCore.td - XCore intrinsics                 -*- tablegen -*-==//
// 
// Copyright (C) 2008 XMOS
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the XCore-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "xcore" in {  // All intrinsics start with "llvm.xcore.".
  def int_xcore_bitrev : Intrinsic<[llvm_i32_ty],[llvm_i32_ty],[IntrNoMem]>;
  def int_xcore_getid : Intrinsic<[llvm_i32_ty],[],[IntrNoMem]>;
}
