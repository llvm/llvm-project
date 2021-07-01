//===- ObjCARCUtil.h - ObjC ARC Utility Functions ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines ARC utility functions which are used by various parts of
/// the compiler.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_OBJCARCUTIL_H
#define LLVM_IR_OBJCARCUTIL_H

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"

namespace llvm {
namespace objcarc {

inline const char *getRVMarkerModuleFlagStr() {
  return "clang.arc.retainAutoreleasedReturnValueMarker";
}

enum AttachedCallOperandBundle : unsigned { RVOB_Retain, RVOB_Claim };

inline AttachedCallOperandBundle
getAttachedCallOperandBundleEnum(bool IsRetain) {
  return IsRetain ? RVOB_Retain : RVOB_Claim;
}

inline bool hasAttachedCallOpBundle(const CallBase *CB) {
  // Ignore the bundle if the return type is void. Global optimization passes
  // can turn the called function's return type to void. That should happen only
  // if the call doesn't return and the call to @llvm.objc.clang.arc.noop.use
  // no longer consumes the function return or is deleted. In that case, it's
  // not necessary to emit the marker instruction or calls to the ARC runtime
  // functions.
  return !CB->getFunctionType()->getReturnType()->isVoidTy() &&
         CB->getOperandBundle(LLVMContext::OB_clang_arc_attachedcall)
             .hasValue();
}

inline bool hasAttachedCallOpBundle(const CallBase *CB, bool IsRetain) {
  assert(hasAttachedCallOpBundle(CB) &&
         "call doesn't have operand bundle clang_arc_attachedcall");
  auto B = CB->getOperandBundle(LLVMContext::OB_clang_arc_attachedcall);
  if (!B.hasValue())
    return false;
  return cast<ConstantInt>(B->Inputs[0])->getZExtValue() ==
         getAttachedCallOperandBundleEnum(IsRetain);
}

} // end namespace objcarc
} // end namespace llvm

#endif
