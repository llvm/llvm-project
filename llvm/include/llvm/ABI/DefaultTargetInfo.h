//===----- DefaultTargetInfo.h - Default ABI classification ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Default ABI classification shared by targets without special rules. This is
/// the llvm::abi analogue of clang::CodeGen::DefaultABIInfo: targets delegate
/// to it for the cases they do not handle themselves.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABI_DEFAULTTARGETINFO_H
#define LLVM_ABI_DEFAULTTARGETINFO_H

#include "llvm/ABI/TargetInfo.h"

namespace llvm {
namespace abi {

/// Self-consistent classification that conforms to no particular ABI.
class LLVM_ABI DefaultTargetInfo : public TargetInfo {
public:
  using TargetInfo::TargetInfo;

  ArgInfo classifyArgumentType(const Type *Ty) const;
  ArgInfo classifyReturnType(const Type *RetTy) const;

  void computeInfo(FunctionInfo &FI) const override;
};

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_DEFAULTTARGETINFO_H
