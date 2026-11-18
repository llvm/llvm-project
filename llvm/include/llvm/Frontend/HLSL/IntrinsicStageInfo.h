//===- IntrinsicStageInfo.h - Intrinsic stage availability -------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_INTRINSICSTAGEINFO_H
#define LLVM_FRONTEND_HLSL_INTRINSICSTAGEINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Compiler.h"
#include <string>

namespace llvm::hlsl {

struct IntrinsicStageInfo {
  Intrinsic::ID ID;
  StringRef BuiltinName;
  StringRef Name;
  ArrayRef<StringRef> Stages;

  LLVM_ABI bool isSupportedIn(StringRef Stage) const;
  LLVM_ABI std::string getStageNames() const;
};

LLVM_ABI ArrayRef<IntrinsicStageInfo> getDXILIntrinsicStageInfos();
LLVM_ABI const IntrinsicStageInfo *
getDXILIntrinsicStageInfo(StringRef BuiltinName);

} // namespace llvm::hlsl

#endif
