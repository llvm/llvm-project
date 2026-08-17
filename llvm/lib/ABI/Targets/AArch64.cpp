//===- AArch64.cpp - AArch64 ABI Implementation ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace abi {

class AArch64TargetInfo : public TargetInfo {
public:
  AArch64TargetInfo(TypeBuilder &TB, AArch64ABIKind Kind)
      : TB(TB), Kind(Kind) {}

  void computeInfo(FunctionInfo &FI) const override {
    FI.getReturnInfo() = ArgInfo::getDirect();
    for (auto &I : FI.arguments())
      I.Info = ArgInfo::getDirect();
  }

private:
  [[maybe_unused]] TypeBuilder &TB;
  [[maybe_unused]] AArch64ABIKind Kind;
};

std::unique_ptr<TargetInfo> createAArch64TargetInfo(TypeBuilder &TB,
                                                    AArch64ABIKind Kind) {
  if (Kind == AArch64ABIKind::Win64)
    llvm_unreachable("Win64 ABI not supported yet");
  return std::make_unique<AArch64TargetInfo>(TB, Kind);
}

} // namespace abi
} // namespace llvm
