//===- TargetCodeGenInfo.cpp - Target CodeGen Info Implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ABI/TargetCodegenInfo.h"
#include "llvm/ABI/ABIFunctionInfo.h"

namespace llvm::abi {

TargetCodeGenInfo::TargetCodeGenInfo(std::unique_ptr<llvm::abi::ABIInfo> Info)
    : Info(std::move(Info)) {}

TargetCodeGenInfo::~TargetCodeGenInfo() = default;

void TargetCodeGenInfo::computeInfo(ABIFunctionInfo &FI) const {
  // Default Impl here
}

} // namespace llvm::abi
