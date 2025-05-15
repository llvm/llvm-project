//===- BPF.h - BPF ABI Implementation ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABI_TARGETS_BPF_H
#define LLVM_ABI_TARGETS_BPF_H

#include "llvm/ABI/ABIInfo.h"
#include "llvm/ABI/Types.h"
#include <memory>

namespace llvm::abi {

class TypeBuilder;

/// Factory function to create a BPF ABI implementation
std::unique_ptr<ABIInfo> createBPFABIInfo(TypeBuilder &TB);

} // namespace llvm::abi

#endif // LLVM_ABI_TARGETS_BPF_H
