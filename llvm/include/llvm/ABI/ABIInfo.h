//===----- ABIInfo.h - ABI information access & encapsulation ----- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// ABI information access & encapsulation
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABI_ABIINFO_H
#define LLVM_ABI_ABIINFO_H

#include "llvm/ABI/ABIFunctionInfo.h"
#include "llvm/ABI/Types.h"
#include <cassert>

namespace llvm {
namespace abi {

/// Abstract base class for target-specific ABI information.
class ABIInfo {
public:
  virtual ~ABIInfo() = default;

  virtual ABIArgInfo classifyReturnType(const Type *RetTy) const = 0;
  virtual ABIArgInfo classifyArgumentType(const Type *ArgTy) const = 0;
  virtual void computeInfo(ABIFunctionInfo &FI) const = 0;
  virtual bool isPassByRef(const Type *Ty) const { return false; }
};

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_ABIINFO_H
