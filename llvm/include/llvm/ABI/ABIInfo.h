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
#include <climits>
#include <cstdint>

namespace llvm {
namespace abi {

struct ABICompatInfo {
  unsigned Version = UINT_MAX;

  struct ABIFlags {
    bool PassInt128VectorsInMem : 1;
    bool ReturnCXXRecordGreaterThan128InMem : 1;
    bool ClassifyIntegerMMXAsSSE : 1;
    bool HonorsRevision98 : 1;

    ABIFlags()
        : PassInt128VectorsInMem(true),
          ReturnCXXRecordGreaterThan128InMem(true),
          ClassifyIntegerMMXAsSSE(true), HonorsRevision98(true) {}

  } Flags;

  ABICompatInfo() : Version(UINT_MAX) {}
  ABICompatInfo(unsigned Ver) : Version(Ver) {}
};

/// Abstract base class for target-specific ABI information.
class ABIInfo {
private:
  ABICompatInfo CompatInfo;

public:
  ABIInfo() : CompatInfo() {}
  explicit ABIInfo(const ABICompatInfo &Info) : CompatInfo(Info) {}

  virtual ~ABIInfo() = default;

  virtual ABIArgInfo classifyReturnType(const Type *RetTy) const = 0;
  virtual ABIArgInfo classifyArgumentType(const Type *ArgTy) const = 0;
  virtual void computeInfo(ABIFunctionInfo &FI) const = 0;
  virtual bool isPassByRef(const Type *Ty) const { return false; }
  const ABICompatInfo &getABICompatInfo() const { return CompatInfo; }

  void setABICompatInfo(const struct ABICompatInfo &Info) { CompatInfo = Info; }
};

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_ABIINFO_H
