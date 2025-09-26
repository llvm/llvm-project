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

namespace llvm {
namespace abi {

enum RecordArgABI {
  /// Pass it using the normal C aggregate rules for the ABI, potentially
  /// introducing extra copies and passing some or all of it in registers.
  RAA_Default = 0,

  /// Pass it on the stack using its defined layout.  The argument must be
  /// evaluated directly into the correct stack position in the arguments area,
  /// and the call machinery must not move it or introduce extra copies.
  RAA_DirectInMemory,

  /// Pass it as a pointer to temporary memory.
  RAA_Indirect
};

struct ABICompatInfo {
  unsigned Version = UINT_MAX;

  struct ABIFlags {
    bool PassInt128VectorsInMem : 1;
    bool ReturnCXXRecordGreaterThan128InMem : 1;
    bool ClassifyIntegerMMXAsSSE : 1;
    bool HonorsRevision98 : 1;
    bool Clang11Compat : 1;

    ABIFlags()
        : PassInt128VectorsInMem(true),
          ReturnCXXRecordGreaterThan128InMem(true),
          ClassifyIntegerMMXAsSSE(true), HonorsRevision98(true),
          Clang11Compat(true) {}

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

  RecordArgABI getRecordArgABI(const RecordType *RT) const;
  RecordArgABI getRecordArgABI(const Type *Ty) const;
  RecordArgABI getRecordArgABI(const RecordType *RT, bool IsCxxRecord) const;
  bool isPromotableInteger(const IntegerType *IT) const;
  virtual void computeInfo(ABIFunctionInfo &FI) const = 0;
  virtual bool isPassByRef(const Type *Ty) const { return false; }
  const ABICompatInfo &getABICompatInfo() const { return CompatInfo; }
  ABIArgInfo getNaturalAlignIndirect(const Type *Ty, bool ByVal = true) const;
  bool isAggregateTypeForABI(const Type *Ty) const;
  bool isZeroSizedType(const Type *Ty) const;
  bool isEmptyRecord(const RecordType *RT) const;
  bool isEmptyField(const FieldInfo &FI) const;

  void setABICompatInfo(const ABICompatInfo &Info) { CompatInfo = Info; }
};

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_ABIINFO_H
