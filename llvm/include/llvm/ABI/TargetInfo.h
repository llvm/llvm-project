//===----- TargetInfo.h - Target ABI information ------------------- C++
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Target-specific ABI information and factory functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABI_TARGETINFO_H
#define LLVM_ABI_TARGETINFO_H

#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/Types.h"
#include <cassert>

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

/// Flags controlling target-specific ABI compatibility behaviour.
/// Construct with the default constructor for the current ABI, or use
/// fromVersion() to get the flags that match a specific Clang version.
struct ABICompatInfo {
  bool PassInt128VectorsInMem : 1;
  bool ReturnCXXRecordGreaterThan128InMem : 1;
  bool ClassifyIntegerMMXAsSSE : 1;
  bool HonorsRevision98 : 1;
  bool Clang11Compat : 1;

  ABICompatInfo()
      : PassInt128VectorsInMem(true), ReturnCXXRecordGreaterThan128InMem(true),
        ClassifyIntegerMMXAsSSE(true), HonorsRevision98(true),
        Clang11Compat(true) {}

  /// Return flags matching the ABI emitted by the given Clang major version.
  // TODO: fill in per-version flag overrides.
  static ABICompatInfo fromVersion(unsigned /*ClangMajor*/) {
    return ABICompatInfo();
  }
};

class TargetInfo {
private:
  ABICompatInfo CompatInfo;

public:
  TargetInfo() : CompatInfo() {}
  explicit TargetInfo(const ABICompatInfo &Info) : CompatInfo(Info) {}

  virtual ~TargetInfo() = default;

  /// Populate FI with the target's ABI-lowering decisions for each argument
  /// and return value.
  virtual void computeInfo(FunctionInfo &FI) const = 0;
  virtual bool isPassByRef(const Type *Ty) const { return false; }
  const ABICompatInfo &getABICompatInfo() const { return CompatInfo; }

protected:
  RecordArgABI getRecordArgABI(const RecordType *RT) const;
  RecordArgABI getRecordArgABI(const Type *Ty) const;
  bool isPromotableInteger(const IntegerType *IT) const;
  ArgInfo getNaturalAlignIndirect(const Type *Ty, bool ByVal = true) const;
  bool isAggregateTypeForABI(const Type *Ty) const;
};

std::unique_ptr<TargetInfo> createBPFTargetInfo(TypeBuilder &TB);

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_TARGETINFO_H
