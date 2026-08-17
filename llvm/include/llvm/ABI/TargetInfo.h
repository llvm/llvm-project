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
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <memory>

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

protected:
  TypeBuilder &TB;

public:
  explicit TargetInfo(TypeBuilder &Builder) : CompatInfo(), TB(Builder) {}
  TargetInfo(TypeBuilder &Builder, const ABICompatInfo &Info)
      : CompatInfo(Info), TB(Builder) {}

  virtual ~TargetInfo() = default;

  /// Populate FI with the target's ABI-lowering decisions for each argument
  /// and return value.
  virtual void computeInfo(FunctionInfo &FI) const = 0;
  virtual bool isPassByRef(const Type *Ty) const { return false; }
  const ABICompatInfo &getABICompatInfo() const { return CompatInfo; }

protected:
  LLVM_ABI RecordArgABI getRecordArgABI(const RecordType *RT) const;
  LLVM_ABI RecordArgABI getRecordArgABI(const Type *Ty) const;
  LLVM_ABI bool isPromotableInteger(const IntegerType *IT) const;
  LLVM_ABI ArgInfo getNaturalAlignIndirect(const Type *Ty,
                                           bool ByVal = true) const;
  LLVM_ABI bool isAggregateTypeForABI(const Type *Ty) const;

  /// If Ty is a transparent union, return its first field type; otherwise
  /// return Ty unchanged.
  LLVM_ABI const Type *useFirstFieldIfTransparentUnion(const Type *Ty) const;

  /// Apply rules for classifying return types that are common to all targets.
  LLVM_ABI bool maybeCommonClassifyReturnType(FunctionInfo &FI) const;

  /// Return true if \p Ty is a valid base type for a homogeneous aggregate.
  virtual bool isHomogeneousAggregateBaseType(const Type *Ty) const {
    return false;
  }

  /// Return true if a homogeneous aggregate with \p Members copies of \p Base
  /// is small enough to be passed in registers for this ABI.
  virtual bool isHomogeneousAggregateSmallEnough(const Type *Base,
                                                 uint64_t Members) const {
    return false;
  }

  /// Return true if zero-length bitfields should be ignored when deciding
  /// whether an aggregate is homogeneous.
  virtual bool isZeroLengthBitfieldPermittedInHomogeneousAggregate() const {
    return false;
  }

  /// Return true if the C++ ABI permits \p RT to be a homogeneous aggregate.
  virtual bool isPermittedToBeHomogeneousAggregate(const RecordType *RT) const {
    return true;
  }

  /// Return true if \p Ty is an ELFv2-style homogeneous aggregate. \p Base is
  /// set to the base element type and \p Members to the number of base
  /// elements.
  LLVM_ABI bool isHomogeneousAggregate(const Type *Ty, const Type *&Base,
                                       uint64_t &Members) const;
};

LLVM_ABI std::unique_ptr<TargetInfo> createBPFTargetInfo(TypeBuilder &TB);

/// The AVX ABI level for X86 targets.
enum class X86AVXABILevel {
  None,
  AVX,
  AVX512,
  Last = AVX512 // must be last
};

LLVM_ABI std::unique_ptr<TargetInfo>
createX86_64TargetInfo(TypeBuilder &TB, X86AVXABILevel AVXLevel,
                       bool Has64BitPointers, const ABICompatInfo &Compat);

enum class AArch64ABIKind {
  AAPCS = 0,
  DarwinPCS,
  Win64,
  AAPCSSoft,
};

/// Target / language flags that affect AArch64 ABI classification.
/// Callers (e.g. Clang) resolve Triple and LangOptions into these flags
/// rather than passing a Triple into the ABI library.
struct AArch64ABIOptions {
  AArch64ABIKind Kind = AArch64ABIKind::AAPCS;
  bool IsILP32 = false;
  bool IsMicrosoftCXXABI = false;

  AArch64ABIOptions() = default;
  AArch64ABIOptions(AArch64ABIKind Kind) : Kind(Kind) {}
};

LLVM_ABI std::unique_ptr<TargetInfo>
createAArch64TargetInfo(TypeBuilder &TB, const AArch64ABIOptions &Opts);

} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_TARGETINFO_H
