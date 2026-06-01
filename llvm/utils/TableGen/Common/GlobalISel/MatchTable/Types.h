//===- Types.h ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains utilities used to represent LLTs for MatchTable-related
/// components.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_COMMON_GLOBALISEL_MATCHTABLE_TYPES_H
#define LLVM_UTILS_TABLEGEN_COMMON_GLOBALISEL_MATCHTABLE_TYPES_H

#include "llvm/CodeGenTypes/LowLevelType.h"
#include <set>
#include <string>
#include <variant>

namespace llvm {
namespace gi {

/// This class stands in for LLT wherever we want to tablegen-erate an
/// equivalent at compiler run-time.
class LLTCodeGen {
private:
  LLT Ty;

public:
  LLTCodeGen() = default;
  LLTCodeGen(const LLT &Ty) : Ty(Ty) {}

  std::string getCxxEnumValue() const;

  void emitCxxEnumValue(raw_ostream &OS) const;
  void emitCxxConstructorCall(raw_ostream &OS) const;

  const LLT &get() const { return Ty; }

  /// This ordering is used for std::unique() and llvm::sort(). There's no
  /// particular logic behind the order but either A < B or B < A must be
  /// true if A != B.
  bool operator<(const LLTCodeGen &Other) const;
  bool operator==(const LLTCodeGen &B) const { return Ty == B.Ty; }
};

// Track all types that are used so we can emit the corresponding enum.
extern std::set<LLTCodeGen> KnownTypes;

/// Convert an MVT to an equivalent LLT if possible, or the invalid LLT() for
/// MVTs that don't map cleanly to an LLT (e.g., iPTR, *any, ...).
std::optional<LLTCodeGen> MVTToLLT(MVT VT);
std::optional<LLTCodeGen> MVTToGenericLLT(MVT VT);

using TempTypeIdx = int64_t;
class LLTCodeGenOrTempType {
public:
  LLTCodeGenOrTempType(const LLTCodeGen &LLT) : Data(LLT) {}
  LLTCodeGenOrTempType(TempTypeIdx TempTy) : Data(TempTy) {}

  bool isLLTCodeGen() const { return std::holds_alternative<LLTCodeGen>(Data); }
  bool isTempTypeIdx() const {
    return std::holds_alternative<TempTypeIdx>(Data);
  }

  const LLTCodeGen &getLLTCodeGen() const {
    assert(isLLTCodeGen());
    return std::get<LLTCodeGen>(Data);
  }

  TempTypeIdx getTempTypeIdx() const {
    assert(isTempTypeIdx());
    return std::get<TempTypeIdx>(Data);
  }

private:
  std::variant<LLTCodeGen, TempTypeIdx> Data;
};

} // namespace gi
} // namespace llvm

#endif
