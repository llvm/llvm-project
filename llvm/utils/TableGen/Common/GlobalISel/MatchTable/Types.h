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

#include "Common/InfoByHwMode.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include <map>
#include <set>
#include <string>
#include <variant>

namespace llvm {
struct TypeSetByHwMode;

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

/// An LLT for each HwMode. Used for types that differ between HwModes so that
/// they can be resolved when the match table is executed.
struct LLTByHwMode : public InfoByHwMode<LLTCodeGen> {
  void insertTypeForMode(unsigned Mode, const LLTCodeGen &Ty) {
    Map.try_emplace(Mode, Ty);
  }

  bool operator==(const LLTByHwMode &Other) const { return Map == Other.Map; }
  bool operator<(const LLTByHwMode &Other) const { return Map < Other.Map; }
};

/// Assigns an index to each LLTByHwMode used in the match table. The index is
/// the operand of the *ByHwMode opcodes and is passed to getLLTForHwMode.
extern std::map<LLTByHwMode, unsigned> LLTByHwModeIDs;
unsigned getLLTByHwModeID(const LLTByHwMode &Ty);

/// Either a single LLT or an LLT for each HwMode.
using LLTOrLLTByHwMode = std::variant<LLTCodeGen, LLTByHwMode>;

/// Convert a TypeSetByHwMode with one type per HwMode to an LLT. If every
/// HwMode has the same type, a single LLTCodeGen is returned. Otherwise an
/// LLTByHwMode is returned. If \p Generic is true, MVTToGenericLLT is used for
/// non-pointer types instead of MVTToLLT. Returns std::nullopt if the type set
/// has more than one type for any HwMode or a type can't be converted.
std::optional<LLTOrLLTByHwMode> getLLTOrLLTByHwMode(const TypeSetByHwMode &VTy,
                                                    bool Generic = false);

using TempTypeIdx = int64_t;
class LLTCodeGenOrTempType {
public:
  LLTCodeGenOrTempType(const LLTCodeGen &LLT) : Data(LLT) {}
  LLTCodeGenOrTempType(TempTypeIdx TempTy) : Data(TempTy) {}
  LLTCodeGenOrTempType(const LLTByHwMode &Ty) : Data(Ty) {}
  LLTCodeGenOrTempType(const LLTOrLLTByHwMode &Ty) {
    if (const auto *LLT = std::get_if<LLTCodeGen>(&Ty))
      Data = *LLT;
    else
      Data = std::get<LLTByHwMode>(Ty);
  }

  bool isLLTCodeGen() const { return std::holds_alternative<LLTCodeGen>(Data); }
  bool isTempTypeIdx() const {
    return std::holds_alternative<TempTypeIdx>(Data);
  }
  bool isLLTByHwMode() const {
    return std::holds_alternative<LLTByHwMode>(Data);
  }

  const LLTCodeGen &getLLTCodeGen() const {
    assert(isLLTCodeGen());
    return std::get<LLTCodeGen>(Data);
  }

  TempTypeIdx getTempTypeIdx() const {
    assert(isTempTypeIdx());
    return std::get<TempTypeIdx>(Data);
  }

  const LLTByHwMode &getLLTByHwMode() const {
    assert(isLLTByHwMode());
    return std::get<LLTByHwMode>(Data);
  }

private:
  std::variant<LLTCodeGen, TempTypeIdx, LLTByHwMode> Data;
};

} // namespace gi
} // namespace llvm

#endif
