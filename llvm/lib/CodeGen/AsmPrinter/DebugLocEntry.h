//===-- llvm/CodeGen/DebugLocEntry.h - Entry in debug_loc list -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DEBUGLOCENTRY_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DEBUGLOCENTRY_H

#include "DebugLocStream.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/Debug.h"

namespace llvm {
class AsmPrinter;

/// This struct describes target specific location.
struct TargetIndexLocation {
  int Index;
  int Offset;

  TargetIndexLocation() = default;
  TargetIndexLocation(unsigned Idx, int64_t Offset)
      : Index(Idx), Offset(Offset) {}

  bool operator==(const TargetIndexLocation &Other) const {
    return Index == Other.Index && Offset == Other.Offset;
  }
};

/// A single location or constant.
class DbgValueLoc {
  /// Any complex address location expression for this DbgValueLoc.
  const DIExpression *Expression;

  /// Type of entry that this represents.
  enum EntryType {
    E_Location,
    E_Integer,
    E_ConstantFP,
    E_ConstantInt,
    E_TargetIndexLocation
  };
  enum EntryType EntryKind;

  /// Either a constant,
  union {
    int64_t Int;
    const ConstantFP *CFP;
    const ConstantInt *CIP;
  } Constant;

  union {
    /// Or a location in the machine frame.
    MachineLocation Loc;
    /// Or a location from target specific location.
    TargetIndexLocation TIL;
  };

public:
  DbgValueLoc(const DIExpression *Expr, int64_t i)
      : Expression(Expr), EntryKind(E_Integer) {
    Constant.Int = i;
  }
  DbgValueLoc(const DIExpression *Expr, const ConstantFP *CFP)
      : Expression(Expr), EntryKind(E_ConstantFP) {
    Constant.CFP = CFP;
  }
  DbgValueLoc(const DIExpression *Expr, const ConstantInt *CIP)
      : Expression(Expr), EntryKind(E_ConstantInt) {
    Constant.CIP = CIP;
  }
  DbgValueLoc(const DIExpression *Expr, MachineLocation Loc)
      : Expression(Expr), EntryKind(E_Location), Loc(Loc) {
    assert(cast<DIExpression>(Expr)->isValid());
  }
  DbgValueLoc(const DIExpression *Expr, TargetIndexLocation Loc)
      : Expression(Expr), EntryKind(E_TargetIndexLocation), TIL(Loc) {}

  bool isLocation() const { return EntryKind == E_Location; }
  bool isTargetIndexLocation() const {
    return EntryKind == E_TargetIndexLocation;
  }
  bool isInt() const { return EntryKind == E_Integer; }
  bool isConstantFP() const { return EntryKind == E_ConstantFP; }
  bool isConstantInt() const { return EntryKind == E_ConstantInt; }
  int64_t getInt() const { return Constant.Int; }
  const ConstantFP *getConstantFP() const { return Constant.CFP; }
  const ConstantInt *getConstantInt() const { return Constant.CIP; }
  MachineLocation getLoc() const { return Loc; }
  TargetIndexLocation getTargetIndexLocation() const { return TIL; }
  bool isFragment() const { return getExpression()->isFragment(); }
  bool isEntryVal() const { return getExpression()->isEntryValue(); }
  const DIExpression *getExpression() const { return Expression; }
  friend bool operator==(const DbgValueLoc &, const DbgValueLoc &);
  friend bool operator<(const DbgValueLoc &, const DbgValueLoc &);
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const {
    if (isLocation()) {
      llvm::dbgs() << "Loc = { reg=" << Loc.getReg() << " ";
      if (Loc.isIndirect())
        llvm::dbgs() << "+0";
      llvm::dbgs() << "} ";
    } else if (isConstantInt())
      Constant.CIP->dump();
    else if (isConstantFP())
      Constant.CFP->dump();
    if (Expression)
      Expression->dump();
  }
#endif
};

/// This struct describes location entries emitted in the .debug_loc
/// section.
class DebugLocEntry {
  /// Begin and end symbols for the address range that this location is valid.
  const MCSymbol *Begin;
  const MCSymbol *End;

  /// A nonempty list of locations/constants belonging to this entry,
  /// sorted by offset.
  SmallVector<DbgValueLoc, 1> Values;

public:
  /// Create a location list entry for the range [\p Begin, \p End).
  ///
  /// \param Vals One or more values describing (parts of) the variable.
  DebugLocEntry(const MCSymbol *Begin, const MCSymbol *End,
                ArrayRef<DbgValueLoc> Vals)
      : Begin(Begin), End(End) {
    addValues(Vals);
  }

  /// Attempt to merge this DebugLocEntry with Next and return
  /// true if the merge was successful. Entries can be merged if they
  /// share the same Loc/Constant and if Next immediately follows this
  /// Entry.
  bool MergeRanges(const DebugLocEntry &Next) {
    // If this and Next are describing the same variable, merge them.
    if ((End == Next.Begin && Values == Next.Values)) {
      End = Next.End;
      return true;
    }
    return false;
  }

  const MCSymbol *getBeginSym() const { return Begin; }
  const MCSymbol *getEndSym() const { return End; }
  ArrayRef<DbgValueLoc> getValues() const { return Values; }
  void addValues(ArrayRef<DbgValueLoc> Vals) {
    Values.append(Vals.begin(), Vals.end());
    sortUniqueValues();
    assert((Values.size() == 1 || all_of(Values, [](DbgValueLoc V) {
              return V.isFragment();
            })) && "must either have a single value or multiple pieces");
  }

  // Sort the pieces by offset.
  // Remove any duplicate entries by dropping all but the first.
  void sortUniqueValues() {
    llvm::sort(Values);
    Values.erase(std::unique(Values.begin(), Values.end(),
                             [](const DbgValueLoc &A, const DbgValueLoc &B) {
                               return A.getExpression() == B.getExpression();
                             }),
                 Values.end());
  }

  /// Lower this entry into a DWARF expression.
  void finalize(const AsmPrinter &AP,
                DebugLocStream::ListBuilder &List,
                const DIBasicType *BT,
                DwarfCompileUnit &TheCU);
};

/// Compare two DbgValueLocs for equality.
inline bool operator==(const DbgValueLoc &A,
                       const DbgValueLoc &B) {
  if (A.EntryKind != B.EntryKind)
    return false;

  if (A.Expression != B.Expression)
    return false;

  switch (A.EntryKind) {
  case DbgValueLoc::E_Location:
    return A.Loc == B.Loc;
  case DbgValueLoc::E_TargetIndexLocation:
    return A.TIL == B.TIL;
  case DbgValueLoc::E_Integer:
    return A.Constant.Int == B.Constant.Int;
  case DbgValueLoc::E_ConstantFP:
    return A.Constant.CFP == B.Constant.CFP;
  case DbgValueLoc::E_ConstantInt:
    return A.Constant.CIP == B.Constant.CIP;
  }
  llvm_unreachable("unhandled EntryKind");
}

/// Compare two fragments based on their offset.
inline bool operator<(const DbgValueLoc &A,
                      const DbgValueLoc &B) {
  return A.getExpression()->getFragmentInfo()->OffsetInBits <
         B.getExpression()->getFragmentInfo()->OffsetInBits;
}

}

#endif
