//===----- HexagonMCChecker.h - Instruction bundle checking ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the checking of insns inside a bundle according to the
// packet constraint rules of the Hexagon ISA.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONMCCHECKER_H
#define HEXAGONMCCHECKER_H

#include "MCTargetDesc/HexagonMCShuffler.h"
#include <queue>
#include <set>

using namespace llvm;

namespace llvm {
class MCOperandInfo;

/// Check for a valid bundle.
class HexagonMCChecker {
  MCContext &Context;
  MCInst &MCB;
  const MCRegisterInfo &RI;
  MCInstrInfo const &MCII;
  MCSubtargetInfo const &STI;
  bool ReportErrors;

  /// Set of definitions: register #, if predicated, if predicated true.
  typedef std::pair<unsigned, bool> PredSense;
  static const PredSense Unconditional;
  typedef std::multiset<PredSense> PredSet;
  typedef std::multiset<PredSense>::iterator PredSetIterator;

  typedef llvm::DenseMap<unsigned, PredSet>::iterator DefsIterator;
  llvm::DenseMap<unsigned, PredSet> Defs;

  /// Information about how a new-value register is defined or used:
  ///   PredReg = predicate register, 0 if use/def not predicated,
  ///   Cond    = true/false for if(PredReg)/if(!PredReg) respectively,
  ///   IsFloat = true if definition produces a floating point value
  ///             (not valid for uses),
  ///   IsNVJ   = true if the use is a new-value branch (not valid for
  ///             definitions).
  struct NewSense {
    unsigned PredReg;
    bool IsFloat, IsNVJ, Cond;
    // The special-case "constructors":
    static NewSense Jmp(bool isNVJ) {
      NewSense NS = {/*PredReg=*/0, /*IsFloat=*/false, /*IsNVJ=*/isNVJ,
                     /*Cond=*/false};
      return NS;
    }
    static NewSense Use(unsigned PR, bool True) {
      NewSense NS = {/*PredReg=*/PR, /*IsFloat=*/false, /*IsNVJ=*/false,
                     /*Cond=*/True};
      return NS;
    }
    static NewSense Def(unsigned PR, bool True, bool Float) {
      NewSense NS = {/*PredReg=*/PR, /*IsFloat=*/Float, /*IsNVJ=*/false,
                     /*Cond=*/True};
      return NS;
    }
  };
  /// Set of definitions that produce new register:
  typedef llvm::SmallVector<NewSense, 2> NewSenseList;
  typedef llvm::DenseMap<unsigned, NewSenseList>::iterator NewDefsIterator;
  llvm::DenseMap<unsigned, NewSenseList> NewDefs;

  /// Set of weak definitions whose clashes should be enforced selectively.
  typedef std::set<unsigned>::iterator SoftDefsIterator;
  std::set<unsigned> SoftDefs;

  /// Set of current definitions committed to the register file.
  typedef std::set<unsigned>::iterator CurDefsIterator;
  std::set<unsigned> CurDefs;

  /// Set of temporary definitions not committed to the register file.
  typedef std::set<unsigned>::iterator TmpDefsIterator;
  std::set<unsigned> TmpDefs;

  /// Set of new predicates used.
  typedef std::set<unsigned>::iterator NewPredsIterator;
  std::set<unsigned> NewPreds;

  /// Set of predicates defined late.
  typedef std::multiset<unsigned>::iterator LatePredsIterator;
  std::multiset<unsigned> LatePreds;

  /// Set of uses.
  typedef std::set<unsigned>::iterator UsesIterator;
  std::set<unsigned> Uses;

  /// Set of new values used: new register, if new-value jump.
  typedef llvm::DenseMap<unsigned, NewSense>::iterator NewUsesIterator;
  llvm::DenseMap<unsigned, NewSense> NewUses;

  /// Pre-defined set of read-only registers.
  typedef std::set<unsigned>::iterator ReadOnlyIterator;
  std::set<unsigned> ReadOnly;

  void init();
  void init(MCInst const &);
  void initReg(MCInst const &, unsigned, unsigned &PredReg, bool &isTrue);

  // Checks performed.
  bool checkBranches();
  bool checkPredicates();
  bool checkNewValues();
  bool checkRegisters();
  bool checkRegistersReadOnly();
  bool checkSolo();
  bool checkShuffle();
  bool checkSlots();

  static void compoundRegisterMap(unsigned &);

  bool isPredicateRegister(unsigned R) const {
    return (Hexagon::P0 == R || Hexagon::P1 == R || Hexagon::P2 == R ||
            Hexagon::P3 == R);
  };
  bool isLoopRegister(unsigned R) const {
    return (Hexagon::SA0 == R || Hexagon::LC0 == R || Hexagon::SA1 == R ||
            Hexagon::LC1 == R);
  };

  bool hasValidNewValueDef(const NewSense &Use, const NewSenseList &Defs) const;

public:
  explicit HexagonMCChecker(MCContext &Context, MCInstrInfo const &MCII,
                            MCSubtargetInfo const &STI, MCInst &mcb,
                            const MCRegisterInfo &ri, bool ReportErrors = true);

  bool check(bool FullCheck = true);
  void reportErrorRegisters(unsigned Register);
  void reportErrorNewValue(unsigned Register);
  void reportError(SMLoc Loc, llvm::Twine const &Msg);
  void reportError(llvm::Twine const &Msg);
  void reportWarning(llvm::Twine const &Msg);
};

} // namespace llvm

#endif // HEXAGONMCCHECKER_H
