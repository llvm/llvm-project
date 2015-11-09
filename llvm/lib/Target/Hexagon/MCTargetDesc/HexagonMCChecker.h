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

#include <map>
#include <set>
#include <queue>
#include "MCTargetDesc/HexagonMCShuffler.h"

using namespace llvm;

namespace llvm {
class MCOperandInfo;

typedef struct {
  unsigned Error, Warning, ShuffleError;
  unsigned Register;
} ErrInfo_T;

class HexagonMCErrInfo {
public:
  enum {
    CHECK_SUCCESS         = 0,
    // Errors.
    CHECK_ERROR_BRANCHES  = 0x00001,
    CHECK_ERROR_NEWP      = 0x00002,
    CHECK_ERROR_NEWV      = 0x00004,
    CHECK_ERROR_REGISTERS = 0x00008,
    CHECK_ERROR_READONLY  = 0x00010,
    CHECK_ERROR_LOOP      = 0x00020,
    CHECK_ERROR_ENDLOOP   = 0x00040,
    CHECK_ERROR_SOLO      = 0x00080,
    CHECK_ERROR_SHUFFLE   = 0x00100,
    CHECK_ERROR_NOSLOTS   = 0x00200,
    CHECK_ERROR_UNKNOWN   = 0x00400,
    // Warnings.
    CHECK_WARN_CURRENT    = 0x10000,
    CHECK_WARN_TEMPORARY  = 0x20000
  };
  ErrInfo_T s;

  void reset() {
    s.Error = CHECK_SUCCESS;
    s.Warning = CHECK_SUCCESS;
    s.ShuffleError = HexagonShuffler::SHUFFLE_SUCCESS;
    s.Register = Hexagon::NoRegister;
  };
  HexagonMCErrInfo() {
    reset();
  };

  void setError(unsigned e, unsigned r = Hexagon::NoRegister)
    { s.Error = e; s.Register = r; };
  void setWarning(unsigned w, unsigned r = Hexagon::NoRegister)
    { s.Warning = w; s.Register = r; };
  void setShuffleError(unsigned e) { s.ShuffleError = e; };
};

/// Check for a valid bundle.
class HexagonMCChecker {
  /// Insn bundle.
  MCInst& MCB;
  MCInst& MCBDX;
  const MCRegisterInfo& RI;
  MCInstrInfo const &MCII;
  MCSubtargetInfo const &STI;
  bool bLoadErrInfo;

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
      NewSense NS = { /*PredReg=*/ 0, /*IsFloat=*/ false, /*IsNVJ=*/ isNVJ,
                      /*Cond=*/ false };
      return NS;
    }
    static NewSense Use(unsigned PR, bool True) {
      NewSense NS = { /*PredReg=*/ PR, /*IsFloat=*/ false, /*IsNVJ=*/ false,
                      /*Cond=*/ True };
      return NS;
    }
    static NewSense Def(unsigned PR, bool True, bool Float) {
      NewSense NS = { /*PredReg=*/ PR, /*IsFloat=*/ Float, /*IsNVJ=*/ false,
                      /*Cond=*/ True };
      return NS;
    }
  };
  /// Set of definitions that produce new register:
  typedef llvm::SmallVector<NewSense,2> NewSenseList;
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

  std::queue<ErrInfo_T> ErrInfoQ;
  HexagonMCErrInfo CrntErrInfo;

  void getErrInfo() {
    if (bLoadErrInfo == true) {
      if (ErrInfoQ.empty()) {
        CrntErrInfo.reset();
      } else {
        CrntErrInfo.s = ErrInfoQ.front();
        ErrInfoQ.pop();
      }
    }
    bLoadErrInfo = false;
  }

  void init();
  void init(MCInst const&);

  // Checks performed.
  bool checkBranches();
  bool checkPredicates();
  bool checkNewValues();
  bool checkRegisters();
  bool checkSolo();
  bool checkShuffle();
  bool checkSlots();

  static void compoundRegisterMap(unsigned&);

  bool isPredicateRegister(unsigned R) const {
    return (Hexagon::P0 == R || Hexagon::P1 == R ||
            Hexagon::P2 == R || Hexagon::P3 == R);
  };
  bool isLoopRegister(unsigned R) const {
    return (Hexagon::SA0 == R || Hexagon::LC0 == R ||
            Hexagon::SA1 == R || Hexagon::LC1 == R);
  };

  bool hasValidNewValueDef(const NewSense &Use,
                           const NewSenseList &Defs) const;

  public:
  explicit HexagonMCChecker(MCInstrInfo const &MCII, MCSubtargetInfo const &STI, MCInst& mcb, MCInst &mcbdx,
                            const MCRegisterInfo& ri);

  bool check();

  /// add a new error/warning
  void addErrInfo(HexagonMCErrInfo &err) { ErrInfoQ.push(err.s); };

  /// Return the error code for the last operation in the insn bundle.
  unsigned getError() { getErrInfo(); return CrntErrInfo.s.Error; };
  unsigned getWarning() { getErrInfo(); return CrntErrInfo.s.Warning; };
  unsigned getShuffleError() { getErrInfo(); return CrntErrInfo.s.ShuffleError; };
  unsigned getErrRegister() { getErrInfo(); return CrntErrInfo.s.Register; };
  bool getNextErrInfo() {
    bLoadErrInfo = true;
    return (ErrInfoQ.empty()) ? false : (getErrInfo(), true);
  }
};

}

#endif // HEXAGONMCCHECKER_H
