//===- HexagonMCChecker.h - Instruction bundle checking ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the checking of insns inside a bundle according to the
// packet constraint rules of the Hexagon ISA.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCCHECKER_H
#define LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCCHECKER_H

#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/SMLoc.h"
#include <set>
#include <utility>

namespace llvm {

class MCContext;
class MCInst;
class MCInstrInfo;
class MCRegisterInfo;
class MCSubtargetInfo;

/// Check for a valid bundle.
class HexagonMCChecker {
  MCContext &Context;
  MCInst &MCB;
  const MCRegisterInfo &RI;
  MCInstrInfo const &MCII;
  MCSubtargetInfo const &STI;
  bool ReportErrors;

  /// Set of definitions: register #, if predicated, if predicated true.
  using PredSense = std::pair<MCRegister, bool>;
  static const PredSense Unconditional;
  using PredSet = std::multiset<PredSense>;
  using PredSetIterator = std::multiset<PredSense>::iterator;

  using DefsIterator = DenseMap<MCRegister, PredSet>::iterator;
  DenseMap<MCRegister, PredSet> Defs;

  /// Set of weak definitions whose clashes should be enforced selectively.
  using SoftDefsIterator = std::set<MCRegister>::iterator;
  std::set<MCRegister> SoftDefs;

  /// Set of temporary definitions not committed to the register file.
  using TmpDefsIterator = std::set<MCRegister>::iterator;
  std::set<MCRegister> TmpDefs;

  /// Set of new predicates used.
  using NewPredsIterator = std::set<MCRegister>::iterator;
  std::set<MCRegister> NewPreds;

  /// Set of predicates defined late.
  using LatePredsIterator = std::multiset<MCRegister>::iterator;
  std::multiset<MCRegister> LatePreds;

  /// Set of uses.
  using UsesIterator = std::set<MCRegister>::iterator;
  std::set<MCRegister> Uses;

  /// Pre-defined set of read-only registers.
  using ReadOnlyIterator = std::set<MCRegister>::iterator;
  std::set<MCRegister> ReadOnly;

  // Contains the vector-pair-registers with the even number
  // first ("v0:1", e.g.) used/def'd in this packet.
  std::set<MCRegister> ReversePairs;

  void init();
  void init(MCInst const &);
  void initReg(MCInst const &, MCRegister, MCRegister &PredReg, bool &isTrue);

  bool registerUsed(MCRegister Register);

  /// \return a tuple of: pointer to the producer instruction or nullptr if
  /// none was found, the operand index, and the PredicateInfo for the
  /// producer.
  std::tuple<MCInst const *, unsigned, HexagonMCInstrInfo::PredicateInfo>
  registerProducer(MCRegister Register,
                   HexagonMCInstrInfo::PredicateInfo Predicated);

  // Checks performed.
  bool checkBranches();
  bool checkPredicates();
  bool checkNewValues();
  bool checkRegisters();
  bool checkRegistersReadOnly();
  void checkRegisterCurDefs();
  bool checkSolo();
  bool checkShuffle();
  bool checkSlots();
  bool checkAXOK();
  bool checkHWLoop();
  bool checkCOFMax1();
  bool checkLegalVecRegPair();
  bool checkValidTmpDst();
  bool checkHVXAccum();

  static void compoundRegisterMap(unsigned &);

  bool isLoopRegister(MCRegister R) const {
    return (Hexagon::SA0 == R || Hexagon::LC0 == R || Hexagon::SA1 == R ||
            Hexagon::LC1 == R);
  }

public:
  explicit HexagonMCChecker(MCContext &Context, MCInstrInfo const &MCII,
                            MCSubtargetInfo const &STI, MCInst &mcb,
                            const MCRegisterInfo &ri, bool ReportErrors = true);
  explicit HexagonMCChecker(HexagonMCChecker const &Check,
                            MCSubtargetInfo const &STI, bool CopyReportErrors);

  bool check(bool FullCheck = true);
  void reportErrorRegisters(MCRegister Register);
  void reportErrorNewValue(MCRegister Register);
  void reportError(SMLoc Loc, Twine const &Msg);
  void reportNote(SMLoc Loc, Twine const &Msg);
  void reportError(Twine const &Msg);
  void reportWarning(Twine const &Msg);
  void reportBranchErrors();
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCCHECKER_H
