//===- ScopDetectionDiagnostic.cpp - Error diagnostics --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Small set of diagnostic helper classes to encapsulate any errors occurred
// during the detection of Scops.
//
// The ScopDetection defines a set of error classes (via Statistic variables)
// that groups a number of individual errors into a group, e.g. non-affinity
// related errors.
// On error we generate an object that carries enough additional information
// to diagnose the error and generate a helpful error message.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopDetectionDiagnostic.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <string>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "polly-detect"

#define SCOP_STAT(NAME, DESC)                                                  \
  {                                                                            \
    "polly-detect", "NAME", "Number of rejected regions: " DESC, {0}, {        \
      false                                                                    \
    }                                                                          \
  }

Statistic RejectStatistics[] = {
    SCOP_STAT(CFG, ""),
    SCOP_STAT(InvalidTerminator, "Unsupported terminator instruction"),
    SCOP_STAT(UnreachableInExit, "Unreachable in exit block"),
    SCOP_STAT(IrreducibleRegion, "Irreducible loops"),
    SCOP_STAT(LastCFG, ""),
    SCOP_STAT(AffFunc, ""),
    SCOP_STAT(UndefCond, "Undefined branch condition"),
    SCOP_STAT(InvalidCond, "Non-integer branch condition"),
    SCOP_STAT(UndefOperand, "Undefined operands in comparison"),
    SCOP_STAT(NonAffBranch, "Non-affine branch condition"),
    SCOP_STAT(NoBasePtr, "No base pointer"),
    SCOP_STAT(UndefBasePtr, "Undefined base pointer"),
    SCOP_STAT(VariantBasePtr, "Variant base pointer"),
    SCOP_STAT(NonAffineAccess, "Non-affine memory accesses"),
    SCOP_STAT(DifferentElementSize, "Accesses with differing sizes"),
    SCOP_STAT(LastAffFunc, ""),
    SCOP_STAT(LoopBound, "Uncomputable loop bounds"),
    SCOP_STAT(LoopHasNoExit, "Loop without exit"),
    SCOP_STAT(LoopHasMultipleExits, "Loop with multiple exits"),
    SCOP_STAT(LoopOnlySomeLatches, "Not all loop latches in scop"),
    SCOP_STAT(FuncCall, "Function call with side effects"),
    SCOP_STAT(NonSimpleMemoryAccess,
              "Compilated access semantics (volatile or atomic)"),
    SCOP_STAT(Alias, "Base address aliasing"),
    SCOP_STAT(Other, ""),
    SCOP_STAT(IntToPtr, "Integer to pointer conversions"),
    SCOP_STAT(Alloca, "Stack allocations"),
    SCOP_STAT(UnknownInst, "Unknown Instructions"),
    SCOP_STAT(Entry, "Contains entry block"),
    SCOP_STAT(Unprofitable, "Assumed to be unprofitable"),
    SCOP_STAT(LastOther, ""),
};

namespace polly {

/// Small string conversion via raw_string_stream.
template <typename T> std::string operator+(Twine LHS, const T &RHS) {
  std::string Buf;
  raw_string_ostream fmt(Buf);
  fmt << RHS;
  fmt.flush();

  return LHS.concat(Buf).str();
}
} // namespace polly

namespace llvm {

// Lexicographic order on (line, col) of our debug locations.
static bool operator<(const DebugLoc &LHS, const DebugLoc &RHS) {
  return LHS.getLine() < RHS.getLine() ||
         (LHS.getLine() == RHS.getLine() && LHS.getCol() < RHS.getCol());
}
} // namespace llvm

namespace polly {

BBPair getBBPairForRegion(const Region *R) {
  return std::make_pair(R->getEntry(), R->getExit());
}

void getDebugLocations(const BBPair &P, DebugLoc &Begin, DebugLoc &End) {
  SmallPtrSet<BasicBlock *, 32> Seen;
  SmallVector<BasicBlock *, 32> Todo;
  Todo.push_back(P.first);
  while (!Todo.empty()) {
    auto *BB = Todo.pop_back_val();
    if (BB == P.second)
      continue;
    if (!Seen.insert(BB).second)
      continue;
    Todo.append(succ_begin(BB), succ_end(BB));
    for (const Instruction &Inst : *BB) {
      DebugLoc DL = Inst.getDebugLoc();
      if (!DL)
        continue;

      Begin = Begin ? std::min(Begin, DL) : DL;
      End = End ? std::max(End, DL) : DL;
    }
  }
}

void emitRejectionRemarks(const BBPair &P, const RejectLog &Log,
                          OptimizationRemarkEmitter &ORE) {
  DebugLoc Begin, End;
  getDebugLocations(P, Begin, End);

  ORE.emit(
      OptimizationRemarkMissed(DEBUG_TYPE, "RejectionErrors", Begin, P.first)
      << "The following errors keep this region from being a Scop.");

  for (RejectReasonPtr RR : Log) {

    if (const DebugLoc &Loc = RR->getDebugLoc())
      ORE.emit(OptimizationRemarkMissed(DEBUG_TYPE, RR->getRemarkName(), Loc,
                                        RR->getRemarkBB())
               << RR->getEndUserMessage());
    else
      ORE.emit(OptimizationRemarkMissed(DEBUG_TYPE, RR->getRemarkName(), Begin,
                                        RR->getRemarkBB())
               << RR->getEndUserMessage());
  }

  /* Check to see if Region is a top level region, getExit = NULL*/
  if (P.second)
    ORE.emit(
        OptimizationRemarkMissed(DEBUG_TYPE, "InvalidScopEnd", End, P.second)
        << "Invalid Scop candidate ends here.");
  else
    ORE.emit(
        OptimizationRemarkMissed(DEBUG_TYPE, "InvalidScopEnd", End, P.first)
        << "Invalid Scop candidate ends here.");
}

//===----------------------------------------------------------------------===//
// RejectReason.

RejectReason::RejectReason(RejectReasonKind K) : Kind(K) {
  RejectStatistics[static_cast<int>(K)]++;
}

const DebugLoc RejectReason::Unknown = DebugLoc();

const DebugLoc &RejectReason::getDebugLoc() const {
  // Allocate an empty DebugLoc and return it a reference to it.
  return Unknown;
}

// RejectLog.
void RejectLog::print(raw_ostream &OS, int level) const {
  int j = 0;
  for (auto Reason : ErrorReports)
    OS.indent(level) << "[" << j++ << "] " << Reason->getMessage() << "\n";
}

//===----------------------------------------------------------------------===//
// ReportCFG.

ReportCFG::ReportCFG(const RejectReasonKind K) : RejectReason(K) {}

bool ReportCFG::classof(const RejectReason *RR) {
  return RR->getKind() >= RejectReasonKind::CFG &&
         RR->getKind() <= RejectReasonKind::LastCFG;
}

//===----------------------------------------------------------------------===//
// ReportInvalidTerminator.

std::string ReportInvalidTerminator::getRemarkName() const {
  return "InvalidTerminator";
}

const Value *ReportInvalidTerminator::getRemarkBB() const { return BB; }

std::string ReportInvalidTerminator::getMessage() const {
  return ("Invalid instruction terminates BB: " + BB->getName()).str();
}

const DebugLoc &ReportInvalidTerminator::getDebugLoc() const {
  return BB->getTerminator()->getDebugLoc();
}

bool ReportInvalidTerminator::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::InvalidTerminator;
}

//===----------------------------------------------------------------------===//
// UnreachableInExit.

std::string ReportUnreachableInExit::getRemarkName() const {
  return "UnreachableInExit";
}

const Value *ReportUnreachableInExit::getRemarkBB() const { return BB; }

std::string ReportUnreachableInExit::getMessage() const {
  std::string BBName = BB->getName();
  return "Unreachable in exit block" + BBName;
}

const DebugLoc &ReportUnreachableInExit::getDebugLoc() const { return DbgLoc; }

std::string ReportUnreachableInExit::getEndUserMessage() const {
  return "Unreachable in exit block.";
}

bool ReportUnreachableInExit::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::UnreachableInExit;
}

//===----------------------------------------------------------------------===//
// ReportIrreducibleRegion.

std::string ReportIrreducibleRegion::getRemarkName() const {
  return "IrreducibleRegion";
}

const Value *ReportIrreducibleRegion::getRemarkBB() const {
  return R->getEntry();
}

std::string ReportIrreducibleRegion::getMessage() const {
  return "Irreducible region encountered: " + R->getNameStr();
}

const DebugLoc &ReportIrreducibleRegion::getDebugLoc() const { return DbgLoc; }

std::string ReportIrreducibleRegion::getEndUserMessage() const {
  return "Irreducible region encountered in control flow.";
}

bool ReportIrreducibleRegion::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::IrreducibleRegion;
}

//===----------------------------------------------------------------------===//
// ReportAffFunc.

ReportAffFunc::ReportAffFunc(const RejectReasonKind K, const Instruction *Inst)
    : RejectReason(K), Inst(Inst) {}

bool ReportAffFunc::classof(const RejectReason *RR) {
  return RR->getKind() >= RejectReasonKind::AffFunc &&
         RR->getKind() <= RejectReasonKind::LastAffFunc;
}

//===----------------------------------------------------------------------===//
// ReportUndefCond.

std::string ReportUndefCond::getRemarkName() const { return "UndefCond"; }

const Value *ReportUndefCond::getRemarkBB() const { return BB; }

std::string ReportUndefCond::getMessage() const {
  return ("Condition based on 'undef' value in BB: " + BB->getName()).str();
}

bool ReportUndefCond::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::UndefCond;
}

//===----------------------------------------------------------------------===//
// ReportInvalidCond.

std::string ReportInvalidCond::getRemarkName() const { return "InvalidCond"; }

const Value *ReportInvalidCond::getRemarkBB() const { return BB; }

std::string ReportInvalidCond::getMessage() const {
  return ("Condition in BB '" + BB->getName()).str() +
         "' neither constant nor an icmp instruction";
}

bool ReportInvalidCond::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::InvalidCond;
}

//===----------------------------------------------------------------------===//
// ReportUndefOperand.

std::string ReportUndefOperand::getRemarkName() const { return "UndefOperand"; }

const Value *ReportUndefOperand::getRemarkBB() const { return BB; }

std::string ReportUndefOperand::getMessage() const {
  return ("undef operand in branch at BB: " + BB->getName()).str();
}

bool ReportUndefOperand::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::UndefOperand;
}

//===----------------------------------------------------------------------===//
// ReportNonAffBranch.

std::string ReportNonAffBranch::getRemarkName() const { return "NonAffBranch"; }

const Value *ReportNonAffBranch::getRemarkBB() const { return BB; }

std::string ReportNonAffBranch::getMessage() const {
  return ("Non affine branch in BB '" + BB->getName()).str() +
         "' with LHS: " + *LHS + " and RHS: " + *RHS;
}

bool ReportNonAffBranch::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::NonAffBranch;
}

//===----------------------------------------------------------------------===//
// ReportNoBasePtr.

std::string ReportNoBasePtr::getRemarkName() const { return "NoBasePtr"; }

const Value *ReportNoBasePtr::getRemarkBB() const { return Inst->getParent(); }

std::string ReportNoBasePtr::getMessage() const { return "No base pointer"; }

bool ReportNoBasePtr::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::NoBasePtr;
}

//===----------------------------------------------------------------------===//
// ReportUndefBasePtr.

std::string ReportUndefBasePtr::getRemarkName() const { return "UndefBasePtr"; }

const Value *ReportUndefBasePtr::getRemarkBB() const {
  return Inst->getParent();
}

std::string ReportUndefBasePtr::getMessage() const {
  return "Undefined base pointer";
}

bool ReportUndefBasePtr::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::UndefBasePtr;
}

//===----------------------------------------------------------------------===//
// ReportVariantBasePtr.

std::string ReportVariantBasePtr::getRemarkName() const {
  return "VariantBasePtr";
}

const Value *ReportVariantBasePtr::getRemarkBB() const {
  return Inst->getParent();
}

std::string ReportVariantBasePtr::getMessage() const {
  return "Base address not invariant in current region:" + *BaseValue;
}

std::string ReportVariantBasePtr::getEndUserMessage() const {
  return "The base address of this array is not invariant inside the loop";
}

bool ReportVariantBasePtr::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::VariantBasePtr;
}

//===----------------------------------------------------------------------===//
// ReportDifferentArrayElementSize

std::string ReportDifferentArrayElementSize::getRemarkName() const {
  return "DifferentArrayElementSize";
}

const Value *ReportDifferentArrayElementSize::getRemarkBB() const {
  return Inst->getParent();
}

std::string ReportDifferentArrayElementSize::getMessage() const {
  return "Access to one array through data types of different size";
}

bool ReportDifferentArrayElementSize::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::DifferentElementSize;
}

std::string ReportDifferentArrayElementSize::getEndUserMessage() const {
  StringRef BaseName = BaseValue->getName();
  std::string Name = BaseName.empty() ? "UNKNOWN" : BaseName;
  return "The array \"" + Name +
         "\" is accessed through elements that differ "
         "in size";
}

//===----------------------------------------------------------------------===//
// ReportNonAffineAccess.

std::string ReportNonAffineAccess::getRemarkName() const {
  return "NonAffineAccess";
}

const Value *ReportNonAffineAccess::getRemarkBB() const {
  return Inst->getParent();
}

std::string ReportNonAffineAccess::getMessage() const {
  return "Non affine access function: " + *AccessFunction;
}

bool ReportNonAffineAccess::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::NonAffineAccess;
}

std::string ReportNonAffineAccess::getEndUserMessage() const {
  StringRef BaseName = BaseValue->getName();
  std::string Name = BaseName.empty() ? "UNKNOWN" : BaseName;
  return "The array subscript of \"" + Name + "\" is not affine";
}

//===----------------------------------------------------------------------===//
// ReportLoopBound.

ReportLoopBound::ReportLoopBound(Loop *L, const SCEV *LoopCount)
    : RejectReason(RejectReasonKind::LoopBound), L(L), LoopCount(LoopCount),
      Loc(L->getStartLoc()) {}

std::string ReportLoopBound::getRemarkName() const { return "LoopBound"; }

const Value *ReportLoopBound::getRemarkBB() const { return L->getHeader(); }

std::string ReportLoopBound::getMessage() const {
  return "Non affine loop bound '" + *LoopCount +
         "' in loop: " + L->getHeader()->getName();
}

const DebugLoc &ReportLoopBound::getDebugLoc() const { return Loc; }

bool ReportLoopBound::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::LoopBound;
}

std::string ReportLoopBound::getEndUserMessage() const {
  return "Failed to derive an affine function from the loop bounds.";
}

//===----------------------------------------------------------------------===//
// ReportLoopHasNoExit.

std::string ReportLoopHasNoExit::getRemarkName() const {
  return "LoopHasNoExit";
}

const Value *ReportLoopHasNoExit::getRemarkBB() const { return L->getHeader(); }

std::string ReportLoopHasNoExit::getMessage() const {
  return "Loop " + L->getHeader()->getName() + " has no exit.";
}

bool ReportLoopHasNoExit::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::LoopHasNoExit;
}

const DebugLoc &ReportLoopHasNoExit::getDebugLoc() const { return Loc; }

std::string ReportLoopHasNoExit::getEndUserMessage() const {
  return "Loop cannot be handled because it has no exit.";
}

//===----------------------------------------------------------------------===//
// ReportLoopHasMultipleExits.

std::string ReportLoopHasMultipleExits::getRemarkName() const {
  return "ReportLoopHasMultipleExits";
}

const Value *ReportLoopHasMultipleExits::getRemarkBB() const {
  return L->getHeader();
}

std::string ReportLoopHasMultipleExits::getMessage() const {
  return "Loop " + L->getHeader()->getName() + " has multiple exits.";
}

bool ReportLoopHasMultipleExits::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::LoopHasMultipleExits;
}

const DebugLoc &ReportLoopHasMultipleExits::getDebugLoc() const { return Loc; }

std::string ReportLoopHasMultipleExits::getEndUserMessage() const {
  return "Loop cannot be handled because it has multiple exits.";
}

//===----------------------------------------------------------------------===//
// ReportLoopOnlySomeLatches

std::string ReportLoopOnlySomeLatches::getRemarkName() const {
  return "LoopHasNoExit";
}

const Value *ReportLoopOnlySomeLatches::getRemarkBB() const {
  return L->getHeader();
}

std::string ReportLoopOnlySomeLatches::getMessage() const {
  return "Not all latches of loop " + L->getHeader()->getName() +
         " part of scop.";
}

bool ReportLoopOnlySomeLatches::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::LoopHasNoExit;
}

const DebugLoc &ReportLoopOnlySomeLatches::getDebugLoc() const { return Loc; }

std::string ReportLoopOnlySomeLatches::getEndUserMessage() const {
  return "Loop cannot be handled because not all latches are part of loop "
         "region.";
}

//===----------------------------------------------------------------------===//
// ReportFuncCall.

ReportFuncCall::ReportFuncCall(Instruction *Inst)
    : RejectReason(RejectReasonKind::FuncCall), Inst(Inst) {}

std::string ReportFuncCall::getRemarkName() const { return "FuncCall"; }

const Value *ReportFuncCall::getRemarkBB() const { return Inst->getParent(); }

std::string ReportFuncCall::getMessage() const {
  return "Call instruction: " + *Inst;
}

const DebugLoc &ReportFuncCall::getDebugLoc() const {
  return Inst->getDebugLoc();
}

std::string ReportFuncCall::getEndUserMessage() const {
  return "This function call cannot be handled. "
         "Try to inline it.";
}

bool ReportFuncCall::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::FuncCall;
}

//===----------------------------------------------------------------------===//
// ReportNonSimpleMemoryAccess

ReportNonSimpleMemoryAccess::ReportNonSimpleMemoryAccess(Instruction *Inst)
    : ReportOther(RejectReasonKind::NonSimpleMemoryAccess), Inst(Inst) {}

std::string ReportNonSimpleMemoryAccess::getRemarkName() const {
  return "NonSimpleMemoryAccess";
}

const Value *ReportNonSimpleMemoryAccess::getRemarkBB() const {
  return Inst->getParent();
}

std::string ReportNonSimpleMemoryAccess::getMessage() const {
  return "Non-simple memory access: " + *Inst;
}

const DebugLoc &ReportNonSimpleMemoryAccess::getDebugLoc() const {
  return Inst->getDebugLoc();
}

std::string ReportNonSimpleMemoryAccess::getEndUserMessage() const {
  return "Volatile memory accesses or memory accesses for atomic types "
         "are not supported.";
}

bool ReportNonSimpleMemoryAccess::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::NonSimpleMemoryAccess;
}

//===----------------------------------------------------------------------===//
// ReportAlias.

ReportAlias::ReportAlias(Instruction *Inst, AliasSet &AS)
    : RejectReason(RejectReasonKind::Alias), Inst(Inst) {
  for (const auto &I : AS)
    Pointers.push_back(I.getValue());
}

std::string ReportAlias::formatInvalidAlias(std::string Prefix,
                                            std::string Suffix) const {
  std::string Message;
  raw_string_ostream OS(Message);

  OS << Prefix;

  for (PointerSnapshotTy::const_iterator PI = Pointers.begin(),
                                         PE = Pointers.end();
       ;) {
    const Value *V = *PI;
    assert(V && "Diagnostic info does not match found LLVM-IR anymore.");

    if (V->getName().empty())
      OS << "\" <unknown> \"";
    else
      OS << "\"" << V->getName() << "\"";

    ++PI;

    if (PI != PE)
      OS << ", ";
    else
      break;
  }

  OS << Suffix;

  return OS.str();
}

std::string ReportAlias::getRemarkName() const { return "Alias"; }

const Value *ReportAlias::getRemarkBB() const { return Inst->getParent(); }

std::string ReportAlias::getMessage() const {
  return formatInvalidAlias("Possible aliasing: ");
}

std::string ReportAlias::getEndUserMessage() const {
  return formatInvalidAlias("Accesses to the arrays ",
                            " may access the same memory.");
}

const DebugLoc &ReportAlias::getDebugLoc() const { return Inst->getDebugLoc(); }

bool ReportAlias::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::Alias;
}

//===----------------------------------------------------------------------===//
// ReportOther.

std::string ReportOther::getRemarkName() const { return "UnknownRejectReason"; }

std::string ReportOther::getMessage() const { return "Unknown reject reason"; }

ReportOther::ReportOther(const RejectReasonKind K) : RejectReason(K) {}

bool ReportOther::classof(const RejectReason *RR) {
  return RR->getKind() >= RejectReasonKind::Other &&
         RR->getKind() <= RejectReasonKind::LastOther;
}

//===----------------------------------------------------------------------===//
// ReportIntToPtr.
ReportIntToPtr::ReportIntToPtr(Instruction *BaseValue)
    : ReportOther(RejectReasonKind::IntToPtr), BaseValue(BaseValue) {}

std::string ReportIntToPtr::getRemarkName() const { return "IntToPtr"; }

const Value *ReportIntToPtr::getRemarkBB() const {
  return BaseValue->getParent();
}

std::string ReportIntToPtr::getMessage() const {
  return "Find bad intToptr prt: " + *BaseValue;
}

const DebugLoc &ReportIntToPtr::getDebugLoc() const {
  return BaseValue->getDebugLoc();
}

bool ReportIntToPtr::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::IntToPtr;
}

//===----------------------------------------------------------------------===//
// ReportAlloca.

ReportAlloca::ReportAlloca(Instruction *Inst)
    : ReportOther(RejectReasonKind::Alloca), Inst(Inst) {}

std::string ReportAlloca::getRemarkName() const { return "Alloca"; }

const Value *ReportAlloca::getRemarkBB() const { return Inst->getParent(); }

std::string ReportAlloca::getMessage() const {
  return "Alloca instruction: " + *Inst;
}

const DebugLoc &ReportAlloca::getDebugLoc() const {
  return Inst->getDebugLoc();
}

bool ReportAlloca::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::Alloca;
}

//===----------------------------------------------------------------------===//
// ReportUnknownInst.

ReportUnknownInst::ReportUnknownInst(Instruction *Inst)
    : ReportOther(RejectReasonKind::UnknownInst), Inst(Inst) {}

std::string ReportUnknownInst::getRemarkName() const { return "UnknownInst"; }

const Value *ReportUnknownInst::getRemarkBB() const {
  return Inst->getParent();
}

std::string ReportUnknownInst::getMessage() const {
  return "Unknown instruction: " + *Inst;
}

const DebugLoc &ReportUnknownInst::getDebugLoc() const {
  return Inst->getDebugLoc();
}

bool ReportUnknownInst::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::UnknownInst;
}

//===----------------------------------------------------------------------===//
// ReportEntry.

ReportEntry::ReportEntry(BasicBlock *BB)
    : ReportOther(RejectReasonKind::Entry), BB(BB) {}

std::string ReportEntry::getRemarkName() const { return "Entry"; }

const Value *ReportEntry::getRemarkBB() const { return BB; }

std::string ReportEntry::getMessage() const {
  return "Region containing entry block of function is invalid!";
}

std::string ReportEntry::getEndUserMessage() const {
  return "Scop contains function entry (not yet supported).";
}

const DebugLoc &ReportEntry::getDebugLoc() const {
  return BB->getTerminator()->getDebugLoc();
}

bool ReportEntry::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::Entry;
}

//===----------------------------------------------------------------------===//
// ReportUnprofitable.

ReportUnprofitable::ReportUnprofitable(Region *R)
    : ReportOther(RejectReasonKind::Unprofitable), R(R) {}

std::string ReportUnprofitable::getRemarkName() const { return "Unprofitable"; }

const Value *ReportUnprofitable::getRemarkBB() const { return R->getEntry(); }

std::string ReportUnprofitable::getMessage() const {
  return "Region can not profitably be optimized!";
}

std::string ReportUnprofitable::getEndUserMessage() const {
  return "No profitable polyhedral optimization found";
}

const DebugLoc &ReportUnprofitable::getDebugLoc() const {
  for (const BasicBlock *BB : R->blocks())
    for (const Instruction &Inst : *BB)
      if (const DebugLoc &DL = Inst.getDebugLoc())
        return DL;

  return R->getEntry()->getTerminator()->getDebugLoc();
}

bool ReportUnprofitable::classof(const RejectReason *RR) {
  return RR->getKind() == RejectReasonKind::Unprofitable;
}
} // namespace polly
