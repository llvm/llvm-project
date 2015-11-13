//=== ScopDetectionDiagnostic.cpp - Error diagnostics --------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "polly/Support/ScopLocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"

#define DEBUG_TYPE "polly-detect"
#include "llvm/Support/Debug.h"

#include <string>

#define BADSCOP_STAT(NAME, DESC)                                               \
  STATISTIC(Bad##NAME##ForScop, "Number of bad regions for Scop: " DESC)

BADSCOP_STAT(CFG, "CFG too complex");
BADSCOP_STAT(LoopBound, "Loop bounds can not be computed");
BADSCOP_STAT(FuncCall, "Function call with side effects appeared");
BADSCOP_STAT(AffFunc, "Expression not affine");
BADSCOP_STAT(Alias, "Found base address alias");
BADSCOP_STAT(SimpleLoop, "Loop not in -loop-simplify form");
BADSCOP_STAT(Other, "Others");

namespace polly {
/// @brief Small string conversion via raw_string_stream.
template <typename T> std::string operator+(Twine LHS, const T &RHS) {
  std::string Buf;
  raw_string_ostream fmt(Buf);
  fmt << RHS;
  fmt.flush();

  return LHS.concat(Buf).str();
}
}

namespace llvm {
// @brief Lexicographic order on (line, col) of our debug locations.
static bool operator<(const llvm::DebugLoc &LHS, const llvm::DebugLoc &RHS) {
  return LHS.getLine() < RHS.getLine() ||
         (LHS.getLine() == RHS.getLine() && LHS.getCol() < RHS.getCol());
}
}

namespace polly {
void getDebugLocations(const Region *R, DebugLoc &Begin, DebugLoc &End) {
  for (const BasicBlock *BB : R->blocks())
    for (const Instruction &Inst : *BB) {
      DebugLoc DL = Inst.getDebugLoc();
      if (!DL)
        continue;

      Begin = Begin ? std::min(Begin, DL) : DL;
      End = End ? std::max(End, DL) : DL;
    }
}

void emitRejectionRemarks(const llvm::Function &F, const RejectLog &Log) {
  LLVMContext &Ctx = F.getContext();

  const Region *R = Log.region();
  DebugLoc Begin, End;

  getDebugLocations(R, Begin, End);

  emitOptimizationRemarkMissed(
      Ctx, DEBUG_TYPE, F, Begin,
      "The following errors keep this region from being a Scop.");

  for (RejectReasonPtr RR : Log) {
    if (const DebugLoc &Loc = RR->getDebugLoc())
      emitOptimizationRemarkMissed(Ctx, DEBUG_TYPE, F, Loc,
                                   RR->getEndUserMessage());
  }

  emitOptimizationRemarkMissed(Ctx, DEBUG_TYPE, F, End,
                               "Invalid Scop candidate ends here.");
}

//===----------------------------------------------------------------------===//
// RejectReason.
const DebugLoc RejectReason::Unknown = DebugLoc();

const llvm::DebugLoc &RejectReason::getDebugLoc() const {
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

ReportCFG::ReportCFG(const RejectReasonKind K) : RejectReason(K) {
  ++BadCFGForScop;
}

bool ReportCFG::classof(const RejectReason *RR) {
  return RR->getKind() >= rrkCFG && RR->getKind() <= rrkLastCFG;
}

//===----------------------------------------------------------------------===//
// ReportInvalidTerminator.

std::string ReportInvalidTerminator::getMessage() const {
  return ("Invalid instruction terminates BB: " + BB->getName()).str();
}

const DebugLoc &ReportInvalidTerminator::getDebugLoc() const {
  return BB->getTerminator()->getDebugLoc();
}

bool ReportInvalidTerminator::classof(const RejectReason *RR) {
  return RR->getKind() == rrkInvalidTerminator;
}

//===----------------------------------------------------------------------===//
// ReportAffFunc.

ReportAffFunc::ReportAffFunc(const RejectReasonKind K, const Instruction *Inst)
    : RejectReason(K), Inst(Inst) {
  ++BadAffFuncForScop;
}

bool ReportAffFunc::classof(const RejectReason *RR) {
  return RR->getKind() >= rrkAffFunc && RR->getKind() <= rrkLastAffFunc;
}

//===----------------------------------------------------------------------===//
// ReportUndefCond.

std::string ReportUndefCond::getMessage() const {
  return ("Condition based on 'undef' value in BB: " + BB->getName()).str();
}

bool ReportUndefCond::classof(const RejectReason *RR) {
  return RR->getKind() == rrkUndefCond;
}

//===----------------------------------------------------------------------===//
// ReportInvalidCond.

std::string ReportInvalidCond::getMessage() const {
  return ("Condition in BB '" + BB->getName()).str() +
         "' neither constant nor an icmp instruction";
}

bool ReportInvalidCond::classof(const RejectReason *RR) {
  return RR->getKind() == rrkInvalidCond;
}

//===----------------------------------------------------------------------===//
// ReportUnsignedCond.

std::string ReportUnsignedCond::getMessage() const {
  return ("Condition in BB '" + BB->getName()).str() +
         "' performs a comparision on (not yet supported) unsigned integers.";
}

std::string ReportUnsignedCond::getEndUserMessage() const {
  return "Unsupported comparision on unsigned integers encountered";
}

bool ReportUnsignedCond::classof(const RejectReason *RR) {
  return RR->getKind() == rrkUnsignedCond;
}

//===----------------------------------------------------------------------===//
// ReportUndefOperand.

std::string ReportUndefOperand::getMessage() const {
  return ("undef operand in branch at BB: " + BB->getName()).str();
}

bool ReportUndefOperand::classof(const RejectReason *RR) {
  return RR->getKind() == rrkUndefOperand;
}

//===----------------------------------------------------------------------===//
// ReportNonAffBranch.

std::string ReportNonAffBranch::getMessage() const {
  return ("Non affine branch in BB '" + BB->getName()).str() + "' with LHS: " +
         *LHS + " and RHS: " + *RHS;
}

bool ReportNonAffBranch::classof(const RejectReason *RR) {
  return RR->getKind() == rrkNonAffBranch;
}

//===----------------------------------------------------------------------===//
// ReportNoBasePtr.

std::string ReportNoBasePtr::getMessage() const { return "No base pointer"; }

bool ReportNoBasePtr::classof(const RejectReason *RR) {
  return RR->getKind() == rrkNoBasePtr;
}

//===----------------------------------------------------------------------===//
// ReportUndefBasePtr.

std::string ReportUndefBasePtr::getMessage() const {
  return "Undefined base pointer";
}

bool ReportUndefBasePtr::classof(const RejectReason *RR) {
  return RR->getKind() == rrkUndefBasePtr;
}

//===----------------------------------------------------------------------===//
// ReportVariantBasePtr.

std::string ReportVariantBasePtr::getMessage() const {
  return "Base address not invariant in current region:" + *BaseValue;
}

std::string ReportVariantBasePtr::getEndUserMessage() const {
  return "The base address of this array is not invariant inside the loop";
}

bool ReportVariantBasePtr::classof(const RejectReason *RR) {
  return RR->getKind() == rrkVariantBasePtr;
}

//===----------------------------------------------------------------------===//
// ReportDifferentArrayElementSize

std::string ReportDifferentArrayElementSize::getMessage() const {
  return "Access to one array through data types of different size";
}

bool ReportDifferentArrayElementSize::classof(const RejectReason *RR) {
  return RR->getKind() == rrkDifferentElementSize;
}

std::string ReportDifferentArrayElementSize::getEndUserMessage() const {
  llvm::StringRef BaseName = BaseValue->getName();
  std::string Name = (BaseName.size() > 0) ? BaseName : "UNKNOWN";
  return "The array \"" + Name + "\" is accessed through elements that differ "
                                 "in size";
}

//===----------------------------------------------------------------------===//
// ReportNonAffineAccess.

std::string ReportNonAffineAccess::getMessage() const {
  return "Non affine access function: " + *AccessFunction;
}

bool ReportNonAffineAccess::classof(const RejectReason *RR) {
  return RR->getKind() == rrkNonAffineAccess;
}

std::string ReportNonAffineAccess::getEndUserMessage() const {
  llvm::StringRef BaseName = BaseValue->getName();
  std::string Name = (BaseName.size() > 0) ? BaseName : "UNKNOWN";
  return "The array subscript of \"" + Name + "\" is not affine";
}

//===----------------------------------------------------------------------===//
// ReportLoopBound.

ReportLoopBound::ReportLoopBound(Loop *L, const SCEV *LoopCount)
    : RejectReason(rrkLoopBound), L(L), LoopCount(LoopCount),
      Loc(L->getStartLoc()) {
  ++BadLoopBoundForScop;
}

std::string ReportLoopBound::getMessage() const {
  return "Non affine loop bound '" + *LoopCount + "' in loop: " +
         L->getHeader()->getName();
}

const DebugLoc &ReportLoopBound::getDebugLoc() const { return Loc; }

bool ReportLoopBound::classof(const RejectReason *RR) {
  return RR->getKind() == rrkLoopBound;
}

std::string ReportLoopBound::getEndUserMessage() const {
  return "Failed to derive an affine function from the loop bounds.";
}

//===----------------------------------------------------------------------===//
// ReportFuncCall.

ReportFuncCall::ReportFuncCall(Instruction *Inst)
    : RejectReason(rrkFuncCall), Inst(Inst) {
  ++BadFuncCallForScop;
}

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
  return RR->getKind() == rrkFuncCall;
}

//===----------------------------------------------------------------------===//
// ReportNonSimpleMemoryAccess

ReportNonSimpleMemoryAccess::ReportNonSimpleMemoryAccess(Instruction *Inst)
    : ReportOther(rrkNonSimpleMemoryAccess), Inst(Inst) {}

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
  return RR->getKind() == rrkNonSimpleMemoryAccess;
}

//===----------------------------------------------------------------------===//
// ReportAlias.

ReportAlias::ReportAlias(Instruction *Inst, AliasSet &AS)
    : RejectReason(rrkAlias), Inst(Inst) {

  for (const auto &I : AS)
    Pointers.push_back(I.getValue());

  ++BadAliasForScop;
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

    if (V->getName().size() == 0)
      OS << "\"" << *V << "\"";
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

std::string ReportAlias::getMessage() const {
  return formatInvalidAlias("Possible aliasing: ");
}

std::string ReportAlias::getEndUserMessage() const {
  return formatInvalidAlias("Accesses to the arrays ",
                            " may access the same memory.");
}

const DebugLoc &ReportAlias::getDebugLoc() const { return Inst->getDebugLoc(); }

bool ReportAlias::classof(const RejectReason *RR) {
  return RR->getKind() == rrkAlias;
}

//===----------------------------------------------------------------------===//
// ReportSimpleLoop.

ReportSimpleLoop::ReportSimpleLoop() : RejectReason(rrkSimpleLoop) {
  ++BadSimpleLoopForScop;
}

std::string ReportSimpleLoop::getMessage() const {
  return "Loop not in simplify form is invalid!";
}

bool ReportSimpleLoop::classof(const RejectReason *RR) {
  return RR->getKind() == rrkSimpleLoop;
}

//===----------------------------------------------------------------------===//
// ReportOther.

std::string ReportOther::getMessage() const { return "Unknown reject reason"; }

ReportOther::ReportOther(const RejectReasonKind K) : RejectReason(K) {
  ++BadOtherForScop;
}

bool ReportOther::classof(const RejectReason *RR) {
  return RR->getKind() >= rrkOther && RR->getKind() <= rrkLastOther;
}

//===----------------------------------------------------------------------===//
// ReportIntToPtr.
ReportIntToPtr::ReportIntToPtr(Instruction *BaseValue)
    : ReportOther(rrkIntToPtr), BaseValue(BaseValue) {}

std::string ReportIntToPtr::getMessage() const {
  return "Find bad intToptr prt: " + *BaseValue;
}

const DebugLoc &ReportIntToPtr::getDebugLoc() const {
  return BaseValue->getDebugLoc();
}

bool ReportIntToPtr::classof(const RejectReason *RR) {
  return RR->getKind() == rrkIntToPtr;
}

//===----------------------------------------------------------------------===//
// ReportAlloca.

ReportAlloca::ReportAlloca(Instruction *Inst)
    : ReportOther(rrkAlloca), Inst(Inst) {}

std::string ReportAlloca::getMessage() const {
  return "Alloca instruction: " + *Inst;
}

const DebugLoc &ReportAlloca::getDebugLoc() const {
  return Inst->getDebugLoc();
}

bool ReportAlloca::classof(const RejectReason *RR) {
  return RR->getKind() == rrkAlloca;
}

//===----------------------------------------------------------------------===//
// ReportUnknownInst.

ReportUnknownInst::ReportUnknownInst(Instruction *Inst)
    : ReportOther(rrkUnknownInst), Inst(Inst) {}

std::string ReportUnknownInst::getMessage() const {
  return "Unknown instruction: " + *Inst;
}

const DebugLoc &ReportUnknownInst::getDebugLoc() const {
  return Inst->getDebugLoc();
}

bool ReportUnknownInst::classof(const RejectReason *RR) {
  return RR->getKind() == rrkUnknownInst;
}

//===----------------------------------------------------------------------===//
// ReportEntry.
ReportEntry::ReportEntry(BasicBlock *BB) : ReportOther(rrkEntry), BB(BB) {}

std::string ReportEntry::getMessage() const {
  return "Region containing entry block of function is invalid!";
}

const DebugLoc &ReportEntry::getDebugLoc() const {
  return BB->getTerminator()->getDebugLoc();
}

bool ReportEntry::classof(const RejectReason *RR) {
  return RR->getKind() == rrkEntry;
}

//===----------------------------------------------------------------------===//
// ReportUnprofitable.
ReportUnprofitable::ReportUnprofitable(Region *R)
    : ReportOther(rrkUnprofitable), R(R) {}

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
  return RR->getKind() == rrkUnprofitable;
}
} // namespace polly
