//===- StackSafetyAnalysis.cpp - Stack memory safety analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>

using namespace llvm;

#define DEBUG_TYPE "stack-safety"

static cl::opt<int> StackSafetyMaxIterations("stack-safety-max-iterations",
                                             cl::init(20), cl::Hidden);

static cl::opt<int> StackSafetyPrint("stack-safety-print", cl::init(0),
                                     cl::Hidden);

namespace {

/// Rewrite an SCEV expression for a memory access address to an expression that
/// represents offset from the given alloca.
class AllocaOffsetRewriter : public SCEVRewriteVisitor<AllocaOffsetRewriter> {
  const Value *AllocaPtr;

public:
  AllocaOffsetRewriter(ScalarEvolution &SE, const Value *AllocaPtr)
      : SCEVRewriteVisitor(SE), AllocaPtr(AllocaPtr) {}

  const SCEV *visitUnknown(const SCEVUnknown *Expr) {
    // FIXME: look through one or several levels of definitions?
    // This can be inttoptr(AllocaPtr) and SCEV would not unwrap
    // it for us.
    if (Expr->getValue() == AllocaPtr)
      return SE.getZero(Expr->getType());
    return Expr;
  }
};

/// Describes use of address in as a function call argument.
struct PassAsArgInfo {
  /// Function being called.
  const GlobalValue *Callee = nullptr;
  /// Index of argument which pass address.
  size_t ParamNo = 0;
  // Offset range of address from base address (alloca or calling function
  // argument).
  // Range should never set to empty-set, that is an invalid access range
  // that can cause empty-set to be propagated with ConstantRange::add
  ConstantRange Offset;
  PassAsArgInfo(const GlobalValue *Callee, size_t ParamNo, ConstantRange Offset)
      : Callee(Callee), ParamNo(ParamNo), Offset(Offset) {}

  StringRef getName() const { return Callee->getName(); }
};

raw_ostream &operator<<(raw_ostream &OS, const PassAsArgInfo &P) {
  return OS << "@" << P.getName() << "(arg" << P.ParamNo << ", " << P.Offset
            << ")";
}

/// Describe uses of address (alloca or parameter) inside of the function.
struct UseInfo {
  // Access range if the address (alloca or parameters).
  // It is allowed to be empty-set when there are no known accesses.
  ConstantRange Range;

  // List of calls which pass address as an argument.
  SmallVector<PassAsArgInfo, 4> Calls;

  explicit UseInfo(unsigned PointerSize) : Range{PointerSize, false} {}

  void updateRange(const ConstantRange &R) {
    assert(!R.isUpperSignWrapped());
    Range = Range.unionWith(R);
    assert(!Range.isUpperSignWrapped());
  }
};

raw_ostream &operator<<(raw_ostream &OS, const UseInfo &U) {
  OS << U.Range;
  for (auto &Call : U.Calls)
    OS << ", " << Call;
  return OS;
}

// Check if we should bailout for such ranges.
bool isUnsafe(const ConstantRange &R) {
  return R.isEmptySet() || R.isFullSet() || R.isUpperSignWrapped();
}

/// Calculate the allocation size of a given alloca. Returns empty range
// in case of confution.
ConstantRange getStaticAllocaSizeRange(const AllocaInst &AI) {
  const DataLayout &DL = AI.getModule()->getDataLayout();
  TypeSize TS = DL.getTypeAllocSize(AI.getAllocatedType());
  unsigned PointerSize = DL.getMaxPointerSizeInBits();
  // Fallback to empty range for alloca size.
  ConstantRange R = ConstantRange::getEmpty(PointerSize);
  if (TS.isScalable())
    return R;
  APInt APSize(PointerSize, TS.getFixedSize(), true);
  if (APSize.isNonPositive())
    return R;
  if (AI.isArrayAllocation()) {
    const auto *C = dyn_cast<ConstantInt>(AI.getArraySize());
    if (!C)
      return R;
    bool Overflow = false;
    APInt Mul = C->getValue();
    if (Mul.isNonPositive())
      return R;
    Mul = Mul.sextOrTrunc(PointerSize);
    APSize = APSize.smul_ov(Mul, Overflow);
    if (Overflow)
      return R;
  }
  R = ConstantRange(APInt::getNullValue(PointerSize), APSize);
  assert(!isUnsafe(R));
  return R;
}

struct FunctionInfo {
  SmallVector<UseInfo, 4> Allocas;
  SmallVector<UseInfo, 4> Params;
  // TODO: describe return value as depending on one or more of its arguments.

  // StackSafetyDataFlowAnalysis counter stored here for faster access.
  int UpdateCount = 0;

  void print(raw_ostream &O, StringRef Name, const Function *F) const {
    // TODO: Consider different printout format after
    // StackSafetyDataFlowAnalysis. Calls and parameters are irrelevant then.
    O << "  @" << Name << ((F && F->isDSOLocal()) ? "" : " dso_preemptable")
      << ((F && F->isInterposable()) ? " interposable" : "") << "\n";

    O << "    args uses:\n";
    size_t Pos = 0;
    for (auto &P : Params) {
      StringRef Name = "<N/A>";
      if (F)
        Name = F->getArg(Pos)->getName();
      O << "      " << Name << "[]: " << P << "\n";
      ++Pos;
    }

    O << "    allocas uses:\n";
    if (F) {
      size_t Pos = 0;
      for (auto &I : instructions(F)) {
        if (const AllocaInst *AI = dyn_cast<AllocaInst>(&I)) {
          auto &AS = Allocas[Pos];
          O << "      " << AI->getName() << "["
            << getStaticAllocaSizeRange(*AI).getUpper() << "]: " << AS << "\n";
          ++Pos;
        }
      }
    } else {
      assert(Allocas.empty());
    }
  }
};

using GVToSSI = std::map<const GlobalValue *, FunctionInfo>;

} // namespace

struct StackSafetyInfo::InfoTy {
  FunctionInfo Info;
};

struct StackSafetyGlobalInfo::InfoTy {
  GVToSSI Info;
  SmallPtrSet<const AllocaInst *, 8> SafeAllocas;
};

namespace {

class StackSafetyLocalAnalysis {
  Function &F;
  const DataLayout &DL;
  ScalarEvolution &SE;
  unsigned PointerSize = 0;

  const ConstantRange UnknownRange;

  ConstantRange offsetFrom(Value *Addr, Value *Base);
  ConstantRange getAccessRange(Value *Addr, Value *Base,
                               ConstantRange SizeRange);
  ConstantRange getAccessRange(Value *Addr, Value *Base, TypeSize Size);
  ConstantRange getMemIntrinsicAccessRange(const MemIntrinsic *MI, const Use &U,
                                           Value *Base);

  bool analyzeAllUses(Value *Ptr, UseInfo &AS);

public:
  StackSafetyLocalAnalysis(Function &F, ScalarEvolution &SE)
      : F(F), DL(F.getParent()->getDataLayout()), SE(SE),
        PointerSize(DL.getPointerSizeInBits()),
        UnknownRange(PointerSize, true) {}

  // Run the transformation on the associated function.
  FunctionInfo run();
};

ConstantRange StackSafetyLocalAnalysis::offsetFrom(Value *Addr, Value *Base) {
  if (!SE.isSCEVable(Addr->getType()))
    return UnknownRange;

  AllocaOffsetRewriter Rewriter(SE, Base);
  const SCEV *Expr = Rewriter.visit(SE.getSCEV(Addr));
  ConstantRange Offset = SE.getSignedRange(Expr);
  if (isUnsafe(Offset))
    return UnknownRange;
  return Offset.sextOrTrunc(PointerSize);
}

ConstantRange
StackSafetyLocalAnalysis::getAccessRange(Value *Addr, Value *Base,
                                         ConstantRange SizeRange) {
  // Zero-size loads and stores do not access memory.
  if (SizeRange.isEmptySet())
    return ConstantRange::getEmpty(PointerSize);
  assert(!isUnsafe(SizeRange));

  ConstantRange Offsets = offsetFrom(Addr, Base);
  if (isUnsafe(Offsets))
    return UnknownRange;

  if (Offsets.signedAddMayOverflow(SizeRange) !=
      ConstantRange::OverflowResult::NeverOverflows)
    return UnknownRange;
  Offsets = Offsets.add(SizeRange);
  if (isUnsafe(Offsets))
    return UnknownRange;
  return Offsets;
}

ConstantRange StackSafetyLocalAnalysis::getAccessRange(Value *Addr, Value *Base,
                                                       TypeSize Size) {
  if (Size.isScalable())
    return UnknownRange;
  APInt APSize(PointerSize, Size.getFixedSize(), true);
  if (APSize.isNegative())
    return UnknownRange;
  return getAccessRange(
      Addr, Base, ConstantRange(APInt::getNullValue(PointerSize), APSize));
}

ConstantRange StackSafetyLocalAnalysis::getMemIntrinsicAccessRange(
    const MemIntrinsic *MI, const Use &U, Value *Base) {
  if (const auto *MTI = dyn_cast<MemTransferInst>(MI)) {
    if (MTI->getRawSource() != U && MTI->getRawDest() != U)
      return ConstantRange::getEmpty(PointerSize);
  } else {
    if (MI->getRawDest() != U)
      return ConstantRange::getEmpty(PointerSize);
  }

  auto *CalculationTy = IntegerType::getIntNTy(SE.getContext(), PointerSize);
  if (!SE.isSCEVable(MI->getLength()->getType()))
    return UnknownRange;

  const SCEV *Expr =
      SE.getTruncateOrZeroExtend(SE.getSCEV(MI->getLength()), CalculationTy);
  ConstantRange Sizes = SE.getSignedRange(Expr);
  if (Sizes.getUpper().isNegative() || isUnsafe(Sizes))
    return UnknownRange;
  Sizes = Sizes.sextOrTrunc(PointerSize);
  ConstantRange SizeRange(APInt::getNullValue(PointerSize),
                          Sizes.getUpper() - 1);
  return getAccessRange(U, Base, SizeRange);
}

/// The function analyzes all local uses of Ptr (alloca or argument) and
/// calculates local access range and all function calls where it was used.
bool StackSafetyLocalAnalysis::analyzeAllUses(Value *Ptr, UseInfo &US) {
  SmallPtrSet<const Value *, 16> Visited;
  SmallVector<const Value *, 8> WorkList;
  WorkList.push_back(Ptr);

  // A DFS search through all uses of the alloca in bitcasts/PHI/GEPs/etc.
  while (!WorkList.empty()) {
    const Value *V = WorkList.pop_back_val();
    for (const Use &UI : V->uses()) {
      const auto *I = cast<const Instruction>(UI.getUser());
      assert(V == UI.get());

      switch (I->getOpcode()) {
      case Instruction::Load: {
        US.updateRange(
            getAccessRange(UI, Ptr, DL.getTypeStoreSize(I->getType())));
        break;
      }

      case Instruction::VAArg:
        // "va-arg" from a pointer is safe.
        break;
      case Instruction::Store: {
        if (V == I->getOperand(0)) {
          // Stored the pointer - conservatively assume it may be unsafe.
          US.updateRange(UnknownRange);
          return false;
        }
        US.updateRange(getAccessRange(
            UI, Ptr, DL.getTypeStoreSize(I->getOperand(0)->getType())));
        break;
      }

      case Instruction::Ret:
        // Information leak.
        // FIXME: Process parameters correctly. This is a leak only if we return
        // alloca.
        US.updateRange(UnknownRange);
        return false;

      case Instruction::Call:
      case Instruction::Invoke: {
        const auto &CB = cast<CallBase>(*I);

        if (I->isLifetimeStartOrEnd())
          break;

        if (const MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I)) {
          US.updateRange(getMemIntrinsicAccessRange(MI, UI, Ptr));
          break;
        }

        // FIXME: consult devirt?
        // Do not follow aliases, otherwise we could inadvertently follow
        // dso_preemptable aliases or aliases with interposable linkage.
        const GlobalValue *Callee =
            dyn_cast<GlobalValue>(CB.getCalledOperand()->stripPointerCasts());
        if (!Callee) {
          US.updateRange(UnknownRange);
          return false;
        }

        assert(isa<Function>(Callee) || isa<GlobalAlias>(Callee));

        int Found = 0;
        for (size_t ArgNo = 0; ArgNo < CB.getNumArgOperands(); ++ArgNo) {
          if (CB.getArgOperand(ArgNo) == V) {
            ++Found;
            US.Calls.emplace_back(Callee, ArgNo, offsetFrom(UI, Ptr));
          }
        }
        if (!Found) {
          US.updateRange(UnknownRange);
          return false;
        }

        break;
      }

      default:
        if (Visited.insert(I).second)
          WorkList.push_back(cast<const Instruction>(I));
      }
    }
  }

  return true;
}

FunctionInfo StackSafetyLocalAnalysis::run() {
  FunctionInfo Info;
  assert(!F.isDeclaration() &&
         "Can't run StackSafety on a function declaration");

  LLVM_DEBUG(dbgs() << "[StackSafety] " << F.getName() << "\n");

  for (auto &I : instructions(F)) {
    if (auto *AI = dyn_cast<AllocaInst>(&I)) {
      Info.Allocas.emplace_back(PointerSize);
      UseInfo &AS = Info.Allocas.back();
      analyzeAllUses(AI, AS);
    }
  }

  for (Argument &A : make_range(F.arg_begin(), F.arg_end())) {
    Info.Params.emplace_back(PointerSize);
    UseInfo &PS = Info.Params.back();
    analyzeAllUses(&A, PS);
  }

  LLVM_DEBUG(Info.print(dbgs(), F.getName(), &F));
  LLVM_DEBUG(dbgs() << "[StackSafety] done\n");
  return Info;
}

class StackSafetyDataFlowAnalysis {
  using FunctionMap = std::map<const GlobalValue *, FunctionInfo>;

  FunctionMap Functions;
  const ConstantRange UnknownRange;

  // Callee-to-Caller multimap.
  DenseMap<const GlobalValue *, SmallVector<const GlobalValue *, 4>> Callers;
  SetVector<const GlobalValue *> WorkList;


  bool updateOneUse(UseInfo &US, bool UpdateToFullSet);
  void updateOneNode(const GlobalValue *Callee, FunctionInfo &FS);
  void updateOneNode(const GlobalValue *Callee) {
    updateOneNode(Callee, Functions.find(Callee)->second);
  }
  void updateAllNodes() {
    for (auto &F : Functions)
      updateOneNode(F.first, F.second);
  }
  void runDataFlow();
#ifndef NDEBUG
  void verifyFixedPoint();
#endif

public:
  StackSafetyDataFlowAnalysis(uint32_t PointerBitWidth, FunctionMap Functions)
      : Functions(std::move(Functions)),
        UnknownRange(ConstantRange::getFull(PointerBitWidth)) {}

  const FunctionMap &run();

  ConstantRange getArgumentAccessRange(const GlobalValue *Callee,
                                       unsigned ParamNo,
                                       const ConstantRange &Offsets) const;
};

ConstantRange StackSafetyDataFlowAnalysis::getArgumentAccessRange(
    const GlobalValue *Callee, unsigned ParamNo,
    const ConstantRange &Offsets) const {
  auto IT = Functions.find(Callee);
  // Unknown callee (outside of LTO domain or an indirect call).
  if (IT == Functions.end())
    return UnknownRange;
  const FunctionInfo &FS = IT->second;
  if (ParamNo >= FS.Params.size()) // possibly vararg
    return UnknownRange;
  auto &Access = FS.Params[ParamNo].Range;
  if (Access.isEmptySet())
    return Access;
  if (Access.isFullSet() || Offsets.isFullSet())
    return UnknownRange;
  if (Offsets.signedAddMayOverflow(Access) !=
      ConstantRange::OverflowResult::NeverOverflows)
    return UnknownRange;
  return Access.add(Offsets);
}

bool StackSafetyDataFlowAnalysis::updateOneUse(UseInfo &US,
                                               bool UpdateToFullSet) {
  bool Changed = false;
  for (auto &CS : US.Calls) {
    assert(!CS.Offset.isEmptySet() &&
           "Param range can't be empty-set, invalid offset range");

    ConstantRange CalleeRange =
        getArgumentAccessRange(CS.Callee, CS.ParamNo, CS.Offset);
    if (!US.Range.contains(CalleeRange)) {
      Changed = true;
      if (UpdateToFullSet)
        US.Range = UnknownRange;
      else
        US.Range = US.Range.unionWith(CalleeRange);
    }
  }
  return Changed;
}

void StackSafetyDataFlowAnalysis::updateOneNode(const GlobalValue *Callee,
                                                FunctionInfo &FS) {
  bool UpdateToFullSet = FS.UpdateCount > StackSafetyMaxIterations;
  bool Changed = false;
  for (auto &PS : FS.Params)
    Changed |= updateOneUse(PS, UpdateToFullSet);

  if (Changed) {
    LLVM_DEBUG(dbgs() << "=== update [" << FS.UpdateCount
                      << (UpdateToFullSet ? ", full-set" : "") << "] " << &FS
                      << "\n");
    // Callers of this function may need updating.
    for (auto &CallerID : Callers[Callee])
      WorkList.insert(CallerID);

    ++FS.UpdateCount;
  }
}

void StackSafetyDataFlowAnalysis::runDataFlow() {
  Callers.clear();
  WorkList.clear();

  SmallVector<const GlobalValue *, 16> Callees;
  for (auto &F : Functions) {
    Callees.clear();
    FunctionInfo &FS = F.second;
    for (auto &PS : FS.Params)
      for (auto &CS : PS.Calls)
        Callees.push_back(CS.Callee);

    llvm::sort(Callees);
    Callees.erase(std::unique(Callees.begin(), Callees.end()), Callees.end());

    for (auto &Callee : Callees)
      Callers[Callee].push_back(F.first);
  }

  updateAllNodes();

  while (!WorkList.empty()) {
    const GlobalValue *Callee = WorkList.back();
    WorkList.pop_back();
    updateOneNode(Callee);
  }
}

#ifndef NDEBUG
void StackSafetyDataFlowAnalysis::verifyFixedPoint() {
  WorkList.clear();
  updateAllNodes();
  assert(WorkList.empty());
}
#endif

const StackSafetyDataFlowAnalysis::FunctionMap &
StackSafetyDataFlowAnalysis::run() {
  runDataFlow();
  LLVM_DEBUG(verifyFixedPoint());
  return Functions;
}

const Function *findCalleeInModule(const GlobalValue *GV) {
  while (GV) {
    if (GV->isInterposable() || !GV->isDSOLocal())
      return nullptr;
    if (const Function *F = dyn_cast<Function>(GV))
      return F;
    const GlobalAlias *A = dyn_cast<GlobalAlias>(GV);
    if (!A)
      return nullptr;
    GV = A->getBaseObject();
    if (GV == A)
      return nullptr;
  }
  return nullptr;
}

void resolveAllCalls(UseInfo &Use) {
  ConstantRange FullSet(Use.Range.getBitWidth(), true);
  for (auto &C : Use.Calls) {
    const Function *F = findCalleeInModule(C.Callee);
    if (F) {
      C.Callee = F;
      continue;
    }

    return Use.updateRange(FullSet);
  }
}

void resolveAllCalls(SmallVectorImpl<UseInfo> &Values) {
  for (auto &V : Values)
    resolveAllCalls(V);
}

GVToSSI createGlobalStackSafetyInfo(
    std::map<const GlobalValue *, FunctionInfo> Functions) {
  GVToSSI SSI;
  if (Functions.empty())
    return SSI;

  // FIXME: Simplify printing and remove copying here.
  auto Copy = Functions;

  for (auto &FI : Copy)
    resolveAllCalls(FI.second.Params);

  uint32_t PointerSize = Copy.begin()
                             ->first->getParent()
                             ->getDataLayout()
                             .getMaxPointerSizeInBits();
  StackSafetyDataFlowAnalysis SSDFA(PointerSize, std::move(Copy));

  for (auto &F : SSDFA.run()) {
    auto FI = F.second;
    size_t Pos = 0;
    auto &SrcF = Functions[F.first];
    for (auto &A : FI.Allocas) {
      resolveAllCalls(A);
      for (auto &C : A.Calls) {
        A.updateRange(
            SSDFA.getArgumentAccessRange(C.Callee, C.ParamNo, C.Offset));
      }
      // FIXME: This is needed only to preserve calls in print() results.
      A.Calls = SrcF.Allocas[Pos].Calls;
      ++Pos;
    }
    Pos = 0;
    for (auto &P : FI.Params) {
      P.Calls = SrcF.Params[Pos].Calls;
      ++Pos;
    }
    SSI[F.first] = std::move(FI);
  }

  return SSI;
}

} // end anonymous namespace

StackSafetyInfo::StackSafetyInfo() = default;

StackSafetyInfo::StackSafetyInfo(Function *F,
                                 std::function<ScalarEvolution &()> GetSE)
    : F(F), GetSE(GetSE) {}

StackSafetyInfo::StackSafetyInfo(StackSafetyInfo &&) = default;

StackSafetyInfo &StackSafetyInfo::operator=(StackSafetyInfo &&) = default;

StackSafetyInfo::~StackSafetyInfo() = default;

const StackSafetyInfo::InfoTy &StackSafetyInfo::getInfo() const {
  if (!Info) {
    StackSafetyLocalAnalysis SSLA(*F, GetSE());
    Info.reset(new InfoTy{SSLA.run()});
  }
  return *Info;
}

void StackSafetyInfo::print(raw_ostream &O) const {
  getInfo().Info.print(O, F->getName(), dyn_cast<Function>(F));
}

const StackSafetyGlobalInfo::InfoTy &StackSafetyGlobalInfo::getInfo() const {
  if (!Info) {
    std::map<const GlobalValue *, FunctionInfo> Functions;
    for (auto &F : M->functions()) {
      if (!F.isDeclaration()) {
        auto FI = GetSSI(F).getInfo().Info;
        Functions.emplace(&F, std::move(FI));
      }
    }
    Info.reset(
        new InfoTy{createGlobalStackSafetyInfo(std::move(Functions)), {}});
    for (auto &KV : Info->Info) {
      if (!KV.first->isDeclaration()) {
        size_t Pos = 0;
        // FIXME: Convert FunctionInfo::Allocas into map<AllocaInst*, UseInfo>
        // and do not rely on alloca index.
        for (auto &I : instructions(*cast<Function>(KV.first))) {
          if (const auto &AI = dyn_cast<AllocaInst>(&I)) {
            if (getStaticAllocaSizeRange(*AI).contains(
                    KV.second.Allocas[Pos].Range)) {
              Info->SafeAllocas.insert(AI);
            }
            ++Pos;
          }
        }
      }
    }
    if (StackSafetyPrint)
      print(errs());
  }
  return *Info;
}

StackSafetyGlobalInfo::StackSafetyGlobalInfo() = default;

StackSafetyGlobalInfo::StackSafetyGlobalInfo(
    Module *M, std::function<const StackSafetyInfo &(Function &F)> GetSSI)
    : M(M), GetSSI(GetSSI) {
  if (StackSafetyPrint > 1)
    getInfo();
}

StackSafetyGlobalInfo::StackSafetyGlobalInfo(StackSafetyGlobalInfo &&) =
    default;

StackSafetyGlobalInfo &
StackSafetyGlobalInfo::operator=(StackSafetyGlobalInfo &&) = default;

StackSafetyGlobalInfo::~StackSafetyGlobalInfo() = default;

bool StackSafetyGlobalInfo::isSafe(const AllocaInst &AI) const {
  const auto &Info = getInfo();
  return Info.SafeAllocas.find(&AI) != Info.SafeAllocas.end();
}

void StackSafetyGlobalInfo::print(raw_ostream &O) const {
  auto &SSI = getInfo().Info;
  if (SSI.empty())
    return;
  const Module &M = *SSI.begin()->first->getParent();
  for (auto &F : M.functions()) {
    if (!F.isDeclaration()) {
      SSI.find(&F)->second.print(O, F.getName(), &F);
      O << "\n";
    }
  }
}

LLVM_DUMP_METHOD void StackSafetyGlobalInfo::dump() const { print(dbgs()); }

AnalysisKey StackSafetyAnalysis::Key;

StackSafetyInfo StackSafetyAnalysis::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  return StackSafetyInfo(&F, [&AM, &F]() -> ScalarEvolution & {
    return AM.getResult<ScalarEvolutionAnalysis>(F);
  });
}

PreservedAnalyses StackSafetyPrinterPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  OS << "'Stack Safety Local Analysis' for function '" << F.getName() << "'\n";
  AM.getResult<StackSafetyAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}

char StackSafetyInfoWrapperPass::ID = 0;

StackSafetyInfoWrapperPass::StackSafetyInfoWrapperPass() : FunctionPass(ID) {
  initializeStackSafetyInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

void StackSafetyInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.setPreservesAll();
}

void StackSafetyInfoWrapperPass::print(raw_ostream &O, const Module *M) const {
  SSI.print(O);
}

bool StackSafetyInfoWrapperPass::runOnFunction(Function &F) {
  auto *SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  SSI = {&F, [SE]() -> ScalarEvolution & { return *SE; }};
  return false;
}

AnalysisKey StackSafetyGlobalAnalysis::Key;

StackSafetyGlobalInfo
StackSafetyGlobalAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  return {&M, [&FAM](Function &F) -> const StackSafetyInfo & {
            return FAM.getResult<StackSafetyAnalysis>(F);
          }};
}

PreservedAnalyses StackSafetyGlobalPrinterPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  OS << "'Stack Safety Analysis' for module '" << M.getName() << "'\n";
  AM.getResult<StackSafetyGlobalAnalysis>(M).print(OS);
  return PreservedAnalyses::all();
}

char StackSafetyGlobalInfoWrapperPass::ID = 0;

StackSafetyGlobalInfoWrapperPass::StackSafetyGlobalInfoWrapperPass()
    : ModulePass(ID) {
  initializeStackSafetyGlobalInfoWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

StackSafetyGlobalInfoWrapperPass::~StackSafetyGlobalInfoWrapperPass() = default;

void StackSafetyGlobalInfoWrapperPass::print(raw_ostream &O,
                                             const Module *M) const {
  SSGI.print(O);
}

void StackSafetyGlobalInfoWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<StackSafetyInfoWrapperPass>();
}

bool StackSafetyGlobalInfoWrapperPass::runOnModule(Module &M) {
  SSGI = {&M, [this](Function &F) -> const StackSafetyInfo & {
            return getAnalysis<StackSafetyInfoWrapperPass>(F).getResult();
          }};
  return false;
}

static const char LocalPassArg[] = "stack-safety-local";
static const char LocalPassName[] = "Stack Safety Local Analysis";
INITIALIZE_PASS_BEGIN(StackSafetyInfoWrapperPass, LocalPassArg, LocalPassName,
                      false, true)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(StackSafetyInfoWrapperPass, LocalPassArg, LocalPassName,
                    false, true)

static const char GlobalPassName[] = "Stack Safety Analysis";
INITIALIZE_PASS_BEGIN(StackSafetyGlobalInfoWrapperPass, DEBUG_TYPE,
                      GlobalPassName, false, true)
INITIALIZE_PASS_DEPENDENCY(StackSafetyInfoWrapperPass)
INITIALIZE_PASS_END(StackSafetyGlobalInfoWrapperPass, DEBUG_TYPE,
                    GlobalPassName, false, true)
