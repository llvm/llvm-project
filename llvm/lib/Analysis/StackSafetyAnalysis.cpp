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

/// Calculate the allocation size of a given alloca. Returns 0 if the
/// size can not be statically determined.
uint64_t getStaticAllocaAllocationSize(const AllocaInst *AI) {
  const DataLayout &DL = AI->getModule()->getDataLayout();
  TypeSize TS = DL.getTypeAllocSize(AI->getAllocatedType());
  if (TS.isScalable())
    return 0;
  uint64_t Size = TS.getFixedSize();
  if (AI->isArrayAllocation()) {
    auto C = dyn_cast<ConstantInt>(AI->getArraySize());
    if (!C)
      return 0;
    Size *= C->getZExtValue();
  }
  return Size;
}

/// Describes uses of allocas and parameters inside of a single function.
struct FunctionInfo {
  SmallVector<UseInfo, 4> Allocas;
  SmallVector<UseInfo, 4> Params;
  const GlobalValue *GV = nullptr;
  // TODO: describe return value as depending on one or more of its arguments.

  // StackSafetyDataFlowAnalysis counter stored here for faster access.
  int UpdateCount = 0;

  FunctionInfo() = default;
  FunctionInfo(const Function *F) : GV(F){};
  explicit FunctionInfo(const GlobalAlias *A);

  bool IsDSOLocal() const { return GV->isDSOLocal(); };

  bool IsInterposable() const { return GV->isInterposable(); };

  StringRef getName() const { return GV->getName(); }

  void print(raw_ostream &O, StringRef Name, const Function *F) const {
    // TODO: Consider different printout format after
    // StackSafetyDataFlowAnalysis. Calls and parameters are irrelevant then.
    O << "  @" << Name << (IsDSOLocal() ? "" : " dso_preemptable")
      << (IsInterposable() ? " interposable" : "") << "\n";

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
        if (auto AI = dyn_cast<AllocaInst>(&I)) {
          auto &AS = Allocas[Pos];
          O << "      " << AI->getName() << "["
            << getStaticAllocaAllocationSize(AI) << "]: " << AS << "\n";
          ++Pos;
        }
      }
    } else {
      assert(Allocas.empty());
    }
  }
};

FunctionInfo::FunctionInfo(const GlobalAlias *A) : GV(A) {
  unsigned PointerSize = A->getParent()->getDataLayout().getPointerSizeInBits();
  const GlobalObject *Aliasee = A->getBaseObject();
  const FunctionType *Type = cast<FunctionType>(Aliasee->getValueType());
  // 'Forward' all parameters to this alias to the aliasee
  for (unsigned ArgNo = 0; ArgNo < Type->getNumParams(); ArgNo++) {
    Params.emplace_back(PointerSize);
    UseInfo &US = Params.back();
    US.Calls.emplace_back(Aliasee, ArgNo, ConstantRange(APInt(PointerSize, 0)));
  }
}

} // namespace

struct StackSafetyInfo::InfoTy {
  FunctionInfo Info;
};

StackSafetyInfo makeSSI(FunctionInfo Info) {
  return StackSafetyInfo(StackSafetyInfo::InfoTy{std::move(Info)});
}

namespace {

// Check if we should bailout for such ranges.
bool isUnsafe(const ConstantRange &R) {
  return R.isEmptySet() || R.isFullSet() || R.isUpperSignWrapped();
}

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

  ConstantRange getRange(uint64_t Lower, uint64_t Upper) const {
    return ConstantRange(APInt(PointerSize, Lower), APInt(PointerSize, Upper));
  }

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
  return getAccessRange(Addr, Base, getRange(0, Size.getFixedSize()));
}

ConstantRange StackSafetyLocalAnalysis::getMemIntrinsicAccessRange(
    const MemIntrinsic *MI, const Use &U, Value *Base) {
  if (auto MTI = dyn_cast<MemTransferInst>(MI)) {
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
  assert(!isUnsafe(Sizes));
  if (Sizes.getUpper().isNegative())
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
      auto I = cast<const Instruction>(UI.getUser());
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

        auto B = CB.arg_begin(), E = CB.arg_end();
        int Found = 0;
        for (auto A = B; A != E; ++A) {
          if (A->get() == V) {
            ++Found;
            ConstantRange Offset = offsetFrom(UI, Ptr);
            US.Calls.emplace_back(Callee, A - B, Offset);
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
  FunctionInfo Info(&F);
  assert(!F.isDeclaration() &&
         "Can't run StackSafety on a function declaration");

  LLVM_DEBUG(dbgs() << "[StackSafety] " << F.getName() << "\n");

  for (auto &I : instructions(F)) {
    if (auto AI = dyn_cast<AllocaInst>(&I)) {
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
  // Callee-to-Caller multimap.
  DenseMap<const GlobalValue *, SmallVector<const GlobalValue *, 4>> Callers;
  SetVector<const GlobalValue *> WorkList;

  unsigned PointerSize = 0;
  const ConstantRange UnknownRange;

  ConstantRange getArgumentAccessRange(const GlobalValue *Callee,
                                       unsigned ParamNo) const;
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
  StackSafetyDataFlowAnalysis(
      Module &M, std::function<const FunctionInfo &(Function &)> FI);
  StackSafetyGlobalInfo run();
};

StackSafetyDataFlowAnalysis::StackSafetyDataFlowAnalysis(
    Module &M, std::function<const FunctionInfo &(Function &)> FI)
    : PointerSize(M.getDataLayout().getPointerSizeInBits()),
      UnknownRange(PointerSize, true) {
  // Without ThinLTO, run the local analysis for every function in the TU and
  // then run the DFA.
  for (auto &F : M.functions())
    if (!F.isDeclaration())
      Functions.emplace(&F, FI(F));
  for (auto &A : M.aliases())
    if (isa<Function>(A.getBaseObject()))
      Functions.emplace(&A, FunctionInfo(&A));
}

ConstantRange
StackSafetyDataFlowAnalysis::getArgumentAccessRange(const GlobalValue *Callee,
                                                    unsigned ParamNo) const {
  auto IT = Functions.find(Callee);
  // Unknown callee (outside of LTO domain or an indirect call).
  if (IT == Functions.end())
    return UnknownRange;
  const FunctionInfo &FS = IT->second;
  // The definition of this symbol may not be the definition in this linkage
  // unit.
  if (!FS.IsDSOLocal() || FS.IsInterposable())
    return UnknownRange;
  if (ParamNo >= FS.Params.size()) // possibly vararg
    return UnknownRange;
  return FS.Params[ParamNo].Range;
}

bool StackSafetyDataFlowAnalysis::updateOneUse(UseInfo &US,
                                               bool UpdateToFullSet) {
  bool Changed = false;
  for (auto &CS : US.Calls) {
    assert(!CS.Offset.isEmptySet() &&
           "Param range can't be empty-set, invalid offset range");

    ConstantRange CalleeRange = getArgumentAccessRange(CS.Callee, CS.ParamNo);
    CalleeRange = CalleeRange.add(CS.Offset);
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
  for (auto &AS : FS.Allocas)
    Changed |= updateOneUse(AS, UpdateToFullSet);
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
    for (auto &AS : FS.Allocas)
      for (auto &CS : AS.Calls)
        Callees.push_back(CS.Callee);
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

StackSafetyGlobalInfo StackSafetyDataFlowAnalysis::run() {
  runDataFlow();
  LLVM_DEBUG(verifyFixedPoint());

  StackSafetyGlobalInfo SSI;
  for (auto &F : Functions)
    SSI.emplace(F.first, makeSSI(F.second));
  return SSI;
}

bool setStackSafetyMetadata(Module &M, const StackSafetyGlobalInfo &SSGI) {
  bool Changed = false;
  unsigned Width = M.getDataLayout().getPointerSizeInBits();
  for (auto &F : M.functions()) {
    if (F.isDeclaration() || F.hasOptNone())
      continue;
    auto Iter = SSGI.find(&F);
    if (Iter == SSGI.end())
      continue;
    const FunctionInfo &Summary = Iter->second.getInfo().Info;
    size_t Pos = 0;
    for (auto &I : instructions(F)) {
      if (auto AI = dyn_cast<AllocaInst>(&I)) {
        auto &AS = Summary.Allocas[Pos];
        ConstantRange AllocaRange{
            APInt(Width, 0), APInt(Width, getStaticAllocaAllocationSize(AI))};
        if (AllocaRange.contains(AS.Range)) {
          AI->setMetadata(M.getMDKindID("stack-safe"),
                          MDNode::get(M.getContext(), None));
          Changed = true;
        }
        ++Pos;
      }
    }
  }
  return Changed;
}

} // end anonymous namespace

StackSafetyInfo::StackSafetyInfo(StackSafetyInfo &&) = default;
StackSafetyInfo &StackSafetyInfo::operator=(StackSafetyInfo &&) = default;

StackSafetyInfo::StackSafetyInfo(InfoTy Info)
    : Info(new InfoTy(std::move(Info))) {}

StackSafetyInfo::~StackSafetyInfo() = default;

void StackSafetyInfo::print(raw_ostream &O, const GlobalValue &F) const {
  Info->Info.print(O, F.getName(), dyn_cast<Function>(&F));
}

static void print(const StackSafetyGlobalInfo &SSI, raw_ostream &O,
                  const Module &M) {
  size_t Count = 0;
  for (auto &F : M.functions())
    if (!F.isDeclaration()) {
      SSI.find(&F)->second.print(O, F);
      O << "\n";
      ++Count;
    }
  for (auto &A : M.aliases()) {
    SSI.find(&A)->second.print(O, A);
    O << "\n";
    ++Count;
  }
  assert(Count == SSI.size() && "Unexpected functions in the result");
}

AnalysisKey StackSafetyAnalysis::Key;

StackSafetyInfo StackSafetyAnalysis::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  StackSafetyLocalAnalysis SSLA(F, AM.getResult<ScalarEvolutionAnalysis>(F));
  return makeSSI(SSLA.run());
}

PreservedAnalyses StackSafetyPrinterPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  OS << "'Stack Safety Local Analysis' for function '" << F.getName() << "'\n";
  AM.getResult<StackSafetyAnalysis>(F).print(OS, F);
  return PreservedAnalyses::all();
}

char StackSafetyInfoWrapperPass::ID = 0;

StackSafetyInfoWrapperPass::StackSafetyInfoWrapperPass() : FunctionPass(ID) {
  initializeStackSafetyInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

void StackSafetyInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.setPreservesAll();
}

void StackSafetyInfoWrapperPass::print(raw_ostream &O, const Module *M) const {
  SSI->print(O, *F);
}

bool StackSafetyInfoWrapperPass::runOnFunction(Function &F) {
  StackSafetyLocalAnalysis SSLA(
      F, getAnalysis<ScalarEvolutionWrapperPass>().getSE());
  SSI = makeSSI(SSLA.run());
  this->F = &F;
  return false;
}

AnalysisKey StackSafetyGlobalAnalysis::Key;

StackSafetyGlobalInfo
StackSafetyGlobalAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  StackSafetyDataFlowAnalysis SSDFA(
      M, [&FAM](Function &F) -> const FunctionInfo & {
        return FAM.getResult<StackSafetyAnalysis>(F).getInfo().Info;
      });
  return SSDFA.run();
}

PreservedAnalyses StackSafetyGlobalPrinterPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  OS << "'Stack Safety Analysis' for module '" << M.getName() << "'\n";
  print(AM.getResult<StackSafetyGlobalAnalysis>(M), OS, M);
  return PreservedAnalyses::all();
}

PreservedAnalyses
StackSafetyGlobalAnnotatorPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &SSGI = AM.getResult<StackSafetyGlobalAnalysis>(M);
  (void)setStackSafetyMetadata(M, SSGI);
  return PreservedAnalyses::all();
}

char StackSafetyGlobalInfoWrapperPass::ID = 0;

StackSafetyGlobalInfoWrapperPass::StackSafetyGlobalInfoWrapperPass(
    bool SetMetadata)
    : ModulePass(ID), SetMetadata(SetMetadata) {
  initializeStackSafetyGlobalInfoWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

void StackSafetyGlobalInfoWrapperPass::print(raw_ostream &O,
                                             const Module *M) const {
  ::print(SSGI, O, *M);
}

void StackSafetyGlobalInfoWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<StackSafetyInfoWrapperPass>();
}

bool StackSafetyGlobalInfoWrapperPass::runOnModule(Module &M) {
  StackSafetyDataFlowAnalysis SSDFA(
      M, [this](Function &F) -> const FunctionInfo & {
        return getAnalysis<StackSafetyInfoWrapperPass>(F)
            .getResult()
            .getInfo()
            .Info;
      });
  SSGI = SSDFA.run();
  return SetMetadata ? setStackSafetyMetadata(M, SSGI) : false;
}

ModulePass *llvm::createStackSafetyGlobalInfoWrapperPass(bool SetMetadata) {
  return new StackSafetyGlobalInfoWrapperPass(SetMetadata);
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
                      GlobalPassName, false, false)
INITIALIZE_PASS_DEPENDENCY(StackSafetyInfoWrapperPass)
INITIALIZE_PASS_END(StackSafetyGlobalInfoWrapperPass, DEBUG_TYPE,
                    GlobalPassName, false, false)
