#include "llvm/Analysis/HeapProvenanceAnalysis.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

AnalysisKey ForwardHeapProvenanceAnalysis::Key;
AnalysisKey BackwardHeapProvenanceAnalysis::Key;
AnalysisKey HeapProvenanceAnalysis::Key;

bool ForwardHeapProvenanceAnalysisResult::invalidate(Module &, const PreservedAnalyses &PA,
                                                     ModuleAnalysisManager::Invalidator &) {
  return false;
}

bool BackwardHeapProvenanceAnalysisResult::invalidate(Module &, const PreservedAnalyses &PA,
                                                      ModuleAnalysisManager::Invalidator &) {
  return false;
}

bool HeapProvenanceAnalysisResult::invalidate(Module &, const PreservedAnalyses &PA,
                                              ModuleAnalysisManager::Invalidator &) {
  return false;
}

static bool isAllocLibCall(const Value *V, const TargetLibraryInfo *TLI) {
  return isAllocationFn(V, TLI);
}

static Value *getFreeLibCallOperand(const CallBase *CB, const TargetLibraryInfo *TLI) {
  if (!CB || !TLI)
    return nullptr;
  const Function *Callee = CB->getCalledFunction();
  if (!Callee)
    return nullptr;
  LibFunc TLIFn;
  if (TLI->getLibFunc(*Callee, TLIFn) && TLI->has(TLIFn) &&
      isLibFreeFunction(Callee, TLIFn)) {
    return CB->getArgOperand(0);
  }
  return nullptr;
}

static bool mergeLattice(const Value *Target, HeapProvenanceLattice &Dest,
                         const HeapProvenanceLattice &Src) {
  using Lattice = HeapProvenanceLattice;
  if (Src.State == Lattice::StateKind::Uninit)
    return false;

  auto MergedDir = static_cast<Lattice::Direction>(Dest.Dir | Src.Dir);

  if (Dest.State == Lattice::StateKind::Uninit) {
    Dest = Src;
    Dest.Dir = MergedDir;
    return true;
  }

  if (Dest.State == Lattice::StateKind::Unknown) {
    if (Dest.Dir != MergedDir) {
      Dest.Dir = MergedDir;
      return true;
    }
    return false;
  }

  if (Src.State == Lattice::StateKind::Unknown ||
      Dest.State == Lattice::StateKind::Unknown) {
    bool Changed =
        (Dest.State != Lattice::StateKind::Unknown) || (Dest.Dir != MergedDir);
    Dest.State = Lattice::StateKind::Unknown;
    Dest.Dir = MergedDir;
    Dest.HeadPayload = {Lattice::Payload::Kind::None, nullptr};
    return Changed;
  }

  bool Changed = false;
  if (Dest.Dir != MergedDir) {
    Dest.Dir = MergedDir;
    Changed = true;
  }

  if (Dest.State == Src.State && Dest.HeadPayload == Src.HeadPayload)
    return Changed;

  if (Dest.isUninit()) {
    Dest.State = Src.State;
    if (isa_and_nonnull<PHINode>(Target) || isa_and_nonnull<SelectInst>(Target))
      Dest.State = Lattice::StateKind::HeapChunkInterior;
    Dest.HeadPayload = Src.HeadPayload;
    return true;
  }

  if (Dest.HeadPayload != Src.HeadPayload) {
    Dest.State = Lattice::StateKind::Unknown;
    Dest.HeadPayload = {Lattice::Payload::Kind::None, nullptr};
    return true;
  }

  return Changed;
}

static bool mergeIntoMap(DenseMap<const Value *, HeapProvenanceLattice> &Map,
                         const Value *Target,
                         const HeapProvenanceLattice &Src) {
  if (!Target || !Target->hasUseList())
    return false;
  return mergeLattice(Target, Map[Target], Src);
}

ForwardHeapProvenanceAnalysis::Result
ForwardHeapProvenanceAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  Result Res;
  SmallVector<const Value *, 64> Worklist;
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    auto &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
    for (Instruction &I : instructions(F)) {
      if (isAllocLibCall(&I, &TLI)) {
        HeapProvenanceLattice Info;
        Info.State = HeapProvenanceLattice::StateKind::HeapChunkHead;
        Info.Dir = HeapProvenanceLattice::Forward;
        Info.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, &I};
        Res.setInfo(&I, Info);
        Worklist.push_back(&I);
      }
    }
  }

  if (Worklist.empty())
    return Res;

  while (!Worklist.empty()) {
    const Value *V = Worklist.pop_back_val();
    HeapProvenanceLattice Info = Res.getInfo(V);
    if (!Info.isValid() || !(Info.Dir & HeapProvenanceLattice::Forward))
      continue;

    HeapProvenanceLattice Derived = Info;
    if (Derived.State == HeapProvenanceLattice::StateKind::HeapChunkHead)
      Derived.State = HeapProvenanceLattice::StateKind::HeapChunkInterior;

    for (const User *U : V->users()) {
      if (auto *GEP = dyn_cast<GEPOperator>(U)) {
        if (GEP->getPointerOperand() == V) {
          if (mergeIntoMap(Res.getMap(), U, Derived))
            Worklist.push_back(U);
        }
      } else if (isa<BitCastInst>(U) || isa<AddrSpaceCastInst>(U) ||
                 isa<PtrToIntInst>(U) || isa<IntToPtrInst>(U) ||
                 isa<BinaryOperator>(U)) {
        if (mergeIntoMap(Res.getMap(), U, Derived))
          Worklist.push_back(U);
      } else if (isa<PHINode>(U) || isa<SelectInst>(U)) {
        if (mergeIntoMap(Res.getMap(), U, Derived))
          Worklist.push_back(U);
      } else if (auto *CB = dyn_cast<CallBase>(U)) {
        for (unsigned i = 0, e = CB->arg_size(); i != e; ++i) {
          if (CB->getArgOperand(i) == V) {
            Function *Callee = CB->getCalledFunction();
            if (!Callee)
              Callee = dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
            if (Callee && !Callee->isDeclaration() && i < Callee->arg_size()) {
              Argument *Arg = Callee->getArg(i);
              if (mergeIntoMap(Res.getMap(), Arg, Derived))
                Worklist.push_back(Arg);
            }
          }
        }
      } else if (auto *RI = dyn_cast<ReturnInst>(U)) {
        Function *Fn = const_cast<Function *>(RI->getParent()->getParent());
        for (const User *FnU : Fn->users()) {
          if (auto *Call = dyn_cast<CallBase>(FnU)) {
            if (mergeIntoMap(Res.getMap(), Call, Derived))
              Worklist.push_back(Call);
          }
        }
      }
    }
  }

  return Res;
}

BackwardHeapProvenanceAnalysis::Result
BackwardHeapProvenanceAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  Result Res;
  SmallVector<const Value *, 64> Worklist;
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    auto &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
    auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
    for (Instruction &I : instructions(F)) {
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (Value *ArgVal = getFreeLibCallOperand(CB, &TLI)) {
          if (!PDT.dominates(CB->getParent(), &F.getEntryBlock()))
            continue;
          HeapProvenanceLattice Info;
          Info.State = HeapProvenanceLattice::StateKind::HeapChunkHead;
          Info.Dir = HeapProvenanceLattice::Backward;
          Info.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, ArgVal};
          if (mergeIntoMap(Res.getMap(), ArgVal, Info))
            Worklist.push_back(ArgVal);
        }
      }
    }
  }

  if (Worklist.empty())
    return Res;

  while (!Worklist.empty()) {
    const Value *V = Worklist.pop_back_val();
    HeapProvenanceLattice Info = Res.getInfo(V);
    if (!Info.isValid() || !(Info.Dir & HeapProvenanceLattice::Backward))
      continue;

    HeapProvenanceLattice BackInfo = Info;
    BackInfo.Dir = HeapProvenanceLattice::Backward;
    if (BackInfo.State == HeapProvenanceLattice::StateKind::HeapChunkHead)
      BackInfo.State = HeapProvenanceLattice::StateKind::HeapChunkInterior;

    if (auto *GEP = dyn_cast<GEPOperator>(const_cast<Value *>(V))) {
      if (mergeIntoMap(Res.getMap(), GEP->getPointerOperand(), BackInfo))
        Worklist.push_back(GEP->getPointerOperand());
    } else if (auto *BC = dyn_cast<BitCastInst>(const_cast<Value *>(V))) {
      if (mergeIntoMap(Res.getMap(), BC->getOperand(0), BackInfo))
        Worklist.push_back(BC->getOperand(0));
    } else if (auto *ASC = dyn_cast<AddrSpaceCastInst>(const_cast<Value *>(V))) {
      if (mergeIntoMap(Res.getMap(), ASC->getOperand(0), BackInfo))
        Worklist.push_back(ASC->getOperand(0));
    } else if (auto *ITP = dyn_cast<IntToPtrInst>(const_cast<Value *>(V))) {
      if (mergeIntoMap(Res.getMap(), ITP->getOperand(0), BackInfo))
        Worklist.push_back(ITP->getOperand(0));
    } else if (auto *PTI = dyn_cast<PtrToIntInst>(const_cast<Value *>(V))) {
      if (mergeIntoMap(Res.getMap(), PTI->getOperand(0), BackInfo))
        Worklist.push_back(PTI->getOperand(0));
    } else if (auto *BO = dyn_cast<BinaryOperator>(const_cast<Value *>(V))) {
      for (Value *Op : BO->operands())
        if (mergeIntoMap(Res.getMap(), Op, BackInfo))
          Worklist.push_back(Op);
    } else if (isa<PHINode>(const_cast<Value *>(V)) || isa<SelectInst>(const_cast<Value *>(V))) {
      // TODO: Propagating backward deallocation provenance across PHI nodes or
      // Select instructions without path-sensitive join conditions is unsound.
      // Stop backward propagation here to maintain lattice join soundness.
    } else if (auto *Arg = dyn_cast<Argument>(const_cast<Value *>(V))) {
      Function *Fn = Arg->getParent();
      for (const User *FnU : Fn->users()) {
        if (auto *CB = dyn_cast<CallBase>(FnU)) {
          if (Arg->getArgNo() < CB->arg_size()) {
            Value *CallOp = CB->getArgOperand(Arg->getArgNo());
            if (mergeIntoMap(Res.getMap(), CallOp, BackInfo))
              Worklist.push_back(CallOp);
          }
        }
      }
    } else if (auto *CB = dyn_cast<CallBase>(const_cast<Value *>(V))) {
      Function *Callee = CB->getCalledFunction();
      if (!Callee)
        Callee = dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
      if (Callee && !Callee->isDeclaration() && V->getType()->isPointerTy()) {
        for (BasicBlock &CalleeBB : *Callee) {
          if (auto *RI = dyn_cast<ReturnInst>(CalleeBB.getTerminator())) {
            if (Value *RetVal = RI->getReturnValue()) {
              if (mergeIntoMap(Res.getMap(), RetVal, BackInfo))
                Worklist.push_back(RetVal);
            }
          }
        }
      }
    }
  }

  return Res;
}

HeapProvenanceAnalysisResult HeapProvenanceAnalysis::analyzeModule(Module &M) {
  ModuleAnalysisManager DummyMAM;
  ForwardHeapProvenanceAnalysis ForwardHPA;
  BackwardHeapProvenanceAnalysis BackwardHPA;
  return Result(ForwardHPA.run(M, DummyMAM), BackwardHPA.run(M, DummyMAM));
}

HeapProvenanceAnalysis::Result
HeapProvenanceAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  auto ForwardRes = MAM.getResult<ForwardHeapProvenanceAnalysis>(M);
  auto BackwardRes = MAM.getResult<BackwardHeapProvenanceAnalysis>(M);
  return Result(std::move(ForwardRes), std::move(BackwardRes));
}

PreservedAnalyses HeapProvenancePrinterPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  auto &Result = MAM.getResult<HeapProvenanceAnalysis>(M);
  OS << "Printing analysis 'Heap Provenance Analysis' for module '"
     << M.getName() << "':\n";
  for (Function &F : M) {
    if (F.isDeclaration()) continue;
    for (Argument &Arg : F.args()) {
      auto Info = Result.getInfo(&Arg);
      if (Info.isValid()) {
        OS << "  argument ";
        Arg.printAsOperand(OS, false);
        OS << ": "
           << (Info.State == HeapProvenanceLattice::StateKind::HeapChunkHead
                   ? "HeapChunkHead"
                   : "HeapChunkInterior")
           << " (" << Info.getExpr() << ")" << Info.getDirectionStr() << "\n";
      }
    }
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto Info = Result.getInfo(&I);
        if (Info.isValid()) {
          OS << "  ";
          I.printAsOperand(OS, false);
          OS << ": "
             << (Info.State == HeapProvenanceLattice::StateKind::HeapChunkHead
                     ? "HeapChunkHead"
                     : "HeapChunkInterior")
             << " (" << Info.getExpr() << ")" << Info.getDirectionStr() << "\n";
        }
      }
    }
  }
  return PreservedAnalyses::all();
}
