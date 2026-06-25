#include "llvm/Analysis/HeapProvenanceAnalysis.h"
#include "llvm/Analysis/SparsePropagation.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
template <> struct LatticeKeyInfo<const Value *> {
  static inline Value *getValueFromLatticeKey(const Value *Key) {
    return const_cast<Value *>(Key);
  }
  static inline const Value *getLatticeKeyFromValue(Value *V) {
    return V;
  }
};
} // namespace llvm

using namespace llvm;

AnalysisKey ForwardHeapProvenanceAnalysis::Key;
AnalysisKey BackwardHeapProvenanceAnalysis::Key;
AnalysisKey HeapProvenanceAnalysis::Key;

bool ForwardHeapProvenanceAnalysisResult::invalidate(Module &, const PreservedAnalyses &PA,
                                                     ModuleAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<ForwardHeapProvenanceAnalysis>();
  return !PAC.preservedWhenStateless();
}

bool BackwardHeapProvenanceAnalysisResult::invalidate(Module &, const PreservedAnalyses &PA,
                                                      ModuleAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<BackwardHeapProvenanceAnalysis>();
  return !PAC.preservedWhenStateless();
}

bool HeapProvenanceAnalysisResult::invalidate(Module &, const PreservedAnalyses &PA,
                                              ModuleAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<HeapProvenanceAnalysis>();
  return !PAC.preservedWhenStateless();
}

static bool isAllocFunc(StringRef Name) {
  return Name == "malloc" || Name == "calloc" || Name == "realloc" ||
         Name == "aligned_alloc" || Name == "strdup" || Name == "strndup" ||
         Name == "g_malloc" || Name == "g_malloc0" || Name == "g_realloc" ||
         Name == "g_strdup" || Name.starts_with("_Zn");
}

static bool isFreeFunc(StringRef Name) {
  return Name == "free" || Name == "realloc" || Name == "cfree" ||
         Name == "g_free" || Name.starts_with("_Zd");
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

  if (Src.State == Lattice::StateKind::Unknown) {
    Dest.State = Lattice::StateKind::Unknown;
    Dest.Dir = MergedDir;
    Dest.HeadPayload = {Lattice::Payload::Kind::None, nullptr};
    return true;
  }

  bool Changed = false;
  if (Dest.Dir != MergedDir) {
    Dest.Dir = MergedDir;
    Changed = true;
  }

  if (Dest.State == Src.State && Dest.HeadPayload == Src.HeadPayload)
    return Changed;

  if (isa_and_nonnull<PHINode>(Target)) {
    Lattice::Payload NewPayload{Lattice::Payload::Kind::Phi, Target};
    if (Dest.State != Lattice::StateKind::HeapChunkInterim ||
        Dest.HeadPayload != NewPayload) {
      Dest.State = Lattice::StateKind::HeapChunkInterim;
      Dest.HeadPayload = NewPayload;
      Changed = true;
    }
    return Changed;
  }

  if (isa_and_nonnull<SelectInst>(Target)) {
    Lattice::Payload NewPayload{Lattice::Payload::Kind::Select, Target};
    if (Dest.State != Lattice::StateKind::HeapChunkInterim ||
        Dest.HeadPayload != NewPayload) {
      Dest.State = Lattice::StateKind::HeapChunkInterim;
      Dest.HeadPayload = NewPayload;
      Changed = true;
    }
    return Changed;
  }

  if (Dest.HeadPayload.Val != Src.HeadPayload.Val) {
    Dest.State = Lattice::StateKind::Unknown;
    Dest.HeadPayload = {Lattice::Payload::Kind::None, nullptr};
    return true;
  }

  return Changed;
}

class ForwardLatticeFunc : public AbstractLatticeFunction<const Value *, HeapProvenanceLattice> {
  DenseMap<const Value *, HeapProvenanceLattice> Seeds;
public:
  ForwardLatticeFunc()
      : AbstractLatticeFunction(HeapProvenanceLattice(), HeapProvenanceLattice(), HeapProvenanceLattice()) {}

  void addSeed(const Value *V, const HeapProvenanceLattice &Info) {
    mergeLattice(V, Seeds[V], Info);
  }
  bool IsSpecialCasedPHI(PHINode *PN) override { return true; }
  HeapProvenanceLattice MergeValues(HeapProvenanceLattice X, HeapProvenanceLattice Y) override {
    mergeLattice(nullptr, X, Y);
    return X;
  }
  HeapProvenanceLattice ComputeLatticeVal(const Value *Key) override {
    auto It = Seeds.find(Key);
    if (It != Seeds.end()) return It->second;
    return getUndefVal();
  }
  void ComputeInstructionState(Instruction &I,
                               SmallDenseMap<const Value *, HeapProvenanceLattice, 16> &ChangedValues,
                               SparseSolver<const Value *, HeapProvenanceLattice> &SS) override {
    auto MergeInto = [&](const Value *Target, const HeapProvenanceLattice &NewI) {
      if (!Target || isa<ConstantPointerNull>(Target) || isa<UndefValue>(Target)) return;
      auto &Dest = ChangedValues[Target];
      if (Dest.isUninit()) {
        auto Existing = SS.getExistingValueState(Target);
        if (Existing.isValid()) Dest = Existing;
      }
      mergeLattice(Target, Dest, NewI);
    };

    auto SeedIt = Seeds.find(&I);
    if (SeedIt != Seeds.end()) MergeInto(&I, SeedIt->second);

    for (const Use &U : I.operands()) {
      const Value *Op = U.get();
      auto OpSeedIt = Seeds.find(Op);
      if (OpSeedIt != Seeds.end()) MergeInto(Op, OpSeedIt->second);

      auto OpInfo = SS.getExistingValueState(Op);
      if (OpInfo.isUninit() || !(OpInfo.Dir & HeapProvenanceLattice::Forward)) continue;

      HeapProvenanceLattice NewI = OpInfo;
      if (NewI.State == HeapProvenanceLattice::StateKind::HeapChunkHead) {
        NewI.State = HeapProvenanceLattice::StateKind::HeapChunkInterim;
        NewI.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, Op};
      }

      if (auto *GEP = dyn_cast<GEPOperator>(&I)) {
        if (GEP->getPointerOperand() == Op)
          ChangedValues[&I] = NewI;
      } else if (isa<BitCastInst>(&I) || isa<AddrSpaceCastInst>(&I) ||
                 isa<PtrToIntInst>(&I) || isa<IntToPtrInst>(&I) ||
                 isa<BinaryOperator>(&I)) {
        ChangedValues[&I] = NewI;
      } else if (isa<PHINode>(&I) || isa<SelectInst>(&I)) {
        MergeInto(&I, NewI);
      } else if (auto *CB = dyn_cast<CallBase>(&I)) {
        Function *Callee = CB->getCalledFunction();
        if (!Callee) Callee = dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
        if (Callee && !Callee->isDeclaration()) {
          for (unsigned idx = 0, e = CB->arg_size(); idx != e; ++idx) {
            if (CB->getArgOperand(idx) == Op && idx < Callee->arg_size()) {
              Argument *Arg = Callee->getArg(idx);
              HeapProvenanceLattice ArgI = NewI;
              ArgI.State = HeapProvenanceLattice::StateKind::HeapChunkInterim;
              ArgI.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, Arg};
              MergeInto(Arg, ArgI);
            }
          }
        }
      }
    }

    if (auto *RI = dyn_cast<ReturnInst>(&I)) {
      if (Value *RetVal = RI->getReturnValue()) {
        auto RetInfo = SS.getExistingValueState(RetVal);
        if (RetInfo.isValid() && (RetInfo.Dir & HeapProvenanceLattice::Forward)) {
          HeapProvenanceLattice NewI = RetInfo;
          if (NewI.State == HeapProvenanceLattice::StateKind::HeapChunkHead) {
            NewI.State = HeapProvenanceLattice::StateKind::HeapChunkInterim;
            NewI.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, RetVal};
          }
          const Function *F = RI->getFunction();
          for (const User *U : F->users()) {
            if (auto *CB = dyn_cast<CallBase>(U)) {
              if (CB->getCalledFunction() == F || CB->getCalledOperand()->stripPointerCasts() == F) {
                HeapProvenanceLattice CBI = NewI;
                CBI.State = HeapProvenanceLattice::StateKind::HeapChunkInterim;
                CBI.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, CB};
                MergeInto(CB, CBI);
              }
            }
          }
        }
      }
    }
  }
};

class BackwardLatticeFunc : public AbstractLatticeFunction<const Value *, HeapProvenanceLattice> {
  DenseMap<const Value *, HeapProvenanceLattice> Seeds;
public:
  BackwardLatticeFunc()
      : AbstractLatticeFunction(HeapProvenanceLattice(), HeapProvenanceLattice(), HeapProvenanceLattice()) {}

  void addSeed(const Value *V, const HeapProvenanceLattice &Info) {
    mergeLattice(V, Seeds[V], Info);
  }
  bool IsSpecialCasedPHI(PHINode *PN) override { return true; }
  HeapProvenanceLattice MergeValues(HeapProvenanceLattice X, HeapProvenanceLattice Y) override {
    mergeLattice(nullptr, X, Y);
    return X;
  }
  HeapProvenanceLattice ComputeLatticeVal(const Value *Key) override {
    auto It = Seeds.find(Key);
    if (It != Seeds.end()) return It->second;
    return getUndefVal();
  }
  void ComputeInstructionState(Instruction &I,
                               SmallDenseMap<const Value *, HeapProvenanceLattice, 16> &ChangedValues,
                               SparseSolver<const Value *, HeapProvenanceLattice> &SS) override {
    auto MergeInto = [&](const Value *Target, const HeapProvenanceLattice &NewI) {
      if (!Target || isa<ConstantPointerNull>(Target) || isa<UndefValue>(Target)) return;
      auto &Dest = ChangedValues[Target];
      if (Dest.isUninit()) {
        auto Existing = SS.getExistingValueState(Target);
        if (Existing.isValid()) Dest = Existing;
      }
      mergeLattice(Target, Dest, NewI);
    };

    auto SeedIt = Seeds.find(&I);
    if (SeedIt != Seeds.end()) MergeInto(&I, SeedIt->second);

    for (const Use &U : I.operands()) {
      const Value *Op = U.get();
      auto OpInfo = ChangedValues[Op];
      if (OpInfo.isUninit())
        OpInfo = SS.getExistingValueState(Op);
      if (OpInfo.isUninit()) {
        auto OpSeedIt = Seeds.find(Op);
        if (OpSeedIt != Seeds.end()) OpInfo = OpSeedIt->second;
      }
      if (OpInfo.isValid() && (OpInfo.Dir & HeapProvenanceLattice::Backward)) {
        HeapProvenanceLattice FwdFromBack = OpInfo;
        if (FwdFromBack.State == HeapProvenanceLattice::StateKind::HeapChunkHead) {
          FwdFromBack.State = HeapProvenanceLattice::StateKind::HeapChunkInterim;
          FwdFromBack.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, Op};
        }
        MergeInto(&I, FwdFromBack);
      }
    }

    auto Info = ChangedValues[&I];
    if (Info.isUninit())
      Info = SS.getExistingValueState(&I);
    if (Info.isValid() && (Info.Dir & HeapProvenanceLattice::Backward)) {
      HeapProvenanceLattice BackInfo = Info;
      BackInfo.Dir = HeapProvenanceLattice::Backward;
      if (BackInfo.State == HeapProvenanceLattice::StateKind::HeapChunkHead) {
        BackInfo.State = HeapProvenanceLattice::StateKind::HeapChunkInterim;
        BackInfo.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, &I};
      }

      if (auto *GEP = dyn_cast<GEPOperator>(&I)) {
        MergeInto(GEP->getPointerOperand(), BackInfo);
      } else if (auto *BC = dyn_cast<BitCastInst>(&I)) {
        MergeInto(BC->getOperand(0), BackInfo);
      } else if (auto *ASC = dyn_cast<AddrSpaceCastInst>(&I)) {
        MergeInto(ASC->getOperand(0), BackInfo);
      } else if (auto *ITP = dyn_cast<IntToPtrInst>(&I)) {
        MergeInto(ITP->getOperand(0), BackInfo);
      } else if (auto *PTI = dyn_cast<PtrToIntInst>(&I)) {
        MergeInto(PTI->getOperand(0), BackInfo);
      } else if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
        for (Value *Op : BO->operands())
          MergeInto(Op, BackInfo);
      } else if (auto *PHI = dyn_cast<PHINode>(&I)) {
        for (Value *InV : PHI->incoming_values())
          MergeInto(InV, BackInfo);
      } else if (auto *Sel = dyn_cast<SelectInst>(&I)) {
        MergeInto(Sel->getTrueValue(), BackInfo);
        MergeInto(Sel->getFalseValue(), BackInfo);
      } else if (auto *CB = dyn_cast<CallBase>(&I)) {
        Function *Callee = CB->getCalledFunction();
        if (!Callee) Callee = dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
        if (Callee && !Callee->isDeclaration() && I.getType()->isPointerTy()) {
          for (BasicBlock &CalleeBB : *Callee) {
            if (auto *RI = dyn_cast<ReturnInst>(CalleeBB.getTerminator())) {
              if (Value *RetVal = RI->getReturnValue()) {
                HeapProvenanceLattice RetI = BackInfo;
                RetI.State = HeapProvenanceLattice::StateKind::HeapChunkInterim;
                RetI.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, RetVal};
                MergeInto(RetVal, RetI);
              }
            }
          }
        }
      }
    }

    for (const Use &U : I.operands()) {
      if (auto *Arg = dyn_cast<Argument>(U.get())) {
        auto ArgInfo = SS.getExistingValueState(Arg);
        if (ArgInfo.isValid() && (ArgInfo.Dir & HeapProvenanceLattice::Backward)) {
          const Function *F = Arg->getParent();
          for (const User *Usr : F->users()) {
            if (auto *CB = dyn_cast<CallBase>(Usr)) {
              if (CB->getCalledFunction() == F || CB->getCalledOperand()->stripPointerCasts() == F) {
                unsigned ArgIdx = Arg->getArgNo();
                if (ArgIdx < CB->arg_size()) {
                  Value *CallerArg = CB->getArgOperand(ArgIdx);
                  HeapProvenanceLattice CallerI = ArgInfo;
                  CallerI.Dir = HeapProvenanceLattice::Backward;
                  CallerI.State = HeapProvenanceLattice::StateKind::HeapChunkInterim;
                  CallerI.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, CallerArg};
                  MergeInto(CallerArg, CallerI);
                }
              }
            }
          }
        }
      }
    }
  }
};

ForwardHeapProvenanceAnalysis::Result
ForwardHeapProvenanceAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  Result Res;
  ForwardLatticeFunc Lattice;

  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          Function *Callee = CB->getCalledFunction();
          if (!Callee)
            Callee = dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
          if (Callee && isAllocFunc(Callee->getName())) {
            HeapProvenanceLattice Info;
            Info.State = HeapProvenanceLattice::StateKind::HeapChunkHead;
            Info.Dir = HeapProvenanceLattice::Forward;
            Info.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, &I};
            Lattice.addSeed(&I, Info);
          }
        }
      }
    }
  }

  SparseSolver<const Value *, HeapProvenanceLattice> Solver(&Lattice);
  for (Function &F : M)
    if (!F.isDeclaration())
      Solver.MarkBlockExecutable(&F.front());

  Solver.Solve();

  for (Function &F : M) {
    for (Argument &Arg : F.args()) {
      auto LV = Solver.getExistingValueState(&Arg);
      if (LV.isValid()) Res.setInfo(&Arg, LV);
    }
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto LV = Solver.getExistingValueState(&I);
        if (LV.isValid()) Res.setInfo(&I, LV);
      }
    }
  }
  return Res;
}

BackwardHeapProvenanceAnalysis::Result
BackwardHeapProvenanceAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  Result Res;
  BackwardLatticeFunc Lattice;

  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          Function *Callee = CB->getCalledFunction();
          if (!Callee)
            Callee = dyn_cast<Function>(CB->getCalledOperand()->stripPointerCasts());
          if (Callee && isFreeFunc(Callee->getName()) && CB->arg_size() > 0) {
            Value *ArgVal = CB->getArgOperand(0);
            HeapProvenanceLattice Info;
            Info.State = HeapProvenanceLattice::StateKind::HeapChunkHead;
            Info.Dir = HeapProvenanceLattice::Backward;
            Info.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref, ArgVal};
            Lattice.addSeed(ArgVal, Info);
          }
        }
      }
    }
  }

  SparseSolver<const Value *, HeapProvenanceLattice> Solver(&Lattice);
  for (Function &F : M)
    if (!F.isDeclaration())
      Solver.MarkBlockExecutable(&F.front());

  Solver.Solve();

  for (Function &F : M) {
    for (Argument &Arg : F.args()) {
      auto LV = Solver.getExistingValueState(&Arg);
      if (LV.isValid()) Res.setInfo(&Arg, LV);
    }
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto LV = Solver.getExistingValueState(&I);
        if (LV.isValid()) Res.setInfo(&I, LV);
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
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    OS << "Printing analysis 'Heap Provenance Analysis' for function '"
       << F.getName() << "':\n";
    for (Argument &Arg : F.args()) {
      auto Info = Result.getInfo(&Arg);
      if (Info.isValid()) {
        OS << "  argument ";
        Arg.printAsOperand(OS, false);
        OS << ": "
           << (Info.State == HeapProvenanceLattice::StateKind::HeapChunkHead
                   ? "HeapChunkHead"
                   : "HeapChunkInterim")
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
                     : "HeapChunkInterim")
             << " (" << Info.getExpr() << ")" << Info.getDirectionStr() << "\n";
        }
      }
    }
  }
  return PreservedAnalyses::all();
}
