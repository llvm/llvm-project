#include "llvm/Analysis/HeapProvenanceAnalysis.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/CFG.h"
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

bool ForwardHeapProvenanceAnalysisResult::invalidate(
    Module &, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &) {
  return false;
}

bool BackwardHeapProvenanceAnalysisResult::invalidate(
    Module &, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &) {
  return false;
}

bool HeapProvenanceAnalysisResult::invalidate(
    Module &, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &) {
  return false;
}

static bool isAllocLibCall(const Value *V, const TargetLibraryInfo *TLI) {
  return isAllocationFn(V, TLI);
}

static Value *getFreeLibCallOperand(const CallBase *CB,
                                    const TargetLibraryInfo *TLI) {
  if (!CB)
    return nullptr;
  if (Value *Freed = getFreedOperand(CB, TLI))
    return Freed;
  if (Value *Reallocated = getReallocatedOperand(CB))
    return Reallocated;
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
              Callee = dyn_cast<Function>(
                  CB->getCalledOperand()->stripPointerCasts());
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
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  CallGraph &CG = MAM.getResult<CallGraphAnalysis>(M);

  for (scc_iterator<CallGraph *> SCCI = scc_begin(&CG); !SCCI.isAtEnd();
       ++SCCI) {
    for (CallGraphNode *CGN : *SCCI) {
      Function *FnPtr = CGN->getFunction();
      if (!FnPtr || FnPtr->isDeclaration())
        continue;
      Function &F = *FnPtr;
      auto &TLI = FAM.getResult<TargetLibraryAnalysis>(F);

      // 1. Fast Bailout & Candidate Discovery
      SmallVector<std::pair<Instruction *, Value *>, 4> Deallocs;
      for (Instruction &I : instructions(F)) {
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          if (Value *ArgVal = getFreeLibCallOperand(CB, &TLI)) {
            Deallocs.push_back({&I, ArgVal});
          } else if (Function *Callee = CB->getCalledFunction()) {
            if (!Callee->isDeclaration()) {
              for (Argument &Arg : Callee->args()) {
                if (Arg.getArgNo() < CB->arg_size()) {
                  HeapProvenanceLattice ArgInfo = Res.getInfo(&Arg);
                  if (ArgInfo.State ==
                      HeapProvenanceLattice::StateKind::HeapChunkHead)
                    Deallocs.push_back({&I, CB->getArgOperand(Arg.getArgNo())});
                }
              }
            }
          }
        }
      }
      if (Deallocs.empty())
        continue;

      SmallPtrSet<const Value *, 16> Candidates;
      SmallVector<const Value *, 16> CandWorklist;
      for (auto &Pair : Deallocs) {
        if (Candidates.insert(Pair.second).second)
          CandWorklist.push_back(Pair.second);
      }
      while (!CandWorklist.empty()) {
        const Value *V = CandWorklist.pop_back_val();
        auto addCand = [&](const Value *Op) {
          if (Candidates.insert(Op).second)
            CandWorklist.push_back(Op);
        };
        if (auto *GEP = dyn_cast<GEPOperator>(const_cast<Value *>(V)))
          addCand(GEP->getPointerOperand());
        else if (auto *BC = dyn_cast<BitCastInst>(const_cast<Value *>(V)))
          addCand(BC->getOperand(0));
        else if (auto *ASC =
                     dyn_cast<AddrSpaceCastInst>(const_cast<Value *>(V)))
          addCand(ASC->getOperand(0));
        else if (auto *ITP = dyn_cast<IntToPtrInst>(const_cast<Value *>(V)))
          addCand(ITP->getOperand(0));
        else if (auto *PTI = dyn_cast<PtrToIntInst>(const_cast<Value *>(V)))
          addCand(PTI->getOperand(0));
        else if (auto *BO = dyn_cast<BinaryOperator>(const_cast<Value *>(V))) {
          for (Value *Op : BO->operands())
            addCand(Op);
        } else if (auto *PHI = dyn_cast<PHINode>(const_cast<Value *>(V))) {
          for (Value *InV : PHI->incoming_values())
            addCand(InV);
        } else if (auto *Sel = dyn_cast<SelectInst>(const_cast<Value *>(V))) {
          addCand(Sel->getTrueValue());
          addCand(Sel->getFalseValue());
        }
      }

      // 2. Sparse Block-Level State Initialization
      DenseMap<const BasicBlock *,
               DenseMap<const Value *, HeapProvenanceLattice>>
          BlockEntryState;
      SmallVector<const BasicBlock *, 16> BBWorklist;
      for (BasicBlock &BB : F) {
        if (succ_empty(&BB)) {
          for (const Value *C : Candidates) {
            HeapProvenanceLattice Info;
            Info.State = HeapProvenanceLattice::StateKind::Unknown;
            Info.Dir = HeapProvenanceLattice::Backward;
            BlockEntryState[&BB][C] = Info;
          }
          BBWorklist.push_back(&BB);
        }
      }
      if (BBWorklist.empty()) {
        for (BasicBlock &BB : F)
          BBWorklist.push_back(&BB);
      }

      SmallPtrSet<const BasicBlock *, 16> InWorklist;
      for (const BasicBlock *BB : BBWorklist)
        InWorklist.insert(BB);

      // 3. Meet & Backward Transfer Worklist Loop
      while (!BBWorklist.empty()) {
        const BasicBlock *BB = BBWorklist.pop_back_val();
        InWorklist.erase(BB);

        DenseMap<const Value *, HeapProvenanceLattice> ExitState;
        if (succ_empty(BB)) {
          ExitState = BlockEntryState[BB];
        } else {
          bool FirstSucc = true;
          for (const BasicBlock *Succ : successors(BB)) {
            auto It = BlockEntryState.find(Succ);
            if (It == BlockEntryState.end())
              continue;
            const auto &SuccState = It->second;
            for (const Value *C : Candidates) {
              auto CIt = SuccState.find(C);
              if (CIt == SuccState.end() || CIt->second.isUninit())
                continue;
              HeapProvenanceLattice SInfo = CIt->second;
              if (auto *PHI = dyn_cast<PHINode>(const_cast<Value *>(C))) {
                if (PHI->getParent() == Succ) {
                  Value *InV = PHI->getIncomingValueForBlock(
                      const_cast<BasicBlock *>(BB));
                  auto InIt = SuccState.find(InV);
                  if (InIt != SuccState.end())
                    SInfo = InIt->second;
                }
              }
              if (FirstSucc)
                ExitState[C] = SInfo;
              else
                mergeLattice(C, ExitState[C], SInfo);
            }
            FirstSucc = false;
          }
        }

        DenseMap<const Value *, HeapProvenanceLattice> CurState = ExitState;
        for (const Instruction &IConst : llvm::reverse(*BB)) {
          Instruction *I = const_cast<Instruction *>(&IConst);
          if (auto *CB = dyn_cast<CallBase>(I)) {
            if (Value *ArgVal = getFreeLibCallOperand(CB, &TLI)) {
              if (Candidates.count(ArgVal)) {
                HeapProvenanceLattice Info;
                Info.State = HeapProvenanceLattice::StateKind::HeapChunkHead;
                Info.Dir = HeapProvenanceLattice::Backward;
                Info.HeadPayload = {HeapProvenanceLattice::Payload::Kind::Ref,
                                    ArgVal};
                CurState[ArgVal] = Info;
              }
            } else if (Function *Callee = CB->getCalledFunction()) {
              if (!Callee->isDeclaration()) {
                for (Argument &Arg : Callee->args()) {
                  if (Arg.getArgNo() < CB->arg_size()) {
                    HeapProvenanceLattice ArgInfo = Res.getInfo(&Arg);
                    if (ArgInfo.State ==
                        HeapProvenanceLattice::StateKind::HeapChunkHead) {
                      Value *CallOp = CB->getArgOperand(Arg.getArgNo());
                      if (Candidates.count(CallOp)) {
                        HeapProvenanceLattice Info;
                        Info.State =
                            HeapProvenanceLattice::StateKind::HeapChunkHead;
                        Info.Dir = HeapProvenanceLattice::Backward;
                        Info.HeadPayload = {
                            HeapProvenanceLattice::Payload::Kind::Ref, CallOp};
                        CurState[CallOp] = Info;
                      }
                    }
                  }
                }
              }
            }
          }
          if (Candidates.count(I)) {
            auto It = CurState.find(I);
            if (It != CurState.end() && It->second.isValid()) {
              HeapProvenanceLattice BackInfo = It->second;
              if (BackInfo.State ==
                  HeapProvenanceLattice::StateKind::HeapChunkHead)
                BackInfo.State =
                    HeapProvenanceLattice::StateKind::HeapChunkInterior;
              auto transferOp = [&](Value *Op) {
                if (Candidates.count(Op))
                  mergeLattice(Op, CurState[Op], BackInfo);
              };
              if (auto *GEP = dyn_cast<GEPOperator>(I))
                transferOp(GEP->getPointerOperand());
              else if (auto *BC = dyn_cast<BitCastInst>(I))
                transferOp(BC->getOperand(0));
              else if (auto *ASC = dyn_cast<AddrSpaceCastInst>(I))
                transferOp(ASC->getOperand(0));
              else if (auto *ITP = dyn_cast<IntToPtrInst>(I))
                transferOp(ITP->getOperand(0));
              else if (auto *PTI = dyn_cast<PtrToIntInst>(I))
                transferOp(PTI->getOperand(0));
              else if (auto *BO = dyn_cast<BinaryOperator>(I)) {
                for (Value *Op : BO->operands())
                  transferOp(Op);
              } else if (auto *PHI = dyn_cast<PHINode>(I)) {
                for (Value *InV : PHI->incoming_values())
                  transferOp(InV);
              } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
                transferOp(Sel->getTrueValue());
                transferOp(Sel->getFalseValue());
              }
            }
          }
        }

        bool Changed = false;
        auto &EntryState = BlockEntryState[BB];
        for (const Value *C : Candidates) {
          if (EntryState[C] != CurState[C]) {
            EntryState[C] = CurState[C];
            Changed = true;
          }
        }
        if (Changed) {
          for (const BasicBlock *Pred : predecessors(BB)) {
            if (InWorklist.insert(Pred).second)
              BBWorklist.push_back(Pred);
          }
        }
      }

      // 4. Populate Res.getMap()
      for (const Value *C : Candidates) {
        if (isa<Argument>(const_cast<Value *>(C))) {
          mergeIntoMap(Res.getMap(), C, BlockEntryState[&F.getEntryBlock()][C]);
        } else if (auto *I = dyn_cast<Instruction>(const_cast<Value *>(C))) {
          mergeIntoMap(Res.getMap(), C, BlockEntryState[I->getParent()][C]);
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
    if (F.isDeclaration())
      continue;
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
