#include "llvm/Analysis/HeapProvenanceAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

AnalysisKey HeapProvenanceAnalysis::Key;

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

static std::string getValueOperandName(const Value *V) {
  std::string Str;
  raw_string_ostream OS(Str);
  V->printAsOperand(OS, false);
  return Str;
}

static void addSymOffset(std::vector<std::string> &Syms, const std::string &S) {
  StringRef SR(S);
  std::string NegS = (SR.size() > 3 && SR.starts_with("-(") && SR.ends_with(")"))
                         ? S.substr(2, S.size() - 3)
                         : "-(" + S + ")";
  for (auto It = Syms.begin(); It != Syms.end(); ++It) {
    if (*It == NegS) {
      Syms.erase(It);
      return;
    }
  }
  Syms.push_back(S);
}

static void extractGEPOffsets(const GEPOperator *GEP, const DataLayout &DL,
                              int64_t &ConstOff,
                              std::vector<std::string> &SymOffs) {
  APInt APIntOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
  if (GEP->accumulateConstantOffset(DL, APIntOffset)) {
    ConstOff += APIntOffset.getSExtValue();
    return;
  }
  for (gep_type_iterator GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP);
       GTI != GTE; ++GTI) {
    Value *Op = GTI.getOperand();
    if (StructType *STy = GTI.getStructTypeOrNull()) {
      ConstantInt *OpC = cast<ConstantInt>(Op);
      const StructLayout *SL = DL.getStructLayout(STy);
      ConstOff += SL->getElementOffset(OpC->getZExtValue());
    } else {
      TypeSize Stride = GTI.getSequentialElementStride(DL);
      int64_t StrideVal = Stride.getKnownMinValue();
      if (ConstantInt *OpC = dyn_cast<ConstantInt>(Op)) {
        ConstOff += OpC->getSExtValue() * StrideVal;
      } else {
        std::string OpName = getValueOperandName(Op);
        if (StrideVal == 1)
          addSymOffset(SymOffs, OpName);
        else
          addSymOffset(SymOffs, OpName + " * " + std::to_string(StrideVal));
      }
    }
  }
}

static HeapProvenanceAnalysisResult::ProvenanceInfo
addOffset(const HeapProvenanceAnalysisResult::ProvenanceInfo &BaseInfo,
          int64_t C, const std::vector<std::string> &Syms) {
  using ProvenanceInfo = HeapProvenanceAnalysisResult::ProvenanceInfo;
  if (!BaseInfo.isValid())
    return BaseInfo;
  ProvenanceInfo Res = BaseInfo;
  Res.State = ProvenanceInfo::RecoverableHeapChunkPtr;
  if (!Res.CustomExpr.empty()) {
    std::string OffStr;
    if (C > 0)
      OffStr += " + " + std::to_string(C);
    else if (C < 0)
      OffStr += " - " + std::to_string(-C);
    for (const auto &S : Syms)
      OffStr += " + " + S;
    Res.CustomExpr = "(" + Res.CustomExpr + ")" + OffStr;
    return Res;
  }
  Res.ConstOffset += C;
  for (const auto &S : Syms)
    addSymOffset(Res.SymOffsets, S);
  if (Res.ConstOffset == 0 && Res.SymOffsets.empty())
    Res.State = ProvenanceInfo::HeapChunkPtr;
  return Res;
}

static HeapProvenanceAnalysisResult::ProvenanceInfo
subOffset(const HeapProvenanceAnalysisResult::ProvenanceInfo &IInfo, int64_t C,
          const std::vector<std::string> &Syms) {
  using ProvenanceInfo = HeapProvenanceAnalysisResult::ProvenanceInfo;
  if (!IInfo.isValid())
    return IInfo;
  ProvenanceInfo Res = IInfo;
  Res.State = ProvenanceInfo::RecoverableHeapChunkPtr;
  if (!Res.CustomExpr.empty()) {
    std::string OffStr;
    if (C > 0)
      OffStr += " - " + std::to_string(C);
    else if (C < 0)
      OffStr += " + " + std::to_string(-C);
    for (const auto &S : Syms)
      OffStr += " - (" + S + ")";
    Res.CustomExpr = "(" + Res.CustomExpr + ")" + OffStr;
    return Res;
  }
  Res.ConstOffset -= C;
  for (const auto &S : Syms)
    addSymOffset(Res.SymOffsets, "-(" + S + ")");
  if (Res.ConstOffset == 0 && Res.SymOffsets.empty())
    Res.State = ProvenanceInfo::HeapChunkPtr;
  return Res;
}

static bool mergeInfo(HeapProvenanceAnalysisResult::ProvenanceInfo &Dest,
                      const HeapProvenanceAnalysisResult::ProvenanceInfo &Src) {
  using ProvenanceInfo = HeapProvenanceAnalysisResult::ProvenanceInfo;
  if (Src.State == ProvenanceInfo::Uninit)
    return false;

  auto MergedDir = static_cast<ProvenanceInfo::Direction>(Dest.Dir | Src.Dir);

  if (Dest.State == ProvenanceInfo::Uninit) {
    Dest = Src;
    Dest.Dir = MergedDir;
    return true;
  }
  if (Dest.State == ProvenanceInfo::Unknown) {
    if (Dest.Dir != MergedDir) {
      Dest.Dir = MergedDir;
      return true;
    }
    return false;
  }
  if (Src.State == ProvenanceInfo::Unknown) {
    if (Dest.Dir != MergedDir) {
      Dest.Dir = MergedDir;
      return true;
    }
    return false;
  }

  bool Changed = false;
  if (Dest.Dir != MergedDir) {
    Dest.Dir = MergedDir;
    Changed = true;
  }

  if (Dest.getExpr() == Src.getExpr()) {
    if (Src.State == ProvenanceInfo::RecoverableHeapChunkPtr &&
        Dest.State != ProvenanceInfo::RecoverableHeapChunkPtr) {
      Dest.State = ProvenanceInfo::RecoverableHeapChunkPtr;
      Changed = true;
    }
    return Changed;
  }

  if (Dest.CustomExpr == "head + dynamic_offset")
    return Changed;

  if (!Dest.CustomExpr.empty()) {
    Dest.CustomExpr = "head + dynamic_offset";
    Dest.State = ProvenanceInfo::RecoverableHeapChunkPtr;
    return true;
  }

  Dest.CustomExpr = "PHI(" + Dest.getExpr() + ", " + Src.getExpr() + ")";
  Dest.State = ProvenanceInfo::RecoverableHeapChunkPtr;
  return true;
}

static bool updateInfo(HeapProvenanceAnalysisResult::ProvenanceInfo &Dest,
                       const HeapProvenanceAnalysisResult::ProvenanceInfo &NewVal) {
  using ProvenanceInfo = HeapProvenanceAnalysisResult::ProvenanceInfo;
  if (Dest.State == NewVal.State && Dest.Dir == NewVal.Dir &&
      Dest.ConstOffset == NewVal.ConstOffset &&
      Dest.SymOffsets == NewVal.SymOffsets &&
      Dest.CustomExpr == NewVal.CustomExpr)
    return false;

  if (!Dest.CustomExpr.empty() && !NewVal.CustomExpr.empty() &&
      Dest.CustomExpr != NewVal.CustomExpr) {
    std::string Widened = "head + dynamic_offset";
    if (Dest.CustomExpr != Widened) {
      Dest = NewVal;
      Dest.CustomExpr = Widened;
      return true;
    }
    return false;
  }

  Dest = NewVal;
  return true;
}

static HeapProvenanceAnalysisResult::ProvenanceInfo
keepForwardOnly(const HeapProvenanceAnalysisResult::ProvenanceInfo &Info) {
  using ProvenanceInfo = HeapProvenanceAnalysisResult::ProvenanceInfo;
  auto Res = Info;
  Res.Dir = static_cast<ProvenanceInfo::Direction>(Info.Dir &
                                                     ProvenanceInfo::Forward);
  if (Res.Dir == ProvenanceInfo::None)
    Res.State = ProvenanceInfo::Uninit;
  return Res;
}

HeapProvenanceAnalysis::Result
HeapProvenanceAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  Result Res;
  const DataLayout &DL = M.getDataLayout();

  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          Function *Callee = CB->getCalledFunction();
          if (!Callee) {
            Value *CalledVal = CB->getCalledOperand()->stripPointerCasts();
            Callee = dyn_cast<Function>(CalledVal);
          }
          if (Callee) {
            StringRef Name = Callee->getName();
            if (isAllocFunc(Name)) {
              Result::ProvenanceInfo Info;
              Info.State = Result::ProvenanceInfo::HeapChunkPtr;
              Info.Dir = Result::ProvenanceInfo::Forward;
              mergeInfo(Res.getMap()[&I], Info);
            }
            if (isFreeFunc(Name)) {
              if (CB->arg_size() > 0) {
                Value *ArgVal = CB->getArgOperand(0);
                Result::ProvenanceInfo Info;
                Info.State = Result::ProvenanceInfo::HeapChunkPtr;
                Info.Dir = Result::ProvenanceInfo::Backward;
                mergeInfo(Res.getMap()[ArgVal], Info);
              }
            }
          }
        }
      }
    }
  }

  bool Changed = true;
  int MaxIters = 50;
  while (Changed && MaxIters-- > 0) {
    Changed = false;

    // Forward propagation
    for (Function &F : M) {
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto *GEP = dyn_cast<GEPOperator>(&I)) {
            Value *Base = GEP->getPointerOperand();
            auto BaseInfo = keepForwardOnly(Res.getInfo(Base));
            if (BaseInfo.isValid()) {
              auto NewInfo = BaseInfo;
              NewInfo.State = Result::ProvenanceInfo::RecoverableHeapChunkPtr;
              int64_t ConstOff = 0;
              std::vector<std::string> SymOffs;
              extractGEPOffsets(GEP, DL, ConstOff, SymOffs);
              NewInfo = addOffset(NewInfo, ConstOff, SymOffs);
              if (mergeInfo(Res.getMap()[&I], NewInfo))
                Changed = true;
            }
          } else if (auto *BC = dyn_cast<BitCastOperator>(&I)) {
            Value *Op = BC->getOperand(0);
            auto OpInfo = keepForwardOnly(Res.getInfo(Op));
            if (OpInfo.isValid()) {
              if (mergeInfo(Res.getMap()[&I], OpInfo))
                Changed = true;
            }
          } else if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(&I)) {
            Value *Op = ASC->getOperand(0);
            auto OpInfo = keepForwardOnly(Res.getInfo(Op));
            if (OpInfo.isValid()) {
              if (mergeInfo(Res.getMap()[&I], OpInfo))
                Changed = true;
            }
          } else if (auto *ITP = dyn_cast<IntToPtrInst>(&I)) {
            Value *Op = ITP->getOperand(0);
            if (auto *BO = dyn_cast<BinaryOperator>(Op)) {
              if (BO->getOpcode() == Instruction::Add ||
                  BO->getOpcode() == Instruction::Sub) {
                Value *LHS = BO->getOperand(0);
                Value *RHS = BO->getOperand(1);
                auto *PTI = dyn_cast<PtrToIntOperator>(LHS);
                if (!PTI) {
                  PTI = dyn_cast<PtrToIntOperator>(RHS);
                  if (PTI && BO->getOpcode() == Instruction::Add)
                    std::swap(LHS, RHS);
                  else
                    PTI = nullptr;
                }
                if (PTI) {
                  Value *BasePtr = PTI->getOperand(0);
                  auto BaseInfo = keepForwardOnly(Res.getInfo(BasePtr));
                  if (BaseInfo.isValid()) {
                    int64_t ConstOff = 0;
                    std::vector<std::string> SymOffs;
                    if (auto *CInt = dyn_cast<ConstantInt>(RHS)) {
                      ConstOff = CInt->getSExtValue();
                    } else {
                      SymOffs.push_back(getValueOperandName(RHS));
                    }
                    if (BO->getOpcode() == Instruction::Sub) {
                      ConstOff = -ConstOff;
                      for (auto &S : SymOffs)
                        S = "-(" + S + ")";
                    }
                    auto NewInfo = addOffset(BaseInfo, ConstOff, SymOffs);
                    if (mergeInfo(Res.getMap()[&I], NewInfo))
                      Changed = true;
                  }
                }
              }
            } else if (auto *PTI = dyn_cast<PtrToIntOperator>(Op)) {
              auto BaseInfo = keepForwardOnly(Res.getInfo(PTI->getOperand(0)));
              if (BaseInfo.isValid()) {
                if (mergeInfo(Res.getMap()[&I], BaseInfo))
                  Changed = true;
              }
            }
          } else if (auto *PHI = dyn_cast<PHINode>(&I)) {
            Result::ProvenanceInfo Temp;
            for (Value *InV : PHI->incoming_values()) {
              if (isa<ConstantPointerNull>(InV) || isa<UndefValue>(InV))
                continue;
              auto InInfo = keepForwardOnly(Res.getInfo(InV));
              if (InInfo.isValid())
                mergeInfo(Temp, InInfo);
            }
            if (Temp.isValid() && updateInfo(Res.getMap()[&I], Temp))
              Changed = true;
          } else if (auto *Sel = dyn_cast<SelectInst>(&I)) {
            Result::ProvenanceInfo Temp;
            for (Value *Op : {Sel->getTrueValue(), Sel->getFalseValue()}) {
              if (isa<ConstantPointerNull>(Op) || isa<UndefValue>(Op))
                continue;
              auto OpInfo = keepForwardOnly(Res.getInfo(Op));
              if (OpInfo.isValid())
                mergeInfo(Temp, OpInfo);
            }
            if (Temp.isValid() && updateInfo(Res.getMap()[&I], Temp))
              Changed = true;
          } else if (auto *CB = dyn_cast<CallBase>(&I)) {
            Function *Callee = CB->getCalledFunction();
            if (!Callee) {
              Value *CalledVal = CB->getCalledOperand()->stripPointerCasts();
              Callee = dyn_cast<Function>(CalledVal);
            }
            if (Callee && !Callee->isDeclaration()) {
              unsigned ArgIdx = 0;
              for (Argument &Arg : Callee->args()) {
                if (ArgIdx < CB->arg_size()) {
                  Value *ArgVal = CB->getArgOperand(ArgIdx);
                  auto ArgInfo = keepForwardOnly(Res.getInfo(ArgVal));
                  if (ArgInfo.isValid()) {
                    if (mergeInfo(Res.getMap()[&Arg], ArgInfo))
                      Changed = true;
                  }
                }
                ArgIdx++;
              }
              if (I.getType()->isPointerTy()) {
                for (BasicBlock &CalleeBB : *Callee) {
                  if (auto *RI =
                          dyn_cast<ReturnInst>(CalleeBB.getTerminator())) {
                    Value *RetVal = RI->getReturnValue();
                    if (RetVal) {
                      auto RetInfo = keepForwardOnly(Res.getInfo(RetVal));
                      if (RetInfo.isValid()) {
                        if (mergeInfo(Res.getMap()[&I], RetInfo))
                          Changed = true;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    // Backward propagation
    for (Function &F : M) {
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          auto Info = Res.getInfo(&I);
          if (!Info.isValid() ||
              (Info.Dir & Result::ProvenanceInfo::Backward) == 0)
            continue;

          if (auto *GEP = dyn_cast<GEPOperator>(&I)) {
            Value *Base = GEP->getPointerOperand();
            int64_t ConstOff = 0;
            std::vector<std::string> SymOffs;
            extractGEPOffsets(GEP, DL, ConstOff, SymOffs);
            auto BaseNew = subOffset(Info, ConstOff, SymOffs);
            BaseNew.Dir = Result::ProvenanceInfo::Backward;
            if (mergeInfo(Res.getMap()[Base], BaseNew))
              Changed = true;
          } else if (auto *BC = dyn_cast<BitCastOperator>(&I)) {
            Value *Op = BC->getOperand(0);
            auto OpNew = Info;
            OpNew.Dir = Result::ProvenanceInfo::Backward;
            if (mergeInfo(Res.getMap()[Op], OpNew))
              Changed = true;
          } else if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(&I)) {
            Value *Op = ASC->getOperand(0);
            auto OpNew = Info;
            OpNew.Dir = Result::ProvenanceInfo::Backward;
            if (mergeInfo(Res.getMap()[Op], OpNew))
              Changed = true;
          } else if (auto *ITP = dyn_cast<IntToPtrInst>(&I)) {
            Value *Op = ITP->getOperand(0);
            if (auto *BO = dyn_cast<BinaryOperator>(Op)) {
              if (BO->getOpcode() == Instruction::Add ||
                  BO->getOpcode() == Instruction::Sub) {
                Value *LHS = BO->getOperand(0);
                Value *RHS = BO->getOperand(1);
                auto *PTI = dyn_cast<PtrToIntOperator>(LHS);
                if (!PTI) {
                  PTI = dyn_cast<PtrToIntOperator>(RHS);
                  if (PTI && BO->getOpcode() == Instruction::Add)
                    std::swap(LHS, RHS);
                  else
                    PTI = nullptr;
                }
                if (PTI) {
                  Value *BasePtr = PTI->getOperand(0);
                  int64_t ConstOff = 0;
                  std::vector<std::string> SymOffs;
                  if (auto *CInt = dyn_cast<ConstantInt>(RHS)) {
                    ConstOff = CInt->getSExtValue();
                  } else {
                    SymOffs.push_back(getValueOperandName(RHS));
                  }
                  if (BO->getOpcode() == Instruction::Sub) {
                    ConstOff = -ConstOff;
                    for (auto &S : SymOffs)
                      S = "-(" + S + ")";
                  }
                  auto BaseNew = subOffset(Info, ConstOff, SymOffs);
                  BaseNew.Dir = Result::ProvenanceInfo::Backward;
                  if (mergeInfo(Res.getMap()[BasePtr], BaseNew))
                    Changed = true;
                }
              }
            } else if (auto *PTI = dyn_cast<PtrToIntOperator>(Op)) {
              Value *BasePtr = PTI->getOperand(0);
              auto BaseNew = Info;
              BaseNew.Dir = Result::ProvenanceInfo::Backward;
              if (mergeInfo(Res.getMap()[BasePtr], BaseNew))
                Changed = true;
            }
          } else if (auto *PHI = dyn_cast<PHINode>(&I)) {
            for (Value *InV : PHI->incoming_values()) {
              if (isa<ConstantPointerNull>(InV) || isa<UndefValue>(InV))
                continue;
              auto InNew = Info;
              InNew.Dir = Result::ProvenanceInfo::Backward;
              if (mergeInfo(Res.getMap()[InV], InNew))
                Changed = true;
            }
          } else if (auto *Sel = dyn_cast<SelectInst>(&I)) {
            for (Value *Op : {Sel->getTrueValue(), Sel->getFalseValue()}) {
              if (isa<ConstantPointerNull>(Op) || isa<UndefValue>(Op))
                continue;
              auto OpNew = Info;
              OpNew.Dir = Result::ProvenanceInfo::Backward;
              if (mergeInfo(Res.getMap()[Op], OpNew))
                Changed = true;
            }
          } else if (auto *CB = dyn_cast<CallBase>(&I)) {
            Function *Callee = CB->getCalledFunction();
            if (!Callee) {
              Value *CalledVal = CB->getCalledOperand()->stripPointerCasts();
              Callee = dyn_cast<Function>(CalledVal);
            }
            if (Callee && !Callee->isDeclaration()) {
              if (I.getType()->isPointerTy()) {
                for (BasicBlock &CalleeBB : *Callee) {
                  if (auto *RI =
                          dyn_cast<ReturnInst>(CalleeBB.getTerminator())) {
                    Value *RetVal = RI->getReturnValue();
                    if (RetVal) {
                      auto RetNew = Info;
                      RetNew.Dir = Result::ProvenanceInfo::Backward;
                      if (mergeInfo(Res.getMap()[RetVal], RetNew))
                        Changed = true;
                    }
                  }
                }
              }
            }
          }
        }
      }

      for (Argument &Arg : F.args()) {
        auto ArgInfo = Res.getInfo(&Arg);
        if (!ArgInfo.isValid() ||
            (ArgInfo.Dir & Result::ProvenanceInfo::Backward) == 0)
          continue;
        for (User *U : F.users()) {
          if (auto *CB = dyn_cast<CallBase>(U)) {
            if (CB->getCalledOperand()->stripPointerCasts() == &F ||
                CB->getCalledFunction() == &F) {
              unsigned ArgIdx = Arg.getArgNo();
              if (ArgIdx < CB->arg_size()) {
                Value *CallArg = CB->getArgOperand(ArgIdx);
                auto CallArgNew = ArgInfo;
                CallArgNew.Dir = Result::ProvenanceInfo::Backward;
                if (mergeInfo(Res.getMap()[CallArg], CallArgNew))
                  Changed = true;
              }
            }
          }
        }
      }
    }
  }

  return Res;
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
           << (Info.State ==
                       HeapProvenanceAnalysisResult::ProvenanceInfo::HeapChunkPtr
                   ? "HeapChunkPtr"
                   : "RecoverableHeapChunkPtr")
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
             << (Info.State ==
                         HeapProvenanceAnalysisResult::ProvenanceInfo::HeapChunkPtr
                     ? "HeapChunkPtr"
                     : "RecoverableHeapChunkPtr")
             << " (" << Info.getExpr() << ")" << Info.getDirectionStr() << "\n";
        }
      }
    }
  }
  return PreservedAnalyses::all();
}
