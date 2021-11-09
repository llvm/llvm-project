//===-------- MIRSampleProfile.cpp: MIRSampleFDO (For FSAFDO) -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the MIRSampleProfile loader, mainly
// for flow sensitive SampleFDO.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRSampleProfile.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/SampleProfileLoaderBaseImpl.h"
#include "llvm/Transforms/Utils/SampleProfileLoaderBaseUtil.h"

using namespace llvm;
using namespace sampleprof;
using namespace llvm::sampleprofutil;
using ProfileCount = Function::ProfileCount;

#define DEBUG_TYPE "fs-profile-loader"

static cl::opt<bool> ShowFSBranchProb(
    "show-fs-branchprob", cl::Hidden, cl::init(false),
    cl::desc("Print setting flow sensitive branch probabilities"));
static cl::opt<unsigned> FSProfileDebugProbDiffThreshold(
    "fs-profile-debug-prob-diff-threshold", cl::init(10),
    cl::desc("Only show debug message if the branch probility is greater than "
             "this value (in percentage)."));

static cl::opt<unsigned> FSProfileDebugBWThreshold(
    "fs-profile-debug-bw-threshold", cl::init(10000),
    cl::desc("Only show debug message if the source branch weight is greater "
             " than this value."));

static cl::opt<bool> ViewBFIBefore("fs-viewbfi-before", cl::Hidden,
                                   cl::init(false),
                                   cl::desc("View BFI before MIR loader"));
static cl::opt<bool> ViewBFIAfter("fs-viewbfi-after", cl::Hidden,
                                  cl::init(false),
                                  cl::desc("View BFI after MIR loader"));

char MIRProfileLoaderPass::ID = 0;

INITIALIZE_PASS_BEGIN(MIRProfileLoaderPass, DEBUG_TYPE,
                      "Load MIR Sample Profile",
                      /* cfg = */ false, /* is_analysis = */ false)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_END(MIRProfileLoaderPass, DEBUG_TYPE, "Load MIR Sample Profile",
                    /* cfg = */ false, /* is_analysis = */ false)

char &llvm::MIRProfileLoaderPassID = MIRProfileLoaderPass::ID;

FunctionPass *llvm::createMIRProfileLoaderPass(std::string File,
                                               std::string RemappingFile,
                                               FSDiscriminatorPass P) {
  return new MIRProfileLoaderPass(File, RemappingFile, P);
}

namespace llvm {

// Internal option used to control BFI display only after MBP pass.
// Defined in CodeGen/MachineBlockFrequencyInfo.cpp:
// -view-block-layout-with-bfi={none | fraction | integer | count}
extern cl::opt<GVDAGType> ViewBlockLayoutWithBFI;

// Command line option to specify the name of the function for CFG dump
// Defined in Analysis/BlockFrequencyInfo.cpp:  -view-bfi-func-name=
extern cl::opt<std::string> ViewBlockFreqFuncName;

namespace afdo_detail {
template <> struct IRTraits<MachineBasicBlock> {
  using InstructionT = MachineInstr;
  using BasicBlockT = MachineBasicBlock;
  using FunctionT = MachineFunction;
  using BlockFrequencyInfoT = MachineBlockFrequencyInfo;
  using LoopT = MachineLoop;
  using LoopInfoPtrT = MachineLoopInfo *;
  using DominatorTreePtrT = MachineDominatorTree *;
  using PostDominatorTreePtrT = MachinePostDominatorTree *;
  using PostDominatorTreeT = MachinePostDominatorTree;
  using OptRemarkEmitterT = MachineOptimizationRemarkEmitter;
  using OptRemarkAnalysisT = MachineOptimizationRemarkAnalysis;
  using PredRangeT = iterator_range<std::vector<MachineBasicBlock *>::iterator>;
  using SuccRangeT = iterator_range<std::vector<MachineBasicBlock *>::iterator>;
  static Function &getFunction(MachineFunction &F) { return F.getFunction(); }
  static const MachineBasicBlock *getEntryBB(const MachineFunction *F) {
    return GraphTraits<const MachineFunction *>::getEntryNode(F);
  }
  static PredRangeT getPredecessors(MachineBasicBlock *BB) {
    return BB->predecessors();
  }
  static SuccRangeT getSuccessors(MachineBasicBlock *BB) {
    return BB->successors();
  }
};
} // namespace afdo_detail

class MIRProfileLoader final
    : public SampleProfileLoaderBaseImpl<MachineBasicBlock> {
public:
  void setInitVals(MachineDominatorTree *MDT, MachinePostDominatorTree *MPDT,
                   MachineLoopInfo *MLI, MachineBlockFrequencyInfo *MBFI,
                   MachineOptimizationRemarkEmitter *MORE) {
    DT = MDT;
    PDT = MPDT;
    LI = MLI;
    BFI = MBFI;
    ORE = MORE;
  }
  void setFSPass(FSDiscriminatorPass Pass) {
    P = Pass;
    LowBit = getFSPassBitBegin(P);
    HighBit = getFSPassBitEnd(P);
    assert(LowBit < HighBit && "HighBit needs to be greater than Lowbit");
  }

  MIRProfileLoader(StringRef Name, StringRef RemapName)
      : SampleProfileLoaderBaseImpl(std::string(Name), std::string(RemapName)) {
  }

  void setBranchProbs(MachineFunction &F);
  bool runOnFunction(MachineFunction &F);
  bool doInitialization(Module &M);
  bool isValid() const { return ProfileIsValid; }

protected:
  friend class SampleCoverageTracker;

  /// Hold the information of the basic block frequency.
  MachineBlockFrequencyInfo *BFI;

  /// PassNum is the sequence number this pass is called, start from 1.
  FSDiscriminatorPass P;

  // LowBit in the FS discriminator used by this instance. Note the number is
  // 0-based. Base discrimnator use bit 0 to bit 11.
  unsigned LowBit;
  // HighwBit in the FS discriminator used by this instance. Note the number
  // is 0-based.
  unsigned HighBit;

  bool ProfileIsValid = true;
};

template <>
void SampleProfileLoaderBaseImpl<
    MachineBasicBlock>::computeDominanceAndLoopInfo(MachineFunction &F) {}

void MIRProfileLoader::setBranchProbs(MachineFunction &F) {
  LLVM_DEBUG(dbgs() << "\nPropagation complete. Setting branch probs\n");
  for (auto &BI : F) {
    MachineBasicBlock *BB = &BI;
    if (BB->succ_size() < 2)
      continue;
    const MachineBasicBlock *EC = EquivalenceClass[BB];
    uint64_t BBWeight = BlockWeights[EC];
    uint64_t SumEdgeWeight = 0;
    for (MachineBasicBlock *Succ : BB->successors()) {
      Edge E = std::make_pair(BB, Succ);
      SumEdgeWeight += EdgeWeights[E];
    }

    if (BBWeight != SumEdgeWeight) {
      LLVM_DEBUG(dbgs() << "BBweight is not equal to SumEdgeWeight: BBWWeight="
                        << BBWeight << " SumEdgeWeight= " << SumEdgeWeight
                        << "\n");
      BBWeight = SumEdgeWeight;
    }
    if (BBWeight == 0) {
      LLVM_DEBUG(dbgs() << "SKIPPED. All branch weights are zero.\n");
      continue;
    }

#ifndef NDEBUG
    uint64_t BBWeightOrig = BBWeight;
#endif
    uint32_t MaxWeight = std::numeric_limits<uint32_t>::max();
    uint32_t Factor = 1;
    if (BBWeight > MaxWeight) {
      Factor = BBWeight / MaxWeight + 1;
      BBWeight /= Factor;
      LLVM_DEBUG(dbgs() << "Scaling weights by " << Factor << "\n");
    }

    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
                                          SE = BB->succ_end();
         SI != SE; ++SI) {
      MachineBasicBlock *Succ = *SI;
      Edge E = std::make_pair(BB, Succ);
      uint64_t EdgeWeight = EdgeWeights[E];
      EdgeWeight /= Factor;

      assert(BBWeight >= EdgeWeight &&
             "BBweight is larger than EdgeWeight -- should not happen.\n");

      BranchProbability OldProb = BFI->getMBPI()->getEdgeProbability(BB, SI);
      BranchProbability NewProb(EdgeWeight, BBWeight);
      if (OldProb == NewProb)
        continue;
      BB->setSuccProbability(SI, NewProb);
#ifndef NDEBUG
      if (!ShowFSBranchProb)
        continue;
      bool Show = false;
      BranchProbability Diff;
      if (OldProb > NewProb)
        Diff = OldProb - NewProb;
      else
        Diff = NewProb - OldProb;
      Show = (Diff >= BranchProbability(FSProfileDebugProbDiffThreshold, 100));
      Show &= (BBWeightOrig >= FSProfileDebugBWThreshold);

      auto DIL = BB->findBranchDebugLoc();
      auto SuccDIL = Succ->findBranchDebugLoc();
      if (Show) {
        dbgs() << "Set branch fs prob: MBB (" << BB->getNumber() << " -> "
               << Succ->getNumber() << "): ";
        if (DIL)
          dbgs() << DIL->getFilename() << ":" << DIL->getLine() << ":"
                 << DIL->getColumn();
        if (SuccDIL)
          dbgs() << "-->" << SuccDIL->getFilename() << ":" << SuccDIL->getLine()
                 << ":" << SuccDIL->getColumn();
        dbgs() << " W=" << BBWeightOrig << "  " << OldProb << " --> " << NewProb
               << "\n";
      }
#endif
    }
  }
}

bool MIRProfileLoader::doInitialization(Module &M) {
  auto &Ctx = M.getContext();

  auto ReaderOrErr = sampleprof::SampleProfileReader::create(Filename, Ctx, P,
                                                             RemappingFilename);
  if (std::error_code EC = ReaderOrErr.getError()) {
    std::string Msg = "Could not open profile: " + EC.message();
    Ctx.diagnose(DiagnosticInfoSampleProfile(Filename, Msg));
    return false;
  }

  Reader = std::move(ReaderOrErr.get());
  Reader->setModule(&M);
  ProfileIsValid = (Reader->read() == sampleprof_error::success);
  Reader->getSummary();

  return true;
}

bool MIRProfileLoader::runOnFunction(MachineFunction &MF) {
  Function &Func = MF.getFunction();
  clearFunctionData(false);
  Samples = Reader->getSamplesFor(Func);
  if (!Samples || Samples->empty())
    return false;

  if (getFunctionLoc(MF) == 0)
    return false;

  DenseSet<GlobalValue::GUID> InlinedGUIDs;
  bool Changed = computeAndPropagateWeights(MF, InlinedGUIDs);

  // Set the new BPI, BFI.
  setBranchProbs(MF);

  return Changed;
}

} // namespace llvm

MIRProfileLoaderPass::MIRProfileLoaderPass(std::string FileName,
                                           std::string RemappingFileName,
                                           FSDiscriminatorPass P)
    : MachineFunctionPass(ID), ProfileFileName(FileName), P(P),
      MIRSampleLoader(
          std::make_unique<MIRProfileLoader>(FileName, RemappingFileName)) {
  LowBit = getFSPassBitBegin(P);
  HighBit = getFSPassBitEnd(P);
  assert(LowBit < HighBit && "HighBit needs to be greater than Lowbit");
}

bool MIRProfileLoaderPass::runOnMachineFunction(MachineFunction &MF) {
  if (!MIRSampleLoader->isValid())
    return false;

  LLVM_DEBUG(dbgs() << "MIRProfileLoader pass working on Func: "
                    << MF.getFunction().getName() << "\n");
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  MIRSampleLoader->setInitVals(
      &getAnalysis<MachineDominatorTree>(),
      &getAnalysis<MachinePostDominatorTree>(), &getAnalysis<MachineLoopInfo>(),
      MBFI, &getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE());

  MF.RenumberBlocks();
  if (ViewBFIBefore && ViewBlockLayoutWithBFI != GVDT_None &&
      (ViewBlockFreqFuncName.empty() ||
       MF.getFunction().getName().equals(ViewBlockFreqFuncName))) {
    MBFI->view("MIR_Prof_loader_b." + MF.getName(), false);
  }

  bool Changed = MIRSampleLoader->runOnFunction(MF);

  if (ViewBFIAfter && ViewBlockLayoutWithBFI != GVDT_None &&
      (ViewBlockFreqFuncName.empty() ||
       MF.getFunction().getName().equals(ViewBlockFreqFuncName))) {
    MBFI->view("MIR_prof_loader_a." + MF.getName(), false);
  }

  return Changed;
}

bool MIRProfileLoaderPass::doInitialization(Module &M) {
  LLVM_DEBUG(dbgs() << "MIRProfileLoader pass working on Module " << M.getName()
                    << "\n");

  MIRSampleLoader->setFSPass(P);
  return MIRSampleLoader->doInitialization(M);
}

void MIRProfileLoaderPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineBlockFrequencyInfo>();
  AU.addRequired<MachineDominatorTree>();
  AU.addRequired<MachinePostDominatorTree>();
  AU.addRequiredTransitive<MachineLoopInfo>();
  AU.addRequired<MachineOptimizationRemarkEmitterPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}
