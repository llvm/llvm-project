//===-- PrefetchInsertion.cpp ---=========-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Prefetch insertion pass implementation.
//===----------------------------------------------------------------------===//
/// Prefetch insertion pass.
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/BasicBlockSectionUtils.h"
#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Target/TargetMachine.h"
#include <map>

using namespace llvm;
#define DEBUG_TYPE "prefetchinsertion"

static cl::opt<bool> UseCodePrefetchInstruction(
    "use-code-prefetch-instruction",
    cl::desc("Whether to use the new prefetchit1 instruction."), cl::init(true),
    cl::Hidden);
static cl::opt<bool> PrefetchNextAddress(
    "prefetch-next-address",
    cl::desc(
        "Whether to prefetch the next address instead of the target address."),
    cl::init(false), cl::Hidden);

namespace {} // end anonymous namespace

namespace llvm {
class PrefetchInsertion : public MachineFunctionPass {
public:
  static char ID;

  BasicBlockSectionsProfileReaderWrapperPass *BBSectionsProfileReader = nullptr;

  PrefetchInsertion() : MachineFunctionPass(ID) {
    initializePrefetchInsertionPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Prefetch Insertion Pass"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Identify basic blocks that need separate sections and prepare to emit them
  /// accordingly.
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace llvm

char PrefetchInsertion::ID = 0;
INITIALIZE_PASS_BEGIN(
    PrefetchInsertion, "prefetch-insertion",
    "Applies path clonings for the -basic-block-sections=list option", false,
    false)
INITIALIZE_PASS_DEPENDENCY(BasicBlockSectionsProfileReaderWrapperPass)
INITIALIZE_PASS_END(
    PrefetchInsertion, "prefetch-insertion",
    "Applies path clonings for the -basic-block-sections=list option", false,
    false)

bool PrefetchInsertion::runOnMachineFunction(MachineFunction &MF) {
  assert(MF.getTarget().getBBSectionsType() == BasicBlockSection::List &&
         "BB Sections list not enabled!");
  if (hasInstrProfHashMismatch(MF))
    return false;
  // errs() << "Running on " << MF.getName() << "\n";
  Function &F = MF.getFunction();
  auto PtrTy = PointerType::getUnqual(F.getParent()->getContext());
  DenseSet<BBPosition> PrefetchTargets =
      getAnalysis<BasicBlockSectionsProfileReaderWrapperPass>()
          .getPrefetchTargetsForFunction(MF.getName());
  // errs() << "Targets: Function: " << F.getName() << " "
  //        << PrefetchTargets.size() << "\n";
  DenseMap<UniqueBBID, SmallVector<unsigned>> PrefetchTargetsByBBID;
  for (const auto &P : PrefetchTargets)
    PrefetchTargetsByBBID[P.BBID].push_back(P.BBOffset);
  for (auto &[BBID, V] : PrefetchTargetsByBBID)
    llvm::sort(V);
  for (auto &BB : MF)
    BB.setPrefetchTargets(PrefetchTargetsByBBID[*BB.getBBID()]);

  for (const BBPosition &P : PrefetchTargets) {
    SmallString<128> PrefetchTargetName("__llvm_prefetch_target_");
    PrefetchTargetName += F.getName();
    PrefetchTargetName += "_";
    PrefetchTargetName += utostr(P.BBID.BaseID);
    PrefetchTargetName += "_";
    PrefetchTargetName += utostr(P.BBOffset);
    F.getParent()->getOrInsertGlobal(PrefetchTargetName, PtrTy);
  }

  SmallVector<PrefetchHint> PrefetchHints =
      getAnalysis<BasicBlockSectionsProfileReaderWrapperPass>()
          .getPrefetchHintsForFunction(MF.getName());
  // errs() << "Hints: Function: " << F.getName() << " " << PrefetchHints.size()
  //        << "\n";
  for (const PrefetchHint &H : PrefetchHints) {
    SmallString<128> PrefetchTargetName("__llvm_prefetch_target_");
    PrefetchTargetName += H.TargetFunctionName;
    PrefetchTargetName += "_";
    PrefetchTargetName += utostr(H.TargetPosition.BBID.BaseID);
    PrefetchTargetName += "_";
    PrefetchTargetName += utostr(H.TargetPosition.BBOffset);
    F.getParent()->getOrInsertGlobal(PrefetchTargetName, PtrTy);
  }

  DenseMap<UniqueBBID, std::map<unsigned, SmallVector<PrefetchTarget>>>
      PrefetchHintsByBBID;
  for (const auto &H : PrefetchHints) {
    PrefetchHintsByBBID[H.SitePosition.BBID][H.SitePosition.BBOffset].push_back(
        PrefetchTarget{H.TargetFunctionName, H.TargetPosition.BBID,
                       H.TargetPosition.BBOffset});
  }
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  for (auto &BB : MF) {
    auto It = PrefetchHintsByBBID.find(*BB.getBBID());
    if (It == PrefetchHintsByBBID.end())
      continue;
    auto BBPrefetchHintIt = It->second.begin();
    unsigned NumInsts = 0;
    auto E = BB.getFirstTerminator();
    unsigned NumCallsites = 0;
    for (auto I = BB.instr_begin();;) {
      auto Current = I;
      if (NumCallsites >= BBPrefetchHintIt->first || Current == E) {
        for (const auto &PrefetchTarget : BBPrefetchHintIt->second) {
          SmallString<128> PrefetchTargetName("__llvm_prefetch_target_");
          PrefetchTargetName += PrefetchTarget.TargetFunction;
          PrefetchTargetName += "_";
          PrefetchTargetName += utostr(PrefetchTarget.TargetBBID.BaseID);
          PrefetchTargetName += "_";
          PrefetchTargetName += utostr(PrefetchTarget.TargetBBOffset);
          auto *GV =
              MF.getFunction().getParent()->getNamedValue(PrefetchTargetName);
          // errs() << "Inserting prefetch for " << GV->getName() << " at "
          //        << MF.getName() << " " << BB.getName() << " " << NumInsts
          //        << "\n";
          MachineInstr *PFetch = MF.CreateMachineInstr(
              UseCodePrefetchInstruction ? TII->get(X86::PREFETCHIT1)
                                         : TII->get(X86::PREFETCHT1),
              Current != BB.instr_end() ? Current->getDebugLoc() : DebugLoc(),
              true);
          PFetch->setFlag(MachineInstr::Prefetch);
          MachineInstrBuilder MIB(MF, PFetch);
          if (!PrefetchNextAddress) {
            MIB.addMemOperand(MF.getMachineMemOperand(
                MachinePointerInfo(GV), MachineMemOperand::MOLoad, /*s=*/8,
                /*base_alignment=*/llvm::Align(1)));
          }
          MIB.addReg(X86::RIP).addImm(1).addReg(X86::NoRegister);
          if (PrefetchNextAddress)
            MIB.addImm(0);
          else
            MIB.addGlobalAddress(GV);
          MIB.addReg(X86::NoRegister);
          BB.insert(Current, PFetch);
        }
        ++BBPrefetchHintIt;
        if (BBPrefetchHintIt == PrefetchHintsByBBID[*BB.getBBID()].end())
          break;
      }
      if (Current != E) {
        // Print the assembly for the instruction.
        if (!Current->isPosition() && !Current->isImplicitDef() &&
            !Current->isKill() && !Current->isDebugInstr()) {
          ++NumInsts;
        }
        if (Current->isCall())
          ++NumCallsites;
        ++I;
      }
    }
  }
  return true;
}

void PrefetchInsertion::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<BasicBlockSectionsProfileReaderWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

FunctionPass *llvm::createPrefetchInsertionPass() {
  return new PrefetchInsertion();
}
