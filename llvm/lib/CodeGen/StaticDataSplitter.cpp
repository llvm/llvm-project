//===- StaticDataSplitter.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass uses branch profile data to assign hotness based section qualifiers
// for the following types of static data:
// - Jump tables
// - Module-internal global variables
// - Constant pools
//
// For the original RFC of this pass please see
// https://discourse.llvm.org/t/rfc-profile-guided-static-data-partitioning/83744

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/StaticDataProfileInfo.h"
#include "llvm/CodeGen/MBFIWrapper.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;

#define DEBUG_TYPE "static-data-splitter"

STATISTIC(NumHotJumpTables, "Number of hot jump tables seen.");
STATISTIC(NumColdJumpTables, "Number of cold jump tables seen.");
STATISTIC(NumUnknownJumpTables,
          "Number of jump tables with unknown hotness. They are from functions "
          "without profile information.");

class StaticDataSplitter : public MachineFunctionPass {
  const MachineBranchProbabilityInfo *MBPI = nullptr;
  const MachineBlockFrequencyInfo *MBFI = nullptr;
  const ProfileSummaryInfo *PSI = nullptr;
  StaticDataProfileInfo *SDPI = nullptr;

  // If the global value is a local linkage global variable, return it.
  // Otherwise, return nullptr.
  const GlobalVariable *getLocalLinkageGlobalVariable(const GlobalValue *GV);

  // Returns true if the global variable is in one of {.rodata, .bss, .data,
  // .data.rel.ro} sections.
  bool inStaticDataSection(const GlobalVariable &GV, const TargetMachine &TM);

  // Returns the constant if the operand refers to a global variable or constant
  // that gets lowered to static data sections. Otherwise, return nullptr.
  const Constant *getConstant(const MachineOperand &Op, const TargetMachine &TM,
                              const MachineConstantPool *MCP);

  // Use profiles to partition static data.
  bool partitionStaticDataWithProfiles(MachineFunction &MF);

  // Update LLVM statistics for a machine function with profiles.
  void updateStatsWithProfiles(const MachineFunction &MF);

  // Update LLVM statistics for a machine function without profiles.
  void updateStatsWithoutProfiles(const MachineFunction &MF);

  void annotateStaticDataWithoutProfiles(const MachineFunction &MF);

public:
  static char ID;

  StaticDataSplitter() : MachineFunctionPass(ID) {
    initializeStaticDataSplitterPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Static Data Splitter"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    AU.addRequired<StaticDataProfileInfoWrapperPass>();
    // This pass does not modify any required analysis results except
    // StaticDataProfileInfoWrapperPass, but StaticDataProfileInfoWrapperPass
    // is made an immutable pass that it won't be re-scheduled by pass manager
    // anyway. So mark setPreservesAll() here for faster compile time.
    AU.setPreservesAll();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

bool StaticDataSplitter::runOnMachineFunction(MachineFunction &MF) {
  MBPI = &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();
  MBFI = &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();

  SDPI = &getAnalysis<StaticDataProfileInfoWrapperPass>()
              .getStaticDataProfileInfo();

  const bool ProfileAvailable = PSI && PSI->hasProfileSummary() && MBFI &&
                                MF.getFunction().hasProfileData();

  if (!ProfileAvailable) {
    annotateStaticDataWithoutProfiles(MF);
    updateStatsWithoutProfiles(MF);
    return false;
  }

  bool Changed = partitionStaticDataWithProfiles(MF);

  updateStatsWithProfiles(MF);
  return Changed;
}

const Constant *
StaticDataSplitter::getConstant(const MachineOperand &Op,
                                const TargetMachine &TM,
                                const MachineConstantPool *MCP) {
  if (!Op.isGlobal() && !Op.isCPI())
    return nullptr;

  if (Op.isGlobal()) {
    // Find global variables with local linkage.
    const GlobalVariable *GV = getLocalLinkageGlobalVariable(Op.getGlobal());
    // Skip those not eligible for annotation or not in static data sections.
    if (!GV || !llvm::memprof::IsAnnotationOK(*GV) ||
        !inStaticDataSection(*GV, TM))
      return nullptr;
    return GV;
  }
  assert(Op.isCPI() && "Op must be constant pool index in this branch");
  int CPI = Op.getIndex();
  if (CPI == -1)
    return nullptr;

  assert(MCP != nullptr && "Constant pool info is not available.");
  const MachineConstantPoolEntry &CPE = MCP->getConstants()[CPI];

  if (CPE.isMachineConstantPoolEntry())
    return nullptr;

  return CPE.Val.ConstVal;
}

bool StaticDataSplitter::partitionStaticDataWithProfiles(MachineFunction &MF) {
  // If any of the static data (jump tables, global variables, constant pools)
  // are captured by the analysis, set `Changed` to true. Note this pass won't
  // invalidate any analysis pass (see `getAnalysisUsage` above), so the main
  // purpose of tracking and conveying the change (to pass manager) is
  // informative as opposed to invalidating any analysis results. As an example
  // of where this information is useful, `PMDataManager::dumpPassInfo` will
  // only dump pass info if a local change happens, otherwise a pass appears as
  // "skipped".
  bool Changed = false;

  MachineJumpTableInfo *MJTI = MF.getJumpTableInfo();

  // Jump table could be used by either terminating instructions or
  // non-terminating ones, so we walk all instructions and use
  // `MachineOperand::isJTI()` to identify jump table operands.
  // Similarly, `MachineOperand::isCPI()` is used to identify constant pool
  // usages in the same loop.
  for (const auto &MBB : MF) {
    std::optional<uint64_t> Count = MBFI->getBlockProfileCount(&MBB);
    for (const MachineInstr &I : MBB) {
      for (const MachineOperand &Op : I.operands()) {
        if (!Op.isJTI() && !Op.isGlobal() && !Op.isCPI())
          continue;

        if (Op.isJTI()) {
          assert(MJTI != nullptr && "Jump table info is not available.");
          const int JTI = Op.getIndex();
          // This is not a source block of jump table.
          if (JTI == -1)
            continue;

          auto Hotness = MachineFunctionDataHotness::Hot;

          // Hotness is based on source basic block hotness.
          // TODO: PSI APIs are about instruction hotness. Introduce API for
          // data access hotness.
          if (Count && PSI->isColdCount(*Count))
            Hotness = MachineFunctionDataHotness::Cold;

          Changed |= MJTI->updateJumpTableEntryHotness(JTI, Hotness);
        } else if (const Constant *C =
                       getConstant(Op, MF.getTarget(), MF.getConstantPool())) {
          SDPI->addConstantProfileCount(C, Count);
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

const GlobalVariable *
StaticDataSplitter::getLocalLinkageGlobalVariable(const GlobalValue *GV) {
  // LLVM IR Verifier requires that a declaration must have valid declaration
  // linkage, and local linkages are not among the valid ones. So there is no
  // need to check GV is not a declaration here.
  return (GV && GV->hasLocalLinkage()) ? dyn_cast<GlobalVariable>(GV) : nullptr;
}

bool StaticDataSplitter::inStaticDataSection(const GlobalVariable &GV,
                                             const TargetMachine &TM) {

  SectionKind Kind = TargetLoweringObjectFile::getKindForGlobal(&GV, TM);
  return Kind.isData() || Kind.isReadOnly() || Kind.isReadOnlyWithRel() ||
         Kind.isBSS();
}

void StaticDataSplitter::updateStatsWithProfiles(const MachineFunction &MF) {
  if (!AreStatisticsEnabled())
    return;

  if (const MachineJumpTableInfo *MJTI = MF.getJumpTableInfo()) {
    for (const auto &JumpTable : MJTI->getJumpTables()) {
      if (JumpTable.Hotness == MachineFunctionDataHotness::Hot) {
        ++NumHotJumpTables;
      } else {
        assert(JumpTable.Hotness == MachineFunctionDataHotness::Cold &&
               "A jump table is either hot or cold when profile information is "
               "available.");
        ++NumColdJumpTables;
      }
    }
  }
}

void StaticDataSplitter::annotateStaticDataWithoutProfiles(
    const MachineFunction &MF) {
  for (const auto &MBB : MF)
    for (const MachineInstr &I : MBB)
      for (const MachineOperand &Op : I.operands())
        if (const Constant *C =
                getConstant(Op, MF.getTarget(), MF.getConstantPool()))
          SDPI->addConstantProfileCount(C, std::nullopt);
}

void StaticDataSplitter::updateStatsWithoutProfiles(const MachineFunction &MF) {
  if (!AreStatisticsEnabled())
    return;

  if (const MachineJumpTableInfo *MJTI = MF.getJumpTableInfo()) {
    NumUnknownJumpTables += MJTI->getJumpTables().size();
  }
}

char StaticDataSplitter::ID = 0;

INITIALIZE_PASS_BEGIN(StaticDataSplitter, DEBUG_TYPE, "Split static data",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(StaticDataProfileInfoWrapperPass)
INITIALIZE_PASS_END(StaticDataSplitter, DEBUG_TYPE, "Split static data", false,
                    false)

MachineFunctionPass *llvm::createStaticDataSplitterPass() {
  return new StaticDataSplitter();
}
