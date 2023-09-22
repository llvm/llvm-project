//===-- BasicBlockSections.cpp ---=========--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BasicBlockSections implementation.
//
// The purpose of this pass is to assign sections to basic blocks when
// -fbasic-block-sections= option is used. Further, with profile information
// only the subset of basic blocks with profiles are placed in separate sections
// and the rest are grouped in a cold section. The exception handling blocks are
// treated specially to ensure they are all in one seciton.
//
// Basic Block Sections
// ====================
//
// With option, -fbasic-block-sections=list, every function may be split into
// clusters of basic blocks. Every cluster will be emitted into a separate
// section with its basic blocks sequenced in the given order. To get the
// optimized performance, the clusters must form an optimal BB layout for the
// function. We insert a symbol at the beginning of every cluster's section to
// allow the linker to reorder the sections in any arbitrary sequence. A global
// order of these sections would encapsulate the function layout.
// For example, consider the following clusters for a function foo (consisting
// of 6 basic blocks 0, 1, ..., 5).
//
// 0 2
// 1 3 5
//
// * Basic blocks 0 and 2 are placed in one section with symbol `foo`
//   referencing the beginning of this section.
// * Basic blocks 1, 3, 5 are placed in a separate section. A new symbol
//   `foo.__part.1` will reference the beginning of this section.
// * Basic block 4 (note that it is not referenced in the list) is placed in
//   one section, and a new symbol `foo.cold` will point to it.
//
// There are a couple of challenges to be addressed:
//
// 1. The last basic block of every cluster should not have any implicit
//    fallthrough to its next basic block, as it can be reordered by the linker.
//    The compiler should make these fallthroughs explicit by adding
//    unconditional jumps..
//
// 2. All inter-cluster branch targets would now need to be resolved by the
//    linker as they cannot be calculated during compile time. This is done
//    using static relocations. Further, the compiler tries to use short branch
//    instructions on some ISAs for small branch offsets. This is not possible
//    for inter-cluster branches as the offset is not determined at compile
//    time, and therefore, long branch instructions have to be used for those.
//
// 3. Debug Information (DebugInfo) and Call Frame Information (CFI) emission
//    needs special handling with basic block sections. DebugInfo needs to be
//    emitted with more relocations as basic block sections can break a
//    function into potentially several disjoint pieces, and CFI needs to be
//    emitted per cluster. This also bloats the object file and binary sizes.
//
// Basic Block Labels
// ==================
//
// With -fbasic-block-sections=labels, we encode the offsets of BB addresses of
// every function into the .llvm_bb_addr_map section. Along with the function
// symbols, this allows for mapping of virtual addresses in PMU profiles back to
// the corresponding basic blocks. This logic is implemented in AsmPrinter. This
// pass only assigns the BBSectionType of every function to ``labels``.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/BasicBlockSectionUtils.h"
#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include <optional>

using namespace llvm;

// Placing the cold clusters in a separate section mitigates against poor
// profiles and allows optimizations such as hugepage mapping to be applied at a
// section granularity. Defaults to ".text.split." which is recognized by lld
// via the `-z keep-text-section-prefix` flag.
cl::opt<std::string> llvm::BBSectionsColdTextPrefix(
    "bbsections-cold-text-prefix",
    cl::desc("The text prefix to use for cold basic block clusters"),
    cl::init(".text.split."), cl::Hidden);

static cl::opt<bool> BBSectionsDetectSourceDrift(
    "bbsections-detect-source-drift",
    cl::desc("This checks if there is a fdo instr. profile hash "
             "mismatch for this function"),
    cl::init(true), cl::Hidden);

namespace {

class BasicBlockSections : public MachineFunctionPass {
public:
  static char ID;

  BasicBlockSectionsProfileReader *BBSectionsProfileReader = nullptr;

  BasicBlockSections() : MachineFunctionPass(ID) {
    initializeBasicBlockSectionsPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Basic Block Sections Analysis";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Identify basic blocks that need separate sections and prepare to emit them
  /// accordingly.
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char BasicBlockSections::ID = 0;
INITIALIZE_PASS_BEGIN(
    BasicBlockSections, "bbsections-prepare",
    "Prepares for basic block sections, by splitting functions "
    "into clusters of basic blocks.",
    false, false)
INITIALIZE_PASS_DEPENDENCY(BasicBlockSectionsProfileReader)
INITIALIZE_PASS_END(BasicBlockSections, "bbsections-prepare",
                    "Prepares for basic block sections, by splitting functions "
                    "into clusters of basic blocks.",
                    false, false)

// This function updates and optimizes the branching instructions of every basic
// block in a given function to account for changes in the layout.
static void
updateBranches(MachineFunction &MF,
               const SmallVector<MachineBasicBlock *> &PreLayoutFallThroughs) {
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  SmallVector<MachineOperand, 4> Cond;
  for (auto &MBB : MF) {
    auto NextMBBI = std::next(MBB.getIterator());
    auto *FTMBB = PreLayoutFallThroughs[MBB.getNumber()];
    // If this block had a fallthrough before we need an explicit unconditional
    // branch to that block if either
    //     1- the block ends a section, which means its next block may be
    //        reorderd by the linker, or
    //     2- the fallthrough block is not adjacent to the block in the new
    //        order.
    if (FTMBB && (MBB.isEndSection() || &*NextMBBI != FTMBB))
      TII->insertUnconditionalBranch(MBB, FTMBB, MBB.findBranchDebugLoc());

    // We do not optimize branches for machine basic blocks ending sections, as
    // their adjacent block might be reordered by the linker.
    if (MBB.isEndSection())
      continue;

    // It might be possible to optimize branches by flipping the branch
    // condition.
    Cond.clear();
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr; // For analyzeBranch.
    if (TII->analyzeBranch(MBB, TBB, FBB, Cond))
      continue;
    MBB.updateTerminator(FTMBB);
  }
}

// This function sorts basic blocks according to the cluster's information.
// All explicitly specified clusters of basic blocks will be ordered
// accordingly. All non-specified BBs go into a separate "Cold" section.
// Additionally, if exception handling landing pads end up in more than one
// clusters, they are moved into a single "Exception" section. Eventually,
// clusters are ordered in increasing order of their IDs, with the "Exception"
// and "Cold" succeeding all other clusters.
// BBProfilesByBBID represents the cluster information for basic blocks. It
// maps from BBID of basic blocks to their cluster information. If this is
// empty, it means unique sections for all basic blocks in the function.
static void assignSections(
    MachineFunction &MF,
    const DenseMap<unsigned, BBProfile<unsigned>> &BBProfilesByBBID) {
  assert(MF.hasBBSections() && "BB Sections is not set for function.");
  // This variable stores the section ID of the cluster containing eh_pads (if
  // all eh_pads are one cluster). If more than one cluster contain eh_pads, we
  // set it equal to ExceptionSectionID.
  std::optional<MBBSectionID> EHPadsSectionID;

  for (auto &MBB : MF) {
    // With the 'all' option, every basic block is placed in a unique section.
    // With the 'list' option, every basic block is placed in a section
    // associated with its cluster, unless we want individual unique sections
    // for every basic block in this function (if BBProfilesByBBID is empty).
    if (MF.getTarget().getBBSectionsType() == llvm::BasicBlockSection::All ||
        BBProfilesByBBID.empty()) {
      // If unique sections are desired for all basic blocks of the function, we
      // set every basic block's section ID equal to its original position in
      // the layout (which is equal to its number). This ensures that basic
      // blocks are ordered canonically.
      MBB.setSectionID(MBB.getNumber());
    } else {
      auto I = BBProfilesByBBID.find(*MBB.getBBID());
      if (I != BBProfilesByBBID.end()) {
        MBB.setSectionID(I->second.ClusterID);
      } else {
        // BB goes into the special cold section if it is not specified in the
        // cluster info map.
        MBB.setSectionID(MBBSectionID::ColdSectionID);
      }
    }

    if (MBB.isEHPad() && EHPadsSectionID != MBB.getSectionID() &&
        EHPadsSectionID != MBBSectionID::ExceptionSectionID) {
      // If we already have one cluster containing eh_pads, this must be updated
      // to ExceptionSectionID. Otherwise, we set it equal to the current
      // section ID.
      EHPadsSectionID = EHPadsSectionID ? MBBSectionID::ExceptionSectionID
                                        : MBB.getSectionID();
    }
  }

  // If EHPads are in more than one section, this places all of them in the
  // special exception section.
  if (EHPadsSectionID == MBBSectionID::ExceptionSectionID)
    for (auto &MBB : MF)
      if (MBB.isEHPad())
        MBB.setSectionID(*EHPadsSectionID);
}

void llvm::sortBasicBlocksAndUpdateBranches(
    MachineFunction &MF, MachineBasicBlockComparator MBBCmp) {
  [[maybe_unused]] const MachineBasicBlock *EntryBlock = &MF.front();
  SmallVector<MachineBasicBlock *> PreLayoutFallThroughs(MF.getNumBlockIDs());
  for (auto &MBB : MF)
    PreLayoutFallThroughs[MBB.getNumber()] =
        MBB.getFallThrough(/*JumpToFallThrough=*/false);

  MF.sort(MBBCmp);
  assert(&MF.front() == EntryBlock &&
         "Entry block should not be displaced by basic block sections");

  // Set IsBeginSection and IsEndSection according to the assigned section IDs.
  MF.assignBeginEndSections();

  // After reordering basic blocks, we must update basic block branches to
  // insert explicit fallthrough branches when required and optimize branches
  // when possible.
  updateBranches(MF, PreLayoutFallThroughs);
}

// If the exception section begins with a landing pad, that landing pad will
// assume a zero offset (relative to @LPStart) in the LSDA. However, a value of
// zero implies "no landing pad." This function inserts a NOP just before the EH
// pad label to ensure a nonzero offset.
void llvm::avoidZeroOffsetLandingPad(MachineFunction &MF) {
  for (auto &MBB : MF) {
    if (MBB.isBeginSection() && MBB.isEHPad()) {
      MachineBasicBlock::iterator MI = MBB.begin();
      while (!MI->isEHLabel())
        ++MI;
      MF.getSubtarget().getInstrInfo()->insertNoop(MBB, MI);
    }
  }
}

// This checks if the source of this function has drifted since this binary was
// profiled previously.  For now, we are piggy backing on what PGO does to
// detect this with instrumented profiles.  PGO emits an hash of the IR and
// checks if the hash has changed.  Advanced basic block layout is usually done
// on top of PGO optimized binaries and hence this check works well in practice.
static bool hasInstrProfHashMismatch(MachineFunction &MF) {
  if (!BBSectionsDetectSourceDrift)
    return false;

  const char MetadataName[] = "instr_prof_hash_mismatch";
  auto *Existing = MF.getFunction().getMetadata(LLVMContext::MD_annotation);
  if (Existing) {
    MDTuple *Tuple = cast<MDTuple>(Existing);
    for (const auto &N : Tuple->operands())
      if (N.equalsStr(MetadataName))
        return true;
  }

  return false;
}

bool BasicBlockSections::runOnMachineFunction(MachineFunction &MF) {
  auto BBSectionsType = MF.getTarget().getBBSectionsType();
  assert(BBSectionsType != BasicBlockSection::None &&
         "BB Sections not enabled!");

  // Check for source drift.  If the source has changed since the profiles
  // were obtained, optimizing basic blocks might be sub-optimal.
  // This only applies to BasicBlockSection::List as it creates
  // clusters of basic blocks using basic block ids. Source drift can
  // invalidate these groupings leading to sub-optimal code generation with
  // regards to performance.
  if (BBSectionsType == BasicBlockSection::List &&
      hasInstrProfHashMismatch(MF))
    return true;
  // Renumber blocks before sorting them. This is useful for accessing the
  // original layout positions and finding the original fallthroughs.
  MF.RenumberBlocks();

  if (BBSectionsType == BasicBlockSection::Labels) {
    MF.setBBSectionsType(BBSectionsType);
    return true;
  }

  DenseMap<unsigned, BBProfile<unsigned>> BBProfilesByBBID;
  if (BBSectionsType == BasicBlockSection::List) {
    auto [HasProfile, RawProfile] =
        getAnalysis<BasicBlockSectionsProfileReader>().getRawProfileForFunction(
            MF.getName());
    if (!HasProfile)
      return true;
    // TODO: Apply the path cloning profile.
    for (const BBProfile<ProfileBBID> &BBP : RawProfile.RawBBProfiles) {
      assert(!BBP.BasicBlockID.CloneID && "Path cloning is not supported yet.");
      BBProfilesByBBID.try_emplace(BBP.BasicBlockID.BBID,
                                   BBProfile<unsigned>{BBP.BasicBlockID.BBID,
                                                       BBP.ClusterID,
                                                       BBP.PositionInCluster});
    }
  }

  MF.setBBSectionsType(BBSectionsType);
  assignSections(MF, BBProfilesByBBID);

  // We make sure that the cluster including the entry basic block precedes all
  // other clusters.
  auto EntryBBSectionID = MF.front().getSectionID();

  // Helper function for ordering BB sections as follows:
  //   * Entry section (section including the entry block).
  //   * Regular sections (in increasing order of their Number).
  //     ...
  //   * Exception section
  //   * Cold section
  auto MBBSectionOrder = [EntryBBSectionID](const MBBSectionID &LHS,
                                            const MBBSectionID &RHS) {
    // We make sure that the section containing the entry block precedes all the
    // other sections.
    if (LHS == EntryBBSectionID || RHS == EntryBBSectionID)
      return LHS == EntryBBSectionID;
    return LHS.Type == RHS.Type ? LHS.Number < RHS.Number : LHS.Type < RHS.Type;
  };

  // We sort all basic blocks to make sure the basic blocks of every cluster are
  // contiguous and ordered accordingly. Furthermore, clusters are ordered in
  // increasing order of their section IDs, with the exception and the
  // cold section placed at the end of the function.
  auto Comparator = [&](const MachineBasicBlock &X,
                        const MachineBasicBlock &Y) {
    auto XSectionID = X.getSectionID();
    auto YSectionID = Y.getSectionID();
    if (XSectionID != YSectionID)
      return MBBSectionOrder(XSectionID, YSectionID);
    // If the two basic block are in the same section, the order is decided by
    // their position within the section.
    if (XSectionID.Type == MBBSectionID::SectionType::Default)
      return BBProfilesByBBID.lookup(*X.getBBID()).PositionInCluster <
             BBProfilesByBBID.lookup(*Y.getBBID()).PositionInCluster;
    return X.getNumber() < Y.getNumber();
  };

  sortBasicBlocksAndUpdateBranches(MF, Comparator);
  avoidZeroOffsetLandingPad(MF);
  return true;
}

void BasicBlockSections::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<BasicBlockSectionsProfileReader>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

MachineFunctionPass *llvm::createBasicBlockSectionsPass() {
  return new BasicBlockSections();
}
