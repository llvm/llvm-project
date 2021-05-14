//===- SIInsertHardClauses.cpp - Insert Hard Clauses ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert s_clause instructions to form hard clauses.
///
/// Clausing load instructions can give cache coherency benefits. Before gfx10,
/// the hardware automatically detected "soft clauses", which were sequences of
/// memory instructions of the same type. In gfx10 this detection was removed,
/// and the s_clause instruction was introduced to explicitly mark "hard
/// clauses".
///
/// It's the scheduler's job to form the clauses by putting similar memory
/// instructions next to each other. Our job is just to insert an s_clause
/// instruction to mark the start of each clause.
///
/// Note that hard clauses are very similar to, but logically distinct from, the
/// groups of instructions that have to be restartable when XNACK is enabled.
/// The rules are slightly different in each case. For example an s_nop
/// instruction breaks a restartable group, but can appear in the middle of a
/// hard clause. (Before gfx10 there wasn't a distinction, and both were called
/// "soft clauses" or just "clauses".)
///
/// The SIFormMemoryClauses pass and GCNHazardRecognizer deal with restartable
/// groups, not hard clauses.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

#define DEBUG_TYPE "si-insert-hard-clauses"

namespace {

enum HardClauseType {
  // Texture, buffer, global or scratch memory instructions.
  HARDCLAUSE_VMEM,
  // Flat (not global or scratch) memory instructions.
  HARDCLAUSE_FLAT,
  // Instructions that access LDS.
  HARDCLAUSE_LDS,
  // Scalar memory instructions.
  HARDCLAUSE_SMEM,
  // VALU instructions.
  HARDCLAUSE_VALU,
  LAST_REAL_HARDCLAUSE_TYPE = HARDCLAUSE_VALU,

  // Internal instructions, which are allowed in the middle of a hard clause,
  // except for s_waitcnt.
  HARDCLAUSE_INTERNAL,
  // Instructions that are not allowed in a hard clause: SALU, export, branch,
  // message, GDS, s_waitcnt and anything else not mentioned above.
  HARDCLAUSE_ILLEGAL,
};

class SIInsertHardClauses : public MachineFunctionPass {
public:
  static char ID;
  const GCNSubtarget *ST = nullptr;

  SIInsertHardClauses() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  HardClauseType getHardClauseType(const MachineInstr &MI) {

    // On current architectures we only get a benefit from clausing loads.
    if (MI.mayLoad()) {
      if (SIInstrInfo::isVMEM(MI) || SIInstrInfo::isSegmentSpecificFLAT(MI)) {
        if (ST->hasNSAClauseBug()) {
          const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(MI.getOpcode());
          if (Info && Info->MIMGEncoding == AMDGPU::MIMGEncGfx10NSA)
            return HARDCLAUSE_ILLEGAL;
        }
        return HARDCLAUSE_VMEM;
      }
      if (SIInstrInfo::isFLAT(MI))
        return HARDCLAUSE_FLAT;
      // TODO: LDS
      if (SIInstrInfo::isSMRD(MI))
        return HARDCLAUSE_SMEM;
    }

    // Don't form VALU clauses. It's not clear what benefit they give, if any.

    // In practice s_nop is the only internal instruction we're likely to see.
    // It's safe to treat the rest as illegal.
    if (MI.getOpcode() == AMDGPU::S_NOP)
      return HARDCLAUSE_INTERNAL;
    return HARDCLAUSE_ILLEGAL;
  }

  // Track information about a clause as we discover it.
  struct ClauseInfo {
    // The type of all (non-internal) instructions in the clause.
    HardClauseType Type = HARDCLAUSE_ILLEGAL;
    // The first (necessarily non-internal) instruction in the clause.
    MachineInstr *First = nullptr;
    // The last non-internal instruction in the clause.
    MachineInstr *Last = nullptr;
    // The length of the clause including any internal instructions in the
    // middle or after the end of the clause.
    unsigned Length = 0;
    // The base operands of *Last.
    SmallVector<const MachineOperand *, 4> BaseOps;
  };

  bool emitClause(const ClauseInfo &CI, const SIInstrInfo *SII) {
    // Get the size of the clause excluding any internal instructions at the
    // end.
    unsigned Size =
        std::distance(CI.First->getIterator(), CI.Last->getIterator()) + 1;
    if (Size < 2)
      return false;
    assert(Size <= 64 && "Hard clause is too long!");

    auto &MBB = *CI.First->getParent();
    auto ClauseMI =
        BuildMI(MBB, *CI.First, DebugLoc(), SII->get(AMDGPU::S_CLAUSE))
            .addImm(Size - 1);
    finalizeBundle(MBB, ClauseMI->getIterator(),
                   std::next(CI.Last->getIterator()));
    return true;
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;

    ST = &MF.getSubtarget<GCNSubtarget>();
    if (!ST->hasHardClauses())
      return false;

    const SIInstrInfo *SII = ST->getInstrInfo();
    const TargetRegisterInfo *TRI = ST->getRegisterInfo();

    bool Changed = false;
    for (auto &MBB : MF) {
      ClauseInfo CI;
      for (auto &MI : MBB) {
        HardClauseType Type = getHardClauseType(MI);

        int64_t Dummy1;
        bool Dummy2;
        unsigned Dummy3;
        SmallVector<const MachineOperand *, 4> BaseOps;
        if (Type <= LAST_REAL_HARDCLAUSE_TYPE) {
          if (!SII->getMemOperandsWithOffsetWidth(MI, BaseOps, Dummy1, Dummy2,
                                                  Dummy3, TRI)) {
            // We failed to get the base operands, so we'll never clause this
            // instruction with any other, so pretend it's illegal.
            Type = HARDCLAUSE_ILLEGAL;
          }
        }

        if (CI.Length == 64 ||
            (CI.Length && Type != HARDCLAUSE_INTERNAL &&
             (Type != CI.Type ||
              // Note that we lie to shouldClusterMemOps about the size of the
              // cluster. When shouldClusterMemOps is called from the machine
              // scheduler it limits the size of the cluster to avoid increasing
              // register pressure too much, but this pass runs after register
              // allocation so there is no need for that kind of limit.
              !SII->shouldClusterMemOps(CI.BaseOps, BaseOps, 2, 2)))) {
          // Finish the current clause.
          Changed |= emitClause(CI, SII);
          CI = ClauseInfo();
        }

        if (CI.Length) {
          // Extend the current clause.
          ++CI.Length;
          if (Type != HARDCLAUSE_INTERNAL) {
            CI.Last = &MI;
            CI.BaseOps = std::move(BaseOps);
          }
        } else if (Type <= LAST_REAL_HARDCLAUSE_TYPE) {
          // Start a new clause.
          CI = ClauseInfo{Type, &MI, &MI, 1, std::move(BaseOps)};
        }
      }

      // Finish the last clause in the basic block if any.
      if (CI.Length)
        Changed |= emitClause(CI, SII);
    }

    return Changed;
  }
};

} // namespace

char SIInsertHardClauses::ID = 0;

char &llvm::SIInsertHardClausesID = SIInsertHardClauses::ID;

INITIALIZE_PASS(SIInsertHardClauses, DEBUG_TYPE, "SI Insert Hard Clauses",
                false, false)
