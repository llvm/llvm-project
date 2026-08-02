//===-- X86FixupInstTuning.cpp - replace instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file does a tuning pass replacing slower machine instructions
// with faster ones. We do this here, as opposed to during normal ISel, as
// attempting to get the "right" instruction can break patterns. This pass
// is not meant search for special cases where an instruction can be transformed
// to another, it is only meant to do transformations where the old instruction
// is always replacable with the new instructions. For example:
//
//      `vpermq ymm` -> `vshufd ymm`
//          -- BAD, not always valid (lane cross/non-repeated mask)
//
//      `vpermilps ymm` -> `vshufd ymm`
//          -- GOOD, always replaceable
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/IR/Analysis.h"

using namespace llvm;

#define DEBUG_TYPE "x86-fixup-inst-tuning"

STATISTIC(NumInstChanges, "Number of instructions changes");

namespace {
class X86FixupInstTuningImpl {
public:
  bool runOnMachineFunction(MachineFunction &MF);

private:
  bool processInstruction(MachineFunction &MF, MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator &I);

  bool processSpills(MachineBasicBlock &MBB, MachineFunction &MF);

  const X86InstrInfo *TII = nullptr;
  const X86Subtarget *ST = nullptr;
  const MCSchedModel *SM = nullptr;
  const X86RegisterInfo *TRI = nullptr;
};

class X86FixupInstTuningLegacy : public MachineFunctionPass {
public:
  static char ID;

  X86FixupInstTuningLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "X86 Fixup Inst Tuning"; }

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool processInstruction(MachineFunction &MF, MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator &I);

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
};
} // end anonymous namespace

char X86FixupInstTuningLegacy ::ID = 0;

INITIALIZE_PASS(X86FixupInstTuningLegacy, DEBUG_TYPE, DEBUG_TYPE, false, false)

FunctionPass *llvm::createX86FixupInstTuningLegacyPass() {
  return new X86FixupInstTuningLegacy();
}

template <typename T>
static std::optional<bool> CmpOptionals(T NewVal, T CurVal) {
  if (NewVal.has_value() && CurVal.has_value() && *NewVal != *CurVal)
    return *NewVal < *CurVal;

  return std::nullopt;
}

bool X86FixupInstTuningImpl::processSpills(MachineBasicBlock &MBB,
                                           MachineFunction &MF) {
  bool Changed = false;

  // Tracks a load->store chain: load from OrigFI into Reg, store Reg to
  // SpillFI. We want to redirect later loads from SpillFI to OrigFI.
  struct SpillEntry {
    SmallVector<MachineOperand, X86::AddrNumOperands> OrigMOs;
    MachineInstr *LoadMI = nullptr;  // The original load from OrigFI.
    MachineInstr *StoreMI = nullptr; // The store to SpillFI.
    MCRegister LoadReg;              // The register used as intermediary.
    unsigned UseCount = 0;           // Number of loads from SpillFI seen.
    unsigned RewriteCount = 0;       // Number successfully rewritten.
    LocationSize Size = LocationSize::beforeOrAfterPointer();
    int64_t Offset = 0;
    int OrigFI = -1;
    bool Invalid = false;
  };

  // Maps SpillFI -> SpillEntry (the redirect info for that slot).
  DenseMap<int, SpillEntry> SpillMap;

  // Maps physical register -> (OrigFI, address operands, LoadMI).
  // Keyed by the canonical (largest) super-register to handle aliasing.
  struct RegEntry {
    SmallVector<MachineOperand, X86::AddrNumOperands> MOs;
    MachineInstr *LoadMI = nullptr;
    int FI = -1;
    LocationSize Size = LocationSize::beforeOrAfterPointer();
  };
  DenseMap<MCRegister, RegEntry> RegToFI;

  auto InvalidateReg = [&](MCRegister Reg) {
    for (MCRegAliasIterator AI(Reg, TRI, /*IncludeSelf=*/true); AI.isValid();
         ++AI)
      RegToFI.erase(*AI);
    for (auto &[FI, SE] : SpillMap) {
      if (SE.Invalid)
        continue;
      bool UsesReg = false;
      for (const MachineOperand &MO : SE.OrigMOs) {
        if (MO.isReg() && MO.getReg() && TRI->regsOverlap(MO.getReg(), Reg)) {
          UsesReg = true;
          break;
        }
      }
      if (UsesReg)
        SE.Invalid = true;
    }
  };

  auto IsSafeToEraseLoad = [&](MachineInstr *LoadMI, MachineInstr *StoreMI,
                               MCRegister Reg) -> bool {
    // Walk forward from LoadMI. The register must not be read by anything
    // other than StoreMI, and must be redefined (or reach end-of-block
    // without being live-out).
    for (MachineBasicBlock::iterator I = std::next(LoadMI->getIterator()),
                                     E = MBB.end();
         I != E; ++I) {
      if (&*I == StoreMI)
        continue;
      if (I->readsRegister(Reg, TRI))
        return false;
      if (I->definesRegister(Reg, TRI))
        return true;
    }
    // Check if the register is live-out of the block.
    for (MCRegAliasIterator AI(Reg, TRI, /*IncludeSelf=*/true); AI.isValid();
         ++AI) {
      for (MachineBasicBlock *Succ : MBB.successors()) {
        if (Succ->isLiveIn(*AI))
          return false;
      }
    }
    return true;
  };

  for (MachineInstr &MI : MBB) {
    if (MI.isCall()) {
      // Calls clobber registers and may read/write memory; clear tracking
      // state.
      RegToFI.clear();
      SpillMap.clear();
      continue;
    }

    int LoadedFI = -1;
    MCRegister LoadedReg(TII->isLoadFromStackSlotPostFE(MI, LoadedFI));
    bool IsX87 = X86::isX87Instruction(MI);

    if (LoadedReg && IsX87) {
      auto It = SpillMap.find(LoadedFI);
      if (It != SpillMap.end() && !MI.memoperands_empty()) {
        // This load reads from a SpillFI we're tracking. Rewrite it to
        // read directly from OrigFI.
        SpillEntry &SE = It->second;
        const MachineMemOperand *MMO = nullptr;
        for (const MachineMemOperand *M : MI.memoperands()) {
          if (M->isLoad()) {
            MMO = M;
            break;
          }
        }
        if (MMO && MMO->getSize() == SE.Size && MMO->getOffset() == SE.Offset) {
          MachineFrameInfo &MFI = MF.getFrameInfo();
          int64_t OrigSize = MFI.getObjectSize(SE.OrigFI);
          if (!SE.Size.hasValue() || OrigSize >= (int64_t)SE.Size.getValue()) {
            SE.UseCount++;
            if (!SE.Invalid) {
              int MemIdx = X86::getFirstAddrOperandIdx(MI);
              if (MemIdx >= 0) {
                for (int i = 0; i < X86::AddrNumOperands; ++i)
                  MI.getOperand(MemIdx + i) = SE.OrigMOs[i];
                // Replace MMO with one describing OrigFI rather than SpillFI.
                MachinePointerInfo PtrInfo =
                    MachinePointerInfo::getFixedStack(MF, SE.OrigFI, SE.Offset);
                MachineMemOperand *NewMMO =
                    MF.getMachineMemOperand(MMO, PtrInfo, MMO->getSize());
                MI.setMemRefs(MF, {NewMMO});
                SE.RewriteCount++;
                Changed = true;
              }
            }
          }
        }
      }
    }

    // MMO-based fallback: handles x87 loads (LD_F32m, LD_F64m, etc.) that
    // aren't recognized by isLoadFromStackSlotPostFE because they aren't
    // in isFrameLoadOpcode, but do have valid FixedStackPseudoSourceValue
    // MMOs. We can rewrite address operands but cannot delete the store
    // (since we can't track all uses via this path reliably).
    if (!LoadedReg && MI.mayLoad() && IsX87) {
      int MemIdx = X86::getFirstAddrOperandIdx(MI);
      if (MemIdx >= 0) {
        for (const MachineMemOperand *MMO : MI.memoperands()) {
          if (!MMO->isLoad())
            continue;
          const auto *FSPV = dyn_cast_or_null<FixedStackPseudoSourceValue>(
              MMO->getPseudoValue());
          if (!FSPV)
            continue;
          int FI = FSPV->getFrameIndex();
          auto It = SpillMap.find(FI);
          if (It != SpillMap.end()) {
            SpillEntry &SE = It->second;
            if (MMO->getSize() == SE.Size && MMO->getOffset() == SE.Offset) {
              MachineFrameInfo &MFI = MF.getFrameInfo();
              int64_t OrigSize = MFI.getObjectSize(SE.OrigFI);
              if (!SE.Size.hasValue() ||
                  OrigSize >= (int64_t)SE.Size.getValue()) {
                SE.UseCount++;
                if (!SE.Invalid) {
                  for (int i = 0; i < X86::AddrNumOperands; ++i)
                    MI.getOperand(MemIdx + i) = SE.OrigMOs[i];
                  MachinePointerInfo PtrInfo =
                      MachinePointerInfo::getFixedStack(MF, SE.OrigFI,
                                                        SE.Offset);
                  MachineMemOperand *NewMMO =
                      MF.getMachineMemOperand(MMO, PtrInfo, MMO->getSize());
                  MI.setMemRefs(MF, {NewMMO});
                  SE.RewriteCount++;
                  Changed = true;
                }
              }
            }
            break;
          }
        }
      }
    }

    int StoredFI = -1;
    MCRegister StoredReg(TII->isStoreToStackSlotPostFE(MI, StoredFI));

    if (StoredReg) {
      // If the stored register was loaded from a known FI, record the chain.
      auto It = RegToFI.find(StoredReg);
      if (It != RegToFI.end() && !MI.memoperands_empty()) {
        const MachineMemOperand *MMO = MI.memoperands().front();
        if (MMO->getSize() == It->second.Size && MMO->getSize().hasValue()) {
          SpillEntry SE;
          SE.OrigMOs = It->second.MOs;
          SE.LoadMI = It->second.LoadMI;
          SE.LoadReg = StoredReg;
          SE.StoreMI = &MI;
          SE.Size = MMO->getSize();
          SE.Offset = MMO->getOffset();
          SE.OrigFI = It->second.FI;
          SpillMap[StoredFI] = SE;
        } else {
          SpillMap.erase(StoredFI);
        }
      } else {
        // The store overwrites SpillFI with unknown data — invalidate.
        SpillMap.erase(StoredFI);
      }
    }

    // If the instruction writes to memory but we couldn't identify it as a
    // simple stack slot store, check MMOs for stores to tracked SpillFIs.
    // Only invalidate what we can't identify.
    if (MI.mayStore() && !StoredReg) {
      bool HasUnknownStore = false;
      for (const MachineMemOperand *MMO : MI.memoperands()) {
        if (!MMO->isStore())
          continue;
        const auto *FSPV = dyn_cast_or_null<FixedStackPseudoSourceValue>(
            MMO->getPseudoValue());
        if (FSPV) {
          // Known FI store — invalidate just that entry.
          int FI = FSPV->getFrameIndex();
          SpillMap.erase(FI);
          SmallVector<int, 4> ToErase;
          for (auto &[KeyFI, SE] : SpillMap) {
            if (SE.OrigFI == FI)
              ToErase.push_back(KeyFI);
          }
          for (int KeyFI : ToErase)
            SpillMap.erase(KeyFI);
        } else {
          HasUnknownStore = true;
        }
      }
      if (HasUnknownStore)
        SpillMap.clear();
    }

    // Any register def (explicit or implicit) invalidates all aliases.
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical())
        InvalidateReg(MO.getReg());
    }

    // Track loads from stack slots
    if (LoadedReg) {
      int MemIdx = X86::getFirstAddrOperandIdx(MI);
      if (MemIdx >= 0 && !MI.memoperands_empty()) {
        bool ClobbersAddr = false;
        for (int i = 0; i < X86::AddrNumOperands; ++i) {
          const MachineOperand &MO = MI.getOperand(MemIdx + i);
          if (MO.isReg() && MO.getReg() &&
              MI.definesRegister(MO.getReg(), TRI)) {
            ClobbersAddr = true;
            break;
          }
        }
        if (!ClobbersAddr) {
          RegEntry RE;
          for (int i = 0; i < X86::AddrNumOperands; ++i)
            RE.MOs.push_back(MI.getOperand(MemIdx + i));
          RE.LoadMI = &MI;
          RE.FI = LoadedFI;
          RE.Size = MI.memoperands().front()->getSize();
          RegToFI[LoadedReg] = RE;
        }
      }
    }
  }

  SmallPtrSet<MachineInstr *, 4> ToErase;
  for (auto &[FI, SE] : SpillMap) {
    // Only delete the store if ALL loads from SpillFI were rewritten.
    if (SE.RewriteCount > 0 && SE.RewriteCount == SE.UseCount) {
      if (SE.StoreMI)
        ToErase.insert(SE.StoreMI);
      // Only delete the load if its register is provably dead.
      if (SE.LoadMI && SE.StoreMI &&
          IsSafeToEraseLoad(SE.LoadMI, SE.StoreMI, SE.LoadReg))
        ToErase.insert(SE.LoadMI);
    }
  }

  for (MachineInstr *MI : ToErase) {
    MI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

bool X86FixupInstTuningImpl::processInstruction(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator &I) {
  MachineInstr &MI = *I;
  unsigned Opc = MI.getOpcode();
  unsigned NumOperands = MI.getDesc().getNumOperands();
  bool OptSize = MF.getFunction().hasOptSize();

  auto GetInstTput = [&](unsigned Opcode) -> std::optional<double> {
    // We already checked that SchedModel exists in `NewOpcPreferable`.
    return MCSchedModel::getReciprocalThroughput(
        *ST, *(SM->getSchedClassDesc(TII->get(Opcode).getSchedClass())));
  };

  auto GetInstLat = [&](unsigned Opcode) -> std::optional<double> {
    // We already checked that SchedModel exists in `NewOpcPreferable`.
    return MCSchedModel::computeInstrLatency(
        *ST, *(SM->getSchedClassDesc(TII->get(Opcode).getSchedClass())));
  };

  auto GetInstSize = [&](unsigned Opcode) -> std::optional<unsigned> {
    if (unsigned Size = TII->get(Opcode).getSize())
      return Size;
    // Zero size means we where unable to compute it.
    return std::nullopt;
  };

  auto NewOpcPreferable = [&](unsigned NewOpc,
                              bool ReplaceInTie = true) -> bool {
    std::optional<bool> Res;
    if (SM->hasInstrSchedModel()) {
      // Compare tput -> lat -> code size.
      Res = CmpOptionals(GetInstTput(NewOpc), GetInstTput(Opc));
      if (Res.has_value())
        return *Res;

      Res = CmpOptionals(GetInstLat(NewOpc), GetInstLat(Opc));
      if (Res.has_value())
        return *Res;
    }

    Res = CmpOptionals(GetInstSize(Opc), GetInstSize(NewOpc));
    if (Res.has_value())
      return *Res;

    // We either have either were unable to get tput/lat/codesize or all values
    // were equal. Return specified option for a tie.
    return ReplaceInTie;
  };

  // `vpermilpd r, i` -> `vshufpd r, r, i`
  // `vpermilpd r, i, k` -> `vshufpd r, r, i, k`
  // `vshufpd` is always as fast or faster than `vpermilpd` and takes
  // 1 less byte of code size for VEX and EVEX encoding.
  auto ProcessVPERMILPDri = [&](unsigned NewOpc) -> bool {
    if (!NewOpcPreferable(NewOpc))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing: " << MI);
    {
      unsigned MaskImm = MI.getOperand(NumOperands - 1).getImm();
      MI.removeOperand(NumOperands - 1);
      MI.addOperand(MI.getOperand(NumOperands - 2));
      MI.setDesc(TII->get(NewOpc));
      MI.addOperand(MachineOperand::CreateImm(MaskImm));
    }
    LLVM_DEBUG(dbgs() << "     With: " << MI);
    return true;
  };

  // `vpermilps r, i` -> `vshufps r, r, i`
  // `vpermilps r, i, k` -> `vshufps r, r, i, k`
  // `vshufps` is always as fast or faster than `vpermilps` and takes
  // 1 less byte of code size for VEX and EVEX encoding.
  auto ProcessVPERMILPSri = [&](unsigned NewOpc) -> bool {
    if (!NewOpcPreferable(NewOpc))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing: " << MI);
    {
      unsigned MaskImm = MI.getOperand(NumOperands - 1).getImm();
      MI.removeOperand(NumOperands - 1);
      MI.addOperand(MI.getOperand(NumOperands - 2));
      MI.setDesc(TII->get(NewOpc));
      MI.addOperand(MachineOperand::CreateImm(MaskImm));
    }
    LLVM_DEBUG(dbgs() << "     With: " << MI);
    return true;
  };

  // `vpermilps m, i` -> `vpshufd m, i` iff no domain delay penalty on shuffles.
  // `vpshufd` is always as fast or faster than `vpermilps` and takes 1 less
  // byte of code size.
  auto ProcessVPERMILPSmi = [&](unsigned NewOpc) -> bool {
    // TODO: Might be work adding bypass delay if -Os/-Oz is enabled as
    // `vpshufd` saves a byte of code size.
    if (!ST->hasNoDomainDelayShuffle() ||
        !NewOpcPreferable(NewOpc, /*ReplaceInTie*/ false))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing: " << MI);
    {
      MI.setDesc(TII->get(NewOpc));
    }
    LLVM_DEBUG(dbgs() << "     With: " << MI);
    return true;
  };

  // `vunpcklpd/vmovlhps r, r` -> `vunpcklqdq r, r`/`vshufpd r, r, 0x00`
  // `vunpckhpd/vmovlhps r, r` -> `vunpckhqdq r, r`/`vshufpd r, r, 0xff`
  // `vunpcklpd r, r, k` -> `vunpcklqdq r, r, k`/`vshufpd r, r, k, 0x00`
  // `vunpckhpd r, r, k` -> `vunpckhqdq r, r, k`/`vshufpd r, r, k, 0xff`
  // `vunpcklpd r, m` -> `vunpcklqdq r, m, k`
  // `vunpckhpd r, m` -> `vunpckhqdq r, m, k`
  // `vunpcklpd r, m, k` -> `vunpcklqdq r, m, k`
  // `vunpckhpd r, m, k` -> `vunpckhqdq r, m, k`
  // 1) If no bypass delay and `vunpck{l|h}qdq` faster than `vunpck{l|h}pd`
  //        -> `vunpck{l|h}qdq`
  // 2) If `vshufpd` faster than `vunpck{l|h}pd`
  //        -> `vshufpd`
  //
  // `vunpcklps` -> `vunpckldq` (for all operand types if no bypass delay)
  auto ProcessUNPCK = [&](unsigned NewOpc, unsigned MaskImm) -> bool {
    if (!NewOpcPreferable(NewOpc, /*ReplaceInTie*/ false))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing: " << MI);
    {
      MI.setDesc(TII->get(NewOpc));
      MI.addOperand(MachineOperand::CreateImm(MaskImm));
    }
    LLVM_DEBUG(dbgs() << "     With: " << MI);
    return true;
  };

  auto ProcessUNPCKToIntDomain = [&](unsigned NewOpc) -> bool {
    // TODO it may be worth it to set ReplaceInTie to `true` as there is no real
    // downside to the integer unpck, but if someone doesn't specify exact
    // target we won't find it faster.
    if (!ST->hasNoDomainDelayShuffle() ||
        !NewOpcPreferable(NewOpc, /*ReplaceInTie*/ false))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing: " << MI);
    {
      MI.setDesc(TII->get(NewOpc));
    }
    LLVM_DEBUG(dbgs() << "     With: " << MI);
    return true;
  };

  auto ProcessUNPCKLPDrr = [&](unsigned NewOpcIntDomain,
                               unsigned NewOpc) -> bool {
    if (ProcessUNPCKToIntDomain(NewOpcIntDomain))
      return true;
    return ProcessUNPCK(NewOpc, 0x00);
  };
  auto ProcessUNPCKHPDrr = [&](unsigned NewOpcIntDomain,
                               unsigned NewOpc) -> bool {
    if (ProcessUNPCKToIntDomain(NewOpcIntDomain))
      return true;
    return ProcessUNPCK(NewOpc, 0xff);
  };

  auto ProcessUNPCKPDrm = [&](unsigned NewOpcIntDomain) -> bool {
    return ProcessUNPCKToIntDomain(NewOpcIntDomain);
  };

  auto ProcessUNPCKPS = [&](unsigned NewOpc) -> bool {
    return ProcessUNPCKToIntDomain(NewOpc);
  };

  // If we're permuting the lower halves of the 256-bit registers, use a
  // subvector insertion instead.
  auto ProcessVPERM2x128ToVINSERT128 = [&](unsigned InsertOpc) -> bool {
    unsigned PermImm = MI.getOperand(NumOperands - 1).getImm();
    // TODO: Handle 0x00/0x02/0x22 when we have test coverage.
    if (PermImm != 0x20 || !NewOpcPreferable(InsertOpc))
      return false;
    Register RHSRegYMM = MI.getOperand(NumOperands - 2).getReg();
    Register RHSRegXMM = TRI->getSubReg(RHSRegYMM, X86::sub_xmm);
    LLVM_DEBUG(dbgs() << "Replacing: " << MI);
    {
      MI.setDesc(TII->get(InsertOpc));
      MI.removeOperand(NumOperands - 1);
      MI.removeOperand(NumOperands - 2);
      // Add the XMM subregister operand.
      MI.addOperand(MachineOperand::CreateReg(RHSRegXMM, /*isDef=*/false,
                                              /*isImp=*/false,
                                              /*isKill=*/false));
      // Add the immediate (1 = insert into high 128-bits).
      MI.addOperand(MachineOperand::CreateImm(1));
    }
    LLVM_DEBUG(dbgs() << "     With: " << MI);
    return true;
  };

  auto ProcessBLENDWToBLENDD = [&](unsigned MovOpc, unsigned NumElts) -> bool {
    if (!ST->hasAVX2() || !NewOpcPreferable(MovOpc))
      return false;
    // Convert to VPBLENDD if scaling the VPBLENDW mask down/up loses no bits.
    APInt MaskW =
        APInt(8, MI.getOperand(NumOperands - 1).getImm(), /*IsSigned=*/false,
              /*implicitTrunc=*/true);
    APInt MaskD = APIntOps::ScaleBitMask(MaskW, 4, /*MatchAllBits=*/true);
    if (MaskW != APIntOps::ScaleBitMask(MaskD, 8, /*MatchAllBits=*/true))
      return false;
    APInt NewMaskD = APInt::getSplat(NumElts, MaskD);
    LLVM_DEBUG(dbgs() << "Replacing: " << MI);
    {
      MI.setDesc(TII->get(MovOpc));
      MI.removeOperand(NumOperands - 1);
      MI.addOperand(MachineOperand::CreateImm(NewMaskD.getZExtValue()));
    }
    LLVM_DEBUG(dbgs() << "     With: " << MI);
    return true;
  };

  auto ProcessBLENDToMOV = [&](unsigned MovOpc, unsigned Mask,
                               unsigned MovImm) -> bool {
    if ((MI.getOperand(NumOperands - 1).getImm() & Mask) != MovImm)
      return false;
    if (!OptSize && !NewOpcPreferable(MovOpc))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing: " << MI);
    {
      MI.setDesc(TII->get(MovOpc));
      MI.removeOperand(NumOperands - 1);
    }
    LLVM_DEBUG(dbgs() << "     With: " << MI);
    return true;
  };

  // Is ADD(X,X) more efficient than SHL(X,1)?
  auto ProcessShiftLeftToAdd = [&](unsigned AddOpc) -> bool {
    if (MI.getOperand(NumOperands - 1).getImm() != 1)
      return false;
    if (!NewOpcPreferable(AddOpc, /*ReplaceInTie*/ true))
      return false;
    LLVM_DEBUG(dbgs() << "Replacing: " << MI);
    {
      MI.setDesc(TII->get(AddOpc));
      MI.removeOperand(NumOperands - 1);
      MI.addOperand(MI.getOperand(NumOperands - 2));
    }
    LLVM_DEBUG(dbgs() << "     With: " << MI);
    return true;
  };

  // `vpermq ymm, ymm, 0x44` -> `vinserti128 ymm, ymm, xmm, 1`
  // `vpermpd ymm, ymm, 0x44` -> `vinsertf128 ymm, ymm, xmm, 1`
  // When the immediate is 0x44, VPERMQ/VPERMPD duplicates the lower 128-bit
  // lane to both lanes. 0x44 = 0b01_00_01_00 means qwords[3:0] = {src[1],
  // src[0], src[1], src[0]} This is equivalent to inserting the lower 128-bits
  // into the upper 128-bit position.
  auto ProcessVPERMQToVINSERT128 = [&](unsigned NewOpc) -> bool {
    if (MI.getOperand(NumOperands - 1).getImm() != 0x44)
      return false;
    if (!NewOpcPreferable(NewOpc, /*ReplaceInTie*/ false))
      return false;

    // Get the XMM subregister of the source YMM register.
    Register SrcReg = MI.getOperand(1).getReg();
    Register XmmReg = TRI->getSubReg(SrcReg, X86::sub_xmm);

    LLVM_DEBUG(dbgs() << "Replacing: " << MI);
    {
      // Transform: VPERMQ $dst, $src, $0x44
      // Into:      VINSERTI128 $dst, $src, $xmm_src, $1
      MI.setDesc(TII->get(NewOpc));
      // Remove the immediate operand.
      MI.removeOperand(NumOperands - 1);
      // Add the XMM subregister operand.
      MI.addOperand(MachineOperand::CreateReg(XmmReg, /*isDef=*/false,
                                              /*isImp=*/false,
                                              /*isKill=*/false));
      // Add the immediate (1 = insert into high 128-bits).
      MI.addOperand(MachineOperand::CreateImm(1));
    }
    LLVM_DEBUG(dbgs() << "     With: " << MI);
    return true;
  };

  switch (Opc) {
  case X86::BLENDPDrri:
    return ProcessBLENDToMOV(X86::MOVSDrr, 0x3, 0x1);
  case X86::VBLENDPDrri:
    return ProcessBLENDToMOV(X86::VMOVSDrr, 0x3, 0x1);

  case X86::BLENDPSrri:
    return ProcessBLENDToMOV(X86::MOVSSrr, 0xF, 0x1) ||
           ProcessBLENDToMOV(X86::MOVSDrr, 0xF, 0x3);
  case X86::VBLENDPSrri:
    return ProcessBLENDToMOV(X86::VMOVSSrr, 0xF, 0x1) ||
           ProcessBLENDToMOV(X86::VMOVSDrr, 0xF, 0x3);

  case X86::VPBLENDWrri:
    // TODO: Add X86::VPBLENDWrmi handling
    // TODO: Add X86::VPBLENDWYrri handling
    // TODO: Add X86::VPBLENDWYrmi handling
    return ProcessBLENDWToBLENDD(X86::VPBLENDDrri, 4);

  case X86::VPERM2F128rri:
    return ProcessVPERM2x128ToVINSERT128(X86::VINSERTF128rri);
  case X86::VPERM2I128rri:
    return ProcessVPERM2x128ToVINSERT128(X86::VINSERTI128rri);

  case X86::VPERMILPDri:
    return ProcessVPERMILPDri(X86::VSHUFPDrri);
  case X86::VPERMILPDYri:
    return ProcessVPERMILPDri(X86::VSHUFPDYrri);
  case X86::VPERMILPDZ128ri:
    return ProcessVPERMILPDri(X86::VSHUFPDZ128rri);
  case X86::VPERMILPDZ256ri:
    return ProcessVPERMILPDri(X86::VSHUFPDZ256rri);
  case X86::VPERMILPDZri:
    return ProcessVPERMILPDri(X86::VSHUFPDZrri);
  case X86::VPERMILPDZ128rikz:
    return ProcessVPERMILPDri(X86::VSHUFPDZ128rrikz);
  case X86::VPERMILPDZ256rikz:
    return ProcessVPERMILPDri(X86::VSHUFPDZ256rrikz);
  case X86::VPERMILPDZrikz:
    return ProcessVPERMILPDri(X86::VSHUFPDZrrikz);
  case X86::VPERMILPDZ128rik:
    return ProcessVPERMILPDri(X86::VSHUFPDZ128rrik);
  case X86::VPERMILPDZ256rik:
    return ProcessVPERMILPDri(X86::VSHUFPDZ256rrik);
  case X86::VPERMILPDZrik:
    return ProcessVPERMILPDri(X86::VSHUFPDZrrik);

  case X86::VPERMILPSri:
    return ProcessVPERMILPSri(X86::VSHUFPSrri);
  case X86::VPERMILPSYri:
    return ProcessVPERMILPSri(X86::VSHUFPSYrri);
  case X86::VPERMILPSZ128ri:
    return ProcessVPERMILPSri(X86::VSHUFPSZ128rri);
  case X86::VPERMILPSZ256ri:
    return ProcessVPERMILPSri(X86::VSHUFPSZ256rri);
  case X86::VPERMILPSZri:
    return ProcessVPERMILPSri(X86::VSHUFPSZrri);
  case X86::VPERMILPSZ128rikz:
    return ProcessVPERMILPSri(X86::VSHUFPSZ128rrikz);
  case X86::VPERMILPSZ256rikz:
    return ProcessVPERMILPSri(X86::VSHUFPSZ256rrikz);
  case X86::VPERMILPSZrikz:
    return ProcessVPERMILPSri(X86::VSHUFPSZrrikz);
  case X86::VPERMILPSZ128rik:
    return ProcessVPERMILPSri(X86::VSHUFPSZ128rrik);
  case X86::VPERMILPSZ256rik:
    return ProcessVPERMILPSri(X86::VSHUFPSZ256rrik);
  case X86::VPERMILPSZrik:
    return ProcessVPERMILPSri(X86::VSHUFPSZrrik);
  case X86::VPERMILPSmi:
    return ProcessVPERMILPSmi(X86::VPSHUFDmi);
  case X86::VPERMILPSYmi:
    // TODO: See if there is a more generic way we can test if the replacement
    // instruction is supported.
    return ST->hasAVX2() ? ProcessVPERMILPSmi(X86::VPSHUFDYmi) : false;
  case X86::VPERMILPSZ128mi:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ128mi);
  case X86::VPERMILPSZ256mi:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ256mi);
  case X86::VPERMILPSZmi:
    return ProcessVPERMILPSmi(X86::VPSHUFDZmi);
  case X86::VPERMILPSZ128mikz:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ128mikz);
  case X86::VPERMILPSZ256mikz:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ256mikz);
  case X86::VPERMILPSZmikz:
    return ProcessVPERMILPSmi(X86::VPSHUFDZmikz);
  case X86::VPERMILPSZ128mik:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ128mik);
  case X86::VPERMILPSZ256mik:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ256mik);
  case X86::VPERMILPSZmik:
    return ProcessVPERMILPSmi(X86::VPSHUFDZmik);
  case X86::VPERMQYri:
    return ProcessVPERMQToVINSERT128(X86::VINSERTI128rri);
  case X86::VPERMPDYri:
    return ProcessVPERMQToVINSERT128(X86::VINSERTF128rri);
  case X86::MOVLHPSrr:
  case X86::UNPCKLPDrr:
    return ProcessUNPCKLPDrr(X86::PUNPCKLQDQrr, X86::SHUFPDrri);
  case X86::VMOVLHPSrr:
  case X86::VUNPCKLPDrr:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQrr, X86::VSHUFPDrri);
  case X86::VUNPCKLPDYrr:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQYrr, X86::VSHUFPDYrri);
    // VMOVLHPS is always 128 bits.
  case X86::VMOVLHPSZrr:
  case X86::VUNPCKLPDZ128rr:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQZ128rr, X86::VSHUFPDZ128rri);
  case X86::VUNPCKLPDZ256rr:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQZ256rr, X86::VSHUFPDZ256rri);
  case X86::VUNPCKLPDZrr:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQZrr, X86::VSHUFPDZrri);
  case X86::VUNPCKLPDZ128rrk:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQZ128rrk, X86::VSHUFPDZ128rrik);
  case X86::VUNPCKLPDZ256rrk:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQZ256rrk, X86::VSHUFPDZ256rrik);
  case X86::VUNPCKLPDZrrk:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQZrrk, X86::VSHUFPDZrrik);
  case X86::VUNPCKLPDZ128rrkz:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQZ128rrkz, X86::VSHUFPDZ128rrikz);
  case X86::VUNPCKLPDZ256rrkz:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQZ256rrkz, X86::VSHUFPDZ256rrikz);
  case X86::VUNPCKLPDZrrkz:
    return ProcessUNPCKLPDrr(X86::VPUNPCKLQDQZrrkz, X86::VSHUFPDZrrikz);
  case X86::UNPCKHPDrr:
    return ProcessUNPCKHPDrr(X86::PUNPCKHQDQrr, X86::SHUFPDrri);
  case X86::VUNPCKHPDrr:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQrr, X86::VSHUFPDrri);
  case X86::VUNPCKHPDYrr:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQYrr, X86::VSHUFPDYrri);
  case X86::VUNPCKHPDZ128rr:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQZ128rr, X86::VSHUFPDZ128rri);
  case X86::VUNPCKHPDZ256rr:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQZ256rr, X86::VSHUFPDZ256rri);
  case X86::VUNPCKHPDZrr:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQZrr, X86::VSHUFPDZrri);
  case X86::VUNPCKHPDZ128rrk:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQZ128rrk, X86::VSHUFPDZ128rrik);
  case X86::VUNPCKHPDZ256rrk:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQZ256rrk, X86::VSHUFPDZ256rrik);
  case X86::VUNPCKHPDZrrk:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQZrrk, X86::VSHUFPDZrrik);
  case X86::VUNPCKHPDZ128rrkz:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQZ128rrkz, X86::VSHUFPDZ128rrikz);
  case X86::VUNPCKHPDZ256rrkz:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQZ256rrkz, X86::VSHUFPDZ256rrikz);
  case X86::VUNPCKHPDZrrkz:
    return ProcessUNPCKHPDrr(X86::VPUNPCKHQDQZrrkz, X86::VSHUFPDZrrikz);
  case X86::UNPCKLPDrm:
    return ProcessUNPCKPDrm(X86::PUNPCKLQDQrm);
  case X86::VUNPCKLPDrm:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQrm);
  case X86::VUNPCKLPDYrm:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQYrm);
  case X86::VUNPCKLPDZ128rm:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQZ128rm);
  case X86::VUNPCKLPDZ256rm:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQZ256rm);
  case X86::VUNPCKLPDZrm:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQZrm);
  case X86::VUNPCKLPDZ128rmk:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQZ128rmk);
  case X86::VUNPCKLPDZ256rmk:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQZ256rmk);
  case X86::VUNPCKLPDZrmk:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQZrmk);
  case X86::VUNPCKLPDZ128rmkz:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQZ128rmkz);
  case X86::VUNPCKLPDZ256rmkz:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQZ256rmkz);
  case X86::VUNPCKLPDZrmkz:
    return ProcessUNPCKPDrm(X86::VPUNPCKLQDQZrmkz);
  case X86::UNPCKHPDrm:
    return ProcessUNPCKPDrm(X86::PUNPCKHQDQrm);
  case X86::VUNPCKHPDrm:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQrm);
  case X86::VUNPCKHPDYrm:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQYrm);
  case X86::VUNPCKHPDZ128rm:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQZ128rm);
  case X86::VUNPCKHPDZ256rm:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQZ256rm);
  case X86::VUNPCKHPDZrm:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQZrm);
  case X86::VUNPCKHPDZ128rmk:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQZ128rmk);
  case X86::VUNPCKHPDZ256rmk:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQZ256rmk);
  case X86::VUNPCKHPDZrmk:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQZrmk);
  case X86::VUNPCKHPDZ128rmkz:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQZ128rmkz);
  case X86::VUNPCKHPDZ256rmkz:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQZ256rmkz);
  case X86::VUNPCKHPDZrmkz:
    return ProcessUNPCKPDrm(X86::VPUNPCKHQDQZrmkz);

  case X86::UNPCKLPSrr:
    return ProcessUNPCKPS(X86::PUNPCKLDQrr);
  case X86::VUNPCKLPSrr:
    return ProcessUNPCKPS(X86::VPUNPCKLDQrr);
  case X86::VUNPCKLPSYrr:
    return ProcessUNPCKPS(X86::VPUNPCKLDQYrr);
  case X86::VUNPCKLPSZ128rr:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ128rr);
  case X86::VUNPCKLPSZ256rr:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ256rr);
  case X86::VUNPCKLPSZrr:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZrr);
  case X86::VUNPCKLPSZ128rrk:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ128rrk);
  case X86::VUNPCKLPSZ256rrk:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ256rrk);
  case X86::VUNPCKLPSZrrk:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZrrk);
  case X86::VUNPCKLPSZ128rrkz:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ128rrkz);
  case X86::VUNPCKLPSZ256rrkz:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ256rrkz);
  case X86::VUNPCKLPSZrrkz:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZrrkz);
  case X86::UNPCKHPSrr:
    return ProcessUNPCKPS(X86::PUNPCKHDQrr);
  case X86::VUNPCKHPSrr:
    return ProcessUNPCKPS(X86::VPUNPCKHDQrr);
  case X86::VUNPCKHPSYrr:
    return ProcessUNPCKPS(X86::VPUNPCKHDQYrr);
  case X86::VUNPCKHPSZ128rr:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ128rr);
  case X86::VUNPCKHPSZ256rr:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ256rr);
  case X86::VUNPCKHPSZrr:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZrr);
  case X86::VUNPCKHPSZ128rrk:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ128rrk);
  case X86::VUNPCKHPSZ256rrk:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ256rrk);
  case X86::VUNPCKHPSZrrk:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZrrk);
  case X86::VUNPCKHPSZ128rrkz:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ128rrkz);
  case X86::VUNPCKHPSZ256rrkz:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ256rrkz);
  case X86::VUNPCKHPSZrrkz:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZrrkz);
  case X86::UNPCKLPSrm:
    return ProcessUNPCKPS(X86::PUNPCKLDQrm);
  case X86::VUNPCKLPSrm:
    return ProcessUNPCKPS(X86::VPUNPCKLDQrm);
  case X86::VUNPCKLPSYrm:
    return ProcessUNPCKPS(X86::VPUNPCKLDQYrm);
  case X86::VUNPCKLPSZ128rm:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ128rm);
  case X86::VUNPCKLPSZ256rm:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ256rm);
  case X86::VUNPCKLPSZrm:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZrm);
  case X86::VUNPCKLPSZ128rmk:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ128rmk);
  case X86::VUNPCKLPSZ256rmk:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ256rmk);
  case X86::VUNPCKLPSZrmk:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZrmk);
  case X86::VUNPCKLPSZ128rmkz:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ128rmkz);
  case X86::VUNPCKLPSZ256rmkz:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZ256rmkz);
  case X86::VUNPCKLPSZrmkz:
    return ProcessUNPCKPS(X86::VPUNPCKLDQZrmkz);
  case X86::UNPCKHPSrm:
    return ProcessUNPCKPS(X86::PUNPCKHDQrm);
  case X86::VUNPCKHPSrm:
    return ProcessUNPCKPS(X86::VPUNPCKHDQrm);
  case X86::VUNPCKHPSYrm:
    return ProcessUNPCKPS(X86::VPUNPCKHDQYrm);
  case X86::VUNPCKHPSZ128rm:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ128rm);
  case X86::VUNPCKHPSZ256rm:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ256rm);
  case X86::VUNPCKHPSZrm:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZrm);
  case X86::VUNPCKHPSZ128rmk:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ128rmk);
  case X86::VUNPCKHPSZ256rmk:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ256rmk);
  case X86::VUNPCKHPSZrmk:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZrmk);
  case X86::VUNPCKHPSZ128rmkz:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ128rmkz);
  case X86::VUNPCKHPSZ256rmkz:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZ256rmkz);
  case X86::VUNPCKHPSZrmkz:
    return ProcessUNPCKPS(X86::VPUNPCKHDQZrmkz);

  case X86::PSLLWri:
    return ProcessShiftLeftToAdd(X86::PADDWrr);
  case X86::VPSLLWri:
    return ProcessShiftLeftToAdd(X86::VPADDWrr);
  case X86::VPSLLWYri:
    return ProcessShiftLeftToAdd(X86::VPADDWYrr);
  case X86::VPSLLWZ128ri:
    return ProcessShiftLeftToAdd(X86::VPADDWZ128rr);
  case X86::VPSLLWZ256ri:
    return ProcessShiftLeftToAdd(X86::VPADDWZ256rr);
  case X86::VPSLLWZri:
    return ProcessShiftLeftToAdd(X86::VPADDWZrr);
  case X86::PSLLDri:
    return ProcessShiftLeftToAdd(X86::PADDDrr);
  case X86::VPSLLDri:
    return ProcessShiftLeftToAdd(X86::VPADDDrr);
  case X86::VPSLLDYri:
    return ProcessShiftLeftToAdd(X86::VPADDDYrr);
  case X86::VPSLLDZ128ri:
    return ProcessShiftLeftToAdd(X86::VPADDDZ128rr);
  case X86::VPSLLDZ256ri:
    return ProcessShiftLeftToAdd(X86::VPADDDZ256rr);
  case X86::VPSLLDZri:
    return ProcessShiftLeftToAdd(X86::VPADDDZrr);
  case X86::PSLLQri:
    return ProcessShiftLeftToAdd(X86::PADDQrr);
  case X86::VPSLLQri:
    return ProcessShiftLeftToAdd(X86::VPADDQrr);
  case X86::VPSLLQYri:
    return ProcessShiftLeftToAdd(X86::VPADDQYrr);
  case X86::VPSLLQZ128ri:
    return ProcessShiftLeftToAdd(X86::VPADDQZ128rr);
  case X86::VPSLLQZ256ri:
    return ProcessShiftLeftToAdd(X86::VPADDQZ256rr);
  case X86::VPSLLQZri:
    return ProcessShiftLeftToAdd(X86::VPADDQZrr);

  default:
    return false;
  }
}

bool X86FixupInstTuningImpl::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "Start X86FixupInstTuning\n";);
  bool Changed = false;
  ST = &MF.getSubtarget<X86Subtarget>();
  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  SM = &ST->getSchedModel();

  for (MachineBasicBlock &MBB : MF) {
    Changed |= processSpills(MBB, MF);
    for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
      if (processInstruction(MF, MBB, I)) {
        ++NumInstChanges;
        Changed = true;
      }
    }
  }
  LLVM_DEBUG(dbgs() << "End X86FixupInstTuning\n";);
  return Changed;
}

bool X86FixupInstTuningLegacy::runOnMachineFunction(MachineFunction &MF) {
  X86FixupInstTuningImpl Impl;
  return Impl.runOnMachineFunction(MF);
}

PreservedAnalyses
X86FixupInstTuningPass::run(MachineFunction &MF,
                            MachineFunctionAnalysisManager &MFAM) {
  X86FixupInstTuningImpl Impl;
  return Impl.runOnMachineFunction(MF)
             ? getMachineFunctionPassPreservedAnalyses()
                   .preserveSet<CFGAnalyses>()
             : PreservedAnalyses::all();
}
