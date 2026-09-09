//===- X86CompressEVEX.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass compresses instructions from EVEX space to legacy/VEX/EVEX space
// when possible in order to reduce code size or facilitate HW decoding.
//
// Possible compression:
//   a. AVX512 instruction (EVEX) -> AVX instruction (VEX)
//   b. Promoted instruction (EVEX) -> pre-promotion instruction (legacy/VEX)
//   c. NDD (EVEX) -> non-NDD (legacy)
//   d. NF_ND (EVEX) -> NF (EVEX)
//   e. NonNF (EVEX) -> NF (EVEX)
//   f. SETZUCCm (EVEX) -> SETCCm (legacy)
//   g. VPMOV*2M (EVEX) + KMOV -> VMOVMSK/VPMOVMSKB (VEX)
//   h. VPMOV*2M (EVEX) + masked VMOV* -> VBLENDV* (VEX)
//
// Compression a, b and c can always reduce code size, with some exceptions
// such as promoted 16-bit CRC32 which is as long as the legacy version.
//
// legacy:
//   crc32w %si, %eax ## encoding: [0x66,0xf2,0x0f,0x38,0xf1,0xc6]
// promoted:
//   crc32w %si, %eax ## encoding: [0x62,0xf4,0x7d,0x08,0xf1,0xc6]
//
// From performance perspective, these should be same (same uops and same EXE
// ports). From a FMV perspective, an older legacy encoding is preferred b/c it
// can execute in more places (broader HW install base). So we will still do
// the compression.
//
// Compression d can help hardware decode (HW may skip reading the NDD
// register) although the instruction length remains unchanged.
//
// Compression e can help hardware skip updating EFLAGS although the instruction
// length remains unchanged.
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include <atomic>
#include <cassert>
#include <cstdint>

using namespace llvm;

#define COMP_EVEX_DESC "Compressing EVEX instrs when possible"
#define COMP_EVEX_NAME "x86-compress-evex"

#define DEBUG_TYPE COMP_EVEX_NAME

extern cl::opt<bool> X86EnableAPXForRelocation;

namespace {
// Including the generated EVEX compression tables.
#define GET_X86_COMPRESS_EVEX_TABLE
#include "X86GenInstrMapping.inc"

class CompressEVEXLegacy : public MachineFunctionPass {
public:
  static char ID;
  CompressEVEXLegacy() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return COMP_EVEX_DESC; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
};

} // end anonymous namespace

char CompressEVEXLegacy::ID = 0;

static bool usesExtendedRegister(const MachineInstr &MI) {
  auto isHiRegIdx = [](MCRegister Reg) {
    // Check for XMM register with indexes between 16 - 31.
    if (Reg >= X86::XMM16 && Reg <= X86::XMM31)
      return true;
    // Check for YMM register with indexes between 16 - 31.
    if (Reg >= X86::YMM16 && Reg <= X86::YMM31)
      return true;
    // Check for GPR with indexes between 16 - 31.
    if (X86II::isApxExtendedReg(Reg))
      return true;
    return false;
  };

  // Check that operands are not ZMM regs or
  // XMM/YMM regs with hi indexes between 16 - 31.
  for (const MachineOperand &MO : MI.explicit_operands()) {
    if (!MO.isReg())
      continue;

    MCRegister Reg = MO.getReg().asMCReg();
    assert(!X86II::isZMMReg(Reg) &&
           "ZMM instructions should not be in the EVEX->VEX tables");
    if (isHiRegIdx(Reg))
      return true;
  }

  return false;
}

// Return true if the EVEX form of \p MI can encode its memory displacement as
// a compressed disp8*N (1 byte) while the VEX/legacy twin would be forced to
// spend a full disp32 (4 bytes). In that window the EVEX encoding is strictly
// shorter overall, despite its 1-2 byte larger prefix, so compressing it to
// VEX would grow code size. Mirrors isDispOrCDisp8 in X86MCCodeEmitter.cpp.
static bool hasShorterEVEXViaCDisp8(const MachineInstr &MI) {
  uint64_t TSFlags = MI.getDesc().TSFlags;
  unsigned CD8_Scale =
      (TSFlags & X86II::CD8_Scale_Mask) >> X86II::CD8_Scale_Shift;
  CD8_Scale = CD8_Scale ? 1U << (CD8_Scale - 1) : 0U;
  // Without a CD8 scale > 1 there is no displacement advantage over VEX.
  if (CD8_Scale <= 1)
    return false;

  int MemOpIdx = X86::getFirstAddrOperandIdx(MI);
  if (MemOpIdx < 0)
    return false;

  const MachineOperand &Disp = MI.getOperand(MemOpIdx + X86::AddrDisp);
  // Only a constant displacement can be range-checked here; symbolic ones
  // (globals, constant pool, jump tables, ...) are resolved later.
  if (!Disp.isImm())
    return false;

  int64_t Val = Disp.getImm();
  // VEX can already use a disp8 in this range, so EVEX offers no saving.
  if (isInt<8>(Val))
    return false;
  // EVEX can use disp8*N only when the value is a multiple of N and the scaled
  // value fits in a signed byte.
  if (Val % static_cast<int64_t>(CD8_Scale) != 0)
    return false;
  return isInt<8>(Val / static_cast<int64_t>(CD8_Scale));
}

// Do any custom cleanup needed to finalize the conversion.
static bool performCustomAdjustments(MachineInstr &MI, unsigned NewOpc) {
  (void)NewOpc;
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case X86::VALIGNDZ128rri:
  case X86::VALIGNDZ128rmi:
  case X86::VALIGNQZ128rri:
  case X86::VALIGNQZ128rmi: {
    assert((NewOpc == X86::VPALIGNRrri || NewOpc == X86::VPALIGNRrmi) &&
           "Unexpected new opcode!");
    unsigned Scale =
        (Opc == X86::VALIGNQZ128rri || Opc == X86::VALIGNQZ128rmi) ? 8 : 4;
    MachineOperand &Imm = MI.getOperand(MI.getNumExplicitOperands() - 1);
    Imm.setImm(Imm.getImm() * Scale);
    break;
  }
  case X86::VSHUFF32X4Z256rmi:
  case X86::VSHUFF32X4Z256rri:
  case X86::VSHUFF64X2Z256rmi:
  case X86::VSHUFF64X2Z256rri:
  case X86::VSHUFI32X4Z256rmi:
  case X86::VSHUFI32X4Z256rri:
  case X86::VSHUFI64X2Z256rmi:
  case X86::VSHUFI64X2Z256rri: {
    assert((NewOpc == X86::VPERM2F128rri || NewOpc == X86::VPERM2I128rri ||
            NewOpc == X86::VPERM2F128rmi || NewOpc == X86::VPERM2I128rmi) &&
           "Unexpected new opcode!");
    MachineOperand &Imm = MI.getOperand(MI.getNumExplicitOperands() - 1);
    int64_t ImmVal = Imm.getImm();
    // Set bit 5, move bit 1 to bit 4, copy bit 0.
    Imm.setImm(0x20 | ((ImmVal & 2) << 3) | (ImmVal & 1));
    break;
  }
  case X86::VRNDSCALEPDZ128rri:
  case X86::VRNDSCALEPDZ128rmi:
  case X86::VRNDSCALEPSZ128rri:
  case X86::VRNDSCALEPSZ128rmi:
  case X86::VRNDSCALEPDZ256rri:
  case X86::VRNDSCALEPDZ256rmi:
  case X86::VRNDSCALEPSZ256rri:
  case X86::VRNDSCALEPSZ256rmi:
  case X86::VRNDSCALESDZrri:
  case X86::VRNDSCALESDZrmi:
  case X86::VRNDSCALESSZrri:
  case X86::VRNDSCALESSZrmi:
  case X86::VRNDSCALESDZrri_Int:
  case X86::VRNDSCALESDZrmi_Int:
  case X86::VRNDSCALESSZrri_Int:
  case X86::VRNDSCALESSZrmi_Int:
    const MachineOperand &Imm = MI.getOperand(MI.getNumExplicitOperands() - 1);
    int64_t ImmVal = Imm.getImm();
    // Ensure that only bits 3:0 of the immediate are used.
    if ((ImmVal & 0xf) != ImmVal)
      return false;
    break;
  }

  return true;
}

static unsigned getMovMskBits(unsigned Opc) {
  switch (Opc) {
  case X86::VPMOVQ2MZ128kr:
  case X86::VPCMPQZ128rri:
    return 2;
  case X86::VPMOVQ2MZ256kr:
  case X86::VPMOVD2MZ128kr:
  case X86::VPCMPQZ256rri:
  case X86::VPCMPDZ128rri:
    return 4;
  case X86::VPMOVD2MZ256kr:
  case X86::VPCMPDZ256rri:
    return 8;
  case X86::VPMOVB2MZ128kr:
  case X86::VPCMPBZ128rri:
    return 16;
  case X86::VPMOVB2MZ256kr:
  case X86::VPCMPBZ256rri:
    return 32;
  default:
    llvm_unreachable("Unknown opcode");
  }
}

static bool isKMovNarrowing(unsigned MaskBits, unsigned KMOVOpc) {
  unsigned KMOVSize = 0;
  switch (KMOVOpc) {
  case X86::KMOVBrk:
    KMOVSize = 8;
    break;
  case X86::KMOVWrk:
    KMOVSize = 16;
    break;
  case X86::KMOVDrk:
    KMOVSize = 32;
    break;
  default:
    llvm_unreachable("Unknown KMOV opcode");
  }

  return KMOVSize < MaskBits;
}

static bool isZeroVector(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case X86::VPXORrr:
  case X86::VPXORYrr:
  case X86::VXORPSrr:
  case X86::VXORPSYrr:
    return MI.getOperand(1).getReg() == MI.getOperand(2).getReg();
  default:
    return false;
  }
}

static bool isAllOnesVector(const MachineInstr &MI, bool Is256Bit) {
  switch (MI.getOpcode()) {
  case X86::VPCMPEQDrr:
    return !Is256Bit && MI.getOperand(1).getReg() == MI.getOperand(2).getReg();
  case X86::VPCMPEQDYrr:
    return MI.getOperand(1).getReg() == MI.getOperand(2).getReg();
  default:
    return false;
  }
}

static MachineInstr *getSignMaskConstantDef(MachineInstr &MI, Register Reg,
                                            bool IsZero, bool Is256Bit,
                                            const TargetRegisterInfo *TRI) {
  for (MachineInstr &DefMI : llvm::reverse(llvm::make_range(
           MI.getParent()->begin(), MachineBasicBlock::iterator(MI)))) {
    if (!DefMI.modifiesRegister(Reg, TRI))
      continue;
    // Stop at the nearest def/clobber; an older matching constant may no
    // longer be the reaching definition.
    if (IsZero ? isZeroVector(DefMI) : isAllOnesVector(DefMI, Is256Bit))
      return &DefMI;
    break;
  }
  return nullptr;
}

static bool isCompressibleBlendVUse(unsigned BlendOpc, unsigned UseOpc) {
  switch (BlendOpc) {
  case X86::VBLENDVPSrrr:
    switch (UseOpc) {
    case X86::VMOVAPSZ128rrk:
    case X86::VMOVUPSZ128rrk:
    case X86::VMOVDQA32Z128rrk:
    case X86::VMOVDQU32Z128rrk:
      return true;
    default:
      return false;
    }
  case X86::VBLENDVPSYrrr:
    switch (UseOpc) {
    case X86::VMOVAPSZ256rrk:
    case X86::VMOVUPSZ256rrk:
    case X86::VMOVDQA32Z256rrk:
    case X86::VMOVDQU32Z256rrk:
      return true;
    default:
      return false;
    }
  case X86::VBLENDVPDrrr:
    switch (UseOpc) {
    case X86::VMOVAPDZ128rrk:
    case X86::VMOVUPDZ128rrk:
    case X86::VMOVDQA64Z128rrk:
    case X86::VMOVDQU64Z128rrk:
      return true;
    default:
      return false;
    }
  case X86::VBLENDVPDYrrr:
    switch (UseOpc) {
    case X86::VMOVAPDZ256rrk:
    case X86::VMOVUPDZ256rrk:
    case X86::VMOVDQA64Z256rrk:
    case X86::VMOVDQU64Z256rrk:
      return true;
    default:
      return false;
    }
  case X86::VPBLENDVBrrr:
    return UseOpc == X86::VMOVDQU8Z128rrk;
  case X86::VPBLENDVBYrrr:
    return UseOpc == X86::VMOVDQU8Z256rrk;
  default:
    return false;
  }
}

// Try to compress mask producer chains:
//   vpmov*2m %xmm0, %k0       ->  (erase this)
//   kmov* %k0, %eax           ->  vmovmskp* %xmm0, %eax
//
//   vpcmpge* $0, %xmm0, %k0   ->  (erase this)  (X >= 0)
//   vpcmpgt* $-1, %xmm0, %k0  ->  (erase this)  (X > -1)
//   kmov* %k0, %eax           ->  vmovmskp* %xmm0, %eax
//                                bounded complement of %eax
//
//   vpmov*2m %xmm0, %k1       ->  (erase this)
//   vmov* %xmm1, %xmm2 {%k1}  ->  vblendv* %xmm0, %xmm2, %xmm1, %xmm2
static bool tryCompressMaskProducer(MachineInstr &MI, MachineBasicBlock &MBB,
                                    const X86Subtarget &ST,
                                    SmallVectorImpl<MachineInstr *> &ToErase) {
  const X86InstrInfo *TII = ST.getInstrInfo();
  const TargetRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo *MRI = &MBB.getParent()->getRegInfo();

  unsigned Opc = MI.getOpcode();
  bool IsSignMaskCmp = Opc == X86::VPCMPBZ128rri || Opc == X86::VPCMPBZ256rri ||
                       Opc == X86::VPCMPDZ128rri || Opc == X86::VPCMPDZ256rri ||
                       Opc == X86::VPCMPQZ128rri || Opc == X86::VPCMPQZ256rri;
  if (!IsSignMaskCmp && Opc != X86::VPMOVD2MZ128kr &&
      Opc != X86::VPMOVD2MZ256kr && Opc != X86::VPMOVQ2MZ128kr &&
      Opc != X86::VPMOVQ2MZ256kr && Opc != X86::VPMOVB2MZ128kr &&
      Opc != X86::VPMOVB2MZ256kr)
    return false;

  if (usesExtendedRegister(MI))
    return false;

  Register MaskReg = MI.getOperand(0).getReg();
  Register SrcVecReg = MI.getOperand(1).getReg();
  MachineInstr *ConstantDef = nullptr;
  bool ConstantDefOnlyFeedsCmp = false;

  if (IsSignMaskCmp) {
    int64_t Pred = MI.getOperand(3).getImm();
    // VPCMP signed predicates: nlt (5) folds X >= 0, nle (6) folds X > -1.
    if (Pred != 5 && Pred != 6)
      return false;
    Register ConstantReg = MI.getOperand(2).getReg();
    bool Is256Bit = Opc == X86::VPCMPBZ256rri || Opc == X86::VPCMPDZ256rri ||
                    Opc == X86::VPCMPQZ256rri;
    // The sign-mask fold is valid only for compares against the reaching
    // zero/all-ones vector definition.
    ConstantDef =
        getSignMaskConstantDef(MI, ConstantReg, Pred == 5, Is256Bit, TRI);
    if (!ConstantDef)
      return false;
    // If the constant feeds only this compare, erase it with the compare.
    ConstantDefOnlyFeedsCmp = !TRI->regsOverlap(ConstantReg, SrcVecReg);
    for (MachineInstr &UseMI :
         llvm::make_range(std::next(MachineBasicBlock::iterator(*ConstantDef)),
                          MachineBasicBlock::iterator(MI)))
      if (UseMI.readsRegister(ConstantReg, TRI)) {
        ConstantDefOnlyFeedsCmp = false;
        break;
      }
  }

  unsigned MovMskOpc = 0;
  unsigned BlendOpc = 0;
  switch (Opc) {
  case X86::VPCMPDZ128rri:
  case X86::VPMOVD2MZ128kr:
    MovMskOpc = X86::VMOVMSKPSrr;
    BlendOpc = X86::VBLENDVPSrrr;
    break;
  case X86::VPCMPDZ256rri:
  case X86::VPMOVD2MZ256kr:
    MovMskOpc = X86::VMOVMSKPSYrr;
    BlendOpc = X86::VBLENDVPSYrrr;
    break;
  case X86::VPCMPQZ128rri:
  case X86::VPMOVQ2MZ128kr:
    MovMskOpc = X86::VMOVMSKPDrr;
    BlendOpc = X86::VBLENDVPDrrr;
    break;
  case X86::VPCMPQZ256rri:
  case X86::VPMOVQ2MZ256kr:
    MovMskOpc = X86::VMOVMSKPDYrr;
    BlendOpc = X86::VBLENDVPDYrrr;
    break;
  case X86::VPCMPBZ128rri:
  case X86::VPMOVB2MZ128kr:
    MovMskOpc = X86::VPMOVMSKBrr;
    BlendOpc = X86::VPBLENDVBrrr;
    break;
  case X86::VPCMPBZ256rri:
  case X86::VPMOVB2MZ256kr:
    MovMskOpc = X86::VPMOVMSKBYrr;
    BlendOpc = X86::VPBLENDVBYrrr;
    break;
  default:
    llvm_unreachable("Unknown VPMOV opcode");
  }

  MachineInstr *KMovMI = nullptr;
  MachineInstr *BlendMI = nullptr;

  for (MachineInstr &CurMI : llvm::make_range(
           std::next(MachineBasicBlock::iterator(MI)), MBB.end())) {
    if (CurMI.readsRegister(MaskReg, TRI)) {
      if (KMovMI || BlendMI)
        return false; // Fail: Mask has MULTIPLE uses

      unsigned UseOpc = CurMI.getOpcode();
      bool IsKMOV = UseOpc == X86::KMOVBrk || UseOpc == X86::KMOVWrk ||
                    UseOpc == X86::KMOVDrk;
      // Only allow non-narrowing KMOV uses of the mask.
      if (IsKMOV && CurMI.getOperand(1).getReg() == MaskReg &&
          !usesExtendedRegister(CurMI) &&
          !isKMovNarrowing(getMovMskBits(Opc), UseOpc)) {
        KMovMI = &CurMI;
        // continue scanning to ensure
        // there are no *other* uses of the mask later in the block.
      } else if (!IsSignMaskCmp && isCompressibleBlendVUse(BlendOpc, UseOpc) &&
                 CurMI.getOperand(2).getReg() == MaskReg &&
                 !usesExtendedRegister(CurMI) &&
                 checkPredicate(BlendOpc, &ST)) {
        BlendMI = &CurMI;
      } else {
        return false;
      }
    }

    if (CurMI.modifiesRegister(MaskReg, TRI)) {
      if (!KMovMI && !BlendMI)
        return false; // Mask clobbered before use
      break;
    }

    if (!KMovMI && !BlendMI && CurMI.modifiesRegister(SrcVecReg, TRI)) {
      return false; // SrcVecReg modified before it could be reused
    }
  }

  if (!KMovMI && !BlendMI)
    return false;

  unsigned MovMskBits = getMovMskBits(Opc);
  // Bounded complements define EFLAGS, unlike VPCMP + KMOV. A 32-bit
  // complement uses NOT, which does not modify EFLAGS.
  if (IsSignMaskCmp && KMovMI) {
    if (KMovMI->getOperand(0).isDead() ||
        (MovMskBits != 32 &&
         MBB.computeRegisterLiveness(
             TRI, X86::EFLAGS,
             std::next(MachineBasicBlock::const_iterator(*KMovMI)),
             MBB.size()) != MachineBasicBlock::LQR_Dead))
      return false;
  }

  // Check if MaskReg is used in any other basic blocks
  for (const MachineInstr &UseMI : MRI->use_instructions(MaskReg))
    if (UseMI.getParent() != &MBB)
      return false;

  // Apply the transformation
  MachineInstr *NewMI = nullptr;
  if (KMovMI) {
    MachineOperand OldDst = KMovMI->getOperand(0);
    KMovMI->setDesc(TII->get(MovMskOpc));
    MachineOperand &NewSrc = KMovMI->getOperand(1);
    NewSrc.setReg(SrcVecReg);
    // setReg() keeps the mask operand's kill flag; take the source's kill
    // state from the VPMOV instead.
    NewSrc.setIsKill(MI.getOperand(1).isKill());
    NewMI = KMovMI;
    if (IsSignMaskCmp) {
      Register DstReg = OldDst.getReg();
      int64_t ComplementMask =
          APInt::getLowBitsSet(32, MovMskBits).getSExtValue();
      unsigned ComplementOpc =
          MovMskBits == 32
              ? X86::NOT32r
              : (isInt<8>(ComplementMask) ? X86::XOR32ri8 : X86::XOR32ri);
      auto MIB = BuildMI(MBB, std::next(MachineBasicBlock::iterator(*KMovMI)),
                         KMovMI->getDebugLoc(), TII->get(ComplementOpc), DstReg)
                     .addReg(DstReg, RegState::Kill);
      if (MovMskBits != 32) {
        MIB.addImm(ComplementMask);
        MIB->findRegisterDefOperand(X86::EFLAGS, TRI)->setIsDead();
      }
      MIB->getOperand(0).setIsRenamable(OldDst.isRenamable());
    }
  } else if (BlendMI) {
    const MachineOperand &MaskVec = MI.getOperand(1);
    const MachineOperand &Dst = BlendMI->getOperand(0);
    const MachineOperand &Passthru = BlendMI->getOperand(1);
    const MachineOperand &Src = BlendMI->getOperand(3);

    // Build a replacement instead of changing BlendMI in place because
    // VMOV*rrk has a tied passthrough operand and a different operand order
    // than VBLENDV.
    auto MIB =
        BuildMI(MBB, *BlendMI, BlendMI->getDebugLoc(), TII->get(BlendOpc))
            .addReg(Dst.getReg(), getRegState(Dst))
            .addReg(Passthru.getReg(), getRegState(Passthru))
            .addReg(Src.getReg(), getRegState(Src))
            .addReg(MaskVec.getReg(), getRegState(MaskVec));
    NewMI = MIB;
    ToErase.push_back(BlendMI);
  }
  assert(NewMI && "Expected a compressed instruction");
  NewMI->setAsmPrinterFlag(X86::AC_EVEX_2_VEX);
  ToErase.push_back(&MI);
  if (ConstantDefOnlyFeedsCmp && MI.getOperand(2).isKill())
    ToErase.push_back(ConstantDef);
  return true;
}

static bool CompressEVEXImpl(MachineInstr &MI, MachineBasicBlock &MBB,
                             const X86Subtarget &ST,
                             SmallVectorImpl<MachineInstr *> &ToErase) {
  uint64_t TSFlags = MI.getDesc().TSFlags;

  // Check for EVEX instructions only.
  if ((TSFlags & X86II::EncodingMask) != X86II::EVEX)
    return false;

  // Instructions with mask or 512-bit vector can't be converted to VEX.
  if (TSFlags & (X86II::EVEX_K | X86II::EVEX_L2))
    return false;

  // Keep the EVEX encoding when there's 1-byte compressed disp8*N.
  if (hasShorterEVEXViaCDisp8(MI))
    return false;

  // Specialized mask-producing folds to MOVMSK/VBLENDV first.
  if (tryCompressMaskProducer(MI, MBB, ST, ToErase))
    return true;

  auto IsRedundantNewDataDest = [&](unsigned &Opc) {
    // $rbx = ADD64rr_ND $rbx, $rax / $rbx = ADD64rr_ND $rax, $rbx
    //   ->
    // $rbx = ADD64rr $rbx, $rax
    const MCInstrDesc &Desc = MI.getDesc();
    Register Reg0 = MI.getOperand(0).getReg();
    const MachineOperand &Op1 = MI.getOperand(1);
    if (!Op1.isReg() || X86::getFirstAddrOperandIdx(MI) == 1 ||
        X86::isCFCMOVCC(MI.getOpcode()))
      return false;
    Register Reg1 = Op1.getReg();
    if (Reg1 == Reg0)
      return true;

    // Op1 and Op2 may be commutable for ND instructions.
    if (!Desc.isCommutable() || Desc.getNumOperands() < 3 ||
        !MI.getOperand(2).isReg() || MI.getOperand(2).getReg() != Reg0)
      return false;
    // Opcode may change after commute, e.g. SHRD -> SHLD
    ST.getInstrInfo()->commuteInstruction(MI, false, 1, 2);
    Opc = MI.getOpcode();
    return true;
  };

  // EVEX_B has several meanings.
  // AVX512:
  //  register form: rounding control or SAE
  //  memory form: broadcast
  //
  // APX:
  //  MAP4: NDD, ZU
  //
  // For AVX512 cases, EVEX prefix is needed in order to carry this information
  // thus preventing the transformation to VEX encoding.
  bool IsND = X86II::hasNewDataDest(TSFlags);
  unsigned Opc = MI.getOpcode();
  bool IsSetZUCCm = Opc == X86::SETZUCCm;
  if (TSFlags & X86II::EVEX_B && !IsND && !IsSetZUCCm)
    return false;
  // MOVBE*rr is special because it has semantic of NDD but not set EVEX_B.
  bool IsNDLike = IsND || Opc == X86::MOVBE32rr || Opc == X86::MOVBE64rr;
  bool IsRedundantNDD = IsNDLike ? IsRedundantNewDataDest(Opc) : false;

  auto GetCompressedOpc = [&](unsigned Opc) -> unsigned {
    ArrayRef<X86TableEntry> Table = ArrayRef(X86CompressEVEXTable);
    const auto I = llvm::lower_bound(Table, Opc);
    if (I == Table.end() || I->OldOpc != Opc)
      return 0;

    if (usesExtendedRegister(MI) || !checkPredicate(I->NewOpc, &ST) ||
        !performCustomAdjustments(MI, I->NewOpc))
      return 0;
    return I->NewOpc;
  };

  Register Dst = MI.getOperand(0).getReg();
  if (IsRedundantNDD) {
    // Redundant NDD ops cannot be safely compressed if either:
    // - the legacy op would introduce a partial write that BreakFalseDeps
    // identified as a potential stall, or
    // - the op is writing to a subregister of a live register, i.e. the
    // full (zeroed) result is used.
    // Both cases are indicated by an implicit def of the superregister.
    if (Dst &&
        (X86::GR16RegClass.contains(Dst) || X86::GR8RegClass.contains(Dst))) {
      Register Super = getX86SubSuperRegister(Dst, 64);
      if (MI.definesRegister(Super, /*TRI=*/nullptr))
        IsRedundantNDD = false;
    }

    // ADDrm/mr instructions with NDD + relocation had been transformed to the
    // instructions without NDD in X86SuppressAPXForRelocation pass. That is to
    // keep backward compatibility with linkers without APX support.
    if (!X86EnableAPXForRelocation)
      assert(!isAddMemInstrWithRelocation(MI) &&
             "Unexpected NDD instruction with relocation!");
  } else if (Opc == X86::ADD32ri_ND || Opc == X86::ADD64ri32_ND ||
             Opc == X86::ADD32rr_ND || Opc == X86::ADD64rr_ND) {
    // Non-redundant NDD ADD can be compressed to LEA when:
    // - No EGPR register used and
    // - EFLAGS is dead.
    if (!usesExtendedRegister(MI) &&
        MI.registerDefIsDead(X86::EFLAGS, /*TRI=*/nullptr)) {
      Register Src1 = MI.getOperand(1).getReg();
      const MachineOperand &Src2 = MI.getOperand(2);
      bool Is32BitReg = Opc == X86::ADD32ri_ND || Opc == X86::ADD32rr_ND;
      const MCInstrDesc &NewDesc =
          ST.getInstrInfo()->get(Is32BitReg ? X86::LEA64_32r : X86::LEA64r);
      if (Is32BitReg)
        Src1 = getX86SubSuperRegister(Src1, 64);
      MachineInstrBuilder MIB = BuildMI(MBB, MI, MI.getDebugLoc(), NewDesc, Dst)
                                    .addReg(Src1)
                                    .addImm(1);
      if (Opc == X86::ADD32ri_ND || Opc == X86::ADD64ri32_ND)
        MIB.addReg(0).add(Src2);
      else if (Is32BitReg)
        MIB.addReg(getX86SubSuperRegister(Src2.getReg(), 64)).addImm(0);
      else
        MIB.add(Src2).addImm(0);
      MIB.addReg(0);
      MI.removeFromParent();
      return true;
    }
  }

  // NonNF -> NF only if it's not a compressible NDD instruction and eflags is
  // dead.
  unsigned NewOpc = IsRedundantNDD
                        ? X86::getNonNDVariant(Opc)
                        : ((IsNDLike && ST.hasNF() &&
                            MI.registerDefIsDead(X86::EFLAGS, /*TRI=*/nullptr))
                               ? X86::getNFVariant(Opc)
                               : GetCompressedOpc(Opc));

  if (!NewOpc)
    return false;
  // NF (No Flags) instructions cannot compress to VEX/legacy encoding.
  // NF_ND can still compress to NF (both remain EVEX).
  assert((IsND || !(TSFlags & X86II::EVEX_NF)) &&
         "Unexpected to compress NF instructions without ND.");

  const MCInstrDesc &NewDesc = ST.getInstrInfo()->get(NewOpc);
  MI.setDesc(NewDesc);
  unsigned AsmComment;
  switch (NewDesc.TSFlags & X86II::EncodingMask) {
  case X86II::LEGACY:
    AsmComment = X86::AC_EVEX_2_LEGACY;
    break;
  case X86II::VEX:
    AsmComment = X86::AC_EVEX_2_VEX;
    break;
  case X86II::EVEX:
    AsmComment = X86::AC_EVEX_2_EVEX;
    assert(IsND && (NewDesc.TSFlags & X86II::EVEX_NF) &&
           "Unknown EVEX2EVEX compression");
    break;
  default:
    llvm_unreachable("Unknown EVEX compression");
  }
  MI.setAsmPrinterFlag(AsmComment);
  if (IsRedundantNDD)
    MI.tieOperands(0, 1);

  return true;
}

static bool runOnMF(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "Start X86CompressEVEXPass\n";);
#ifndef NDEBUG
  // Make sure the tables are sorted.
  static std::atomic<bool> TableChecked(false);
  if (!TableChecked.load(std::memory_order_relaxed)) {
    assert(llvm::is_sorted(X86CompressEVEXTable) &&
           "X86CompressEVEXTable is not sorted!");
    TableChecked.store(true, std::memory_order_relaxed);
  }
#endif
  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  if (!ST.hasAVX512() && !ST.hasEGPR() && !ST.hasNDD() && !ST.hasZU())
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    SmallVector<MachineInstr *, 4> ToErase;

    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      Changed |= CompressEVEXImpl(MI, MBB, ST, ToErase);
    }

    for (MachineInstr *MI : ToErase) {
      MI->eraseFromParent();
    }
  }
  LLVM_DEBUG(dbgs() << "End X86CompressEVEXPass\n";);
  return Changed;
}

INITIALIZE_PASS(CompressEVEXLegacy, COMP_EVEX_NAME, COMP_EVEX_DESC, false,
                false)

FunctionPass *llvm::createX86CompressEVEXLegacyPass() {
  return new CompressEVEXLegacy();
}

bool CompressEVEXLegacy::runOnMachineFunction(MachineFunction &MF) {
  return runOnMF(MF);
}

PreservedAnalyses
X86CompressEVEXPass::run(MachineFunction &MF,
                         MachineFunctionAnalysisManager &MFAM) {
  bool Changed = runOnMF(MF);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
