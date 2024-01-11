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
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86InstComments.h"
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include <atomic>
#include <cassert>
#include <cstdint>

using namespace llvm;

// Including the generated EVEX compression tables.
struct X86CompressEVEXTableEntry {
  uint16_t OldOpc;
  uint16_t NewOpc;

  bool operator<(const X86CompressEVEXTableEntry &RHS) const {
    return OldOpc < RHS.OldOpc;
  }

  friend bool operator<(const X86CompressEVEXTableEntry &TE, unsigned Opc) {
    return TE.OldOpc < Opc;
  }
};
#include "X86GenCompressEVEXTables.inc"

#define COMP_EVEX_DESC "Compressing EVEX instrs when possible"
#define COMP_EVEX_NAME "x86-compress-evex"

#define DEBUG_TYPE COMP_EVEX_NAME

namespace {

class CompressEVEXPass : public MachineFunctionPass {
public:
  static char ID;
  CompressEVEXPass() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return COMP_EVEX_DESC; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }
};

} // end anonymous namespace

char CompressEVEXPass::ID = 0;

static bool usesExtendedRegister(const MachineInstr &MI) {
  auto isHiRegIdx = [](unsigned Reg) {
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

    Register Reg = MO.getReg();
    assert(!X86II::isZMMReg(Reg) &&
           "ZMM instructions should not be in the EVEX->VEX tables");
    if (isHiRegIdx(Reg))
      return true;
  }

  return false;
}

static bool checkVEXInstPredicate(unsigned OldOpc, const X86Subtarget &ST) {
  switch (OldOpc) {
  default:
    return true;
  case X86::VCVTNEPS2BF16Z128rm:
  case X86::VCVTNEPS2BF16Z128rr:
  case X86::VCVTNEPS2BF16Z256rm:
  case X86::VCVTNEPS2BF16Z256rr:
    return ST.hasAVXNECONVERT();
  case X86::VPDPBUSDSZ128m:
  case X86::VPDPBUSDSZ128r:
  case X86::VPDPBUSDSZ256m:
  case X86::VPDPBUSDSZ256r:
  case X86::VPDPBUSDZ128m:
  case X86::VPDPBUSDZ128r:
  case X86::VPDPBUSDZ256m:
  case X86::VPDPBUSDZ256r:
  case X86::VPDPWSSDSZ128m:
  case X86::VPDPWSSDSZ128r:
  case X86::VPDPWSSDSZ256m:
  case X86::VPDPWSSDSZ256r:
  case X86::VPDPWSSDZ128m:
  case X86::VPDPWSSDZ128r:
  case X86::VPDPWSSDZ256m:
  case X86::VPDPWSSDZ256r:
    return ST.hasAVXVNNI();
  case X86::VPMADD52HUQZ128m:
  case X86::VPMADD52HUQZ128r:
  case X86::VPMADD52HUQZ256m:
  case X86::VPMADD52HUQZ256r:
  case X86::VPMADD52LUQZ128m:
  case X86::VPMADD52LUQZ128r:
  case X86::VPMADD52LUQZ256m:
  case X86::VPMADD52LUQZ256r:
    return ST.hasAVXIFMA();
  }
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
    assert((NewOpc == X86::VPERM2F128rr || NewOpc == X86::VPERM2I128rr ||
            NewOpc == X86::VPERM2F128rm || NewOpc == X86::VPERM2I128rm) &&
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
  case X86::VRNDSCALESDZr:
  case X86::VRNDSCALESDZm:
  case X86::VRNDSCALESSZr:
  case X86::VRNDSCALESSZm:
  case X86::VRNDSCALESDZr_Int:
  case X86::VRNDSCALESDZm_Int:
  case X86::VRNDSCALESSZr_Int:
  case X86::VRNDSCALESSZm_Int:
    const MachineOperand &Imm = MI.getOperand(MI.getNumExplicitOperands() - 1);
    int64_t ImmVal = Imm.getImm();
    // Ensure that only bits 3:0 of the immediate are used.
    if ((ImmVal & 0xf) != ImmVal)
      return false;
    break;
  }

  return true;
}

static bool CompressEVEXImpl(MachineInstr &MI, const X86Subtarget &ST) {
  uint64_t TSFlags = MI.getDesc().TSFlags;

  // Check for EVEX instructions only.
  if ((TSFlags & X86II::EncodingMask) != X86II::EVEX)
    return false;

  // Instructions with mask or 512-bit vector can't be converted to VEX.
  if (TSFlags & (X86II::EVEX_K | X86II::EVEX_L2))
    return false;

  // EVEX_B has several meanings.
  // AVX512:
  //  register form: rounding control or SAE
  //  memory form: broadcast
  //
  // APX:
  //  MAP4: NDD
  //
  // For AVX512 cases, EVEX prefix is needed in order to carry this information
  // thus preventing the transformation to VEX encoding.
  if (TSFlags & X86II::EVEX_B)
    return false;

  ArrayRef<X86CompressEVEXTableEntry> Table = ArrayRef(X86CompressEVEXTable);

  unsigned Opc = MI.getOpcode();
  const auto *I = llvm::lower_bound(Table, Opc);
  if (I == Table.end() || I->OldOpc != Opc)
    return false;

  if (usesExtendedRegister(MI) || !checkVEXInstPredicate(Opc, ST) ||
      !performCustomAdjustments(MI, I->NewOpc))
    return false;

  const MCInstrDesc &NewDesc = ST.getInstrInfo()->get(I->NewOpc);
  MI.setDesc(NewDesc);
  uint64_t Encoding = NewDesc.TSFlags & X86II::EncodingMask;
  auto AsmComment =
      (Encoding == X86II::VEX) ? X86::AC_EVEX_2_VEX : X86::AC_EVEX_2_LEGACY;
  MI.setAsmPrinterFlag(AsmComment);
  return true;
}

bool CompressEVEXPass::runOnMachineFunction(MachineFunction &MF) {
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
  if (!ST.hasAVX512() && !ST.hasEGPR() && !ST.hasNDD())
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Traverse the basic block.
    for (MachineInstr &MI : MBB)
      Changed |= CompressEVEXImpl(MI, ST);
  }

  return Changed;
}

INITIALIZE_PASS(CompressEVEXPass, COMP_EVEX_NAME, COMP_EVEX_DESC, false, false)

FunctionPass *llvm::createX86CompressEVEXPass() {
  return new CompressEVEXPass();
}
