//===- X86EvexToVex.cpp ---------------------------------------------------===//
// Compress EVEX instructions to VEX encoding when possible to reduce code size
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file defines the pass that goes over all AVX-512 instructions which
/// are encoded using the EVEX prefix and if possible replaces them by their
/// corresponding VEX encoding which is usually shorter by 2 bytes.
/// EVEX instructions may be encoded via the VEX prefix when the AVX-512
/// instruction has a corresponding AVX/AVX2 opcode and when it does not
/// use the xmm or the mask registers or xmm/ymm registers with indexes
/// higher than 15.
/// The pass applies code reduction on the generated code for AVX-512 instrs.
//
//===----------------------------------------------------------------------===//

#include "InstPrinter/X86InstComments.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

// Including the generated EVEX2VEX tables.
struct X86EvexToVexCompressTableEntry {
  uint16_t EvexOpcode;
  uint16_t VexOpcode;
};
#include "X86GenEVEX2VEXTables.inc"

#define EVEX2VEX_DESC "Compressing EVEX instrs to VEX encoding when possible"
#define EVEX2VEX_NAME "x86-evex-to-vex-compress"

#define DEBUG_TYPE EVEX2VEX_NAME

namespace {

class EvexToVexInstPass : public MachineFunctionPass {

  /// X86EvexToVexCompressTable - Evex to Vex encoding opcode map.
  using EvexToVexTableType = DenseMap<unsigned, uint16_t>;
  EvexToVexTableType EvexToVex128Table;
  EvexToVexTableType EvexToVex256Table;

  /// For EVEX instructions that can be encoded using VEX encoding, replace
  /// them by the VEX encoding in order to reduce size.
  bool CompressEvexToVexImpl(MachineInstr &MI) const;

  /// For initializing the hash map tables of all AVX-512 EVEX
  /// corresponding to AVX/AVX2 opcodes.
  void AddTableEntry(EvexToVexTableType &EvexToVexTable, uint16_t EvexOp,
                     uint16_t VexOp);

public:
  static char ID;

  EvexToVexInstPass() : MachineFunctionPass(ID) {
    initializeEvexToVexInstPassPass(*PassRegistry::getPassRegistry());

    // Initialize the EVEX to VEX 128 table map.
    for (X86EvexToVexCompressTableEntry Entry : X86EvexToVex128CompressTable) {
      AddTableEntry(EvexToVex128Table, Entry.EvexOpcode, Entry.VexOpcode);
    }

    // Initialize the EVEX to VEX 256 table map.
    for (X86EvexToVexCompressTableEntry Entry : X86EvexToVex256CompressTable) {
      AddTableEntry(EvexToVex256Table, Entry.EvexOpcode, Entry.VexOpcode);
    }
  }

  StringRef getPassName() const override { return EVEX2VEX_DESC; }

  /// Loop over all of the basic blocks, replacing EVEX instructions
  /// by equivalent VEX instructions when possible for reducing code size.
  bool runOnMachineFunction(MachineFunction &MF) override;

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  /// Machine instruction info used throughout the class.
  const X86InstrInfo *TII;
};

} // end anonymous namespace

char EvexToVexInstPass::ID = 0;

bool EvexToVexInstPass::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();

  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  if (!ST.hasAVX512())
    return false;

  bool Changed = false;

  /// Go over all basic blocks in function and replace
  /// EVEX encoded instrs by VEX encoding when possible.
  for (MachineBasicBlock &MBB : MF) {

    // Traverse the basic block.
    for (MachineInstr &MI : MBB)
      Changed |= CompressEvexToVexImpl(MI);
  }

  return Changed;
}

void EvexToVexInstPass::AddTableEntry(EvexToVexTableType &EvexToVexTable,
                                      uint16_t EvexOp, uint16_t VexOp) {
  EvexToVexTable[EvexOp] = VexOp;
}

static bool usesExtendedRegister(const MachineInstr &MI) {
  auto isHiRegIdx = [](unsigned Reg) {
    // Check for XMM register with indexes between 16 - 31.
    if (Reg >= X86::XMM16 && Reg <= X86::XMM31)
      return true;

    // Check for YMM register with indexes between 16 - 31.
    if (Reg >= X86::YMM16 && Reg <= X86::YMM31)
      return true;

    return false;
  };

  // Check that operands are not ZMM regs or
  // XMM/YMM regs with hi indexes between 16 - 31.
  for (const MachineOperand &MO : MI.explicit_operands()) {
    if (!MO.isReg())
      continue;

    unsigned Reg = MO.getReg();

    assert(!(Reg >= X86::ZMM0 && Reg <= X86::ZMM31) &&
           "ZMM instructions should not be in the EVEX->VEX tables");

    if (isHiRegIdx(Reg))
      return true;
  }

  return false;
}

// Do any custom cleanup needed to finalize the conversion.
static void performCustomAdjustments(MachineInstr &MI, unsigned NewOpc) {
  (void)NewOpc;
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case X86::VALIGNDZ128rri:
  case X86::VALIGNDZ128rmi:
  case X86::VALIGNQZ128rri:
  case X86::VALIGNQZ128rmi:
    assert((NewOpc == X86::VPALIGNRrri || NewOpc == X86::VPALIGNRrmi) &&
           "Unexpected new opcode!");
    unsigned Scale = (Opc == X86::VALIGNQZ128rri ||
                      Opc == X86::VALIGNQZ128rmi) ? 8 : 4;
    MachineOperand &Imm = MI.getOperand(MI.getNumExplicitOperands()-1);
    Imm.setImm(Imm.getImm() * Scale);
    break;
  }
}


// For EVEX instructions that can be encoded using VEX encoding
// replace them by the VEX encoding in order to reduce size.
bool EvexToVexInstPass::CompressEvexToVexImpl(MachineInstr &MI) const {
  // VEX format.
  // # of bytes: 0,2,3  1      1      0,1   0,1,2,4  0,1
  //  [Prefixes] [VEX]  OPCODE ModR/M [SIB] [DISP]  [IMM]
  //
  // EVEX format.
  //  # of bytes: 4    1      1      1      4       / 1         1
  //  [Prefixes]  EVEX Opcode ModR/M [SIB] [Disp32] / [Disp8*N] [Immediate]

  const MCInstrDesc &Desc = MI.getDesc();

  // Check for EVEX instructions only.
  if ((Desc.TSFlags & X86II::EncodingMask) != X86II::EVEX)
    return false;

  // Check for EVEX instructions with mask or broadcast as in these cases
  // the EVEX prefix is needed in order to carry this information
  // thus preventing the transformation to VEX encoding.
  if (Desc.TSFlags & (X86II::EVEX_K | X86II::EVEX_B))
    return false;

  // Check for non EVEX_V512 instrs only.
  // EVEX_V512 instr: bit EVEX_L2 = 1; bit VEX_L = 0.
  if ((Desc.TSFlags & X86II::EVEX_L2) && !(Desc.TSFlags & X86II::VEX_L))
    return false;

  // EVEX_V128 instr: bit EVEX_L2 = 0, bit VEX_L = 0.
  bool IsEVEX_V128 =
      (!(Desc.TSFlags & X86II::EVEX_L2) && !(Desc.TSFlags & X86II::VEX_L));

  // EVEX_V256 instr: bit EVEX_L2 = 0, bit VEX_L = 1.
  bool IsEVEX_V256 =
      (!(Desc.TSFlags & X86II::EVEX_L2) && (Desc.TSFlags & X86II::VEX_L));

  unsigned NewOpc = 0;

  // Check for EVEX_V256 instructions.
  if (IsEVEX_V256) {
    // Search for opcode in the EvexToVex256 table.
    auto It = EvexToVex256Table.find(MI.getOpcode());
    if (It != EvexToVex256Table.end())
      NewOpc = It->second;
  }
  // Check for EVEX_V128 or Scalar instructions.
  else if (IsEVEX_V128) {
    // Search for opcode in the EvexToVex128 table.
    auto It = EvexToVex128Table.find(MI.getOpcode());
    if (It != EvexToVex128Table.end())
      NewOpc = It->second;
  }

  if (!NewOpc)
    return false;

  if (usesExtendedRegister(MI))
    return false;

  performCustomAdjustments(MI, NewOpc);

  MI.setDesc(TII->get(NewOpc));
  MI.setAsmPrinterFlag(AC_EVEX_2_VEX);
  return true;
}

INITIALIZE_PASS(EvexToVexInstPass, EVEX2VEX_NAME, EVEX2VEX_DESC, false, false)

FunctionPass *llvm::createX86EvexToVexInsts() {
  return new EvexToVexInstPass();
}
