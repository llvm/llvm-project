//===-- ARMExpandPseudoInsts.cpp - Expand pseudo instructions -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions to allow proper scheduling, if-conversion, and other late
// optimizations. This pass should be run after register allocation but before
// the post-regalloc scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMConstantPoolValue.h"
#include "ARMMachineFunctionInfo.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h" // FIXME: for debug only. remove!
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "arm-pseudo"

static cl::opt<bool>
VerifyARMPseudo("verify-arm-pseudo-expand", cl::Hidden,
                cl::desc("Verify machine code after expanding ARM pseudos"));

namespace {
  class ARMExpandPseudo : public MachineFunctionPass {
  public:
    static char ID;
    ARMExpandPseudo() : MachineFunctionPass(ID) {}

    const ARMBaseInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    const ARMSubtarget *STI;
    ARMFunctionInfo *AFI;

    bool runOnMachineFunction(MachineFunction &Fn) override;

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::AllVRegsAllocated);
    }

    const char *getPassName() const override {
      return "ARM pseudo instruction expansion pass";
    }

  private:
    void TransferImpOps(MachineInstr &OldMI,
                        MachineInstrBuilder &UseMI, MachineInstrBuilder &DefMI);
    bool ExpandMI(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator MBBI,
                  MachineBasicBlock::iterator &NextMBBI);
    bool ExpandMBB(MachineBasicBlock &MBB);
    void ExpandVLD(MachineBasicBlock::iterator &MBBI);
    void ExpandVST(MachineBasicBlock::iterator &MBBI);
    void ExpandLaneOp(MachineBasicBlock::iterator &MBBI);
    void ExpandVTBL(MachineBasicBlock::iterator &MBBI,
                    unsigned Opc, bool IsExt);
    void ExpandMOV32BitImm(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator &MBBI);
    bool ExpandCMP_SWAP(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MBBI, unsigned LdrexOp,
                        unsigned StrexOp, unsigned UxtOp,
                        MachineBasicBlock::iterator &NextMBBI);

    bool ExpandCMP_SWAP_64(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           MachineBasicBlock::iterator &NextMBBI);
  };
  char ARMExpandPseudo::ID = 0;
}

/// TransferImpOps - Transfer implicit operands on the pseudo instruction to
/// the instructions created from the expansion.
void ARMExpandPseudo::TransferImpOps(MachineInstr &OldMI,
                                     MachineInstrBuilder &UseMI,
                                     MachineInstrBuilder &DefMI) {
  const MCInstrDesc &Desc = OldMI.getDesc();
  for (unsigned i = Desc.getNumOperands(), e = OldMI.getNumOperands();
       i != e; ++i) {
    const MachineOperand &MO = OldMI.getOperand(i);
    assert(MO.isReg() && MO.getReg());
    if (MO.isUse())
      UseMI.addOperand(MO);
    else
      DefMI.addOperand(MO);
  }
}

namespace {
  // Constants for register spacing in NEON load/store instructions.
  // For quad-register load-lane and store-lane pseudo instructors, the
  // spacing is initially assumed to be EvenDblSpc, and that is changed to
  // OddDblSpc depending on the lane number operand.
  enum NEONRegSpacing {
    SingleSpc,
    EvenDblSpc,
    OddDblSpc
  };

  // Entries for NEON load/store information table.  The table is sorted by
  // PseudoOpc for fast binary-search lookups.
  struct NEONLdStTableEntry {
    uint16_t PseudoOpc;
    uint16_t RealOpc;
    bool IsLoad;
    bool isUpdating;
    bool hasWritebackOperand;
    uint8_t RegSpacing; // One of type NEONRegSpacing
    uint8_t NumRegs; // D registers loaded or stored
    uint8_t RegElts; // elements per D register; used for lane ops
    // FIXME: Temporary flag to denote whether the real instruction takes
    // a single register (like the encoding) or all of the registers in
    // the list (like the asm syntax and the isel DAG). When all definitions
    // are converted to take only the single encoded register, this will
    // go away.
    bool copyAllListRegs;

    // Comparison methods for binary search of the table.
    bool operator<(const NEONLdStTableEntry &TE) const {
      return PseudoOpc < TE.PseudoOpc;
    }
    friend bool operator<(const NEONLdStTableEntry &TE, unsigned PseudoOpc) {
      return TE.PseudoOpc < PseudoOpc;
    }
    friend bool LLVM_ATTRIBUTE_UNUSED operator<(unsigned PseudoOpc,
                                                const NEONLdStTableEntry &TE) {
      return PseudoOpc < TE.PseudoOpc;
    }
  };
}

static const NEONLdStTableEntry NEONLdStTable[] = {
{ ARM::VLD1LNq16Pseudo,     ARM::VLD1LNd16,     true, false, false, EvenDblSpc, 1, 4 ,true},
{ ARM::VLD1LNq16Pseudo_UPD, ARM::VLD1LNd16_UPD, true, true, true,  EvenDblSpc, 1, 4 ,true},
{ ARM::VLD1LNq32Pseudo,     ARM::VLD1LNd32,     true, false, false, EvenDblSpc, 1, 2 ,true},
{ ARM::VLD1LNq32Pseudo_UPD, ARM::VLD1LNd32_UPD, true, true, true,  EvenDblSpc, 1, 2 ,true},
{ ARM::VLD1LNq8Pseudo,      ARM::VLD1LNd8,      true, false, false, EvenDblSpc, 1, 8 ,true},
{ ARM::VLD1LNq8Pseudo_UPD,  ARM::VLD1LNd8_UPD, true, true, true,  EvenDblSpc, 1, 8 ,true},

{ ARM::VLD1d64QPseudo,      ARM::VLD1d64Q,     true,  false, false, SingleSpc,  4, 1 ,false},
{ ARM::VLD1d64QPseudoWB_fixed,  ARM::VLD1d64Qwb_fixed,   true,  true, false, SingleSpc,  4, 1 ,false},
{ ARM::VLD1d64TPseudo,      ARM::VLD1d64T,     true,  false, false, SingleSpc,  3, 1 ,false},
{ ARM::VLD1d64TPseudoWB_fixed,  ARM::VLD1d64Twb_fixed,   true,  true, false, SingleSpc,  3, 1 ,false},

{ ARM::VLD2LNd16Pseudo,     ARM::VLD2LNd16,     true, false, false, SingleSpc,  2, 4 ,true},
{ ARM::VLD2LNd16Pseudo_UPD, ARM::VLD2LNd16_UPD, true, true, true,  SingleSpc,  2, 4 ,true},
{ ARM::VLD2LNd32Pseudo,     ARM::VLD2LNd32,     true, false, false, SingleSpc,  2, 2 ,true},
{ ARM::VLD2LNd32Pseudo_UPD, ARM::VLD2LNd32_UPD, true, true, true,  SingleSpc,  2, 2 ,true},
{ ARM::VLD2LNd8Pseudo,      ARM::VLD2LNd8,      true, false, false, SingleSpc,  2, 8 ,true},
{ ARM::VLD2LNd8Pseudo_UPD,  ARM::VLD2LNd8_UPD, true, true, true,  SingleSpc,  2, 8 ,true},
{ ARM::VLD2LNq16Pseudo,     ARM::VLD2LNq16,     true, false, false, EvenDblSpc, 2, 4 ,true},
{ ARM::VLD2LNq16Pseudo_UPD, ARM::VLD2LNq16_UPD, true, true, true,  EvenDblSpc, 2, 4 ,true},
{ ARM::VLD2LNq32Pseudo,     ARM::VLD2LNq32,     true, false, false, EvenDblSpc, 2, 2 ,true},
{ ARM::VLD2LNq32Pseudo_UPD, ARM::VLD2LNq32_UPD, true, true, true,  EvenDblSpc, 2, 2 ,true},

{ ARM::VLD2q16Pseudo,       ARM::VLD2q16,      true,  false, false, SingleSpc,  4, 4 ,false},
{ ARM::VLD2q16PseudoWB_fixed,   ARM::VLD2q16wb_fixed, true, true, false,  SingleSpc,  4, 4 ,false},
{ ARM::VLD2q16PseudoWB_register,   ARM::VLD2q16wb_register, true, true, true,  SingleSpc,  4, 4 ,false},
{ ARM::VLD2q32Pseudo,       ARM::VLD2q32,      true,  false, false, SingleSpc,  4, 2 ,false},
{ ARM::VLD2q32PseudoWB_fixed,   ARM::VLD2q32wb_fixed, true, true, false,  SingleSpc,  4, 2 ,false},
{ ARM::VLD2q32PseudoWB_register,   ARM::VLD2q32wb_register, true, true, true,  SingleSpc,  4, 2 ,false},
{ ARM::VLD2q8Pseudo,        ARM::VLD2q8,       true,  false, false, SingleSpc,  4, 8 ,false},
{ ARM::VLD2q8PseudoWB_fixed,    ARM::VLD2q8wb_fixed, true, true, false,  SingleSpc,  4, 8 ,false},
{ ARM::VLD2q8PseudoWB_register,    ARM::VLD2q8wb_register, true, true, true,  SingleSpc,  4, 8 ,false},

{ ARM::VLD3DUPd16Pseudo,     ARM::VLD3DUPd16,     true, false, false, SingleSpc, 3, 4,true},
{ ARM::VLD3DUPd16Pseudo_UPD, ARM::VLD3DUPd16_UPD, true, true, true,  SingleSpc, 3, 4,true},
{ ARM::VLD3DUPd32Pseudo,     ARM::VLD3DUPd32,     true, false, false, SingleSpc, 3, 2,true},
{ ARM::VLD3DUPd32Pseudo_UPD, ARM::VLD3DUPd32_UPD, true, true, true,  SingleSpc, 3, 2,true},
{ ARM::VLD3DUPd8Pseudo,      ARM::VLD3DUPd8,      true, false, false, SingleSpc, 3, 8,true},
{ ARM::VLD3DUPd8Pseudo_UPD,  ARM::VLD3DUPd8_UPD, true, true, true,  SingleSpc, 3, 8,true},

{ ARM::VLD3LNd16Pseudo,     ARM::VLD3LNd16,     true, false, false, SingleSpc,  3, 4 ,true},
{ ARM::VLD3LNd16Pseudo_UPD, ARM::VLD3LNd16_UPD, true, true, true,  SingleSpc,  3, 4 ,true},
{ ARM::VLD3LNd32Pseudo,     ARM::VLD3LNd32,     true, false, false, SingleSpc,  3, 2 ,true},
{ ARM::VLD3LNd32Pseudo_UPD, ARM::VLD3LNd32_UPD, true, true, true,  SingleSpc,  3, 2 ,true},
{ ARM::VLD3LNd8Pseudo,      ARM::VLD3LNd8,      true, false, false, SingleSpc,  3, 8 ,true},
{ ARM::VLD3LNd8Pseudo_UPD,  ARM::VLD3LNd8_UPD, true, true, true,  SingleSpc,  3, 8 ,true},
{ ARM::VLD3LNq16Pseudo,     ARM::VLD3LNq16,     true, false, false, EvenDblSpc, 3, 4 ,true},
{ ARM::VLD3LNq16Pseudo_UPD, ARM::VLD3LNq16_UPD, true, true, true,  EvenDblSpc, 3, 4 ,true},
{ ARM::VLD3LNq32Pseudo,     ARM::VLD3LNq32,     true, false, false, EvenDblSpc, 3, 2 ,true},
{ ARM::VLD3LNq32Pseudo_UPD, ARM::VLD3LNq32_UPD, true, true, true,  EvenDblSpc, 3, 2 ,true},

{ ARM::VLD3d16Pseudo,       ARM::VLD3d16,      true,  false, false, SingleSpc,  3, 4 ,true},
{ ARM::VLD3d16Pseudo_UPD,   ARM::VLD3d16_UPD, true, true, true,  SingleSpc,  3, 4 ,true},
{ ARM::VLD3d32Pseudo,       ARM::VLD3d32,      true,  false, false, SingleSpc,  3, 2 ,true},
{ ARM::VLD3d32Pseudo_UPD,   ARM::VLD3d32_UPD, true, true, true,  SingleSpc,  3, 2 ,true},
{ ARM::VLD3d8Pseudo,        ARM::VLD3d8,       true,  false, false, SingleSpc,  3, 8 ,true},
{ ARM::VLD3d8Pseudo_UPD,    ARM::VLD3d8_UPD, true, true, true,  SingleSpc,  3, 8 ,true},

{ ARM::VLD3q16Pseudo_UPD,    ARM::VLD3q16_UPD, true, true, true,  EvenDblSpc, 3, 4 ,true},
{ ARM::VLD3q16oddPseudo,     ARM::VLD3q16,     true,  false, false, OddDblSpc,  3, 4 ,true},
{ ARM::VLD3q16oddPseudo_UPD, ARM::VLD3q16_UPD, true, true, true,  OddDblSpc,  3, 4 ,true},
{ ARM::VLD3q32Pseudo_UPD,    ARM::VLD3q32_UPD, true, true, true,  EvenDblSpc, 3, 2 ,true},
{ ARM::VLD3q32oddPseudo,     ARM::VLD3q32,     true,  false, false, OddDblSpc,  3, 2 ,true},
{ ARM::VLD3q32oddPseudo_UPD, ARM::VLD3q32_UPD, true, true, true,  OddDblSpc,  3, 2 ,true},
{ ARM::VLD3q8Pseudo_UPD,     ARM::VLD3q8_UPD, true, true, true,  EvenDblSpc, 3, 8 ,true},
{ ARM::VLD3q8oddPseudo,      ARM::VLD3q8,      true,  false, false, OddDblSpc,  3, 8 ,true},
{ ARM::VLD3q8oddPseudo_UPD,  ARM::VLD3q8_UPD, true, true, true,  OddDblSpc,  3, 8 ,true},

{ ARM::VLD4DUPd16Pseudo,     ARM::VLD4DUPd16,     true, false, false, SingleSpc, 4, 4,true},
{ ARM::VLD4DUPd16Pseudo_UPD, ARM::VLD4DUPd16_UPD, true, true, true,  SingleSpc, 4, 4,true},
{ ARM::VLD4DUPd32Pseudo,     ARM::VLD4DUPd32,     true, false, false, SingleSpc, 4, 2,true},
{ ARM::VLD4DUPd32Pseudo_UPD, ARM::VLD4DUPd32_UPD, true, true, true,  SingleSpc, 4, 2,true},
{ ARM::VLD4DUPd8Pseudo,      ARM::VLD4DUPd8,      true, false, false, SingleSpc, 4, 8,true},
{ ARM::VLD4DUPd8Pseudo_UPD,  ARM::VLD4DUPd8_UPD, true, true, true,  SingleSpc, 4, 8,true},

{ ARM::VLD4LNd16Pseudo,     ARM::VLD4LNd16,     true, false, false, SingleSpc,  4, 4 ,true},
{ ARM::VLD4LNd16Pseudo_UPD, ARM::VLD4LNd16_UPD, true, true, true,  SingleSpc,  4, 4 ,true},
{ ARM::VLD4LNd32Pseudo,     ARM::VLD4LNd32,     true, false, false, SingleSpc,  4, 2 ,true},
{ ARM::VLD4LNd32Pseudo_UPD, ARM::VLD4LNd32_UPD, true, true, true,  SingleSpc,  4, 2 ,true},
{ ARM::VLD4LNd8Pseudo,      ARM::VLD4LNd8,      true, false, false, SingleSpc,  4, 8 ,true},
{ ARM::VLD4LNd8Pseudo_UPD,  ARM::VLD4LNd8_UPD, true, true, true,  SingleSpc,  4, 8 ,true},
{ ARM::VLD4LNq16Pseudo,     ARM::VLD4LNq16,     true, false, false, EvenDblSpc, 4, 4 ,true},
{ ARM::VLD4LNq16Pseudo_UPD, ARM::VLD4LNq16_UPD, true, true, true,  EvenDblSpc, 4, 4 ,true},
{ ARM::VLD4LNq32Pseudo,     ARM::VLD4LNq32,     true, false, false, EvenDblSpc, 4, 2 ,true},
{ ARM::VLD4LNq32Pseudo_UPD, ARM::VLD4LNq32_UPD, true, true, true,  EvenDblSpc, 4, 2 ,true},

{ ARM::VLD4d16Pseudo,       ARM::VLD4d16,      true,  false, false, SingleSpc,  4, 4 ,true},
{ ARM::VLD4d16Pseudo_UPD,   ARM::VLD4d16_UPD, true, true, true,  SingleSpc,  4, 4 ,true},
{ ARM::VLD4d32Pseudo,       ARM::VLD4d32,      true,  false, false, SingleSpc,  4, 2 ,true},
{ ARM::VLD4d32Pseudo_UPD,   ARM::VLD4d32_UPD, true, true, true,  SingleSpc,  4, 2 ,true},
{ ARM::VLD4d8Pseudo,        ARM::VLD4d8,       true,  false, false, SingleSpc,  4, 8 ,true},
{ ARM::VLD4d8Pseudo_UPD,    ARM::VLD4d8_UPD, true, true, true,  SingleSpc,  4, 8 ,true},

{ ARM::VLD4q16Pseudo_UPD,    ARM::VLD4q16_UPD, true, true, true,  EvenDblSpc, 4, 4 ,true},
{ ARM::VLD4q16oddPseudo,     ARM::VLD4q16,     true,  false, false, OddDblSpc,  4, 4 ,true},
{ ARM::VLD4q16oddPseudo_UPD, ARM::VLD4q16_UPD, true, true, true,  OddDblSpc,  4, 4 ,true},
{ ARM::VLD4q32Pseudo_UPD,    ARM::VLD4q32_UPD, true, true, true,  EvenDblSpc, 4, 2 ,true},
{ ARM::VLD4q32oddPseudo,     ARM::VLD4q32,     true,  false, false, OddDblSpc,  4, 2 ,true},
{ ARM::VLD4q32oddPseudo_UPD, ARM::VLD4q32_UPD, true, true, true,  OddDblSpc,  4, 2 ,true},
{ ARM::VLD4q8Pseudo_UPD,     ARM::VLD4q8_UPD, true, true, true,  EvenDblSpc, 4, 8 ,true},
{ ARM::VLD4q8oddPseudo,      ARM::VLD4q8,      true,  false, false, OddDblSpc,  4, 8 ,true},
{ ARM::VLD4q8oddPseudo_UPD,  ARM::VLD4q8_UPD, true, true, true,  OddDblSpc,  4, 8 ,true},

{ ARM::VST1LNq16Pseudo,     ARM::VST1LNd16,    false, false, false, EvenDblSpc, 1, 4 ,true},
{ ARM::VST1LNq16Pseudo_UPD, ARM::VST1LNd16_UPD, false, true, true,  EvenDblSpc, 1, 4 ,true},
{ ARM::VST1LNq32Pseudo,     ARM::VST1LNd32,    false, false, false, EvenDblSpc, 1, 2 ,true},
{ ARM::VST1LNq32Pseudo_UPD, ARM::VST1LNd32_UPD, false, true, true,  EvenDblSpc, 1, 2 ,true},
{ ARM::VST1LNq8Pseudo,      ARM::VST1LNd8,     false, false, false, EvenDblSpc, 1, 8 ,true},
{ ARM::VST1LNq8Pseudo_UPD,  ARM::VST1LNd8_UPD, false, true, true,  EvenDblSpc, 1, 8 ,true},

{ ARM::VST1d64QPseudo,      ARM::VST1d64Q,     false, false, false, SingleSpc,  4, 1 ,false},
{ ARM::VST1d64QPseudoWB_fixed,  ARM::VST1d64Qwb_fixed, false, true, false,  SingleSpc,  4, 1 ,false},
{ ARM::VST1d64QPseudoWB_register, ARM::VST1d64Qwb_register, false, true, true,  SingleSpc,  4, 1 ,false},
{ ARM::VST1d64TPseudo,      ARM::VST1d64T,     false, false, false, SingleSpc,  3, 1 ,false},
{ ARM::VST1d64TPseudoWB_fixed,  ARM::VST1d64Twb_fixed, false, true, false,  SingleSpc,  3, 1 ,false},
{ ARM::VST1d64TPseudoWB_register,  ARM::VST1d64Twb_register, false, true, true,  SingleSpc,  3, 1 ,false},

{ ARM::VST2LNd16Pseudo,     ARM::VST2LNd16,     false, false, false, SingleSpc, 2, 4 ,true},
{ ARM::VST2LNd16Pseudo_UPD, ARM::VST2LNd16_UPD, false, true, true,  SingleSpc, 2, 4 ,true},
{ ARM::VST2LNd32Pseudo,     ARM::VST2LNd32,     false, false, false, SingleSpc, 2, 2 ,true},
{ ARM::VST2LNd32Pseudo_UPD, ARM::VST2LNd32_UPD, false, true, true,  SingleSpc, 2, 2 ,true},
{ ARM::VST2LNd8Pseudo,      ARM::VST2LNd8,      false, false, false, SingleSpc, 2, 8 ,true},
{ ARM::VST2LNd8Pseudo_UPD,  ARM::VST2LNd8_UPD, false, true, true,  SingleSpc, 2, 8 ,true},
{ ARM::VST2LNq16Pseudo,     ARM::VST2LNq16,     false, false, false, EvenDblSpc, 2, 4,true},
{ ARM::VST2LNq16Pseudo_UPD, ARM::VST2LNq16_UPD, false, true, true,  EvenDblSpc, 2, 4,true},
{ ARM::VST2LNq32Pseudo,     ARM::VST2LNq32,     false, false, false, EvenDblSpc, 2, 2,true},
{ ARM::VST2LNq32Pseudo_UPD, ARM::VST2LNq32_UPD, false, true, true,  EvenDblSpc, 2, 2,true},

{ ARM::VST2q16Pseudo,       ARM::VST2q16,      false, false, false, SingleSpc,  4, 4 ,false},
{ ARM::VST2q16PseudoWB_fixed,   ARM::VST2q16wb_fixed, false, true, false,  SingleSpc,  4, 4 ,false},
{ ARM::VST2q16PseudoWB_register,   ARM::VST2q16wb_register, false, true, true,  SingleSpc,  4, 4 ,false},
{ ARM::VST2q32Pseudo,       ARM::VST2q32,      false, false, false, SingleSpc,  4, 2 ,false},
{ ARM::VST2q32PseudoWB_fixed,   ARM::VST2q32wb_fixed, false, true, false,  SingleSpc,  4, 2 ,false},
{ ARM::VST2q32PseudoWB_register,   ARM::VST2q32wb_register, false, true, true,  SingleSpc,  4, 2 ,false},
{ ARM::VST2q8Pseudo,        ARM::VST2q8,       false, false, false, SingleSpc,  4, 8 ,false},
{ ARM::VST2q8PseudoWB_fixed,    ARM::VST2q8wb_fixed, false, true, false,  SingleSpc,  4, 8 ,false},
{ ARM::VST2q8PseudoWB_register,    ARM::VST2q8wb_register, false, true, true,  SingleSpc,  4, 8 ,false},

{ ARM::VST3LNd16Pseudo,     ARM::VST3LNd16,     false, false, false, SingleSpc, 3, 4 ,true},
{ ARM::VST3LNd16Pseudo_UPD, ARM::VST3LNd16_UPD, false, true, true,  SingleSpc, 3, 4 ,true},
{ ARM::VST3LNd32Pseudo,     ARM::VST3LNd32,     false, false, false, SingleSpc, 3, 2 ,true},
{ ARM::VST3LNd32Pseudo_UPD, ARM::VST3LNd32_UPD, false, true, true,  SingleSpc, 3, 2 ,true},
{ ARM::VST3LNd8Pseudo,      ARM::VST3LNd8,      false, false, false, SingleSpc, 3, 8 ,true},
{ ARM::VST3LNd8Pseudo_UPD,  ARM::VST3LNd8_UPD, false, true, true,  SingleSpc, 3, 8 ,true},
{ ARM::VST3LNq16Pseudo,     ARM::VST3LNq16,     false, false, false, EvenDblSpc, 3, 4,true},
{ ARM::VST3LNq16Pseudo_UPD, ARM::VST3LNq16_UPD, false, true, true,  EvenDblSpc, 3, 4,true},
{ ARM::VST3LNq32Pseudo,     ARM::VST3LNq32,     false, false, false, EvenDblSpc, 3, 2,true},
{ ARM::VST3LNq32Pseudo_UPD, ARM::VST3LNq32_UPD, false, true, true,  EvenDblSpc, 3, 2,true},

{ ARM::VST3d16Pseudo,       ARM::VST3d16,      false, false, false, SingleSpc,  3, 4 ,true},
{ ARM::VST3d16Pseudo_UPD,   ARM::VST3d16_UPD, false, true, true,  SingleSpc,  3, 4 ,true},
{ ARM::VST3d32Pseudo,       ARM::VST3d32,      false, false, false, SingleSpc,  3, 2 ,true},
{ ARM::VST3d32Pseudo_UPD,   ARM::VST3d32_UPD, false, true, true,  SingleSpc,  3, 2 ,true},
{ ARM::VST3d8Pseudo,        ARM::VST3d8,       false, false, false, SingleSpc,  3, 8 ,true},
{ ARM::VST3d8Pseudo_UPD,    ARM::VST3d8_UPD, false, true, true,  SingleSpc,  3, 8 ,true},

{ ARM::VST3q16Pseudo_UPD,    ARM::VST3q16_UPD, false, true, true,  EvenDblSpc, 3, 4 ,true},
{ ARM::VST3q16oddPseudo,     ARM::VST3q16,     false, false, false, OddDblSpc,  3, 4 ,true},
{ ARM::VST3q16oddPseudo_UPD, ARM::VST3q16_UPD, false, true, true,  OddDblSpc,  3, 4 ,true},
{ ARM::VST3q32Pseudo_UPD,    ARM::VST3q32_UPD, false, true, true,  EvenDblSpc, 3, 2 ,true},
{ ARM::VST3q32oddPseudo,     ARM::VST3q32,     false, false, false, OddDblSpc,  3, 2 ,true},
{ ARM::VST3q32oddPseudo_UPD, ARM::VST3q32_UPD, false, true, true,  OddDblSpc,  3, 2 ,true},
{ ARM::VST3q8Pseudo_UPD,     ARM::VST3q8_UPD, false, true, true,  EvenDblSpc, 3, 8 ,true},
{ ARM::VST3q8oddPseudo,      ARM::VST3q8,      false, false, false, OddDblSpc,  3, 8 ,true},
{ ARM::VST3q8oddPseudo_UPD,  ARM::VST3q8_UPD, false, true, true,  OddDblSpc,  3, 8 ,true},

{ ARM::VST4LNd16Pseudo,     ARM::VST4LNd16,     false, false, false, SingleSpc, 4, 4 ,true},
{ ARM::VST4LNd16Pseudo_UPD, ARM::VST4LNd16_UPD, false, true, true,  SingleSpc, 4, 4 ,true},
{ ARM::VST4LNd32Pseudo,     ARM::VST4LNd32,     false, false, false, SingleSpc, 4, 2 ,true},
{ ARM::VST4LNd32Pseudo_UPD, ARM::VST4LNd32_UPD, false, true, true,  SingleSpc, 4, 2 ,true},
{ ARM::VST4LNd8Pseudo,      ARM::VST4LNd8,      false, false, false, SingleSpc, 4, 8 ,true},
{ ARM::VST4LNd8Pseudo_UPD,  ARM::VST4LNd8_UPD, false, true, true,  SingleSpc, 4, 8 ,true},
{ ARM::VST4LNq16Pseudo,     ARM::VST4LNq16,     false, false, false, EvenDblSpc, 4, 4,true},
{ ARM::VST4LNq16Pseudo_UPD, ARM::VST4LNq16_UPD, false, true, true,  EvenDblSpc, 4, 4,true},
{ ARM::VST4LNq32Pseudo,     ARM::VST4LNq32,     false, false, false, EvenDblSpc, 4, 2,true},
{ ARM::VST4LNq32Pseudo_UPD, ARM::VST4LNq32_UPD, false, true, true,  EvenDblSpc, 4, 2,true},

{ ARM::VST4d16Pseudo,       ARM::VST4d16,      false, false, false, SingleSpc,  4, 4 ,true},
{ ARM::VST4d16Pseudo_UPD,   ARM::VST4d16_UPD, false, true, true,  SingleSpc,  4, 4 ,true},
{ ARM::VST4d32Pseudo,       ARM::VST4d32,      false, false, false, SingleSpc,  4, 2 ,true},
{ ARM::VST4d32Pseudo_UPD,   ARM::VST4d32_UPD, false, true, true,  SingleSpc,  4, 2 ,true},
{ ARM::VST4d8Pseudo,        ARM::VST4d8,       false, false, false, SingleSpc,  4, 8 ,true},
{ ARM::VST4d8Pseudo_UPD,    ARM::VST4d8_UPD, false, true, true,  SingleSpc,  4, 8 ,true},

{ ARM::VST4q16Pseudo_UPD,    ARM::VST4q16_UPD, false, true, true,  EvenDblSpc, 4, 4 ,true},
{ ARM::VST4q16oddPseudo,     ARM::VST4q16,     false, false, false, OddDblSpc,  4, 4 ,true},
{ ARM::VST4q16oddPseudo_UPD, ARM::VST4q16_UPD, false, true, true,  OddDblSpc,  4, 4 ,true},
{ ARM::VST4q32Pseudo_UPD,    ARM::VST4q32_UPD, false, true, true,  EvenDblSpc, 4, 2 ,true},
{ ARM::VST4q32oddPseudo,     ARM::VST4q32,     false, false, false, OddDblSpc,  4, 2 ,true},
{ ARM::VST4q32oddPseudo_UPD, ARM::VST4q32_UPD, false, true, true,  OddDblSpc,  4, 2 ,true},
{ ARM::VST4q8Pseudo_UPD,     ARM::VST4q8_UPD, false, true, true,  EvenDblSpc, 4, 8 ,true},
{ ARM::VST4q8oddPseudo,      ARM::VST4q8,      false, false, false, OddDblSpc,  4, 8 ,true},
{ ARM::VST4q8oddPseudo_UPD,  ARM::VST4q8_UPD, false, true, true,  OddDblSpc,  4, 8 ,true}
};

/// LookupNEONLdSt - Search the NEONLdStTable for information about a NEON
/// load or store pseudo instruction.
static const NEONLdStTableEntry *LookupNEONLdSt(unsigned Opcode) {
#ifndef NDEBUG
  // Make sure the table is sorted.
  static bool TableChecked = false;
  if (!TableChecked) {
    assert(std::is_sorted(std::begin(NEONLdStTable), std::end(NEONLdStTable)) &&
           "NEONLdStTable is not sorted!");
    TableChecked = true;
  }
#endif

  auto I = std::lower_bound(std::begin(NEONLdStTable),
                            std::end(NEONLdStTable), Opcode);
  if (I != std::end(NEONLdStTable) && I->PseudoOpc == Opcode)
    return I;
  return nullptr;
}

/// GetDSubRegs - Get 4 D subregisters of a Q, QQ, or QQQQ register,
/// corresponding to the specified register spacing.  Not all of the results
/// are necessarily valid, e.g., a Q register only has 2 D subregisters.
static void GetDSubRegs(unsigned Reg, NEONRegSpacing RegSpc,
                        const TargetRegisterInfo *TRI, unsigned &D0,
                        unsigned &D1, unsigned &D2, unsigned &D3) {
  if (RegSpc == SingleSpc) {
    D0 = TRI->getSubReg(Reg, ARM::dsub_0);
    D1 = TRI->getSubReg(Reg, ARM::dsub_1);
    D2 = TRI->getSubReg(Reg, ARM::dsub_2);
    D3 = TRI->getSubReg(Reg, ARM::dsub_3);
  } else if (RegSpc == EvenDblSpc) {
    D0 = TRI->getSubReg(Reg, ARM::dsub_0);
    D1 = TRI->getSubReg(Reg, ARM::dsub_2);
    D2 = TRI->getSubReg(Reg, ARM::dsub_4);
    D3 = TRI->getSubReg(Reg, ARM::dsub_6);
  } else {
    assert(RegSpc == OddDblSpc && "unknown register spacing");
    D0 = TRI->getSubReg(Reg, ARM::dsub_1);
    D1 = TRI->getSubReg(Reg, ARM::dsub_3);
    D2 = TRI->getSubReg(Reg, ARM::dsub_5);
    D3 = TRI->getSubReg(Reg, ARM::dsub_7);
  }
}

/// ExpandVLD - Translate VLD pseudo instructions with Q, QQ or QQQQ register
/// operands to real VLD instructions with D register operands.
void ARMExpandPseudo::ExpandVLD(MachineBasicBlock::iterator &MBBI) {
  MachineInstr &MI = *MBBI;
  MachineBasicBlock &MBB = *MI.getParent();

  const NEONLdStTableEntry *TableEntry = LookupNEONLdSt(MI.getOpcode());
  assert(TableEntry && TableEntry->IsLoad && "NEONLdStTable lookup failed");
  NEONRegSpacing RegSpc = (NEONRegSpacing)TableEntry->RegSpacing;
  unsigned NumRegs = TableEntry->NumRegs;

  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                                    TII->get(TableEntry->RealOpc));
  unsigned OpIdx = 0;

  bool DstIsDead = MI.getOperand(OpIdx).isDead();
  unsigned DstReg = MI.getOperand(OpIdx++).getReg();
  unsigned D0, D1, D2, D3;
  GetDSubRegs(DstReg, RegSpc, TRI, D0, D1, D2, D3);
  MIB.addReg(D0, RegState::Define | getDeadRegState(DstIsDead));
  if (NumRegs > 1 && TableEntry->copyAllListRegs)
    MIB.addReg(D1, RegState::Define | getDeadRegState(DstIsDead));
  if (NumRegs > 2 && TableEntry->copyAllListRegs)
    MIB.addReg(D2, RegState::Define | getDeadRegState(DstIsDead));
  if (NumRegs > 3 && TableEntry->copyAllListRegs)
    MIB.addReg(D3, RegState::Define | getDeadRegState(DstIsDead));

  if (TableEntry->isUpdating)
    MIB.addOperand(MI.getOperand(OpIdx++));

  // Copy the addrmode6 operands.
  MIB.addOperand(MI.getOperand(OpIdx++));
  MIB.addOperand(MI.getOperand(OpIdx++));
  // Copy the am6offset operand.
  if (TableEntry->hasWritebackOperand)
    MIB.addOperand(MI.getOperand(OpIdx++));

  // For an instruction writing double-spaced subregs, the pseudo instruction
  // has an extra operand that is a use of the super-register.  Record the
  // operand index and skip over it.
  unsigned SrcOpIdx = 0;
  if (RegSpc == EvenDblSpc || RegSpc == OddDblSpc)
    SrcOpIdx = OpIdx++;

  // Copy the predicate operands.
  MIB.addOperand(MI.getOperand(OpIdx++));
  MIB.addOperand(MI.getOperand(OpIdx++));

  // Copy the super-register source operand used for double-spaced subregs over
  // to the new instruction as an implicit operand.
  if (SrcOpIdx != 0) {
    MachineOperand MO = MI.getOperand(SrcOpIdx);
    MO.setImplicit(true);
    MIB.addOperand(MO);
  }
  // Add an implicit def for the super-register.
  MIB.addReg(DstReg, RegState::ImplicitDefine | getDeadRegState(DstIsDead));
  TransferImpOps(MI, MIB, MIB);

  // Transfer memoperands.
  MIB->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());

  MI.eraseFromParent();
}

/// ExpandVST - Translate VST pseudo instructions with Q, QQ or QQQQ register
/// operands to real VST instructions with D register operands.
void ARMExpandPseudo::ExpandVST(MachineBasicBlock::iterator &MBBI) {
  MachineInstr &MI = *MBBI;
  MachineBasicBlock &MBB = *MI.getParent();

  const NEONLdStTableEntry *TableEntry = LookupNEONLdSt(MI.getOpcode());
  assert(TableEntry && !TableEntry->IsLoad && "NEONLdStTable lookup failed");
  NEONRegSpacing RegSpc = (NEONRegSpacing)TableEntry->RegSpacing;
  unsigned NumRegs = TableEntry->NumRegs;

  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                                    TII->get(TableEntry->RealOpc));
  unsigned OpIdx = 0;
  if (TableEntry->isUpdating)
    MIB.addOperand(MI.getOperand(OpIdx++));

  // Copy the addrmode6 operands.
  MIB.addOperand(MI.getOperand(OpIdx++));
  MIB.addOperand(MI.getOperand(OpIdx++));
  // Copy the am6offset operand.
  if (TableEntry->hasWritebackOperand)
    MIB.addOperand(MI.getOperand(OpIdx++));

  bool SrcIsKill = MI.getOperand(OpIdx).isKill();
  bool SrcIsUndef = MI.getOperand(OpIdx).isUndef();
  unsigned SrcReg = MI.getOperand(OpIdx++).getReg();
  unsigned D0, D1, D2, D3;
  GetDSubRegs(SrcReg, RegSpc, TRI, D0, D1, D2, D3);
  MIB.addReg(D0, getUndefRegState(SrcIsUndef));
  if (NumRegs > 1 && TableEntry->copyAllListRegs)
    MIB.addReg(D1, getUndefRegState(SrcIsUndef));
  if (NumRegs > 2 && TableEntry->copyAllListRegs)
    MIB.addReg(D2, getUndefRegState(SrcIsUndef));
  if (NumRegs > 3 && TableEntry->copyAllListRegs)
    MIB.addReg(D3, getUndefRegState(SrcIsUndef));

  // Copy the predicate operands.
  MIB.addOperand(MI.getOperand(OpIdx++));
  MIB.addOperand(MI.getOperand(OpIdx++));

  if (SrcIsKill && !SrcIsUndef) // Add an implicit kill for the super-reg.
    MIB->addRegisterKilled(SrcReg, TRI, true);
  else if (!SrcIsUndef)
    MIB.addReg(SrcReg, RegState::Implicit); // Add implicit uses for src reg.
  TransferImpOps(MI, MIB, MIB);

  // Transfer memoperands.
  MIB->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());

  MI.eraseFromParent();
}

/// ExpandLaneOp - Translate VLD*LN and VST*LN instructions with Q, QQ or QQQQ
/// register operands to real instructions with D register operands.
void ARMExpandPseudo::ExpandLaneOp(MachineBasicBlock::iterator &MBBI) {
  MachineInstr &MI = *MBBI;
  MachineBasicBlock &MBB = *MI.getParent();

  const NEONLdStTableEntry *TableEntry = LookupNEONLdSt(MI.getOpcode());
  assert(TableEntry && "NEONLdStTable lookup failed");
  NEONRegSpacing RegSpc = (NEONRegSpacing)TableEntry->RegSpacing;
  unsigned NumRegs = TableEntry->NumRegs;
  unsigned RegElts = TableEntry->RegElts;

  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                                    TII->get(TableEntry->RealOpc));
  unsigned OpIdx = 0;
  // The lane operand is always the 3rd from last operand, before the 2
  // predicate operands.
  unsigned Lane = MI.getOperand(MI.getDesc().getNumOperands() - 3).getImm();

  // Adjust the lane and spacing as needed for Q registers.
  assert(RegSpc != OddDblSpc && "unexpected register spacing for VLD/VST-lane");
  if (RegSpc == EvenDblSpc && Lane >= RegElts) {
    RegSpc = OddDblSpc;
    Lane -= RegElts;
  }
  assert(Lane < RegElts && "out of range lane for VLD/VST-lane");

  unsigned D0 = 0, D1 = 0, D2 = 0, D3 = 0;
  unsigned DstReg = 0;
  bool DstIsDead = false;
  if (TableEntry->IsLoad) {
    DstIsDead = MI.getOperand(OpIdx).isDead();
    DstReg = MI.getOperand(OpIdx++).getReg();
    GetDSubRegs(DstReg, RegSpc, TRI, D0, D1, D2, D3);
    MIB.addReg(D0, RegState::Define | getDeadRegState(DstIsDead));
    if (NumRegs > 1)
      MIB.addReg(D1, RegState::Define | getDeadRegState(DstIsDead));
    if (NumRegs > 2)
      MIB.addReg(D2, RegState::Define | getDeadRegState(DstIsDead));
    if (NumRegs > 3)
      MIB.addReg(D3, RegState::Define | getDeadRegState(DstIsDead));
  }

  if (TableEntry->isUpdating)
    MIB.addOperand(MI.getOperand(OpIdx++));

  // Copy the addrmode6 operands.
  MIB.addOperand(MI.getOperand(OpIdx++));
  MIB.addOperand(MI.getOperand(OpIdx++));
  // Copy the am6offset operand.
  if (TableEntry->hasWritebackOperand)
    MIB.addOperand(MI.getOperand(OpIdx++));

  // Grab the super-register source.
  MachineOperand MO = MI.getOperand(OpIdx++);
  if (!TableEntry->IsLoad)
    GetDSubRegs(MO.getReg(), RegSpc, TRI, D0, D1, D2, D3);

  // Add the subregs as sources of the new instruction.
  unsigned SrcFlags = (getUndefRegState(MO.isUndef()) |
                       getKillRegState(MO.isKill()));
  MIB.addReg(D0, SrcFlags);
  if (NumRegs > 1)
    MIB.addReg(D1, SrcFlags);
  if (NumRegs > 2)
    MIB.addReg(D2, SrcFlags);
  if (NumRegs > 3)
    MIB.addReg(D3, SrcFlags);

  // Add the lane number operand.
  MIB.addImm(Lane);
  OpIdx += 1;

  // Copy the predicate operands.
  MIB.addOperand(MI.getOperand(OpIdx++));
  MIB.addOperand(MI.getOperand(OpIdx++));

  // Copy the super-register source to be an implicit source.
  MO.setImplicit(true);
  MIB.addOperand(MO);
  if (TableEntry->IsLoad)
    // Add an implicit def for the super-register.
    MIB.addReg(DstReg, RegState::ImplicitDefine | getDeadRegState(DstIsDead));
  TransferImpOps(MI, MIB, MIB);
  // Transfer memoperands.
  MIB->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
  MI.eraseFromParent();
}

/// ExpandVTBL - Translate VTBL and VTBX pseudo instructions with Q or QQ
/// register operands to real instructions with D register operands.
void ARMExpandPseudo::ExpandVTBL(MachineBasicBlock::iterator &MBBI,
                                 unsigned Opc, bool IsExt) {
  MachineInstr &MI = *MBBI;
  MachineBasicBlock &MBB = *MI.getParent();

  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(Opc));
  unsigned OpIdx = 0;

  // Transfer the destination register operand.
  MIB.addOperand(MI.getOperand(OpIdx++));
  if (IsExt)
    MIB.addOperand(MI.getOperand(OpIdx++));

  bool SrcIsKill = MI.getOperand(OpIdx).isKill();
  unsigned SrcReg = MI.getOperand(OpIdx++).getReg();
  unsigned D0, D1, D2, D3;
  GetDSubRegs(SrcReg, SingleSpc, TRI, D0, D1, D2, D3);
  MIB.addReg(D0);

  // Copy the other source register operand.
  MIB.addOperand(MI.getOperand(OpIdx++));

  // Copy the predicate operands.
  MIB.addOperand(MI.getOperand(OpIdx++));
  MIB.addOperand(MI.getOperand(OpIdx++));

  // Add an implicit kill and use for the super-reg.
  MIB.addReg(SrcReg, RegState::Implicit | getKillRegState(SrcIsKill));
  TransferImpOps(MI, MIB, MIB);
  MI.eraseFromParent();
}

static bool IsAnAddressOperand(const MachineOperand &MO) {
  // This check is overly conservative.  Unless we are certain that the machine
  // operand is not a symbol reference, we return that it is a symbol reference.
  // This is important as the load pair may not be split up Windows.
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
  case MachineOperand::MO_Immediate:
  case MachineOperand::MO_CImmediate:
  case MachineOperand::MO_FPImmediate:
    return false;
  case MachineOperand::MO_MachineBasicBlock:
    return true;
  case MachineOperand::MO_FrameIndex:
    return false;
  case MachineOperand::MO_ConstantPoolIndex:
  case MachineOperand::MO_TargetIndex:
  case MachineOperand::MO_JumpTableIndex:
  case MachineOperand::MO_ExternalSymbol:
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_BlockAddress:
    return true;
  case MachineOperand::MO_RegisterMask:
  case MachineOperand::MO_RegisterLiveOut:
    return false;
  case MachineOperand::MO_Metadata:
  case MachineOperand::MO_MCSymbol:
    return true;
  case MachineOperand::MO_CFIIndex:
    return false;
  case MachineOperand::MO_IntrinsicID:
    llvm_unreachable("should not exist post-isel");
  }
  llvm_unreachable("unhandled machine operand type");
}

void ARMExpandPseudo::ExpandMOV32BitImm(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator &MBBI) {
  MachineInstr &MI = *MBBI;
  unsigned Opcode = MI.getOpcode();
  unsigned PredReg = 0;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);
  unsigned DstReg = MI.getOperand(0).getReg();
  bool DstIsDead = MI.getOperand(0).isDead();
  bool isCC = Opcode == ARM::MOVCCi32imm || Opcode == ARM::t2MOVCCi32imm;
  const MachineOperand &MO = MI.getOperand(isCC ? 2 : 1);
  bool RequiresBundling = STI->isTargetWindows() && IsAnAddressOperand(MO);
  MachineInstrBuilder LO16, HI16;

  if (!STI->hasV6T2Ops() &&
      (Opcode == ARM::MOVi32imm || Opcode == ARM::MOVCCi32imm)) {
    // FIXME Windows CE supports older ARM CPUs
    assert(!STI->isTargetWindows() && "Windows on ARM requires ARMv7+");

    // Expand into a movi + orr.
    LO16 = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM::MOVi), DstReg);
    HI16 = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM::ORRri))
      .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead))
      .addReg(DstReg);

    assert (MO.isImm() && "MOVi32imm w/ non-immediate source operand!");
    unsigned ImmVal = (unsigned)MO.getImm();
    unsigned SOImmValV1 = ARM_AM::getSOImmTwoPartFirst(ImmVal);
    unsigned SOImmValV2 = ARM_AM::getSOImmTwoPartSecond(ImmVal);
    LO16 = LO16.addImm(SOImmValV1);
    HI16 = HI16.addImm(SOImmValV2);
    LO16->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
    HI16->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
    LO16.addImm(Pred).addReg(PredReg).addReg(0);
    HI16.addImm(Pred).addReg(PredReg).addReg(0);
    TransferImpOps(MI, LO16, HI16);
    MI.eraseFromParent();
    return;
  }

  unsigned LO16Opc = 0;
  unsigned HI16Opc = 0;
  if (Opcode == ARM::t2MOVi32imm || Opcode == ARM::t2MOVCCi32imm) {
    LO16Opc = ARM::t2MOVi16;
    HI16Opc = ARM::t2MOVTi16;
  } else {
    LO16Opc = ARM::MOVi16;
    HI16Opc = ARM::MOVTi16;
  }

  LO16 = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(LO16Opc), DstReg);
  HI16 = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(HI16Opc))
    .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead))
    .addReg(DstReg);

  switch (MO.getType()) {
  case MachineOperand::MO_Immediate: {
    unsigned Imm = MO.getImm();
    unsigned Lo16 = Imm & 0xffff;
    unsigned Hi16 = (Imm >> 16) & 0xffff;
    LO16 = LO16.addImm(Lo16);
    HI16 = HI16.addImm(Hi16);
    break;
  }
  case MachineOperand::MO_ExternalSymbol: {
    const char *ES = MO.getSymbolName();
    unsigned TF = MO.getTargetFlags();
    LO16 = LO16.addExternalSymbol(ES, TF | ARMII::MO_LO16);
    HI16 = HI16.addExternalSymbol(ES, TF | ARMII::MO_HI16);
    break;
  }
  default: {
    const GlobalValue *GV = MO.getGlobal();
    unsigned TF = MO.getTargetFlags();
    LO16 = LO16.addGlobalAddress(GV, MO.getOffset(), TF | ARMII::MO_LO16);
    HI16 = HI16.addGlobalAddress(GV, MO.getOffset(), TF | ARMII::MO_HI16);
    break;
  }
  }

  LO16->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
  HI16->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
  LO16.addImm(Pred).addReg(PredReg);
  HI16.addImm(Pred).addReg(PredReg);

  if (RequiresBundling)
    finalizeBundle(MBB, LO16->getIterator(), MBBI->getIterator());

  TransferImpOps(MI, LO16, HI16);
  MI.eraseFromParent();
}

static void addPostLoopLiveIns(MachineBasicBlock *MBB, LivePhysRegs &LiveRegs) {
  for (auto I = LiveRegs.begin(); I != LiveRegs.end(); ++I)
    MBB->addLiveIn(*I);
}

/// Expand a CMP_SWAP pseudo-inst to an ldrex/strex loop as simply as
/// possible. This only gets used at -O0 so we don't care about efficiency of the
/// generated code.
bool ARMExpandPseudo::ExpandCMP_SWAP(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     unsigned LdrexOp, unsigned StrexOp,
                                     unsigned UxtOp,
                                     MachineBasicBlock::iterator &NextMBBI) {
  bool IsThumb = STI->isThumb();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();
  MachineOperand &Dest = MI.getOperand(0);
  unsigned StatusReg = MI.getOperand(1).getReg();
  MachineOperand &Addr = MI.getOperand(2);
  MachineOperand &Desired = MI.getOperand(3);
  MachineOperand &New = MI.getOperand(4);

  LivePhysRegs LiveRegs(&TII->getRegisterInfo());
  LiveRegs.addLiveOuts(MBB);
  for (auto I = std::prev(MBB.end()); I != MBBI; --I)
    LiveRegs.stepBackward(*I);

  MachineFunction *MF = MBB.getParent();
  auto LoadCmpBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto StoreBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto DoneBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  MF->insert(++MBB.getIterator(), LoadCmpBB);
  MF->insert(++LoadCmpBB->getIterator(), StoreBB);
  MF->insert(++StoreBB->getIterator(), DoneBB);

  if (UxtOp) {
    MachineInstrBuilder MIB =
        BuildMI(MBB, MBBI, DL, TII->get(UxtOp), Desired.getReg())
            .addReg(Desired.getReg(), RegState::Kill);
    if (!IsThumb)
      MIB.addImm(0);
    AddDefaultPred(MIB);
  }

  // .Lloadcmp:
  //     ldrex rDest, [rAddr]
  //     cmp rDest, rDesired
  //     bne .Ldone
  LoadCmpBB->addLiveIn(Addr.getReg());
  LoadCmpBB->addLiveIn(Dest.getReg());
  LoadCmpBB->addLiveIn(Desired.getReg());
  addPostLoopLiveIns(LoadCmpBB, LiveRegs);

  MachineInstrBuilder MIB;
  MIB = BuildMI(LoadCmpBB, DL, TII->get(LdrexOp), Dest.getReg());
  MIB.addReg(Addr.getReg());
  if (LdrexOp == ARM::t2LDREX)
    MIB.addImm(0); // a 32-bit Thumb ldrex (only) allows an offset.
  AddDefaultPred(MIB);

  unsigned CMPrr = IsThumb ? ARM::tCMPhir : ARM::CMPrr;
  AddDefaultPred(BuildMI(LoadCmpBB, DL, TII->get(CMPrr))
                     .addReg(Dest.getReg(), getKillRegState(Dest.isDead()))
                     .addOperand(Desired));
  unsigned Bcc = IsThumb ? ARM::tBcc : ARM::Bcc;
  BuildMI(LoadCmpBB, DL, TII->get(Bcc))
      .addMBB(DoneBB)
      .addImm(ARMCC::NE)
      .addReg(ARM::CPSR, RegState::Kill);
  LoadCmpBB->addSuccessor(DoneBB);
  LoadCmpBB->addSuccessor(StoreBB);

  // .Lstore:
  //     strex rStatus, rNew, [rAddr]
  //     cmp rStatus, #0
  //     bne .Lloadcmp
  StoreBB->addLiveIn(Addr.getReg());
  StoreBB->addLiveIn(New.getReg());
  addPostLoopLiveIns(StoreBB, LiveRegs);


  MIB = BuildMI(StoreBB, DL, TII->get(StrexOp), StatusReg);
  MIB.addOperand(New);
  MIB.addOperand(Addr);
  if (StrexOp == ARM::t2STREX)
    MIB.addImm(0); // a 32-bit Thumb strex (only) allows an offset.
  AddDefaultPred(MIB);

  unsigned CMPri = IsThumb ? ARM::t2CMPri : ARM::CMPri;
  AddDefaultPred(BuildMI(StoreBB, DL, TII->get(CMPri))
                     .addReg(StatusReg, RegState::Kill)
                     .addImm(0));
  BuildMI(StoreBB, DL, TII->get(Bcc))
      .addMBB(LoadCmpBB)
      .addImm(ARMCC::NE)
      .addReg(ARM::CPSR, RegState::Kill);
  StoreBB->addSuccessor(LoadCmpBB);
  StoreBB->addSuccessor(DoneBB);

  DoneBB->splice(DoneBB->end(), &MBB, MI, MBB.end());
  DoneBB->transferSuccessors(&MBB);
  addPostLoopLiveIns(DoneBB, LiveRegs);

  MBB.addSuccessor(LoadCmpBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();
  return true;
}

/// ARM's ldrexd/strexd take a consecutive register pair (represented as a
/// single GPRPair register), Thumb's take two separate registers so we need to
/// extract the subregs from the pair.
static void addExclusiveRegPair(MachineInstrBuilder &MIB, MachineOperand &Reg,
                                unsigned Flags, bool IsThumb,
                                const TargetRegisterInfo *TRI) {
  if (IsThumb) {
    unsigned RegLo = TRI->getSubReg(Reg.getReg(), ARM::gsub_0);
    unsigned RegHi = TRI->getSubReg(Reg.getReg(), ARM::gsub_1);
    MIB.addReg(RegLo, Flags | getKillRegState(Reg.isDead()));
    MIB.addReg(RegHi, Flags | getKillRegState(Reg.isDead()));
  } else
    MIB.addReg(Reg.getReg(), Flags | getKillRegState(Reg.isDead()));
}

/// Expand a 64-bit CMP_SWAP to an ldrexd/strexd loop.
bool ARMExpandPseudo::ExpandCMP_SWAP_64(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        MachineBasicBlock::iterator &NextMBBI) {
  bool IsThumb = STI->isThumb();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();
  MachineOperand &Dest = MI.getOperand(0);
  unsigned StatusReg = MI.getOperand(1).getReg();
  MachineOperand &Addr = MI.getOperand(2);
  MachineOperand &Desired = MI.getOperand(3);
  MachineOperand &New = MI.getOperand(4);

  unsigned DestLo = TRI->getSubReg(Dest.getReg(), ARM::gsub_0);
  unsigned DestHi = TRI->getSubReg(Dest.getReg(), ARM::gsub_1);
  unsigned DesiredLo = TRI->getSubReg(Desired.getReg(), ARM::gsub_0);
  unsigned DesiredHi = TRI->getSubReg(Desired.getReg(), ARM::gsub_1);

  LivePhysRegs LiveRegs(&TII->getRegisterInfo());
  LiveRegs.addLiveOuts(MBB);
  for (auto I = std::prev(MBB.end()); I != MBBI; --I)
    LiveRegs.stepBackward(*I);

  MachineFunction *MF = MBB.getParent();
  auto LoadCmpBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto StoreBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  auto DoneBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  MF->insert(++MBB.getIterator(), LoadCmpBB);
  MF->insert(++LoadCmpBB->getIterator(), StoreBB);
  MF->insert(++StoreBB->getIterator(), DoneBB);

  // .Lloadcmp:
  //     ldrexd rDestLo, rDestHi, [rAddr]
  //     cmp rDestLo, rDesiredLo
  //     sbcs rStatus<dead>, rDestHi, rDesiredHi
  //     bne .Ldone
  LoadCmpBB->addLiveIn(Addr.getReg());
  LoadCmpBB->addLiveIn(Dest.getReg());
  LoadCmpBB->addLiveIn(Desired.getReg());
  addPostLoopLiveIns(LoadCmpBB, LiveRegs);

  unsigned LDREXD = IsThumb ? ARM::t2LDREXD : ARM::LDREXD;
  MachineInstrBuilder MIB;
  MIB = BuildMI(LoadCmpBB, DL, TII->get(LDREXD));
  addExclusiveRegPair(MIB, Dest, RegState::Define, IsThumb, TRI);
  MIB.addReg(Addr.getReg());
  AddDefaultPred(MIB);

  unsigned CMPrr = IsThumb ? ARM::tCMPhir : ARM::CMPrr;
  AddDefaultPred(BuildMI(LoadCmpBB, DL, TII->get(CMPrr))
                     .addReg(DestLo, getKillRegState(Dest.isDead()))
                     .addReg(DesiredLo, getKillRegState(Desired.isDead())));

  unsigned SBCrr = IsThumb ? ARM::t2SBCrr : ARM::SBCrr;
  MIB = BuildMI(LoadCmpBB, DL, TII->get(SBCrr))
            .addReg(StatusReg, RegState::Define | RegState::Dead)
            .addReg(DestHi, getKillRegState(Dest.isDead()))
            .addReg(DesiredHi, getKillRegState(Desired.isDead()));
  AddDefaultPred(MIB);
  MIB.addReg(ARM::CPSR, RegState::Kill);

  unsigned Bcc = IsThumb ? ARM::tBcc : ARM::Bcc;
  BuildMI(LoadCmpBB, DL, TII->get(Bcc))
      .addMBB(DoneBB)
      .addImm(ARMCC::NE)
      .addReg(ARM::CPSR, RegState::Kill);
  LoadCmpBB->addSuccessor(DoneBB);
  LoadCmpBB->addSuccessor(StoreBB);

  // .Lstore:
  //     strexd rStatus, rNewLo, rNewHi, [rAddr]
  //     cmp rStatus, #0
  //     bne .Lloadcmp
  StoreBB->addLiveIn(Addr.getReg());
  StoreBB->addLiveIn(New.getReg());
  addPostLoopLiveIns(StoreBB, LiveRegs);

  unsigned STREXD = IsThumb ? ARM::t2STREXD : ARM::STREXD;
  MIB = BuildMI(StoreBB, DL, TII->get(STREXD), StatusReg);
  addExclusiveRegPair(MIB, New, 0, IsThumb, TRI);
  MIB.addOperand(Addr);
  AddDefaultPred(MIB);

  unsigned CMPri = IsThumb ? ARM::t2CMPri : ARM::CMPri;
  AddDefaultPred(BuildMI(StoreBB, DL, TII->get(CMPri))
                     .addReg(StatusReg, RegState::Kill)
                     .addImm(0));
  BuildMI(StoreBB, DL, TII->get(Bcc))
      .addMBB(LoadCmpBB)
      .addImm(ARMCC::NE)
      .addReg(ARM::CPSR, RegState::Kill);
  StoreBB->addSuccessor(LoadCmpBB);
  StoreBB->addSuccessor(DoneBB);

  DoneBB->splice(DoneBB->end(), &MBB, MI, MBB.end());
  DoneBB->transferSuccessors(&MBB);
  addPostLoopLiveIns(DoneBB, LiveRegs);

  MBB.addSuccessor(LoadCmpBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();
  return true;
}


bool ARMExpandPseudo::ExpandMI(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI,
                               MachineBasicBlock::iterator &NextMBBI) {
  MachineInstr &MI = *MBBI;
  unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
    default:
      return false;

    case ARM::TCRETURNdi:
    case ARM::TCRETURNri: {
      MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
      assert(MBBI->isReturn() &&
             "Can only insert epilog into returning blocks");
      unsigned RetOpcode = MBBI->getOpcode();
      DebugLoc dl = MBBI->getDebugLoc();
      const ARMBaseInstrInfo &TII = *static_cast<const ARMBaseInstrInfo *>(
          MBB.getParent()->getSubtarget().getInstrInfo());

      // Tail call return: adjust the stack pointer and jump to callee.
      MBBI = MBB.getLastNonDebugInstr();
      MachineOperand &JumpTarget = MBBI->getOperand(0);

      // Jump to label or value in register.
      if (RetOpcode == ARM::TCRETURNdi) {
        unsigned TCOpcode =
            STI->isThumb()
                ? (STI->isTargetMachO() ? ARM::tTAILJMPd : ARM::tTAILJMPdND)
                : ARM::TAILJMPd;
        MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII.get(TCOpcode));
        if (JumpTarget.isGlobal())
          MIB.addGlobalAddress(JumpTarget.getGlobal(), JumpTarget.getOffset(),
                               JumpTarget.getTargetFlags());
        else {
          assert(JumpTarget.isSymbol());
          MIB.addExternalSymbol(JumpTarget.getSymbolName(),
                                JumpTarget.getTargetFlags());
        }

        // Add the default predicate in Thumb mode.
        if (STI->isThumb())
          MIB.addImm(ARMCC::AL).addReg(0);
      } else if (RetOpcode == ARM::TCRETURNri) {
        BuildMI(MBB, MBBI, dl,
                TII.get(STI->isThumb() ? ARM::tTAILJMPr : ARM::TAILJMPr))
            .addReg(JumpTarget.getReg(), RegState::Kill);
      }

      auto NewMI = std::prev(MBBI);
      for (unsigned i = 1, e = MBBI->getNumOperands(); i != e; ++i)
        NewMI->addOperand(MBBI->getOperand(i));

      // Delete the pseudo instruction TCRETURN.
      MBB.erase(MBBI);
      MBBI = NewMI;
      return true;
    }
    case ARM::VMOVScc:
    case ARM::VMOVDcc: {
      unsigned newOpc = Opcode == ARM::VMOVScc ? ARM::VMOVS : ARM::VMOVD;
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(newOpc),
              MI.getOperand(1).getReg())
        .addOperand(MI.getOperand(2))
        .addImm(MI.getOperand(3).getImm()) // 'pred'
        .addOperand(MI.getOperand(4));

      MI.eraseFromParent();
      return true;
    }
    case ARM::t2MOVCCr:
    case ARM::MOVCCr: {
      unsigned Opc = AFI->isThumbFunction() ? ARM::t2MOVr : ARM::MOVr;
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(Opc),
              MI.getOperand(1).getReg())
        .addOperand(MI.getOperand(2))
        .addImm(MI.getOperand(3).getImm()) // 'pred'
        .addOperand(MI.getOperand(4))
        .addReg(0); // 's' bit

      MI.eraseFromParent();
      return true;
    }
    case ARM::MOVCCsi: {
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM::MOVsi),
              (MI.getOperand(1).getReg()))
        .addOperand(MI.getOperand(2))
        .addImm(MI.getOperand(3).getImm())
        .addImm(MI.getOperand(4).getImm()) // 'pred'
        .addOperand(MI.getOperand(5))
        .addReg(0); // 's' bit

      MI.eraseFromParent();
      return true;
    }
    case ARM::MOVCCsr: {
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM::MOVsr),
              (MI.getOperand(1).getReg()))
        .addOperand(MI.getOperand(2))
        .addOperand(MI.getOperand(3))
        .addImm(MI.getOperand(4).getImm())
        .addImm(MI.getOperand(5).getImm()) // 'pred'
        .addOperand(MI.getOperand(6))
        .addReg(0); // 's' bit

      MI.eraseFromParent();
      return true;
    }
    case ARM::t2MOVCCi16:
    case ARM::MOVCCi16: {
      unsigned NewOpc = AFI->isThumbFunction() ? ARM::t2MOVi16 : ARM::MOVi16;
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(NewOpc),
              MI.getOperand(1).getReg())
        .addImm(MI.getOperand(2).getImm())
        .addImm(MI.getOperand(3).getImm()) // 'pred'
        .addOperand(MI.getOperand(4));
      MI.eraseFromParent();
      return true;
    }
    case ARM::t2MOVCCi:
    case ARM::MOVCCi: {
      unsigned Opc = AFI->isThumbFunction() ? ARM::t2MOVi : ARM::MOVi;
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(Opc),
              MI.getOperand(1).getReg())
        .addImm(MI.getOperand(2).getImm())
        .addImm(MI.getOperand(3).getImm()) // 'pred'
        .addOperand(MI.getOperand(4))
        .addReg(0); // 's' bit

      MI.eraseFromParent();
      return true;
    }
    case ARM::t2MVNCCi:
    case ARM::MVNCCi: {
      unsigned Opc = AFI->isThumbFunction() ? ARM::t2MVNi : ARM::MVNi;
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(Opc),
              MI.getOperand(1).getReg())
        .addImm(MI.getOperand(2).getImm())
        .addImm(MI.getOperand(3).getImm()) // 'pred'
        .addOperand(MI.getOperand(4))
        .addReg(0); // 's' bit

      MI.eraseFromParent();
      return true;
    }
    case ARM::t2MOVCClsl:
    case ARM::t2MOVCClsr:
    case ARM::t2MOVCCasr:
    case ARM::t2MOVCCror: {
      unsigned NewOpc;
      switch (Opcode) {
      case ARM::t2MOVCClsl: NewOpc = ARM::t2LSLri; break;
      case ARM::t2MOVCClsr: NewOpc = ARM::t2LSRri; break;
      case ARM::t2MOVCCasr: NewOpc = ARM::t2ASRri; break;
      case ARM::t2MOVCCror: NewOpc = ARM::t2RORri; break;
      default: llvm_unreachable("unexpeced conditional move");
      }
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(NewOpc),
              MI.getOperand(1).getReg())
        .addOperand(MI.getOperand(2))
        .addImm(MI.getOperand(3).getImm())
        .addImm(MI.getOperand(4).getImm()) // 'pred'
        .addOperand(MI.getOperand(5))
        .addReg(0); // 's' bit
      MI.eraseFromParent();
      return true;
    }
    case ARM::Int_eh_sjlj_dispatchsetup: {
      MachineFunction &MF = *MI.getParent()->getParent();
      const ARMBaseInstrInfo *AII =
        static_cast<const ARMBaseInstrInfo*>(TII);
      const ARMBaseRegisterInfo &RI = AII->getRegisterInfo();
      // For functions using a base pointer, we rematerialize it (via the frame
      // pointer) here since eh.sjlj.setjmp and eh.sjlj.longjmp don't do it
      // for us. Otherwise, expand to nothing.
      if (RI.hasBasePointer(MF)) {
        int32_t NumBytes = AFI->getFramePtrSpillOffset();
        unsigned FramePtr = RI.getFrameRegister(MF);
        assert(MF.getSubtarget().getFrameLowering()->hasFP(MF) &&
               "base pointer without frame pointer?");

        if (AFI->isThumb2Function()) {
          emitT2RegPlusImmediate(MBB, MBBI, MI.getDebugLoc(), ARM::R6,
                                 FramePtr, -NumBytes, ARMCC::AL, 0, *TII);
        } else if (AFI->isThumbFunction()) {
          emitThumbRegPlusImmediate(MBB, MBBI, MI.getDebugLoc(), ARM::R6,
                                    FramePtr, -NumBytes, *TII, RI);
        } else {
          emitARMRegPlusImmediate(MBB, MBBI, MI.getDebugLoc(), ARM::R6,
                                  FramePtr, -NumBytes, ARMCC::AL, 0,
                                  *TII);
        }
        // If there's dynamic realignment, adjust for it.
        if (RI.needsStackRealignment(MF)) {
          MachineFrameInfo &MFI = MF.getFrameInfo();
          unsigned MaxAlign = MFI.getMaxAlignment();
          assert (!AFI->isThumb1OnlyFunction());
          // Emit bic r6, r6, MaxAlign
          assert(MaxAlign <= 256 && "The BIC instruction cannot encode "
                                    "immediates larger than 256 with all lower "
                                    "bits set.");
          unsigned bicOpc = AFI->isThumbFunction() ?
            ARM::t2BICri : ARM::BICri;
          AddDefaultCC(AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(),
                                              TII->get(bicOpc), ARM::R6)
                                      .addReg(ARM::R6, RegState::Kill)
                                      .addImm(MaxAlign-1)));
        }

      }
      MI.eraseFromParent();
      return true;
    }

    case ARM::MOVsrl_flag:
    case ARM::MOVsra_flag: {
      // These are just fancy MOVs instructions.
      AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM::MOVsi),
                             MI.getOperand(0).getReg())
                     .addOperand(MI.getOperand(1))
                     .addImm(ARM_AM::getSORegOpc((Opcode == ARM::MOVsrl_flag ?
                                                  ARM_AM::lsr : ARM_AM::asr),
                                                 1)))
        .addReg(ARM::CPSR, RegState::Define);
      MI.eraseFromParent();
      return true;
    }
    case ARM::RRX: {
      // This encodes as "MOVs Rd, Rm, rrx
      MachineInstrBuilder MIB =
        AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(),TII->get(ARM::MOVsi),
                               MI.getOperand(0).getReg())
                       .addOperand(MI.getOperand(1))
                       .addImm(ARM_AM::getSORegOpc(ARM_AM::rrx, 0)))
        .addReg(0);
      TransferImpOps(MI, MIB, MIB);
      MI.eraseFromParent();
      return true;
    }
    case ARM::tTPsoft:
    case ARM::TPsoft: {
      MachineInstrBuilder MIB;
      if (Opcode == ARM::tTPsoft)
        MIB = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                      TII->get( ARM::tBL))
              .addImm((unsigned)ARMCC::AL).addReg(0)
              .addExternalSymbol("__aeabi_read_tp", 0);
      else
        MIB = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                      TII->get( ARM::BL))
              .addExternalSymbol("__aeabi_read_tp", 0);

      MIB->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
      TransferImpOps(MI, MIB, MIB);
      MI.eraseFromParent();
      return true;
    }
    case ARM::tLDRpci_pic:
    case ARM::t2LDRpci_pic: {
      unsigned NewLdOpc = (Opcode == ARM::tLDRpci_pic)
        ? ARM::tLDRpci : ARM::t2LDRpci;
      unsigned DstReg = MI.getOperand(0).getReg();
      bool DstIsDead = MI.getOperand(0).isDead();
      MachineInstrBuilder MIB1 =
        AddDefaultPred(BuildMI(MBB, MBBI, MI.getDebugLoc(),
                               TII->get(NewLdOpc), DstReg)
                       .addOperand(MI.getOperand(1)));
      MIB1->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
      MachineInstrBuilder MIB2 = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                                         TII->get(ARM::tPICADD))
        .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead))
        .addReg(DstReg)
        .addOperand(MI.getOperand(2));
      TransferImpOps(MI, MIB1, MIB2);
      MI.eraseFromParent();
      return true;
    }

    case ARM::LDRLIT_ga_abs:
    case ARM::LDRLIT_ga_pcrel:
    case ARM::LDRLIT_ga_pcrel_ldr:
    case ARM::tLDRLIT_ga_abs:
    case ARM::tLDRLIT_ga_pcrel: {
      unsigned DstReg = MI.getOperand(0).getReg();
      bool DstIsDead = MI.getOperand(0).isDead();
      const MachineOperand &MO1 = MI.getOperand(1);
      const GlobalValue *GV = MO1.getGlobal();
      bool IsARM =
          Opcode != ARM::tLDRLIT_ga_pcrel && Opcode != ARM::tLDRLIT_ga_abs;
      bool IsPIC =
          Opcode != ARM::LDRLIT_ga_abs && Opcode != ARM::tLDRLIT_ga_abs;
      unsigned LDRLITOpc = IsARM ? ARM::LDRi12 : ARM::tLDRpci;
      unsigned PICAddOpc =
          IsARM
              ? (Opcode == ARM::LDRLIT_ga_pcrel_ldr ? ARM::PICLDR : ARM::PICADD)
              : ARM::tPICADD;

      // We need a new const-pool entry to load from.
      MachineConstantPool *MCP = MBB.getParent()->getConstantPool();
      unsigned ARMPCLabelIndex = 0;
      MachineConstantPoolValue *CPV;

      if (IsPIC) {
        unsigned PCAdj = IsARM ? 8 : 4;
        ARMPCLabelIndex = AFI->createPICLabelUId();
        CPV = ARMConstantPoolConstant::Create(GV, ARMPCLabelIndex,
                                              ARMCP::CPValue, PCAdj);
      } else
        CPV = ARMConstantPoolConstant::Create(GV, ARMCP::no_modifier);

      MachineInstrBuilder MIB =
          BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(LDRLITOpc), DstReg)
            .addConstantPoolIndex(MCP->getConstantPoolIndex(CPV, 4));
      if (IsARM)
        MIB.addImm(0);
      AddDefaultPred(MIB);

      if (IsPIC) {
        MachineInstrBuilder MIB =
          BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(PICAddOpc))
            .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead))
            .addReg(DstReg)
            .addImm(ARMPCLabelIndex);

        if (IsARM)
          AddDefaultPred(MIB);
      }

      MI.eraseFromParent();
      return true;
    }
    case ARM::MOV_ga_pcrel:
    case ARM::MOV_ga_pcrel_ldr:
    case ARM::t2MOV_ga_pcrel: {
      // Expand into movw + movw. Also "add pc" / ldr [pc] in PIC mode.
      unsigned LabelId = AFI->createPICLabelUId();
      unsigned DstReg = MI.getOperand(0).getReg();
      bool DstIsDead = MI.getOperand(0).isDead();
      const MachineOperand &MO1 = MI.getOperand(1);
      const GlobalValue *GV = MO1.getGlobal();
      unsigned TF = MO1.getTargetFlags();
      bool isARM = Opcode != ARM::t2MOV_ga_pcrel;
      unsigned LO16Opc = isARM ? ARM::MOVi16_ga_pcrel : ARM::t2MOVi16_ga_pcrel;
      unsigned HI16Opc = isARM ? ARM::MOVTi16_ga_pcrel :ARM::t2MOVTi16_ga_pcrel;
      unsigned LO16TF = TF | ARMII::MO_LO16;
      unsigned HI16TF = TF | ARMII::MO_HI16;
      unsigned PICAddOpc = isARM
        ? (Opcode == ARM::MOV_ga_pcrel_ldr ? ARM::PICLDR : ARM::PICADD)
        : ARM::tPICADD;
      MachineInstrBuilder MIB1 = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                                         TII->get(LO16Opc), DstReg)
        .addGlobalAddress(GV, MO1.getOffset(), TF | LO16TF)
        .addImm(LabelId);

      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(HI16Opc), DstReg)
        .addReg(DstReg)
        .addGlobalAddress(GV, MO1.getOffset(), TF | HI16TF)
        .addImm(LabelId);

      MachineInstrBuilder MIB3 = BuildMI(MBB, MBBI, MI.getDebugLoc(),
                                         TII->get(PICAddOpc))
        .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead))
        .addReg(DstReg).addImm(LabelId);
      if (isARM) {
        AddDefaultPred(MIB3);
        if (Opcode == ARM::MOV_ga_pcrel_ldr)
          MIB3->setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
      }
      TransferImpOps(MI, MIB1, MIB3);
      MI.eraseFromParent();
      return true;
    }

    case ARM::MOVi32imm:
    case ARM::MOVCCi32imm:
    case ARM::t2MOVi32imm:
    case ARM::t2MOVCCi32imm:
      ExpandMOV32BitImm(MBB, MBBI);
      return true;

    case ARM::SUBS_PC_LR: {
      MachineInstrBuilder MIB =
          BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM::SUBri), ARM::PC)
              .addReg(ARM::LR)
              .addOperand(MI.getOperand(0))
              .addOperand(MI.getOperand(1))
              .addOperand(MI.getOperand(2))
              .addReg(ARM::CPSR, RegState::Undef);
      TransferImpOps(MI, MIB, MIB);
      MI.eraseFromParent();
      return true;
    }
    case ARM::VLDMQIA: {
      unsigned NewOpc = ARM::VLDMDIA;
      MachineInstrBuilder MIB =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(NewOpc));
      unsigned OpIdx = 0;

      // Grab the Q register destination.
      bool DstIsDead = MI.getOperand(OpIdx).isDead();
      unsigned DstReg = MI.getOperand(OpIdx++).getReg();

      // Copy the source register.
      MIB.addOperand(MI.getOperand(OpIdx++));

      // Copy the predicate operands.
      MIB.addOperand(MI.getOperand(OpIdx++));
      MIB.addOperand(MI.getOperand(OpIdx++));

      // Add the destination operands (D subregs).
      unsigned D0 = TRI->getSubReg(DstReg, ARM::dsub_0);
      unsigned D1 = TRI->getSubReg(DstReg, ARM::dsub_1);
      MIB.addReg(D0, RegState::Define | getDeadRegState(DstIsDead))
        .addReg(D1, RegState::Define | getDeadRegState(DstIsDead));

      // Add an implicit def for the super-register.
      MIB.addReg(DstReg, RegState::ImplicitDefine | getDeadRegState(DstIsDead));
      TransferImpOps(MI, MIB, MIB);
      MIB.setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
      MI.eraseFromParent();
      return true;
    }

    case ARM::VSTMQIA: {
      unsigned NewOpc = ARM::VSTMDIA;
      MachineInstrBuilder MIB =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(NewOpc));
      unsigned OpIdx = 0;

      // Grab the Q register source.
      bool SrcIsKill = MI.getOperand(OpIdx).isKill();
      unsigned SrcReg = MI.getOperand(OpIdx++).getReg();

      // Copy the destination register.
      MIB.addOperand(MI.getOperand(OpIdx++));

      // Copy the predicate operands.
      MIB.addOperand(MI.getOperand(OpIdx++));
      MIB.addOperand(MI.getOperand(OpIdx++));

      // Add the source operands (D subregs).
      unsigned D0 = TRI->getSubReg(SrcReg, ARM::dsub_0);
      unsigned D1 = TRI->getSubReg(SrcReg, ARM::dsub_1);
      MIB.addReg(D0, SrcIsKill ? RegState::Kill : 0)
         .addReg(D1, SrcIsKill ? RegState::Kill : 0);

      if (SrcIsKill)      // Add an implicit kill for the Q register.
        MIB->addRegisterKilled(SrcReg, TRI, true);

      TransferImpOps(MI, MIB, MIB);
      MIB.setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
      MI.eraseFromParent();
      return true;
    }

    case ARM::VLD2q8Pseudo:
    case ARM::VLD2q16Pseudo:
    case ARM::VLD2q32Pseudo:
    case ARM::VLD2q8PseudoWB_fixed:
    case ARM::VLD2q16PseudoWB_fixed:
    case ARM::VLD2q32PseudoWB_fixed:
    case ARM::VLD2q8PseudoWB_register:
    case ARM::VLD2q16PseudoWB_register:
    case ARM::VLD2q32PseudoWB_register:
    case ARM::VLD3d8Pseudo:
    case ARM::VLD3d16Pseudo:
    case ARM::VLD3d32Pseudo:
    case ARM::VLD1d64TPseudo:
    case ARM::VLD1d64TPseudoWB_fixed:
    case ARM::VLD3d8Pseudo_UPD:
    case ARM::VLD3d16Pseudo_UPD:
    case ARM::VLD3d32Pseudo_UPD:
    case ARM::VLD3q8Pseudo_UPD:
    case ARM::VLD3q16Pseudo_UPD:
    case ARM::VLD3q32Pseudo_UPD:
    case ARM::VLD3q8oddPseudo:
    case ARM::VLD3q16oddPseudo:
    case ARM::VLD3q32oddPseudo:
    case ARM::VLD3q8oddPseudo_UPD:
    case ARM::VLD3q16oddPseudo_UPD:
    case ARM::VLD3q32oddPseudo_UPD:
    case ARM::VLD4d8Pseudo:
    case ARM::VLD4d16Pseudo:
    case ARM::VLD4d32Pseudo:
    case ARM::VLD1d64QPseudo:
    case ARM::VLD1d64QPseudoWB_fixed:
    case ARM::VLD4d8Pseudo_UPD:
    case ARM::VLD4d16Pseudo_UPD:
    case ARM::VLD4d32Pseudo_UPD:
    case ARM::VLD4q8Pseudo_UPD:
    case ARM::VLD4q16Pseudo_UPD:
    case ARM::VLD4q32Pseudo_UPD:
    case ARM::VLD4q8oddPseudo:
    case ARM::VLD4q16oddPseudo:
    case ARM::VLD4q32oddPseudo:
    case ARM::VLD4q8oddPseudo_UPD:
    case ARM::VLD4q16oddPseudo_UPD:
    case ARM::VLD4q32oddPseudo_UPD:
    case ARM::VLD3DUPd8Pseudo:
    case ARM::VLD3DUPd16Pseudo:
    case ARM::VLD3DUPd32Pseudo:
    case ARM::VLD3DUPd8Pseudo_UPD:
    case ARM::VLD3DUPd16Pseudo_UPD:
    case ARM::VLD3DUPd32Pseudo_UPD:
    case ARM::VLD4DUPd8Pseudo:
    case ARM::VLD4DUPd16Pseudo:
    case ARM::VLD4DUPd32Pseudo:
    case ARM::VLD4DUPd8Pseudo_UPD:
    case ARM::VLD4DUPd16Pseudo_UPD:
    case ARM::VLD4DUPd32Pseudo_UPD:
      ExpandVLD(MBBI);
      return true;

    case ARM::VST2q8Pseudo:
    case ARM::VST2q16Pseudo:
    case ARM::VST2q32Pseudo:
    case ARM::VST2q8PseudoWB_fixed:
    case ARM::VST2q16PseudoWB_fixed:
    case ARM::VST2q32PseudoWB_fixed:
    case ARM::VST2q8PseudoWB_register:
    case ARM::VST2q16PseudoWB_register:
    case ARM::VST2q32PseudoWB_register:
    case ARM::VST3d8Pseudo:
    case ARM::VST3d16Pseudo:
    case ARM::VST3d32Pseudo:
    case ARM::VST1d64TPseudo:
    case ARM::VST3d8Pseudo_UPD:
    case ARM::VST3d16Pseudo_UPD:
    case ARM::VST3d32Pseudo_UPD:
    case ARM::VST1d64TPseudoWB_fixed:
    case ARM::VST1d64TPseudoWB_register:
    case ARM::VST3q8Pseudo_UPD:
    case ARM::VST3q16Pseudo_UPD:
    case ARM::VST3q32Pseudo_UPD:
    case ARM::VST3q8oddPseudo:
    case ARM::VST3q16oddPseudo:
    case ARM::VST3q32oddPseudo:
    case ARM::VST3q8oddPseudo_UPD:
    case ARM::VST3q16oddPseudo_UPD:
    case ARM::VST3q32oddPseudo_UPD:
    case ARM::VST4d8Pseudo:
    case ARM::VST4d16Pseudo:
    case ARM::VST4d32Pseudo:
    case ARM::VST1d64QPseudo:
    case ARM::VST4d8Pseudo_UPD:
    case ARM::VST4d16Pseudo_UPD:
    case ARM::VST4d32Pseudo_UPD:
    case ARM::VST1d64QPseudoWB_fixed:
    case ARM::VST1d64QPseudoWB_register:
    case ARM::VST4q8Pseudo_UPD:
    case ARM::VST4q16Pseudo_UPD:
    case ARM::VST4q32Pseudo_UPD:
    case ARM::VST4q8oddPseudo:
    case ARM::VST4q16oddPseudo:
    case ARM::VST4q32oddPseudo:
    case ARM::VST4q8oddPseudo_UPD:
    case ARM::VST4q16oddPseudo_UPD:
    case ARM::VST4q32oddPseudo_UPD:
      ExpandVST(MBBI);
      return true;

    case ARM::VLD1LNq8Pseudo:
    case ARM::VLD1LNq16Pseudo:
    case ARM::VLD1LNq32Pseudo:
    case ARM::VLD1LNq8Pseudo_UPD:
    case ARM::VLD1LNq16Pseudo_UPD:
    case ARM::VLD1LNq32Pseudo_UPD:
    case ARM::VLD2LNd8Pseudo:
    case ARM::VLD2LNd16Pseudo:
    case ARM::VLD2LNd32Pseudo:
    case ARM::VLD2LNq16Pseudo:
    case ARM::VLD2LNq32Pseudo:
    case ARM::VLD2LNd8Pseudo_UPD:
    case ARM::VLD2LNd16Pseudo_UPD:
    case ARM::VLD2LNd32Pseudo_UPD:
    case ARM::VLD2LNq16Pseudo_UPD:
    case ARM::VLD2LNq32Pseudo_UPD:
    case ARM::VLD3LNd8Pseudo:
    case ARM::VLD3LNd16Pseudo:
    case ARM::VLD3LNd32Pseudo:
    case ARM::VLD3LNq16Pseudo:
    case ARM::VLD3LNq32Pseudo:
    case ARM::VLD3LNd8Pseudo_UPD:
    case ARM::VLD3LNd16Pseudo_UPD:
    case ARM::VLD3LNd32Pseudo_UPD:
    case ARM::VLD3LNq16Pseudo_UPD:
    case ARM::VLD3LNq32Pseudo_UPD:
    case ARM::VLD4LNd8Pseudo:
    case ARM::VLD4LNd16Pseudo:
    case ARM::VLD4LNd32Pseudo:
    case ARM::VLD4LNq16Pseudo:
    case ARM::VLD4LNq32Pseudo:
    case ARM::VLD4LNd8Pseudo_UPD:
    case ARM::VLD4LNd16Pseudo_UPD:
    case ARM::VLD4LNd32Pseudo_UPD:
    case ARM::VLD4LNq16Pseudo_UPD:
    case ARM::VLD4LNq32Pseudo_UPD:
    case ARM::VST1LNq8Pseudo:
    case ARM::VST1LNq16Pseudo:
    case ARM::VST1LNq32Pseudo:
    case ARM::VST1LNq8Pseudo_UPD:
    case ARM::VST1LNq16Pseudo_UPD:
    case ARM::VST1LNq32Pseudo_UPD:
    case ARM::VST2LNd8Pseudo:
    case ARM::VST2LNd16Pseudo:
    case ARM::VST2LNd32Pseudo:
    case ARM::VST2LNq16Pseudo:
    case ARM::VST2LNq32Pseudo:
    case ARM::VST2LNd8Pseudo_UPD:
    case ARM::VST2LNd16Pseudo_UPD:
    case ARM::VST2LNd32Pseudo_UPD:
    case ARM::VST2LNq16Pseudo_UPD:
    case ARM::VST2LNq32Pseudo_UPD:
    case ARM::VST3LNd8Pseudo:
    case ARM::VST3LNd16Pseudo:
    case ARM::VST3LNd32Pseudo:
    case ARM::VST3LNq16Pseudo:
    case ARM::VST3LNq32Pseudo:
    case ARM::VST3LNd8Pseudo_UPD:
    case ARM::VST3LNd16Pseudo_UPD:
    case ARM::VST3LNd32Pseudo_UPD:
    case ARM::VST3LNq16Pseudo_UPD:
    case ARM::VST3LNq32Pseudo_UPD:
    case ARM::VST4LNd8Pseudo:
    case ARM::VST4LNd16Pseudo:
    case ARM::VST4LNd32Pseudo:
    case ARM::VST4LNq16Pseudo:
    case ARM::VST4LNq32Pseudo:
    case ARM::VST4LNd8Pseudo_UPD:
    case ARM::VST4LNd16Pseudo_UPD:
    case ARM::VST4LNd32Pseudo_UPD:
    case ARM::VST4LNq16Pseudo_UPD:
    case ARM::VST4LNq32Pseudo_UPD:
      ExpandLaneOp(MBBI);
      return true;

    case ARM::VTBL3Pseudo: ExpandVTBL(MBBI, ARM::VTBL3, false); return true;
    case ARM::VTBL4Pseudo: ExpandVTBL(MBBI, ARM::VTBL4, false); return true;
    case ARM::VTBX3Pseudo: ExpandVTBL(MBBI, ARM::VTBX3, true); return true;
    case ARM::VTBX4Pseudo: ExpandVTBL(MBBI, ARM::VTBX4, true); return true;

    case ARM::CMP_SWAP_8:
      if (STI->isThumb())
        return ExpandCMP_SWAP(MBB, MBBI, ARM::t2LDREXB, ARM::t2STREXB,
                              ARM::tUXTB, NextMBBI);
      else
        return ExpandCMP_SWAP(MBB, MBBI, ARM::LDREXB, ARM::STREXB,
                              ARM::UXTB, NextMBBI);
    case ARM::CMP_SWAP_16:
      if (STI->isThumb())
        return ExpandCMP_SWAP(MBB, MBBI, ARM::t2LDREXH, ARM::t2STREXH,
                              ARM::tUXTH, NextMBBI);
      else
        return ExpandCMP_SWAP(MBB, MBBI, ARM::LDREXH, ARM::STREXH,
                              ARM::UXTH, NextMBBI);
    case ARM::CMP_SWAP_32:
      if (STI->isThumb())
        return ExpandCMP_SWAP(MBB, MBBI, ARM::t2LDREX, ARM::t2STREX, 0,
                              NextMBBI);
      else
        return ExpandCMP_SWAP(MBB, MBBI, ARM::LDREX, ARM::STREX, 0, NextMBBI);

    case ARM::CMP_SWAP_64:
      return ExpandCMP_SWAP_64(MBB, MBBI, NextMBBI);
  }
}

bool ARMExpandPseudo::ExpandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= ExpandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool ARMExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  STI = &static_cast<const ARMSubtarget &>(MF.getSubtarget());
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();
  AFI = MF.getInfo<ARMFunctionInfo>();

  bool Modified = false;
  for (MachineFunction::iterator MFI = MF.begin(), E = MF.end(); MFI != E;
       ++MFI)
    Modified |= ExpandMBB(*MFI);
  if (VerifyARMPseudo)
    MF.verify(this, "After expanding ARM pseudo instructions.");
  return Modified;
}

/// createARMExpandPseudoPass - returns an instance of the pseudo instruction
/// expansion pass.
FunctionPass *llvm::createARMExpandPseudoPass() {
  return new ARMExpandPseudo();
}
