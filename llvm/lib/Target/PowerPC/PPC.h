//===-- PPC.h - Top-level interface for PowerPC Target ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// PowerPC back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPC_H
#define LLVM_LIB_TARGET_POWERPC_PPC_H

#include "llvm/Support/CodeGen.h"

// GCC #defines PPC on Linux but we use it as our namespace name
#undef PPC

namespace llvm {
  class PPCTargetMachine;
  class PassRegistry;
  class FunctionPass;
  class MachineInstr;
  class MachineOperand;
  class AsmPrinter;
  class MCInst;
  class MCOperand;
  class ModulePass;
  
  FunctionPass *createPPCCTRLoops();
#ifndef NDEBUG
  FunctionPass *createPPCCTRLoopsVerify();
#endif
  FunctionPass *createPPCLoopInstrFormPrepPass(PPCTargetMachine &TM);
  FunctionPass *createPPCTOCRegDepsPass();
  FunctionPass *createPPCEarlyReturnPass();
  FunctionPass *createPPCVSXCopyPass();
  FunctionPass *createPPCVSXFMAMutatePass();
  FunctionPass *createPPCVSXSwapRemovalPass();
  FunctionPass *createPPCReduceCRLogicalsPass();
  FunctionPass *createPPCMIPeepholePass();
  FunctionPass *createPPCBranchSelectionPass();
  FunctionPass *createPPCBranchCoalescingPass();
  FunctionPass *createPPCISelDag(PPCTargetMachine &TM, CodeGenOpt::Level OL);
  FunctionPass *createPPCTLSDynamicCallPass();
  FunctionPass *createPPCBoolRetToIntPass();
  FunctionPass *createPPCExpandISELPass();
  FunctionPass *createPPCPreEmitPeepholePass();
  void LowerPPCMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                    AsmPrinter &AP);
  bool LowerPPCMachineOperandToMCOperand(const MachineOperand &MO,
                                         MCOperand &OutMO, AsmPrinter &AP);

  void initializePPCCTRLoopsPass(PassRegistry&);
#ifndef NDEBUG
  void initializePPCCTRLoopsVerifyPass(PassRegistry&);
#endif
  void initializePPCLoopInstrFormPrepPass(PassRegistry&);
  void initializePPCTOCRegDepsPass(PassRegistry&);
  void initializePPCEarlyReturnPass(PassRegistry&);
  void initializePPCVSXCopyPass(PassRegistry&);
  void initializePPCVSXFMAMutatePass(PassRegistry&);
  void initializePPCVSXSwapRemovalPass(PassRegistry&);
  void initializePPCReduceCRLogicalsPass(PassRegistry&);
  void initializePPCBSelPass(PassRegistry&);
  void initializePPCBranchCoalescingPass(PassRegistry&);
  void initializePPCBoolRetToIntPass(PassRegistry&);
  void initializePPCExpandISELPass(PassRegistry &);
  void initializePPCPreEmitPeepholePass(PassRegistry &);
  void initializePPCTLSDynamicCallPass(PassRegistry &);
  void initializePPCMIPeepholePass(PassRegistry&);

  extern char &PPCVSXFMAMutateID;

  ModulePass *createPPCLowerMASSVEntriesPass();
  void initializePPCLowerMASSVEntriesPass(PassRegistry &);
  extern char &PPCLowerMASSVEntriesID;
  
  namespace PPCII {

  /// Target Operand Flag enum.
  enum TOF {
    //===------------------------------------------------------------------===//
    // PPC Specific MachineOperand flags.
    MO_NO_FLAG,

    /// On a symbol operand "FOO", this indicates that the reference is actually
    /// to "FOO@plt".  This is used for calls and jumps to external functions
    /// and for PIC calls on 32-bit ELF systems.
    MO_PLT = 1,

    /// MO_PIC_FLAG - If this bit is set, the symbol reference is relative to
    /// the function's picbase, e.g. lo16(symbol-picbase).
    MO_PIC_FLAG = 2,

    /// MO_PCREL_FLAG - If this bit is set, the symbol reference is relative to
    /// the current instruction address(pc), e.g., var@pcrel. Fixup is VK_PCREL.
    MO_PCREL_FLAG = 4,

    /// MO_GOT_FLAG - If this bit is set the symbol reference is to be computed
    /// via the GOT. For example when combined with the MO_PCREL_FLAG it should
    /// produce the relocation @got@pcrel. Fixup is VK_PPC_GOT_PCREL.
    MO_GOT_FLAG = 8,

    // MO_PCREL_OPT_FLAG - If this bit is set the operand is part of a
    // PC Relative linker optimization.
    MO_PCREL_OPT_FLAG = 16,

    /// MO_TLSGD_FLAG - If this bit is set the symbol reference is relative to
    /// TLS General Dynamic model.
    MO_TLSGD_FLAG = 32,

    /// MO_TPREL_FLAG - If this bit is set the symbol reference is relative to
    /// TLS Initial Exec model.
    MO_TPREL_FLAG = 64,

    /// MO_GOT_TLSGD_PCREL_FLAG - A combintaion of flags, if these bits are set
    /// they should produce the relocation @got@tlsgd@pcrel.
    /// Fix up is VK_PPC_GOT_TLSGD_PCREL
    MO_GOT_TLSGD_PCREL_FLAG = MO_PCREL_FLAG | MO_GOT_FLAG | MO_TLSGD_FLAG,

    /// MO_GOT_TPREL_PCREL_FLAG - A combintaion of flags, if these bits are set
    /// they should produce the relocation @got@tprel@pcrel.
    /// Fix up is VK_PPC_GOT_TPREL_PCREL
    MO_GOT_TPREL_PCREL_FLAG = MO_GOT_FLAG | MO_TPREL_FLAG | MO_PCREL_FLAG,

    /// The next are not flags but distinct values.
    MO_ACCESS_MASK = 0xf00,

    /// MO_LO, MO_HA - lo16(symbol) and ha16(symbol)
    MO_LO = 1 << 8,
    MO_HA = 2 << 8,

    MO_TPREL_LO = 4 << 8,
    MO_TPREL_HA = 3 << 8,

    /// These values identify relocations on immediates folded
    /// into memory operations.
    MO_DTPREL_LO = 5 << 8,
    MO_TLSLD_LO = 6 << 8,
    MO_TOC_LO = 7 << 8,

    // Symbol for VK_PPC_TLS fixup attached to an ADD instruction
    MO_TLS = 8 << 8
  };
  } // end namespace PPCII

} // end namespace llvm;

#endif
