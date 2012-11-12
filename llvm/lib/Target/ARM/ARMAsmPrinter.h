//===-- ARMAsmPrinter.h - Print machine code to an ARM .s file --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ARM Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMASMPRINTER_H
#define ARMASMPRINTER_H

#include "ARM.h"
#include "ARMTargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MCOperand;

namespace ARM {
  enum DW_ISA {
    DW_ISA_ARM_thumb = 1,
    DW_ISA_ARM_arm = 2
  };
}

class LLVM_LIBRARY_VISIBILITY ARMAsmPrinter : public AsmPrinter {

  /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
  /// make the right decision when printing asm code for different targets.
  const ARMSubtarget *Subtarget;

  /// AFI - Keep a pointer to ARMFunctionInfo for the current
  /// MachineFunction.
  ARMFunctionInfo *AFI;

  /// MCP - Keep a pointer to constantpool entries of the current
  /// MachineFunction.
  const MachineConstantPool *MCP;

  /// InConstantPool - Maintain state when emitting a sequence of constant
  /// pool entries so we can properly mark them as data regions.
  bool InConstantPool;
public:
  explicit ARMAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
    : AsmPrinter(TM, Streamer), AFI(NULL), MCP(NULL), InConstantPool(false) {
      Subtarget = &TM.getSubtarget<ARMSubtarget>();
    }

  virtual const char *getPassName() const LLVM_OVERRIDE {
    return "ARM Assembly Printer";
  }

  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O,
                    const char *Modifier = 0);

  virtual bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &O) LLVM_OVERRIDE;
  virtual bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                                     unsigned AsmVariant, const char *ExtraCode,
                                     raw_ostream &O) LLVM_OVERRIDE;

  void EmitJumpTable(const MachineInstr *MI);
  void EmitJump2Table(const MachineInstr *MI);
  virtual void EmitInstruction(const MachineInstr *MI) LLVM_OVERRIDE;
  virtual bool runOnMachineFunction(MachineFunction &F) LLVM_OVERRIDE;

  virtual void EmitConstantPool() LLVM_OVERRIDE {
    // we emit constant pools customly!
  }
  virtual void EmitFunctionBodyEnd() LLVM_OVERRIDE;
  virtual void EmitFunctionEntryLabel() LLVM_OVERRIDE;
  virtual void EmitStartOfAsmFile(Module &M) LLVM_OVERRIDE;
  virtual void EmitEndOfAsmFile(Module &M) LLVM_OVERRIDE;
  virtual void EmitXXStructor(const Constant *CV) LLVM_OVERRIDE;

  // lowerOperand - Convert a MachineOperand into the equivalent MCOperand.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp);

private:
  // Helpers for EmitStartOfAsmFile() and EmitEndOfAsmFile()
  void emitAttributes();

  // Helper for ELF .o only
  void emitARMAttributeSection();

  // Generic helper used to emit e.g. ARMv5 mul pseudos
  void EmitPatchedInstruction(const MachineInstr *MI, unsigned TargetOpc);

  void EmitUnwindingInstruction(const MachineInstr *MI);

  // emitPseudoExpansionLowering - tblgen'erated.
  bool emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                                   const MachineInstr *MI);

public:
  void PrintDebugValueComment(const MachineInstr *MI, raw_ostream &OS);

  virtual MachineLocation
    getDebugValueLocation(const MachineInstr *MI) const LLVM_OVERRIDE;

  /// EmitDwarfRegOp - Emit dwarf register operation.
  virtual void EmitDwarfRegOp(const MachineLocation &MLoc) const LLVM_OVERRIDE;

  virtual unsigned getISAEncoding() LLVM_OVERRIDE {
    // ARM/Darwin adds ISA to the DWARF info for each function.
    if (!Subtarget->isTargetDarwin())
      return 0;
    return Subtarget->isThumb() ?
      ARM::DW_ISA_ARM_thumb : ARM::DW_ISA_ARM_arm;
  }

private:
  MCOperand GetSymbolRef(const MachineOperand &MO, const MCSymbol *Symbol);
  MCSymbol *GetARMJTIPICJumpTableLabel2(unsigned uid, unsigned uid2) const;

  MCSymbol *GetARMSJLJEHLabel(void) const;

  MCSymbol *GetARMGVSymbol(const GlobalValue *GV);

public:
  /// EmitMachineConstantPoolValue - Print a machine constantpool value to
  /// the .s file.
  virtual void
    EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) LLVM_OVERRIDE;
};
} // end namespace llvm

#endif
