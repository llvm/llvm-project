//===------- LeonPasses.h - Define passes specific to LEON ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPARC_LEON_PASSES_H
#define LLVM_LIB_TARGET_SPARC_LEON_PASSES_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"

#include "Sparc.h"
#include "SparcSubtarget.h"

namespace llvm {
class LLVM_LIBRARY_VISIBILITY LEONMachineFunctionPass
    : public MachineFunctionPass {
protected:
  const SparcSubtarget *Subtarget;
  const int LAST_OPERAND = -1;

  // this vector holds free registers that we allocate in groups for some of the
  // LEON passes
  std::vector<int> UsedRegisters;

protected:
  LEONMachineFunctionPass(TargetMachine &tm, char &ID);
  LEONMachineFunctionPass(char &ID);

  int GetRegIndexForOperand(MachineInstr &MI, int OperandIndex);
  void clearUsedRegisterList() { UsedRegisters.clear(); }

  void markRegisterUsed(int registerIndex) {
    UsedRegisters.push_back(registerIndex);
  }
  int getUnusedFPRegister(MachineRegisterInfo &MRI);
};

class LLVM_LIBRARY_VISIBILITY ReplaceSDIV : public LEONMachineFunctionPass {
public:
  static char ID;

  ReplaceSDIV();
  ReplaceSDIV(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "ReplaceSDIV: Leon erratum fix:  do not emit SDIV, but emit SDIVCC "
           "instead";
  }
};

class LLVM_LIBRARY_VISIBILITY FixCALL : public LEONMachineFunctionPass {
public:
  static char ID;

  FixCALL(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "FixCALL: Leon erratum fix: restrict the size of the immediate "
           "operand of the CALL instruction to 20 bits";
  }
};

class LLVM_LIBRARY_VISIBILITY RestoreExecAddress : public LEONMachineFunctionPass {
public:
  static char ID;

  RestoreExecAddress(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction& MF) override;

  const char *getPassName() const override {
    return "RestoreExecAddress: Leon erratum fix: ensure execution "
           "address is restored for bad floating point trap handlers.";
  }
};

class LLVM_LIBRARY_VISIBILITY IgnoreZeroFlag : public LEONMachineFunctionPass {
public:
  static char ID;

  IgnoreZeroFlag(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "IgnoreZeroFlag: Leon erratum fix: do not rely on the zero bit "
           "flag on a divide overflow for SDIVCC and UDIVCC";
  }
};

class LLVM_LIBRARY_VISIBILITY FillDataCache : public LEONMachineFunctionPass {
public:
  static char ID;
  static bool CacheFilled;

  FillDataCache(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction& MF) override;

  const char *getPassName() const override {
    return "FillDataCache: Leon erratum fix: fill data cache with values at application startup";
  }
};

class LLVM_LIBRARY_VISIBILITY InsertNOPDoublePrecision
    : public LEONMachineFunctionPass {
public:
  static char ID;

  InsertNOPDoublePrecision(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "InsertNOPDoublePrecision: Leon erratum fix: insert a NOP before "
           "the double precision floating point instruction";
  }
};

class LLVM_LIBRARY_VISIBILITY FixFSMULD : public LEONMachineFunctionPass {
public:
  static char ID;

  FixFSMULD(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "FixFSMULD: Leon erratum fix: do not utilize FSMULD";
  }
};

class LLVM_LIBRARY_VISIBILITY ReplaceFMULS : public LEONMachineFunctionPass {
public:
  static char ID;

  ReplaceFMULS(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "ReplaceFMULS: Leon erratum fix: Replace FMULS instruction with a "
           "routine using conversions/double precision operations to replace "
           "FMULS";
  }
};

class LLVM_LIBRARY_VISIBILITY PreventRoundChange
    : public LEONMachineFunctionPass {
public:
  static char ID;

  PreventRoundChange(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "PreventRoundChange: Leon erratum fix: prevent any rounding mode "
           "change request: use only the round-to-nearest rounding mode";
  }
};

class LLVM_LIBRARY_VISIBILITY FixAllFDIVSQRT : public LEONMachineFunctionPass {
public:
  static char ID;

  FixAllFDIVSQRT(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "FixAllFDIVSQRT: Leon erratum fix: Fix FDIVS/FDIVD/FSQRTS/FSQRTD "
           "instructions with NOPs and floating-point store";
  }
};

class LLVM_LIBRARY_VISIBILITY InsertNOPLoad : public LEONMachineFunctionPass {
public:
  static char ID;

  InsertNOPLoad(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "InsertNOPLoad: Leon erratum fix: Insert a NOP instruction after "
           "every single-cycle load instruction when the next instruction is "
           "another load/store instruction";
  }
};

class LLVM_LIBRARY_VISIBILITY InsertNOPsLoadStore
    : public LEONMachineFunctionPass {
public:
  static char ID;

  InsertNOPsLoadStore(TargetMachine &tm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "InsertNOPsLoadStore: Leon Erratum Fix: Insert NOPs between "
           "single-precision loads and the store, so the number of "
           "instructions between is 4";
  }
};
} // namespace lllvm

#endif // LLVM_LIB_TARGET_SPARC_LEON_PASSES_H
