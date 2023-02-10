//===- ARMRandezvousShadowStack.cpp - ARM Randezvous Shadow Stack ---------===//
//
// Copyright (c) 2021-2022, University of Rochester
//
// Part of the Randezvous Project, under the Apache License v2.0 with
// LLVM Exceptions.  See LICENSE.txt in the llvm directory for license
// information.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of a pass that instruments ARM machine
// code to save/load the return address to/from a randomized compact shadow
// stack.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-randezvous-shadow-stack"

#include "ARMRandezvousCLR.h"
#include "ARMRandezvousOptions.h"
#include "ARMRandezvousShadowStack.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

STATISTIC(NumPrologues, "Number of prologues transformed to use shadow stack");
STATISTIC(NumEpilogues, "Number of epilogues transformed to use shadow stack");
STATISTIC(NumNullified, "Number of return addresses nullified");

char ARMRandezvousShadowStack::ID = 0;

ARMRandezvousShadowStack::ARMRandezvousShadowStack() : ModulePass(ID) {
}

StringRef
ARMRandezvousShadowStack::getPassName() const {
  return "ARM Randezvous Shadow Stack Pass";
}

void
ARMRandezvousShadowStack::getAnalysisUsage(AnalysisUsage & AU) const {
  // We need this to access MachineFunctions
  AU.addRequired<MachineModuleInfoWrapperPass>();

  AU.setPreservesCFG();
  ModulePass::getAnalysisUsage(AU);
}

void
ARMRandezvousShadowStack::releaseMemory() {
  TrapBlocks.clear();
}

//
// Method: createShadowStack()
//
// Description:
//   This method creates a GlobalVariable as the shadow stack.  The shadow
//   stack is initialized either as zeroed memory or with addresses of randomly
//   picked trap blocks.
//
// Input:
//   M - A reference to the Module in which to create the shadow stack.
//
// Return value:
//   A pointer to the created GlobalVariable.
//
GlobalVariable *
ARMRandezvousShadowStack::createShadowStack(Module & M) {
  // Create types for the shadow stack
  uint64_t PtrSize = M.getDataLayout().getPointerSize();
  LLVMContext & Ctx = M.getContext();
  PointerType * RetAddrTy = PointerType::getUnqual(Type::getInt8Ty(Ctx));
  ArrayType * SSTy = ArrayType::get(RetAddrTy,
                                    RandezvousShadowStackSize / PtrSize);

  // Create the shadow stack
  Constant * CSS = M.getOrInsertGlobal(ShadowStackName, SSTy);
  GlobalVariable * SS = dyn_cast<GlobalVariable>(CSS);
  assert(SS != nullptr && "Shadow stack has wrong type!");
  SS->setLinkage(GlobalVariable::LinkOnceAnyLinkage);

  // Initialize the shadow stack if not initialized
  if (!SS->hasInitializer()) {
    Constant * SSInit = nullptr;
    if (EnableRandezvousDecoyPointers) {
      // Initialize the shadow stack with an array of random values; they are
      // either random trap block addresses or purely random values with the
      // LSB set
      std::vector<Constant *> SSInitArray;
      for (unsigned i = 0; i < SSTy->getNumElements(); ++i) {
        if (!TrapBlocks.empty()) {
          uint64_t Idx = (*RNG)() % TrapBlocks.size();
          const BasicBlock * BB = TrapBlocks[Idx]->getBasicBlock();
          SSInitArray.push_back(BlockAddress::get(const_cast<BasicBlock *>(BB)));
        } else {
          APInt A(8 * PtrSize, (*RNG)() | 0x1);
          SSInitArray.push_back(Constant::getIntegerValue(RetAddrTy, A));
        }
      }
      SSInit = ConstantArray::get(SSTy, SSInitArray);
    } else {
      // Initialize the shadow stack with zeros
      SSInit = ConstantArray::getNullValue(SSTy);
    }
    SS->setInitializer(SSInit);
  }

  // Add the shadow stack to @llvm.used
  appendToUsed(M, { SS });

  return SS;
}

//
// Method: createInitFunction()
//
// Description:
//   This method creates a function (both Function and MachineFunction) that
//   initializes the reserved registers for the shadow stack.
//
// Inputs:
//   M  - A reference to the Module in which to create the function.
//   SS - A reference to the shadow stack global variable.
//
// Return value:
//   A pointer to the created Function.
//
Function *
ARMRandezvousShadowStack::createInitFunction(Module & M, GlobalVariable & SS) {
  // Create types for the init function
  LLVMContext & Ctx = M.getContext();
  FunctionType * FuncTy = FunctionType::get(Type::getVoidTy(Ctx), false);

  // Create the init function
  FunctionCallee FC = M.getOrInsertFunction(InitFuncName, FuncTy);
  Function * F = dyn_cast<Function>(FC.getCallee());
  assert(F != nullptr && "Init function has wrong type!");
  MachineModuleInfo & MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  MachineFunction & MF = MMI.getOrCreateMachineFunction(*F);

  // Set necessary attributes and properties
  F->setLinkage(GlobalVariable::LinkOnceAnyLinkage);
  if (!F->hasFnAttribute(Attribute::Naked)) {
    F->addFnAttr(Attribute::Naked);
  }
  if (!F->hasFnAttribute(Attribute::NoUnwind)) {
    F->addFnAttr(Attribute::NoUnwind);
  }
  if (!F->hasFnAttribute(Attribute::WillReturn)) {
    F->addFnAttr(Attribute::WillReturn);
  }
  using Property = MachineFunctionProperties::Property;
  if (!MF.getProperties().hasProperty(Property::NoVRegs)) {
    MF.getProperties().set(Property::NoVRegs);
  }

  // Create a basic block if not created
  if (F->empty()) {
    assert(MF.empty() && "Machine IR basic block already there!");

    // Build an IR basic block
    BasicBlock * BB = BasicBlock::Create(Ctx, "", F);
    IRBuilder<> IRB(BB);
    IRB.CreateRetVoid(); // At this point, what the IR basic block contains
                         // doesn't matter so just place a return there

    // Build machine IR basic block(s)
    const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();
    MachineBasicBlock * MBB = MF.CreateMachineBasicBlock(BB);
    MachineBasicBlock * MBB2 = nullptr;
    MachineBasicBlock * MBB3 = nullptr;
    MachineBasicBlock * RetMBB = MBB;
    MF.push_back(MBB);
    // MOVi16 SSPtrReg, @SS_lo
    BuildMI(MBB, DebugLoc(), TII->get(ARM::t2MOVi16), ShadowStackPtrReg)
    .addGlobalAddress(&SS, 0, ARMII::MO_LO16)
    .add(predOps(ARMCC::AL));
    // MOVTi16 SSPtrReg, @SS_hi
    BuildMI(MBB, DebugLoc(), TII->get(ARM::t2MOVTi16), ShadowStackPtrReg)
    .addReg(ShadowStackPtrReg)
    .addGlobalAddress(&SS, 0, ARMII::MO_HI16)
    .add(predOps(ARMCC::AL));
    if (RandezvousRNGAddress != 0) {
      // User provided an RNG address, so load a random stride from the RNG
      if (ARM_AM::getT2SOImmVal(RandezvousRNGAddress) != -1) {
        // Use MOVi if the address can be encoded in Thumb modified constant
        BuildMI(MBB, DebugLoc(), TII->get(ARM::t2MOVi), ARM::R0)
        .addImm(RandezvousRNGAddress)
        .add(predOps(ARMCC::AL))
        .add(condCodeOp()); // No 'S' bit
      } else {
        // Otherwise use MOVi16/MOVTi16 to encode lower/upper 16 bits of the
        // address
        BuildMI(MBB, DebugLoc(), TII->get(ARM::t2MOVi16), ARM::R0)
        .addImm(RandezvousRNGAddress & 0xffff)
        .add(predOps(ARMCC::AL));
        BuildMI(MBB, DebugLoc(), TII->get(ARM::t2MOVTi16), ARM::R0)
        .addReg(ARM::R0)
        .addImm((RandezvousRNGAddress >> 16) & 0xffff)
        .add(predOps(ARMCC::AL));
      }

      MBB2 = MF.CreateMachineBasicBlock(BB);
      MF.push_back(MBB2);
      MBB->addSuccessor(MBB2);
      MBB2->addSuccessor(MBB2);
      // LDRi12 SSStrideReg, [R0, #0]
      BuildMI(MBB2, DebugLoc(), TII->get(ARM::t2LDRi12), ShadowStackStrideReg)
      .addReg(ARM::R0)
      .addImm(0)
      .add(predOps(ARMCC::AL));
      // CMPi8 SSStrideReg, #0
      BuildMI(MBB2, DebugLoc(), TII->get(ARM::t2CMPri))
      .addReg(ShadowStackStrideReg)
      .addImm(0)
      .add(predOps(ARMCC::AL));
      // BEQ MBB2
      BuildMI(MBB2, DebugLoc(), TII->get(ARM::t2Bcc))
      .addMBB(MBB2)
      .addImm(ARMCC::EQ)
      .addReg(ARM::CPSR, RegState::Kill);

      MBB3 = MF.CreateMachineBasicBlock(BB);
      MF.push_back(MBB3);
      MBB2->addSuccessor(MBB3);
      // BFC SSStrideReg, #(SSStrideLength - 1), #(33 - SSStrideLength)
      BuildMI(MBB3, DebugLoc(), TII->get(ARM::t2BFC), ShadowStackStrideReg)
      .addReg(ShadowStackStrideReg)
      .addImm((1 << (RandezvousShadowStackStrideLength - 1)) - 1)
      .add(predOps(ARMCC::AL));
      // BFC SSStrideReg, #0, #2
      BuildMI(MBB3, DebugLoc(), TII->get(ARM::t2BFC), ShadowStackStrideReg)
      .addReg(ShadowStackStrideReg)
      .addImm(~0x3)
      .add(predOps(ARMCC::AL));
      RetMBB = MBB3;
    } else {
      // Generate a static random stride
      uint64_t Stride = (*RNG)();
      Stride &= (1ul << (RandezvousShadowStackStrideLength - 1)) - 1;
      Stride &= ~0x3ul;
      if (ARM_AM::getT2SOImmVal(Stride) != -1) {
        // Use MOVi if the stride can be encoded in Thumb modified constant
        BuildMI(MBB, DebugLoc(), TII->get(ARM::t2MOVi), ShadowStackStrideReg)
        .addImm(Stride)
        .add(predOps(ARMCC::AL))
        .add(condCodeOp()); // No 'S' bit
      } else {
        // Otherwise use MOVi16/MOVTi16 to encode lower/upper 16 bits of the
        // stride
        BuildMI(MBB, DebugLoc(), TII->get(ARM::t2MOVi16), ShadowStackStrideReg)
        .addImm(Stride & 0xffff)
        .add(predOps(ARMCC::AL));
        BuildMI(MBB, DebugLoc(), TII->get(ARM::t2MOVTi16), ShadowStackStrideReg)
        .addReg(ShadowStackStrideReg)
        .addImm((Stride >> 16) & 0xffff)
        .add(predOps(ARMCC::AL));
      }
    }
    // BX_RET
    BuildMI(RetMBB, DebugLoc(), TII->get(ARM::tBX_RET))
    .add(predOps(ARMCC::AL));
  }

  // Add the init function to @llvm.used
  appendToUsed(M, { F });

  return F;
}

//
// Method: pushToShadowStack()
//
// Description:
//   This method modifies a PUSH instruction to not save LR to the stack and
//   inserts new instructions that save LR to the shadow stack.
//
// Inputs:
//   MI     - A reference to a PUSH instruction that saves LR to the stack.
//   LR     - A reference to the LR operand of the PUSH.
//   Stride - A static stride to use.
//
// Return value:
//   true - The machine code was modified.
//
bool
ARMRandezvousShadowStack::pushToShadowStack(MachineInstr & MI,
                                            MachineOperand & LR,
                                            uint32_t Stride) {
  MachineFunction & MF = *MI.getMF();
  const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();
  const DebugLoc & DL = MI.getDebugLoc();

  Register PredReg;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);

  // Build the following instruction sequence
  //
  // STR_POST LR, [SSPtrReg], #Stride
  // ADDrr    SSPtrReg, SSPtrReg, SSStrideReg
  std::vector<MachineInstr *> NewInsts;
  NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2STR_POST), ShadowStackPtrReg)
                     .addReg(ARM::LR)
                     .addReg(ShadowStackPtrReg)
                     .addImm(Stride)
                     .add(predOps(Pred, PredReg)));
  NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2ADDrr), ShadowStackPtrReg)
                     .addReg(ShadowStackPtrReg)
                     .addReg(ShadowStackStrideReg)
                     .add(predOps(Pred, PredReg))
                     .add(condCodeOp()));

  // Now insert these new instructions into the basic block
  insertInstsBefore(MI, NewInsts);

  // At last, replace the old PUSH with a new one that doesn't push LR to the
  // stack
  switch (MI.getOpcode()) {
  case ARM::t2STMDB_UPD:
    // STMDB_UPD should store at least two registers; if it happens to be two,
    // we replace it with a STR_PRE
    assert(MI.getNumExplicitOperands() >= 6 && "Buggy STMDB_UPD!");
    if (MI.getNumExplicitOperands() > 6) {
      MI.removeOperand(MI.getOperandNo(&LR));
    } else {
      unsigned Idx = MI.getOperandNo(&LR);
      Idx = Idx == 4 ? 5 : 4;
      insertInstBefore(MI, BuildMI(MF, DL, TII->get(ARM::t2STR_PRE), ARM::SP)
                           .add(MI.getOperand(Idx))
                           .addReg(ARM::SP)
                           .addImm(-4)
                           .add(predOps(Pred, PredReg))
                           .setMIFlags(MI.getFlags()));
      removeInst(MI);
    }
    break;

  case ARM::tPUSH:
    // PUSH should store at least one register; if it happens to be one, we
    // just remove it
    assert(MI.getNumExplicitOperands() >= 3 && "Buggy PUSH!");
    if (MI.getNumExplicitOperands() > 3) {
      MI.removeOperand(MI.getOperandNo(&LR));
    } else {
      removeInst(MI);
    }
    break;

  // ARM::t2STR_PRE
  default:
    // STR_PRE only stores one register, so we just remove it
    removeInst(MI);
    break;
  }

  ++NumPrologues;
  return true;
}

//
// Method: popFromShadowStack()
//
// Description:
//   This method modifies a POP instruction to not write to PC/LR and inserts
//   new instructions that load the return address from the shadow stack into
//   PC/LR.
//
// Inputs:
//   MI     - A reference to a POP instruction that writes to LR or PC.
//   PCLR   - A reference to the PC or LR operand of the POP.
//   Stride - A static stride to use.
//
// Return value:
//   true - The machine code was modified.
//
bool
ARMRandezvousShadowStack::popFromShadowStack(MachineInstr & MI,
                                             MachineOperand & PCLR,
                                             uint32_t Stride) {
  MachineFunction & MF = *MI.getMF();
  const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();
  const DebugLoc & DL = MI.getDebugLoc();

  Register PredReg;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);

  // Build the following instruction sequence
  //
  // SUBrr    SSPtrReg, SSPtrReg, SSStrideReg
  // LDR_PRE  PC/LR, [SSPtrReg, #-Stride]!
  std::vector<MachineInstr *> NewInsts;
  NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2SUBrr), ShadowStackPtrReg)
                     .addReg(ShadowStackPtrReg)
                     .addReg(ShadowStackStrideReg)
                     .add(predOps(Pred, PredReg))
                     .add(condCodeOp()));
  NewInsts.push_back(BuildMI(MF, DL, TII->get(PCLR.getReg() == ARM::PC ?
                                              ARM::t2LDR_PRE_RET :
                                              ARM::t2LDR_PRE),
                             PCLR.getReg())
                     .addReg(ShadowStackPtrReg, RegState::Define)
                     .addReg(ShadowStackPtrReg)
                     .addImm(-Stride)
                     .add(predOps(Pred, PredReg)));

  // Now insert these new instructions into the basic block
  insertInstsAfter(MI, NewInsts);

  // Replace the old POP with a new one that doesn't write to PC/LR
  switch (MI.getOpcode()) {
  case ARM::t2LDMIA_RET:
    MI.setDesc(TII->get(ARM::t2LDMIA_UPD));
    NewInsts[1]->copyImplicitOps(MF, MI);
    for (unsigned i = MI.getNumOperands() - 1, e = MI.getNumExplicitOperands();
         i >= e; --i) {
      MI.removeOperand(i);
    }
    LLVM_FALLTHROUGH;
  case ARM::t2LDMIA_UPD:
    // LDMIA_UPD should load at least two registers; if it happens to be two,
    // we replace it with a LDR_POST
    assert(MI.getNumExplicitOperands() >= 6 && "Buggy LDMIA_UPD!");
    if (MI.getNumExplicitOperands() > 6) {
      MI.removeOperand(MI.getOperandNo(&PCLR));
    } else {
      unsigned Idx = MI.getOperandNo(&PCLR);
      Idx = Idx == 4 ? 5 : 4;
      insertInstAfter(MI, BuildMI(MF, DL, TII->get(ARM::t2LDR_POST),
                                  MI.getOperand(Idx).getReg())
                          .addReg(ARM::SP, RegState::Define)
                          .addReg(ARM::SP)
                          .addImm(4)
                          .add(predOps(Pred, PredReg))
                          .setMIFlags(MI.getFlags()));
      removeInst(MI);
    }
    break;

  case ARM::tPOP_RET:
    MI.setDesc(TII->get(ARM::tPOP));
    NewInsts[1]->copyImplicitOps(MF, MI);
    for (unsigned i = MI.getNumOperands() - 1, e = MI.getNumExplicitOperands();
         i >= e; --i) {
      MI.removeOperand(i);
    }
    LLVM_FALLTHROUGH;
  case ARM::tPOP:
    // POP should load at least one register; if it happens to be one, we just
    // remove it
    assert(MI.getNumExplicitOperands() >= 3 && "Buggy POP!");
    if (MI.getNumExplicitOperands() > 3) {
      MI.removeOperand(MI.getOperandNo(&PCLR));
    } else {
      removeInst(MI);
    }
    break;

  // ARM::t2LDR_POST
  default:
    // LDR_POST only loads one register, so we just remove it
    removeInst(MI);
    break;
  }

  if (EnableRandezvousRAN) {
    // Nullify the return address in the shadow stack
    nullifyReturnAddress(*NewInsts[1], NewInsts[1]->getOperand(0));
  }

  ++NumEpilogues;
  return true;
}

//
// Method: nullifyReturnAddress()
//
// Description:
//   This method nullifies an in-memory return address by either zeroing it out
//   or filling it with a null value (either the address of a randomly picked
//   trap block or a purely random value).
//
// Inputs:
//   MI   - A reference to a POP or LDR instruction that writes to LR or PC.
//   PCLR - A reference to the PC or LR operand of MI.
//
// Return value:
//   true - The machine code was modified.
//
bool
ARMRandezvousShadowStack::nullifyReturnAddress(MachineInstr & MI,
                                               MachineOperand & PCLR) {
  MachineFunction & MF = *MI.getMF();
  const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();
  const DebugLoc & DL = MI.getDebugLoc();

  Register PredReg;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);

  // Mark LR as restored since we're going to use LR to hold the return address
  // in all the cases
  MachineFrameInfo & MFI = MF.getFrameInfo();
  if (MFI.isCalleeSavedInfoValid()) {
    for (CalleeSavedInfo & CSI : MFI.getCalleeSavedInfo()) {
      if (CSI.getReg() == ARM::LR) {
        CSI.setRestored(true);
        break;
      }
    }
  }

  // We need to use a scratch register as the source register of a store.  If
  // no free register is around, spill and use R4.
  std::vector<Register> FreeRegs = findFreeRegistersAfter(MI);
  bool Spill = FreeRegs.empty();
  Register FreeReg = Spill ? ARM::R4 : FreeRegs[0];

  std::vector<MachineInstr *> NewInsts;
  switch (MI.getOpcode()) {
  // LDMIA_RET SP!, {..., PC} -> LDMIA_UPD SP!, {..., LR}
  //                             MOVi16    FreeReg, #0
  //                             STRi8     FreeReg, [SP, #-4]
  //                             BX_RET
  case ARM::t2LDMIA_RET:
    assert(PCLR.getReg() == ARM::PC && "Buggy POP!");
    MI.setDesc(TII->get(ARM::t2LDMIA_UPD));
    PCLR.setReg(ARM::LR);
    insertInstAfter(MI, BuildMI(MF, DL, TII->get(ARM::tBX_RET))
                        .add(predOps(Pred, PredReg)));
    LLVM_FALLTHROUGH;
  // LDMIA_UPD SP!, {..., LR} -> LDMIA_UPD SP!, {..., LR}
  //                             MOVi16    FreeReg, #0
  //                             STRi8     FreeReg, [SP, #-4]
  case ARM::t2LDMIA_UPD:
  // LDR_POST LR, [SP], #4 -> LDR_POST LR, [SP], #4
  //                          MOVi16   FreeReg, #0
  //                          STRi8    FreeReg, [SP, #-4]
  case ARM::t2LDR_POST:
    assert(PCLR.getReg() == ARM::LR && "Buggy POP!");
    if (Spill) {
      NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::tPUSH))
                         .add(predOps(Pred, PredReg))
                         .addReg(FreeReg));
    }
    NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2MOVi16), FreeReg)
                       .addImm(0)
                       .add(predOps(Pred, PredReg)));
    NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2STRi8))
                       .addReg(FreeReg)
                       .addReg(ARM::SP)
                       .addImm(-4)
                       .add(predOps(Pred, PredReg)));
    if (Spill) {
      NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::tPOP))
                         .add(predOps(Pred, PredReg))
                         .addReg(FreeReg));
    }
    insertInstsAfter(MI, NewInsts);
    break;

  // POP(_RET) {..., PC} -> LDMIA_UPD SP!, {..., LR}
  //                        MOVi16    FreeReg, #0
  //                        STRi8     FreeReg, [SP, #-4]
  //                        BX_RET
  case ARM::tPOP:
  case ARM::tPOP_RET: {
    assert(PCLR.getReg() == ARM::PC && "Buggy POP!");
    MachineInstrBuilder MIB = BuildMI(MF, DL, TII->get(ARM::t2LDMIA_UPD), ARM::SP)
                              .addReg(ARM::SP);
    for (MachineOperand & MO : MI.explicit_operands()) {
      if (MO.isReg() && MO.getReg() == ARM::PC) {
        MIB.addReg(ARM::LR, RegState::Define);
      } else {
        MIB.add(MO);
      }
    }
    NewInsts.push_back(MIB);
    if (Spill) {
      NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::tPUSH))
                         .add(predOps(Pred, PredReg))
                         .addReg(FreeReg));
    }
    NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2MOVi16), FreeReg)
                       .addImm(0)
                       .add(predOps(Pred, PredReg)));
    NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2STRi8))
                       .addReg(FreeReg)
                       .addReg(ARM::SP)
                       .addImm(-4)
                       .add(predOps(Pred, PredReg)));
    if (Spill) {
      NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::tPOP))
                         .add(predOps(Pred, PredReg))
                         .addReg(FreeReg));
    }
    NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::tBX_RET))
                       .add(predOps(Pred, PredReg)));
    insertInstsAfter(MI, NewInsts);
    removeInst(MI);
    break;
  }

  // LDR_PRE_RET PC, [SSPtrReg, #imm]! -> LDR_PRE LR, [SSPtrReg, #imm]!
  //                                      MOVi16  FreeReg, #0
  //                                      STRi12  FreeReg, [SSPtrReg, #0]
  //                                      BX_RET
  //
  //                                   or LDR_PRE LR, [SSPtrReg, #imm]!
  //                                      MOVi16  FreeReg, #null-lo16
  //                                      MOVTi16 FreeReg, #null-hi16
  //                                      STRi12  FreeReg, [SSPtrReg, #0]
  //                                      BX_RET
  case ARM::t2LDR_PRE_RET:
    assert(PCLR.getReg() == ARM::PC && "Buggy POP!");
    MI.setDesc(TII->get(ARM::t2LDR_PRE));
    PCLR.setReg(ARM::LR);
    insertInstAfter(MI, BuildMI(MF, DL, TII->get(ARM::tBX_RET))
                        .add(predOps(Pred, PredReg)));
    LLVM_FALLTHROUGH;
  // LDR_PRE LR, [SSPtrReg, #imm]! -> LDR_PRE LR, [SSPtrReg, #imm]!
  //                                  MOVi16  FreeReg, #0
  //                                  STRi12  FreeReg, [SSPtrReg, #0]
  //
  //                               or LDR_PRE LR, [SSPtrReg, #imm]!
  //                                  MOVi16  FreeReg, #null-lo16
  //                                  MOVTi16 FreeReg, #null-hi16
  //                                  STRi12  FreeReg, [SSPtrReg, #0]
  default:
    assert(MI.getOpcode() == ARM::t2LDR_PRE && "Unrecognized POP!");
    assert(MI.getOperand(1).getReg() == ShadowStackPtrReg && "Buggy POP!");
    assert(PCLR.getReg() == ARM::LR && "Buggy POP!");
    if (Spill) {
      NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::tPUSH))
                         .add(predOps(Pred, PredReg))
                         .addReg(FreeReg));
    }
    if (EnableRandezvousDecoyPointers) {
      if (!TrapBlocks.empty()) {
        // Use the address of a trap block as the null value
        uint64_t Idx = (*RNG)() % TrapBlocks.size();
        const BasicBlock * BB = TrapBlocks[Idx]->getBasicBlock();
        BlockAddress * BA = BlockAddress::get(const_cast<BasicBlock *>(BB));
        NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2MOVi16), FreeReg)
                           .addBlockAddress(BA, 0, ARMII::MO_LO16)
                           .add(predOps(Pred, PredReg)));
        NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2MOVTi16), FreeReg)
                           .addReg(FreeReg)
                           .addBlockAddress(BA, 0, ARMII::MO_HI16)
                           .add(predOps(Pred, PredReg)));
      } else {
        // Use a random value with the LSB set as the null value
        uint32_t NullValue = (*RNG)() | 0x1;
        if (ARM_AM::getT2SOImmVal(NullValue) != -1) {
          NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2MOVi), FreeReg)
                             .addImm(NullValue)
                             .add(predOps(Pred, PredReg))
                             .add(condCodeOp())); // No 'S' bit
        } else {
          NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2MOVi16), FreeReg)
                             .addImm(NullValue & 0xffff)
                             .add(predOps(Pred, PredReg)));
          NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2MOVTi16), FreeReg)
                             .addReg(FreeReg)
                             .addImm((NullValue >> 16) & 0xffff)
                             .add(predOps(Pred, PredReg)));
        }
      }
    } else {
      NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2MOVi16), FreeReg)
                         .addImm(0)
                         .add(predOps(Pred, PredReg)));
    }
    NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::t2STRi12))
                       .addReg(FreeReg)
                       .addReg(ShadowStackPtrReg)
                       .addImm(0)
                       .add(predOps(Pred, PredReg)));
    if (Spill) {
      NewInsts.push_back(BuildMI(MF, DL, TII->get(ARM::tPOP))
                         .add(predOps(Pred, PredReg))
                         .addReg(FreeReg));
    }
    insertInstsAfter(MI, NewInsts);
    break;
  }

  ++NumNullified;
  return true;
}

//
// Method: runOnModule()
//
// Description:
//   This method is called when the PassManager wants this pass to transform
//   the specified Module.  This method
//
//   * creates a global variable as the shadow stack,
//
//   * creates a function that initializes the reserved registers for the
//     shadow stack, and
//
//   * transforms the Module to utilize the shadow stack for saving/restoring
//     return addresses and/or to nullify a saved return address on returns.
//
// Input:
//   M - A reference to the Module to transform.
//
// Output:
//   M - The transformed Module.
//
// Return value:
//   true  - The Module was transformed.
//   false - The Module was not transformed.
//
bool
ARMRandezvousShadowStack::runOnModule(Module & M) {
  if (!EnableRandezvousShadowStack && !EnableRandezvousRAN) {
    return false;
  }

  MachineModuleInfo & MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  Twine RNGName = getPassName() + "-" + Twine(RandezvousShadowStackSeed);
  RNG = M.createRNG(RNGName.str());

  // Find trap blocks inserted by CLR
  for (Function & F : M) {
    MachineFunction * MF = MMI.getMachineFunction(F);
    if (MF != nullptr) {
      for (MachineBasicBlock & MBB : *MF) {
        if (MBB.isRandezvousTrapBlock()) {
          TrapBlocks.push_back(&MBB);
        }
      }
    }
  }

  if (EnableRandezvousShadowStack) {
    assert((RandezvousShadowStackStrideLength > 2 &&
            RandezvousShadowStackStrideLength <= 32) && "Invalid stride length!");

    // Create and initialize a global variable for the shadow stack
    GlobalVariable * SS = createShadowStack(M);

    // Create an init function that:
    // * loads the address of the shadow stack to the shadow stack pointer
    //   register, and
    // * generates a random stride (either dynamic or static) to the shadow
    //   stack stride register
    createInitFunction(M, *SS);
  }

  // Instrument pushes and pops in each function
  bool changed = false;
  for (Function & F : M) {
    MachineFunction * MF = MMI.getMachineFunction(F);
    if (MF == nullptr) {
      continue;
    }

    // Find out all pushes that write LR to the stack and all pops that read a
    // return address from the stack to LR or PC
    std::vector<std::pair<MachineInstr *, MachineOperand *> > Pushes;
    std::vector<std::pair<MachineInstr *, MachineOperand *> > Pops;
    for (MachineBasicBlock & MBB : *MF) {
      for (MachineInstr & MI : MBB) {
        switch (MI.getOpcode()) {
        // Frame-setup instructions in function prologue
        case ARM::t2STR_PRE:
        case ARM::t2STMDB_UPD:
          // STR_PRE and STMDB_UPD are considered as PUSH if they write to SP!
          if (MI.getOperand(0).getReg() != ARM::SP) {
            break;
          }
          LLVM_FALLTHROUGH;
        case ARM::tPUSH:
          if (MI.getFlag(MachineInstr::FrameSetup)) {
            for (MachineOperand & MO : MI.explicit_operands()) {
              if (MO.isReg() && MO.getReg() == ARM::LR) {
                Pushes.push_back(std::make_pair(&MI, &MO));
                break;
              }
            }
          }
          break;

        // Frame-destroy instructions in function epilogue
        case ARM::t2LDR_POST:
        case ARM::t2LDMIA_UPD:
        case ARM::t2LDMIA_RET:
          // LDR_POST and LDMIA_(UPD|RET) are considered as POP if they read
          // from SP!
          if (MI.getOperand(1).getReg() != ARM::SP) {
            break;
          }
          LLVM_FALLTHROUGH;
        case ARM::tPOP:
        case ARM::tPOP_RET:
          if (MI.getFlag(MachineInstr::FrameDestroy)) {
            // Handle 2 cases:
            // (1) Pop writing to LR
            // (2) Pop writing to PC
            for (MachineOperand & MO : MI.explicit_operands()) {
              if (MO.isReg()) {
                if (MO.getReg() == ARM::LR || MO.getReg() == ARM::PC) {
                  Pops.push_back(std::make_pair(&MI, &MO));
                  break;
                }
              }
            }
          }
          break;

        default:
          break;
        }
      }
    }

    // Instrument each push and pop
    if (EnableRandezvousShadowStack) {
      // Generate a per-function static stride
      uint32_t Stride = (*RNG)();
      Stride &= (1ul << (RandezvousShadowStackStrideLength - 1)) - 1;
      Stride &= ~0x3ul;
      // Limit the static stride to be within 8 bits, so that it can fit in
      // STR_POST and LDR_PRE as an immediate
      Stride &= 0xfful;
      // Don't generate an empty stride; either the dynamic stride or the
      // static stride needs to make sure of it, so just do it on the static to
      // leave more room for the dynamic
      if (Stride == 0u) {
        Stride = 4u;
      }

      for (auto & MIMO : Pushes) {
        changed |= pushToShadowStack(*MIMO.first, *MIMO.second, Stride);
      }
      for (auto & MIMO : Pops) {
        changed |= popFromShadowStack(*MIMO.first, *MIMO.second, Stride);
      }
    } else if (EnableRandezvousRAN) {
      for (auto & MIMO : Pops) {
        changed |= nullifyReturnAddress(*MIMO.first, *MIMO.second);
      }
    }
  }

  return changed;
}

ModulePass *
llvm::createARMRandezvousShadowStack(void) {
  return new ARMRandezvousShadowStack();
}