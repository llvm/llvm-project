// RUN: llvm-tblgen --gen-compress-inst-emitter -I %p/../../include -I %S %s -o - | FileCheck %s

include "Common/RegClassByHwModeCommon.td"

def IsPtr64 : Predicate<"Subtarget->isPtr64()">;
defvar Ptr32 = DefaultMode;
def Ptr64 : HwMode<[IsPtr64]>;
def PtrRC : RegClassByHwMode<[Ptr32, Ptr64], [XRegs, YRegs]>;


def X_MOV : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins XRegs:$src);
  let AsmString = "x_mov $dst, $src";
  let opcode = 0;
}

def X_MOV_SMALL : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins XRegs:$src);
  let AsmString = "x_mov.small $dst, $src";
  let opcode = 1;
  let Size = 1;
}
def X_MOV_ZERO : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins);
  let AsmString = "x_mov.zero $dst";
  let opcode = 2;
  let Size = 1;
}
def X_MOV_TIED : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins XRegs:$src);
  let Constraints = "$src = $dst";
  let AsmString = "x_mov.tied $dst, $src";
  let opcode = 3;
  let Size = 1;
}

def PTR_MOV : TestInstruction {
  let OutOperandList = (outs PtrRC:$dst);
  let InOperandList = (ins PtrRC:$src);
  let AsmString = "ptr_mov $dst, $src";
  let opcode = 3;
}

def PTR_MOV_SMALL : TestInstruction {
  let OutOperandList = (outs PtrRC:$dst);
  let InOperandList = (ins PtrRC:$src);
  let AsmString = "ptr_mov.small $dst, $src";
  let opcode = 4;
  let Size = 1;
}
def PTR_MOV_ZERO : TestInstruction {
  let OutOperandList = (outs PtrRC:$dst);
  let InOperandList = (ins);
  let AsmString = "ptr_mov.zero $dst";
  let opcode = 3;
  let Size = 1;
}
def PTR_MOV_TIED : TestInstruction {
  let OutOperandList = (outs PtrRC:$dst);
  let InOperandList = (ins PtrRC:$src);
  let Constraints = "$src = $dst";
  let AsmString = "ptr_mov.tied $dst, $src";
  let opcode = 3;
  let Size = 1;
}

def : CompressPat<(X_MOV XRegs:$dst, X0),
                  (X_MOV_ZERO XRegs:$dst)>;
def : CompressPat<(X_MOV XRegs:$dst, XRegs:$dst),
                  (X_MOV_TIED XRegs:$dst)>;
def : CompressPat<(X_MOV XRegs:$dst, XRegs:$src),
                  (X_MOV_SMALL XRegs:$dst, XRegs:$src)>;
// TODO: Should also be able to use a fixed register with RegClassByHwMode
// def : CompressPat<(PTR_MOV PtrRC:$dst, X0),
//                   (PTR_MOV_ZERO PtrRC:$dst)>;
def : CompressPat<(PTR_MOV PtrRC:$dst, PtrRC:$dst),
                  (PTR_MOV_TIED PtrRC:$dst)>;
def : CompressPat<(PTR_MOV PtrRC:$dst, PtrRC:$src),
                  (PTR_MOV_SMALL PtrRC:$dst, PtrRC:$src)>;

// CHECK:     static bool compressInst(MCInst &OutInst,
// CHECK-NEXT:                          const MCInst &MI,
// CHECK-NEXT:                          const MCSubtargetInfo &STI) {
// CHECK-NEXT:   {{\[\[}}maybe_unused]] unsigned HwModeId = STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo);
// CHECK-NEXT:   switch (MI.getOpcode()) {
// CHECK-NEXT:   default: return false;
// CHECK-NEXT:   case MyTarget::PTR_MOV: {
// CHECK-NEXT:     if (MI.getOperand(1).isReg() && MI.getOperand(0).isReg() &&
// CHECK-NEXT:         (MI.getOperand(1).getReg() == MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTargetRegClassByHwModeTables[HwModeId][MyTarget::PtrRC]].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // ptr_mov.tied $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::PTR_MOV_TIED);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     if (MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTargetRegClassByHwModeTables[HwModeId][MyTarget::PtrRC]].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTargetRegClassByHwModeTables[HwModeId][MyTarget::PtrRC]].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // ptr_mov.small $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::PTR_MOV_SMALL);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case PTR_MOV
// CHECK-NEXT:   case MyTarget::X_MOV: {
// CHECK-NEXT:     if (MI.getOperand(1).isReg() &&
// CHECK-NEXT:         (MI.getOperand(1).getReg() == MyTarget::X0) &&
// CHECK-NEXT:         MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(0).getReg())) {
// CHECK-NEXT:       // x_mov.zero $dst
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::X_MOV_ZERO);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     if (MI.getOperand(1).isReg() && MI.getOperand(0).isReg() &&
// CHECK-NEXT:         (MI.getOperand(1).getReg() == MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // x_mov.tied $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::X_MOV_TIED);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     if (MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // x_mov.small $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::X_MOV_SMALL);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case X_MOV
// CHECK-NEXT:   } // switch
// CHECK-NEXT:   return false;
// CHECK-NEXT: }

// CHECK:      static bool uncompressInst(MCInst &OutInst,
// CHECK-NEXT:                            const MCInst &MI,
// CHECK-NEXT:                            const MCSubtargetInfo &STI) {
// CHECK-NEXT:   {{\[\[}}maybe_unused]] unsigned HwModeId = STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo);
// CHECK-NEXT:   switch (MI.getOpcode()) {
// CHECK-NEXT:   default: return false;
// CHECK-NEXT:   case MyTarget::PTR_MOV_SMALL: {
// CHECK-NEXT:     if (MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTargetRegClassByHwModeTables[HwModeId][MyTarget::PtrRC]].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTargetRegClassByHwModeTables[HwModeId][MyTarget::PtrRC]].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // ptr_mov $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::PTR_MOV);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case PTR_MOV_SMALL
// CHECK-NEXT:   case MyTarget::PTR_MOV_TIED: {
// CHECK-NEXT:     if (MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTargetRegClassByHwModeTables[HwModeId][MyTarget::PtrRC]].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTargetRegClassByHwModeTables[HwModeId][MyTarget::PtrRC]].contains(MI.getOperand(0).getReg())) {
// CHECK-NEXT:       // ptr_mov $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::PTR_MOV);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case PTR_MOV_TIED
// CHECK-NEXT:   case MyTarget::X_MOV_SMALL: {
// CHECK-NEXT:     if (MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // x_mov $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::X_MOV);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case X_MOV_SMALL
// CHECK-NEXT:   case MyTarget::X_MOV_TIED: {
// CHECK-NEXT:     if (MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(0).getReg())) {
// CHECK-NEXT:       // x_mov $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::X_MOV);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case X_MOV_TIED
// CHECK-NEXT:   case MyTarget::X_MOV_ZERO: {
// CHECK-NEXT:     if (MI.getOperand(0).isReg() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(0).getReg())) {
// CHECK-NEXT:       // x_mov $dst, $src
// CHECK-NEXT:       OutInst.setOpcode(MyTarget::X_MOV);
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       OutInst.addOperand(MCOperand::createReg(MyTarget::X0));
// CHECK-NEXT:       OutInst.setLoc(MI.getLoc());
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case X_MOV_ZERO
// CHECK-NEXT:   } // switch
// CHECK-NEXT:   return false;
// CHECK-NEXT: }

// CHECK:      static bool isCompressibleInst(const MachineInstr &MI,
// CHECK-NEXT:                                const MyTargetSubtarget &STI) {
// CHECK-NEXT:   {{\[\[}}maybe_unused]] unsigned HwModeId = STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo);
// CHECK-NEXT:   switch (MI.getOpcode()) {
// CHECK-NEXT:   default: return false;
// CHECK-NEXT:   case MyTarget::PTR_MOV: {
// CHECK-NEXT:     if (MI.getOperand(0).isReg() && MI.getOperand(0).getReg().isPhysical() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTargetRegClassByHwModeTables[HwModeId][MyTarget::PtrRC]].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() && MI.getOperand(1).getReg().isPhysical() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTargetRegClassByHwModeTables[HwModeId][MyTarget::PtrRC]].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // ptr_mov.small $dst, $src
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     if (MI.getOperand(1).isReg() && MI.getOperand(0).isReg() &&
// CHECK-NEXT:         (MI.getOperand(1).getReg() == MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() && MI.getOperand(1).getReg().isPhysical() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTargetRegClassByHwModeTables[HwModeId][MyTarget::PtrRC]].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // ptr_mov.tied $dst, $src
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case PTR_MOV
// CHECK-NEXT:   case MyTarget::X_MOV: {
// CHECK-NEXT:     if (MI.getOperand(0).isReg() && MI.getOperand(0).getReg().isPhysical() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() && MI.getOperand(1).getReg().isPhysical() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // x_mov.small $dst, $src
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     if (MI.getOperand(1).isReg() && MI.getOperand(0).isReg() &&
// CHECK-NEXT:         (MI.getOperand(1).getReg() == MI.getOperand(0).getReg()) &&
// CHECK-NEXT:         MI.getOperand(1).isReg() && MI.getOperand(1).getReg().isPhysical() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(1).getReg())) {
// CHECK-NEXT:       // x_mov.tied $dst, $src
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       // Operand: src
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     if (MI.getOperand(1).isReg() &&
// CHECK-NEXT:         (MI.getOperand(1).getReg() == MyTarget::X0) &&
// CHECK-NEXT:         MI.getOperand(0).isReg() && MI.getOperand(0).getReg().isPhysical() &&
// CHECK-NEXT:         MyTargetMCRegisterClasses[MyTarget::XRegsRegClassID].contains(MI.getOperand(0).getReg())) {
// CHECK-NEXT:       // x_mov.zero $dst
// CHECK-NEXT:       // Operand: dst
// CHECK-NEXT:       return true;
// CHECK-NEXT:     } // if
// CHECK-NEXT:     break;
// CHECK-NEXT:   } // case X_MOV
// CHECK-NEXT:   } // switch
// CHECK-NEXT:   return false;
// CHECK-NEXT: }

def MyTargetISA : InstrInfo;
def MyTargetAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}
def MyTarget : Target {
  let InstructionSet = MyTargetISA;
  let AssemblyWriters = [MyTargetAsmWriter];
}
