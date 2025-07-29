// RUN: llvm-tblgen -gen-compress-inst-emitter -I %p/../../../include %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }
def ArchAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyWriters = [ArchAsmWriter];
}

def Reg0 : Register<"reg0"> {
  let HWEncoding{4-0} = 0;
}
def Reg1 : Register<"reg1"> {
  let HWEncoding{4-0} = 1;
}

def Regs : RegisterClass<"Arch", [i32], 32, (add Reg0, Reg1)>;
def RegsC : RegisterClass<"Arch", [i32], 32, (sub Regs, Reg0)>;

def simm6 : Operand<i32>, ImmLeaf<i32, [{return isInt<6>(Imm);}]> {
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isInt<6>(Imm);
  }];
}

def simm12 : Operand<i32>, ImmLeaf<i32, [{return isInt<12>(Imm);}]> {
  let MCOperandPredicate = [{
    int64_t Imm;
    if (!MCOp.evaluateAsConstantImm(Imm))
      return false;
    return isInt<12>(Imm);
  }];
}

def MemOpnd : Operand<iPTR> {
  let MIOperandInfo = (ops Regs, simm12);
}

def MemOpndC : Operand<iPTR> {
  let MIOperandInfo = (ops RegsC, simm6);
}

def BigInst : Instruction {
  let Namespace = "MyNS";
  let OutOperandList = (outs Regs:$dst);
  let InOperandList = (ins MemOpnd:$addr);
  let Size = 4;
  let AsmString = "big $dst, $addr";
}

def SmallInst : Instruction {
  let Namespace = "MyNS";
  let OutOperandList = (outs RegsC:$dst);
  let InOperandList = (ins MemOpndC:$addr);
  let Size = 2;
  let AsmString = "small $dst, $addr";
}

def : CompressPat<(BigInst RegsC:$dst, RegsC:$src, simm6:$imm),
                  (SmallInst RegsC:$dst, RegsC:$src, simm6:$imm)>;

def BigInst2 : Instruction {
  let Namespace = "MyNS";
  let OutOperandList = (outs Regs:$dst);
  let InOperandList = (ins MemOpnd:$addr);
  let Size = 4;
  let AsmString = "big $dst, $addr";
}

def SmallInst2 : Instruction {
  let Namespace = "MyNS";
  let OutOperandList = (outs RegsC:$dst);
  let InOperandList = (ins RegsC:$src, simm6:$imm);
  let Size = 2;
  let AsmString = "small $dst, $src, $imm";
}

def : CompressPat<(BigInst2 RegsC:$dst, RegsC:$src, simm6:$imm),
                  (SmallInst2 RegsC:$dst, RegsC:$src, simm6:$imm)>;

def BigInst3 : Instruction {
  let Namespace = "MyNS";
  let OutOperandList = (outs Regs:$dst);
  let InOperandList = (ins Regs:$src, simm12:$imm);
  let Size = 4;
  let AsmString = "big $dst, $src, $imm";
}

def SmallInst3 : Instruction {
  let Namespace = "MyNS";
  let OutOperandList = (outs RegsC:$dst);
  let InOperandList = (ins MemOpndC:$addr);
  let Size = 2;
  let AsmString = "small $dst, $addr";
}

def : CompressPat<(BigInst3 RegsC:$dst, RegsC:$src, simm6:$imm),
                  (SmallInst3 RegsC:$dst, RegsC:$src, simm6:$imm)>;

// CHECK-LABEL: ArchValidateMCOperandForCompress
// CHECK: // simm6
// CHECK: return isInt<6>(Imm);

// CHECK-LABEL: compressInst
// CHECK: case Arch::BigInst
// CHECK-NEXT: if (MI.getOperand(0).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT: MI.getOperand(1).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(1).getReg()) &&
// CHECK-NEXT: ArchValidateMCOperandForCompress(MI.getOperand(2), STI, 1 /* simm6 */)) {
// CHECK-NEXT: // small $dst, $addr
// CHECK-NEXT: OutInst.setOpcode(Arch::SmallInst);
// CHECK-NEXT: // Operand: dst
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT: // Operand: addr
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(2));
// CHECK-NEXT: OutInst.setLoc(MI.getLoc());
// CHECK-NEXT: return true;
// CHECK-NEXT: } // if
// CHECK: case Arch::BigInst2
// CHECK-NEXT: if (MI.getOperand(0).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT: MI.getOperand(1).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(1).getReg()) &&
// CHECK-NEXT: ArchValidateMCOperandForCompress(MI.getOperand(2), STI, 1 /* simm6 */)) {
// CHECK-NEXT: // small $dst, $src, $imm
// CHECK-NEXT: OutInst.setOpcode(Arch::SmallInst2);
// CHECK-NEXT: // Operand: dst
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT: // Operand: src
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT: // Operand: imm
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(2));
// CHECK-NEXT: OutInst.setLoc(MI.getLoc());
// CHECK-NEXT: return true;
// CHECK-NEXT: } // if
// CHECK: case Arch::BigInst3
// CHECK-NEXT: if (MI.getOperand(0).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT: MI.getOperand(1).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(1).getReg()) &&
// CHECK-NEXT: ArchValidateMCOperandForCompress(MI.getOperand(2), STI, 1 /* simm6 */)) {
// CHECK-NEXT: // small $dst, $addr
// CHECK-NEXT: OutInst.setOpcode(Arch::SmallInst3);
// CHECK-NEXT: // Operand: dst
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT: // Operand: addr
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(2));
// CHECK-NEXT: OutInst.setLoc(MI.getLoc());
// CHECK-NEXT: return true;
// CHECK-NEXT: } // if

// CHECK-LABEL: ArchValidateMCOperandForUncompress
// CHECK: // simm6
// CHECK: return isInt<6>(Imm);

// CHECK-LABEL: uncompressInst
// CHECK: case Arch::SmallInst:
// CHECK-NEXT:  if (MI.getOperand(0).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT: MI.getOperand(1).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(1).getReg()) &&
// CHECK-NEXT: ArchValidateMCOperandForUncompress(MI.getOperand(2), STI, 1 /* simm6 */) &&
// CHECK-NEXT: ArchValidateMCOperandForUncompress(MI.getOperand(2), STI, 2 /* simm12 */))
// CHECK-NEXT: // big $dst, $addr
// CHECK-NEXT: OutInst.setOpcode(Arch::BigInst);
// CHECK-NEXT: // Operand: dst
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT: // Operand: addr
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(2));
// CHECK-NEXT: OutInst.setLoc(MI.getLoc());
// CHECK-NEXT: return true;
// CHECK-NEXT: } // if
// CHECK: case Arch::SmallInst2:
// CHECK-NEXT:  if (MI.getOperand(0).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT: MI.getOperand(1).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(1).getReg()) &&
// CHECK-NEXT: ArchValidateMCOperandForUncompress(MI.getOperand(2), STI, 1 /* simm6 */) &&
// CHECK-NEXT: ArchValidateMCOperandForUncompress(MI.getOperand(2), STI, 2 /* simm12 */)) {
// CHECK-NEXT: // big $dst, $addr
// CHECK-NEXT: OutInst.setOpcode(Arch::BigInst2);
// CHECK-NEXT: // Operand: dst
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT: // Operand: addr
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(2));
// CHECK-NEXT: OutInst.setLoc(MI.getLoc());
// CHECK-NEXT: return true;
// CHECK-NEXT: } // if
// CHECK: case Arch::SmallInst3:
// CHECK-NEXT:  if (MI.getOperand(0).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT: MI.getOperand(1).isReg() &&
// CHECK-NEXT: ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(1).getReg()) &&
// CHECK-NEXT: ArchValidateMCOperandForUncompress(MI.getOperand(2), STI, 1 /* simm6 */) &&
// CHECK-NEXT: ArchValidateMCOperandForUncompress(MI.getOperand(2), STI, 2 /* simm12 */)) {
// CHECK-NEXT: // big $dst, $src, $imm
// CHECK-NEXT: OutInst.setOpcode(Arch::BigInst3);
// CHECK-NEXT: // Operand: dst
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(0));
// CHECK-NEXT: // Operand: src
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(1));
// CHECK-NEXT: // Operand: imm
// CHECK-NEXT: OutInst.addOperand(MI.getOperand(2));
// CHECK-NEXT: OutInst.setLoc(MI.getLoc());
// CHECK-NEXT: return true;
// CHECK-NEXT: } // if

// CHECK-LABEL: ArchValidateMachineOperand
// CHECK: // simm6
// CHECK: return isInt<6>(Imm);

// CHECK-LABEL: isCompressibleInst
// CHECK:  case Arch::BigInst: {
// CHECK-NEXT:  if (MI.getOperand(0).isReg() && MI.getOperand(0).getReg().isPhysical() &&
// CHECK-NEXT:    ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:    MI.getOperand(1).isReg() && MI.getOperand(1).getReg().isPhysical() &&
// CHECK-NEXT:    ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(1).getReg()) &&
// CHECK-NEXT:    MI.getOperand(2).isImm() &&
// CHECK-NEXT:    ArchValidateMachineOperand(MI.getOperand(2), &STI, 1 /* simm6 */)) {
// CHECK-NEXT:    // small $dst, $addr
// CHECK-NEXT:    // Operand: dst
// CHECK-NEXT:    // Operand: addr
// CHECK-NEXT:    return true;
// CHECK-NEXT: } // if
// CHECK:  case Arch::BigInst2: {
// CHECK-NEXT:  if (MI.getOperand(0).isReg() && MI.getOperand(0).getReg().isPhysical() &&
// CHECK-NEXT:    ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:    MI.getOperand(1).isReg() && MI.getOperand(1).getReg().isPhysical() &&
// CHECK-NEXT:    ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(1).getReg()) &&
// CHECK-NEXT:    MI.getOperand(2).isImm() &&
// CHECK-NEXT:    ArchValidateMachineOperand(MI.getOperand(2), &STI, 1 /* simm6 */)) {
// CHECK-NEXT:    // small $dst, $src, $imm
// CHECK-NEXT:    // Operand: dst
// CHECK-NEXT:    // Operand: src
// CHECK-NEXT:    // Operand: imm
// CHECK-NEXT:    return true;
// CHECK-NEXT: } // if
// CHECK:  case Arch::BigInst3: {
// CHECK-NEXT:  if (MI.getOperand(0).isReg() && MI.getOperand(0).getReg().isPhysical() &&
// CHECK-NEXT:    ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(0).getReg()) &&
// CHECK-NEXT:    MI.getOperand(1).isReg() && MI.getOperand(1).getReg().isPhysical() &&
// CHECK-NEXT:    ArchMCRegisterClasses[Arch::RegsCRegClassID].contains(MI.getOperand(1).getReg()) &&
// CHECK-NEXT:    MI.getOperand(2).isImm() &&
// CHECK-NEXT:    ArchValidateMachineOperand(MI.getOperand(2), &STI, 1 /* simm6 */)) {
// CHECK-NEXT:    // small $dst, $addr
// CHECK-NEXT:    // Operand: dst
// CHECK-NEXT:    // Operand: addr
// CHECK-NEXT:    return true;
// CHECK-NEXT: } // if
