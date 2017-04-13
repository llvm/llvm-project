// RUN: llvm-tblgen -gen-global-isel -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

//===- Define the necessary boilerplate for our test target. --------------===//

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}

def complex : Operand<i32>, ComplexPattern<i32, 2, "SelectComplexPattern", []> {
  let MIOperandInfo = (ops i32imm, i32imm);
}
def gi_complex :
    GIComplexOperandMatcher<s32, (ops i32imm, i32imm), "selectComplexPattern">,
    GIComplexPatternEquiv<complex>;

def m1 : OperandWithDefaultOps <i32, (ops (i32 -1))>;
def Z : OperandWithDefaultOps <i32, (ops R0)>;
def m1Z : OperandWithDefaultOps <i32, (ops (i32 -1), R0)>;

//===- Test the function definition boilerplate. --------------------------===//

// CHECK: bool MyTargetInstructionSelector::selectImpl(MachineInstr &I) const {
// CHECK: MachineFunction &MF = *I.getParent()->getParent();
// CHECK: const MachineRegisterInfo &MRI = MF.getRegInfo();

//===- Test a pattern with multiple ComplexPattern operands. --------------===//
//

// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:    MachineInstr &MI0 = I;
// CHECK-NEXT:    if (MI0.getNumOperands() < 4)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if ((MI0.getOpcode() == TargetOpcode::G_SELECT) &&
// CHECK-NEXT:        ((/* dst */ (MRI.getType(MI0.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src1 */ (MRI.getType(MI0.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src2 */ (MRI.getType(MI0.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        (selectComplexPattern(MI0.getOperand(2), TempOp0, TempOp1)))) &&
// CHECK-NEXT:        ((/* src3 */ (MRI.getType(MI0.getOperand(3).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        (selectComplexPattern(MI0.getOperand(3), TempOp2, TempOp3))))) {
// CHECK-NEXT:          // (select:i32 GPR32:i32:$src1, complex:i32:$src2, complex:i32:$src3) => (INSN2:i32 GPR32:i32:$src1, complex:i32:$src3, complex:i32:$src2)
// CHECK-NEXT:          MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::INSN2));
// CHECK-NEXT:          MIB.add(MI0.getOperand(0)/*dst*/);
// CHECK-NEXT:          MIB.add(MI0.getOperand(1)/*src1*/);
// CHECK-NEXT:          MIB.add(TempOp2);
// CHECK-NEXT:          MIB.add(TempOp3);
// CHECK-NEXT:          MIB.add(TempOp0);
// CHECK-NEXT:          MIB.add(TempOp1);
// CHECK-NEXT:          for (const auto *FromMI : {&MI0, })
// CHECK-NEXT:            for (const auto &MMO : FromMI->memoperands())
// CHECK-NEXT:              MIB.addMemOperand(MMO);
// CHECK-NEXT:          I.eraseFromParent();
// CHECK-NEXT:          MachineInstr &NewI = *MIB;
// CHECK-NEXT:          constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:          return true;
// CHECK-NEXT:        }

def : GINodeEquiv<G_SELECT, select>;
def INSN2 : I<(outs GPR32:$dst), (ins GPR32:$src1, complex:$src2, complex:$src3), []>;
def : Pat<(select GPR32:$src1, complex:$src2, complex:$src3),
          (INSN2 GPR32:$src1, complex:$src3, complex:$src2)>;

//===- Test a simple pattern with regclass operands. ----------------------===//

// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:    MachineInstr &MI0 = I;
// CHECK-NEXT:    if (MI0.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if ((MI0.getOpcode() == TargetOpcode::G_ADD) &&
// CHECK-NEXT:        ((/* dst */ (MRI.getType(MI0.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src1 */ (MRI.getType(MI0.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src2 */ (MRI.getType(MI0.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(2).getReg(), MRI, TRI)))))) {

// CHECK-NEXT:      // (add:i32 GPR32:i32:$src1, GPR32:i32:$src2) => (ADD:i32 GPR32:i32:$src1, GPR32:i32:$src2)
// CHECK-NEXT:      I.setDesc(TII.get(MyTarget::ADD));
// CHECK-NEXT:      MachineInstr &NewI = I;
// CHECK-NEXT:      constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:      return true;
// CHECK-NEXT:    }
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }()) { return true; }


def ADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2),
            [(set GPR32:$dst, (add GPR32:$src1, GPR32:$src2))]>;

//===- Test a nested instruction match. -----------------------------------===//

// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:    MachineInstr &MI0 = I;
// CHECK-NEXT:    if (MI0.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if (!MI0.getOperand(1).isReg())
// CHECK-NEXT:      return false;
// CHECK-NEXT:    MachineInstr &MI1 = *MRI.getVRegDef(MI0.getOperand(1).getReg());
// CHECK-NEXT:    if (MI1.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if ((MI0.getOpcode() == TargetOpcode::G_MUL) &&
// CHECK-NEXT:        ((/* dst */ (MRI.getType(MI0.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* Operand 1 */ (MRI.getType(MI0.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        (((MI1.getOpcode() == TargetOpcode::G_ADD) &&
// CHECK-NEXT:        ((/* Operand 0 */ (MRI.getType(MI1.getOperand(0).getReg()) == (LLT::scalar(32))))) &&
// CHECK-NEXT:        ((/* src1 */ (MRI.getType(MI1.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI1.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src2 */ (MRI.getType(MI1.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI1.getOperand(2).getReg(), MRI, TRI))))))
// CHECK-NEXT:        ))) &&
// CHECK-NEXT:        ((/* src3 */ (MRI.getType(MI0.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(2).getReg(), MRI, TRI)))))) {
// CHECK-NEXT:        if (!isObviouslySafeToFold(MI1)) return false;
// CHECK-NEXT:        // (mul:i32 (add:i32 GPR32:i32:$src1, GPR32:i32:$src2), GPR32:i32:$src3)  =>  (MULADD:i32 GPR32:i32:$src1, GPR32:i32:$src2, GPR32:i32:$src3)
// CHECK-NEXT:     MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::MULADD));
// CHECK-NEXT:     MIB.add(MI0.getOperand(0)/*dst*/);
// CHECK-NEXT:     MIB.add(MI1.getOperand(1)/*src1*/);
// CHECK-NEXT:     MIB.add(MI1.getOperand(2)/*src2*/);
// CHECK-NEXT:     MIB.add(MI0.getOperand(2)/*src3*/);
// CHECK-NEXT:     for (const auto *FromMI : {&MI0, &MI1, })
// CHECK-NEXT:       for (const auto &MMO : FromMI->memoperands())
// CHECK-NEXT:         MIB.addMemOperand(MMO);
// CHECK-NEXT:     I.eraseFromParent();
// CHECK-NEXT:     MachineInstr &NewI = *MIB;
// CHECK-NEXT:     constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:     return true;
// CHECK-NEXT:   }

// We also get a second rule by commutativity.
// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:    MachineInstr &MI0 = I;
// CHECK-NEXT:    if (MI0.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if (!MI0.getOperand(2).isReg())
// CHECK-NEXT:      return false;
// CHECK-NEXT:    MachineInstr &MI1 = *MRI.getVRegDef(MI0.getOperand(2).getReg());
// CHECK-NEXT:    if (MI1.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if ((MI0.getOpcode() == TargetOpcode::G_MUL) &&
// CHECK-NEXT:        ((/* dst */ (MRI.getType(MI0.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src3 */ (MRI.getType(MI0.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* Operand 2 */ (MRI.getType(MI0.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        (((MI1.getOpcode() == TargetOpcode::G_ADD) &&
// CHECK-NEXT:        ((/* Operand 0 */ (MRI.getType(MI1.getOperand(0).getReg()) == (LLT::scalar(32))))) &&
// CHECK-NEXT:        ((/* src1 */ (MRI.getType(MI1.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI1.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src2 */ (MRI.getType(MI1.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI1.getOperand(2).getReg(), MRI, TRI))))))
// CHECK-NEXT:        )))) {
// CHECK-NEXT:        if (!isObviouslySafeToFold(MI1)) return false;
// CHECK-NEXT:        // (mul:i32 GPR32:i32:$src3, (add:i32 GPR32:i32:$src1, GPR32:i32:$src2))  =>  (MULADD:i32 GPR32:i32:$src1, GPR32:i32:$src2, GPR32:i32:$src3)
// CHECK-NEXT:     MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::MULADD));
// CHECK-NEXT:     MIB.add(MI0.getOperand(0)/*dst*/);
// CHECK-NEXT:     MIB.add(MI1.getOperand(1)/*src1*/);
// CHECK-NEXT:     MIB.add(MI1.getOperand(2)/*src2*/);
// CHECK-NEXT:     MIB.add(MI0.getOperand(1)/*src3*/);
// CHECK-NEXT:     for (const auto *FromMI : {&MI0, &MI1, })
// CHECK-NEXT:       for (const auto &MMO : FromMI->memoperands())
// CHECK-NEXT:         MIB.addMemOperand(MMO);
// CHECK-NEXT:     I.eraseFromParent();
// CHECK-NEXT:     MachineInstr &NewI = *MIB;
// CHECK-NEXT:     constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:     return true;
// CHECK-NEXT:   }

def MULADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2, GPR32:$src3),
               [(set GPR32:$dst,
                     (mul (add GPR32:$src1, GPR32:$src2), GPR32:$src3))]>;

//===- Test another simple pattern with regclass operands. ----------------===//

// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:    MachineInstr &MI0 = I;
// CHECK-NEXT:    if (MI0.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if ((MI0.getOpcode() == TargetOpcode::G_MUL) &&
// CHECK-NEXT:        ((/* dst */ (MRI.getType(MI0.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src1 */ (MRI.getType(MI0.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src2 */ (MRI.getType(MI0.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(2).getReg(), MRI, TRI)))))) {
// CHECK-NEXT:      // (mul:i32 GPR32:i32:$src1, GPR32:i32:$src2) => (MUL:i32 GPR32:i32:$src2, GPR32:i32:$src1)
// CHECK-NEXT:      MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::MUL));
// CHECK-NEXT:      MIB.add(MI0.getOperand(0)/*dst*/);
// CHECK-NEXT:      MIB.add(MI0.getOperand(2)/*src2*/);
// CHECK-NEXT:      MIB.add(MI0.getOperand(1)/*src1*/);
// CHECK-NEXT:     for (const auto *FromMI : {&MI0, })
// CHECK-NEXT:       for (const auto &MMO : FromMI->memoperands())
// CHECK-NEXT:         MIB.addMemOperand(MMO);
// CHECK-NEXT:      I.eraseFromParent();
// CHECK-NEXT:      MachineInstr &NewI = *MIB;
// CHECK-NEXT:      constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:      return true;
// CHECK-NEXT:    }
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }()) { return true; }

def MUL : I<(outs GPR32:$dst), (ins GPR32:$src2, GPR32:$src1),
             [(set GPR32:$dst, (mul GPR32:$src1, GPR32:$src2))]>;

//===- Test a pattern with ComplexPattern operands. -----------------------===//
//

// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:    MachineInstr &MI0 = I;
// CHECK-NEXT:    if (MI0.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if ((MI0.getOpcode() == TargetOpcode::G_SUB) &&
// CHECK-NEXT:        ((/* dst */ (MRI.getType(MI0.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src1 */ (MRI.getType(MI0.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src2 */ (MRI.getType(MI0.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        (selectComplexPattern(MI0.getOperand(2), TempOp0, TempOp1))))) {
// CHECK-NEXT:          // (sub:i32 GPR32:i32:$src1, complex:i32:$src2) => (INSN1:i32 GPR32:i32:$src1, complex:i32:$src2)
// CHECK-NEXT:          MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::INSN1));
// CHECK-NEXT:          MIB.add(MI0.getOperand(0)/*dst*/);
// CHECK-NEXT:          MIB.add(MI0.getOperand(1)/*src1*/);
// CHECK-NEXT:          MIB.add(TempOp0);
// CHECK-NEXT:          MIB.add(TempOp1);
// CHECK-NEXT:          for (const auto *FromMI : {&MI0, })
// CHECK-NEXT:            for (const auto &MMO : FromMI->memoperands())
// CHECK-NEXT:              MIB.addMemOperand(MMO);
// CHECK-NEXT:          I.eraseFromParent();
// CHECK-NEXT:          MachineInstr &NewI = *MIB;
// CHECK-NEXT:          constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:          return true;
// CHECK-NEXT:        }

def INSN1 : I<(outs GPR32:$dst), (ins GPR32:$src1, complex:$src2), []>;
def : Pat<(sub GPR32:$src1, complex:$src2), (INSN1 GPR32:$src1, complex:$src2)>;

//===- Test a simple pattern with a default operand. ----------------------===//
//

// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:    MachineInstr &MI0 = I;
// CHECK-NEXT:    if (MI0.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if ((MI0.getOpcode() == TargetOpcode::G_XOR) &&
// CHECK-NEXT:        ((/* dst */ (MRI.getType(MI0.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src1 */ (MRI.getType(MI0.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* Operand 2 */ (MRI.getType(MI0.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        (isOperandImmEqual(MI0.getOperand(2), -2, MRI))))) {
// CHECK-NEXT:      // (xor:i32 GPR32:i32:$src1, -2:i32) => (XORI:i32 GPR32:i32:$src1)
// CHECK-NEXT:      MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::XORI));
// CHECK-NEXT:      MIB.add(MI0.getOperand(0)/*dst*/);
// CHECK-NEXT:      MIB.addImm(-1);
// CHECK-NEXT:      MIB.add(MI0.getOperand(1)/*src1*/);
// CHECK-NEXT:      for (const auto *FromMI : {&MI0, })
// CHECK-NEXT:        for (const auto &MMO : FromMI->memoperands())
// CHECK-NEXT:          MIB.addMemOperand(MMO);
// CHECK-NEXT:      I.eraseFromParent();
// CHECK-NEXT:      MachineInstr &NewI = *MIB;
// CHECK-NEXT:      constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:      return true;
// CHECK-NEXT:    }
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }()) { return true; }

// The -2 is just to distinguish it from the 'not' case below.
def XORI : I<(outs GPR32:$dst), (ins m1:$src2, GPR32:$src1),
             [(set GPR32:$dst, (xor GPR32:$src1, -2))]>;

//===- Test a simple pattern with a default register operand. -------------===//
//

// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:    MachineInstr &MI0 = I;
// CHECK-NEXT:    if (MI0.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if ((MI0.getOpcode() == TargetOpcode::G_XOR) &&
// CHECK-NEXT:        ((/* dst */ (MRI.getType(MI0.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src1 */ (MRI.getType(MI0.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* Operand 2 */ (MRI.getType(MI0.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        (isOperandImmEqual(MI0.getOperand(2), -3, MRI))))) {
// CHECK-NEXT:      // (xor:i32 GPR32:i32:$src1, -3:i32) => (XOR:i32 GPR32:i32:$src1)
// CHECK-NEXT:      MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::XOR));
// CHECK-NEXT:      MIB.add(MI0.getOperand(0)/*dst*/);
// CHECK-NEXT:      MIB.addReg(MyTarget::R0);
// CHECK-NEXT:      MIB.add(MI0.getOperand(1)/*src1*/);
// CHECK-NEXT:      for (const auto *FromMI : {&MI0, })
// CHECK-NEXT:        for (const auto &MMO : FromMI->memoperands())
// CHECK-NEXT:          MIB.addMemOperand(MMO);
// CHECK-NEXT:      I.eraseFromParent();
// CHECK-NEXT:      MachineInstr &NewI = *MIB;
// CHECK-NEXT:      constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:      return true;
// CHECK-NEXT:    }
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }()) { return true; }

// The -3 is just to distinguish it from the 'not' case below and the other default op case above.
def XOR : I<(outs GPR32:$dst), (ins Z:$src2, GPR32:$src1),
            [(set GPR32:$dst, (xor GPR32:$src1, -3))]>;

//===- Test a simple pattern with a multiple default operands. ------------===//
//

// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:    MachineInstr &MI0 = I;
// CHECK-NEXT:    if (MI0.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if ((MI0.getOpcode() == TargetOpcode::G_XOR) &&
// CHECK-NEXT:        ((/* dst */ (MRI.getType(MI0.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* src1 */ (MRI.getType(MI0.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* Operand 2 */ (MRI.getType(MI0.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        (isOperandImmEqual(MI0.getOperand(2), -4, MRI))))) {
// CHECK-NEXT:      // (xor:i32 GPR32:i32:$src1, -4:i32) => (XORlike:i32 GPR32:i32:$src1)
// CHECK-NEXT:      MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::XORlike));
// CHECK-NEXT:      MIB.add(MI0.getOperand(0)/*dst*/);
// CHECK-NEXT:      MIB.addImm(-1);
// CHECK-NEXT:      MIB.addReg(MyTarget::R0);
// CHECK-NEXT:      MIB.add(MI0.getOperand(1)/*src1*/);
// CHECK-NEXT:      for (const auto *FromMI : {&MI0, })
// CHECK-NEXT:        for (const auto &MMO : FromMI->memoperands())
// CHECK-NEXT:          MIB.addMemOperand(MMO);
// CHECK-NEXT:      I.eraseFromParent();
// CHECK-NEXT:      MachineInstr &NewI = *MIB;
// CHECK-NEXT:      constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:      return true;
// CHECK-NEXT:    }
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }()) { return true; }

// The -4 is just to distinguish it from the other 'not' cases.
def XORlike : I<(outs GPR32:$dst), (ins m1Z:$src2, GPR32:$src1),
                [(set GPR32:$dst, (xor GPR32:$src1, -4))]>;

//===- Test a simple pattern with constant immediate operands. ------------===//
//
// This must precede the 3-register variants because constant immediates have
// priority over register banks.

// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:    MachineInstr &MI0 = I;
// CHECK-NEXT:    if (MI0.getNumOperands() < 3)
// CHECK-NEXT:      return false;
// CHECK-NEXT:    if ((MI0.getOpcode() == TargetOpcode::G_XOR) &&
// CHECK-NEXT:        ((/* dst */ (MRI.getType(MI0.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* Wm */ (MRI.getType(MI0.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:         ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(MI0.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT:        ((/* Operand 2 */ (MRI.getType(MI0.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:        (isOperandImmEqual(MI0.getOperand(2), -1, MRI))))) {
// CHECK-NEXT:      // (xor:i32 GPR32:i32:$Wm, -1:i32) => (ORN:i32 R0:i32, GPR32:i32:$Wm)
// CHECK-NEXT:      MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::ORN));
// CHECK-NEXT:      MIB.add(MI0.getOperand(0)/*dst*/);
// CHECK-NEXT:      MIB.addReg(MyTarget::R0);
// CHECK-NEXT:      MIB.add(MI0.getOperand(1)/*Wm*/);
// CHECK-NEXT:      for (const auto *FromMI : {&MI0, })
// CHECK-NEXT:        for (const auto &MMO : FromMI->memoperands())
// CHECK-NEXT:          MIB.addMemOperand(MMO);
// CHECK-NEXT:      I.eraseFromParent();
// CHECK-NEXT:      MachineInstr &NewI = *MIB;
// CHECK-NEXT:      constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:      return true;
// CHECK-NEXT:    }
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }()) { return true; }

def ORN : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def : Pat<(not GPR32:$Wm), (ORN R0, GPR32:$Wm)>;

//===- Test a pattern with an MBB operand. --------------------------------===//

// CHECK-LABEL: if ([&]() {
// CHECK-NEXT:      MachineInstr &MI0 = I;
// CHECK-NEXT:      if (MI0.getNumOperands() < 1)
// CHECK-NEXT:        return false;
// CHECK-NEXT:      if ((MI0.getOpcode() == TargetOpcode::G_BR) &&
// CHECK-NEXT:          ((/* target */ (MI0.getOperand(0).isMBB())))) {
                    
// CHECK-NEXT:       // (br (bb:Other):$target) => (BR (bb:Other):$target)
// CHECK-NEXT:       I.setDesc(TII.get(MyTarget::BR));
// CHECK-NEXT:       MachineInstr &NewI = I;
// CHECK-NEXT:       constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:       return true;
// CHECK-NEXT:    }
// CHECK-NEXT:    return false;
// CHECK-NEXT:  }()) { return true; }

def BR : I<(outs), (ins unknown:$target),
            [(br bb:$target)]>;
