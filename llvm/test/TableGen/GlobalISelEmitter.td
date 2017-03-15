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

//===- Test the function definition boilerplate. --------------------------===//

// CHECK: bool MyTargetInstructionSelector::selectImpl(MachineInstr &I) const {
// CHECK: MachineFunction &MF = *I.getParent()->getParent();
// CHECK: const MachineRegisterInfo &MRI = MF.getRegInfo();

//===- Test a simple pattern with regclass operands. ----------------------===//

// CHECK-LABEL: if ((I.getOpcode() == TargetOpcode::G_ADD) &&
// CHECK-NEXT: ((/* dst */ (MRI.getType(I.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT: ((/* src1 */ (MRI.getType(I.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT: ((/* src2 */ (MRI.getType(I.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(2).getReg(), MRI, TRI)))))) {

// CHECK-NEXT:   // (add:i32 GPR32:i32:$src1, GPR32:i32:$src2) => (ADD:i32 GPR32:i32:$src1, GPR32:i32:$src2)
// CHECK-NEXT:   I.setDesc(TII.get(MyTarget::ADD));
// CHECK-NEXT:   MachineInstr &NewI = I;
// CHECK:        constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

def ADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2),
            [(set GPR32:$dst, (add GPR32:$src1, GPR32:$src2))]>;

// CHECK-LABEL: if ((I.getOpcode() == TargetOpcode::G_MUL) &&
// CHECK-NEXT: ((/* dst */ (MRI.getType(I.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT: ((/* src1 */ (MRI.getType(I.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT: ((/* src2 */ (MRI.getType(I.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(2).getReg(), MRI, TRI)))))) {

// CHECK-NEXT:   // (mul:i32 GPR32:i32:$src1, GPR32:i32:$src2) => (MUL:i32 GPR32:i32:$src2, GPR32:i32:$src1)
// CHECK-NEXT:   MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::MUL));
// CHECK-NEXT:   MIB.add(I.getOperand(0)/*dst*/);
// CHECK-NEXT:   MIB.add(I.getOperand(2)/*src2*/);
// CHECK-NEXT:   MIB.add(I.getOperand(1)/*src1*/);
// CHECK-NEXT:   MIB.setMemRefs(I.memoperands_begin(), I.memoperands_end());
// CHECK-NEXT:   I.eraseFromParent();
// CHECK-NEXT:   MachineInstr &NewI = *MIB;
// CHECK:        constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

def MUL : I<(outs GPR32:$dst), (ins GPR32:$src2, GPR32:$src1),
             [(set GPR32:$dst, (mul GPR32:$src1, GPR32:$src2))]>;

//===- Test a simple pattern with constant immediate operands. ------------===//
//
// This must precede the 3-register variants because constant immediates have
// priority over register banks.

// CHECK-LABEL: if ((I.getOpcode() == TargetOpcode::G_XOR) &&
// CHECK-NEXT: ((/* dst */ (MRI.getType(I.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT: ((/* Wm */ (MRI.getType(I.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT: ((/* Operand 2 */ (MRI.getType(I.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT: (isOperandImmEqual(I.getOperand(2), -1, MRI))))) {

// CHECK-NEXT:   // (xor:i32 GPR32:i32:$Wm, -1:i32) => (ORN:i32 R0:i32, GPR32:i32:$Wm)
// CHECK-NEXT:   MachineInstrBuilder MIB = BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(MyTarget::ORN));
// CHECK-NEXT:   MIB.add(I.getOperand(0)/*dst*/);
// CHECK-NEXT:   MIB.addReg(MyTarget::R0);
// CHECK-NEXT:   MIB.add(I.getOperand(1)/*Wm*/);
// CHECK-NEXT:   MIB.setMemRefs(I.memoperands_begin(), I.memoperands_end());
// CHECK-NEXT:   I.eraseFromParent();
// CHECK-NEXT:   MachineInstr &NewI = *MIB;
// CHECK:        constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

def ORN : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2), []>;
def : Pat<(not GPR32:$Wm), (ORN R0, GPR32:$Wm)>;

//===- Test a pattern with an MBB operand. --------------------------------===//

// CHECK-LABEL: if ((I.getOpcode() == TargetOpcode::G_BR) &&
// CHECK-NEXT: ((/* target */ (I.getOperand(0).isMBB())))) {

// CHECK-NEXT:   // (br (bb:Other):$target) => (BR (bb:Other):$target)
// CHECK-NEXT:   I.setDesc(TII.get(MyTarget::BR));
// CHECK-NEXT:   MachineInstr &NewI = I;
// CHECK:        constrainSelectedInstRegOperands(NewI, TII, TRI, RBI);
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

def BR : I<(outs), (ins unknown:$target),
            [(br bb:$target)]>;
