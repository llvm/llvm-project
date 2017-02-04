// RUN: llvm-tblgen -gen-global-isel -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

//===- Define the necessary boilerplate for our test target. --------------===//

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def R0 : Register<"r0">;
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
// CHECK: const MachineRegisterInfo &MRI = I.getParent()->getParent()->getRegInfo();


//===- Test a simple pattern with regclass operands. ----------------------===//

// CHECK: if ((I.getOpcode() == TargetOpcode::G_ADD) &&
// CHECK-NEXT: (((MRI.getType(I.getOperand(0).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(0).getReg(), MRI, TRI))))) &&
// CHECK-NEXT: (((MRI.getType(I.getOperand(1).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(1).getReg(), MRI, TRI))))) &&
// CHECK-NEXT: (((MRI.getType(I.getOperand(2).getReg()) == (LLT::scalar(32))) &&
// CHECK-NEXT:  ((&RBI.getRegBankFromRegClass(MyTarget::GPR32RegClass) == RBI.getRegBank(I.getOperand(2).getReg(), MRI, TRI)))))) {

// CHECK-NEXT:   // (add:i32 GPR32:i32:$src1, GPR32:i32:$src2) => (ADD:i32 GPR32:i32:$src1, GPR32:i32:$src2)
// CHECK-NEXT:   I.setDesc(TII.get(MyTarget::ADD));
// CHECK-NEXT:   constrainSelectedInstRegOperands(I, TII, TRI, RBI);
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

def ADD : I<(outs GPR32:$dst), (ins GPR32:$src1, GPR32:$src2),
            [(set GPR32:$dst, (add GPR32:$src1, GPR32:$src2))]>;

//===- Test a pattern with an MBB operand. --------------------------------===//

// CHECK: if ((I.getOpcode() == TargetOpcode::G_BR) &&
// CHECK-NEXT: (((I.getOperand(0).isMBB())))) {

// CHECK-NEXT:   // (br (bb:Other):$target) => (BR (bb:Other):$target)
// CHECK-NEXT:   I.setDesc(TII.get(MyTarget::BR));
// CHECK-NEXT:   constrainSelectedInstRegOperands(I, TII, TRI, RBI);
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

def BR : I<(outs), (ins unknown:$target),
            [(br bb:$target)]>;
