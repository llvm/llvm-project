// The test incorrectly classifies the pattern as a typecast and it ends up with a
// typecast DAG with zero arguements, leading to llvm-tblgen crash.
// This test will check if the error is gracefully handled without any crash.
 
// RUN: not llvm-tblgen -gen-dag-isel -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

class MyReg<string n>
  : Register<n> {
  let Namespace = "MyTarget";
}

def X0 : MyReg<"x0">;
def X1 : MyReg<"x1">;

def XRegs : RegisterClass<"MyTarget", [i64], 64, (add X0, X1)>;

class TestInstruction : Instruction {
  let Size = 2;
  let Namespace = "MyTarget";
  let hasSideEffects = false;
  let hasExtraSrcRegAllocReq = false;
  let hasExtraDefRegAllocReq = false;

  field bits<16> Inst;
  bits<3> dst;
  bits<3> src;
  bits<3> opcode;

  let Inst{2-0} = dst;
  let Inst{5-3} = src;
  let Inst{7-5} = opcode;
}

def MY_LOAD : TestInstruction {
  let OutOperandList = (outs XRegs:$dst);
  let InOperandList = (ins ptr_rc:$ptr);
  let AsmString = "my_load $dst, $ptr";
  let opcode = 0;
}

// CHECK: [[@LINE+2]]:1: error: {{.*}} Type cast only takes one operand!
// CHECK: [[@LINE+1]]:1: error: {{.*}} cannot update node type from unknown operand!
def : Pat<
  (i64 (load (iPTR:$src))),
  (MY_LOAD $val, $src)
>;

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }
