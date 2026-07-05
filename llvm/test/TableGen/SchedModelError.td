// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s -DFILE=%s

include "llvm/Target/Target.td"

def TestTarget : Target;

// CHECK: [[FILE]]:[[@LINE+1]]:5: error: No schedule information for instruction 'TestInst' in SchedMachineModel 'TestSchedModel'
def TestInst : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins);
  bits<8> Inst = 0b00101010;
}

def TestSchedModel : SchedMachineModel {
  let CompleteModel = 1;
}

def TestProcessor : ProcessorModel<"testprocessor", TestSchedModel, []>;
