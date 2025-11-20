// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s

// Make sure we don't form ReadAdvances with ValidWrites entries that are not
// associated with any instructions.

include "llvm/Target/Target.td"

def TargetX : Target;

def WriteX : SchedWrite;
def WriteY : SchedWrite;
def ReadX : SchedRead;

def InstX : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins);
  let SchedRW = [WriteX, ReadX];
}

def SchedModelX: SchedMachineModel {
  let CompleteModel = 0;
}

let SchedModel = SchedModelX in {
  def : ReadAdvance<ReadX, 1, [WriteX, WriteY]>;
  // CHECK: error: ReadAdvance referencing a ValidWrite that is not used by any instruction (WriteY)
}

def ProcessorX: ProcessorModel<"ProcessorX", SchedModelX, []>;
