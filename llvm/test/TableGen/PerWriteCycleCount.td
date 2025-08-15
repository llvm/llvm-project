// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s
// RUN: not llvm-tblgen -gen-subtarget -I %p/../../include -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s

// Make sure that ReadAdvance entries with multiple writes are correctly
// handled.

include "llvm/Target/Target.td"

def MyTarget : Target;

let OutOperandList = (outs), InOperandList = (ins) in {
  def Inst_A : Instruction;
  def Inst_B : Instruction;
  def Inst_C : Instruction;
}

let CompleteModel = 0 in {
  def SchedModel_A: SchedMachineModel;
}

def Read_D : SchedRead;
def Read_E : SchedRead;

// CHECK: extern const llvm::MCReadAdvanceEntry MyTargetReadAdvanceTable[] = {
// CHECK-NEXT:  {0,  0,  0}, // Invalid
// CHECK-NEXT:  {0,  1,  1}, // #1
// CHECK-NEXT:  {0,  2,  3}, // #2
// CHECK-NEXT:  {0,  3,  2} // #3
// CHECK-NEXT: }; // MyTargetReadAdvanceTable

let SchedModel = SchedModel_A in {
  def Write_A : SchedWriteRes<[]>;
  def Write_B : SchedWriteRes<[]>;
  def Write_C : SchedWriteRes<[]>;

  def : InstRW<[Write_A], (instrs Inst_A)>;
  def : InstRW<[Write_B], (instrs Inst_B)>;
  def : InstRW<[Write_C, Read_D], (instrs Inst_C)>;

  def : ReadAdvance<Read_D, 2, [Write_A, Write_B, Write_C], [-1, 1]>;

#ifdef ERROR1
// ERROR1: error: assertion failed: cannot have more `tunables' than `writes'
  def : ReadAdvance<Read_E, 2, [Write_A, Write_B, Write_C], [1, 2, 3, 4]>;
#endif
}

def ProcessorA: ProcessorModel<"ProcessorA", SchedModel_A, []>;
