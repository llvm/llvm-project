// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s

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
}

def ProcessorA: ProcessorModel<"ProcessorA", SchedModel_A, []>;
