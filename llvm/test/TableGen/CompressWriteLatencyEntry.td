// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s

// Make sure that ReadAdvance entries are correctly processed.
// Not all ProcReadAdvance definitions implicitly inherit from SchedRead.
// Some ProcReadAdvances are subclasses of ReadAdvance.

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

// CHECK: extern const llvm::MCWriteLatencyEntry MyTargetWriteLatencyTable[] = {
// CHECK-NEXT:  { 0,  0}, // Invalid
// CHECK-NEXT:  { 1,  0}, // #1 Write_A_Write_C
// CHECK-NEXT:  { 1,  2} // #2 Write_B
// CHECK-NEXT: }; // MyTargetWriteLatencyTable

// CHECK: extern const llvm::MCReadAdvanceEntry MyTargetReadAdvanceTable[] = {
// CHECK-NEXT:  {0,  0,  0}, // Invalid
// CHECK-NEXT:  {0,  2,  1} // #1
// CHECK-NEXT: }; // MyTargetReadAdvanceTable

// CHECK:  static const llvm::MCSchedClassDesc SchedModel_ASchedClasses[] = {
// CHECK-NEXT:  {DBGFIELD("InvalidSchedClass")  8191, false, false, false, 0, 0,  0, 0,  0, 0},
// CHECK-NEXT:  {DBGFIELD("Inst_A")             1, false, false, false,  0, 0,  1, 1,  0, 0}, // #1
// CHECK-NEXT:  {DBGFIELD("Inst_B")             1, false, false, false,  0, 0,  2, 1,  0, 0}, // #2
// CHECK-NEXT:  {DBGFIELD("Inst_C")             1, false, false, false,  0, 0,  1, 1,  1, 1}, // #3
// CHECK-NEXT: }; // SchedModel_ASchedClasses

let SchedModel = SchedModel_A in {
  def Write_A : SchedWriteRes<[]>;
  def Write_B : SchedWriteRes<[]>;
  def Write_C : SchedWriteRes<[]>;

  def : InstRW<[Write_A], (instrs Inst_A)>;
  def : InstRW<[Write_B], (instrs Inst_B)>;
  def : InstRW<[Write_C, Read_D], (instrs Inst_C)>;

  def : ReadAdvance<Read_D, 1, [Write_B]>;
}

def ProcessorA: ProcessorModel<"ProcessorA", SchedModel_A, []>;
