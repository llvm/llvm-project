// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s 2>&1 | FileCheck %s
// Check if it is valid MCSchedClassDesc if didn't have the resources.

include "llvm/Target/Target.td"

def MyTarget : Target;

let OutOperandList = (outs), InOperandList = (ins) in {
  def Inst_A : Instruction;
  def Inst_B : Instruction;
}

let CompleteModel = 0 in {
  def SchedModel_A: SchedMachineModel;
  def SchedModel_B: SchedMachineModel;
  def SchedModel_C: SchedMachineModel;
}

// Inst_B didn't have the resoures, and it is invalid.
// CHECK: SchedModel_ASchedClassIndices[] = {
// CHECK-NEXT: 0, // #0
// CHECK-NEXT: 1, // #1
// CHECK-NEXT: 0, // #2
let SchedModel = SchedModel_A in {
  def Write_A : SchedWriteRes<[]>;
  def : InstRW<[Write_A], (instrs Inst_A)>;
}

// Inst_A didn't have the resoures, and it is invalid.
// CHECK: SchedModel_BSchedClassIndices[] = {
// CHECK-NEXT: 0, // #0
// CHECK-NEXT: 0, // #1
// CHECK-NEXT: 1, // #2
let SchedModel = SchedModel_B in {
  def Write_B: SchedWriteRes<[]>;
  def : InstRW<[Write_B], (instrs Inst_B)>;
}

// CHECK: SchedModel_CSchedClassIndices[] = {
// CHECK-NEXT: 0, // #0
// CHECK-NEXT: 1, // #1
// CHECK-NEXT: 1, // #2
let SchedModel = SchedModel_C in {
  def Write_C: SchedWriteRes<[]>;
  def : InstRW<[Write_C], (instrs Inst_A, Inst_B)>;
}

def ProcessorA: ProcessorModel<"ProcessorA", SchedModel_A, []>;
def ProcessorB: ProcessorModel<"ProcessorB", SchedModel_B, []>;
def ProcessorC: ProcessorModel<"ProcessorC", SchedModel_C, []>;
