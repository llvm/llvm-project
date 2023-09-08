// RUN: llvm-tblgen -gen-subtarget -DCORRECT -I %p/../../include %s 2>&1 | \
// RUN:   FileCheck %s  --check-prefix=CORRECT

// RUN: not llvm-tblgen -gen-subtarget -DWRONG_SIZE -I %p/../../include %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=WRONG_SIZE

// RUN: not llvm-tblgen -gen-subtarget -DWRONG_VALUE -I %p/../../include %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=WRONG_VALUE

// RUN: not llvm-tblgen -gen-subtarget -DNEGATIVE_INVALID -I %p/../../include %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NEGATIVE_INVALID

// Make sure that AcquireAtCycle in WriteRes is used to generate the
// correct data.

include "llvm/Target/Target.td"

def MyTarget : Target;

let BufferSize = 0 in {
def ResX0 : ProcResource<1>; // X0
def ResX1 : ProcResource<1>; // X1
def ResX2 : ProcResource<1>; // X2
}

let OutOperandList = (outs), InOperandList = (ins) in {
  def Inst_A : Instruction;
  def Inst_B : Instruction;
}

let CompleteModel = 0 in {
  def SchedModel_A: SchedMachineModel;
}

def WriteInst_A : SchedWrite;
def WriteInst_B : SchedWrite;

let SchedModel = SchedModel_A in {
// Check the generated data when there are no semantic issues.
#ifdef CORRECT
// CORRECT-LABEL: llvm::MCWriteProcResEntry MyTargetWriteProcResTable[] = {
// CORRECT-NEXT: { 0, 0, 0 }, // Invalid
def : WriteRes<WriteInst_A, [ResX0, ResX1, ResX2]> {
// CORRECT-NEXT: { 1, 2, 0}, // #1
// CORRECT-NEXT: { 2, 4, 1}, // #2
// CORRECT-NEXT: { 3, 3, 2}, // #3
    let ReleaseAtCycles = [2, 4, 3];
    let AcquireAtCycles = [0, 1, 2];
}
def : WriteRes<WriteInst_B, [ResX2]> {
// If unspecified, AcquireAtCycle is set to 0.
// CORRECT-NEXT: { 3, 1, 0} // #4
    let ReleaseAtCycles = [1];
}
#endif // CORRECT

#ifdef WRONG_SIZE
// WRONG_SIZE: AcquireAtCycle.td:[[@LINE+1]]:1: error: Inconsistent resource cycles: size(AcquireAtCycles) != size(ProcResources): 2 vs 3
def : WriteRes<WriteInst_A, [ResX0, ResX1, ResX2]> {
    let ReleaseAtCycles = [2, 4, 3];
    let AcquireAtCycles = [0, 1];
}
#endif

#ifdef WRONG_VALUE
// WRONG_VALUE: AcquireAtCycle.td:[[@LINE+1]]:1: error: Inconsistent resource cycles: AcquireAtCycles < ReleaseAtCycles must hold
def : WriteRes<WriteInst_A, [ResX0, ResX1, ResX2]> {
    let ReleaseAtCycles = [2, 4, 3];
    let AcquireAtCycles = [0, 1, 8];
}
#endif

#ifdef NEGATIVE_INVALID
// NEGATIVE_INVALID: AcquireAtCycle.td:[[@LINE+1]]:1: error: Invalid value: AcquireAtCycle must be a non-negative value.
def : WriteRes<WriteInst_A, [ResX0]> {
    let ReleaseAtCycles = [2];
    let AcquireAtCycles = [-1];
}
#endif

def : InstRW<[WriteInst_A], (instrs Inst_A)>;
def : InstRW<[WriteInst_B], (instrs Inst_B)>;
}

def ProcessorA: ProcessorModel<"ProcessorA", SchedModel_A, []>;

