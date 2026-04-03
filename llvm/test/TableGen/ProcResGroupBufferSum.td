// RUN: llvm-tblgen -gen-subtarget -DPOSITIVE_BUFFER -I %p/../../include %s 2>&1 | FileCheck %s --check-prefix=POSITIVE-BUFFER
// RUN: not llvm-tblgen -gen-subtarget -DDEFAULT_BUFFER -I %p/../../include %s 2>&1 | FileCheck %s --check-prefix=NON-POSITIVE-BUFFER
// RUN: not llvm-tblgen -gen-subtarget -DZERO_BUFFER -I %p/../../include %s 2>&1 | FileCheck %s --check-prefix=NON-POSITIVE-BUFFER

include "llvm/Target/Target.td"

def MyTarget : Target;

let OutOperandList = (outs), InOperandList = (ins) in {
  def Inst_A : Instruction;
}

let CompleteModel = 0 in {
  def SchedModel_A : SchedMachineModel;
}

def WriteInst_A : SchedWrite;

let SchedModel = SchedModel_A in {

def ResA : ProcResource<1> {
  let BufferSize = 3;
}

def ResB : ProcResource<1> {
  let BufferSize = 5;
}

def ResC : ProcResource<1> {
  let BufferSize = 2;
}

#ifdef POSITIVE_BUFFER
// Verify that ProcResGroupBufferSum computes BufferSize as the sum of
// individual resource buffer sizes: 3 + 5 + 2 = 10.
def ResGroup : ProcResGroupBufferSum<[ResA, ResB, ResC]>;

def : WriteRes<WriteInst_A, [ResGroup]>;
def : InstRW<[WriteInst_A], (instrs Inst_A)>;

} // let SchedModel = SchedModel_A

def ProcessorA : ProcessorModel<"ProcessorA", SchedModel_A, []>;

// POSITIVE-BUFFER: // {Name, NumUnits, SuperIdx, BufferSize, SubUnitsIdxBegin}
// POSITIVE-BUFFER: static const llvm::MCProcResourceDesc SchedModel_AProcResources[]
// POSITIVE-BUFFER-DAG: {"ResA", 1, 0, 3, nullptr}
// POSITIVE-BUFFER-DAG: {"ResB", 1, 0, 5, nullptr}
// POSITIVE-BUFFER-DAG: {"ResC", 1, 0, 2, nullptr}
// POSITIVE-BUFFER: {"ResGroup", 3, 0, 10, SchedModel_AProcResourceSubUnits
#endif

#ifdef DEFAULT_BUFFER
// Verify that using a resource with default BufferSize (-1) triggers an error.
def ResDefault : ProcResource<1>;
def ResGroup : ProcResGroupBufferSum<[ResA, ResDefault]>;
#endif

#ifdef ZERO_BUFFER
// Verify that using a resource with BufferSize = 0 triggers an error.
def ResZero : ProcResource<1> {
  let BufferSize = 0;
}
def ResGroup : ProcResGroupBufferSum<[ResA, ResZero]>;
#endif

// NON-POSITIVE-BUFFER: assertion failed: All resources in a ProcResGroupBufferSum must have BufferSize >= 1
