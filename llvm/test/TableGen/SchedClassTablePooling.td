// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s -o - | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;

let OutOperandList = (outs), InOperandList = (ins) in {
  def InstA : Instruction;
  def InstB : Instruction;
}

let CompleteModel = 0 in {
  def ModelA : SchedMachineModel;
  def ModelB : SchedMachineModel;
}

let SchedModel = ModelA in {
  def WriteA0 : SchedWriteRes<[]>;
  def WriteB0 : SchedWriteRes<[]>;

  def : InstRW<[WriteA0], (instrs InstA)>;
  def : InstRW<[WriteB0], (instrs InstB)>;
}

let SchedModel = ModelB in {
  def WriteA1 : SchedWriteRes<[]>;
  def WriteB1 : SchedWriteRes<[]>;

  def : InstRW<[WriteA1], (instrs InstA)>;
  def : InstRW<[WriteB1], (instrs InstB)>;
}

def ProcessorA : ProcessorModel<"processor-a", ModelA, []>;
def ProcessorB : ProcessorModel<"processor-b", ModelB, []>;

// All builds share identical descriptors through compact index tables.
// CHECK: static const llvm::MCSchedClassDesc MyTargetSchedClassPool[] = {
// CHECK: static_assert(sizeof(MyTargetSchedClassPool) / sizeof(MyTargetSchedClassPool[0]) <= (1ULL << 16));
// CHECK: static const uint16_t ModelASchedClassIndices[] = {
// CHECK: static_assert(sizeof(ModelASchedClassIndices[0]) == 2);
// CHECK: static_assert(sizeof(ModelASchedClassIndices) / sizeof(ModelASchedClassIndices[0]) == 3);
// CHECK: static const uint16_t ModelBSchedClassIndices[] = {
// CHECK: static_assert(sizeof(ModelBSchedClassIndices[0]) == 2);
// CHECK: static_assert(sizeof(ModelBSchedClassIndices) / sizeof(ModelBSchedClassIndices[0]) == 3);

// Names are stored separately and indexed by the original scheduling class.
// CHECK: static const uint32_t MyTargetSchedClassNameOffsets[] = {
// CHECK: static_assert(sizeof(MyTargetSchedClassNameOffsets[0]) == 4);
// CHECK: static_assert(sizeof(MyTargetSchedClassNameOffsets) / sizeof(MyTargetSchedClassNameOffsets[0]) == 3);

// CHECK: extern const llvm::MCSchedModel MyTargetSchedModels[] = {
// CHECK: { // ModelA
// CHECK:   MyTargetSchedClassPool,
// CHECK:   DBGVAL_OR_NULLPTR(&MyTargetSchedClassNames), // SchedClassNames
// CHECK-NEXT:   DBGVAL_OR_NULLPTR(MyTargetSchedClassNameOffsets), // SchedClassNameOffsets
// CHECK:   ModelASchedClassIndices,

// CHECK: { // ModelB
// CHECK:   MyTargetSchedClassPool,
// CHECK:   DBGVAL_OR_NULLPTR(&MyTargetSchedClassNames), // SchedClassNames
// CHECK-NEXT:   DBGVAL_OR_NULLPTR(MyTargetSchedClassNameOffsets), // SchedClassNameOffsets
// CHECK:   ModelBSchedClassIndices,
