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

// Debug builds retain the per-processor tables so class names are available.
// CHECK: #if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
// CHECK: static const llvm::MCSchedClassDesc ModelASchedClasses[] = {
// CHECK: static const llvm::MCSchedClassDesc ModelBSchedClasses[] = {

// Release builds share identical descriptors through a compact index table.
// CHECK: #else
// CHECK: static const llvm::MCSchedClassDesc MyTargetSchedClassPool[] = {
// CHECK: static_assert(sizeof(MyTargetSchedClassPool) / sizeof(MyTargetSchedClassPool[0]) <= (1ULL << 16));
// CHECK: static const uint16_t ModelASchedClassIndices[] = {
// CHECK: static_assert(sizeof(ModelASchedClassIndices[0]) == 2);
// CHECK: static_assert(sizeof(ModelASchedClassIndices) / sizeof(ModelASchedClassIndices[0]) == 3);
// CHECK: static const uint16_t ModelBSchedClassIndices[] = {
// CHECK: static_assert(sizeof(ModelBSchedClassIndices[0]) == 2);
// CHECK: static_assert(sizeof(ModelBSchedClassIndices) / sizeof(ModelBSchedClassIndices[0]) == 3);

// CHECK: static const llvm::MCSchedModel ModelA = {
// CHECK: #if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
// CHECK-NEXT:   ModelASchedClasses,
// CHECK-NEXT: #else
// CHECK-NEXT:   MyTargetSchedClassPool,
// CHECK-NEXT: #endif
// CHECK: #if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
// CHECK-NEXT:   nullptr,
// CHECK-NEXT: #else
// CHECK-NEXT:   ModelASchedClassIndices,
// CHECK-NEXT: #endif

// CHECK: static const llvm::MCSchedModel ModelB = {
// CHECK: #if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
// CHECK-NEXT:   ModelBSchedClasses,
// CHECK-NEXT: #else
// CHECK-NEXT:   MyTargetSchedClassPool,
// CHECK-NEXT: #endif
// CHECK: #if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
// CHECK-NEXT:   nullptr,
// CHECK-NEXT: #else
// CHECK-NEXT:   ModelBSchedClassIndices,
// CHECK-NEXT: #endif
