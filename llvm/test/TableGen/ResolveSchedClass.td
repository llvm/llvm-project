// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s -o - | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def FeatureFoo : SubtargetFeature<"foo", "HasFoo", "true", "enable foo">;

def ResX0 : ProcResource<1>;

let OutOperandList = (outs), InOperandList = (ins) in
def Inst_A : Instruction;

def SchedModel_A: SchedMachineModel {
  let CompleteModel = false;
}

let SchedModel = SchedModel_A in {
def SchedWriteResA : SchedWriteRes<[ResX0]> {
  let Latency = 2;
}
def SchedWriteResB : SchedWriteRes<[ResX0]> {
  let Latency = 4;
}

// Check SchedPredicate with subtarget feature.
def FeatureFooPred : FeatureSchedPredicate<FeatureFoo>;

def Variant : SchedWriteVariant<[
  SchedVar<FeatureFooPred, [SchedWriteResA]>,
  SchedVar<NoSchedPred,    [SchedWriteResB]>
]>;

def : InstRW<[Variant], (instrs Inst_A)>;
}

def ProcessorA: ProcessorModel<"ProcessorA", SchedModel_A, []>;

// CHECK:  unsigned resolveVariantSchedClassImpl(unsigned SchedClass,
// CHECK-NEXT:      const MCInst *MI, const MCInstrInfo *MCII, const MCSubtargetInfo &STI, unsigned CPUID)
// CHECK:  case {{.*}}: // Inst_A
// CHECK-NEXT:   if (CPUID == {{.*}}) { // SchedModel_A
// CHECK-NEXT:     if (STI.hasFeature(TestTarget::FeatureFoo))
// CHECK-NEXT:       return {{.*}}; // SchedWriteResA
// CHECK-NEXT:     return {{.*}}; // SchedWriteResB

// CHECK: unsigned resolveVariantSchedClass(unsigned SchedClass,
// CHECK-NEXT:    const MCInst *MI, const MCInstrInfo *MCII,
// CHECK-NEXT:    unsigned CPUID) const override {
// CHECK-NEXT:   return TestTarget_MC::resolveVariantSchedClassImpl(SchedClass, MI, MCII, *this, CPUID);
// CHECK-NEXT: }

// CHECK: unsigned TestTargetGenSubtargetInfo
// CHECK-NEXT: ::resolveSchedClass(unsigned SchedClass, const MachineInstr *MI, const TargetSchedModel *SchedModel) const {
// CHECK-NEXT:   switch (SchedClass) {
// CHECK-NEXT:   case {{.*}}: // Inst_A
// CHECK-NEXT:     if (SchedModel->getProcessorID() == {{.*}}) { // SchedModel_A
// CHECK-NEXT:       if (this->hasFeature(TestTarget::FeatureFoo))
// CHECK-NEXT:         return {{.*}}; // SchedWriteResA
// CHECK-NEXT:       return {{.*}}; // SchedWriteResB
