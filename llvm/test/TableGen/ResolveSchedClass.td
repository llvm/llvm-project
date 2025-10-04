// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s -o - | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

// CHECK:  unsigned resolveVariantSchedClassImpl(unsigned SchedClass,
// CHECK-NEXT:      const MCInst *MI, const MCInstrInfo *MCII, const MCSubtargetInfo &STI, unsigned CPUID)

// CHECK: unsigned resolveVariantSchedClass(unsigned SchedClass,
// CHECK-NEXT:    const MCInst *MI, const MCInstrInfo *MCII,
// CHECK-NEXT:    unsigned CPUID) const override {
// CHECK-NEXT:   return TestTarget_MC::resolveVariantSchedClassImpl(SchedClass, MI, MCII, *this, CPUID);
// CHECK-NEXT: }
