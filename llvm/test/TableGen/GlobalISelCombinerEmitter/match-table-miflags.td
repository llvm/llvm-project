// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def MIFlagsTest : GICombineRule<
  (defs root:$dst),
  (match (G_SEXT $dst, $tmp), (G_ZEXT $tmp, $src, (MIFlags FmReassoc, FmNsz, (not FmNoNans, FmArcp))):$mi),
  (apply (G_MUL $dst, $src, $src, (MIFlags $mi, FmReassoc, (not FmNsz, FmArcp))))>;

def MyCombiner: GICombiner<"GenMyCombiner", [MIFlagsTest]>;

// CHECK:      const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4(68), // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_SEXT),
// CHECK-NEXT:       // MIs[0] dst
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       // MIs[0] tmp
// CHECK-NEXT:       GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_ZEXT),
// CHECK-NEXT:       GIM_MIFlags, /*MI*/1, GIMT_Encode4(MachineInstr::FmNsz | MachineInstr::FmReassoc),
// CHECK-NEXT:       GIM_MIFlagsNot, /*MI*/1, GIMT_Encode4(MachineInstr::FmArcp | MachineInstr::FmNoNans),
// CHECK-NEXT:       // MIs[1] src
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:       // Combiner Rule #0: MIFlagsTest
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::G_MUL),
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
// CHECK-NEXT:       GIR_CopyMIFlags, /*InsnID*/0, /*OldInsnID*/1,
// CHECK-NEXT:       GIR_SetMIFlags, /*InsnID*/0, GIMT_Encode4(MachineInstr::FmReassoc),
// CHECK-NEXT:       GIR_UnsetMIFlags, /*InsnID*/0, GIMT_Encode4(MachineInstr::FmNsz | MachineInstr::FmArcp),
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 0: @68
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     };
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
