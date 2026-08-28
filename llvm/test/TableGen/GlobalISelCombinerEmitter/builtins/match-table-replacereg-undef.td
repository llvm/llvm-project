// RUN: llvm-tblgen -I %p/../../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def ReplaceWithUndef : GICombineRule<
  (defs root:$dst),
  (match  (G_FNEG $dst, $src)),
  (apply  (GIReplaceRegWithUndef $dst))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  ReplaceWithUndef
]>;

//      CHECK: const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4([[L32:[0-9]+]]), // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FNEG),
// CHECK-NEXT:       // MIs[0] dst
// CHECK-NEXT:       GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// CHECK-NEXT:       // MIs[0] src
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/255,
// CHECK-NEXT:       // Combiner Rule #0: ReplaceWithUndef
// CHECK-NEXT:       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_IMPLICIT_DEF),
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(static_cast<uint16_t>(RegState::Define)),
// CHECK-NEXT:       GIR_ReplaceRegWithTempReg, /*OldInsnID*/0, /*OldOpIdx*/0, /*TempRegID*/0,
// CHECK-NEXT:       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:     // Label 0: @[[L32]]
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     }; // Size: {{[0-9]+}} bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
