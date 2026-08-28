// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

// G_FCONSTANT, like G_CONSTANT, can be built directly in an apply pattern
// with a GITypeOf-typed literal. The literal is interpreted as a
// floating-point value (the pattern grammar has no fp literals).
def Test0 : GICombineRule<
  (defs root:$dst),
  (match (G_FNEG $t, $src),
         (G_FNEG $dst, $t)),
  (apply (G_FCONSTANT $dst, (GITypeOf<"$dst"> 0)))>;

// CHECK:      const uint8_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static uint8_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ GIMT_Encode4([[L34:[0-9]+]]), // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GIMT_Encode2(GICXXPred_Simple_IsRule0Enabled),
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, GIMT_Encode2(TargetOpcode::G_FNEG),
// CHECK-NEXT:       // MIs[0] dst
// CHECK-NEXT:       GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/255,
// CHECK-NEXT:       // MIs[0] t
// CHECK-NEXT:       GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/1, GIMT_Encode2(TargetOpcode::G_FNEG),
// CHECK-NEXT:       // MIs[1] src
// CHECK-NEXT:       // No operand predicates
// CHECK-NEXT:       GIM_CheckIsSafeToFold, /*NumInsns*/1,
// CHECK-NEXT:       // Combiner Rule #0: Test0
// CHECK-NEXT:       GIR_BuildRootMI, /*Opcode*/GIMT_Encode2(TargetOpcode::G_FCONSTANT),
// CHECK-NEXT:       GIR_RootToRootCopy, /*OpIdx*/0, // dst
// CHECK-NEXT:       GIR_AddCFPImm, /*InsnID*/0, /*Type*/255, /*Imm*/GIMT_Encode8(0),
// CHECK-NEXT:       GIR_EraseRootFromParent_Done,
// CHECK-NEXT:     // Label 0: @[[L34]]
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     }; // Size: {{[0-9]+}} bytes
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }

def MyCombiner: GICombiner<"GenMyCombiner", [
  Test0
]>;
