// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def Test0 : GICombineRule<
  (defs root:$dst),
  (match (G_MUL $dst, $src, -1)),
  (apply (G_SUB $dst, (GITypeOf<"$src"> 0), $tmp),
         (G_CONSTANT GITypeOf<"$dst">:$tmp, (GITypeOf<"$src"> 42)))>;

// CHECK:      const int64_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static int64_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ 57, // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule0Enabled,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_MUL,
// CHECK-NEXT:       // MIs[0] dst
// CHECK-NEXT:       GIM_RecordRegType, /*MI*/0, /*Op*/0, /*TempTypeIdx*/-1,
// CHECK-NEXT:       // MIs[0] src
// CHECK-NEXT:       GIM_RecordRegType, /*MI*/0, /*Op*/1, /*TempTypeIdx*/-2,
// CHECK-NEXT:       // MIs[0] Operand 2
// CHECK-NEXT:       GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/-2,
// CHECK-NEXT:       GIR_BuildConstant, /*TempRegID*/1, /*Val*/0,
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/-1,
// CHECK-NEXT:       // Combiner Rule #0: Test0
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::G_CONSTANT,
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_AddCImm, /*InsnID*/0, /*Type*/-2, /*Imm*/42,
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::G_SUB,
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 0: @57
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     };
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }

def MyCombiner: GICombiner<"GenMyCombiner", [
  Test0
]>;
