// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | \
// RUN: FileCheck %s

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def InstTest0 : GICombineRule<
  (defs root:$a),
  (match  (G_MUL i32:$x, i32:$b, i32:$c),
          (G_MUL $a, i32:$b, i32:$x)),
  (apply  (G_ADD i64:$tmp, $b, i32:$c),
          (G_ADD i8:$a, $b, i64:$tmp))>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  InstTest0,
]>;

// CHECK:       const int64_t *GenMyCombiner::getMatchTable() const {
// CHECK-NEXT:   constexpr static int64_t MatchTable0[] = {
// CHECK-NEXT:     GIM_Try, /*On fail goto*//*Label 0*/ 73, // Rule ID 0 //
// CHECK-NEXT:       GIM_CheckSimplePredicate, GICXXPred_Simple_IsRule0Enabled,
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/0, TargetOpcode::G_MUL,
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s8,
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:       GIM_RecordInsnIgnoreCopies, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
// CHECK-NEXT:       GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
// CHECK-NEXT:       GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
// CHECK-NEXT:       // MIs[1] b
// CHECK-NEXT:       GIM_CheckIsSameOperandIgnoreCopies, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
// CHECK-NEXT:       GIM_CheckIsSafeToFold, /*InsnID*/1,
// CHECK-NEXT:       GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
// CHECK-NEXT:       // Combiner Rule #0: InstTest0
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::G_ADD,
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // b
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // c
// CHECK-NEXT:       GIR_EraseFromParent, /*InsnID*/0,
// CHECK-NEXT:       GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::G_ADD,
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/0, // a
// CHECK-NEXT:       GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // b
// CHECK-NEXT:       GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/0,
// CHECK-NEXT:       GIR_Done,
// CHECK-NEXT:     // Label 0: @73
// CHECK-NEXT:     GIM_Reject,
// CHECK-NEXT:     };
// CHECK-NEXT:   return MatchTable0;
// CHECK-NEXT: }
