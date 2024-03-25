// RUN: llvm-tblgen -I %p/../../../include -gen-global-isel-combiner \
// RUN:     -combiners=MyCombiner %s | FileCheck %s

// Checks that temporary registers defined in apply patterns
// are emitted with RegState::Define.

include "llvm/Target/Target.td"
include "llvm/Target/GlobalISel/Combine.td"

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

def Test0 : GICombineRule<
  (defs root:$dst),
  (match (G_ADD $dst, $lhs, $rhs)),
  (apply (G_UDIVREM $tmp, $dst, $lhs, $rhs))
>;

def Test1 : GICombineRule<
  (defs root:$dst),
  (match (G_ADD $dst, $lhs, $rhs)),
  (apply (G_UDIVREM $dst, $tmp, $lhs, $rhs))
>;

def Test2 : GICombineRule<
  (defs root:$dst),
  (match (G_ADD $dst, $lhs, $rhs)),
  (apply (G_ADD $tmp, 0, $lhs),
         (G_ADD $dst, $tmp, $rhs))
>;

def MyCombiner: GICombiner<"GenMyCombiner", [
  Test0,
  Test1,
  Test2,
]>;

// CHECK:       // Combiner Rule #0: Test0
// CHECK-NEXT:  GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::G_UDIVREM),
// CHECK-NEXT:  GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:  GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:  GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // lhs
// CHECK-NEXT:  GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // rhs

// CHECK:       // Combiner Rule #1: Test1
// CHECK-NEXT:  GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::G_UDIVREM),
// CHECK-NEXT:  GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
// CHECK-NEXT:  GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:  GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // lhs
// CHECK-NEXT:  GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // rhs

// CHECK:       // Combiner Rule #2: Test2
// CHECK-NEXT:  GIR_BuildMI, /*InsnID*/0, /*Opcode*/GIMT_Encode2(TargetOpcode::G_ADD),
// CHECK-NEXT:  GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/GIMT_Encode2(RegState::Define),
// CHECK-NEXT:  GIR_AddSimpleTempRegister, /*InsnID*/0, /*TempRegID*/1,
// CHECK-NEXT:  GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // lhs
