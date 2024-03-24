// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

let Namespace = "TestNamespace" in {

def R0 : Register<"r0">;

foreach i = 0...127 in {
def GPR#i : RegisterClass<"TestTarget", [i32], 32,
                          (add R0)>;
}

def GPRAbove127 : RegisterClass<"TestTarget", [i32], 32,
                                (add R0)>;
} // end Namespace TestNamespace

// CHECK:      OPC_CheckOpcode, TARGET_VAL(ISD::ADD),
// CHECK-NEXT: OPC_RecordChild0, // #0 = $src
// CHECK-NEXT: OPC_Scope, 12, /*->18*/ // 2 children in Scope
// CHECK-NEXT: OPC_CheckChild1Integer, 0,
// CHECK-NEXT: OPC_EmitInteger32, 0|128,2/*256*/,
// CHECK-NEXT: OPC_MorphNodeTo1None, TARGET_VAL(TargetOpcode::COPY_TO_REGCLASS),
// CHECK-NEXT:     MVT::i32, 2/*#Ops*/, 1, 0,
def : Pat<(i32 (add i32:$src, (i32 0))),
          (COPY_TO_REGCLASS GPRAbove127, GPR0:$src)>;

// CHECK:      OPC_CheckChild1Integer, 2,
// CHECK-NEXT: OPC_EmitStringInteger32, TestNamespace::GPR127RegClassID,
// CHECK-NEXT: OPC_MorphNodeTo1None, TARGET_VAL(TargetOpcode::COPY_TO_REGCLASS),
// CHECK-NEXT:     MVT::i32, 2/*#Ops*/, 1, 0,
def : Pat<(i32 (add i32:$src, (i32 1))),
          (COPY_TO_REGCLASS GPR127, GPR0:$src)>;
