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
// CHECK-NEXT: OPC_Scope, 14, /*->20*/ // 2 children in Scope
// CHECK-NEXT: OPC_CheckChild1Integer, 0,
// CHECK-NEXT: OPC_EmitInteger, MVT::i32, 0|128,1/*128*/,
// CHECK-NEXT: OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::COPY_TO_REGCLASS), 0,
// CHECK-NEXT:     MVT::i32, 2/*#Ops*/, 1, 0,
def : Pat<(i32 (add i32:$src, (i32 0))),
          (COPY_TO_REGCLASS GPRAbove127, GPR0:$src)>;

// CHECK:      OPC_CheckChild1Integer, 1,
// CHECK-NEXT: OPC_EmitInteger, MVT::i32, TestNamespace::GPR127RegClassID,
// CHECK-NEXT: OPC_MorphNodeTo1, TARGET_VAL(TargetOpcode::COPY_TO_REGCLASS), 0,
// CHECK-NEXT:     MVT::i32, 2/*#Ops*/, 1, 0,
def : Pat<(i32 (add i32:$src, (i32 1))),
          (COPY_TO_REGCLASS GPR127, GPR0:$src)>;
