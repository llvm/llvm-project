// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;


def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR : RegisterClass<"MyTarget", [i32, f32], 32, (add R0)>;

def a_nearbyint : SDNode<
    "MyTargetISD::A_NEARBYINT",
    SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisFP<1>]>,
    [SDNPOptChain]
>;

def I_NEARBYINT : Instruction {
  let OutOperandList = (outs GPR:$out);
  let InOperandList = (ins GPR:$x);
}

def : Pat<(a_nearbyint GPR:$x), (I_NEARBYINT GPR:$x)>;
def : Pat<(a_nearbyint (a_nearbyint GPR:$x)), (I_NEARBYINT GPR:$x)>;

// CHECK-LABEL: OPC_CheckOpcode, TARGET_VAL(MyTargetISD::A_NEARBYINT),
// CHECK-NEXT: OPC_RecordOptionalChain,
// CHECK-NEXT: OPC_Scope, 17
// CHECK-NEXT:  OPC_MoveChild0,
// CHECK-NEXT:  OPC_CheckOpcode, TARGET_VAL(MyTargetISD::A_NEARBYINT),
// CHECK-NEXT:  OPC_RecordOptionalChain,
// CHECK-NEXT:  OPC_CheckFoldableChainNode,
// CHECK-NEXT:  OPC_RecordChild0,
// CHECK-NEXT:  OPC_MoveParent,
// CHECK-NEXT:  OPC_EmitMergeInputChains, 0, 
// CHECK-NEXT:  OPC_MorphNodeTo1, TARGET_VAL(::I_NEARBYINT), 0,
// CHECK-NEXT:      /*MVT::f32*/12, 1/*#Ops*/, 2,
 
// CHECK:      /*Scope*/ 10,
// CHECK-NEXT:  OPC_RecordChild0,
// CHECK-NEXT:  OPC_EmitMergeInputChains, 0, 
// CHECK-NEXT:  OPC_MorphNodeTo1, TARGET_VAL(::I_NEARBYINT), 0,
// CHECK-NEXT:      /*MVT::f32*/12, 1/*#Ops*/, 1, 
