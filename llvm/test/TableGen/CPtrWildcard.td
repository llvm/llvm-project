// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s -o - | FileCheck %s

// Create an intrinsic that uses cPTR to overload on capability pointer types,
// and verify that we can match it correct in SelectionDAG.

// CHECK:        static const unsigned char MatcherTable[] = {
// CHECK-NEXT: /*     0*/ OPC_CheckOpcode, TARGET_VAL(ISD::INTRINSIC_WO_CHAIN),
// CHECK-NEXT:/*     3*/ OPC_CheckChild0Integer, [[#]],
// CHECK-NEXT:/*     5*/ OPC_RecordChild1, // #0 = $src
// CHECK-NEXT:/*     6*/ OPC_Scope, 9, /*->17*/ // 2 children in Scope
// CHECK-NEXT:/*     8*/  OPC_CheckChild1Type, /*MVT::c64*/126|128,1/*254*/,
// CHECK-NEXT:/*    11*/  OPC_MorphNodeTo1None, TARGET_VAL(MyTarget::C64_TO_I64),
// CHECK-NEXT:                /*MVT::i64*/8, 1/*#Ops*/, 0,
// CHECK-NEXT:            // Src: (intrinsic_wo_chain:{ *:[i64] } [[#]]:{ *:[iPTR] }, c64:{ *:[c64] }:$src) - Complexity = 8
// CHECK-NEXT:            // Dst: (C64_TO_I64:{ *:[i64] } ?:{ *:[c64] }:$src)
// CHECK-NEXT:/*    17*/ /*Scope*/ 9, /*->27*/
// CHECK-NEXT:/*    18*/  OPC_CheckChild1Type, /*MVT::c128*/127|128,1/*255*/,
// CHECK-NEXT:/*    21*/  OPC_MorphNodeTo1None, TARGET_VAL(MyTarget::C128_TO_I64),
// CHECK-NEXT:                /*MVT::i64*/8, 1/*#Ops*/, 0,
// CHECK-NEXT:            // Src: (intrinsic_wo_chain:{ *:[i64] } [[#]]:{ *:[iPTR] }, c128:{ *:[c128] }:$src) - Complexity = 8
// CHECK-NEXT:            // Dst: (C128_TO_I64:{ *:[i64] } ?:{ *:[c128] }:$src)
// CHECK-NEXT:/*    27*/ 0, /*End of Scope*/
// CHECK-NEXT:    0
// CHECK-NEXT:  }; // Total Array size is 29 bytes

include "llvm/Target/Target.td"

def my_cap_ty : LLVMQualPointerType<200> {
  let VT = cPTR;
}

def int_cap_get_length :
  Intrinsic<[llvm_i64_ty],
            [my_cap_ty],
            [IntrNoMem, IntrWillReturn]>;

class CapReg<string n> : Register<n> {
    let Namespace = "MyTarget";
}

def C64  : CapReg<"c0">;
def C64s
    : RegisterClass<"MyTarget", [i64, c64], 64,
                    (add C64)>;

def C128  : CapReg<"c0">;
def C128s
    : RegisterClass<"MyTarget", [c128], 64,
                    (add C128)>;

def C64_TO_I64 : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = (outs C64s:$dst);
  let InOperandList = (ins C64s:$src);
}

def C128_TO_I64 : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = (outs C64s:$dst);
  let InOperandList = (ins C128s:$src);
}

def : Pat<
  (int_cap_get_length c64:$src),
  (C64_TO_I64 $src)
>;

def : Pat<
  (int_cap_get_length c128:$src),
  (C128_TO_I64 $src)
>;

def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }
