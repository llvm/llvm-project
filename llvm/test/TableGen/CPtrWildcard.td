// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s -o - | FileCheck %s

// Create an intrinsic that uses cPTR to overload on capability pointer types,
// and verify that we can match it correct in SelectionDAG.

// CHECK:        static const uint8_t MatcherTable[] = {
// CHECK-NEXT: /*     0*/ OPC_CheckOpcodeISD_INTRINSIC_WO_CHAIN,
// CHECK-NEXT:/*     1*/ OPC_CheckChild0Integer, [[#]],
// CHECK-NEXT:/*     3*/ OPC_RecordChild1, // #0 = $src
// CHECK-NEXT:/*     4*/ OPC_Scope /*2 children */, 9, // ->15
// CHECK-NEXT:/*     6*/  OPC_CheckChild1Type, /*MVT::c64*/5|128,2/*261*/,
// CHECK-NEXT:/*     9*/  OPC_MorphNodeTo1None, TARGET_VAL(MyTarget::C64_TO_I64),
// CHECK-NEXT:                MVT::i64, 1/*#Ops*/, /*OperandList*/0, // Ops = #0
// CHECK-NEXT:            // Src: (intrinsic_wo_chain:{ *:[i64] } [[#]]:{ *:[iPTR] }, c64:{ *:[c64] }:$src) - Complexity = 8
// CHECK-NEXT:            // Dst: (C64_TO_I64:{ *:[i64] } ?:{ *:[c64] }:$src)
// CHECK-NEXT:/*    15*/ /*Scope*/ 9, // ->25
// CHECK-NEXT:/*    16*/  OPC_CheckChild1Type, /*MVT::c128*/6|128,2/*262*/,
// CHECK-NEXT:/*    19*/  OPC_MorphNodeTo1None, TARGET_VAL(MyTarget::C128_TO_I64),
// CHECK-NEXT:                MVT::i64, 1/*#Ops*/, /*OperandList*/0, // Ops = #0
// CHECK-NEXT:            // Src: (intrinsic_wo_chain:{ *:[i64] } [[#]]:{ *:[iPTR] }, c128:{ *:[c128] }:$src) - Complexity = 8
// CHECK-NEXT:            // Dst: (C128_TO_I64:{ *:[i64] } ?:{ *:[c128] }:$src)
// CHECK-NEXT:/*    25*/ 0, // End of Scope
// CHECK-NEXT:  }; // Total Array size is 26 bytes

// CHECK:        static const uint8_t OperandLists[] = {
// CHECK-NEXT:     /* 0 */ 0,
// CHECK-NEXT:   }

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
