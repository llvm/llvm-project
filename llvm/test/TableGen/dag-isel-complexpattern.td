// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestTargetInstrInfo : InstrInfo;

def TestTarget : Target {
  let InstructionSet = TestTargetInstrInfo;
}

def REG32 : Register<"REG32">;
def REG64 : Register<"REG64">;
def GPR32 : RegisterClass<"TestTarget", [i32], 32, (add REG32)>;
def GPR64 : RegisterClass<"TestTarget", [i64], 64, (add REG64)>;

def CP32 : ComplexPattern<i32, 0, "SelectCP32">;

// Without using ComplexPattern's type, this pattern would be ambiguous as to
// what integer type is being used, since both i32 and i64 are legal, and used
// to erroneously happen, whilst using a leaf value like CP32:$a/b instead of
// (CP32) still worked.
def INSTR : Instruction {
// CHECK-LABEL: OPC_CheckOpcode, TARGET_VAL(ISD::STORE)
// CHECK: OPC_CheckTypeI32
// CHECK: OPC_CheckComplexPat0, /*#*/1, // SelectCP32:$
// CHECK: Src: (st (add:{ *:[i32] } (CP32:{ *:[i32] }), (CP32:{ *:[i32] })), i64:{ *:[i64] }:$addr)
  let OutOperandList = (outs);
  let InOperandList = (ins GPR64:$addr);
  let Pattern = [(store (add (CP32), (CP32)), i64:$addr)];
}
