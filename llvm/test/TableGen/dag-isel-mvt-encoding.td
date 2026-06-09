// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s -o - | FileCheck %s

// Verify the direct/escaped MVT boundary and representative values on both
// sides of the old VBR boundary at value 127.

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo;
def TestTarget : Target {
  let InstructionSet = TestInstrInfo;
}

def R0 : Register<"r0">;
def GPR : RegisterClass<"TestTarget", [i32], 32, (add R0)>;

class TestRegisterClass<ValueType VT>
    : RegisterClass<"TestTarget", [VT], VT.Size, (add R0)>;

def RC127 : TestRegisterClass<v64bf16>;
def RC128 : TestRegisterClass<v128bf16>;
def RC254 : TestRegisterClass<x86amx>;
def RC255 : TestRegisterClass<i64x8>;
def RC262 : TestRegisterClass<c128>;

class TestNode<ValueType VT, string Opcode>
    : SDNode<Opcode, SDTypeProfile<1, 0, [SDTCisVT<0, VT>]>>;

def Node127 : TestNode<v64bf16, "TestISD::NODE127">;
def Node128 : TestNode<v128bf16, "TestISD::NODE128">;
def Node254 : TestNode<x86amx, "TestISD::NODE254">;
def Node255 : TestNode<i64x8, "TestISD::NODE255">;
def Node262 : TestNode<c128, "TestISD::NODE262">;

class TestInstruction<RegisterClass RC> : Instruction {
  let OutOperandList = (outs RC:$dst);
  let InOperandList = (ins);
}

def Inst127 : TestInstruction<RC127>;
def Inst128 : TestInstruction<RC128>;
def Inst254 : TestInstruction<RC254>;
def Inst255 : TestInstruction<RC255>;
def Inst262 : TestInstruction<RC262>;

def : Pat<(Node127), (Inst127)>;
def : Pat<(Node128), (Inst128)>;
def : Pat<(Node254), (Inst254)>;
def : Pat<(Node255), (Inst255)>;
def : Pat<(Node262), (Inst262)>;

// CHECK-DAG: MVT::v64bf16, 0/*#Ops*/
// CHECK-DAG: MVT::v128bf16, 0/*#Ops*/
// CHECK-DAG: MVT::x86amx, 0/*#Ops*/
// CHECK-DAG: 255,/*MVT::i64x8*/0, 0/*#Ops*/
// CHECK-DAG: 255,/*MVT::c128*/7, 0/*#Ops*/
// CHECK: Total Array size is 44 bytes
