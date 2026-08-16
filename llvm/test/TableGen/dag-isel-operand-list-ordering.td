// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s -o %t
// RUN: FileCheck %s < %t

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo;
def TestTarget : Target {
  let InstructionSet = TestInstrInfo;
}

class TestReg<int Index> : Register<"R"#Index, []>;
foreach I = 0...3 in
  def "R"#I : TestReg<I>;

def Reg : RegisterClass<"TestTarget", [i32], 32, (sequence "R%d", 0, 3)>;

class TestInstruction<string Name> : Instruction {
  let Namespace = "Test";
  let OutOperandList = (outs Reg:$dst);
  let InOperandList = (ins Reg:$src0, Reg:$src1);
  let AsmString = Name;
}

def CommonAdd : TestInstruction<"common-add">;
def CommonSub : TestInstruction<"common-sub">;
def RareMul : TestInstruction<"rare-mul">;

def : Pat<(add i32:$x, i32:$y), (CommonAdd Reg:$x, Reg:$y)>;
def : Pat<(sub i32:$x, i32:$y), (CommonSub Reg:$x, Reg:$y)>;
def : Pat<(mul i32:$x, i32:$y), (RareMul Reg:$y, Reg:$x)>;

// The reverse-lexicographical suffix order places (1, 0) before (0, 1).
// Frequency ordering instead gives the twice-used (0, 1) sequence offset 0.
// CHECK: OPC_MorphNodeTo1None, TARGET_VAL(Test::CommonAdd),
// CHECK-NEXT: MVT::i32, 2/*#Ops*/, /*OperandList*/0,
// CHECK: OPC_MorphNodeTo1None, TARGET_VAL(Test::CommonSub),
// CHECK-NEXT: MVT::i32, 2/*#Ops*/, /*OperandList*/0,
// CHECK: OPC_MorphNodeTo1None, TARGET_VAL(Test::RareMul),
// CHECK-NEXT: MVT::i32, 2/*#Ops*/, /*OperandList*/2,

// CHECK-LABEL: static const uint8_t OperandLists[] = {
// CHECK-NEXT:  /* 0 */ 0, 1,
// CHECK-NEXT:  /* 2 */ 1, 0,
