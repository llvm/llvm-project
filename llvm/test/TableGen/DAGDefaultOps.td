// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include %s -o %t
// RUN: FileCheck --check-prefix=ADD %s < %t
// RUN: FileCheck --check-prefix=ADDINT %s < %t
// RUN: FileCheck --check-prefix=SUB %s < %t
// RUN: FileCheck --check-prefix=MULINT %s < %t

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo;
def TestTarget : Target {
    let InstructionSet = TestInstrInfo;
}

class TestEncoding : Instruction {
  field bits<32> Inst;
}

class TestReg<int index> : Register<"R"#index, []> {
    let HWEncoding{15...4} = 0;
    let HWEncoding{3...0} = !cast<bits<4>>(index);
}
foreach i = 0...15 in
  def "R"#i : TestReg<i>;

def Reg : RegisterClass<"TestTarget", [i32], 32, (sequence "R%d", 0, 15)>;

def IntOperand: Operand<i32>;
def OptionalIntOperand: OperandWithDefaultOps<i32, (ops (i32 0))>;

class RRI<string Mnemonic, bits<4> Opcode> : TestEncoding {
  dag OutOperandList = (outs Reg:$dest);
  dag InOperandList = (ins Reg:$src1, Reg:$src2, OptionalIntOperand:$imm);
  string AsmString = Mnemonic # " $dest1, $src1, $src2, #$imm";
  string AsmVariantName = "";
  field bits<4> dest;
  field bits<4> src1;
  field bits<4> src2;
  field bits<16> imm;
  let Inst{31...28} = Opcode;
  let Inst{27...24} = dest;
  let Inst{23...20} = src1;
  let Inst{19...16} = src2;
  let Inst{15...0} = imm;
}

def AddRRI : RRI<"add", 0b0001>;

// I define one of these intrinsics with IntrNoMem and the other
// without it, so that they'll match different top-level DAG opcodes
// (INTRINSIC_WO_CHAIN and INTRINSIC_W_CHAIN), which makes the
// FileCheck-herding easier, because every case I want to detect
// should show up as a separate top-level switch element.
def int_addplus1 : Intrinsic<
    [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_mul3 : Intrinsic<
    [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty]>;

def AddPat  : Pat<(add i32:$x, i32:$y),
                  (AddRRI Reg:$x, Reg:$y)>;
def Add1Pat : Pat<(int_addplus1 i32:$x, i32:$y),
                  (AddRRI Reg:$x, Reg:$y, (i32 1))>;

def SubRRI : RRI<"sub", 0b0010> {
  let Pattern = [(set Reg:$dest, (sub Reg:$src1, Reg:$src2))];
}

def MulRRI : RRI<"mul", 0b0011> {
  let Pattern = [(set Reg:$dest, (int_mul3 Reg:$src1, Reg:$src2, i32:$imm))];
}

def MulIRR : RRI<"mul2", 0b0100> {
  let InOperandList = (ins OptionalIntOperand:$imm, Reg:$src1, Reg:$src2);
}
def MulIRRPat : Pat<(mul i32:$x, i32:$y), (MulIRR Reg:$x, Reg:$y)>;

// ADD: SwitchOpcode{{.*}}TARGET_VAL(ISD::ADD)
// ADD-NEXT: OPC_RecordChild0
// ADD-NEXT: OPC_RecordChild1
// ADD-NEXT: OPC_EmitInteger32, 0
// ADD-NEXT: OPC_MorphNodeTo1None, TARGET_VAL(::AddRRI)

// ADDINT: SwitchOpcode{{.*}}TARGET_VAL(ISD::INTRINSIC_WO_CHAIN)
// ADDINT-NEXT: OPC_CheckChild0Integer
// ADDINT-NEXT: OPC_RecordChild1
// ADDINT-NEXT: OPC_RecordChild2
// ADDINT-NEXT: OPC_EmitInteger32, 2
// ADDINT-NEXT: OPC_MorphNodeTo1None, TARGET_VAL(::AddRRI)

// SUB: SwitchOpcode{{.*}}TARGET_VAL(ISD::SUB)
// SUB-NEXT: OPC_RecordChild0
// SUB-NEXT: OPC_RecordChild1
// SUB-NEXT: OPC_EmitInteger32, 0
// SUB-NEXT: OPC_MorphNodeTo1None, TARGET_VAL(::SubRRI)

// MULINT: SwitchOpcode{{.*}}TARGET_VAL(ISD::INTRINSIC_W_CHAIN)
// MULINT-NEXT: OPC_RecordNode
// MULINT-NEXT: OPC_CheckChild1Integer
// MULINT-NEXT: OPC_RecordChild2
// MULINT-NEXT: OPC_RecordChild3
// MULINT-NEXT: OPC_RecordChild4
// MULINT-NEXT: OPC_EmitMergeInputChains
// MULINT-NEXT: OPC_MorphNodeTo1Chain, TARGET_VAL(::MulRRI)

// MUL: SwitchOpcode{{.*}}TARGET_VAL(ISD::MUL)
// MUL-NEXT: OPC_EmitInteger32, 0
// MUL-NEXT: OPC_RecordChild0
// MUL-NEXT: OPC_RecordChild1
// MUL-NEXT: OPC_MorphNodeTo1Chain, TARGET_VAL(::MulRRI)
