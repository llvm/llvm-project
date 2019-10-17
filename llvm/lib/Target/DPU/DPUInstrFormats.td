//===-- DPUInstrFormats.td - DPU Instruction Formats -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Describe DPU instructions format
//
//===----------------------------------------------------------------------===//


class DPUInstruction : Instruction {
  field bits<48> Inst;

  field dag OutOperandList;
  field dag InOperandList;

  field string AsmString;

  field list<dag> Pattern;

  let Namespace = "DPU";
  let DecoderNamespace = "DPU";
  let Size = 6;
  field bits<48> SoftFail = 0;
}

class PseudoDPUInstruction<dag outs, dag ins, string asmstr, list<dag> pattern> : DPUInstruction {
  dag OutOperandList = outs;
  dag InOperandList  = ins;
  let AsmString      = asmstr;
  let Pattern        = pattern;
  let isPseudo       = 1;
  let isCodeGenOnly  = 1;
}

class ConditionClassAsmOperand<string cc> : AsmOperandClass {
  let Name = cc;
  let RenderMethod = "addConditionOperands";
  let ParserMethod = "parseAnyConditionAs" # cc;
  let PredicateMethod = "is" # cc;
}

class ImmAsmOperandClass<int nr_bits, string signedness> : AsmOperandClass {
  let Name = signedness # "Imm" # nr_bits;
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseAnyImmediate";
  let PredicateMethod = "is" # signedness # "Imm<" # nr_bits # ">";
}

def EndianAsmOperandClass : AsmOperandClass {
  let Name = "Endian";
  let RenderMethod = "addEndianOperands";
  let ParserMethod = "parseAnyEndianness";
  let PredicateMethod = "isEndian";
}


class ConditionClassOperand<string cc> : Operand<i32> {
  let PrintMethod = "printCondition";
  let ParserMatchClass = ConditionClassAsmOperand<cc>;
  let EncoderMethod = "getConditionEncoding";
}

class ImmOperand<int nr_bits, string signedness, ValueType super_op> : Operand<super_op> {
  let PrintMethod = "print" # signedness # "Imm<" # nr_bits # ">";
  let ParserMatchClass = ImmAsmOperandClass<nr_bits, signedness>;
}

class SImmOperand<int nr_bits, ValueType super_op> : ImmOperand<nr_bits, "S", super_op>, ImmLeaf<super_op, !strconcat("return (Imm >= -(1L << (", !cast<string>(nr_bits), " - 1))) && (Imm < (1L << (", !cast<string>(nr_bits), " - 1)));")>;
class UImmOperand<int nr_bits, ValueType super_op> : ImmOperand<nr_bits, "U", super_op>, ImmLeaf<super_op, !strconcat("return (Imm >= 0) && (Imm < (1L << ", !cast<string>(nr_bits), "));")>;
class SUImmOperand<int nr_bits, ValueType super_op> : ImmOperand<nr_bits, "SU", super_op>, ImmLeaf<super_op, !strconcat("return (Imm >= -(1L << (", !cast<string>(nr_bits), " - 1))) && (Imm < (1L << ", !cast<string>(nr_bits), "));")>;
class PCImmOperand<int nr_bits, ValueType super_op> : ImmOperand<nr_bits, "PC", super_op>, ImmLeaf<super_op, !strconcat("return (Imm >= -(1L << (", !cast<string>(nr_bits), " - 1))) && (Imm < (1L << ", !cast<string>(nr_bits), "));")>;

def endian : Operand<OtherVT> {
  let PrintMethod = "printEndianness";
  let ParserMatchClass = EndianAsmOperandClass;
}

def  s8_imm : SImmOperand< 8, i32>;
def s11_imm : SImmOperand<11, i32>;
def s12_imm : SImmOperand<12, i32>;
def s16_imm : SImmOperand<16, i32>;
def s17_imm : SImmOperand<17, i32>;
def s24_imm : SImmOperand<24, i32>;
def s27_imm : SImmOperand<27, i32>;
def s28_imm : SImmOperand<28, i32>;

def  su8_imm : SUImmOperand< 8, i32>;
def su16_imm : SUImmOperand<16, i32>;
def u32_imm : SUImmOperand<32, i32>;
def u5_imm  : UImmOperand< 5, i32>;
def u8_imm  : UImmOperand< 8, i32>;

def pc16 : PCImmOperand<16, OtherVT> {
}
def pc24 : PCImmOperand<24, i32> {
}
def pc28 : PCImmOperand<28, i32> {
}
def pcbb : PCImmOperand<28, OtherVT>;

def acquire_cc      : ConditionClassOperand<"Acquire_cc">;
def add_cc          : ConditionClassOperand<"Add_cc">;
def add_nz_cc       : ConditionClassOperand<"Add_nz_cc">;
def boot_cc         : ConditionClassOperand<"Boot_cc">;
def const_cc_ge0    : ConditionClassOperand<"Const_cc_ge0">;
def const_cc_geu    : ConditionClassOperand<"Const_cc_geu">;
def const_cc_zero   : ConditionClassOperand<"Const_cc_zero">;
def count_cc        : ConditionClassOperand<"Count_cc">;
def count_nz_cc     : ConditionClassOperand<"Count_nz_cc">;
def div_cc          : ConditionClassOperand<"Div_cc">;
def div_nz_cc       : ConditionClassOperand<"Div_nz_cc">;
def ext_sub_set_cc  : ConditionClassOperand<"Ext_sub_set_cc">;
def false_cc        : ConditionClassOperand<"False_cc">;
def imm_shift_cc    : ConditionClassOperand<"Imm_shift_cc">;
def imm_shift_nz_cc : ConditionClassOperand<"Imm_shift_nz_cc">;
def log_cc          : ConditionClassOperand<"Log_cc">;
def log_nz_cc       : ConditionClassOperand<"Log_nz_cc">;
def log_set_cc      : ConditionClassOperand<"Log_set_cc">;
def mul_cc          : ConditionClassOperand<"Mul_cc">;
def mul_nz_cc       : ConditionClassOperand<"Mul_nz_cc">;
def no_cc           : ConditionClassOperand<"No_cc">;
def release_cc      : ConditionClassOperand<"Release_cc">;
def shift_cc        : ConditionClassOperand<"Shift_cc">;
def shift_nz_cc     : ConditionClassOperand<"Shift_nz_cc">;
def sub_cc          : ConditionClassOperand<"Sub_cc">;
def sub_nz_cc       : ConditionClassOperand<"Sub_nz_cc">;
def sub_set_cc      : ConditionClassOperand<"Sub_set_cc">;
def true_cc         : ConditionClassOperand<"True_cc">;
def true_false_cc   : ConditionClassOperand<"True_false_cc">;

def ACQUIRErici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s16_imm:$imm, acquire_cc:$cc, pc16:$pc);
	let AsmString        = "acquire $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isBarrier        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<16> imm; bits<2> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{25-24} = cc{1-0};
	let Inst{30-26} = imm{12-8};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{15-13};
	let Inst{47-42} = 0b011111;
}

def ADDrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u32_imm:$imm);
	let AsmString        = "add $rc, $ra, $imm";
	let Pattern          = [(set SimpleReg:$rc, (add SimpleRegOrCst:$ra, u32_imm:$imm))];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<32> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = 0b11;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0000;
}

def ADDzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "add $zero, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<32> imm; bits<5> rb;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{38-37} = 0b11;
	let Inst{39} = imm{7};
	let Inst{47-40} = 0b01100000;
}

def ADDrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "add $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s11_imm:$imm, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "add $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def ADDrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "add $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s27_imm:$imm, log_set_cc:$cc);
	let AsmString        = "add $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<27> imm; bits<5> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def ADDrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "add $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100000000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s27_imm:$imm, false_cc:$cc);
	let AsmString        = "add $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<27> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100000000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def ADDssi: DPUInstruction {
	let OutOperandList   = (outs SafeReg:$sc);
	let InOperandList    = (ins SafeRegOrCst:$sa, s17_imm:$imm);
	let AsmString        = "add $sc, $sa, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<17> imm; bits<5> sa; bits<5> sc;
	let Inst{4-0} = imm{16-12};
	let Inst{5} = imm{16};
	let Inst{6} = imm{16};
	let Inst{7} = imm{16};
	let Inst{8} = imm{16};
	let Inst{9} = imm{16};
	let Inst{10} = imm{16};
	let Inst{11} = imm{16};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100010000;
	let Inst{38-34} = sa{4-0};
	let Inst{41-39} = sc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = sc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "add $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "add $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def ADDrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "add $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "add $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def ADDrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "add $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (add SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "add $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def ADDsss: DPUInstruction {
	let OutOperandList   = (outs SafeReg:$sc);
	let InOperandList    = (ins SafeRegOrCst:$sa, SafeReg:$sb);
	let AsmString        = "add $sc, $sa, $sb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isAdd            = 1;
	bits<5> sa; bits<5> sb; bits<5> sc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = sa{2-0};
	let Inst{31-20} = 0b000100001100;
	let Inst{33-32} = sa{4-3};
	let Inst{38-34} = sb{4-0};
	let Inst{41-39} = sc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = sc{4-3};
	let Inst{47-46} = 0b00;
}

def ADD_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "add.s $dc, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{39-37} = 0b111;
	let Inst{43-40} = dc{4-1};
	let Inst{44} = imm{7};
	let Inst{47-45} = 0b000;
}

def ADD_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "add.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "add.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "add.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "add.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "add.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "add.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "add.u $dc, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{39-37} = 0b011;
	let Inst{43-40} = dc{4-1};
	let Inst{44} = imm{7};
	let Inst{47-45} = 0b000;
}

def ADD_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "add.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "add.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "add.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "add.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "add.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADD_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "add.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDCrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u32_imm:$imm);
	let AsmString        = "addc $rc, $ra, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = 0b11;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0001;
}

def ADDCzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "addc $zero, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{38-37} = 0b11;
	let Inst{39} = imm{7};
	let Inst{47-40} = 0b01100001;
}

def ADDCrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "addc $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDCzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s11_imm:$imm, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "addc $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def ADDCrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "addc $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDCzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s27_imm:$imm, log_set_cc:$cc);
	let AsmString        = "addc $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<27> imm; bits<5> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def ADDCrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "addc $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100100000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDCzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s27_imm:$imm, false_cc:$cc);
	let AsmString        = "addc $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<27> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100100000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def ADDCrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "addc $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDCzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "addc $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def ADDCrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "addc $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDCzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "addc $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def ADDCrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "addc $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b001000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDCzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "addc $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b001000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def ADDC_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "addc.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "addc.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "addc.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "addc.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "addc.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "addc.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b001000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "addc.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "addc.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "addc.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, add_nz_cc:$cc, pc16:$pc);
	let AsmString        = "addc.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "addc.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDC_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "addc.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b001000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def ADDSrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s17_imm:$imm);
	let AsmString        = "adds $rc, $ra, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<17> imm; bits<5> ra; bits<5> rc;
	let Inst{4-0} = imm{16-12};
	let Inst{5} = imm{16};
	let Inst{6} = imm{16};
	let Inst{7} = imm{16};
	let Inst{8} = imm{16};
	let Inst{9} = imm{16};
	let Inst{10} = imm{16};
	let Inst{11} = imm{16};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100010000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ADDSrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "adds $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rb; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = ra{2-0};
	let Inst{31-20} = 0b000100001100;
	let Inst{33-32} = ra{4-3};
	let Inst{38-34} = rb{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def ANDrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$ra, u32_imm:$imm);
	let AsmString        = "and $rc, $ra, $imm";
	let Pattern          = [(set SimpleReg:$rc, (and SimpleReg:$ra, u32_imm:$imm))];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = 0b11;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0101;
}

def ANDzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "and $zero, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{38-37} = 0b11;
	let Inst{39} = imm{7};
	let Inst{47-40} = 0b01100101;
}

def ANDrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "and $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s12_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "and $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<12> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{11};
	let Inst{47-45} = 0b100;
}

def ANDrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "and $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, log_set_cc:$cc);
	let AsmString        = "and $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<4> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def ANDrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "and $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110100000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, false_cc:$cc);
	let AsmString        = "and $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110100000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def ANDrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "and $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "and $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ANDrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "and $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "and $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ANDrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "and $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (and SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "and $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def AND_Srki: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins CstReg:$ra, u32_imm:$imm);
	let AsmString        = "and.s $dc, $ra, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = 0b11;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0101;
}

def AND_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "and.s $dc, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{39-37} = 0b111;
	let Inst{43-40} = dc{4-1};
	let Inst{44} = imm{7};
	let Inst{47-45} = 0b001;
}

def AND_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "and.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "and.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "and.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "and.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "and.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "and.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Urki: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins CstReg:$ra, u32_imm:$imm);
	let AsmString        = "and.u $dc, $ra, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = 0b11;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0101;
}

def AND_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "and.u $dc, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{39-37} = 0b011;
	let Inst{43-40} = dc{4-1};
	let Inst{44} = imm{7};
	let Inst{47-45} = 0b001;
}

def AND_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "and.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "and.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "and.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "and.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "and.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def AND_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "and.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDNrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "andn $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111100;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDNzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s12_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "andn $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<12> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111100;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{11};
	let Inst{47-45} = 0b100;
}

def ANDNrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "andn $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111100;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDNzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, log_set_cc:$cc);
	let AsmString        = "andn $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<4> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111100;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def ANDNrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "andn $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDNzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, false_cc:$cc);
	let AsmString        = "andn $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111000000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def ANDNrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "andn $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDNzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "andn $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ANDNrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "andn $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDNzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "andn $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ANDNrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "andn $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (and (not SimpleRegOrCst:$ra), SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDNzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "andn $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ANDNrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm);
	let AsmString        = "andn $rc, $ra, $imm";
	let Pattern          = [(set SimpleReg:$rc, (and (not SimpleRegOrCst:$ra), s24_imm:$imm))];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ANDN_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "andn.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111100;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "andn.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111100;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "andn.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "andn.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "andn.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "andn.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "andn.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111100;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "andn.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111100;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "andn.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "andn.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "andn.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ANDN_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "andn.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASRrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "asr $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ASRzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "asr $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ASRrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "asr $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b10000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ASRzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "asr $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b10000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ASRrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "asr $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (sra SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b10000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ASRzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "asr $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b10000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ASRrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "asr $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ASRzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "asr $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ASRrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "asr $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ASRzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "asr $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ASRrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "asr $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (sra SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ASRzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "asr $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ASR_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "asr.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "asr.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b10000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "asr.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b10000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "asr.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "asr.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "asr.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "asr.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "asr.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b10000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "asr.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b10000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "asr.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "asr.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ASR_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "asr.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def BKP: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins);
	let AsmString        = "bkp";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let Inst{47-0} = 0b011111100110001100100000000000000000000000000000;
}

def BOOTrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, boot_cc:$cc, pc16:$pc);
	let AsmString        = "boot $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<8> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011111011;
}

def BOOTri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm);
	let AsmString        = "boot $ra, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<8> imm; bits<5> ra;
	let Inst{15-0} = 0b0000000000000000;
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100100000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011111011;
}

def CALLrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, pc24:$off);
	let AsmString        = "call $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isCall           = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1100000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CALLzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, pc28:$off);
	let AsmString        = "call $zero, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isBarrier        = 1;
	let isTerminator     = 1;
	bits<28> off; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1100000000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = off{27};
	let Inst{47-45} = 0b100;
}

def CALLrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "call $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isCall           = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000000000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CALLzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "call $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isCall           = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000000000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CALLrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "call $rc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isCall           = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100000000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CALLri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins pc24:$off);
	let AsmString        = "call $rc, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isCall           = 1;
	bits<24> off; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{38-24} = 0b110001100000000;
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CAOrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cao $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CAOzrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cao $zero, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CAOrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "cao $rc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{23-0} = 0b000000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CAOzrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "cao $zero, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra;
	let Inst{23-0} = 0b000000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CAOrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "cao $rc, $ra";
	let Pattern          = [(set SimpleReg:$rc, (ctpop SimpleRegOrCst:$ra))];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100110000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CAOzr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra);
	let AsmString        = "cao $zero, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra;
	let Inst{33-0} = 0b1100110000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CAO_Srrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cao.s $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CAO_Srrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "cao.s $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b000000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CAO_Srr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "cao.s $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CAO_Urrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cao.u $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CAO_Urrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "cao.u $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b000000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CAO_Urr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "cao.u $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLOrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "clo $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00010000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CLOzrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "clo $zero, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00010000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CLOrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "clo $rc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{23-0} = 0b000100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CLOzrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "clo $zero, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra;
	let Inst{23-0} = 0b000100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CLOrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "clo $rc, $ra";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Clo SimpleRegOrCst:$ra))];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100110000000100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CLOzr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra);
	let AsmString        = "clo $zero, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra;
	let Inst{33-0} = 0b1100110000000100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CLO_Srrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "clo.s $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00010000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLO_Srrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "clo.s $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b000100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLO_Srr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "clo.s $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000000100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLO_Urrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "clo.u $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00010000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLO_Urrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "clo.u $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b000100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLO_Urr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "clo.u $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000000100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLR_RUNrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, boot_cc:$cc, pc16:$pc);
	let AsmString        = "clr_run $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<8> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011111001;
}

def CLSrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cls $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00100000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CLSzrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cls $zero, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00100000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CLSrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "cls $rc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{23-0} = 0b001000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CLSzrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "cls $zero, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra;
	let Inst{23-0} = 0b001000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CLSrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "cls $rc, $ra";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Cls SimpleRegOrCst:$ra))];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100110000001000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CLSzr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra);
	let AsmString        = "cls $zero, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra;
	let Inst{33-0} = 0b1100110000001000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CLS_Srrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cls.s $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00100000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLS_Srrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "cls.s $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b001000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLS_Srr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "cls.s $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000001000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLS_Urrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cls.u $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00100000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLS_Urrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "cls.u $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b001000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLS_Urr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "cls.u $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000001000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLZrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "clz $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00110000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CLZzrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "clz $zero, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00110000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CLZrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "clz $rc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{23-0} = 0b001100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CLZzrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "clz $zero, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra;
	let Inst{23-0} = 0b001100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CLZrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "clz $rc, $ra";
	let Pattern          = [(set SimpleReg:$rc, (ctlz SimpleRegOrCst:$ra))];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100110000001100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CLZzr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra);
	let AsmString        = "clz $zero, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra;
	let Inst{33-0} = 0b1100110000001100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CLZ_Srrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "clz.s $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00110000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLZ_Srrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "clz.s $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b001100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLZ_Srr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "clz.s $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000001100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLZ_Urrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, count_nz_cc:$cc, pc16:$pc);
	let AsmString        = "clz.u $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00110000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLZ_Urrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "clz.u $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b001100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CLZ_Urr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "clz.u $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000001100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CMPB4rrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cmpb4 $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CMPB4zrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cmpb4 $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CMPB4rrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "cmpb4 $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CMPB4zrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "cmpb4 $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CMPB4rrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "cmpb4 $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100000000000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def CMPB4zrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "cmpb4 $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100000000000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def CMPB4_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cmpb4.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CMPB4_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "cmpb4.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CMPB4_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "cmpb4.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100000000000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CMPB4_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "cmpb4.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CMPB4_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "cmpb4.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def CMPB4_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "cmpb4.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100000000000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def DIV_STEPrrrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, DoubleReg:$db, u5_imm:$shift, div_cc:$cc, pc16:$pc);
	let AsmString        = "div_step $dc, $ra, $db, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> db; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b1;
	let Inst{19-17} = db{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b1000;
}

def DIV_STEPrrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, DoubleReg:$db, u5_imm:$shift);
	let AsmString        = "div_step $dc, $ra, $db, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> db; bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b10000000000000000;
	let Inst{19-17} = db{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTSBrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extsb $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01010000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTSBzrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extsb $zero, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01010000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTSBrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extsb $rc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{23-0} = 0b010100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTSBzrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extsb $zero, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra;
	let Inst{23-0} = 0b010100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTSBrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "extsb $rc, $ra";
	let Pattern          = [(set SimpleReg:$rc, (sext_inreg SimpleRegOrCst:$ra, i8))];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100110000010100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTSBzr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra);
	let AsmString        = "extsb $zero, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra;
	let Inst{33-0} = 0b1100110000010100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTSB_Srrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extsb.s $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01010000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTSB_Srrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extsb.s $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b010100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTSB_Srr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "extsb.s $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000010100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTSHrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extsh $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01110000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTSHzrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extsh $zero, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01110000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTSHrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extsh $rc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{23-0} = 0b011100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTSHzrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extsh $zero, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra;
	let Inst{23-0} = 0b011100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTSHrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "extsh $rc, $ra";
	let Pattern          = [(set SimpleReg:$rc, (sext_inreg SimpleRegOrCst:$ra, i16))];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100110000011100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTSHzr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra);
	let AsmString        = "extsh $zero, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra;
	let Inst{33-0} = 0b1100110000011100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTSH_Srrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extsh.s $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01110000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTSH_Srrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extsh.s $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b011100000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTSH_Srr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "extsh.s $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000011100000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTUBrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extub $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTUBzrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extub $zero, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTUBrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extub $rc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{23-0} = 0b010000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTUBzrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extub $zero, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra;
	let Inst{23-0} = 0b010000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTUBrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "extub $rc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100110000010000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTUBzr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra);
	let AsmString        = "extub $zero, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra;
	let Inst{33-0} = 0b1100110000010000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTUB_Urrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extub.u $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTUB_Urrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extub.u $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b010000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTUB_Urr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "extub.u $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000010000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTUHrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extuh $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01100000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTUHzrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extuh $zero, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01100000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTUHrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extuh $rc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{23-0} = 0b011000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTUHzrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extuh $zero, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra;
	let Inst{23-0} = 0b011000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTUHrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "extuh $rc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100110000011000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def EXTUHzr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra);
	let AsmString        = "extuh $zero, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra;
	let Inst{33-0} = 0b1100110000011000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def EXTUH_Urrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "extuh.u $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b01100000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTUH_Urrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "extuh.u $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b011000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def EXTUH_Urr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "extuh.u $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100110000011000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def FAULTi: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins s24_imm:$imm);
	let AsmString        = "fault $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{47-24} = 0b011111100110001100100000;
}

def HASHrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "hash $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110001;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def HASHzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s12_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "hash $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<12> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{11};
	let Inst{47-45} = 0b100;
}

def HASHrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "hash $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110001;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def HASHzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, log_set_cc:$cc);
	let AsmString        = "hash $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<4> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def HASHrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "hash $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100010000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def HASHzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, false_cc:$cc);
	let AsmString        = "hash $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100010000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def HASHrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "hash $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def HASHzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "hash $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def HASHrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "hash $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def HASHzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "hash $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def HASHrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "hash $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000000001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def HASHzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "hash $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000000001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def HASHrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm);
	let AsmString        = "hash $rc, $ra, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100010000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def HASH_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "hash.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "hash.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "hash.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100010000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "hash.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "hash.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "hash.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000000001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "hash.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "hash.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "hash.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100010000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "hash.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "hash.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def HASH_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "hash.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000000001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def JEQrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s11_imm:$imm, pc16:$pc);
	let AsmString        = "jeq $ra, $imm, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110000010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def JEQrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, pc16:$pc);
	let AsmString        = "jeq $ra, $rb, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100000101100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JGESrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s11_imm:$imm, pc16:$pc);
	let AsmString        = "jges $ra, $imm, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def JGESrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, pc16:$pc);
	let AsmString        = "jges $ra, $rb, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100101111100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JGEUrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s11_imm:$imm, pc16:$pc);
	let AsmString        = "jgeu $ra, $imm, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010101;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def JGEUrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, pc16:$pc);
	let AsmString        = "jgeu $ra, $rb, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100101011100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JGTSrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s11_imm:$imm, pc16:$pc);
	let AsmString        = "jgts $ra, $imm, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110011001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def JGTSrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, pc16:$pc);
	let AsmString        = "jgts $ra, $rb, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100110011100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JGTUrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s11_imm:$imm, pc16:$pc);
	let AsmString        = "jgtu $ra, $imm, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110011011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def JGTUrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, pc16:$pc);
	let AsmString        = "jgtu $ra, $rb, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100110111100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JLESrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s11_imm:$imm, pc16:$pc);
	let AsmString        = "jles $ra, $imm, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110011000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def JLESrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, pc16:$pc);
	let AsmString        = "jles $ra, $rb, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100110001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JLEUrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s11_imm:$imm, pc16:$pc);
	let AsmString        = "jleu $ra, $imm, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110011010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def JLEUrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, pc16:$pc);
	let AsmString        = "jleu $ra, $rb, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100110101100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JLTSrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s11_imm:$imm, pc16:$pc);
	let AsmString        = "jlts $ra, $imm, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010110;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def JLTSrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, pc16:$pc);
	let AsmString        = "jlts $ra, $rb, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100101101100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JLTUrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s11_imm:$imm, pc16:$pc);
	let AsmString        = "jltu $ra, $imm, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010100;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def JLTUrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, pc16:$pc);
	let AsmString        = "jltu $ra, $rb, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100101001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JNEQrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s11_imm:$imm, pc16:$pc);
	let AsmString        = "jneq $ra, $imm, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110000011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def JNEQrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, pc16:$pc);
	let AsmString        = "jneq $ra, $rb, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100000111100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JNZri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, pc16:$pc);
	let AsmString        = "jnz $ra, $pc";
	let Pattern          = [(brcond SimpleRegOrCst:$ra, bb:$pc)];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{33-16} = 0b111000001100000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def JUMPri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, pc28:$off);
	let AsmString        = "jump $ra, $off";
	let Pattern          = [(brind (add SimpleRegOrCst:$ra, pc28:$off))];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isBarrier        = 1;
	let isTerminator     = 1;
	let isIndirectBranch = 1;
	bits<28> off; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1100000000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = off{27};
	let Inst{47-45} = 0b100;
}

def JUMPi: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins pcbb:$off);
	let AsmString        = "jump $off";
	let Pattern          = [(br bb:$off)];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isBarrier        = 1;
	let isTerminator     = 1;
	bits<28> off;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{38-24} = 0b110001100000000;
	let Inst{41-39} = off{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = off{27};
	let Inst{47-45} = 0b100;
}

def JUMPr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "jump $ra";
	let Pattern          = [(brind SimpleRegOrCst:$ra)];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isBarrier        = 1;
	let isTerminator     = 1;
	let isIndirectBranch = 1;
	bits<5> ra;
	let Inst{33-0} = 0b1100000000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def JZri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, pc16:$pc);
	let AsmString        = "jz $ra, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{33-16} = 0b111000001000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def LBSerri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lbs $endian, $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b001;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110100;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LBSersi: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s24_imm:$off);
	let AsmString        = "lbs $endian, $rc, $sa, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> sa; bits<5> sc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b001;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110101;
	let Inst{38-34} = sa{4-0};
	let Inst{43-39} = sc{4-0};
	let Inst{47-44} = 0b0111;
}

def LBSrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lbs $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101000001;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LBS_Serri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lbs.s $endian, $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b001;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110110;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LBS_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lbs.s $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101100001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LBSSrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lbss $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101010001;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LBUerri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lbu $endian, $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b000;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110100;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LBUersi: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s24_imm:$off);
	let AsmString        = "lbu $endian, $rc, $sa, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> sa; bits<5> sc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b000;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110101;
	let Inst{38-34} = sa{4-0};
	let Inst{43-39} = sc{4-0};
	let Inst{47-44} = 0b0111;
}

def LBUrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lbu $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LBU_Uerri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lbu.u $endian, $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b000;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110110;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LBU_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lbu.u $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LBUSrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lbus $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101010000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LDerri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "ld $endian, $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b110;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110100;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b0111;
}

def LDersi: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s24_imm:$off);
	let AsmString        = "ld $endian, $dc, $sa, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> sa; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b110;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110101;
	let Inst{38-34} = sa{4-0};
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b0111;
}

def LDrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "ld $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101000110;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b0111;
}

def LDMArri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, u8_imm:$immDma);
	let AsmString        = "ldma $ra, $rb, $immDma";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<8> immDma; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{31-24} = immDma{7-0};
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011100000;
}

def LDMA_DBGrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, u8_imm:$immDma);
	let AsmString        = "ldma_dbg $ra, $rb, $immDma";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<8> immDma; bits<5> ra;
	let Inst{16-0} = 0b00000000000000100;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{31-24} = immDma{7-0};
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011100000;
}

def LDMAIrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, u8_imm:$immDma);
	let AsmString        = "ldmai $ra, $rb, $immDma";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<8> immDma; bits<5> ra;
	let Inst{16-0} = 0b00000000000000001;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{31-24} = immDma{7-0};
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011100000;
}

def LDSrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lds $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101010110;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b0111;
}

def LHSerri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lhs $endian, $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b011;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110100;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LHSersi: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s24_imm:$off);
	let AsmString        = "lhs $endian, $rc, $sa, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> sa; bits<5> sc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b011;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110101;
	let Inst{38-34} = sa{4-0};
	let Inst{43-39} = sc{4-0};
	let Inst{47-44} = 0b0111;
}

def LHSrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lhs $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101000011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LHS_Serri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lhs.s $endian, $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b011;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110110;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LHS_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lhs.s $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101100011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LHSSrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lhss $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101010011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LHUerri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lhu $endian, $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b010;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110100;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LHUersi: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s24_imm:$off);
	let AsmString        = "lhu $endian, $rc, $sa, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> sa; bits<5> sc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b010;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110101;
	let Inst{38-34} = sa{4-0};
	let Inst{43-39} = sc{4-0};
	let Inst{47-44} = 0b0111;
}

def LHUrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lhu $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101000010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LHU_Uerri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lhu.u $endian, $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b010;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110110;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LHU_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lhu.u $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101100010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LHUSrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lhus $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101010010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LSLrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0100;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0100;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b01000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b01000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (shl SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b01000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b01000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (shl SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0100;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0100;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1rrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1 $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0110;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1zrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1 $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0110;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1rric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl1 $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b01100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1zric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl1 $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b01100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1rri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl1 $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lsl1 SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b01100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1zri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl1 $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b01100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1rrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1 $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1zrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1 $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1rrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl1 $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1zrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl1 $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1rrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl1 $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lsl1 SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1zrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl1 $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0110;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl1.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl1.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl1.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl1.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0110;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl1.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl1.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl1.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl1.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1Xrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1x $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0111;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1Xzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1x $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0111;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1Xrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl1x $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b01110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1Xzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl1x $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b01110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1Xrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl1x $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lsl1x SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b01110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1Xzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl1x $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b01110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1Xrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1x $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1Xzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1x $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1Xrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl1x $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1Xzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl1x $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1Xrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl1x $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lsl1x SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL1Xzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl1x $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL1X_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1x.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0111;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl1x.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl1x.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1x.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl1x.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl1x.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1x.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0111;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsl1x.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl1x.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl1x.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsl1x.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL1X_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsl1x.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_ADDrrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl_add $rc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL_ADDzrrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl_add $zero, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL_ADDrrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl_add $rc, $rb, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (add SimpleReg:$rb, (shl SimpleRegOrCst:$ra, u5_imm:$shift)))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL_ADDzrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl_add $zero, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL_ADD_Srrrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl_add.s $dc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_ADD_Srrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl_add.s $dc, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_ADD_Urrrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl_add.u $dc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_ADD_Urrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl_add.u $dc, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_SUBrrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl_sub $rc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL_SUBzrrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl_sub $zero, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL_SUBrrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl_sub $rc, $rb, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (sub SimpleReg:$rb, (shl SimpleRegOrCst:$ra, u5_imm:$shift)))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSL_SUBzrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl_sub $zero, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSL_SUB_Srrrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl_sub.s $dc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_SUB_Srrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl_sub.s $dc, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_SUB_Urrrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsl_sub.u $dc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSL_SUB_Urrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsl_sub.u $dc, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLXrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lslx $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0101;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLXzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lslx $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0101;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLXrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lslx $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b01010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLXzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lslx $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b01010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLXrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lslx $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lslx SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b01010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLXzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lslx $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b01010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLXrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lslx $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLXzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lslx $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLXrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lslx $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLXzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lslx $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLXrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lslx $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lslx SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSLXzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lslx $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSLX_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lslx.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0101;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lslx.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lslx.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lslx.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lslx.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lslx.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lslx.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0101;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lslx.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lslx.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b01010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lslx.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lslx.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSLX_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lslx.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1100;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1100;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b11000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b11000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (srl SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b11000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b11000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (srl SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1100;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1100;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11000000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1rrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1 $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1110;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1zrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1 $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1110;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1rric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr1 $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b11100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1zric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr1 $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b11100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1rri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr1 $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lsr1 SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b11100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1zri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr1 $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b11100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1rrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1 $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1zrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1 $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1rrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr1 $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1zrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr1 $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1rrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr1 $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lsr1 SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1zrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr1 $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1110;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr1.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr1.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr1.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr1.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1110;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr1.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr1.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr1.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr1.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1Xrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1x $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1111;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1Xzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1x $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1111;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1Xrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr1x $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b11110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1Xzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr1x $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b11110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1Xrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr1x $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lsr1x SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b11110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1Xzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr1x $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b11110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1Xrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1x $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1Xzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1x $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1Xrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr1x $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1Xzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr1x $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1Xrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr1x $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lsr1x SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR1Xzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr1x $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR1X_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1x.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1111;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr1x.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr1x.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1x.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr1x.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr1x.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1x.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1111;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsr1x.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr1x.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11110000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr1x.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsr1x.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR1X_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsr1x.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_ADDrrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr_add $rc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR_ADDzrrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr_add $zero, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR_ADDrrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr_add $rc, $rb, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (add SimpleReg:$rb, (srl SimpleRegOrCst:$ra, u5_imm:$shift)))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSR_ADDzrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr_add $zero, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSR_ADD_Srrrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr_add.s $dc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_ADD_Srrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr_add.s $dc, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_ADD_Urrrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsr_add.u $dc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSR_ADD_Urrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsr_add.u $dc, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRXrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsrx $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1101;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRXzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsrx $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1101;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRXrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsrx $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b11010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRXzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsrx $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b11010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRXrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsrx $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lsrx SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b11010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRXzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsrx $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b11010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRXrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsrx $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRXzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsrx $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRXrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsrx $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRXzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsrx $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRXrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsrx $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Lsrx SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def LSRXzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsrx $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def LSRX_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsrx.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1101;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsrx.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsrx.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsrx.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsrx.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsrx.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsrx.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1101;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "lsrx.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "lsrx.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b11010000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "lsrx.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "lsrx.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LSRX_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "lsrx.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def LWerri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lw $endian, $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b100;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110100;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LWersi: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s24_imm:$off);
	let AsmString        = "lw $endian, $rc, $sa, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> sa; bits<5> sc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b100;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110101;
	let Inst{38-34} = sa{4-0};
	let Inst{43-39} = sc{4-0};
	let Inst{47-44} = 0b0111;
}

def LWrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lw $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101000100;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def LW_Serri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lw.s $endian, $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b100;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110110;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LW_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lw.s $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101100100;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LW_Uerri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lw.u $endian, $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<1> endian; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{26-24} = 0b100;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110110;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LW_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lw.u $dc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> dc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101100100;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0111;
}

def LWSrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off);
	let AsmString        = "lws $rc, $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra; bits<5> rc;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1101010100;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0111;
}

def MOVDrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins DoubleReg:$db, true_false_cc:$cc, pc16:$pc);
	let AsmString        = "movd $dc, $db, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> db; bits<4> cc; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b1;
	let Inst{19-17} = db{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0010;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = 0b11000;
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b1000;
}

def MOVDrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins DoubleReg:$db);
	let AsmString        = "movd $dc, $db";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> db; bits<5> dc;
	let Inst{16-0} = 0b10000000000000000;
	let Inst{19-17} = db{2-0};
	let Inst{31-20} = 0b001000000000;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = 0b11000;
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b1000;
}

def MOVErrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "move $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def MOVErici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "move $rc, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<8> imm; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{38-28} = 0b11000111011;
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def MOVEri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins u32_imm:$imm);
	let AsmString        = "move $rc, $imm";
	let Pattern          = [(set SimpleReg:$rc, u32_imm:$imm)];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{38-32} = 0b1100011;
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0110;
}

def MOVErr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "move $rc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1110110000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def MOVE_Srrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "move.s $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def MOVE_Srici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "move.s $dc, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<8> imm; bits<4> cc; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{39-28} = 0b111000111011;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def MOVE_Srr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "move.s $dc, $ra";
	let Pattern          = [(set DoubleReg:$dc, (sext SimpleRegOrCst:$ra))];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1110110000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def MOVE_Sri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins u32_imm:$imm);
	let AsmString        = "move.s $dc, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{39-32} = 0b11101011;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0101;
}

def MOVE_Uri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins u32_imm:$imm);
	let AsmString        = "move.u $dc, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{39-32} = 0b01101011;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b0101;
}

def MOVE_Urici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "move.u $dc, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<8> imm; bits<4> cc; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{39-28} = 0b011000111011;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def MOVE_Urrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "move.u $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def MOVE_Urr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "move.u $dc, $ra";
	let Pattern          = [(set DoubleReg:$dc, (zext SimpleRegOrCst:$ra))];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1110110000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def MUL_SH_SHrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_sh $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_SHzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_sh $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_SHrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_sh $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_SHzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_sh $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_SHrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_sh $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_SHzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_sh $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_SH_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_sh.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_SH_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_sh.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_SH_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_sh.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_SLrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_sl $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_SLzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_sl $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_SLrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_sl $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_SLzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_sl $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_SLrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_sl $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_SLzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_sl $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_SL_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_sl.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_SL_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_sl.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_SL_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_sl.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_UHrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_uh $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_UHzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_uh $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_UHrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_uh $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_UHzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_uh $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_UHrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_uh $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_UHzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_uh $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_UH_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_uh.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_UH_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_uh.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_UH_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_uh.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_ULrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_ul $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_ULzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_ul $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_ULrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_ul $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_ULzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_ul $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_ULrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_ul $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SH_ULzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_ul $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SH_UL_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sh_ul.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_UL_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sh_ul.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SH_UL_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sh_ul.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_SHrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_sh $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_SHzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_sh $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_SHrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_sh $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_SHzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_sh $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_SHrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_sh $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_SHzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_sh $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_SH_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_sh.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_SH_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_sh.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_SH_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_sh.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_SLrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_sl $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_SLzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_sl $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_SLrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_sl $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_SLzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_sl $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_SLrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_sl $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Mul8SS SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_SLzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_sl $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_SL_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_sl.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_SL_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_sl.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0100;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_SL_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_sl.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_UHrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_uh $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_UHzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_uh $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_UHrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_uh $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_UHzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_uh $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_UHrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_uh $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_UHzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_uh $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_UH_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_uh.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_UH_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_uh.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_UH_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_uh.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_ULrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_ul $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_ULzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_ul $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_ULrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_ul $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_ULzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_ul $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_ULrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_ul $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Mul8SU SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_SL_ULzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_ul $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_SL_UL_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_sl_ul.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_UL_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_sl_ul.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_SL_UL_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_sl_ul.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_STEPrrrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCstButZero:$ra, DoubleReg:$db, u5_imm:$shift, boot_cc:$cc, pc16:$pc);
	let AsmString        = "mul_step $dc, $ra, $db, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> db; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b1;
	let Inst{19-17} = db{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b1000;
}

def MUL_STEPrrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCstButZero:$ra, DoubleReg:$db, u5_imm:$shift);
	let AsmString        = "mul_step $dc, $ra, $db, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> db; bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b10000000000000000;
	let Inst{19-17} = db{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b1000;
}

def MUL_UH_UHrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_uh_uh $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0011;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UH_UHzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_uh_uh $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0011;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UH_UHrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_uh_uh $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UH_UHzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_uh_uh $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UH_UHrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_uh_uh $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UH_UHzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_uh_uh $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UH_UH_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_uh_uh.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0011;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UH_UH_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_uh_uh.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UH_UH_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_uh_uh.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UH_ULrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_uh_ul $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UH_ULzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_uh_ul $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UH_ULrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_uh_ul $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UH_ULzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_uh_ul $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UH_ULrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_uh_ul $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UH_ULzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_uh_ul $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UH_UL_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_uh_ul.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UH_UL_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_uh_ul.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UH_UL_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_uh_ul.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UL_UHrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_ul_uh $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UL_UHzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_ul_uh $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UL_UHrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_ul_uh $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UL_UHzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_ul_uh $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UL_UHrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_ul_uh $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UL_UHzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_ul_uh $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UL_UH_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_ul_uh.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UL_UH_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_ul_uh.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UL_UH_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_ul_uh.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UL_ULrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_ul_ul $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UL_ULzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_ul_ul $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UL_ULrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_ul_ul $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UL_ULzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_ul_ul $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UL_ULrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_ul_ul $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (SDT_Mul8UU SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def MUL_UL_ULzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_ul_ul $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def MUL_UL_UL_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, mul_nz_cc:$cc, pc16:$pc);
	let AsmString        = "mul_ul_ul.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UL_UL_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "mul_ul_ul.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def MUL_UL_UL_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "mul_ul_ul.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b000000000000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def NANDrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nand $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111111;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NANDzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s12_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nand $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<12> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{11};
	let Inst{47-45} = 0b100;
}

def NANDrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nand $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111111;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NANDzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nand $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<4> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def NANDrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "nand $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111110000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NANDzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, false_cc:$cc);
	let AsmString        = "nand $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111110000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def NANDrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nand $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NANDzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nand $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def NANDrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nand $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NANDzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nand $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def NANDrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nand $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (not (and SimpleRegOrCst:$ra, SimpleReg:$rb)))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NANDzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nand $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def NANDrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm);
	let AsmString        = "nand $rc, $ra, $imm";
	let Pattern          = [(set SimpleReg:$rc, (not (and SimpleRegOrCst:$ra, s24_imm:$imm)))];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111110000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NAND_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nand.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nand.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "nand.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111110000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nand.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nand.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nand.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nand.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nand.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "nand.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111110000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nand.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nand.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1111;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NAND_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nand.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001111;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NEGrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "neg $rc, $ra";
	let Pattern          = [(set SimpleReg:$rc, (ineg SimpleRegOrCst:$ra))];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100000000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0010;
}

def NEGrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "neg $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def NOP: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins);
	let AsmString        = "nop";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let Inst{47-0} = 0b011111000110001100000000000000000000000000000000;
}

def NORrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nor $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111110;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NORzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s12_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nor $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<12> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111110;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{11};
	let Inst{47-45} = 0b100;
}

def NORrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nor $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111110;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NORzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nor $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<4> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111110;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def NORrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "nor $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111100000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NORzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, false_cc:$cc);
	let AsmString        = "nor $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111100000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def NORrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nor $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NORzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nor $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def NORrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nor $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NORzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nor $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def NORrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nor $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (not (or SimpleRegOrCst:$ra, SimpleReg:$rb)))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NORzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nor $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def NORrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm);
	let AsmString        = "nor $rc, $ra, $imm";
	let Pattern          = [(set SimpleReg:$rc, (not (or SimpleRegOrCst:$ra, s24_imm:$imm)))];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111100000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NOR_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nor.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111110;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nor.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111110;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "nor.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nor.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nor.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nor.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nor.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111110;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nor.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111110;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "nor.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nor.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nor.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1110;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOR_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nor.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001110;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NOTrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "not $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b11111111;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NOTrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "not $rc, $ra";
	let Pattern          = [(set SimpleReg:$rc, (not SimpleRegOrCst:$ra))];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1111111111111111111111111111111111;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0100;
}

def NXORrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nxor $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111001;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NXORzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s12_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nxor $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<12> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{11};
	let Inst{47-45} = 0b100;
}

def NXORrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nxor $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111001;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NXORzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nxor $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<4> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def NXORrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "nxor $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NXORzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, false_cc:$cc);
	let AsmString        = "nxor $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def NXORrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nxor $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NXORzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nxor $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def NXORrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nxor $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NXORzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nxor $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def NXORrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nxor $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (not (xor SimpleRegOrCst:$ra, SimpleReg:$rb)))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NXORzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nxor $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def NXORrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm);
	let AsmString        = "nxor $rc, $ra, $imm";
	let Pattern          = [(set SimpleReg:$rc, (not (xor SimpleRegOrCst:$ra, s24_imm:$imm)))];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def NXOR_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nxor.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nxor.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "nxor.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nxor.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nxor.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nxor.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nxor.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "nxor.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111001;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "nxor.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "nxor.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "nxor.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1001;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def NXOR_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "nxor.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u32_imm:$imm);
	let AsmString        = "or $rc, $ra, $imm";
	let Pattern          = [(set SimpleReg:$rc, (or SimpleRegOrCst:$ra, u32_imm:$imm))];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = 0b11;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0110;
}

def ORzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "or $zero, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{38-37} = 0b11;
	let Inst{39} = imm{7};
	let Inst{47-40} = 0b01100110;
}

def ORrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "or $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s12_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "or $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<12> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{11};
	let Inst{47-45} = 0b100;
}

def ORrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "or $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, log_set_cc:$cc);
	let AsmString        = "or $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<4> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def ORrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "or $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110110000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, false_cc:$cc);
	let AsmString        = "or $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110110000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def ORrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "or $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "or $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ORrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "or $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "or $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ORrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "or $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (or SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "or $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def OR_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "or.s $dc, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{39-37} = 0b111;
	let Inst{43-40} = dc{4-1};
	let Inst{44} = imm{7};
	let Inst{47-45} = 0b010;
}

def OR_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "or.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "or.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "or.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110110000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "or.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "or.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "or.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "or.u $dc, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{39-37} = 0b011;
	let Inst{43-40} = dc{4-1};
	let Inst{44} = imm{7};
	let Inst{47-45} = 0b010;
}

def OR_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "or.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "or.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "or.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110110000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "or.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "or.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1011;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def OR_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "or.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORNrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "orn $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORNzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s12_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "orn $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<12> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111101;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{11};
	let Inst{47-45} = 0b100;
}

def ORNrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "orn $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORNzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, log_set_cc:$cc);
	let AsmString        = "orn $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<4> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111101;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def ORNrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "orn $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111010000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORNzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, false_cc:$cc);
	let AsmString        = "orn $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111010000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def ORNrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "orn $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORNzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "orn $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ORNrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "orn $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORNzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "orn $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ORNrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "orn $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (or (not SimpleRegOrCst:$ra), SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORNzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "orn $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ORNrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm);
	let AsmString        = "orn $rc, $ra, $imm";
	let Pattern          = [(set SimpleReg:$rc, (or (not SimpleRegOrCst:$ra), s24_imm:$imm))];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111010000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ORN_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "orn.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "orn.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "orn.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111010000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "orn.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "orn.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "orn.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "orn.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "orn.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "orn.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1111010000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "orn.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "orn.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1101;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ORN_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "orn.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def READ_RUNrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, boot_cc:$cc, pc16:$pc);
	let AsmString        = "read_run $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<8> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110011;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011111000;
}

def RELEASErici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s16_imm:$imm, release_cc:$cc, pc16:$pc);
	let AsmString        = "release $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isBarrier        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<16> imm; bits<2> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{25-24} = cc{1-0};
	let Inst{30-26} = imm{12-8};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{15-13};
	let Inst{47-42} = 0b011111;
}

def RESUMErici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, boot_cc:$cc, pc16:$pc);
	let AsmString        = "resume $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<8> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011111010;
}

def RESUMEri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm);
	let AsmString        = "resume $ra, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<8> imm; bits<5> ra;
	let Inst{15-0} = 0b0000000000000000;
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1100100000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011111010;
}

def ROLrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0010;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ROLzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0010;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ROLrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "rol $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b00100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ROLzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "rol $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b00100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ROLrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "rol $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (rotl SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b00100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ROLzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "rol $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b00100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ROLrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ROLzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ROLrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "rol $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ROLzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "rol $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ROLrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rol $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (rotl SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ROLzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rol $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ROL_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0010;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "rol.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b00100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "rol.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b00100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "rol.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rol.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b0010;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "rol.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b00100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "rol.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b00100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "rol.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rol.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100000010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_ADDrrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol_add $rc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ROL_ADDzrrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol_add $zero, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ROL_ADDrrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "rol_add $rc, $rb, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (add SimpleReg:$rb, (rotl SimpleRegOrCst:$ra, u5_imm:$shift)))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def ROL_ADDzrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "rol_add $zero, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ROL_ADD_Srrrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol_add.s $dc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_ADD_Srrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "rol_add.s $dc, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_ADD_Urrrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift, div_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rol_add.u $dc, $rb, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROL_ADD_Urrri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "rol_add.u $dc, $rb, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{31-29} = 0b000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def RORrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "ror $rc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1010;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def RORzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "ror $zero, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> shift; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1010;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def RORrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "ror $rc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b10100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def RORzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "ror $zero, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra;
	let Inst{19-0} = 0b10100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def RORrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "ror $rc, $ra, $shift";
	let Pattern          = [(set SimpleReg:$rc, (rotr SimpleRegOrCst:$ra, u5_imm:$shift))];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> rc;
	let Inst{19-0} = 0b10100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def RORzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "ror $zero, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra;
	let Inst{19-0} = 0b10100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def RORrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "ror $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def RORzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "ror $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def RORrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "ror $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def RORzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "ror $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def RORrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "ror $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (rotr SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def RORzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "ror $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def ROR_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "ror.s $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1010;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "ror.s $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b10100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Srri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "ror.s $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b10100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "ror.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "ror.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "ror.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, imm_shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "ror.u $dc, $ra, $shift, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> shift; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = 0b1010;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{29} = cc{4};
	let Inst{33-30} = 0b1101;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift, log_set_cc:$cc);
	let AsmString        = "ror.u $dc, $ra, $shift, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b10100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Urri: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u5_imm:$shift);
	let AsmString        = "ror.u $dc, $ra, $shift";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> shift; bits<5> ra; bits<5> dc;
	let Inst{19-0} = 0b10100000000000000000;
	let Inst{23-20} = shift{3-0};
	let Inst{27-24} = 0b0000;
	let Inst{28} = shift{4};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, shift_nz_cc:$cc, pc16:$pc);
	let AsmString        = "ror.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{28} = 0b1;
	let Inst{29} = cc{4};
	let Inst{31-30} = 0b11;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "ror.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1010;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1101;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def ROR_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "ror.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110100001010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def RSUBrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rsub $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def RSUBzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rsub $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def RSUBrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_set_cc:$cc);
	let AsmString        = "rsub $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def RSUBzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, sub_set_cc:$cc);
	let AsmString        = "rsub $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def RSUBrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rsub $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b010000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def RSUBzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rsub $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b010000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def RSUB_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rsub.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUB_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_set_cc:$cc);
	let AsmString        = "rsub.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUB_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rsub.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b010000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUB_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rsub.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUB_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_set_cc:$cc);
	let AsmString        = "rsub.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUB_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rsub.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b010000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUBCrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rsubc $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def RSUBCzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rsubc $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def RSUBCrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_set_cc:$cc);
	let AsmString        = "rsubc $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def RSUBCzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, sub_set_cc:$cc);
	let AsmString        = "rsubc $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def RSUBCrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rsubc $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b011000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def RSUBCzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rsubc $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b011000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def RSUBC_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rsubc.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUBC_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_set_cc:$cc);
	let AsmString        = "rsubc.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUBC_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rsubc.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b011000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUBC_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "rsubc.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUBC_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_set_cc:$cc);
	let AsmString        = "rsubc.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{31-29} = 0b011;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def RSUBC_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "rsubc.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b011000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SATSrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sats $rc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def SATSzrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sats $zero, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def SATSrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "sats $rc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{23-0} = 0b000000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def SATSzrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "sats $zero, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra;
	let Inst{23-0} = 0b000000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def SATSrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "sats $rc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rc;
	let Inst{33-0} = 0b1100100000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def SATSzr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra);
	let AsmString        = "sats $zero, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra;
	let Inst{33-0} = 0b1100100000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def SATS_Srrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sats.s $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def SATS_Srrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "sats.s $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b000000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def SATS_Srr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "sats.s $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100100000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def SATS_Urrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sats.u $dc, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def SATS_Urrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, log_set_cc:$cc);
	let AsmString        = "sats.u $dc, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{23-0} = 0b000000000000000000000000;
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b110010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def SATS_Urr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra);
	let AsmString        = "sats.u $dc, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> dc;
	let Inst{33-0} = 0b1100100000000000000000000000000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def SBerir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sb $endian, $ra, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> rb; bits<1> endian; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b00;
	let Inst{27} = endian{0};
	let Inst{31-28} = 0b0100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SBerii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s12_imm:$off, su8_imm:$imm);
	let AsmString        = "sb $endian, $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<8> imm; bits<12> off; bits<1> endian; bits<5> ra;
	let Inst{3-0} = imm{7-4};
	let Inst{11-4} = 0b00000000;
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b00;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110100;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SBesir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sb $endian, $sa, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> sb; bits<1> endian; bits<5> sa;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = sb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b00;
	let Inst{27} = endian{0};
	let Inst{31-28} = 0b0101;
	let Inst{33-32} = sb{4-3};
	let Inst{38-34} = sa{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SBesii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s12_imm:$off, su8_imm:$imm);
	let AsmString        = "sb $endian, $sa, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<8> imm; bits<12> off; bits<1> endian; bits<5> sa;
	let Inst{3-0} = imm{7-4};
	let Inst{11-4} = 0b00000000;
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b00;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110101;
	let Inst{38-34} = sa{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SBrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, su8_imm:$imm);
	let AsmString        = "sb $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<8> imm; bits<12> off; bits<5> ra;
	let Inst{3-0} = imm{7-4};
	let Inst{11-4} = 0b00000000;
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110100000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SBrir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sb $ra, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> rb; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{31-25} = 0b0100000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SB_IDerii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s12_imm:$off, su8_imm:$imm);
	let AsmString        = "sb_id $endian, $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<8> imm; bits<12> off; bits<1> endian; bits<5> ra;
	let Inst{3-0} = imm{7-4};
	let Inst{11-4} = 0b00000000;
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b00;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110110;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SB_IDrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, su8_imm:$imm);
	let AsmString        = "sb_id $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<8> imm; bits<12> off; bits<5> ra;
	let Inst{3-0} = imm{7-4};
	let Inst{11-4} = 0b00000000;
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110110000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SB_IDri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off);
	let AsmString        = "sb_id $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<12> off; bits<5> ra;
	let Inst{11-0} = 0b000000000000;
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = 0b0000;
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110110000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SBSrir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sbs $ra, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> rb; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{31-25} = 0b0101000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SBSrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, s8_imm:$imm);
	let AsmString        = "sbs $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<8> imm; bits<12> off; bits<5> ra;
	let Inst{3-0} = imm{7-4};
	let Inst{11-4} = 0b00000000;
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110101000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SDerir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off, DoubleReg:$db);
	let AsmString        = "sd $endian, $ra, $off, $db";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> db; bits<1> endian; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b1;
	let Inst{19-17} = db{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b11;
	let Inst{27} = endian{0};
	let Inst{31-28} = 0b0100;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SDerii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sd $endian, $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<1> endian; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b11;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110100;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SDesir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s24_imm:$off, DoubleReg:$db);
	let AsmString        = "sd $endian, $sa, $off, $db";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> db; bits<1> endian; bits<5> sa;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b1;
	let Inst{19-17} = db{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b11;
	let Inst{27} = endian{0};
	let Inst{31-28} = 0b0101;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = sa{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SDesii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sd $endian, $sa, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<1> endian; bits<5> sa;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b11;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110101;
	let Inst{38-34} = sa{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SDrir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off, DoubleReg:$db);
	let AsmString        = "sd $ra, $off, $db";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> db; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b1;
	let Inst{19-17} = db{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{31-25} = 0b0100011;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SDrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sd $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110100011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SD_IDerii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sd_id $endian, $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<1> endian; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b11;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110110;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SD_IDrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sd_id $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110110011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SD_IDri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off);
	let AsmString        = "sd_id $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<12> off; bits<5> ra;
	let Inst{11-0} = 0b000000000000;
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = 0b0000;
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110110011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SDMArri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, u8_imm:$immDma);
	let AsmString        = "sdma $ra, $rb, $immDma";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<8> immDma; bits<5> ra;
	let Inst{16-0} = 0b00000000000000010;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{31-24} = immDma{7-0};
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011100000;
}

def SDMA_DBGrri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, u8_imm:$immDma);
	let AsmString        = "sdma_dbg $ra, $rb, $immDma";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<8> immDma; bits<5> ra;
	let Inst{16-0} = 0b00000000000000110;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{31-24} = immDma{7-0};
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011100000;
}

def SDSrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sds $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110101011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SDSrir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off, DoubleReg:$db);
	let AsmString        = "sds $ra, $off, $db";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> db; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b1;
	let Inst{19-17} = db{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{31-25} = 0b0101011;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SHerir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sh $endian, $ra, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> rb; bits<1> endian; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b01;
	let Inst{27} = endian{0};
	let Inst{31-28} = 0b0100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SHerii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s12_imm:$off, su16_imm:$imm);
	let AsmString        = "sh $endian, $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<1> endian; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b01;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110100;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SHesir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sh $endian, $sa, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> sb; bits<1> endian; bits<5> sa;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = sb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b01;
	let Inst{27} = endian{0};
	let Inst{31-28} = 0b0101;
	let Inst{33-32} = sb{4-3};
	let Inst{38-34} = sa{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SHesii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s12_imm:$off, su16_imm:$imm);
	let AsmString        = "sh $endian, $sa, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<1> endian; bits<5> sa;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b01;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110101;
	let Inst{38-34} = sa{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SHrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, su16_imm:$imm);
	let AsmString        = "sh $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110100001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SHrir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sh $ra, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> rb; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{31-25} = 0b0100001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SH_IDerii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s12_imm:$off, su16_imm:$imm);
	let AsmString        = "sh_id $endian, $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<1> endian; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b01;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110110;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SH_IDrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, su16_imm:$imm);
	let AsmString        = "sh_id $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110110001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SH_IDri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off);
	let AsmString        = "sh_id $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<12> off; bits<5> ra;
	let Inst{11-0} = 0b000000000000;
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = 0b0000;
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110110001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SHSrir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "shs $ra, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> rb; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{31-25} = 0b0101001;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SHSrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "shs $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110101001;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def STOPci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins boot_cc:$cc, pc16:$pc);
	let AsmString        = "stop $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b00000000;
	let Inst{27-24} = cc{3-0};
	let Inst{47-28} = 0b01111110111100110010;
}

def STOP: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins);
	let AsmString        = "stop";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let Inst{47-0} = 0b011111101111001100100000000000000000000000000000;
}

def SUBrir: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins u32_imm:$imm, SimpleRegOrCst:$ra);
	let AsmString        = "sub $rc, $imm, $ra";
	let Pattern          = [(set SimpleReg:$rc, (sub u32_imm:$imm, SimpleRegOrCst:$ra))];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = 0b11;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0010;
}

def SUBzir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, u32_imm:$imm, SimpleReg:$rb);
	let AsmString        = "sub $zero, $imm, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{38-37} = 0b11;
	let Inst{39} = imm{7};
	let Inst{47-40} = 0b01100010;
}

def SUBrirci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins s8_imm:$imm, SimpleRegOrCst:$ra, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub $rc, $imm, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBzirci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, s11_imm:$imm, SimpleRegOrCst:$ra, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub $zero, $imm, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def SUBrirc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, sub_set_cc:$cc);
	let AsmString        = "sub $rc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBzirc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, s27_imm:$imm, SimpleRegOrCst:$ra, sub_set_cc:$cc);
	let AsmString        = "sub $zero, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<27> imm; bits<5> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def SUBrirf: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, false_cc:$cc);
	let AsmString        = "sub $rc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1101000000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBzirf: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, s27_imm:$imm, SimpleRegOrCst:$ra, false_cc:$cc);
	let AsmString        = "sub $zero, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<27> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1101000000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def SUBrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<6> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s11_imm:$imm, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<6> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def SUBrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, ext_sub_set_cc:$cc);
	let AsmString        = "sub $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<6> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s27_imm:$imm, ext_sub_set_cc:$cc);
	let AsmString        = "sub $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<27> imm; bits<6> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def SUBrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "sub $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110000000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s27_imm:$imm, false_cc:$cc);
	let AsmString        = "sub $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<27> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110000000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def SUBssi: DPUInstruction {
	let OutOperandList   = (outs SafeReg:$sc);
	let InOperandList    = (ins SafeRegOrCst:$sa, s17_imm:$imm);
	let AsmString        = "sub $sc, $sa, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<17> imm; bits<5> sa; bits<5> sc;
	let Inst{4-0} = imm{16-12};
	let Inst{5} = imm{16};
	let Inst{6} = imm{16};
	let Inst{7} = imm{16};
	let Inst{8} = imm{16};
	let Inst{9} = imm{16};
	let Inst{10} = imm{16};
	let Inst{11} = imm{16};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010000;
	let Inst{38-34} = sa{4-0};
	let Inst{41-39} = sc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = sc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBsss: DPUInstruction {
	let OutOperandList   = (outs SafeReg:$sc);
	let InOperandList    = (ins SafeRegOrCst:$sa, SafeReg:$sb);
	let AsmString        = "sub $sc, $sa, $sb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> sa; bits<5> sb; bits<5> sc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = sa{2-0};
	let Inst{31-20} = 0b010100001100;
	let Inst{33-32} = sa{4-3};
	let Inst{38-34} = sb{4-0};
	let Inst{41-39} = sc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = sc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<6> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<6> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def SUBrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, ext_sub_set_cc:$cc);
	let AsmString        = "sub $rc, $ra, $rb, $cc";
	let Pattern          = [(set SimpleReg:$rc, (DPUSetCC (i32 imm:$cc), SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<6> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, ext_sub_set_cc:$cc);
	let AsmString        = "sub $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<6> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def SUBrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "sub $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (sub SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "sub $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def SUB_Srirci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s8_imm:$imm, SimpleRegOrCst:$ra, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub.s $dc, $imm, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Srirc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, sub_set_cc:$cc);
	let AsmString        = "sub.s $dc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Srirf: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, false_cc:$cc);
	let AsmString        = "sub.s $dc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1101000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<6> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, ext_sub_set_cc:$cc);
	let AsmString        = "sub.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<6> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "sub.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<6> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, ext_sub_set_cc:$cc);
	let AsmString        = "sub.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<6> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "sub.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Urirci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s8_imm:$imm, SimpleRegOrCst:$ra, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub.u $dc, $imm, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Urirc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, sub_set_cc:$cc);
	let AsmString        = "sub.u $dc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11010;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Urirf: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, false_cc:$cc);
	let AsmString        = "sub.u $dc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1101000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<6> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, ext_sub_set_cc:$cc);
	let AsmString        = "sub.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<6> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "sub.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "sub.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<6> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, ext_sub_set_cc:$cc);
	let AsmString        = "sub.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<6> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b0;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUB_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "sub.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b100000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBCrir: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins u32_imm:$imm, SimpleRegOrCst:$ra);
	let AsmString        = "subc $rc, $imm, $ra";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = 0b11;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0011;
}

def SUBCzir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, u32_imm:$imm, SimpleReg:$rb);
	let AsmString        = "subc $zero, $imm, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{38-37} = 0b11;
	let Inst{39} = imm{7};
	let Inst{47-40} = 0b01100011;
}

def SUBCrirci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins s8_imm:$imm, SimpleRegOrCst:$ra, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc $rc, $imm, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBCzirci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, s11_imm:$imm, SimpleRegOrCst:$ra, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc $zero, $imm, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<5> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def SUBCrirc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, sub_set_cc:$cc);
	let AsmString        = "subc $rc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBCzirc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, s27_imm:$imm, SimpleRegOrCst:$ra, sub_set_cc:$cc);
	let AsmString        = "subc $zero, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<27> imm; bits<5> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11011;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def SUBCrirf: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, false_cc:$cc);
	let AsmString        = "subc $rc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1101100000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBCzirf: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, s27_imm:$imm, SimpleRegOrCst:$ra, false_cc:$cc);
	let AsmString        = "subc $zero, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<27> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1101100000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def SUBCrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<6> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBCzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s11_imm:$imm, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<11> imm; bits<6> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{47-42} = 0b001111;
}

def SUBCrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, ext_sub_set_cc:$cc);
	let AsmString        = "subc $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<6> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBCzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s27_imm:$imm, ext_sub_set_cc:$cc);
	let AsmString        = "subc $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<27> imm; bits<6> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def SUBCrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "subc $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110100000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBCzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s27_imm:$imm, false_cc:$cc);
	let AsmString        = "subc $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<27> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110100000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{47-42} = 0b001111;
}

def SUBCrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<6> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBCzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<6> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def SUBCrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, ext_sub_set_cc:$cc);
	let AsmString        = "subc $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<6> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBCzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, ext_sub_set_cc:$cc);
	let AsmString        = "subc $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<6> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def SUBCrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "subc $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b101000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBCzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "subc $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b101000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b001111000;
}

def SUBC_Srirci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s8_imm:$imm, SimpleRegOrCst:$ra, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc.s $dc, $imm, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Srirc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, sub_set_cc:$cc);
	let AsmString        = "subc.s $dc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Srirf: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, false_cc:$cc);
	let AsmString        = "subc.s $dc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1101100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<6> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, ext_sub_set_cc:$cc);
	let AsmString        = "subc.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<6> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "subc.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<6> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, ext_sub_set_cc:$cc);
	let AsmString        = "subc.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<6> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "subc.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b101000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Urirci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s8_imm:$imm, SimpleRegOrCst:$ra, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc.u $dc, $imm, $ra, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<5> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Urirc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, sub_set_cc:$cc);
	let AsmString        = "subc.u $dc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{33-29} = 0b11011;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Urirf: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins s24_imm:$imm, SimpleRegOrCst:$ra, false_cc:$cc);
	let AsmString        = "subc.u $dc, $imm, $ra, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1101100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<6> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, ext_sub_set_cc:$cc);
	let AsmString        = "subc.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<6> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{33-31} = 0b111;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "subc.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110100000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, sub_nz_cc:$cc, pc16:$pc);
	let AsmString        = "subc.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<6> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, ext_sub_set_cc:$cc);
	let AsmString        = "subc.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<6> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1100;
	let Inst{28-24} = cc{4-0};
	let Inst{29} = 0b1;
	let Inst{30} = cc{5};
	let Inst{31} = 0b1;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBC_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "subc.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b101000001100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def SUBSrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "subs $rc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> ra; bits<5> rb; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = ra{2-0};
	let Inst{31-20} = 0b010100001100;
	let Inst{33-32} = ra{4-3};
	let Inst{38-34} = rb{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SUBSrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s17_imm:$imm);
	let AsmString        = "subs $rc, $ra, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<17> imm; bits<5> ra; bits<5> rc;
	let Inst{4-0} = imm{16-12};
	let Inst{5} = imm{16};
	let Inst{6} = imm{16};
	let Inst{7} = imm{16};
	let Inst{8} = imm{16};
	let Inst{9} = imm{16};
	let Inst{10} = imm{16};
	let Inst{11} = imm{16};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110010000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def SWerir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sw $endian, $ra, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> rb; bits<1> endian; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b10;
	let Inst{27} = endian{0};
	let Inst{31-28} = 0b0100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SWerii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sw $endian, $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<1> endian; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b10;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110100;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SWesir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sw $endian, $sa, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> sb; bits<1> endian; bits<5> sa;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = sb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b10;
	let Inst{27} = endian{0};
	let Inst{31-28} = 0b0101;
	let Inst{33-32} = sb{4-3};
	let Inst{38-34} = sa{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SWesii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SafeRegOrCst:$sa, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sw $endian, $sa, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<1> endian; bits<5> sa;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b10;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110101;
	let Inst{38-34} = sa{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SWrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sw $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110100010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SWrir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sw $ra, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> rb; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{31-25} = 0b0100010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SW_IDerii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins endian:$endian, SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sw_id $endian, $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<1> endian; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{26-25} = 0b10;
	let Inst{27} = endian{0};
	let Inst{33-28} = 0b110110;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SW_IDrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sw_id $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110110010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SW_IDri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off);
	let AsmString        = "sw_id $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<12> off; bits<5> ra;
	let Inst{11-0} = 0b000000000000;
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = 0b0000;
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110110010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SWAPDrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins DoubleReg:$db, true_false_cc:$cc, pc16:$pc);
	let AsmString        = "swapd $dc, $db, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> db; bits<4> cc; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b1;
	let Inst{19-17} = db{2-0};
	let Inst{23-20} = 0b0000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1010;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = 0b11000;
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b1000;
}

def SWAPDrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins DoubleReg:$db);
	let AsmString        = "swapd $dc, $db";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> db; bits<5> dc;
	let Inst{16-0} = 0b10000000000000000;
	let Inst{19-17} = db{2-0};
	let Inst{31-20} = 0b101000000000;
	let Inst{33-32} = db{4-3};
	let Inst{38-34} = 0b11000;
	let Inst{43-39} = dc{4-0};
	let Inst{47-44} = 0b1000;
}

def SWSrir: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$off, SimpleReg:$rb);
	let AsmString        = "sws $ra, $off, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> rb; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{31-25} = 0b0101010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def SWSrii: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s12_imm:$off, s16_imm:$imm);
	let AsmString        = "sws $ra, $off, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<16> imm; bits<12> off; bits<5> ra;
	let Inst{11-0} = imm{15-4};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = imm{3-0};
	let Inst{23-20} = off{3-0};
	let Inst{24} = off{7};
	let Inst{33-25} = 0b110101010;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = off{6-4};
	let Inst{47-42} = 0b011111;
}

def TELLri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleReg:$ra, s24_imm:$off);
	let AsmString        = "tell $ra, $off";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> off; bits<5> ra;
	let Inst{11-0} = off{23-12};
	let Inst{12} = off{11};
	let Inst{13} = off{10};
	let Inst{14} = off{9};
	let Inst{15} = off{8};
	let Inst{19-16} = off{7-4};
	let Inst{23-20} = off{3-0};
	let Inst{33-24} = 0b1100000000;
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b011111000;
}

def TIMErci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins true_cc:$cc, pc16:$pc);
	let AsmString        = "time $rc, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b10000000;
	let Inst{27-24} = cc{3-0};
	let Inst{38-28} = 0b11000000010;
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def TIMEzci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, true_cc:$cc, pc16:$pc);
	let AsmString        = "time $zero, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b10000000;
	let Inst{27-24} = cc{3-0};
	let Inst{47-28} = 0b00111100011000000010;
}

def TIMEr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins);
	let AsmString        = "time $rc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rc;
	let Inst{38-0} = 0b110000000100000100000000000000000000000;
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def TIMEz: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero);
	let AsmString        = "time $zero";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let Inst{47-0} = 0b001111000110000000100000100000000000000000000000;
}

def TIME_Srci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins true_cc:$cc, pc16:$pc);
	let AsmString        = "time.s $dc, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b10000000;
	let Inst{27-24} = cc{3-0};
	let Inst{39-28} = 0b111000000010;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def TIME_Sr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins);
	let AsmString        = "time.s $dc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> dc;
	let Inst{39-0} = 0b1110000000100000100000000000000000000000;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def TIME_Urci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins true_cc:$cc, pc16:$pc);
	let AsmString        = "time.u $dc, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<4> cc; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{23-16} = 0b10000000;
	let Inst{27-24} = cc{3-0};
	let Inst{39-28} = 0b011000000010;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def TIME_Ur: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins);
	let AsmString        = "time.u $dc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> dc;
	let Inst{39-0} = 0b0110000000100000100000000000000000000000;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def TIME_CFGrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$rb, true_cc:$cc, pc16:$pc);
	let AsmString        = "time_cfg $rc, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0010;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = 0b11001;
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def TIME_CFGzrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, true_cc:$cc, pc16:$pc);
	let AsmString        = "time_cfg $zero, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0010;
	let Inst{33-32} = rb{4-3};
	let Inst{47-34} = 0b00111100011001;
}

def TIME_CFGrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleReg:$rb);
	let AsmString        = "time_cfg $rc, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b001000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = 0b11001;
	let Inst{41-39} = rc{2-0};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = rc{4-3};
	let Inst{47-46} = 0b00;
}

def TIME_CFGzr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb);
	let AsmString        = "time_cfg $zero, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b001000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{47-34} = 0b00111100011001;
}

def TIME_CFGr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins SimpleReg:$rb);
	let AsmString        = "time_cfg $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b001000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{47-34} = 0b00111100011001;
}

def TIME_CFG_Srrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, true_cc:$cc, pc16:$pc);
	let AsmString        = "time_cfg.s $dc, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0010;
	let Inst{33-32} = rb{4-3};
	let Inst{39-34} = 0b111001;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def TIME_CFG_Srr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb);
	let AsmString        = "time_cfg.s $dc, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b001000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{39-34} = 0b111001;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def TIME_CFG_Urrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb, true_cc:$cc, pc16:$pc);
	let AsmString        = "time_cfg.u $dc, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> dc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b0010;
	let Inst{33-32} = rb{4-3};
	let Inst{39-34} = 0b011001;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def TIME_CFG_Urr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleReg:$rb);
	let AsmString        = "time_cfg.u $dc, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b001000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{39-34} = 0b011001;
	let Inst{41-40} = dc{2-1};
	let Inst{43-42} = 0b11;
	let Inst{45-44} = dc{4-3};
	let Inst{47-46} = 0b01;
}

def XORrri: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, u32_imm:$imm);
	let AsmString        = "xor $rc, $ra, $imm";
	let Pattern          = [(set SimpleReg:$rc, (xor SimpleRegOrCst:$ra, u32_imm:$imm))];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = 0b11;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b0100;
}

def XORzri: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleReg:$rb, u32_imm:$imm);
	let AsmString        = "xor $zero, $rb, $imm";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<32> imm; bits<5> rb;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = imm{3-0};
	let Inst{31-24} = imm{31-24};
	let Inst{33-32} = rb{4-3};
	let Inst{36-34} = imm{6-4};
	let Inst{38-37} = 0b11;
	let Inst{39} = imm{7};
	let Inst{47-40} = 0b01100100;
}

def XORrrici: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "xor $rc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def XORzrici: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s12_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "xor $zero, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<12> imm; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{10-8};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{11};
	let Inst{47-45} = 0b100;
}

def XORrric: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "xor $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def XORzric: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, log_set_cc:$cc);
	let AsmString        = "xor $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<4> cc; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def XORrrif: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "xor $rc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> rc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110000000;
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def XORzrif: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, s28_imm:$imm, false_cc:$cc);
	let AsmString        = "xor $zero, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<28> imm; bits<5> ra;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110000000;
	let Inst{38-34} = ra{4-0};
	let Inst{41-39} = imm{26-24};
	let Inst{43-42} = 0b11;
	let Inst{44} = imm{27};
	let Inst{47-45} = 0b100;
}

def XORrrrci: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "xor $rc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc; bits<5> rc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def XORzrrci: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "xor $zero, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<16> pc; bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def XORrrrc: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "xor $rc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def XORzrrc: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "xor $zero, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def XORrrr: DPUInstruction {
	let OutOperandList   = (outs SimpleReg:$rc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "xor $rc, $ra, $rb";
	let Pattern          = [(set SimpleReg:$rc, (xor SimpleRegOrCst:$ra, SimpleReg:$rb))];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> rc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{43-39} = rc{4-0};
	let Inst{47-44} = 0b1000;
}

def XORzrr: DPUInstruction {
	let OutOperandList   = (outs);
	let InOperandList    = (ins ZeroRegister:$zero, SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "xor $zero, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{47-39} = 0b100011000;
}

def XOR_Srrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "xor.s $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Srric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "xor.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Srrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "xor.s $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Srrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "xor.s $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Srrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "xor.s $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Srrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "xor.s $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b1;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Urrici: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s8_imm:$imm, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "xor.u $dc, $ra, $imm, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> ra; bits<16> pc; bits<8> imm; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Urric: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, log_set_cc:$cc);
	let AsmString        = "xor.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{27-24} = cc{3-0};
	let Inst{33-28} = 0b111000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Urrif: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, s24_imm:$imm, false_cc:$cc);
	let AsmString        = "xor.u $dc, $ra, $imm, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<24> imm; bits<5> ra; bits<5> dc;
	let Inst{11-0} = imm{23-12};
	let Inst{12} = imm{11};
	let Inst{13} = imm{10};
	let Inst{14} = imm{9};
	let Inst{15} = imm{8};
	let Inst{19-16} = imm{7-4};
	let Inst{23-20} = imm{3-0};
	let Inst{33-24} = 0b1110000000;
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Urrrci: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_nz_cc:$cc, pc16:$pc);
	let AsmString        = "xor.u $dc, $ra, $rb, $cc, $pc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	let isBranch         = 1;
	let isCompare        = 1;
	let isTerminator     = 1;
	bits<5> dc; bits<5> rb; bits<5> ra; bits<16> pc; bits<4> cc;
	let Inst{15-0} = pc{15-0};
	let Inst{16} = 0b0;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Urrrc: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb, log_set_cc:$cc);
	let AsmString        = "xor.u $dc, $ra, $rb, $cc";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<4> cc; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{23-20} = 0b1000;
	let Inst{27-24} = cc{3-0};
	let Inst{31-28} = 0b1100;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

def XOR_Urrr: DPUInstruction {
	let OutOperandList   = (outs DoubleReg:$dc);
	let InOperandList    = (ins SimpleRegOrCst:$ra, SimpleReg:$rb);
	let AsmString        = "xor.u $dc, $ra, $rb";
	let Pattern          = [];
	let isAsCheapAsAMove = 1;
	bits<5> rb; bits<5> ra; bits<5> dc;
	let Inst{16-0} = 0b00000000000000000;
	let Inst{19-17} = rb{2-0};
	let Inst{31-20} = 0b110000001000;
	let Inst{33-32} = rb{4-3};
	let Inst{38-34} = ra{4-0};
	let Inst{39} = 0b0;
	let Inst{43-40} = dc{4-1};
	let Inst{47-44} = 0b1001;
}

