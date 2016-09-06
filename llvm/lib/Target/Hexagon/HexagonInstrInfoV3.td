//=- HexagonInstrInfoV3.td - Target Desc. for Hexagon Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V3 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

def callv3 : SDNode<"HexagonISD::CALL", SDT_SPCall,
           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;

def callv3nr : SDNode<"HexagonISD::CALLnr", SDT_SPCall,
           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;

//===----------------------------------------------------------------------===//
// J +
//===----------------------------------------------------------------------===//
// Call subroutine.
let isCall = 1, hasSideEffects = 1, isPredicable = 1,
    isExtended = 0, isExtendable = 1, opExtendable = 0,
    isExtentSigned = 1, opExtentBits = 24, opExtentAlign = 2 in
class T_Call<bit CSR, string ExtStr>
  : JInst<(outs), (ins calltarget:$dst),
      "call " # ExtStr # "$dst", [], "", J_tc_2early_SLOT23> {
  let BaseOpcode = "call";
  bits<24> dst;

  let Defs = !if (CSR, VolatileV3.Regs, []);
  let IClass = 0b0101;
  let Inst{27-25} = 0b101;
  let Inst{24-16,13-1} = dst{23-2};
  let Inst{0} = 0b0;
}

let isCall = 1, hasSideEffects = 1, isPredicated = 1,
    isExtended = 0, isExtendable = 1, opExtendable = 1,
    isExtentSigned = 1, opExtentBits = 17, opExtentAlign = 2 in
class T_CallPred<bit CSR, bit IfTrue, string ExtStr>
  : JInst<(outs), (ins PredRegs:$Pu, calltarget:$dst),
      CondStr<"$Pu", IfTrue, 0>.S # "call " # ExtStr # "$dst",
      [], "", J_tc_2early_SLOT23> {
  let BaseOpcode = "call";
  let isPredicatedFalse = !if(IfTrue,0,1);
  bits<2> Pu;
  bits<17> dst;

  let Defs = !if (CSR, VolatileV3.Regs, []);
  let IClass = 0b0101;
  let Inst{27-24} = 0b1101;
  let Inst{23-22,20-16,13,7-1} = dst{16-2};
  let Inst{21} = !if(IfTrue,0,1);
  let Inst{11} = 0b0;
  let Inst{9-8} = Pu;
}

multiclass T_Calls<bit CSR, string ExtStr> {
  def NAME : T_Call<CSR, ExtStr>;
  def t    : T_CallPred<CSR, 1, ExtStr>;
  def f    : T_CallPred<CSR, 0, ExtStr>;
}

defm J2_call: T_Calls<1, "">, PredRel;

let isCodeGenOnly = 1, isCall = 1, hasSideEffects = 1,
    Defs = VolatileV3.Regs in
def PS_call_nr : T_Call<1, "">, PredRel;

let isCodeGenOnly = 1, isCall = 1, hasSideEffects = 1,
    Defs = [PC, R31, R6, R7, P0] in
def PS_call_stk :  T_Call<0, "">, PredRel;

//===----------------------------------------------------------------------===//
// J -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// JR +
//===----------------------------------------------------------------------===//
// Call subroutine from register.

let isCodeGenOnly = 1, Defs = VolatileV3.Regs in {
  def PS_callr_nr : JUMPR_MISC_CALLR<0, 1>; // Call, no return.
}

//===----------------------------------------------------------------------===//
// JR -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU64/ALU +
//===----------------------------------------------------------------------===//

let Defs = [USR_OVF], Itinerary = ALU64_tc_2_SLOT23 in
def A2_addpsat : T_ALU64_arith<"add", 0b011, 0b101, 1, 0, 1>;

class T_ALU64_addsp_hl<string suffix, bits<3> MinOp>
  : T_ALU64_rr<"add", suffix, 0b0011, 0b011, MinOp, 0, 0, "">;

def A2_addspl : T_ALU64_addsp_hl<":raw:lo", 0b110>;
def A2_addsph : T_ALU64_addsp_hl<":raw:hi", 0b111>;

let hasSideEffects = 0, isAsmParserOnly = 1 in
def A2_addsp : ALU64_rr<(outs DoubleRegs:$Rd),
  (ins IntRegs:$Rs, DoubleRegs:$Rt), "$Rd = add($Rs, $Rt)",
  [(set (i64 DoubleRegs:$Rd), (i64 (add (i64 (sext (i32 IntRegs:$Rs))),
                                        (i64 DoubleRegs:$Rt))))],
  "", ALU64_tc_1_SLOT23>;


let hasSideEffects = 0 in
class T_XTYPE_MIN_MAX_P<bit isMax, bit isUnsigned>
  : ALU64Inst<(outs DoubleRegs:$Rd), (ins DoubleRegs:$Rt, DoubleRegs:$Rs),
  "$Rd = "#!if(isMax,"max","min")#!if(isUnsigned,"u","")
          #"($Rt, $Rs)", [], "", ALU64_tc_2_SLOT23> {
  bits<5> Rd;
  bits<5> Rs;
  bits<5> Rt;

  let IClass = 0b1101;

  let Inst{27-23} = 0b00111;
  let Inst{22-21} = !if(isMax, 0b10, 0b01);
  let Inst{20-16} = !if(isMax, Rt, Rs);
  let Inst{12-8} = !if(isMax, Rs, Rt);
  let Inst{7} = 0b1;
  let Inst{6} = !if(isMax, 0b0, 0b1);
  let Inst{5} = isUnsigned;
  let Inst{4-0} = Rd;
}

def A2_minp  : T_XTYPE_MIN_MAX_P<0, 0>;
def A2_minup : T_XTYPE_MIN_MAX_P<0, 1>;
def A2_maxp  : T_XTYPE_MIN_MAX_P<1, 0>;
def A2_maxup : T_XTYPE_MIN_MAX_P<1, 1>;

multiclass MinMax_pats_p<PatFrag Op, InstHexagon Inst, InstHexagon SwapInst> {
  defm: T_MinMax_pats<Op, DoubleRegs, i64, Inst, SwapInst>;
}

let AddedComplexity = 200 in {
  defm: MinMax_pats_p<setge,  A2_maxp,  A2_minp>;
  defm: MinMax_pats_p<setgt,  A2_maxp,  A2_minp>;
  defm: MinMax_pats_p<setle,  A2_minp,  A2_maxp>;
  defm: MinMax_pats_p<setlt,  A2_minp,  A2_maxp>;
  defm: MinMax_pats_p<setuge, A2_maxup, A2_minup>;
  defm: MinMax_pats_p<setugt, A2_maxup, A2_minup>;
  defm: MinMax_pats_p<setule, A2_minup, A2_maxup>;
  defm: MinMax_pats_p<setult, A2_minup, A2_maxup>;
}

//===----------------------------------------------------------------------===//
// ALU64/ALU -
//===----------------------------------------------------------------------===//




//def : Pat <(brcond (i1 (seteq (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegEzt (i32 IntRegs:$src1), bb:$offset)>;

//def : Pat <(brcond (i1 (setne (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegNzt (i32 IntRegs:$src1), bb:$offset)>;

//def : Pat <(brcond (i1 (setle (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegLezt (i32 IntRegs:$src1), bb:$offset)>;

//def : Pat <(brcond (i1 (setge (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegGezt (i32 IntRegs:$src1), bb:$offset)>;

//def : Pat <(brcond (i1 (setgt (i32 IntRegs:$src1), -1)), bb:$offset),
//      (JMP_RegGezt (i32 IntRegs:$src1), bb:$offset)>;

// Map call instruction
def : Pat<(callv3 I32:$dst),
          (J2_callr I32:$dst)>;
def : Pat<(callv3 tglobaladdr:$dst),
          (J2_call tglobaladdr:$dst)>;
def : Pat<(callv3 texternalsym:$dst),
          (J2_call texternalsym:$dst)>;
def : Pat<(callv3 tglobaltlsaddr:$dst),
          (J2_call tglobaltlsaddr:$dst)>;

def : Pat<(callv3nr I32:$dst),
          (PS_callr_nr I32:$dst)>;
def : Pat<(callv3nr tglobaladdr:$dst),
          (PS_call_nr tglobaladdr:$dst)>;
def : Pat<(callv3nr texternalsym:$dst),
          (PS_call_nr texternalsym:$dst)>;

//===----------------------------------------------------------------------===//
// :raw form of vrcmpys:hi/lo insns
//===----------------------------------------------------------------------===//
// Vector reduce complex multiply by scalar.
let Defs = [USR_OVF], hasSideEffects = 0 in
class T_vrcmpRaw<string HiLo, bits<3>MajOp>:
  MInst<(outs DoubleRegs:$Rdd),
         (ins DoubleRegs:$Rss, DoubleRegs:$Rtt),
         "$Rdd = vrcmpys($Rss, $Rtt):<<1:sat:raw:"#HiLo, []> {
    bits<5> Rdd;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1000;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = Rss;
    let Inst{12-8}  = Rtt;
    let Inst{7-5}   = 0b100;
    let Inst{4-0}   = Rdd;
}

def M2_vrcmpys_s1_h: T_vrcmpRaw<"hi", 0b101>;
def M2_vrcmpys_s1_l: T_vrcmpRaw<"lo", 0b111>;

// Assembler mapped to M2_vrcmpys_s1_h or M2_vrcmpys_s1_l
let hasSideEffects = 0, isAsmParserOnly = 1 in
def M2_vrcmpys_s1
 : MInst<(outs DoubleRegs:$Rdd), (ins DoubleRegs:$Rss, IntRegs:$Rt),
 "$Rdd=vrcmpys($Rss,$Rt):<<1:sat">;

// Vector reduce complex multiply by scalar with accumulation.
let Defs = [USR_OVF], hasSideEffects = 0 in
class T_vrcmpys_acc<string HiLo, bits<3>MajOp>:
  MInst <(outs DoubleRegs:$Rxx),
         (ins DoubleRegs:$_src_, DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Rxx += vrcmpys($Rss, $Rtt):<<1:sat:raw:"#HiLo, [],
  "$Rxx = $_src_"> {
    bits<5> Rxx;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1010;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = Rss;
    let Inst{12-8}  = Rtt;
    let Inst{7-5}   = 0b100;
    let Inst{4-0}   = Rxx;
  }

def M2_vrcmpys_acc_s1_h: T_vrcmpys_acc<"hi", 0b101>;
def M2_vrcmpys_acc_s1_l: T_vrcmpys_acc<"lo", 0b111>;

// Assembler mapped to M2_vrcmpys_acc_s1_h or M2_vrcmpys_acc_s1_l

let isAsmParserOnly = 1 in
def M2_vrcmpys_acc_s1
  : MInst <(outs DoubleRegs:$dst),
           (ins DoubleRegs:$dst2, DoubleRegs:$src1, IntRegs:$src2),
           "$dst += vrcmpys($src1, $src2):<<1:sat", [],
           "$dst2 = $dst">;

def M2_vrcmpys_s1rp_h : T_MType_vrcmpy <"vrcmpys", 0b101, 0b110, 1>;
def M2_vrcmpys_s1rp_l : T_MType_vrcmpy <"vrcmpys", 0b101, 0b111, 0>;

// Assembler mapped to M2_vrcmpys_s1rp_h or M2_vrcmpys_s1rp_l
let isAsmParserOnly = 1 in
def M2_vrcmpys_s1rp
  : MInst <(outs IntRegs:$Rd), (ins DoubleRegs:$Rss, IntRegs:$Rt),
  "$Rd=vrcmpys($Rss,$Rt):<<1:rnd:sat">;


// S2_cabacdecbin: Cabac decode bin.
let Defs = [P0], isPredicateLate = 1, Itinerary = S_3op_tc_1_SLOT23 in
def S2_cabacdecbin : T_S3op_64 < "decbin", 0b11, 0b110, 0>;
