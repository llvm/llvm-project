//==- HexagonInstrInfo.td - Target Description for Hexagon -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

include "HexagonInstrFormats.td"
include "HexagonOperands.td"
include "HexagonInstrEnc.td"
// Pattern fragment that combines the value type and the register class
// into a single parameter.
// The pat frags in the definitions below need to have a named register,
// otherwise i32 will be assumed regardless of the register class. The
// name of the register does not matter.
def I1  : PatLeaf<(i1 PredRegs:$R)>;
def I32 : PatLeaf<(i32 IntRegs:$R)>;
def I64 : PatLeaf<(i64 DoubleRegs:$R)>;
def F32 : PatLeaf<(f32 IntRegs:$R)>;
def F64 : PatLeaf<(f64 DoubleRegs:$R)>;

// Pattern fragments to extract the low and high subregisters from a
// 64-bit value.
def LoReg: OutPatFrag<(ops node:$Rs),
                      (EXTRACT_SUBREG (i64 $Rs), subreg_loreg)>;
def HiReg: OutPatFrag<(ops node:$Rs),
                      (EXTRACT_SUBREG (i64 $Rs), subreg_hireg)>;

def orisadd: PatFrag<(ops node:$Addr, node:$off),
    (or node:$Addr, node:$off), [{ return orIsAdd(N); }]>;

// SDNode for converting immediate C to C-1.
def DEC_CONST_SIGNED : SDNodeXForm<imm, [{
   // Return the byte immediate const-1 as an SDNode.
   int32_t imm = N->getSExtValue();
   return XformSToSM1Imm(imm, SDLoc(N));
}]>;

// SDNode for converting immediate C to C-2.
def DEC2_CONST_SIGNED : SDNodeXForm<imm, [{
   // Return the byte immediate const-2 as an SDNode.
   int32_t imm = N->getSExtValue();
   return XformSToSM2Imm(imm, SDLoc(N));
}]>;

// SDNode for converting immediate C to C-3.
def DEC3_CONST_SIGNED : SDNodeXForm<imm, [{
   // Return the byte immediate const-3 as an SDNode.
   int32_t imm = N->getSExtValue();
   return XformSToSM3Imm(imm, SDLoc(N));
}]>;

// SDNode for converting immediate C to C-1.
def DEC_CONST_UNSIGNED : SDNodeXForm<imm, [{
   // Return the byte immediate const-1 as an SDNode.
   uint32_t imm = N->getZExtValue();
   return XformUToUM1Imm(imm, SDLoc(N));
}]>;

//===----------------------------------------------------------------------===//
// Compare
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, isCompare = 1, InputType = "imm", isExtendable = 1,
    opExtendable = 2 in
class T_CMP <string mnemonic, bits<2> MajOp, bit isNot, Operand ImmOp>
  : ALU32Inst <(outs PredRegs:$dst),
               (ins IntRegs:$src1, ImmOp:$src2),
  "$dst = "#!if(isNot, "!","")#mnemonic#"($src1, #$src2)",
  [], "",ALU32_2op_tc_2early_SLOT0123 >, ImmRegRel {
    bits<2> dst;
    bits<5> src1;
    bits<10> src2;
    let CextOpcode = mnemonic;
    let opExtentBits  = !if(!eq(mnemonic, "cmp.gtu"), 9, 10);
    let isExtentSigned = !if(!eq(mnemonic, "cmp.gtu"), 0, 1);

    let IClass = 0b0111;

    let Inst{27-24} = 0b0101;
    let Inst{23-22} = MajOp;
    let Inst{21}    = !if(!eq(mnemonic, "cmp.gtu"), 0, src2{9});
    let Inst{20-16} = src1;
    let Inst{13-5}  = src2{8-0};
    let Inst{4}     = isNot;
    let Inst{3-2}   = 0b00;
    let Inst{1-0}   = dst;
  }

def C2_cmpeqi   : T_CMP <"cmp.eq",  0b00, 0, s10Ext>;
def C2_cmpgti   : T_CMP <"cmp.gt",  0b01, 0, s10Ext>;
def C2_cmpgtui  : T_CMP <"cmp.gtu", 0b10, 0, u9Ext>;

class T_CMP_pat <InstHexagon MI, PatFrag OpNode, PatLeaf ImmPred>
  : Pat<(i1 (OpNode (i32 IntRegs:$src1), ImmPred:$src2)),
        (MI IntRegs:$src1, ImmPred:$src2)>;

def : T_CMP_pat <C2_cmpeqi,  seteq,  s10ImmPred>;
def : T_CMP_pat <C2_cmpgti,  setgt,  s10ImmPred>;
def : T_CMP_pat <C2_cmpgtui, setugt, u9ImmPred>;

//===----------------------------------------------------------------------===//
// ALU32/ALU +
//===----------------------------------------------------------------------===//
// Add.

def SDT_Int32Leaf  : SDTypeProfile<1, 0, [SDTCisVT<0, i32>]>;
def SDT_Int32Unary : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;

def SDTHexagonI64I32I32 : SDTypeProfile<1, 2,
  [SDTCisVT<0, i64>, SDTCisVT<1, i32>, SDTCisSameAs<1, 2>]>;

def HexagonCOMBINE : SDNode<"HexagonISD::COMBINE", SDTHexagonI64I32I32>;
def HexagonPACKHL  : SDNode<"HexagonISD::PACKHL",  SDTHexagonI64I32I32>;

let hasSideEffects = 0, hasNewValue = 1, InputType = "reg" in
class T_ALU32_3op<string mnemonic, bits<3> MajOp, bits<3> MinOp, bit OpsRev,
                  bit IsComm>
  : ALU32_rr<(outs IntRegs:$Rd), (ins IntRegs:$Rs, IntRegs:$Rt),
             "$Rd = "#mnemonic#"($Rs, $Rt)",
             [], "", ALU32_3op_tc_1_SLOT0123>, ImmRegRel, PredRel {
  let isCommutable = IsComm;
  let BaseOpcode = mnemonic#_rr;
  let CextOpcode = mnemonic;

  bits<5> Rs;
  bits<5> Rt;
  bits<5> Rd;

  let IClass = 0b1111;
  let Inst{27} = 0b0;
  let Inst{26-24} = MajOp;
  let Inst{23-21} = MinOp;
  let Inst{20-16} = !if(OpsRev,Rt,Rs);
  let Inst{12-8} = !if(OpsRev,Rs,Rt);
  let Inst{4-0} = Rd;
}

let hasSideEffects = 0, hasNewValue = 1 in
class T_ALU32_3op_pred<string mnemonic, bits<3> MajOp, bits<3> MinOp,
                       bit OpsRev, bit PredNot, bit PredNew>
  : ALU32_rr<(outs IntRegs:$Rd), (ins PredRegs:$Pu, IntRegs:$Rs, IntRegs:$Rt),
             "if ("#!if(PredNot,"!","")#"$Pu"#!if(PredNew,".new","")#") "#
             "$Rd = "#mnemonic#"($Rs, $Rt)",
             [], "", ALU32_3op_tc_1_SLOT0123>, ImmRegRel, PredNewRel {
  let isPredicated = 1;
  let isPredicatedFalse = PredNot;
  let isPredicatedNew = PredNew;
  let BaseOpcode = mnemonic#_rr;
  let CextOpcode = mnemonic;

  bits<2> Pu;
  bits<5> Rs;
  bits<5> Rt;
  bits<5> Rd;

  let IClass = 0b1111;
  let Inst{27} = 0b1;
  let Inst{26-24} = MajOp;
  let Inst{23-21} = MinOp;
  let Inst{20-16} = !if(OpsRev,Rt,Rs);
  let Inst{13} = PredNew;
  let Inst{12-8} = !if(OpsRev,Rs,Rt);
  let Inst{7} = PredNot;
  let Inst{6-5} = Pu;
  let Inst{4-0} = Rd;
}

class T_ALU32_combineh<string Op1, string Op2, bits<3> MajOp, bits<3> MinOp,
                      bit OpsRev>
  : T_ALU32_3op<"", MajOp, MinOp, OpsRev, 0> {
  let AsmString = "$Rd = combine($Rs"#Op1#", $Rt"#Op2#")";
}

def A2_combine_hh : T_ALU32_combineh<".h", ".h", 0b011, 0b100, 1>;
def A2_combine_hl : T_ALU32_combineh<".h", ".l", 0b011, 0b101, 1>;
def A2_combine_lh : T_ALU32_combineh<".l", ".h", 0b011, 0b110, 1>;
def A2_combine_ll : T_ALU32_combineh<".l", ".l", 0b011, 0b111, 1>;

class T_ALU32_3op_sfx<string mnemonic, string suffix, bits<3> MajOp,
                      bits<3> MinOp, bit OpsRev, bit IsComm>
  : T_ALU32_3op<"", MajOp, MinOp, OpsRev, IsComm> {
  let AsmString = "$Rd = "#mnemonic#"($Rs, $Rt)"#suffix;
}

def A2_svaddh   : T_ALU32_3op<"vaddh",   0b110, 0b000, 0, 1>;
def A2_svsubh   : T_ALU32_3op<"vsubh",   0b110, 0b100, 1, 0>;

let Defs = [USR_OVF], Itinerary = ALU32_3op_tc_2_SLOT0123 in {
  def A2_svaddhs  : T_ALU32_3op_sfx<"vaddh",  ":sat", 0b110, 0b001, 0, 1>;
  def A2_addsat   : T_ALU32_3op_sfx<"add",    ":sat", 0b110, 0b010, 0, 1>;
  def A2_svadduhs : T_ALU32_3op_sfx<"vadduh", ":sat", 0b110, 0b011, 0, 1>;
  def A2_svsubhs  : T_ALU32_3op_sfx<"vsubh",  ":sat", 0b110, 0b101, 1, 0>;
  def A2_subsat   : T_ALU32_3op_sfx<"sub",    ":sat", 0b110, 0b110, 1, 0>;
  def A2_svsubuhs : T_ALU32_3op_sfx<"vsubuh", ":sat", 0b110, 0b111, 1, 0>;
}

let Itinerary = ALU32_3op_tc_2_SLOT0123 in
def A2_svavghs  : T_ALU32_3op_sfx<"vavgh",  ":rnd", 0b111, 0b001, 0, 1>;

def A2_svavgh   : T_ALU32_3op<"vavgh",   0b111, 0b000, 0, 1>;
def A2_svnavgh  : T_ALU32_3op<"vnavgh",  0b111, 0b011, 1, 0>;

multiclass T_ALU32_3op_p<string mnemonic, bits<3> MajOp, bits<3> MinOp,
                         bit OpsRev> {
  def t    : T_ALU32_3op_pred<mnemonic, MajOp, MinOp, OpsRev, 0, 0>;
  def f    : T_ALU32_3op_pred<mnemonic, MajOp, MinOp, OpsRev, 1, 0>;
  def tnew : T_ALU32_3op_pred<mnemonic, MajOp, MinOp, OpsRev, 0, 1>;
  def fnew : T_ALU32_3op_pred<mnemonic, MajOp, MinOp, OpsRev, 1, 1>;
}

multiclass T_ALU32_3op_A2<string mnemonic, bits<3> MajOp, bits<3> MinOp,
                          bit OpsRev, bit IsComm> {
  let isPredicable = 1 in
  def  A2_#NAME  : T_ALU32_3op  <mnemonic, MajOp, MinOp, OpsRev, IsComm>;
  defm A2_p#NAME : T_ALU32_3op_p<mnemonic, MajOp, MinOp, OpsRev>;
}

defm add : T_ALU32_3op_A2<"add", 0b011, 0b000, 0, 1>;
defm and : T_ALU32_3op_A2<"and", 0b001, 0b000, 0, 1>;
defm or  : T_ALU32_3op_A2<"or",  0b001, 0b001, 0, 1>;
defm sub : T_ALU32_3op_A2<"sub", 0b011, 0b001, 1, 0>;
defm xor : T_ALU32_3op_A2<"xor", 0b001, 0b011, 0, 1>;

// Pats for instruction selection.
class BinOp32_pat<SDNode Op, InstHexagon MI, ValueType ResT>
  : Pat<(ResT (Op (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))),
        (ResT (MI IntRegs:$Rs, IntRegs:$Rt))>;

def: BinOp32_pat<add, A2_add, i32>;
def: BinOp32_pat<and, A2_and, i32>;
def: BinOp32_pat<or,  A2_or,  i32>;
def: BinOp32_pat<sub, A2_sub, i32>;
def: BinOp32_pat<xor, A2_xor, i32>;

// A few special cases producing register pairs:
let OutOperandList = (outs DoubleRegs:$Rd), hasNewValue = 0 in {
  def S2_packhl    : T_ALU32_3op  <"packhl",  0b101, 0b100, 0, 0>;

  let isPredicable = 1 in
    def A2_combinew  : T_ALU32_3op  <"combine", 0b101, 0b000, 0, 0>;

  // Conditional combinew uses "newt/f" instead of "t/fnew".
  def C2_ccombinewt    : T_ALU32_3op_pred<"combine", 0b101, 0b000, 0, 0, 0>;
  def C2_ccombinewf    : T_ALU32_3op_pred<"combine", 0b101, 0b000, 0, 1, 0>;
  def C2_ccombinewnewt : T_ALU32_3op_pred<"combine", 0b101, 0b000, 0, 0, 1>;
  def C2_ccombinewnewf : T_ALU32_3op_pred<"combine", 0b101, 0b000, 0, 1, 1>;
}

def: BinOp32_pat<HexagonCOMBINE, A2_combinew, i64>;
def: BinOp32_pat<HexagonPACKHL,  S2_packhl,   i64>;

let hasSideEffects = 0, hasNewValue = 1, isCompare = 1, InputType = "reg"  in
class T_ALU32_3op_cmp<string mnemonic, bits<2> MinOp, bit IsNeg, bit IsComm>
  : ALU32_rr<(outs PredRegs:$Pd), (ins IntRegs:$Rs, IntRegs:$Rt),
             "$Pd = "#mnemonic#"($Rs, $Rt)",
             [], "", ALU32_3op_tc_1_SLOT0123>, ImmRegRel {
  let CextOpcode = mnemonic;
  let isCommutable = IsComm;
  bits<5> Rs;
  bits<5> Rt;
  bits<2> Pd;

  let IClass = 0b1111;
  let Inst{27-24} = 0b0010;
  let Inst{22-21} = MinOp;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{4} = IsNeg;
  let Inst{3-2} = 0b00;
  let Inst{1-0} = Pd;
}

let Itinerary = ALU32_3op_tc_2early_SLOT0123 in {
  def C2_cmpeq   : T_ALU32_3op_cmp< "cmp.eq",  0b00, 0, 1>;
  def C2_cmpgt   : T_ALU32_3op_cmp< "cmp.gt",  0b10, 0, 0>;
  def C2_cmpgtu  : T_ALU32_3op_cmp< "cmp.gtu", 0b11, 0, 0>;
}

// Patfrag to convert the usual comparison patfrags (e.g. setlt) to ones
// that reverse the order of the operands.
class RevCmp<PatFrag F> : PatFrag<(ops node:$rhs, node:$lhs), F.Fragment>;

// Pats for compares. They use PatFrags as operands, not SDNodes,
// since seteq/setgt/etc. are defined as ParFrags.
class T_cmp32_rr_pat<InstHexagon MI, PatFrag Op, ValueType VT>
  : Pat<(VT (Op (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))),
        (VT (MI IntRegs:$Rs, IntRegs:$Rt))>;

def: T_cmp32_rr_pat<C2_cmpeq,  seteq, i1>;
def: T_cmp32_rr_pat<C2_cmpgt,  setgt, i1>;
def: T_cmp32_rr_pat<C2_cmpgtu, setugt, i1>;

def: T_cmp32_rr_pat<C2_cmpgt,  RevCmp<setlt>,  i1>;
def: T_cmp32_rr_pat<C2_cmpgtu, RevCmp<setult>, i1>;

let CextOpcode = "MUX", InputType = "reg", hasNewValue = 1 in
def C2_mux: ALU32_rr<(outs IntRegs:$Rd),
                     (ins PredRegs:$Pu, IntRegs:$Rs, IntRegs:$Rt),
      "$Rd = mux($Pu, $Rs, $Rt)", [], "", ALU32_3op_tc_1_SLOT0123>, ImmRegRel {
  bits<5> Rd;
  bits<2> Pu;
  bits<5> Rs;
  bits<5> Rt;

  let CextOpcode = "mux";
  let InputType = "reg";
  let hasSideEffects = 0;
  let IClass = 0b1111;

  let Inst{27-24} = 0b0100;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{6-5} = Pu;
  let Inst{4-0} = Rd;
}

def: Pat<(i32 (select (i1 PredRegs:$Pu), (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))),
         (C2_mux PredRegs:$Pu, IntRegs:$Rs, IntRegs:$Rt)>;

// Combines the two immediates into a double register.
// Increase complexity to make it greater than any complexity of a combine
// that involves a register.

let isReMaterializable = 1, isMoveImm = 1, isAsCheapAsAMove = 1,
    isExtentSigned = 1, isExtendable = 1, opExtentBits = 8, opExtendable = 1,
    AddedComplexity = 75 in
def A2_combineii: ALU32Inst <(outs DoubleRegs:$Rdd), (ins s8Ext:$s8, s8Imm:$S8),
  "$Rdd = combine(#$s8, #$S8)",
  [(set (i64 DoubleRegs:$Rdd),
        (i64 (HexagonCOMBINE(i32 s32ImmPred:$s8), (i32 s8ImmPred:$S8))))]> {
    bits<5> Rdd;
    bits<8> s8;
    bits<8> S8;

    let IClass = 0b0111;
    let Inst{27-23} = 0b11000;
    let Inst{22-16} = S8{7-1};
    let Inst{13}    = S8{0};
    let Inst{12-5}  = s8;
    let Inst{4-0}   = Rdd;
  }

//===----------------------------------------------------------------------===//
// Template class for predicated ADD of a reg and an Immediate value.
//===----------------------------------------------------------------------===//
let hasNewValue = 1, hasSideEffects = 0 in
class T_Addri_Pred <bit PredNot, bit PredNew>
  : ALU32_ri <(outs IntRegs:$Rd),
              (ins PredRegs:$Pu, IntRegs:$Rs, s8Ext:$s8),
  !if(PredNot, "if (!$Pu", "if ($Pu")#!if(PredNew,".new) $Rd = ",
  ") $Rd = ")#"add($Rs, #$s8)"> {
    bits<5> Rd;
    bits<2> Pu;
    bits<5> Rs;
    bits<8> s8;

    let isPredicatedNew = PredNew;
    let IClass = 0b0111;

    let Inst{27-24} = 0b0100;
    let Inst{23}    = PredNot;
    let Inst{22-21} = Pu;
    let Inst{20-16} = Rs;
    let Inst{13}    = PredNew;
    let Inst{12-5}  = s8;
    let Inst{4-0}   = Rd;
  }

//===----------------------------------------------------------------------===//
// A2_addi: Add a signed immediate to a register.
//===----------------------------------------------------------------------===//
let hasNewValue = 1, hasSideEffects = 0 in
class T_Addri <Operand immOp>
  : ALU32_ri <(outs IntRegs:$Rd),
              (ins IntRegs:$Rs, immOp:$s16),
  "$Rd = add($Rs, #$s16)", [], "", ALU32_ADDI_tc_1_SLOT0123> {
    bits<5> Rd;
    bits<5> Rs;
    bits<16> s16;

    let IClass = 0b1011;

    let Inst{27-21} = s16{15-9};
    let Inst{20-16} = Rs;
    let Inst{13-5}  = s16{8-0};
    let Inst{4-0}   = Rd;
  }

//===----------------------------------------------------------------------===//
// Multiclass for ADD of a register and an immediate value.
//===----------------------------------------------------------------------===//
multiclass Addri_Pred<string mnemonic, bit PredNot> {
  let isPredicatedFalse = PredNot in {
    def NAME     : T_Addri_Pred<PredNot, 0>;
    // Predicate new
    def NAME#new : T_Addri_Pred<PredNot, 1>;
  }
}

let isExtendable = 1, isExtentSigned = 1, InputType = "imm" in
multiclass Addri_base<string mnemonic, SDNode OpNode> {
  let CextOpcode = mnemonic, BaseOpcode = mnemonic#_ri in {
    let opExtendable = 2, opExtentBits = 16, isPredicable = 1 in
    def A2_#NAME : T_Addri<s16Ext>;

    let opExtendable = 3, opExtentBits = 8, isPredicated = 1 in {
      defm A2_p#NAME#t : Addri_Pred<mnemonic, 0>;
      defm A2_p#NAME#f : Addri_Pred<mnemonic, 1>;
    }
  }
}

defm addi : Addri_base<"add", add>, ImmRegRel, PredNewRel;

def: Pat<(i32 (add I32:$Rs, s32ImmPred:$s16)),
         (i32 (A2_addi I32:$Rs, imm:$s16))>;

let hasNewValue = 1, hasSideEffects = 0, isPseudo = 1 in
def A2_iconst
  : ALU32_ri <(outs IntRegs:$Rd),
              (ins s23_2Imm:$s23_2),
  "$Rd = iconst(#$s23_2)"> {}

//===----------------------------------------------------------------------===//
// Template class used for the following ALU32 instructions.
// Rd=and(Rs,#s10)
// Rd=or(Rs,#s10)
//===----------------------------------------------------------------------===//
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 10,
InputType = "imm", hasNewValue = 1 in
class T_ALU32ri_logical <string mnemonic, SDNode OpNode, bits<2> MinOp>
  : ALU32_ri <(outs IntRegs:$Rd),
              (ins IntRegs:$Rs, s10Ext:$s10),
  "$Rd = "#mnemonic#"($Rs, #$s10)" ,
  [(set (i32 IntRegs:$Rd), (OpNode (i32 IntRegs:$Rs), s32ImmPred:$s10))]> {
    bits<5> Rd;
    bits<5> Rs;
    bits<10> s10;
    let CextOpcode = mnemonic;

    let IClass = 0b0111;

    let Inst{27-24} = 0b0110;
    let Inst{23-22} = MinOp;
    let Inst{21}    = s10{9};
    let Inst{20-16} = Rs;
    let Inst{13-5}  = s10{8-0};
    let Inst{4-0}   = Rd;
  }

def A2_orir  : T_ALU32ri_logical<"or", or, 0b10>, ImmRegRel;
def A2_andir : T_ALU32ri_logical<"and", and, 0b00>, ImmRegRel;

// Subtract register from immediate
// Rd32=sub(#s10,Rs32)
let isExtendable = 1, CextOpcode = "sub", opExtendable = 1, isExtentSigned = 1,
    opExtentBits = 10, InputType = "imm", hasNewValue = 1, hasSideEffects = 0 in
def A2_subri: ALU32_ri <(outs IntRegs:$Rd), (ins s10Ext:$s10, IntRegs:$Rs),
  "$Rd = sub(#$s10, $Rs)", []>, ImmRegRel {
    bits<5> Rd;
    bits<10> s10;
    bits<5> Rs;

    let IClass = 0b0111;

    let Inst{27-22} = 0b011001;
    let Inst{21}    = s10{9};
    let Inst{20-16} = Rs;
    let Inst{13-5}  = s10{8-0};
    let Inst{4-0}   = Rd;
  }

// Nop.
let hasSideEffects = 0 in
def A2_nop: ALU32Inst <(outs), (ins), "nop" > {
  let IClass = 0b0111;
  let Inst{27-24} = 0b1111;
}

def: Pat<(sub s32ImmPred:$s10, IntRegs:$Rs),
         (A2_subri imm:$s10, IntRegs:$Rs)>;

// Rd = not(Rs) gets mapped to Rd=sub(#-1, Rs).
def: Pat<(not (i32 IntRegs:$src1)),
         (A2_subri -1, IntRegs:$src1)>;

let hasSideEffects = 0, hasNewValue = 1 in
class T_tfr16<bit isHi>
  : ALU32Inst <(outs IntRegs:$Rx), (ins IntRegs:$src1, u16Imm:$u16),
  "$Rx"#!if(isHi, ".h", ".l")#" = #$u16",
  [], "$src1 = $Rx" > {
    bits<5> Rx;
    bits<16> u16;

    let IClass = 0b0111;
    let Inst{27-26} = 0b00;
    let Inst{25-24} = !if(isHi, 0b10, 0b01);
    let Inst{23-22} = u16{15-14};
    let Inst{21}    = 0b1;
    let Inst{20-16} = Rx;
    let Inst{13-0}  = u16{13-0};
  }

def A2_tfril: T_tfr16<0>;
def A2_tfrih: T_tfr16<1>;

// Conditional transfer is an alias to conditional "Rd = add(Rs, #0)".
let isPredicated = 1, hasNewValue = 1, opNewValue = 0 in
class T_tfr_pred<bit isPredNot, bit isPredNew>
  : ALU32Inst<(outs IntRegs:$dst),
              (ins PredRegs:$src1, IntRegs:$src2),
              "if ("#!if(isPredNot, "!", "")#
              "$src1"#!if(isPredNew, ".new", "")#
              ") $dst = $src2"> {
    bits<5> dst;
    bits<2> src1;
    bits<5> src2;

    let isPredicatedFalse = isPredNot;
    let isPredicatedNew = isPredNew;
    let IClass = 0b0111;

    let Inst{27-24} = 0b0100;
    let Inst{23} = isPredNot;
    let Inst{13} = isPredNew;
    let Inst{12-5} = 0;
    let Inst{4-0} = dst;
    let Inst{22-21} = src1;
    let Inst{20-16} = src2;
  }

let isPredicable = 1 in
class T_tfr : ALU32Inst<(outs IntRegs:$dst), (ins IntRegs:$src),
              "$dst = $src"> {
    bits<5> dst;
    bits<5> src;

    let IClass = 0b0111;

    let Inst{27-21} = 0b0000011;
    let Inst{20-16} = src;
    let Inst{13}    = 0b0;
    let Inst{4-0}   = dst;
  }

let InputType = "reg", hasNewValue = 1, hasSideEffects = 0 in
multiclass tfr_base<string CextOp> {
  let CextOpcode = CextOp, BaseOpcode = CextOp in {
    def NAME : T_tfr;

    // Predicate
    def t : T_tfr_pred<0, 0>;
    def f : T_tfr_pred<1, 0>;
    // Predicate new
    def tnew : T_tfr_pred<0, 1>;
    def fnew : T_tfr_pred<1, 1>;
  }
}

// Assembler mapped to C2_ccombinew[t|f|newt|newf].
// Please don't add bits to this instruction as it'll be converted into
// 'combine' before object code emission.
let isPredicated = 1 in
class T_tfrp_pred<bit PredNot, bit PredNew>
  : ALU32_rr <(outs DoubleRegs:$dst),
              (ins PredRegs:$src1, DoubleRegs:$src2),
  "if ("#!if(PredNot, "!", "")#"$src1"
        #!if(PredNew, ".new", "")#") $dst = $src2" > {
    let isPredicatedFalse = PredNot;
    let isPredicatedNew = PredNew;
  }

// Assembler mapped to A2_combinew.
// Please don't add bits to this instruction as it'll be converted into
// 'combine' before object code emission.
class T_tfrp : ALU32Inst <(outs DoubleRegs:$dst),
               (ins DoubleRegs:$src),
    "$dst = $src">;

let hasSideEffects = 0 in
multiclass TFR64_base<string BaseName> {
  let BaseOpcode = BaseName in {
    let isPredicable = 1 in
    def NAME : T_tfrp;
    // Predicate
    def t : T_tfrp_pred <0, 0>;
    def f : T_tfrp_pred <1, 0>;
    // Predicate new
    def tnew : T_tfrp_pred <0, 1>;
    def fnew : T_tfrp_pred <1, 1>;
  }
}

let InputType = "imm", isExtendable = 1, isExtentSigned = 1, opExtentBits = 12,
    isMoveImm = 1, opExtendable = 2, BaseOpcode = "TFRI", CextOpcode = "TFR",
    hasSideEffects = 0, isPredicated = 1, hasNewValue = 1 in
class T_TFRI_Pred<bit PredNot, bit PredNew>
  : ALU32_ri<(outs IntRegs:$Rd), (ins PredRegs:$Pu, s12Ext:$s12),
    "if ("#!if(PredNot,"!","")#"$Pu"#!if(PredNew,".new","")#") $Rd = #$s12",
    [], "", ALU32_2op_tc_1_SLOT0123>, ImmRegRel, PredNewRel {
  let isPredicatedFalse = PredNot;
  let isPredicatedNew = PredNew;

  bits<5> Rd;
  bits<2> Pu;
  bits<12> s12;

  let IClass = 0b0111;
  let Inst{27-24} = 0b1110;
  let Inst{23} = PredNot;
  let Inst{22-21} = Pu;
  let Inst{20} = 0b0;
  let Inst{19-16,12-5} = s12;
  let Inst{13} = PredNew;
  let Inst{4-0} = Rd;
}

def C2_cmoveit    : T_TFRI_Pred<0, 0>;
def C2_cmoveif    : T_TFRI_Pred<1, 0>;
def C2_cmovenewit : T_TFRI_Pred<0, 1>;
def C2_cmovenewif : T_TFRI_Pred<1, 1>;

let InputType = "imm", isExtendable = 1, isExtentSigned = 1,
    CextOpcode = "TFR", BaseOpcode = "TFRI", hasNewValue = 1, opNewValue = 0,
    isAsCheapAsAMove = 1 , opExtendable = 1, opExtentBits = 16, isMoveImm = 1,
    isPredicated = 0, isPredicable = 1, isReMaterializable = 1 in
def A2_tfrsi : ALU32Inst<(outs IntRegs:$Rd), (ins s16Ext:$s16), "$Rd = #$s16",
    [(set (i32 IntRegs:$Rd), s32ImmPred:$s16)], "", ALU32_2op_tc_1_SLOT0123>,
    ImmRegRel, PredRel {
  bits<5> Rd;
  bits<16> s16;

  let IClass = 0b0111;
  let Inst{27-24} = 0b1000;
  let Inst{23-22,20-16,13-5} = s16;
  let Inst{4-0} = Rd;
}

defm A2_tfr  : tfr_base<"TFR">, ImmRegRel, PredNewRel;
let isAsmParserOnly = 1 in
defm A2_tfrp : TFR64_base<"TFR64">, PredNewRel;

// Assembler mapped
let isReMaterializable = 1, isMoveImm = 1, isAsCheapAsAMove = 1,
    isAsmParserOnly = 1 in
def A2_tfrpi : ALU64_rr<(outs DoubleRegs:$dst), (ins s8Imm64:$src1),
                      "$dst = #$src1",
                      [(set (i64 DoubleRegs:$dst), s8Imm64Pred:$src1)]>;

// TODO: see if this instruction can be deleted..
let isExtendable = 1, opExtendable = 1, opExtentBits = 6,
    isAsmParserOnly = 1 in {
def TFRI64_V4 : ALU64_rr<(outs DoubleRegs:$dst), (ins u64Imm:$src1),
                         "$dst = #$src1">;
def TFRI64_V2_ext : ALU64_rr<(outs DoubleRegs:$dst),
                             (ins s8Ext:$src1, s8Imm:$src2),
                             "$dst = combine(##$src1, #$src2)">;
}

//===----------------------------------------------------------------------===//
// ALU32/ALU -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ALU32/PERM +
//===----------------------------------------------------------------------===//
// Scalar mux register immediate.
let hasSideEffects = 0, isExtentSigned = 1, CextOpcode = "MUX",
    InputType = "imm", hasNewValue = 1, isExtendable = 1, opExtentBits = 8 in
class T_MUX1 <bit MajOp, dag ins, string AsmStr>
      : ALU32Inst <(outs IntRegs:$Rd), ins, AsmStr>, ImmRegRel {
  bits<5> Rd;
  bits<2> Pu;
  bits<8> s8;
  bits<5> Rs;

  let IClass = 0b0111;
  let Inst{27-24} = 0b0011;
  let Inst{23} = MajOp;
  let Inst{22-21} = Pu;
  let Inst{20-16} = Rs;
  let Inst{13}    = 0b0;
  let Inst{12-5}  = s8;
  let Inst{4-0}   = Rd;
}

let opExtendable = 2 in
def C2_muxri : T_MUX1<0b1, (ins PredRegs:$Pu, s8Ext:$s8, IntRegs:$Rs),
                           "$Rd = mux($Pu, #$s8, $Rs)">;

let opExtendable = 3 in
def C2_muxir : T_MUX1<0b0, (ins PredRegs:$Pu, IntRegs:$Rs, s8Ext:$s8),
                           "$Rd = mux($Pu, $Rs, #$s8)">;

def : Pat<(i32 (select I1:$Pu, s32ImmPred:$s8, I32:$Rs)),
          (C2_muxri I1:$Pu, s32ImmPred:$s8, I32:$Rs)>;

def : Pat<(i32 (select I1:$Pu, I32:$Rs, s32ImmPred:$s8)),
          (C2_muxir I1:$Pu, I32:$Rs, s32ImmPred:$s8)>;

// C2_muxii: Scalar mux immediates.
let isExtentSigned = 1, hasNewValue = 1, isExtendable = 1,
    opExtentBits = 8, opExtendable = 2 in
def C2_muxii: ALU32Inst <(outs IntRegs:$Rd),
                         (ins PredRegs:$Pu, s8Ext:$s8, s8Imm:$S8),
  "$Rd = mux($Pu, #$s8, #$S8)" ,
  [(set (i32 IntRegs:$Rd),
        (i32 (select I1:$Pu, s32ImmPred:$s8, s8ImmPred:$S8)))] > {
    bits<5> Rd;
    bits<2> Pu;
    bits<8> s8;
    bits<8> S8;

    let IClass = 0b0111;

    let Inst{27-25} = 0b101;
    let Inst{24-23} = Pu;
    let Inst{22-16} = S8{7-1};
    let Inst{13}    = S8{0};
    let Inst{12-5}  = s8;
    let Inst{4-0}   = Rd;
  }

let isCodeGenOnly = 1, isPseudo = 1 in
def PS_pselect : ALU64_rr<(outs DoubleRegs:$Rd),
      (ins PredRegs:$Pu, DoubleRegs:$Rs, DoubleRegs:$Rt),
      ".error \"should not emit\" ", []>;


//===----------------------------------------------------------------------===//
// template class for non-predicated alu32_2op instructions
// - aslh, asrh, sxtb, sxth, zxth
//===----------------------------------------------------------------------===//
let hasNewValue = 1, opNewValue = 0 in
class T_ALU32_2op <string mnemonic, bits<3> minOp> :
  ALU32Inst <(outs IntRegs:$Rd), (ins IntRegs:$Rs),
             "$Rd = "#mnemonic#"($Rs)", [] > {
  bits<5> Rd;
  bits<5> Rs;

  let IClass = 0b0111;

  let Inst{27-24} = 0b0000;
  let Inst{23-21} = minOp;
  let Inst{13} = 0b0;
  let Inst{4-0} = Rd;
  let Inst{20-16} = Rs;
}

//===----------------------------------------------------------------------===//
// template class for predicated alu32_2op instructions
// - aslh, asrh, sxtb, sxth, zxtb, zxth
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
class T_ALU32_2op_Pred <string mnemonic, bits<3> minOp, bit isPredNot,
                        bit isPredNew > :
  ALU32Inst <(outs IntRegs:$Rd), (ins PredRegs:$Pu, IntRegs:$Rs),
             !if(isPredNot, "if (!$Pu", "if ($Pu")
             #!if(isPredNew, ".new) ",") ")#"$Rd = "#mnemonic#"($Rs)"> {
  bits<5> Rd;
  bits<2> Pu;
  bits<5> Rs;

  let IClass = 0b0111;

  let Inst{27-24} = 0b0000;
  let Inst{23-21} = minOp;
  let Inst{13} = 0b1;
  let Inst{11} = isPredNot;
  let Inst{10} = isPredNew;
  let Inst{4-0} = Rd;
  let Inst{9-8} = Pu;
  let Inst{20-16} = Rs;
}

multiclass ALU32_2op_Pred<string mnemonic, bits<3> minOp, bit PredNot> {
  let isPredicatedFalse = PredNot in {
    def NAME : T_ALU32_2op_Pred<mnemonic, minOp, PredNot, 0>;

    // Predicate new
    let isPredicatedNew = 1 in
    def NAME#new : T_ALU32_2op_Pred<mnemonic, minOp, PredNot, 1>;
  }
}

multiclass ALU32_2op_base<string mnemonic, bits<3> minOp> {
  let BaseOpcode = mnemonic in {
    let isPredicable = 1, hasSideEffects = 0 in
    def A2_#NAME : T_ALU32_2op<mnemonic, minOp>;

    let isPredicated = 1, hasSideEffects = 0 in {
      defm A4_p#NAME#t : ALU32_2op_Pred<mnemonic, minOp, 0>;
      defm A4_p#NAME#f : ALU32_2op_Pred<mnemonic, minOp, 1>;
    }
  }
}

defm aslh : ALU32_2op_base<"aslh", 0b000>, PredNewRel;
defm asrh : ALU32_2op_base<"asrh", 0b001>, PredNewRel;
defm sxtb : ALU32_2op_base<"sxtb", 0b101>, PredNewRel;
defm sxth : ALU32_2op_base<"sxth", 0b111>, PredNewRel;
defm zxth : ALU32_2op_base<"zxth", 0b110>, PredNewRel;

// Rd=zxtb(Rs): assembler mapped to Rd=and(Rs,#255).
// Compiler would want to generate 'zxtb' instead of 'and' becuase 'zxtb' has
// predicated forms while 'and' doesn't. Since integrated assembler can't
// handle 'mapped' instructions, we need to encode 'zxtb' same as 'and' where
// immediate operand is set to '255'.

let hasNewValue = 1, opNewValue = 0 in
class T_ZXTB: ALU32Inst < (outs IntRegs:$Rd), (ins IntRegs:$Rs),
  "$Rd = zxtb($Rs)", [] > { // Rd = and(Rs,255)
    bits<5> Rd;
    bits<5> Rs;
    bits<10> s10 = 255;

    let IClass = 0b0111;

    let Inst{27-22} = 0b011000;
    let Inst{4-0} = Rd;
    let Inst{20-16} = Rs;
    let Inst{21} = s10{9};
    let Inst{13-5} = s10{8-0};
}

//Rd=zxtb(Rs): assembler mapped to "Rd=and(Rs,#255)
multiclass ZXTB_base <string mnemonic, bits<3> minOp> {
  let BaseOpcode = mnemonic in {
    let isPredicable = 1, hasSideEffects = 0 in
    def A2_#NAME : T_ZXTB;

    let isPredicated = 1, hasSideEffects = 0 in {
      defm A4_p#NAME#t : ALU32_2op_Pred<mnemonic, minOp, 0>;
      defm A4_p#NAME#f : ALU32_2op_Pred<mnemonic, minOp, 1>;
    }
  }
}

defm zxtb : ZXTB_base<"zxtb",0b100>, PredNewRel;

def: Pat<(shl I32:$src1, (i32 16)),   (A2_aslh I32:$src1)>;
def: Pat<(sra I32:$src1, (i32 16)),   (A2_asrh I32:$src1)>;
def: Pat<(sext_inreg I32:$src1, i8),  (A2_sxtb I32:$src1)>;
def: Pat<(sext_inreg I32:$src1, i16), (A2_sxth I32:$src1)>;

//===----------------------------------------------------------------------===//
// Template class for vector add and avg
//===----------------------------------------------------------------------===//

class T_VectALU_64 <string opc, bits<3> majOp, bits<3> minOp,
                   bit isSat, bit isRnd, bit isCrnd, bit SwapOps >
  : ALU64_rr < (outs DoubleRegs:$Rdd),
                (ins DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Rdd = "#opc#"($Rss, $Rtt)"#!if(isRnd, ":rnd", "")
                             #!if(isCrnd,":crnd","")
                             #!if(isSat, ":sat", ""),
  [], "", ALU64_tc_2_SLOT23 > {
    bits<5> Rdd;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1101;

    let Inst{27-24} = 0b0011;
    let Inst{23-21} = majOp;
    let Inst{20-16} = !if (SwapOps, Rtt, Rss);
    let Inst{12-8} = !if (SwapOps, Rss, Rtt);
    let Inst{7-5} = minOp;
    let Inst{4-0} = Rdd;
  }

// ALU64 - Vector add
// Rdd=vadd[u][bhw](Rss,Rtt)
let Itinerary = ALU64_tc_1_SLOT23 in {
  def A2_vaddub  : T_VectALU_64 < "vaddub", 0b000, 0b000, 0, 0, 0, 0>;
  def A2_vaddh   : T_VectALU_64 < "vaddh",  0b000, 0b010, 0, 0, 0, 0>;
  def A2_vaddw   : T_VectALU_64 < "vaddw",  0b000, 0b101, 0, 0, 0, 0>;
}

// Rdd=vadd[u][bhw](Rss,Rtt):sat
let Defs = [USR_OVF] in {
  def A2_vaddubs : T_VectALU_64 < "vaddub", 0b000, 0b001, 1, 0, 0, 0>;
  def A2_vaddhs  : T_VectALU_64 < "vaddh",  0b000, 0b011, 1, 0, 0, 0>;
  def A2_vadduhs : T_VectALU_64 < "vadduh", 0b000, 0b100, 1, 0, 0, 0>;
  def A2_vaddws  : T_VectALU_64 < "vaddw",  0b000, 0b110, 1, 0, 0, 0>;
}

// ALU64 - Vector average
// Rdd=vavg[u][bhw](Rss,Rtt)
let Itinerary = ALU64_tc_1_SLOT23 in {
  def A2_vavgub : T_VectALU_64 < "vavgub", 0b010, 0b000, 0, 0, 0, 0>;
  def A2_vavgh  : T_VectALU_64 < "vavgh",  0b010, 0b010, 0, 0, 0, 0>;
  def A2_vavguh : T_VectALU_64 < "vavguh", 0b010, 0b101, 0, 0, 0, 0>;
  def A2_vavgw  : T_VectALU_64 < "vavgw",  0b011, 0b000, 0, 0, 0, 0>;
  def A2_vavguw : T_VectALU_64 < "vavguw", 0b011, 0b011, 0, 0, 0, 0>;
}

// Rdd=vavg[u][bhw](Rss,Rtt)[:rnd|:crnd]
def A2_vavgubr : T_VectALU_64 < "vavgub", 0b010, 0b001, 0, 1, 0, 0>;
def A2_vavghr  : T_VectALU_64 < "vavgh",  0b010, 0b011, 0, 1, 0, 0>;
def A2_vavghcr : T_VectALU_64 < "vavgh",  0b010, 0b100, 0, 0, 1, 0>;
def A2_vavguhr : T_VectALU_64 < "vavguh", 0b010, 0b110, 0, 1, 0, 0>;

def A2_vavgwr  : T_VectALU_64 < "vavgw",  0b011, 0b001, 0, 1, 0, 0>;
def A2_vavgwcr : T_VectALU_64 < "vavgw",  0b011, 0b010, 0, 0, 1, 0>;
def A2_vavguwr : T_VectALU_64 < "vavguw", 0b011, 0b100, 0, 1, 0, 0>;

// Rdd=vnavg[bh](Rss,Rtt)
let Itinerary = ALU64_tc_1_SLOT23 in {
  def A2_vnavgh   : T_VectALU_64 < "vnavgh", 0b100, 0b000, 0, 0, 0, 1>;
  def A2_vnavgw   : T_VectALU_64 < "vnavgw", 0b100, 0b011, 0, 0, 0, 1>;
}

// Rdd=vnavg[bh](Rss,Rtt)[:rnd|:crnd]:sat
let Defs = [USR_OVF] in {
  def A2_vnavghr  : T_VectALU_64 < "vnavgh", 0b100, 0b001, 1, 1, 0, 1>;
  def A2_vnavghcr : T_VectALU_64 < "vnavgh", 0b100, 0b010, 1, 0, 1, 1>;
  def A2_vnavgwr  : T_VectALU_64 < "vnavgw", 0b100, 0b100, 1, 1, 0, 1>;
  def A2_vnavgwcr : T_VectALU_64 < "vnavgw", 0b100, 0b110, 1, 0, 1, 1>;
}

// Rdd=vsub[u][bh](Rss,Rtt)
let Itinerary = ALU64_tc_1_SLOT23 in {
  def A2_vsubub  : T_VectALU_64 < "vsubub", 0b001, 0b000, 0, 0, 0, 1>;
  def A2_vsubh   : T_VectALU_64 < "vsubh",  0b001, 0b010, 0, 0, 0, 1>;
  def A2_vsubw   : T_VectALU_64 < "vsubw",  0b001, 0b101, 0, 0, 0, 1>;
}

// Rdd=vsub[u][bh](Rss,Rtt):sat
let Defs = [USR_OVF] in {
  def A2_vsububs : T_VectALU_64 < "vsubub", 0b001, 0b001, 1, 0, 0, 1>;
  def A2_vsubhs  : T_VectALU_64 < "vsubh",  0b001, 0b011, 1, 0, 0, 1>;
  def A2_vsubuhs : T_VectALU_64 < "vsubuh", 0b001, 0b100, 1, 0, 0, 1>;
  def A2_vsubws  : T_VectALU_64 < "vsubw",  0b001, 0b110, 1, 0, 0, 1>;
}

// Rdd=vmax[u][bhw](Rss,Rtt)
def A2_vmaxb  : T_VectALU_64 < "vmaxb",  0b110, 0b110, 0, 0, 0, 1>;
def A2_vmaxub : T_VectALU_64 < "vmaxub", 0b110, 0b000, 0, 0, 0, 1>;
def A2_vmaxh  : T_VectALU_64 < "vmaxh",  0b110, 0b001, 0, 0, 0, 1>;
def A2_vmaxuh : T_VectALU_64 < "vmaxuh", 0b110, 0b010, 0, 0, 0, 1>;
def A2_vmaxw  : T_VectALU_64 < "vmaxw",  0b110, 0b011, 0, 0, 0, 1>;
def A2_vmaxuw : T_VectALU_64 < "vmaxuw", 0b101, 0b101, 0, 0, 0, 1>;

// Rdd=vmin[u][bhw](Rss,Rtt)
def A2_vminb  : T_VectALU_64 < "vminb",  0b110, 0b111, 0, 0, 0, 1>;
def A2_vminub : T_VectALU_64 < "vminub", 0b101, 0b000, 0, 0, 0, 1>;
def A2_vminh  : T_VectALU_64 < "vminh",  0b101, 0b001, 0, 0, 0, 1>;
def A2_vminuh : T_VectALU_64 < "vminuh", 0b101, 0b010, 0, 0, 0, 1>;
def A2_vminw  : T_VectALU_64 < "vminw",  0b101, 0b011, 0, 0, 0, 1>;
def A2_vminuw : T_VectALU_64 < "vminuw", 0b101, 0b100, 0, 0, 0, 1>;

//===----------------------------------------------------------------------===//
// Template class for vector compare
//===----------------------------------------------------------------------===//
let hasSideEffects = 0 in
class T_vcmp <string Str, bits<4> minOp>
  : ALU64_rr <(outs PredRegs:$Pd),
              (ins DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Pd = "#Str#"($Rss, $Rtt)", [],
  "", ALU64_tc_2early_SLOT23> {
    bits<2> Pd;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1101;

    let Inst{27-23} = 0b00100;
    let Inst{13} = minOp{3};
    let Inst{7-5} = minOp{2-0};
    let Inst{1-0} = Pd;
    let Inst{20-16} = Rss;
    let Inst{12-8} = Rtt;
  }

class T_vcmp_pat<InstHexagon MI, PatFrag Op, ValueType T>
  : Pat<(i1 (Op (T DoubleRegs:$Rss), (T DoubleRegs:$Rtt))),
        (i1 (MI DoubleRegs:$Rss, DoubleRegs:$Rtt))>;

// Vector compare bytes
def A2_vcmpbeq  : T_vcmp <"vcmpb.eq",  0b0110>;
def A2_vcmpbgtu : T_vcmp <"vcmpb.gtu", 0b0111>;

// Vector compare halfwords
def A2_vcmpheq  : T_vcmp <"vcmph.eq",  0b0011>;
def A2_vcmphgt  : T_vcmp <"vcmph.gt",  0b0100>;
def A2_vcmphgtu : T_vcmp <"vcmph.gtu", 0b0101>;

// Vector compare words
def A2_vcmpweq  : T_vcmp <"vcmpw.eq",  0b0000>;
def A2_vcmpwgt  : T_vcmp <"vcmpw.gt",  0b0001>;
def A2_vcmpwgtu : T_vcmp <"vcmpw.gtu", 0b0010>;

def: T_vcmp_pat<A2_vcmpbeq,  seteq,  v8i8>;
def: T_vcmp_pat<A2_vcmpbgtu, setugt, v8i8>;
def: T_vcmp_pat<A2_vcmpheq,  seteq,  v4i16>;
def: T_vcmp_pat<A2_vcmphgt,  setgt,  v4i16>;
def: T_vcmp_pat<A2_vcmphgtu, setugt, v4i16>;
def: T_vcmp_pat<A2_vcmpweq,  seteq,  v2i32>;
def: T_vcmp_pat<A2_vcmpwgt,  setgt,  v2i32>;
def: T_vcmp_pat<A2_vcmpwgtu, setugt, v2i32>;

//===----------------------------------------------------------------------===//
// ALU32/PERM -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ALU32/PRED +
//===----------------------------------------------------------------------===//
// No bits needed.  If cmp.ge is found the assembler parser will
// transform it to cmp.gt subtracting 1 from the immediate.
let isPseudo = 1 in {
def C2_cmpgei: ALU32Inst <
  (outs PredRegs:$Pd), (ins IntRegs:$Rs, s8Ext:$s8),
  "$Pd = cmp.ge($Rs, #$s8)">;
def C2_cmpgeui: ALU32Inst <
  (outs PredRegs:$Pd), (ins IntRegs:$Rs, u8Ext:$s8),
  "$Pd = cmp.geu($Rs, #$s8)">;
}


//===----------------------------------------------------------------------===//
// ALU32/PRED -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// ALU64/ALU +
//===----------------------------------------------------------------------===//
// Add.
//===----------------------------------------------------------------------===//
// Template Class
// Add/Subtract halfword
// Rd=add(Rt.L,Rs.[HL])[:sat]
// Rd=sub(Rt.L,Rs.[HL])[:sat]
// Rd=add(Rt.[LH],Rs.[HL])[:sat][:<16]
// Rd=sub(Rt.[LH],Rs.[HL])[:sat][:<16]
//===----------------------------------------------------------------------===//

let  hasNewValue = 1, opNewValue = 0 in
class T_XTYPE_ADD_SUB <bits<2> LHbits, bit isSat, bit hasShift, bit isSub>
  : ALU64Inst <(outs IntRegs:$Rd), (ins IntRegs:$Rt, IntRegs:$Rs),
  "$Rd = "#!if(isSub,"sub","add")#"($Rt."
          #!if(hasShift, !if(LHbits{1},"h","l"),"l") #", $Rs."
          #!if(hasShift, !if(LHbits{0},"h)","l)"), !if(LHbits{1},"h)","l)"))
          #!if(isSat,":sat","")
          #!if(hasShift,":<<16",""), [], "", ALU64_tc_1_SLOT23> {
    bits<5> Rd;
    bits<5> Rt;
    bits<5> Rs;
    let IClass = 0b1101;

    let Inst{27-23} = 0b01010;
    let Inst{22} = hasShift;
    let Inst{21} = isSub;
    let Inst{7} = isSat;
    let Inst{6-5} = LHbits;
    let Inst{4-0} = Rd;
    let Inst{12-8} = Rt;
    let Inst{20-16} = Rs;
  }

//Rd=sub(Rt.L,Rs.[LH])
def A2_subh_l16_ll : T_XTYPE_ADD_SUB <0b00, 0, 0, 1>;
def A2_subh_l16_hl : T_XTYPE_ADD_SUB <0b10, 0, 0, 1>;

//Rd=add(Rt.L,Rs.[LH])
def A2_addh_l16_ll : T_XTYPE_ADD_SUB <0b00, 0, 0, 0>;
def A2_addh_l16_hl : T_XTYPE_ADD_SUB <0b10, 0, 0, 0>;

let Itinerary = ALU64_tc_2_SLOT23, Defs = [USR_OVF] in {
  //Rd=sub(Rt.L,Rs.[LH]):sat
  def A2_subh_l16_sat_ll : T_XTYPE_ADD_SUB <0b00, 1, 0, 1>;
  def A2_subh_l16_sat_hl : T_XTYPE_ADD_SUB <0b10, 1, 0, 1>;

  //Rd=add(Rt.L,Rs.[LH]):sat
  def A2_addh_l16_sat_ll : T_XTYPE_ADD_SUB <0b00, 1, 0, 0>;
  def A2_addh_l16_sat_hl : T_XTYPE_ADD_SUB <0b10, 1, 0, 0>;
}

//Rd=sub(Rt.[LH],Rs.[LH]):<<16
def A2_subh_h16_ll : T_XTYPE_ADD_SUB <0b00, 0, 1, 1>;
def A2_subh_h16_lh : T_XTYPE_ADD_SUB <0b01, 0, 1, 1>;
def A2_subh_h16_hl : T_XTYPE_ADD_SUB <0b10, 0, 1, 1>;
def A2_subh_h16_hh : T_XTYPE_ADD_SUB <0b11, 0, 1, 1>;

//Rd=add(Rt.[LH],Rs.[LH]):<<16
def A2_addh_h16_ll : T_XTYPE_ADD_SUB <0b00, 0, 1, 0>;
def A2_addh_h16_lh : T_XTYPE_ADD_SUB <0b01, 0, 1, 0>;
def A2_addh_h16_hl : T_XTYPE_ADD_SUB <0b10, 0, 1, 0>;
def A2_addh_h16_hh : T_XTYPE_ADD_SUB <0b11, 0, 1, 0>;

let Itinerary = ALU64_tc_2_SLOT23, Defs = [USR_OVF] in {
  //Rd=sub(Rt.[LH],Rs.[LH]):sat:<<16
  def A2_subh_h16_sat_ll : T_XTYPE_ADD_SUB <0b00, 1, 1, 1>;
  def A2_subh_h16_sat_lh : T_XTYPE_ADD_SUB <0b01, 1, 1, 1>;
  def A2_subh_h16_sat_hl : T_XTYPE_ADD_SUB <0b10, 1, 1, 1>;
  def A2_subh_h16_sat_hh : T_XTYPE_ADD_SUB <0b11, 1, 1, 1>;

  //Rd=add(Rt.[LH],Rs.[LH]):sat:<<16
  def A2_addh_h16_sat_ll : T_XTYPE_ADD_SUB <0b00, 1, 1, 0>;
  def A2_addh_h16_sat_lh : T_XTYPE_ADD_SUB <0b01, 1, 1, 0>;
  def A2_addh_h16_sat_hl : T_XTYPE_ADD_SUB <0b10, 1, 1, 0>;
  def A2_addh_h16_sat_hh : T_XTYPE_ADD_SUB <0b11, 1, 1, 0>;
}

// Add halfword.
def: Pat<(sext_inreg (add I32:$src1, I32:$src2), i16),
         (A2_addh_l16_ll I32:$src1, I32:$src2)>;

def: Pat<(sra (add (shl I32:$src1, (i32 16)), I32:$src2), (i32 16)),
         (A2_addh_l16_hl I32:$src1, I32:$src2)>;

def: Pat<(shl (add I32:$src1, I32:$src2), (i32 16)),
         (A2_addh_h16_ll I32:$src1, I32:$src2)>;

// Subtract halfword.
def: Pat<(sext_inreg (sub I32:$src1, I32:$src2), i16),
         (A2_subh_l16_ll I32:$src1, I32:$src2)>;

def: Pat<(shl (sub I32:$src1, I32:$src2), (i32 16)),
         (A2_subh_h16_ll I32:$src1, I32:$src2)>;

let hasSideEffects = 0, hasNewValue = 1 in
def S2_parityp: ALU64Inst<(outs IntRegs:$Rd),
      (ins DoubleRegs:$Rs, DoubleRegs:$Rt),
      "$Rd = parity($Rs, $Rt)", [], "", ALU64_tc_2_SLOT23> {
  bits<5> Rd;
  bits<5> Rs;
  bits<5> Rt;

  let IClass = 0b1101;
  let Inst{27-24} = 0b0000;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{4-0} = Rd;
}

let hasNewValue = 1, opNewValue = 0, hasSideEffects = 0 in
class T_XTYPE_MIN_MAX < bit isMax, bit isUnsigned >
  : ALU64Inst < (outs IntRegs:$Rd), (ins IntRegs:$Rt, IntRegs:$Rs),
  "$Rd = "#!if(isMax,"max","min")#!if(isUnsigned,"u","")
          #"($Rt, $Rs)", [], "", ALU64_tc_2_SLOT23> {
    bits<5> Rd;
    bits<5> Rt;
    bits<5> Rs;

    let IClass = 0b1101;

    let Inst{27-23} = 0b01011;
    let Inst{22-21} = !if(isMax, 0b10, 0b01);
    let Inst{7} = isUnsigned;
    let Inst{4-0} = Rd;
    let Inst{12-8} = !if(isMax, Rs, Rt);
    let Inst{20-16} = !if(isMax, Rt, Rs);
  }

def A2_min  : T_XTYPE_MIN_MAX < 0, 0 >;
def A2_minu : T_XTYPE_MIN_MAX < 0, 1 >;
def A2_max  : T_XTYPE_MIN_MAX < 1, 0 >;
def A2_maxu : T_XTYPE_MIN_MAX < 1, 1 >;

// Here, depending on  the operand being selected, we'll either generate a
// min or max instruction.
// Ex:
// (a>b)?a:b --> max(a,b) => Here check performed is '>' and the value selected
// is the larger of two. So, the corresponding HexagonInst is passed in 'Inst'.
// (a>b)?b:a --> min(a,b) => Here check performed is '>' but the smaller value
// is selected and the corresponding HexagonInst is passed in 'SwapInst'.

multiclass T_MinMax_pats <PatFrag Op, RegisterClass RC, ValueType VT,
                          InstHexagon Inst, InstHexagon SwapInst> {
  def: Pat<(select (i1 (Op (VT RC:$src1), (VT RC:$src2))),
                   (VT RC:$src1), (VT RC:$src2)),
           (Inst RC:$src1, RC:$src2)>;
  def: Pat<(select (i1 (Op (VT RC:$src1), (VT RC:$src2))),
                   (VT RC:$src2), (VT RC:$src1)),
           (SwapInst RC:$src1, RC:$src2)>;
}


multiclass MinMax_pats <PatFrag Op, InstHexagon Inst, InstHexagon SwapInst> {
  defm: T_MinMax_pats<Op, IntRegs, i32, Inst, SwapInst>;

  def: Pat<(sext_inreg (i32 (select (i1 (Op (i32 PositiveHalfWord:$src1),
                                            (i32 PositiveHalfWord:$src2))),
                                    (i32 PositiveHalfWord:$src1),
                                    (i32 PositiveHalfWord:$src2))), i16),
           (Inst IntRegs:$src1, IntRegs:$src2)>;

  def: Pat<(sext_inreg (i32 (select (i1 (Op (i32 PositiveHalfWord:$src1),
                                            (i32 PositiveHalfWord:$src2))),
                                    (i32 PositiveHalfWord:$src2),
                                    (i32 PositiveHalfWord:$src1))), i16),
           (SwapInst IntRegs:$src1, IntRegs:$src2)>;
}

let AddedComplexity = 200 in {
  defm: MinMax_pats<setge,  A2_max,  A2_min>;
  defm: MinMax_pats<setgt,  A2_max,  A2_min>;
  defm: MinMax_pats<setle,  A2_min,  A2_max>;
  defm: MinMax_pats<setlt,  A2_min,  A2_max>;
  defm: MinMax_pats<setuge, A2_maxu, A2_minu>;
  defm: MinMax_pats<setugt, A2_maxu, A2_minu>;
  defm: MinMax_pats<setule, A2_minu, A2_maxu>;
  defm: MinMax_pats<setult, A2_minu, A2_maxu>;
}

class T_cmp64_rr<string mnemonic, bits<3> MinOp, bit IsComm>
  : ALU64_rr<(outs PredRegs:$Pd), (ins DoubleRegs:$Rs, DoubleRegs:$Rt),
             "$Pd = "#mnemonic#"($Rs, $Rt)", [], "", ALU64_tc_2early_SLOT23> {
  let isCompare = 1;
  let isCommutable = IsComm;
  let hasSideEffects = 0;

  bits<2> Pd;
  bits<5> Rs;
  bits<5> Rt;

  let IClass = 0b1101;
  let Inst{27-21} = 0b0010100;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{7-5} = MinOp;
  let Inst{1-0} = Pd;
}

def C2_cmpeqp  : T_cmp64_rr<"cmp.eq",  0b000, 1>;
def C2_cmpgtp  : T_cmp64_rr<"cmp.gt",  0b010, 0>;
def C2_cmpgtup : T_cmp64_rr<"cmp.gtu", 0b100, 0>;

class T_cmp64_rr_pat<InstHexagon MI, PatFrag CmpOp>
  : Pat<(i1 (CmpOp (i64 DoubleRegs:$Rs), (i64 DoubleRegs:$Rt))),
        (i1 (MI DoubleRegs:$Rs, DoubleRegs:$Rt))>;

def: T_cmp64_rr_pat<C2_cmpeqp,  seteq>;
def: T_cmp64_rr_pat<C2_cmpgtp,  setgt>;
def: T_cmp64_rr_pat<C2_cmpgtup, setugt>;
def: T_cmp64_rr_pat<C2_cmpgtp,  RevCmp<setlt>>;
def: T_cmp64_rr_pat<C2_cmpgtup, RevCmp<setult>>;

def C2_vmux : ALU64_rr<(outs DoubleRegs:$Rd),
      (ins PredRegs:$Pu, DoubleRegs:$Rs, DoubleRegs:$Rt),
      "$Rd = vmux($Pu, $Rs, $Rt)", [], "", ALU64_tc_1_SLOT23> {
  let hasSideEffects = 0;

  bits<5> Rd;
  bits<2> Pu;
  bits<5> Rs;
  bits<5> Rt;

  let IClass = 0b1101;
  let Inst{27-24} = 0b0001;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{6-5} = Pu;
  let Inst{4-0} = Rd;
}

class T_ALU64_rr<string mnemonic, string suffix, bits<4> RegType,
                 bits<3> MajOp, bits<3> MinOp, bit OpsRev, bit IsComm,
                 string Op2Pfx>
  : ALU64_rr<(outs DoubleRegs:$Rd), (ins DoubleRegs:$Rs, DoubleRegs:$Rt),
             "$Rd = " #mnemonic# "($Rs, " #Op2Pfx# "$Rt)" #suffix, [],
             "", ALU64_tc_1_SLOT23> {
  let hasSideEffects = 0;
  let isCommutable = IsComm;

  bits<5> Rs;
  bits<5> Rt;
  bits<5> Rd;

  let IClass = 0b1101;
  let Inst{27-24} = RegType;
  let Inst{23-21} = MajOp;
  let Inst{20-16} = !if (OpsRev,Rt,Rs);
  let Inst{12-8} = !if (OpsRev,Rs,Rt);
  let Inst{7-5} = MinOp;
  let Inst{4-0} = Rd;
}

class T_ALU64_arith<string mnemonic, bits<3> MajOp, bits<3> MinOp, bit IsSat,
                    bit OpsRev, bit IsComm>
  : T_ALU64_rr<mnemonic, !if(IsSat,":sat",""), 0b0011, MajOp, MinOp, OpsRev,
               IsComm, "">;

def A2_addp : T_ALU64_arith<"add", 0b000, 0b111, 0, 0, 1>;
def A2_subp : T_ALU64_arith<"sub", 0b001, 0b111, 0, 1, 0>;

def: Pat<(i64 (add I64:$Rs, I64:$Rt)), (A2_addp I64:$Rs, I64:$Rt)>;
def: Pat<(i64 (sub I64:$Rs, I64:$Rt)), (A2_subp I64:$Rs, I64:$Rt)>;

class T_ALU64_logical<string mnemonic, bits<3> MinOp, bit OpsRev, bit IsComm,
                      bit IsNeg>
  : T_ALU64_rr<mnemonic, "", 0b0011, 0b111, MinOp, OpsRev, IsComm,
               !if(IsNeg,"~","")>;

def A2_andp : T_ALU64_logical<"and", 0b000, 0, 1, 0>;
def A2_orp  : T_ALU64_logical<"or",  0b010, 0, 1, 0>;
def A2_xorp : T_ALU64_logical<"xor", 0b100, 0, 1, 0>;

def: Pat<(i64 (and I64:$Rs, I64:$Rt)), (A2_andp I64:$Rs, I64:$Rt)>;
def: Pat<(i64 (or  I64:$Rs, I64:$Rt)), (A2_orp  I64:$Rs, I64:$Rt)>;
def: Pat<(i64 (xor I64:$Rs, I64:$Rt)), (A2_xorp I64:$Rs, I64:$Rt)>;

//===----------------------------------------------------------------------===//
// ALU64/ALU -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU64/BIT +
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// ALU64/BIT -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU64/PERM +
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
// ALU64/PERM -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CR +
//===----------------------------------------------------------------------===//
// Logical reductions on predicates.

// Looping instructions.

// Pipelined looping instructions.

// Logical operations on predicates.
let hasSideEffects = 0 in
class T_LOGICAL_1OP<string MnOp, bits<2> OpBits>
    : CRInst<(outs PredRegs:$Pd), (ins PredRegs:$Ps),
             "$Pd = " # MnOp # "($Ps)", [], "", CR_tc_2early_SLOT23> {
  bits<2> Pd;
  bits<2> Ps;

  let IClass = 0b0110;
  let Inst{27-23} = 0b10111;
  let Inst{22-21} = OpBits;
  let Inst{20} = 0b0;
  let Inst{17-16} = Ps;
  let Inst{13} = 0b0;
  let Inst{1-0} = Pd;
}

def C2_any8 : T_LOGICAL_1OP<"any8", 0b00>;
def C2_all8 : T_LOGICAL_1OP<"all8", 0b01>;
def C2_not  : T_LOGICAL_1OP<"not",  0b10>;

def: Pat<(i1 (not (i1 PredRegs:$Ps))),
         (C2_not PredRegs:$Ps)>;

let hasSideEffects = 0 in
class T_LOGICAL_2OP<string MnOp, bits<3> OpBits, bit IsNeg, bit Rev>
    : CRInst<(outs PredRegs:$Pd), (ins PredRegs:$Ps, PredRegs:$Pt),
             "$Pd = " # MnOp # "($Ps, " # !if (IsNeg,"!","") # "$Pt)",
             [], "", CR_tc_2early_SLOT23> {
  bits<2> Pd;
  bits<2> Ps;
  bits<2> Pt;

  let IClass = 0b0110;
  let Inst{27-24} = 0b1011;
  let Inst{23-21} = OpBits;
  let Inst{20} = 0b0;
  let Inst{17-16} = !if(Rev,Pt,Ps);  // Rs and Rt are reversed for some
  let Inst{13} = 0b0;                // instructions.
  let Inst{9-8} = !if(Rev,Ps,Pt);
  let Inst{1-0} = Pd;
}

def C2_and  : T_LOGICAL_2OP<"and", 0b000, 0, 1>;
def C2_or   : T_LOGICAL_2OP<"or",  0b001, 0, 1>;
def C2_xor  : T_LOGICAL_2OP<"xor", 0b010, 0, 0>;
def C2_andn : T_LOGICAL_2OP<"and", 0b011, 1, 1>;
def C2_orn  : T_LOGICAL_2OP<"or",  0b111, 1, 1>;

def: Pat<(i1 (and I1:$Ps, I1:$Pt)),       (C2_and  I1:$Ps, I1:$Pt)>;
def: Pat<(i1 (or  I1:$Ps, I1:$Pt)),       (C2_or   I1:$Ps, I1:$Pt)>;
def: Pat<(i1 (xor I1:$Ps, I1:$Pt)),       (C2_xor  I1:$Ps, I1:$Pt)>;
def: Pat<(i1 (and I1:$Ps, (not I1:$Pt))), (C2_andn I1:$Ps, I1:$Pt)>;
def: Pat<(i1 (or  I1:$Ps, (not I1:$Pt))), (C2_orn  I1:$Ps, I1:$Pt)>;

let hasSideEffects = 0, hasNewValue = 1 in
def C2_vitpack : SInst<(outs IntRegs:$Rd), (ins PredRegs:$Ps, PredRegs:$Pt),
      "$Rd = vitpack($Ps, $Pt)", [], "", S_2op_tc_1_SLOT23> {
  bits<5> Rd;
  bits<2> Ps;
  bits<2> Pt;

  let IClass = 0b1000;
  let Inst{27-24} = 0b1001;
  let Inst{22-21} = 0b00;
  let Inst{17-16} = Ps;
  let Inst{9-8} = Pt;
  let Inst{4-0} = Rd;
}

let hasSideEffects = 0 in
def C2_mask : SInst<(outs DoubleRegs:$Rd), (ins PredRegs:$Pt),
      "$Rd = mask($Pt)", [], "", S_2op_tc_1_SLOT23> {
  bits<5> Rd;
  bits<2> Pt;

  let IClass = 0b1000;
  let Inst{27-24} = 0b0110;
  let Inst{9-8} = Pt;
  let Inst{4-0} = Rd;
}

// User control register transfer.
//===----------------------------------------------------------------------===//
// CR -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// JR +
//===----------------------------------------------------------------------===//

def retflag : SDNode<"HexagonISD::RET_FLAG", SDTNone,
                     [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def eh_return: SDNode<"HexagonISD::EH_RETURN", SDTNone, [SDNPHasChain]>;

class CondStr<string CReg, bit True, bit New> {
  string S = "if (" # !if(True,"","!") # CReg # !if(New,".new","") # ") ";
}
class JumpOpcStr<string Mnemonic, bit New, bit Taken> {
  string S = Mnemonic # !if(Taken, ":t", ":nt");
}

let isBranch = 1, isBarrier = 1, Defs = [PC], hasSideEffects = 0,
    isPredicable = 1,
    isExtendable = 1, opExtendable = 0, isExtentSigned = 1,
    opExtentBits = 24, opExtentAlign = 2, InputType = "imm" in
class T_JMP<string ExtStr>
  : JInst_CJUMP_UCJUMP<(outs), (ins brtarget:$dst),
      "jump " # ExtStr # "$dst",
      [], "", J_tc_2early_CJUMP_UCJUMP_ARCHDEPSLOT> {
    bits<24> dst;
    let IClass = 0b0101;

    let Inst{27-25} = 0b100;
    let Inst{24-16} = dst{23-15};
    let Inst{13-1} = dst{14-2};
}

let isBranch = 1, Defs = [PC], hasSideEffects = 0, isPredicated = 1,
    isExtendable = 1, opExtendable = 1, isExtentSigned = 1,
    opExtentBits = 17, opExtentAlign = 2, InputType = "imm" in
class T_JMP_c<bit PredNot, bit isPredNew, bit isTak, string ExtStr>
  : JInst_CJUMP_UCJUMP<(outs), (ins PredRegs:$src, brtarget:$dst),
      CondStr<"$src", !if(PredNot,0,1), isPredNew>.S #
        JumpOpcStr<"jump", isPredNew, isTak>.S # " " #
        ExtStr # "$dst",
      [], "", J_tc_2early_CJUMP_UCJUMP_ARCHDEPSLOT>, ImmRegRel {
    let isTaken = isTak;
    let isPredicatedFalse = PredNot;
    let isPredicatedNew = isPredNew;
    bits<2> src;
    bits<17> dst;

    let IClass = 0b0101;

    let Inst{27-24} = 0b1100;
    let Inst{21} = PredNot;
    let Inst{12} = isTak;
    let Inst{11} = isPredNew;
    let Inst{9-8} = src;
    let Inst{23-22} = dst{16-15};
    let Inst{20-16} = dst{14-10};
    let Inst{13} = dst{9};
    let Inst{7-1} = dst{8-2};
  }

multiclass JMP_Pred<bit PredNot, string ExtStr> {
  def NAME       : T_JMP_c<PredNot, 0, 0, ExtStr>; // not taken
  // Predicate new
  def NAME#newpt : T_JMP_c<PredNot, 1, 1, ExtStr>; // taken
  def NAME#new   : T_JMP_c<PredNot, 1, 0, ExtStr>; // not taken
}

multiclass JMP_base<string BaseOp, string ExtStr> {
  let BaseOpcode = BaseOp in {
    def NAME : T_JMP<ExtStr>;
    defm t : JMP_Pred<0, ExtStr>;
    defm f : JMP_Pred<1, ExtStr>;
  }
}

// Jumps to address stored in a register, JUMPR_MISC
// if ([[!]P[.new]]) jumpr[:t/nt] Rs
let isBranch = 1, isIndirectBranch = 1, isBarrier = 1, Defs = [PC],
    isPredicable = 1, hasSideEffects = 0, InputType = "reg" in
class T_JMPr
  : JRInst<(outs), (ins IntRegs:$dst),
      "jumpr $dst", [], "", J_tc_2early_SLOT2> {
    bits<5> dst;

    let IClass = 0b0101;
    let Inst{27-21} = 0b0010100;
    let Inst{20-16} = dst;
}

let isBranch = 1, isIndirectBranch = 1, Defs = [PC], isPredicated = 1,
    hasSideEffects = 0, InputType = "reg" in
class T_JMPr_c <bit PredNot, bit isPredNew, bit isTak>
  : JRInst <(outs), (ins PredRegs:$src, IntRegs:$dst),
      CondStr<"$src", !if(PredNot,0,1), isPredNew>.S #
        JumpOpcStr<"jumpr", isPredNew, isTak>.S # " $dst", [],
      "", J_tc_2early_SLOT2> {

    let isTaken = isTak;
    let isPredicatedFalse = PredNot;
    let isPredicatedNew = isPredNew;
    bits<2> src;
    bits<5> dst;

    let IClass = 0b0101;

    let Inst{27-22} = 0b001101;
    let Inst{21} = PredNot;
    let Inst{20-16} = dst;
    let Inst{12} = isTak;
    let Inst{11} = isPredNew;
    let Inst{9-8} = src;
}

multiclass JMPR_Pred<bit PredNot> {
  def NAME        : T_JMPr_c<PredNot, 0, 0>; // not taken
  // Predicate new
  def NAME#newpt  : T_JMPr_c<PredNot, 1, 1>; // taken
  def NAME#new    : T_JMPr_c<PredNot, 1, 0>; // not taken
}

multiclass JMPR_base<string BaseOp> {
  let BaseOpcode = BaseOp in {
    def NAME : T_JMPr;
    defm t : JMPR_Pred<0>;
    defm f : JMPR_Pred<1>;
  }
}

let isCall = 1, hasSideEffects = 1 in
class JUMPR_MISC_CALLR<bit isPred, bit isPredNot,
               dag InputDag = (ins IntRegs:$Rs)>
  : JRInst<(outs), InputDag,
      !if(isPred, !if(isPredNot, "if (!$Pu) callr $Rs",
                                 "if ($Pu) callr $Rs"),
                                 "callr $Rs"),
      [], "", J_tc_2early_SLOT2> {
    bits<5> Rs;
    bits<2> Pu;
    let isPredicated = isPred;
    let isPredicatedFalse = isPredNot;

    let IClass = 0b0101;
    let Inst{27-25} = 0b000;
    let Inst{24-23} = !if (isPred, 0b10, 0b01);
    let Inst{22} = 0;
    let Inst{21} = isPredNot;
    let Inst{9-8} = !if (isPred, Pu, 0b00);
    let Inst{20-16} = Rs;

  }

let Defs = VolatileV3.Regs in {
  def J2_callrt : JUMPR_MISC_CALLR<1, 0, (ins PredRegs:$Pu, IntRegs:$Rs)>;
  def J2_callrf : JUMPR_MISC_CALLR<1, 1, (ins PredRegs:$Pu, IntRegs:$Rs)>;
}

let isTerminator = 1, hasSideEffects = 0 in {
  defm J2_jump : JMP_base<"JMP", "">, PredNewRel;

  defm J2_jumpr : JMPR_base<"JMPr">, PredNewRel;

  let isReturn = 1, isCodeGenOnly = 1 in
  defm PS_jmpret : JMPR_base<"JMPret">, PredNewRel;
}

let validSubTargets  = HasV60SubT in
multiclass JMPpt_base<string BaseOp> {
  let BaseOpcode = BaseOp in {
    def tpt : T_JMP_c <0, 0, 1, "">; // Predicate true - taken
    def fpt : T_JMP_c <1, 0, 1, "">; // Predicate false - taken
  }
}

let validSubTargets  = HasV60SubT in
multiclass JMPRpt_base<string BaseOp> {
  let BaseOpcode = BaseOp in {
    def tpt : T_JMPr_c<0, 0, 1>; // predicate true - taken
    def fpt : T_JMPr_c<1, 0, 1>; // predicate false - taken
  }
}

defm J2_jumpr : JMPRpt_base<"JMPr">;
defm J2_jump  : JMPpt_base<"JMP">;

def: Pat<(br bb:$dst),
         (J2_jump brtarget:$dst)>;
def: Pat<(retflag),
         (PS_jmpret (i32 R31))>;
def: Pat<(brcond (i1 PredRegs:$src1), bb:$offset),
         (J2_jumpt PredRegs:$src1, bb:$offset)>;

// A return through builtin_eh_return.
let isReturn = 1, isTerminator = 1, isBarrier = 1, hasSideEffects = 0,
    isCodeGenOnly = 1, Defs = [PC], Uses = [R28], isPredicable = 0 in
def EH_RETURN_JMPR : T_JMPr;

def: Pat<(eh_return),
         (EH_RETURN_JMPR (i32 R31))>;
def: Pat<(brind (i32 IntRegs:$dst)),
         (J2_jumpr IntRegs:$dst)>;

//===----------------------------------------------------------------------===//
// JR -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LD +
//===----------------------------------------------------------------------===//

// Load - Base with Immediate offset addressing mode
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, AddedComplexity = 20 in
class T_load_io <string mnemonic, RegisterClass RC, bits<4> MajOp,
                 Operand ImmOp>
  : LDInst<(outs RC:$dst), (ins IntRegs:$src1, ImmOp:$offset),
  "$dst = "#mnemonic#"($src1 + #$offset)", []>, AddrModeRel {
    bits<4> name;
    bits<5> dst;
    bits<5> src1;
    bits<14> offset;
    bits<11> offsetBits;

    string ImmOpStr = !cast<string>(ImmOp);
    let offsetBits = !if (!eq(ImmOpStr, "s11_3Ext"), offset{13-3},
                     !if (!eq(ImmOpStr, "s11_2Ext"), offset{12-2},
                     !if (!eq(ImmOpStr, "s11_1Ext"), offset{11-1},
                                      /* s11_0Ext */ offset{10-0})));
    let opExtentBits = !if (!eq(ImmOpStr, "s11_3Ext"), 14,
                       !if (!eq(ImmOpStr, "s11_2Ext"), 13,
                       !if (!eq(ImmOpStr, "s11_1Ext"), 12,
                                        /* s11_0Ext */ 11)));
    let hasNewValue = !if (!eq(!cast<string>(RC), "DoubleRegs"), 0, 1);

    let IClass = 0b1001;

    let Inst{27}    = 0b0;
    let Inst{26-25} = offsetBits{10-9};
    let Inst{24-21} = MajOp;
    let Inst{20-16} = src1;
    let Inst{13-5}  = offsetBits{8-0};
    let Inst{4-0}   = dst;
  }

let opExtendable = 3, isExtentSigned = 0, isPredicated = 1 in
class T_pload_io <string mnemonic, RegisterClass RC, bits<4>MajOp,
                  Operand ImmOp, bit isNot, bit isPredNew>
  : LDInst<(outs RC:$dst),
           (ins PredRegs:$src1, IntRegs:$src2, ImmOp:$offset),
  "if ("#!if(isNot, "!$src1", "$src1")
       #!if(isPredNew, ".new", "")
       #") $dst = "#mnemonic#"($src2 + #$offset)",
  [],"", V2LDST_tc_ld_SLOT01> , AddrModeRel {
    bits<5> dst;
    bits<2> src1;
    bits<5> src2;
    bits<9> offset;
    bits<6> offsetBits;
    string ImmOpStr = !cast<string>(ImmOp);

    let offsetBits = !if (!eq(ImmOpStr, "u6_3Ext"), offset{8-3},
                     !if (!eq(ImmOpStr, "u6_2Ext"), offset{7-2},
                     !if (!eq(ImmOpStr, "u6_1Ext"), offset{6-1},
                                      /* u6_0Ext */ offset{5-0})));
    let opExtentBits = !if (!eq(ImmOpStr, "u6_3Ext"), 9,
                       !if (!eq(ImmOpStr, "u6_2Ext"), 8,
                       !if (!eq(ImmOpStr, "u6_1Ext"), 7,
                                        /* u6_0Ext */ 6)));
    let hasNewValue = !if (!eq(ImmOpStr, "u6_3Ext"), 0, 1);
    let isPredicatedNew = isPredNew;
    let isPredicatedFalse = isNot;

    let IClass = 0b0100;

    let Inst{27}    = 0b0;
    let Inst{27}    = 0b0;
    let Inst{26}    = isNot;
    let Inst{25}    = isPredNew;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = src2;
    let Inst{13}    = 0b0;
    let Inst{12-11} = src1;
    let Inst{10-5}  = offsetBits;
    let Inst{4-0}   = dst;
  }

let isExtendable = 1, hasSideEffects = 0, addrMode = BaseImmOffset in
multiclass LD_Idxd<string mnemonic, string CextOp, RegisterClass RC,
                   Operand ImmOp, Operand predImmOp, bits<4>MajOp> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_indexed in {
    let isPredicable = 1 in
    def L2_#NAME#_io : T_load_io <mnemonic, RC, MajOp, ImmOp>;

    // Predicated
    def L2_p#NAME#t_io : T_pload_io <mnemonic, RC, MajOp, predImmOp, 0, 0>;
    def L2_p#NAME#f_io : T_pload_io <mnemonic, RC, MajOp, predImmOp, 1, 0>;

    // Predicated new
    def L2_p#NAME#tnew_io : T_pload_io <mnemonic, RC, MajOp, predImmOp, 0, 1>;
    def L2_p#NAME#fnew_io : T_pload_io <mnemonic, RC, MajOp, predImmOp, 1, 1>;
  }
}

let accessSize = ByteAccess in {
  defm loadrb:  LD_Idxd <"memb", "LDrib", IntRegs, s11_0Ext, u6_0Ext, 0b1000>;
  defm loadrub: LD_Idxd <"memub", "LDriub", IntRegs, s11_0Ext, u6_0Ext, 0b1001>;
}

let accessSize = HalfWordAccess, opExtentAlign = 1 in {
  defm loadrh:  LD_Idxd <"memh", "LDrih", IntRegs, s11_1Ext, u6_1Ext, 0b1010>;
  defm loadruh: LD_Idxd <"memuh", "LDriuh", IntRegs, s11_1Ext, u6_1Ext, 0b1011>;
}

let accessSize = WordAccess, opExtentAlign = 2 in
defm loadri: LD_Idxd <"memw", "LDriw", IntRegs, s11_2Ext, u6_2Ext, 0b1100>;

let accessSize = DoubleWordAccess, opExtentAlign = 3 in
defm loadrd: LD_Idxd <"memd", "LDrid", DoubleRegs, s11_3Ext, u6_3Ext, 0b1110>;

let accessSize = HalfWordAccess, opExtentAlign = 1 in {
  def L2_loadbsw2_io:   T_load_io<"membh",  IntRegs, 0b0001, s11_1Ext>;
  def L2_loadbzw2_io:   T_load_io<"memubh", IntRegs, 0b0011, s11_1Ext>;
}

let accessSize = WordAccess, opExtentAlign = 2 in {
  def L2_loadbzw4_io: T_load_io<"memubh", DoubleRegs, 0b0101, s11_2Ext>;
  def L2_loadbsw4_io: T_load_io<"membh",  DoubleRegs, 0b0111, s11_2Ext>;
}

let addrMode = BaseImmOffset, isExtendable = 1, hasSideEffects = 0,
    opExtendable = 3, isExtentSigned = 1  in
class T_loadalign_io <string str, bits<4> MajOp, Operand ImmOp>
  : LDInst<(outs DoubleRegs:$dst),
           (ins DoubleRegs:$src1, IntRegs:$src2, ImmOp:$offset),
  "$dst = "#str#"($src2 + #$offset)", [],
  "$src1 = $dst">, AddrModeRel {
    bits<4> name;
    bits<5> dst;
    bits<5> src2;
    bits<12> offset;
    bits<11> offsetBits;

    let offsetBits = !if (!eq(!cast<string>(ImmOp), "s11_1Ext"), offset{11-1},
                                                  /* s11_0Ext */ offset{10-0});
    let IClass = 0b1001;

    let Inst{27}    = 0b0;
    let Inst{26-25} = offsetBits{10-9};
    let Inst{24-21} = MajOp;
    let Inst{20-16} = src2;
    let Inst{13-5}  = offsetBits{8-0};
    let Inst{4-0}   = dst;
  }

let accessSize = HalfWordAccess, opExtentBits = 12, opExtentAlign = 1 in
def L2_loadalignh_io: T_loadalign_io <"memh_fifo", 0b0010, s11_1Ext>;

let accessSize = ByteAccess, opExtentBits = 11 in
def L2_loadalignb_io: T_loadalign_io <"memb_fifo", 0b0100, s11_0Ext>;

// Patterns to select load-indexed (i.e. load from base+offset).
multiclass Loadx_pat<PatFrag Load, ValueType VT, PatLeaf ImmPred,
                     InstHexagon MI> {
  def: Pat<(VT (Load AddrFI:$fi)), (VT (MI AddrFI:$fi, 0))>;
  def: Pat<(VT (Load (add (i32 AddrFI:$fi), ImmPred:$Off))),
           (VT (MI AddrFI:$fi, imm:$Off))>;
  def: Pat<(VT (Load (orisadd (i32 AddrFI:$fi), ImmPred:$Off))),
           (VT (MI AddrFI:$fi, imm:$Off))>;
  def: Pat<(VT (Load (add (i32 IntRegs:$Rs), ImmPred:$Off))),
           (VT (MI IntRegs:$Rs, imm:$Off))>;
  def: Pat<(VT (Load (i32 IntRegs:$Rs))), (VT (MI IntRegs:$Rs, 0))>;
}

let AddedComplexity = 20 in {
  defm: Loadx_pat<load,           i32, s30_2ImmPred, L2_loadri_io>;
  defm: Loadx_pat<load,           i64, s29_3ImmPred, L2_loadrd_io>;
  defm: Loadx_pat<atomic_load_8 , i32, s32_0ImmPred, L2_loadrub_io>;
  defm: Loadx_pat<atomic_load_16, i32, s31_1ImmPred, L2_loadruh_io>;
  defm: Loadx_pat<atomic_load_32, i32, s30_2ImmPred, L2_loadri_io>;
  defm: Loadx_pat<atomic_load_64, i64, s29_3ImmPred, L2_loadrd_io>;

  defm: Loadx_pat<extloadi1,      i32, s32_0ImmPred, L2_loadrub_io>;
  defm: Loadx_pat<extloadi8,      i32, s32_0ImmPred, L2_loadrub_io>;
  defm: Loadx_pat<extloadi16,     i32, s31_1ImmPred, L2_loadruh_io>;
  defm: Loadx_pat<sextloadi8,     i32, s32_0ImmPred, L2_loadrb_io>;
  defm: Loadx_pat<sextloadi16,    i32, s31_1ImmPred, L2_loadrh_io>;
  defm: Loadx_pat<zextloadi1,     i32, s32_0ImmPred, L2_loadrub_io>;
  defm: Loadx_pat<zextloadi8,     i32, s32_0ImmPred, L2_loadrub_io>;
  defm: Loadx_pat<zextloadi16,    i32, s31_1ImmPred, L2_loadruh_io>;
  // No sextloadi1.
}

// Sign-extending loads of i1 need to replicate the lowest bit throughout
// the 32-bit value. Since the loaded value can only be 0 or 1, 0-v should
// do the trick.
let AddedComplexity = 20 in
def: Pat<(i32 (sextloadi1 (i32 IntRegs:$Rs))),
         (A2_subri 0, (L2_loadrub_io IntRegs:$Rs, 0))>;

//===----------------------------------------------------------------------===//
// Post increment load
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Template class for non-predicated post increment loads with immediate offset.
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, addrMode = PostInc in
class T_load_pi <string mnemonic, RegisterClass RC, Operand ImmOp,
                     bits<4> MajOp >
  : LDInstPI <(outs RC:$dst, IntRegs:$dst2),
  (ins IntRegs:$src1, ImmOp:$offset),
  "$dst = "#mnemonic#"($src1++#$offset)" ,
  [],
  "$src1 = $dst2" > ,
  PredNewRel {
    bits<5> dst;
    bits<5> src1;
    bits<7> offset;
    bits<4> offsetBits;

    string ImmOpStr = !cast<string>(ImmOp);
    let offsetBits = !if (!eq(ImmOpStr, "s4_3Imm"), offset{6-3},
                     !if (!eq(ImmOpStr, "s4_2Imm"), offset{5-2},
                     !if (!eq(ImmOpStr, "s4_1Imm"), offset{4-1},
                                      /* s4_0Imm */ offset{3-0})));
    let hasNewValue = !if (!eq(ImmOpStr, "s4_3Imm"), 0, 1);

    let IClass = 0b1001;

    let Inst{27-25} = 0b101;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = src1;
    let Inst{13-12} = 0b00;
    let Inst{8-5} = offsetBits;
    let Inst{4-0}   = dst;
  }

//===----------------------------------------------------------------------===//
// Template class for predicated post increment loads with immediate offset.
//===----------------------------------------------------------------------===//
let isPredicated = 1, hasSideEffects = 0, addrMode = PostInc in
class T_pload_pi <string mnemonic, RegisterClass RC, Operand ImmOp,
                          bits<4> MajOp, bit isPredNot, bit isPredNew >
  : LDInst <(outs RC:$dst, IntRegs:$dst2),
            (ins PredRegs:$src1, IntRegs:$src2, ImmOp:$offset),
  !if(isPredNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
  ") ")#"$dst = "#mnemonic#"($src2++#$offset)",
  [] ,
  "$src2 = $dst2" > ,
  PredNewRel {
    bits<5> dst;
    bits<2> src1;
    bits<5> src2;
    bits<7> offset;
    bits<4> offsetBits;

    let isPredicatedNew = isPredNew;
    let isPredicatedFalse = isPredNot;

    string ImmOpStr = !cast<string>(ImmOp);
    let offsetBits = !if (!eq(ImmOpStr, "s4_3Imm"), offset{6-3},
                     !if (!eq(ImmOpStr, "s4_2Imm"), offset{5-2},
                     !if (!eq(ImmOpStr, "s4_1Imm"), offset{4-1},
                                      /* s4_0Imm */ offset{3-0})));
    let hasNewValue = !if (!eq(ImmOpStr, "s4_3Imm"), 0, 1);

    let IClass = 0b1001;

    let Inst{27-25} = 0b101;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = src2;
    let Inst{13} = 0b1;
    let Inst{12} = isPredNew;
    let Inst{11} = isPredNot;
    let Inst{10-9} = src1;
    let Inst{8-5}  = offsetBits;
    let Inst{4-0}  = dst;
  }

//===----------------------------------------------------------------------===//
// Multiclass for post increment loads with immediate offset.
//===----------------------------------------------------------------------===//

multiclass LD_PostInc <string mnemonic, string BaseOp, RegisterClass RC,
                       Operand ImmOp, bits<4> MajOp> {
  let BaseOpcode = "POST_"#BaseOp in {
    let isPredicable = 1 in
    def L2_#NAME#_pi : T_load_pi < mnemonic, RC, ImmOp, MajOp>;

    // Predicated
    def L2_p#NAME#t_pi : T_pload_pi < mnemonic, RC, ImmOp, MajOp, 0, 0>;
    def L2_p#NAME#f_pi : T_pload_pi < mnemonic, RC, ImmOp, MajOp, 1, 0>;

    // Predicated new
    def L2_p#NAME#tnew_pi : T_pload_pi < mnemonic, RC, ImmOp, MajOp, 0, 1>;
    def L2_p#NAME#fnew_pi : T_pload_pi < mnemonic, RC, ImmOp, MajOp, 1, 1>;
  }
}

// post increment byte loads with immediate offset
let accessSize = ByteAccess in {
  defm loadrb  : LD_PostInc <"memb",  "LDrib", IntRegs, s4_0Imm, 0b1000>;
  defm loadrub : LD_PostInc <"memub", "LDriub", IntRegs, s4_0Imm, 0b1001>;
}

// post increment halfword loads with immediate offset
let accessSize = HalfWordAccess, opExtentAlign = 1 in {
  defm loadrh  : LD_PostInc <"memh",  "LDrih", IntRegs, s4_1Imm, 0b1010>;
  defm loadruh : LD_PostInc <"memuh", "LDriuh", IntRegs, s4_1Imm, 0b1011>;
}

// post increment word loads with immediate offset
let accessSize = WordAccess, opExtentAlign = 2 in
defm loadri : LD_PostInc <"memw", "LDriw", IntRegs, s4_2Imm, 0b1100>;

// post increment doubleword loads with immediate offset
let accessSize = DoubleWordAccess, opExtentAlign = 3 in
defm loadrd : LD_PostInc <"memd", "LDrid", DoubleRegs, s4_3Imm, 0b1110>;

// Rd=memb[u]h(Rx++#s4:1)
// Rdd=memb[u]h(Rx++#s4:2)
let accessSize = HalfWordAccess, opExtentAlign = 1 in {
  def L2_loadbsw2_pi   : T_load_pi <"membh", IntRegs, s4_1Imm, 0b0001>;
  def L2_loadbzw2_pi   : T_load_pi <"memubh", IntRegs, s4_1Imm, 0b0011>;
}
let accessSize = WordAccess, opExtentAlign = 2, hasNewValue = 0 in {
  def L2_loadbsw4_pi   : T_load_pi <"membh", DoubleRegs, s4_2Imm, 0b0111>;
  def L2_loadbzw4_pi   : T_load_pi <"memubh", DoubleRegs, s4_2Imm, 0b0101>;
}

//===----------------------------------------------------------------------===//
// Template class for post increment fifo loads with immediate offset.
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, addrMode = PostInc in
class T_loadalign_pi <string mnemonic, Operand ImmOp, bits<4> MajOp >
  : LDInstPI <(outs DoubleRegs:$dst, IntRegs:$dst2),
  (ins DoubleRegs:$src1, IntRegs:$src2, ImmOp:$offset),
  "$dst = "#mnemonic#"($src2++#$offset)" ,
  [], "$src2 = $dst2, $src1 = $dst" > ,
  PredNewRel {
    bits<5> dst;
    bits<5> src2;
    bits<5> offset;
    bits<4> offsetBits;

    let offsetBits = !if (!eq(!cast<string>(ImmOp), "s4_1Imm"), offset{4-1},
                                                  /* s4_0Imm */ offset{3-0});
    let IClass = 0b1001;

    let Inst{27-25} = 0b101;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = src2;
    let Inst{13-12} = 0b00;
    let Inst{8-5} = offsetBits;
    let Inst{4-0}   = dst;
  }

// Ryy=memh_fifo(Rx++#s4:1)
// Ryy=memb_fifo(Rx++#s4:0)
let accessSize = ByteAccess in
def L2_loadalignb_pi : T_loadalign_pi <"memb_fifo", s4_0Imm, 0b0100>;

let accessSize = HalfWordAccess, opExtentAlign = 1 in
def L2_loadalignh_pi : T_loadalign_pi <"memh_fifo", s4_1Imm, 0b0010>;

//===----------------------------------------------------------------------===//
// Template class for post increment loads with register offset.
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, addrMode = PostInc in
class T_load_pr <string mnemonic, RegisterClass RC, bits<4> MajOp,
                       MemAccessSize AccessSz>
  : LDInstPI <(outs RC:$dst, IntRegs:$_dst_),
              (ins IntRegs:$src1, ModRegs:$src2),
  "$dst = "#mnemonic#"($src1++$src2)" ,
  [], "$src1 = $_dst_" > {
    bits<5> dst;
    bits<5> src1;
    bits<1> src2;

    let accessSize = AccessSz;
    let IClass = 0b1001;

    let Inst{27-25} = 0b110;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = src1;
    let Inst{13}    = src2;
    let Inst{12}    = 0b0;
    let Inst{7}     = 0b0;
    let Inst{4-0}   = dst;
  }

let hasNewValue = 1 in {
  def L2_loadrb_pr  : T_load_pr <"memb",  IntRegs, 0b1000, ByteAccess>;
  def L2_loadrub_pr : T_load_pr <"memub", IntRegs, 0b1001, ByteAccess>;
  def L2_loadrh_pr  : T_load_pr <"memh",  IntRegs, 0b1010, HalfWordAccess>;
  def L2_loadruh_pr : T_load_pr <"memuh", IntRegs, 0b1011, HalfWordAccess>;
  def L2_loadri_pr  : T_load_pr <"memw",  IntRegs, 0b1100, WordAccess>;

  def L2_loadbzw2_pr : T_load_pr <"memubh", IntRegs, 0b0011, HalfWordAccess>;
}

def L2_loadrd_pr   : T_load_pr <"memd", DoubleRegs, 0b1110, DoubleWordAccess>;
def L2_loadbzw4_pr : T_load_pr <"memubh", DoubleRegs, 0b0101, WordAccess>;

// Load predicate.
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in
def LDriw_pred : LDInst<(outs PredRegs:$dst),
                        (ins IntRegs:$addr, s11_2Ext:$off),
                        ".error \"should not emit\"", []>;
// Load modifier.
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in
def LDriw_mod : LDInst<(outs ModRegs:$dst),
                        (ins IntRegs:$addr, s11_2Ext:$off),
                        ".error \"should not emit\"", []>;

let Defs = [R29, R30, R31], Uses = [R30], hasSideEffects = 0 in
  def L2_deallocframe : LDInst<(outs), (ins),
                     "deallocframe",
                     []> {
    let IClass = 0b1001;

    let Inst{27-16} = 0b000000011110;
    let Inst{13} = 0b0;
    let Inst{4-0} = 0b11110;
}

// Load / Post increment circular addressing mode.
let Uses = [CS], hasSideEffects = 0, addrMode = PostInc in
class T_load_pcr<string mnemonic, RegisterClass RC, bits<4> MajOp>
  : LDInst <(outs RC:$dst, IntRegs:$_dst_),
            (ins IntRegs:$Rz, ModRegs:$Mu),
  "$dst = "#mnemonic#"($Rz ++ I:circ($Mu))", [],
  "$Rz = $_dst_" > {
    bits<5> dst;
    bits<5> Rz;
    bit Mu;

    let hasNewValue = !if (!eq(!cast<string>(RC), "DoubleRegs"), 0, 1);
    let IClass = 0b1001;

    let Inst{27-25} = 0b100;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = Rz;
    let Inst{13} = Mu;
    let Inst{12} = 0b0;
    let Inst{9} = 0b1;
    let Inst{7} = 0b0;
    let Inst{4-0} = dst;
 }

let accessSize = ByteAccess in {
  def L2_loadrb_pcr  : T_load_pcr <"memb",  IntRegs, 0b1000>;
  def L2_loadrub_pcr : T_load_pcr <"memub", IntRegs, 0b1001>;
}

let accessSize = HalfWordAccess in {
  def L2_loadrh_pcr   : T_load_pcr <"memh",   IntRegs, 0b1010>;
  def L2_loadruh_pcr  : T_load_pcr <"memuh",  IntRegs, 0b1011>;
  def L2_loadbsw2_pcr : T_load_pcr <"membh",  IntRegs, 0b0001>;
  def L2_loadbzw2_pcr : T_load_pcr <"memubh", IntRegs, 0b0011>;
}

let accessSize = WordAccess in {
  def  L2_loadri_pcr  : T_load_pcr <"memw", IntRegs, 0b1100>;
  let hasNewValue = 0 in {
    def L2_loadbzw4_pcr : T_load_pcr <"memubh", DoubleRegs, 0b0101>;
    def L2_loadbsw4_pcr : T_load_pcr <"membh",  DoubleRegs, 0b0111>;
  }
}

let accessSize = DoubleWordAccess in
def L2_loadrd_pcr  : T_load_pcr <"memd", DoubleRegs, 0b1110>;

// Load / Post increment circular addressing mode.
let Uses = [CS], hasSideEffects = 0, addrMode = PostInc in
class T_loadalign_pcr<string mnemonic, bits<4> MajOp, MemAccessSize AccessSz >
  : LDInst <(outs DoubleRegs:$dst, IntRegs:$_dst_),
            (ins DoubleRegs:$_src_, IntRegs:$Rz, ModRegs:$Mu),
  "$dst = "#mnemonic#"($Rz ++ I:circ($Mu))", [],
  "$Rz = $_dst_, $dst = $_src_" > {
    bits<5> dst;
    bits<5> Rz;
    bit Mu;

    let accessSize = AccessSz;
    let IClass = 0b1001;

    let Inst{27-25} = 0b100;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = Rz;
    let Inst{13}    = Mu;
    let Inst{12}    = 0b0;
    let Inst{9}     = 0b1;
    let Inst{7}     = 0b0;
    let Inst{4-0}   = dst;
 }

def L2_loadalignb_pcr : T_loadalign_pcr <"memb_fifo", 0b0100, ByteAccess>;
def L2_loadalignh_pcr : T_loadalign_pcr <"memh_fifo", 0b0010, HalfWordAccess>;

//===----------------------------------------------------------------------===//
// Circular loads with immediate offset.
//===----------------------------------------------------------------------===//
let Uses = [CS], mayLoad = 1, hasSideEffects = 0, addrMode = PostInc in
class T_load_pci <string mnemonic, RegisterClass RC,
                  Operand ImmOp, bits<4> MajOp>
  : LDInstPI<(outs RC:$dst, IntRegs:$_dst_),
             (ins IntRegs:$Rz, ImmOp:$offset, ModRegs:$Mu),
  "$dst = "#mnemonic#"($Rz ++ #$offset:circ($Mu))", [],
  "$Rz = $_dst_"> {
    bits<5> dst;
    bits<5> Rz;
    bits<1> Mu;
    bits<7> offset;
    bits<4> offsetBits;

    string ImmOpStr = !cast<string>(ImmOp);
    let hasNewValue = !if (!eq(!cast<string>(RC), "DoubleRegs"), 0, 1);
    let offsetBits = !if (!eq(ImmOpStr, "s4_3Imm"), offset{6-3},
                     !if (!eq(ImmOpStr, "s4_2Imm"), offset{5-2},
                     !if (!eq(ImmOpStr, "s4_1Imm"), offset{4-1},
                                      /* s4_0Imm */ offset{3-0})));
    let IClass      = 0b1001;
    let Inst{27-25} = 0b100;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = Rz;
    let Inst{13}    = Mu;
    let Inst{12}    = 0b0;
    let Inst{9}     = 0b0;
    let Inst{8-5}   = offsetBits;
    let Inst{4-0}   = dst;
  }

// Byte variants of circ load
let accessSize = ByteAccess in {
  def L2_loadrb_pci  : T_load_pci <"memb",  IntRegs, s4_0Imm, 0b1000>;
  def L2_loadrub_pci : T_load_pci <"memub", IntRegs, s4_0Imm, 0b1001>;
}

// Half word variants of circ load
let accessSize = HalfWordAccess in {
  def L2_loadrh_pci   : T_load_pci <"memh",   IntRegs, s4_1Imm, 0b1010>;
  def L2_loadruh_pci  : T_load_pci <"memuh",  IntRegs, s4_1Imm, 0b1011>;
  def L2_loadbzw2_pci : T_load_pci <"memubh", IntRegs, s4_1Imm, 0b0011>;
  def L2_loadbsw2_pci : T_load_pci <"membh",  IntRegs, s4_1Imm, 0b0001>;
}

// Word variants of circ load
let accessSize = WordAccess in
def L2_loadri_pci   : T_load_pci <"memw",   IntRegs,    s4_2Imm, 0b1100>;

let accessSize = WordAccess, hasNewValue = 0 in {
  def L2_loadbzw4_pci : T_load_pci <"memubh", DoubleRegs, s4_2Imm, 0b0101>;
  def L2_loadbsw4_pci : T_load_pci <"membh",  DoubleRegs, s4_2Imm, 0b0111>;
}

let accessSize = DoubleWordAccess, hasNewValue = 0 in
def L2_loadrd_pci : T_load_pci <"memd", DoubleRegs, s4_3Imm, 0b1110>;


// TODO: memb_fifo and memh_fifo must take destination register as input.
// One-off circ loads - not enough in common to break into a class.
let accessSize = ByteAccess in
def L2_loadalignb_pci : T_load_pci <"memb_fifo", DoubleRegs, s4_0Imm, 0b0100>;

let accessSize = HalfWordAccess, opExtentAlign = 1 in
def L2_loadalignh_pci : T_load_pci <"memh_fifo", DoubleRegs, s4_1Imm, 0b0010>;

// L[24]_load[wd]_locked: Load word/double with lock.
let isSoloAX = 1 in
class T_load_locked <string mnemonic, RegisterClass RC>
  : LD0Inst <(outs RC:$dst),
             (ins IntRegs:$src),
    "$dst = "#mnemonic#"($src)"> {
    bits<5> dst;
    bits<5> src;
    let IClass = 0b1001;
    let Inst{27-21} = 0b0010000;
    let Inst{20-16} = src;
    let Inst{13-12} = !if (!eq(mnemonic, "memd_locked"), 0b01, 0b00);
    let Inst{5}   = 0;
    let Inst{4-0} = dst;
}
let hasNewValue = 1, accessSize = WordAccess, opNewValue = 0 in
  def L2_loadw_locked : T_load_locked <"memw_locked", IntRegs>;
let accessSize = DoubleWordAccess in
  def L4_loadd_locked : T_load_locked <"memd_locked", DoubleRegs>;

// S[24]_store[wd]_locked: Store word/double conditionally.
let isSoloAX = 1, isPredicateLate = 1 in
class T_store_locked <string mnemonic, RegisterClass RC>
  : ST0Inst <(outs PredRegs:$Pd), (ins IntRegs:$Rs, RC:$Rt),
    mnemonic#"($Rs, $Pd) = $Rt"> {
    bits<2> Pd;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1010;
    let Inst{27-23} = 0b00001;
    let Inst{22} = !if (!eq(mnemonic, "memw_locked"), 0b0, 0b1);
    let Inst{21} = 0b1;
    let Inst{20-16} = Rs;
    let Inst{12-8} = Rt;
    let Inst{1-0} = Pd;
}

let accessSize = WordAccess in
def S2_storew_locked : T_store_locked <"memw_locked", IntRegs>;

let accessSize = DoubleWordAccess in
def S4_stored_locked : T_store_locked <"memd_locked", DoubleRegs>;

//===----------------------------------------------------------------------===//
// Bit-reversed loads with auto-increment register
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, addrMode = PostInc in
class T_load_pbr<string mnemonic, RegisterClass RC,
                            MemAccessSize addrSize, bits<4> majOp>
  : LDInst
    <(outs RC:$dst, IntRegs:$_dst_),
     (ins IntRegs:$Rz, ModRegs:$Mu),
     "$dst = "#mnemonic#"($Rz ++ $Mu:brev)" ,
      [] , "$Rz = $_dst_" > {

      let accessSize = addrSize;

      bits<5> dst;
      bits<5> Rz;
      bits<1> Mu;

      let IClass = 0b1001;

      let Inst{27-25} = 0b111;
      let Inst{24-21} = majOp;
      let Inst{20-16} = Rz;
      let Inst{13} = Mu;
      let Inst{12} = 0b0;
      let Inst{7} = 0b0;
      let Inst{4-0} = dst;
  }

let hasNewValue =1, opNewValue = 0 in {
  def L2_loadrb_pbr   : T_load_pbr <"memb",  IntRegs, ByteAccess, 0b1000>;
  def L2_loadrub_pbr  : T_load_pbr <"memub", IntRegs, ByteAccess, 0b1001>;
  def L2_loadrh_pbr   : T_load_pbr <"memh",  IntRegs, HalfWordAccess, 0b1010>;
  def L2_loadruh_pbr  : T_load_pbr <"memuh", IntRegs, HalfWordAccess, 0b1011>;
  def L2_loadbsw2_pbr : T_load_pbr <"membh", IntRegs, HalfWordAccess, 0b0001>;
  def L2_loadbzw2_pbr : T_load_pbr <"memubh", IntRegs, HalfWordAccess, 0b0011>;
  def L2_loadri_pbr : T_load_pbr <"memw", IntRegs, WordAccess, 0b1100>;
}

def L2_loadbzw4_pbr : T_load_pbr <"memubh", DoubleRegs, WordAccess, 0b0101>;
def L2_loadbsw4_pbr : T_load_pbr <"membh",  DoubleRegs, WordAccess, 0b0111>;
def L2_loadrd_pbr : T_load_pbr <"memd", DoubleRegs, DoubleWordAccess, 0b1110>;

def L2_loadalignb_pbr :T_load_pbr <"memb_fifo", DoubleRegs, ByteAccess, 0b0100>;
def L2_loadalignh_pbr :T_load_pbr <"memh_fifo", DoubleRegs,
                                   HalfWordAccess, 0b0010>;

//===----------------------------------------------------------------------===//
// LD -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/ALU +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTYPE/ALU -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/COMPLEX +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTYPE/COMPLEX -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/MPYH +
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Template Class
// MPYS / Multipy signed/unsigned halfwords
//Rd=mpy[u](Rs.[H|L],Rt.[H|L])[:<<1][:rnd][:sat]
//===----------------------------------------------------------------------===//

let hasNewValue = 1, opNewValue = 0 in
class T_M2_mpy < bits<2> LHbits, bit isSat, bit isRnd,
                 bit hasShift, bit isUnsigned>
  : MInst < (outs IntRegs:$Rd), (ins IntRegs:$Rs, IntRegs:$Rt),
  "$Rd = "#!if(isUnsigned,"mpyu","mpy")#"($Rs."#!if(LHbits{1},"h","l")
                                       #", $Rt."#!if(LHbits{0},"h)","l)")
                                       #!if(hasShift,":<<1","")
                                       #!if(isRnd,":rnd","")
                                       #!if(isSat,":sat",""),
  [], "", M_tc_3x_SLOT23 > {
    bits<5> Rd;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1100;
    let Inst{23} = hasShift;
    let Inst{22} = isUnsigned;
    let Inst{21} = isRnd;
    let Inst{7} = isSat;
    let Inst{6-5} = LHbits;
    let Inst{4-0} = Rd;
    let Inst{20-16} = Rs;
    let Inst{12-8} = Rt;
  }

//Rd=mpy(Rs.[H|L],Rt.[H|L])[:<<1]
def M2_mpy_ll_s1: T_M2_mpy<0b00, 0, 0, 1, 0>;
def M2_mpy_ll_s0: T_M2_mpy<0b00, 0, 0, 0, 0>;
def M2_mpy_lh_s1: T_M2_mpy<0b01, 0, 0, 1, 0>;
def M2_mpy_lh_s0: T_M2_mpy<0b01, 0, 0, 0, 0>;
def M2_mpy_hl_s1: T_M2_mpy<0b10, 0, 0, 1, 0>;
def M2_mpy_hl_s0: T_M2_mpy<0b10, 0, 0, 0, 0>;
def M2_mpy_hh_s1: T_M2_mpy<0b11, 0, 0, 1, 0>;
def M2_mpy_hh_s0: T_M2_mpy<0b11, 0, 0, 0, 0>;

//Rd=mpyu(Rs.[H|L],Rt.[H|L])[:<<1]
def M2_mpyu_ll_s1: T_M2_mpy<0b00, 0, 0, 1, 1>;
def M2_mpyu_ll_s0: T_M2_mpy<0b00, 0, 0, 0, 1>;
def M2_mpyu_lh_s1: T_M2_mpy<0b01, 0, 0, 1, 1>;
def M2_mpyu_lh_s0: T_M2_mpy<0b01, 0, 0, 0, 1>;
def M2_mpyu_hl_s1: T_M2_mpy<0b10, 0, 0, 1, 1>;
def M2_mpyu_hl_s0: T_M2_mpy<0b10, 0, 0, 0, 1>;
def M2_mpyu_hh_s1: T_M2_mpy<0b11, 0, 0, 1, 1>;
def M2_mpyu_hh_s0: T_M2_mpy<0b11, 0, 0, 0, 1>;

//Rd=mpy(Rs.[H|L],Rt.[H|L])[:<<1]:rnd
def M2_mpy_rnd_ll_s1: T_M2_mpy <0b00, 0, 1, 1, 0>;
def M2_mpy_rnd_ll_s0: T_M2_mpy <0b00, 0, 1, 0, 0>;
def M2_mpy_rnd_lh_s1: T_M2_mpy <0b01, 0, 1, 1, 0>;
def M2_mpy_rnd_lh_s0: T_M2_mpy <0b01, 0, 1, 0, 0>;
def M2_mpy_rnd_hl_s1: T_M2_mpy <0b10, 0, 1, 1, 0>;
def M2_mpy_rnd_hl_s0: T_M2_mpy <0b10, 0, 1, 0, 0>;
def M2_mpy_rnd_hh_s1: T_M2_mpy <0b11, 0, 1, 1, 0>;
def M2_mpy_rnd_hh_s0: T_M2_mpy <0b11, 0, 1, 0, 0>;

//Rd=mpy(Rs.[H|L],Rt.[H|L])[:<<1][:sat]
//Rd=mpy(Rs.[H|L],Rt.[H|L])[:<<1][:rnd][:sat]
let Defs = [USR_OVF] in {
  def M2_mpy_sat_ll_s1: T_M2_mpy <0b00, 1, 0, 1, 0>;
  def M2_mpy_sat_ll_s0: T_M2_mpy <0b00, 1, 0, 0, 0>;
  def M2_mpy_sat_lh_s1: T_M2_mpy <0b01, 1, 0, 1, 0>;
  def M2_mpy_sat_lh_s0: T_M2_mpy <0b01, 1, 0, 0, 0>;
  def M2_mpy_sat_hl_s1: T_M2_mpy <0b10, 1, 0, 1, 0>;
  def M2_mpy_sat_hl_s0: T_M2_mpy <0b10, 1, 0, 0, 0>;
  def M2_mpy_sat_hh_s1: T_M2_mpy <0b11, 1, 0, 1, 0>;
  def M2_mpy_sat_hh_s0: T_M2_mpy <0b11, 1, 0, 0, 0>;

  def M2_mpy_sat_rnd_ll_s1: T_M2_mpy <0b00, 1, 1, 1, 0>;
  def M2_mpy_sat_rnd_ll_s0: T_M2_mpy <0b00, 1, 1, 0, 0>;
  def M2_mpy_sat_rnd_lh_s1: T_M2_mpy <0b01, 1, 1, 1, 0>;
  def M2_mpy_sat_rnd_lh_s0: T_M2_mpy <0b01, 1, 1, 0, 0>;
  def M2_mpy_sat_rnd_hl_s1: T_M2_mpy <0b10, 1, 1, 1, 0>;
  def M2_mpy_sat_rnd_hl_s0: T_M2_mpy <0b10, 1, 1, 0, 0>;
  def M2_mpy_sat_rnd_hh_s1: T_M2_mpy <0b11, 1, 1, 1, 0>;
  def M2_mpy_sat_rnd_hh_s0: T_M2_mpy <0b11, 1, 1, 0, 0>;
}

//===----------------------------------------------------------------------===//
// Template Class
// MPYS / Multipy signed/unsigned halfwords and add/subtract the
// result from the accumulator.
//Rx [-+]= mpy[u](Rs.[H|L],Rt.[H|L])[:<<1][:sat]
//===----------------------------------------------------------------------===//

let hasNewValue = 1, opNewValue = 0 in
class T_M2_mpy_acc < bits<2> LHbits, bit isSat, bit isNac,
                 bit hasShift, bit isUnsigned >
  : MInst_acc<(outs IntRegs:$Rx), (ins IntRegs:$dst2, IntRegs:$Rs, IntRegs:$Rt),
  "$Rx "#!if(isNac,"-= ","+= ")#!if(isUnsigned,"mpyu","mpy")
                              #"($Rs."#!if(LHbits{1},"h","l")
                              #", $Rt."#!if(LHbits{0},"h)","l)")
                              #!if(hasShift,":<<1","")
                              #!if(isSat,":sat",""),
  [], "$dst2 = $Rx", M_tc_3x_SLOT23 > {
    bits<5> Rx;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;
    let Inst{27-24} = 0b1110;
    let Inst{23} = hasShift;
    let Inst{22} = isUnsigned;
    let Inst{21} = isNac;
    let Inst{7} = isSat;
    let Inst{6-5} = LHbits;
    let Inst{4-0} = Rx;
    let Inst{20-16} = Rs;
    let Inst{12-8} = Rt;
  }

//Rx += mpy(Rs.[H|L],Rt.[H|L])[:<<1]
def M2_mpy_acc_ll_s1: T_M2_mpy_acc <0b00, 0, 0, 1, 0>;
def M2_mpy_acc_ll_s0: T_M2_mpy_acc <0b00, 0, 0, 0, 0>;
def M2_mpy_acc_lh_s1: T_M2_mpy_acc <0b01, 0, 0, 1, 0>;
def M2_mpy_acc_lh_s0: T_M2_mpy_acc <0b01, 0, 0, 0, 0>;
def M2_mpy_acc_hl_s1: T_M2_mpy_acc <0b10, 0, 0, 1, 0>;
def M2_mpy_acc_hl_s0: T_M2_mpy_acc <0b10, 0, 0, 0, 0>;
def M2_mpy_acc_hh_s1: T_M2_mpy_acc <0b11, 0, 0, 1, 0>;
def M2_mpy_acc_hh_s0: T_M2_mpy_acc <0b11, 0, 0, 0, 0>;

//Rx += mpyu(Rs.[H|L],Rt.[H|L])[:<<1]
def M2_mpyu_acc_ll_s1: T_M2_mpy_acc <0b00, 0, 0, 1, 1>;
def M2_mpyu_acc_ll_s0: T_M2_mpy_acc <0b00, 0, 0, 0, 1>;
def M2_mpyu_acc_lh_s1: T_M2_mpy_acc <0b01, 0, 0, 1, 1>;
def M2_mpyu_acc_lh_s0: T_M2_mpy_acc <0b01, 0, 0, 0, 1>;
def M2_mpyu_acc_hl_s1: T_M2_mpy_acc <0b10, 0, 0, 1, 1>;
def M2_mpyu_acc_hl_s0: T_M2_mpy_acc <0b10, 0, 0, 0, 1>;
def M2_mpyu_acc_hh_s1: T_M2_mpy_acc <0b11, 0, 0, 1, 1>;
def M2_mpyu_acc_hh_s0: T_M2_mpy_acc <0b11, 0, 0, 0, 1>;

//Rx -= mpy(Rs.[H|L],Rt.[H|L])[:<<1]
def M2_mpy_nac_ll_s1: T_M2_mpy_acc <0b00, 0, 1, 1, 0>;
def M2_mpy_nac_ll_s0: T_M2_mpy_acc <0b00, 0, 1, 0, 0>;
def M2_mpy_nac_lh_s1: T_M2_mpy_acc <0b01, 0, 1, 1, 0>;
def M2_mpy_nac_lh_s0: T_M2_mpy_acc <0b01, 0, 1, 0, 0>;
def M2_mpy_nac_hl_s1: T_M2_mpy_acc <0b10, 0, 1, 1, 0>;
def M2_mpy_nac_hl_s0: T_M2_mpy_acc <0b10, 0, 1, 0, 0>;
def M2_mpy_nac_hh_s1: T_M2_mpy_acc <0b11, 0, 1, 1, 0>;
def M2_mpy_nac_hh_s0: T_M2_mpy_acc <0b11, 0, 1, 0, 0>;

//Rx -= mpyu(Rs.[H|L],Rt.[H|L])[:<<1]
def M2_mpyu_nac_ll_s1: T_M2_mpy_acc <0b00, 0, 1, 1, 1>;
def M2_mpyu_nac_ll_s0: T_M2_mpy_acc <0b00, 0, 1, 0, 1>;
def M2_mpyu_nac_lh_s1: T_M2_mpy_acc <0b01, 0, 1, 1, 1>;
def M2_mpyu_nac_lh_s0: T_M2_mpy_acc <0b01, 0, 1, 0, 1>;
def M2_mpyu_nac_hl_s1: T_M2_mpy_acc <0b10, 0, 1, 1, 1>;
def M2_mpyu_nac_hl_s0: T_M2_mpy_acc <0b10, 0, 1, 0, 1>;
def M2_mpyu_nac_hh_s1: T_M2_mpy_acc <0b11, 0, 1, 1, 1>;
def M2_mpyu_nac_hh_s0: T_M2_mpy_acc <0b11, 0, 1, 0, 1>;

//Rx += mpy(Rs.[H|L],Rt.[H|L])[:<<1]:sat
def M2_mpy_acc_sat_ll_s1: T_M2_mpy_acc <0b00, 1, 0, 1, 0>;
def M2_mpy_acc_sat_ll_s0: T_M2_mpy_acc <0b00, 1, 0, 0, 0>;
def M2_mpy_acc_sat_lh_s1: T_M2_mpy_acc <0b01, 1, 0, 1, 0>;
def M2_mpy_acc_sat_lh_s0: T_M2_mpy_acc <0b01, 1, 0, 0, 0>;
def M2_mpy_acc_sat_hl_s1: T_M2_mpy_acc <0b10, 1, 0, 1, 0>;
def M2_mpy_acc_sat_hl_s0: T_M2_mpy_acc <0b10, 1, 0, 0, 0>;
def M2_mpy_acc_sat_hh_s1: T_M2_mpy_acc <0b11, 1, 0, 1, 0>;
def M2_mpy_acc_sat_hh_s0: T_M2_mpy_acc <0b11, 1, 0, 0, 0>;

//Rx -= mpy(Rs.[H|L],Rt.[H|L])[:<<1]:sat
def M2_mpy_nac_sat_ll_s1: T_M2_mpy_acc <0b00, 1, 1, 1, 0>;
def M2_mpy_nac_sat_ll_s0: T_M2_mpy_acc <0b00, 1, 1, 0, 0>;
def M2_mpy_nac_sat_lh_s1: T_M2_mpy_acc <0b01, 1, 1, 1, 0>;
def M2_mpy_nac_sat_lh_s0: T_M2_mpy_acc <0b01, 1, 1, 0, 0>;
def M2_mpy_nac_sat_hl_s1: T_M2_mpy_acc <0b10, 1, 1, 1, 0>;
def M2_mpy_nac_sat_hl_s0: T_M2_mpy_acc <0b10, 1, 1, 0, 0>;
def M2_mpy_nac_sat_hh_s1: T_M2_mpy_acc <0b11, 1, 1, 1, 0>;
def M2_mpy_nac_sat_hh_s0: T_M2_mpy_acc <0b11, 1, 1, 0, 0>;

//===----------------------------------------------------------------------===//
// Template Class
// MPYS / Multipy signed/unsigned halfwords and add/subtract the
// result from the 64-bit destination register.
//Rxx [-+]= mpy[u](Rs.[H|L],Rt.[H|L])[:<<1][:sat]
//===----------------------------------------------------------------------===//

class T_M2_mpyd_acc < bits<2> LHbits, bit isNac, bit hasShift, bit isUnsigned>
  : MInst_acc<(outs DoubleRegs:$Rxx),
              (ins DoubleRegs:$dst2, IntRegs:$Rs, IntRegs:$Rt),
  "$Rxx "#!if(isNac,"-= ","+= ")#!if(isUnsigned,"mpyu","mpy")
                                #"($Rs."#!if(LHbits{1},"h","l")
                                #", $Rt."#!if(LHbits{0},"h)","l)")
                                #!if(hasShift,":<<1",""),
  [], "$dst2 = $Rxx", M_tc_3x_SLOT23 > {
    bits<5> Rxx;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b0110;
    let Inst{23} = hasShift;
    let Inst{22} = isUnsigned;
    let Inst{21} = isNac;
    let Inst{7} = 0;
    let Inst{6-5} = LHbits;
    let Inst{4-0} = Rxx;
    let Inst{20-16} = Rs;
    let Inst{12-8} = Rt;
  }

def M2_mpyd_acc_hh_s0: T_M2_mpyd_acc <0b11, 0, 0, 0>;
def M2_mpyd_acc_hl_s0: T_M2_mpyd_acc <0b10, 0, 0, 0>;
def M2_mpyd_acc_lh_s0: T_M2_mpyd_acc <0b01, 0, 0, 0>;
def M2_mpyd_acc_ll_s0: T_M2_mpyd_acc <0b00, 0, 0, 0>;

def M2_mpyd_acc_hh_s1: T_M2_mpyd_acc <0b11, 0, 1, 0>;
def M2_mpyd_acc_hl_s1: T_M2_mpyd_acc <0b10, 0, 1, 0>;
def M2_mpyd_acc_lh_s1: T_M2_mpyd_acc <0b01, 0, 1, 0>;
def M2_mpyd_acc_ll_s1: T_M2_mpyd_acc <0b00, 0, 1, 0>;

def M2_mpyd_nac_hh_s0: T_M2_mpyd_acc <0b11, 1, 0, 0>;
def M2_mpyd_nac_hl_s0: T_M2_mpyd_acc <0b10, 1, 0, 0>;
def M2_mpyd_nac_lh_s0: T_M2_mpyd_acc <0b01, 1, 0, 0>;
def M2_mpyd_nac_ll_s0: T_M2_mpyd_acc <0b00, 1, 0, 0>;

def M2_mpyd_nac_hh_s1: T_M2_mpyd_acc <0b11, 1, 1, 0>;
def M2_mpyd_nac_hl_s1: T_M2_mpyd_acc <0b10, 1, 1, 0>;
def M2_mpyd_nac_lh_s1: T_M2_mpyd_acc <0b01, 1, 1, 0>;
def M2_mpyd_nac_ll_s1: T_M2_mpyd_acc <0b00, 1, 1, 0>;

def M2_mpyud_acc_hh_s0: T_M2_mpyd_acc <0b11, 0, 0, 1>;
def M2_mpyud_acc_hl_s0: T_M2_mpyd_acc <0b10, 0, 0, 1>;
def M2_mpyud_acc_lh_s0: T_M2_mpyd_acc <0b01, 0, 0, 1>;
def M2_mpyud_acc_ll_s0: T_M2_mpyd_acc <0b00, 0, 0, 1>;

def M2_mpyud_acc_hh_s1: T_M2_mpyd_acc <0b11, 0, 1, 1>;
def M2_mpyud_acc_hl_s1: T_M2_mpyd_acc <0b10, 0, 1, 1>;
def M2_mpyud_acc_lh_s1: T_M2_mpyd_acc <0b01, 0, 1, 1>;
def M2_mpyud_acc_ll_s1: T_M2_mpyd_acc <0b00, 0, 1, 1>;

def M2_mpyud_nac_hh_s0: T_M2_mpyd_acc <0b11, 1, 0, 1>;
def M2_mpyud_nac_hl_s0: T_M2_mpyd_acc <0b10, 1, 0, 1>;
def M2_mpyud_nac_lh_s0: T_M2_mpyd_acc <0b01, 1, 0, 1>;
def M2_mpyud_nac_ll_s0: T_M2_mpyd_acc <0b00, 1, 0, 1>;

def M2_mpyud_nac_hh_s1: T_M2_mpyd_acc <0b11, 1, 1, 1>;
def M2_mpyud_nac_hl_s1: T_M2_mpyd_acc <0b10, 1, 1, 1>;
def M2_mpyud_nac_lh_s1: T_M2_mpyd_acc <0b01, 1, 1, 1>;
def M2_mpyud_nac_ll_s1: T_M2_mpyd_acc <0b00, 1, 1, 1>;

//===----------------------------------------------------------------------===//
// Template Class -- Vector Multipy
// Used for complex multiply real or imaginary, dual multiply and even halfwords
//===----------------------------------------------------------------------===//
class T_M2_vmpy < string opc, bits<3> MajOp, bits<3> MinOp, bit hasShift,
                  bit isRnd, bit isSat >
  : MInst <(outs DoubleRegs:$Rdd), (ins DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Rdd = "#opc#"($Rss, $Rtt)"#!if(hasShift,":<<1","")
                              #!if(isRnd,":rnd","")
                              #!if(isSat,":sat",""),
  [] > {
    bits<5> Rdd;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1000;
    let Inst{23-21} = MajOp;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = Rdd;
    let Inst{20-16} = Rss;
    let Inst{12-8} = Rtt;
  }

// Vector complex multiply imaginary: Rdd=vcmpyi(Rss,Rtt)[:<<1]:sat
let Defs = [USR_OVF] in {
def M2_vcmpy_s1_sat_i: T_M2_vmpy <"vcmpyi", 0b110, 0b110, 1, 0, 1>;
def M2_vcmpy_s0_sat_i: T_M2_vmpy <"vcmpyi", 0b010, 0b110, 0, 0, 1>;

// Vector complex multiply real: Rdd=vcmpyr(Rss,Rtt)[:<<1]:sat
def M2_vcmpy_s1_sat_r: T_M2_vmpy <"vcmpyr", 0b101, 0b110, 1, 0, 1>;
def M2_vcmpy_s0_sat_r: T_M2_vmpy <"vcmpyr", 0b001, 0b110, 0, 0, 1>;

// Vector dual multiply: Rdd=vdmpy(Rss,Rtt)[:<<1]:sat
def M2_vdmpys_s1: T_M2_vmpy <"vdmpy", 0b100, 0b100, 1, 0, 1>;
def M2_vdmpys_s0: T_M2_vmpy <"vdmpy", 0b000, 0b100, 0, 0, 1>;

// Vector multiply even halfwords: Rdd=vmpyeh(Rss,Rtt)[:<<1]:sat
def M2_vmpy2es_s1: T_M2_vmpy <"vmpyeh", 0b100, 0b110, 1, 0, 1>;
def M2_vmpy2es_s0: T_M2_vmpy <"vmpyeh", 0b000, 0b110, 0, 0, 1>;

//Rdd=vmpywoh(Rss,Rtt)[:<<1][:rnd]:sat
def M2_mmpyh_s0:  T_M2_vmpy <"vmpywoh", 0b000, 0b111, 0, 0, 1>;
def M2_mmpyh_s1:  T_M2_vmpy <"vmpywoh", 0b100, 0b111, 1, 0, 1>;
def M2_mmpyh_rs0: T_M2_vmpy <"vmpywoh", 0b001, 0b111, 0, 1, 1>;
def M2_mmpyh_rs1: T_M2_vmpy <"vmpywoh", 0b101, 0b111, 1, 1, 1>;

//Rdd=vmpyweh(Rss,Rtt)[:<<1][:rnd]:sat
def M2_mmpyl_s0:  T_M2_vmpy <"vmpyweh", 0b000, 0b101, 0, 0, 1>;
def M2_mmpyl_s1:  T_M2_vmpy <"vmpyweh", 0b100, 0b101, 1, 0, 1>;
def M2_mmpyl_rs0: T_M2_vmpy <"vmpyweh", 0b001, 0b101, 0, 1, 1>;
def M2_mmpyl_rs1: T_M2_vmpy <"vmpyweh", 0b101, 0b101, 1, 1, 1>;

//Rdd=vmpywouh(Rss,Rtt)[:<<1][:rnd]:sat
def M2_mmpyuh_s0:  T_M2_vmpy <"vmpywouh", 0b010, 0b111, 0, 0, 1>;
def M2_mmpyuh_s1:  T_M2_vmpy <"vmpywouh", 0b110, 0b111, 1, 0, 1>;
def M2_mmpyuh_rs0: T_M2_vmpy <"vmpywouh", 0b011, 0b111, 0, 1, 1>;
def M2_mmpyuh_rs1: T_M2_vmpy <"vmpywouh", 0b111, 0b111, 1, 1, 1>;

//Rdd=vmpyweuh(Rss,Rtt)[:<<1][:rnd]:sat
def M2_mmpyul_s0:  T_M2_vmpy <"vmpyweuh", 0b010, 0b101, 0, 0, 1>;
def M2_mmpyul_s1:  T_M2_vmpy <"vmpyweuh", 0b110, 0b101, 1, 0, 1>;
def M2_mmpyul_rs0: T_M2_vmpy <"vmpyweuh", 0b011, 0b101, 0, 1, 1>;
def M2_mmpyul_rs1: T_M2_vmpy <"vmpyweuh", 0b111, 0b101, 1, 1, 1>;
}

let hasNewValue = 1, opNewValue = 0 in
class T_MType_mpy <string mnemonic, bits<4> RegTyBits, RegisterClass RC,
                   bits<3> MajOp, bits<3> MinOp, bit isSat = 0, bit isRnd = 0,
                   string op2Suffix = "", bit isRaw = 0, bit isHi = 0 >
  : MInst <(outs IntRegs:$dst), (ins RC:$src1, RC:$src2),
  "$dst = "#mnemonic
           #"($src1, $src2"#op2Suffix#")"
           #!if(MajOp{2}, ":<<1", "")
           #!if(isRnd, ":rnd", "")
           #!if(isSat, ":sat", "")
           #!if(isRaw, !if(isHi, ":raw:hi", ":raw:lo"), ""), [] > {
    bits<5> dst;
    bits<5> src1;
    bits<5> src2;

    let IClass = 0b1110;

    let Inst{27-24} = RegTyBits;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = src1;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = src2;
    let Inst{7-5}   = MinOp;
    let Inst{4-0}   = dst;
  }

class T_MType_vrcmpy <string mnemonic, bits<3> MajOp, bits<3> MinOp, bit isHi>
  : T_MType_mpy <mnemonic, 0b1001, DoubleRegs, MajOp, MinOp, 1, 1, "", 1, isHi>;

class T_MType_dd  <string mnemonic, bits<3> MajOp, bits<3> MinOp,
                   bit isSat = 0, bit isRnd = 0 >
  : T_MType_mpy <mnemonic, 0b1001, DoubleRegs, MajOp, MinOp, isSat, isRnd>;

class T_MType_rr1  <string mnemonic, bits<3> MajOp, bits<3> MinOp,
                    bit isSat = 0, bit isRnd = 0 >
  : T_MType_mpy<mnemonic, 0b1101, IntRegs, MajOp, MinOp, isSat, isRnd>;

class T_MType_rr2 <string mnemonic, bits<3> MajOp, bits<3> MinOp,
                   bit isSat = 0, bit isRnd = 0, string op2str = "" >
  : T_MType_mpy<mnemonic, 0b1101, IntRegs, MajOp, MinOp, isSat, isRnd, op2str>;

def M2_vradduh    : T_MType_dd <"vradduh", 0b000, 0b001, 0, 0>;
def M2_vdmpyrs_s0 : T_MType_dd <"vdmpy",   0b000, 0b000, 1, 1>;
def M2_vdmpyrs_s1 : T_MType_dd <"vdmpy",   0b100, 0b000, 1, 1>;

let CextOpcode = "mpyi", InputType = "reg" in
def M2_mpyi    : T_MType_rr1 <"mpyi", 0b000, 0b000>, ImmRegRel;

def M2_mpy_up  : T_MType_rr1 <"mpy",  0b000, 0b001>;
def M2_mpyu_up : T_MType_rr1 <"mpyu", 0b010, 0b001>;

def M2_dpmpyss_rnd_s0 : T_MType_rr1 <"mpy", 0b001, 0b001, 0, 1>;

def M2_vmpy2s_s0pack : T_MType_rr1 <"vmpyh", 0b001, 0b111, 1, 1>;
def M2_vmpy2s_s1pack : T_MType_rr1 <"vmpyh", 0b101, 0b111, 1, 1>;

def M2_hmmpyh_rs1 : T_MType_rr2 <"mpy", 0b101, 0b100, 1, 1, ".h">;
def M2_hmmpyl_rs1 : T_MType_rr2 <"mpy", 0b111, 0b100, 1, 1, ".l">;

def M2_cmpyrs_s0  : T_MType_rr2 <"cmpy", 0b001, 0b110, 1, 1>;
def M2_cmpyrs_s1  : T_MType_rr2 <"cmpy", 0b101, 0b110, 1, 1>;
def M2_cmpyrsc_s0 : T_MType_rr2 <"cmpy", 0b011, 0b110, 1, 1, "*">;
def M2_cmpyrsc_s1 : T_MType_rr2 <"cmpy", 0b111, 0b110, 1, 1, "*">;

// V4 Instructions
def M2_vraddh : T_MType_dd <"vraddh", 0b001, 0b111, 0>;
def M2_mpysu_up : T_MType_rr1 <"mpysu", 0b011, 0b001, 0>;
def M2_mpy_up_s1 : T_MType_rr1 <"mpy", 0b101, 0b010, 0>;
def M2_mpy_up_s1_sat : T_MType_rr1 <"mpy", 0b111, 0b000, 1>;

def M2_hmmpyh_s1 : T_MType_rr2 <"mpy", 0b101, 0b000, 1, 0, ".h">;
def M2_hmmpyl_s1 : T_MType_rr2 <"mpy", 0b101, 0b001, 1, 0, ".l">;

def: Pat<(i32 (mul   I32:$src1, I32:$src2)), (M2_mpyi    I32:$src1, I32:$src2)>;
def: Pat<(i32 (mulhs I32:$src1, I32:$src2)), (M2_mpy_up  I32:$src1, I32:$src2)>;
def: Pat<(i32 (mulhu I32:$src1, I32:$src2)), (M2_mpyu_up I32:$src1, I32:$src2)>;

let hasNewValue = 1, opNewValue = 0 in
class T_MType_mpy_ri <bit isNeg, Operand ImmOp, list<dag> pattern>
  : MInst < (outs IntRegs:$Rd), (ins IntRegs:$Rs, ImmOp:$u8),
  "$Rd ="#!if(isNeg, "- ", "+ ")#"mpyi($Rs, #$u8)" ,
   pattern, "", M_tc_3x_SLOT23> {
    bits<5> Rd;
    bits<5> Rs;
    bits<8> u8;

    let IClass = 0b1110;

    let Inst{27-24} = 0b0000;
    let Inst{23} = isNeg;
    let Inst{13} = 0b0;
    let Inst{4-0} = Rd;
    let Inst{20-16} = Rs;
    let Inst{12-5} = u8;
  }

let isExtendable = 1, opExtentBits = 8, opExtendable = 2 in
def M2_mpysip : T_MType_mpy_ri <0, u8Ext,
                [(set (i32 IntRegs:$Rd), (mul IntRegs:$Rs, u32ImmPred:$u8))]>;

def M2_mpysin :  T_MType_mpy_ri <1, u8Imm,
                [(set (i32 IntRegs:$Rd), (ineg (mul IntRegs:$Rs,
                                                    u8ImmPred:$u8)))]>;

// Assember mapped to M2_mpyi
let isAsmParserOnly = 1 in
def M2_mpyui : MInst<(outs IntRegs:$dst),
                     (ins IntRegs:$src1, IntRegs:$src2),
  "$dst = mpyui($src1, $src2)">;

// Rd=mpyi(Rs,#m9)
// s9 is NOT the same as m9 - but it works.. so far.
// Assembler maps to either Rd=+mpyi(Rs,#u8) or Rd=-mpyi(Rs,#u8)
// depending on the value of m9. See Arch Spec.
let isExtendable = 1, opExtendable = 2, isExtentSigned = 1, opExtentBits = 9,
    CextOpcode = "mpyi", InputType = "imm", hasNewValue = 1,
    isAsmParserOnly = 1 in
def M2_mpysmi : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, s9Ext:$src2),
    "$dst = mpyi($src1, #$src2)",
    [(set (i32 IntRegs:$dst), (mul (i32 IntRegs:$src1),
                                   s32ImmPred:$src2))]>, ImmRegRel;

let hasNewValue = 1, isExtendable = 1,  opExtentBits = 8, opExtendable = 3,
    InputType = "imm" in
class T_MType_acc_ri <string mnemonic, bits<3> MajOp, Operand ImmOp,
                      list<dag> pattern = []>
 : MInst < (outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2, ImmOp:$src3),
  "$dst "#mnemonic#"($src2, #$src3)",
  pattern, "$src1 = $dst", M_tc_2_SLOT23> {
    bits<5> dst;
    bits<5> src2;
    bits<8> src3;

    let IClass = 0b1110;

    let Inst{27-26} = 0b00;
    let Inst{25-23} = MajOp;
    let Inst{20-16} = src2;
    let Inst{13} = 0b0;
    let Inst{12-5} = src3;
    let Inst{4-0} = dst;
  }

let InputType = "reg", hasNewValue = 1 in
class T_MType_acc_rr <string mnemonic, bits<3> MajOp, bits<3> MinOp,
                      bit isSwap = 0, list<dag> pattern = [], bit hasNot = 0,
                      bit isSat = 0, bit isShift = 0>
  : MInst < (outs IntRegs:$dst),
            (ins IntRegs:$src1, IntRegs:$src2, IntRegs:$src3),
  "$dst "#mnemonic#"($src2, "#!if(hasNot, "~$src3)","$src3)")
                          #!if(isShift, ":<<1", "")
                          #!if(isSat, ":sat", ""),
  pattern, "$src1 = $dst", M_tc_2_SLOT23 > {
    bits<5> dst;
    bits<5> src2;
    bits<5> src3;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1111;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = !if(isSwap, src3, src2);
    let Inst{13} = 0b0;
    let Inst{12-8} = !if(isSwap, src2, src3);
    let Inst{7-5} = MinOp;
    let Inst{4-0} = dst;
  }

let CextOpcode = "MPYI_acc", Itinerary = M_tc_3x_SLOT23 in {
  def M2_macsip : T_MType_acc_ri <"+= mpyi", 0b010, u8Ext,
                  [(set (i32 IntRegs:$dst),
                        (add (mul IntRegs:$src2, u32ImmPred:$src3),
                             IntRegs:$src1))]>, ImmRegRel;

  def M2_maci   : T_MType_acc_rr <"+= mpyi", 0b000, 0b000, 0,
                 [(set (i32 IntRegs:$dst),
                       (add (mul IntRegs:$src2, IntRegs:$src3),
                            IntRegs:$src1))]>, ImmRegRel;
}

let CextOpcode = "ADD_acc" in {
  let isExtentSigned = 1 in
  def M2_accii : T_MType_acc_ri <"+= add", 0b100, s8Ext,
                 [(set (i32 IntRegs:$dst),
                       (add (add (i32 IntRegs:$src2), s32ImmPred:$src3),
                            (i32 IntRegs:$src1)))]>, ImmRegRel;

  def M2_acci  : T_MType_acc_rr <"+= add",  0b000, 0b001, 0,
                 [(set (i32 IntRegs:$dst),
                       (add (add (i32 IntRegs:$src2), (i32 IntRegs:$src3)),
                            (i32 IntRegs:$src1)))]>, ImmRegRel;
}

let CextOpcode = "SUB_acc" in {
  let isExtentSigned = 1 in
  def M2_naccii : T_MType_acc_ri <"-= add", 0b101, s8Ext>, ImmRegRel;

  def M2_nacci  : T_MType_acc_rr <"-= add",  0b100, 0b001, 0>, ImmRegRel;
}

let Itinerary = M_tc_3x_SLOT23 in
def M2_macsin : T_MType_acc_ri <"-= mpyi", 0b011, u8Ext>;

def M2_xor_xacc : T_MType_acc_rr < "^= xor", 0b100, 0b011, 0>;
def M2_subacc : T_MType_acc_rr <"+= sub",  0b000, 0b011, 1>;

class T_MType_acc_pat1 <InstHexagon MI, SDNode firstOp, SDNode secOp,
                        PatLeaf ImmPred>
  : Pat <(secOp IntRegs:$src1, (firstOp IntRegs:$src2, ImmPred:$src3)),
         (MI IntRegs:$src1, IntRegs:$src2, ImmPred:$src3)>;

class T_MType_acc_pat2 <InstHexagon MI, SDNode firstOp, SDNode secOp>
  : Pat <(i32 (secOp IntRegs:$src1, (firstOp IntRegs:$src2, IntRegs:$src3))),
         (MI IntRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

def : T_MType_acc_pat2 <M2_xor_xacc, xor, xor>;
def : T_MType_acc_pat1 <M2_macsin, mul, sub, u32ImmPred>;

def : T_MType_acc_pat1 <M2_naccii, add, sub, s32ImmPred>;
def : T_MType_acc_pat2 <M2_nacci, add, sub>;

//===----------------------------------------------------------------------===//
// Template Class -- XType Vector Instructions
//===----------------------------------------------------------------------===//
class T_XTYPE_Vect < string opc, bits<3> MajOp, bits<3> MinOp, bit isConj >
  : MInst <(outs DoubleRegs:$Rdd), (ins DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Rdd = "#opc#"($Rss, $Rtt"#!if(isConj,"*)",")"),
  [] > {
    bits<5> Rdd;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1000;
    let Inst{23-21} = MajOp;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = Rdd;
    let Inst{20-16} = Rss;
    let Inst{12-8} = Rtt;
  }

class T_XTYPE_Vect_acc < string opc, bits<3> MajOp, bits<3> MinOp, bit isConj >
  : MInst <(outs DoubleRegs:$Rdd),
           (ins DoubleRegs:$dst2, DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Rdd += "#opc#"($Rss, $Rtt"#!if(isConj,"*)",")"),
  [], "$dst2 = $Rdd",M_tc_3x_SLOT23 > {
    bits<5> Rdd;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1010;
    let Inst{23-21} = MajOp;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = Rdd;
    let Inst{20-16} = Rss;
    let Inst{12-8} = Rtt;
  }

class T_XTYPE_Vect_diff < bits<3> MajOp, string opc >
  : MInst <(outs DoubleRegs:$Rdd), (ins DoubleRegs:$Rtt, DoubleRegs:$Rss),
  "$Rdd = "#opc#"($Rtt, $Rss)",
  [], "",M_tc_2_SLOT23 > {
    bits<5> Rdd;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1000;
    let Inst{23-21} = MajOp;
    let Inst{7-5} = 0b000;
    let Inst{4-0} = Rdd;
    let Inst{20-16} = Rss;
    let Inst{12-8} = Rtt;
  }

// Vector reduce add unsigned bytes: Rdd32=vrmpybu(Rss32,Rtt32)
def A2_vraddub: T_XTYPE_Vect <"vraddub", 0b010, 0b001, 0>;
def A2_vraddub_acc: T_XTYPE_Vect_acc <"vraddub", 0b010, 0b001, 0>;

// Vector sum of absolute differences unsigned bytes: Rdd=vrsadub(Rss,Rtt)
def A2_vrsadub: T_XTYPE_Vect <"vrsadub", 0b010, 0b010, 0>;
def A2_vrsadub_acc: T_XTYPE_Vect_acc <"vrsadub", 0b010, 0b010, 0>;

// Vector absolute difference: Rdd=vabsdiffh(Rtt,Rss)
def M2_vabsdiffh: T_XTYPE_Vect_diff<0b011, "vabsdiffh">;

// Vector absolute difference words: Rdd=vabsdiffw(Rtt,Rss)
def M2_vabsdiffw: T_XTYPE_Vect_diff<0b001, "vabsdiffw">;

// Vector reduce complex multiply real or imaginary:
// Rdd[+]=vrcmpy[ir](Rss,Rtt[*])
def M2_vrcmpyi_s0:  T_XTYPE_Vect <"vrcmpyi", 0b000, 0b000, 0>;
def M2_vrcmpyi_s0c: T_XTYPE_Vect <"vrcmpyi", 0b010, 0b000, 1>;
def M2_vrcmaci_s0:  T_XTYPE_Vect_acc <"vrcmpyi", 0b000, 0b000, 0>;
def M2_vrcmaci_s0c: T_XTYPE_Vect_acc <"vrcmpyi", 0b010, 0b000, 1>;

def M2_vrcmpyr_s0:  T_XTYPE_Vect <"vrcmpyr", 0b000, 0b001, 0>;
def M2_vrcmpyr_s0c: T_XTYPE_Vect <"vrcmpyr", 0b011, 0b001, 1>;
def M2_vrcmacr_s0:  T_XTYPE_Vect_acc <"vrcmpyr", 0b000, 0b001, 0>;
def M2_vrcmacr_s0c: T_XTYPE_Vect_acc <"vrcmpyr", 0b011, 0b001, 1>;

// Vector reduce halfwords:
// Rdd[+]=vrmpyh(Rss,Rtt)
def M2_vrmpy_s0: T_XTYPE_Vect <"vrmpyh", 0b000, 0b010, 0>;
def M2_vrmac_s0: T_XTYPE_Vect_acc <"vrmpyh", 0b000, 0b010, 0>;

//===----------------------------------------------------------------------===//
// Template Class -- Vector Multipy with accumulation.
// Used for complex multiply real or imaginary, dual multiply and even halfwords
//===----------------------------------------------------------------------===//
let Defs = [USR_OVF] in
class T_M2_vmpy_acc_sat < string opc, bits<3> MajOp, bits<3> MinOp,
                          bit hasShift, bit isRnd >
  : MInst <(outs DoubleRegs:$Rxx),
           (ins DoubleRegs:$dst2, DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Rxx += "#opc#"($Rss, $Rtt)"#!if(hasShift,":<<1","")
                               #!if(isRnd,":rnd","")#":sat",
  [], "$dst2 = $Rxx",M_tc_3x_SLOT23 > {
    bits<5> Rxx;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1010;
    let Inst{23-21} = MajOp;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = Rxx;
    let Inst{20-16} = Rss;
    let Inst{12-8} = Rtt;
  }

class T_M2_vmpy_acc < string opc, bits<3> MajOp, bits<3> MinOp,
                      bit hasShift, bit isRnd >
  : MInst <(outs DoubleRegs:$Rxx),
           (ins DoubleRegs:$dst2, DoubleRegs:$Rss, DoubleRegs:$Rtt),
  "$Rxx += "#opc#"($Rss, $Rtt)"#!if(hasShift,":<<1","")
                               #!if(isRnd,":rnd",""),
  [], "$dst2 = $Rxx",M_tc_3x_SLOT23 > {
    bits<5> Rxx;
    bits<5> Rss;
    bits<5> Rtt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1010;
    let Inst{23-21} = MajOp;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = Rxx;
    let Inst{20-16} = Rss;
    let Inst{12-8} = Rtt;
  }

// Vector multiply word by signed half with accumulation
// Rxx+=vmpyw[eo]h(Rss,Rtt)[:<<1][:rnd]:sat
def M2_mmacls_s1:  T_M2_vmpy_acc_sat <"vmpyweh", 0b100, 0b101, 1, 0>;
def M2_mmacls_s0:  T_M2_vmpy_acc_sat <"vmpyweh", 0b000, 0b101, 0, 0>;
def M2_mmacls_rs1: T_M2_vmpy_acc_sat <"vmpyweh", 0b101, 0b101, 1, 1>;
def M2_mmacls_rs0: T_M2_vmpy_acc_sat <"vmpyweh", 0b001, 0b101, 0, 1>;

def M2_mmachs_s1:  T_M2_vmpy_acc_sat <"vmpywoh", 0b100, 0b111, 1, 0>;
def M2_mmachs_s0:  T_M2_vmpy_acc_sat <"vmpywoh", 0b000, 0b111, 0, 0>;
def M2_mmachs_rs1: T_M2_vmpy_acc_sat <"vmpywoh", 0b101, 0b111, 1, 1>;
def M2_mmachs_rs0: T_M2_vmpy_acc_sat <"vmpywoh", 0b001, 0b111, 0, 1>;

// Vector multiply word by unsigned half with accumulation
// Rxx+=vmpyw[eo]uh(Rss,Rtt)[:<<1][:rnd]:sat
def M2_mmaculs_s1:  T_M2_vmpy_acc_sat <"vmpyweuh", 0b110, 0b101, 1, 0>;
def M2_mmaculs_s0:  T_M2_vmpy_acc_sat <"vmpyweuh", 0b010, 0b101, 0, 0>;
def M2_mmaculs_rs1: T_M2_vmpy_acc_sat <"vmpyweuh", 0b111, 0b101, 1, 1>;
def M2_mmaculs_rs0: T_M2_vmpy_acc_sat <"vmpyweuh", 0b011, 0b101, 0, 1>;

def M2_mmacuhs_s1:  T_M2_vmpy_acc_sat <"vmpywouh", 0b110, 0b111, 1, 0>;
def M2_mmacuhs_s0:  T_M2_vmpy_acc_sat <"vmpywouh", 0b010, 0b111, 0, 0>;
def M2_mmacuhs_rs1: T_M2_vmpy_acc_sat <"vmpywouh", 0b111, 0b111, 1, 1>;
def M2_mmacuhs_rs0: T_M2_vmpy_acc_sat <"vmpywouh", 0b011, 0b111, 0, 1>;

// Vector multiply even halfwords with accumulation
// Rxx+=vmpyeh(Rss,Rtt)[:<<1][:sat]
def M2_vmac2es:    T_M2_vmpy_acc     <"vmpyeh", 0b001, 0b010, 0, 0>;
def M2_vmac2es_s1: T_M2_vmpy_acc_sat <"vmpyeh", 0b100, 0b110, 1, 0>;
def M2_vmac2es_s0: T_M2_vmpy_acc_sat <"vmpyeh", 0b000, 0b110, 0, 0>;

// Vector dual multiply with accumulation
// Rxx+=vdmpy(Rss,Rtt)[:sat]
def M2_vdmacs_s1: T_M2_vmpy_acc_sat <"vdmpy", 0b100, 0b100, 1, 0>;
def M2_vdmacs_s0: T_M2_vmpy_acc_sat <"vdmpy", 0b000, 0b100, 0, 0>;

// Vector complex multiply real or imaginary with accumulation
// Rxx+=vcmpy[ir](Rss,Rtt):sat
def M2_vcmac_s0_sat_r: T_M2_vmpy_acc_sat <"vcmpyr", 0b001, 0b100, 0, 0>;
def M2_vcmac_s0_sat_i: T_M2_vmpy_acc_sat <"vcmpyi", 0b010, 0b100, 0, 0>;

//===----------------------------------------------------------------------===//
// Template Class -- Multiply signed/unsigned halfwords with and without
// saturation and rounding
//===----------------------------------------------------------------------===//
class T_M2_mpyd < bits<2> LHbits, bit isRnd, bit hasShift, bit isUnsigned >
  : MInst < (outs DoubleRegs:$Rdd), (ins IntRegs:$Rs, IntRegs:$Rt),
  "$Rdd = "#!if(isUnsigned,"mpyu","mpy")#"($Rs."#!if(LHbits{1},"h","l")
                                       #", $Rt."#!if(LHbits{0},"h)","l)")
                                       #!if(hasShift,":<<1","")
                                       #!if(isRnd,":rnd",""),
  [] > {
    bits<5> Rdd;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b0100;
    let Inst{23} = hasShift;
    let Inst{22} = isUnsigned;
    let Inst{21} = isRnd;
    let Inst{6-5} = LHbits;
    let Inst{4-0} = Rdd;
    let Inst{20-16} = Rs;
    let Inst{12-8} = Rt;
}

def M2_mpyd_hh_s0: T_M2_mpyd<0b11, 0, 0, 0>;
def M2_mpyd_hl_s0: T_M2_mpyd<0b10, 0, 0, 0>;
def M2_mpyd_lh_s0: T_M2_mpyd<0b01, 0, 0, 0>;
def M2_mpyd_ll_s0: T_M2_mpyd<0b00, 0, 0, 0>;

def M2_mpyd_hh_s1: T_M2_mpyd<0b11, 0, 1, 0>;
def M2_mpyd_hl_s1: T_M2_mpyd<0b10, 0, 1, 0>;
def M2_mpyd_lh_s1: T_M2_mpyd<0b01, 0, 1, 0>;
def M2_mpyd_ll_s1: T_M2_mpyd<0b00, 0, 1, 0>;

def M2_mpyd_rnd_hh_s0: T_M2_mpyd<0b11, 1, 0, 0>;
def M2_mpyd_rnd_hl_s0: T_M2_mpyd<0b10, 1, 0, 0>;
def M2_mpyd_rnd_lh_s0: T_M2_mpyd<0b01, 1, 0, 0>;
def M2_mpyd_rnd_ll_s0: T_M2_mpyd<0b00, 1, 0, 0>;

def M2_mpyd_rnd_hh_s1: T_M2_mpyd<0b11, 1, 1, 0>;
def M2_mpyd_rnd_hl_s1: T_M2_mpyd<0b10, 1, 1, 0>;
def M2_mpyd_rnd_lh_s1: T_M2_mpyd<0b01, 1, 1, 0>;
def M2_mpyd_rnd_ll_s1: T_M2_mpyd<0b00, 1, 1, 0>;

//Rdd=mpyu(Rs.[HL],Rt.[HL])[:<<1]
def M2_mpyud_hh_s0: T_M2_mpyd<0b11, 0, 0, 1>;
def M2_mpyud_hl_s0: T_M2_mpyd<0b10, 0, 0, 1>;
def M2_mpyud_lh_s0: T_M2_mpyd<0b01, 0, 0, 1>;
def M2_mpyud_ll_s0: T_M2_mpyd<0b00, 0, 0, 1>;

def M2_mpyud_hh_s1: T_M2_mpyd<0b11, 0, 1, 1>;
def M2_mpyud_hl_s1: T_M2_mpyd<0b10, 0, 1, 1>;
def M2_mpyud_lh_s1: T_M2_mpyd<0b01, 0, 1, 1>;
def M2_mpyud_ll_s1: T_M2_mpyd<0b00, 0, 1, 1>;

//===----------------------------------------------------------------------===//
// Template Class for xtype mpy:
// Vector multiply
// Complex multiply
// multiply 32X32 and use full result
//===----------------------------------------------------------------------===//
let hasSideEffects = 0 in
class T_XTYPE_mpy64 <string mnemonic, bits<3> MajOp, bits<3> MinOp,
                     bit isSat, bit hasShift, bit isConj>
   : MInst <(outs DoubleRegs:$Rdd),
            (ins IntRegs:$Rs, IntRegs:$Rt),
  "$Rdd = "#mnemonic#"($Rs, $Rt"#!if(isConj,"*)",")")
                                #!if(hasShift,":<<1","")
                                #!if(isSat,":sat",""),
  [] > {
    bits<5> Rdd;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b0101;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{12-8} = Rt;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = Rdd;
  }

//===----------------------------------------------------------------------===//
// Template Class for xtype mpy with accumulation into 64-bit:
// Vector multiply
// Complex multiply
// multiply 32X32 and use full result
//===----------------------------------------------------------------------===//
class T_XTYPE_mpy64_acc <string op1, string op2, bits<3> MajOp, bits<3> MinOp,
                         bit isSat, bit hasShift, bit isConj>
  : MInst <(outs DoubleRegs:$Rxx),
           (ins DoubleRegs:$dst2, IntRegs:$Rs, IntRegs:$Rt),
  "$Rxx "#op2#"= "#op1#"($Rs, $Rt"#!if(isConj,"*)",")")
                                   #!if(hasShift,":<<1","")
                                   #!if(isSat,":sat",""),

  [] , "$dst2 = $Rxx" > {
    bits<5> Rxx;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b0111;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{12-8} = Rt;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = Rxx;
  }

// MPY - Multiply and use full result
// Rdd = mpy[u](Rs,Rt)
def M2_dpmpyss_s0 : T_XTYPE_mpy64 < "mpy", 0b000, 0b000, 0, 0, 0>;
def M2_dpmpyuu_s0 : T_XTYPE_mpy64 < "mpyu", 0b010, 0b000, 0, 0, 0>;

// Rxx[+-]= mpy[u](Rs,Rt)
def M2_dpmpyss_acc_s0 : T_XTYPE_mpy64_acc < "mpy",  "+", 0b000, 0b000, 0, 0, 0>;
def M2_dpmpyss_nac_s0 : T_XTYPE_mpy64_acc < "mpy",  "-", 0b001, 0b000, 0, 0, 0>;
def M2_dpmpyuu_acc_s0 : T_XTYPE_mpy64_acc < "mpyu", "+", 0b010, 0b000, 0, 0, 0>;
def M2_dpmpyuu_nac_s0 : T_XTYPE_mpy64_acc < "mpyu", "-", 0b011, 0b000, 0, 0, 0>;

// Complex multiply real or imaginary
// Rxx=cmpy[ir](Rs,Rt)
def M2_cmpyi_s0 : T_XTYPE_mpy64 < "cmpyi", 0b000, 0b001, 0, 0, 0>;
def M2_cmpyr_s0 : T_XTYPE_mpy64 < "cmpyr", 0b000, 0b010, 0, 0, 0>;

// Rxx+=cmpy[ir](Rs,Rt)
def M2_cmaci_s0 : T_XTYPE_mpy64_acc < "cmpyi", "+", 0b000, 0b001, 0, 0, 0>;
def M2_cmacr_s0 : T_XTYPE_mpy64_acc < "cmpyr", "+", 0b000, 0b010, 0, 0, 0>;

// Complex multiply
// Rdd=cmpy(Rs,Rt)[:<<]:sat
def M2_cmpys_s0 : T_XTYPE_mpy64 < "cmpy", 0b000, 0b110, 1, 0, 0>;
def M2_cmpys_s1 : T_XTYPE_mpy64 < "cmpy", 0b100, 0b110, 1, 1, 0>;

// Rdd=cmpy(Rs,Rt*)[:<<]:sat
def M2_cmpysc_s0 : T_XTYPE_mpy64 < "cmpy", 0b010, 0b110, 1, 0, 1>;
def M2_cmpysc_s1 : T_XTYPE_mpy64 < "cmpy", 0b110, 0b110, 1, 1, 1>;

// Rxx[-+]=cmpy(Rs,Rt)[:<<1]:sat
def M2_cmacs_s0  : T_XTYPE_mpy64_acc < "cmpy", "+", 0b000, 0b110, 1, 0, 0>;
def M2_cnacs_s0  : T_XTYPE_mpy64_acc < "cmpy", "-", 0b000, 0b111, 1, 0, 0>;
def M2_cmacs_s1  : T_XTYPE_mpy64_acc < "cmpy", "+", 0b100, 0b110, 1, 1, 0>;
def M2_cnacs_s1  : T_XTYPE_mpy64_acc < "cmpy", "-", 0b100, 0b111, 1, 1, 0>;

// Rxx[-+]=cmpy(Rs,Rt*)[:<<1]:sat
def M2_cmacsc_s0 : T_XTYPE_mpy64_acc < "cmpy", "+", 0b010, 0b110, 1, 0, 1>;
def M2_cnacsc_s0 : T_XTYPE_mpy64_acc < "cmpy", "-", 0b010, 0b111, 1, 0, 1>;
def M2_cmacsc_s1 : T_XTYPE_mpy64_acc < "cmpy", "+", 0b110, 0b110, 1, 1, 1>;
def M2_cnacsc_s1 : T_XTYPE_mpy64_acc < "cmpy", "-", 0b110, 0b111, 1, 1, 1>;

// Vector multiply halfwords
// Rdd=vmpyh(Rs,Rt)[:<<]:sat
//let Defs = [USR_OVF] in {
  def M2_vmpy2s_s1 : T_XTYPE_mpy64 < "vmpyh", 0b100, 0b101, 1, 1, 0>;
  def M2_vmpy2s_s0 : T_XTYPE_mpy64 < "vmpyh", 0b000, 0b101, 1, 0, 0>;
//}

// Rxx+=vmpyh(Rs,Rt)[:<<1][:sat]
def M2_vmac2     : T_XTYPE_mpy64_acc < "vmpyh", "+", 0b001, 0b001, 0, 0, 0>;
def M2_vmac2s_s1 : T_XTYPE_mpy64_acc < "vmpyh", "+", 0b100, 0b101, 1, 1, 0>;
def M2_vmac2s_s0 : T_XTYPE_mpy64_acc < "vmpyh", "+", 0b000, 0b101, 1, 0, 0>;

def: Pat<(i64 (mul (i64 (anyext (i32 IntRegs:$src1))),
                   (i64 (anyext (i32 IntRegs:$src2))))),
         (M2_dpmpyuu_s0 IntRegs:$src1, IntRegs:$src2)>;

def: Pat<(i64 (mul (i64 (sext (i32 IntRegs:$src1))),
                   (i64 (sext (i32 IntRegs:$src2))))),
         (M2_dpmpyss_s0 IntRegs:$src1, IntRegs:$src2)>;

def: Pat<(i64 (mul (is_sext_i32:$src1),
                   (is_sext_i32:$src2))),
         (M2_dpmpyss_s0 (LoReg DoubleRegs:$src1), (LoReg DoubleRegs:$src2))>;

// Multiply and accumulate, use full result.
// Rxx[+-]=mpy(Rs,Rt)

def: Pat<(i64 (add (i64 DoubleRegs:$src1),
                   (mul (i64 (sext (i32 IntRegs:$src2))),
                        (i64 (sext (i32 IntRegs:$src3)))))),
         (M2_dpmpyss_acc_s0 DoubleRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

def: Pat<(i64 (sub (i64 DoubleRegs:$src1),
                   (mul (i64 (sext (i32 IntRegs:$src2))),
                        (i64 (sext (i32 IntRegs:$src3)))))),
         (M2_dpmpyss_nac_s0 DoubleRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

def: Pat<(i64 (add (i64 DoubleRegs:$src1),
                   (mul (i64 (anyext (i32 IntRegs:$src2))),
                        (i64 (anyext (i32 IntRegs:$src3)))))),
         (M2_dpmpyuu_acc_s0 DoubleRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

def: Pat<(i64 (add (i64 DoubleRegs:$src1),
                   (mul (i64 (zext (i32 IntRegs:$src2))),
                        (i64 (zext (i32 IntRegs:$src3)))))),
         (M2_dpmpyuu_acc_s0 DoubleRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

def: Pat<(i64 (sub (i64 DoubleRegs:$src1),
                   (mul (i64 (anyext (i32 IntRegs:$src2))),
                        (i64 (anyext (i32 IntRegs:$src3)))))),
         (M2_dpmpyuu_nac_s0 DoubleRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

def: Pat<(i64 (sub (i64 DoubleRegs:$src1),
                   (mul (i64 (zext (i32 IntRegs:$src2))),
                        (i64 (zext (i32 IntRegs:$src3)))))),
         (M2_dpmpyuu_nac_s0 DoubleRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

//===----------------------------------------------------------------------===//
// MTYPE/MPYH -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/MPYS +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTYPE/MPYS -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/VB +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTYPE/VB -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MTYPE/VH  +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MTYPE/VH  -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ST +
//===----------------------------------------------------------------------===//
///
// Store doubleword.
//===----------------------------------------------------------------------===//
// Template class for non-predicated post increment stores with immediate offset
//===----------------------------------------------------------------------===//
let isPredicable = 1, hasSideEffects = 0, addrMode = PostInc in
class T_store_pi <string mnemonic, RegisterClass RC, Operand ImmOp,
                 bits<4> MajOp, bit isHalf >
  : STInst <(outs IntRegs:$_dst_),
            (ins IntRegs:$src1, ImmOp:$offset, RC:$src2),
  mnemonic#"($src1++#$offset) = $src2"#!if(isHalf, ".h", ""),
  [], "$src1 = $_dst_" >,
  AddrModeRel {
    bits<5> src1;
    bits<5> src2;
    bits<7> offset;
    bits<4> offsetBits;

    string ImmOpStr = !cast<string>(ImmOp);
    let offsetBits = !if (!eq(ImmOpStr, "s4_3Imm"), offset{6-3},
                     !if (!eq(ImmOpStr, "s4_2Imm"), offset{5-2},
                     !if (!eq(ImmOpStr, "s4_1Imm"), offset{4-1},
                                      /* s4_0Imm */ offset{3-0})));
    // Store upper-half and store doubleword cannot be NV.
    let isNVStorable = !if (!eq(ImmOpStr, "s4_3Imm"), 0, !if(isHalf,0,1));

    let IClass = 0b1010;

    let Inst{27-25} = 0b101;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = src1;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = src2;
    let Inst{7}     = 0b0;
    let Inst{6-3}   = offsetBits;
    let Inst{1}     = 0b0;
  }

//===----------------------------------------------------------------------===//
// Template class for predicated post increment stores with immediate offset
//===----------------------------------------------------------------------===//
let isPredicated = 1, hasSideEffects = 0, addrMode = PostInc in
class T_pstore_pi <string mnemonic, RegisterClass RC, Operand ImmOp,
                   bits<4> MajOp, bit isHalf, bit isPredNot, bit isPredNew>
  : STInst <(outs IntRegs:$_dst_),
            (ins PredRegs:$src1, IntRegs:$src2, ImmOp:$offset, RC:$src3),
  !if(isPredNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
  ") ")#mnemonic#"($src2++#$offset) = $src3"#!if(isHalf, ".h", ""),
  [], "$src2 = $_dst_" >,
  AddrModeRel {
    bits<2> src1;
    bits<5> src2;
    bits<7> offset;
    bits<5> src3;
    bits<4> offsetBits;

    string ImmOpStr = !cast<string>(ImmOp);
    let offsetBits = !if (!eq(ImmOpStr, "s4_3Imm"), offset{6-3},
                     !if (!eq(ImmOpStr, "s4_2Imm"), offset{5-2},
                     !if (!eq(ImmOpStr, "s4_1Imm"), offset{4-1},
                                      /* s4_0Imm */ offset{3-0})));

    // Store upper-half and store doubleword cannot be NV.
    let isNVStorable = !if (!eq(ImmOpStr, "s4_3Imm"), 0, !if(isHalf,0,1));
    let isPredicatedNew = isPredNew;
    let isPredicatedFalse = isPredNot;

    let IClass = 0b1010;

    let Inst{27-25} = 0b101;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = src2;
    let Inst{13} = 0b1;
    let Inst{12-8} = src3;
    let Inst{7} = isPredNew;
    let Inst{6-3} = offsetBits;
    let Inst{2} = isPredNot;
    let Inst{1-0} = src1;
  }

multiclass ST_PostInc<string mnemonic, string BaseOp, RegisterClass RC,
                      Operand ImmOp, bits<4> MajOp, bit isHalf = 0 > {

  let BaseOpcode = "POST_"#BaseOp in {
    def S2_#NAME#_pi : T_store_pi <mnemonic, RC, ImmOp, MajOp, isHalf>;

    // Predicated
    def S2_p#NAME#t_pi : T_pstore_pi <mnemonic, RC, ImmOp, MajOp, isHalf, 0, 0>;
    def S2_p#NAME#f_pi : T_pstore_pi <mnemonic, RC, ImmOp, MajOp, isHalf, 1, 0>;

    // Predicated new
    def S2_p#NAME#tnew_pi : T_pstore_pi <mnemonic, RC, ImmOp, MajOp,
                                          isHalf, 0, 1>;
    def S2_p#NAME#fnew_pi : T_pstore_pi <mnemonic, RC, ImmOp, MajOp,
                                          isHalf, 1, 1>;
  }
}

let accessSize = ByteAccess in
defm storerb: ST_PostInc <"memb", "STrib", IntRegs, s4_0Imm, 0b1000>;

let accessSize = HalfWordAccess in
defm storerh: ST_PostInc <"memh", "STrih", IntRegs, s4_1Imm, 0b1010>;

let accessSize = WordAccess in
defm storeri: ST_PostInc <"memw", "STriw", IntRegs, s4_2Imm, 0b1100>;

let accessSize = DoubleWordAccess in
defm storerd: ST_PostInc <"memd", "STrid", DoubleRegs, s4_3Imm, 0b1110>;

let accessSize = HalfWordAccess, isNVStorable = 0 in
defm storerf: ST_PostInc <"memh", "STrih_H", IntRegs, s4_1Imm, 0b1011, 1>;

class Storepi_pat<PatFrag Store, PatFrag Value, PatFrag Offset,
                  InstHexagon MI>
  : Pat<(Store Value:$src1, I32:$src2, Offset:$offset),
        (MI I32:$src2, imm:$offset, Value:$src1)>;

def: Storepi_pat<post_truncsti8,  I32, s4_0ImmPred, S2_storerb_pi>;
def: Storepi_pat<post_truncsti16, I32, s4_1ImmPred, S2_storerh_pi>;
def: Storepi_pat<post_store,      I32, s4_2ImmPred, S2_storeri_pi>;
def: Storepi_pat<post_store,      I64, s4_3ImmPred, S2_storerd_pi>;

//===----------------------------------------------------------------------===//
// Template class for post increment stores with register offset.
//===----------------------------------------------------------------------===//
class T_store_pr <string mnemonic, RegisterClass RC, bits<3> MajOp,
                     MemAccessSize AccessSz, bit isHalf = 0>
  : STInst <(outs IntRegs:$_dst_),
            (ins IntRegs:$src1, ModRegs:$src2, RC:$src3),
  mnemonic#"($src1++$src2) = $src3"#!if(isHalf, ".h", ""),
  [], "$src1 = $_dst_" > {
    bits<5> src1;
    bits<1> src2;
    bits<5> src3;
    let accessSize = AccessSz;

    // Store upper-half and store doubleword cannot be NV.
    let isNVStorable = !if(!eq(mnemonic,"memd"), 0, !if(isHalf,0,1));

    let IClass = 0b1010;

    let Inst{27-24} = 0b1101;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = src1;
    let Inst{13} = src2;
    let Inst{12-8} = src3;
    let Inst{7} = 0b0;
  }

def S2_storerb_pr : T_store_pr<"memb", IntRegs, 0b000, ByteAccess>;
def S2_storerh_pr : T_store_pr<"memh", IntRegs, 0b010, HalfWordAccess>;
def S2_storeri_pr : T_store_pr<"memw", IntRegs, 0b100, WordAccess>;
def S2_storerd_pr : T_store_pr<"memd", DoubleRegs, 0b110, DoubleWordAccess>;
def S2_storerf_pr : T_store_pr<"memh", IntRegs, 0b011, HalfWordAccess, 1>;

let opExtendable = 1, isExtentSigned = 1, isPredicable = 1 in
class T_store_io <string mnemonic, RegisterClass RC, Operand ImmOp,
                  bits<3> MajOp, bit isH = 0>
  : STInst <(outs),
            (ins IntRegs:$src1, ImmOp:$src2, RC:$src3),
  mnemonic#"($src1+#$src2) = $src3"#!if(isH,".h","")>,
  AddrModeRel, ImmRegRel {
    bits<5> src1;
    bits<14> src2; // Actual address offset
    bits<5> src3;
    bits<11> offsetBits; // Represents offset encoding

    string ImmOpStr = !cast<string>(ImmOp);

    let opExtentBits = !if (!eq(ImmOpStr, "s11_3Ext"), 14,
                       !if (!eq(ImmOpStr, "s11_2Ext"), 13,
                       !if (!eq(ImmOpStr, "s11_1Ext"), 12,
                                        /* s11_0Ext */ 11)));
    let offsetBits = !if (!eq(ImmOpStr, "s11_3Ext"), src2{13-3},
                     !if (!eq(ImmOpStr, "s11_2Ext"), src2{12-2},
                     !if (!eq(ImmOpStr, "s11_1Ext"), src2{11-1},
                                      /* s11_0Ext */ src2{10-0})));
    // Store upper-half and store doubleword cannot be NV.
    let isNVStorable = !if (!eq(mnemonic, "memd"), 0, !if(isH,0,1));
    let IClass = 0b1010;

    let Inst{27} = 0b0;
    let Inst{26-25} = offsetBits{10-9};
    let Inst{24} = 0b1;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = src1;
    let Inst{13} = offsetBits{8};
    let Inst{12-8} = src3;
    let Inst{7-0} = offsetBits{7-0};
  }

let opExtendable = 2, isPredicated = 1 in
class T_pstore_io <string mnemonic, RegisterClass RC, Operand ImmOp,
                   bits<3>MajOp, bit PredNot, bit isPredNew, bit isH = 0>
  : STInst <(outs),
            (ins PredRegs:$src1, IntRegs:$src2, ImmOp:$src3, RC:$src4),
  !if(PredNot, "if (!$src1", "if ($src1")#!if(isPredNew, ".new) ",
  ") ")#mnemonic#"($src2+#$src3) = $src4"#!if(isH,".h",""),
  [],"",V2LDST_tc_st_SLOT01 >,
   AddrModeRel, ImmRegRel {
    bits<2> src1;
    bits<5> src2;
    bits<9> src3; // Actual address offset
    bits<5> src4;
    bits<6> offsetBits; // Represents offset encoding

    let isPredicatedNew = isPredNew;
    let isPredicatedFalse = PredNot;

    string ImmOpStr = !cast<string>(ImmOp);
    let opExtentBits = !if (!eq(ImmOpStr, "u6_3Ext"), 9,
                       !if (!eq(ImmOpStr, "u6_2Ext"), 8,
                       !if (!eq(ImmOpStr, "u6_1Ext"), 7,
                                        /* u6_0Ext */ 6)));
    let offsetBits = !if (!eq(ImmOpStr, "u6_3Ext"), src3{8-3},
                     !if (!eq(ImmOpStr, "u6_2Ext"), src3{7-2},
                     !if (!eq(ImmOpStr, "u6_1Ext"), src3{6-1},
                                      /* u6_0Ext */ src3{5-0})));
    // Store upper-half and store doubleword cannot be NV.
    let isNVStorable = !if (!eq(mnemonic, "memd"), 0, !if(isH,0,1));

    let IClass = 0b0100;

    let Inst{27} = 0b0;
    let Inst{26} = PredNot;
    let Inst{25} = isPredNew;
    let Inst{24} = 0b0;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = src2;
    let Inst{13} = offsetBits{5};
    let Inst{12-8} = src4;
    let Inst{7-3} = offsetBits{4-0};
    let Inst{1-0} = src1;
  }

let isExtendable = 1, hasSideEffects = 0 in
multiclass ST_Idxd<string mnemonic, string CextOp, RegisterClass RC,
                 Operand ImmOp, Operand predImmOp, bits<3> MajOp, bit isH = 0> {
  let CextOpcode = CextOp, BaseOpcode = CextOp#_indexed in {
    def S2_#NAME#_io : T_store_io <mnemonic, RC, ImmOp, MajOp, isH>;

    // Predicated
    def S2_p#NAME#t_io : T_pstore_io<mnemonic, RC, predImmOp, MajOp, 0, 0, isH>;
    def S2_p#NAME#f_io : T_pstore_io<mnemonic, RC, predImmOp, MajOp, 1, 0, isH>;

    // Predicated new
    def S4_p#NAME#tnew_io : T_pstore_io <mnemonic, RC, predImmOp,
                                         MajOp, 0, 1, isH>;
    def S4_p#NAME#fnew_io : T_pstore_io <mnemonic, RC, predImmOp,
                                         MajOp, 1, 1, isH>;
  }
}

let addrMode = BaseImmOffset, InputType = "imm" in {
  let accessSize = ByteAccess in
    defm storerb: ST_Idxd < "memb", "STrib", IntRegs, s11_0Ext, u6_0Ext, 0b000>;

  let accessSize = HalfWordAccess, opExtentAlign = 1 in
    defm storerh: ST_Idxd < "memh", "STrih", IntRegs, s11_1Ext, u6_1Ext, 0b010>;

  let accessSize = WordAccess, opExtentAlign = 2 in
    defm storeri: ST_Idxd < "memw", "STriw", IntRegs, s11_2Ext, u6_2Ext, 0b100>;

  let accessSize = DoubleWordAccess, isNVStorable = 0, opExtentAlign = 3 in
    defm storerd: ST_Idxd < "memd", "STrid", DoubleRegs, s11_3Ext,
                            u6_3Ext, 0b110>;

  let accessSize = HalfWordAccess, opExtentAlign = 1 in
    defm storerf: ST_Idxd < "memh", "STrif", IntRegs, s11_1Ext,
                            u6_1Ext, 0b011, 1>;
}

// Patterns for generating stores, where the address takes different forms:
// - frameindex,
// - frameindex + offset,
// - base + offset,
// - simple (base address without offset).
// These would usually be used together (via Storex_pat defined below), but
// in some cases one may want to apply different properties (such as
// AddedComplexity) to the individual patterns.
class Storex_fi_pat<PatFrag Store, PatFrag Value, InstHexagon MI>
  : Pat<(Store Value:$Rs, AddrFI:$fi), (MI AddrFI:$fi, 0, Value:$Rs)>;
multiclass Storex_fi_add_pat<PatFrag Store, PatFrag Value, PatFrag ImmPred,
                             InstHexagon MI> {
  def: Pat<(Store Value:$Rs, (add (i32 AddrFI:$fi), ImmPred:$Off)),
           (MI AddrFI:$fi, imm:$Off, Value:$Rs)>;
  def: Pat<(Store Value:$Rs, (orisadd (i32 AddrFI:$fi), ImmPred:$Off)),
           (MI AddrFI:$fi, imm:$Off, Value:$Rs)>;
}
multiclass Storex_add_pat<PatFrag Store, PatFrag Value, PatFrag ImmPred,
                          InstHexagon MI> {
  def: Pat<(Store Value:$Rt, (add (i32 IntRegs:$Rs), ImmPred:$Off)),
           (MI IntRegs:$Rs, imm:$Off, Value:$Rt)>;
  def: Pat<(Store Value:$Rt, (orisadd (i32 IntRegs:$Rs), ImmPred:$Off)),
           (MI IntRegs:$Rs, imm:$Off, Value:$Rt)>;
}
class Storex_simple_pat<PatFrag Store, PatFrag Value, InstHexagon MI>
  : Pat<(Store Value:$Rt, (i32 IntRegs:$Rs)),
        (MI IntRegs:$Rs, 0, Value:$Rt)>;

// Patterns for generating stores, where the address takes different forms,
// and where the value being stored is transformed through the value modifier
// ValueMod.  The address forms are same as above.
class Storexm_fi_pat<PatFrag Store, PatFrag Value, PatFrag ValueMod,
                     InstHexagon MI>
  : Pat<(Store Value:$Rs, AddrFI:$fi),
        (MI AddrFI:$fi, 0, (ValueMod Value:$Rs))>;
multiclass Storexm_fi_add_pat<PatFrag Store, PatFrag Value, PatFrag ImmPred,
                              PatFrag ValueMod, InstHexagon MI> {
  def: Pat<(Store Value:$Rs, (add (i32 AddrFI:$fi), ImmPred:$Off)),
           (MI AddrFI:$fi, imm:$Off, (ValueMod Value:$Rs))>;
  def: Pat<(Store Value:$Rs, (orisadd (i32 AddrFI:$fi), ImmPred:$Off)),
           (MI AddrFI:$fi, imm:$Off, (ValueMod Value:$Rs))>;
}
multiclass Storexm_add_pat<PatFrag Store, PatFrag Value, PatFrag ImmPred,
                           PatFrag ValueMod, InstHexagon MI> {
  def: Pat<(Store Value:$Rt, (add (i32 IntRegs:$Rs), ImmPred:$Off)),
           (MI IntRegs:$Rs, imm:$Off, (ValueMod Value:$Rt))>;
  def: Pat<(Store Value:$Rt, (orisadd (i32 IntRegs:$Rs), ImmPred:$Off)),
           (MI IntRegs:$Rs, imm:$Off, (ValueMod Value:$Rt))>;
}
class Storexm_simple_pat<PatFrag Store, PatFrag Value, PatFrag ValueMod,
                         InstHexagon MI>
  : Pat<(Store Value:$Rt, (i32 IntRegs:$Rs)),
        (MI IntRegs:$Rs, 0, (ValueMod Value:$Rt))>;

multiclass Storex_pat<PatFrag Store, PatFrag Value, PatLeaf ImmPred,
                      InstHexagon MI> {
  def:  Storex_fi_pat     <Store, Value,          MI>;
  defm: Storex_fi_add_pat <Store, Value, ImmPred, MI>;
  defm: Storex_add_pat    <Store, Value, ImmPred, MI>;
}

multiclass Storexm_pat<PatFrag Store, PatFrag Value, PatLeaf ImmPred,
                       PatFrag ValueMod, InstHexagon MI> {
  def:  Storexm_fi_pat     <Store, Value,          ValueMod, MI>;
  defm: Storexm_fi_add_pat <Store, Value, ImmPred, ValueMod, MI>;
  defm: Storexm_add_pat    <Store, Value, ImmPred, ValueMod, MI>;
}

// Regular stores in the DAG have two operands: value and address.
// Atomic stores also have two, but they are reversed: address, value.
// To use atomic stores with the patterns, they need to have their operands
// swapped. This relies on the knowledge that the F.Fragment uses names
// "ptr" and "val".
class SwapSt<PatFrag F>
  : PatFrag<(ops node:$val, node:$ptr), F.Fragment, F.PredicateCode,
            F.OperandTransform>;

let AddedComplexity = 20 in {
  defm: Storex_pat<truncstorei8,    I32, s32_0ImmPred, S2_storerb_io>;
  defm: Storex_pat<truncstorei16,   I32, s31_1ImmPred, S2_storerh_io>;
  defm: Storex_pat<store,           I32, s30_2ImmPred, S2_storeri_io>;
  defm: Storex_pat<store,           I64, s29_3ImmPred, S2_storerd_io>;

  defm: Storex_pat<SwapSt<atomic_store_8>,  I32, s32_0ImmPred, S2_storerb_io>;
  defm: Storex_pat<SwapSt<atomic_store_16>, I32, s31_1ImmPred, S2_storerh_io>;
  defm: Storex_pat<SwapSt<atomic_store_32>, I32, s30_2ImmPred, S2_storeri_io>;
  defm: Storex_pat<SwapSt<atomic_store_64>, I64, s29_3ImmPred, S2_storerd_io>;
}

// Simple patterns should be tried with the least priority.
def: Storex_simple_pat<truncstorei8,    I32, S2_storerb_io>;
def: Storex_simple_pat<truncstorei16,   I32, S2_storerh_io>;
def: Storex_simple_pat<store,           I32, S2_storeri_io>;
def: Storex_simple_pat<store,           I64, S2_storerd_io>;

def: Storex_simple_pat<SwapSt<atomic_store_8>,  I32, S2_storerb_io>;
def: Storex_simple_pat<SwapSt<atomic_store_16>, I32, S2_storerh_io>;
def: Storex_simple_pat<SwapSt<atomic_store_32>, I32, S2_storeri_io>;
def: Storex_simple_pat<SwapSt<atomic_store_64>, I64, S2_storerd_io>;

let AddedComplexity = 20 in {
  defm: Storexm_pat<truncstorei8,  I64, s32_0ImmPred, LoReg, S2_storerb_io>;
  defm: Storexm_pat<truncstorei16, I64, s31_1ImmPred, LoReg, S2_storerh_io>;
  defm: Storexm_pat<truncstorei32, I64, s30_2ImmPred, LoReg, S2_storeri_io>;
}

def: Storexm_simple_pat<truncstorei8,  I64, LoReg, S2_storerb_io>;
def: Storexm_simple_pat<truncstorei16, I64, LoReg, S2_storerh_io>;
def: Storexm_simple_pat<truncstorei32, I64, LoReg, S2_storeri_io>;

// Store predicate.
let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in
def STriw_pred : STInst<(outs),
      (ins IntRegs:$addr, s11_2Ext:$off, PredRegs:$src1),
      ".error \"should not emit\"", []>;
// Store modifier.
let isExtendable = 1, opExtendable = 1, isExtentSigned = 1, opExtentBits = 13,
    isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 0 in
def STriw_mod : STInst<(outs),
      (ins IntRegs:$addr, s11_2Ext:$off, ModRegs:$src1),
      ".error \"should not emit\"", []>;

// S2_allocframe: Allocate stack frame.
let Defs = [R29, R30], Uses = [R29, R31, R30],
    hasSideEffects = 0, accessSize = DoubleWordAccess in
def S2_allocframe: ST0Inst <
  (outs), (ins u11_3Imm:$u11_3),
  "allocframe(#$u11_3)" > {
    bits<14> u11_3;

    let IClass = 0b1010;
    let Inst{27-16} = 0b000010011101;
    let Inst{13-11} = 0b000;
    let Inst{10-0} = u11_3{13-3};
  }

// S2_storer[bhwdf]_pci: Store byte/half/word/double.
// S2_storer[bhwdf]_pci -> S2_storerbnew_pci
let Uses = [CS], addrMode = PostInc in
class T_store_pci <string mnemonic, RegisterClass RC,
                         Operand Imm, bits<4>MajOp,
                         MemAccessSize AlignSize, string RegSrc = "Rt">
  : STInst <(outs IntRegs:$_dst_),
  (ins IntRegs:$Rz, Imm:$offset, ModRegs:$Mu, RC:$Rt),
  #mnemonic#"($Rz ++ #$offset:circ($Mu)) = $"#RegSrc#"",
  [] ,
  "$Rz = $_dst_" > {
    bits<5> Rz;
    bits<7> offset;
    bits<1> Mu;
    bits<5> Rt;
    let accessSize = AlignSize;
    let isNVStorable = !if(!eq(mnemonic,"memd"), 0,
                       !if(!eq(RegSrc,"Rt.h"), 0, 1));

    let IClass = 0b1010;
    let Inst{27-25} = 0b100;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = Rz;
    let Inst{13} = Mu;
    let Inst{12-8} = Rt;
    let Inst{7} = 0b0;
    let Inst{6-3} =
      !if (!eq(!cast<string>(AlignSize), "DoubleWordAccess"), offset{6-3},
      !if (!eq(!cast<string>(AlignSize), "WordAccess"),       offset{5-2},
      !if (!eq(!cast<string>(AlignSize), "HalfWordAccess"),   offset{4-1},
                                       /* ByteAccess */       offset{3-0})));
    let Inst{1} = 0b0;
  }

def S2_storerb_pci : T_store_pci<"memb", IntRegs, s4_0Imm, 0b1000,
                                 ByteAccess>;
def S2_storerh_pci : T_store_pci<"memh", IntRegs, s4_1Imm, 0b1010,
                                 HalfWordAccess>;
def S2_storerf_pci : T_store_pci<"memh", IntRegs, s4_1Imm, 0b1011,
                                 HalfWordAccess, "Rt.h">;
def S2_storeri_pci : T_store_pci<"memw", IntRegs, s4_2Imm, 0b1100,
                                 WordAccess>;
def S2_storerd_pci : T_store_pci<"memd", DoubleRegs, s4_3Imm, 0b1110,
                                 DoubleWordAccess>;

let Uses = [CS], isNewValue = 1, mayStore = 1, isNVStore = 1, opNewValue = 4,
    addrMode = PostInc in
class T_storenew_pci <string mnemonic, Operand Imm,
                             bits<2>MajOp, MemAccessSize AlignSize>
  : NVInst < (outs IntRegs:$_dst_),
  (ins IntRegs:$Rz, Imm:$offset, ModRegs:$Mu, IntRegs:$Nt),
  #mnemonic#"($Rz ++ #$offset:circ($Mu)) = $Nt.new",
  [],
  "$Rz = $_dst_"> {
    bits<5> Rz;
    bits<6> offset;
    bits<1> Mu;
    bits<3> Nt;

    let accessSize = AlignSize;

    let IClass = 0b1010;
    let Inst{27-21} = 0b1001101;
    let Inst{20-16} = Rz;
    let Inst{13} = Mu;
    let Inst{12-11} = MajOp;
    let Inst{10-8} = Nt;
    let Inst{7} = 0b0;
    let Inst{6-3} =
      !if (!eq(!cast<string>(AlignSize), "WordAccess"),     offset{5-2},
      !if (!eq(!cast<string>(AlignSize), "HalfWordAccess"), offset{4-1},
                                       /* ByteAccess */     offset{3-0}));
    let Inst{1} = 0b0;
  }

def S2_storerbnew_pci : T_storenew_pci <"memb", s4_0Imm, 0b00, ByteAccess>;
def S2_storerhnew_pci : T_storenew_pci <"memh", s4_1Imm, 0b01, HalfWordAccess>;
def S2_storerinew_pci : T_storenew_pci <"memw", s4_2Imm, 0b10, WordAccess>;

//===----------------------------------------------------------------------===//
// Circular stores with auto-increment register
//===----------------------------------------------------------------------===//
let Uses = [CS], addrMode = PostInc in
class T_store_pcr <string mnemonic, RegisterClass RC, bits<4>MajOp,
                               MemAccessSize AlignSize, string RegSrc = "Rt">
  : STInst <(outs IntRegs:$_dst_),
  (ins IntRegs:$Rz, ModRegs:$Mu, RC:$Rt),
  #mnemonic#"($Rz ++ I:circ($Mu)) = $"#RegSrc#"",
  [],
  "$Rz = $_dst_" > {
    bits<5> Rz;
    bits<1> Mu;
    bits<5> Rt;

    let accessSize = AlignSize;
    let isNVStorable = !if(!eq(mnemonic,"memd"), 0,
                       !if(!eq(RegSrc,"Rt.h"), 0, 1));

    let IClass = 0b1010;
    let Inst{27-25} = 0b100;
    let Inst{24-21} = MajOp;
    let Inst{20-16} = Rz;
    let Inst{13} = Mu;
    let Inst{12-8} = Rt;
    let Inst{7} = 0b0;
    let Inst{1} = 0b1;
  }

def S2_storerb_pcr : T_store_pcr<"memb", IntRegs, 0b1000, ByteAccess>;
def S2_storerh_pcr : T_store_pcr<"memh", IntRegs, 0b1010, HalfWordAccess>;
def S2_storeri_pcr : T_store_pcr<"memw", IntRegs, 0b1100, WordAccess>;
def S2_storerd_pcr : T_store_pcr<"memd", DoubleRegs, 0b1110, DoubleWordAccess>;
def S2_storerf_pcr : T_store_pcr<"memh", IntRegs, 0b1011,
                                 HalfWordAccess, "Rt.h">;

//===----------------------------------------------------------------------===//
// Circular .new stores with auto-increment register
//===----------------------------------------------------------------------===//
let Uses = [CS], isNewValue = 1, mayStore = 1, isNVStore = 1, opNewValue = 3,
    addrMode = PostInc in
class T_storenew_pcr <string mnemonic, bits<2>MajOp,
                                   MemAccessSize AlignSize>
  : NVInst <(outs IntRegs:$_dst_),
  (ins IntRegs:$Rz, ModRegs:$Mu, IntRegs:$Nt),
  #mnemonic#"($Rz ++ I:circ($Mu)) = $Nt.new" ,
  [] ,
  "$Rz = $_dst_"> {
    bits<5> Rz;
    bits<1> Mu;
    bits<3> Nt;

    let accessSize = AlignSize;

    let IClass = 0b1010;
    let Inst{27-21} = 0b1001101;
    let Inst{20-16} = Rz;
    let Inst{13} = Mu;
    let Inst{12-11} = MajOp;
    let Inst{10-8} = Nt;
    let Inst{7} = 0b0;
    let Inst{1} = 0b1;
  }

def S2_storerbnew_pcr : T_storenew_pcr <"memb", 0b00, ByteAccess>;
def S2_storerhnew_pcr : T_storenew_pcr <"memh", 0b01, HalfWordAccess>;
def S2_storerinew_pcr : T_storenew_pcr <"memw", 0b10, WordAccess>;

//===----------------------------------------------------------------------===//
// Bit-reversed stores with auto-increment register
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, addrMode = PostInc in
class T_store_pbr<string mnemonic, RegisterClass RC,
                            MemAccessSize addrSize, bits<3> majOp,
                            bit isHalf = 0>
  : STInst
    <(outs IntRegs:$_dst_),
     (ins IntRegs:$Rz, ModRegs:$Mu, RC:$src),
     #mnemonic#"($Rz ++ $Mu:brev) = $src"#!if (!eq(isHalf, 1), ".h", ""),
     [], "$Rz = $_dst_" > {

      let accessSize = addrSize;

      bits<5> Rz;
      bits<1> Mu;
      bits<5> src;

      let IClass = 0b1010;

      let Inst{27-24} = 0b1111;
      let Inst{23-21} = majOp;
      let Inst{7} = 0b0;
      let Inst{20-16} = Rz;
      let Inst{13} = Mu;
      let Inst{12-8} = src;
    }

let isNVStorable = 1 in {
  let BaseOpcode = "S2_storerb_pbr" in
  def S2_storerb_pbr : T_store_pbr<"memb", IntRegs, ByteAccess,
                                             0b000>, NewValueRel;
  let BaseOpcode = "S2_storerh_pbr" in
  def S2_storerh_pbr : T_store_pbr<"memh", IntRegs, HalfWordAccess,
                                             0b010>, NewValueRel;
  let BaseOpcode = "S2_storeri_pbr" in
  def S2_storeri_pbr : T_store_pbr<"memw", IntRegs, WordAccess,
                                             0b100>, NewValueRel;
}

def S2_storerf_pbr : T_store_pbr<"memh", IntRegs, HalfWordAccess, 0b011, 1>;
def S2_storerd_pbr : T_store_pbr<"memd", DoubleRegs, DoubleWordAccess, 0b110>;

//===----------------------------------------------------------------------===//
// Bit-reversed .new stores with auto-increment register
//===----------------------------------------------------------------------===//
let isNewValue = 1, mayStore = 1, isNVStore = 1, opNewValue = 3,
    hasSideEffects = 0, addrMode = PostInc in
class T_storenew_pbr<string mnemonic, MemAccessSize addrSize, bits<2> majOp>
  : NVInst <(outs IntRegs:$_dst_),
            (ins IntRegs:$Rz, ModRegs:$Mu, IntRegs:$Nt),
     #mnemonic#"($Rz ++ $Mu:brev) = $Nt.new", [],
     "$Rz = $_dst_">, NewValueRel {
    let accessSize = addrSize;
    bits<5> Rz;
    bits<1> Mu;
    bits<3> Nt;

    let IClass = 0b1010;

    let Inst{27-21} = 0b1111101;
    let Inst{12-11} = majOp;
    let Inst{7} = 0b0;
    let Inst{20-16} = Rz;
    let Inst{13} = Mu;
    let Inst{10-8} = Nt;
  }

let BaseOpcode = "S2_storerb_pbr" in
def S2_storerbnew_pbr : T_storenew_pbr<"memb", ByteAccess, 0b00>;

let BaseOpcode = "S2_storerh_pbr" in
def S2_storerhnew_pbr : T_storenew_pbr<"memh", HalfWordAccess, 0b01>;

let BaseOpcode = "S2_storeri_pbr" in
def S2_storerinew_pbr : T_storenew_pbr<"memw", WordAccess, 0b10>;

//===----------------------------------------------------------------------===//
// ST -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Template class for S_2op instructions.
//===----------------------------------------------------------------------===//
let hasSideEffects = 0 in
class T_S2op_1 <string mnemonic, bits<4> RegTyBits, RegisterClass RCOut,
                RegisterClass RCIn, bits<2> MajOp, bits<3> MinOp, bit isSat>
  : SInst <(outs RCOut:$dst), (ins RCIn:$src),
  "$dst = "#mnemonic#"($src)"#!if(isSat, ":sat", ""),
  [], "", S_2op_tc_1_SLOT23 > {
    bits<5> dst;
    bits<5> src;

    let IClass = 0b1000;

    let Inst{27-24} = RegTyBits;
    let Inst{23-22} = MajOp;
    let Inst{21} = 0b0;
    let Inst{20-16} = src;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = dst;
  }

class T_S2op_1_di <string mnemonic, bits<2> MajOp, bits<3> MinOp>
  : T_S2op_1 <mnemonic, 0b0100, DoubleRegs, IntRegs, MajOp, MinOp, 0>;

let hasNewValue = 1 in
class T_S2op_1_id <string mnemonic, bits<2> MajOp, bits<3> MinOp, bit isSat = 0>
  : T_S2op_1 <mnemonic, 0b1000, IntRegs, DoubleRegs, MajOp, MinOp, isSat>;

let hasNewValue = 1 in
class T_S2op_1_ii <string mnemonic, bits<2> MajOp, bits<3> MinOp, bit isSat = 0>
  : T_S2op_1 <mnemonic, 0b1100, IntRegs, IntRegs, MajOp, MinOp, isSat>;

// Vector sign/zero extend
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  def S2_vsxtbh : T_S2op_1_di <"vsxtbh", 0b00, 0b000>;
  def S2_vsxthw : T_S2op_1_di <"vsxthw", 0b00, 0b100>;
  def S2_vzxtbh : T_S2op_1_di <"vzxtbh", 0b00, 0b010>;
  def S2_vzxthw : T_S2op_1_di <"vzxthw", 0b00, 0b110>;
}

// Vector splat bytes/halfwords
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
  def S2_vsplatrb : T_S2op_1_ii <"vsplatb", 0b01, 0b111>;
  def S2_vsplatrh : T_S2op_1_di <"vsplath", 0b01, 0b010>;
}

// Sign extend word to doubleword
def A2_sxtw   : T_S2op_1_di <"sxtw", 0b01, 0b000>;

def: Pat <(i64 (sext I32:$src)), (A2_sxtw I32:$src)>;

// Vector saturate and pack
let Defs = [USR_OVF] in {
  def S2_svsathb  : T_S2op_1_ii <"vsathb", 0b10, 0b000>;
  def S2_svsathub : T_S2op_1_ii <"vsathub", 0b10, 0b010>;
  def S2_vsathb   : T_S2op_1_id <"vsathb", 0b00, 0b110>;
  def S2_vsathub  : T_S2op_1_id <"vsathub", 0b00, 0b000>;
  def S2_vsatwh   : T_S2op_1_id <"vsatwh", 0b00, 0b010>;
  def S2_vsatwuh  : T_S2op_1_id <"vsatwuh", 0b00, 0b100>;
}

// Vector truncate
def S2_vtrunohb : T_S2op_1_id <"vtrunohb", 0b10, 0b000>;
def S2_vtrunehb : T_S2op_1_id <"vtrunehb", 0b10, 0b010>;

// Swizzle the bytes of a word
def A2_swiz : T_S2op_1_ii <"swiz", 0b10, 0b111>;

// Saturate
let Defs = [USR_OVF] in {
  def A2_sat   : T_S2op_1_id <"sat", 0b11, 0b000>;
  def A2_satb  : T_S2op_1_ii <"satb", 0b11, 0b111>;
  def A2_satub : T_S2op_1_ii <"satub", 0b11, 0b110>;
  def A2_sath  : T_S2op_1_ii <"sath", 0b11, 0b100>;
  def A2_satuh : T_S2op_1_ii <"satuh", 0b11, 0b101>;
  def A2_roundsat : T_S2op_1_id <"round", 0b11, 0b001, 0b1>;
}

let Itinerary = S_2op_tc_2_SLOT23 in {
  // Vector round and pack
  def S2_vrndpackwh   : T_S2op_1_id <"vrndwh", 0b10, 0b100>;

  let Defs = [USR_OVF] in
  def S2_vrndpackwhs  : T_S2op_1_id <"vrndwh", 0b10, 0b110, 1>;

  // Bit reverse
  def S2_brev : T_S2op_1_ii <"brev", 0b01, 0b110>;

  // Absolute value word
  def A2_abs    : T_S2op_1_ii <"abs", 0b10, 0b100>;

  let Defs = [USR_OVF] in
  def A2_abssat : T_S2op_1_ii <"abs", 0b10, 0b101, 1>;

  // Negate with saturation
  let Defs = [USR_OVF] in
  def A2_negsat : T_S2op_1_ii <"neg", 0b10, 0b110, 1>;
}

def: Pat<(i32 (select (i1 (setlt (i32 IntRegs:$src), 0)),
                      (i32 (sub 0, (i32 IntRegs:$src))),
                      (i32 IntRegs:$src))),
         (A2_abs IntRegs:$src)>;

let AddedComplexity = 50 in
def: Pat<(i32 (xor (add (sra (i32 IntRegs:$src), (i32 31)),
                        (i32 IntRegs:$src)),
                   (sra (i32 IntRegs:$src), (i32 31)))),
         (A2_abs IntRegs:$src)>;

class T_S2op_2 <string mnemonic, bits<4> RegTyBits, RegisterClass RCOut,
                RegisterClass RCIn, bits<3> MajOp, bits<3> MinOp,
                bit isSat, bit isRnd, list<dag> pattern = []>
  : SInst <(outs RCOut:$dst),
  (ins RCIn:$src, u5Imm:$u5),
  "$dst = "#mnemonic#"($src, #$u5)"#!if(isSat, ":sat", "")
                                   #!if(isRnd, ":rnd", ""),
  pattern, "", S_2op_tc_2_SLOT23> {
    bits<5> dst;
    bits<5> src;
    bits<5> u5;

    let IClass = 0b1000;

    let Inst{27-24} = RegTyBits;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = src;
    let Inst{13} = 0b0;
    let Inst{12-8} = u5;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = dst;
  }

class T_S2op_2_di <string mnemonic, bits<3> MajOp, bits<3> MinOp>
  : T_S2op_2 <mnemonic, 0b1000, DoubleRegs, IntRegs, MajOp, MinOp, 0, 0>;

let hasNewValue = 1 in
class T_S2op_2_id <string mnemonic, bits<3> MajOp, bits<3> MinOp>
  : T_S2op_2 <mnemonic, 0b1000, IntRegs, DoubleRegs, MajOp, MinOp, 0, 0>;

let hasNewValue = 1 in
class T_S2op_2_ii <string mnemonic, bits<3> MajOp, bits<3> MinOp,
                   bit isSat = 0, bit isRnd = 0, list<dag> pattern = []>
  : T_S2op_2 <mnemonic, 0b1100, IntRegs, IntRegs, MajOp, MinOp,
              isSat, isRnd, pattern>;

class T_S2op_shift <string mnemonic, bits<3> MajOp, bits<3> MinOp, SDNode OpNd>
  : T_S2op_2_ii <mnemonic, MajOp, MinOp, 0, 0,
    [(set (i32 IntRegs:$dst), (OpNd (i32 IntRegs:$src),
                                    (u5ImmPred:$u5)))]>;

// Vector arithmetic shift right by immediate with truncate and pack
def S2_asr_i_svw_trun : T_S2op_2_id <"vasrw", 0b110, 0b010>;

// Arithmetic/logical shift right/left by immediate
let Itinerary = S_2op_tc_1_SLOT23 in {
  def S2_asr_i_r : T_S2op_shift <"asr", 0b000, 0b000, sra>;
  def S2_lsr_i_r : T_S2op_shift <"lsr", 0b000, 0b001, srl>;
  def S2_asl_i_r : T_S2op_shift <"asl", 0b000, 0b010, shl>;
}

// Shift left by immediate with saturation
let Defs = [USR_OVF] in
def S2_asl_i_r_sat : T_S2op_2_ii <"asl", 0b010, 0b010, 1>;

// Shift right with round
def S2_asr_i_r_rnd : T_S2op_2_ii <"asr", 0b010, 0b000, 0, 1>;

let isAsmParserOnly = 1 in
def S2_asr_i_r_rnd_goodsyntax
  : SInst <(outs IntRegs:$dst), (ins  IntRegs:$src, u5Imm:$u5),
  "$dst = asrrnd($src, #$u5)",
  [], "", S_2op_tc_1_SLOT23>;

let isAsmParserOnly = 1 in
def A2_not: ALU32_rr<(outs IntRegs:$dst),(ins IntRegs:$src),
  "$dst = not($src)">;

def: Pat<(i32 (sra (i32 (add (i32 (sra I32:$src1, u5ImmPred:$src2)),
                             (i32 1))),
                   (i32 1))),
         (S2_asr_i_r_rnd IntRegs:$src1, u5ImmPred:$src2)>;

class T_S2op_3<string opc, bits<2>MajOp, bits<3>minOp, bits<1> sat = 0>
  : SInst<(outs DoubleRegs:$Rdd), (ins DoubleRegs:$Rss),
           "$Rdd = "#opc#"($Rss)"#!if(!eq(sat, 1),":sat","")> {
  bits<5> Rss;
  bits<5> Rdd;
  let IClass = 0b1000;
  let Inst{27-24} = 0;
  let Inst{23-22} = MajOp;
  let Inst{20-16} = Rss;
  let Inst{7-5} = minOp;
  let Inst{4-0} = Rdd;
}

def A2_absp : T_S2op_3 <"abs", 0b10, 0b110>;
def A2_negp : T_S2op_3 <"neg", 0b10, 0b101>;
def A2_notp : T_S2op_3 <"not", 0b10, 0b100>;

// Innterleave/deinterleave
def S2_interleave   : T_S2op_3 <"interleave",   0b11, 0b101>;
def S2_deinterleave : T_S2op_3 <"deinterleave", 0b11, 0b100>;

// Vector Complex conjugate
def A2_vconj : T_S2op_3 <"vconj", 0b10, 0b111, 1>;

// Vector saturate without pack
def S2_vsathb_nopack  : T_S2op_3 <"vsathb",  0b00, 0b111>;
def S2_vsathub_nopack : T_S2op_3 <"vsathub", 0b00, 0b100>;
def S2_vsatwh_nopack  : T_S2op_3 <"vsatwh",  0b00, 0b110>;
def S2_vsatwuh_nopack : T_S2op_3 <"vsatwuh", 0b00, 0b101>;

// Vector absolute value halfwords with and without saturation
// Rdd64=vabsh(Rss64)[:sat]
def A2_vabsh    : T_S2op_3 <"vabsh", 0b01, 0b100>;
def A2_vabshsat : T_S2op_3 <"vabsh", 0b01, 0b101, 1>;

// Vector absolute value words with and without saturation
def A2_vabsw    : T_S2op_3 <"vabsw", 0b01, 0b110>;
def A2_vabswsat : T_S2op_3 <"vabsw", 0b01, 0b111, 1>;

def : Pat<(not (i64 DoubleRegs:$src1)),
          (A2_notp DoubleRegs:$src1)>;

//===----------------------------------------------------------------------===//
// STYPE/BIT +
//===----------------------------------------------------------------------===//
// Bit count

let hasSideEffects = 0, hasNewValue = 1 in
class T_COUNT_LEADING<string MnOp, bits<3> MajOp, bits<3> MinOp, bit Is32,
                dag Out, dag Inp>
    : SInst<Out, Inp, "$Rd = "#MnOp#"($Rs)", [], "", S_2op_tc_1_SLOT23> {
  bits<5> Rs;
  bits<5> Rd;
  let IClass = 0b1000;
  let Inst{27} = 0b1;
  let Inst{26} = Is32;
  let Inst{25-24} = 0b00;
  let Inst{23-21} = MajOp;
  let Inst{20-16} = Rs;
  let Inst{7-5} = MinOp;
  let Inst{4-0} = Rd;
}

class T_COUNT_LEADING_32<string MnOp, bits<3> MajOp, bits<3> MinOp>
    : T_COUNT_LEADING<MnOp, MajOp, MinOp, 0b1,
                      (outs IntRegs:$Rd), (ins IntRegs:$Rs)>;

class T_COUNT_LEADING_64<string MnOp, bits<3> MajOp, bits<3> MinOp>
    : T_COUNT_LEADING<MnOp, MajOp, MinOp, 0b0,
                      (outs IntRegs:$Rd), (ins DoubleRegs:$Rs)>;

def S2_cl0     : T_COUNT_LEADING_32<"cl0",     0b000, 0b101>;
def S2_cl1     : T_COUNT_LEADING_32<"cl1",     0b000, 0b110>;
def S2_ct0     : T_COUNT_LEADING_32<"ct0",     0b010, 0b100>;
def S2_ct1     : T_COUNT_LEADING_32<"ct1",     0b010, 0b101>;
def S2_cl0p    : T_COUNT_LEADING_64<"cl0",     0b010, 0b010>;
def S2_cl1p    : T_COUNT_LEADING_64<"cl1",     0b010, 0b100>;
def S2_clb     : T_COUNT_LEADING_32<"clb",     0b000, 0b100>;
def S2_clbp    : T_COUNT_LEADING_64<"clb",     0b010, 0b000>;
def S2_clbnorm : T_COUNT_LEADING_32<"normamt", 0b000, 0b111>;

// Count leading zeros.
def: Pat<(i32 (ctlz I32:$Rs)), (S2_cl0 I32:$Rs)>;
def: Pat<(i32 (trunc (ctlz I64:$Rss))), (S2_cl0p I64:$Rss)>;

// Count trailing zeros: 32-bit.
def: Pat<(i32 (cttz I32:$Rs)), (S2_ct0 I32:$Rs)>;

// Count leading ones.
def: Pat<(i32 (ctlz (not I32:$Rs))), (S2_cl1 I32:$Rs)>;
def: Pat<(i32 (trunc (ctlz (not I64:$Rss)))), (S2_cl1p I64:$Rss)>;

// Count trailing ones: 32-bit.
def: Pat<(i32 (cttz (not I32:$Rs))), (S2_ct1 I32:$Rs)>;

// The 64-bit counts leading/trailing are defined in HexagonInstrInfoV4.td.

// Bit set/clear/toggle

let hasSideEffects = 0, hasNewValue = 1 in
class T_SCT_BIT_IMM<string MnOp, bits<3> MinOp>
    : SInst<(outs IntRegs:$Rd), (ins IntRegs:$Rs, u5Imm:$u5),
            "$Rd = "#MnOp#"($Rs, #$u5)", [], "", S_2op_tc_1_SLOT23> {
  bits<5> Rd;
  bits<5> Rs;
  bits<5> u5;
  let IClass = 0b1000;
  let Inst{27-21} = 0b1100110;
  let Inst{20-16} = Rs;
  let Inst{13} = 0b0;
  let Inst{12-8} = u5;
  let Inst{7-5} = MinOp;
  let Inst{4-0} = Rd;
}

let hasSideEffects = 0, hasNewValue = 1 in
class T_SCT_BIT_REG<string MnOp, bits<2> MinOp>
    : SInst<(outs IntRegs:$Rd), (ins IntRegs:$Rs, IntRegs:$Rt),
            "$Rd = "#MnOp#"($Rs, $Rt)", [], "", S_3op_tc_1_SLOT23> {
  bits<5> Rd;
  bits<5> Rs;
  bits<5> Rt;
  let IClass = 0b1100;
  let Inst{27-22} = 0b011010;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{7-6} = MinOp;
  let Inst{4-0} = Rd;
}

def S2_clrbit_i    : T_SCT_BIT_IMM<"clrbit",    0b001>;
def S2_setbit_i    : T_SCT_BIT_IMM<"setbit",    0b000>;
def S2_togglebit_i : T_SCT_BIT_IMM<"togglebit", 0b010>;
def S2_clrbit_r    : T_SCT_BIT_REG<"clrbit",    0b01>;
def S2_setbit_r    : T_SCT_BIT_REG<"setbit",    0b00>;
def S2_togglebit_r : T_SCT_BIT_REG<"togglebit", 0b10>;

def: Pat<(i32 (and (i32 IntRegs:$Rs), (not (shl 1, u5ImmPred:$u5)))),
         (S2_clrbit_i IntRegs:$Rs, u5ImmPred:$u5)>;
def: Pat<(i32 (or (i32 IntRegs:$Rs), (shl 1, u5ImmPred:$u5))),
         (S2_setbit_i IntRegs:$Rs, u5ImmPred:$u5)>;
def: Pat<(i32 (xor (i32 IntRegs:$Rs), (shl 1, u5ImmPred:$u5))),
         (S2_togglebit_i IntRegs:$Rs, u5ImmPred:$u5)>;
def: Pat<(i32 (and (i32 IntRegs:$Rs), (not (shl 1, (i32 IntRegs:$Rt))))),
         (S2_clrbit_r IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(i32 (or (i32 IntRegs:$Rs), (shl 1, (i32 IntRegs:$Rt)))),
         (S2_setbit_r IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(i32 (xor (i32 IntRegs:$Rs), (shl 1, (i32 IntRegs:$Rt)))),
         (S2_togglebit_r IntRegs:$Rs, IntRegs:$Rt)>;

// Bit test

let hasSideEffects = 0 in
class T_TEST_BIT_IMM<string MnOp, bits<3> MajOp>
    : SInst<(outs PredRegs:$Pd), (ins IntRegs:$Rs, u5Imm:$u5),
            "$Pd = "#MnOp#"($Rs, #$u5)",
            [], "", S_2op_tc_2early_SLOT23> {
  bits<2> Pd;
  bits<5> Rs;
  bits<5> u5;
  let IClass = 0b1000;
  let Inst{27-24} = 0b0101;
  let Inst{23-21} = MajOp;
  let Inst{20-16} = Rs;
  let Inst{13} = 0;
  let Inst{12-8} = u5;
  let Inst{1-0} = Pd;
}

let hasSideEffects = 0 in
class T_TEST_BIT_REG<string MnOp, bit IsNeg>
    : SInst<(outs PredRegs:$Pd), (ins IntRegs:$Rs, IntRegs:$Rt),
            "$Pd = "#MnOp#"($Rs, $Rt)",
            [], "", S_3op_tc_2early_SLOT23> {
  bits<2> Pd;
  bits<5> Rs;
  bits<5> Rt;
  let IClass = 0b1100;
  let Inst{27-22} = 0b011100;
  let Inst{21} = IsNeg;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{1-0} = Pd;
}

def S2_tstbit_i : T_TEST_BIT_IMM<"tstbit", 0b000>;
def S2_tstbit_r : T_TEST_BIT_REG<"tstbit", 0>;

let AddedComplexity = 20 in { // Complexity greater than cmp reg-imm.
  def: Pat<(i1 (setne (and (shl 1, u5ImmPred:$u5), (i32 IntRegs:$Rs)), 0)),
           (S2_tstbit_i IntRegs:$Rs, u5ImmPred:$u5)>;
  def: Pat<(i1 (setne (and (shl 1, (i32 IntRegs:$Rt)), (i32 IntRegs:$Rs)), 0)),
           (S2_tstbit_r IntRegs:$Rs, IntRegs:$Rt)>;
  def: Pat<(i1 (trunc (i32 IntRegs:$Rs))),
           (S2_tstbit_i IntRegs:$Rs, 0)>;
  def: Pat<(i1 (trunc (i64 DoubleRegs:$Rs))),
           (S2_tstbit_i (LoReg DoubleRegs:$Rs), 0)>;
}

let hasSideEffects = 0 in
class T_TEST_BITS_IMM<string MnOp, bits<2> MajOp, bit IsNeg>
    : SInst<(outs PredRegs:$Pd), (ins IntRegs:$Rs, u6Imm:$u6),
            "$Pd = "#MnOp#"($Rs, #$u6)",
            [], "", S_2op_tc_2early_SLOT23> {
  bits<2> Pd;
  bits<5> Rs;
  bits<6> u6;
  let IClass = 0b1000;
  let Inst{27-24} = 0b0101;
  let Inst{23-22} = MajOp;
  let Inst{21} = IsNeg;
  let Inst{20-16} = Rs;
  let Inst{13-8} = u6;
  let Inst{1-0} = Pd;
}

let hasSideEffects = 0 in
class T_TEST_BITS_REG<string MnOp, bits<2> MajOp, bit IsNeg>
    : SInst<(outs PredRegs:$Pd), (ins IntRegs:$Rs, IntRegs:$Rt),
            "$Pd = "#MnOp#"($Rs, $Rt)",
            [], "", S_3op_tc_2early_SLOT23> {
  bits<2> Pd;
  bits<5> Rs;
  bits<5> Rt;
  let IClass = 0b1100;
  let Inst{27-24} = 0b0111;
  let Inst{23-22} = MajOp;
  let Inst{21} = IsNeg;
  let Inst{20-16} = Rs;
  let Inst{12-8} = Rt;
  let Inst{1-0} = Pd;
}

def C2_bitsclri : T_TEST_BITS_IMM<"bitsclr", 0b10, 0>;
def C2_bitsclr  : T_TEST_BITS_REG<"bitsclr", 0b10, 0>;
def C2_bitsset  : T_TEST_BITS_REG<"bitsset", 0b01, 0>;

let AddedComplexity = 20 in { // Complexity greater than compare reg-imm.
  def: Pat<(i1 (seteq (and (i32 IntRegs:$Rs), u6ImmPred:$u6), 0)),
           (C2_bitsclri IntRegs:$Rs, u6ImmPred:$u6)>;
  def: Pat<(i1 (seteq (and (i32 IntRegs:$Rs), (i32 IntRegs:$Rt)), 0)),
           (C2_bitsclr IntRegs:$Rs, IntRegs:$Rt)>;
}

let AddedComplexity = 10 in   // Complexity greater than compare reg-reg.
def: Pat<(i1 (seteq (and (i32 IntRegs:$Rs), (i32 IntRegs:$Rt)), IntRegs:$Rt)),
         (C2_bitsset IntRegs:$Rs, IntRegs:$Rt)>;

//===----------------------------------------------------------------------===//
// STYPE/BIT -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/COMPLEX +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// STYPE/COMPLEX -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XTYPE/PERM +
//===----------------------------------------------------------------------===//

def: Pat<(or (or (shl (or (shl (i32 (extloadi8 (add (i32 IntRegs:$b), 3))),
                               (i32 8)),
                          (i32 (zextloadi8 (add (i32 IntRegs:$b), 2)))),
                      (i32 16)),
                 (shl (i32 (zextloadi8 (add (i32 IntRegs:$b), 1))), (i32 8))),
             (zextloadi8 (i32 IntRegs:$b))),
         (A2_swiz (L2_loadri_io IntRegs:$b, 0))>;

//===----------------------------------------------------------------------===//
// XTYPE/PERM -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/PRED +
//===----------------------------------------------------------------------===//

// Predicate transfer.
let hasSideEffects = 0, hasNewValue = 1 in
def C2_tfrpr : SInst<(outs IntRegs:$Rd), (ins PredRegs:$Ps),
      "$Rd = $Ps", [], "", S_2op_tc_1_SLOT23> {
  bits<5> Rd;
  bits<2> Ps;

  let IClass = 0b1000;
  let Inst{27-24} = 0b1001;
  let Inst{22} = 0b1;
  let Inst{17-16} = Ps;
  let Inst{4-0} = Rd;
}

// Transfer general register to predicate.
let hasSideEffects = 0 in
def C2_tfrrp: SInst<(outs PredRegs:$Pd), (ins IntRegs:$Rs),
      "$Pd = $Rs", [], "", S_2op_tc_2early_SLOT23> {
  bits<2> Pd;
  bits<5> Rs;

  let IClass = 0b1000;
  let Inst{27-21} = 0b0101010;
  let Inst{20-16} = Rs;
  let Inst{1-0} = Pd;
}

let hasSideEffects = 0, isCodeGenOnly = 1 in
def C2_pxfer_map: SInst<(outs PredRegs:$dst), (ins PredRegs:$src),
     "$dst = $src">;


// Patterns for loads of i1:
def: Pat<(i1 (load AddrFI:$fi)),
         (C2_tfrrp (L2_loadrub_io AddrFI:$fi, 0))>;
def: Pat<(i1 (load (add (i32 IntRegs:$Rs), s32ImmPred:$Off))),
         (C2_tfrrp (L2_loadrub_io IntRegs:$Rs, imm:$Off))>;
def: Pat<(i1 (load (i32 IntRegs:$Rs))),
         (C2_tfrrp (L2_loadrub_io IntRegs:$Rs, 0))>;

def I1toI32: OutPatFrag<(ops node:$Rs),
                        (C2_muxii (i1 $Rs), 1, 0)>;

def I32toI1: OutPatFrag<(ops node:$Rs),
                        (i1 (C2_tfrrp (i32 $Rs)))>;

defm: Storexm_pat<store, I1, s32ImmPred, I1toI32, S2_storerb_io>;
def: Storexm_simple_pat<store, I1, I1toI32, S2_storerb_io>;

//===----------------------------------------------------------------------===//
// STYPE/PRED -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/SHIFT +
//===----------------------------------------------------------------------===//
class S_2OpInstImm<string Mnemonic, bits<3>MajOp, bits<3>MinOp,
                   Operand Imm, list<dag> pattern = [], bit isRnd = 0>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, Imm:$src2),
           "$dst = "#Mnemonic#"($src1, #$src2)"#!if(isRnd, ":rnd", ""),
           pattern> {
  bits<5> src1;
  bits<5> dst;
  let IClass = 0b1000;
  let Inst{27-24} = 0;
  let Inst{23-21} = MajOp;
  let Inst{20-16} = src1;
  let Inst{7-5} = MinOp;
  let Inst{4-0} = dst;
}

class S_2OpInstImmI6<string Mnemonic, SDNode OpNode, bits<3>MinOp>
  : S_2OpInstImm<Mnemonic, 0b000, MinOp, u6Imm,
  [(set (i64 DoubleRegs:$dst), (OpNode (i64 DoubleRegs:$src1),
                                        u6ImmPred:$src2))]> {
  bits<6> src2;
  let Inst{13-8} = src2;
}

// Shift by immediate.
def S2_asr_i_p : S_2OpInstImmI6<"asr", sra, 0b000>;
def S2_asl_i_p : S_2OpInstImmI6<"asl", shl, 0b010>;
def S2_lsr_i_p : S_2OpInstImmI6<"lsr", srl, 0b001>;

// Shift left by small amount and add.
let AddedComplexity = 100, hasNewValue = 1, hasSideEffects = 0 in
def S2_addasl_rrri: SInst <(outs IntRegs:$Rd),
                           (ins IntRegs:$Rt, IntRegs:$Rs, u3Imm:$u3),
  "$Rd = addasl($Rt, $Rs, #$u3)" ,
  [(set (i32 IntRegs:$Rd), (add (i32 IntRegs:$Rt),
                                (shl (i32 IntRegs:$Rs), u3ImmPred:$u3)))],
  "", S_3op_tc_2_SLOT23> {
    bits<5> Rd;
    bits<5> Rt;
    bits<5> Rs;
    bits<3> u3;

    let IClass = 0b1100;

    let Inst{27-21} = 0b0100000;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = Rt;
    let Inst{7-5}   = u3;
    let Inst{4-0}   = Rd;
  }

//===----------------------------------------------------------------------===//
// STYPE/SHIFT -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/VH +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// STYPE/VH -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STYPE/VW +
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// STYPE/VW -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SYSTEM/SUPER +
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SYSTEM/USER +
//===----------------------------------------------------------------------===//
def HexagonBARRIER: SDNode<"HexagonISD::BARRIER", SDTNone, [SDNPHasChain]>;

let hasSideEffects = 1, isSoloAX = 1 in
def Y2_barrier : SYSInst<(outs), (ins),
                     "barrier",
                     [(HexagonBARRIER)],"",ST_tc_st_SLOT0> {
  let Inst{31-28} = 0b1010;
  let Inst{27-21} = 0b1000000;
}

//===----------------------------------------------------------------------===//
// SYSTEM/SUPER -
//===----------------------------------------------------------------------===//

// Generate frameindex addresses. The main reason for the offset operand is
// that every instruction that is allowed to have frame index as an operand
// will then have that operand followed by an immediate operand (the offset).
// This simplifies the frame-index elimination code.
//
let isMoveImm = 1, isAsCheapAsAMove = 1, isReMaterializable = 1,
    isPseudo = 1, isCodeGenOnly = 1, hasSideEffects = 0 in {
  def PS_fi  : ALU32_ri<(outs IntRegs:$Rd),
                        (ins IntRegs:$fi, s32Imm:$off), "">;
  def PS_fia : ALU32_ri<(outs IntRegs:$Rd),
                        (ins IntRegs:$Rs, IntRegs:$fi, s32Imm:$off), "">;
}

def: Pat<(orisadd (i32 AddrFI:$Rs), s32ImmPred:$off),
         (PS_fi (i32 AddrFI:$Rs), s32ImmPred:$off)>;

//===----------------------------------------------------------------------===//
// CRUSER - Type.
//===----------------------------------------------------------------------===//
// HW loop
let isExtendable = 1, isExtentSigned = 1, opExtentBits = 9, opExtentAlign = 2,
    opExtendable = 0, hasSideEffects = 0 in
class LOOP_iBase<string mnemonic, Operand brOp, bit mustExtend = 0>
         : CRInst<(outs), (ins brOp:$offset, u10Imm:$src2),
           #mnemonic#"($offset, #$src2)",
           [], "" , CR_tc_3x_SLOT3> {
    bits<9> offset;
    bits<10> src2;

    let IClass = 0b0110;

    let Inst{27-22} = 0b100100;
    let Inst{21} = !if (!eq(mnemonic, "loop0"), 0b0, 0b1);
    let Inst{20-16} = src2{9-5};
    let Inst{12-8} = offset{8-4};
    let Inst{7-5} = src2{4-2};
    let Inst{4-3} = offset{3-2};
    let Inst{1-0} = src2{1-0};
}

let isExtendable = 1, isExtentSigned = 1, opExtentBits = 9, opExtentAlign = 2,
    opExtendable = 0, hasSideEffects = 0 in
class LOOP_rBase<string mnemonic, Operand brOp, bit mustExtend = 0>
         : CRInst<(outs), (ins brOp:$offset, IntRegs:$src2),
           #mnemonic#"($offset, $src2)",
           [], "" ,CR_tc_3x_SLOT3> {
    bits<9> offset;
    bits<5> src2;

    let IClass = 0b0110;

    let Inst{27-22} = 0b000000;
    let Inst{21} = !if (!eq(mnemonic, "loop0"), 0b0, 0b1);
    let Inst{20-16} = src2;
    let Inst{12-8} = offset{8-4};
    let Inst{4-3} = offset{3-2};
  }

multiclass LOOP_ri<string mnemonic> {
  def i : LOOP_iBase<mnemonic, brtarget>;
  def r : LOOP_rBase<mnemonic, brtarget>;

  let isCodeGenOnly = 1, isExtended = 1, opExtendable = 0 in {
    def iext: LOOP_iBase<mnemonic, brtargetExt, 1>;
    def rext: LOOP_rBase<mnemonic, brtargetExt, 1>;
  }
}


let Defs = [SA0, LC0, USR] in
defm J2_loop0 : LOOP_ri<"loop0">;

// Interestingly only loop0's appear to set usr.lpcfg
let Defs = [SA1, LC1] in
defm J2_loop1 : LOOP_ri<"loop1">;

let isBranch = 1, isTerminator = 1, hasSideEffects = 0,
    Defs = [PC, LC0], Uses = [SA0, LC0] in {
def ENDLOOP0 : Endloop<(outs), (ins brtarget:$offset),
                       ":endloop0",
                       []>;
}

let isBranch = 1, isTerminator = 1, hasSideEffects = 0,
    Defs = [PC, LC1], Uses = [SA1, LC1] in {
def ENDLOOP1 : Endloop<(outs), (ins brtarget:$offset),
                       ":endloop1",
                       []>;
}

// Pipelined loop instructions, sp[123]loop0
let Defs = [LC0, SA0, P3, USR], hasSideEffects = 0,
    isExtentSigned = 1, isExtendable = 1, opExtentBits = 9, opExtentAlign = 2,
    opExtendable = 0, isPredicateLate = 1 in
class SPLOOP_iBase<string SP, bits<2> op>
  : CRInst <(outs), (ins brtarget:$r7_2, u10Imm:$U10),
  "p3 = sp"#SP#"loop0($r7_2, #$U10)" > {
    bits<9> r7_2;
    bits<10> U10;

    let IClass = 0b0110;

    let Inst{22-21} = op;
    let Inst{27-23} = 0b10011;
    let Inst{20-16} = U10{9-5};
    let Inst{12-8} = r7_2{8-4};
    let Inst{7-5} = U10{4-2};
    let Inst{4-3} = r7_2{3-2};
    let Inst{1-0} = U10{1-0};
  }

let Defs = [LC0, SA0, P3, USR], hasSideEffects = 0,
    isExtentSigned = 1, isExtendable = 1, opExtentBits = 9, opExtentAlign = 2,
    opExtendable = 0, isPredicateLate = 1 in
class SPLOOP_rBase<string SP, bits<2> op>
  : CRInst <(outs), (ins brtarget:$r7_2, IntRegs:$Rs),
  "p3 = sp"#SP#"loop0($r7_2, $Rs)" > {
    bits<9> r7_2;
    bits<5> Rs;

    let IClass = 0b0110;

    let Inst{22-21} = op;
    let Inst{27-23} = 0b00001;
    let Inst{20-16} = Rs;
    let Inst{12-8} = r7_2{8-4};
    let Inst{4-3} = r7_2{3-2};
  }

multiclass SPLOOP_ri<string mnemonic, bits<2> op> {
  def i : SPLOOP_iBase<mnemonic, op>;
  def r : SPLOOP_rBase<mnemonic, op>;
}

defm J2_ploop1s : SPLOOP_ri<"1", 0b01>;
defm J2_ploop2s : SPLOOP_ri<"2", 0b10>;
defm J2_ploop3s : SPLOOP_ri<"3", 0b11>;

// if (Rs[!>=<]=#0) jump:[t/nt]
let Defs = [PC], isPredicated = 1, isBranch = 1, hasSideEffects = 0,
    hasSideEffects = 0 in
class J2_jump_0_Base<string compare, bit isTak, bits<2> op>
  : CRInst <(outs), (ins IntRegs:$Rs, brtarget:$r13_2),
  "if ($Rs"#compare#"#0) jump"#!if(isTak, ":t", ":nt")#" $r13_2" > {
    bits<5> Rs;
    bits<15> r13_2;

    let IClass = 0b0110;

    let Inst{27-24} = 0b0001;
    let Inst{23-22} = op;
    let Inst{12} = isTak;
    let Inst{21} = r13_2{14};
    let Inst{20-16} = Rs;
    let Inst{11-1} = r13_2{12-2};
    let Inst{13} = r13_2{13};
  }

multiclass J2_jump_compare_0<string compare, bits<2> op> {
  def NAME    : J2_jump_0_Base<compare, 0, op>;
  def NAME#pt : J2_jump_0_Base<compare, 1, op>;
}

defm J2_jumprz    : J2_jump_compare_0<"!=", 0b00>;
defm J2_jumprgtez : J2_jump_compare_0<">=", 0b01>;
defm J2_jumprnz   : J2_jump_compare_0<"==", 0b10>;
defm J2_jumprltez : J2_jump_compare_0<"<=", 0b11>;

// Transfer to/from Control/GPR Guest/GPR
let hasSideEffects = 0 in
class TFR_CR_RS_base<RegisterClass CTRC, RegisterClass RC, bit isDouble>
  : CRInst <(outs CTRC:$dst), (ins RC:$src),
  "$dst = $src", [], "", CR_tc_3x_SLOT3> {
    bits<5> dst;
    bits<5> src;

    let IClass = 0b0110;

    let Inst{27-25} = 0b001;
    let Inst{24} = isDouble;
    let Inst{23-21} = 0b001;
    let Inst{20-16} = src;
    let Inst{4-0} = dst;
  }

def A2_tfrrcr : TFR_CR_RS_base<CtrRegs, IntRegs, 0b0>;
def A4_tfrpcp : TFR_CR_RS_base<CtrRegs64, DoubleRegs, 0b1>;
def : InstAlias<"m0 = $Rs", (A2_tfrrcr C6, IntRegs:$Rs)>;
def : InstAlias<"m1 = $Rs", (A2_tfrrcr C7, IntRegs:$Rs)>;

let hasSideEffects = 0 in
class TFR_RD_CR_base<RegisterClass RC, RegisterClass CTRC, bit isSingle>
  : CRInst <(outs RC:$dst), (ins CTRC:$src),
  "$dst = $src", [], "", CR_tc_3x_SLOT3> {
    bits<5> dst;
    bits<5> src;

    let IClass = 0b0110;

    let Inst{27-26} = 0b10;
    let Inst{25} = isSingle;
    let Inst{24-21} = 0b0000;
    let Inst{20-16} = src;
    let Inst{4-0} = dst;
  }

let hasNewValue = 1, opNewValue = 0 in
def A2_tfrcrr : TFR_RD_CR_base<IntRegs, CtrRegs, 1>;
def A4_tfrcpp : TFR_RD_CR_base<DoubleRegs, CtrRegs64, 0>;
def : InstAlias<"$Rd = m0", (A2_tfrcrr IntRegs:$Rd, C6)>;
def : InstAlias<"$Rd = m1", (A2_tfrcrr IntRegs:$Rd, C7)>;

// Y4_trace: Send value to etm trace.
let isSoloAX = 1, hasSideEffects = 0 in
def Y4_trace: CRInst <(outs), (ins IntRegs:$Rs),
  "trace($Rs)"> {
    bits<5> Rs;

    let IClass = 0b0110;
    let Inst{27-21} = 0b0010010;
    let Inst{20-16} = Rs;
  }

// Support for generating global address.
// Taken from X86InstrInfo.td.
def SDTHexagonCONST32 : SDTypeProfile<1, 1, [SDTCisVT<0, i32>,
                                             SDTCisVT<1, i32>,
                                             SDTCisPtrTy<0>]>;
def HexagonCONST32    : SDNode<"HexagonISD::CONST32",    SDTHexagonCONST32>;
def HexagonCONST32_GP : SDNode<"HexagonISD::CONST32_GP", SDTHexagonCONST32>;

// HI/LO Instructions
let isReMaterializable = 1, isMoveImm = 1, hasSideEffects = 0,
    hasNewValue = 1, opNewValue = 0 in
class REG_IMMED<string RegHalf, bit Rs, bits<3> MajOp, bit MinOp>
  : ALU32_ri<(outs IntRegs:$dst),
              (ins u16Imm:$imm_value),
              "$dst"#RegHalf#" = $imm_value", []> {
    bits<5> dst;
    bits<32> imm_value;
    let IClass = 0b0111;

    let Inst{27} = Rs;
    let Inst{26-24} = MajOp;
    let Inst{21} = MinOp;
    let Inst{20-16} = dst;
    let Inst{23-22} = imm_value{15-14};
    let Inst{13-0} = imm_value{13-0};
}

let isAsmParserOnly = 1 in {
  def LO : REG_IMMED<".l", 0b0, 0b001, 0b1>;
  def HI : REG_IMMED<".h", 0b0, 0b010, 0b1>;
}

let isReMaterializable = 1, isMoveImm = 1, isAsmParserOnly = 1 in {
  def CONST32 : CONSTLDInst<(outs IntRegs:$Rd), (ins i32imm:$v),
                "$Rd = CONST32(#$v)", [(set I32:$Rd, imm:$v)]>;
  def CONST64 : CONSTLDInst<(outs DoubleRegs:$Rd), (ins i64imm:$v),
                "$Rd = CONST64(#$v)", [(set I64:$Rd, imm:$v)]>;
}

// Map TLS addressses to A2_tfrsi.
def: Pat<(HexagonCONST32 tglobaltlsaddr:$addr), (A2_tfrsi s16Ext:$addr)>;
def: Pat<(HexagonCONST32 bbl:$label),           (A2_tfrsi s16Ext:$label)>;

let hasSideEffects = 0, isReMaterializable = 1, isPseudo = 1,
    isCodeGenOnly = 1 in
def PS_true : SInst<(outs PredRegs:$dst), (ins), "",
              [(set (i1 PredRegs:$dst), 1)]>;

let hasSideEffects = 0, isReMaterializable = 1, isPseudo = 1,
    isCodeGenOnly = 1 in
def PS_false : SInst<(outs PredRegs:$dst), (ins), "",
               [(set (i1 PredRegs:$dst), 0)]>;

// Pseudo instructions.
def SDT_SPCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32> ]>;
def SDT_SPCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32> ]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_SPCallSeqStart,
                    [SDNPHasChain, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_SPCallSeqEnd,
                    [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def SDT_SPCall  : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;

// For tailcalls a HexagonTCRet SDNode has 3 SDNode Properties - a chain,
// Optional Flag and Variable Arguments.
// Its 1 Operand has pointer type.
def HexagonTCRet : SDNode<"HexagonISD::TC_RETURN", SDT_SPCall,
                          [SDNPHasChain,  SDNPOptInGlue, SDNPVariadic]>;

let Defs = [R29, R30], Uses = [R31, R30, R29], isPseudo = 1 in
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt),
                              ".error \"should not emit\" ",
                              [(callseq_start timm:$amt)]>;

let Defs = [R29, R30, R31], Uses = [R29], isPseudo = 1 in
def ADJCALLSTACKUP : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                             ".error \"should not emit\" ",
                             [(callseq_end timm:$amt1, timm:$amt2)]>;

// Call subroutine indirectly.
let Defs = VolatileV3.Regs in
def J2_callr : JUMPR_MISC_CALLR<0, 1>;

// Indirect tail-call.
let isPseudo = 1, isCall = 1, isReturn = 1, isBarrier = 1, isPredicable = 0,
    isTerminator = 1, isCodeGenOnly = 1 in
def PS_tailcall_r : T_JMPr;

// Direct tail-calls.
let isPseudo = 1, isCall = 1, isReturn = 1, isBarrier = 1, isPredicable = 0,
    isTerminator = 1, isCodeGenOnly = 1 in
def PS_tailcall_i : JInst<(outs), (ins calltarget:$dst), "", []>;

//Tail calls.
def: Pat<(HexagonTCRet tglobaladdr:$dst),
         (PS_tailcall_i tglobaladdr:$dst)>;
def: Pat<(HexagonTCRet texternalsym:$dst),
         (PS_tailcall_i texternalsym:$dst)>;
def: Pat<(HexagonTCRet I32:$dst),
         (PS_tailcall_r I32:$dst)>;

// Map from r0 = and(r1, 65535) to r0 = zxth(r1)
def: Pat<(and (i32 IntRegs:$src1), 65535),
         (A2_zxth IntRegs:$src1)>;

// Map from r0 = and(r1, 255) to r0 = zxtb(r1).
def: Pat<(and (i32 IntRegs:$src1), 255),
         (A2_zxtb IntRegs:$src1)>;

// Map Add(p1, true) to p1 = not(p1).
//     Add(p1, false) should never be produced,
//     if it does, it got to be mapped to NOOP.
def: Pat<(add (i1 PredRegs:$src1), -1),
         (C2_not PredRegs:$src1)>;

// Map from p0 = pnot(p0); r0 = mux(p0, #i, #j) => r0 = mux(p0, #j, #i).
def: Pat<(select (not (i1 PredRegs:$src1)), s8ImmPred:$src2, s32ImmPred:$src3),
         (C2_muxii PredRegs:$src1, s32ImmPred:$src3, s8ImmPred:$src2)>;

// Map from p0 = pnot(p0); r0 = select(p0, #i, r1)
// => r0 = C2_muxir(p0, r1, #i)
def: Pat<(select (not (i1 PredRegs:$src1)), s32ImmPred:$src2,
                 (i32 IntRegs:$src3)),
         (C2_muxir PredRegs:$src1, IntRegs:$src3, s32ImmPred:$src2)>;

// Map from p0 = pnot(p0); r0 = mux(p0, r1, #i)
// => r0 = C2_muxri (p0, #i, r1)
def: Pat<(select (not (i1 PredRegs:$src1)), IntRegs:$src2, s32ImmPred:$src3),
         (C2_muxri PredRegs:$src1, s32ImmPred:$src3, IntRegs:$src2)>;

// Map from p0 = pnot(p0); if (p0) jump => if (!p0) jump.
def: Pat<(brcond (not (i1 PredRegs:$src1)), bb:$offset),
         (J2_jumpf PredRegs:$src1, bb:$offset)>;

// Map from Rdd = sign_extend_inreg(Rss, i32) -> Rdd = A2_sxtw(Rss.lo).
def: Pat<(i64 (sext_inreg (i64 DoubleRegs:$src1), i32)),
         (A2_sxtw (LoReg DoubleRegs:$src1))>;

// Map from Rdd = sign_extend_inreg(Rss, i16) -> Rdd = A2_sxtw(A2_sxth(Rss.lo)).
def: Pat<(i64 (sext_inreg (i64 DoubleRegs:$src1), i16)),
         (A2_sxtw (A2_sxth (LoReg DoubleRegs:$src1)))>;

// Map from Rdd = sign_extend_inreg(Rss, i8) -> Rdd = A2_sxtw(A2_sxtb(Rss.lo)).
def: Pat<(i64 (sext_inreg (i64 DoubleRegs:$src1), i8)),
         (A2_sxtw (A2_sxtb (LoReg DoubleRegs:$src1)))>;

// We want to prevent emitting pnot's as much as possible.
// Map brcond with an unsupported setcc to a J2_jumpf.
def : Pat <(brcond (i1 (setne (i32 IntRegs:$src1), (i32 IntRegs:$src2))),
                        bb:$offset),
      (J2_jumpf (C2_cmpeq (i32 IntRegs:$src1), (i32 IntRegs:$src2)),
                bb:$offset)>;

def : Pat <(brcond (i1 (setne (i32 IntRegs:$src1), s10ImmPred:$src2)),
                        bb:$offset),
      (J2_jumpf (C2_cmpeqi (i32 IntRegs:$src1), s10ImmPred:$src2), bb:$offset)>;

def: Pat<(brcond (i1 (setne (i1 PredRegs:$src1), (i1 -1))), bb:$offset),
         (J2_jumpf PredRegs:$src1, bb:$offset)>;

def: Pat<(brcond (i1 (setne (i1 PredRegs:$src1), (i1 0))), bb:$offset),
         (J2_jumpt PredRegs:$src1, bb:$offset)>;

// cmp.lt(Rs, Imm) -> !cmp.ge(Rs, Imm) -> !cmp.gt(Rs, Imm-1)
def: Pat<(brcond (i1 (setlt (i32 IntRegs:$src1), s8ImmPred:$src2)), bb:$offset),
        (J2_jumpf (C2_cmpgti IntRegs:$src1, (DEC_CONST_SIGNED s8ImmPred:$src2)),
                  bb:$offset)>;

// Map from a 64-bit select to an emulated 64-bit mux.
// Hexagon does not support 64-bit MUXes; so emulate with combines.
def: Pat<(select (i1 PredRegs:$src1), (i64 DoubleRegs:$src2),
                 (i64 DoubleRegs:$src3)),
         (A2_combinew (C2_mux PredRegs:$src1, (HiReg DoubleRegs:$src2),
                                              (HiReg DoubleRegs:$src3)),
                      (C2_mux PredRegs:$src1, (LoReg DoubleRegs:$src2),
                                              (LoReg DoubleRegs:$src3)))>;

// Map from a 1-bit select to logical ops.
// From LegalizeDAG.cpp: (B1 ? B2 : B3) <=> (B1 & B2)|(!B1&B3).
def: Pat<(select (i1 PredRegs:$src1), (i1 PredRegs:$src2), (i1 PredRegs:$src3)),
         (C2_or (C2_and PredRegs:$src1, PredRegs:$src2),
                (C2_and (C2_not PredRegs:$src1), PredRegs:$src3))>;

// Map for truncating from 64 immediates to 32 bit immediates.
def: Pat<(i32 (trunc (i64 DoubleRegs:$src))),
         (LoReg DoubleRegs:$src)>;

// Map for truncating from i64 immediates to i1 bit immediates.
def: Pat<(i1 (trunc (i64 DoubleRegs:$src))),
         (C2_tfrrp (LoReg DoubleRegs:$src))>;

// rs <= rt -> !(rs > rt).
let AddedComplexity = 30 in
def: Pat<(i1 (setle (i32 IntRegs:$src1), s32ImmPred:$src2)),
         (C2_not (C2_cmpgti IntRegs:$src1, s32ImmPred:$src2))>;

// rs <= rt -> !(rs > rt).
def : Pat<(i1 (setle (i32 IntRegs:$src1), (i32 IntRegs:$src2))),
      (i1 (C2_not (C2_cmpgt (i32 IntRegs:$src1), (i32 IntRegs:$src2))))>;

// Rss <= Rtt -> !(Rss > Rtt).
def: Pat<(i1 (setle (i64 DoubleRegs:$src1), (i64 DoubleRegs:$src2))),
         (C2_not (C2_cmpgtp DoubleRegs:$src1, DoubleRegs:$src2))>;

// Map cmpne -> cmpeq.
// Hexagon_TODO: We should improve on this.
// rs != rt -> !(rs == rt).
let AddedComplexity = 30 in
def: Pat<(i1 (setne (i32 IntRegs:$src1), s32ImmPred:$src2)),
         (C2_not (C2_cmpeqi IntRegs:$src1, s32ImmPred:$src2))>;

// Convert setne back to xor for hexagon since we compute w/ pred registers.
def: Pat<(i1 (setne (i1 PredRegs:$src1), (i1 PredRegs:$src2))),
         (C2_xor PredRegs:$src1, PredRegs:$src2)>;

// Map cmpne(Rss) -> !cmpew(Rss).
// rs != rt -> !(rs == rt).
def: Pat<(i1 (setne (i64 DoubleRegs:$src1), (i64 DoubleRegs:$src2))),
         (C2_not (C2_cmpeqp DoubleRegs:$src1, DoubleRegs:$src2))>;

// Map cmpge(Rs, Rt) -> !(cmpgt(Rs, Rt).
// rs >= rt -> !(rt > rs).
def : Pat <(i1 (setge (i32 IntRegs:$src1), (i32 IntRegs:$src2))),
      (i1 (C2_not (i1 (C2_cmpgt (i32 IntRegs:$src2), (i32 IntRegs:$src1)))))>;

// cmpge(Rs, Imm) -> cmpgt(Rs, Imm-1)
let AddedComplexity = 30 in
def: Pat<(i1 (setge (i32 IntRegs:$src1), s32ImmPred:$src2)),
         (C2_cmpgti IntRegs:$src1, (DEC_CONST_SIGNED s32ImmPred:$src2))>;

// Map cmpge(Rss, Rtt) -> !cmpgt(Rtt, Rss).
// rss >= rtt -> !(rtt > rss).
def: Pat<(i1 (setge (i64 DoubleRegs:$src1), (i64 DoubleRegs:$src2))),
         (C2_not (C2_cmpgtp DoubleRegs:$src2, DoubleRegs:$src1))>;

// Map cmplt(Rs, Imm) -> !cmpge(Rs, Imm).
// !cmpge(Rs, Imm) -> !cmpgt(Rs, Imm-1).
// rs < rt -> !(rs >= rt).
let AddedComplexity = 30 in
def: Pat<(i1 (setlt (i32 IntRegs:$src1), s32ImmPred:$src2)),
         (C2_not (C2_cmpgti IntRegs:$src1,
                            (DEC_CONST_SIGNED s32ImmPred:$src2)))>;

// Generate cmpgeu(Rs, #0) -> cmpeq(Rs, Rs)
def: Pat<(i1 (setuge (i32 IntRegs:$src1), 0)),
         (C2_cmpeq IntRegs:$src1, IntRegs:$src1)>;

// Generate cmpgeu(Rs, #u8) -> cmpgtu(Rs, #u8 -1)
def: Pat<(i1 (setuge (i32 IntRegs:$src1), u32ImmPred:$src2)),
         (C2_cmpgtui IntRegs:$src1, (DEC_CONST_UNSIGNED u32ImmPred:$src2))>;

// Generate cmpgtu(Rs, #u9)
def: Pat<(i1 (setugt (i32 IntRegs:$src1), u32ImmPred:$src2)),
         (C2_cmpgtui IntRegs:$src1, u32ImmPred:$src2)>;

// Map from Rs >= Rt -> !(Rt > Rs).
// rs >= rt -> !(rt > rs).
def: Pat<(i1 (setuge (i64 DoubleRegs:$src1), (i64 DoubleRegs:$src2))),
         (C2_not (C2_cmpgtup DoubleRegs:$src2, DoubleRegs:$src1))>;

// Map from cmpleu(Rss, Rtt) -> !cmpgtu(Rss, Rtt-1).
// Map from (Rs <= Rt) -> !(Rs > Rt).
def: Pat<(i1 (setule (i64 DoubleRegs:$src1), (i64 DoubleRegs:$src2))),
         (C2_not (C2_cmpgtup DoubleRegs:$src1, DoubleRegs:$src2))>;

// Sign extends.
// i1 -> i32
def: Pat<(i32 (sext (i1 PredRegs:$src1))),
         (C2_muxii PredRegs:$src1, -1, 0)>;

// i1 -> i64
def: Pat<(i64 (sext (i1 PredRegs:$src1))),
         (A2_combinew (A2_tfrsi -1), (C2_muxii PredRegs:$src1, -1, 0))>;

// Zero extends.
// i1 -> i32
def: Pat<(i32 (zext (i1 PredRegs:$src1))),
         (C2_muxii PredRegs:$src1, 1, 0)>;

// Map from Rs = Pd to Pd = mux(Pd, #1, #0)
def: Pat<(i32 (anyext (i1 PredRegs:$src1))),
         (C2_muxii PredRegs:$src1, 1, 0)>;

// Map from Rss = Pd to Rdd = sxtw (mux(Pd, #1, #0))
def: Pat<(i64 (anyext (i1 PredRegs:$src1))),
         (A2_sxtw (C2_muxii PredRegs:$src1, 1, 0))>;

// Clear the sign bit in a 64-bit register.
def ClearSign : OutPatFrag<(ops node:$Rss),
  (A2_combinew (S2_clrbit_i (HiReg $Rss), 31), (LoReg $Rss))>;

def MulHU : OutPatFrag<(ops node:$Rss, node:$Rtt),
  (A2_addp
    (M2_dpmpyuu_acc_s0
      (S2_lsr_i_p
        (A2_addp
          (M2_dpmpyuu_acc_s0
            (S2_lsr_i_p (M2_dpmpyuu_s0 (LoReg $Rss), (LoReg $Rtt)), 32),
            (HiReg $Rss),
            (LoReg $Rtt)),
          (A2_combinew (A2_tfrsi 0),
                       (LoReg (M2_dpmpyuu_s0 (LoReg $Rss), (HiReg $Rtt))))),
        32),
      (HiReg $Rss),
      (HiReg $Rtt)),
    (S2_lsr_i_p (M2_dpmpyuu_s0 (LoReg $Rss), (HiReg $Rtt)), 32))>;

// Multiply 64-bit unsigned and use upper result.
def : Pat <(mulhu I64:$Rss, I64:$Rtt), (MulHU $Rss, $Rtt)>;

// Multiply 64-bit signed and use upper result.
//
// For two signed 64-bit integers A and B, let A' and B' denote A and B
// with the sign bit cleared. Then A = -2^63*s(A) + A', where s(A) is the
// sign bit of A (and identically for B). With this notation, the signed
// product A*B can be written as:
//   AB = (-2^63 s(A) + A') * (-2^63 s(B) + B')
//      = 2^126 s(A)s(B) - 2^63 [s(A)B'+s(B)A'] + A'B'
//      = 2^126 s(A)s(B) + 2^63 [s(A)B'+s(B)A'] + A'B' - 2*2^63 [s(A)B'+s(B)A']
//      = (unsigned product AB) - 2^64 [s(A)B'+s(B)A']

def : Pat <(mulhs I64:$Rss, I64:$Rtt),
  (A2_subp
    (MulHU $Rss, $Rtt),
    (A2_addp
      (A2_andp (S2_asr_i_p $Rss, 63), (ClearSign $Rtt)),
      (A2_andp (S2_asr_i_p $Rtt, 63), (ClearSign $Rss))))>;

// Hexagon specific ISD nodes.
def SDTHexagonALLOCA : SDTypeProfile<1, 2,
      [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def HexagonALLOCA : SDNode<"HexagonISD::ALLOCA", SDTHexagonALLOCA,
      [SDNPHasChain]>;

// The reason for the custom inserter is to record all ALLOCA instructions
// in MachineFunctionInfo.
let Defs = [R29], isCodeGenOnly = 1, isPseudo = 1, hasSideEffects = 1,
    usesCustomInserter = 1 in
def PS_alloca: ALU32Inst<(outs IntRegs:$Rd),
      (ins IntRegs:$Rs, u32Imm:$A), "",
      [(set (i32 IntRegs:$Rd),
            (HexagonALLOCA (i32 IntRegs:$Rs), (i32 imm:$A)))]>;

let isCodeGenOnly = 1, isPseudo = 1, Uses = [R30], hasSideEffects = 0 in
def PS_aligna : ALU32Inst<(outs IntRegs:$Rd), (ins u32Imm:$A), "", []>;

def HexagonJT:     SDNode<"HexagonISD::JT", SDTIntUnaryOp>;
def HexagonCP:     SDNode<"HexagonISD::CP", SDTIntUnaryOp>;

def: Pat<(HexagonJT tjumptable:$dst), (A2_tfrsi s16Ext:$dst)>;
def: Pat<(HexagonCP tconstpool:$dst), (A2_tfrsi s16Ext:$dst)>;

// XTYPE/SHIFT
//
//===----------------------------------------------------------------------===//
// Template Class
// Shift by immediate/register and accumulate/logical
//===----------------------------------------------------------------------===//

// Rx[+-&|]=asr(Rs,#u5)
// Rx[+-&|^]=lsr(Rs,#u5)
// Rx[+-&|^]=asl(Rs,#u5)

let hasNewValue = 1, opNewValue = 0 in
class T_shift_imm_acc_r <string opc1, string opc2, SDNode OpNode1,
                         SDNode OpNode2, bits<3> majOp, bits<2> minOp>
  : SInst_acc<(outs IntRegs:$Rx),
              (ins IntRegs:$src1, IntRegs:$Rs, u5Imm:$u5),
  "$Rx "#opc2#opc1#"($Rs, #$u5)",
  [(set (i32 IntRegs:$Rx),
         (OpNode2 (i32 IntRegs:$src1),
                  (OpNode1 (i32 IntRegs:$Rs), u5ImmPred:$u5)))],
  "$src1 = $Rx", S_2op_tc_2_SLOT23> {
    bits<5> Rx;
    bits<5> Rs;
    bits<5> u5;

    let IClass = 0b1000;

    let Inst{27-24} = 0b1110;
    let Inst{23-22} = majOp{2-1};
    let Inst{13} = 0b0;
    let Inst{7} = majOp{0};
    let Inst{6-5} = minOp;
    let Inst{4-0} = Rx;
    let Inst{20-16} = Rs;
    let Inst{12-8} = u5;
  }

// Rx[+-&|]=asr(Rs,Rt)
// Rx[+-&|^]=lsr(Rs,Rt)
// Rx[+-&|^]=asl(Rs,Rt)

let hasNewValue = 1, opNewValue = 0 in
class T_shift_reg_acc_r <string opc1, string opc2, SDNode OpNode1,
                         SDNode OpNode2, bits<2> majOp, bits<2> minOp>
  : SInst_acc<(outs IntRegs:$Rx),
              (ins IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt),
  "$Rx "#opc2#opc1#"($Rs, $Rt)",
  [(set (i32 IntRegs:$Rx),
         (OpNode2 (i32 IntRegs:$src1),
                  (OpNode1 (i32 IntRegs:$Rs), (i32 IntRegs:$Rt))))],
  "$src1 = $Rx", S_3op_tc_2_SLOT23 > {
    bits<5> Rx;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1100;

    let Inst{27-24} = 0b1100;
    let Inst{23-22} = majOp;
    let Inst{7-6} = minOp;
    let Inst{4-0} = Rx;
    let Inst{20-16} = Rs;
    let Inst{12-8} = Rt;
  }

// Rxx[+-&|]=asr(Rss,#u6)
// Rxx[+-&|^]=lsr(Rss,#u6)
// Rxx[+-&|^]=asl(Rss,#u6)

class T_shift_imm_acc_p <string opc1, string opc2, SDNode OpNode1,
                         SDNode OpNode2, bits<3> majOp, bits<2> minOp>
  : SInst_acc<(outs DoubleRegs:$Rxx),
              (ins DoubleRegs:$src1, DoubleRegs:$Rss, u6Imm:$u6),
  "$Rxx "#opc2#opc1#"($Rss, #$u6)",
  [(set (i64 DoubleRegs:$Rxx),
        (OpNode2 (i64 DoubleRegs:$src1),
                 (OpNode1 (i64 DoubleRegs:$Rss), u6ImmPred:$u6)))],
  "$src1 = $Rxx", S_2op_tc_2_SLOT23> {
    bits<5> Rxx;
    bits<5> Rss;
    bits<6> u6;

    let IClass = 0b1000;

    let Inst{27-24} = 0b0010;
    let Inst{23-22} = majOp{2-1};
    let Inst{7} = majOp{0};
    let Inst{6-5} = minOp;
    let Inst{4-0} = Rxx;
    let Inst{20-16} = Rss;
    let Inst{13-8} = u6;
  }


// Rxx[+-&|]=asr(Rss,Rt)
// Rxx[+-&|^]=lsr(Rss,Rt)
// Rxx[+-&|^]=asl(Rss,Rt)
// Rxx[+-&|^]=lsl(Rss,Rt)

class T_shift_reg_acc_p <string opc1, string opc2, SDNode OpNode1,
                         SDNode OpNode2, bits<3> majOp, bits<2> minOp>
  : SInst_acc<(outs DoubleRegs:$Rxx),
              (ins DoubleRegs:$src1, DoubleRegs:$Rss, IntRegs:$Rt),
  "$Rxx "#opc2#opc1#"($Rss, $Rt)",
  [(set (i64 DoubleRegs:$Rxx),
        (OpNode2 (i64 DoubleRegs:$src1),
                 (OpNode1 (i64 DoubleRegs:$Rss), (i32 IntRegs:$Rt))))],
  "$src1 = $Rxx", S_3op_tc_2_SLOT23> {
    bits<5> Rxx;
    bits<5> Rss;
    bits<5> Rt;

    let IClass = 0b1100;

    let Inst{27-24} = 0b1011;
    let Inst{23-21} = majOp;
    let Inst{20-16} = Rss;
    let Inst{12-8} = Rt;
    let Inst{7-6} = minOp;
    let Inst{4-0} = Rxx;
  }

//===----------------------------------------------------------------------===//
// Multi-class for the shift instructions with logical/arithmetic operators.
//===----------------------------------------------------------------------===//

multiclass xtype_imm_base<string OpcStr1, string OpcStr2, SDNode OpNode1,
                         SDNode OpNode2, bits<3> majOp, bits<2> minOp > {
  def _i_r#NAME : T_shift_imm_acc_r< OpcStr1, OpcStr2, OpNode1,
                                     OpNode2, majOp, minOp >;
  def _i_p#NAME : T_shift_imm_acc_p< OpcStr1, OpcStr2, OpNode1,
                                     OpNode2, majOp, minOp >;
}

multiclass xtype_imm_acc<string opc1, SDNode OpNode, bits<2>minOp> {
  let AddedComplexity = 100 in
  defm _acc  : xtype_imm_base< opc1, "+= ", OpNode, add, 0b001, minOp>;

  defm _nac  : xtype_imm_base< opc1, "-= ", OpNode, sub, 0b000, minOp>;
  defm _and  : xtype_imm_base< opc1, "&= ", OpNode, and, 0b010, minOp>;
  defm _or   : xtype_imm_base< opc1, "|= ", OpNode,  or, 0b011, minOp>;
}

multiclass xtype_xor_imm_acc<string opc1, SDNode OpNode, bits<2>minOp> {
let AddedComplexity = 100 in
  defm _xacc  : xtype_imm_base< opc1, "^= ", OpNode, xor, 0b100, minOp>;
}

defm S2_asr : xtype_imm_acc<"asr", sra, 0b00>;

defm S2_lsr : xtype_imm_acc<"lsr", srl, 0b01>,
              xtype_xor_imm_acc<"lsr", srl, 0b01>;

defm S2_asl : xtype_imm_acc<"asl", shl, 0b10>,
              xtype_xor_imm_acc<"asl", shl, 0b10>;

multiclass xtype_reg_acc_r<string opc1, SDNode OpNode, bits<2>minOp> {
  let AddedComplexity = 100 in
  def _acc : T_shift_reg_acc_r <opc1, "+= ", OpNode, add, 0b11, minOp>;

  def _nac : T_shift_reg_acc_r <opc1, "-= ", OpNode, sub, 0b10, minOp>;
  def _and : T_shift_reg_acc_r <opc1, "&= ", OpNode, and, 0b01, minOp>;
  def _or  : T_shift_reg_acc_r <opc1, "|= ", OpNode,  or, 0b00, minOp>;
}

multiclass xtype_reg_acc_p<string opc1, SDNode OpNode, bits<2>minOp> {
  let AddedComplexity = 100 in
  def _acc : T_shift_reg_acc_p <opc1, "+= ", OpNode, add, 0b110, minOp>;

  def _nac : T_shift_reg_acc_p <opc1, "-= ", OpNode, sub, 0b100, minOp>;
  def _and : T_shift_reg_acc_p <opc1, "&= ", OpNode, and, 0b010, minOp>;
  def _or  : T_shift_reg_acc_p <opc1, "|= ", OpNode,  or, 0b000, minOp>;
  def _xor : T_shift_reg_acc_p <opc1, "^= ", OpNode, xor, 0b011, minOp>;
}

multiclass xtype_reg_acc<string OpcStr, SDNode OpNode, bits<2> minOp > {
  defm _r_r : xtype_reg_acc_r <OpcStr, OpNode, minOp>;
  defm _r_p : xtype_reg_acc_p <OpcStr, OpNode, minOp>;
}

defm S2_asl : xtype_reg_acc<"asl", shl, 0b10>;
defm S2_asr : xtype_reg_acc<"asr", sra, 0b00>;
defm S2_lsr : xtype_reg_acc<"lsr", srl, 0b01>;
defm S2_lsl : xtype_reg_acc<"lsl", shl, 0b11>;

//===----------------------------------------------------------------------===//
let hasSideEffects = 0 in
class T_S3op_1 <string mnemonic, RegisterClass RC, bits<2> MajOp, bits<3> MinOp,
                bit SwapOps, bit isSat = 0, bit isRnd = 0, bit hasShift = 0>
  : SInst <(outs RC:$dst),
           (ins DoubleRegs:$src1, DoubleRegs:$src2),
  "$dst = "#mnemonic#"($src1, $src2)"#!if(isRnd, ":rnd", "")
                                     #!if(hasShift,":>>1","")
                                     #!if(isSat, ":sat", ""),
  [], "", S_3op_tc_2_SLOT23 > {
    bits<5> dst;
    bits<5> src1;
    bits<5> src2;

    let IClass = 0b1100;

    let Inst{27-24} = 0b0001;
    let Inst{23-22} = MajOp;
    let Inst{20-16} = !if (SwapOps, src2, src1);
    let Inst{12-8}  = !if (SwapOps, src1, src2);
    let Inst{7-5}   = MinOp;
    let Inst{4-0}   = dst;
  }

class T_S3op_64 <string mnemonic, bits<2> MajOp, bits<3> MinOp, bit SwapOps,
                 bit isSat = 0, bit isRnd = 0, bit hasShift = 0 >
  : T_S3op_1 <mnemonic, DoubleRegs, MajOp, MinOp, SwapOps,
              isSat, isRnd, hasShift>;

let Itinerary = S_3op_tc_1_SLOT23 in {
  def S2_shuffeb : T_S3op_64 < "shuffeb", 0b00, 0b010, 0>;
  def S2_shuffeh : T_S3op_64 < "shuffeh", 0b00, 0b110, 0>;
  def S2_shuffob : T_S3op_64 < "shuffob", 0b00, 0b100, 1>;
  def S2_shuffoh : T_S3op_64 < "shuffoh", 0b10, 0b000, 1>;

  def S2_vtrunewh : T_S3op_64 < "vtrunewh", 0b10, 0b010, 0>;
  def S2_vtrunowh : T_S3op_64 < "vtrunowh", 0b10, 0b100, 0>;
}

def S2_lfsp : T_S3op_64 < "lfs", 0b10, 0b110, 0>;

let hasSideEffects = 0 in
class T_S3op_2 <string mnemonic, bits<3> MajOp, bit SwapOps>
  : SInst < (outs DoubleRegs:$Rdd),
            (ins DoubleRegs:$Rss, DoubleRegs:$Rtt, PredRegs:$Pu),
  "$Rdd = "#mnemonic#"($Rss, $Rtt, $Pu)",
  [], "", S_3op_tc_1_SLOT23 > {
    bits<5> Rdd;
    bits<5> Rss;
    bits<5> Rtt;
    bits<2> Pu;

    let IClass = 0b1100;

    let Inst{27-24} = 0b0010;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = !if (SwapOps, Rtt, Rss);
    let Inst{12-8} = !if (SwapOps, Rss, Rtt);
    let Inst{6-5} = Pu;
    let Inst{4-0} = Rdd;
  }

def S2_valignrb  : T_S3op_2 < "valignb",  0b000, 1>;
def S2_vsplicerb : T_S3op_2 < "vspliceb", 0b100, 0>;

//===----------------------------------------------------------------------===//
// Template class used by vector shift, vector rotate, vector neg,
// 32-bit shift, 64-bit shifts, etc.
//===----------------------------------------------------------------------===//

let hasSideEffects = 0 in
class T_S3op_3 <string mnemonic, RegisterClass RC, bits<2> MajOp,
                 bits<2> MinOp, bit isSat = 0, list<dag> pattern = [] >
  : SInst <(outs RC:$dst),
           (ins RC:$src1, IntRegs:$src2),
  "$dst = "#mnemonic#"($src1, $src2)"#!if(isSat, ":sat", ""),
  pattern, "", S_3op_tc_1_SLOT23> {
    bits<5> dst;
    bits<5> src1;
    bits<5> src2;

    let IClass = 0b1100;

    let Inst{27-24} = !if(!eq(!cast<string>(RC), "IntRegs"), 0b0110, 0b0011);
    let Inst{23-22} = MajOp;
    let Inst{20-16} = src1;
    let Inst{12-8} = src2;
    let Inst{7-6} = MinOp;
    let Inst{4-0} = dst;
  }

let hasNewValue = 1 in
class T_S3op_shift32 <string mnemonic, SDNode OpNode, bits<2> MinOp>
  : T_S3op_3 <mnemonic, IntRegs, 0b01, MinOp, 0,
    [(set (i32 IntRegs:$dst), (OpNode (i32 IntRegs:$src1),
                                      (i32 IntRegs:$src2)))]>;

let hasNewValue = 1, Itinerary = S_3op_tc_2_SLOT23 in
class T_S3op_shift32_Sat <string mnemonic, bits<2> MinOp>
  : T_S3op_3 <mnemonic, IntRegs, 0b00, MinOp, 1, []>;


class T_S3op_shift64 <string mnemonic, SDNode OpNode, bits<2> MinOp>
  : T_S3op_3 <mnemonic, DoubleRegs, 0b10, MinOp, 0,
    [(set (i64 DoubleRegs:$dst), (OpNode (i64 DoubleRegs:$src1),
                                         (i32 IntRegs:$src2)))]>;


class T_S3op_shiftVect <string mnemonic, bits<2> MajOp, bits<2> MinOp>
  : T_S3op_3 <mnemonic, DoubleRegs, MajOp, MinOp, 0, []>;


// Shift by register
// Rdd=[asr|lsr|asl|lsl](Rss,Rt)

def S2_asr_r_p : T_S3op_shift64 < "asr", sra, 0b00>;
def S2_lsr_r_p : T_S3op_shift64 < "lsr", srl, 0b01>;
def S2_asl_r_p : T_S3op_shift64 < "asl", shl, 0b10>;
def S2_lsl_r_p : T_S3op_shift64 < "lsl", shl, 0b11>;

// Rd=[asr|lsr|asl|lsl](Rs,Rt)

def S2_asr_r_r : T_S3op_shift32<"asr", sra, 0b00>;
def S2_lsr_r_r : T_S3op_shift32<"lsr", srl, 0b01>;
def S2_asl_r_r : T_S3op_shift32<"asl", shl, 0b10>;
def S2_lsl_r_r : T_S3op_shift32<"lsl", shl, 0b11>;

// Shift by register with saturation
// Rd=asr(Rs,Rt):sat
// Rd=asl(Rs,Rt):sat

let Defs = [USR_OVF] in {
  def S2_asr_r_r_sat : T_S3op_shift32_Sat<"asr", 0b00>;
  def S2_asl_r_r_sat : T_S3op_shift32_Sat<"asl", 0b10>;
}

let hasNewValue = 1, hasSideEffects = 0 in
class T_S3op_8 <string opc, bits<3> MinOp, bit isSat, bit isRnd, bit hasShift, bit hasSplat = 0>
  : SInst < (outs IntRegs:$Rd),
            (ins DoubleRegs:$Rss, IntRegs:$Rt),
  "$Rd = "#opc#"($Rss, $Rt"#!if(hasSplat, "*", "")#")"
                           #!if(hasShift, ":<<1", "")
                           #!if(isRnd, ":rnd", "")
                           #!if(isSat, ":sat", ""),
  [], "", S_3op_tc_1_SLOT23 > {
    bits<5> Rd;
    bits<5> Rss;
    bits<5> Rt;

    let IClass = 0b1100;

    let Inst{27-24} = 0b0101;
    let Inst{20-16} = Rss;
    let Inst{12-8}  = Rt;
    let Inst{7-5}   = MinOp;
    let Inst{4-0}   = Rd;
  }

def S2_asr_r_svw_trun : T_S3op_8<"vasrw", 0b010, 0, 0, 0>;

let Defs = [USR_OVF], Itinerary = S_3op_tc_2_SLOT23 in
def S2_vcrotate : T_S3op_shiftVect < "vcrotate", 0b11, 0b00>;

let hasSideEffects = 0 in
class T_S3op_7 <string mnemonic, bit MajOp >
  : SInst <(outs DoubleRegs:$Rdd),
           (ins DoubleRegs:$Rss, DoubleRegs:$Rtt, u3Imm:$u3),
  "$Rdd = "#mnemonic#"($Rss, $Rtt, #$u3)" ,
  [], "", S_3op_tc_1_SLOT23 > {
    bits<5> Rdd;
    bits<5> Rss;
    bits<5> Rtt;
    bits<3> u3;

    let IClass = 0b1100;

    let Inst{27-24} = 0b0000;
    let Inst{23}    = MajOp;
    let Inst{20-16} = !if(MajOp, Rss, Rtt);
    let Inst{12-8}  =  !if(MajOp, Rtt, Rss);
    let Inst{7-5}   = u3;
    let Inst{4-0}   = Rdd;
  }

def S2_valignib  : T_S3op_7 < "valignb", 0>;
def S2_vspliceib : T_S3op_7 < "vspliceb", 1>;

//===----------------------------------------------------------------------===//
// Template class for 'insert bitfield' instructions
//===----------------------------------------------------------------------===//
let hasSideEffects = 0 in
class T_S3op_insert <string mnemonic, RegisterClass RC>
  : SInst <(outs RC:$dst),
           (ins RC:$src1, RC:$src2, DoubleRegs:$src3),
  "$dst = "#mnemonic#"($src2, $src3)" ,
  [], "$src1 = $dst", S_3op_tc_1_SLOT23 > {
    bits<5> dst;
    bits<5> src2;
    bits<5> src3;

    let IClass = 0b1100;

    let Inst{27-26} = 0b10;
    let Inst{25-24} = !if(!eq(!cast<string>(RC), "IntRegs"), 0b00, 0b10);
    let Inst{23}    = 0b0;
    let Inst{20-16} = src2;
    let Inst{12-8}  = src3;
    let Inst{4-0}   = dst;
  }

let hasSideEffects = 0 in
class T_S2op_insert <bits<4> RegTyBits, RegisterClass RC, Operand ImmOp>
  : SInst <(outs RC:$dst), (ins RC:$dst2, RC:$src1, ImmOp:$src2, ImmOp:$src3),
  "$dst = insert($src1, #$src2, #$src3)",
  [], "$dst2 = $dst", S_2op_tc_2_SLOT23> {
    bits<5> dst;
    bits<5> src1;
    bits<6> src2;
    bits<6> src3;
    bit bit23;
    bit bit13;
    string ImmOpStr = !cast<string>(ImmOp);

    let bit23 = !if (!eq(ImmOpStr, "u6Imm"), src3{5}, 0);
    let bit13 = !if (!eq(ImmOpStr, "u6Imm"), src2{5}, 0);

    let IClass = 0b1000;

    let Inst{27-24} = RegTyBits;
    let Inst{23}    = bit23;
    let Inst{22-21} = src3{4-3};
    let Inst{20-16} = src1;
    let Inst{13}    = bit13;
    let Inst{12-8}  = src2{4-0};
    let Inst{7-5}   = src3{2-0};
    let Inst{4-0}   = dst;
  }

// Rx=insert(Rs,Rtt)
// Rx=insert(Rs,#u5,#U5)
let hasNewValue = 1 in {
  def S2_insert_rp : T_S3op_insert <"insert", IntRegs>;
  def S2_insert    : T_S2op_insert <0b1111, IntRegs, u5Imm>;
}

// Rxx=insert(Rss,Rtt)
// Rxx=insert(Rss,#u6,#U6)
def S2_insertp_rp : T_S3op_insert<"insert", DoubleRegs>;
def S2_insertp    : T_S2op_insert <0b0011, DoubleRegs, u6Imm>;


def SDTHexagonINSERT:
  SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                       SDTCisInt<0>, SDTCisVT<3, i32>, SDTCisVT<4, i32>]>;
def SDTHexagonINSERTRP:
  SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                       SDTCisInt<0>, SDTCisVT<3, i64>]>;

def HexagonINSERT   : SDNode<"HexagonISD::INSERT",   SDTHexagonINSERT>;
def HexagonINSERTRP : SDNode<"HexagonISD::INSERTRP", SDTHexagonINSERTRP>;

def: Pat<(HexagonINSERT I32:$Rs, I32:$Rt, u5ImmPred:$u1, u5ImmPred:$u2),
         (S2_insert I32:$Rs, I32:$Rt, u5ImmPred:$u1, u5ImmPred:$u2)>;
def: Pat<(HexagonINSERT I64:$Rs, I64:$Rt, u6ImmPred:$u1, u6ImmPred:$u2),
         (S2_insertp I64:$Rs, I64:$Rt, u6ImmPred:$u1, u6ImmPred:$u2)>;
def: Pat<(HexagonINSERTRP I32:$Rs, I32:$Rt, I64:$Ru),
         (S2_insert_rp I32:$Rs, I32:$Rt, I64:$Ru)>;
def: Pat<(HexagonINSERTRP I64:$Rs, I64:$Rt, I64:$Ru),
         (S2_insertp_rp I64:$Rs, I64:$Rt, I64:$Ru)>;

let AddedComplexity = 100 in
def: Pat<(or (or (shl (HexagonINSERT (i32 (zextloadi8 (add I32:$b, 2))),
                                     (i32 (extloadi8  (add I32:$b, 3))),
                                     24, 8),
                      (i32 16)),
                 (shl (i32 (zextloadi8 (add I32:$b, 1))), (i32 8))),
             (zextloadi8 I32:$b)),
         (A2_swiz (L2_loadri_io I32:$b, 0))>;


//===----------------------------------------------------------------------===//
// Template class for 'extract bitfield' instructions
//===----------------------------------------------------------------------===//
let hasNewValue = 1, hasSideEffects = 0 in
class T_S3op_extract <string mnemonic, bits<2> MinOp>
  : SInst <(outs IntRegs:$Rd), (ins IntRegs:$Rs, DoubleRegs:$Rtt),
  "$Rd = "#mnemonic#"($Rs, $Rtt)",
  [], "", S_3op_tc_2_SLOT23 > {
    bits<5> Rd;
    bits<5> Rs;
    bits<5> Rtt;

    let IClass = 0b1100;

    let Inst{27-22} = 0b100100;
    let Inst{20-16} = Rs;
    let Inst{12-8}  = Rtt;
    let Inst{7-6}   = MinOp;
    let Inst{4-0}   = Rd;
  }

let hasSideEffects = 0 in
class T_S2op_extract <string mnemonic, bits<4> RegTyBits,
                      RegisterClass RC, Operand ImmOp>
  : SInst <(outs RC:$dst), (ins RC:$src1, ImmOp:$src2, ImmOp:$src3),
  "$dst = "#mnemonic#"($src1, #$src2, #$src3)",
  [], "", S_2op_tc_2_SLOT23> {
    bits<5> dst;
    bits<5> src1;
    bits<6> src2;
    bits<6> src3;
    bit bit23;
    bit bit13;
    string ImmOpStr = !cast<string>(ImmOp);

    let bit23 = !if (!eq(ImmOpStr, "u6Imm"), src3{5},
                !if (!eq(mnemonic, "extractu"), 0, 1));

    let bit13 = !if (!eq(ImmOpStr, "u6Imm"), src2{5}, 0);

    let IClass = 0b1000;

    let Inst{27-24} = RegTyBits;
    let Inst{23}    = bit23;
    let Inst{22-21} = src3{4-3};
    let Inst{20-16} = src1;
    let Inst{13}    = bit13;
    let Inst{12-8}  = src2{4-0};
    let Inst{7-5}   = src3{2-0};
    let Inst{4-0}   = dst;
  }

// Extract bitfield

// Rdd=extractu(Rss,Rtt)
// Rdd=extractu(Rss,#u6,#U6)
def S2_extractup_rp : T_S3op_64 < "extractu", 0b00, 0b000, 0>;
def S2_extractup    : T_S2op_extract <"extractu", 0b0001, DoubleRegs, u6Imm>;

// Rd=extractu(Rs,Rtt)
// Rd=extractu(Rs,#u5,#U5)
let hasNewValue = 1 in {
  def S2_extractu_rp : T_S3op_extract<"extractu", 0b00>;
  def S2_extractu    : T_S2op_extract <"extractu", 0b1101, IntRegs, u5Imm>;
}

def SDTHexagonEXTRACTU:
  SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisInt<1>,
                       SDTCisVT<2, i32>, SDTCisVT<3, i32>]>;
def SDTHexagonEXTRACTURP:
  SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisInt<1>,
                       SDTCisVT<2, i64>]>;

def HexagonEXTRACTU   : SDNode<"HexagonISD::EXTRACTU",   SDTHexagonEXTRACTU>;
def HexagonEXTRACTURP : SDNode<"HexagonISD::EXTRACTURP", SDTHexagonEXTRACTURP>;

def: Pat<(HexagonEXTRACTU I32:$src1, u5ImmPred:$src2, u5ImmPred:$src3),
         (S2_extractu I32:$src1, u5ImmPred:$src2, u5ImmPred:$src3)>;
def: Pat<(HexagonEXTRACTU I64:$src1, u6ImmPred:$src2, u6ImmPred:$src3),
         (S2_extractup I64:$src1, u6ImmPred:$src2, u6ImmPred:$src3)>;
def: Pat<(HexagonEXTRACTURP I32:$src1, I64:$src2),
         (S2_extractu_rp I32:$src1, I64:$src2)>;
def: Pat<(HexagonEXTRACTURP I64:$src1, I64:$src2),
         (S2_extractup_rp I64:$src1, I64:$src2)>;

// Change the sign of the immediate for Rd=-mpyi(Rs,#u8)
def: Pat<(mul (i32 IntRegs:$src1), (ineg n8ImmPred:$src2)),
         (M2_mpysin IntRegs:$src1, u8ImmPred:$src2)>;

//===----------------------------------------------------------------------===//
// :raw for of tableindx[bdhw] insns
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
class tableidxRaw<string OpStr, bits<2>MinOp>
  : SInst <(outs IntRegs:$Rx),
           (ins IntRegs:$_dst_, IntRegs:$Rs, u4Imm:$u4, s6Imm:$S6),
           "$Rx = "#OpStr#"($Rs, #$u4, #$S6):raw",
    [], "$Rx = $_dst_" > {
    bits<5> Rx;
    bits<5> Rs;
    bits<4> u4;
    bits<6> S6;

    let IClass = 0b1000;

    let Inst{27-24} = 0b0111;
    let Inst{23-22} = MinOp;
    let Inst{21}    = u4{3};
    let Inst{20-16} = Rs;
    let Inst{13-8}  = S6;
    let Inst{7-5}   = u4{2-0};
    let Inst{4-0}   = Rx;
  }

def S2_tableidxb : tableidxRaw<"tableidxb", 0b00>;
def S2_tableidxh : tableidxRaw<"tableidxh", 0b01>;
def S2_tableidxw : tableidxRaw<"tableidxw", 0b10>;
def S2_tableidxd : tableidxRaw<"tableidxd", 0b11>;

//===----------------------------------------------------------------------===//
// Template class for 'table index' instructions which are assembler mapped
// to their :raw format.
//===----------------------------------------------------------------------===//
let isPseudo = 1 in
class tableidx_goodsyntax <string mnemonic>
  : SInst <(outs IntRegs:$Rx),
           (ins IntRegs:$_dst_, IntRegs:$Rs, u4Imm:$u4, u5Imm:$u5),
           "$Rx = "#mnemonic#"($Rs, #$u4, #$u5)",
           [], "$Rx = $_dst_" >;

def S2_tableidxb_goodsyntax : tableidx_goodsyntax<"tableidxb">;
def S2_tableidxh_goodsyntax : tableidx_goodsyntax<"tableidxh">;
def S2_tableidxw_goodsyntax : tableidx_goodsyntax<"tableidxw">;
def S2_tableidxd_goodsyntax : tableidx_goodsyntax<"tableidxd">;

//===----------------------------------------------------------------------===//
// V3 Instructions +
//===----------------------------------------------------------------------===//

include "HexagonInstrInfoV3.td"

//===----------------------------------------------------------------------===//
// V3 Instructions -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// V4 Instructions +
//===----------------------------------------------------------------------===//

include "HexagonInstrInfoV4.td"

//===----------------------------------------------------------------------===//
// V4 Instructions -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// V5 Instructions +
//===----------------------------------------------------------------------===//

include "HexagonInstrInfoV5.td"

//===----------------------------------------------------------------------===//
// V5 Instructions -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// V60 Instructions +
//===----------------------------------------------------------------------===//

include "HexagonInstrInfoV60.td"

//===----------------------------------------------------------------------===//
// V60 Instructions -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU32/64/Vector +
//===----------------------------------------------------------------------===///

include "HexagonInstrInfoVector.td"

include "HexagonInstrAlias.td"
include "HexagonSystemInst.td"

