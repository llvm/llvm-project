//=- HexagonInstrInfoV5.td - Target Desc. for Hexagon Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V5 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XTYPE/MPY
//===----------------------------------------------------------------------===//

  //Rdd[+]=vrmpybsu(Rss,Rtt)
let Predicates = [HasV5T] in {
  def M5_vrmpybsu: T_XTYPE_Vect<"vrmpybsu", 0b110, 0b001, 0>;
  def M5_vrmacbsu: T_XTYPE_Vect_acc<"vrmpybsu", 0b110, 0b001, 0>;

  //Rdd[+]=vrmpybu(Rss,Rtt)
  def M5_vrmpybuu: T_XTYPE_Vect<"vrmpybu", 0b100, 0b001, 0>;
  def M5_vrmacbuu: T_XTYPE_Vect_acc<"vrmpybu", 0b100, 0b001, 0>;

  def M5_vdmpybsu: T_M2_vmpy<"vdmpybsu", 0b101, 0b001, 0, 0, 1>;
  def M5_vdmacbsu: T_M2_vmpy_acc_sat <"vdmpybsu", 0b001, 0b001, 0, 0>;
}

// Vector multiply bytes
// Rdd=vmpyb[s]u(Rs,Rt)
let Predicates = [HasV5T] in {
  def M5_vmpybsu: T_XTYPE_mpy64 <"vmpybsu", 0b010, 0b001, 0, 0, 0>;
  def M5_vmpybuu: T_XTYPE_mpy64 <"vmpybu",  0b100, 0b001, 0, 0, 0>;

  // Rxx+=vmpyb[s]u(Rs,Rt)
  def M5_vmacbsu: T_XTYPE_mpy64_acc <"vmpybsu", "+", 0b110, 0b001, 0, 0, 0>;
  def M5_vmacbuu: T_XTYPE_mpy64_acc <"vmpybu", "+", 0b100, 0b001, 0, 0, 0>;

  // Rd=vaddhub(Rss,Rtt):sat
  let hasNewValue = 1, opNewValue = 0 in
    def A5_vaddhubs: T_S3op_1 <"vaddhub", IntRegs, 0b01, 0b001, 0, 1>;
}

def S2_asr_i_p_rnd : S_2OpInstImm<"asr", 0b110, 0b111, u6_0Imm, [], 1>,
      Requires<[HasV5T]> {
  bits<6> src2;
  let Inst{13-8} = src2;
}

let isAsmParserOnly = 1 in
def S2_asr_i_p_rnd_goodsyntax
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, u6_0Imm:$src2),
    "$dst = asrrnd($src1, #$src2)">;

def C4_fastcorner9 : T_LOGICAL_2OP<"fastcorner9", 0b000, 0, 0>,
  Requires<[HasV5T]> {
  let Inst{13,7,4} = 0b111;
}

def C4_fastcorner9_not : T_LOGICAL_2OP<"!fastcorner9", 0b000, 0, 0>,
  Requires<[HasV5T]> {
  let Inst{20,13,7,4} = 0b1111;
}

let hasNewValue = 1, validSubTargets = HasV5SubT in
def S5_popcountp : ALU64_rr<(outs IntRegs:$Rd), (ins DoubleRegs:$Rss),
  "$Rd = popcount($Rss)", [], "", S_2op_tc_2_SLOT23>,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rss;

    let IClass = 0b1000;

    let Inst{27-21} = 0b1000011;
    let Inst{7-5} = 0b011;
    let Inst{4-0} = Rd;
    let Inst{20-16} = Rss;
  }

let isFP = 1, hasNewValue = 1, opNewValue = 0 in
class T_MInstFloat <string mnemonic, bits<3> MajOp, bits<3> MinOp>
  : MInst<(outs IntRegs:$Rd),
          (ins IntRegs:$Rs, IntRegs:$Rt),
  "$Rd = "#mnemonic#"($Rs, $Rt)", [],
  "" , M_tc_3or4x_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1011;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{13} = 0b0;
    let Inst{12-8} = Rt;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = Rd;
  }

let isCommutable = 1 in {
  def F2_sfadd : T_MInstFloat < "sfadd", 0b000, 0b000>;
  def F2_sfmpy : T_MInstFloat < "sfmpy", 0b010, 0b000>;
}

def F2_sfsub : T_MInstFloat < "sfsub", 0b000, 0b001>;

let Itinerary = M_tc_3x_SLOT23 in {
  def F2_sfmax : T_MInstFloat < "sfmax", 0b100, 0b000>;
  def F2_sfmin : T_MInstFloat < "sfmin", 0b100, 0b001>;
}

let Itinerary = M_tc_3or4x_SLOT23 in {
def F2_sffixupn : T_MInstFloat < "sffixupn", 0b110, 0b000>;
def F2_sffixupd : T_MInstFloat < "sffixupd", 0b110, 0b001>;
}

// F2_sfrecipa: Reciprocal approximation for division.
let Uses = [USR], isPredicateLate = 1, isFP = 1,
    hasSideEffects = 0, hasNewValue = 1, Itinerary = M_tc_3or4x_SLOT23 in
def F2_sfrecipa: MInst <
  (outs IntRegs:$Rd, PredRegs:$Pe),
  (ins IntRegs:$Rs, IntRegs:$Rt),
  "$Rd, $Pe = sfrecipa($Rs, $Rt)">,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<2> Pe;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;
    let Inst{27-21} = 0b1011111;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = Rt;
    let Inst{7}     = 0b1;
    let Inst{6-5}   = Pe;
    let Inst{4-0}   = Rd;
  }

// F2_dfcmpeq: Floating point compare for equal.
let Uses = [USR], isCompare = 1, isFP = 1 in
class T_fcmp <string mnemonic, RegisterClass RC, bits<3> MinOp,
              list<dag> pattern = [] >
  : ALU64Inst <(outs PredRegs:$dst), (ins RC:$src1, RC:$src2),
  "$dst = "#mnemonic#"($src1, $src2)", pattern,
  "" , ALU64_tc_2early_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<2> dst;
    bits<5> src1;
    bits<5> src2;

    let IClass = 0b1101;

    let Inst{27-21} = 0b0010111;
    let Inst{20-16} = src1;
    let Inst{12-8}  = src2;
    let Inst{7-5}   = MinOp;
    let Inst{1-0}   = dst;
  }

class T_fcmp64 <string mnemonic, PatFrag OpNode, bits<3> MinOp>
  : T_fcmp <mnemonic, DoubleRegs, MinOp, []> {
  let IClass = 0b1101;
  let Inst{27-21} = 0b0010111;
}

class T_fcmp32 <string mnemonic, PatFrag OpNode, bits<3> MinOp>
  : T_fcmp <mnemonic, IntRegs, MinOp, []> {
  let IClass = 0b1100;
  let Inst{27-21} = 0b0111111;
}

def F2_dfcmpeq : T_fcmp64<"dfcmp.eq", setoeq, 0b000>;
def F2_dfcmpgt : T_fcmp64<"dfcmp.gt", setogt, 0b001>;
def F2_dfcmpge : T_fcmp64<"dfcmp.ge", setoge, 0b010>;
def F2_dfcmpuo : T_fcmp64<"dfcmp.uo", setuo,  0b011>;

def F2_sfcmpge : T_fcmp32<"sfcmp.ge", setoge, 0b000>;
def F2_sfcmpuo : T_fcmp32<"sfcmp.uo", setuo,  0b001>;
def F2_sfcmpeq : T_fcmp32<"sfcmp.eq", setoeq, 0b011>;
def F2_sfcmpgt : T_fcmp32<"sfcmp.gt", setogt, 0b100>;

// F2 convert template classes:
let Uses = [USR], isFP = 1 in
class F2_RDD_RSS_CONVERT<string mnemonic, bits<3> MinOp,
                         string chop ="">
  : SInst <(outs DoubleRegs:$Rdd), (ins DoubleRegs:$Rss),
   "$Rdd = "#mnemonic#"($Rss)"#chop, [], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rdd;
     bits<5> Rss;

     let IClass = 0b1000;

     let Inst{27-21} = 0b0000111;
     let Inst{20-16} = Rss;
     let Inst{7-5} = MinOp;
     let Inst{4-0} = Rdd;
  }

let Uses = [USR], isFP = 1 in
class F2_RDD_RS_CONVERT<string mnemonic, bits<3> MinOp,
                        string chop ="">
  : SInst <(outs DoubleRegs:$Rdd), (ins IntRegs:$Rs),
   "$Rdd = "#mnemonic#"($Rs)"#chop, [], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rdd;
     bits<5> Rs;

     let IClass = 0b1000;

     let Inst{27-21} = 0b0100100;
     let Inst{20-16} = Rs;
     let Inst{7-5} = MinOp;
     let Inst{4-0} = Rdd;
  }

let Uses = [USR], isFP = 1, hasNewValue = 1 in
class F2_RD_RSS_CONVERT<string mnemonic, bits<3> MinOp,
                        string chop ="">
  : SInst <(outs IntRegs:$Rd), (ins DoubleRegs:$Rss),
   "$Rd = "#mnemonic#"($Rss)"#chop, [], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rd;
     bits<5> Rss;

     let IClass = 0b1000;

     let Inst{27-24} = 0b1000;
     let Inst{23-21} = MinOp;
     let Inst{20-16} = Rss;
     let Inst{7-5} = 0b001;
     let Inst{4-0} = Rd;
  }

let Uses = [USR], isFP = 1, hasNewValue = 1 in
class F2_RD_RS_CONVERT<string mnemonic, bits<3> MajOp, bits<3> MinOp,
                        string chop ="">
  : SInst <(outs IntRegs:$Rd), (ins IntRegs:$Rs),
   "$Rd = "#mnemonic#"($Rs)"#chop, [], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rd;
     bits<5> Rs;

     let IClass = 0b1000;

     let Inst{27-24} = 0b1011;
     let Inst{23-21} = MajOp;
     let Inst{20-16} = Rs;
     let Inst{7-5} = MinOp;
     let Inst{4-0} = Rd;
  }

// Convert single precision to double precision and vice-versa.
def F2_conv_sf2df : F2_RDD_RS_CONVERT <"convert_sf2df", 0b000>;
def F2_conv_df2sf : F2_RD_RSS_CONVERT <"convert_df2sf", 0b000>;

// Convert Integer to Floating Point.
def F2_conv_d2sf : F2_RD_RSS_CONVERT <"convert_d2sf", 0b010>;
def F2_conv_ud2sf : F2_RD_RSS_CONVERT <"convert_ud2sf", 0b001>;
def F2_conv_uw2sf : F2_RD_RS_CONVERT <"convert_uw2sf", 0b001, 0b000>;
def F2_conv_w2sf : F2_RD_RS_CONVERT <"convert_w2sf", 0b010, 0b000>;
def F2_conv_d2df : F2_RDD_RSS_CONVERT <"convert_d2df", 0b011>;
def F2_conv_ud2df : F2_RDD_RSS_CONVERT <"convert_ud2df", 0b010>;
def F2_conv_uw2df : F2_RDD_RS_CONVERT <"convert_uw2df", 0b001>;
def F2_conv_w2df : F2_RDD_RS_CONVERT <"convert_w2df", 0b010>;

// Convert Floating Point to Integer.
def F2_conv_df2uw_chop : F2_RD_RSS_CONVERT <"convert_df2uw", 0b101, ":chop">;
def F2_conv_df2w_chop : F2_RD_RSS_CONVERT <"convert_df2w", 0b111, ":chop">;
def F2_conv_sf2uw_chop : F2_RD_RS_CONVERT <"convert_sf2uw", 0b011, 0b001,
                                           ":chop">;
def F2_conv_sf2w_chop : F2_RD_RS_CONVERT <"convert_sf2w", 0b100, 0b001,
                                          ":chop">;
def F2_conv_df2d_chop : F2_RDD_RSS_CONVERT <"convert_df2d", 0b110, ":chop">;
def F2_conv_df2ud_chop : F2_RDD_RSS_CONVERT <"convert_df2ud", 0b111, ":chop">;
def F2_conv_sf2d_chop : F2_RDD_RS_CONVERT <"convert_sf2d", 0b110, ":chop">;
def F2_conv_sf2ud_chop : F2_RDD_RS_CONVERT <"convert_sf2ud", 0b101, ":chop">;

// Convert Floating Point to Integer: non-chopped.
let AddedComplexity = 20, Predicates = [HasV5T] in {
  def F2_conv_df2d : F2_RDD_RSS_CONVERT <"convert_df2d", 0b000>;
  def F2_conv_df2ud : F2_RDD_RSS_CONVERT <"convert_df2ud", 0b001>;
  def F2_conv_sf2ud : F2_RDD_RS_CONVERT <"convert_sf2ud", 0b011>;
  def F2_conv_sf2d : F2_RDD_RS_CONVERT <"convert_sf2d", 0b100>;
  def F2_conv_df2uw : F2_RD_RSS_CONVERT <"convert_df2uw", 0b011>;
  def F2_conv_df2w : F2_RD_RSS_CONVERT <"convert_df2w", 0b100>;
  def F2_conv_sf2uw : F2_RD_RS_CONVERT <"convert_sf2uw", 0b011, 0b000>;
  def F2_conv_sf2w : F2_RD_RS_CONVERT <"convert_sf2w", 0b100, 0b000>;
}

// Fix up radicand.
let Uses = [USR], isFP = 1, hasNewValue = 1 in
def F2_sffixupr: SInst<(outs IntRegs:$Rd), (ins IntRegs:$Rs),
  "$Rd = sffixupr($Rs)",
  [], "" , S_2op_tc_3or4x_SLOT23>, Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rs;

    let IClass = 0b1000;

    let Inst{27-21} = 0b1011101;
    let Inst{20-16} = Rs;
    let Inst{7-5}   = 0b000;
    let Inst{4-0}   = Rd;
  }

// F2_sffma: Floating-point fused multiply add.
let Uses = [USR], isFP = 1, hasNewValue = 1 in
class T_sfmpy_acc <bit isSub, bit isLib>
  : MInst<(outs IntRegs:$Rx),
          (ins IntRegs:$dst2, IntRegs:$Rs, IntRegs:$Rt),
  "$Rx "#!if(isSub, "-=","+=")#" sfmpy($Rs, $Rt)"#!if(isLib, ":lib",""),
  [], "$dst2 = $Rx" , M_tc_3or4x_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<5> Rx;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-21} = 0b1111000;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = Rt;
    let Inst{7}     = 0b1;
    let Inst{6}     = isLib;
    let Inst{5}     = isSub;
    let Inst{4-0}   = Rx;
  }

def F2_sffma: T_sfmpy_acc <0, 0>;
def F2_sffms: T_sfmpy_acc <1, 0>;
def F2_sffma_lib: T_sfmpy_acc <0, 1>;
def F2_sffms_lib: T_sfmpy_acc <1, 1>;

// Floating-point fused multiply add w/ additional scaling (2**pu).
let Uses = [USR], isFP = 1, hasNewValue = 1 in
def F2_sffma_sc: MInst <
  (outs IntRegs:$Rx),
  (ins IntRegs:$dst2, IntRegs:$Rs, IntRegs:$Rt, PredRegs:$Pu),
  "$Rx += sfmpy($Rs, $Rt, $Pu):scale" ,
  [], "$dst2 = $Rx" , M_tc_3or4x_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<5> Rx;
    bits<5> Rs;
    bits<5> Rt;
    bits<2> Pu;

    let IClass = 0b1110;

    let Inst{27-21} = 0b1111011;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = Rt;
    let Inst{7}     = 0b1;
    let Inst{6-5}   = Pu;
    let Inst{4-0}   = Rx;
  }

//===----------------------------------------------------------------------===//
// :natural forms of vasrh and vasrhub insns
//===----------------------------------------------------------------------===//
// S5_asrhub_rnd_sat: Vector arithmetic shift right by immediate with round,
// saturate, and pack.
let Defs = [USR_OVF], hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
class T_ASRHUB<bit isSat>
  : SInst <(outs IntRegs:$Rd),
  (ins DoubleRegs:$Rss, u4_0Imm:$u4),
  "$Rd = vasrhub($Rss, #$u4):"#!if(isSat, "sat", "raw"),
  [], "", S_2op_tc_2_SLOT23>,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rss;
    bits<4> u4;

    let IClass = 0b1000;

    let Inst{27-21} = 0b1000011;
    let Inst{20-16} = Rss;
    let Inst{13-12} = 0b00;
    let Inst{11-8} = u4;
    let Inst{7-6} = 0b10;
    let Inst{5} = isSat;
    let Inst{4-0} = Rd;
  }

def S5_asrhub_rnd_sat : T_ASRHUB <0>;
def S5_asrhub_sat : T_ASRHUB <1>;

let isAsmParserOnly = 1 in
def S5_asrhub_rnd_sat_goodsyntax
  : SInst <(outs IntRegs:$Rd), (ins DoubleRegs:$Rss, u4_0Imm:$u4),
  "$Rd = vasrhub($Rss, #$u4):rnd:sat">, Requires<[HasV5T]>;

// S5_vasrhrnd: Vector arithmetic shift right by immediate with round.
let hasSideEffects = 0 in
def S5_vasrhrnd : SInst <(outs DoubleRegs:$Rdd),
                         (ins DoubleRegs:$Rss, u4_0Imm:$u4),
  "$Rdd = vasrh($Rss, #$u4):raw">,
  Requires<[HasV5T]> {
    bits<5> Rdd;
    bits<5> Rss;
    bits<4> u4;

    let IClass = 0b1000;

    let Inst{27-21} = 0b0000001;
    let Inst{20-16} = Rss;
    let Inst{13-12} = 0b00;
    let Inst{11-8}  = u4;
    let Inst{7-5}   = 0b000;
    let Inst{4-0}   = Rdd;
  }

let isAsmParserOnly = 1 in
def S5_vasrhrnd_goodsyntax
  : SInst <(outs DoubleRegs:$Rdd), (ins DoubleRegs:$Rss, u4_0Imm:$u4),
  "$Rdd = vasrh($Rss,#$u4):rnd">, Requires<[HasV5T]>;

// Floating point reciprocal square root approximation
let Uses = [USR], isPredicateLate = 1, isFP = 1,
    hasSideEffects = 0, hasNewValue = 1, opNewValue = 0,
    validSubTargets = HasV5SubT in
def F2_sfinvsqrta: SInst <
  (outs IntRegs:$Rd, PredRegs:$Pe),
  (ins IntRegs:$Rs),
  "$Rd, $Pe = sfinvsqrta($Rs)" > ,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<2> Pe;
    bits<5> Rs;

    let IClass = 0b1000;

    let Inst{27-21} = 0b1011111;
    let Inst{20-16} = Rs;
    let Inst{7} = 0b0;
    let Inst{6-5} = Pe;
    let Inst{4-0} = Rd;
  }

// Complex multiply 32x16
let Defs = [USR_OVF], Itinerary = S_3op_tc_3x_SLOT23 in {
  def M4_cmpyi_whc : T_S3op_8<"cmpyiwh", 0b101, 1, 1, 1, 1>;
  def M4_cmpyr_whc : T_S3op_8<"cmpyrwh", 0b111, 1, 1, 1, 1>;
}

// Classify floating-point value
let Uses = [USR], isFP = 1 in
def F2_sfclass : T_TEST_BIT_IMM<"sfclass", 0b111>, Requires<[HasV5T]>;

let Uses = [USR], isFP = 1 in
def F2_dfclass: ALU64Inst<(outs PredRegs:$Pd), (ins DoubleRegs:$Rss, u5_0Imm:$u5),
  "$Pd = dfclass($Rss, #$u5)",
  [], "" , ALU64_tc_2early_SLOT23 > , Requires<[HasV5T]> {
    bits<2> Pd;
    bits<5> Rss;
    bits<5> u5;

    let IClass = 0b1101;
    let Inst{27-21} = 0b1100100;
    let Inst{20-16} = Rss;
    let Inst{12-10} = 0b000;
    let Inst{9-5}   = u5;
    let Inst{4-3}   = 0b10;
    let Inst{1-0}   = Pd;
  }

// Instructions to create floating point constant
class T_fimm <string mnemonic, RegisterClass RC, bits<4> RegType, bit isNeg>
  : ALU64Inst<(outs RC:$dst), (ins u10_0Imm:$src),
  "$dst = "#mnemonic#"(#$src)"#!if(isNeg, ":neg", ":pos"),
  [], "", ALU64_tc_2_SLOT23>, Requires<[HasV5T]> {
    bits<5> dst;
    bits<10> src;

    let IClass = 0b1101;
    let Inst{27-24} = RegType;
    let Inst{23}    = 0b0;
    let Inst{22}    = isNeg;
    let Inst{21}    = src{9};
    let Inst{13-5}  = src{8-0};
    let Inst{4-0}   = dst;
  }

let hasNewValue = 1, opNewValue = 0 in {
  def F2_sfimm_p : T_fimm <"sfmake", IntRegs, 0b0110, 0>;
  def F2_sfimm_n : T_fimm <"sfmake", IntRegs, 0b0110, 1>;
}

def F2_dfimm_p : T_fimm <"dfmake", DoubleRegs, 0b1001, 0>;
def F2_dfimm_n : T_fimm <"dfmake", DoubleRegs, 0b1001, 1>;
