//===-- PPCInstrFutureMMA.td - Future Instruction Set ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the instructions introduced for the Future CPU for MMA.
//
//===----------------------------------------------------------------------===//

class XX3Form_AT3_XABp5_P1<bits<6> opcode, bits<8> xo, dag OOL, dag IOL,
                           string asmstr, list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, NoItinerary> {
  bits<3> AT;
  bits<5> XAp;
  bits<5> XBp;
  bits<1> P;

  let Pattern = pattern;

  let Inst{6-8} = AT{2-0};
  let Inst{9-10} = 0;
  let Inst{11-14} = XAp{3-0};
  let Inst{15} = P;
  let Inst{16-19} = XBp{3-0};
  let Inst{20} = 0;
  let Inst{21-28} = xo;
  let Inst{29} = XAp{4};
  let Inst{30} = XBp{4};
  let Inst{31} = 0;
}

class XX2Form_AT3_XBp5_P2<bits<6> opcode, bits<9> xo, dag OOL, dag IOL,
                          string asmstr, list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, NoItinerary> {
  bits<3> AT;
  bits<5> XBp;
  bits<2> P;

  let Pattern = pattern;

  let Inst{6-8} = AT{2-0};
  let Inst{9-14} = 0;
  let Inst{15} = P{0};
  let Inst{16-19} = XBp{3-0};
  let Inst{20} = P{1};
  let Inst{21-29} = xo;
  let Inst{30} = XBp{4};
  let Inst{31} = 0;
}

class XForm_ATB3<bits<6> opcode, bits<5> o, bits<10> xo, dag OOL, dag IOL,
                 string asmstr, list<dag> pattern>
  : I <opcode, OOL, IOL, asmstr, NoItinerary> {
  bits<3> AT;
  bits<3> AB;

  let Pattern = pattern;

  let Inst{6-8} = AT{2-0};
  let Inst{9-10} = 0;
  let Inst{11-15} = o;
  let Inst{16-18} = AB{2-0};
  let Inst{19-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31} = 0;
}

class XX3Form_AT3_XAp5B6<bits<6> opcode, bits<8> xo, dag OOL, dag IOL,
                           string asmstr, InstrItinClass itin,
                           list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<3> AT;
  bits<5> XAp;
  bits<6> XB;

  let Pattern = pattern;

  let Inst{6-8} = AT;
  let Inst{9-10} = 0;
  let Inst{11-14} = XAp{3-0};
  let Inst{15} = 0;
  let Inst{16-20} = XB{4-0};
  let Inst{21-28} = xo;
  let Inst{29}    = XAp{4};
  let Inst{30}    = XB{5};
  let Inst{31} = 0;
}

class MMIRR_XX3Form_X8YP4_XAp5B6<bits<6> opcode, bits<8> xo, dag OOL, dag IOL,
                                 string asmstr, InstrItinClass itin,
                                 list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<3> AT;
  bits<5> XAp;
  bits<6> XB;
  bits<8> XMSK;
  bits<4> YMSK;
  bits<4> PMSK;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-7} = 3;
  let Inst{8-11} = 9;
  let Inst{12-15} = 0;
  let Inst{16-19} = PMSK;
  let Inst{20-27} = XMSK;
  let Inst{28-31} = YMSK;

  // The instruction.
  let Inst{38-40} = AT;
  let Inst{41-42} = 0;
  let Inst{43-46} = XAp{3-0};
  let Inst{47} = 0;
  let Inst{48-52} = XB{4-0};
  let Inst{53-60} = xo;
  let Inst{61} = XAp{4};
  let Inst{62} = XB{5};
  let Inst{63} = 0;
}

class MMIRR_XX3Form_X8Y4P2_XAp5B6<bits<6> opcode, bits<8> xo, dag OOL, dag IOL,
                                 string asmstr, InstrItinClass itin,
                                 list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<3> AT;
  bits<5> XAp;
  bits<6> XB;
  bits<8> XMSK;
  bits<4> YMSK;
  bits<2> PMSK;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-7} = 3;
  let Inst{8-11} = 9;
  let Inst{12-15} = 0;
  let Inst{16-17} = PMSK;
  let Inst{18-19} = 0;
  let Inst{20-27} = XMSK;
  let Inst{28-31} = YMSK;

  // The instruction.
  let Inst{38-40} = AT;
  let Inst{41-42} = 0;
  let Inst{43-46} = XAp{3-0};
  let Inst{47} = 0;
  let Inst{48-52} = XB{4-0};
  let Inst{53-60} = xo;
  let Inst{61} = XAp{4};
  let Inst{62} = XB{5};
  let Inst{63} = 0;
}

multiclass DMR_UM_XOEO<bits<6> opcode, bits<8> xo, dag IOL, string asmbase,
                       string asmstr> {
  let Predicates = [MMA, IsISAFuture] in {
  def NAME :
    XX3Form_AT3_XAp5B6<opcode, !or(xo, 0x01), (outs dmr:$AT), IOL,
                     !strconcat(asmbase#" ", asmstr), IIC_VecFP, []>,
    RegConstraint<"@earlyclobber $AT">;
  def PP :
    XX3Form_AT3_XAp5B6<opcode, xo, (outs dmr:$AT), !con((ins dmr:$ATi), IOL),
                     !strconcat(asmbase#"pp ", asmstr), IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  }
}

multiclass DMR_UM_M448_XOEO<bits<6> opcode, bits<8> xo, dag IOL, string asmbase,
                            string asmstr> {
  defm NAME : DMR_UM_XOEO<opcode, xo, IOL, asmbase, asmstr>;
  let Predicates = [MMA, PrefixInstrs, IsISAFuture] in {
  def PM#NAME :
    MMIRR_XX3Form_X8YP4_XAp5B6<
      opcode, !or(xo, 0x01), (outs dmr:$AT),
      !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u4imm:$PMSK)),
      !strconcat("pm"#asmbase#" ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"@earlyclobber $AT">;
  def PM#NAME#PP :
    MMIRR_XX3Form_X8YP4_XAp5B6<
      opcode, xo, (outs dmr:$AT),
      !con((ins dmr:$ATi),
           !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u4imm:$PMSK))),
      !strconcat("pm"#asmbase#"pp ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  }
}

multiclass DMR_BF16_UM_XOEO<bits<6> opcode, bits<8> xo, dag IOL, string asmbase,
                       string asmstr> {
  let Predicates = [MMA, IsISAFuture] in {
  def NAME :
    XX3Form_AT3_XAp5B6<opcode, !or(xo, 0x11), (outs dmr:$AT), IOL,
                     !strconcat(asmbase#" ", asmstr), IIC_VecFP, []>,
    RegConstraint<"@earlyclobber $AT">;
  def PP :
    XX3Form_AT3_XAp5B6<opcode, xo, (outs dmr:$AT), !con((ins dmr:$ATi), IOL),
                     !strconcat(asmbase#"pp ", asmstr), IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  }
}

multiclass DMR_BF16_UM_M284_XOEO<bits<6> opcode, bits<8> xo, dag IOL, string asmbase,
                            string asmstr> {
  defm NAME : DMR_BF16_UM_XOEO<opcode, xo, IOL, asmbase, asmstr>;
  let Predicates = [MMA, PrefixInstrs, IsISAFuture] in {
  def PM#NAME :
    MMIRR_XX3Form_X8Y4P2_XAp5B6<
      opcode, !or(xo, 0x11), (outs dmr:$AT),
      !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u2imm:$PMSK)),
      !strconcat("pm"#asmbase#" ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"@earlyclobber $AT">;
  def PM#NAME#PP :
    MMIRR_XX3Form_X8Y4P2_XAp5B6<
      opcode, xo, (outs dmr:$AT),
      !con((ins dmr:$ATi), !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u2imm:$PMSK))),
      !strconcat("pm"#asmbase#"pp ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  }
}

multiclass DMR_F16_UM_M284_XOEO<bits<6> opcode, bits<8> xo, dag IOL, string asmbase,
                            string asmstr> {
  defm NAME : DMR_UM_XOEO<opcode, xo, IOL, asmbase, asmstr>;
  let Predicates = [MMA, PrefixInstrs, IsISAFuture] in {
  def PM#NAME :
    MMIRR_XX3Form_X8Y4P2_XAp5B6<
      opcode, !or(xo, 0x01), (outs dmr:$AT),
      !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u2imm:$PMSK)),
      !strconcat("pm"#asmbase#" ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"@earlyclobber $AT">;
  def PM#NAME#PP :
    MMIRR_XX3Form_X8Y4P2_XAp5B6<
      opcode, xo, (outs dmr:$AT),
      !con((ins dmr:$ATi), !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u2imm:$PMSK))),
      !strconcat("pm"#asmbase#"pp ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  }
}

multiclass DMR_NEG_UM_M284_XOXORf939a0<bits<6> opcode, bits<8> xo, dag IOL,
                                  string asmbase, string asmstr> {
  defm NAME : DMR_BF16_UM_M284_XOEO<opcode, xo, IOL, asmbase, asmstr>;
  let Predicates = [MMA, IsISAFuture] in {
  def PN : XX3Form_AT3_XAp5B6<
             opcode, !xor(xo, 0xF9), (outs dmr:$AT), !con((ins dmr:$ATi), IOL),
             !strconcat(asmbase#"pn ", asmstr), IIC_VecFP, []>,
           RegConstraint<"$ATi = $AT">;
  def NP : XX3Form_AT3_XAp5B6<
             opcode, !xor(xo, 0x39), (outs dmr:$AT), !con((ins dmr:$ATi), IOL),
             !strconcat(asmbase#"np ", asmstr), IIC_VecFP, []>,
           RegConstraint<"$ATi = $AT">;
  def NN : XX3Form_AT3_XAp5B6<
             opcode, !xor(xo, 0xA0), (outs dmr:$AT), !con((ins dmr:$ATi), IOL),
             !strconcat(asmbase#"nn ", asmstr), IIC_VecFP, []>,
           RegConstraint<"$ATi = $AT">;
  }
  let Predicates = [MMA, PrefixInstrs, IsISAFuture] in {
   def PM#NAME#PN :
    MMIRR_XX3Form_X8Y4P2_XAp5B6<
      opcode, !xor(xo, 0xF9), (outs dmr:$AT),
      !con((ins dmr:$ATi), !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u2imm:$PMSK))),
      !strconcat("pm"#asmbase#"pn ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  def PM#NAME#NP :
    MMIRR_XX3Form_X8Y4P2_XAp5B6<
      opcode, !xor(xo, 0x39), (outs dmr:$AT),
      !con((ins dmr:$ATi), !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u2imm:$PMSK))),
      !strconcat("pm"#asmbase#"np ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  def PM#NAME#NN :
    MMIRR_XX3Form_X8Y4P2_XAp5B6<
      opcode, !xor(xo, 0xA0), (outs dmr:$AT),
      !con((ins dmr:$ATi), !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u2imm:$PMSK))),
      !strconcat("pm"#asmbase#"nn ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  }
}

multiclass DMR_NEG_UM_M284_XOXORd11188<bits<6> opcode, bits<8> xo, dag IOL,
                                  string asmbase, string asmstr> {
  defm NAME : DMR_F16_UM_M284_XOEO<opcode, xo, IOL, asmbase, asmstr>;
  let Predicates = [MMA, IsISAFuture] in {
  def PN : XX3Form_AT3_XAp5B6<
             opcode, !xor(xo, 0xD1), (outs dmr:$AT), !con((ins dmr:$ATi), IOL),
             !strconcat(asmbase#"pn ", asmstr), IIC_VecFP, []>,
           RegConstraint<"$ATi = $AT">;
  def NP : XX3Form_AT3_XAp5B6<
             opcode, !xor(xo, 0x11), (outs dmr:$AT), !con((ins dmr:$ATi), IOL),
             !strconcat(asmbase#"np ", asmstr), IIC_VecFP, []>,
           RegConstraint<"$ATi = $AT">;
  def NN : XX3Form_AT3_XAp5B6<
             opcode, !xor(xo, 0x88), (outs dmr:$AT), !con((ins dmr:$ATi), IOL),
             !strconcat(asmbase#"nn ", asmstr), IIC_VecFP, []>,
           RegConstraint<"$ATi = $AT">;
  }
  let Predicates = [MMA, PrefixInstrs, IsISAFuture] in {
   def PM#NAME#PN :
    MMIRR_XX3Form_X8Y4P2_XAp5B6<
      opcode, !xor(xo, 0xD1), (outs dmr:$AT),
      !con((ins dmr:$ATi), !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u2imm:$PMSK))),
      !strconcat("pm"#asmbase#"pn ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  def PM#NAME#NP :
    MMIRR_XX3Form_X8Y4P2_XAp5B6<
      opcode, !xor(xo, 0x11), (outs dmr:$AT),
      !con((ins dmr:$ATi), !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u2imm:$PMSK))),
      !strconcat("pm"#asmbase#"np ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  def PM#NAME#NN :
    MMIRR_XX3Form_X8Y4P2_XAp5B6<
      opcode, !xor(xo, 0x88), (outs dmr:$AT),
      !con((ins dmr:$ATi), !con(IOL, (ins u8imm:$XMSK, u4imm:$YMSK, u2imm:$PMSK))),
      !strconcat("pm"#asmbase#"nn ", asmstr#", $XMSK, $YMSK, $PMSK"),
      IIC_VecFP, []>,
    RegConstraint<"$ATi = $AT">;
  }
}

class XForm_AT3_T1_AB3<bits<6> opcode, bits<5> o, bits<10> xo, dag OOL, dag IOL,
                       string asmstr, list<dag> pattern>
  : I <opcode, OOL, IOL, asmstr, NoItinerary> {
  bits<3> AT;
  bits<3> AB;
  bits<1> T;

  let Pattern = pattern;

  let Inst{6-8} = AT{2-0};
  let Inst{9} = 0;
  let Inst{10} = T;
  let Inst{11-15} = o;
  let Inst{16-18} = AB{2-0};
  let Inst{19-20} = 0;
  let Inst{21-30} = xo;
  let Inst{31} = 0;
}

class XForm_ATp2_SR5<bits<6> opcode, bits<5> o, bits<10> xo, dag OOL, dag IOL,
                     string asmstr, list<dag> pattern>
  : I <opcode, OOL, IOL, asmstr, NoItinerary> {
  bits<2> ATp;
  bits<5> SR;

  let Pattern = pattern;

  let Inst{6-7} = ATp{1-0};
  let Inst{8-10} = 0;
  let Inst{11-15} = o;
  let Inst{16-20} = SR{4-0};
  let Inst{21-30} = xo;
  let Inst{31} = 0;
}

class XX2Form_AT3_XB6_ID2_E1_BL2<bits<6> opcode, bits<9> xo, dag OOL, dag IOL,
                          string asmstr, list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, NoItinerary> {
  bits<3> AT;
  bits<6> XB;
  bits<2> ID;
  bits<1> E;
  bits<2> BL;

  let Pattern = pattern;

  let Inst{6-8} = AT{2-0};
  let Inst{9-10} = 0;
  let Inst{11-12} = ID{1-0};
  let Inst{13} = E;
  let Inst{14-15} = BL{1-0};
  let Inst{16-20} = XB{4-0};
  let Inst{21-29} = xo;
  let Inst{30} = XB{5};
  let Inst{31} = 0;
}

let Predicates = [IsISAFuture] in {
  def DMXXEXTFDMR512 : XX3Form_AT3_XABp5_P1<60, 226,
                                            (outs vsrprc:$XAp, vsrprc:$XBp),
                                            (ins wacc:$AT),
                                            "dmxxextfdmr512 $XAp, $XBp, $AT, 0", []> {
    let P = 0;
  }

  def DMXXEXTFDMR512_HI : XX3Form_AT3_XABp5_P1<60, 226,
                                               (outs vsrprc:$XAp, vsrprc:$XBp),
                                               (ins wacc_hi:$AT),
                                               "dmxxextfdmr512 $XAp, $XBp, $AT, 1", []> {
    let P = 1;
  }

  def DMXXINSTDMR512 : XX3Form_AT3_XABp5_P1<60, 234, (outs wacc:$AT),
                                             (ins vsrprc:$XAp, vsrprc:$XBp),
                                             "dmxxinstdmr512 $AT, $XAp, $XBp, 0", []> {
    let P = 0;
  }

  def DMXXINSTDMR512_HI : XX3Form_AT3_XABp5_P1<60, 234, (outs wacc_hi:$AT),
                                                (ins vsrprc:$XAp, vsrprc:$XBp),
                                                "dmxxinstdmr512 $AT, $XAp, $XBp, 1", []> {
    let P = 1;
  }

  def DMXXEXTFDMR256 : XX2Form_AT3_XBp5_P2<60, 484, (outs vsrprc:$XBp),
                                           (ins dmrrowp:$AT, u2imm:$P),
                                           "dmxxextfdmr256 $XBp, $AT, $P", []>;

  def DMXXINSTDMR256 : XX2Form_AT3_XBp5_P2<60, 485, (outs dmrrowp:$AT),
                                            (ins vsrprc:$XBp, u2imm:$P),
                                            "dmxxinstdmr256 $AT, $XBp, $P", []>;

  def DMMR : XForm_ATB3<31, 6, 177, (outs dmr:$AT), (ins dmr:$AB),
                        "dmmr $AT, $AB", 
                        [(set v1024i1:$AT, (int_ppc_mma_dmmr v1024i1:$AB))]>;

  def DMXOR : XForm_ATB3<31, 7, 177, (outs dmr:$AT), (ins dmr:$ATi, dmr:$AB),
                         "dmxor $AT, $AB",
                         [(set v1024i1:$AT, (int_ppc_mma_dmxor v1024i1:$ATi, v1024i1:$AB))]>,
                         RegConstraint<"$ATi = $AT">;

  def DMSETDMRZ : XForm_AT3<31, 2, 177, (outs dmr:$AT), (ins),
                            "dmsetdmrz $AT", NoItinerary,
                            [(set v1024i1:$AT, (int_ppc_mma_dmsetdmrz))]>;
}

// MMA+ accumulating/non-accumulating instructions.

// DMXVI8GERX4, DMXVI8GERX4PP, PMDMXVI8GERX4,  PMDMXVI8GERX4PP
defm DMXVI8GERX4 : DMR_UM_M448_XOEO<59, 10, (ins vsrprc:$XAp, vsrc:$XB),
                               "dmxvi8gerx4", "$AT, $XAp, $XB">;

let Predicates = [MMA, IsISAFuture] in {
  def DMXVI8GERX4SPP :
    XX3Form_AT3_XAp5B6<59, 98, (outs dmr:$AT), (ins dmr:$ATi, vsrprc:$XAp, vsrc:$XB),
                     "dmxvi8gerx4spp $AT, $XAp, $XB", IIC_VecGeneral, []>,
    RegConstraint<"$ATi = $AT">;
}

let Predicates = [MMA, PrefixInstrs, IsISAFuture] in {
  def PMDMXVI8GERX4SPP :
    MMIRR_XX3Form_X8YP4_XAp5B6<59, 98, (outs dmr:$AT),
                            (ins dmr:$ATi, vsrprc:$XAp,vsrc:$XB, u8imm:$XMSK,
                             u4imm:$YMSK, u4imm:$PMSK),
                            "pmdmxvi8gerx4spp $AT, $XAp, $XB, $XMSK, $YMSK, $PMSK",
                            IIC_VecGeneral, []>,
    RegConstraint<"$ATi = $AT">;
}

// DMXVBF16GERX2, DMXVBF16GERX2PP, DMXVBF16GERX2PN, dMXVBF16GERX2NP, DMXVBF16GERX2NN
// PMDMXVBF16GERX2, PMDMXVBF16GERX2PP, PMDMXVBF16GERX2PN, PMDMXVBF16GERX2NP, PMDMXVBF16GERX2NN
defm DMXVBF16GERX2 : DMR_NEG_UM_M284_XOXORf939a0<59, 74, (ins vsrprc:$XAp, vsrc:$XB),
                                         "dmxvbf16gerx2", "$AT, $XAp, $XB">;

// DMXVF16GERX2, DMXVF16GERX2PP, DMXVF16GERX2PN, dMXVF16GERX2NP, DMXVF16GERX2NN
// PMDMXVF16GERX2, PMDMXVF16GERX2PP, PMDMXVF16GERX2PN, PMDMXVF16GERX2NP, PMDMXVF16GERX2NN
defm DMXVF16GERX2 : DMR_NEG_UM_M284_XOXORd11188<59, 66, (ins vsrprc:$XAp, vsrc:$XB),
                                         "dmxvf16gerx2", "$AT, $XAp, $XB">;

// DMF cryptography [support] Instructions
let Predicates = [IsISAFuture] in {
  def DMSHA2HASH :
    XForm_AT3_T1_AB3<31, 14, 177, (outs dmr:$AT), (ins dmr:$ATi, dmr:$AB, u1imm:$T),
                     "dmsha2hash $AT, $AB, $T",
                     [(set v1024i1:$AT, (int_ppc_mma_dmsha2hash v1024i1:$ATi, v1024i1:$AB, timm:$T))]>,
                     RegConstraint<"$ATi = $AT">;

  def DMSHA3HASH :
    XForm_ATp2_SR5<31, 15, 177, (outs dmrp:$ATp), (ins dmrp:$ATpi , u5imm:$SR),
                   "dmsha3hash $ATp, $SR",
                   [(set v2048i1:$ATp, (int_ppc_mma_dmsha3hash v2048i1:$ATpi, timm:$SR))]>,
                   RegConstraint<"$ATpi = $ATp">;

  def DMXXSHAPAD :
    XX2Form_AT3_XB6_ID2_E1_BL2<60, 421, (outs dmr:$AT),
                               (ins dmr:$ATi, vsrc:$XB, u2imm:$ID, u1imm:$E, u2imm:$BL),
                               "dmxxshapad $AT, $XB, $ID, $E, $BL", []>,
                               RegConstraint<"$ATi = $AT">;
}

// MMA+ Intrinsics
let Predicates = [MMA, IsISAFuture] in {
  def : Pat<(v1024i1 (int_ppc_mma_dmxvi8gerx4 v256i1:$XAp, v16i8:$XB)),
            (DMXVI8GERX4 $XAp, RCCp.BToVSRC)>;
  def : Pat<(v1024i1 (int_ppc_mma_dmxvi8gerx4pp v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVI8GERX4PP $ATi, $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvi8gerx4spp v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVI8GERX4SPP $ATi, $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvbf16gerx2 v256i1:$XAp, v16i8:$XB)),
            (DMXVBF16GERX2 $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvbf16gerx2pp v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVBF16GERX2PP $ATi, $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvbf16gerx2pn v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVBF16GERX2PN $ATi, $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvbf16gerx2np v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVBF16GERX2NP $ATi, $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvbf16gerx2nn v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVBF16GERX2NN $ATi, $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvf16gerx2 v256i1:$XAp, v16i8:$XB)),
            (DMXVF16GERX2 $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvf16gerx2pp v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVF16GERX2PP $ATi, $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvf16gerx2pn v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVF16GERX2PN $ATi, $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvf16gerx2np v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVF16GERX2NP $ATi, $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvf16gerx2nn v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVF16GERX2NN $ATi, $XAp, RCCp.BToVSRC)>;
}

let Predicates = [MMA, PrefixInstrs, IsISAFuture] in {
  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvi8gerx4 v256i1:$XAp, v16i8:$XB, Msk8Imm:$XMSK,
                                            Msk4Imm:$YMSK, Msk4Imm:$PMSK)),
            (PMDMXVI8GERX4 $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                        Msk4Imm:$YMSK, Msk4Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvi8gerx4pp v1024i1:$ATi, v256i1:$XAp, v16i8:$XB,
                                              Msk8Imm:$XMSK, Msk4Imm:$YMSK,
                                              Msk4Imm:$PMSK)),
            (PMDMXVI8GERX4PP $ATi, $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                          Msk4Imm:$YMSK, Msk4Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvi8gerx4spp v1024i1:$ATi, v256i1:$XAp, v16i8:$XB,
                                               Msk8Imm:$XMSK, Msk4Imm:$YMSK,
                                               Msk4Imm:$PMSK)),
            (PMDMXVI8GERX4SPP $ATi, $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                           Msk4Imm:$YMSK, Msk4Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvbf16gerx2 v256i1:$XAp, v16i8:$XB, Msk8Imm:$XMSK,
                                            Msk4Imm:$YMSK, Msk2Imm:$PMSK)),
            (PMDMXVBF16GERX2 $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                        Msk4Imm:$YMSK, Msk2Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvbf16gerx2pp v1024i1:$ATi, v256i1:$XAp, v16i8:$XB,
                                              Msk8Imm:$XMSK, Msk4Imm:$YMSK,
                                              Msk2Imm:$PMSK)),
            (PMDMXVBF16GERX2PP $ATi, $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                          Msk4Imm:$YMSK, Msk2Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvbf16gerx2pn v1024i1:$ATi, v256i1:$XAp, v16i8:$XB,
                                               Msk8Imm:$XMSK, Msk4Imm:$YMSK,
                                               Msk2Imm:$PMSK)),
            (PMDMXVBF16GERX2PN $ATi, $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                           Msk4Imm:$YMSK, Msk2Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvbf16gerx2np v1024i1:$ATi, v256i1:$XAp, v16i8:$XB,
                                              Msk8Imm:$XMSK, Msk4Imm:$YMSK,
                                              Msk2Imm:$PMSK)),
            (PMDMXVBF16GERX2NP $ATi, $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                          Msk4Imm:$YMSK, Msk2Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvbf16gerx2nn v1024i1:$ATi, v256i1:$XAp, v16i8:$XB,
                                               Msk8Imm:$XMSK, Msk4Imm:$YMSK,
                                               Msk2Imm:$PMSK)),
            (PMDMXVBF16GERX2NN $ATi, $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                           Msk4Imm:$YMSK, Msk2Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvf16gerx2 v256i1:$XAp, v16i8:$XB, Msk8Imm:$XMSK,
                                             Msk4Imm:$YMSK, Msk2Imm:$PMSK)),
            (PMDMXVF16GERX2 $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                          Msk4Imm:$YMSK, Msk2Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvf16gerx2pp v1024i1:$ATi, v256i1:$XAp, v16i8:$XB,
                                              Msk8Imm:$XMSK, Msk4Imm:$YMSK,
                                              Msk2Imm:$PMSK)),
            (PMDMXVF16GERX2PP $ATi, $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                          Msk4Imm:$YMSK, Msk2Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvf16gerx2pn v1024i1:$ATi, v256i1:$XAp, v16i8:$XB,
                                               Msk8Imm:$XMSK, Msk4Imm:$YMSK,
                                               Msk2Imm:$PMSK)),
            (PMDMXVF16GERX2PN $ATi, $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                           Msk4Imm:$YMSK, Msk2Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvf16gerx2np v1024i1:$ATi, v256i1:$XAp, v16i8:$XB,
                                              Msk8Imm:$XMSK, Msk4Imm:$YMSK,
                                              Msk2Imm:$PMSK)),
            (PMDMXVF16GERX2NP $ATi, $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                          Msk4Imm:$YMSK, Msk2Imm:$PMSK)>;

  def : Pat<(v1024i1 (int_ppc_mma_pmdmxvf16gerx2nn v1024i1:$ATi, v256i1:$XAp, v16i8:$XB,
                                               Msk8Imm:$XMSK, Msk4Imm:$YMSK,
                                               Msk2Imm:$PMSK)),
            (PMDMXVF16GERX2NN $ATi, $XAp, RCCp.BToVSRC, Msk8Imm:$XMSK,
                           Msk4Imm:$YMSK, Msk2Imm:$PMSK)>;
}

// Cryptography Intrinsic
let Predicates = [IsISAFuture] in {
  def : Pat<(v1024i1 (int_ppc_mma_dmxxshapad v1024i1:$ATi, v16i8:$XB, timm:$ID,
                      timm:$E, timm:$BL)), (DMXXSHAPAD $ATi, RCCp.BToVSRC, $ID, $E, $BL)>;
}

// MMA+ Instruction aliases
let Predicates = [IsISAFuture] in {
  def : InstAlias<"dmsha256hash $AT, $AB",
                  (DMSHA2HASH dmr:$AT, dmr:$AB, 0)>;

  def : InstAlias<"dmsha512hash $AT, $AB",
                  (DMSHA2HASH dmr:$AT, dmr:$AB, 1)>;

  def : InstAlias<"dmsha3dw $ATp",
                  (DMSHA3HASH dmrp:$ATp, 0)>;

  def : InstAlias<"dmcryshash $ATp",
                  (DMSHA3HASH dmrp:$ATp, 12)>;

  def : InstAlias<"dmxxsha3512pad $AT, $XB, $E",
                  (DMXXSHAPAD dmr:$AT, vsrc:$XB, 0, u1imm:$E, 0)>;

  def : InstAlias<"dmxxsha3384pad $AT, $XB, $E",
                  (DMXXSHAPAD dmr:$AT, vsrc:$XB, 0, u1imm:$E, 1)>;

  def : InstAlias<"dmxxsha3256pad $AT, $XB, $E",
                  (DMXXSHAPAD dmr:$AT, vsrc:$XB, 0, u1imm:$E, 2)>;

  def : InstAlias<"dmxxsha3224pad $AT, $XB, $E",
                  (DMXXSHAPAD dmr:$AT, vsrc:$XB, 0, u1imm:$E, 3)>;

  def : InstAlias<"dmxxshake256pad $AT, $XB, $E",
                  (DMXXSHAPAD dmr:$AT, vsrc:$XB, 1, u1imm:$E, 0)>;

  def : InstAlias<"dmxxshake128pad $AT, $XB, $E",
                  (DMXXSHAPAD dmr:$AT, vsrc:$XB, 1, u1imm:$E, 1)>;

  def : InstAlias<"dmxxsha384512pad $AT, $XB",
                  (DMXXSHAPAD dmr:$AT, vsrc:$XB, 2, 0, 0)>;

  def : InstAlias<"dmxxsha224256pad $AT, $XB",
                  (DMXXSHAPAD dmr:$AT, vsrc:$XB, 3, 0, 0)>;
}
