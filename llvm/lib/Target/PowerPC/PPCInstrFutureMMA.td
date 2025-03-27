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
  bits<6> XAp;
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
    RegConstraint<"$ATi = $AT">, NoEncode<"$ATi">;
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
    RegConstraint<"$ATi = $AT">, NoEncode<"$ATi">;
  }
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

  def DMXXINSTFDMR512 : XX3Form_AT3_XABp5_P1<60, 234, (outs wacc:$AT),
                                             (ins vsrprc:$XAp, vsrprc:$XBp),
                                             "dmxxinstfdmr512 $AT, $XAp, $XBp, 0", []> {
    let P = 0;
  }

  def DMXXINSTFDMR512_HI : XX3Form_AT3_XABp5_P1<60, 234, (outs wacc_hi:$AT),
                                                (ins vsrprc:$XAp, vsrprc:$XBp),
                                                "dmxxinstfdmr512 $AT, $XAp, $XBp, 1", []> {
    let P = 1;
  }

  def DMXXEXTFDMR256 : XX2Form_AT3_XBp5_P2<60, 484, (outs vsrprc:$XBp),
                                           (ins dmrrowp:$AT, u2imm:$P),
                                           "dmxxextfdmr256 $XBp, $AT, $P", []>;

  def DMXXINSTFDMR256 : XX2Form_AT3_XBp5_P2<60, 485, (outs dmrrowp:$AT),
                                            (ins vsrprc:$XBp, u2imm:$P),
                                            "dmxxinstfdmr256 $AT, $XBp, $P", []>;

  def DMMR : XForm_ATB3<31, 6, 177, (outs dmr:$AT), (ins dmr:$AB),
                        "dmmr $AT, $AB", 
                        [(set v1024i1:$AT, (int_ppc_mma_dmmr v1024i1:$AB))]>;

  def DMXOR : XForm_ATB3<31, 7, 177, (outs dmr:$AT), (ins dmr:$ATi, dmr:$AB),
                         "dmxor $AT, $AB",
                         [(set v1024i1:$AT, (int_ppc_mma_dmxor v1024i1:$ATi, v1024i1:$AB))]>,
                         RegConstraint<"$ATi = $AT">, NoEncode<"$ATi">;

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
    RegConstraint<"$ATi = $AT">, NoEncode<"$ATi">;
}

let Predicates = [MMA, PrefixInstrs, IsISAFuture] in {
  def PMDMXVI8GERX4SPP :
    MMIRR_XX3Form_X8YP4_XAp5B6<59, 98, (outs dmr:$AT),
                            (ins dmr:$ATi, vsrprc:$XAp,vsrc:$XB, u8imm:$XMSK,
                             u4imm:$YMSK, u4imm:$PMSK),
                            "pmdmxvi8gerx4spp $AT, $XAp, $XB, $XMSK, $YMSK, $PMSK",
                            IIC_VecGeneral, []>,
    RegConstraint<"$ATi = $AT">, NoEncode<"$ATi">;
}

// MMA+ Intrinsics
let Predicates = [MMA, IsISAFuture] in {
  def : Pat<(v1024i1 (int_ppc_mma_dmxvi8gerx4 v256i1:$XAp, v16i8:$XB)),
            (DMXVI8GERX4 $XAp, RCCp.BToVSRC)>;
  def : Pat<(v1024i1 (int_ppc_mma_dmxvi8gerx4pp v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVI8GERX4PP $ATi, $XAp, RCCp.BToVSRC)>;

  def : Pat<(v1024i1 (int_ppc_mma_dmxvi8gerx4spp v1024i1:$ATi, v256i1:$XAp, v16i8:$XB)),
            (DMXVI8GERX4SPP $ATi, $XAp, RCCp.BToVSRC)>;
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
}
