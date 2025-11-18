//===-- PPCInstrFuture.td - Future Instruction Set --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the instructions introduced for the Future CPU.
//
//===----------------------------------------------------------------------===//

class XForm_RS5<bits<6> opcode, bits<10> xo, dag OOL, dag IOL, string asmstr,
                list<dag> pattern> : I<opcode, OOL, IOL, asmstr, NoItinerary> {
  bits<5> RS;

  let Pattern = pattern;

  let Inst{6...10} = RS;
  let Inst{11...20} = 0;
  let Inst{21...30} = xo;
  let Inst{31} = 0;
}

class XOForm_RTAB5_L1<bits<6> opcode, bits<9> xo, dag OOL, dag IOL,
                      string asmstr, list<dag> pattern>
    : I<opcode, OOL, IOL, asmstr, NoItinerary> {
  bits<5> RT;
  bits<5> RA;
  bits<5> RB;
  bit L;

  let Pattern = pattern;

  bit RC = 0; // set by isRecordForm

  let Inst{6...10} = RT;
  let Inst{11...15} = RA;
  let Inst{16...20} = RB;
  let Inst{21} = L;
  let Inst{22...30} = xo;
  let Inst{31} = RC;
}

multiclass XOForm_RTAB5_L1r<bits<6> opcode, bits<9> xo, dag OOL, dag IOL,
                            string asmbase, string asmstr, list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XOForm_RTAB5_L1<opcode, xo, OOL, IOL,
                               !strconcat(asmbase, !strconcat(" ", asmstr)),
                               pattern>,
               RecFormRel;
    let Defs = [CR0] in def _rec
        : XOForm_RTAB5_L1<opcode, xo, OOL, IOL,
                          !strconcat(asmbase, !strconcat(". ", asmstr)), []>,
        isRecordForm, RecFormRel;
  }
}

class VXForm_VRTB5_Base<bits<11> xo, dag OOL, dag IOL, string asmstr,
                        list<dag> pattern>
    : I<4, OOL, IOL, asmstr, NoItinerary> {
  bits<5> VRT;
  bits<5> VRB;

  let Pattern = pattern;

  let Inst{6...10} = VRT;
  let Inst{16...20} = VRB;
  let Inst{21...31} = xo;
}

class VXForm_VRTB5<bits<11> xo, bits<5> R, dag OOL, dag IOL, string asmstr,
                   list<dag> pattern>
    : VXForm_VRTB5_Base<xo, OOL, IOL, asmstr, pattern> {

  let Inst{11...15} = R;
}

class VXForm_VRTB5_UIM2<bits<11> xo, bits<3> R, dag OOL, dag IOL, string asmstr,
                        list<dag> pattern>
    : VXForm_VRTB5_Base<xo, OOL, IOL, asmstr, pattern> {
  bits<2> UIM;

  let Inst{11...13} = R;
  let Inst{14...15} = UIM;
}

class VXForm_VRTB5_UIM1<bits<11> xo, bits<4> R, dag OOL, dag IOL, string asmstr,
                        list<dag> pattern>
    : VXForm_VRTB5_Base<xo, OOL, IOL, asmstr, pattern> {
  bits<1> UIM;

  let Inst{11...14} = R;
  let Inst{15} = UIM;
}

class VXForm_VRTB5_UIM3<bits<11> xo, bits<2> R, dag OOL, dag IOL, string asmstr,
                        list<dag> pattern>
    : VXForm_VRTB5_Base<xo, OOL, IOL, asmstr, pattern> {
  bits<3> UIM;

  let Inst{11...12} = R;
  let Inst{13...15} = UIM;
}

class VXForm_VRTAB5<bits<11> xo, dag OOL, dag IOL, string asmstr,
                    list<dag> pattern>
    : VXForm_VRTB5_Base<xo, OOL, IOL, asmstr, pattern> {
  bits<5> VRA;

  let Inst{11...15} = VRA;
}

class XX3Form_XTBp5_M2<bits<9> xo, dag OOL, dag IOL, string asmstr,
                       list<dag> pattern>
    : I<60, OOL, IOL, asmstr, NoItinerary> {

  bits<5> XTp;
  bits<5> XBp;
  bits<2> M;

  let Pattern = pattern;

  let Inst{6...9} = XTp{3...0};
  let Inst {10} = XTp{4};
  let Inst{15} = M{0};
  let Inst{16...19} = XBp{3...0};
  let Inst{20} = M{1};
  let Inst{21...29} = xo;
  let Inst{30} = XBp{4};
}

class XX3Form_XTABp5_M2<bits<8> xo, dag OOL, dag IOL, string asmstr,
                        list<dag> pattern>
    : I<60, OOL, IOL, asmstr, NoItinerary> {

  bits<5> XTp;
  bits<5> XAp;
  bits<5> XBp;
  bits<2> M;

  let Pattern = pattern;

  let Inst{6...9} = XTp{3...0};
  let Inst{10} = XTp{4};
  let Inst{11...14} = XAp{3...0};
  let Inst{15} = M{0};
  let Inst{16...19} = XBp{3...0};
  let Inst{20} = M{1};
  let Inst{21...28} = xo;
  let Inst{29} = XAp{4};
  let Inst{30} = XBp{4};
}

class XX3Form_XTAB6_P1<bits<5> xo, dag OOL, dag IOL, string asmstr,
                       list<dag> pattern>
    : I<60, OOL, IOL, asmstr, NoItinerary> {

  bits<6> XT;
  bits<6> XA;
  bits<6> XB;
  bits<1> P;

  let Pattern = pattern;

  let Inst{6...10} = XT{4...0};
  let Inst{11...15} = XA{4...0};
  let Inst{16...20} = XB{4...0};
  let Inst{21...22} = 3;
  let Inst{23} = P;
  let Inst{24...28} = xo;
  let Inst{29} = XA{5};
  let Inst{30} = XB{5};
  let Inst{31} = XT{5};
}

class XX3Form_XTAB6<bits<6> opcode, bits<8> xo, dag OOL, dag IOL, string asmstr,
                    list<dag> pattern>
    : I<opcode, OOL, IOL, asmstr, NoItinerary> {

  bits<6> XT;
  bits<6> XA;
  bits<6> XB;

  let Pattern = pattern;

  let Inst{6...10} = XT{4...0};
  let Inst{11...15} = XA{4...0};
  let Inst{16...20} = XB{4...0};
  let Inst{21...28} = xo;
  let Inst{29} = XA{5};
  let Inst{30} = XB{5};
  let Inst{31} = XT{5};
}

class XForm_RBS5<bits<6> opCode, bits<10> xo, dag OOL, dag IOL, string asmstr,
                 InstrItinClass itin, list<dag> pattern>
    : I<opCode, OOL, IOL, asmstr, itin> {

  bits<5> RB;
  bits<5> RS;

  let Pattern = pattern;

  let Inst{6...10} = RS;
  let Inst{11...15} = 0;
  let Inst{16...20} = RB;
  let Inst{21...30} = xo;
  let Inst{31} = 0;
}

class XX3Form_XTAB6_S<bits<5> xo, dag OOL, dag IOL, string asmstr,
                       list<dag> pattern>
    : I<59, OOL, IOL, asmstr, NoItinerary> {
  bits<6> XT;
  bits<6> XA;
  bits<6> XB;

  let Pattern = pattern;

  let Inst{6...10} = XT{4...0};
  let Inst{11...15} = XA{4...0};
  let Inst{16...20} = XB{4...0};
  let Inst{24...28} = xo;
  let Inst{29} = XA{5};
  let Inst{30} = XB{5};
  let Inst{31} = XT{5};
}

class XX3Form_XTAB6_S3<bits<5> xo, dag OOL, dag IOL, string asmstr,
                       list<dag> pattern>
    : XX3Form_XTAB6_S<xo, OOL, IOL, asmstr, pattern> {

  bits<3> S;
  let Inst{21...23} = S;
}

class XX3Form_XTAB6_3S1<bits<5> xo, dag OOL, dag IOL, string asmstr,
                       list<dag> pattern>
    : XX3Form_XTAB6_S<xo, OOL, IOL, asmstr, pattern> {

  bits<1> S0;
  bits<1> S1;
  bits<1> S2;

  let Inst{21} = S0;
  let Inst{22} = S1;
  let Inst{23} = S2;
}

class XX3Form_XTAB6_2S1<bits<5> xo, dag OOL, dag IOL, string asmstr,
                       list<dag> pattern>
    : XX3Form_XTAB6_S<xo, OOL, IOL, asmstr, pattern> {

  bits<1> S1;
  bits<1> S2;

  let Inst{21} = 0;
  let Inst{22} = S1;
  let Inst{23} = S2;
}

class XX3Form_XTAB6_P<bits<7> xo, dag OOL, dag IOL, string asmstr,
                      list<dag> pattern>
    : I<59, OOL, IOL, asmstr, NoItinerary> {

  bits<6> XT;
  bits<6> XA;
  bits<6> XB;
  bits<1> P;

  let Pattern = pattern;

  let Inst{6...10} = XT{4...0};
  let Inst{11...15} = XA{4...0};
  let Inst{16...20} = XB{4...0};
  let Inst{21} = P;
  let Inst{22...28} = xo;
  let Inst{29} = XA{5};
  let Inst{30} = XB{5};
  let Inst{31} = XT{5};
}

// Prefix instruction classes.

class 8RR_XX4Form_XTABC6_P<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                           InstrItinClass itin, list<dag> pattern>
    : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<6> XT;
  bits<6> XA;
  bits<6> XB;
  bits<6> XC;
  bits<1> P;

  let Pattern = pattern;

  // The prefix.
  let Inst{6...7} = 1;
  let Inst{8...11} = 0;

  // The instruction.
  let Inst{38...42} = XT{4...0};
  let Inst{43...47} = XA{4...0};
  let Inst{48...52} = XB{4...0};
  let Inst{53...57} = XC{4...0};
  let Inst{58} = 1;
  let Inst{59} = P;
  let Inst{60} = XC{5};
  let Inst{61} = XA{5};
  let Inst{62} = XB{5};
  let Inst{63} = XT{5};
}

//-------------------------- Instruction definitions -------------------------//
// Predicate combinations available:
// [IsISAFuture]
// [HasVSX, IsISAFuture]
// [HasVSX, PrefixInstrs, IsISAFuture]

let Predicates = [IsISAFuture] in {
  defm SUBFUS : XOForm_RTAB5_L1r<31, 72, (outs g8rc:$RT),
                                 (ins g8rc:$RA, g8rc:$RB, u1imm:$L), "subfus",
                                 "$RT, $L, $RA, $RB", []>;
  def TLBSYNCIO
      : XForm_RS5<31, 564, (outs), (ins g8rc:$RS), "tlbsyncio $RS", []>;
  def PTESYNCIO
      : XForm_RS5<31, 596, (outs), (ins g8rc:$RS), "ptesyncio $RS", []>;
  def TLBIEP : XForm_RSB5_UIMM2_2UIMM1<31, 50, (outs),
                                       (ins gprc:$RB, gprc:$RS, u2imm:$RIC,
                                           u1imm:$PRS, u1imm:$R),
                                       "tlbiep $RB, $RS, $RIC, $PRS, $R", []>;
  def TLBIEIO
      : XForm_RSB5_UIMM2<31, 18, (outs), (ins g8rc:$RB, g8rc:$RS, u2imm:$RIC),
                         "tlbieio $RB, $RS, $RIC", []>;
  def MTLPL : XForm_RBS5<31, 275, (outs), (ins gprc:$RB, gprc:$RS),
                         "mtlpl $RB, $RS", IIC_SprMTSPR, []>;
  let Interpretation64Bit = 1, isCodeGenOnly = 1 in {
    def TLBIEP8
        : XForm_RSB5_UIMM2_2UIMM1<31, 50, (outs),
                                  (ins g8rc:$RB, g8rc:$RS, u2imm:$RIC,
                                      u1imm:$PRS, u1imm:$R),
                                  "tlbiep $RB, $RS, $RIC, $PRS, $R", []>;
    def MTLPL8 : XForm_RBS5<31, 275, (outs), (ins g8rc:$RB, g8rc:$RS),
                            "mtlpl $RB, $RS", IIC_SprMTSPR, []>, isPPC64;
  }
}

let Predicates = [HasVSX, IsISAFuture] in {
  let mayLoad = 1 in {
    def LXVRL : XX1Form_memOp<31, 525, (outs vsrc:$XT),
                              (ins (memr $RA):$addr, g8rc:$RB),
                              "lxvrl $XT, $addr, $RB", IIC_LdStLoad, []>;
    def LXVRLL : XX1Form_memOp<31, 557, (outs vsrc:$XT),
                               (ins (memr $RA):$addr, g8rc:$RB),
                               "lxvrll $XT, $addr, $RB", IIC_LdStLoad, []>;
    def LXVPRL : XForm_XTp5_RAB5<31, 589, (outs vsrprc:$XTp),
                                 (ins (memr $RA):$addr, g8rc:$RB),
                                 "lxvprl $XTp, $addr, $RB", IIC_LdStLFD, []>;
    def LXVPRLL : XForm_XTp5_RAB5<31, 621, (outs vsrprc:$XTp),
                                  (ins (memr $RA):$addr, g8rc:$RB),
                                  "lxvprll $XTp, $addr, $RB", IIC_LdStLFD, []>;
    def LXVPB32X
        : XForm_XTp5_RAB5<31, 877, (outs vsrprc:$XTp),
                          (ins (memr $RA):$addr, g8rc:$RB),
                          "lxvpb32x $XTp, $addr, $RB", IIC_LdStLFD, []>;
  }

  let mayStore = 1 in {
    def STXVRL : XX1Form_memOp<31, 653, (outs),
                               (ins vsrc:$XT, (memr $RA):$addr, g8rc:$RB),
                               "stxvrl $XT, $addr, $RB", IIC_LdStLoad, []>;
    def STXVRLL : XX1Form_memOp<31, 685, (outs),
                                (ins vsrc:$XT, (memr $RA):$addr, g8rc:$RB),
                                "stxvrll $XT, $addr, $RB", IIC_LdStLoad, []>;
    def STXVPRL : XForm_XTp5_RAB5<31, 717, (outs),
                                  (ins vsrprc:$XTp, (memr $RA):$addr, g8rc:$RB),
                                  "stxvprl $XTp, $addr, $RB", IIC_LdStLFD, []>;
    def STXVPRLL
        : XForm_XTp5_RAB5<31, 749, (outs),
                          (ins vsrprc:$XTp, (memr $RA):$addr, g8rc:$RB),
                          "stxvprll $XTp, $addr, $RB", IIC_LdStLFD, []>;
    def STXVPB32X
        : XForm_XTp5_RAB5<31, 1005, (outs),
                          (ins vsrprc:$XTp, (memr $RA):$addr, g8rc:$RB),
                          "stxvpb32x $XTp, $addr, $RB", IIC_LdStLFD, []>;
  }

  def VUPKHSNTOB : VXForm_VRTB5<387, 0, (outs vrrc:$VRT), (ins vrrc:$VRB),
                                "vupkhsntob $VRT, $VRB", []>;
  def VUPKLSNTOB : VXForm_VRTB5<387, 1, (outs vrrc:$VRT), (ins vrrc:$VRB),
                                "vupklsntob $VRT, $VRB", []>;
  def VUPKINT4TOBF16
      : VXForm_VRTB5_UIM2<387, 2, (outs vrrc:$VRT), (ins vrrc:$VRB, u2imm:$UIM),
                          "vupkint4tobf16 $VRT, $VRB, $UIM", []>;
  def VUPKINT8TOBF16
      : VXForm_VRTB5_UIM1<387, 1, (outs vrrc:$VRT), (ins vrrc:$VRB, u1imm:$UIM),
                          "vupkint8tobf16 $VRT, $VRB, $UIM", []>;
  def VUPKINT8TOFP32
      : VXForm_VRTB5_UIM2<387, 3, (outs vrrc:$VRT), (ins vrrc:$VRB, u2imm:$UIM),
                          "vupkint8tofp32 $VRT, $VRB, $UIM", []>;
  def VUPKINT4TOFP32
      : VXForm_VRTB5_UIM3<387, 2, (outs vrrc:$VRT), (ins vrrc:$VRB, u3imm:$UIM),
                          "vupkint4tofp32 $VRT, $VRB, $UIM", []>;

  def VUCMPRHN : VXForm_VRTAB5<3, (outs vrrc:$VRT), (ins vrrc:$VRA, vrrc:$VRB),
                               "vucmprhn $VRT, $VRA, $VRB", []>;
  def VUCMPRLN : VXForm_VRTAB5<67, (outs vrrc:$VRT), (ins vrrc:$VRA, vrrc:$VRB),
                               "vucmprln $VRT, $VRA, $VRB", []>;
  def VUCMPRHB
      : VXForm_VRTAB5<131, (outs vrrc:$VRT), (ins vrrc:$VRA, vrrc:$VRB),
                      "vucmprhb $VRT, $VRA, $VRB", []>;
  def VUCMPRLB
      : VXForm_VRTAB5<195, (outs vrrc:$VRT), (ins vrrc:$VRA, vrrc:$VRB),
                      "vucmprlb $VRT, $VRA, $VRB", []>;
  def VUCMPRHH
      : VXForm_VRTAB5<259, (outs vrrc:$VRT), (ins vrrc:$VRA, vrrc:$VRB),
                      "vucmprhh $VRT, $VRA, $VRB", []>;
  def VUCMPRLH
      : VXForm_VRTAB5<323, (outs vrrc:$VRT), (ins vrrc:$VRA, vrrc:$VRB),
                      "vucmprlh $VRT, $VRA, $VRB", []>;

  def XVRLW : XX3Form_XTAB6<60, 184, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                            "xvrlw $XT, $XA, $XB",
                            [(set v4i32:$XT, (int_ppc_vsx_xvrlw v4i32:$XA,
                                                 v4i32:$XB))]>;

  // AES Acceleration Instructions
  def XXAESENCP : XX3Form_XTABp5_M2<194, (outs vsrprc:$XTp),
                                    (ins vsrprc:$XAp, vsrprc:$XBp, u2imm:$M),
                                    "xxaesencp $XTp, $XAp, $XBp, $M", []>;
  def XXAESDECP : XX3Form_XTABp5_M2<202, (outs vsrprc:$XTp),
                                    (ins vsrprc:$XAp, vsrprc:$XBp, u2imm:$M),
                                    "xxaesdecp $XTp, $XAp, $XBp, $M", []>;
  def XXAESGENLKP : XX3Form_XTBp5_M2<420, (outs vsrprc:$XTp),
                                     (ins vsrprc:$XBp, u2imm:$M),
                                     "xxaesgenlkp $XTp, $XBp, $M", []>;
  def XXGFMUL128 : XX3Form_XTAB6_P1<26, (outs vsrc:$XT),
                                    (ins vsrc:$XA, vsrc:$XB, u1imm:$P),
                                    "xxgfmul128 $XT, $XA, $XB, $P", []>;

  // VSX Vector Integer Arithmetic Instructions
  def XVADDUWM : XX3Form_XTAB6<60, 131, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                               "xvadduwm $XT, $XA, $XB", []>;
  def XVADDUHM : XX3Form_XTAB6<60, 139, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                               "xvadduhm $XT, $XA, $XB", []>;
  def XVSUBUWM: XX3Form_XTAB6<60, 147, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvsubuwm $XT, $XA, $XB", []>;
  def XVSUBUHM: XX3Form_XTAB6<60, 155, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvsubuhm $XT, $XA, $XB", []>;
  def XVMULUWM: XX3Form_XTAB6<60, 163, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvmuluwm $XT, $XA, $XB", []>;
  def XVMULUHM: XX3Form_XTAB6<60, 171, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvmuluhm $XT, $XA, $XB", []>;
  def XVMULHSW: XX3Form_XTAB6<60, 179, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvmulhsw $XT, $XA, $XB", []>;
  def XVMULHSH: XX3Form_XTAB6<60, 187, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvmulhsh $XT, $XA, $XB", []>;
  def XVMULHUW: XX3Form_XTAB6<60, 114, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvmulhuw $XT, $XA, $XB", []>;
  def XVMULHUH: XX3Form_XTAB6<60, 122, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                              "xvmulhuh $XT, $XA, $XB", []>;

  // Elliptic Curve Cryptography Acceleration Instructions.
  def XXMULMUL
      : XX3Form_XTAB6_S3<1, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB, u3imm:$S),
                         "xxmulmul $XT, $XA, $XB, $S", []>;
  def XXMULMULHIADD
      : XX3Form_XTAB6_3S1<9, (outs vsrc:$XT),
                          (ins vsrc:$XA, vsrc:$XB, u1imm:$S0, u1imm:$S1,
                              u1imm:$S2),
                          "xxmulmulhiadd $XT, $XA, $XB, $S0, $S1, $S2", []>;
  def XXMULMULLOADD
      : XX3Form_XTAB6_2S1<17, (outs vsrc:$XT),
                          (ins vsrc:$XA, vsrc:$XB, u1imm:$S1, u1imm:$S2),
                          "xxmulmulloadd $XT, $XA, $XB, $S1, $S2", []>;
  def XXSSUMUDM
      : XX3Form_XTAB6_P<25, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB, u1imm:$P),
                        "xxssumudm $XT, $XA, $XB, $P", []>;
  def XXSSUMUDMC
      : XX3Form_XTAB6_P<57, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB, u1imm:$P),
                        "xxssumudmc $XT, $XA, $XB, $P", []>;
  def XSADDADDUQM
      : XX3Form_XTAB6<59, 96, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsaddadduqm $XT, $XA, $XB", []>;
  def XSADDADDSUQM
      : XX3Form_XTAB6<59, 104, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsaddaddsuqm $XT, $XA, $XB", []>;
  def XSADDSUBUQM
      : XX3Form_XTAB6<59, 112, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsaddsubuqm $XT, $XA, $XB", []>;
  def XSADDSUBSUQM
      : XX3Form_XTAB6<59, 224, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsaddsubsuqm $XT, $XA, $XB", []>;
  def XSMERGE2T1UQM
      : XX3Form_XTAB6<59, 232, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsmerge2t1uqm $XT, $XA, $XB", []>;
  def XSMERGE2T2UQM
      : XX3Form_XTAB6<59, 240, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsmerge2t2uqm $XT, $XA, $XB", []>;
  def XSMERGE2T3UQM
      : XX3Form_XTAB6<59, 89, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsmerge2t3uqm $XT, $XA, $XB", []>;
  def XSMERGE3T1UQM
      : XX3Form_XTAB6<59, 121, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsmerge3t1uqm $XT, $XA, $XB", []>;
  def XSREBASE2T1UQM
      : XX3Form_XTAB6<59, 145, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsrebase2t1uqm $XT, $XA, $XB", []>;
  def XSREBASE2T2UQM
      : XX3Form_XTAB6<59, 177, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsrebase2t2uqm $XT, $XA, $XB", []>;
  def XSREBASE2T3UQM
      : XX3Form_XTAB6<59, 209, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsrebase2t3uqm $XT, $XA, $XB", []>;
  def XSREBASE2T4UQM
      : XX3Form_XTAB6<59, 217, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsrebase2t4uqm $XT, $XA, $XB", []>;
  def XSREBASE3T1UQM
      : XX3Form_XTAB6<59, 241, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsrebase3t1uqm $XT, $XA, $XB", []>;
  def XSREBASE3T2UQM
      : XX3Form_XTAB6<59, 249, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsrebase3t2uqm $XT, $XA, $XB", []>;
  def XSREBASE3T3UQM
      : XX3Form_XTAB6<59, 195, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB),
                      "xsrebase3t3uqm $XT, $XA, $XB", []>;
}

let Predicates = [HasVSX, PrefixInstrs, IsISAFuture] in {
  def XXSSUMUDMCEXT
      : 8RR_XX4Form_XTABC6_P<
            34, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB, vsrc:$XC, u1imm:$P),
            "xxssumudmcext $XT, $XA, $XB, $XC, $P", IIC_VecGeneral, []>;
}

//---------------------------- Anonymous Patterns ----------------------------//
// Predicate combinations available:

// Load/Store VSX Vector with Right Length (Left-justified).
def : Pat<(v4i32 (int_ppc_vsx_lxvrl addr:$RA, i64:$RB)), (LXVRL $RA, $RB)>;
def : Pat<(v4i32 (int_ppc_vsx_lxvrll addr:$RA, i64:$RB)), (LXVRLL $RA, $RB)>;
def : Pat<(int_ppc_vsx_stxvrl v4i32:$XT, addr:$RA, i64:$RB), (STXVRL $XT, $RA,
                                                                 $RB)>;
def : Pat<(int_ppc_vsx_stxvrll v4i32:$XT, addr:$RA, i64:$RB), (STXVRLL $XT, $RA,
                                                                  $RB)>;

// Load/Store VSX Vector pair with Right Length (Left-justified).
def : Pat<(v256i1 (int_ppc_vsx_lxvprl addr:$RA, i64:$RB)), (LXVPRL $RA, $RB)>;
def : Pat<(v256i1 (int_ppc_vsx_lxvprll addr:$RA, i64:$RB)), (LXVPRLL $RA, $RB)>;
def : Pat<(int_ppc_vsx_stxvprl v256i1:$XTp, addr:$RA, i64:$RB), (STXVPRL $XTp,
                                                                    $RA, $RB)>;
def : Pat<(int_ppc_vsx_stxvprll v256i1:$XTp, addr:$RA, i64:$RB), (STXVPRLL $XTp,
                                                                     $RA, $RB)>;
let Predicates = [HasVSX, IsISAFuture] in {
  def : Pat<(v4i32 (rotl v4i32:$vA, v4i32:$vB)), (v4i32 (XVRLW v4i32:$vA,
                                                     v4i32:$vB))>;
}

//---------------------------- Instruction aliases ---------------------------//
// Predicate combinations available:
// [HasVSX, IsISAFuture]

let Predicates = [HasVSX, IsISAFuture] in {
  def : InstAlias<"xxaes128encp $XTp, $XAp, $XBp",
                  (XXAESENCP vsrprc:$XTp, vsrprc:$XAp, vsrprc:$XBp, 0)>;
  def : InstAlias<"xxaes192encp $XTp, $XAp, $XBp",
                  (XXAESENCP vsrprc:$XTp, vsrprc:$XAp, vsrprc:$XBp, 1)>;
  def : InstAlias<"xxaes256encp $XTp, $XAp, $XBp",
                  (XXAESENCP vsrprc:$XTp, vsrprc:$XAp, vsrprc:$XBp, 2)>;
  def : InstAlias<"xxaes128decp $XTp, $XAp, $XBp",
                  (XXAESDECP vsrprc:$XTp, vsrprc:$XAp, vsrprc:$XBp, 0)>;
  def : InstAlias<"xxaes192decp $XTp, $XAp, $XBp",
                  (XXAESDECP vsrprc:$XTp, vsrprc:$XAp, vsrprc:$XBp, 1)>;
  def : InstAlias<"xxaes256decp $XTp, $XAp, $XBp",
                  (XXAESDECP vsrprc:$XTp, vsrprc:$XAp, vsrprc:$XBp, 2)>;
  def : InstAlias<"xxaes128genlkp $XTp, $XBp", (XXAESGENLKP vsrprc:$XTp,
                                                   vsrprc:$XBp, 0)>;
  def : InstAlias<"xxaes192genlkp $XTp, $XBp", (XXAESGENLKP vsrprc:$XTp,
                                                   vsrprc:$XBp, 1)>;
  def : InstAlias<"xxaes256genlkp $XTp, $XBp", (XXAESGENLKP vsrprc:$XTp,
                                                   vsrprc:$XBp, 2)>;
  def : InstAlias<"xxgfmul128gcm $XT, $XA, $XB", (XXGFMUL128 vsrc:$XT, vsrc:$XA,
                                                     vsrc:$XB, 0)>;
  def : InstAlias<"xxgfmul128xts $XT, $XA, $XB", (XXGFMUL128 vsrc:$XT, vsrc:$XA,
                                                     vsrc:$XB, 1)>;
}
