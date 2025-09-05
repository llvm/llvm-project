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

class XOForm_RTAB5_L1<bits<6> opcode, bits<9> xo, dag OOL, dag IOL,
                      string asmstr, list<dag> pattern>
         : I<opcode, OOL, IOL, asmstr, NoItinerary> {
  bits<5> RT;
  bits<5> RA;
  bits<5> RB;
  bit L;

  let Pattern = pattern;

  bit RC = 0;    // set by isRecordForm

  let Inst{6...10}  = RT;
  let Inst{11...15} = RA;
  let Inst{16...20} = RB;
  let Inst{21}    = L;
  let Inst{22...30} = xo;
  let Inst{31}    = RC;
}

multiclass XOForm_RTAB5_L1r<bits<6> opcode, bits<9> xo, dag OOL, dag IOL,
                            string asmbase, string asmstr,
                            list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : XOForm_RTAB5_L1<opcode, xo, OOL, IOL,
                               !strconcat(asmbase, !strconcat(" ", asmstr)),
                               pattern>, RecFormRel;
    let Defs = [CR0] in
    def _rec : XOForm_RTAB5_L1<opcode, xo, OOL, IOL,
                               !strconcat(asmbase, !strconcat(". ", asmstr)),
                               []>, isRecordForm, RecFormRel;
  }
}

class VXForm_VRTB5<bits<11> xo, bits<5> R, dag OOL, dag IOL, string asmstr,
                   list<dag> pattern> : I<4, OOL, IOL, asmstr, NoItinerary> {
  bits<5> VRT;
  bits<5> VRB;

  let Pattern = pattern;

  let Inst{6...10} = VRT;
  let Inst{11...15} = R;
  let Inst{16...20} = VRB;
  let Inst{21...31} = xo;
}

class VXForm_VRTB5_UIM2<bits<11> xo, bits<3> R, dag OOL, dag IOL, string asmstr,
                        list<dag> pattern>
    : I<4, OOL, IOL, asmstr, NoItinerary> {
  bits<5> VRT;
  bits<5> VRB;
  bits<2> UIM;

  let Pattern = pattern;

  let Inst{6...10} = VRT;
  let Inst{11...13} = R;
  let Inst{14...15} = UIM;
  let Inst{16...20} = VRB;
  let Inst{21...31} = xo;
}

class VXForm_VRTB5_UIM1<bits<11> xo, bits<4> R, dag OOL, dag IOL, string asmstr,
                        list<dag> pattern>
    : I<4, OOL, IOL, asmstr, NoItinerary> {
  bits<5> VRT;
  bits<5> VRB;
  bits<1> UIM;

  let Pattern = pattern;

  let Inst{6...10} = VRT;
  let Inst{11...14} = R;
  let Inst{15} = UIM;
  let Inst{16...20} = VRB;
  let Inst{21...31} = xo;
}

class VXForm_VRTB5_UIM3<bits<11> xo, bits<2> R, dag OOL, dag IOL, string asmstr,
                        list<dag> pattern>
    : I<4, OOL, IOL, asmstr, NoItinerary> {
  bits<5> VRT;
  bits<5> VRB;
  bits<3> UIM;

  let Pattern = pattern;

  let Inst{6...10} = VRT;
  let Inst{11...12} = R;
  let Inst{13...15} = UIM;
  let Inst{16...20} = VRB;
  let Inst{21...31} = xo;
}

class VXForm_VRTAB5<bits<11> xo, dag OOL, dag IOL, string asmstr,
                    list<dag> pattern> : I<4, OOL, IOL, asmstr, NoItinerary> {
  bits<5> VRT;
  bits<5> VRA;
  bits<5> VRB;

  let Pattern = pattern;

  let Inst{6...10} = VRT;
  let Inst{11...15} = VRA;
  let Inst{16...20} = VRB;
  let Inst{21...31} = xo;
}

let Predicates = [IsISAFuture] in {
  defm SUBFUS : XOForm_RTAB5_L1r<31, 72, (outs g8rc:$RT),
                                 (ins g8rc:$RA, g8rc:$RB, u1imm:$L),
                                 "subfus",  "$RT, $L, $RA, $RB", []>;
}

let Predicates = [HasVSX, IsISAFuture] in {
  let mayLoad = 1 in {
    def LXVRL
        : XX1Form_memOp<31, 525, (outs vsrc:$XT), (ins (memr $RA):$addr, g8rc:$RB),
                        "lxvrl $XT, $addr, $RB", IIC_LdStLoad, []>;
    def LXVRLL
        : XX1Form_memOp<31, 557, (outs vsrc:$XT), (ins (memr $RA):$addr, g8rc:$RB),
                        "lxvrll $XT, $addr, $RB", IIC_LdStLoad, []>;
    def LXVPRL
        : XForm_XTp5_XAB5<31, 589, (outs vsrprc:$XTp), (ins (memr $RA):$addr, g8rc:$RB),
                          "lxvprl $XTp, $addr, $RB", IIC_LdStLFD, []>;
    def LXVPRLL
        : XForm_XTp5_XAB5<31, 621, (outs vsrprc:$XTp), (ins (memr $RA):$addr, g8rc:$RB),
                          "lxvprll $XTp, $addr, $RB", IIC_LdStLFD, []>;
  }

  let mayStore = 1 in {
    def STXVRL
        : XX1Form_memOp<31, 653, (outs), (ins vsrc:$XT, (memr $RA):$addr, g8rc:$RB),
                        "stxvrl $XT, $addr, $RB", IIC_LdStLoad, []>;
    def STXVRLL
        : XX1Form_memOp<31, 685, (outs), (ins vsrc:$XT, (memr $RA):$addr, g8rc:$RB),
                        "stxvrll $XT, $addr, $RB", IIC_LdStLoad, []>;
    def STXVPRL : XForm_XTp5_XAB5<31, 717, (outs),
                                  (ins vsrprc:$XTp, (memr $RA):$addr, g8rc:$RB),
                                  "stxvprl $XTp, $addr, $RB", IIC_LdStLFD, []>;
    def STXVPRLL : XForm_XTp5_XAB5<31, 749, (outs),
                                   (ins vsrprc:$XTp, (memr $RA):$addr, g8rc:$RB),
                                   "stxvprll $XTp, $addr, $RB", IIC_LdStLFD, []>;
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
}
