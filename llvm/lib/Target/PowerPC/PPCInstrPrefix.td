//===----------------------------------------------------------------------===//
// PowerPC ISA 3.1 specific type constraints.
//

def SDT_PPCSplat32 : SDTypeProfile<1, 3, [ SDTCisVT<0, v2i64>,
  SDTCisVec<1>, SDTCisInt<2>, SDTCisInt<3>
]>;

//===----------------------------------------------------------------------===//
// ISA 3.1 specific PPCISD nodes.
//

def PPCxxsplti32dx : SDNode<"PPCISD::XXSPLTI32DX", SDT_PPCSplat32, []>;

//===----------------------------------------------------------------------===//

// PC Relative flag (for instructions that use the address of the prefix for
// address computations).
class isPCRel { bit PCRel = 1; }

// Top-level class for prefixed instructions.
class PI<bits<6> pref, bits<6> opcode, dag OOL, dag IOL, string asmstr,
         InstrItinClass itin> : Instruction {
  field bits<64> Inst;
  field bits<64> SoftFail = 0;
  bit PCRel = 0; // Default value, set by isPCRel.
  let Size = 8;

  let Namespace = "PPC";
  let OutOperandList = OOL;
  let InOperandList = IOL;
  let AsmString = asmstr;
  let Itinerary = itin;
  let Inst{0-5} = pref;
  let Inst{32-37} = opcode;

  bits<1> PPC970_First = 0;
  bits<1> PPC970_Single = 0;
  bits<1> PPC970_Cracked = 0;
  bits<3> PPC970_Unit = 0;

  /// These fields correspond to the fields in PPCInstrInfo.h.  Any changes to
  /// these must be reflected there!  See comments there for what these are.
  let TSFlags{0}   = PPC970_First;
  let TSFlags{1}   = PPC970_Single;
  let TSFlags{2}   = PPC970_Cracked;
  let TSFlags{5-3} = PPC970_Unit;

  bits<1> Prefixed = 1;  // This is a prefixed instruction.
  let TSFlags{7}  = Prefixed;

  // For cases where multiple instruction definitions really represent the
  // same underlying instruction but with one definition for 64-bit arguments
  // and one for 32-bit arguments, this bit breaks the degeneracy between
  // the two forms and allows TableGen to generate mapping tables.
  bit Interpretation64Bit = 0;

  // Fields used for relation models.
  string BaseName = "";
}

// VX-Form: [ PO VT R VB RC XO ]
class VXForm_VTB5_RC<bits<10> xo, bits<5> R, dag OOL, dag IOL, string asmstr,
                      InstrItinClass itin, list<dag> pattern>
  : I<4, OOL, IOL, asmstr, itin> {
  bits<5> VT;
  bits<5> VB;
  bit RC = 0;

  let Pattern = pattern;

  let Inst{6-10} = VT;
  let Inst{11-15} = R;
  let Inst{16-20} = VB;
  let Inst{21} = RC;
  let Inst{22-31} = xo;
}

// Multiclass definition to account for record and non-record form
// instructions of VXRForm.
multiclass VXForm_VTB5_RCr<bits<10> xo, bits<5> R, dag OOL, dag IOL,
                            string asmbase, string asmstr,
                            InstrItinClass itin, list<dag> pattern> {
  let BaseName = asmbase in {
    def NAME : VXForm_VTB5_RC<xo, R, OOL, IOL,
                               !strconcat(asmbase, !strconcat(" ", asmstr)),
                               itin, pattern>, RecFormRel;
    let Defs = [CR6] in
    def _rec : VXForm_VTB5_RC<xo, R, OOL, IOL,
                               !strconcat(asmbase, !strconcat(". ", asmstr)),
                               itin, []>, isRecordForm, RecFormRel;
  }
}

class MLS_DForm_R_SI34_RTA5_MEM<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                                InstrItinClass itin, list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<5> FRS;
  bits<39> D_RA;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-7} = 2;
  let Inst{8-10} = 0;
  let Inst{11} = PCRel;
  let Inst{12-13} = 0;
  let Inst{14-31} = D_RA{33-16}; // d0

  // The instruction.
  let Inst{38-42} = FRS{4-0};
  let Inst{43-47} = D_RA{38-34}; // RA
  let Inst{48-63} = D_RA{15-0}; // d1
}

class MLS_DForm_R_SI34_RTA5<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                            InstrItinClass itin, list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<5> RT;
  bits<5> RA;
  bits<34> SI;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-7} = 2;
  let Inst{8-10} = 0;
  let Inst{11} = PCRel;
  let Inst{12-13} = 0;
  let Inst{14-31} = SI{33-16};

  // The instruction.
  let Inst{38-42} = RT;
  let Inst{43-47} = RA;
  let Inst{48-63} = SI{15-0};
}

class MLS_DForm_SI34_RT5<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                         InstrItinClass itin, list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<5> RT;
  bits<34> SI;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-7} = 2;
  let Inst{8-10} = 0;
  let Inst{11} = 0;
  let Inst{12-13} = 0;
  let Inst{14-31} = SI{33-16};

  // The instruction.
  let Inst{38-42} = RT;
  let Inst{43-47} = 0;
  let Inst{48-63} = SI{15-0};
}

multiclass MLS_DForm_R_SI34_RTA5_p<bits<6> opcode, dag OOL, dag IOL,
                                   dag PCRel_IOL, string asmstr,
                                   InstrItinClass itin> {
  def NAME : MLS_DForm_R_SI34_RTA5<opcode, OOL, IOL,
                                   !strconcat(asmstr, ", 0"), itin, []>;
  def pc : MLS_DForm_R_SI34_RTA5<opcode, OOL, PCRel_IOL,
                                 !strconcat(asmstr, ", 1"), itin, []>, isPCRel;
}

class 8LS_DForm_R_SI34_RTA5<bits<6> opcode, dag OOL, dag IOL, string asmstr,
                            InstrItinClass itin, list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<5> RT;
  bits<39> D_RA;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-10} = 0;
  let Inst{11} = PCRel;
  let Inst{12-13} = 0;
  let Inst{14-31} = D_RA{33-16}; // d0

  // The instruction.
  let Inst{38-42} = RT{4-0};
  let Inst{43-47} = D_RA{38-34}; // RA
  let Inst{48-63} = D_RA{15-0}; // d1
}

// 8LS:D-Form: [ 1 0 0 // R // d0
//               PO TX T RA d1 ]
class 8LS_DForm_R_SI34_XT6_RA5<bits<5> opcode, dag OOL, dag IOL, string asmstr,
                               InstrItinClass itin, list<dag> pattern>
  : PI<1, { opcode, ? }, OOL, IOL, asmstr, itin> {
  bits<6> XT;
  bits<39> D_RA;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-7} = 0;
  let Inst{8} = 0;
  let Inst{9-10} = 0; // reserved
  let Inst{11} = PCRel;
  let Inst{12-13} = 0; // reserved
  let Inst{14-31} = D_RA{33-16}; // d0

  // The instruction.
  let Inst{37} = XT{5};
  let Inst{38-42} = XT{4-0};
  let Inst{43-47} = D_RA{38-34}; // RA
  let Inst{48-63} = D_RA{15-0}; // d1
}

// X-Form: [PO T IMM VRB XO TX]
class XForm_XT6_IMM5_VB5<bits<6> opcode, bits<10> xo, dag OOL, dag IOL,
                         string asmstr, InstrItinClass itin, list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<6> XT;
  bits<5> VRB;
  bits<5> IMM;

  let Pattern = pattern;
  let Inst{6-10} = XT{4-0};
  let Inst{11-15} = IMM;
  let Inst{16-20} = VRB;
  let Inst{21-30} = xo;
  let Inst{31} = XT{5};
}

class 8RR_XX4Form_IMM8_XTAB6<bits<6> opcode, bits<2> xo,
                             dag OOL, dag IOL, string asmstr,
                             InstrItinClass itin, list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
    bits<6> XT;
    bits<6> XA;
    bits<6> XB;
    bits<6> XC;
    bits<8> IMM;

    let Pattern = pattern;

    // The prefix.
    let Inst{6-7} = 1;
    let Inst{8} = 0;
    let Inst{9-11} = 0;
    let Inst{12-13} = 0;
    let Inst{14-23} = 0;
    let Inst{24-31} = IMM;

    // The instruction.
    let Inst{38-42} = XT{4-0};
    let Inst{43-47} = XA{4-0};
    let Inst{48-52} = XB{4-0};
    let Inst{53-57} = XC{4-0};
    let Inst{58-59} = xo;
    let Inst{60} = XC{5};
    let Inst{61} = XA{5};
    let Inst{62} = XB{5};
    let Inst{63} = XT{5};
}

class VXForm_RD5_N3_VB5<bits<11> xo, dag OOL, dag IOL, string asmstr,
                        InstrItinClass itin, list<dag> pattern>
  : I<4, OOL, IOL, asmstr, itin> {
  bits<5> RD;
  bits<5> VB;
  bits<3> N;

  let Pattern = pattern;

  let Inst{6-10}  = RD;
  let Inst{11-12} = 0;
  let Inst{13-15} = N;
  let Inst{16-20} = VB;
  let Inst{21-31} = xo;
}


// VX-Form: [PO VRT RA VRB XO].
// Destructive (insert) forms are suffixed with _ins.
class VXForm_VTB5_RA5_ins<bits<11> xo, string opc, list<dag> pattern>
  : VXForm_1<xo, (outs vrrc:$vD), (ins vrrc:$vDi, gprc:$rA, vrrc:$vB),
             !strconcat(opc, " $vD, $rA, $vB"), IIC_VecGeneral, pattern>,
             RegConstraint<"$vDi = $vD">, NoEncode<"$vDi">;

// VX-Form: [PO VRT RA RB XO].
// Destructive (insert) forms are suffixed with _ins.
class VXForm_VRT5_RAB5_ins<bits<11> xo, string opc, list<dag> pattern>
  : VXForm_1<xo, (outs vrrc:$vD), (ins vrrc:$vDi, gprc:$rA, gprc:$rB),
             !strconcat(opc, " $vD, $rA, $rB"), IIC_VecGeneral, pattern>,
             RegConstraint<"$vDi = $vD">, NoEncode<"$vDi">;

// VX-Form: [ PO BF // VRA VRB XO ]
class VXForm_BF3_VAB5<bits<11> xo, dag OOL, dag IOL, string asmstr,
                      InstrItinClass itin, list<dag> pattern>
  : I<4, OOL, IOL, asmstr, itin> {
  bits<3> BF;
  bits<5> VA;
  bits<5> VB;

  let Pattern = pattern;

  let Inst{6-8} = BF;
  let Inst{9-10} = 0;
  let Inst{11-15} = VA;
  let Inst{16-20} = VB;
  let Inst{21-31} = xo;
}

// VN-Form: [PO VRT VRA VRB PS SD XO]
// SD is "Shift Direction"
class VNForm_VTAB5_SD3<bits<6> xo, bits<2> ps, dag OOL, dag IOL, string asmstr,
                       InstrItinClass itin, list<dag> pattern>
    : I<4, OOL, IOL, asmstr, itin> {
  bits<5> VRT;
  bits<5> VRA;
  bits<5> VRB;
  bits<3> SD;

  let Pattern = pattern;

  let Inst{6-10}  = VRT;
  let Inst{11-15} = VRA;
  let Inst{16-20} = VRB;
  let Inst{21-22} = ps;
  let Inst{23-25} = SD;
  let Inst{26-31} = xo;
}

class VXForm_RD5_MP_VB5<bits<11> xo, bits<4> eo, dag OOL, dag IOL,
                        string asmstr, InstrItinClass itin, list<dag> pattern>
  : I<4, OOL, IOL, asmstr, itin> {
  bits<5> RD;
  bits<5> VB;
  bit MP;

  let Pattern = pattern;

  let Inst{6-10}  = RD;
  let Inst{11-14} = eo;
  let Inst{15} = MP;
  let Inst{16-20} = VB;
  let Inst{21-31} = xo;
}

// 8RR:D-Form: [ 1 1 0 // // imm0
//               PO T XO TX imm1 ].
class 8RR_DForm_IMM32_XT6<bits<6> opcode, bits<4> xo, dag OOL, dag IOL,
                          string asmstr, InstrItinClass itin,
                          list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<6> XT;
  bits<32> IMM32;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-7} = 1;
  let Inst{8-11} = 0;
  let Inst{12-13} = 0; // reserved
  let Inst{14-15} = 0; // reserved
  let Inst{16-31} = IMM32{31-16};

  // The instruction.
  let Inst{38-42} = XT{4-0};
  let Inst{43-46} = xo;
  let Inst{47} = XT{5};
  let Inst{48-63} = IMM32{15-0};
}

// 8RR:D-Form: [ 1 1 0 // // imm0
//               PO T XO IX TX imm1 ].
class 8RR_DForm_IMM32_XT6_IX<bits<6> opcode, bits<3> xo, dag OOL, dag IOL,
                             string asmstr, InstrItinClass itin,
                             list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<6> XT;
  bit IX;
  bits<32> IMM32;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-7} = 1;
  let Inst{8-11} = 0;
  let Inst{12-13} = 0; // reserved
  let Inst{14-15} = 0; // reserved
  let Inst{16-31} = IMM32{31-16};

  // The instruction.
  let Inst{38-42} = XT{4-0};
  let Inst{43-45} = xo;
  let Inst{46} = IX;
  let Inst{47} = XT{5};
  let Inst{48-63} = IMM32{15-0};
}

class 8RR_XX4Form_XTABC6<bits<6> opcode, bits<2> xo, dag OOL, dag IOL,
                         string asmstr, InstrItinClass itin, list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<6> XT;
  bits<6> XA;
  bits<6> XB;
  bits<6> XC;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-7} = 1;
  let Inst{8-11} = 0;
  let Inst{12-13} = 0;
  let Inst{14-31} = 0;

  // The instruction.
  let Inst{38-42} = XT{4-0};
  let Inst{43-47} = XA{4-0};
  let Inst{48-52} = XB{4-0};
  let Inst{53-57} = XC{4-0};
  let Inst{58-59} = xo;
  let Inst{60} = XC{5};
  let Inst{61} = XA{5};
  let Inst{62} = XB{5};
  let Inst{63} = XT{5};
}

class 8RR_XX4Form_IMM3_XTABC6<bits<6> opcode, bits<2> xo, dag OOL, dag IOL,
                              string asmstr, InstrItinClass itin,
                              list<dag> pattern>
  : PI<1, opcode, OOL, IOL, asmstr, itin> {
  bits<6> XT;
  bits<6> XA;
  bits<6> XB;
  bits<6> XC;
  bits<3> IMM;

  let Pattern = pattern;

  // The prefix.
  let Inst{6-7} = 1;
  let Inst{8-11} = 0;
  let Inst{12-13} = 0;
  let Inst{14-28} = 0;
  let Inst{29-31} = IMM;

  // The instruction.
  let Inst{38-42} = XT{4-0};
  let Inst{43-47} = XA{4-0};
  let Inst{48-52} = XB{4-0};
  let Inst{53-57} = XC{4-0};
  let Inst{58-59} = xo;
  let Inst{60} = XC{5};
  let Inst{61} = XA{5};
  let Inst{62} = XB{5};
  let Inst{63} = XT{5};
}

// [PO BF / XO2 B XO BX /]
class XX2_BF3_XO5_XB6_XO9<bits<6> opcode, bits<5> xo2, bits<9> xo, dag OOL,
                          dag IOL, string asmstr, InstrItinClass itin,
                          list<dag> pattern>
  : I<opcode, OOL, IOL, asmstr, itin> {
  bits<3> BF;
  bits<6> XB;

  let Pattern = pattern;

  let Inst{6-8}   = BF;
  let Inst{9-10}  = 0;
  let Inst{11-15} = xo2;
  let Inst{16-20} = XB{4-0};
  let Inst{21-29} = xo;
  let Inst{30}    = XB{5};
  let Inst{31}    = 0;
}

multiclass MLS_DForm_R_SI34_RTA5_MEM_p<bits<6> opcode, dag OOL, dag IOL,
                                       dag PCRel_IOL, string asmstr,
                                       InstrItinClass itin> {
  def NAME : MLS_DForm_R_SI34_RTA5_MEM<opcode, OOL, IOL,
                                       !strconcat(asmstr, ", 0"), itin, []>;
  def pc : MLS_DForm_R_SI34_RTA5_MEM<opcode, OOL, PCRel_IOL,
                                     !strconcat(asmstr, ", 1"), itin, []>,
                                     isPCRel;
}

multiclass 8LS_DForm_R_SI34_RTA5_p<bits<6> opcode, dag OOL, dag IOL,
                                   dag PCRel_IOL, string asmstr,
                                   InstrItinClass itin> {
  def NAME : 8LS_DForm_R_SI34_RTA5<opcode, OOL, IOL,
                                   !strconcat(asmstr, ", 0"), itin, []>;
  def pc : 8LS_DForm_R_SI34_RTA5<opcode, OOL, PCRel_IOL,
                                 !strconcat(asmstr, ", 1"), itin, []>, isPCRel;
}

multiclass 8LS_DForm_R_SI34_XT6_RA5_p<bits<5> opcode, dag OOL, dag IOL,
                                      dag PCRel_IOL, string asmstr,
                                      InstrItinClass itin> {
  def NAME : 8LS_DForm_R_SI34_XT6_RA5<opcode, OOL, IOL,
                                      !strconcat(asmstr, ", 0"), itin, []>;
  def pc : 8LS_DForm_R_SI34_XT6_RA5<opcode, OOL, PCRel_IOL,
                                    !strconcat(asmstr, ", 1"), itin, []>,
                                    isPCRel;
}

def PrefixInstrs : Predicate<"Subtarget->hasPrefixInstrs()">;
def IsISA3_1 : Predicate<"Subtarget->isISA3_1()">;
def PairedVectorMemops : Predicate<"PPCSubTarget->pairedVectorMemops()">;

let Predicates = [PrefixInstrs] in {
  let Interpretation64Bit = 1, isCodeGenOnly = 1 in {
    defm PADDI8 :
      MLS_DForm_R_SI34_RTA5_p<14, (outs g8rc:$RT), (ins g8rc:$RA, s34imm:$SI),
                              (ins immZero:$RA, s34imm_pcrel:$SI),
                              "paddi $RT, $RA, $SI", IIC_LdStLFD>;
    let isReMaterializable = 1, isAsCheapAsAMove = 1, isMoveImm = 1 in {
      def PLI8 : MLS_DForm_SI34_RT5<14, (outs g8rc:$RT),
                                    (ins s34imm:$SI),
                                    "pli $RT, $SI", IIC_IntSimple, []>;
    }
  }
  defm PADDI :
    MLS_DForm_R_SI34_RTA5_p<14, (outs gprc:$RT), (ins gprc:$RA, s34imm:$SI),
                            (ins immZero:$RA, s34imm_pcrel:$SI),
                            "paddi $RT, $RA, $SI", IIC_LdStLFD>;
  let isReMaterializable = 1, isAsCheapAsAMove = 1, isMoveImm = 1 in {
    def PLI : MLS_DForm_SI34_RT5<14, (outs gprc:$RT),
                                 (ins s34imm:$SI),
                                 "pli $RT, $SI", IIC_IntSimple, []>;
  }

  let mayLoad = 1, mayStore = 0 in {
    defm PLXV :
      8LS_DForm_R_SI34_XT6_RA5_p<25, (outs vsrc:$XT), (ins memri34:$D_RA),
                                 (ins memri34_pcrel:$D_RA), "plxv $XT, $D_RA",
                                 IIC_LdStLFD>;
    defm PLFS :
      MLS_DForm_R_SI34_RTA5_MEM_p<48, (outs f4rc:$FRT), (ins memri34:$D_RA),
                                  (ins memri34_pcrel:$D_RA), "plfs $FRT, $D_RA",
                                  IIC_LdStLFD>;
    defm PLFD :
      MLS_DForm_R_SI34_RTA5_MEM_p<50, (outs f8rc:$FRT), (ins memri34:$D_RA),
                                  (ins  memri34_pcrel:$D_RA), "plfd $FRT, $D_RA",
                                  IIC_LdStLFD>;
    defm PLXSSP :
      8LS_DForm_R_SI34_RTA5_p<43, (outs vfrc:$VRT), (ins memri34:$D_RA),
                              (ins memri34_pcrel:$D_RA), "plxssp $VRT, $D_RA",
                              IIC_LdStLFD>;
    defm PLXSD :
      8LS_DForm_R_SI34_RTA5_p<42, (outs vfrc:$VRT), (ins memri34:$D_RA),
                              (ins memri34_pcrel:$D_RA), "plxsd $VRT, $D_RA",
                              IIC_LdStLFD>;
    let Interpretation64Bit = 1, isCodeGenOnly = 1 in {
      defm PLBZ8 :
        MLS_DForm_R_SI34_RTA5_MEM_p<34, (outs g8rc:$RT), (ins memri34:$D_RA),
                                    (ins memri34_pcrel:$D_RA), "plbz $RT, $D_RA",
                                    IIC_LdStLFD>;
      defm PLHZ8 :
        MLS_DForm_R_SI34_RTA5_MEM_p<40, (outs g8rc:$RT), (ins memri34:$D_RA),
                                    (ins memri34_pcrel:$D_RA), "plhz $RT, $D_RA",
                                    IIC_LdStLFD>;
      defm PLHA8 :
        MLS_DForm_R_SI34_RTA5_MEM_p<42, (outs g8rc:$RT), (ins memri34:$D_RA),
                                    (ins memri34_pcrel:$D_RA), "plha $RT, $D_RA",
                                    IIC_LdStLFD>;
      defm PLWA8 :
        8LS_DForm_R_SI34_RTA5_p<41, (outs g8rc:$RT), (ins memri34:$D_RA),
                                (ins memri34_pcrel:$D_RA), "plwa $RT, $D_RA",
                                IIC_LdStLFD>;
      defm PLWZ8 :
        MLS_DForm_R_SI34_RTA5_MEM_p<32, (outs g8rc:$RT), (ins memri34:$D_RA),
                                    (ins memri34_pcrel:$D_RA), "plwz $RT, $D_RA",
                                    IIC_LdStLFD>;
    }
    defm PLBZ :
      MLS_DForm_R_SI34_RTA5_MEM_p<34, (outs gprc:$RT), (ins memri34:$D_RA),
                                  (ins memri34_pcrel:$D_RA), "plbz $RT, $D_RA",
                                  IIC_LdStLFD>;
    defm PLHZ :
      MLS_DForm_R_SI34_RTA5_MEM_p<40, (outs gprc:$RT), (ins memri34:$D_RA),
                                  (ins memri34_pcrel:$D_RA), "plhz $RT, $D_RA",
                                  IIC_LdStLFD>;
    defm PLHA :
      MLS_DForm_R_SI34_RTA5_MEM_p<42, (outs gprc:$RT), (ins memri34:$D_RA),
                                  (ins memri34_pcrel:$D_RA), "plha $RT, $D_RA",
                                  IIC_LdStLFD>;
    defm PLWZ :
      MLS_DForm_R_SI34_RTA5_MEM_p<32, (outs gprc:$RT), (ins memri34:$D_RA),
                                  (ins memri34_pcrel:$D_RA), "plwz $RT, $D_RA",
                                  IIC_LdStLFD>;
    defm PLWA :
      8LS_DForm_R_SI34_RTA5_p<41, (outs gprc:$RT), (ins memri34:$D_RA),
                              (ins memri34_pcrel:$D_RA), "plwa $RT, $D_RA",
                              IIC_LdStLFD>;
    defm PLD :
      8LS_DForm_R_SI34_RTA5_p<57, (outs g8rc:$RT), (ins memri34:$D_RA),
                              (ins memri34_pcrel:$D_RA), "pld $RT, $D_RA",
                              IIC_LdStLFD>;
  }

  let mayStore = 1, mayLoad = 0 in {
    defm PSTXV :
      8LS_DForm_R_SI34_XT6_RA5_p<27, (outs), (ins vsrc:$XS, memri34:$D_RA),
                                 (ins vsrc:$XS, memri34_pcrel:$D_RA),
                                 "pstxv $XS, $D_RA", IIC_LdStLFD>;
    defm PSTFS :
      MLS_DForm_R_SI34_RTA5_MEM_p<52, (outs), (ins f4rc:$FRS, memri34:$D_RA),
                                  (ins f4rc:$FRS, memri34_pcrel:$D_RA),
                                  "pstfs $FRS, $D_RA", IIC_LdStLFD>;
    defm PSTFD :
      MLS_DForm_R_SI34_RTA5_MEM_p<54, (outs), (ins f8rc:$FRS, memri34:$D_RA),
                                  (ins f8rc:$FRS, memri34_pcrel:$D_RA),
                                  "pstfd $FRS, $D_RA", IIC_LdStLFD>;
    defm PSTXSSP :
      8LS_DForm_R_SI34_RTA5_p<47, (outs), (ins vfrc:$VRS, memri34:$D_RA),
                              (ins vfrc:$VRS, memri34_pcrel:$D_RA),
                              "pstxssp $VRS, $D_RA", IIC_LdStLFD>;
    defm PSTXSD :
      8LS_DForm_R_SI34_RTA5_p<46, (outs), (ins vfrc:$VRS, memri34:$D_RA),
                              (ins vfrc:$VRS, memri34_pcrel:$D_RA),
                              "pstxsd $VRS, $D_RA", IIC_LdStLFD>;
    let Interpretation64Bit = 1, isCodeGenOnly = 1 in {
      defm PSTB8 :
        MLS_DForm_R_SI34_RTA5_MEM_p<38, (outs), (ins g8rc:$RS, memri34:$D_RA),
                                    (ins g8rc:$RS, memri34_pcrel:$D_RA),
                                    "pstb $RS, $D_RA", IIC_LdStLFD>;
      defm PSTH8 :
        MLS_DForm_R_SI34_RTA5_MEM_p<44, (outs), (ins g8rc:$RS, memri34:$D_RA),
                                    (ins g8rc:$RS, memri34_pcrel:$D_RA),
                                    "psth $RS, $D_RA", IIC_LdStLFD>;
      defm PSTW8 :
        MLS_DForm_R_SI34_RTA5_MEM_p<36, (outs), (ins g8rc:$RS, memri34:$D_RA),
                                    (ins g8rc:$RS, memri34_pcrel:$D_RA),
                                    "pstw $RS, $D_RA", IIC_LdStLFD>;
    }
    defm PSTB :
      MLS_DForm_R_SI34_RTA5_MEM_p<38, (outs), (ins gprc:$RS, memri34:$D_RA),
                                  (ins gprc:$RS, memri34_pcrel:$D_RA),
                                  "pstb $RS, $D_RA", IIC_LdStLFD>;
    defm PSTH :
      MLS_DForm_R_SI34_RTA5_MEM_p<44, (outs), (ins gprc:$RS, memri34:$D_RA),
                                  (ins gprc:$RS, memri34_pcrel:$D_RA),
                                  "psth $RS, $D_RA", IIC_LdStLFD>;
    defm PSTW :
      MLS_DForm_R_SI34_RTA5_MEM_p<36, (outs), (ins gprc:$RS, memri34:$D_RA),
                                  (ins gprc:$RS, memri34_pcrel:$D_RA),
                                  "pstw $RS, $D_RA", IIC_LdStLFD>;
    defm PSTD :
      8LS_DForm_R_SI34_RTA5_p<61, (outs), (ins g8rc:$RS, memri34:$D_RA),
                              (ins g8rc:$RS, memri34_pcrel:$D_RA),
                              "pstd $RS, $D_RA", IIC_LdStLFD>;
  }
}

// TODO: We have an added complexity of 500 here. This is only a temporary
// solution to have tablegen consider these patterns first. The way we do
// addressing for PowerPC is complex depending on available D form, X form, or
// aligned D form loads/stores like DS and DQ forms. The prefixed
// instructions in this file also add additional PC Relative loads/stores
// and D form loads/stores with 34 bit immediates. It is very difficult to force
// instruction selection to consistently pick these first without the current
// added complexity. Once pc-relative implementation is complete, a set of
// follow-up patches will address this refactoring and the AddedComplexity will
// be removed.
let Predicates = [PCRelativeMemops], AddedComplexity = 500 in {
  // Load i32
  def : Pat<(i32 (zextloadi8  (PPCmatpcreladdr pcreladdr:$ga))),
            (PLBZpc $ga, 0)>;
  def : Pat<(i32 (extloadi8   (PPCmatpcreladdr pcreladdr:$ga))),
            (PLBZpc $ga, 0)>;
  def : Pat<(i32 (sextloadi16 (PPCmatpcreladdr pcreladdr:$ga))),
            (PLHApc $ga, 0)>;
  def : Pat<(i32 (zextloadi16 (PPCmatpcreladdr pcreladdr:$ga))),
            (PLHZpc $ga, 0)>;
  def : Pat<(i32 (extloadi16 (PPCmatpcreladdr pcreladdr:$ga))),
            (PLHZpc $ga, 0)>;
  def : Pat<(i32 (load (PPCmatpcreladdr pcreladdr:$ga))), (PLWZpc $ga, 0)>;

  // Store i32
  def : Pat<(truncstorei8 i32:$RS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTBpc $RS, $ga, 0)>;
  def : Pat<(truncstorei16 i32:$RS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTHpc $RS, $ga, 0)>;
  def : Pat<(store i32:$RS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTWpc $RS, $ga, 0)>;

  // Load i64
  def : Pat<(i64 (zextloadi8  (PPCmatpcreladdr pcreladdr:$ga))),
            (PLBZ8pc $ga, 0)>;
  def : Pat<(i64 (extloadi8   (PPCmatpcreladdr pcreladdr:$ga))),
            (PLBZ8pc $ga, 0)>;
  def : Pat<(i64 (sextloadi16 (PPCmatpcreladdr pcreladdr:$ga))),
            (PLHA8pc $ga, 0)>;
  def : Pat<(i64 (zextloadi16 (PPCmatpcreladdr pcreladdr:$ga))),
            (PLHZ8pc $ga, 0)>;
  def : Pat<(i64 (extloadi16 (PPCmatpcreladdr pcreladdr:$ga))),
            (PLHZ8pc $ga, 0)>;
  def : Pat<(i64 (zextloadi32 (PPCmatpcreladdr pcreladdr:$ga))),
            (PLWZ8pc $ga, 0)>;
  def : Pat<(i64 (sextloadi32 (PPCmatpcreladdr pcreladdr:$ga))),
            (PLWA8pc $ga, 0)>;
  def : Pat<(i64 (extloadi32 (PPCmatpcreladdr pcreladdr:$ga))),
            (PLWZ8pc $ga, 0)>;
  def : Pat<(i64 (load (PPCmatpcreladdr pcreladdr:$ga))), (PLDpc $ga, 0)>;

  // Store i64
  def : Pat<(truncstorei8 i64:$RS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTB8pc $RS, $ga, 0)>;
  def : Pat<(truncstorei16 i64:$RS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTH8pc $RS, $ga, 0)>;
  def : Pat<(truncstorei32 i64:$RS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTW8pc $RS, $ga, 0)>;
  def : Pat<(store i64:$RS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTDpc $RS, $ga, 0)>;

  // Load f32
  def : Pat<(f32 (load (PPCmatpcreladdr pcreladdr:$addr))), (PLFSpc $addr, 0)>;

  // Store f32
  def : Pat<(store f32:$FRS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTFSpc $FRS, $ga, 0)>;

  // Load f64
  def : Pat<(f64 (extloadf32 (PPCmatpcreladdr pcreladdr:$addr))),
            (COPY_TO_REGCLASS (PLFSpc $addr, 0), VSFRC)>;
  def : Pat<(f64 (load (PPCmatpcreladdr pcreladdr:$addr))), (PLFDpc $addr, 0)>;

  // Store f64
  def : Pat<(store f64:$FRS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTFDpc $FRS, $ga, 0)>;

  // Load f128
  def : Pat<(f128 (load (PPCmatpcreladdr pcreladdr:$addr))),
            (COPY_TO_REGCLASS (PLXVpc $addr, 0), VRRC)>;

  // Store f128
  def : Pat<(store f128:$XS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTXVpc (COPY_TO_REGCLASS $XS, VSRC), $ga, 0)>;

  // Load v4i32
  def : Pat<(v4i32 (load (PPCmatpcreladdr pcreladdr:$addr))), (PLXVpc $addr, 0)>;

  // Store v4i32
  def : Pat<(store v4i32:$XS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTXVpc $XS, $ga, 0)>;

  // Load v2i64
  def : Pat<(v2i64 (load (PPCmatpcreladdr pcreladdr:$addr))), (PLXVpc $addr, 0)>;

  // Store v2i64
  def : Pat<(store v2i64:$XS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTXVpc $XS, $ga, 0)>;

  // Load v4f32
  def : Pat<(v4f32 (load (PPCmatpcreladdr pcreladdr:$addr))), (PLXVpc $addr, 0)>;

  // Store v4f32
  def : Pat<(store v4f32:$XS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTXVpc $XS, $ga, 0)>;

  // Load v2f64
  def : Pat<(v2f64 (load (PPCmatpcreladdr pcreladdr:$addr))), (PLXVpc $addr, 0)>;

  // Store v2f64
  def : Pat<(store v2f64:$XS, (PPCmatpcreladdr pcreladdr:$ga)),
            (PSTXVpc $XS, $ga, 0)>;

  // Atomic Load
  def : Pat<(atomic_load_8 (PPCmatpcreladdr pcreladdr:$ga)),
            (PLBZpc $ga, 0)>;
  def : Pat<(atomic_load_16 (PPCmatpcreladdr pcreladdr:$ga)),
            (PLHZpc $ga, 0)>;
  def : Pat<(atomic_load_32 (PPCmatpcreladdr pcreladdr:$ga)),
            (PLWZpc $ga, 0)>;
  def : Pat<(atomic_load_64 (PPCmatpcreladdr pcreladdr:$ga)),
            (PLDpc $ga, 0)>;

  // Atomic Store
  def : Pat<(atomic_store_8 (PPCmatpcreladdr pcreladdr:$ga), i32:$RS),
            (PSTBpc $RS, $ga, 0)>;
  def : Pat<(atomic_store_16 (PPCmatpcreladdr pcreladdr:$ga), i32:$RS),
            (PSTHpc $RS, $ga, 0)>;
  def : Pat<(atomic_store_32 (PPCmatpcreladdr pcreladdr:$ga), i32:$RS),
            (PSTWpc $RS, $ga, 0)>;
  def : Pat<(atomic_store_8 (PPCmatpcreladdr pcreladdr:$ga), i64:$RS),
            (PSTB8pc $RS, $ga, 0)>;
  def : Pat<(atomic_store_16 (PPCmatpcreladdr pcreladdr:$ga), i64:$RS),
            (PSTH8pc $RS, $ga, 0)>;
  def : Pat<(atomic_store_32 (PPCmatpcreladdr pcreladdr:$ga), i64:$RS),
            (PSTW8pc $RS, $ga, 0)>;
  def : Pat<(atomic_store_64 (PPCmatpcreladdr pcreladdr:$ga), i64:$RS),
            (PSTDpc $RS, $ga, 0)>;

  // Special Cases For PPCstore_scal_int_from_vsr
  def : Pat<(PPCstore_scal_int_from_vsr
              (f64 (PPCcv_fp_to_sint_in_vsr f64:$src)),
              (PPCmatpcreladdr pcreladdr:$dst), 8),
            (PSTXSDpc (XSCVDPSXDS f64:$src), $dst, 0)>;
  def : Pat<(PPCstore_scal_int_from_vsr
              (f64 (PPCcv_fp_to_sint_in_vsr f128:$src)),
              (PPCmatpcreladdr pcreladdr:$dst), 8),
            (PSTXSDpc (COPY_TO_REGCLASS (XSCVQPSDZ f128:$src), VFRC), $dst, 0)>;

  def : Pat<(PPCstore_scal_int_from_vsr
              (f64 (PPCcv_fp_to_uint_in_vsr f64:$src)),
              (PPCmatpcreladdr pcreladdr:$dst), 8),
            (PSTXSDpc (XSCVDPUXDS f64:$src), $dst, 0)>;
  def : Pat<(PPCstore_scal_int_from_vsr
              (f64 (PPCcv_fp_to_uint_in_vsr f128:$src)),
              (PPCmatpcreladdr pcreladdr:$dst), 8),
            (PSTXSDpc (COPY_TO_REGCLASS (XSCVQPUDZ f128:$src), VFRC), $dst, 0)>;

  // If the PPCmatpcreladdr node is not caught by any other pattern it should be
  // caught here and turned into a paddi instruction to materialize the address.
  def : Pat<(PPCmatpcreladdr pcreladdr:$addr), (PADDI8pc 0, $addr)>;
}

let Predicates = [PrefixInstrs] in {
  def XXSPLTIW : 8RR_DForm_IMM32_XT6<32, 3, (outs vsrc:$XT),
                                     (ins i32imm:$IMM32),
                                     "xxspltiw $XT, $IMM32", IIC_VecGeneral,
                                     []>;
  def XXSPLTIDP : 8RR_DForm_IMM32_XT6<32, 2, (outs vsrc:$XT),
                                      (ins i32imm:$IMM32),
                                      "xxspltidp $XT, $IMM32", IIC_VecGeneral,
                                      [(set v2f64:$XT,
                                            (PPCxxspltidp i32:$IMM32))]>;
  def XXSPLTI32DX :
      8RR_DForm_IMM32_XT6_IX<32, 0, (outs vsrc:$XT),
                             (ins vsrc:$XTi, u1imm:$IX, i32imm:$IMM32),
                             "xxsplti32dx $XT, $IX, $IMM32", IIC_VecGeneral,
                             [(set v2i64:$XT,
                                   (PPCxxsplti32dx v2i64:$XTi, i32:$IX,
                                                   i32:$IMM32))]>,
                             RegConstraint<"$XTi = $XT">, NoEncode<"$XTi">;
  def XXPERMX :
    8RR_XX4Form_IMM3_XTABC6<34, 0, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB,
                            vsrc:$XC, u3imm:$UIM),
                            "xxpermx $XT, $XA, $XB, $XC, $UIM",
                            IIC_VecPerm, []>;
  def XXBLENDVB :
    8RR_XX4Form_XTABC6<33, 0, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB,
                       vsrc:$XC), "xxblendvb $XT, $XA, $XB, $XC",
                       IIC_VecGeneral, []>;
  def XXBLENDVH :
    8RR_XX4Form_XTABC6<33, 1, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB,
                       vsrc:$XC), "xxblendvh $XT, $XA, $XB, $XC",
                       IIC_VecGeneral, []>;
  def XXBLENDVW :
    8RR_XX4Form_XTABC6<33, 2, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB,
                       vsrc:$XC), "xxblendvw $XT, $XA, $XB, $XC",
                       IIC_VecGeneral, []>;
  def XXBLENDVD :
    8RR_XX4Form_XTABC6<33, 3, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB,
                       vsrc:$XC), "xxblendvd $XT, $XA, $XB, $XC",
                       IIC_VecGeneral, []>;
}

let Predicates = [IsISA3_1] in {
  def VSLDBI : VNForm_VTAB5_SD3<22, 0, (outs vrrc:$VRT),
                                (ins vrrc:$VRA, vrrc:$VRB, u3imm:$SH),
                                "vsldbi $VRT, $VRA, $VRB, $SH",
                                IIC_VecGeneral, 
                                [(set v16i8:$VRT,
                                      (int_ppc_altivec_vsldbi v16i8:$VRA,
                                                              v16i8:$VRB,
                                                              i32:$SH))]>;
  def VSRDBI : VNForm_VTAB5_SD3<22, 1, (outs vrrc:$VRT),
                                (ins vrrc:$VRA, vrrc:$VRB, u3imm:$SH),
                                "vsrdbi $VRT, $VRA, $VRB, $SH",
                                IIC_VecGeneral,
                                [(set v16i8:$VRT,
                                      (int_ppc_altivec_vsrdbi v16i8:$VRA,
                                                              v16i8:$VRB, 
                                                              i32:$SH))]>;
  defm VSTRIBR : VXForm_VTB5_RCr<13, 1, (outs vrrc:$vT), (ins vrrc:$vB),
                                 "vstribr", "$vT, $vB", IIC_VecGeneral, []>;
  defm VSTRIBL : VXForm_VTB5_RCr<13, 0, (outs vrrc:$vT), (ins vrrc:$vB),
                                 "vstribl", "$vT, $vB", IIC_VecGeneral, []>;
  defm VSTRIHR : VXForm_VTB5_RCr<13, 3, (outs vrrc:$vT), (ins vrrc:$vB),
                                 "vstrihr", "$vT, $vB", IIC_VecGeneral, []>;
  defm VSTRIHL : VXForm_VTB5_RCr<13, 2, (outs vrrc:$vT), (ins vrrc:$vB),
                                 "vstrihl", "$vT, $vB", IIC_VecGeneral, []>;
  def VINSW :
    VXForm_1<207, (outs vrrc:$vD), (ins vrrc:$vDi, u4imm:$UIM, gprc:$rB),
             "vinsw $vD, $rB, $UIM", IIC_VecGeneral,
             [(set v4i32:$vD,
                   (int_ppc_altivec_vinsw v4i32:$vDi, i32:$rB, timm:$UIM))]>,
             RegConstraint<"$vDi = $vD">, NoEncode<"$vDi">;
  def VINSD :
    VXForm_1<463, (outs vrrc:$vD), (ins vrrc:$vDi, u4imm:$UIM, g8rc:$rB),
             "vinsd $vD, $rB, $UIM", IIC_VecGeneral,
             [(set v2i64:$vD,
                   (int_ppc_altivec_vinsd v2i64:$vDi, i64:$rB, timm:$UIM))]>,
             RegConstraint<"$vDi = $vD">, NoEncode<"$vDi">;
  def VINSBVLX :
    VXForm_VTB5_RA5_ins<15, "vinsbvlx",
                        [(set v16i8:$vD,
                              (int_ppc_altivec_vinsbvlx v16i8:$vDi, i32:$rA,
                                                        v16i8:$vB))]>;
  def VINSBVRX :
    VXForm_VTB5_RA5_ins<271, "vinsbvrx",
                        [(set v16i8:$vD,
                              (int_ppc_altivec_vinsbvrx v16i8:$vDi, i32:$rA,
                                                        v16i8:$vB))]>;
  def VINSHVLX :
    VXForm_VTB5_RA5_ins<79, "vinshvlx",
                        [(set v8i16:$vD,
                              (int_ppc_altivec_vinshvlx v8i16:$vDi, i32:$rA,
                                                        v8i16:$vB))]>;
  def VINSHVRX :
    VXForm_VTB5_RA5_ins<335, "vinshvrx",
                        [(set v8i16:$vD,
                              (int_ppc_altivec_vinshvrx v8i16:$vDi, i32:$rA,
                                                        v8i16:$vB))]>;
  def VINSWVLX :
    VXForm_VTB5_RA5_ins<143, "vinswvlx",
                        [(set v4i32:$vD,
                              (int_ppc_altivec_vinswvlx v4i32:$vDi, i32:$rA,
                                                        v4i32:$vB))]>;
  def VINSWVRX :
    VXForm_VTB5_RA5_ins<399, "vinswvrx",
                        [(set v4i32:$vD,
                              (int_ppc_altivec_vinswvrx v4i32:$vDi, i32:$rA,
                                                        v4i32:$vB))]>;
  def VINSBLX :
    VXForm_VRT5_RAB5_ins<527, "vinsblx",
                         [(set v16i8:$vD,
                               (int_ppc_altivec_vinsblx v16i8:$vDi, i32:$rA,
                                                        i32:$rB))]>;
  def VINSBRX :
    VXForm_VRT5_RAB5_ins<783, "vinsbrx",
                         [(set v16i8:$vD,
                               (int_ppc_altivec_vinsbrx v16i8:$vDi, i32:$rA,
                                                        i32:$rB))]>;
  def VINSHLX :
    VXForm_VRT5_RAB5_ins<591, "vinshlx",
                         [(set v8i16:$vD,
                               (int_ppc_altivec_vinshlx v8i16:$vDi, i32:$rA,
                                                        i32:$rB))]>;
  def VINSHRX :
    VXForm_VRT5_RAB5_ins<847, "vinshrx",
                         [(set v8i16:$vD,
                               (int_ppc_altivec_vinshrx v8i16:$vDi, i32:$rA,
                                                        i32:$rB))]>;
  def VINSWLX :
    VXForm_VRT5_RAB5_ins<655, "vinswlx",
                         [(set v4i32:$vD,
                               (int_ppc_altivec_vinswlx v4i32:$vDi, i32:$rA,
                                                        i32:$rB))]>;
  def VINSWRX :
    VXForm_VRT5_RAB5_ins<911, "vinswrx",
                         [(set v4i32:$vD,
                               (int_ppc_altivec_vinswrx v4i32:$vDi, i32:$rA,
                                                        i32:$rB))]>;
  def VINSDLX :
    VXForm_1<719, (outs vrrc:$vD), (ins vrrc:$vDi, g8rc:$rA, g8rc:$rB),
             "vinsdlx $vD, $rA, $rB", IIC_VecGeneral,
              [(set v2i64:$vD,
                    (int_ppc_altivec_vinsdlx v2i64:$vDi, i64:$rA, i64:$rB))]>,
              RegConstraint<"$vDi = $vD">, NoEncode<"$vDi">;
  def VINSDRX :
    VXForm_1<975, (outs vrrc:$vD), (ins vrrc:$vDi, g8rc:$rA, g8rc:$rB),
             "vinsdrx $vD, $rA, $rB", IIC_VecGeneral,
              [(set v2i64:$vD,
                    (int_ppc_altivec_vinsdrx v2i64:$vDi, i64:$rA, i64:$rB))]>,
              RegConstraint<"$vDi = $vD">, NoEncode<"$vDi">;
  def VEXTRACTBM : VXForm_RD5_XO5_RS5<1602, 8, (outs gprc:$rD), (ins vrrc:$vB),
                                      "vextractbm $rD, $vB", IIC_VecGeneral,
                                      []>;
  def VEXTRACTHM : VXForm_RD5_XO5_RS5<1602, 9, (outs gprc:$rD), (ins vrrc:$vB),
                                      "vextracthm $rD, $vB", IIC_VecGeneral,
                                      []>;
  def VEXTRACTWM : VXForm_RD5_XO5_RS5<1602, 10, (outs gprc:$rD), (ins vrrc:$vB),
                                      "vextractwm $rD, $vB", IIC_VecGeneral,
                                      []>;
  def VEXTRACTDM : VXForm_RD5_XO5_RS5<1602, 11, (outs gprc:$rD), (ins vrrc:$vB),
                                      "vextractdm $rD, $vB", IIC_VecGeneral,
                                      []>;
  def VEXTRACTQM : VXForm_RD5_XO5_RS5<1602, 12, (outs gprc:$rD), (ins vrrc:$vB),
                                      "vextractqm $rD, $vB", IIC_VecGeneral,
                                      []>;
  def VEXPANDBM : VXForm_RD5_XO5_RS5<1602, 0, (outs vrrc:$vD), (ins vrrc:$vB),
                                     "vexpandbm $vD, $vB", IIC_VecGeneral,
                                     []>;
  def VEXPANDHM : VXForm_RD5_XO5_RS5<1602, 1, (outs vrrc:$vD), (ins vrrc:$vB),
                                     "vexpandhm $vD, $vB", IIC_VecGeneral,
                                     []>;
  def VEXPANDWM : VXForm_RD5_XO5_RS5<1602, 2, (outs vrrc:$vD), (ins vrrc:$vB),
                                     "vexpandwm $vD, $vB", IIC_VecGeneral,
                                     []>;
  def VEXPANDDM : VXForm_RD5_XO5_RS5<1602, 3, (outs vrrc:$vD), (ins vrrc:$vB),
                                     "vexpanddm $vD, $vB", IIC_VecGeneral,
                                     []>;
  def VEXPANDQM : VXForm_RD5_XO5_RS5<1602, 4, (outs vrrc:$vD), (ins vrrc:$vB),
                                     "vexpandqm $vD, $vB", IIC_VecGeneral,
                                     []>;
  def MTVSRBM : VXForm_RD5_XO5_RS5<1602, 16, (outs vrrc:$vD), (ins g8rc:$rB),
                                   "mtvsrbm $vD, $rB", IIC_VecGeneral,
                                   []>;
  def MTVSRHM : VXForm_RD5_XO5_RS5<1602, 17, (outs vrrc:$vD), (ins g8rc:$rB),
                                   "mtvsrhm $vD, $rB", IIC_VecGeneral,
                                   []>;
  def MTVSRWM : VXForm_RD5_XO5_RS5<1602, 18, (outs vrrc:$vD), (ins g8rc:$rB),
                                   "mtvsrwm $vD, $rB", IIC_VecGeneral,
                                   []>;
  def MTVSRDM : VXForm_RD5_XO5_RS5<1602, 19, (outs vrrc:$vD), (ins g8rc:$rB),
                                   "mtvsrdm $vD, $rB", IIC_VecGeneral,
                                   []>;
  def MTVSRQM : VXForm_RD5_XO5_RS5<1602, 20, (outs vrrc:$vD), (ins g8rc:$rB),
                                   "mtvsrqm $vD, $rB", IIC_VecGeneral,
                                   []>;
  def MTVSRBMI : DXForm<4, 10, (outs vrrc:$vD), (ins u16imm64:$D),
                        "mtvsrbmi $vD, $D", IIC_VecGeneral,
                        []>;
  def VCNTMBB : VXForm_RD5_MP_VB5<1602, 12, (outs g8rc:$rD),
                                  (ins vrrc:$vB, u1imm:$MP),
                                  "vcntmbb $rD, $vB, $MP", IIC_VecGeneral,
                                  []>;
  def VCNTMBH : VXForm_RD5_MP_VB5<1602, 13, (outs g8rc:$rD),
                                  (ins vrrc:$vB, u1imm:$MP),
                                  "vcntmbh $rD, $vB, $MP", IIC_VecGeneral,
                                  []>;
  def VCNTMBW : VXForm_RD5_MP_VB5<1602, 14, (outs g8rc:$rD),
                                  (ins vrrc:$vB, u1imm:$MP),
                                  "vcntmbw $rD, $vB, $MP", IIC_VecGeneral,
                                  []>;
  def VCNTMBD : VXForm_RD5_MP_VB5<1602, 15, (outs g8rc:$rD),
                                  (ins vrrc:$vB, u1imm:$MP),
                                  "vcntmbd $rD, $vB, $MP", IIC_VecGeneral,
                                  []>;
  def VEXTDUBVLX : VAForm_1a<24, (outs vrrc:$vD),
                             (ins vrrc:$vA, vrrc:$vB, gprc:$rC),
                             "vextdubvlx $vD, $vA, $vB, $rC",
                             IIC_VecGeneral,
                             [(set v2i64:$vD,
                                   (int_ppc_altivec_vextdubvlx v16i8:$vA,
                                                               v16i8:$vB,
                                                               i32:$rC))]>;
  def VEXTDUBVRX : VAForm_1a<25, (outs vrrc:$vD),
                             (ins vrrc:$vA, vrrc:$vB, gprc:$rC),
                             "vextdubvrx $vD, $vA, $vB, $rC",
                             IIC_VecGeneral,
                             [(set v2i64:$vD,
                                   (int_ppc_altivec_vextdubvrx v16i8:$vA,
                                                               v16i8:$vB,
                                                               i32:$rC))]>;
  def VEXTDUHVLX : VAForm_1a<26, (outs vrrc:$vD),
                             (ins vrrc:$vA, vrrc:$vB, gprc:$rC),
                             "vextduhvlx $vD, $vA, $vB, $rC",
                             IIC_VecGeneral,
                             [(set v2i64:$vD,
                                   (int_ppc_altivec_vextduhvlx v8i16:$vA,
                                                               v8i16:$vB,
                                                               i32:$rC))]>;
  def VEXTDUHVRX : VAForm_1a<27, (outs vrrc:$vD),
                             (ins vrrc:$vA, vrrc:$vB, gprc:$rC),
                             "vextduhvrx $vD, $vA, $vB, $rC",
                             IIC_VecGeneral,
                             [(set v2i64:$vD,
                                   (int_ppc_altivec_vextduhvrx v8i16:$vA,
                                                               v8i16:$vB,
                                                               i32:$rC))]>;
  def VEXTDUWVLX : VAForm_1a<28, (outs vrrc:$vD),
                             (ins vrrc:$vA, vrrc:$vB, gprc:$rC),
                             "vextduwvlx $vD, $vA, $vB, $rC",
                             IIC_VecGeneral,
                             [(set v2i64:$vD,
                                   (int_ppc_altivec_vextduwvlx v4i32:$vA,
                                                               v4i32:$vB,
                                                               i32:$rC))]>;
  def VEXTDUWVRX : VAForm_1a<29, (outs vrrc:$vD),
                             (ins vrrc:$vA, vrrc:$vB, gprc:$rC),
                             "vextduwvrx $vD, $vA, $vB, $rC",
                             IIC_VecGeneral,
                             [(set v2i64:$vD,
                                   (int_ppc_altivec_vextduwvrx v4i32:$vA,
                                                               v4i32:$vB,
                                                               i32:$rC))]>;
  def VEXTDDVLX : VAForm_1a<30, (outs vrrc:$vD),
                            (ins vrrc:$vA, vrrc:$vB, gprc:$rC),
                            "vextddvlx $vD, $vA, $vB, $rC",
                            IIC_VecGeneral,
                            [(set v2i64:$vD,
                                  (int_ppc_altivec_vextddvlx v2i64:$vA,
                                                             v2i64:$vB,
                                                             i32:$rC))]>;
  def VEXTDDVRX : VAForm_1a<31, (outs vrrc:$vD),
                            (ins vrrc:$vA, vrrc:$vB, gprc:$rC),
                            "vextddvrx $vD, $vA, $vB, $rC",
                            IIC_VecGeneral,
                            [(set v2i64:$vD,
                                  (int_ppc_altivec_vextddvrx v2i64:$vA,
                                                             v2i64:$vB,
                                                             i32:$rC))]>;
   def VPDEPD : VXForm_1<1485, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vpdepd $vD, $vA, $vB", IIC_VecGeneral,
                         [(set v2i64:$vD,
                         (int_ppc_altivec_vpdepd v2i64:$vA, v2i64:$vB))]>;
   def VPEXTD : VXForm_1<1421, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vpextd $vD, $vA, $vB", IIC_VecGeneral,
                         [(set v2i64:$vD,
                         (int_ppc_altivec_vpextd v2i64:$vA, v2i64:$vB))]>;
   def PDEPD : XForm_6<31, 156, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                       "pdepd $rA, $rS, $rB", IIC_IntGeneral,
                       [(set i64:$rA, (int_ppc_pdepd i64:$rS, i64:$rB))]>;
   def PEXTD : XForm_6<31, 188, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                       "pextd $rA, $rS, $rB", IIC_IntGeneral,
                       [(set i64:$rA, (int_ppc_pextd i64:$rS, i64:$rB))]>;
   def VCFUGED : VXForm_1<1357, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                          "vcfuged $vD, $vA, $vB", IIC_VecGeneral,
                          [(set v2i64:$vD,
                          (int_ppc_altivec_vcfuged v2i64:$vA, v2i64:$vB))]>;
   def VGNB : VXForm_RD5_N3_VB5<1228, (outs g8rc:$rD), (ins vrrc:$vB, u3imm:$N),
                                "vgnb $rD, $vB, $N", IIC_VecGeneral,
                                [(set i64:$rD,
                                (int_ppc_altivec_vgnb v1i128:$vB, timm:$N))]>;
   def CFUGED : XForm_6<31, 220, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                        "cfuged $rA, $rS, $rB", IIC_IntGeneral,
                        [(set i64:$rA, (int_ppc_cfuged i64:$rS, i64:$rB))]>;
   def XXEVAL :
     8RR_XX4Form_IMM8_XTAB6<34, 1, (outs vsrc:$XT), (ins vsrc:$XA, vsrc:$XB,
                            vsrc:$XC, u8imm:$IMM),
                            "xxeval $XT, $XA, $XB, $XC, $IMM", IIC_VecGeneral,
                            [(set v2i64:$XT, (int_ppc_vsx_xxeval v2i64:$XA,
                                  v2i64:$XB, v2i64:$XC, timm:$IMM))]>;
   def VCLZDM : VXForm_1<1924, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vclzdm $vD, $vA, $vB", IIC_VecGeneral,
                         [(set v2i64:$vD,
                         (int_ppc_altivec_vclzdm v2i64:$vA, v2i64:$vB))]>;
   def VCTZDM : VXForm_1<1988, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vctzdm $vD, $vA, $vB", IIC_VecGeneral,
                         [(set v2i64:$vD,
                         (int_ppc_altivec_vctzdm v2i64:$vA, v2i64:$vB))]>;
   def CNTLZDM : XForm_6<31, 59, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                         "cntlzdm $rA, $rS, $rB", IIC_IntGeneral,
                         [(set i64:$rA,
                         (int_ppc_cntlzdm i64:$rS, i64:$rB))]>;
   def CNTTZDM : XForm_6<31, 571, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                         "cnttzdm $rA, $rS, $rB", IIC_IntGeneral,
                         [(set i64:$rA,
                         (int_ppc_cnttzdm i64:$rS, i64:$rB))]>;
   def XXGENPCVBM :
     XForm_XT6_IMM5_VB5<60, 916, (outs vsrc:$XT), (ins vrrc:$VRB, s5imm:$IMM),
                        "xxgenpcvbm $XT, $VRB, $IMM", IIC_VecGeneral, []>;
   def XXGENPCVHM :
     XForm_XT6_IMM5_VB5<60, 917, (outs vsrc:$XT), (ins vrrc:$VRB, s5imm:$IMM),
                        "xxgenpcvhm $XT, $VRB, $IMM", IIC_VecGeneral, []>;
   def XXGENPCVWM :
     XForm_XT6_IMM5_VB5<60, 948, (outs vsrc:$XT), (ins vrrc:$VRB, s5imm:$IMM),
                        "xxgenpcvwm $XT, $VRB, $IMM", IIC_VecGeneral, []>;
   def XXGENPCVDM :
     XForm_XT6_IMM5_VB5<60, 949, (outs vsrc:$XT), (ins vrrc:$VRB, s5imm:$IMM),
                        "xxgenpcvdm $XT, $VRB, $IMM", IIC_VecGeneral, []>;
   def VCLRLB : VXForm_1<397, (outs vrrc:$vD), (ins vrrc:$vA, gprc:$rB),
                         "vclrlb $vD, $vA, $rB", IIC_VecGeneral,
                         [(set v16i8:$vD,
                               (int_ppc_altivec_vclrlb v16i8:$vA, i32:$rB))]>;
   def VCLRRB : VXForm_1<461, (outs vrrc:$vD), (ins vrrc:$vA, gprc:$rB),
                         "vclrrb $vD, $vA, $rB", IIC_VecGeneral,
                         [(set v16i8:$vD,
                               (int_ppc_altivec_vclrrb v16i8:$vA, i32:$rB))]>;
  def VMULLD : VXForm_1<457, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vmulld $vD, $vA, $vB", IIC_VecGeneral,
                        [(set v2i64:$vD, (mul v2i64:$vA, v2i64:$vB))]>;
  def VMULHSW : VXForm_1<905, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vmulhsw $vD, $vA, $vB", IIC_VecGeneral,
                         [(set v4i32:$vD, (mulhs v4i32:$vA, v4i32:$vB))]>;
  def VMULHUW : VXForm_1<649, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vmulhuw $vD, $vA, $vB", IIC_VecGeneral,
                         [(set v4i32:$vD, (mulhu v4i32:$vA, v4i32:$vB))]>;
  def VMULHSD : VXForm_1<969, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vmulhsd $vD, $vA, $vB", IIC_VecGeneral,
                         [(set v2i64:$vD, (mulhs v2i64:$vA, v2i64:$vB))]>;
  def VMULHUD : VXForm_1<713, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vmulhud $vD, $vA, $vB", IIC_VecGeneral,
                         [(set v2i64:$vD, (mulhu v2i64:$vA, v2i64:$vB))]>;
  def VMODSW : VXForm_1<1931, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vmodsw $vD, $vA, $vB", IIC_VecGeneral,
                        [(set v4i32:$vD, (srem v4i32:$vA, v4i32:$vB))]>;
  def VMODUW : VXForm_1<1675, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vmoduw $vD, $vA, $vB", IIC_VecGeneral,
                        [(set v4i32:$vD, (urem v4i32:$vA, v4i32:$vB))]>;
  def VMODSD : VXForm_1<1995, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vmodsd $vD, $vA, $vB", IIC_VecGeneral,
                        [(set v2i64:$vD, (srem v2i64:$vA, v2i64:$vB))]>;
  def VMODUD : VXForm_1<1739, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vmodud $vD, $vA, $vB", IIC_VecGeneral,
                        [(set v2i64:$vD, (urem v2i64:$vA, v2i64:$vB))]>;
  def VDIVSW : VXForm_1<395, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vdivsw $vD, $vA, $vB", IIC_VecGeneral,
                        [(set v4i32:$vD, (sdiv v4i32:$vA, v4i32:$vB))]>;
  def VDIVUW : VXForm_1<139, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vdivuw $vD, $vA, $vB", IIC_VecGeneral,
                        [(set v4i32:$vD, (udiv v4i32:$vA, v4i32:$vB))]>;
  def VDIVSD : VXForm_1<459, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vdivsd $vD, $vA, $vB", IIC_VecGeneral,
                        [(set v2i64:$vD, (sdiv v2i64:$vA, v2i64:$vB))]>;
  def VDIVUD : VXForm_1<203, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vdivud $vD, $vA, $vB", IIC_VecGeneral,
                        [(set v2i64:$vD, (udiv v2i64:$vA, v2i64:$vB))]>;
  def VDIVESW : VXForm_1<907, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vdivesw $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VDIVEUW : VXForm_1<651, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vdiveuw $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VDIVESD : VXForm_1<971, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vdivesd $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VDIVEUD : VXForm_1<715, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vdiveud $vD, $vA, $vB", IIC_VecGeneral, []>;
  def XVTLSBB : XX2_BF3_XO5_XB6_XO9<60, 2, 475, (outs crrc:$BF), (ins vsrc:$XB),
                                    "xvtlsbb $BF, $XB", IIC_VecGeneral, []>;

  // The XFormMemOp flag for the following 8 instructions is set on
  // the instruction format.
  let mayLoad = 1, mayStore = 0 in {
    def LXVRBX : X_XT6_RA5_RB5<31, 13, "lxvrbx", vsrc, []>;
    def LXVRHX : X_XT6_RA5_RB5<31, 45, "lxvrhx", vsrc, []>;
    def LXVRWX : X_XT6_RA5_RB5<31, 77, "lxvrwx", vsrc, []>;
    def LXVRDX : X_XT6_RA5_RB5<31, 109, "lxvrdx", vsrc, []>;
  }

  let mayLoad = 0, mayStore = 1 in {
    def STXVRBX : X_XS6_RA5_RB5<31, 141, "stxvrbx", vsrc, []>;
    def STXVRHX : X_XS6_RA5_RB5<31, 173, "stxvrhx", vsrc, []>;
    def STXVRWX : X_XS6_RA5_RB5<31, 205, "stxvrwx", vsrc, []>;
    def STXVRDX : X_XS6_RA5_RB5<31, 237, "stxvrdx", vsrc, []>;
  }

  def VMULESD : VXForm_1<968, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vmulesd $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VMULEUD : VXForm_1<712, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vmuleud $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VMULOSD : VXForm_1<456, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vmulosd $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VMULOUD : VXForm_1<200, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vmuloud $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VMSUMCUD : VAForm_1a<23, (outs vrrc:$vD),
                           (ins vrrc:$vA, vrrc:$vB, vrrc:$vC),
                           "vmsumcud $vD, $vA, $vB, $vC", IIC_VecGeneral, []>;
  def VDIVSQ : VXForm_1<267, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vdivsq $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VDIVUQ : VXForm_1<11, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vdivuq $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VDIVESQ : VXForm_1<779, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vdivesq $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VDIVEUQ : VXForm_1<523, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                         "vdiveuq $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VCMPEQUQ : VCMP <455, "vcmpequq $vD, $vA, $vB" , v1i128>;
  def VCMPGTSQ : VCMP <903, "vcmpgtsq $vD, $vA, $vB" , v1i128>;
  def VCMPGTUQ : VCMP <647, "vcmpgtuq $vD, $vA, $vB" , v1i128>;
  def VCMPEQUQ_rec : VCMPo <455, "vcmpequq. $vD, $vA, $vB" , v1i128>;
  def VCMPGTSQ_rec : VCMPo <903, "vcmpgtsq. $vD, $vA, $vB" , v1i128>;
  def VCMPGTUQ_rec : VCMPo <647, "vcmpgtuq. $vD, $vA, $vB" , v1i128>;
  def VMODSQ : VXForm_1<1803, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vmodsq $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VMODUQ : VXForm_1<1547, (outs vrrc:$vD), (ins vrrc:$vA, vrrc:$vB),
                        "vmoduq $vD, $vA, $vB", IIC_VecGeneral, []>;
  def VEXTSD2Q : VXForm_RD5_XO5_RS5<1538, 27, (outs vrrc:$vD), (ins vrrc:$vB),
                                    "vextsd2q $vD, $vB", IIC_VecGeneral, []>;
  def VCMPUQ : VXForm_BF3_VAB5<257, (outs crrc:$BF), (ins vrrc:$vA, vrrc:$vB),
                               "vcmpuq $BF, $vA, $vB", IIC_VecGeneral, []>;
  def VCMPSQ : VXForm_BF3_VAB5<321, (outs crrc:$BF), (ins vrrc:$vA, vrrc:$vB),
                               "vcmpsq $BF, $vA, $vB", IIC_VecGeneral, []>;
  def VRLQNM : VX1_VT5_VA5_VB5<325, "vrlqnm", []>;
  def VRLQMI : VXForm_1<69, (outs vrrc:$vD),
                        (ins vrrc:$vA, vrrc:$vB, vrrc:$vDi),
                        "vrlqmi $vD, $vA, $vB", IIC_VecFP, []>,
                        RegConstraint<"$vDi = $vD">, NoEncode<"$vDi">;
  def VSLQ : VX1_VT5_VA5_VB5<261, "vslq", []>;
  def VSRAQ : VX1_VT5_VA5_VB5<773, "vsraq", []>;
  def VSRQ : VX1_VT5_VA5_VB5<517, "vsrq", []>;
  def VRLQ : VX1_VT5_VA5_VB5<5, "vrlq", []>;
  def XSCVQPUQZ : X_VT5_XO5_VB5<63, 0, 836, "xscvqpuqz", []>;
  def XSCVQPSQZ : X_VT5_XO5_VB5<63, 8, 836, "xscvqpsqz", []>;
  def XSCVUQQP : X_VT5_XO5_VB5<63, 3, 836, "xscvuqqp", []>;
  def XSCVSQQP : X_VT5_XO5_VB5<63, 11, 836, "xscvsqqp", []>;
}

//---------------------------- Anonymous Patterns ----------------------------//
let Predicates = [IsISA3_1] in {
  def : Pat<(v16i8 (int_ppc_vsx_xxgenpcvbm v16i8:$VRB, imm:$IMM)),
            (v16i8 (COPY_TO_REGCLASS (XXGENPCVBM $VRB, imm:$IMM), VRRC))>;
  def : Pat<(v8i16 (int_ppc_vsx_xxgenpcvhm v8i16:$VRB, imm:$IMM)),
            (v8i16 (COPY_TO_REGCLASS (XXGENPCVHM $VRB, imm:$IMM), VRRC))>;
  def : Pat<(v4i32 (int_ppc_vsx_xxgenpcvwm v4i32:$VRB, imm:$IMM)),
            (v4i32 (COPY_TO_REGCLASS (XXGENPCVWM $VRB, imm:$IMM), VRRC))>;
  def : Pat<(v2i64 (int_ppc_vsx_xxgenpcvdm v2i64:$VRB, imm:$IMM)),
            (v2i64 (COPY_TO_REGCLASS (XXGENPCVDM $VRB, imm:$IMM), VRRC))>;
  def : Pat<(i32 (int_ppc_vsx_xvtlsbb v16i8:$XB, 1)),
            (EXTRACT_SUBREG (XVTLSBB (COPY_TO_REGCLASS $XB, VSRC)), sub_lt)>;
  def : Pat<(i32 (int_ppc_vsx_xvtlsbb v16i8:$XB, 0)),
            (EXTRACT_SUBREG (XVTLSBB (COPY_TO_REGCLASS $XB, VSRC)), sub_eq)>;
  def : Pat<(v1i128 (shl v1i128:$VRA, v1i128:$VRB)),
            (v1i128 (VSLQ v1i128:$VRA, v1i128:$VRB))>;
  def : Pat<(v1i128 (PPCshl v1i128:$VRA, v1i128:$VRB)),
            (v1i128 (VSLQ v1i128:$VRA, v1i128:$VRB))>;
  def : Pat<(v1i128 (srl v1i128:$VRA, v1i128:$VRB)),
            (v1i128 (VSRQ v1i128:$VRA, v1i128:$VRB))>;
  def : Pat<(v1i128 (PPCsrl v1i128:$VRA, v1i128:$VRB)),
            (v1i128 (VSRQ v1i128:$VRA, v1i128:$VRB))>;
  def : Pat<(v1i128 (sra v1i128:$VRA, v1i128:$VRB)),
            (v1i128 (VSRAQ v1i128:$VRA, v1i128:$VRB))>;
  def : Pat<(v1i128 (PPCsra v1i128:$VRA, v1i128:$VRB)),
            (v1i128 (VSRAQ v1i128:$VRA, v1i128:$VRB))>;
}

let AddedComplexity = 400, Predicates = [IsISA3_1] in {
  def : Pat<(truncstorei8 (i32 (vector_extract v16i8:$rS, 0)), xoaddr:$src),
            (STXVRBX (COPY_TO_REGCLASS $rS, VSRC), xoaddr:$src)>;
  def : Pat<(truncstorei16 (i32 (vector_extract v8i16:$rS, 0)), xoaddr:$src),
            (STXVRHX (COPY_TO_REGCLASS $rS, VSRC), xoaddr:$src)>;
  def : Pat<(store (i32 (vector_extract v4i32:$rS, 0)), xoaddr:$src),
            (STXVRWX (COPY_TO_REGCLASS $rS, VSRC), xoaddr:$src)>;
  def : Pat<(store (i64 (vector_extract v2i64:$rS, 0)), xoaddr:$src),
            (STXVRDX (COPY_TO_REGCLASS $rS, VSRC), xoaddr:$src)>;
}

let AddedComplexity = 400, Predicates = [PrefixInstrs] in {
 def : Pat<(v4i32 (build_vector i32immNonAllOneNonZero:$A,
                                i32immNonAllOneNonZero:$A,
                                i32immNonAllOneNonZero:$A,
                                i32immNonAllOneNonZero:$A)),
           (v4i32 (XXSPLTIW imm:$A))>;
 def : Pat<(f32 nzFPImmAsi32:$A),
           (COPY_TO_REGCLASS (XXSPLTIDP (getFPAs32BitInt fpimm:$A)),
                             VSFRC)>;
 def : Pat<(f64 nzFPImmAsi32:$A),
           (COPY_TO_REGCLASS (XXSPLTIDP (getFPAs32BitInt fpimm:$A)),
                             VSFRC)>;
}

let Predicates = [PrefixInstrs] in {
  def : Pat<(v16i8 (int_ppc_vsx_xxpermx v16i8:$A, v16i8:$B, v16i8:$C, timm:$D)),
            (COPY_TO_REGCLASS (XXPERMX (COPY_TO_REGCLASS $A, VSRC),
                                       (COPY_TO_REGCLASS $B, VSRC),
                                       (COPY_TO_REGCLASS $C, VSRC), $D), VSRC)>;
  def : Pat<(v16i8 (int_ppc_vsx_xxblendvb v16i8:$A, v16i8:$B, v16i8:$C)),
            (COPY_TO_REGCLASS
                   (XXBLENDVB (COPY_TO_REGCLASS $A, VSRC),
                              (COPY_TO_REGCLASS $B, VSRC),
                              (COPY_TO_REGCLASS $C, VSRC)), VSRC)>;
  def : Pat<(v8i16 (int_ppc_vsx_xxblendvh v8i16:$A, v8i16:$B, v8i16:$C)),
            (COPY_TO_REGCLASS
                   (XXBLENDVH (COPY_TO_REGCLASS $A, VSRC),
                              (COPY_TO_REGCLASS $B, VSRC),
                              (COPY_TO_REGCLASS $C, VSRC)), VSRC)>;
  def : Pat<(int_ppc_vsx_xxblendvw v4i32:$A, v4i32:$B, v4i32:$C),
            (XXBLENDVW $A, $B, $C)>;
  def : Pat<(int_ppc_vsx_xxblendvd v2i64:$A, v2i64:$B, v2i64:$C),
            (XXBLENDVD $A, $B, $C)>;
}

