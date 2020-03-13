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

def PrefixInstrs : Predicate<"PPCSubTarget->hasPrefixInstrs()">;

let Predicates = [PrefixInstrs] in {
  let Interpretation64Bit = 1, isCodeGenOnly = 1 in {
    defm PADDI8 :
      MLS_DForm_R_SI34_RTA5_p<14, (outs g8rc:$RT), (ins g8rc:$RA, s34imm:$SI),
                              (ins immZero:$RA, s34imm:$SI),
                              "paddi $RT, $RA, $SI", IIC_LdStLFD>;
    let isReMaterializable = 1, isAsCheapAsAMove = 1, isMoveImm = 1 in {
      def PLI8 : MLS_DForm_SI34_RT5<14, (outs g8rc:$RT),
                                    (ins s34imm:$SI),
                                    "pli $RT, $SI", IIC_IntSimple, []>;
    }
  }
  defm PADDI :
    MLS_DForm_R_SI34_RTA5_p<14, (outs gprc:$RT), (ins gprc:$RA, s34imm:$SI),
                            (ins immZero:$RA, s34imm:$SI),
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

