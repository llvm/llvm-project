//==== SPUInstrFormats.td - Cell SPU Instruction Formats ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Cell SPU instruction formats. Note that these are notationally similar to
// PowerPC, like "A-Form". But the sizes of operands and fields differ.

// This was kiped from the PPC instruction formats (seemed like a good idea...)

class SPUInstr<dag OOL, dag IOL, string asmstr, InstrItinClass itin>
        : Instruction {
  field bits<32> Inst;

  let Namespace = "SPU";
  let OutOperandList = OOL;
  let InOperandList = IOL;
  let AsmString = asmstr;
  let Itinerary = itin;
}

// RR Format
class RRForm<bits<11> opcode, dag OOL, dag IOL, string asmstr, 
              InstrItinClass itin, list<dag> pattern>
         : SPUInstr<OOL, IOL, asmstr, itin> {
  bits<7> RA;
  bits<7> RB;
  bits<7> RT;

  let Pattern = pattern;

  let Inst{0-10} = opcode;
  let Inst{11-17} = RB;
  let Inst{18-24} = RA;
  let Inst{25-31} = RT;
}

let RB = 0 in {
  // RR Format, where RB is zeroed (dont care):
  class RRForm_1<bits<11> opcode, dag OOL, dag IOL, string asmstr, 
                 InstrItinClass itin, list<dag> pattern>
           : RRForm<opcode, OOL, IOL, asmstr, itin, pattern>
  { }

  let RA = 0 in {
    // RR Format, where RA and RB are zeroed (dont care):
    // Used for reads from status control registers (see FPSCRRr32)
    class RRForm_2<bits<11> opcode, dag OOL, dag IOL, string asmstr,
                   InstrItinClass itin, list<dag> pattern>
             : RRForm<opcode, OOL, IOL, asmstr, itin, pattern>
    { }
  }
}

let RT = 0 in {
  // RR Format, where RT is zeroed (don't care), or as the instruction handbook
  // says, "RT is a false target." Used in "Halt if" instructions
  class RRForm_3<bits<11> opcode, dag OOL, dag IOL, string asmstr,
                 InstrItinClass itin, list<dag> pattern>
      : RRForm<opcode, OOL, IOL, asmstr, itin, pattern>
  { }
}

// RRR Format
class RRRForm<bits<4> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
        : SPUInstr<OOL, IOL, asmstr, itin>
{
  bits<7> RA;
  bits<7> RB;
  bits<7> RC;
  bits<7> RT;

  let Pattern = pattern;

  let Inst{0-3} = opcode;
  let Inst{4-10} = RT;
  let Inst{11-17} = RB;
  let Inst{18-24} = RA;
  let Inst{25-31} = RC;
}

// RI7 Format
class RI7Form<bits<11> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
        : SPUInstr<OOL, IOL, asmstr, itin>
{
  bits<7> i7;
  bits<7> RA;
  bits<7> RT;

  let Pattern = pattern;

  let Inst{0-10} = opcode;
  let Inst{11-17} = i7;
  let Inst{18-24} = RA;
  let Inst{25-31} = RT;
}

// CVTIntFp Format
class CVTIntFPForm<bits<10> opcode, dag OOL, dag IOL, string asmstr,
                   InstrItinClass itin, list<dag> pattern>
        : SPUInstr<OOL, IOL, asmstr, itin>
{
  bits<7> RA;
  bits<7> RT;

  let Pattern = pattern;

  let Inst{0-9} = opcode;
  let Inst{10-17} = 0;
  let Inst{18-24} = RA;
  let Inst{25-31} = RT;
}

let RA = 0 in {
  class BICondForm<bits<11> opcode, dag OOL, dag IOL, string asmstr, list<dag> pattern>
           : RRForm<opcode, OOL, IOL, asmstr, BranchResolv, pattern>
  { }

  let RT = 0 in {
    // Branch instruction format (without D/E flag settings)
    class BRForm<bits<11> opcode, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin, list<dag> pattern>
          : RRForm<opcode, OOL, IOL, asmstr, itin, pattern>
    { }

    class BIForm<bits<11> opcode, string asmstr, list<dag> pattern>
             : RRForm<opcode, (outs), (ins R32C:$func), asmstr, BranchResolv,
                      pattern>
    { }

    let RB = 0 in {
      // Return instruction (bi, branch indirect), RA is zero (LR):
      class RETForm<string asmstr, list<dag> pattern>
             : BRForm<0b00010101100, (outs), (ins), asmstr, BranchResolv,
                      pattern>
      { }
    }
  }
}

// Branch indirect external data forms:
class BISLEDForm<bits<2> DE_flag, string asmstr, list<dag> pattern>
         : SPUInstr<(outs), (ins indcalltarget:$func), asmstr, BranchResolv>
{
  bits<7> Rcalldest;

  let Pattern = pattern;

  let Inst{0-10} = 0b11010101100;
  let Inst{11} = 0;
  let Inst{12-13} = DE_flag;
  let Inst{14-17} = 0b0000;
  let Inst{18-24} = Rcalldest;
  let Inst{25-31} = 0b0000000;
}

// RI10 Format
class RI10Form<bits<8> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
        : SPUInstr<OOL, IOL, asmstr, itin>
{
  bits<10> i10;
  bits<7> RA;
  bits<7> RT;

  let Pattern = pattern;

  let Inst{0-7} = opcode;
  let Inst{8-17} = i10;
  let Inst{18-24} = RA;
  let Inst{25-31} = RT;
}

// RI10 Format, where the constant is zero (or effectively ignored by the
// SPU)
let i10 = 0 in {
  class RI10Form_1<bits<8> opcode, dag OOL, dag IOL, string asmstr,
                   InstrItinClass itin, list<dag> pattern>
          : RI10Form<opcode, OOL, IOL, asmstr, itin, pattern>
  { }
}

// RI10 Format, where RT is ignored.
// This format is used primarily by the Halt If ... Immediate set of
// instructions
let RT = 0 in {
  class RI10Form_2<bits<8> opcode, dag OOL, dag IOL, string asmstr,
                   InstrItinClass itin, list<dag> pattern>
        : RI10Form<opcode, OOL, IOL, asmstr, itin, pattern>
  { }
}

// RI16 Format
class RI16Form<bits<9> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
        : SPUInstr<OOL, IOL, asmstr, itin>
{
  bits<16> i16;
  bits<7> RT;

  let Pattern = pattern;

  let Inst{0-8} = opcode;
  let Inst{9-24} = i16;
  let Inst{25-31} = RT;
}

// Specialized version of the RI16 Format for unconditional branch relative and
// branch absolute, branch and set link. Note that for branch and set link, the
// link register doesn't have to be $lr, but this is actually hard coded into
// the instruction pattern.

let RT = 0 in {
  class UncondBranch<bits<9> opcode, dag OOL, dag IOL, string asmstr,
                     list<dag> pattern>
    : RI16Form<opcode, OOL, IOL, asmstr, BranchResolv, pattern>
  { }

  class BranchSetLink<bits<9> opcode, dag OOL, dag IOL, string asmstr,
                      list<dag> pattern>
        : RI16Form<opcode, OOL, IOL, asmstr, BranchResolv, pattern>
  { }
}

//===----------------------------------------------------------------------===//
// Specialized versions of RI16:
//===----------------------------------------------------------------------===//

// RI18 Format
class RI18Form<bits<7> opcode, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin, list<dag> pattern>
        : SPUInstr<OOL, IOL, asmstr, itin>
{
  bits<18> i18;
  bits<7> RT;

  let Pattern = pattern;

  let Inst{0-6} = opcode;
  let Inst{7-24} = i18;
  let Inst{25-31} = RT;
}

//===----------------------------------------------------------------------===//
// Instruction formats for intrinsics:
//===----------------------------------------------------------------------===//

// RI10 Format for v8i16 intrinsics
class RI10_Int_v8i16<bits<8> opcode, string opc, InstrItinClass itin,
                     Intrinsic IntID> :
  RI10Form<opcode, (outs VECREG:$rT), (ins s10imm:$val, VECREG:$rA),
           !strconcat(opc, " $rT, $rA, $val"), itin,
           [(set (v8i16 VECREG:$rT), (IntID (v8i16 VECREG:$rA),
                                            i16ImmSExt10:$val))] >;

class RI10_Int_v4i32<bits<8> opcode, string opc, InstrItinClass itin,
                     Intrinsic IntID> :
  RI10Form<opcode, (outs VECREG:$rT), (ins s10imm:$val, VECREG:$rA),
           !strconcat(opc, " $rT, $rA, $val"), itin,
           [(set (v4i32 VECREG:$rT), (IntID (v4i32 VECREG:$rA),
                                            i32ImmSExt10:$val))] >;

// RR Format for v8i16 intrinsics
class RR_Int_v8i16<bits<11> opcode, string opc, InstrItinClass itin,
                   Intrinsic IntID> :
  RRForm<opcode, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
         !strconcat(opc, " $rT, $rA, $rB"), itin,
         [(set (v8i16 VECREG:$rT), (IntID (v8i16 VECREG:$rA),
                                          (v8i16 VECREG:$rB)))] >;

// RR Format for v4i32 intrinsics
class RR_Int_v4i32<bits<11> opcode, string opc, InstrItinClass itin,
                   Intrinsic IntID> :
  RRForm<opcode, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
         !strconcat(opc, " $rT, $rA, $rB"), itin,
         [(set (v4i32 VECREG:$rT), (IntID (v4i32 VECREG:$rA),
                                          (v4i32 VECREG:$rB)))] >;

//===----------------------------------------------------------------------===//
// Pseudo instructions, like call frames:
//===----------------------------------------------------------------------===//

class Pseudo<dag OOL, dag IOL, string asmstr, list<dag> pattern>
    : SPUInstr<OOL, IOL, asmstr, NoItinerary> {
  let OutOperandList = OOL;
  let InOperandList = IOL;
  let AsmString   = asmstr;
  let Pattern = pattern;
  let Inst{31-0} = 0;
}
