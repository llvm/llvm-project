//===-- Nios2InstrFormats.td - Nios2 Instruction Formats ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe NIOS2 instructions format
//
//
//===----------------------------------------------------------------------===//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<6> val> {
  bits<6> Value = val;
}

def Pseudo : Format<0>;
def FrmI : Format<1>;
def FrmR : Format<2>;
def FrmJ : Format<3>;
def FrmOther : Format<4>; // Instruction w/ a custom format

def isNios2r1  : Predicate<"Subtarget->isNios2r1()">;
def isNios2r2  : Predicate<"Subtarget->isNios2r2()">;

class PredicateControl {
  // Predicates related to specific target CPU features
  list<Predicate> FeaturePredicates = [];
  // Predicates for the instruction group membership in given ISA
  list<Predicate> InstrPredicates = [];
  
  list<Predicate> Predicates = !listconcat(FeaturePredicates, InstrPredicates);
}

//===----------------------------------------------------------------------===//
// Base classes for 32-bit, 16-bit and pseudo instructions
//===----------------------------------------------------------------------===//

class Nios2Inst32<dag outs, dag ins, string asmstr, list<dag> pattern,
                  InstrItinClass itin, Format f>: Instruction,
		                                  PredicateControl {
  field bits<32> Inst;
  Format Form = f;

  let Namespace = "Nios2";
  let Size = 4;

  bits<6> Opcode = 0;

  // Bottom 6 bits are the 'opcode' field
  let Inst{5-0} = Opcode;

  let OutOperandList = outs;
  let InOperandList  = ins;

  let AsmString = asmstr;
  let Pattern   = pattern;
  let Itinerary = itin;

  // Attributes specific to Nios2 instructions:

  // TSFlags layout should be kept in sync with Nios2InstrInfo.h.
  let TSFlags{5-0} = Form.Value;
  let DecoderNamespace = "Nios2";
  field bits<32> SoftFail = 0;
}

class Nios2Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern,
      InstrItinClass Itin = IIPseudo>:
  Nios2Inst32<outs, ins, asmstr, pattern, Itin, Pseudo> {

  let isCodeGenOnly = 1;
  let isPseudo = 1;
}

//===----------------------------------------------------------------------===//
// Base classes for R1 and R2 instructions
//===----------------------------------------------------------------------===//

class Nios2R1Inst32<dag outs, dag ins, string asmstr, list<dag> pattern,
                    InstrItinClass itin, Format f>: 
      Nios2Inst32<outs, ins, asmstr, pattern, itin, f> {
  let DecoderNamespace = "Nios2";
  let InstrPredicates = [isNios2r1];
}

class Nios2R2Inst32<dag outs, dag ins, string asmstr, list<dag> pattern,
                    InstrItinClass itin, Format f>: 
      Nios2Inst32<outs, ins, asmstr, pattern, itin, f> {
  let DecoderNamespace = "Nios2r2";
  let InstrPredicates = [isNios2r2];
}

//===----------------------------------------------------------------------===//
// Format I instruction class in Nios2 : <|A|B|immediate|opcode|>
//===----------------------------------------------------------------------===//

class FI<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
         InstrItinClass itin>: Nios2R1Inst32<outs, ins, asmstr,
	                                     pattern, itin, FrmI> {

  bits<5>  rA;
  bits<5>  rB;
  bits<16> imm;

  let Opcode = op;

  let Inst{31-27} = rA;
  let Inst{26-22} = rB;
  let Inst{21-6} = imm;
}


//===----------------------------------------------------------------------===//
// Format R instruction : <|A|B|C|opx|imm|opcode|>
//===----------------------------------------------------------------------===//

class FR<bits<6> opx, dag outs, dag ins, string asmstr, list<dag> pattern,
         InstrItinClass itin>: Nios2R1Inst32<outs, ins, asmstr,
	                                     pattern, itin, FrmR> {
  bits<5> rA;
  bits<5> rB;
  bits<5> rC;
  bits<5> imm = 0;

  let Opcode = 0x3a; /* opcode is always 0x3a for R instr. */

  let Inst{31-27} = rA;
  let Inst{26-22} = rB;
  let Inst{21-17} = rC;
  let Inst{16-11} = opx; /* opx stands for opcode extension */
  let Inst{10-6}  = imm; /* optional 5-bit immediate value */
}

//===----------------------------------------------------------------------===//
// Format J instruction class in Nios2 : <|address|opcode|>
//===----------------------------------------------------------------------===//

class FJ<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
         InstrItinClass itin>:
      Nios2R1Inst32<outs, ins, asmstr, pattern, itin, FrmJ> {
  bits<26> addr;
  let Opcode = op;
  let Inst{31-6} = addr;
}

//===----------------------------------------------------------------------===//
// Multiclasses for common instructions of both R1 and R2:
//===----------------------------------------------------------------------===//

// Multiclass for instructions that have R format in R1 and F3X6 format in R2
// and their opx values differ between R1 and R2
multiclass CommonInstr_R_F3X6_opx<bits<6> opxR1, bits<6> opxR2, dag outs,
                                  dag ins, string asmstr, list<dag> pattern,
                                  InstrItinClass itin> {
  def NAME#_R1 : FR<opxR1, outs, ins, asmstr, pattern, itin>;
}

// Multiclass for instructions that have R format in R1 and F3X6 format in R2
// and their opx values are the same in R1 and R2
multiclass CommonInstr_R_F3X6<bits<6> opx, dag outs, dag ins, string asmstr,
                              list<dag> pattern, InstrItinClass itin> :
  CommonInstr_R_F3X6_opx<opx, opx, outs, ins, asmstr, pattern, itin>;

// Multiclass for instructions that have I format in R1 and F2I16 format in R2
// and their op code values differ between R1 and R2
multiclass CommonInstr_I_F2I16_op<bits<6> opR1, bits<6> opR2, dag outs, dag ins,
                                  string asmstr, list<dag> pattern,
                                  InstrItinClass itin> {
  def NAME#_R1 : FI<opR1, outs, ins, asmstr, pattern, itin>;
}

// Multiclass for instructions that have I format in R1 and F2I16 format in R2
// and their op code values are the same in R1 and R2
multiclass CommonInstr_I_F2I16<bits<6> op, dag outs, dag ins, string asmstr,
                               list<dag> pattern, InstrItinClass itin> :
  CommonInstr_I_F2I16_op<op, op, outs, ins, asmstr, pattern, itin>;
