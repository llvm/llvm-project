//==- HexagonInstrFormats.td - Hexagon Instruction Formats --*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                         Hexagon Intruction Flags +
//
//                    *** Must match HexagonBaseInfo.h ***
//===----------------------------------------------------------------------===//

class Type<bits<5> t> {
  bits<5> Value = t;
}
def TypePSEUDO : Type<0>;
def TypeALU32  : Type<1>;
def TypeCR     : Type<2>;
def TypeJR     : Type<3>;
def TypeJ      : Type<4>;
def TypeLD     : Type<5>;
def TypeST     : Type<6>;
def TypeSYSTEM : Type<7>;
def TypeXTYPE  : Type<8>;
def TypeMARKER : Type<31>;

//===----------------------------------------------------------------------===//
//                         Intruction Class Declaration +
//===----------------------------------------------------------------------===//

class InstHexagon<dag outs, dag ins, string asmstr, list<dag> pattern,
                  string cstr, InstrItinClass itin, Type type> : Instruction {
  field bits<32> Inst;

  let Namespace = "Hexagon";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  let Constraints = cstr;
  let Itinerary = itin;
  let Size = 4;

  // *** Must match HexagonBaseInfo.h ***
  // Instruction type according to the ISA.
  Type HexagonType = type;
  let TSFlags{4-0} = HexagonType.Value;
  // Solo instructions, i.e., those that cannot be in a packet with others.
  bits<1> isHexagonSolo = 0;
  let TSFlags{5} = isHexagonSolo;
  // Predicated instructions.
  bits<1> isPredicated = 0;
  let TSFlags{6} = isPredicated;

  // Dot new value store instructions.
  bits<1> isNVStore = 0;
  let TSFlags{8} = isNVStore;

  // Fields used for relation models.
  string BaseOpcode = "";
  string CextOpcode = "";
  string PredSense = "";
  string PNewValue = "";
  string InputType = "";    // Input is "imm" or "reg" type.
  // *** The code above must match HexagonBaseInfo.h ***
}

//===----------------------------------------------------------------------===//
//                         Intruction Classes Definitions +
//===----------------------------------------------------------------------===//

// LD Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class LDInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", LD, TypeLD> {
  bits<5> rd;
  bits<5> rs;
  bits<13> imm13;
}

class LDInst2<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", LD, TypeLD> {
  bits<5> rd;
  bits<5> rs;
  bits<13> imm13;
  let mayLoad = 1;
}

// LD Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class LDInstPost<dag outs, dag ins, string asmstr, list<dag> pattern,
                 string cstr>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, LD, TypeLD> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<13> imm13;
}

// ST Instruction Class in V2/V3 can take SLOT0 only.
// ST Instruction Class in V4    can take SLOT0 & SLOT1.
// Definition of the instruction class CHANGED from V2/V3 to V4.
class STInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", ST, TypeST> {
  bits<5> rd;
  bits<5> rs;
  bits<13> imm13;
}

class STInst2<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", ST, TypeST> {
  bits<5> rd;
  bits<5> rs;
  bits<13> imm13;
  let mayStore = 1;
}

// SYSTEM Instruction Class in V4 can take SLOT0 only
// In V2/V3 we used ST for this but in v4 ST can take SLOT0 or SLOT1.
class SYSInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", SYS, TypeSYSTEM> {
  bits<5> rd;
  bits<5> rs;
  bits<13> imm13;
}

// ST Instruction Class in V2/V3 can take SLOT0 only.
// ST Instruction Class in V4    can take SLOT0 & SLOT1.
// Definition of the instruction class CHANGED from V2/V3 to V4.
class STInstPost<dag outs, dag ins, string asmstr, list<dag> pattern,
                 string cstr>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, ST, TypeST> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<13> imm13;
}

// ALU32 Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class ALU32Type<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstHexagon<outs, ins, asmstr, pattern, "", ALU32, TypeALU32> {
  bits<5>  rd;
  bits<5>  rs;
  bits<5>  rt;
  bits<16> imm16;
  bits<16> imm16_2;
}

// ALU64 Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from ALU64 to XTYPE from V2/V3 to V4.
class ALU64Type<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstHexagon<outs, ins, asmstr, pattern, "", ALU64, TypeXTYPE> {
  bits<5>  rd;
  bits<5>  rs;
  bits<5>  rt;
  bits<16> imm16;
  bits<16> imm16_2;
}

class ALU64_acc<dag outs, dag ins, string asmstr, list<dag> pattern,
   string cstr>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, ALU64, TypeXTYPE> {
  bits<5>  rd;
  bits<5>  rs;
  bits<5>  rt;
  bits<16> imm16;
  bits<16> imm16_2;
}

// M Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from M to XTYPE from V2/V3 to V4.
class MInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", M, TypeXTYPE> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
}

// M Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from M to XTYPE from V2/V3 to V4.
class MInst_acc<dag outs, dag ins, string asmstr, list<dag> pattern,
    string cstr>
    : InstHexagon<outs, ins, asmstr, pattern, cstr, M, TypeXTYPE> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
}

// S Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from S to XTYPE from V2/V3 to V4.
class SInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", S, TypeXTYPE> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
}

// S Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from S to XTYPE from V2/V3 to V4.
class SInst_acc<dag outs, dag ins, string asmstr, list<dag> pattern,
   string cstr>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, S, TypeXTYPE> {
//  : InstHexagon<outs, ins, asmstr, pattern, cstr,  S> {
//  : InstHexagon<outs, ins, asmstr, pattern, cstr, !if(V4T, XTYPE_V4, S)> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
}

// J Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class JType<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", J, TypeJ> {
  bits<16> imm16;
}

// JR Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class JRType<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", JR, TypeJR> {
  bits<5>  rs;
  bits<5>  pu; // Predicate register
}

// CR Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class CRInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", CR, TypeCR> {
  bits<5> rs;
  bits<10> imm10;
}

class Marker<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", MARKER, TypeMARKER> {
  let isCodeGenOnly = 1;
  let isPseudo = 1;
}

class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", PSEUDO, TypePSEUDO> {
  let isCodeGenOnly = 1;
  let isPseudo = 1;
}

//===----------------------------------------------------------------------===//
//                         Intruction Classes Definitions -
//===----------------------------------------------------------------------===//


//
// ALU32 patterns
//.
class ALU32_rr<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU32Type<outs, ins, asmstr, pattern> {
}

class ALU32_ir<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU32Type<outs, ins, asmstr, pattern> {
   let rt{0-4} = 0;
}

class ALU32_ri<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU32Type<outs, ins, asmstr, pattern> {
  let rt{0-4} = 0;
}

class ALU32_ii<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU32Type<outs, ins, asmstr, pattern> {
  let rt{0-4} = 0;
}

//
// ALU64 patterns.
//
class ALU64_rr<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU64Type<outs, ins, asmstr, pattern> {
}

class ALU64_ri<dag outs, dag ins, string asmstr, list<dag> pattern>
   : ALU64Type<outs, ins, asmstr, pattern> {
  let rt{0-4} = 0;
}

// J Type Instructions.
class JInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : JType<outs, ins, asmstr, pattern> {
}

// JR type Instructions.
class JRInst<dag outs, dag ins, string asmstr, list<dag> pattern>
  : JRType<outs, ins, asmstr, pattern> {
}


// Post increment ST Instruction.
class STInstPI<dag outs, dag ins, string asmstr, list<dag> pattern,
               string cstr>
  : STInstPost<outs, ins, asmstr, pattern, cstr> {
  let rt{0-4} = 0;
}

class STInst2PI<dag outs, dag ins, string asmstr, list<dag> pattern,
                string cstr>
  : STInstPost<outs, ins, asmstr, pattern, cstr> {
  let rt{0-4} = 0;
  let mayStore = 1;
}

// Post increment LD Instruction.
class LDInstPI<dag outs, dag ins, string asmstr, list<dag> pattern,
               string cstr>
  : LDInstPost<outs, ins, asmstr, pattern, cstr> {
  let rt{0-4} = 0;
}

class LDInst2PI<dag outs, dag ins, string asmstr, list<dag> pattern,
                string cstr>
  : LDInstPost<outs, ins, asmstr, pattern, cstr> {
  let rt{0-4} = 0;
  let mayLoad = 1;
}

//===----------------------------------------------------------------------===//
// V4 Instruction Format Definitions +
//===----------------------------------------------------------------------===//

include "HexagonInstrFormatsV4.td"

//===----------------------------------------------------------------------===//
// V4 Instruction Format Definitions +
//===----------------------------------------------------------------------===//
