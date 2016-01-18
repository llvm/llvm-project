//==- HexagonInstrFormats.td - Hexagon Instruction Formats --*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V4 instruction classes in TableGen format.
//
//===----------------------------------------------------------------------===//

//----------------------------------------------------------------------------//
//                         Hexagon Instruction Flags
//
//                        *** Must match BaseInfo.h ***
//----------------------------------------------------------------------------//

def TypeMEMOP    : IType<9>;
def TypeNV       : IType<10>;
def TypeDUPLEX   : IType<11>;
def TypeCOMPOUND : IType<12>;
def TypeAG_VX    : IType<28>;
def TypeAG_VM    : IType<29>;
def TypePREFIX   : IType<30>;

//                      Duplex Instruction Class Declaration
//===----------------------------------------------------------------------===//

class OpcodeDuplex {
  field bits<32> Inst = ?; // Default to an invalid insn.
  bits<4> IClass = 0; // ICLASS
  bits<13> ISubHi = 0; // Low sub-insn
  bits<13> ISubLo = 0; // High sub-insn

  let Inst{31-29} = IClass{3-1};
  let Inst{13}    = IClass{0};
  let Inst{15-14} = 0;
  let Inst{28-16} = ISubHi;
  let Inst{12-0}  = ISubLo;
}

class InstDuplex<bits<4> iClass, list<dag> pattern = [],
                 string cstr = "">
  : Instruction, OpcodeDuplex {
  let Namespace = "Hexagon";
  IType Type = TypeDUPLEX;  // uses slot 0,1
  let isCodeGenOnly = 1;
  let hasSideEffects = 0;
  dag OutOperandList = (outs);
  dag InOperandList = (ins);
  let IClass = iClass;
  let Constraints = cstr;
  let Itinerary = DUPLEX;
  let Size = 4;

  // SoftFail is a field the disassembler can use to provide a way for
  // instructions to not match without killing the whole decode process. It is
  // mainly used for ARM, but Tablegen expects this field to exist or it fails
  // to build the decode table.
  field bits<32> SoftFail = 0;

  // *** Must match MCTargetDesc/HexagonBaseInfo.h ***

  let TSFlags{4-0} = Type.Value;

  // Predicated instructions.
  bits<1> isPredicated = 0;
  let TSFlags{6} = isPredicated;
  bits<1> isPredicatedFalse = 0;
  let TSFlags{7} = isPredicatedFalse;
  bits<1> isPredicatedNew = 0;
  let TSFlags{8} = isPredicatedNew;

  // New-value insn helper fields.
  bits<1> isNewValue = 0;
  let TSFlags{9} = isNewValue; // New-value consumer insn.
  bits<1> hasNewValue = 0;
  let TSFlags{10} = hasNewValue; // New-value producer insn.
  bits<3> opNewValue = 0;
  let TSFlags{13-11} = opNewValue; // New-value produced operand.
  bits<1> isNVStorable = 0;
  let TSFlags{14} = isNVStorable; // Store that can become new-value store.
  bits<1> isNVStore = 0;
  let TSFlags{15} = isNVStore; // New-value store insn.

  // Immediate extender helper fields.
  bits<1> isExtendable = 0;
  let TSFlags{16} = isExtendable; // Insn may be extended.
  bits<1> isExtended = 0;
  let TSFlags{17} = isExtended; // Insn must be extended.
  bits<3> opExtendable = 0;
  let TSFlags{20-18} = opExtendable; // Which operand may be extended.
  bits<1> isExtentSigned = 0;
  let TSFlags{21} = isExtentSigned; // Signed or unsigned range.
  bits<5> opExtentBits = 0;
  let TSFlags{26-22} = opExtentBits; //Number of bits of range before extending.
  bits<2> opExtentAlign = 0;
  let TSFlags{28-27} = opExtentAlign; // Alignment exponent before extending.
}

//----------------------------------------------------------------------------//
//                         Instruction Classes Definitions
//----------------------------------------------------------------------------//

//
// NV type instructions.
//
class NVInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "", InstrItinClass itin = NCJ_tc_3or4stall_SLOT0>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeNV>, OpcodeHexagon;

class NVInst_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "", InstrItinClass itin = NCJ_tc_3or4stall_SLOT0>
  : NVInst<outs, ins, asmstr, pattern, cstr, itin>;

// Definition of Post increment new value store.
class NVInstPost_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "", InstrItinClass itin = ST_tc_st_SLOT0>
  : NVInst<outs, ins, asmstr, pattern, cstr, itin>;

// Post increment ST Instruction.
let mayStore = 1 in
class NVInstPI_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "", InstrItinClass itin = ST_tc_st_SLOT0>
  : NVInst<outs, ins, asmstr, pattern, cstr, itin>;

// New-value conditional branch.
class NCJInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : NVInst<outs, ins, asmstr, pattern, cstr>;

let mayLoad = 1, mayStore = 1 in
class MEMInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "", InstrItinClass itin = V4LDST_tc_st_SLOT0>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeMEMOP>,
    OpcodeHexagon;

class MEMInst_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                 string cstr = "", InstrItinClass itin = V4LDST_tc_st_SLOT0>
  : MEMInst<outs, ins, asmstr, pattern, cstr, itin>;

let isCodeGenOnly = 1 in
class EXTENDERInst<dag outs, dag ins, string asmstr, list<dag> pattern = []>
  : InstHexagon<outs, ins, asmstr, pattern, "", EXTENDER_tc_1_SLOT0123,
                TypePREFIX>, OpcodeHexagon;

class SUBInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, "", PREFIX, TypeDUPLEX>,
    OpcodeHexagon;

class CJInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, COMPOUND, TypeCOMPOUND>,
    OpcodeHexagon;
