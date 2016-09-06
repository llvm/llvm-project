//==- HexagonInstrFormats.td - Hexagon Instruction Formats --*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                         Hexagon Instruction Flags +
//
//                    *** Must match HexagonBaseInfo.h ***
//===----------------------------------------------------------------------===//

class IType<bits<5> t> {
  bits<5> Value = t;
}
def TypePSEUDO : IType<0>;
def TypeALU32  : IType<1>;
def TypeCR     : IType<2>;
def TypeJR     : IType<3>;
def TypeJ      : IType<4>;
def TypeLD     : IType<5>;
def TypeST     : IType<6>;
def TypeSYSTEM : IType<7>;
def TypeXTYPE  : IType<8>;
def TypeENDLOOP: IType<31>;

// Maintain list of valid subtargets for each instruction.
class SubTarget<bits<6> value> {
  bits<6> Value = value;
}

def HasAnySubT    : SubTarget<0x3f>;  // 111111
def HasV5SubT     : SubTarget<0x3e>;  // 111110
def HasV55SubT    : SubTarget<0x3c>;  // 111100
def HasV60SubT    : SubTarget<0x38>;  // 111000

// Addressing modes for load/store instructions
class AddrModeType<bits<3> value> {
  bits<3> Value = value;
}

def NoAddrMode     : AddrModeType<0>;  // No addressing mode
def Absolute       : AddrModeType<1>;  // Absolute addressing mode
def AbsoluteSet    : AddrModeType<2>;  // Absolute set addressing mode
def BaseImmOffset  : AddrModeType<3>;  // Indirect with offset
def BaseLongOffset : AddrModeType<4>;  // Indirect with long offset
def BaseRegOffset  : AddrModeType<5>;  // Indirect with register offset
def PostInc        : AddrModeType<6>;  // Post increment addressing mode

class MemAccessSize<bits<4> value> {
  bits<4> Value = value;
}

def NoMemAccess      : MemAccessSize<0>;// Not a memory acces instruction.
def ByteAccess       : MemAccessSize<1>;// Byte access instruction (memb).
def HalfWordAccess   : MemAccessSize<2>;// Half word access instruction (memh).
def WordAccess       : MemAccessSize<3>;// Word access instruction (memw).
def DoubleWordAccess : MemAccessSize<4>;// Double word access instruction (memd)
def Vector64Access   : MemAccessSize<7>;// Vector access instruction (memv)
def Vector128Access  : MemAccessSize<8>;// Vector access instruction (memv)


//===----------------------------------------------------------------------===//
//                         Instruction Class Declaration +
//===----------------------------------------------------------------------===//

class OpcodeHexagon {
  field bits<32> Inst = ?; // Default to an invalid insn.
  bits<4> IClass = 0; // ICLASS

  let Inst{31-28} = IClass;

  bits<1> zero = 0;
}

class InstHexagon<dag outs, dag ins, string asmstr, list<dag> pattern,
                  string cstr, InstrItinClass itin, IType type>
  : Instruction {
  let Namespace = "Hexagon";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  let Constraints = cstr;
  let Itinerary = itin;
  let Size = 4;

  // SoftFail is a field the disassembler can use to provide a way for
  // instructions to not match without killing the whole decode process. It is
  // mainly used for ARM, but Tablegen expects this field to exist or it fails
  // to build the decode table.
  field bits<32> SoftFail = 0;

  // *** Must match MCTargetDesc/HexagonBaseInfo.h ***

  // Instruction type according to the ISA.
  IType Type = type;
  let TSFlags{4-0} = Type.Value;

  // Solo instructions, i.e., those that cannot be in a packet with others.
  bits<1> isSolo = 0;
  let TSFlags{5} = isSolo;
  // Packed only with A or X-type instructions.
  bits<1> isSoloAX = 0;
  let TSFlags{6} = isSoloAX;
  // Only A-type instruction in first slot or nothing.
  bits<1> isSoloAin1 = 0;
  let TSFlags{7} = isSoloAin1;

  // Predicated instructions.
  bits<1> isPredicated = 0;
  let TSFlags{8} = isPredicated;
  bits<1> isPredicatedFalse = 0;
  let TSFlags{9} = isPredicatedFalse;
  bits<1> isPredicatedNew = 0;
  let TSFlags{10} = isPredicatedNew;
  bits<1> isPredicateLate = 0;
  let TSFlags{11} = isPredicateLate; // Late predicate producer insn.

  // New-value insn helper fields.
  bits<1> isNewValue = 0;
  let TSFlags{12} = isNewValue; // New-value consumer insn.
  bits<1> hasNewValue = 0;
  let TSFlags{13} = hasNewValue; // New-value producer insn.
  bits<3> opNewValue = 0;
  let TSFlags{16-14} = opNewValue; // New-value produced operand.
  bits<1> isNVStorable = 0;
  let TSFlags{17} = isNVStorable; // Store that can become new-value store.
  bits<1> isNVStore = 0;
  let TSFlags{18} = isNVStore; // New-value store insn.
  bits<1> isCVLoadable = 0;
  let TSFlags{19} = isCVLoadable; // Load that can become cur-value load.
  bits<1> isCVLoad = 0;
  let TSFlags{20} = isCVLoad; // Cur-value load insn.

  // Immediate extender helper fields.
  bits<1> isExtendable = 0;
  let TSFlags{21} = isExtendable; // Insn may be extended.
  bits<1> isExtended = 0;
  let TSFlags{22} = isExtended; // Insn must be extended.
  bits<3> opExtendable = 0;
  let TSFlags{25-23} = opExtendable; // Which operand may be extended.
  bits<1> isExtentSigned = 0;
  let TSFlags{26} = isExtentSigned; // Signed or unsigned range.
  bits<5> opExtentBits = 0;
  let TSFlags{31-27} = opExtentBits; //Number of bits of range before extending.
  bits<2> opExtentAlign = 0;
  let TSFlags{33-32} = opExtentAlign; // Alignment exponent before extending.

  // If an instruction is valid on a subtarget, set the corresponding
  // bit from validSubTargets.
  // By default, instruction is valid on all subtargets.
  SubTarget validSubTargets = HasAnySubT;
  let TSFlags{39-34} = validSubTargets.Value;

  // Addressing mode for load/store instructions.
  AddrModeType addrMode = NoAddrMode;
  let TSFlags{42-40} = addrMode.Value;

  // Memory access size for mem access instructions (load/store)
  MemAccessSize accessSize = NoMemAccess;
  let TSFlags{46-43} = accessSize.Value;

  bits<1> isTaken = 0;
  let TSFlags {47} = isTaken; // Branch prediction.

  bits<1> isFP = 0;
  let TSFlags {48} = isFP; // Floating-point.

  bits<1> hasNewValue2 = 0;
  let TSFlags{50} = hasNewValue2; // Second New-value producer insn.
  bits<3> opNewValue2 = 0;
  let TSFlags{53-51} = opNewValue2; // Second New-value produced operand.

  bits<1> isAccumulator = 0;
  let TSFlags{54} = isAccumulator;

  // Fields used for relation models.
  bit isNonTemporal = 0;
  string isNT = ""; // set to "true" for non-temporal vector stores.
  string BaseOpcode = "";
  string CextOpcode = "";
  string PredSense = "";
  string PNewValue = "";
  string NValueST  = "";    // Set to "true" for new-value stores.
  string InputType = "";    // Input is "imm" or "reg" type.
  string isFloat = "false"; // Set to "true" for the floating-point load/store.
  string isBrTaken = !if(isTaken, "true", "false"); // Set to "true"/"false" for jump instructions

  let PredSense = !if(isPredicated, !if(isPredicatedFalse, "false", "true"),
                                    "");
  let PNewValue = !if(isPredicatedNew, "new", "");
  let NValueST = !if(isNVStore, "true", "false");
  let isNT = !if(isNonTemporal, "true", "false");

  // *** Must match MCTargetDesc/HexagonBaseInfo.h ***
}

//===----------------------------------------------------------------------===//
//                         Instruction Classes Definitions +
//===----------------------------------------------------------------------===//

// LD Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
let mayLoad = 1 in
class LDInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "", InstrItinClass itin = LD_tc_ld_SLOT01>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeLD>, OpcodeHexagon;

let mayLoad = 1 in
class LDInst2<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : LDInst<outs, ins, asmstr, pattern, cstr>;

class CONSTLDInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                  string cstr = "">
  : LDInst<outs, ins, asmstr, pattern, cstr>;

// LD Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class LDInstPost<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                 string cstr = "">
  : LDInst<outs, ins, asmstr, pattern, cstr>;

let mayLoad = 1 in
class LD0Inst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "", InstrItinClass itin=LD_tc_ld_SLOT0>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeLD>, OpcodeHexagon;

let mayLoad = 1 in
class LD1Inst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "", InstrItinClass itin=LD_tc_ld_SLOT0>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeLD>;

// ST Instruction Class in V2/V3 can take SLOT0 only.
// ST Instruction Class in V4    can take SLOT0 & SLOT1.
// Definition of the instruction class CHANGED from V2/V3 to V4.
let mayStore = 1 in
class STInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "", InstrItinClass itin = ST_tc_st_SLOT01>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeST>, OpcodeHexagon;

class STInst2<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : STInst<outs, ins, asmstr, pattern, cstr>;

let mayStore = 1 in
class ST0Inst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "", InstrItinClass itin = ST_tc_ld_SLOT0>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeST>, OpcodeHexagon;

// Same as ST0Inst but doesn't derive from OpcodeHexagon.
let mayStore = 1 in
class ST1Inst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "", InstrItinClass itin = ST_tc_st_SLOT0>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeST>;

// ST Instruction Class in V2/V3 can take SLOT0 only.
// ST Instruction Class in V4    can take SLOT0 & SLOT1.
// Definition of the instruction class CHANGED from V2/V3 to V4.
class STInstPost<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                 string cstr = "", InstrItinClass itin = ST_tc_st_SLOT01>
  : STInst<outs, ins, asmstr, pattern, cstr, itin>;

// SYSTEM Instruction Class in V4 can take SLOT0 only
// In V2/V3 we used ST for this but in v4 ST can take SLOT0 or SLOT1.
class SYSInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "",  InstrItinClass itin = ST_tc_3stall_SLOT0>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeSYSTEM>,
    OpcodeHexagon;

// ALU32 Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class ALU32Inst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "", InstrItinClass itin = ALU32_2op_tc_1_SLOT0123>
 : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeALU32>, OpcodeHexagon;

// ALU64 Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from ALU64 to XTYPE from V2/V3 to V4.
class ALU64Inst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "", InstrItinClass itin = ALU64_tc_2_SLOT23>
   : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeXTYPE>,
     OpcodeHexagon;

class ALU64_acc<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "", InstrItinClass itin = ALU64_tc_2_SLOT23>
  : ALU64Inst<outs, ins, asmstr, pattern, cstr, itin>;


// M Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from M to XTYPE from V2/V3 to V4.
class MInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
            string cstr = "", InstrItinClass itin = M_tc_3x_SLOT23>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeXTYPE>,
    OpcodeHexagon;

// Same as above but doesn't derive from OpcodeHexagon
class MInst2<dag outs, dag ins, string asmstr, list<dag> pattern = [],
            string cstr = "", InstrItinClass itin = M_tc_3x_SLOT23>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeXTYPE>;

// M Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from M to XTYPE from V2/V3 to V4.
class MInst_acc<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "", InstrItinClass itin = M_tc_2_SLOT23>
    : MInst<outs, ins, asmstr, pattern, cstr, itin>;

// S Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from S to XTYPE from V2/V3 to V4.
class SInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
            string cstr = "", InstrItinClass itin = S_2op_tc_1_SLOT23>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeXTYPE>,
    OpcodeHexagon;

class SInst2<dag outs, dag ins, string asmstr, list<dag> pattern = [],
            string cstr = "", InstrItinClass itin = S_2op_tc_1_SLOT23>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeXTYPE>;

// S Instruction Class in V2/V3.
// XTYPE Instruction Class in V4.
// Definition of the instruction class NOT CHANGED.
// Name of the Instruction Class changed from S to XTYPE from V2/V3 to V4.
class SInst_acc<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "", InstrItinClass itin = S_3op_tc_1_SLOT23>
  : SInst<outs, ins, asmstr, pattern, cstr, itin>;

// J Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class JInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
            string cstr = "", InstrItinClass itin = J_tc_2early_SLOT23>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeJ>, OpcodeHexagon;

class JInst_CJUMP_UCJUMP<dag outs, dag ins, string asmstr, list<dag> pattern = [],
            string cstr = "", InstrItinClass itin = J_tc_2early_CJUMP_UCJUMP_ARCHDEPSLOT>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeJ>, OpcodeHexagon;

// JR Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class JRInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "", InstrItinClass itin = J_tc_2early_SLOT2>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeJR>, OpcodeHexagon;

// CR Instruction Class in V2/V3/V4.
// Definition of the instruction class NOT CHANGED.
class CRInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "", InstrItinClass itin = CR_tc_2early_SLOT3>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeCR>, OpcodeHexagon;

let isCodeGenOnly = 1, isPseudo = 1 in
class Endloop<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "", InstrItinClass itin = J_tc_2early_SLOT0123>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeENDLOOP>,
    OpcodeHexagon;

let isCodeGenOnly = 1, isPseudo = 1 in
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, PSEUDO, TypePSEUDO>,
    OpcodeHexagon;

let isCodeGenOnly = 1, isPseudo = 1 in
class PseudoM<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr="">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, PSEUDOM, TypePSEUDO>,
    OpcodeHexagon;

//===----------------------------------------------------------------------===//
//                         Instruction Classes Definitions -
//===----------------------------------------------------------------------===//


//
// ALU32 patterns
//.
class ALU32_rr<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "", InstrItinClass itin = ALU32_2op_tc_1_SLOT0123>
   : ALU32Inst<outs, ins, asmstr, pattern, cstr, itin>;

class ALU32_ir<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "", InstrItinClass itin = ALU32_2op_tc_1_SLOT0123>
   : ALU32Inst<outs, ins, asmstr, pattern, cstr, itin>;

class ALU32_ri<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "", InstrItinClass itin = ALU32_2op_tc_1_SLOT0123>
   : ALU32Inst<outs, ins, asmstr, pattern, cstr, itin>;

class ALU32_ii<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "", InstrItinClass itin = ALU32_2op_tc_1_SLOT0123>
   : ALU32Inst<outs, ins, asmstr, pattern, cstr, itin>;

//
// ALU64 patterns.
//
class ALU64_rr<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "", InstrItinClass itin = ALU64_tc_1_SLOT23>
   : ALU64Inst<outs, ins, asmstr, pattern, cstr, itin>;

class ALU64_ri<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "", InstrItinClass itin = ALU64_tc_1_SLOT23>
   : ALU64Inst<outs, ins, asmstr, pattern, cstr, itin>;

// Post increment ST Instruction.
class STInstPI<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "">
  : STInst<outs, ins, asmstr, pattern, cstr>;

// Post increment LD Instruction.
class LDInstPI<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "">
  : LDInst<outs, ins, asmstr, pattern, cstr>;

//===----------------------------------------------------------------------===//
// V4 Instruction Format Definitions +
//===----------------------------------------------------------------------===//

include "HexagonInstrFormatsV4.td"

//===----------------------------------------------------------------------===//
// V4 Instruction Format Definitions +
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// V60 Instruction Format Definitions +
//===----------------------------------------------------------------------===//

include "HexagonInstrFormatsV60.td"

//===----------------------------------------------------------------------===//
// V60 Instruction Format Definitions +
//===----------------------------------------------------------------------===//
