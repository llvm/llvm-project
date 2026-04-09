//===-- LX32InstrFormats.td - LX32 Instruction Format Definitions ---------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the encoding-level building blocks for all LX32
// instruction formats.  Every class here models exactly one concern:
//
//   Field mixins  — carry a single named bit-field (rd, rs1, rs2, funct3,
//                   and the immediate variants).
//   LX32Opcode   — pairs a 7-bit opcode value with a human-readable name.
//   LXInstCommon — the root Instruction subclass shared by all formats.
//   LXInst       — places the 7-bit opcode in Inst{6-0}.
//   LXInstR/I/S/B/U/J — one class per ISA format; each sets the remaining
//                   bit fields according to the LX32 encoding specification.
//   RegisterOperand helpers — typed wrappers around GPR/GPRNOX0 for use in
//                   instruction operand lists.
//
// Nothing in this file sets semantic attributes (mayLoad, mayStore,
// isBranch, etc.).  Those belong in LX32InstrInfo.td where the actual
// instruction definitions live.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Minimal field mixins — single source of truth for bit-field names
//===----------------------------------------------------------------------===//

/// Destination register field (bits [11:7] in all formats that have one).
class LXInstRD { bits<5> rd; }

/// Source register 1 field (bits [19:15]).
class LXInstRS1 { bits<5> rs1; }

/// Source register 2 field (bits [24:20]).
class LXInstRS2 { bits<5> rs2; }

/// Convenience mixin: both source registers.
class LXInstRS1_RS2 : LXInstRS1, LXInstRS2 {}

/// funct3 discriminator field (bits [14:12]).
class LXInstFunct3<bits<3> funct3_val> {
  bits<3> funct3 = funct3_val;
}

//===----------------------------------------------------------------------===//
// Immediate field mixins
//===----------------------------------------------------------------------===//

/// 12-bit sign-extended immediate.  Used by I-type and S-type.
class LXImm12 { bits<12> imm12; }

/// 13-bit sign-extended branch offset.
/// Models bits [12:1] of the byte offset — bit 0 is always zero and is
/// not stored.  The B-type encoder scrambles these bits across the word;
/// see LXInstBBase below for the exact mapping.
class LXImm13 { bits<13> imm13; }

/// 20-bit immediate.  Used by U-type (LUI / AUIPC) placed in Inst[31:12].
class LXImm20 { bits<20> imm20; }

/// 21-bit sign-extended jump offset.
/// Models bits [20:1] of the byte offset — bit 0 is always zero and is
/// not stored.  The J-type encoder scrambles these bits; see LXInstJBase.
class LXImm21 { bits<21> imm21; }

//===----------------------------------------------------------------------===//
// Opcode descriptor
//
// LX32Opcode pairs the 7-bit opcode value with a name string used in
// diagnostics.  One instance per opcode group (e.g., OPC_ARITH = 0110011).
//===----------------------------------------------------------------------===//

class LX32Opcode<string name, bits<7> val> {
  string  Name = name;
  bits<7> Val  = val;
}

//===----------------------------------------------------------------------===//
// Instruction base classes
//===----------------------------------------------------------------------===//

/// LXInstCommon — root for all LX32 instructions.
///
/// Defines the 32-bit Inst field, wires OutOperandList / InOperandList,
/// and assembles the AsmString from the opcode mnemonic and argument string.
/// The AsmString logic: if argstr is non-empty the result is
/// "opcodestr\targstr"; if argstr is empty the result is just "opcodestr"
/// (used for pseudo-instructions like "ret" and "nop").
class LXInstCommon<dag outs, dag ins, string opcodestr, string argstr>
    : Instruction {
  let Namespace       = "LX32";
  field bits<32> Inst;
  let Size            = 4;
  dag OutOperandList  = outs;
  dag InOperandList   = ins;
  let AsmString       = !if(!empty(argstr),
                            opcodestr,
                            opcodestr # "\t" # argstr);
}

/// LXInst — places the 7-bit opcode in Inst{6-0} and inherits LXInstCommon.
/// All format-specific classes derive from this.
class LXInst<LX32Opcode opcode, dag outs, dag ins,
             string opcodestr, string argstr>
    : LXInstCommon<outs, ins, opcodestr, argstr> {
  let Inst{6-0} = opcode.Val;
}

//===----------------------------------------------------------------------===//
// RegisterOperand helpers
//
// Typed wrappers used in instruction operand lists.  Using RegisterOperand
// (rather than raw RegisterClass) lets the assembler parser attach the
// correct register class to each operand position without guessing.
//===----------------------------------------------------------------------===//

def LX32GPROut      : RegisterOperand<GPR>;
def LX32GPRIn       : RegisterOperand<GPR>;
def LX32GPRNOX0Out  : RegisterOperand<GPRNOX0>;

//===----------------------------------------------------------------------===//
// R-Type format  —  register-register operations
//
//  31      25 24    20 19    15 14  12 11     7 6      0
//  [ funct7 ] [ rs2  ] [ rs1  ] [fn3] [  rd  ] [opcode]
//===----------------------------------------------------------------------===//

/// LXInstRBase — R-type without the funct7 field.
/// Used as an intermediate when funct7 is supplied by a derived class.
class LXInstRBase<bits<3> funct3, LX32Opcode opcode,
                  dag outs, dag ins, string opcodestr, string argstr>
    : LXInst<opcode, outs, ins, opcodestr, argstr>,
      LXInstRD, LXInstRS1_RS2, LXInstFunct3<funct3> {
  let Inst{11-7}  = rd;
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{14-12} = funct3;
}

/// LXInstR — full R-type including funct7.
class LXInstR<bits<7> funct7, bits<3> funct3, LX32Opcode opcode,
              dag outs, dag ins, string opcodestr, string argstr>
    : LXInstRBase<funct3, opcode, outs, ins, opcodestr, argstr> {
  let Inst{31-25} = funct7;
}

//===----------------------------------------------------------------------===//
// I-Type format  —  immediate arithmetic, loads, JALR
//
//  31          20 19    15 14  12 11     7 6      0
//  [   imm12   ] [ rs1  ] [fn3] [  rd  ] [opcode]
//===----------------------------------------------------------------------===//

/// LXInstIBase — I-type without semantic attributes.
class LXInstIBase<bits<3> funct3, LX32Opcode opcode,
                  dag outs, dag ins, string opcodestr, string argstr>
    : LXInst<opcode, outs, ins, opcodestr, argstr>,
      LXInstRD, LXInstRS1, LXInstFunct3<funct3>, LXImm12 {
  let Inst{11-7}  = rd;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{31-20} = imm12;
}

/// LXInstI — I-type for immediate arithmetic and JALR.
/// Semantic attributes (mayLoad, hasSideEffects, etc.) are set in
/// LX32InstrInfo.td via let-blocks at the instruction-definition site.
class LXInstI<bits<3> funct3, LX32Opcode opcode,
              dag outs, dag ins, string opcodestr, string argstr>
    : LXInstIBase<funct3, opcode, outs, ins, opcodestr, argstr> {}

//===----------------------------------------------------------------------===//
// S-Type format  —  stores
//
//  31      25 24    20 19    15 14  12 11     7 6      0
//  [imm12[11:5]] [rs2] [ rs1 ] [fn3] [imm12[4:0]] [opcode]
//===----------------------------------------------------------------------===//

/// LXInstMemBase — S-type field layout.
/// imm12 is split: the upper 7 bits go to Inst{31:25}, the lower 5 bits to
/// Inst{11:7}.  No rd field in stores.
class LXInstMemBase<bits<3> funct3, LX32Opcode opcode,
                    dag outs, dag ins, string opcodestr, string argstr>
    : LXInst<opcode, outs, ins, opcodestr, argstr>,
      LXInstRS1_RS2, LXInstFunct3<funct3>, LXImm12 {
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{14-12} = funct3;
  let Inst{31-25} = imm12{11-5};
  let Inst{11-7}  = imm12{4-0};
}

/// LXInstS — S-type for store instructions.
class LXInstS<bits<3> funct3, LX32Opcode opcode,
              dag outs, dag ins, string opcodestr, string argstr>
    : LXInstMemBase<funct3, opcode, outs, ins, opcodestr, argstr> {
  let mayStore       = 1;
  let hasSideEffects = 1;
}

//===----------------------------------------------------------------------===//
// B-Type format  —  conditional branches
//
//  31      25 24    20 19    15 14  12 11     7 6      0
//  [12|10:5]  [ rs2 ] [ rs1 ] [fn3] [4:1|11]  [opcode]
//
// imm13 represents bits [12:1] of the signed byte offset.
// Bit 0 is always zero (instructions are 4-byte aligned) and is not encoded.
// The bit scrambling here matches the LX32 B-type specification:
//   Inst[31]    ← imm13[12]   (sign bit)
//   Inst[30:25] ← imm13[10:5]
//   Inst[11:8]  ← imm13[4:1]
//   Inst[7]     ← imm13[11]
//===----------------------------------------------------------------------===//

/// LXInstBBase — B-type field layout and bit scrambling.
class LXInstBBase<bits<3> funct3, LX32Opcode opcode,
                  dag outs, dag ins, string opcodestr, string argstr>
    : LXInst<opcode, outs, ins, opcodestr, argstr>,
      LXInstRS1_RS2, LXInstFunct3<funct3>, LXImm13 {
  let Inst{19-15} = rs1;
  let Inst{24-20} = rs2;
  let Inst{14-12} = funct3;
  let Inst{31}    = imm13{12};
  let Inst{30-25} = imm13{10-5};
  let Inst{11-8}  = imm13{4-1};
  let Inst{7}     = imm13{11};
}

/// LXInstB — B-type for conditional branches.
/// isBranch, isTerminator, and hasSideEffects are set here because they
/// apply uniformly to every B-type instruction.
class LXInstB<bits<3> funct3, LX32Opcode opcode,
              dag outs, dag ins, string opcodestr, string argstr>
    : LXInstBBase<funct3, opcode, outs, ins, opcodestr, argstr> {
  let isBranch       = 1;
  let isTerminator   = 1;
  let hasSideEffects = 1;
}

//===----------------------------------------------------------------------===//
// U-Type format  —  upper immediate (LUI / AUIPC)
//
//  31          12 11     7 6      0
//  [   imm20   ] [  rd  ] [opcode]
//===----------------------------------------------------------------------===//

/// LXInstUBase — U-type field layout.
class LXInstUBase<LX32Opcode opcode,
                  dag outs, dag ins, string opcodestr, string argstr>
    : LXInst<opcode, outs, ins, opcodestr, argstr>, LXInstRD, LXImm20 {
  let Inst{11-7}  = rd;
  let Inst{31-12} = imm20;
}

/// LXInstU — U-type.  Semantic attributes set per-instruction in
/// LX32InstrInfo.td.
class LXInstU<LX32Opcode opcode,
              dag outs, dag ins, string opcodestr, string argstr>
    : LXInstUBase<opcode, outs, ins, opcodestr, argstr> {}

//===----------------------------------------------------------------------===//
// J-Type format  —  unconditional jump-and-link (JAL)
//
//  31      12 11     7 6      0
//  [imm21 scrambled] [  rd  ] [opcode]
//
// imm21 represents bits [20:1] of the signed byte offset.
// Bit 0 is always zero and is not encoded.
// The bit scrambling matches the LX32 J-type specification:
//   Inst[31]    ← imm21[20]   (sign bit)
//   Inst[30:21] ← imm21[10:1]
//   Inst[20]    ← imm21[11]
//   Inst[19:12] ← imm21[19:12]
//===----------------------------------------------------------------------===//

/// LXInstJBase — J-type field layout and bit scrambling.
class LXInstJBase<LX32Opcode opcode,
                  dag outs, dag ins, string opcodestr, string argstr>
    : LXInst<opcode, outs, ins, opcodestr, argstr>, LXInstRD, LXImm21 {
  let Inst{11-7}  = rd;
  let Inst{31}    = imm21{20};
  let Inst{30-21} = imm21{10-1};
  let Inst{20}    = imm21{11};
  let Inst{19-12} = imm21{19-12};
}

/// LXInstJ — J-type for JAL.
/// isBranch, isTerminator, isBarrier, and hasSideEffects are set here
/// because they apply to every J-type instruction.
class LXInstJ<LX32Opcode opcode,
              dag outs, dag ins, string opcodestr, string argstr>
    : LXInstJBase<opcode, outs, ins, opcodestr, argstr> {
  let isBranch       = 1;
  let isTerminator   = 1;
  let isBarrier      = 1;
  let hasSideEffects = 1;
}
