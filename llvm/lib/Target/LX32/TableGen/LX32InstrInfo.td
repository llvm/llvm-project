//===-- LX32InstrInfo.td - LX32 Instruction Definitions -------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file defines the complete LX32 instruction set at the TableGen level.
// It is organized into the following sections:
//
//   Section 0   — Custom SDNodes and pattern fragments
//   Section 0.5 — Operand transformers (HI20 / LO12)
//   Section 1   — Operand type definitions
//   Section 2   — Opcode map
//   Section 3   — R-type instructions (register-register ALU)
//   Section 4   — I-type instructions (immediate ALU)
//   Section 5   — Load instructions
//   Section 6   — Store instructions
//   Section 7   — Branch instructions (B-type)
//   Section 8   — Jump instructions (JAL / JALR)
//   Section 9   — Upper-immediate instructions (LUI / AUIPC)
//   Section 10  — Constant-materialization patterns
//   Section 11  — Pseudo-instructions
//   Section 12  — Codegen-only pseudo overrides (stack adjustment, guards)
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Section 0 — Custom SDNodes and Pattern Fragments
//===----------------------------------------------------------------------===//

// LX32ret — custom DAG node for the LX32 return sequence.
//
// SDTypeProfile<0, 0, []> means zero results and zero fixed operands.
// The SDNPHasChain flag makes the node part of the chain so it is always
// scheduled last.  SDNPOptInGlue allows an optional glue edge from the
// CopyToReg nodes that load the return value into a0/a1.  SDNPVariadic
// permits the glue chain to hold an arbitrary number of glued nodes.
def SDT_LX32Ret : SDTypeProfile<0, 0, []>;
def LX32ret : SDNode<"LX32ISD::RET", SDT_LX32Ret,
                     [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

// Load pattern fragments — model sign- and zero-extending sub-word loads.
// These wrap the generic LLVM load nodes so that the instruction patterns
// in Section 5 can match them with a single, readable name.
def lx32_lh  : PatFrag<(ops node:$ptr), (i32 (sextloadi16 node:$ptr))>;
def lx32_lhu : PatFrag<(ops node:$ptr), (i32 (zextloadi16 node:$ptr))>;
def lx32_lb  : PatFrag<(ops node:$ptr), (i32 (sextloadi8  node:$ptr))>;
def lx32_lbu : PatFrag<(ops node:$ptr), (i32 (zextloadi8  node:$ptr))>;

//===----------------------------------------------------------------------===//
// Section 0.5 — Operand Transformers
//
// HI20 and LO12 split a 32-bit constant into the two pieces needed for
// the LUI + ADDI materialization sequence:
//
//   LUI  rd,      ((imm + 0x800) >> 12) & 0xFFFFF   ← upper 20 bits
//   ADDI rd, rd,  sign_extend_12(imm & 0xFFF)        ← lower 12 bits
//
// The +0x800 bias in HI20 compensates for the sign extension that ADDI
// applies to its 12-bit immediate: if bit 11 of the low part is 1, ADDI
// sign-extends it to a negative value, so LUI must load one higher.
//
// LO12 sign-extends the 12-bit result so that ADDI receives a value in
// the range [-2048, 2047].  Using getZExtValue() & 0xFFF without sign
// extension would produce values up to 4095, which do not fit in simm12
// and would silently truncate or mis-encode.
//===----------------------------------------------------------------------===//

def HI20 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(
      ((uint64_t)N->getZExtValue() + 0x800) >> 12, SDLoc(N), MVT::i32);
}]>;

def LO12 : SDNodeXForm<imm, [{
  // Sign-extend the low 12 bits so that the result is a valid simm12.
  return CurDAG->getTargetConstant(
      SignExtend64<12>(N->getZExtValue() & 0xfff), SDLoc(N), MVT::i32);
}]>;

//===----------------------------------------------------------------------===//
// Section 1 — Operand Definitions
//
// bare_symbol — a symbolic address operand with no relocation modifier.
//   Used by pseudo-instructions such as PseudoLA and PseudoCALL that take
//   a label or function name as an operand.
//
// simm12 / uimm5 / uimm12 / uimm20 — typed immediates.
//   Each specifies an EncoderMethod so that the MCCodeEmitter knows which
//   C++ function to call when encoding the operand into a binary word.
//   "getImmOpValue" is the generic encoder that simply writes the immediate
//   bits directly — suitable for any immediate that requires no bit
//   shuffling beyond what the format class already handles.
//
// simm13 / simm21 — branch and jump target operands.
//   These use specialized encoder methods because the branch (B-type) and
//   jump (J-type) formats scatter the immediate bits across the instruction
//   word in a non-contiguous order.  The encoder methods handle relocations
//   (symbols resolved at link time) by emitting fixup records.
//
// imm12 / imm5 — PatLeaf predicates.
//   Used in instruction patterns (not in operand lists) to restrict which
//   constant values a pattern may match.  TableGen would crash if a raw
//   Operand were used directly in a pattern — PatLeaf is the correct idiom.
//===----------------------------------------------------------------------===//

def bare_symbol : Operand<i32>;
def simm12 : Operand<i32>     { let EncoderMethod = "getImmOpValue"; }
def uimm5  : Operand<i32>     { let EncoderMethod = "getImmOpValue"; }
def uimm12 : Operand<i32>     { let EncoderMethod = "getImmOpValue"; }
def uimm20 : Operand<i32>     { let EncoderMethod = "getImmOpValue"; }
def simm13 : Operand<OtherVT> { let EncoderMethod = "getBranchTargetOpValue"; }
def simm21 : Operand<OtherVT> { let EncoderMethod = "getJumpTargetOpValue"; }

// Pattern predicates — restrict immediate matching without using Operand
// directly in a pattern dag, which would trigger a TableGen assertion.
def imm12 : PatLeaf<(i32 imm), [{ return isInt<12>(N->getSExtValue()); }]>;
def imm5  : PatLeaf<(i32 imm), [{ return isUInt<5>(N->getZExtValue()); }]>;

//===----------------------------------------------------------------------===//
// Section 2 — Opcode Map
//
// One LX32Opcode instance per 7-bit opcode group.  The Name field is used
// only in diagnostics; the Val field is what gets placed in Inst[6:0].
//===----------------------------------------------------------------------===//

def OPC_LOAD   : LX32Opcode<"load",   0b0000011>;
def OPC_STORE  : LX32Opcode<"store",  0b0100011>;
def OPC_BRANCH : LX32Opcode<"branch", 0b1100011>;
def OPC_JAL    : LX32Opcode<"jal",    0b1101111>;
def OPC_JALR   : LX32Opcode<"jalr",   0b1100111>;
def OPC_LUI    : LX32Opcode<"lui",    0b0110111>;
def OPC_AUIPC  : LX32Opcode<"auipc",  0b0010111>;
def OPC_IMM    : LX32Opcode<"imm",    0b0010011>;
def OPC_ARITH  : LX32Opcode<"arith",  0b0110011>;
def OPC_SYSTEM : LX32Opcode<"system", 0b1110011>;

//===----------------------------------------------------------------------===//
// Section 3 — R-Type Instructions (register-register ALU)
//
// All R-type instructions share:
//   hasSideEffects = 0  — pure computation, no observable side effects
//   mayLoad = 0         — no memory reads
//   mayStore = 0        — no memory writes
//
// The class ALU_R_Pat bundles the LXInstR encoding with a pattern list so
// that each instruction can declare its DAG pattern inline.
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
class ALU_R_Pat<bits<7> f7, bits<3> f3, string op, list<dag> pat>
    : LXInstR<f7, f3, OPC_ARITH,
              (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
              op, "$rd, $rs1, $rs2"> {
  let Pattern = pat;
}

def ADD  : ALU_R_Pat<0b0000000, 0b000, "add",  [(set GPR:$rd, (add  GPR:$rs1, GPR:$rs2))]>;
def SUB  : ALU_R_Pat<0b0100000, 0b000, "sub",  [(set GPR:$rd, (sub  GPR:$rs1, GPR:$rs2))]>;
def SLL  : ALU_R_Pat<0b0000000, 0b001, "sll",  [(set GPR:$rd, (shl  GPR:$rs1, GPR:$rs2))]>;
def SLT  : ALU_R_Pat<0b0000000, 0b010, "slt",  [(set GPR:$rd, (setlt  GPR:$rs1, GPR:$rs2))]>;
def SLTU : ALU_R_Pat<0b0000000, 0b011, "sltu", [(set GPR:$rd, (setult GPR:$rs1, GPR:$rs2))]>;
def XOR  : ALU_R_Pat<0b0000000, 0b100, "xor",  [(set GPR:$rd, (xor  GPR:$rs1, GPR:$rs2))]>;
def SRL  : ALU_R_Pat<0b0000000, 0b101, "srl",  [(set GPR:$rd, (srl  GPR:$rs1, GPR:$rs2))]>;
def SRA  : ALU_R_Pat<0b0100000, 0b101, "sra",  [(set GPR:$rd, (sra  GPR:$rs1, GPR:$rs2))]>;
def OR   : ALU_R_Pat<0b0000000, 0b110, "or",   [(set GPR:$rd, (or   GPR:$rs1, GPR:$rs2))]>;
def AND  : ALU_R_Pat<0b0000000, 0b111, "and",  [(set GPR:$rd, (and  GPR:$rs1, GPR:$rs2))]>;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===----------------------------------------------------------------------===//
// Section 4 — I-Type Instructions (immediate ALU)
//
// ALU_I_Pat uses simm12 as the operand type (for the assembler and encoder)
// and imm12 as the PatLeaf in the pattern (for the instruction selector).
// The distinction is necessary: simm12 is an Operand (used at MC level),
// while imm12 is a PatLeaf (used at SelectionDAG level).  Mixing them would
// cause a TableGen assertion.
//
// Shifts (SLLI, SRLI, SRAI) use uimm5 as the operand type because only bits
// [4:0] are meaningful; bits [11:5] of the immediate field carry the funct7
// discriminator (0b0000000 for logical shifts, 0b0100000 for arithmetic).
// Those bits are set statically via "let imm12{11-5} = ..." at definition
// time so the encoder always produces the correct word.
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
class ALU_I_Pat<bits<3> f3, string op, list<dag> pat>
    : LXInstI<f3, OPC_IMM,
              (outs GPR:$rd), (ins GPR:$rs1, simm12:$imm12),
              op, "$rd, $rs1, $imm12"> {
  let Pattern = pat;
}

def ADDI  : ALU_I_Pat<0b000, "addi",  [(set GPR:$rd, (add  GPR:$rs1, imm12:$imm12))]>;
def SLTI  : ALU_I_Pat<0b010, "slti",  [(set GPR:$rd, (setlt  GPR:$rs1, imm12:$imm12))]>;
def SLTIU : ALU_I_Pat<0b011, "sltiu", [(set GPR:$rd, (setult GPR:$rs1, imm12:$imm12))]>;
def XORI  : ALU_I_Pat<0b100, "xori",  [(set GPR:$rd, (xor  GPR:$rs1, imm12:$imm12))]>;
def ORI   : ALU_I_Pat<0b110, "ori",   [(set GPR:$rd, (or   GPR:$rs1, imm12:$imm12))]>;
def ANDI  : ALU_I_Pat<0b111, "andi",  [(set GPR:$rd, (and  GPR:$rs1, imm12:$imm12))]>;

// Shift-immediate instructions.
// uimm5 is used for the shift amount operand (only 5 bits are meaningful).
// imm12{11-5} is set statically to encode the funct7 discriminator:
//   SLLI / SRLI: 0b0000000 (logical shift)
//   SRAI:        0b0100000 (arithmetic right shift)
// imm5 in the pattern (PatLeaf) restricts matching to constants in [0, 31].
def SLLI : LXInstI<0b001, OPC_IMM,
                   (outs GPR:$rd), (ins GPR:$rs1, uimm5:$imm12),
                   "slli", "$rd, $rs1, $imm12"> {
  let Pattern      = [(set GPR:$rd, (shl GPR:$rs1, imm5:$imm12))];
  let imm12{11-5}  = 0b0000000;
}
def SRLI : LXInstI<0b101, OPC_IMM,
                   (outs GPR:$rd), (ins GPR:$rs1, uimm5:$imm12),
                   "srli", "$rd, $rs1, $imm12"> {
  let Pattern      = [(set GPR:$rd, (srl GPR:$rs1, imm5:$imm12))];
  let imm12{11-5}  = 0b0000000;
}
def SRAI : LXInstI<0b101, OPC_IMM,
                   (outs GPR:$rd), (ins GPR:$rs1, uimm5:$imm12),
                   "srai", "$rd, $rs1, $imm12"> {
  let Pattern      = [(set GPR:$rd, (sra GPR:$rs1, imm5:$imm12))];
  let imm12{11-5}  = 0b0100000;
}
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===----------------------------------------------------------------------===//
// Section 5 — Load Instructions
//
// All loads use the I-type encoding: rd = MEM[rs1 + simm12].
// The assembly syntax is "rd, offset(base)" rather than "rd, rs1, imm"
// to match conventional assembler notation.
//
// Pattern fragments (lx32_lh, lx32_lhu, lx32_lb, lx32_lbu) are defined in
// Section 0 and wrap the LLVM sign/zero-extending load nodes so that each
// instruction below can match the correct load variant with a single name.
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
class Load_Pat<bits<3> f3, string op, list<dag> pat>
    : LXInstI<f3, OPC_LOAD,
              (outs GPR:$rd), (ins GPR:$rs1, simm12:$imm12),
              op, "$rd, ${imm12}(${rs1})"> {
  let Pattern = pat;
}

// LW — load 32-bit word (sign-extended to 32 bits, though LX32 is already
//      32-bit so no extension occurs).
def LW  : Load_Pat<0b010, "lw",  [(set GPR:$rd, (load      (add GPR:$rs1, imm12:$imm12)))]>;

// LH — load 16-bit halfword, sign-extended to 32 bits.
def LH  : Load_Pat<0b001, "lh",  [(set GPR:$rd, (lx32_lh   (add GPR:$rs1, imm12:$imm12)))]>;

// LHU — load 16-bit halfword, zero-extended to 32 bits.
def LHU : Load_Pat<0b101, "lhu", [(set GPR:$rd, (lx32_lhu  (add GPR:$rs1, imm12:$imm12)))]>;

// LB — load 8-bit byte, sign-extended to 32 bits.
def LB  : Load_Pat<0b000, "lb",  [(set GPR:$rd, (lx32_lb   (add GPR:$rs1, imm12:$imm12)))]>;

// LBU — load 8-bit byte, zero-extended to 32 bits.
def LBU : Load_Pat<0b100, "lbu", [(set GPR:$rd, (lx32_lbu  (add GPR:$rs1, imm12:$imm12)))]>;
} // hasSideEffects = 0, mayLoad = 1, mayStore = 0

//===----------------------------------------------------------------------===//
// Section 6 — Store Instructions
//
// Stores use the S-type encoding: MEM[rs1 + simm12] = rs2.
// The assembly syntax is "rs2, offset(base)".
//
// Note on truncating stores:
//   SW stores all 32 bits — pattern uses the generic (store ...) node.
//   SH stores the low 16 bits — pattern uses (truncstorei16 ...) directly.
//   SB stores the low 8 bits  — pattern uses (truncstorei8  ...) directly.
//
//   The original implementation used PatFrag helpers (lx32_sh / lx32_sb)
//   to wrap truncstorei16/truncstorei8.  Those fragments are incorrect
//   because truncstorei* are chain nodes and PatFrag does not propagate
//   chain/memory flags — the selector would not recognize the pattern or
//   would generate incorrect code.  Using the LLVM DAG nodes directly in
//   the pattern is the right approach, as shown for SW and used here.
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
class Store_Pat<bits<3> f3, string op, list<dag> pat>
    : LXInstS<f3, OPC_STORE,
              (outs), (ins GPR:$rs2, GPR:$rs1, simm12:$imm12),
              op, "$rs2, ${imm12}(${rs1})"> {
  let Pattern = pat;
}

// SW — store 32-bit word.
def SW : Store_Pat<0b010, "sw",
    [(store GPR:$rs2, (add GPR:$rs1, imm12:$imm12))]>;

// SH — store low 16 bits (truncating store).
def SH : Store_Pat<0b001, "sh",
    [(truncstorei16 GPR:$rs2, (add GPR:$rs1, imm12:$imm12))]>;

// SB — store low 8 bits (truncating store).
def SB : Store_Pat<0b000, "sb",
    [(truncstorei8 GPR:$rs2, (add GPR:$rs1, imm12:$imm12))]>;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 1

// Additional addressing patterns for pointer-dereference IR where the address
// node is already a plain base register and no explicit add-immediate appears.
def : Pat<(load GPR:$rs1), (LW GPR:$rs1, 0)>;
def : Pat<(store GPR:$rs2, GPR:$rs1), (SW GPR:$rs2, GPR:$rs1, 0)>;

//===----------------------------------------------------------------------===//
// Section 7 — Branch Instructions (B-type)
//
// Branches compare two registers and jump to a PC-relative address if the
// condition holds.  They have no result register.
//
// The operand simm13 carries the 13-bit signed offset (bits [12:1] of the
// byte offset; bit 0 is implicit zero).  The EncoderMethod
// "getBranchTargetOpValue" handles both literal offsets and symbolic labels
// (the latter emit a fixup record for the linker).
//
// No DAG patterns are defined here.  Branches are lowered to machine
// instructions by LX32ISelLowering::LowerBR_CC (Day 9).  Defining patterns
// directly for branches is fragile because the DAG conditional branch node
// (ISD::BRCOND) requires a separate condition-code lowering step.
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    isTerminator = 1, isBranch = 1 in {
class Branch_B<bits<3> funct3, string opcodestr>
    : LXInstB<funct3, OPC_BRANCH,
              (outs), (ins GPR:$rs1, GPR:$rs2, simm13:$imm13),
              opcodestr, "$rs1, $rs2, $imm13">;

def BEQ  : Branch_B<0b000, "beq">;
def BNE  : Branch_B<0b001, "bne">;
def BLT  : Branch_B<0b100, "blt">;
def BGE  : Branch_B<0b101, "bge">;
def BLTU : Branch_B<0b110, "bltu">;
def BGEU : Branch_B<0b111, "bgeu">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===----------------------------------------------------------------------===//
// Section 8 — Jump Instructions (JAL / JALR)
//
// JAL (J-type) — unconditional jump with link.
//   Writes PC+4 into rd, then jumps to PC + sign_extend(imm21).
//   rd = X0 is used for unconditional jumps without link ("j" pseudo).
//   rd = X1 (ra) is used for calls ("jal ra, target").
//   isCall is NOT set here because JAL with rd=X0 is not a call.
//   PseudoCALL (Section 11) handles the call case explicitly and carries
//   Defs = [X1] so the register allocator knows ra is clobbered.
//
// JALR (I-type) — indirect jump with link.
//   Writes PC+4 into rd, then jumps to (rs1 + sign_extend(imm12)) & ~1.
//   Used to implement function returns (JALR x0, ra, 0 = "ret") and
//   indirect calls.  Like JAL, isCall and Defs are not set here; callers
//   use PseudoRET or PseudoCALL instead of emitting JALR directly.
//
// TODO (Day 7): if JALR is ever emitted directly as a call (e.g., for
//   indirect calls through a function pointer), add a variant with
//   isCall = 1 and Defs = [X1] so the RA treats ra as clobbered.
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    isTerminator = 1, isBarrier = 1 in {
  def JAL  : LXInstJ<OPC_JAL,
                     (outs GPR:$rd), (ins simm21:$imm21),
                     "jal", "$rd, $imm21">;
  def JALR : LXInstI<0b000, OPC_JALR,
                     (outs GPR:$rd), (ins GPR:$rs1, simm12:$imm12),
                     "jalr", "$rd, ${imm12}(${rs1})">;
}

//===----------------------------------------------------------------------===//
// Section 9 — Upper-Immediate Instructions (LUI / AUIPC)
//
// LUI  — loads a 20-bit immediate into bits [31:12] of rd; bits [11:0] = 0.
//   Used as the first instruction of the two-instruction constant
//   materialization sequence (LUI + ADDI) for 32-bit constants.
//
// AUIPC — adds a 20-bit immediate (shifted left 12) to the current PC and
//   places the result in rd.  Used for PC-relative addressing of global
//   symbols: AUIPC rd, hi20(sym); LW rd, lo12(sym)(rd).
//
// Both use uimm20 as the operand type (20 unsigned bits, range [0, 0xFFFFF]).
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
  def LUI   : LXInstU<OPC_LUI,   (outs GPR:$rd), (ins uimm20:$imm20),
                      "lui",   "$rd, $imm20">;
  def AUIPC : LXInstU<OPC_AUIPC, (outs GPR:$rd), (ins uimm20:$imm20),
                      "auipc", "$rd, $imm20">;
}

//===----------------------------------------------------------------------===//
// Section 10 — Constant-Materialization Patterns
//
// Two patterns cover all 32-bit integer constants:
//
//   Small constants (fits in simm12): ADDI rd, x0, imm
//     The imm12 PatLeaf (Section 1) restricts this pattern to values in
//     the range [-2048, 2047].
//
//   Large constants (does not fit in simm12): LUI rd, hi20; ADDI rd, rd, lo12
//     HI20 and LO12 (Section 0.5) compute the two pieces.  LO12 produces a
//     sign-extended 12-bit value so that ADDI receives a valid simm12.
//
// TODO (Day 10): when LO12 == 0, this pattern emits "ADDI rd, rd, 0" which
//   is a no-op.  The ISelDAGToDAG pass (Day 10) will suppress that case in
//   C++ to avoid the redundant instruction.
//===----------------------------------------------------------------------===//

// Small constant: fits directly in the ADDI immediate field.
def : Pat<(i32 imm12:$imm), (ADDI X0, imm12:$imm)>;

// Large constant: upper 20 bits via LUI, lower 12 bits via ADDI.
def : Pat<(i32 imm:$imm), (ADDI (LUI (HI20 imm:$imm)), (LO12 imm:$imm))>;

//===----------------------------------------------------------------------===//
// Section 11 — Pseudo-Instructions
//
// Pseudo-instructions are emitted by the backend but have no direct hardware
// encoding.  They are expanded to real instructions either in the AsmPrinter
// (for assembler-level pseudos like "ret") or in LX32InstrInfo::
// expandPostRAPseudo (for codegen pseudos like PseudoRET, Day 7).
//
// LX32Pseudo base class: sets isPseudo = 1 and accepts a pattern list so
//   that individual pseudos can declare DAG patterns where appropriate.
//
// PseudoRET — models a function return.  Expands to "JALR x0, ra, 0".
//   Pattern [(LX32ret)] causes the instruction selector to emit PseudoRET
//   whenever it encounters the LX32ISD::RET node from ISelLowering (Day 9).
//   Uses = [X1] informs the RA that the return address register is consumed.
//
// PseudoCALL — models a direct call to a register-held address.
//   No pattern: the call sequence is built explicitly in LX32ISelLowering::
//   LowerCall (Day 9), which emits this pseudo with the callee address.
//   isCall = 1 triggers call-site handling in the frame lowering.
//   Defs = [X1] tells the RA that ra (X1) is clobbered by the call.
//   hasSideEffects = 1 prevents the optimizer from eliminating calls.
//
// PseudoLA — load address of a symbol into a register.
//   Expands to "AUIPC rd, hi20(addr); ADDI rd, rd, lo12(addr)" in the
//   AsmPrinter.  No DAG pattern: materialization of global addresses is
//   handled in LX32ISelLowering::lowerGlobalAddress (Day 9).
//
// PseudoNOP — explicit no-operation.  Expands to "ADDI x0, x0, 0".
//   Used by the assembler and occasionally by peephole passes.
//===----------------------------------------------------------------------===//

class LX32Pseudo<dag outs, dag ins, string opcodestr, list<dag> pattern>
    : LXInstCommon<outs, ins, opcodestr, ""> {
  let isPseudo = 1;
  let Pattern  = pattern;
}

let isPseudo = 1, hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {

  def PseudoRET : LX32Pseudo<(outs), (ins), "ret", [(LX32ret)]> {
    let isReturn    = 1;
    let isTerminator = 1;
    let isBarrier   = 1;
    // Keep return-value registers live across RA so return-copy chains are not
    // deleted as dead defs before pseudo expansion.
    let Uses        = [X1, X10, X11];
  }

  def PseudoBR : LX32Pseudo<(outs), (ins simm21:$target), "br\t$target", []> {
    let isTerminator = 1;
    let isBranch     = 1;
    let isBarrier    = 1;
  }

  def PseudoCALL : LX32Pseudo<(outs), (ins bare_symbol:$target), "call\t$target", []> {
    let isCall         = 1;
    // Model ABI caller-saved clobbers so RA spills/reloads across calls.
    let Defs           = [X1, X5, X6, X7,
                          X10, X11, X12, X13, X14, X15, X16, X17,
                          X28, X29, X30, X31];
    let hasSideEffects = 1;
  }

  def PseudoLA  : LX32Pseudo<(outs GPR:$rd), (ins bare_symbol:$addr),
                              "la\t$rd, $addr", []>;

  def PseudoNOP : LX32Pseudo<(outs), (ins), "nop", []>;

} // isPseudo = 1

//===----------------------------------------------------------------------===//
// Section 12 — Codegen-Only Pseudo Overrides
//
// These pseudos exist only during code generation and are never seen by the
// assembler.  isCodeGenOnly = 1 prevents them from appearing in the
// instruction enum exposed to the MC layer.
//
// LX32_LOAD_STACK_GUARD — loads the stack-canary value used by
//   -fstack-protector.  Expanded in the backend to a load from the
//   platform-defined canary address.
//
// LX32_PREALLOCATED_ARG — marker for a pre-allocated outgoing argument slot.
//   Used by the preallocated-call-frame lowering pass.
//
// ADJCALLSTACKDOWN / ADJCALLSTACKUP — abstract stack-pointer adjustments
//   that bracket call sequences.  Expanded by LX32FrameLowering::
//   eliminateCallFramePseudoInstr (Day 8) into real ADDI sp, sp, ±N
//   instructions or eliminated entirely when the frame has reserved call
//   frame space.
//===----------------------------------------------------------------------===//

let isPseudo = 1, isCodeGenOnly = 1, Namespace = "LX32" in {

  class PseudoCondBr<string Name>
      : LX32Pseudo<(outs), (ins GPR:$rs1, GPR:$rs2, simm13:$target), Name, []> {
    let hasSideEffects = 0;
    let mayLoad        = 0;
    let mayStore       = 0;
    let isTerminator = 1;
    let isBranch     = 1;
  }

  def PseudoBEQ  : PseudoCondBr<"pbeq\t$rs1, $rs2, $target">;
  def PseudoBNE  : PseudoCondBr<"pbne\t$rs1, $rs2, $target">;
  def PseudoBLT  : PseudoCondBr<"pblt\t$rs1, $rs2, $target">;
  def PseudoBGE  : PseudoCondBr<"pbge\t$rs1, $rs2, $target">;
  def PseudoBLTU : PseudoCondBr<"pbltu\t$rs1, $rs2, $target">;
  def PseudoBGEU : PseudoCondBr<"pbgeu\t$rs1, $rs2, $target">;

  def LX32_LOAD_STACK_GUARD : LX32Pseudo<(outs GPR:$dst), (ins),
                                          "load_stack_guard", []> {
    let mayLoad        = 1;
    let hasSideEffects = 0;
    let mayStore       = 0;
  }

  def LX32_PREALLOCATED_ARG : LX32Pseudo<(outs), (ins i32imm:$p),
                                          "preallocated_arg", []> {
    let hasSideEffects = 1;
    let mayLoad        = 0;
    let mayStore       = 0;
  }

  // ADJCALLSTACKDOWN amt1, amt2
  //   Emitted before a call.  amt1 is the amount to decrement sp;
  //   amt2 is the pre-allocated frame size (non-zero when hasReservedCallFrame
  //   returns false).
  def ADJCALLSTACKDOWN : LX32Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                                    "adjcallstackdown", []> {
    let hasSideEffects = 1;
    let mayLoad        = 0;
    let mayStore       = 0;
  }

  // ADJCALLSTACKUP amt1, amt2
  //   Emitted after a call.  Mirrors ADJCALLSTACKDOWN.
  def ADJCALLSTACKUP : LX32Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                                  "adjcallstackup", []> {
    let hasSideEffects = 1;
    let mayLoad        = 0;
    let mayStore       = 0;
  }

} // isPseudo = 1, isCodeGenOnly = 1
