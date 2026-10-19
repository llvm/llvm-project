//===-- AGXInstructions.td - AGX code generation pseudos ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pseudo instructions used by instruction selection, and their selection
// patterns. Each pseudo has the operand list of the real instruction it maps
// to (see AGXInstrFormats.td), with typed register classes:
//
//   * every ALU source is a (SrcMods, Src*Op) pair,
//   * the destination cache hint of real ALU instructions is omitted.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand helpers
//===----------------------------------------------------------------------===//

// Device memory index: a 32-bit register or a signed 16-bit immediate.
def IndexOp : AGXRegOperand<GPR32, "OPERAND_SRC">;
// Threadgroup memory base: a 16-bit register, a 16-bit uniform or zero.
def LocalBaseOp : AGXRegOperand<SRC16, "OPERAND_SRC">;
// Threadgroup memory index: a 16-bit register or a 16-bit immediate.
def LocalIndexOp : AGXRegOperand<GPR16, "OPERAND_SRC">;

// Encodings of the memory format field.
defvar FMT_I8 = 0;
defvar FMT_I16 = 1;
defvar FMT_I32 = 2;

// AGXSrcMods flags.
defvar MOD_ABS = 1;
defvar MOD_NEG = 2;
defvar MOD_SX = 4;

// 8-bit floating point immediates.
defvar FPIMM_NEG_ZERO = 0x80;

class AGXALUPseudo<Instruction real, dag outs, dag ins, list<dag> pattern = [],
                   list<SchedReadWrite> sched = [WriteALU]>
    : AGXPseudo<outs, ins, pattern> {
  let G13Real = real;
  let IsALU = 1;
  let SchedRW = sched;
}

//===----------------------------------------------------------------------===//
// Moves
//===----------------------------------------------------------------------===//

let isAsCheapAsAMove = 1, isReMaterializable = 1 in {
def MOV_IMM16 : AGXALUPseudo<MOV_IMM16_g13, (outs GPR16:$dst),
                             (ins i32imm:$imm), [], [WriteMovImm]>;
def MOV_IMM32 : AGXALUPseudo<MOV_IMM32_g13, (outs GPR32:$dst),
                             (ins i32imm:$imm), [], [WriteMovImm]>;
}

def imm_to_i32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), SDLoc(N), MVT::i32);
}]>;

def : Pat<(i16 imm:$imm), (MOV_IMM16 (imm_to_i32 $imm))>;
def : Pat<(i32 imm:$imm), (MOV_IMM32 imm:$imm)>;

def bitcast_fpimm_to_i32 : SDNodeXForm<fpimm, [{
  return CurDAG->getTargetConstant(
      N->getValueAPF().bitcastToAPInt().getZExtValue(), SDLoc(N), MVT::i32);
}]>;

def : Pat<(f32 fpimm:$imm), (MOV_IMM32 (bitcast_fpimm_to_i32 $imm))>;
def : Pat<(f16 fpimm:$imm), (MOV_IMM16 (bitcast_fpimm_to_i32 $imm))>;
def : Pat<(bf16 fpimm:$imm), (MOV_IMM16 (bitcast_fpimm_to_i32 $imm))>;

def imm_lo32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 0xffffffff, SDLoc(N),
                                   MVT::i32);
}]>;
def imm_hi32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() >> 32, SDLoc(N),
                                   MVT::i32);
}]>;

def : Pat<(i64 imm:$imm),
          (REG_SEQUENCE GPR64, (MOV_IMM32 (imm_lo32 $imm)), sub0,
                               (MOV_IMM32 (imm_hi32 $imm)), sub1)>;

def GET_SR : AGXALUPseudo<GET_SR_g13, (outs GPR32:$dst), (ins i32imm:$sr),
                          [], [WriteSR]>;

class SRPat<Intrinsic intr, int sr> : Pat<(i32 (intr)), (GET_SR sr)>;

def : SRPat<int_agx_threadgroup_position_in_grid_x, 0>;
def : SRPat<int_agx_threadgroup_position_in_grid_y, 1>;
def : SRPat<int_agx_threadgroup_position_in_grid_z, 2>;
def : SRPat<int_agx_threads_per_threadgroup_x, 4>;
def : SRPat<int_agx_threads_per_threadgroup_y, 5>;
def : SRPat<int_agx_threads_per_threadgroup_z, 6>;
def : SRPat<int_agx_core_id, 20>;
def : SRPat<int_agx_thread_position_in_threadgroup_x, 48>;
def : SRPat<int_agx_thread_position_in_threadgroup_y, 49>;
def : SRPat<int_agx_thread_position_in_threadgroup_z, 50>;
def : SRPat<int_agx_thread_index_in_threadgroup, 51>;
def : SRPat<int_agx_thread_index_in_simdgroup, 52>;
def : SRPat<int_agx_simdgroup_index_in_threadgroup, 53>;
def : SRPat<int_agx_active_thread_index_in_quadgroup, 56>;
def : SRPat<int_agx_total_active_threads_in_quadgroup, 57>;
def : SRPat<int_agx_active_thread_index_in_simdgroup, 58>;
def : SRPat<int_agx_total_active_threads_in_simdgroup, 59>;
def : SRPat<int_agx_thread_position_in_grid_x, 80>;
def : SRPat<int_agx_thread_position_in_grid_y, 81>;
def : SRPat<int_agx_thread_position_in_grid_z, 82>;
def : Pat<(i32 (int_agx_read_sr timm:$sr)), (GET_SR timm:$sr)>;

//===----------------------------------------------------------------------===//
// Bitwise operations
//===----------------------------------------------------------------------===//

// Truth tables, see BITOP_g13.
defvar TT_AND = 0b1000;
defvar TT_OR = 0b1110;
defvar TT_XOR = 0b0110;
defvar TT_MOV = 0b1010;
defvar TT_NOT = 0b0101;

multiclass BitopPseudos<RegisterClass rc, RegisterOperand src, ValueType vt,
                        ComplexPattern srcPat> {
  def NAME : AGXALUPseudo<BITOP_g13, (outs rc:$dst),
                          (ins i32imm:$tt, SrcMods:$src0_mods, src:$src0,
                               SrcMods:$src1_mods, src:$src1)>;

  def : Pat<(vt (and srcPat:$a, srcPat:$b)),
            (!cast<Instruction>(NAME) TT_AND, 0, $a, 0, $b)>;
  def : Pat<(vt (or srcPat:$a, srcPat:$b)),
            (!cast<Instruction>(NAME) TT_OR, 0, $a, 0, $b)>;
  def : Pat<(vt (xor srcPat:$a, srcPat:$b)),
            (!cast<Instruction>(NAME) TT_XOR, 0, $a, 0, $b)>;
  def : Pat<(vt (not srcPat:$a)),
            (!cast<Instruction>(NAME) TT_NOT, 0, $a, 0, (vt 0))>;
}

defm BITOP_I16 : BitopPseudos<GPR16, Src16Op, i16, AGXIntSrc16>;
defm BITOP_I32 : BitopPseudos<GPR32, Src32Op, i32, AGXIntSrc32>;

// Zero extension and register moves with a narrower source.
def BITOP_I32_FROM16 : AGXALUPseudo<BITOP_g13, (outs GPR32:$dst),
                                    (ins i32imm:$tt, SrcMods:$src0_mods,
                                         Src16Op:$src0, SrcMods:$src1_mods,
                                         Src16Op:$src1)>;

def : Pat<(i32 (zext i16:$a)), (BITOP_I32_FROM16 TT_MOV, 0, $a, 0, (i16 0))>;
def : Pat<(i32 (anyext i16:$a)), (BITOP_I32_FROM16 TT_MOV, 0, $a, 0, (i16 0))>;

def : Pat<(i64 (and i64:$a, i64:$b)),
          (REG_SEQUENCE GPR64,
            (BITOP_I32 TT_AND, 0, (EXTRACT_SUBREG $a, sub0), 0,
                       (EXTRACT_SUBREG $b, sub0)), sub0,
            (BITOP_I32 TT_AND, 0, (EXTRACT_SUBREG $a, sub1), 0,
                       (EXTRACT_SUBREG $b, sub1)), sub1)>;
def : Pat<(i64 (or i64:$a, i64:$b)),
          (REG_SEQUENCE GPR64,
            (BITOP_I32 TT_OR, 0, (EXTRACT_SUBREG $a, sub0), 0,
                       (EXTRACT_SUBREG $b, sub0)), sub0,
            (BITOP_I32 TT_OR, 0, (EXTRACT_SUBREG $a, sub1), 0,
                       (EXTRACT_SUBREG $b, sub1)), sub1)>;
def : Pat<(i64 (xor i64:$a, i64:$b)),
          (REG_SEQUENCE GPR64,
            (BITOP_I32 TT_XOR, 0, (EXTRACT_SUBREG $a, sub0), 0,
                       (EXTRACT_SUBREG $b, sub0)), sub0,
            (BITOP_I32 TT_XOR, 0, (EXTRACT_SUBREG $a, sub1), 0,
                       (EXTRACT_SUBREG $b, sub1)), sub1)>;

def BITREV_I32 : AGXALUPseudo<BITREV_g13, (outs GPR32:$dst),
                              (ins SrcMods:$src0_mods, Src32Op:$src0), [],
                              [WriteIMul]>;
def POPCOUNT_I32 : AGXALUPseudo<POPCOUNT_g13, (outs GPR32:$dst),
                                (ins SrcMods:$src0_mods, Src32Op:$src0), [],
                                [WriteIMul]>;
// Index of the most significant set bit, or -1 for zero.
def FFS_I32 : AGXALUPseudo<FFS_g13, (outs GPR32:$dst),
                           (ins SrcMods:$src0_mods, Src32Op:$src0), [],
                           [WriteIMul]>;
def INTL_I32 : AGXALUPseudo<INTL_g13, (outs GPR32:$dst),
                            (ins SrcMods:$src0_mods, Src32Op:$src0,
                                 SrcMods:$src1_mods, Src32Op:$src1), [],
                            [WriteIMul]>;

def : Pat<(i32 (bitreverse AGXIntSrc32:$a)), (BITREV_I32 0, $a)>;
def : Pat<(i32 (ctpop AGXIntSrc32:$a)), (POPCOUNT_I32 0, $a)>;
def : Pat<(i32 (int_agx_interleave AGXIntSrc32:$a, AGXIntSrc32:$b)),
          (INTL_I32 0, $a, 0, $b)>;

//===----------------------------------------------------------------------===//
// Integer arithmetic
//===----------------------------------------------------------------------===//

multiclass IAddPseudos<Instruction real, RegisterClass rc,
                       RegisterOperand src, ValueType vt, ComplexPattern pat,
                       SDPatternOperator node,
                       list<SchedReadWrite> sched = [WriteALU]> {
  def NAME : AGXALUPseudo<real, (outs rc:$dst),
                          (ins SrcMods:$src0_mods, src:$src0,
                               SrcMods:$src1_mods, src:$src1, i32imm:$shift),
                          [], sched>;
  def : Pat<(vt (node pat:$a, pat:$b)),
            (!cast<Instruction>(NAME) 0, $a, 0, $b, 0)>;
}

defm IADD_I16 : IAddPseudos<IADD_g13, GPR16, Src16Op, i16,
                            AGXIntSrc16, add>;
defm IADD_I32 : IAddPseudos<IADD_g13, GPR32, Src32Op, i32,
                            AGXIntSrc32, add>;
defm IADD_I64 : IAddPseudos<IADD_g13, GPR64, Src64Op, i64,
                            AGXIntSrc64, add, [WriteALU64]>;
defm ISUB_I16 : IAddPseudos<ISUB_g13, GPR16, Src16Op, i16,
                            AGXIntSrc16, sub>;
defm ISUB_I32 : IAddPseudos<ISUB_g13, GPR32, Src32Op, i32,
                            AGXIntSrc32, sub>;
defm ISUB_I64 : IAddPseudos<ISUB_g13, GPR64, Src64Op, i64,
                            AGXIntSrc64, sub, [WriteALU64]>;

// Saturating add and subtract. Signedness comes from the sign extension
// flags of the sources.
def IADD_SAT_I32 : AGXALUPseudo<IADD_SAT_g13, (outs GPR32:$dst),
                                (ins SrcMods:$src0_mods, Src32Op:$src0,
                                     SrcMods:$src1_mods, Src32Op:$src1,
                                     i32imm:$shift)>;
def ISUB_SAT_I32 : AGXALUPseudo<ISUB_SAT_g13, (outs GPR32:$dst),
                                (ins SrcMods:$src0_mods, Src32Op:$src0,
                                     SrcMods:$src1_mods, Src32Op:$src1,
                                     i32imm:$shift)>;

def : Pat<(i32 (uaddsat AGXIntSrc32:$a, AGXIntSrc32:$b)),
          (IADD_SAT_I32 0, $a, 0, $b, 0)>;
def : Pat<(i32 (usubsat AGXIntSrc32:$a, AGXIntSrc32:$b)),
          (ISUB_SAT_I32 0, $a, 0, $b, 0)>;
def : Pat<(i32 (saddsat i32:$a, i32:$b)),
          (IADD_SAT_I32 MOD_SX, $a, MOD_SX, $b, 0)>;
def : Pat<(i32 (ssubsat i32:$a, i32:$b)),
          (ISUB_SAT_I32 MOD_SX, $a, MOD_SX, $b, 0)>;

// Sign and zero extension through iadd with a zero addend.
def IADD_I32_FROM16 : AGXALUPseudo<IADD_g13, (outs GPR32:$dst),
                                   (ins SrcMods:$src0_mods, Src16Op:$src0,
                                        SrcMods:$src1_mods, Src16Op:$src1,
                                        i32imm:$shift)>;
def IADD_I64_FROM32 : AGXALUPseudo<IADD_g13, (outs GPR64:$dst),
                                   (ins SrcMods:$src0_mods, Src32Op:$src0,
                                        SrcMods:$src1_mods, Src32Op:$src1,
                                        i32imm:$shift), [], [WriteALU64]>;

// 64-bit address arithmetic: base + (extended 32-bit index << shift).
def IADD_I64_I32 : AGXALUPseudo<IADD_g13, (outs GPR64:$dst),
                                (ins SrcMods:$src0_mods, Src64Op:$src0,
                                     SrcMods:$src1_mods, Src32Op:$src1,
                                     i32imm:$shift), [], [WriteALU64]>;

// The iadd shift amount, 0 to 4.
def agx_iadd_shift : ImmLeaf<i32, [{ return Imm >= 0 && Imm <= 4; }]>;

def : Pat<(i64 (shl (zext i32:$x), agx_iadd_shift:$s)),
          (IADD_I64_FROM32 0, (i32 0), 0, $x, imm:$s)>;
def : Pat<(i64 (shl (sext i32:$x), agx_iadd_shift:$s)),
          (IADD_I64_FROM32 0, (i32 0), MOD_SX, $x, imm:$s)>;
def : Pat<(i64 (add i64:$a, (shl (zext i32:$x), agx_iadd_shift:$s))),
          (IADD_I64_I32 0, $a, 0, $x, imm:$s)>;
def : Pat<(i64 (add i64:$a, (shl (sext i32:$x), agx_iadd_shift:$s))),
          (IADD_I64_I32 0, $a, MOD_SX, $x, imm:$s)>;
def : Pat<(i64 (add i64:$a, (zext i32:$x))), (IADD_I64_I32 0, $a, 0, $x, 0)>;
def : Pat<(i64 (add i64:$a, (sext i32:$x))),
          (IADD_I64_I32 0, $a, MOD_SX, $x, 0)>;
def : Pat<(i64 (add i64:$a, (shl i64:$b, agx_iadd_shift:$s))),
          (IADD_I64 0, $a, 0, $b, imm:$s)>;

def : Pat<(i32 (sext i16:$a)), (IADD_I32_FROM16 MOD_SX, $a, 0, (i16 0), 0)>;
def : Pat<(i64 (sext i32:$a)), (IADD_I64_FROM32 MOD_SX, $a, 0, (i32 0), 0)>;
def : Pat<(i64 (zext i32:$a)), (IADD_I64_FROM32 0, $a, 0, (i32 0), 0)>;
def : Pat<(i64 (anyext i32:$a)), (IADD_I64_FROM32 0, $a, 0, (i32 0), 0)>;
def : Pat<(i64 (zext i16:$a)),
          (IADD_I64_FROM32 0, (BITOP_I32_FROM16 TT_MOV, 0, $a, 0, (i16 0)), 0,
                           (i32 0), 0)>;
def : Pat<(i64 (anyext i16:$a)),
          (IADD_I64_FROM32 0, (BITOP_I32_FROM16 TT_MOV, 0, $a, 0, (i16 0)), 0,
                           (i32 0), 0)>;
def : Pat<(i64 (sext i16:$a)),
          (IADD_I64_FROM32 MOD_SX, (IADD_I32_FROM16 MOD_SX, $a, 0, (i16 0), 0),
                           0, (i32 0), 0)>;
def : Pat<(i32 (sext_inreg i32:$a, i16)),
          (IADD_I32_FROM16 MOD_SX, (EXTRACT_SUBREG $a, lo16), 0, (i16 0), 0)>;
def : Pat<(i64 (sext_inreg i64:$a, i32)),
          (IADD_I64_FROM32 MOD_SX, (EXTRACT_SUBREG $a, sub0), 0, (i32 0), 0)>;

def : Pat<(i16 (trunc i32:$a)), (EXTRACT_SUBREG $a, lo16)>;
def : Pat<(i32 (trunc i64:$a)), (EXTRACT_SUBREG $a, sub0)>;
def : Pat<(i16 (trunc i64:$a)),
          (i16 (EXTRACT_SUBREG (i32 (EXTRACT_SUBREG $a, sub0)), lo16))>;

// imadd: D = A * B + C.
multiclass IMAddPseudos<RegisterClass rc, RegisterOperand src,
                        RegisterOperand addend, ValueType vt, ComplexPattern pat,
                        list<SchedReadWrite> sched = [WriteIMul]> {
  def NAME : AGXALUPseudo<IMADD_g13, (outs rc:$dst),
                          (ins SrcMods:$src0_mods, src:$src0,
                               SrcMods:$src1_mods, src:$src1,
                               SrcMods:$src2_mods, addend:$src2, i32imm:$shift),
                          [], sched>;
  def : Pat<(vt (mul pat:$a, pat:$b)),
            (!cast<Instruction>(NAME) 0, $a, 0, $b, 0, (vt 0), 0)>;
  def : Pat<(vt (add (mul pat:$a, pat:$b), pat:$c)),
            (!cast<Instruction>(NAME) 0, $a, 0, $b, 0, $c, 0)>;
}

defm IMADD_I16 : IMAddPseudos<GPR16, Src16Op, Src16Op, i16, AGXIntSrc16>;
defm IMADD_I32 : IMAddPseudos<GPR32, Src32Op, Src32Op, i32, AGXIntSrc32>;

// 32 x 32 -> 64-bit multiply-add.
def IMADD_I64_FROM32 : AGXALUPseudo<IMADD_g13, (outs GPR64:$dst),
                                    (ins SrcMods:$src0_mods, Src32Op:$src0,
                                         SrcMods:$src1_mods, Src32Op:$src1,
                                         SrcMods:$src2_mods, Src64Op:$src2,
                                         i32imm:$shift), [], [WriteALU64]>;

def : Pat<(i64 (mul (zext i32:$a), (zext i32:$b))),
          (IMADD_I64_FROM32 0, $a, 0, $b, 0, (i64 0), 0)>;
def : Pat<(i64 (mul (sext i32:$a), (sext i32:$b))),
          (IMADD_I64_FROM32 MOD_SX, $a, MOD_SX, $b, 0, (i64 0), 0)>;
def : Pat<(i32 (mulhu i32:$a, i32:$b)),
          (EXTRACT_SUBREG (IMADD_I64_FROM32 0, $a, 0, $b, 0, (i64 0), 0), sub1)>;
def : Pat<(i32 (mulhs i32:$a, i32:$b)),
          (EXTRACT_SUBREG (IMADD_I64_FROM32 MOD_SX, $a, MOD_SX, $b, 0,
                                            (i64 0), 0), sub1)>;

//===----------------------------------------------------------------------===//
// Shifts and bit fields
//===----------------------------------------------------------------------===//

multiclass BitfieldPseudo<Instruction real, RegisterClass rc,
                          RegisterOperand src> {
  def NAME : AGXALUPseudo<real, (outs rc:$dst),
                          (ins SrcMods:$src0_mods, src:$src0,
                               SrcMods:$src1_mods, src:$src1,
                               SrcMods:$src2_mods, src:$src2, i32imm:$mask),
                          [], [WriteIMul]>;
}

defm BFI_I16 : BitfieldPseudo<BFI_g13, GPR16, Src16Op>;
defm BFI_I32 : BitfieldPseudo<BFI_g13, GPR32, Src32Op>;
defm BFEIL_I16 : BitfieldPseudo<BFEIL_g13, GPR16, Src16Op>;
defm BFEIL_I32 : BitfieldPseudo<BFEIL_g13, GPR32, Src32Op>;
defm EXTR_I32 : BitfieldPseudo<EXTR_g13, GPR32, Src32Op>;

def ASR_I16 : AGXALUPseudo<ASR_g13, (outs GPR16:$dst),
                           (ins SrcMods:$src0_mods, Src16Op:$src0,
                                SrcMods:$src1_mods, Src16Op:$src1), [],
                           [WriteIMul]>;
def ASR_I32 : AGXALUPseudo<ASR_g13, (outs GPR32:$dst),
                           (ins SrcMods:$src0_mods, Src32Op:$src0,
                                SrcMods:$src1_mods, Src32Op:$src1), [],
                           [WriteIMul]>;

// shl x, s = bfi 0, x, s (full mask); srl x, s = bfeil 0, x, s.
def : Pat<(i32 (shl AGXIntSrc32:$a, AGXIntSrc32:$s)),
          (BFI_I32 0, (i32 0), 0, $a, 0, $s, 0)>;
def : Pat<(i32 (srl AGXIntSrc32:$a, AGXIntSrc32:$s)),
          (BFEIL_I32 0, (i32 0), 0, $a, 0, $s, 0)>;
def : Pat<(i32 (sra AGXIntSrc32:$a, AGXIntSrc32:$s)),
          (ASR_I32 0, $a, 0, $s)>;
def : Pat<(i16 (shl AGXIntSrc16:$a, AGXIntSrc16:$s)),
          (BFI_I16 0, (i16 0), 0, $a, 0, $s, 0)>;
def : Pat<(i16 (srl AGXIntSrc16:$a, AGXIntSrc16:$s)),
          (BFEIL_I16 0, (i16 0), 0, $a, 0, $s, 0)>;
def : Pat<(i16 (sra AGXIntSrc16:$a, AGXIntSrc16:$s)),
          (ASR_I16 0, $a, 0, $s)>;

// Rotates: extr reads the register pair (B:A).
def : Pat<(i32 (rotr i32:$a, AGXIntSrc32:$s)),
          (EXTR_I32 0, $a, 0, $a, 0, $s, 0)>;
def : Pat<(i32 (fshr i32:$hi, i32:$lo, AGXIntSrc32:$s)),
          (EXTR_I32 0, $lo, 0, $hi, 0, $s, 0)>;

def : Pat<(i32 (sext_inreg i32:$a, i8)),
          (ASR_I32 0, (BFI_I32 0, (i32 0), 0, $a, 0, (i32 24), 0), 0, (i32 24))>;
def : Pat<(i16 (sext_inreg i16:$a, i8)),
          (ASR_I16 0, (BFI_I16 0, (i16 0), 0, $a, 0, (i16 8), 0), 0, (i16 8))>;
def : Pat<(i32 (sext_inreg i32:$a, i1)),
          (ISUB_I32 0, (i32 0), 0, (BITOP_I32 TT_AND, 0, $a, 0, (i32 1)), 0)>;
def : Pat<(i16 (sext_inreg i16:$a, i1)),
          (ISUB_I16 0, (i16 0), 0, (BITOP_I16 TT_AND, 0, $a, 0, (i16 1)), 0)>;

// Count leading zeros: 31 - ffs(x); ffs(0) is -1.
def : Pat<(i32 (ctlz AGXIntSrc32:$a)),
          (ISUB_I32 0, (i32 31), 0, (FFS_I32 0, $a), 0)>;
def : Pat<(i32 (cttz AGXIntSrc32:$a)),
          (ISUB_I32 0, (i32 31), 0, (FFS_I32 0, (BITREV_I32 0, $a)), 0)>;
def : Pat<(i64 (ctpop i64:$a)),
          (IADD_I64_FROM32 0, (IADD_I32 0, (POPCOUNT_I32 0, (EXTRACT_SUBREG $a, sub0)),
                                        0, (POPCOUNT_I32 0, (EXTRACT_SUBREG $a, sub1)),
                                        0), 0, (i32 0), 0)>;

//===----------------------------------------------------------------------===//
// Compare and select
//===----------------------------------------------------------------------===//

// Integer conditions: ueq 0, ult 1, ugt 2, seq 4, slt 5, sgt 6.
// Float conditions: eq 0, lt 1, gt 2, ltn 3, gte 5, lte 6, gtn 7.
defvar IC_UEQ = 0;
defvar IC_ULT = 1;
defvar IC_UGT = 2;
defvar IC_SLT = 5;
defvar IC_SGT = 6;
defvar FC_EQ = 0;
defvar FC_LT = 1;
defvar FC_GT = 2;
defvar FC_LTN = 3;
defvar FC_GTE = 5;
defvar FC_LTE = 6;
defvar FC_GTN = 7;

multiclass ICmpselPseudo<RegisterClass rc, RegisterOperand cmp,
                         RegisterOperand sel> {
  def NAME : AGXALUPseudo<ICMPSEL_g13, (outs rc:$dst),
                          (ins i32imm:$cc, SrcMods:$src0_mods, cmp:$src0,
                               SrcMods:$src1_mods, cmp:$src1,
                               SrcMods:$src2_mods, sel:$src2,
                               SrcMods:$src3_mods, sel:$src3), [], [WriteFP]>;
}

multiclass FCmpselPseudo<RegisterClass rc, RegisterOperand cmp,
                         RegisterOperand sel> {
  def NAME : AGXALUPseudo<FCMPSEL_g13, (outs rc:$dst),
                          (ins i32imm:$cc, SrcMods:$src0_mods, cmp:$src0,
                               SrcMods:$src1_mods, cmp:$src1,
                               SrcMods:$src2_mods, sel:$src2,
                               SrcMods:$src3_mods, sel:$src3), [], [WriteFP]>;
}

// Naming: <compare type>_<result type>.
defm ICMPSEL_I32 : ICmpselPseudo<GPR32, Src32Op, Src32Op>;
defm ICMPSEL_I16 : ICmpselPseudo<GPR16, Src16Op, Src16Op>;
defm ICMPSEL_I16_I32 : ICmpselPseudo<GPR32, Src16Op, Src32Op>;
defm ICMPSEL_I32_I16 : ICmpselPseudo<GPR16, Src32Op, Src16Op>;
defm FCMPSEL_F32 : FCmpselPseudo<GPR32, Src32Op, Src32Op>;
defm FCMPSEL_F16 : FCmpselPseudo<GPR16, Src16Op, Src16Op>;
defm FCMPSEL_F16_I32 : FCmpselPseudo<GPR32, Src16Op, Src32Op>;
defm FCMPSEL_F32_I16 : FCmpselPseudo<GPR16, Src32Op, Src16Op>;

// setcc: (cond, hardware condition, swap true/false).
class AGXCond<CondCode cond, int hw, bit swap> {
  CondCode CC = cond;
  int HW = hw;
  bit Swap = swap;
}

defvar AGXIntConds = [AGXCond<SETEQ, IC_UEQ, 0>, AGXCond<SETNE, IC_UEQ, 1>,
                      AGXCond<SETULT, IC_ULT, 0>, AGXCond<SETUGE, IC_ULT, 1>,
                      AGXCond<SETUGT, IC_UGT, 0>, AGXCond<SETULE, IC_UGT, 1>,
                      AGXCond<SETLT, IC_SLT, 0>, AGXCond<SETGE, IC_SLT, 1>,
                      AGXCond<SETGT, IC_SGT, 0>, AGXCond<SETLE, IC_SGT, 1>];

// Ordered conditions select their predicate; unordered conditions select the
// inverse ordered predicate with swapped results.
defvar AGXFPConds = [AGXCond<SETOEQ, FC_EQ, 0>, AGXCond<SETUNE, FC_EQ, 1>,
                     AGXCond<SETOLT, FC_LT, 0>, AGXCond<SETUGE, FC_LT, 1>,
                     AGXCond<SETOGT, FC_GT, 0>, AGXCond<SETULE, FC_GT, 1>,
                     AGXCond<SETOGE, FC_GTE, 0>, AGXCond<SETULT, FC_GTE, 1>,
                     AGXCond<SETOLE, FC_LTE, 0>, AGXCond<SETUGT, FC_LTE, 1>,
                     AGXCond<SETEQ, FC_EQ, 0>, AGXCond<SETNE, FC_EQ, 1>,
                     AGXCond<SETLT, FC_LT, 0>, AGXCond<SETGE, FC_GTE, 0>,
                     AGXCond<SETGT, FC_GT, 0>, AGXCond<SETLE, FC_LTE, 0>];

multiclass SetCCPats<Instruction inst, ValueType cvt, ValueType rvt,
                     ComplexPattern cpat, ComplexPattern spat,
                     list<AGXCond> conds, bit fp> {
  foreach C = conds in {
    if fp then {
      def : Pat<(rvt (setcc (cpat cvt:$a, i32:$am), (cpat cvt:$b, i32:$bm),
                            C.CC)),
                !if(C.Swap, (inst C.HW, $am, $a, $bm, $b, 0, (rvt 0), 0, (rvt 1)),
                            (inst C.HW, $am, $a, $bm, $b, 0, (rvt 1), 0, (rvt 0)))>;
      def : Pat<(rvt (select (i32 (setcc (cpat cvt:$a, i32:$am),
                                         (cpat cvt:$b, i32:$bm), C.CC)),
                             spat:$x, spat:$y)),
                !if(C.Swap, (inst C.HW, $am, $a, $bm, $b, 0, $y, 0, $x),
                            (inst C.HW, $am, $a, $bm, $b, 0, $x, 0, $y))>;
    } else {
      def : Pat<(rvt (setcc cpat:$a, cpat:$b, C.CC)),
                !if(C.Swap, (inst C.HW, 0, $a, 0, $b, 0, (rvt 0), 0, (rvt 1)),
                            (inst C.HW, 0, $a, 0, $b, 0, (rvt 1), 0, (rvt 0)))>;
      def : Pat<(rvt (select (i32 (setcc cpat:$a, cpat:$b, C.CC)),
                             spat:$x, spat:$y)),
                !if(C.Swap, (inst C.HW, 0, $a, 0, $b, 0, $y, 0, $x),
                            (inst C.HW, 0, $a, 0, $b, 0, $x, 0, $y))>;
    }
  }
}

defm : SetCCPats<ICMPSEL_I32, i32, i32, AGXIntSrc32, AGXIntSrc32,
                 AGXIntConds, 0>;
defm : SetCCPats<ICMPSEL_I16_I32, i16, i32, AGXIntSrc16, AGXIntSrc32,
                 AGXIntConds, 0>;
defm : SetCCPats<FCMPSEL_F32, f32, i32, AGXFPSrc32, AGXIntSrc32,
                 AGXFPConds, 1>;
defm : SetCCPats<FCMPSEL_F16_I32, f16, i32, AGXFPSrc16, AGXIntSrc32,
                 AGXFPConds, 1>;

// Selects of other value types on a comparison.
multiclass SelectCCPats<Instruction inst, ValueType cvt, ValueType rvt,
                        ComplexPattern cpat, list<AGXCond> conds, bit fp> {
  foreach C = conds in {
    if fp then
      def : Pat<(rvt (select (i32 (setcc (cpat cvt:$a, i32:$am),
                                         (cpat cvt:$b, i32:$bm), C.CC)),
                             rvt:$x, rvt:$y)),
                !if(C.Swap, (inst C.HW, $am, $a, $bm, $b, 0, $y, 0, $x),
                            (inst C.HW, $am, $a, $bm, $b, 0, $x, 0, $y))>;
    else
      def : Pat<(rvt (select (i32 (setcc cpat:$a, cpat:$b, C.CC)),
                             rvt:$x, rvt:$y)),
                !if(C.Swap, (inst C.HW, 0, $a, 0, $b, 0, $y, 0, $x),
                            (inst C.HW, 0, $a, 0, $b, 0, $x, 0, $y))>;
  }
}

defm : SelectCCPats<ICMPSEL_I32, i32, f32, AGXIntSrc32, AGXIntConds, 0>;
defm : SelectCCPats<ICMPSEL_I16_I32, i16, f32, AGXIntSrc16, AGXIntConds, 0>;
defm : SelectCCPats<FCMPSEL_F32, f32, f32, AGXFPSrc32, AGXFPConds, 1>;
defm : SelectCCPats<FCMPSEL_F16_I32, f16, f32, AGXFPSrc16, AGXFPConds, 1>;
foreach VT = [i16, f16, bf16] in {
defm : SelectCCPats<ICMPSEL_I32_I16, i32, VT, AGXIntSrc32, AGXIntConds, 0>;
defm : SelectCCPats<ICMPSEL_I16, i16, VT, AGXIntSrc16, AGXIntConds, 0>;
defm : SelectCCPats<FCMPSEL_F32_I16, f32, VT, AGXFPSrc32, AGXFPConds, 1>;
defm : SelectCCPats<FCMPSEL_F16, f16, VT, AGXFPSrc16, AGXFPConds, 1>;
}

// Select on a boolean (0/1 in a 32-bit register).
def : Pat<(i32 (select i32:$c, AGXIntSrc32:$x, AGXIntSrc32:$y)),
          (ICMPSEL_I32 IC_UEQ, 0, $c, 0, (i32 0), 0, $y, 0, $x)>;
def : Pat<(i16 (select i32:$c, AGXIntSrc16:$x, AGXIntSrc16:$y)),
          (ICMPSEL_I32_I16 IC_UEQ, 0, $c, 0, (i32 0), 0, $y, 0, $x)>;
def : Pat<(f32 (select i32:$c, f32:$x, f32:$y)),
          (ICMPSEL_I32 IC_UEQ, 0, $c, 0, (i32 0), 0, $y, 0, $x)>;
foreach VT = [f16, bf16] in
def : Pat<(VT (select i32:$c, VT:$x, VT:$y)),
          (ICMPSEL_I32_I16 IC_UEQ, 0, $c, 0, (i32 0), 0, $y, 0, $x)>;

def : Pat<(i64 (select i32:$c, i64:$x, i64:$y)),
          (REG_SEQUENCE GPR64,
            (ICMPSEL_I32 IC_UEQ, 0, $c, 0, (i32 0), 0, (EXTRACT_SUBREG $y, sub0),
                         0, (EXTRACT_SUBREG $x, sub0)), sub0,
            (ICMPSEL_I32 IC_UEQ, 0, $c, 0, (i32 0), 0, (EXTRACT_SUBREG $y, sub1),
                         0, (EXTRACT_SUBREG $x, sub1)), sub1)>;

// Integer min/max.
def : Pat<(i32 (smin i32:$a, i32:$b)),
          (ICMPSEL_I32 IC_SLT, 0, $a, 0, $b, 0, $a, 0, $b)>;
def : Pat<(i32 (smax i32:$a, i32:$b)),
          (ICMPSEL_I32 IC_SGT, 0, $a, 0, $b, 0, $a, 0, $b)>;
def : Pat<(i32 (umin i32:$a, i32:$b)),
          (ICMPSEL_I32 IC_ULT, 0, $a, 0, $b, 0, $a, 0, $b)>;
def : Pat<(i32 (umax i32:$a, i32:$b)),
          (ICMPSEL_I32 IC_UGT, 0, $a, 0, $b, 0, $a, 0, $b)>;
def : Pat<(i16 (smin i16:$a, i16:$b)),
          (ICMPSEL_I16 IC_SLT, 0, $a, 0, $b, 0, $a, 0, $b)>;
def : Pat<(i16 (smax i16:$a, i16:$b)),
          (ICMPSEL_I16 IC_SGT, 0, $a, 0, $b, 0, $a, 0, $b)>;
def : Pat<(i16 (umin i16:$a, i16:$b)),
          (ICMPSEL_I16 IC_ULT, 0, $a, 0, $b, 0, $a, 0, $b)>;
def : Pat<(i16 (umax i16:$a, i16:$b)),
          (ICMPSEL_I16 IC_UGT, 0, $a, 0, $b, 0, $a, 0, $b)>;

// Float min/max: the "n" conditions return the other operand for a NaN.
def : Pat<(f32 (fminnum f32:$a, f32:$b)),
          (FCMPSEL_F32 FC_LTN, 0, $a, 0, $b, 0, $a, 0, $b)>;
def : Pat<(f32 (fmaxnum f32:$a, f32:$b)),
          (FCMPSEL_F32 FC_GTN, 0, $a, 0, $b, 0, $a, 0, $b)>;
def : Pat<(f16 (fminnum f16:$a, f16:$b)),
          (FCMPSEL_F16 FC_LTN, 0, $a, 0, $b, 0, $a, 0, $b)>;
def : Pat<(f16 (fmaxnum f16:$a, f16:$b)),
          (FCMPSEL_F16 FC_GTN, 0, $a, 0, $b, 0, $a, 0, $b)>;

//===----------------------------------------------------------------------===//
// Floating point arithmetic
//===----------------------------------------------------------------------===//

multiclass FBinPseudo<Instruction real, RegisterClass rc, RegisterOperand src,
                      list<SchedReadWrite> sched = [WriteFP]> {
  def NAME : AGXALUPseudo<real, (outs rc:$dst),
                          (ins SrcMods:$src0_mods, src:$src0,
                               SrcMods:$src1_mods, src:$src1), [], sched>;
}

defm FADD32_F32 : FBinPseudo<FADD32_g13, GPR32, Src32Op>;
defm FMUL32_F32 : FBinPseudo<FMUL32_g13, GPR32, Src32Op>;
defm FADD16_F16 : FBinPseudo<FADD16_g13, GPR16, Src16Op>;
defm FMUL16_F16 : FBinPseudo<FMUL16_g13, GPR16, Src16Op>;
defm FADD32_SAT_F32 : FBinPseudo<FADD32_SAT_g13, GPR32, Src32Op>;
defm FADD16_SAT_F16 : FBinPseudo<FADD16_SAT_g13, GPR16, Src16Op>;
// Conversions between half and single precision through fadd32 -0.0.
def FADD32_F32_F16 : AGXALUPseudo<FADD32_g13, (outs GPR32:$dst),
                                  (ins SrcMods:$src0_mods, Src16Op:$src0,
                                       SrcMods:$src1_mods, Src16Op:$src1), [],
                                  [WriteFP]>;
def FADD32_F16_F32 : AGXALUPseudo<FADD32_g13, (outs GPR16:$dst),
                                  (ins SrcMods:$src0_mods, Src32Op:$src0,
                                       SrcMods:$src1_mods, Src32Op:$src1), [],
                                  [WriteFP]>;

def : Pat<(f32 (fadd (AGXFPSrc32 f32:$a, i32:$am), (AGXFPSrc32 f32:$b, i32:$bm))),
          (FADD32_F32 $am, $a, $bm, $b)>;
def : Pat<(f32 (fmul (AGXFPSrc32 f32:$a, i32:$am), (AGXFPSrc32 f32:$b, i32:$bm))),
          (FMUL32_F32 $am, $a, $bm, $b)>;
def : Pat<(f16 (fadd (AGXFPSrc16 f16:$a, i32:$am), (AGXFPSrc16 f16:$b, i32:$bm))),
          (FADD16_F16 $am, $a, $bm, $b)>;
def : Pat<(f16 (fmul (AGXFPSrc16 f16:$a, i32:$am), (AGXFPSrc16 f16:$b, i32:$bm))),
          (FMUL16_F16 $am, $a, $bm, $b)>;

// A negation or absolute value of its own is a bit operation: an addition
// with a source modifier would flush a denormal operand. (Those that feed a
// floating point operation are its source modifiers.)
def : Pat<(f32 (fneg f32:$a)),
          (BITOP_I32 TT_XOR, 0, $a, 0, (MOV_IMM32 0x80000000))>;
def : Pat<(f32 (fabs f32:$a)),
          (BITOP_I32 TT_AND, 0, $a, 0, (MOV_IMM32 0x7fffffff))>;
def : Pat<(f16 (fneg f16:$a)),
          (FADD16_F16 MOD_NEG, $a, 0, (i16 FPIMM_NEG_ZERO))>;
def : Pat<(f16 (fabs f16:$a)),
          (FADD16_F16 MOD_ABS, $a, 0, (i16 FPIMM_NEG_ZERO))>;
def : Pat<(f32 (fcanonicalize (AGXFPSrc32 f32:$a, i32:$am))),
          (FADD32_F32 $am, $a, 0, (i32 FPIMM_NEG_ZERO))>;

// Clamp to [0, 1].
def agx_fpimm0 : FPImmLeaf<fAny, [{ return Imm.isExactlyValue(+0.0); }]>;
def agx_fpimm1 : FPImmLeaf<fAny, [{ return Imm.isExactlyValue(+1.0); }]>;

def : Pat<(f32 (fminnum (fmaxnum (AGXFPSrc32 f32:$a, i32:$am), (f32 agx_fpimm0)),
                        (f32 agx_fpimm1))),
          (FADD32_SAT_F32 $am, $a, 0, (i32 FPIMM_NEG_ZERO))>;

def : Pat<(f32 (fpextend (AGXFPSrc16 f16:$a, i32:$am))),
          (FADD32_F32_F16 $am, $a, 0, (i16 FPIMM_NEG_ZERO))>;
def : Pat<(f16 (fpround (AGXFPSrc32 f32:$a, i32:$am))),
          (FADD32_F16_F32 $am, $a, 0, (i32 FPIMM_NEG_ZERO))>;

multiclass FMAPseudo<Instruction real, RegisterClass rc, RegisterOperand src> {
  def NAME : AGXALUPseudo<real, (outs rc:$dst),
                          (ins SrcMods:$src0_mods, src:$src0,
                               SrcMods:$src1_mods, src:$src1,
                               SrcMods:$src2_mods, src:$src2), [], [WriteFP]>;
}

defm FMADD32_F32 : FMAPseudo<FMADD32_g13, GPR32, Src32Op>;
defm FMADD16_F16 : FMAPseudo<FMADD16_g13, GPR16, Src16Op>;

def : Pat<(f32 (fma (AGXFPSrc32 f32:$a, i32:$am), (AGXFPSrc32 f32:$b, i32:$bm),
                    (AGXFPSrc32 f32:$c, i32:$cm))),
          (FMADD32_F32 $am, $a, $bm, $b, $cm, $c)>;
def : Pat<(f16 (fma (AGXFPSrc16 f16:$a, i32:$am), (AGXFPSrc16 f16:$b, i32:$bm),
                    (AGXFPSrc16 f16:$c, i32:$cm))),
          (FMADD16_F16 $am, $a, $bm, $b, $cm, $c)>;

multiclass FUnaryPseudo<Instruction real, SDPatternOperator node,
                        list<SchedReadWrite> sched = [WriteFRound]> {
  def _F32 : AGXALUPseudo<real, (outs GPR32:$dst),
                          (ins SrcMods:$src0_mods, Src32Op:$src0), [], sched>;
  def _F16 : AGXALUPseudo<real, (outs GPR16:$dst),
                          (ins SrcMods:$src0_mods, Src16Op:$src0), [], sched>;
  def : Pat<(f32 (node (AGXFPSrc32 f32:$a, i32:$am))),
            (!cast<Instruction>(NAME#"_F32") $am, $a)>;
  def : Pat<(f16 (node (AGXFPSrc16 f16:$a, i32:$am))),
            (!cast<Instruction>(NAME#"_F16") $am, $a)>;
}

defm FLOOR : FUnaryPseudo<FLOOR_g13, ffloor>;
defm CEIL : FUnaryPseudo<CEIL_g13, fceil>;
defm TRUNC : FUnaryPseudo<TRUNC_g13, ftrunc>;
defm RINT : FUnaryPseudo<RINT_g13, froundeven>;
defm RCP : FUnaryPseudo<RCP_g13, int_agx_rcp, [WriteSFU]>;
defm RSQRT : FUnaryPseudo<RSQRT_g13, int_agx_rsqrt, [WriteSFUSlow]>;
defm RSQRT_SPECIAL : FUnaryPseudo<RSQRT_SPECIAL_g13, int_agx_rsqrt_special,
                                  [WriteSFUSlow]>;
defm LOG2 : FUnaryPseudo<LOG2_g13, flog2, [WriteSFU]>;
defm EXP2 : FUnaryPseudo<EXP2_g13, fexp2, [WriteSFU]>;
defm SIN_PT_1 : FUnaryPseudo<SIN_PT_1_g13, int_agx_sin_pt_1>;
defm SIN_PT_2 : FUnaryPseudo<SIN_PT_2_g13, int_agx_sin_pt_2, [WriteSFU]>;
defm DFDX : FUnaryPseudo<DFDX_g13, int_agx_dfdx>;
defm DFDY : FUnaryPseudo<DFDY_g13, int_agx_dfdy>;

def : Pat<(f32 (frint (AGXFPSrc32 f32:$a, i32:$am))), (RINT_F32 $am, $a)>;
def : Pat<(f32 (fnearbyint (AGXFPSrc32 f32:$a, i32:$am))), (RINT_F32 $am, $a)>;
def : Pat<(f16 (frint (AGXFPSrc16 f16:$a, i32:$am))), (RINT_F16 $am, $a)>;
def : Pat<(f16 (fnearbyint (AGXFPSrc16 f16:$a, i32:$am))), (RINT_F16 $am, $a)>;

// sqrt(x) = x * rsqrt_special(x).
def : Pat<(f32 (fsqrt f32:$a)),
          (FMUL32_F32 0, $a, 0, (RSQRT_SPECIAL_F32 0, $a))>;
def : Pat<(f16 (fsqrt f16:$a)),
          (FMUL16_F16 0, $a, 0, (RSQRT_SPECIAL_F16 0, $a))>;

def : Pat<(f32 (AGXsin_quadrants f32:$q)),
          (FMUL32_F32 0, (SIN_PT_2_F32 0, (SIN_PT_1_F32 0, $q)), 0,
                      (SIN_PT_1_F32 0, $q))>;

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

// Modes of the convert instruction.
defvar CVT_U8_TO_F = 0;
defvar CVT_S8_TO_F = 1;
defvar CVT_F_TO_U16 = 4;
defvar CVT_F_TO_S16 = 5;
defvar CVT_U16_TO_F = 6;
defvar CVT_S16_TO_F = 7;
defvar CVT_F_TO_U32 = 8;
defvar CVT_F_TO_S32 = 9;
defvar CVT_U32_TO_F = 10;
defvar CVT_S32_TO_F = 11;
defvar ROUND_RTZ = 0;
defvar ROUND_RTE = 1;

multiclass ConvertPseudo<RegisterClass rc, RegisterOperand src> {
  def NAME : AGXALUPseudo<CONVERT_g13, (outs rc:$dst),
                          (ins i32imm:$mode, SrcMods:$src0_mods, src:$src0,
                               i32imm:$round), [], [WriteIMul]>;
}

// Naming: CONVERT_<dst bits>_<src bits>.
defm CONVERT_32_32 : ConvertPseudo<GPR32, Src32Op>;
defm CONVERT_32_16 : ConvertPseudo<GPR32, Src16Op>;
defm CONVERT_16_32 : ConvertPseudo<GPR16, Src32Op>;
defm CONVERT_16_16 : ConvertPseudo<GPR16, Src16Op>;

def : Pat<(i32 (fp_to_sint f32:$a)), (CONVERT_32_32 CVT_F_TO_S32, 0, $a, ROUND_RTZ)>;
def : Pat<(i32 (fp_to_uint f32:$a)), (CONVERT_32_32 CVT_F_TO_U32, 0, $a, ROUND_RTZ)>;
def : Pat<(i32 (fp_to_sint f16:$a)), (CONVERT_32_16 CVT_F_TO_S32, 0, $a, ROUND_RTZ)>;
def : Pat<(i32 (fp_to_uint f16:$a)), (CONVERT_32_16 CVT_F_TO_U32, 0, $a, ROUND_RTZ)>;
def : Pat<(i16 (fp_to_sint f32:$a)), (CONVERT_16_32 CVT_F_TO_S16, 0, $a, ROUND_RTZ)>;
def : Pat<(i16 (fp_to_uint f32:$a)), (CONVERT_16_32 CVT_F_TO_U16, 0, $a, ROUND_RTZ)>;
def : Pat<(i16 (fp_to_sint f16:$a)), (CONVERT_16_16 CVT_F_TO_S16, 0, $a, ROUND_RTZ)>;
def : Pat<(i16 (fp_to_uint f16:$a)), (CONVERT_16_16 CVT_F_TO_U16, 0, $a, ROUND_RTZ)>;
def : Pat<(f32 (sint_to_fp i32:$a)), (CONVERT_32_32 CVT_S32_TO_F, 0, $a, ROUND_RTE)>;
def : Pat<(f32 (uint_to_fp i32:$a)), (CONVERT_32_32 CVT_U32_TO_F, 0, $a, ROUND_RTE)>;
def : Pat<(f32 (sint_to_fp i16:$a)), (CONVERT_32_16 CVT_S16_TO_F, 0, $a, ROUND_RTE)>;
def : Pat<(f32 (uint_to_fp i16:$a)), (CONVERT_32_16 CVT_U16_TO_F, 0, $a, ROUND_RTE)>;
def : Pat<(f16 (sint_to_fp i32:$a)), (CONVERT_16_32 CVT_S32_TO_F, 0, $a, ROUND_RTE)>;
def : Pat<(f16 (uint_to_fp i32:$a)), (CONVERT_16_32 CVT_U32_TO_F, 0, $a, ROUND_RTE)>;
def : Pat<(f16 (sint_to_fp i16:$a)), (CONVERT_16_16 CVT_S16_TO_F, 0, $a, ROUND_RTE)>;
def : Pat<(f16 (uint_to_fp i16:$a)), (CONVERT_16_16 CVT_U16_TO_F, 0, $a, ROUND_RTE)>;

// Bit casts between same sized types are free.
def : Pat<(i32 (bitconvert f32:$a)), (COPY_TO_REGCLASS $a, GPR32)>;
def : Pat<(f32 (bitconvert i32:$a)), (COPY_TO_REGCLASS $a, GPR32)>;
def : Pat<(i16 (bitconvert f16:$a)), (COPY_TO_REGCLASS $a, GPR16)>;
def : Pat<(f16 (bitconvert i16:$a)), (COPY_TO_REGCLASS $a, GPR16)>;
def : Pat<(i16 (bitconvert bf16:$a)), (COPY_TO_REGCLASS $a, GPR16)>;
def : Pat<(bf16 (bitconvert i16:$a)), (COPY_TO_REGCLASS $a, GPR16)>;
def : Pat<(i64 (bitconvert f64:$a)), (COPY_TO_REGCLASS $a, GPR64)>;
def : Pat<(f64 (bitconvert i64:$a)), (COPY_TO_REGCLASS $a, GPR64)>;

//===----------------------------------------------------------------------===//
// Device memory
//===----------------------------------------------------------------------===//

class DeviceLoadPseudo<RegisterClass rc>
    : AGXPseudo<(outs rc:$R),
                (ins i32imm:$g, i32imm:$fmt, i32imm:$mask, Base64Op:$A,
                     IndexOp:$O, i32imm:$sgn, i32imm:$shift)> {
  let G13Real = DEVICE_LOAD_g13;
  let mayLoad = 1;
  let IsAsync = 1;
  let IsMemory = 1;
  let SchedRW = [WriteDeviceLoad];
}

class DeviceStorePseudo<RegisterClass rc>
    : AGXPseudo<(outs),
                (ins i32imm:$g, i32imm:$fmt, i32imm:$mask, rc:$R,
                     Base64Op:$A, IndexOp:$O, i32imm:$sgn, i32imm:$shift)> {
  let G13Real = DEVICE_STORE_g13;
  let mayStore = 1;
  let IsMemory = 1;
  let SchedRW = [WriteDeviceStore];
}

def DEVICE_LOAD_16 : DeviceLoadPseudo<GPR16>;
def DEVICE_LOAD_32 : DeviceLoadPseudo<GPR32>;
def DEVICE_LOAD_64 : DeviceLoadPseudo<GPR32x2>;
def DEVICE_LOAD_96 : DeviceLoadPseudo<GPR96>;
def DEVICE_LOAD_128 : DeviceLoadPseudo<GPR128>;
def DEVICE_STORE_16 : DeviceStorePseudo<GPR16>;
def DEVICE_STORE_32 : DeviceStorePseudo<GPR32>;
def DEVICE_STORE_64 : DeviceStorePseudo<GPR32x2>;
def DEVICE_STORE_96 : DeviceStorePseudo<GPR96>;
def DEVICE_STORE_128 : DeviceStorePseudo<GPR128>;

multiclass DeviceLoadPat<PatFrag frag, ValueType vt, Instruction inst,
                         ComplexPattern addr, int fmt, int mask> {
  def : Pat<(vt (frag (addr i64:$base, i32:$idx, i32:$sgn, i32:$shift))),
            (inst 0, fmt, mask, $base, $idx, $sgn, $shift)>;
}

multiclass DeviceStorePat<PatFrag frag, ValueType vt, Instruction inst,
                          ComplexPattern addr, int fmt, int mask> {
  def : Pat<(frag vt:$val, (addr i64:$base, i32:$idx, i32:$sgn, i32:$shift)),
            (inst 0, fmt, mask, $val, $base, $idx, $sgn, $shift)>;
}

defm : DeviceLoadPat<device_load, i32, DEVICE_LOAD_32, AGXDeviceAddr32, FMT_I32, 1>;
defm : DeviceLoadPat<device_load, f32, DEVICE_LOAD_32, AGXDeviceAddr32, FMT_I32, 1>;
defm : DeviceLoadPat<device_load, i64, DEVICE_LOAD_64, AGXDeviceAddr32, FMT_I32, 3>;
defm : DeviceLoadPat<device_load, f64, DEVICE_LOAD_64, AGXDeviceAddr32, FMT_I32, 3>;
defm : DeviceLoadPat<device_load, i16, DEVICE_LOAD_16, AGXDeviceAddr16, FMT_I16, 1>;
defm : DeviceLoadPat<device_load, f16, DEVICE_LOAD_16, AGXDeviceAddr16, FMT_I16, 1>;
defm : DeviceLoadPat<device_load, bf16, DEVICE_LOAD_16, AGXDeviceAddr16, FMT_I16, 1>;
defm : DeviceLoadPat<device_extloadi8, i16, DEVICE_LOAD_16, AGXDeviceAddr8, FMT_I8, 1>;
defm : DeviceLoadPat<device_zextloadi8, i16, DEVICE_LOAD_16, AGXDeviceAddr8, FMT_I8, 1>;
defm : DeviceLoadPat<device_extloadi8, i32, DEVICE_LOAD_32, AGXDeviceAddr8, FMT_I8, 1>;
defm : DeviceLoadPat<device_zextloadi8, i32, DEVICE_LOAD_32, AGXDeviceAddr8, FMT_I8, 1>;
defm : DeviceLoadPat<device_extloadi16, i32, DEVICE_LOAD_32, AGXDeviceAddr16, FMT_I16, 1>;
defm : DeviceLoadPat<device_zextloadi16, i32, DEVICE_LOAD_32, AGXDeviceAddr16, FMT_I16, 1>;

defm : DeviceStorePat<device_store, i32, DEVICE_STORE_32, AGXDeviceAddr32, FMT_I32, 1>;
defm : DeviceStorePat<device_store, f32, DEVICE_STORE_32, AGXDeviceAddr32, FMT_I32, 1>;
defm : DeviceStorePat<device_store, i64, DEVICE_STORE_64, AGXDeviceAddr32, FMT_I32, 3>;
defm : DeviceStorePat<device_store, f64, DEVICE_STORE_64, AGXDeviceAddr32, FMT_I32, 3>;
defm : DeviceStorePat<device_store, i16, DEVICE_STORE_16, AGXDeviceAddr16, FMT_I16, 1>;
defm : DeviceStorePat<device_store, f16, DEVICE_STORE_16, AGXDeviceAddr16, FMT_I16, 1>;
defm : DeviceStorePat<device_store, bf16, DEVICE_STORE_16, AGXDeviceAddr16, FMT_I16, 1>;
defm : DeviceStorePat<device_truncstorei8, i16, DEVICE_STORE_16, AGXDeviceAddr8, FMT_I8, 1>;
defm : DeviceStorePat<device_truncstorei8, i32, DEVICE_STORE_32, AGXDeviceAddr8, FMT_I8, 1>;
defm : DeviceStorePat<device_truncstorei16, i32, DEVICE_STORE_32, AGXDeviceAddr16, FMT_I16, 1>;

//===----------------------------------------------------------------------===//
// Threadgroup memory
//===----------------------------------------------------------------------===//

class LocalLoadPseudo<RegisterClass rc>
    : AGXPseudo<(outs rc:$R),
                (ins i32imm:$fmt, i32imm:$mask, LocalBaseOp:$A,
                     LocalIndexOp:$O)> {
  let G13Real = THREADGROUP_LOAD_g13;
  let mayLoad = 1;
  let IsMemory = 1;
  let SchedRW = [WriteLocalLoad];
}

class LocalStorePseudo<RegisterClass rc>
    : AGXPseudo<(outs),
                (ins i32imm:$fmt, i32imm:$mask, rc:$R, LocalBaseOp:$A,
                     LocalIndexOp:$O)> {
  let G13Real = THREADGROUP_STORE_g13;
  let mayStore = 1;
  let IsMemory = 1;
  let SchedRW = [WriteLocalStore];
}

def LOCAL_LOAD_16 : LocalLoadPseudo<GPR16>;
def LOCAL_LOAD_32 : LocalLoadPseudo<GPR32>;
def LOCAL_LOAD_64 : LocalLoadPseudo<GPR32x2>;
def LOCAL_STORE_16 : LocalStorePseudo<GPR16>;
def LOCAL_STORE_32 : LocalStorePseudo<GPR32>;
def LOCAL_STORE_64 : LocalStorePseudo<GPR32x2>;

multiclass LocalLoadPat<PatFrag frag, ValueType vt, Instruction inst,
                        ComplexPattern addr, int fmt, int mask> {
  def : Pat<(vt (frag (addr i16:$base, i16:$idx))),
            (inst fmt, mask, $base, $idx)>;
}

multiclass LocalStorePat<PatFrag frag, ValueType vt, Instruction inst,
                         ComplexPattern addr, int fmt, int mask> {
  def : Pat<(frag vt:$val, (addr i16:$base, i16:$idx)),
            (inst fmt, mask, $val, $base, $idx)>;
}

defm : LocalLoadPat<local_load, i32, LOCAL_LOAD_32, AGXLocalAddr32, FMT_I32, 1>;
defm : LocalLoadPat<local_load, f32, LOCAL_LOAD_32, AGXLocalAddr32, FMT_I32, 1>;
defm : LocalLoadPat<local_load, i64, LOCAL_LOAD_64, AGXLocalAddr32, FMT_I32, 3>;
defm : LocalLoadPat<local_load, f64, LOCAL_LOAD_64, AGXLocalAddr32, FMT_I32, 3>;
defm : LocalLoadPat<local_load, i16, LOCAL_LOAD_16, AGXLocalAddr16, FMT_I16, 1>;
defm : LocalLoadPat<local_load, f16, LOCAL_LOAD_16, AGXLocalAddr16, FMT_I16, 1>;
defm : LocalLoadPat<local_extloadi8, i16, LOCAL_LOAD_16, AGXLocalAddr8, FMT_I8, 1>;
defm : LocalLoadPat<local_zextloadi8, i16, LOCAL_LOAD_16, AGXLocalAddr8, FMT_I8, 1>;
defm : LocalLoadPat<local_extloadi8, i32, LOCAL_LOAD_32, AGXLocalAddr8, FMT_I8, 1>;
defm : LocalLoadPat<local_zextloadi8, i32, LOCAL_LOAD_32, AGXLocalAddr8, FMT_I8, 1>;
defm : LocalLoadPat<local_extloadi16, i32, LOCAL_LOAD_32, AGXLocalAddr16, FMT_I16, 1>;
defm : LocalLoadPat<local_zextloadi16, i32, LOCAL_LOAD_32, AGXLocalAddr16, FMT_I16, 1>;

defm : LocalStorePat<local_store, i32, LOCAL_STORE_32, AGXLocalAddr32, FMT_I32, 1>;
defm : LocalStorePat<local_store, f32, LOCAL_STORE_32, AGXLocalAddr32, FMT_I32, 1>;
defm : LocalStorePat<local_store, i64, LOCAL_STORE_64, AGXLocalAddr32, FMT_I32, 3>;
defm : LocalStorePat<local_store, f64, LOCAL_STORE_64, AGXLocalAddr32, FMT_I32, 3>;
defm : LocalStorePat<local_store, i16, LOCAL_STORE_16, AGXLocalAddr16, FMT_I16, 1>;
defm : LocalStorePat<local_store, f16, LOCAL_STORE_16, AGXLocalAddr16, FMT_I16, 1>;
defm : LocalStorePat<local_truncstorei8, i16, LOCAL_STORE_16, AGXLocalAddr8, FMT_I8, 1>;
defm : LocalStorePat<local_truncstorei8, i32, LOCAL_STORE_32, AGXLocalAddr8, FMT_I8, 1>;
defm : LocalStorePat<local_truncstorei16, i32, LOCAL_STORE_32, AGXLocalAddr16, FMT_I16, 1>;

//===----------------------------------------------------------------------===//
// Stack memory
//===----------------------------------------------------------------------===//

// Byte offset into the thread's stack.
def StackIndexOp : AGXRegOperand<GPR32, "OPERAND_SRC">;

class StackLoadPseudo<RegisterClass rc>
    : AGXPseudo<(outs rc:$R),
                (ins i32imm:$g, i32imm:$fmt, i32imm:$mask, StackIndexOp:$O)> {
  let G13Real = STACK_LOAD_g13;
  let mayLoad = 1;
  let IsAsync = 1;
  let IsMemory = 1;
  let SchedRW = [WriteDeviceLoad];
}

class StackStorePseudo<RegisterClass rc>
    : AGXPseudo<(outs),
                (ins i32imm:$g, i32imm:$fmt, i32imm:$mask, rc:$R,
                     StackIndexOp:$O)> {
  let G13Real = STACK_STORE_g13;
  let mayStore = 1;
  let IsMemory = 1;
  let SchedRW = [WriteDeviceStore];
}

def STACK_LOAD_16 : StackLoadPseudo<GPR16>;
def STACK_LOAD_32 : StackLoadPseudo<GPR32>;
def STACK_LOAD_64 : StackLoadPseudo<GPR32x2>;
def STACK_LOAD_96 : StackLoadPseudo<GPR96>;
def STACK_LOAD_128 : StackLoadPseudo<GPR128>;
def STACK_STORE_16 : StackStorePseudo<GPR16>;
def STACK_STORE_32 : StackStorePseudo<GPR32>;
def STACK_STORE_64 : StackStorePseudo<GPR32x2>;
def STACK_STORE_96 : StackStorePseudo<GPR96>;
def STACK_STORE_128 : StackStorePseudo<GPR128>;

def : Pat<(i32 (private_load (AGXStackAddr i32:$off))),
          (STACK_LOAD_32 0, FMT_I32, 1, $off)>;
def : Pat<(f32 (private_load (AGXStackAddr i32:$off))),
          (STACK_LOAD_32 0, FMT_I32, 1, $off)>;
def : Pat<(i64 (private_load (AGXStackAddr i32:$off))),
          (STACK_LOAD_64 0, FMT_I32, 3, $off)>;
def : Pat<(i16 (private_load (AGXStackAddr i32:$off))),
          (STACK_LOAD_16 0, FMT_I16, 1, $off)>;
def : Pat<(f16 (private_load (AGXStackAddr i32:$off))),
          (STACK_LOAD_16 0, FMT_I16, 1, $off)>;
def : Pat<(i16 (private_extloadi8 (AGXStackAddr i32:$off))),
          (STACK_LOAD_16 0, FMT_I8, 1, $off)>;
def : Pat<(i16 (private_zextloadi8 (AGXStackAddr i32:$off))),
          (STACK_LOAD_16 0, FMT_I8, 1, $off)>;
def : Pat<(i32 (private_extloadi8 (AGXStackAddr i32:$off))),
          (STACK_LOAD_32 0, FMT_I8, 1, $off)>;
def : Pat<(i32 (private_zextloadi8 (AGXStackAddr i32:$off))),
          (STACK_LOAD_32 0, FMT_I8, 1, $off)>;
def : Pat<(i32 (private_extloadi16 (AGXStackAddr i32:$off))),
          (STACK_LOAD_32 0, FMT_I16, 1, $off)>;
def : Pat<(i32 (private_zextloadi16 (AGXStackAddr i32:$off))),
          (STACK_LOAD_32 0, FMT_I16, 1, $off)>;

def : Pat<(private_store i32:$v, (AGXStackAddr i32:$off)),
          (STACK_STORE_32 0, FMT_I32, 1, $v, $off)>;
def : Pat<(private_store f32:$v, (AGXStackAddr i32:$off)),
          (STACK_STORE_32 0, FMT_I32, 1, $v, $off)>;
def : Pat<(private_store i64:$v, (AGXStackAddr i32:$off)),
          (STACK_STORE_64 0, FMT_I32, 3, $v, $off)>;
def : Pat<(private_store i16:$v, (AGXStackAddr i32:$off)),
          (STACK_STORE_16 0, FMT_I16, 1, $v, $off)>;
def : Pat<(private_store f16:$v, (AGXStackAddr i32:$off)),
          (STACK_STORE_16 0, FMT_I16, 1, $v, $off)>;
def : Pat<(private_truncstorei8 i16:$v, (AGXStackAddr i32:$off)),
          (STACK_STORE_16 0, FMT_I8, 1, $v, $off)>;
def : Pat<(private_truncstorei8 i32:$v, (AGXStackAddr i32:$off)),
          (STACK_STORE_32 0, FMT_I8, 1, $v, $off)>;
def : Pat<(private_truncstorei16 i32:$v, (AGXStackAddr i32:$off)),
          (STACK_STORE_32 0, FMT_I16, 1, $v, $off)>;

// The address of a stack object, as a byte offset.
let isReMaterializable = 1 in
def FRAME_ADDR : AGXALUPseudo<MOV_IMM32_g13, (outs GPR32:$dst),
                              (ins i32imm:$imm), [], [WriteMovImm]>;
def agx_to_tframeindex : SDNodeXForm<frameindex, [{
  return CurDAG->getTargetFrameIndex(N->getIndex(), MVT::i32);
}]>;

def : Pat<(i32 frameindex:$fi), (FRAME_ADDR (agx_to_tframeindex $fi))>;

let hasSideEffects = 1 in
def STACK_ADJUST : AGXPseudo<(outs), (ins i32imm:$size)> {
  let G13Real = STACK_ADJUST_g13;
}

//===----------------------------------------------------------------------===//
// Atomics
//===----------------------------------------------------------------------===//

// Atomic operation numbers.
defvar AOP_ADD = 0;
defvar AOP_SUB = 1;
defvar AOP_XCHG = 2;
defvar AOP_CMPXCHG = 3;
defvar AOP_UMIN = 4;
defvar AOP_IMIN = 5;
defvar AOP_UMAX = 6;
defvar AOP_IMAX = 7;
defvar AOP_AND = 8;
defvar AOP_OR = 9;
defvar AOP_XOR = 10;

let mayLoad = 1, mayStore = 1, hasSideEffects = 1, IsMemory = 1 in {
def DEVICE_ATOMIC : AGXPseudo<(outs GPR32:$R),
                              (ins i32imm:$op, i32imm:$g, Base64Op:$A,
                                   IndexOp:$O, i32imm:$sgn, GPR32:$S,
                                   i32imm:$coherent)> {
  let G13Real = ATOMIC_g13;
  let IsAsync = 1;
  let SchedRW = [WriteAtomic];
}

// The data operand is the pair {new value, compare value}.
def DEVICE_ATOMIC_CMPXCHG : AGXPseudo<(outs GPR32:$R),
                                      (ins i32imm:$op, i32imm:$g, Base64Op:$A,
                                           IndexOp:$O, i32imm:$sgn, GPR64:$S,
                                           i32imm:$coherent)> {
  let G13Real = ATOMIC_g13;
  let IsAsync = 1;
  let SchedRW = [WriteAtomic];
}

def LOCAL_ATOMIC : AGXPseudo<(outs GPR32:$R),
                             (ins i32imm:$op, LocalBaseOp:$A, LocalIndexOp:$O,
                                  GPR32:$S)> {
  let G13Real = THREADGROUP_ATOMIC_g13;
  let SchedRW = [WriteAtomic];
}

def LOCAL_ATOMIC_CMPXCHG : AGXPseudo<(outs GPR32:$R),
                                     (ins i32imm:$op, LocalBaseOp:$A,
                                          LocalIndexOp:$O, GPR64:$S)> {
  let G13Real = THREADGROUP_ATOMIC_g13;
  let SchedRW = [WriteAtomic];
}
}

class AtomicFrag<SDPatternOperator op, list<int> as>
    : PatFrag<(ops node:$ptr, node:$val), (op node:$ptr, node:$val)> {
  let IsAtomic = 1;
  let MemoryVT = i32;
  let AddressSpaces = as;
}

class AtomicCmpSwapFrag<list<int> as>
    : PatFrag<(ops node:$ptr, node:$cmp, node:$new),
              (atomic_cmp_swap node:$ptr, node:$cmp, node:$new)> {
  let IsAtomic = 1;
  let MemoryVT = i32;
  let AddressSpaces = as;
}

multiclass AGXAtomicPats<SDPatternOperator node, int op> {
  def _device : AtomicFrag<node, DeviceAS>;
  def _local : AtomicFrag<node, LocalAS>;

  // Atomics take the full address in the base register; the element index
  // is always zero.
  def : Pat<(i32 (!cast<PatFrag>(NAME#"_device") i64:$ptr, i32:$val)),
            (DEVICE_ATOMIC op, 0, $ptr, (i32 0), 0, $val, 0)>;
  def : Pat<(i32 (!cast<PatFrag>(NAME#"_local") i32:$ptr, i32:$val)),
            (LOCAL_ATOMIC op, (EXTRACT_SUBREG $ptr, lo16), (i16 0), $val)>;
}

defm agx_atomic_add : AGXAtomicPats<atomic_load_add, AOP_ADD>;
defm agx_atomic_sub : AGXAtomicPats<atomic_load_sub, AOP_SUB>;
defm agx_atomic_xchg : AGXAtomicPats<atomic_swap, AOP_XCHG>;
defm agx_atomic_umin : AGXAtomicPats<atomic_load_umin, AOP_UMIN>;
defm agx_atomic_imin : AGXAtomicPats<atomic_load_min, AOP_IMIN>;
defm agx_atomic_umax : AGXAtomicPats<atomic_load_umax, AOP_UMAX>;
defm agx_atomic_imax : AGXAtomicPats<atomic_load_max, AOP_IMAX>;
defm agx_atomic_and : AGXAtomicPats<atomic_load_and, AOP_AND>;
defm agx_atomic_or : AGXAtomicPats<atomic_load_or, AOP_OR>;
defm agx_atomic_xor : AGXAtomicPats<atomic_load_xor, AOP_XOR>;

def agx_cmpxchg_device : AtomicCmpSwapFrag<DeviceAS>;
def agx_cmpxchg_local : AtomicCmpSwapFrag<LocalAS>;

def : Pat<(i32 (agx_cmpxchg_device i64:$ptr, i32:$cmp, i32:$new)),
          (DEVICE_ATOMIC_CMPXCHG AOP_CMPXCHG, 0, $ptr, (i32 0), 0,
             (REG_SEQUENCE GPR64, $new, sub0, $cmp, sub1), 0)>;
def : Pat<(i32 (agx_cmpxchg_local i32:$ptr, i32:$cmp, i32:$new)),
          (LOCAL_ATOMIC_CMPXCHG AOP_CMPXCHG, (EXTRACT_SUBREG $ptr, lo16),
             (i16 0), (REG_SEQUENCE GPR64, $new, sub0, $cmp, sub1))>;

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, isConvergent = 1 in {
def THREADGROUP_BARRIER : AGXPseudo<(outs), (ins),
                                    [(int_agx_threadgroup_barrier)]> {
  let G13Real = THREADGROUP_BARRIER_g13;
  let SchedRW = [WriteBarrier];
}
}

let hasSideEffects = 1 in
def MEMORY_BARRIER : AGXPseudo<(outs), (ins i32imm:$a, i32imm:$b, i32imm:$c),
                               [(int_agx_memory_barrier timm:$a, timm:$b,
                                                        timm:$c)]> {
  let G13Real = MEMORY_BARRIER_g13;
  let SchedRW = [WriteBarrier];
}

// Device scope barrier, as emitted for fences.
def : Pat<(atomic_fence (i64 timm), (i64 timm)), (MEMORY_BARRIER 1, 2, 9)>;

let hasSideEffects = 1 in
def WAIT : AGXPseudo<(outs), (ins i32imm:$slot), [(int_agx_wait timm:$slot)]> {
  let G13Real = WAIT_g13;
  let SchedRW = [WriteControl];
}

//===----------------------------------------------------------------------===//
// SIMD-group operations
//===----------------------------------------------------------------------===//

def shuffle_imm_xform : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), SDLoc(N), MVT::i16);
}]>;
def shuffle_imm : ImmLeaf<i32, [{ return Imm >= 0 && Imm < 256; }],
                          shuffle_imm_xform>;

multiclass ShufflePseudo<Instruction real, SDPatternOperator intr> {
  let isConvergent = 1 in {
  def _32 : AGXALUPseudo<real, (outs GPR32:$dst),
                         (ins SrcMods:$src0_mods, Src32Op:$src0,
                              SrcMods:$src1_mods, Src16Op:$src1), [],
                         [WriteShuffle]>;
  def _16 : AGXALUPseudo<real, (outs GPR16:$dst),
                         (ins SrcMods:$src0_mods, Src16Op:$src0,
                              SrcMods:$src1_mods, Src16Op:$src1), [],
                         [WriteShuffle]>;
  }
  foreach VT = [i32, f32] in
  def : Pat<(VT (intr VT:$v, i32:$idx)),
            (!cast<Instruction>(NAME#"_32") 0, $v, 0,
                                            (EXTRACT_SUBREG $idx, lo16))>;
  foreach VT = [i16, f16] in
  def : Pat<(VT (intr VT:$v, i32:$idx)),
            (!cast<Instruction>(NAME#"_16") 0, $v, 0,
                                            (EXTRACT_SUBREG $idx, lo16))>;
  // Immediate indices are the same in every thread.
  let AddedComplexity = 10 in {
  foreach VT = [i32, f32] in
  def : Pat<(VT (intr VT:$v, shuffle_imm:$idx)),
            (!cast<Instruction>(NAME#"_32") 0, $v, 0, (i16 (shuffle_imm_xform $idx)))>;
  foreach VT = [i16, f16] in
  def : Pat<(VT (intr VT:$v, shuffle_imm:$idx)),
            (!cast<Instruction>(NAME#"_16") 0, $v, 0, (i16 (shuffle_imm_xform $idx)))>;
  }
}

defm SIMD_SHUFFLE : ShufflePseudo<SIMD_SHUFFLE_g13, int_agx_simd_shuffle>;
defm SIMD_SHUFFLE_XOR : ShufflePseudo<SIMD_SHUFFLE_XOR_g13, int_agx_simd_shuffle_xor>;
defm SIMD_SHUFFLE_UP : ShufflePseudo<SIMD_SHUFFLE_UP_g13, int_agx_simd_shuffle_up>;
defm SIMD_SHUFFLE_DOWN : ShufflePseudo<SIMD_SHUFFLE_DOWN_g13, int_agx_simd_shuffle_down>;
defm QUAD_SHUFFLE : ShufflePseudo<QUAD_SHUFFLE_g13, int_agx_quad_shuffle>;
defm QUAD_SHUFFLE_XOR : ShufflePseudo<QUAD_SHUFFLE_XOR_g13, int_agx_quad_shuffle_xor>;
defm QUAD_SHUFFLE_UP : ShufflePseudo<QUAD_SHUFFLE_UP_g13, int_agx_quad_shuffle_up>;
defm QUAD_SHUFFLE_DOWN : ShufflePseudo<QUAD_SHUFFLE_DOWN_g13, int_agx_quad_shuffle_down>;

let isConvergent = 1 in {
def ICMP_BALLOT : AGXALUPseudo<ICMP_BALLOT_g13, (outs GPR32:$dst),
                               (ins i32imm:$cc, SrcMods:$src0_mods,
                                    Src32Op:$src0, SrcMods:$src1_mods,
                                    Src32Op:$src1), [], [WriteShuffle]>;
def ICMP_QUAD_BALLOT : AGXALUPseudo<ICMP_QUAD_BALLOT_g13, (outs GPR32:$dst),
                                    (ins i32imm:$cc, SrcMods:$src0_mods,
                                         Src32Op:$src0, SrcMods:$src1_mods,
                                         Src32Op:$src1), [], [WriteShuffle]>;
}

// ballot(c) = icmp_ballot nueq c, 0 (the inverted ueq condition is 8).
def : Pat<(i32 (AGXballot i32:$c)),
          (ICMP_BALLOT 8, 0, $c, 0, (i32 0))>;
def : Pat<(i32 (AGXquad_ballot i32:$c)),
          (BITOP_I32 TT_AND, 0, (ICMP_QUAD_BALLOT 8, 0, $c, 0, (i32 0)), 0,
                     (i32 15))>;

foreach Op = AGXSIMDOps in
def AGXSIMDOpImm#Op.Op : TImmLeaf<i32, "return Imm == "#Op.Op#";">;

foreach Kind = AGXReduceKinds in
foreach Op = AGXSIMDOps in {
  defvar Real = !cast<Instruction>(Kind.Name#"_"#Op.Name#"_g13");
  let isConvergent = 1 in
  def Kind.Name#_#Op.Name#_32
      : AGXALUPseudo<Real, (outs GPR32:$dst),
                     (ins SrcMods:$src0_mods, Src32Op:$src0), [],
                     [!if(!eq(Kind.Opc, 1), WriteSIMDReduce,
                          !if(!eq(Kind.Opc, 3), WriteSIMDReduce,
                              WriteQuadReduce))]>;
  defvar Intr = !cond(!eq(Kind.Opc, 0) : int_agx_quad_reduce,
                      !eq(Kind.Opc, 1) : int_agx_simd_reduce,
                      !eq(Kind.Opc, 2) : int_agx_quad_prefix,
                      true : int_agx_simd_prefix);
  foreach VT = [i32, f32] in
  def : Pat<(VT (Intr VT:$v, (i32 !cast<TImmLeaf>("AGXSIMDOpImm"#Op.Op)))),
            (!cast<Instruction>(Kind.Name#"_"#Op.Name#"_32") 0, $v)>;
}

let isConvergent = 1 in
def SIMD_MATRIX_FMADD32 : AGXALUPseudo<SIMD_MATRIX_FMADD32_g13,
                                       (outs GPR64:$dst),
                                       (ins SrcMods:$src0_mods, Src64Op:$src0,
                                            SrcMods:$src1_mods, Src64Op:$src1,
                                            SrcMods:$src2_mods, Src64Op:$src2),
                                       [], [WriteSIMDReduce]>;

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

// Exec mask instructions, operating on the nesting counter r0l.
let hasSideEffects = 1, Uses = [R0L], Defs = [R0L], IsExecMaskOp = 1,
    SchedRW = [WriteControl] in {
def IF_ICMP : AGXPseudo<(outs), (ins i32imm:$nest, i32imm:$cc,
                                     SrcMods:$src0_mods, Src32Op:$src0,
                                     SrcMods:$src1_mods, Src32Op:$src1,
                                     i32imm:$n)> {
  let G13Real = IF_ICMP_g13;
}
def ELSE_ICMP : AGXPseudo<(outs), (ins i32imm:$nest, i32imm:$cc,
                                       SrcMods:$src0_mods, Src32Op:$src0,
                                       SrcMods:$src1_mods, Src32Op:$src1,
                                       i32imm:$n)> {
  let G13Real = ELSE_ICMP_g13;
}
def WHILE_ICMP : AGXPseudo<(outs), (ins i32imm:$nest, i32imm:$cc,
                                        SrcMods:$src0_mods, Src32Op:$src0,
                                        SrcMods:$src1_mods, Src32Op:$src1,
                                        i32imm:$n)> {
  let G13Real = WHILE_ICMP_g13;
}
def IF_FCMP : AGXPseudo<(outs), (ins i32imm:$nest, i32imm:$cc,
                                     SrcMods:$src0_mods, Src32Op:$src0,
                                     SrcMods:$src1_mods, Src32Op:$src1,
                                     i32imm:$n)> {
  let G13Real = IF_FCMP_g13;
}
def ELSE_FCMP : AGXPseudo<(outs), (ins i32imm:$nest, i32imm:$cc,
                                       SrcMods:$src0_mods, Src32Op:$src0,
                                       SrcMods:$src1_mods, Src32Op:$src1,
                                       i32imm:$n)> {
  let G13Real = ELSE_FCMP_g13;
}
def POP_EXEC : AGXPseudo<(outs), (ins i32imm:$nest, i32imm:$n)> {
  let G13Real = POP_EXEC_g13;
}
}

// mov_imm r0l, n: initialise or set the nesting counter.
let hasSideEffects = 1, Defs = [R0L] in
def SET_NEST : AGXPseudo<(outs), (ins i32imm:$imm)> {
  let G13Real = MOV_IMM16_g13;
}

let isBranch = 1, isTerminator = 1, Uses = [R0L] in {
def JMP_EXEC_ANY : AGXPseudo<(outs), (ins AGXBrTarget:$target)> {
  let G13Real = JMP_EXEC_ANY_g13;
}
def JMP_EXEC_NONE : AGXPseudo<(outs), (ins AGXBrTarget:$target)> {
  let G13Real = JMP_EXEC_NONE_g13;
}
}

// Structured control flow, expanded by AGXLowerControlFlow.
let isBranch = 1, isTerminator = 1, hasSideEffects = 1, Uses = [R0L],
    Defs = [R0L] in {
// Deactivate the threads where $cond is zero; branch to $target if no thread
// remains active.
def AGX_IF : AGXCodeGenPseudo<(outs), (ins GPR32:$cond, AGXBrTarget:$target),
                              [(AGXif i32:$cond, bb:$target)]>;
// Swap the active threads of the innermost if; branch to $target if none.
def AGX_ELSE : AGXCodeGenPseudo<(outs), (ins AGXBrTarget:$target),
                                [(AGXelse bb:$target)]>;
// Deactivate the threads where $brk is nonzero; branch to $target (the
// loop header) if any thread remains active.
def AGX_LOOP : AGXCodeGenPseudo<(outs), (ins GPR32:$brk, AGXBrTarget:$target),
                                [(AGXloop i32:$brk, bb:$target)]>;
}

let hasSideEffects = 1, Uses = [R0L], Defs = [R0L] in {
// Reactivate the threads deactivated by the innermost construct.
def AGX_END_CF : AGXCodeGenPseudo<(outs), (ins), [(AGXend_cf)]>;
// Enter a loop: protect the threads that are already inactive.
def AGX_LOOP_BEGIN : AGXCodeGenPseudo<(outs), (ins), [(AGXloop_begin)]>;
}

let isBranch = 1, isTerminator = 1, isBarrier = 1 in
def AGX_BR : AGXCodeGenPseudo<(outs), (ins AGXBrTarget:$target), [(br bb:$target)]>;

let isTerminator = 1, isReturn = 1, isBarrier = 1, hasSideEffects = 1 in
def AGX_STOP : AGXPseudo<(outs), (ins), [(AGXstop)]> {
  let G13Real = STOP_g13;
}

let isTrap = 1, isTerminator = 1, isBarrier = 1, hasSideEffects = 1 in
def AGX_TRAP : AGXPseudo<(outs), (ins), [(trap)]> {
  let G13Real = TRAP_g13;
}

def : Pat<(debugtrap), (AGX_TRAP)>;
