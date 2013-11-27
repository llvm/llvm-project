//===-- SparcInstr64Bit.td - 64-bit instructions for Sparc Target ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains instruction definitions and patterns needed for 64-bit
// code generation on SPARC v9.
//
// Some SPARC v9 instructions are defined in SparcInstrInfo.td because they can
// also be used in 32-bit code running on a SPARC v9 CPU.
//
//===----------------------------------------------------------------------===//

let Predicates = [Is64Bit] in {
// The same integer registers are used for i32 and i64 values.
// When registers hold i32 values, the high bits are don't care.
// This give us free trunc and anyext.
def : Pat<(i64 (anyext i32:$val)), (COPY_TO_REGCLASS $val, I64Regs)>;
def : Pat<(i32 (trunc i64:$val)), (COPY_TO_REGCLASS $val, IntRegs)>;

} // Predicates = [Is64Bit]


//===----------------------------------------------------------------------===//
// 64-bit Shift Instructions.
//===----------------------------------------------------------------------===//
//
// The 32-bit shift instructions are still available. The left shift srl
// instructions shift all 64 bits, but it only accepts a 5-bit shift amount.
//
// The srl instructions only shift the low 32 bits and clear the high 32 bits.
// Finally, sra shifts the low 32 bits and sign-extends to 64 bits.

let Predicates = [Is64Bit] in {

def : Pat<(i64 (zext i32:$val)), (SRLri $val, 0)>;
def : Pat<(i64 (sext i32:$val)), (SRAri $val, 0)>;

def : Pat<(i64 (and i64:$val, 0xffffffff)), (SRLri $val, 0)>;
def : Pat<(i64 (sext_inreg i64:$val, i32)), (SRAri $val, 0)>;

defm SLLX : F3_S<"sllx", 0b100101, 1, shl, i64, I64Regs>;
defm SRLX : F3_S<"srlx", 0b100110, 1, srl, i64, I64Regs>;
defm SRAX : F3_S<"srax", 0b100111, 1, sra, i64, I64Regs>;

} // Predicates = [Is64Bit]


//===----------------------------------------------------------------------===//
// 64-bit Immediates.
//===----------------------------------------------------------------------===//
//
// All 32-bit immediates can be materialized with sethi+or, but 64-bit
// immediates may require more code. There may be a point where it is
// preferable to use a constant pool load instead, depending on the
// microarchitecture.

// Single-instruction patterns.

// The ALU instructions want their simm13 operands as i32 immediates.
def as_i32imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), MVT::i32);
}]>;
def : Pat<(i64 simm13:$val), (ORri (i64 G0), (as_i32imm $val))>;
def : Pat<(i64 SETHIimm:$val), (SETHIi (HI22 $val))>;

// Double-instruction patterns.

// All unsigned i32 immediates can be handled by sethi+or.
def uimm32 : PatLeaf<(imm), [{ return isUInt<32>(N->getZExtValue()); }]>;
def : Pat<(i64 uimm32:$val), (ORri (SETHIi (HI22 $val)), (LO10 $val))>,
      Requires<[Is64Bit]>;

// All negative i33 immediates can be handled by sethi+xor.
def nimm33 : PatLeaf<(imm), [{
  int64_t Imm = N->getSExtValue();
  return Imm < 0 && isInt<33>(Imm);
}]>;
// Bits 10-31 inverted. Same as assembler's %hix.
def HIX22 : SDNodeXForm<imm, [{
  uint64_t Val = (~N->getZExtValue() >> 10) & ((1u << 22) - 1);
  return CurDAG->getTargetConstant(Val, MVT::i32);
}]>;
// Bits 0-9 with ones in bits 10-31. Same as assembler's %lox.
def LOX10 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(~(~N->getZExtValue() & 0x3ff), MVT::i32);
}]>;
def : Pat<(i64 nimm33:$val), (XORri (SETHIi (HIX22 $val)), (LOX10 $val))>,
      Requires<[Is64Bit]>;

// More possible patterns:
//
//   (sllx sethi, n)
//   (sllx simm13, n)
//
// 3 instrs:
//
//   (xor (sllx sethi), simm13)
//   (sllx (xor sethi, simm13))
//
// 4 instrs:
//
//   (or sethi, (sllx sethi))
//   (xnor sethi, (sllx sethi))
//
// 5 instrs:
//
//   (or (sllx sethi), (or sethi, simm13))
//   (xnor (sllx sethi), (or sethi, simm13))
//   (or (sllx sethi), (sllx sethi))
//   (xnor (sllx sethi), (sllx sethi))
//
// Worst case is 6 instrs:
//
//   (or (sllx (or sethi, simmm13)), (or sethi, simm13))

// Bits 42-63, same as assembler's %hh.
def HH22 : SDNodeXForm<imm, [{
  uint64_t Val = (N->getZExtValue() >> 42) & ((1u << 22) - 1);
  return CurDAG->getTargetConstant(Val, MVT::i32);
}]>;
// Bits 32-41, same as assembler's %hm.
def HM10 : SDNodeXForm<imm, [{
  uint64_t Val = (N->getZExtValue() >> 32) & ((1u << 10) - 1);
  return CurDAG->getTargetConstant(Val, MVT::i32);
}]>;
def : Pat<(i64 imm:$val),
          (ORrr (SLLXri (ORri (SETHIi (HH22 $val)), (HM10 $val)), (i32 32)),
                (ORri (SETHIi (HI22 $val)), (LO10 $val)))>,
      Requires<[Is64Bit]>;


//===----------------------------------------------------------------------===//
// 64-bit Integer Arithmetic and Logic.
//===----------------------------------------------------------------------===//

let Predicates = [Is64Bit] in {

// Register-register instructions.

def : Pat<(and i64:$a, i64:$b), (ANDrr $a, $b)>;
def : Pat<(or  i64:$a, i64:$b), (ORrr  $a, $b)>;
def : Pat<(xor i64:$a, i64:$b), (XORrr $a, $b)>;

def : Pat<(and i64:$a, (not i64:$b)), (ANDNrr $a, $b)>;
def : Pat<(or  i64:$a, (not i64:$b)), (ORNrr  $a, $b)>;
def : Pat<(xor i64:$a, (not i64:$b)), (XNORrr $a, $b)>;

def : Pat<(add i64:$a, i64:$b), (ADDrr $a, $b)>;
def : Pat<(sub i64:$a, i64:$b), (SUBrr $a, $b)>;

def : Pat<(SPcmpicc i64:$a, i64:$b), (CMPrr $a, $b)>;

def : Pat<(tlsadd i64:$a, i64:$b, tglobaltlsaddr:$sym),
          (TLS_ADDrr $a, $b, $sym)>;

// Register-immediate instructions.

def : Pat<(and i64:$a, (i64 simm13:$b)), (ANDri $a, (as_i32imm $b))>;
def : Pat<(or  i64:$a, (i64 simm13:$b)), (ORri  $a, (as_i32imm $b))>;
def : Pat<(xor i64:$a, (i64 simm13:$b)), (XORri $a, (as_i32imm $b))>;

def : Pat<(add i64:$a, (i64 simm13:$b)), (ADDri $a, (as_i32imm $b))>;
def : Pat<(sub i64:$a, (i64 simm13:$b)), (SUBri $a, (as_i32imm $b))>;

def : Pat<(SPcmpicc i64:$a, (i64 simm13:$b)), (CMPri $a, (as_i32imm $b))>;

def : Pat<(ctpop i64:$src), (POPCrr $src)>;

// "LEA" form of add
def LEAX_ADDri : F3_2<2, 0b000000,
                     (outs I64Regs:$dst), (ins MEMri:$addr),
                     "add ${addr:arith}, $dst",
                     [(set iPTR:$dst, ADDRri:$addr)]>;

} // Predicates = [Is64Bit]


//===----------------------------------------------------------------------===//
// 64-bit Integer Multiply and Divide.
//===----------------------------------------------------------------------===//

let Predicates = [Is64Bit] in {

def MULXrr : F3_1<2, 0b001001,
                  (outs I64Regs:$rd), (ins I64Regs:$rs1, I64Regs:$rs2),
                  "mulx $rs1, $rs2, $rd",
                  [(set i64:$rd, (mul i64:$rs1, i64:$rs2))]>;
def MULXri : F3_2<2, 0b001001,
                  (outs IntRegs:$rd), (ins IntRegs:$rs1, i64imm:$i),
                  "mulx $rs1, $i, $rd",
                  [(set i64:$rd, (mul i64:$rs1, (i64 simm13:$i)))]>;

// Division can trap.
let hasSideEffects = 1 in {
def SDIVXrr : F3_1<2, 0b101101,
                   (outs I64Regs:$rd), (ins I64Regs:$rs1, I64Regs:$rs2),
                   "sdivx $rs1, $rs2, $rd",
                   [(set i64:$rd, (sdiv i64:$rs1, i64:$rs2))]>;
def SDIVXri : F3_2<2, 0b101101,
                   (outs IntRegs:$rd), (ins IntRegs:$rs1, i64imm:$i),
                   "sdivx $rs1, $i, $rd",
                   [(set i64:$rd, (sdiv i64:$rs1, (i64 simm13:$i)))]>;

def UDIVXrr : F3_1<2, 0b001101,
                   (outs I64Regs:$rd), (ins I64Regs:$rs1, I64Regs:$rs2),
                   "udivx $rs1, $rs2, $rd",
                   [(set i64:$rd, (udiv i64:$rs1, i64:$rs2))]>;
def UDIVXri : F3_2<2, 0b001101,
                   (outs IntRegs:$rd), (ins IntRegs:$rs1, i64imm:$i),
                   "udivx $rs1, $i, $rd",
                   [(set i64:$rd, (udiv i64:$rs1, (i64 simm13:$i)))]>;
} // hasSideEffects = 1

} // Predicates = [Is64Bit]


//===----------------------------------------------------------------------===//
// 64-bit Loads and Stores.
//===----------------------------------------------------------------------===//
//
// All the 32-bit loads and stores are available. The extending loads are sign
// or zero-extending to 64 bits. The LDrr and LDri instructions load 32 bits
// zero-extended to i64. Their mnemonic is lduw in SPARC v9 (Load Unsigned
// Word).
//
// SPARC v9 adds 64-bit loads as well as a sign-extending ldsw i32 loads.

let Predicates = [Is64Bit] in {

// 64-bit loads.
def LDXrr  : F3_1<3, 0b001011,
                  (outs I64Regs:$dst), (ins MEMrr:$addr),
                  "ldx [$addr], $dst",
                  [(set i64:$dst, (load ADDRrr:$addr))]>;
def LDXri  : F3_2<3, 0b001011,
                  (outs I64Regs:$dst), (ins MEMri:$addr),
                  "ldx [$addr], $dst",
                  [(set i64:$dst, (load ADDRri:$addr))]>;
let mayLoad = 1 in
  def TLS_LDXrr : F3_1<3, 0b001011,
                       (outs IntRegs:$dst), (ins MEMrr:$addr, TLSSym:$sym),
                       "ldx [$addr], $dst, $sym",
                       [(set i64:$dst,
                           (tlsld ADDRrr:$addr, tglobaltlsaddr:$sym))]>;

// Extending loads to i64.
def : Pat<(i64 (zextloadi1 ADDRrr:$addr)), (LDUBrr ADDRrr:$addr)>;
def : Pat<(i64 (zextloadi1 ADDRri:$addr)), (LDUBri ADDRri:$addr)>;
def : Pat<(i64 (extloadi1 ADDRrr:$addr)), (LDUBrr ADDRrr:$addr)>;
def : Pat<(i64 (extloadi1 ADDRri:$addr)), (LDUBri ADDRri:$addr)>;

def : Pat<(i64 (zextloadi8 ADDRrr:$addr)), (LDUBrr ADDRrr:$addr)>;
def : Pat<(i64 (zextloadi8 ADDRri:$addr)), (LDUBri ADDRri:$addr)>;
def : Pat<(i64 (extloadi8 ADDRrr:$addr)),  (LDUBrr ADDRrr:$addr)>;
def : Pat<(i64 (extloadi8 ADDRri:$addr)),  (LDUBri ADDRri:$addr)>;
def : Pat<(i64 (sextloadi8 ADDRrr:$addr)), (LDSBrr ADDRrr:$addr)>;
def : Pat<(i64 (sextloadi8 ADDRri:$addr)), (LDSBri ADDRri:$addr)>;

def : Pat<(i64 (zextloadi16 ADDRrr:$addr)), (LDUHrr ADDRrr:$addr)>;
def : Pat<(i64 (zextloadi16 ADDRri:$addr)), (LDUHri ADDRri:$addr)>;
def : Pat<(i64 (extloadi16 ADDRrr:$addr)),  (LDUHrr ADDRrr:$addr)>;
def : Pat<(i64 (extloadi16 ADDRri:$addr)),  (LDUHri ADDRri:$addr)>;
def : Pat<(i64 (sextloadi16 ADDRrr:$addr)), (LDSHrr ADDRrr:$addr)>;
def : Pat<(i64 (sextloadi16 ADDRri:$addr)), (LDSHri ADDRri:$addr)>;

def : Pat<(i64 (zextloadi32 ADDRrr:$addr)), (LDrr ADDRrr:$addr)>;
def : Pat<(i64 (zextloadi32 ADDRri:$addr)), (LDri ADDRri:$addr)>;
def : Pat<(i64 (extloadi32 ADDRrr:$addr)),  (LDrr ADDRrr:$addr)>;
def : Pat<(i64 (extloadi32 ADDRri:$addr)),  (LDri ADDRri:$addr)>;

// Sign-extending load of i32 into i64 is a new SPARC v9 instruction.
def LDSWrr : F3_1<3, 0b001011,
                 (outs I64Regs:$dst), (ins MEMrr:$addr),
                 "ldsw [$addr], $dst",
                 [(set i64:$dst, (sextloadi32 ADDRrr:$addr))]>;
def LDSWri : F3_2<3, 0b001011,
                 (outs I64Regs:$dst), (ins MEMri:$addr),
                 "ldsw [$addr], $dst",
                 [(set i64:$dst, (sextloadi32 ADDRri:$addr))]>;

// 64-bit stores.
def STXrr  : F3_1<3, 0b001110,
                 (outs), (ins MEMrr:$addr, I64Regs:$src),
                 "stx $src, [$addr]",
                 [(store i64:$src, ADDRrr:$addr)]>;
def STXri  : F3_2<3, 0b001110,
                 (outs), (ins MEMri:$addr, I64Regs:$src),
                 "stx $src, [$addr]",
                 [(store i64:$src, ADDRri:$addr)]>;

// Truncating stores from i64 are identical to the i32 stores.
def : Pat<(truncstorei8  i64:$src, ADDRrr:$addr), (STBrr ADDRrr:$addr, $src)>;
def : Pat<(truncstorei8  i64:$src, ADDRri:$addr), (STBri ADDRri:$addr, $src)>;
def : Pat<(truncstorei16 i64:$src, ADDRrr:$addr), (STHrr ADDRrr:$addr, $src)>;
def : Pat<(truncstorei16 i64:$src, ADDRri:$addr), (STHri ADDRri:$addr, $src)>;
def : Pat<(truncstorei32 i64:$src, ADDRrr:$addr), (STrr  ADDRrr:$addr, $src)>;
def : Pat<(truncstorei32 i64:$src, ADDRri:$addr), (STri  ADDRri:$addr, $src)>;

// store 0, addr -> store %g0, addr
def : Pat<(store (i64 0), ADDRrr:$dst), (STXrr ADDRrr:$dst, (i64 G0))>;
def : Pat<(store (i64 0), ADDRri:$dst), (STXri ADDRri:$dst, (i64 G0))>;

} // Predicates = [Is64Bit]


//===----------------------------------------------------------------------===//
// 64-bit Conditionals.
//===----------------------------------------------------------------------===//
//
// Flag-setting instructions like subcc and addcc set both icc and xcc flags.
// The icc flags correspond to the 32-bit result, and the xcc are for the
// full 64-bit result.
//
// We reuse CMPICC SDNodes for compares, but use new BRXCC branch nodes for
// 64-bit compares. See LowerBR_CC.

let Predicates = [Is64Bit] in {

let Uses = [ICC] in
def BPXCC : BranchSP<(ins brtarget:$imm22, CCOp:$cond),
                     "b$cond %xcc, $imm22",
                     [(SPbrxcc bb:$imm22, imm:$cond)]>;

// Conditional moves on %xcc.
let Uses = [ICC], Constraints = "$f = $rd" in {
def MOVXCCrr : Pseudo<(outs IntRegs:$rd),
                      (ins IntRegs:$rs2, IntRegs:$f, CCOp:$cond),
                      "mov$cond %xcc, $rs2, $rd",
                      [(set i32:$rd,
                       (SPselectxcc i32:$rs2, i32:$f, imm:$cond))]>;
def MOVXCCri : Pseudo<(outs IntRegs:$rd),
                      (ins i32imm:$i, IntRegs:$f, CCOp:$cond),
                      "mov$cond %xcc, $i, $rd",
                      [(set i32:$rd,
                       (SPselectxcc simm11:$i, i32:$f, imm:$cond))]>;
def FMOVS_XCC : Pseudo<(outs FPRegs:$rd),
                      (ins FPRegs:$rs2, FPRegs:$f, CCOp:$cond),
                      "fmovs$cond %xcc, $rs2, $rd",
                      [(set f32:$rd,
                       (SPselectxcc f32:$rs2, f32:$f, imm:$cond))]>;
def FMOVD_XCC : Pseudo<(outs DFPRegs:$rd),
                      (ins DFPRegs:$rs2, DFPRegs:$f, CCOp:$cond),
                      "fmovd$cond %xcc, $rs2, $rd",
                      [(set f64:$rd,
                       (SPselectxcc f64:$rs2, f64:$f, imm:$cond))]>;
} // Uses, Constraints

//===----------------------------------------------------------------------===//
// 64-bit Floating Point Conversions.
//===----------------------------------------------------------------------===//

let Predicates = [Is64Bit] in {

def FXTOS : F3_3u<2, 0b110100, 0b010000100,
                 (outs FPRegs:$dst), (ins DFPRegs:$src),
                 "fxtos $src, $dst",
                 [(set FPRegs:$dst, (SPxtof DFPRegs:$src))]>;
def FXTOD : F3_3u<2, 0b110100, 0b010001000,
                 (outs DFPRegs:$dst), (ins DFPRegs:$src),
                 "fxtod $src, $dst",
                 [(set DFPRegs:$dst, (SPxtof DFPRegs:$src))]>;
def FXTOQ : F3_3u<2, 0b110100, 0b010001100,
                 (outs QFPRegs:$dst), (ins DFPRegs:$src),
                 "fxtoq $src, $dst",
                 [(set QFPRegs:$dst, (SPxtof DFPRegs:$src))]>,
                 Requires<[HasHardQuad]>;

def FSTOX : F3_3u<2, 0b110100, 0b010000001,
                 (outs DFPRegs:$dst), (ins FPRegs:$src),
                 "fstox $src, $dst",
                 [(set DFPRegs:$dst, (SPftox FPRegs:$src))]>;
def FDTOX : F3_3u<2, 0b110100, 0b010000010,
                 (outs DFPRegs:$dst), (ins DFPRegs:$src),
                 "fdtox $src, $dst",
                 [(set DFPRegs:$dst, (SPftox DFPRegs:$src))]>;
def FQTOX : F3_3u<2, 0b110100, 0b010000011,
                 (outs DFPRegs:$dst), (ins QFPRegs:$src),
                 "fqtox $src, $dst",
                 [(set DFPRegs:$dst, (SPftox QFPRegs:$src))]>,
                 Requires<[HasHardQuad]>;

} // Predicates = [Is64Bit]

def : Pat<(SPselectxcc i64:$t, i64:$f, imm:$cond),
          (MOVXCCrr $t, $f, imm:$cond)>;
def : Pat<(SPselectxcc (i64 simm11:$t), i64:$f, imm:$cond),
          (MOVXCCri (as_i32imm $t), $f, imm:$cond)>;

def : Pat<(SPselecticc i64:$t, i64:$f, imm:$cond),
          (MOVICCrr $t, $f, imm:$cond)>;
def : Pat<(SPselecticc (i64 simm11:$t), i64:$f, imm:$cond),
          (MOVICCri (as_i32imm $t), $f, imm:$cond)>;

def : Pat<(SPselectfcc i64:$t, i64:$f, imm:$cond),
          (MOVFCCrr $t, $f, imm:$cond)>;
def : Pat<(SPselectfcc (i64 simm11:$t), i64:$f, imm:$cond),
          (MOVFCCri (as_i32imm $t), $f, imm:$cond)>;

} // Predicates = [Is64Bit]
