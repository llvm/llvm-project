//==----- RISCVMacroFusion.td - Macro Fusion Definitions -----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the macro fusion predicators.

// Fuse LUI followed by ADDI or ADDIW:
//   rd = imm[31:0] which decomposes to
//   lui rd, imm[31:12]
//   addi(w) rd, rd, imm[11:0]
def TuneLUIADDIFusion
  : SimpleFusion<"lui-addi-fusion", "HasLUIADDIFusion",
                 "Enable LUI+ADDI macro fusion",
                 CheckOpcode<[LUI]>,
                 CheckOpcode<[ADDI, ADDIW]>>;

// Fuse AUIPC followed by ADDI:
//   auipc rd, imm20
//   addi rd, rd, imm12
def TuneAUIPCADDIFusion
  : SimpleFusion<"auipc-addi-fusion", "HasAUIPCADDIFusion",
                 "Enable AUIPC+ADDI macrofusion",
                 CheckOpcode<[AUIPC]>,
                 CheckOpcode<[ADDI]>>;

// Fuse zero extension of halfword:
//   slli rd, rs1, 48
//   srli rd, rd, 48
def TuneZExtHFusion
  : SimpleFusion<"zexth-fusion", "HasZExtHFusion",
                 "Enable SLLI+SRLI to be fused to zero extension of halfword",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 48>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 48>
                 ]>>;

// Fuse zero extension of word:
//   slli rd, rs1, 32
//   srli rd, rd, 32
def TuneZExtWFusion
  : SimpleFusion<"zextw-fusion", "HasZExtWFusion",
                 "Enable SLLI+SRLI to be fused to zero extension of word",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 32>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 32>
                 ]>>;

// Fuse shifted zero extension of word:
//   slli rd, rs1, 32
//   srli rd, rd, x
//   where 0 <= x < 32
def TuneShiftedZExtWFusion
  : SimpleFusion<"shifted-zextw-fusion", "HasShiftedZExtWFusion",
                 "Enable SLLI+SRLI to be fused when computing (shifted) word zero extension",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 32>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperandRange<2, 0, 31>
                 ]>>;

// Fuse load with add:
//   add rd, rs1, rs2
//   ld rd, 0(rd)
def TuneLDADDFusion
  : SimpleFusion<"ld-add-fusion", "HasLDADDFusion", "Enable LD+ADD macrofusion",
                 CheckOpcode<[ADD]>,
                 CheckAll<[
                   CheckOpcode<[LD]>,
                   CheckIsImmOperand<2>,
                   CheckImmOperand<2, 0>
                 ]>>;

// Get lower 16 bits:
// slliw r1, r0, 16
// srliw r1, r1, 16
def GetLower16BitsFusion
  : SimpleFusion<"get-lower-16bits-fusion", "HasGetLower16BitsFusion",
                 "Enable SLLIW+SRLIW to be fused to get lower 16 bits",
                 CheckAll<[
                   CheckOpcode<[SLLIW]>,
                   CheckImmOperand<2, 16>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLIW]>,
                   CheckImmOperand<2, 16>
                 ]>>;

// Sign-extend a 16-bit number:
// slliw r1, r0, 16
// sraiw r1, r1, 16
def SExtHFusion
  : SimpleFusion<"sign-extend-16bits-fusion","HasSExtHFusion",
                 "Enable SLLIW+SRAIW to be fused to sign-extend a 16-bit number",
                 CheckAll<[
                   CheckOpcode<[SLLIW]>,
                   CheckImmOperand<2, 16>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRAIW]>,
                   CheckImmOperand<2, 16>
                 ]>>;

// These should be covered by Zba extension.
// * shift left by one and add:
//   slli r1, r0, 1
//   add r1, r1, r2
// * shift left by two and add:
//   slli r1, r0, 2
//   add r1, r1, r2
// * shift left by three and add:
//   slli r1, r0, 3
//   add r1, r1, r2
let IsCommutable = 1 in
def ShiftNAddFusion
  : SimpleFusion<"shift-n-add-fusion", "HasShiftNAddFusion",
                 "Enable SLLI+ADD to be fused to shift left by 1/2/3 and add",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckAny<[CheckImmOperand<2, 1>,
                             CheckImmOperand<2, 2>,
                             CheckImmOperand<2, 3>]>
                 ]>,
                 CheckOpcode<[ADD]>>;

// * Shift zero-extended word left by 1:
//   slli r1, r0, 32
//   srli r1, r0, 31
// * Shift zero-extended word left by 2:
//   slli r1, r0, 32
//   srli r1, r0, 30
// * Shift zero-extended word left by 3:
//   slli r1, r0, 32
//   srli r1, r0, 29
def ShiftZExtByNFusion
  : SimpleFusion<"shift-zext-by-n-fusion", "HasShiftZExtByNFusion",
                 "Enable SLLI+SRLI to be fused to shift zero-extended word left by 1/2/3",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckImmOperand<2, 32>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckAny<[CheckImmOperand<2, 29>,
                             CheckImmOperand<2, 30>,
                             CheckImmOperand<2, 31>]>
                 ]>>;

// Get the second byte:
// srli r1, r0, 8
// andi r1, r1, 255
def GetSecondByteFusion
  : SimpleFusion<"get-second-byte-fusion", "HasGetSecondByteFusion",
                 "Enable SRLI+ANDI to be fused to get the second byte",
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckImmOperand<2, 8>
                 ]>,
                 CheckAll<[
                   CheckOpcode<[ANDI]>,
                   CheckImmOperand<2, 255>
                 ]>>;

// Shift left by four and add:
// slli r1, r0, 4
// add r1, r1, r2
let IsCommutable = 1 in
def ShiftLeft4AddFusion
  : SimpleFusion<"shift-left-four-add-fusion", "HasShiftLeft4AddFusion",
                 "Enable SLLI+ADD to be fused to shift left by four and add",
                 CheckAll<[
                   CheckOpcode<[SLLI]>,
                   CheckImmOperand<2, 4>
                 ]>,
                 CheckOpcode<[ADD]>>;

// * Shift right by 29 and add:
//   srli r1, r0, 29
//   add r1, r1, r2
// * Shift right by 30 and add:
//   srli r1, r0, 30
//   add r1, r1, r2
// * Shift right by 31 and add:
//   srli r1, r0, 31
//   add r1, r1, r2
// * Shift right by 32 and add:
//   srli r1, r0, 32
//   add r1, r1, r2
let IsCommutable = 1 in
def ShiftRightNAddFusion
  : SimpleFusion<"shift-right-n-add-fusion", "HasShiftRightNAddFusion",
                 "Enable SRLI+add to be fused to shift right by 29/30/31/32 and add",
                 CheckAll<[
                   CheckOpcode<[SRLI]>,
                   CheckAny<[CheckImmOperand<2, 29>,
                             CheckImmOperand<2, 30>,
                             CheckImmOperand<2, 31>,
                             CheckImmOperand<2, 32>]>
                 ]>,
                 CheckOpcode<[ADD]>>;

// Add one if odd, otherwise unchanged:
// andi r1, r0, 1
// add r1, r1, r2
// Add one if odd (in word format), otherwise unchanged:
// andi r1, r0, 1
// addw r1, r1, r2
let IsCommutable = 1 in
def AddOneIfOddFusion
  : SimpleFusion<"add-one-if-odd-fusion", "HasAddOneIfOddFusion",
                 "Enable ANDI+ADDW to be fused to add one if odd",
                 CheckAll<[
                   CheckOpcode<[ANDI]>,
                   CheckImmOperand<2, 1>
                 ]>,
                 CheckOpcode<[ADD, ADDW]>>;

// * Add word and extract its lower 1 bit:
//   andw r1, r1, r0
//   andi r1, r1, 1
// * Add word and extract its lower 8 bits:
//   andw r1, r1, r0
//   andi r1, r1, 255
def AddAndExtractNBitsFusion
  : SimpleFusion<"add-and-extract-n-bits-fusion", "HasAddAndExtractNBitsFusion",
                 "Enable ADDW+ANDI to be fused to get lower 16 bits",
                 CheckOpcode<[ADDW]>,
                 CheckAll<[
                   CheckOpcode<[ANDI]>,
                   CheckAny<[CheckImmOperand<2, 1>,
                             CheckImmOperand<2, 255>]>
                 ]>>;

// * Add word and zext.h:
//   andw r1, r1, r0
//   zext.h r1, r1
// * Add word and sext.h:
//   andw r1, r1, r0
//   sext.h r1, r1
def AddwAndExtFusion
  : SimpleFusion<"addw-and-ext-fusion", "HasAddwAndExtFusion",
                 "Enable ADDW+ZEXT_H/SEXT_H to be fused",
                 CheckOpcode<[ADDW]>,
                 CheckOpcode<[ZEXT_H_RV32, ZEXT_H_RV64, SEXT_H]>>;

// Logic operation and extract its LSB:
// <logic op> r1, r1, r0
// andi r1, r1, 1
def LogicOpAndExtractLSBFusion
  : SimpleFusion<"logic-op-and-extract-lsb-fusion", "HasLogicOpAndExtractLSBFusion",
                 "Enable AND/OR/XOR/ANDI/ORI/XORI/ORC_B+ANDI to be fused to logic operation and extract its LSB",
                 CheckOpcode<[AND, OR, XOR, ANDI, ORI, XORI, ORC_B]>,
                 CheckAll<[
                   CheckOpcode<[ANDI]>,
                   CheckImmOperand<2, 1>
                 ]>>;

// Logic operation and extract its lower 16 bits:
// <logic op> r1, r1, r0
// zext.h r1, r1
def LogicOpAndExtractLow16BitsFusion
  : SimpleFusion<"logic-op-and-extract-low-16bits-fusion", "HasLogicOpAndExtractLow16BitsFusion",
                 "Enable AND/OR/XOR/ANDI/ORI/XORI/ORC_B+ZEXT_H to be fused to logic operation and extract its lower 16 bits",
                 CheckOpcode<[AND, OR, XOR, ANDI, ORI, XORI, ORC_B]>,
                 CheckOpcode<[ZEXT_H_RV32, ZEXT_H_RV64]>>;

// OR(Cat(src1(63, 8), 0.U(8.W)), src2):
// andi r1, r0, -256
// or r1, r1, r2
let IsCommutable = 1 in
def OrCatFusion
  : SimpleFusion<"or-cat-fusion", "HasOrCatFusion",
                 "Enable SLLIW+SRLIW to be fused to get lower 16 bits",
                 CheckAll<[
                   CheckOpcode<[ANDI]>,
                   CheckImmOperand<2, -256>
                 ]>,
                 CheckOpcode<[OR]>>;

// Multiply 7-bit data with 32-bit data:
// andi r1, r0, 127
// mulw r1, r1, r2
let IsCommutable = 1 in
def Mul7BitsWith32BitsFusion
  : SimpleFusion<"mul-7bits-with-32bit-fusion", "HasMul7BitsWith32BitsFusion",
                 "Enable ANDI+MULW to be fused to multiply 7-bit data with 32-bit data",
                 CheckAll<[
                   CheckOpcode<[ANDI]>,
                   CheckImmOperand<2, 127>
                 ]>,
                 CheckOpcode<[MULW]>>;
