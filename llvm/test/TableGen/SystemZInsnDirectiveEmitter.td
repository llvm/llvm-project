// RUN: llvm-tblgen -gen-insn-directive-match-table %p/../../lib/Target/SystemZ/SystemZ.td \
// RUN:   -I %p/../../lib/Target/SystemZ -I %p/../../include | FileCheck %s

// Verify that the SystemZ .insn directive match table is generated correctly.
// Checks cover:
//   - Representative entries from several format families to confirm format
//     name, opcode symbol, operand count, and MCK kinds.
//   - X-imm (union signed/unsigned) operands are mapped to the right MCK.
//   - Entries are sorted by format.
// Checks must appear in output order (FileCheck is sequential).

// Single-operand format.
// CHECK: {"e", SystemZ::InsnE, 1, {MCK_U16Imm}},

// I format: opcode + 1 immediate.
// CHECK: {"i", SystemZ::InsnI, 2, {MCK_U16Imm, MCK_U8Imm}},

// MII: opcode + mask + two PC-relative immediates of different widths.
// CHECK: {"mii", SystemZ::InsnMII, 4, {MCK_U48Imm, MCK_U4Imm, MCK_PCRel12, MCK_PCRel24}},

// RI alias uses signed 16-bit immediate; RI-a subformat uses the union X16Imm.
// CHECK: {"ri", SystemZ::InsnRI, 3, {MCK_U32Imm, MCK_AnyReg, MCK_S16Imm}},
// CHECK: {"ri_a", SystemZ::InsnRIa, 3, {MCK_U32Imm, MCK_AnyReg, MCK_X16Imm}},

// RIE-c uses X8Imm (union of signed/unsigned 8-bit) for its immediate field.
// CHECK: {"rie_c", SystemZ::InsnRIEc, 5, {MCK_U48Imm, MCK_AnyReg, MCK_X8Imm, MCK_U4Imm, MCK_PCRel16}},

// RIL-a uses X32Imm (union of signed/unsigned 32-bit).
// CHECK: {"ril_a", SystemZ::InsnRILa, 3, {MCK_U48Imm, MCK_AnyReg, MCK_X32Imm}},

// RSL-a uses a BDL address with 4-bit length.
// CHECK: {"rsl_a", SystemZ::InsnRSLa, 2, {MCK_U48Imm, MCK_BDLAddr64Disp12Len4}},

// RSL-b uses a BDL address with 8-bit length.
// CHECK: {"rsl_b", SystemZ::InsnRSLb, 4, {MCK_U48Imm, MCK_AnyReg, MCK_BDLAddr64Disp12Len8, MCK_U4Imm}},

// SI, SIL and SIY use X-imm for the immediate field.
// CHECK: {"si", SystemZ::InsnSI, 3, {MCK_U32Imm, MCK_BDAddr64Disp12, MCK_X8Imm}},
// CHECK: {"sil", SystemZ::InsnSIL, 3, {MCK_U48Imm, MCK_BDAddr64Disp12, MCK_X16Imm}},
// CHECK: {"siy", SystemZ::InsnSIY, 3, {MCK_U48Imm, MCK_BDAddr64Disp20, MCK_X8Imm}},

// SMI (branch-prediction): opcode + mask + PC-relative 16 + BD address.
// CHECK: {"smi", SystemZ::InsnSMI, 4, {MCK_U48Imm, MCK_U4Imm, MCK_PCRel16, MCK_BDAddr64Disp12}},

// SS-a uses an 8-bit-length BDL for the first operand.
// CHECK: {"ss_a", SystemZ::InsnSSa, 3, {MCK_U48Imm, MCK_BDLAddr64Disp12Len8, MCK_BDAddr64Disp12}},

// VRI-a uses X16Imm.
// CHECK: {"vri_a", SystemZ::InsnVRIa, 4, {MCK_U48Imm, MCK_VR128, MCK_X16Imm, MCK_U4Imm}},

// Verify format-then-opcode sort order: vrr_j comes before vrr_k.
// CHECK: {"vrr_j",
// CHECK: {"vrr_k",

// VRS-d (z14): opcode + vector + BD address + GP register.
// CHECK: {"vrs_d", SystemZ::InsnVRSd, 4, {MCK_U48Imm, MCK_VR128, MCK_BDAddr64Disp12, MCK_AnyReg}},
