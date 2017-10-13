//===- TargetGlobalISel.td - Common code for GlobalISel ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent interfaces used to support
// SelectionDAG instruction selection patterns (specified in
// TargetSelectionDAG.td) when generating GlobalISel instruction selectors.
//
// This is intended as a compatibility layer, to enable reuse of target
// descriptions written for SelectionDAG without requiring explicit GlobalISel
// support.  It will eventually supersede SelectionDAG patterns.
//
//===----------------------------------------------------------------------===//

// Declare that a generic Instruction is 'equivalent' to an SDNode, that is,
// SelectionDAG patterns involving the SDNode can be transformed to match the
// Instruction instead.
class GINodeEquiv<Instruction i, SDNode node> {
  Instruction I = i;
  SDNode Node = node;
}

// These are defined in the same order as the G_* instructions.
def : GINodeEquiv<G_ANYEXT, anyext>;
def : GINodeEquiv<G_SEXT, sext>;
def : GINodeEquiv<G_ZEXT, zext>;
def : GINodeEquiv<G_TRUNC, trunc>;
def : GINodeEquiv<G_BITCAST, bitconvert>;
// G_INTTOPTR - SelectionDAG has no equivalent.
// G_PTRTOINT - SelectionDAG has no equivalent.
def : GINodeEquiv<G_CONSTANT, imm>;
def : GINodeEquiv<G_FCONSTANT, fpimm>;
def : GINodeEquiv<G_ADD, add>;
def : GINodeEquiv<G_SUB, sub>;
def : GINodeEquiv<G_MUL, mul>;
def : GINodeEquiv<G_SDIV, sdiv>;
def : GINodeEquiv<G_UDIV, udiv>;
def : GINodeEquiv<G_SREM, srem>;
def : GINodeEquiv<G_UREM, urem>;
def : GINodeEquiv<G_AND, and>;
def : GINodeEquiv<G_OR, or>;
def : GINodeEquiv<G_XOR, xor>;
def : GINodeEquiv<G_SHL, shl>;
def : GINodeEquiv<G_LSHR, srl>;
def : GINodeEquiv<G_ASHR, sra>;
def : GINodeEquiv<G_SELECT, select>;
def : GINodeEquiv<G_FNEG, fneg>;
def : GINodeEquiv<G_FPEXT, fpextend>;
def : GINodeEquiv<G_FPTRUNC, fpround>;
def : GINodeEquiv<G_FPTOSI, fp_to_sint>;
def : GINodeEquiv<G_FPTOUI, fp_to_uint>;
def : GINodeEquiv<G_SITOFP, sint_to_fp>;
def : GINodeEquiv<G_UITOFP, uint_to_fp>;
def : GINodeEquiv<G_FADD, fadd>;
def : GINodeEquiv<G_FSUB, fsub>;
def : GINodeEquiv<G_FMA, fma>;
def : GINodeEquiv<G_FMUL, fmul>;
def : GINodeEquiv<G_FDIV, fdiv>;
def : GINodeEquiv<G_FREM, frem>;
def : GINodeEquiv<G_FPOW, fpow>;
def : GINodeEquiv<G_FEXP2, fexp2>;
def : GINodeEquiv<G_FLOG2, flog2>;
def : GINodeEquiv<G_INTRINSIC, intrinsic_wo_chain>;
// ISD::INTRINSIC_VOID can also be handled with G_INTRINSIC_W_SIDE_EFFECTS.
def : GINodeEquiv<G_INTRINSIC_W_SIDE_EFFECTS, intrinsic_void>;
def : GINodeEquiv<G_INTRINSIC_W_SIDE_EFFECTS, intrinsic_w_chain>;
def : GINodeEquiv<G_BR, br>;
def : GINodeEquiv<G_BSWAP, bswap>;

// Specifies the GlobalISel equivalents for SelectionDAG's ComplexPattern.
// Should be used on defs that subclass GIComplexOperandMatcher<>.
class GIComplexPatternEquiv<ComplexPattern seldag> {
  ComplexPattern SelDAGEquivalent = seldag;
}
