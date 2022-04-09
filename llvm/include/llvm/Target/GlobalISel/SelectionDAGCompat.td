//===- TargetGlobalISel.td - Common code for GlobalISel ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

  // SelectionDAG has separate nodes for atomic and non-atomic memory operations
  // (ISD::LOAD, ISD::ATOMIC_LOAD, ISD::STORE, ISD::ATOMIC_STORE) but GlobalISel
  // stores this information in the MachineMemoryOperand.
  bit CheckMMOIsNonAtomic = false;
  bit CheckMMOIsAtomic = false;

  // SelectionDAG has one node for all loads and uses predicates to
  // differentiate them. GlobalISel on the other hand uses separate opcodes.
  // When this is true, the resulting opcode is G_LOAD/G_SEXTLOAD/G_ZEXTLOAD
  // depending on the predicates on the node.
  Instruction IfSignExtend = ?;
  Instruction IfZeroExtend = ?;

  // SelectionDAG has one setcc for all compares. This differentiates
  // for G_ICMP and G_FCMP.
  Instruction IfFloatingPoint = ?;
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
// timm must not be materialized and therefore has no GlobalISel equivalent
def : GINodeEquiv<G_FCONSTANT, fpimm>;
def : GINodeEquiv<G_IMPLICIT_DEF, undef>;
def : GINodeEquiv<G_FRAME_INDEX, frameindex>;
def : GINodeEquiv<G_BLOCK_ADDR, blockaddress>;
def : GINodeEquiv<G_ADD, add>;
def : GINodeEquiv<G_SUB, sub>;
def : GINodeEquiv<G_MUL, mul>;
def : GINodeEquiv<G_UMULH, mulhu>;
def : GINodeEquiv<G_SMULH, mulhs>;
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
def : GINodeEquiv<G_SADDSAT, saddsat>;
def : GINodeEquiv<G_UADDSAT, uaddsat>;
def : GINodeEquiv<G_SSUBSAT, ssubsat>;
def : GINodeEquiv<G_USUBSAT, usubsat>;
def : GINodeEquiv<G_SSHLSAT, sshlsat>;
def : GINodeEquiv<G_USHLSAT, ushlsat>;
def : GINodeEquiv<G_SMULFIX, smulfix>;
def : GINodeEquiv<G_UMULFIX, umulfix>;
def : GINodeEquiv<G_SMULFIXSAT, smulfixsat>;
def : GINodeEquiv<G_UMULFIXSAT, umulfixsat>;
def : GINodeEquiv<G_SDIVFIX, sdivfix>;
def : GINodeEquiv<G_UDIVFIX, udivfix>;
def : GINodeEquiv<G_SDIVFIXSAT, sdivfixsat>;
def : GINodeEquiv<G_UDIVFIXSAT, udivfixsat>;
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
def : GINodeEquiv<G_FMAD, fmad>;
def : GINodeEquiv<G_FMUL, fmul>;
def : GINodeEquiv<G_FDIV, fdiv>;
def : GINodeEquiv<G_FREM, frem>;
def : GINodeEquiv<G_FPOW, fpow>;
def : GINodeEquiv<G_FEXP2, fexp2>;
def : GINodeEquiv<G_FLOG2, flog2>;
def : GINodeEquiv<G_FCANONICALIZE, fcanonicalize>;
def : GINodeEquiv<G_INTRINSIC, intrinsic_wo_chain>;
// ISD::INTRINSIC_VOID can also be handled with G_INTRINSIC_W_SIDE_EFFECTS.
def : GINodeEquiv<G_INTRINSIC_W_SIDE_EFFECTS, intrinsic_void>;
def : GINodeEquiv<G_INTRINSIC_W_SIDE_EFFECTS, intrinsic_w_chain>;
def : GINodeEquiv<G_BR, br>;
def : GINodeEquiv<G_BSWAP, bswap>;
def : GINodeEquiv<G_BITREVERSE, bitreverse>;
def : GINodeEquiv<G_FSHL, fshl>;
def : GINodeEquiv<G_FSHR, fshr>;
def : GINodeEquiv<G_CTLZ, ctlz>;
def : GINodeEquiv<G_CTTZ, cttz>;
def : GINodeEquiv<G_CTLZ_ZERO_UNDEF, ctlz_zero_undef>;
def : GINodeEquiv<G_CTTZ_ZERO_UNDEF, cttz_zero_undef>;
def : GINodeEquiv<G_CTPOP, ctpop>;
def : GINodeEquiv<G_EXTRACT_VECTOR_ELT, extractelt>;
def : GINodeEquiv<G_CONCAT_VECTORS, concat_vectors>;
def : GINodeEquiv<G_BUILD_VECTOR, build_vector>;
def : GINodeEquiv<G_FCEIL, fceil>;
def : GINodeEquiv<G_FCOS, fcos>;
def : GINodeEquiv<G_FSIN, fsin>;
def : GINodeEquiv<G_FABS, fabs>;
def : GINodeEquiv<G_FSQRT, fsqrt>;
def : GINodeEquiv<G_FFLOOR, ffloor>;
def : GINodeEquiv<G_FRINT, frint>;
def : GINodeEquiv<G_FNEARBYINT, fnearbyint>;
def : GINodeEquiv<G_INTRINSIC_TRUNC, ftrunc>;
def : GINodeEquiv<G_INTRINSIC_ROUND, fround>;
def : GINodeEquiv<G_INTRINSIC_LRINT, lrint>;
def : GINodeEquiv<G_FCOPYSIGN, fcopysign>;
def : GINodeEquiv<G_SMIN, smin>;
def : GINodeEquiv<G_SMAX, smax>;
def : GINodeEquiv<G_UMIN, umin>;
def : GINodeEquiv<G_UMAX, umax>;
def : GINodeEquiv<G_ABS, abs>;
def : GINodeEquiv<G_FMINNUM, fminnum>;
def : GINodeEquiv<G_FMAXNUM, fmaxnum>;
def : GINodeEquiv<G_FMINNUM_IEEE, fminnum_ieee>;
def : GINodeEquiv<G_FMAXNUM_IEEE, fmaxnum_ieee>;
def : GINodeEquiv<G_READCYCLECOUNTER, readcyclecounter>;
def : GINodeEquiv<G_ROTR, rotr>;
def : GINodeEquiv<G_ROTL, rotl>;
def : GINodeEquiv<G_LROUND, lround>;
def : GINodeEquiv<G_LLROUND, llround>;

def : GINodeEquiv<G_STRICT_FADD, strict_fadd>;
def : GINodeEquiv<G_STRICT_FSUB, strict_fsub>;
def : GINodeEquiv<G_STRICT_FMUL, strict_fmul>;
def : GINodeEquiv<G_STRICT_FDIV, strict_fdiv>;
def : GINodeEquiv<G_STRICT_FREM, strict_frem>;
def : GINodeEquiv<G_STRICT_FMA, strict_fma>;
def : GINodeEquiv<G_STRICT_FSQRT, strict_fsqrt>;

// Broadly speaking G_LOAD is equivalent to ISD::LOAD but there are some
// complications that tablegen must take care of. For example, Predicates such
// as isSignExtLoad require that this is not a perfect 1:1 mapping since a
// sign-extending load is (G_SEXTLOAD x) in GlobalISel. Additionally,
// G_LOAD handles both atomic and non-atomic loads where as SelectionDAG had
// separate nodes for them. This GINodeEquiv maps the non-atomic loads to
// G_LOAD with a non-atomic MachineMemOperand.
def : GINodeEquiv<G_LOAD, ld> {
  let CheckMMOIsNonAtomic = true;
  let IfSignExtend = G_SEXTLOAD;
  let IfZeroExtend = G_ZEXTLOAD;
}

def : GINodeEquiv<G_ICMP, setcc> {
  let IfFloatingPoint = G_FCMP;
}

// Broadly speaking G_STORE is equivalent to ISD::STORE but there are some
// complications that tablegen must take care of. For example, predicates such
// as isTruncStore require that this is not a perfect 1:1 mapping since a
// truncating store is (G_STORE (G_TRUNCATE x)) in GlobalISel. Additionally,
// G_STORE handles both atomic and non-atomic stores where as SelectionDAG had
// separate nodes for them. This GINodeEquiv maps the non-atomic stores to
// G_STORE with a non-atomic MachineMemOperand.
def : GINodeEquiv<G_STORE, st> { let CheckMMOIsNonAtomic = true; }

def : GINodeEquiv<G_LOAD, atomic_load> {
  let CheckMMOIsNonAtomic = false;
  let CheckMMOIsAtomic = true;
  let IfSignExtend = G_SEXTLOAD;
  let IfZeroExtend = G_ZEXTLOAD;
}

// Operands are swapped for atomic_store vs. regular store
def : GINodeEquiv<G_STORE, atomic_store> {
  let CheckMMOIsNonAtomic = false;
  let CheckMMOIsAtomic = true;
}

def : GINodeEquiv<G_ATOMIC_CMPXCHG, atomic_cmp_swap>;
def : GINodeEquiv<G_ATOMICRMW_XCHG, atomic_swap>;
def : GINodeEquiv<G_ATOMICRMW_ADD, atomic_load_add>;
def : GINodeEquiv<G_ATOMICRMW_SUB, atomic_load_sub>;
def : GINodeEquiv<G_ATOMICRMW_AND, atomic_load_and>;
def : GINodeEquiv<G_ATOMICRMW_NAND, atomic_load_nand>;
def : GINodeEquiv<G_ATOMICRMW_OR, atomic_load_or>;
def : GINodeEquiv<G_ATOMICRMW_XOR, atomic_load_xor>;
def : GINodeEquiv<G_ATOMICRMW_MIN, atomic_load_min>;
def : GINodeEquiv<G_ATOMICRMW_MAX, atomic_load_max>;
def : GINodeEquiv<G_ATOMICRMW_UMIN, atomic_load_umin>;
def : GINodeEquiv<G_ATOMICRMW_UMAX, atomic_load_umax>;
def : GINodeEquiv<G_ATOMICRMW_FADD, atomic_load_fadd>;
def : GINodeEquiv<G_ATOMICRMW_FSUB, atomic_load_fsub>;
def : GINodeEquiv<G_ATOMICRMW_FMAX, atomic_load_fmax>;
def : GINodeEquiv<G_ATOMICRMW_FMIN, atomic_load_fmin>;
def : GINodeEquiv<G_FENCE, atomic_fence>;

// Specifies the GlobalISel equivalents for SelectionDAG's ComplexPattern.
// Should be used on defs that subclass GIComplexOperandMatcher<>.
class GIComplexPatternEquiv<ComplexPattern seldag> {
  ComplexPattern SelDAGEquivalent = seldag;
}

// Specifies the GlobalISel equivalents for SelectionDAG's SDNodeXForm.
// Should be used on defs that subclass GICustomOperandRenderer<>.
class GISDNodeXFormEquiv<SDNodeXForm seldag> {
  SDNodeXForm SelDAGEquivalent = seldag;
}
