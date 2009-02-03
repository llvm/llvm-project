//===- SPUNodes.td - Specialized SelectionDAG nodes used for CellSPU ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Type profiles and SelectionDAG nodes used by CellSPU
//
//===----------------------------------------------------------------------===//

// Type profile for a call sequence
def SDT_SPUCallSeq : SDTypeProfile<0, 1, [ SDTCisVT<0, i32> ]>;

// SPU_GenControl: Type profile for generating control words for insertions
def SPU_GenControl : SDTypeProfile<1, 1, []>;
def SPUshufmask    : SDNode<"SPUISD::SHUFFLE_MASK", SPU_GenControl, []>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_SPUCallSeq,
                           [SDNPHasChain, SDNPOutFlag]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_SPUCallSeq,
                           [SDNPHasChain, SDNPOutFlag]>;
//===----------------------------------------------------------------------===//
// Operand constraints:
//===----------------------------------------------------------------------===//

def SDT_SPUCall   : SDTypeProfile<0, -1, [SDTCisInt<0>]>;
def SPUcall       : SDNode<"SPUISD::CALL", SDT_SPUCall,
                           [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

// Operand type constraints for vector shuffle/permute operations
def SDT_SPUshuffle   : SDTypeProfile<1, 3, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>
]>;

// Vector binary operator type constraints (needs a further constraint to
// ensure that operand 0 is a vector...):

def SPUVecBinop: SDTypeProfile<1, 2, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>
]>;

// Trinary operators, e.g., addx, carry generate
def SPUIntTrinaryOp : SDTypeProfile<1, 3, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>, SDTCisInt<0>
]>;

// SELECT_MASK type constraints: There are several variations for the various
// vector types (this avoids having to bit_convert all over the place.)
def SPUselmask_type: SDTypeProfile<1, 1, [
  SDTCisInt<1>
]>;

// SELB type constraints:
def SPUselb_type: SDTypeProfile<1, 3, [
  SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, SDTCisSameAs<0, 3> ]>;

// SPU Vector shift pseudo-instruction type constraints
def SPUvecshift_type: SDTypeProfile<1, 2, [
  SDTCisSameAs<0, 1>, SDTCisInt<2>]>;

// "marker" type for i64 operators that need a shuffle mask
// (i.e., uses cg or bg or another instruction that needs to
// use shufb to get things in the right place.)
// Op0: The result
// Op1, 2: LHS, RHS
// Op3: Carry-generate shuffle mask

def SPUmarker_type : SDTypeProfile<1, 3, [
  SDTCisInt<0>, SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2> ]>;

//===----------------------------------------------------------------------===//
// Synthetic/pseudo-instructions
//===----------------------------------------------------------------------===//

// SPU CNTB:
def SPUcntb : SDNode<"SPUISD::CNTB", SDTIntUnaryOp>;

// SPU vector shuffle node, matched by the SPUISD::SHUFB enum (see
// SPUISelLowering.h):
def SPUshuffle: SDNode<"SPUISD::SHUFB", SDT_SPUshuffle, []>;

// Shift left quadword by bits and bytes
def SPUshlquad_l_bits: SDNode<"SPUISD::SHLQUAD_L_BITS", SPUvecshift_type, []>;
def SPUshlquad_l_bytes: SDNode<"SPUISD::SHLQUAD_L_BYTES", SPUvecshift_type, []>;

// Vector shifts (ISD::SHL,SRL,SRA are for _integers_ only):
def SPUvec_shl: SDNode<"SPUISD::VEC_SHL", SPUvecshift_type, []>;
def SPUvec_srl: SDNode<"SPUISD::VEC_SRL", SPUvecshift_type, []>;
def SPUvec_sra: SDNode<"SPUISD::VEC_SRA", SPUvecshift_type, []>;

def SPUvec_rotl: SDNode<"SPUISD::VEC_ROTL", SPUvecshift_type, []>;
def SPUvec_rotr: SDNode<"SPUISD::VEC_ROTR", SPUvecshift_type, []>;

// Vector rotate left, bits shifted out of the left are rotated in on the right
def SPUrotbytes_left: SDNode<"SPUISD::ROTBYTES_LEFT",
                             SPUvecshift_type, []>;

// Vector rotate left by bytes, but the count is given in bits and the SPU
// internally converts it to bytes (saves an instruction to mask off lower
// three bits)
def SPUrotbytes_left_bits : SDNode<"SPUISD::ROTBYTES_LEFT_BITS",
                                   SPUvecshift_type>;

// SPU form select mask for bytes, immediate
def SPUselmask: SDNode<"SPUISD::SELECT_MASK", SPUselmask_type, []>;

// SPU select bits instruction
def SPUselb: SDNode<"SPUISD::SELB", SPUselb_type, []>;

def SDTprefslot2vec: SDTypeProfile<1, 1, []>;
def SPUprefslot2vec: SDNode<"SPUISD::PREFSLOT2VEC", SDTprefslot2vec, []>;

def SPU_vec_demote   : SDTypeProfile<1, 1, []>;
def SPUvec2prefslot: SDNode<"SPUISD::VEC2PREFSLOT", SPU_vec_demote, []>;

// Address high and low components, used for [r+r] type addressing
def SPUhi : SDNode<"SPUISD::Hi", SDTIntBinOp, []>;
def SPUlo : SDNode<"SPUISD::Lo", SDTIntBinOp, []>;

// PC-relative address
def SPUpcrel : SDNode<"SPUISD::PCRelAddr", SDTIntBinOp, []>;

// A-Form local store addresses
def SPUaform : SDNode<"SPUISD::AFormAddr", SDTIntBinOp, []>;

// Indirect [D-Form "imm($reg)" and X-Form "$reg($reg)"] addresses
def SPUindirect : SDNode<"SPUISD::IndirectAddr", SDTIntBinOp, []>;

// i64 markers: supplies extra operands used to generate the i64 operator
// instruction sequences
def SPUadd64 : SDNode<"SPUISD::ADD64_MARKER", SPUmarker_type, []>;
def SPUsub64 : SDNode<"SPUISD::SUB64_MARKER", SPUmarker_type, []>;
def SPUmul64 : SDNode<"SPUISD::MUL64_MARKER", SPUmarker_type, []>;

//===----------------------------------------------------------------------===//
// Constraints: (taken from PPCInstrInfo.td)
//===----------------------------------------------------------------------===//

class RegConstraint<string C> {
  string Constraints = C;
}

class NoEncode<string E> {
  string DisableEncoding = E;
}

//===----------------------------------------------------------------------===//
// Return (flag isn't quite what it means: the operations are flagged so that
// instruction scheduling doesn't disassociate them.)
//===----------------------------------------------------------------------===//

def retflag     : SDNode<"SPUISD::RET_FLAG", SDTNone,
                         [SDNPHasChain, SDNPOptInFlag]>;
