//===-------------- VVPInstrInfo.td - VVP_* SDNode patterns ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the VE Vector Predicated SDNodes (VVP SDNodes).  VVP
// SDNodes are an intermediate isel layer between the vector SDNodes emitted by
// LLVM and the actual VE vector instructions. For example:
//
//  ADD(x,y)   -->   VVP_ADD(x,y,mask,evl)   -->   VADDSWSXrvml(x,y,mask,evl)
//     ^                      ^                            ^
//  The standard     The VVP layer SDNode.        The VE vector instruction.
//  SDNode.
//
// TODO explain how VVP nodes relate to VP SDNodes once VP ISel is uptream.
//===----------------------------------------------------------------------===//

// Binary Operators {

// BinaryOp(x,y,mask,vl)
def SDTIntBinOpVVP : SDTypeProfile<1, 4, [     // vp_add, vp_and, etc.
  SDTCisSameAs<0, 1>,
  SDTCisSameAs<0, 2>,
  SDTCisInt<0>,
  SDTCisSameNumEltsAs<0, 3>,
  IsVLVT<4>
]>;

// BinaryFPOp(x,y,mask,vl)
def SDTFPBinOpVVP : SDTypeProfile<1, 4, [      // vvp_fadd, etc.
  SDTCisSameAs<0, 1>,
  SDTCisSameAs<0, 2>,
  SDTCisFP<0>,
  SDTCisInt<3>,
  SDTCisSameNumEltsAs<0, 3>,
  IsVLVT<4>
]>;

// Select(OnTrue, OnFalse, SelMask, vl)
def SDTSelectVVP : SDTypeProfile<1, 4, [       // vp_select, vp_merge
  SDTCisVec<0>,
  SDTCisSameNumEltsAs<0, 3>,
  SDTCisSameAs<0, 1>,
  SDTCisSameAs<1, 2>,
  IsVLVT<4>
]>;

// SetCC (lhs, rhs, cc, mask, vl)
def SDTSetCCVVP : SDTypeProfile<1, 5, [        // vp_setcc
  SDTCisVec<0>,
  SDTCisVec<1>,
  SDTCisSameNumEltsAs<0, 1>,
  SDTCisSameAs<1, 2>,
  SDTCisVT<3, OtherVT>,
  SDTCisInt<4>,
  SDTCisSameNumEltsAs<0, 4>,
  IsVLVT<5>
]>;


// Binary operator commutative pattern.
class vvp_commutative<SDNode RootOp> :
  PatFrags<
  (ops node:$lhs, node:$rhs, node:$mask, node:$vlen),
  [(RootOp node:$lhs, node:$rhs, node:$mask, node:$vlen),
   (RootOp node:$rhs, node:$lhs, node:$mask, node:$vlen)]>;

// VVP node definitions.
def vvp_add    : SDNode<"VEISD::VVP_ADD",  SDTIntBinOpVVP>;
def c_vvp_add  : vvp_commutative<vvp_add>;

def vvp_sub    : SDNode<"VEISD::VVP_SUB",  SDTIntBinOpVVP>;

def vvp_mul    : SDNode<"VEISD::VVP_MUL",  SDTIntBinOpVVP>;
def c_vvp_mul  : vvp_commutative<vvp_mul>;

def vvp_sdiv   : SDNode<"VEISD::VVP_SDIV", SDTIntBinOpVVP>;
def vvp_udiv   : SDNode<"VEISD::VVP_UDIV", SDTIntBinOpVVP>;

def vvp_and    : SDNode<"VEISD::VVP_AND",  SDTIntBinOpVVP>;
def c_vvp_and  : vvp_commutative<vvp_and>;

def vvp_or     : SDNode<"VEISD::VVP_OR",  SDTIntBinOpVVP>;
def c_vvp_or   : vvp_commutative<vvp_or>;

def vvp_xor    : SDNode<"VEISD::VVP_XOR",  SDTIntBinOpVVP>;
def c_vvp_xor  : vvp_commutative<vvp_xor>;

def vvp_srl    : SDNode<"VEISD::VVP_SRL",  SDTIntBinOpVVP>;
def vvp_sra    : SDNode<"VEISD::VVP_SRA",  SDTIntBinOpVVP>;
def vvp_shl    : SDNode<"VEISD::VVP_SHL",  SDTIntBinOpVVP>;

def vvp_fadd    : SDNode<"VEISD::VVP_FADD",  SDTFPBinOpVVP>;
def c_vvp_fadd  : vvp_commutative<vvp_fadd>;
def vvp_fsub    : SDNode<"VEISD::VVP_FSUB",  SDTFPBinOpVVP>;
def vvp_fmul    : SDNode<"VEISD::VVP_FMUL",  SDTFPBinOpVVP>;
def c_vvp_fmul  : vvp_commutative<vvp_fmul>;
def vvp_fdiv    : SDNode<"VEISD::VVP_FDIV",  SDTFPBinOpVVP>;

// } Binary Operators

def vvp_select : SDNode<"VEISD::VVP_SELECT", SDTSelectVVP>;

// setcc (lhs, rhs, cc, mask, vl)
def vvp_setcc  : SDNode<"VEISD::VVP_SETCC", SDTSetCCVVP>;
