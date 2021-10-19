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

// Binary operator commutative pattern.
class vvp_commutative<SDNode RootOp> :
  PatFrags<
  (ops node:$lhs, node:$rhs, node:$mask, node:$vlen),
  [(RootOp node:$lhs, node:$rhs, node:$mask, node:$vlen),
   (RootOp node:$rhs, node:$lhs, node:$mask, node:$vlen)]>;

// VVP node definitions.
def vvp_add    : SDNode<"VEISD::VVP_ADD",  SDTIntBinOpVVP>;
def c_vvp_add  : vvp_commutative<vvp_add>;

def vvp_and    : SDNode<"VEISD::VVP_AND",  SDTIntBinOpVVP>;
def c_vvp_and  : vvp_commutative<vvp_and>;

// } Binary Operators
