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

def : GINodeEquiv<G_ADD, add>;
def : GINodeEquiv<G_SUB, sub>;
def : GINodeEquiv<G_MUL, mul>;

def : GINodeEquiv<G_OR, or>;
def : GINodeEquiv<G_XOR, xor>;
def : GINodeEquiv<G_AND, and>;

def : GINodeEquiv<G_SHL, shl>;
def : GINodeEquiv<G_LSHR, srl>;
def : GINodeEquiv<G_ASHR, sra>;

def : GINodeEquiv<G_SDIV, sdiv>;
def : GINodeEquiv<G_UDIV, udiv>;
def : GINodeEquiv<G_SREM, srem>;
def : GINodeEquiv<G_UREM, urem>;

def : GINodeEquiv<G_BR, br>;
