//===-- WebAssemblyGISel.td - Wasm GlobalISel Patterns -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains patterns that are relevant to GlobalISel, including
/// GIComplexOperandMatcher definitions for equivalent SelectionDAG
/// ComplexPatterns.
//
//===----------------------------------------------------------------------===//

include "WebAssembly.td"
include "WebAssemblyCombine.td"

defvar ModeAddr32 = DefaultMode;
def ModeAddr64 : HwMode<[HasAddr64]>;

def p0i : ValueTypeByHwMode<[ModeAddr32, ModeAddr64],
                            [i32, i64]>;
def p0 : PtrValueTypeByHwMode<p0i, 0>;


def : Pat<(p0 (ptradd p0:$lhs, p0i:$rhs)),
          (ADD_I32 I32:$lhs, I32:$rhs)>, Requires<[HasAddr32]>;
def : Pat<(p0 (ptradd p0:$lhs, p0i:$rhs)),
          (ADD_I64 I64:$lhs, I64:$rhs)>, Requires<[HasAddr64]>;

def : Pat<(frameindex:$fi),
          (COPY_I32 (to_tframeindex $fi))>, Requires<[HasAddr32]>;
def : Pat<(frameindex:$fi),
          (COPY_I64 (to_tframeindex $fi))>, Requires<[HasAddr64]>;

// G_ICMP between p0
multiclass ComparisonP0<CondCode cond, string Name> {
  def : Pat<(setcc p0:$lhs, p0:$rhs, cond),
            (!cast<NI>(Name # "_I32") I32:$lhs, I32:$rhs)>, Requires<[HasAddr32]>;
  def : Pat<(setcc p0:$lhs, p0:$rhs, cond),
            (!cast<NI>(Name # "_I64") I64:$lhs, I64:$rhs)>, Requires<[HasAddr64]>;
}

defm : ComparisonP0<SETEQ, "EQ">;
defm : ComparisonP0<SETNE, "NE">;
defm : ComparisonP0<SETLT,  "LT_S">;
defm : ComparisonP0<SETULT, "LT_U">;
defm : ComparisonP0<SETGT,  "GT_S">;
defm : ComparisonP0<SETUGT, "GT_U">;
defm : ComparisonP0<SETLE,  "LE_S">;
defm : ComparisonP0<SETULE, "LE_U">;
defm : ComparisonP0<SETGE,  "GE_S">;
defm : ComparisonP0<SETUGE, "GE_U">;

// G_SELECT of p0
def : Pat<(select I32:$cond, p0:$lhs, p0:$rhs),
          (SELECT_I32 I32:$lhs, I32:$rhs, I32:$cond)>, Requires<[HasAddr32]>;
def : Pat<(select I32:$cond, p0:$lhs, p0:$rhs),
          (SELECT_I64 I64:$lhs, I64:$rhs, I32:$cond)>, Requires<[HasAddr64]>;

// ISD::SELECT requires its operand to conform to getBooleanContents, but
// WebAssembly's select interprets any non-zero value as true, so we can fold
// a setne with 0 into a select.
def : Pat<(select (i32 (setne I32:$cond, 0)), p0:$lhs, p0:$rhs),
          (SELECT_I32 I32:$lhs, I32:$rhs, I32:$cond)>, Requires<[HasAddr32]>;
def : Pat<(select (i32 (setne I32:$cond, 0)), p0:$lhs, p0:$rhs),
          (SELECT_I64 I64:$lhs, I64:$rhs, I32:$cond)>, Requires<[HasAddr64]>;

// And again, this time with seteq instead of setne and the arms reversed.
def : Pat<(select (i32 (seteq I32:$cond, 0)), p0:$lhs, p0:$rhs),
          (SELECT_I32 I32:$rhs, I32:$lhs, I32:$cond)>, Requires<[HasAddr32]>;
def : Pat<(select (i32 (seteq I32:$cond, 0)), p0:$lhs, p0:$rhs),
          (SELECT_I64 I64:$rhs, I64:$lhs, I32:$cond)>, Requires<[HasAddr64]>;

//===----------------------------------------------------------------------===//
// Complex pattern equivalents
//===----------------------------------------------------------------------===//

def gi_AddrOps32 : GIComplexOperandMatcher<s32, "selectAddrOperands32">,
                   GIComplexPatternEquiv<AddrOps32>;

def gi_AddrOps64 : GIComplexOperandMatcher<s64, "selectAddrOperands64">,
                   GIComplexPatternEquiv<AddrOps64>;
