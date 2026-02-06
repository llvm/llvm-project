//===-- WebAssemblyGIsel.td - WASM GlobalISel Patterns -----*- tablegen -*-===//
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


//===----------------------------------------------------------------------===//
// Pointer types and related patterns
//===----------------------------------------------------------------------===//

defvar ModeAddr32 = DefaultMode;
def ModeAddr64 : HwMode<[HasAddr64]>;

def Addr0VT : ValueTypeByHwMode<[ModeAddr32, ModeAddr64],
                               [i32,  i64]>;

def p0 : PtrValueTypeByHwMode<Addr0VT, 0>;

// G_CONSTANT with p0
def : Pat<(p0 (imm:$addr)),
          (CONST_I32 imm:$addr)>, Requires<[HasAddr32]>;
def : Pat<(p0 (imm:$addr)),
          (CONST_I64 imm:$addr)>, Requires<[HasAddr64]>;

// G_LOAD of p0
def : Pat<(p0 (load (AddrOps32 offset32_op:$offset, I32:$addr))),
          (LOAD_I32_A32 0,
                        offset32_op:$offset,
                        I32:$addr)>,
          Requires<[HasAddr32]>;

def : Pat<(p0 (load (AddrOps64 offset64_op:$offset, I64:$addr))),
          (LOAD_I64_A64 0,
                        offset64_op:$offset,
                        I64:$addr)>,
          Requires<[HasAddr64]>;

// G_STORE of p0
def : Pat<(store p0:$val, (AddrOps32 offset32_op:$offset, I32:$addr)),
        (STORE_I32_A32 0,
                       offset32_op:$offset,
                       I32:$addr,
                       p0:$val)>,
        Requires<[HasAddr32]>;

def : Pat<(store p0:$val, (AddrOps64 offset64_op:$offset, I64:$addr)),
        (STORE_I64_A64 0,
                       offset64_op:$offset,
                       I64:$addr,
                       p0:$val)>,
        Requires<[HasAddr64]>;

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

//===----------------------------------------------------------------------===//
// Miscallenous patterns
//===----------------------------------------------------------------------===//

def : Pat<(i32 (fp_to_sint_sat_gi F32:$src)), (I32_TRUNC_S_SAT_F32 F32:$src)>;
def : Pat<(i32 (fp_to_uint_sat_gi F32:$src)), (I32_TRUNC_U_SAT_F32 F32:$src)>;
def : Pat<(i32 (fp_to_sint_sat_gi F64:$src)), (I32_TRUNC_S_SAT_F64 F64:$src)>;
def : Pat<(i32 (fp_to_uint_sat_gi F64:$src)), (I32_TRUNC_U_SAT_F64 F64:$src)>;
def : Pat<(i64 (fp_to_sint_sat_gi F32:$src)), (I64_TRUNC_S_SAT_F32 F32:$src)>;
def : Pat<(i64 (fp_to_uint_sat_gi F32:$src)), (I64_TRUNC_U_SAT_F32 F32:$src)>;
def : Pat<(i64 (fp_to_sint_sat_gi F64:$src)), (I64_TRUNC_S_SAT_F64 F64:$src)>;
def : Pat<(i64 (fp_to_uint_sat_gi F64:$src)), (I64_TRUNC_U_SAT_F64 F64:$src)>;

def : GINodeEquiv<G_DEBUGTRAP, debugtrap>;

def : Pat<(i32 (ctlz_zero_undef I32:$src)), (CLZ_I32 I32:$src)>;
def : Pat<(i64 (ctlz_zero_undef I64:$src)), (CLZ_I64 I64:$src)>;
def : Pat<(i32 (cttz_zero_undef I32:$src)), (CTZ_I32 I32:$src)>;
def : Pat<(i64 (cttz_zero_undef I64:$src)), (CTZ_I64 I64:$src)>;


def : Pat<(f32 (strict_fmul F32:$lhs, F32:$rhs)), (MUL_F32 F32:$lhs, F32:$rhs)>;
def : Pat<(f64 (strict_fmul F64:$lhs, F64:$rhs)), (MUL_F64 F64:$lhs, F64:$rhs)>;

//===----------------------------------------------------------------------===//
// Complex pattern equivalents
//===----------------------------------------------------------------------===//

def gi_AddrOps32 : GIComplexOperandMatcher<s32, "selectAddrOperands32">,
                   GIComplexPatternEquiv<AddrOps32>;

def gi_AddrOps64 : GIComplexOperandMatcher<s64, "selectAddrOperands64">,
                   GIComplexPatternEquiv<AddrOps64>;
