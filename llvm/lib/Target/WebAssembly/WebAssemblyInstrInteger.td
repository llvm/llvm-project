// WebAssemblyInstrInteger.td-WebAssembly Integer codegen -------*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Integer operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

defm ADD : BinaryInt<add, "add">;
defm SUB : BinaryInt<sub, "sub">;
defm MUL : BinaryInt<mul, "mul">;
defm DIV_S : BinaryInt<sdiv, "div_s">;
defm DIV_U : BinaryInt<udiv, "div_u">;
defm REM_S : BinaryInt<srem, "rem_s">;
defm REM_U : BinaryInt<urem, "rem_u">;
defm AND : BinaryInt<and, "and">;
defm OR : BinaryInt<or, "or">;
defm XOR : BinaryInt<xor, "xor">;
defm SHL : BinaryInt<shl, "shl">;
defm SHR_U : BinaryInt<srl, "shr_u">;
defm SHR_S : BinaryInt<sra, "shr_s">;

defm EQ : ComparisonInt<SETEQ, "eq">;
defm NE : ComparisonInt<SETNE, "ne">;
defm LT_S : ComparisonInt<SETLT, "lt_s">;
defm LE_S : ComparisonInt<SETLE, "le_s">;
defm LT_U : ComparisonInt<SETULT, "lt_u">;
defm LE_U : ComparisonInt<SETULE, "le_u">;
defm GT_S : ComparisonInt<SETGT, "gt_s">;
defm GE_S : ComparisonInt<SETGE, "ge_s">;
defm GT_U : ComparisonInt<SETUGT, "gt_u">;
defm GE_U : ComparisonInt<SETUGE, "ge_u">;

defm CLZ : UnaryInt<ctlz, "clz">;
defm CTZ : UnaryInt<cttz, "ctz">;
defm POPCNT : UnaryInt<ctpop, "popcnt">;

// Expand the "don't care" operations to supported operations.
def : Pat<(ctlz_zero_undef I32:$src), (CLZ_I32 I32:$src)>;
def : Pat<(ctlz_zero_undef I64:$src), (CLZ_I64 I64:$src)>;
def : Pat<(cttz_zero_undef I32:$src), (CTZ_I32 I32:$src)>;
def : Pat<(cttz_zero_undef I64:$src), (CTZ_I64 I64:$src)>;

def SELECT_I32 : I<(outs I32:$dst), (ins I32:$cond, I32:$lhs, I32:$rhs),
                   [(set I32:$dst, (select I32:$cond, I32:$lhs, I32:$rhs))],
                   "i32.select $dst, $cond, $lhs, $rhs">;
def SELECT_I64 : I<(outs I64:$dst), (ins I32:$cond, I64:$lhs, I64:$rhs),
                   [(set I64:$dst, (select I32:$cond, I64:$lhs, I64:$rhs))],
                   "i64.select $dst, $cond, $lhs, $rhs">;
