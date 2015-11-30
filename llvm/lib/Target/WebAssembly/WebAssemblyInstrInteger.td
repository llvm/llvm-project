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

let Defs = [ARGUMENTS] in {

// The spaces after the names are for aesthetic purposes only, to make
// operands line up vertically after tab expansion.
let isCommutable = 1 in
defm ADD : BinaryInt<add, "add ">;
defm SUB : BinaryInt<sub, "sub ">;
let isCommutable = 1 in
defm MUL : BinaryInt<mul, "mul ">;
// Divide and remainder trap on a zero denominator.
let hasSideEffects = 1 in {
defm DIV_S : BinaryInt<sdiv, "div_s">;
defm DIV_U : BinaryInt<udiv, "div_u">;
defm REM_S : BinaryInt<srem, "rem_s">;
defm REM_U : BinaryInt<urem, "rem_u">;
} // hasSideEffects = 1
let isCommutable = 1 in {
defm AND : BinaryInt<and, "and ">;
defm OR : BinaryInt<or, "or  ">;
defm XOR : BinaryInt<xor, "xor ">;
} // isCommutable = 1
defm SHL : BinaryInt<shl, "shl ">;
defm SHR_U : BinaryInt<srl, "shr_u">;
defm SHR_S : BinaryInt<sra, "shr_s">;

let isCommutable = 1 in {
defm EQ : ComparisonInt<SETEQ, "eq  ">;
defm NE : ComparisonInt<SETNE, "ne  ">;
} // isCommutable = 1
defm LT_S : ComparisonInt<SETLT, "lt_s">;
defm LE_S : ComparisonInt<SETLE, "le_s">;
defm LT_U : ComparisonInt<SETULT, "lt_u">;
defm LE_U : ComparisonInt<SETULE, "le_u">;
defm GT_S : ComparisonInt<SETGT, "gt_s">;
defm GE_S : ComparisonInt<SETGE, "ge_s">;
defm GT_U : ComparisonInt<SETUGT, "gt_u">;
defm GE_U : ComparisonInt<SETUGE, "ge_u">;

defm CLZ : UnaryInt<ctlz, "clz ">;
defm CTZ : UnaryInt<cttz, "ctz ">;
defm POPCNT : UnaryInt<ctpop, "popcnt">;

} // Defs = [ARGUMENTS]

// Expand the "don't care" operations to supported operations.
def : Pat<(ctlz_zero_undef I32:$src), (CLZ_I32 I32:$src)>;
def : Pat<(ctlz_zero_undef I64:$src), (CLZ_I64 I64:$src)>;
def : Pat<(cttz_zero_undef I32:$src), (CTZ_I32 I32:$src)>;
def : Pat<(cttz_zero_undef I64:$src), (CTZ_I64 I64:$src)>;

let Defs = [ARGUMENTS] in {

def SELECT_I32 : I<(outs I32:$dst), (ins I32:$cond, I32:$lhs, I32:$rhs),
                   [(set I32:$dst, (select I32:$cond, I32:$lhs, I32:$rhs))],
                   "i32.select\t$dst, $cond, $lhs, $rhs">;
def SELECT_I64 : I<(outs I64:$dst), (ins I32:$cond, I64:$lhs, I64:$rhs),
                   [(set I64:$dst, (select I32:$cond, I64:$lhs, I64:$rhs))],
                   "i64.select\t$dst, $cond, $lhs, $rhs">;

} // Defs = [ARGUMENTS]

// ISD::SELECT requires its operand to conform to getBooleanContents, but
// WebAssembly's select interprets any non-zero value as true, so we can fold
// a setne with 0 into a select.
def : Pat<(select (i32 (setne I32:$cond, 0)), I32:$lhs, I32:$rhs),
          (SELECT_I32 I32:$cond, I32:$lhs, I32:$rhs)>;
def : Pat<(select (i32 (setne I32:$cond, 0)), I64:$lhs, I64:$rhs),
          (SELECT_I64 I32:$cond, I64:$lhs, I64:$rhs)>;

// And again, this time with seteq instead of setne and the arms reversed.
def : Pat<(select (i32 (seteq I32:$cond, 0)), I32:$lhs, I32:$rhs),
          (SELECT_I32 I32:$cond, I32:$rhs, I32:$lhs)>;
def : Pat<(select (i32 (seteq I32:$cond, 0)), I64:$lhs, I64:$rhs),
          (SELECT_I64 I32:$cond, I64:$rhs, I64:$lhs)>;
