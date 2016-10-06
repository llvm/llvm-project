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
defm ADD : BinaryInt<add, "add ", 0x40, 0x5b>;
defm SUB : BinaryInt<sub, "sub ", 0x41, 0x5c>;
let isCommutable = 1 in
defm MUL : BinaryInt<mul, "mul ", 0x42, 0x5d>;
// Divide and remainder trap on a zero denominator.
let hasSideEffects = 1 in {
defm DIV_S : BinaryInt<sdiv, "div_s", 0x43, 0x5e>;
defm DIV_U : BinaryInt<udiv, "div_u", 0x44, 0x5f>;
defm REM_S : BinaryInt<srem, "rem_s", 0x45, 0x60>;
defm REM_U : BinaryInt<urem, "rem_u", 0x46, 0x61>;
} // hasSideEffects = 1
let isCommutable = 1 in {
defm AND : BinaryInt<and, "and ", 0x47, 0x62>;
defm OR : BinaryInt<or, "or  ", 0x48, 0x63>;
defm XOR : BinaryInt<xor, "xor ", 0x49, 0x64>;
} // isCommutable = 1
defm SHL : BinaryInt<shl, "shl ", 0x4a, 0x65>;
defm SHR_U : BinaryInt<srl, "shr_u", 0x4b, 0x66>;
defm SHR_S : BinaryInt<sra, "shr_s", 0x4c, 0x67>;
defm ROTL : BinaryInt<rotl, "rotl", 0xb7, 0xb9>;
defm ROTR : BinaryInt<rotr, "rotr", 0xb6, 0xb8>;

let isCommutable = 1 in {
defm EQ : ComparisonInt<SETEQ, "eq  ", 0x4d, 0x68>;
defm NE : ComparisonInt<SETNE, "ne  ", 0x4e, 0x69>;
} // isCommutable = 1
defm LT_S : ComparisonInt<SETLT, "lt_s", 0x4f, 0x6a>;
defm LE_S : ComparisonInt<SETLE, "le_s", 0x50, 0x6b>;
defm LT_U : ComparisonInt<SETULT, "lt_u", 0x51, 0x6c>;
defm LE_U : ComparisonInt<SETULE, "le_u", 0x52, 0x6d>;
defm GT_S : ComparisonInt<SETGT, "gt_s", 0x53, 0x6e>;
defm GE_S : ComparisonInt<SETGE, "ge_s", 0x54, 0x6f>;
defm GT_U : ComparisonInt<SETUGT, "gt_u", 0x55, 0x70>;
defm GE_U : ComparisonInt<SETUGE, "ge_u", 0x56, 0x71>;

defm CLZ : UnaryInt<ctlz, "clz ", 0x57, 0x72>;
defm CTZ : UnaryInt<cttz, "ctz ", 0x58, 0x73>;
defm POPCNT : UnaryInt<ctpop, "popcnt", 0x59, 0x74>;

def EQZ_I32 : I<(outs I32:$dst), (ins I32:$src),
                [(set I32:$dst, (setcc I32:$src, 0, SETEQ))],
                "i32.eqz \t$dst, $src", 0x5a>;
def EQZ_I64 : I<(outs I32:$dst), (ins I64:$src),
                [(set I32:$dst, (setcc I64:$src, 0, SETEQ))],
                "i64.eqz \t$dst, $src", 0xba>;

} // Defs = [ARGUMENTS]

// Optimize away an explicit mask on a rotate count.
def : Pat<(rotl I32:$lhs, (and I32:$rhs, 31)), (ROTL_I32 I32:$lhs, I32:$rhs)>;
def : Pat<(rotr I32:$lhs, (and I32:$rhs, 31)), (ROTR_I32 I32:$lhs, I32:$rhs)>;
def : Pat<(rotl I64:$lhs, (and I64:$rhs, 63)), (ROTL_I64 I64:$lhs, I64:$rhs)>;
def : Pat<(rotr I64:$lhs, (and I64:$rhs, 63)), (ROTR_I64 I64:$lhs, I64:$rhs)>;

let Defs = [ARGUMENTS] in {

def SELECT_I32 : I<(outs I32:$dst), (ins I32:$lhs, I32:$rhs, I32:$cond),
                   [(set I32:$dst, (select I32:$cond, I32:$lhs, I32:$rhs))],
                   "i32.select\t$dst, $lhs, $rhs, $cond">;
def SELECT_I64 : I<(outs I64:$dst), (ins I64:$lhs, I64:$rhs, I32:$cond),
                   [(set I64:$dst, (select I32:$cond, I64:$lhs, I64:$rhs))],
                   "i64.select\t$dst, $lhs, $rhs, $cond">;

} // Defs = [ARGUMENTS]

// ISD::SELECT requires its operand to conform to getBooleanContents, but
// WebAssembly's select interprets any non-zero value as true, so we can fold
// a setne with 0 into a select.
def : Pat<(select (i32 (setne I32:$cond, 0)), I32:$lhs, I32:$rhs),
          (SELECT_I32 I32:$lhs, I32:$rhs, I32:$cond)>;
def : Pat<(select (i32 (setne I32:$cond, 0)), I64:$lhs, I64:$rhs),
          (SELECT_I64 I64:$lhs, I64:$rhs, I32:$cond)>;

// And again, this time with seteq instead of setne and the arms reversed.
def : Pat<(select (i32 (seteq I32:$cond, 0)), I32:$lhs, I32:$rhs),
          (SELECT_I32 I32:$rhs, I32:$lhs, I32:$cond)>;
def : Pat<(select (i32 (seteq I32:$cond, 0)), I64:$lhs, I64:$rhs),
          (SELECT_I64 I64:$rhs, I64:$lhs, I32:$cond)>;
