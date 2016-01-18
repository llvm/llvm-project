// WebAssemblyInstrFloat.td-WebAssembly Float codegen support ---*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Floating-point operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

defm ADD : BinaryFP<fadd, "add">;
defm SUB : BinaryFP<fsub, "sub">;
defm MUL : BinaryFP<fmul, "mul">;
defm DIV : BinaryFP<fdiv, "div">;
defm SQRT : UnaryFP<fsqrt, "sqrt">;

defm ABS : UnaryFP<fabs, "abs">;
defm NEG : UnaryFP<fneg, "neg">;
defm COPYSIGN : BinaryFP<fcopysign, "copysign">;

defm CEIL : UnaryFP<fceil, "ceil">;
defm FLOOR : UnaryFP<ffloor, "floor">;
defm TRUNC : UnaryFP<ftrunc, "trunc">;
defm NEAREST : UnaryFP<fnearbyint, "nearest">;

// WebAssembly doesn't expose inexact exceptions, so map frint to fnearbyint.
def : Pat<(frint f32:$src), (NEAREST_F32 f32:$src)>;
def : Pat<(frint f64:$src), (NEAREST_F64 f64:$src)>;

defm EQ : ComparisonFP<SETOEQ, "eq">;
defm NE : ComparisonFP<SETUNE, "ne">;
defm LT : ComparisonFP<SETOLT, "lt">;
defm LE : ComparisonFP<SETOLE, "le">;
defm GT : ComparisonFP<SETOGT, "gt">;
defm GE : ComparisonFP<SETOGE, "ge">;

// Don't care floating-point comparisons, supported via other comparisons.
def : Pat<(seteq f32:$lhs, f32:$rhs), (EQ_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(setne f32:$lhs, f32:$rhs), (NE_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(setlt f32:$lhs, f32:$rhs), (LT_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(setle f32:$lhs, f32:$rhs), (LE_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(setgt f32:$lhs, f32:$rhs), (GT_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(setge f32:$lhs, f32:$rhs), (GE_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(seteq f64:$lhs, f64:$rhs), (EQ_F64 f64:$lhs, f64:$rhs)>;
def : Pat<(setne f64:$lhs, f64:$rhs), (NE_F64 f64:$lhs, f64:$rhs)>;
def : Pat<(setlt f64:$lhs, f64:$rhs), (LT_F64 f64:$lhs, f64:$rhs)>;
def : Pat<(setle f64:$lhs, f64:$rhs), (LE_F64 f64:$lhs, f64:$rhs)>;
def : Pat<(setgt f64:$lhs, f64:$rhs), (GT_F64 f64:$lhs, f64:$rhs)>;
def : Pat<(setge f64:$lhs, f64:$rhs), (GE_F64 f64:$lhs, f64:$rhs)>;

/*
 * TODO(jfb): Add the following for 32-bit and 64-bit.
 *
 * f32.min: minimum (binary operator); if either operand is NaN, returns NaN
 * f32.max: maximum (binary operator); if either operand is NaN, returns NaN
 */

def SELECT_F32 : I<(outs F32:$dst), (ins I32:$cond, F32:$lhs, F32:$rhs),
                   [(set F32:$dst, (select I32:$cond, F32:$lhs, F32:$rhs))],
                   "f32.select $dst, $cond, $lhs, $rhs">;
def SELECT_F64 : I<(outs F64:$dst), (ins I32:$cond, F64:$lhs, F64:$rhs),
                   [(set F64:$dst, (select I32:$cond, F64:$lhs, F64:$rhs))],
                   "f64.select $dst, $cond, $lhs, $rhs">;
