//===-- WebAssemblyInstrConv.td-WebAssembly Conversion support -*- tablegen -*-=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly datatype conversions, truncations, reinterpretations,
/// promotions, and demotions operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

def i64__WRAP_i32 : I<(outs I32:$dst), (ins I64:$src),
                     [(set I32:$dst, (trunc I64:$src))],
                     "i32.wrap/i64 $dst, $src">;

def I32__EXTEND_S_I64 : I<(outs I64:$dst), (ins I32:$src),
                           [(set I64:$dst, (sext I32:$src))],
                           "i64.extend_s/i32 $dst, $src">;
def I32__EXTEND_U_I64 : I<(outs I64:$dst), (ins I32:$src),
                          [(set I64:$dst, (zext I32:$src))],
                          "i64.extend_u/i32 $dst, $src">;

def F32__TRUNC_S_I32 : I<(outs I32:$dst), (ins F32:$src),
                         [(set I32:$dst, (fp_to_sint F32:$src))],
                         "i32.trunc_s/f32 $dst, $src">;
def F32__TRUNC_U_I32 : I<(outs I32:$dst), (ins F32:$src),
                         [(set I32:$dst, (fp_to_uint F32:$src))],
                         "i32.trunc_u/f32 $dst, $src">;
def F32__TRUNC_S_I64 : I<(outs I64:$dst), (ins F32:$src),
                         [(set I64:$dst, (fp_to_sint F32:$src))],
                         "i64.trunc_s/f32 $dst, $src">;
def F32__TRUNC_U_I64 : I<(outs I64:$dst), (ins F32:$src),
                         [(set I64:$dst, (fp_to_uint F32:$src))],
                         "i64.trunc_u/f32 $dst, $src">;
def F64__TRUNC_S_I32 : I<(outs I32:$dst), (ins F64:$src),
                         [(set I32:$dst, (fp_to_sint F64:$src))],
                         "i32.trunc_s/f64 $dst, $src">;
def F64__TRUNC_U_I32 : I<(outs I32:$dst), (ins F64:$src),
                         [(set I32:$dst, (fp_to_uint F64:$src))],
                         "i32.trunc_u/f64 $dst, $src">;
def F64__TRUNC_S_I64 : I<(outs I64:$dst), (ins F64:$src),
                         [(set I64:$dst, (fp_to_sint F64:$src))],
                         "i64.trunc_s/f64 $dst, $src">;
def F64__TRUNC_U_I64 : I<(outs I64:$dst), (ins F64:$src),
                         [(set I64:$dst, (fp_to_uint F64:$src))],
                         "i64.trunc_u/f64 $dst, $src">;

def I32__CONVERT_S_F32 : I<(outs F32:$dst), (ins I32:$src),
                           [(set F32:$dst, (sint_to_fp I32:$src))],
                           "f32.convert_s/i32 $dst, $src">;
def I32__CONVERT_U_F32 : I<(outs F32:$dst), (ins I32:$src),
                           [(set F32:$dst, (uint_to_fp I32:$src))],
                           "f32.convert_u/i32 $dst, $src">;
def I32__CONVERT_S_F64 : I<(outs F64:$dst), (ins I32:$src),
                           [(set F64:$dst, (sint_to_fp I32:$src))],
                           "f64.convert_s/i32 $dst, $src">;
def I32__CONVERT_U_F64 : I<(outs F64:$dst), (ins I32:$src),
                           [(set F64:$dst, (uint_to_fp I32:$src))],
                           "f64.convert_u/i32 $dst, $src">;
def I64__CONVERT_S_F32 : I<(outs F32:$dst), (ins I64:$src),
                           [(set F32:$dst, (sint_to_fp I64:$src))],
                           "f32.convert_s/i64 $dst, $src">;
def I64__CONVERT_U_F32 : I<(outs F32:$dst), (ins I64:$src),
                           [(set F32:$dst, (uint_to_fp I64:$src))],
                           "f32.convert_u/i64 $dst, $src">;
def I64__CONVERT_S_F64 : I<(outs F64:$dst), (ins I64:$src),
                           [(set F64:$dst, (sint_to_fp I64:$src))],
                           "f64.convert_s/i64 $dst, $src">;
def I64__CONVERT_U_F64 : I<(outs F64:$dst), (ins I64:$src),
                           [(set F64:$dst, (uint_to_fp I64:$src))],
                           "f64.convert_u/i64 $dst, $src">;

def F32__PROMOTE_F64 : I<(outs F64:$dst), (ins F32:$src),
                         [(set F64:$dst, (fextend F32:$src))],
                         "f64.promote/f32 $dst, $src">;
def F64__DEMOTE_F32 : I<(outs F32:$dst), (ins F64:$src),
                        [(set F32:$dst, (fround F64:$src))],
                        "f32.demote/f64 $dst, $src">;

def F32__REINTERPRET_I32 : I<(outs I32:$dst), (ins F32:$src),
                             [(set I32:$dst, (bitconvert F32:$src))],
                             "i32.reinterpret/f32 $dst, $src">;
def I32__REINTERPRET_F32 : I<(outs F32:$dst), (ins I32:$src),
                             [(set F32:$dst, (bitconvert I32:$src))],
                             "f32.reinterpret/i32 $dst, $src">;
def F64__REINTERPRET_I64 : I<(outs I64:$dst), (ins F64:$src),
                             [(set I64:$dst, (bitconvert F64:$src))],
                             "i64.reinterpret/f64 $dst, $src">;
def I64__REINTERPRET_F64 : I<(outs F64:$dst), (ins I64:$src),
                             [(set F64:$dst, (bitconvert I64:$src))],
                             "f64.reinterpret/i64 $dst, $src">;
