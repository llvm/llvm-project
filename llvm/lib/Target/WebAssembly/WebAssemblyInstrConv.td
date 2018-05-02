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
/// WebAssembly datatype conversions, truncations, reinterpretations,
/// promotions, and demotions operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

let Defs = [ARGUMENTS] in {

def I32_WRAP_I64 : I<(outs I32:$dst), (ins I64:$src),
                      [(set I32:$dst, (trunc I64:$src))],
                      "i32.wrap/i64\t$dst, $src", 0xa7>;

def I64_EXTEND_S_I32 : I<(outs I64:$dst), (ins I32:$src),
                          [(set I64:$dst, (sext I32:$src))],
                          "i64.extend_s/i32\t$dst, $src", 0xac>;
def I64_EXTEND_U_I32 : I<(outs I64:$dst), (ins I32:$src),
                         [(set I64:$dst, (zext I32:$src))],
                         "i64.extend_u/i32\t$dst, $src", 0xad>;

let Predicates = [HasSignExt] in {
def I32_EXTEND8_S_I32 : I<(outs I32:$dst), (ins I32:$src),
                          [(set I32:$dst, (sext_inreg I32:$src, i8))],
                          "i32.extend8_s\t$dst, $src", 0xc0>;
def I32_EXTEND16_S_I32 : I<(outs I32:$dst), (ins I32:$src),
                           [(set I32:$dst, (sext_inreg I32:$src, i16))],
                           "i32.extend16_s\t$dst, $src", 0xc1>;
def I64_EXTEND8_S_I64 : I<(outs I64:$dst), (ins I64:$src),
                           [(set I64:$dst, (sext_inreg I64:$src, i8))],
                           "i64.extend8_s\t$dst, $src", 0xc2>;
def I64_EXTEND16_S_I64 : I<(outs I64:$dst), (ins I64:$src),
                           [(set I64:$dst, (sext_inreg I64:$src, i16))],
                           "i64.extend16_s\t$dst, $src", 0xc3>;
def I64_EXTEND32_S_I64 : I<(outs I64:$dst), (ins I64:$src),
                           [(set I64:$dst, (sext_inreg I64:$src, i32))],
                           "i64.extend32_s\t$dst, $src", 0xc4>;
} // Predicates = [HasSignExt]

} // defs = [ARGUMENTS]

// Expand a "don't care" extend into zero-extend (chosen over sign-extend
// somewhat arbitrarily, although it favors popular hardware architectures
// and is conceptually a simpler operation).
def : Pat<(i64 (anyext I32:$src)), (I64_EXTEND_U_I32 I32:$src)>;

let Defs = [ARGUMENTS] in {

// Conversion from floating point to integer instructions which don't trap on
// overflow or invalid.
def I32_TRUNC_S_SAT_F32 : I<(outs I32:$dst), (ins F32:$src),
                            [(set I32:$dst, (fp_to_sint F32:$src))],
                            "i32.trunc_s:sat/f32\t$dst, $src", 0xfc00>,
                            Requires<[HasNontrappingFPToInt]>;
def I32_TRUNC_U_SAT_F32 : I<(outs I32:$dst), (ins F32:$src),
                            [(set I32:$dst, (fp_to_uint F32:$src))],
                            "i32.trunc_u:sat/f32\t$dst, $src", 0xfc01>,
                            Requires<[HasNontrappingFPToInt]>;
def I64_TRUNC_S_SAT_F32 : I<(outs I64:$dst), (ins F32:$src),
                            [(set I64:$dst, (fp_to_sint F32:$src))],
                            "i64.trunc_s:sat/f32\t$dst, $src", 0xfc04>,
                            Requires<[HasNontrappingFPToInt]>;
def I64_TRUNC_U_SAT_F32 : I<(outs I64:$dst), (ins F32:$src),
                            [(set I64:$dst, (fp_to_uint F32:$src))],
                            "i64.trunc_u:sat/f32\t$dst, $src", 0xfc05>,
                            Requires<[HasNontrappingFPToInt]>;
def I32_TRUNC_S_SAT_F64 : I<(outs I32:$dst), (ins F64:$src),
                            [(set I32:$dst, (fp_to_sint F64:$src))],
                            "i32.trunc_s:sat/f64\t$dst, $src", 0xfc02>,
                            Requires<[HasNontrappingFPToInt]>;
def I32_TRUNC_U_SAT_F64 : I<(outs I32:$dst), (ins F64:$src),
                            [(set I32:$dst, (fp_to_uint F64:$src))],
                            "i32.trunc_u:sat/f64\t$dst, $src", 0xfc03>,
                            Requires<[HasNontrappingFPToInt]>;
def I64_TRUNC_S_SAT_F64 : I<(outs I64:$dst), (ins F64:$src),
                            [(set I64:$dst, (fp_to_sint F64:$src))],
                            "i64.trunc_s:sat/f64\t$dst, $src", 0xfc06>,
                            Requires<[HasNontrappingFPToInt]>;
def I64_TRUNC_U_SAT_F64 : I<(outs I64:$dst), (ins F64:$src),
                            [(set I64:$dst, (fp_to_uint F64:$src))],
                            "i64.trunc_u:sat/f64\t$dst, $src", 0xfc07>,
                            Requires<[HasNontrappingFPToInt]>;

// Conversion from floating point to integer pseudo-instructions which don't
// trap on overflow or invalid.
let usesCustomInserter = 1, isCodeGenOnly = 1 in {
def FP_TO_SINT_I32_F32 : I<(outs I32:$dst), (ins F32:$src),
                        [(set I32:$dst, (fp_to_sint F32:$src))], "", 0>,
                        Requires<[NotHasNontrappingFPToInt]>;
def FP_TO_UINT_I32_F32 : I<(outs I32:$dst), (ins F32:$src),
                        [(set I32:$dst, (fp_to_uint F32:$src))], "", 0>,
                        Requires<[NotHasNontrappingFPToInt]>;
def FP_TO_SINT_I64_F32 : I<(outs I64:$dst), (ins F32:$src),
                        [(set I64:$dst, (fp_to_sint F32:$src))], "", 0>,
                        Requires<[NotHasNontrappingFPToInt]>;
def FP_TO_UINT_I64_F32 : I<(outs I64:$dst), (ins F32:$src),
                        [(set I64:$dst, (fp_to_uint F32:$src))], "", 0>,
                        Requires<[NotHasNontrappingFPToInt]>;
def FP_TO_SINT_I32_F64 : I<(outs I32:$dst), (ins F64:$src),
                        [(set I32:$dst, (fp_to_sint F64:$src))], "", 0>,
                        Requires<[NotHasNontrappingFPToInt]>;
def FP_TO_UINT_I32_F64 : I<(outs I32:$dst), (ins F64:$src),
                        [(set I32:$dst, (fp_to_uint F64:$src))], "", 0>,
                        Requires<[NotHasNontrappingFPToInt]>;
def FP_TO_SINT_I64_F64 : I<(outs I64:$dst), (ins F64:$src),
                        [(set I64:$dst, (fp_to_sint F64:$src))], "", 0>,
                        Requires<[NotHasNontrappingFPToInt]>;
def FP_TO_UINT_I64_F64 : I<(outs I64:$dst), (ins F64:$src),
                        [(set I64:$dst, (fp_to_uint F64:$src))], "", 0>,
                        Requires<[NotHasNontrappingFPToInt]>;
} // usesCustomInserter, isCodeGenOnly = 1

// Conversion from floating point to integer traps on overflow and invalid.
let hasSideEffects = 1 in {
def I32_TRUNC_S_F32 : I<(outs I32:$dst), (ins F32:$src),
                        [], "i32.trunc_s/f32\t$dst, $src", 0xa8>;
def I32_TRUNC_U_F32 : I<(outs I32:$dst), (ins F32:$src),
                        [], "i32.trunc_u/f32\t$dst, $src", 0xa9>;
def I64_TRUNC_S_F32 : I<(outs I64:$dst), (ins F32:$src),
                        [], "i64.trunc_s/f32\t$dst, $src", 0xae>;
def I64_TRUNC_U_F32 : I<(outs I64:$dst), (ins F32:$src),
                        [], "i64.trunc_u/f32\t$dst, $src", 0xaf>;
def I32_TRUNC_S_F64 : I<(outs I32:$dst), (ins F64:$src),
                        [], "i32.trunc_s/f64\t$dst, $src", 0xaa>;
def I32_TRUNC_U_F64 : I<(outs I32:$dst), (ins F64:$src),
                        [], "i32.trunc_u/f64\t$dst, $src", 0xab>;
def I64_TRUNC_S_F64 : I<(outs I64:$dst), (ins F64:$src),
                        [], "i64.trunc_s/f64\t$dst, $src", 0xb0>;
def I64_TRUNC_U_F64 : I<(outs I64:$dst), (ins F64:$src),
                        [], "i64.trunc_u/f64\t$dst, $src", 0xb1>;
} // hasSideEffects = 1

def F32_CONVERT_S_I32 : I<(outs F32:$dst), (ins I32:$src),
                          [(set F32:$dst, (sint_to_fp I32:$src))],
                          "f32.convert_s/i32\t$dst, $src", 0xb2>;
def F32_CONVERT_U_I32 : I<(outs F32:$dst), (ins I32:$src),
                          [(set F32:$dst, (uint_to_fp I32:$src))],
                          "f32.convert_u/i32\t$dst, $src", 0xb3>;
def F64_CONVERT_S_I32 : I<(outs F64:$dst), (ins I32:$src),
                          [(set F64:$dst, (sint_to_fp I32:$src))],
                          "f64.convert_s/i32\t$dst, $src", 0xb7>;
def F64_CONVERT_U_I32 : I<(outs F64:$dst), (ins I32:$src),
                          [(set F64:$dst, (uint_to_fp I32:$src))],
                          "f64.convert_u/i32\t$dst, $src", 0xb8>;
def F32_CONVERT_S_I64 : I<(outs F32:$dst), (ins I64:$src),
                          [(set F32:$dst, (sint_to_fp I64:$src))],
                          "f32.convert_s/i64\t$dst, $src", 0xb4>;
def F32_CONVERT_U_I64 : I<(outs F32:$dst), (ins I64:$src),
                          [(set F32:$dst, (uint_to_fp I64:$src))],
                          "f32.convert_u/i64\t$dst, $src", 0xb5>;
def F64_CONVERT_S_I64 : I<(outs F64:$dst), (ins I64:$src),
                          [(set F64:$dst, (sint_to_fp I64:$src))],
                          "f64.convert_s/i64\t$dst, $src", 0xb9>;
def F64_CONVERT_U_I64 : I<(outs F64:$dst), (ins I64:$src),
                          [(set F64:$dst, (uint_to_fp I64:$src))],
                          "f64.convert_u/i64\t$dst, $src", 0xba>;

def F64_PROMOTE_F32 : I<(outs F64:$dst), (ins F32:$src),
                        [(set F64:$dst, (fpextend F32:$src))],
                        "f64.promote/f32\t$dst, $src", 0xbb>;
def F32_DEMOTE_F64 : I<(outs F32:$dst), (ins F64:$src),
                       [(set F32:$dst, (fpround F64:$src))],
                       "f32.demote/f64\t$dst, $src", 0xb6>;

def I32_REINTERPRET_F32 : I<(outs I32:$dst), (ins F32:$src),
                            [(set I32:$dst, (bitconvert F32:$src))],
                            "i32.reinterpret/f32\t$dst, $src", 0xbc>;
def F32_REINTERPRET_I32 : I<(outs F32:$dst), (ins I32:$src),
                            [(set F32:$dst, (bitconvert I32:$src))],
                            "f32.reinterpret/i32\t$dst, $src", 0xbe>;
def I64_REINTERPRET_F64 : I<(outs I64:$dst), (ins F64:$src),
                            [(set I64:$dst, (bitconvert F64:$src))],
                            "i64.reinterpret/f64\t$dst, $src", 0xbd>;
def F64_REINTERPRET_I64 : I<(outs F64:$dst), (ins I64:$src),
                            [(set F64:$dst, (bitconvert I64:$src))],
                            "f64.reinterpret/i64\t$dst, $src", 0xbf>;

} // Defs = [ARGUMENTS]
