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

//===----------------------------------------------------------------------===//
// Miscellaneous patterns
//===----------------------------------------------------------------------===//

def : Pat<(i32 (fp_to_sint_sat_gi F32:$src)), (I32_TRUNC_S_SAT_F32 F32:$src)>;
def : Pat<(i32 (fp_to_uint_sat_gi F32:$src)), (I32_TRUNC_U_SAT_F32 F32:$src)>;
def : Pat<(i32 (fp_to_sint_sat_gi F64:$src)), (I32_TRUNC_S_SAT_F64 F64:$src)>;
def : Pat<(i32 (fp_to_uint_sat_gi F64:$src)), (I32_TRUNC_U_SAT_F64 F64:$src)>;
def : Pat<(i64 (fp_to_sint_sat_gi F32:$src)), (I64_TRUNC_S_SAT_F32 F32:$src)>;
def : Pat<(i64 (fp_to_uint_sat_gi F32:$src)), (I64_TRUNC_U_SAT_F32 F32:$src)>;
def : Pat<(i64 (fp_to_sint_sat_gi F64:$src)), (I64_TRUNC_S_SAT_F64 F64:$src)>;
def : Pat<(i64 (fp_to_uint_sat_gi F64:$src)), (I64_TRUNC_U_SAT_F64 F64:$src)>;
