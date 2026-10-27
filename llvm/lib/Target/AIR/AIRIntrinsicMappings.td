//===- AIRIntrinsicMappings.td ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mapping of LLVM intrinsic names to their Apple AIR equivalents. The records
// declared here are consumed by -gen-searchable-tables to produce a binary-
// searchable table used by the LLVMToAIRIntrinsics pass.
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"

// One record per function, for every scalar and vector type. The AIR name of
// a call is built from the call's own overload suffix:
//
//   "air." ["fast_"] AIRName "." ["s." | "u."] <overload suffix>
//
// so llvm.sin.v4f32 becomes air.sin.v4f32, and llvm.smin.i32 air.min.s.i32.
// "fast_" is used when the function has a relaxed variant, the call allows
// approximate functions (afn) and the element type is float; that is the
// convention of Metal.jl and floor, and calls without afn keep the precise
// function. Intrinsics without a record are left alone: some (llvm.copysign,
// llvm.memcpy, the saturating integer ops) are accepted by Metal as they are.
//
// Every record here is exercised on hardware by llvm/utils/air-gpu.

class AIRFunction<string llvm, string air, bits<2> sign = 0, bit fast = 0,
                  bits<4> numArgs = 0> {
  string LLVMName = llvm;       // Base name of the LLVM intrinsic.
  string AIRName = air;         // AIR function, without "air." and type suffix.
  bits<2> Signedness = sign;    // 0: none, 1: signed (".s"), 2: unsigned (".u").
  bit HasFast = fast;           // Whether air.fast_<AIRName> exists.
  bits<4> NumArgs = numArgs;    // Leading operands AIR takes; 0 for all.
  // Whether the AIR function lowers back to this intrinsic (AGXLowerAIR):
  // false where several intrinsics share one AIR function.
  bit Invertible = 1;
}

def AIRFunctions : GenericTable {
  let FilterClass = "AIRFunction";
  let Fields = ["LLVMName", "AIRName", "Signedness", "HasFast", "NumArgs",
                "Invertible"];
}

class AIRFloatFunction<string llvm, string air>
    : AIRFunction<llvm, air, 0, 1>;

// Floating point, with relaxed variants.
def : AIRFloatFunction<"llvm.fabs", "fabs">;
def : AIRFloatFunction<"llvm.floor", "floor">;
def : AIRFloatFunction<"llvm.ceil", "ceil">;
def : AIRFloatFunction<"llvm.trunc", "trunc">;
def : AIRFloatFunction<"llvm.rint", "rint">;
def : AIRFloatFunction<"llvm.round", "round">;
def : AIRFloatFunction<"llvm.sqrt", "sqrt">;
def : AIRFloatFunction<"llvm.exp", "exp">;
def : AIRFloatFunction<"llvm.exp2", "exp2">;
def : AIRFloatFunction<"llvm.exp10", "exp10">;
def : AIRFloatFunction<"llvm.log", "log">;
def : AIRFloatFunction<"llvm.log2", "log2">;
def : AIRFloatFunction<"llvm.log10", "log10">;
def : AIRFloatFunction<"llvm.sin", "sin">;
def : AIRFloatFunction<"llvm.cos", "cos">;
def : AIRFloatFunction<"llvm.tan", "tan">;
def : AIRFloatFunction<"llvm.asin", "asin">;
def : AIRFloatFunction<"llvm.acos", "acos">;
def : AIRFloatFunction<"llvm.atan", "atan">;
def : AIRFloatFunction<"llvm.atan2", "atan2">;
def : AIRFloatFunction<"llvm.sinh", "sinh">;
def : AIRFloatFunction<"llvm.cosh", "cosh">;
def : AIRFloatFunction<"llvm.tanh", "tanh">;
def : AIRFloatFunction<"llvm.asinh", "asinh">;
def : AIRFloatFunction<"llvm.acosh", "acosh">;
def : AIRFloatFunction<"llvm.atanh", "atanh">;
def : AIRFloatFunction<"llvm.pow", "pow">;
def : AIRFloatFunction<"llvm.minnum", "fmin">;
def : AIRFloatFunction<"llvm.maxnum", "fmax">;
// AIRNaNMinMax has already given these the NaN behaviour of LLVM; what is
// left maps onto fmin/fmax.
let Invertible = 0 in {
def : AIRFloatFunction<"llvm.minimum", "fmin">;
def : AIRFloatFunction<"llvm.maximum", "fmax">;
}

// Floating point, no relaxed variant.
def : AIRFunction<"llvm.fma", "fma">;

// Integer.
def : AIRFunction<"llvm.smin", "min", 1>;
def : AIRFunction<"llvm.smax", "max", 1>;
def : AIRFunction<"llvm.umin", "min", 2>;
def : AIRFunction<"llvm.umax", "max", 2>;
def : AIRFunction<"llvm.abs", "abs", 1, 0, 1>;
def : AIRFunction<"llvm.ctpop", "popcount">;
def : AIRFunction<"llvm.ctlz", "clz">;
def : AIRFunction<"llvm.cttz", "ctz">;
def : AIRFunction<"llvm.bitreverse", "reverse_bits">;
