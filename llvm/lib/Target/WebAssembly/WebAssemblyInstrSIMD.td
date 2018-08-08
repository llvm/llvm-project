// WebAssemblyInstrSIMD.td - WebAssembly SIMD codegen support -*- tablegen -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly SIMD operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

let Defs = [ARGUMENTS] in {

let isCommutable = 1 in
defm ADD : SIMDBinaryInt<add, "add ", 24>;
defm SUB : SIMDBinaryInt<sub, "sub ", 28>;
let isCommutable = 1 in
defm MUL : SIMDBinaryInt<mul, "mul ", 32>;

let isCommutable = 1 in
defm ADD : SIMDBinaryFP<fadd, "add ", 122>;
defm SUB : SIMDBinaryFP<fsub, "sub ", 124>;
defm DIV : SIMDBinaryFP<fdiv, "div ", 126>;
let isCommutable = 1 in
defm MUL : SIMDBinaryFP<fmul, "mul ", 128>;

} // Defs = [ARGUMENTS]
