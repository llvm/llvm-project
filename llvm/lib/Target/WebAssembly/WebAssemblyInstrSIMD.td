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
/// \brief WebAssembly SIMD operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

let isCommutable = 1 in {
defm ADD : SIMDBinary<add, fadd, "add ">;
defm MUL: SIMDBinary<mul, fmul, "mul ">;
} // isCommutable = 1
defm SUB: SIMDBinary<sub, fsub, "sub ">;
