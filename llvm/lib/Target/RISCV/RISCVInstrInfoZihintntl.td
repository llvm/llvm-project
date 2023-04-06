//===RISCVInstrInfoZihintntl.td - 'Zihintntl' instructions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions from Non-Temporal Locality 
/// Hints extension document (zihintntl).
///
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 4 in {
  def PseudoNTLALL :  Pseudo<(outs), (ins), [], "ntl.all">, 
                             PseudoInstExpansion<(ADD X0, X0, X5)>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 2 in {
  def PseudoCNTLALL :  Pseudo<(outs), (ins), [], "c.ntl.all">,
                              PseudoInstExpansion<(C_ADD_HINT X0, X0, X5)>;
}
