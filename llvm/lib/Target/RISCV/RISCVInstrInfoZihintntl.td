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
  def PseudoNTLP1   :  Pseudo<(outs), (ins), [], "ntl.p1">, 
                             PseudoInstExpansion<(ADD X0, X0, X2)>;
  def PseudoNTLPALL :  Pseudo<(outs), (ins), [], "ntl.pall">, 
                             PseudoInstExpansion<(ADD X0, X0, X3)>;
  def PseudoNTLS1   :  Pseudo<(outs), (ins), [], "ntl.s1">, 
                             PseudoInstExpansion<(ADD X0, X0, X4)>;
  def PseudoNTLALL  :  Pseudo<(outs), (ins), [], "ntl.all">, 
                             PseudoInstExpansion<(ADD X0, X0, X5)>;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Size = 2 in {
  def PseudoCNTLP1   :  Pseudo<(outs), (ins), [], "c.ntl.p1">,
                              PseudoInstExpansion<(C_ADD_HINT X0, X0, X2)>;
  def PseudoCNTLPALL :  Pseudo<(outs), (ins), [], "c.ntl.pall">,
                              PseudoInstExpansion<(C_ADD_HINT X0, X0, X3)>;
  def PseudoCNTLS1   :  Pseudo<(outs), (ins), [], "c.ntl.s1">,
                              PseudoInstExpansion<(C_ADD_HINT X0, X0, X4)>;
  def PseudoCNTLALL  :  Pseudo<(outs), (ins), [], "c.ntl.all">,
                              PseudoInstExpansion<(C_ADD_HINT X0, X0, X5)>;
}
