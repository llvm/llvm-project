//===-- RISCVInstrInfoZvqdot.td - 'Zvqdotq' instructions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvqdotq'
// extension.
// This version is still experimental as the 'Zvqdotq' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvqdotq] in {
  def VQDOT_VV   : VALUVV<0b101100, OPMVV, "vqdot.vv">;
  def VQDOT_VX   : VALUVX<0b101100, OPMVX, "vqdot.vx">;
  def VQDOTU_VV  : VALUVV<0b101000, OPMVV, "vqdotu.vv">;
  def VQDOTU_VX  : VALUVX<0b101000, OPMVX, "vqdotu.vx">;
  def VQDOTSU_VV : VALUVV<0b101010, OPMVV, "vqdotsu.vv">;
  def VQDOTSU_VX : VALUVX<0b101010, OPMVX, "vqdotsu.vx">;
  def VQDOTUS_VX : VALUVX<0b101110, OPMVX, "vqdotus.vx">;
} // Predicates = [HasStdExtZvqdotq]
