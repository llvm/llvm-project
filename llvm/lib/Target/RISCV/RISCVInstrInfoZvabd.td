//===-- RISCVInstrInfoZvabd.td - 'Zvabd' instructions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions for 'Zvabd' (Vector Absolute
/// Difference).
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Definitions
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtZvabd] in {
  defm VABS_V : VALU_MV_VS2<"vabs.v", 0b010010, 0b10000>;

  def VABD_VV  : VALUVV<0b010001, OPMVV, "vabd.vv">;
  def VABDU_VV : VALUVV<0b010011, OPMVV, "vabdu.vv">;

  let Constraints = "@earlyclobber $vd", RVVConstraint = WidenV in {
    def VWABDA_VV  : VALUVV<0b010101, OPMVV, "vwabda.vv">;
    def VWABDAU_VV : VALUVV<0b010110, OPMVV, "vwabdau.vv">;
  } // Constraints = "@earlyclobber $vd", RVVConstraint = WidenV
} // Predicates = [HasStdExtZvabd]
