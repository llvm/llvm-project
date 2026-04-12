//===---- RISCVInstrInfoZvzip.td - 'Zvzip' instructions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the Zvzip standard extension
// for reordering structured data in vector registers.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvzip], Constraints = "@earlyclobber $vd" in {
  let VS1VS2Constraint = WidenV in
  def VZIP_VV : VALUVV<0b111110, OPMVV, "vzip.vv">;

  let VS1VS2Constraint = VS2Constraint in {
    def VUNZIPE_V : VALUVs2<0b010010, 0b01011, OPMVV, "vunzipe.v">;
    def VUNZIPO_V : VALUVs2<0b010010, 0b01111, OPMVV, "vunzipo.v">;
  }

  let VS1VS2Constraint = Vrgather in {
    def VPAIRE_VV : VALUVV<0b001111, OPIVV, "vpaire.vv">;
    def VPAIRO_VV : VALUVV<0b001111, OPMVV, "vpairo.vv">;
  }
} // Predicates = [HasStdExtZvzip]
