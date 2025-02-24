//===-- RISCVInstrInfoXRivos.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by Rivos Inc.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// XRivosVizip
//===----------------------------------------------------------------------===//


let Predicates = [HasVendorXRivosVizip], DecoderNamespace = "XRivos",
  Constraints = "@earlyclobber $vd", RVVConstraint = Vrgather,
  Inst<6-0> = OPC_CUSTOM_2.Value in  {
defm RV_VZIPEVEN_V : VALU_IV_V<"rv.vzipeven", 0b001100>;
defm RV_VZIPODD_V : VALU_IV_V<"rv.vzipodd", 0b011100>;
defm RV_VZIP2A_V : VALU_IV_V<"rv.vzip2a", 0b000100>;
defm RV_VZIP2B_V : VALU_IV_V<"rv.vzip2b", 0b010100>;
defm RV_VUNZIP2A_V : VALU_IV_V<"rv.vunzip2a", 0b001000>;
defm RV_VUNZIP2B_V : VALU_IV_V<"rv.vunzip2b", 0b011000>;
}
