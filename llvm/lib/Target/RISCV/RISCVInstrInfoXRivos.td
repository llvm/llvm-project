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
defm RI_VZIPEVEN_V : VALU_IV_V<"ri.vzipeven", 0b001100>;
defm RI_VZIPODD_V : VALU_IV_V<"ri.vzipodd", 0b011100>;
defm RI_VZIP2A_V : VALU_IV_V<"ri.vzip2a", 0b000100>;
defm RI_VZIP2B_V : VALU_IV_V<"ri.vzip2b", 0b010100>;
defm RI_VUNZIP2A_V : VALU_IV_V<"ri.vunzip2a", 0b001000>;
defm RI_VUNZIP2B_V : VALU_IV_V<"ri.vunzip2b", 0b011000>;
}
