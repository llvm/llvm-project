//===- RISCVInstrInfoXSfvfofp4min.td - 'XSfvfofp4min' -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the SiFive custom 'XSfvfofp4min' extension, providing
// vector conversion instructions for OFP4.
// spec: https://www.sifive.com/document-file/ofp4-conversion-instruction-xsfvfofp4min-extension
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXSfvfofp4min], AltFmtType = IS_NOT_ALTFMT in
  def VFEXT_VF2 : VEXT_MV_VS2<"sf.vfext.vf2", 0b010010, 0b10110>;
