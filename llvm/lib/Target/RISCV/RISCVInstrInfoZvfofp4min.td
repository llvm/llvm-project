//===- RISCVInstrInfoZvfofp4min.td - 'Zvfofp4min' ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvfofp4min'
// extension, providing vector conversion instructions for OFP4.
// This version is still experimental as the 'Zvfofp4min' extension hasn't been
// ratified yet.
// spec: https://github.com/aswaterman/riscv-misc/blob/4404b28f3f5f5f737281624bf9da14e7649f1854/isa/zvfbfa.adoc#ofp4-conversion-extension-zvfofp4min-version-01
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvfofp4min] in
  defm VFEXT_VF2 : VALU_MV_VS2<"vfext.vf2", 0b010010, 0b10110>;

