//===- RISCVInstrInfoZvfofp8min.td - 'Zvfofp8min' ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvfofp8min'
// extension, providing vector conversion instructions for OFP8.
// This version is still experimental as the 'Zvfofp8min' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvfofp8min], Constraints = "@earlyclobber $vd",
    mayRaiseFPException = true, Uses = [FRM, VL, VTYPE] in {
  defm VFNCVTBF16_SAT_F_F_W
      : VNCVTF_FV_VS2<"vfncvtbf16.sat.f.f.w", 0b010010, 0b11111>;
  defm VFNCVT_F_F_Q : VNCVTF_FV_VS2<"vfncvt.f.f.q", 0b010010, 0b11001>;
  defm VFNCVT_SAT_F_F_Q : VNCVTF_FV_VS2<"vfncvt.sat.f.f.q", 0b010010, 0b11011>;
}
