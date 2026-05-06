//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvvmm'
// extension for integer matrix multiply-accumulate.
// This version is still experimental as the 'Zvvmm' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

class VIMEMACVV<bits<6> funct6, string opcodestr>
    : RVInstVV<funct6, OPIVV, (outs VR:$vd_wb),
               (ins VR:$vd, VR:$vs1, VR:$vs2), opcodestr,
               "$vd, $vs1, $vs2"> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Constraints = "$vd = $vd_wb";
  let vm = 1;
  let VMConstraint = false;
}

let Predicates = [HasStdExtZvvmm] in {
  def VMMACC_VV : VIMEMACVV<0b111000, "vmmacc.vv">;
  def VWMMACC_VV : VIMEMACVV<0b111001, "vwmmacc.vv">;
  def VQMMACC_VV : VIMEMACVV<0b111010, "vqmmacc.vv">;
  def V8WMMACC_VV : VIMEMACVV<0b111011, "v8wmmacc.vv">;
} // Predicates = [HasStdExtZvvmm]
