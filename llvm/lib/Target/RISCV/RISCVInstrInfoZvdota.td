//===-- RISCVInstrInfoZvdota.td - 'Zvdota' instructions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions for 'Zvdota'.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Definitions
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// op vd, vs2, vs1, vm
class ZvdotVALUVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd_wb),
               (ins VR:$vd, VR:$vs2, VR:$vs1, VMaskOp:$vm),
               opcodestr, "$vd, $vs2, $vs1$vm"> {
  let Inst{6-0} = OPC_OP_VE.Value;
  let Constraints = "@earlyclobber $vd_wb, $vd = $vd_wb";
  let VS1VS2Constraint = Vrgather;
}
}

let Predicates = [HasStdExtZvqwdota8iOrZvqwdota16i], DestEEW = EEWSEWx4 in {
  def VQWDOTAU_VV  : ZvdotVALUVV<0b100110, OPIVV, "vqwdotau.vv">;
  def VQWDOTAS_VV  : ZvdotVALUVV<0b100111, OPIVV, "vqwdotas.vv">;
} // Predicates = [HasStdExtZvqwdota8iOrZvqwdota16i]

let Predicates = [HasStdExtZvfqwdota8f],
    mayRaiseFPException = true, DestEEW = EEWSEWx4 in {
  def VFQWDOTA_VV : ZvdotVALUVV<0b100110, OPFVV, "vfqwdota.vv">;
  def VFQWDOTA_ALT_VV : ZvdotVALUVV<0b100111, OPFVV, "vfqwdota.alt.vv">;
} // Predicates = [HasStdExtZvfqwdota8f]

let Predicates = [HasStdExtZvfwdota16bf],
    mayRaiseFPException = true, DestEEW = EEWSEWx2 in {
  def VFWDOTA_VV : ZvdotVALUVV<0b100100, OPFVV, "vfwdota.vv">;
} // Predicates = [HasStdExtZvfwdota16bf]
