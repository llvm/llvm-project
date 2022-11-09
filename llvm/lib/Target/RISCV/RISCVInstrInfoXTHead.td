//===-- RISCVInstrInfoXTHead.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by T-Head of Alibaba.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//
class THInstVdotVV<bits<6> funct6, RISCVVFormat opv, dag outs, dag ins,
                   string opcodestr, string argstr>
    : RVInstVV<funct6, opv, outs, ins, opcodestr, argstr> {
  let Inst{26} = 0;
  let Opcode = OPC_CUSTOM_0.Value;
  let DecoderNamespace = "THeadV";
}

class THInstVdotVX<bits<6> funct6, RISCVVFormat opv, dag outs, dag ins,
                   string opcodestr, string argstr>
    : RVInstVX<funct6, opv, outs, ins, opcodestr, argstr> {
  let Inst{26} = 1;
  let Opcode = OPC_CUSTOM_0.Value;
  let DecoderNamespace = "THeadV";
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// op vd, vs1, vs2, vm (reverse the order of vs1 and vs2)
class THVdotALUrVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : THInstVdotVV<funct6, opv, (outs VR:$vd),
                   (ins VR:$vs1, VR:$vs2, VMaskOp:$vm),
                   opcodestr, "$vd, $vs1, $vs2$vm">;

// op vd, rs1, vs2, vm (reverse the order of rs1 and vs2)
class THVdotALUrVX<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : THInstVdotVX<funct6, opv, (outs VR:$vd),
                   (ins GPR:$rs1, VR:$vs2, VMaskOp:$vm),
                   opcodestr, "$vd, $rs1, $vs2$vm">;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===----------------------------------------------------------------------===//
// Combination of instruction classes.
// Use these multiclasses to define instructions more easily.
//===----------------------------------------------------------------------===//
multiclass THVdotVMAQA_VX<string opcodestr, bits<6> funct6> {
  def _VX : THVdotALUrVX<funct6, OPMVX, opcodestr # ".vx">;
}

multiclass THVdotVMAQA<string opcodestr, bits<6> funct6> {
  def _VV   : THVdotALUrVV<funct6, OPMVX, opcodestr # ".vv">;
  defm ""   : THVdotVMAQA_VX<opcodestr, funct6>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasVendorXTHeadVdot],
    Constraints = "@earlyclobber $vd",
    RVVConstraint = WidenV in {
defm THVdotVMAQA      : THVdotVMAQA<"th.vmaqa",     0b100000>;
defm THVdotVMAQAU     : THVdotVMAQA<"th.vmaqau",    0b100010>;
defm THVdotVMAQASU    : THVdotVMAQA<"th.vmaqasu",   0b100100>;
defm THVdotVMAQAUS    : THVdotVMAQA_VX<"th.vmaqaus",0b100110>;
}
