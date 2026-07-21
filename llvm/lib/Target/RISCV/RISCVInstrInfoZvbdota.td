//===-- RISCVInstrInfoZvbdota.td - 'Zvbdota' instructions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zvbdota'
// extension, providing vector batched dot-product instructions.
// This version is still experimental as the 'Zvbdota' extension hasn't been
// ratified yet. Version 0.2.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction class definitions
//===----------------------------------------------------------------------===//

// Base class for Zvbdota integer instructions (uses OPC_OP_VE)
let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    VS1VS2Constraint = Vrgather, Constraints = "@earlyclobber $vd_wb, $vd = $vd_wb" in
class ZvbdotaVV<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd_wb),
               (ins VR:$vd, VRM8:$vs2, VR:$vs1, uimm3:$ci, VMaskOp:$vm),
               opcodestr, "$vd, $vs2, $vs1, $ci$vm"> {
  bits<3> ci;
  let Inst{24-23} = vs2{4-3};
  let Inst{22-20} = ci;
  let Inst{6-0} = OPC_OP_VE.Value;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// Integer batched dot-product
multiclass VBdotVV<string opcodestr, bits<6> funct6> {
  def _VV : ZvbdotaVV<funct6, OPIVV, opcodestr>;
}

// Floating-point batched dot-product
multiclass VBdotFVV<string opcodestr, bits<6> funct6> {
  def _VV : ZvbdotaVV<funct6, OPFVV, opcodestr>;
}

// Integer Batched Dot-Product Instructions for 8-bit elements
let Predicates = [HasStdExtZvqwbdota8iOrZvqwbdota16i] in {
defm VQWBDOTAU : VBdotVV<"vqwbdotau.vv", 0b101110>;  // Unsigned (funct6 = 0x2e)
defm VQWBDOTAS : VBdotVV<"vqwbdotas.vv", 0b101111>;  // Signed (funct6 = 0x2f)
} // Predicates = [HasStdExtZvqwbdota8iOrZvqwbdota16i]

// BF16 Batched Dot-Product Instructions
let Predicates = [HasStdExtZvfwbdota16bf] in
defm VFWBDOTA : VBdotFVV<"vfwbdota.vv", 0b101100>; // funct6 = 0x2c

// OCP FP8 Batched Dot-Product Instructions
let Predicates = [HasStdExtZvfqwbdota8f] in {
defm VFQWBDOTA : VBdotFVV<"vfqwbdota.vv", 0b101110>;         // E4M3 (funct6 = 0x2e)
defm VFQWBDOTA_ALT : VBdotFVV<"vfqwbdota.alt.vv", 0b101111>; // E5M2 (funct6 = 0x2f)
} // Predicates = [HasStdExtZvfqwbdota8f]

// FP32 Batched Dot-Product Instructions
let Predicates = [HasStdExtZvfbdota32f] in
defm VFBDOTA : VBdotFVV<"vfbdota.vv", 0b101011>;  // funct6 = 0x2b
