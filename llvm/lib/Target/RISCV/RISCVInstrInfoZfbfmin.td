//===-- RISCVInstrInfoZfbfmin.td - 'Zfbfmin' instructions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zfbfmin'
// extension, providing scalar conversion instructions for BFloat16.
// This version is still experimental as the 'Zfbfmin' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfbfmin] in {
def FCVT_BF16_S : FPUnaryOp_r_frm<0b0100010, 0b01000, FPR16, FPR32, "fcvt.bf16.s">,
                  Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]>;
def FCVT_S_BF16 : FPUnaryOp_r_frm<0b0100000, 0b00110, FPR32, FPR16, "fcvt.s.bf16">,
                  Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]>;
} // Predicates = [HasStdExtZfbfmin]
