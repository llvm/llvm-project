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
// RISC-V specific DAG Nodes.
//===----------------------------------------------------------------------===//
 
def SDT_RISCVFP_ROUND_BF16
    : SDTypeProfile<1, 1, [SDTCisVT<0, bf16>, SDTCisVT<1, f32>]>;
def SDT_RISCVFP_EXTEND_BF16
    : SDTypeProfile<1, 1, [SDTCisVT<0, f32>, SDTCisVT<1, bf16>]>;
 
def riscv_fpround_bf16
    : SDNode<"RISCVISD::FP_ROUND_BF16", SDT_RISCVFP_ROUND_BF16>;
def riscv_fpextend_bf16
    : SDNode<"RISCVISD::FP_EXTEND_BF16", SDT_RISCVFP_EXTEND_BF16>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfbfmin] in {
def FCVT_BF16_S : FPUnaryOp_r_frm<0b0100010, 0b01000, FPR16, FPR32, "fcvt.bf16.s">,
                  Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]>;
def FCVT_S_BF16 : FPUnaryOp_r_frm<0b0100000, 0b00110, FPR32, FPR16, "fcvt.s.bf16">,
                  Sched<[WriteFCvtF32ToF16, ReadFCvtF32ToF16]>;
} // Predicates = [HasStdExtZfbfmin]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//
 
let Predicates = [HasStdExtZfbfmin] in {
/// Loads
def : LdPat<load, FLH, bf16>;

/// Stores
def : StPat<store, FSH, FPR16, bf16>;

/// Float conversion operations
// f32 -> bf16, bf16 -> f32
def : Pat<(bf16 (riscv_fpround_bf16 FPR32:$rs1)), 
          (FCVT_BF16_S FPR32:$rs1, FRM_DYN)>;
def : Pat<(riscv_fpextend_bf16 (bf16 FPR16:$rs1)), 
          (FCVT_S_BF16 FPR16:$rs1, FRM_DYN)>;

// Moves (no conversion)
def : Pat<(bf16 (riscv_fmv_h_x GPR:$src)), (FMV_H_X GPR:$src)>;
def : Pat<(riscv_fmv_x_anyexth (bf16 FPR16:$src)), (FMV_X_H FPR16:$src)>;
def : Pat<(riscv_fmv_x_signexth (bf16 FPR16:$src)), (FMV_X_H FPR16:$src)>;
} // Predicates = [HasStdExtZfbfmin]
