//===-- RISCVInstrInfoZfa.td - RISC-V 'Zfa' instructions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zfa'
// additional floating-point extension, version 0.1.
// This version is still experimental as the 'Zfa' extension hasn't been
// ratified yet.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

// 5-bit floating-point immediate encodings.
def LoadFPImmOperand : AsmOperandClass {
  let Name = "LoadFPImm";
  let ParserMethod = "parseFPImm";
  let RenderMethod = "addFPImmOperands";
  let DiagnosticType = "InvalidLoadFPImm";
}

def loadfp32imm : Operand<f32> {
  let ParserMatchClass = LoadFPImmOperand;
  let PrintMethod = "printFPImmOperand";
}

def loadfp64imm : Operand<f64> {
  let ParserMatchClass = LoadFPImmOperand;
  let PrintMethod = "printFPImmOperand";
}

def loadfp16imm : Operand<f16> {
  let ParserMatchClass = LoadFPImmOperand;
  let PrintMethod = "printFPImmOperand";
}

def RTZArg : AsmOperandClass {
  let Name = "RTZArg";
  let RenderMethod = "addFRMArgOperands";
  let DiagnosticType = "InvalidRTZArg";
  let ParserMethod = "parseFRMArg";
}

def rtzarg : Operand<XLenVT> {
  let ParserMatchClass = RTZArg;
  let PrintMethod = "printFRMArg";
  let DecoderMethod = "decodeFRMArg";
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1 in
class FPBinaryOp_rr<bits<7> funct7, bits<3> funct3, DAGOperand rdty,
                    DAGOperand rsty, string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP_FP, (outs rdty:$rd),
              (ins rsty:$rs1, rsty:$rs2), opcodestr, "$rd, $rs1, $rs2">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class FPUnaryOp_imm<bits<7> funct7, bits<5> rs2val, bits<3> funct3, RISCVOpcode opcode,
                    dag outs, dag ins, string opcodestr, string argstr>
    : RVInst<outs, ins, opcodestr, argstr, [], InstFormatI> {
  bits<5> imm;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = rs2val;
  let Inst{19-15} = imm;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Opcode = opcode.Value;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, mayRaiseFPException = 1,
    UseNamedOperandTable = 1, hasPostISelHook = 1 in
class FPUnaryOp_r_rtz<bits<7> funct7, bits<5> rs2val, DAGOperand rdty,
                      DAGOperand rs1ty, string opcodestr>
    : RVInstRFrm<funct7, OPC_OP_FP, (outs rdty:$rd),
                 (ins rs1ty:$rs1, rtzarg:$frm), opcodestr,
                  "$rd, $rs1$frm"> {
  let rs2 = rs2val;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfa] in {
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def FLI_S : FPUnaryOp_imm<0b1111000, 0b00001, 0b000, OPC_OP_FP, (outs FPR32:$rd),
            (ins loadfp32imm:$imm), "fli.s", "$rd, $imm">;

def FMINM_S: FPALU_rr<0b0010100, 0b010, "fminm.s", FPR32, /*Commutable*/ 1>;
def FMAXM_S: FPALU_rr<0b0010100, 0b011, "fmaxm.s", FPR32, /*Commutable*/ 1>;

def FROUND_S : FPUnaryOp_r_frm<0b0100000, 0b00100, FPR32, FPR32, "fround.s">;
def FROUNDNX_S : FPUnaryOp_r_frm<0b0100000, 0b00101, FPR32, FPR32, "froundnx.s">;

def FLTQ_S : FPCmp_rr<0b1010000, 0b101, "fltq.s", FPR32>;
def FLEQ_S : FPCmp_rr<0b1010000, 0b100, "fleq.s", FPR32>;
} // Predicates = [HasStdExtZfa]

let Predicates = [HasStdExtZfa, HasStdExtD] in {
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def FLI_D : FPUnaryOp_imm<0b1111001, 0b00001, 0b000, OPC_OP_FP, (outs FPR64:$rd),
            (ins loadfp64imm:$imm), "fli.d", "$rd, $imm">;

def FMINM_D: FPALU_rr<0b0010101, 0b010, "fminm.d", FPR64, /*Commutable*/ 1>;
def FMAXM_D: FPALU_rr<0b0010101, 0b011, "fmaxm.d", FPR64, /*Commutable*/ 1>;

def FROUND_D : FPUnaryOp_r_frm<0b0100001, 0b00100, FPR64, FPR64, "fround.d">;
def FROUNDNX_D : FPUnaryOp_r_frm<0b0100001, 0b00101, FPR64, FPR64, "froundnx.d">;

def FCVTMOD_W_D
    : FPUnaryOp_r_rtz<0b1100001, 0b01000, GPR, FPR64, "fcvtmod.w.d">,
      Sched<[WriteFCvtF64ToI32, ReadFCvtF64ToI32]>;

def FLTQ_D : FPCmp_rr<0b1010001, 0b101, "fltq.d", FPR64>;
def FLEQ_D : FPCmp_rr<0b1010001, 0b100, "fleq.d", FPR64>;
} // Predicates = [HasStdExtZfa, HasStdExtD]

let Predicates = [HasStdExtZfa, HasStdExtD, IsRV32] in {
let mayRaiseFPException = 0 in {
def FMVH_X_D : FPUnaryOp_r<0b1110001, 0b00001, 0b000, GPR, FPR64, "fmvh.x.d">,
               Sched<[WriteFMovF32ToI32, ReadFMovF32ToI32]>;
def FMVP_D_X : FPBinaryOp_rr<0b1011001, 0b000, FPR64, GPR, "fmvp.d.x">,
               Sched<[WriteFMovI32ToF32, ReadFMovI32ToF32]>;
}

let isCodeGenOnly = 1, mayRaiseFPException = 0 in
def FMV_X_W_FPR64 : FPUnaryOp_r<0b1110000, 0b00000, 0b000, GPR, FPR64,
                                "fmv.x.w">,
                    Sched<[WriteFMovF32ToI32, ReadFMovF32ToI32]>;
} // Predicates = [HasStdExtZfa, HasStdExtD, IsRV32]

let Predicates = [HasStdExtZfa, HasStdExtZfhOrZvfh] in
let isReMaterializable = 1, isAsCheapAsAMove = 1 in
def FLI_H : FPUnaryOp_imm<0b1111010, 0b00001, 0b000, OPC_OP_FP, (outs FPR16:$rd),
            (ins loadfp16imm:$imm), "fli.h", "$rd, $imm">;

let Predicates = [HasStdExtZfa, HasStdExtZfh] in {
def FMINM_H: FPALU_rr<0b0010110, 0b010, "fminm.h", FPR16, /*Commutable*/ 1>;
def FMAXM_H: FPALU_rr<0b0010110, 0b011, "fmaxm.h", FPR16, /*Commutable*/ 1>;

def FROUND_H : FPUnaryOp_r_frm<0b0100010, 0b00100, FPR16, FPR16, "fround.h">;
def FROUNDNX_H : FPUnaryOp_r_frm<0b0100010, 0b00101, FPR16, FPR16, "froundnx.h">;

def FLTQ_H : FPCmp_rr<0b1010010, 0b101, "fltq.h", FPR16>;
def FLEQ_H : FPCmp_rr<0b1010010, 0b100, "fleq.h", FPR16>;
} // Predicates = [HasStdExtZfa, HasStdExtZfh]

//===----------------------------------------------------------------------===//
// Pseudo-instructions and codegen patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfa] in {
def : InstAlias<"fgtq.s $rd, $rs, $rt",
                (FLTQ_S GPR:$rd, FPR32:$rt, FPR32:$rs), 0>;
def : InstAlias<"fgeq.s $rd, $rs, $rt",
                (FLEQ_S GPR:$rd, FPR32:$rt, FPR32:$rs), 0>;
}

let Predicates = [HasStdExtZfa, HasStdExtD] in {
def : InstAlias<"fgtq.d $rd, $rs, $rt",
                (FLTQ_D GPR:$rd, FPR64:$rt, FPR64:$rs), 0>;
def : InstAlias<"fgeq.d $rd, $rs, $rt",
                (FLEQ_D GPR:$rd, FPR64:$rt, FPR64:$rs), 0>;
}

let Predicates = [HasStdExtZfa, HasStdExtZfh] in {
def : InstAlias<"fgtq.h $rd, $rs, $rt",
                (FLTQ_H GPR:$rd, FPR16:$rt, FPR16:$rs), 0>;
def : InstAlias<"fgeq.h $rd, $rs, $rt",
                (FLEQ_H GPR:$rd, FPR16:$rt, FPR16:$rs), 0>;
}

//===----------------------------------------------------------------------===//
// Codegen patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZfa] in {
def: PatFprFpr<fminimum, FMINM_S, FPR32>;
def: PatFprFpr<fmaximum, FMAXM_S, FPR32>;

// frint rounds according to the current rounding mode and detects
// inexact conditions.
def: Pat<(any_frint FPR32:$rs1), (FROUNDNX_S FPR32:$rs1, 0b111)>;

// fnearbyint is like frint but does not detect inexact conditions.
def: Pat<(any_fnearbyint FPR32:$rs1), (FROUND_S FPR32:$rs1, 0b111)>;

def: Pat<(any_fround FPR32:$rs1), (FROUND_S FPR32:$rs1, 0b100)>;
def: Pat<(any_ffloor FPR32:$rs1), (FROUND_S FPR32:$rs1, 0b011)>;
def: Pat<(any_fceil FPR32:$rs1), (FROUND_S FPR32:$rs1, 0b010)>;
def: Pat<(any_ftrunc FPR32:$rs1), (FROUND_S FPR32:$rs1, 0b001)>;

def: PatSetCC<FPR32, strict_fsetcc, SETLT, FLTQ_S>;
def: PatSetCC<FPR32, strict_fsetcc, SETOLT, FLTQ_S>;
def: PatSetCC<FPR32, strict_fsetcc, SETLE, FLEQ_S>;
def: PatSetCC<FPR32, strict_fsetcc, SETOLE, FLEQ_S>;
} // Predicates = [HasStdExtZfa]

let Predicates = [HasStdExtZfa, HasStdExtD] in {
def: PatFprFpr<fminimum, FMINM_D, FPR64>;
def: PatFprFpr<fmaximum, FMAXM_D, FPR64>;

// frint rounds according to the current rounding mode and detects
// inexact conditions.
def: Pat<(any_frint FPR64:$rs1), (FROUNDNX_D FPR64:$rs1, 0b111)>;

// fnearbyint is like frint but does not detect inexact conditions.
def: Pat<(any_fnearbyint FPR64:$rs1), (FROUND_D FPR64:$rs1, 0b111)>;

def: Pat<(any_fround FPR64:$rs1), (FROUND_D FPR64:$rs1, 0b100)>;
def: Pat<(any_froundeven FPR64:$rs1), (FROUND_D FPR64:$rs1, 0b000)>;
def: Pat<(any_ffloor FPR64:$rs1), (FROUND_D FPR64:$rs1, 0b011)>;
def: Pat<(any_fceil FPR64:$rs1), (FROUND_D FPR64:$rs1, 0b010)>;
def: Pat<(any_ftrunc FPR64:$rs1), (FROUND_D FPR64:$rs1, 0b001)>;

def: PatSetCC<FPR64, strict_fsetcc, SETLT, FLTQ_D>;
def: PatSetCC<FPR64, strict_fsetcc, SETOLT, FLTQ_D>;
def: PatSetCC<FPR64, strict_fsetcc, SETLE, FLEQ_D>;
def: PatSetCC<FPR64, strict_fsetcc, SETOLE, FLEQ_D>;
} // Predicates = [HasStdExtZfa, HasStdExtD]

let Predicates = [HasStdExtZfa, HasStdExtD, IsRV32] in {
def : Pat<(RISCVBuildPairF64 GPR:$rs1, GPR:$rs2),
          (FMVP_D_X GPR:$rs1, GPR:$rs2)>;
}

let Predicates = [HasStdExtZfa, HasStdExtZfh] in {
def: PatFprFpr<fminimum, FMINM_H, FPR16>;
def: PatFprFpr<fmaximum, FMAXM_H, FPR16>;

// frint rounds according to the current rounding mode and detects
// inexact conditions.
def: Pat<(any_frint FPR16:$rs1), (FROUNDNX_H FPR16:$rs1, 0b111)>;

// fnearbyint is like frint but does not detect inexact conditions.
def: Pat<(any_fnearbyint FPR16:$rs1), (FROUND_H FPR16:$rs1, 0b111)>;

def: Pat<(any_fround FPR16:$rs1), (FROUND_H FPR16:$rs1, 0b100)>;
def: Pat<(any_froundeven FPR16:$rs1), (FROUND_H FPR16:$rs1, 0b000)>;
def: Pat<(any_ffloor FPR16:$rs1), (FROUND_H FPR16:$rs1, 0b011)>;
def: Pat<(any_fceil FPR16:$rs1), (FROUND_H FPR16:$rs1, 0b010)>;
def: Pat<(any_ftrunc FPR16:$rs1), (FROUND_H FPR16:$rs1, 0b001)>;

def: PatSetCC<FPR16, strict_fsetcc, SETLT, FLTQ_H>;
def: PatSetCC<FPR16, strict_fsetcc, SETOLT, FLTQ_H>;
def: PatSetCC<FPR16, strict_fsetcc, SETLE, FLEQ_H>;
def: PatSetCC<FPR16, strict_fsetcc, SETOLE, FLEQ_H>;
} // Predicates = [HasStdExtZfa, HasStdExtZfh]
