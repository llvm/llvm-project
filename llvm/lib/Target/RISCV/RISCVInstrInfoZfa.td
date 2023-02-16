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
def FMINM_S: FPALU_rr<0b0010100, 0b010, "fminm.s", FPR32, /*Commutable*/ 1>;
def FMAXM_S: FPALU_rr<0b0010100, 0b011, "fmaxm.s", FPR32, /*Commutable*/ 1>;

def FROUND_S : FPUnaryOp_r_frm<0b0100000, 0b00100, FPR32, FPR32, "fround.s">;
def FROUNDNX_S : FPUnaryOp_r_frm<0b0100000, 0b00101, FPR32, FPR32, "froundnx.s">;

def FLTQ_S : FPCmp_rr<0b1010000, 0b101, "fltq.s", FPR32, /*Commutable*/ 1>;
def FLEQ_S : FPCmp_rr<0b1010000, 0b100, "fleq.s", FPR32, /*Commutable*/ 1>;
} // Predicates = [HasStdExtZfa]

let Predicates = [HasStdExtZfa, HasStdExtD] in {
def FMINM_D: FPALU_rr<0b0010101, 0b010, "fminm.d", FPR64, /*Commutable*/ 1>;
def FMAXM_D: FPALU_rr<0b0010101, 0b011, "fmaxm.d", FPR64, /*Commutable*/ 1>;

def FROUND_D : FPUnaryOp_r_frm<0b0100001, 0b00100, FPR64, FPR64, "fround.d">;
def FROUNDNX_D : FPUnaryOp_r_frm<0b0100001, 0b00101, FPR64, FPR64, "froundnx.d">;

def FCVTMOD_W_D
    : FPUnaryOp_r_rtz<0b1100001, 0b01000, GPR, FPR64, "fcvtmod.w.d">,
      Sched<[WriteFCvtF64ToI32, ReadFCvtF64ToI32]>;

def FLTQ_D : FPCmp_rr<0b1010001, 0b101, "fltq.d", FPR64, /*Commutable*/ 1>;
def FLEQ_D : FPCmp_rr<0b1010001, 0b100, "fleq.d", FPR64, /*Commutable*/ 1>;
} // Predicates = [HasStdExtZfa, HasStdExtD]

let Predicates = [HasStdExtZfa, HasStdExtD, IsRV32] in {
def FMVH_X_D : FPUnaryOp_r<0b1110001, 0b00001, 0b000, GPR, FPR64, "fmvh.x.d">,
               Sched<[WriteFMovF32ToI32, ReadFMovF32ToI32]>;
def FMVP_D_X : FPBinaryOp_rr<0b1011001, 0b000, FPR64, GPR, "fmvp.d.x">,
               Sched<[WriteFMovI32ToF32, ReadFMovI32ToF32]>;
} // Predicates = [HasStdExtZfa, HasStdExtD, IsRV32]

let Predicates = [HasStdExtZfa, HasStdExtZfh] in {
def FMINM_H: FPALU_rr<0b0010110, 0b010, "fminm.h", FPR16, /*Commutable*/ 1>;
def FMAXM_H: FPALU_rr<0b0010110, 0b011, "fmaxm.h", FPR16, /*Commutable*/ 1>;

def FROUND_H : FPUnaryOp_r_frm<0b0100010, 0b00100, FPR16, FPR16, "fround.h">;
def FROUNDNX_H : FPUnaryOp_r_frm<0b0100010, 0b00101, FPR16, FPR16, "froundnx.h">;

def FLTQ_H : FPCmp_rr<0b1010010, 0b101, "fltq.h", FPR16, /*Commutable*/ 1>;
def FLEQ_H : FPCmp_rr<0b1010010, 0b100, "fleq.h", FPR16, /*Commutable*/ 1>;
} // Predicates = [HasStdExtZfa, HasStdExtZfh]
