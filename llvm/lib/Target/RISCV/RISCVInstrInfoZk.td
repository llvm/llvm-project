//===- RISCVInstrInfoZk.td - RISC-V Scalar Crypto instructions - tablegen -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V instructions from the standard 'Zk',
// Scalar Cryptography Instructions extension, version 1.0.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand and SDNode transformation definitions.
//===----------------------------------------------------------------------===//

def RnumArg : AsmOperandClass {
  let Name = "RnumArg";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidRnumArg";
}

def rnum : Operand<XLenVT>, ImmLeaf<XLenVT, [{return (Imm >= 0 && Imm <= 10);}]> {
  let ParserMatchClass = RnumArg;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<4>";
  let OperandType = "OPERAND_RVKRNUM";
  let OperandNamespace = "RISCVOp";
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//
let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVKUnary<bits<12> imm12_in, bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_OP_IMM, (outs GPR:$rd), (ins GPR:$rs1),
              opcodestr, "$rd, $rs1">{
  let imm12 = imm12_in;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVKByteSelect<bits<5> funct5, string opcodestr>
    : RVInstR<{0b00, funct5}, 0b000, OPC_OP, (outs GPR:$rd),
              (ins GPR:$rs1, GPR:$rs2, uimm2:$bs),
              opcodestr, "$rd, $rs1, $rs2, $bs">{
  bits<2> bs;
  let Inst{31-30} = bs;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class RVKUnary_rnum<bits<7> funct7, bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_OP_IMM, (outs GPR:$rd), (ins GPR:$rs1, rnum:$rnum),
              opcodestr, "$rd, $rs1, $rnum">{
    bits<4> rnum;
    let Inst{31-25} = funct7;
    let Inst{24} = 1;
    let Inst{23-20} = rnum;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//
let Predicates = [HasStdExtZknd, IsRV32] in {
def AES32DSI  : RVKByteSelect<0b10101, "aes32dsi">;
def AES32DSMI : RVKByteSelect<0b10111, "aes32dsmi">;
} // Predicates = [HasStdExtZknd, IsRV32]

let Predicates = [HasStdExtZknd, IsRV64] in {
def AES64DS  : ALU_rr<0b0011101, 0b000, "aes64ds">;
def AES64DSM : ALU_rr<0b0011111, 0b000, "aes64dsm">;

def AES64IM  : RVKUnary<0b001100000000, 0b001, "aes64im">;
} // Predicates = [HasStdExtZknd, IsRV64]

let Predicates = [HasStdExtZkndOrZkne, IsRV64] in {
def AES64KS2  : ALU_rr<0b0111111, 0b000, "aes64ks2">;

def AES64KS1I : RVKUnary_rnum<0b0011000, 0b001, "aes64ks1i">;
} // Predicates = [HasStdExtZkndOrZkne, IsRV64]

let Predicates = [HasStdExtZkne, IsRV32] in {
def AES32ESI  : RVKByteSelect<0b10001, "aes32esi">;
def AES32ESMI : RVKByteSelect<0b10011, "aes32esmi">;
} // Predicates = [HasStdExtZkne, IsRV32]

let Predicates = [HasStdExtZkne, IsRV64] in {
def AES64ES   : ALU_rr<0b0011001, 0b000, "aes64es">;
def AES64ESM  : ALU_rr<0b0011011, 0b000, "aes64esm">;
} // Predicates = [HasStdExtZkne, IsRV64]

let Predicates = [HasStdExtZknh] in {
def SHA256SIG0 : RVKUnary<0b000100000010, 0b001, "sha256sig0">;
def SHA256SIG1 : RVKUnary<0b000100000011, 0b001, "sha256sig1">;
def SHA256SUM0 : RVKUnary<0b000100000000, 0b001, "sha256sum0">;
def SHA256SUM1 : RVKUnary<0b000100000001, 0b001, "sha256sum1">;
} // Predicates = [HasStdExtZknh]

let Predicates = [HasStdExtZknh, IsRV32] in {
def SHA512SIG0H : ALU_rr<0b0101110, 0b000, "sha512sig0h">;
def SHA512SIG0L : ALU_rr<0b0101010, 0b000, "sha512sig0l">;
def SHA512SIG1H : ALU_rr<0b0101111, 0b000, "sha512sig1h">;
def SHA512SIG1L : ALU_rr<0b0101011, 0b000, "sha512sig1l">;
def SHA512SUM0R : ALU_rr<0b0101000, 0b000, "sha512sum0r">;
def SHA512SUM1R : ALU_rr<0b0101001, 0b000, "sha512sum1r">;
} // [HasStdExtZknh, IsRV32]

let Predicates = [HasStdExtZknh, IsRV64] in {
def SHA512SIG0 : RVKUnary<0b000100000110, 0b001, "sha512sig0">;
def SHA512SIG1 : RVKUnary<0b000100000111, 0b001, "sha512sig1">;
def SHA512SUM0 : RVKUnary<0b000100000100, 0b001, "sha512sum0">;
def SHA512SUM1 : RVKUnary<0b000100000101, 0b001, "sha512sum1">;
} // Predicates = [HasStdExtZknh, IsRV64]

let Predicates = [HasStdExtZksed] in {
def SM4ED : RVKByteSelect<0b11000, "sm4ed">;
def SM4KS : RVKByteSelect<0b11010, "sm4ks">;
} // Predicates = [HasStdExtZksed]

let Predicates = [HasStdExtZksh] in {
def SM3P0 : RVKUnary<0b000100001000, 0b001, "sm3p0">;
def SM3P1 : RVKUnary<0b000100001001, 0b001, "sm3p1">;
} // Predicates = [HasStdExtZksh]
