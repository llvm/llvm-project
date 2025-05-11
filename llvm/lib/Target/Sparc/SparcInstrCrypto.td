//===----------- SparcInstrCrypto.td - cryptographic extensions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains instruction formats, definitions and patterns needed for
// cryptographic instructions on SPARC.
//===----------------------------------------------------------------------===//


// Convenience template for 4-operand instructions
class FourOpImm<string OpcStr, bits<6> op3val, bits<4> op5val,
        RegisterClass RC>
      : F3_4<op3val, op5val, (outs RC:$rd), (ins RC:$rs1, RC:$rs2, simm5Op:$rs3),
        !strconcat(OpcStr, " $rs1, $rs2, $rs3, $rd")>;

let Predicates = [HasCrypto] in {
def AES_EROUND01 : FourOp<"aes_eround01", 0b011001, 0b0000, DFPRegs>;
def AES_EROUND23 : FourOp<"aes_eround23", 0b011001, 0b0001, DFPRegs>;
def AES_DROUND01 : FourOp<"aes_dround01", 0b011001, 0b0010, DFPRegs>;
def AES_DROUND23 : FourOp<"aes_dround23", 0b011001, 0b0011, DFPRegs>;
def AES_EROUND01_LAST : FourOp<"aes_eround01_l", 0b011001, 0b0100, DFPRegs>;
def AES_EROUND23_LAST : FourOp<"aes_eround23_l", 0b011001, 0b0101, DFPRegs>;
def AES_DROUND01_LAST : FourOp<"aes_dround01_l", 0b011001, 0b0110, DFPRegs>;
def AES_DROUND23_LAST : FourOp<"aes_dround23_l", 0b011001, 0b0111, DFPRegs>;
def AES_KEXPAND0  : F3_3<2, 0b110110, 0b100110000,
                    (outs DFPRegs:$rd), (ins DFPRegs:$rs1, DFPRegs:$rs2),
                    "aes_kexpand0 $rs1, $rs2, $rd", []>;
def AES_KEXPAND1 : FourOpImm<"aes_kexpand1", 0b011001, 0b1000, DFPRegs>;
def AES_KEXPAND2  : F3_3<2, 0b110110, 0b100110001,
                    (outs DFPRegs:$rd), (ins DFPRegs:$rs1, DFPRegs:$rs2),
                    "aes_kexpand2 $rs1, $rs2, $rd", []>;

def CAMELLIA_F : FourOp<"camellia_f", 0b011001, 0b1100, DFPRegs>;
def CAMELLIA_FL  : F3_3<2, 0b110110, 0b100111100,
                    (outs DFPRegs:$rd), (ins DFPRegs:$rs1, DFPRegs:$rs2),
                    "camellia_fl $rs1, $rs2, $rd", []>;
def CAMELLIA_FLI : F3_3<2, 0b110110, 0b100111101,
                    (outs DFPRegs:$rd), (ins DFPRegs:$rs1, DFPRegs:$rs2),
                    "camellia_fli $rs1, $rs2, $rd", []>;

def CRC32C : F3_3<2, 0b110110, 0b101000111,
                    (outs DFPRegs:$rd), (ins DFPRegs:$rs1, DFPRegs:$rs2),
                    "crc32c $rs1, $rs2, $rd", []>;

def DES_ROUND : FourOp<"des_round", 0b011001, 0b1001, DFPRegs>;
let rs2 = 0 in {
def DES_IP  : F3_3<2, 0b110110, 0b100110100,
                    (outs DFPRegs:$rd), (ins DFPRegs:$rs1),
                    "des_ip $rs1, $rd", []>;
def DES_IIP  : F3_3<2, 0b110110, 0b100110101,
                    (outs DFPRegs:$rd), (ins DFPRegs:$rs1),
                    "des_iip $rs1, $rd", []>;
}
def DES_KEXPAND : F3_3<2, 0b110110, 0b100110110,
                    (outs DFPRegs:$rd), (ins DFPRegs:$rs1, simm5Op:$rs2),
                    "des_kexpand $rs1, $rs2, $rd", []>;

let rs1 = 0, rs2 = 0, rd = 0 in {
let Uses = [D0, D1, D2, D5, D6, D7, D8, D9, D10, D11],
    Defs = [D0, D1, D2, D3, D4, D5, D6, D7] in
def MD5 : F3_3<2, 0b110110, 0b101000000, (outs), (ins), "md5", []>;
let Uses = [D0, D1, D2, D4, D5, D6, D7, D8, D9, D10, D11],
    Defs = [D0, D1, D2] in
def SHA1 : F3_3<2, 0b110110, 0b101000001, (outs), (ins), "sha1", []>;
let Uses = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11],
    Defs = [D0, D1, D2, D3] in
def SHA256 : F3_3<2, 0b110110, 0b101000010, (outs), (ins), "sha256", []>;
let Uses = [D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11,
            D12, D13, D14, D15, D16, D17, D18, D19, D20, D21, D22, D23],
    Defs = [D0, D1, D2, D3, D4, D5, D6, D7] in
def SHA512 : F3_3<2, 0b110110, 0b101000011, (outs), (ins), "sha512", []>;
}

// These instructions use and clobber all DFP and non-reserved Int registers.
let rs1 = 0, rd = 0,
Uses = [ D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
         D8,  D9, D10, D11, D12, D13, D14, D15,
        D16, D17, D18, D19, D20, D21, D22, D23,
        D24, D25, D26, D27, D28, D29, D30, D31,
         O0,  O1,  O2,  O3,  O4,  O5,
         L0,  L1,  L2,  L3,  L4,  L5,  L6,  L7,
         I0,  I1,  I2,  I3,  I4,  I5 ],
Defs = [ D0,  D1,  D2,  D3,  D4,  D5,  D6,  D7,
         D8,  D9, D10, D11, D12, D13, D14, D15,
        D16, D17, D18, D19, D20, D21, D22, D23,
        D24, D25, D26, D27, D28, D29, D30, D31,
         O0,  O1,  O2,  O3,  O4,  O5,
         L0,  L1,  L2,  L3,  L4,  L5,  L6,  L7,
         I0,  I1,  I2,  I3,  I4,  I5 ] in {
def MPMUL   : F3_3<2, 0b110110, 0b101001000, (outs), (ins simm5Op:$rs2), "mpmul $rs2", []>;
def MONTMUL : F3_3<2, 0b110110, 0b101001001, (outs), (ins simm5Op:$rs2), "montmul $rs2", []>;
def MONTSQR : F3_3<2, 0b110110, 0b101001010, (outs), (ins simm5Op:$rs2), "montsqr $rs2", []>;
}
} // Predicates = [HasCrypto]
