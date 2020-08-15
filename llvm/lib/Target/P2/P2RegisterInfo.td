//===-- P2RegisterInfo.td - P2 Register defs -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the P2 register file
//
//  propeller doesn't have a real register file, other than it's 16 special
//  registers at the end of COG ram. We'll extend that to include 16 general
//  purpose registers before that, called r0-r31
//
//
//===----------------------------------------------------------------------===//

// each register is a 9 bit address
class P2Reg<bits<9> Enc, string n> : Register<n> {
    let HWEncoding{8-0} = Enc; // encode each register by it's cog RAM address

    let Namespace = "P2";
}

//===----------------------------------------------------------------------===//
//@Registers
//===----------------------------------------------------------------------===//
// The register string, such as "9" or "gp" will show on "llvm-objdump -d"
//@ All registers definition
let Namespace = "P2" in {
    //@ General Purpose Registers
    // this will be modified as we adjust how the backend will work since each of the 512 "registers" can store either instructions
    // or data

    def R0  : P2Reg<0x1d0,  "r0">,  DwarfRegNum<[0]>;
    def R1  : P2Reg<0x1d1,  "r1">,  DwarfRegNum<[1]>;
    def R2  : P2Reg<0x1d2,  "r2">,  DwarfRegNum<[2]>;
    def R3  : P2Reg<0x1d3,  "r3">,  DwarfRegNum<[3]>;
    def R4  : P2Reg<0x1d4,  "r4">,  DwarfRegNum<[4]>;
    def R5  : P2Reg<0x1d5,  "r5">,  DwarfRegNum<[5]>;
    def R6  : P2Reg<0x1d6,  "r6">,  DwarfRegNum<[6]>;
    def R7  : P2Reg<0x1d7,  "r7">,  DwarfRegNum<[7]>;
    def R8  : P2Reg<0x1d8,  "r8">,  DwarfRegNum<[8]>;
    def R9  : P2Reg<0x1d9,  "r9">,  DwarfRegNum<[9]>;
    def R10 : P2Reg<0x1da, "r10">, DwarfRegNum<[10]>;
    def R11 : P2Reg<0x1db, "r11">, DwarfRegNum<[11]>;
    def R12 : P2Reg<0x1dc, "r12">, DwarfRegNum<[12]>;
    def R13 : P2Reg<0x1dd, "r13">, DwarfRegNum<[13]>;
    def R14 : P2Reg<0x1de, "r14">, DwarfRegNum<[14]>;
    def R15 : P2Reg<0x1df, "r15">, DwarfRegNum<[15]>;
    def R16 : P2Reg<0x1e0, "r16">, DwarfRegNum<[16]>;
    def R17 : P2Reg<0x1e1, "r17">, DwarfRegNum<[17]>;
    def R18 : P2Reg<0x1e2, "r18">, DwarfRegNum<[18]>;
    def R19 : P2Reg<0x1e3, "r19">, DwarfRegNum<[19]>;
    def R20 : P2Reg<0x1e4, "r20">, DwarfRegNum<[20]>;
    def R21 : P2Reg<0x1e5, "r21">, DwarfRegNum<[21]>;
    def R22 : P2Reg<0x1e6, "r22">, DwarfRegNum<[22]>;
    def R23 : P2Reg<0x1e7, "r23">, DwarfRegNum<[23]>;
    def R24 : P2Reg<0x1e8, "r24">, DwarfRegNum<[24]>;
    def R25 : P2Reg<0x1e9, "r25">, DwarfRegNum<[25]>;
    def R26 : P2Reg<0x1ea, "r26">, DwarfRegNum<[26]>;
    def R27 : P2Reg<0x1eb, "r27">, DwarfRegNum<[27]>;
    def R28 : P2Reg<0x1ec, "r28">, DwarfRegNum<[28]>;
    def R29 : P2Reg<0x1ed, "r29">, DwarfRegNum<[29]>;
    def R30 : P2Reg<0x1ee, "r30">, DwarfRegNum<[30]>;
    def R31 : P2Reg<0x1ef, "r31">, DwarfRegNum<[31]>;

    def SW  : P2Reg<0, "sw">, DwarfRegNum<[32]>; // this is an implicit register representing the status word. it can not be directly accessed.
    def QX  : P2Reg<0, "qx">, DwarfRegNum<[33]>; // this is an implicit register for x result of cordic
    def QY  : P2Reg<0, "qy">, DwarfRegNum<[34]>; // this is an implicit register for y result of cordic
    def N   : P2Reg<0, "n">, DwarfRegNum<[35]>; // implicit register for result of augs/augd instructions

    def IJMP3 : P2Reg<0x1f0, "ijmp3">, DwarfRegNum<[36]>;
    def IRET3 : P2Reg<0x1f1, "iret3">, DwarfRegNum<[37]>;
    def IJMP2 : P2Reg<0x1f2, "ijmp2">, DwarfRegNum<[38]>;
    def IRET2 : P2Reg<0x1f3, "iret2">, DwarfRegNum<[39]>;
    def IJMP1 : P2Reg<0x1f4, "ijmp1">, DwarfRegNum<[40]>;
    def IRET1 : P2Reg<0x1f5, "iret1">, DwarfRegNum<[41]>;

    def PA : P2Reg<0x1f6, "pa">, DwarfRegNum<[42]>;
    def PB : P2Reg<0x1f7, "pb">, DwarfRegNum<[43]>;
    def PTRA : P2Reg<0x1f8, "ptra">, DwarfRegNum<[44]>; // this is also the stack pointer
    def PTRB : P2Reg<0x1f9, "ptrb">, DwarfRegNum<[45]>;
    def DIRA : P2Reg<0x1fa, "dira">, DwarfRegNum<[46]>;
    def DIRB : P2Reg<0x1fb, "dirb">, DwarfRegNum<[47]>;
    def OUTA : P2Reg<0x1fc, "outa">, DwarfRegNum<[48]>;
    def OUTB : P2Reg<0x1fd, "outb">, DwarfRegNum<[49]>;
    def INA  : P2Reg<0x1fe, "ina">, DwarfRegNum<[50]>;
    def INB  : P2Reg<0x1ff, "inb">, DwarfRegNum<[51]>;
}

//===----------------------------------------------------------------------===//
//@Register Classes
//===----------------------------------------------------------------------===//


def P2GPR : RegisterClass<"P2", [i32], 32, (add
    // general purpose registers
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16,
    R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31,
    // interrupt registers
    IJMP3, IRET3, IJMP2, IRET2, IJMP1, IRET1,
    // cog pointer regsiters, PTRA is the stack pointer
    PA, PB, PTRA, PTRB,
    // i/o registers
    DIRA, DIRB, OUTA, OUTB, INA, INB
)>;

def P2Implicit : RegisterClass<"P2", [i32], 32, (add SW, QX, QY, N)>;

