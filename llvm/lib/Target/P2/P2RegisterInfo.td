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
//  purpose registers before that, called r0-r15
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

    def R0  : P2Reg<0x1e0,  "r0">,  DwarfRegNum<[0]>;
    def R1  : P2Reg<0x1e1,  "r1">,  DwarfRegNum<[1]>;
    def R2  : P2Reg<0x1e2,  "r2">,  DwarfRegNum<[2]>;
    def R3  : P2Reg<0x1e3,  "r3">,  DwarfRegNum<[3]>;
    def R4  : P2Reg<0x1e4,  "r4">,  DwarfRegNum<[4]>;
    def R5  : P2Reg<0x1e5,  "r5">,  DwarfRegNum<[5]>;
    def R6  : P2Reg<0x1e6,  "r6">,  DwarfRegNum<[6]>;
    def R7  : P2Reg<0x1e7,  "r7">,  DwarfRegNum<[7]>;
    def R8  : P2Reg<0x1e8,  "r8">,  DwarfRegNum<[8]>;
    def R9  : P2Reg<0x1e9,  "r9">,  DwarfRegNum<[9]>;
    def R10 : P2Reg<0x1ea, "r10">, DwarfRegNum<[10]>;
    def R11 : P2Reg<0x1eb, "r11">, DwarfRegNum<[11]>;
    def R12 : P2Reg<0x1ec, "r12">, DwarfRegNum<[12]>;
    def R13 : P2Reg<0x1ed, "r13">, DwarfRegNum<[13]>;
    def R14 : P2Reg<0x1ee, "r14">, DwarfRegNum<[14]>;
    def R15 : P2Reg<0x1ef, "r15">, DwarfRegNum<[15]>;

    def SW  : P2Reg<0, "sw">, DwarfRegNum<[20]>; // this is an implicit register representing the status word. it can not be directly accessed.
    def QX  : P2Reg<0, "qx">, DwarfRegNum<[21]>; // this is an implicit register for x result of cordic
    def QY  : P2Reg<0, "qy">, DwarfRegNum<[22]>; // this is an implicit register for y result of cordic
    def N   : P2Reg<0, "n">, DwarfRegNum<[23]>; // implicit register for result of augs/augd instructions

    def PTRA : P2Reg<0x1f8, "ptra">, DwarfRegNum<[40]>; // this is also the stack pointer
    def PTRB : P2Reg<0x1f9, "ptrb">, DwarfRegNum<[41]>;
    def DIRA : P2Reg<0x1fa, "dira">, DwarfRegNum<[42]>;
    def DIRB : P2Reg<0x1fb, "dirb">, DwarfRegNum<[43]>;
    def OUTA : P2Reg<0x1fc, "outa">, DwarfRegNum<[44]>;
    def OUTB : P2Reg<0x1fd, "outb">, DwarfRegNum<[45]>;
}

//===----------------------------------------------------------------------===//
//@Register Classes
//===----------------------------------------------------------------------===//


def P2GPR : RegisterClass<"P2", [i32], 32, (add
    // general purpose registers
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15,
    // cog pointer regsiters, PTRA is the stack pointer
    PTRA, PTRB,
    // i/o registers
    DIRA, DIRB, OUTA, OUTB
)>;

def P2Implicit : RegisterClass<"P2", [i32], 32, (add SW, QX, QY, N)>;

