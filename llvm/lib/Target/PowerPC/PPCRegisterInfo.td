//===-- PPCRegisterInfo.td - The PowerPC Register File -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

let Namespace = "PPC" in {
def sub_lt : SubRegIndex;
def sub_gt : SubRegIndex;
def sub_eq : SubRegIndex;
def sub_un : SubRegIndex;
def sub_32 : SubRegIndex;
}


class PPCReg<string n> : Register<n> {
  let Namespace = "PPC";
}

// We identify all our registers with a 5-bit ID, for consistency's sake.

// GPR - One of the 32 32-bit general-purpose registers
class GPR<bits<5> num, string n> : PPCReg<n> {
  field bits<5> Num = num;
}

// GP8 - One of the 32 64-bit general-purpose registers
class GP8<GPR SubReg, string n> : PPCReg<n> {
  field bits<5> Num = SubReg.Num;
  let SubRegs = [SubReg];
  let SubRegIndices = [sub_32];
}

// SPR - One of the 32-bit special-purpose registers
class SPR<bits<10> num, string n> : PPCReg<n> {
  field bits<10> Num = num;
}

// FPR - One of the 32 64-bit floating-point registers
class FPR<bits<5> num, string n> : PPCReg<n> {
  field bits<5> Num = num;
}

// VR - One of the 32 128-bit vector registers
class VR<bits<5> num, string n> : PPCReg<n> {
  field bits<5> Num = num;
}

// CR - One of the 8 4-bit condition registers
class CR<bits<3> num, string n, list<Register> subregs> : PPCReg<n> {
  field bits<3> Num = num;
  let SubRegs = subregs;
}

// CRBIT - One of the 32 1-bit condition register fields
class CRBIT<bits<5> num, string n> : PPCReg<n> {
  field bits<5> Num = num;
}


// General-purpose registers
def R0  : GPR< 0,  "r0">, DwarfRegNum<[-2, 0]>;
def R1  : GPR< 1,  "r1">, DwarfRegNum<[-2, 1]>;
def R2  : GPR< 2,  "r2">, DwarfRegNum<[-2, 2]>;
def R3  : GPR< 3,  "r3">, DwarfRegNum<[-2, 3]>;
def R4  : GPR< 4,  "r4">, DwarfRegNum<[-2, 4]>;
def R5  : GPR< 5,  "r5">, DwarfRegNum<[-2, 5]>;
def R6  : GPR< 6,  "r6">, DwarfRegNum<[-2, 6]>;
def R7  : GPR< 7,  "r7">, DwarfRegNum<[-2, 7]>;
def R8  : GPR< 8,  "r8">, DwarfRegNum<[-2, 8]>;
def R9  : GPR< 9,  "r9">, DwarfRegNum<[-2, 9]>;
def R10 : GPR<10, "r10">, DwarfRegNum<[-2, 10]>;
def R11 : GPR<11, "r11">, DwarfRegNum<[-2, 11]>;
def R12 : GPR<12, "r12">, DwarfRegNum<[-2, 12]>;
def R13 : GPR<13, "r13">, DwarfRegNum<[-2, 13]>;
def R14 : GPR<14, "r14">, DwarfRegNum<[-2, 14]>;
def R15 : GPR<15, "r15">, DwarfRegNum<[-2, 15]>;
def R16 : GPR<16, "r16">, DwarfRegNum<[-2, 16]>;
def R17 : GPR<17, "r17">, DwarfRegNum<[-2, 17]>;
def R18 : GPR<18, "r18">, DwarfRegNum<[-2, 18]>;
def R19 : GPR<19, "r19">, DwarfRegNum<[-2, 19]>;
def R20 : GPR<20, "r20">, DwarfRegNum<[-2, 20]>;
def R21 : GPR<21, "r21">, DwarfRegNum<[-2, 21]>;
def R22 : GPR<22, "r22">, DwarfRegNum<[-2, 22]>;
def R23 : GPR<23, "r23">, DwarfRegNum<[-2, 23]>;
def R24 : GPR<24, "r24">, DwarfRegNum<[-2, 24]>;
def R25 : GPR<25, "r25">, DwarfRegNum<[-2, 25]>;
def R26 : GPR<26, "r26">, DwarfRegNum<[-2, 26]>;
def R27 : GPR<27, "r27">, DwarfRegNum<[-2, 27]>;
def R28 : GPR<28, "r28">, DwarfRegNum<[-2, 28]>;
def R29 : GPR<29, "r29">, DwarfRegNum<[-2, 29]>;
def R30 : GPR<30, "r30">, DwarfRegNum<[-2, 30]>;
def R31 : GPR<31, "r31">, DwarfRegNum<[-2, 31]>;

// 64-bit General-purpose registers
def X0  : GP8< R0,  "r0">, DwarfRegNum<[0, -2]>;
def X1  : GP8< R1,  "r1">, DwarfRegNum<[1, -2]>;
def X2  : GP8< R2,  "r2">, DwarfRegNum<[2, -2]>;
def X3  : GP8< R3,  "r3">, DwarfRegNum<[3, -2]>;
def X4  : GP8< R4,  "r4">, DwarfRegNum<[4, -2]>;
def X5  : GP8< R5,  "r5">, DwarfRegNum<[5, -2]>;
def X6  : GP8< R6,  "r6">, DwarfRegNum<[6, -2]>;
def X7  : GP8< R7,  "r7">, DwarfRegNum<[7, -2]>;
def X8  : GP8< R8,  "r8">, DwarfRegNum<[8, -2]>;
def X9  : GP8< R9,  "r9">, DwarfRegNum<[9, -2]>;
def X10 : GP8<R10, "r10">, DwarfRegNum<[10, -2]>;
def X11 : GP8<R11, "r11">, DwarfRegNum<[11, -2]>;
def X12 : GP8<R12, "r12">, DwarfRegNum<[12, -2]>;
def X13 : GP8<R13, "r13">, DwarfRegNum<[13, -2]>;
def X14 : GP8<R14, "r14">, DwarfRegNum<[14, -2]>;
def X15 : GP8<R15, "r15">, DwarfRegNum<[15, -2]>;
def X16 : GP8<R16, "r16">, DwarfRegNum<[16, -2]>;
def X17 : GP8<R17, "r17">, DwarfRegNum<[17, -2]>;
def X18 : GP8<R18, "r18">, DwarfRegNum<[18, -2]>;
def X19 : GP8<R19, "r19">, DwarfRegNum<[19, -2]>;
def X20 : GP8<R20, "r20">, DwarfRegNum<[20, -2]>;
def X21 : GP8<R21, "r21">, DwarfRegNum<[21, -2]>;
def X22 : GP8<R22, "r22">, DwarfRegNum<[22, -2]>;
def X23 : GP8<R23, "r23">, DwarfRegNum<[23, -2]>;
def X24 : GP8<R24, "r24">, DwarfRegNum<[24, -2]>;
def X25 : GP8<R25, "r25">, DwarfRegNum<[25, -2]>;
def X26 : GP8<R26, "r26">, DwarfRegNum<[26, -2]>;
def X27 : GP8<R27, "r27">, DwarfRegNum<[27, -2]>;
def X28 : GP8<R28, "r28">, DwarfRegNum<[28, -2]>;
def X29 : GP8<R29, "r29">, DwarfRegNum<[29, -2]>;
def X30 : GP8<R30, "r30">, DwarfRegNum<[30, -2]>;
def X31 : GP8<R31, "r31">, DwarfRegNum<[31, -2]>;

// Floating-point registers
def F0  : FPR< 0,  "f0">, DwarfRegNum<[32, 32]>;
def F1  : FPR< 1,  "f1">, DwarfRegNum<[33, 33]>;
def F2  : FPR< 2,  "f2">, DwarfRegNum<[34, 34]>;
def F3  : FPR< 3,  "f3">, DwarfRegNum<[35, 35]>;
def F4  : FPR< 4,  "f4">, DwarfRegNum<[36, 36]>;
def F5  : FPR< 5,  "f5">, DwarfRegNum<[37, 37]>;
def F6  : FPR< 6,  "f6">, DwarfRegNum<[38, 38]>;
def F7  : FPR< 7,  "f7">, DwarfRegNum<[39, 39]>;
def F8  : FPR< 8,  "f8">, DwarfRegNum<[40, 40]>;
def F9  : FPR< 9,  "f9">, DwarfRegNum<[41, 41]>;
def F10 : FPR<10, "f10">, DwarfRegNum<[42, 42]>;
def F11 : FPR<11, "f11">, DwarfRegNum<[43, 43]>;
def F12 : FPR<12, "f12">, DwarfRegNum<[44, 44]>;
def F13 : FPR<13, "f13">, DwarfRegNum<[45, 45]>;
def F14 : FPR<14, "f14">, DwarfRegNum<[46, 46]>;
def F15 : FPR<15, "f15">, DwarfRegNum<[47, 47]>;
def F16 : FPR<16, "f16">, DwarfRegNum<[48, 48]>;
def F17 : FPR<17, "f17">, DwarfRegNum<[49, 49]>;
def F18 : FPR<18, "f18">, DwarfRegNum<[50, 50]>;
def F19 : FPR<19, "f19">, DwarfRegNum<[51, 51]>;
def F20 : FPR<20, "f20">, DwarfRegNum<[52, 52]>;
def F21 : FPR<21, "f21">, DwarfRegNum<[53, 53]>;
def F22 : FPR<22, "f22">, DwarfRegNum<[54, 54]>;
def F23 : FPR<23, "f23">, DwarfRegNum<[55, 55]>;
def F24 : FPR<24, "f24">, DwarfRegNum<[56, 56]>;
def F25 : FPR<25, "f25">, DwarfRegNum<[57, 57]>;
def F26 : FPR<26, "f26">, DwarfRegNum<[58, 58]>;
def F27 : FPR<27, "f27">, DwarfRegNum<[59, 59]>;
def F28 : FPR<28, "f28">, DwarfRegNum<[60, 60]>;
def F29 : FPR<29, "f29">, DwarfRegNum<[61, 61]>;
def F30 : FPR<30, "f30">, DwarfRegNum<[62, 62]>;
def F31 : FPR<31, "f31">, DwarfRegNum<[63, 63]>;

// Vector registers
def V0  : VR< 0,  "v0">, DwarfRegNum<[77, 77]>;
def V1  : VR< 1,  "v1">, DwarfRegNum<[78, 78]>;
def V2  : VR< 2,  "v2">, DwarfRegNum<[79, 79]>;
def V3  : VR< 3,  "v3">, DwarfRegNum<[80, 80]>;
def V4  : VR< 4,  "v4">, DwarfRegNum<[81, 81]>;
def V5  : VR< 5,  "v5">, DwarfRegNum<[82, 82]>;
def V6  : VR< 6,  "v6">, DwarfRegNum<[83, 83]>;
def V7  : VR< 7,  "v7">, DwarfRegNum<[84, 84]>;
def V8  : VR< 8,  "v8">, DwarfRegNum<[85, 85]>;
def V9  : VR< 9,  "v9">, DwarfRegNum<[86, 86]>;
def V10 : VR<10, "v10">, DwarfRegNum<[87, 87]>;
def V11 : VR<11, "v11">, DwarfRegNum<[88, 88]>;
def V12 : VR<12, "v12">, DwarfRegNum<[89, 89]>;
def V13 : VR<13, "v13">, DwarfRegNum<[90, 90]>;
def V14 : VR<14, "v14">, DwarfRegNum<[91, 91]>;
def V15 : VR<15, "v15">, DwarfRegNum<[92, 92]>;
def V16 : VR<16, "v16">, DwarfRegNum<[93, 93]>;
def V17 : VR<17, "v17">, DwarfRegNum<[94, 94]>;
def V18 : VR<18, "v18">, DwarfRegNum<[95, 95]>;
def V19 : VR<19, "v19">, DwarfRegNum<[96, 96]>;
def V20 : VR<20, "v20">, DwarfRegNum<[97, 97]>;
def V21 : VR<21, "v21">, DwarfRegNum<[98, 98]>;
def V22 : VR<22, "v22">, DwarfRegNum<[99, 99]>;
def V23 : VR<23, "v23">, DwarfRegNum<[100, 100]>;
def V24 : VR<24, "v24">, DwarfRegNum<[101, 101]>;
def V25 : VR<25, "v25">, DwarfRegNum<[102, 102]>;
def V26 : VR<26, "v26">, DwarfRegNum<[103, 103]>;
def V27 : VR<27, "v27">, DwarfRegNum<[104, 104]>;
def V28 : VR<28, "v28">, DwarfRegNum<[105, 105]>;
def V29 : VR<29, "v29">, DwarfRegNum<[106, 106]>;
def V30 : VR<30, "v30">, DwarfRegNum<[107, 107]>;
def V31 : VR<31, "v31">, DwarfRegNum<[108, 108]>;

// Condition register bits
def CR0LT : CRBIT< 0, "0">;
def CR0GT : CRBIT< 1, "1">;
def CR0EQ : CRBIT< 2, "2">;
def CR0UN : CRBIT< 3, "3">;
def CR1LT : CRBIT< 4, "4">;
def CR1GT : CRBIT< 5, "5">;
def CR1EQ : CRBIT< 6, "6">;
def CR1UN : CRBIT< 7, "7">;
def CR2LT : CRBIT< 8, "8">;
def CR2GT : CRBIT< 9, "9">;
def CR2EQ : CRBIT<10, "10">;
def CR2UN : CRBIT<11, "11">;
def CR3LT : CRBIT<12, "12">;
def CR3GT : CRBIT<13, "13">;
def CR3EQ : CRBIT<14, "14">;
def CR3UN : CRBIT<15, "15">;
def CR4LT : CRBIT<16, "16">;
def CR4GT : CRBIT<17, "17">;
def CR4EQ : CRBIT<18, "18">;
def CR4UN : CRBIT<19, "19">;
def CR5LT : CRBIT<20, "20">;
def CR5GT : CRBIT<21, "21">;
def CR5EQ : CRBIT<22, "22">;
def CR5UN : CRBIT<23, "23">;
def CR6LT : CRBIT<24, "24">;
def CR6GT : CRBIT<25, "25">;
def CR6EQ : CRBIT<26, "26">;
def CR6UN : CRBIT<27, "27">;
def CR7LT : CRBIT<28, "28">;
def CR7GT : CRBIT<29, "29">;
def CR7EQ : CRBIT<30, "30">;
def CR7UN : CRBIT<31, "31">;

// Condition registers
let SubRegIndices = [sub_lt, sub_gt, sub_eq, sub_un] in {
def CR0 : CR<0, "cr0", [CR0LT, CR0GT, CR0EQ, CR0UN]>, DwarfRegNum<[68, 68]>;
def CR1 : CR<1, "cr1", [CR1LT, CR1GT, CR1EQ, CR1UN]>, DwarfRegNum<[69, 69]>;
def CR2 : CR<2, "cr2", [CR2LT, CR2GT, CR2EQ, CR2UN]>, DwarfRegNum<[70, 70]>;
def CR3 : CR<3, "cr3", [CR3LT, CR3GT, CR3EQ, CR3UN]>, DwarfRegNum<[71, 71]>;
def CR4 : CR<4, "cr4", [CR4LT, CR4GT, CR4EQ, CR4UN]>, DwarfRegNum<[72, 72]>;
def CR5 : CR<5, "cr5", [CR5LT, CR5GT, CR5EQ, CR5UN]>, DwarfRegNum<[73, 73]>;
def CR6 : CR<6, "cr6", [CR6LT, CR6GT, CR6EQ, CR6UN]>, DwarfRegNum<[74, 74]>;
def CR7 : CR<7, "cr7", [CR7LT, CR7GT, CR7EQ, CR7UN]>, DwarfRegNum<[75, 75]>;
}

// Link register
def LR  : SPR<8, "lr">, DwarfRegNum<[-2, 65]>;
//let Aliases = [LR] in
def LR8 : SPR<8, "lr">, DwarfRegNum<[65, -2]>;

// Count register
def CTR  : SPR<9, "ctr">, DwarfRegNum<[-2, 66]>;
def CTR8 : SPR<9, "ctr">, DwarfRegNum<[66, -2]>;

// VRsave register
def VRSAVE: SPR<256, "VRsave">, DwarfRegNum<[109]>;

// Carry bit.  In the architecture this is really bit 0 of the XER register
// (which really is SPR register 1);  this is the only bit interesting to a
// compiler.
def CARRY: SPR<1, "ca">;

// FP rounding mode:  bits 30 and 31 of the FP status and control register
// This is not allocated as a normal register; it appears only in
// Uses and Defs.  The ABI says it needs to be preserved by a function,
// but this is not achieved by saving and restoring it as with
// most registers, it has to be done in code; to make this work all the
// return and call instructions are described as Uses of RM, so instructions
// that do nothing but change RM will not get deleted.
// Also, in the architecture it is not really a SPR; 512 is arbitrary.
def RM: SPR<512, "**ROUNDING MODE**">;

/// Register classes
// Allocate volatiles first
// then nonvolatiles in reverse order since stmw/lmw save from rN to r31
def GPRC : RegisterClass<"PPC", [i32], 32, (add (sequence "R%u", 2, 12),
                                                (sequence "R%u", 30, 13),
                                                R31, R0, R1, LR)>;

def G8RC : RegisterClass<"PPC", [i64], 64, (add (sequence "X%u", 2, 12),
                                                (sequence "X%u", 30, 14),
                                                X31, X13, X0, X1, LR8)>;

// Allocate volatiles first, then non-volatiles in reverse order. With the SVR4
// ABI the size of the Floating-point register save area is determined by the
// allocated non-volatile register with the lowest register number, as FP
// register N is spilled to offset 8 * (32 - N) below the back chain word of the
// previous stack frame. By allocating non-volatiles in reverse order we make
// sure that the Floating-point register save area is always as small as
// possible because there aren't any unused spill slots.
def F8RC : RegisterClass<"PPC", [f64], 64, (add (sequence "F%u", 0, 13),
                                                (sequence "F%u", 31, 14))>;
def F4RC : RegisterClass<"PPC", [f32], 32, (add F8RC)>;

def VRRC : RegisterClass<"PPC", [v16i8,v8i16,v4i32,v4f32], 128,
                         (add V2, V3, V4, V5, V0, V1, V6, V7, V8, V9, V10, V11,
                             V12, V13, V14, V15, V16, V17, V18, V19, V31, V30,
                             V29, V28, V27, V26, V25, V24, V23, V22, V21, V20)>;

def CRBITRC : RegisterClass<"PPC", [i32], 32,
  (add CR0LT, CR0GT, CR0EQ, CR0UN,
       CR1LT, CR1GT, CR1EQ, CR1UN,
       CR2LT, CR2GT, CR2EQ, CR2UN,
       CR3LT, CR3GT, CR3EQ, CR3UN,
       CR4LT, CR4GT, CR4EQ, CR4UN,
       CR5LT, CR5GT, CR5EQ, CR5UN,
       CR6LT, CR6GT, CR6EQ, CR6UN,
       CR7LT, CR7GT, CR7EQ, CR7UN)>
{
  let CopyCost = -1;
}

def CRRC : RegisterClass<"PPC", [i32], 32, (add CR0, CR1, CR5, CR6,
                                                CR7, CR2, CR3, CR4)>;

// The CTR registers are not allocatable because they're used by the
// decrement-and-branch instructions, and thus need to stay live across
// multiple basic blocks.
def CTRRC : RegisterClass<"PPC", [i32], 32, (add CTR)> {
  let isAllocatable = 0;
}
def CTRRC8 : RegisterClass<"PPC", [i64], 64, (add CTR8)> {
  let isAllocatable = 0;
}

def VRSAVERC : RegisterClass<"PPC", [i32], 32, (add VRSAVE)>;
def CARRYRC : RegisterClass<"PPC", [i32], 32, (add CARRY)> {
  let CopyCost = -1;
}
