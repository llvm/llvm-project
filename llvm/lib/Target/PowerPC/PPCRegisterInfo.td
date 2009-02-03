//===- PPCRegisterInfo.td - The PowerPC Register File ------*- tablegen -*-===//
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
class CR<bits<3> num, string n> : PPCReg<n> {
  field bits<3> Num = num;
}

// CRBIT - One of the 32 1-bit condition register fields
class CRBIT<bits<5> num, string n> : PPCReg<n> {
  field bits<5> Num = num;
}


// General-purpose registers
def R0  : GPR< 0,  "r0">, DwarfRegNum<[0]>;
def R1  : GPR< 1,  "r1">, DwarfRegNum<[1]>;
def R2  : GPR< 2,  "r2">, DwarfRegNum<[2]>;
def R3  : GPR< 3,  "r3">, DwarfRegNum<[3]>;
def R4  : GPR< 4,  "r4">, DwarfRegNum<[4]>;
def R5  : GPR< 5,  "r5">, DwarfRegNum<[5]>;
def R6  : GPR< 6,  "r6">, DwarfRegNum<[6]>;
def R7  : GPR< 7,  "r7">, DwarfRegNum<[7]>;
def R8  : GPR< 8,  "r8">, DwarfRegNum<[8]>;
def R9  : GPR< 9,  "r9">, DwarfRegNum<[9]>;
def R10 : GPR<10, "r10">, DwarfRegNum<[10]>;
def R11 : GPR<11, "r11">, DwarfRegNum<[11]>;
def R12 : GPR<12, "r12">, DwarfRegNum<[12]>;
def R13 : GPR<13, "r13">, DwarfRegNum<[13]>;
def R14 : GPR<14, "r14">, DwarfRegNum<[14]>;
def R15 : GPR<15, "r15">, DwarfRegNum<[15]>;
def R16 : GPR<16, "r16">, DwarfRegNum<[16]>;
def R17 : GPR<17, "r17">, DwarfRegNum<[17]>;
def R18 : GPR<18, "r18">, DwarfRegNum<[18]>;
def R19 : GPR<19, "r19">, DwarfRegNum<[19]>;
def R20 : GPR<20, "r20">, DwarfRegNum<[20]>;
def R21 : GPR<21, "r21">, DwarfRegNum<[21]>;
def R22 : GPR<22, "r22">, DwarfRegNum<[22]>;
def R23 : GPR<23, "r23">, DwarfRegNum<[23]>;
def R24 : GPR<24, "r24">, DwarfRegNum<[24]>;
def R25 : GPR<25, "r25">, DwarfRegNum<[25]>;
def R26 : GPR<26, "r26">, DwarfRegNum<[26]>;
def R27 : GPR<27, "r27">, DwarfRegNum<[27]>;
def R28 : GPR<28, "r28">, DwarfRegNum<[28]>;
def R29 : GPR<29, "r29">, DwarfRegNum<[29]>;
def R30 : GPR<30, "r30">, DwarfRegNum<[30]>;
def R31 : GPR<31, "r31">, DwarfRegNum<[31]>;

// 64-bit General-purpose registers
def X0  : GP8< R0,  "r0">, DwarfRegNum<[0]>;
def X1  : GP8< R1,  "r1">, DwarfRegNum<[1]>;
def X2  : GP8< R2,  "r2">, DwarfRegNum<[2]>;
def X3  : GP8< R3,  "r3">, DwarfRegNum<[3]>;
def X4  : GP8< R4,  "r4">, DwarfRegNum<[4]>;
def X5  : GP8< R5,  "r5">, DwarfRegNum<[5]>;
def X6  : GP8< R6,  "r6">, DwarfRegNum<[6]>;
def X7  : GP8< R7,  "r7">, DwarfRegNum<[7]>;
def X8  : GP8< R8,  "r8">, DwarfRegNum<[8]>;
def X9  : GP8< R9,  "r9">, DwarfRegNum<[9]>;
def X10 : GP8<R10, "r10">, DwarfRegNum<[10]>;
def X11 : GP8<R11, "r11">, DwarfRegNum<[11]>;
def X12 : GP8<R12, "r12">, DwarfRegNum<[12]>;
def X13 : GP8<R13, "r13">, DwarfRegNum<[13]>;
def X14 : GP8<R14, "r14">, DwarfRegNum<[14]>;
def X15 : GP8<R15, "r15">, DwarfRegNum<[15]>;
def X16 : GP8<R16, "r16">, DwarfRegNum<[16]>;
def X17 : GP8<R17, "r17">, DwarfRegNum<[17]>;
def X18 : GP8<R18, "r18">, DwarfRegNum<[18]>;
def X19 : GP8<R19, "r19">, DwarfRegNum<[19]>;
def X20 : GP8<R20, "r20">, DwarfRegNum<[20]>;
def X21 : GP8<R21, "r21">, DwarfRegNum<[21]>;
def X22 : GP8<R22, "r22">, DwarfRegNum<[22]>;
def X23 : GP8<R23, "r23">, DwarfRegNum<[23]>;
def X24 : GP8<R24, "r24">, DwarfRegNum<[24]>;
def X25 : GP8<R25, "r25">, DwarfRegNum<[25]>;
def X26 : GP8<R26, "r26">, DwarfRegNum<[26]>;
def X27 : GP8<R27, "r27">, DwarfRegNum<[27]>;
def X28 : GP8<R28, "r28">, DwarfRegNum<[28]>;
def X29 : GP8<R29, "r29">, DwarfRegNum<[29]>;
def X30 : GP8<R30, "r30">, DwarfRegNum<[30]>;
def X31 : GP8<R31, "r31">, DwarfRegNum<[31]>;

// Floating-point registers
def F0  : FPR< 0,  "f0">, DwarfRegNum<[32]>;
def F1  : FPR< 1,  "f1">, DwarfRegNum<[33]>;
def F2  : FPR< 2,  "f2">, DwarfRegNum<[34]>;
def F3  : FPR< 3,  "f3">, DwarfRegNum<[35]>;
def F4  : FPR< 4,  "f4">, DwarfRegNum<[36]>;
def F5  : FPR< 5,  "f5">, DwarfRegNum<[37]>;
def F6  : FPR< 6,  "f6">, DwarfRegNum<[38]>;
def F7  : FPR< 7,  "f7">, DwarfRegNum<[39]>;
def F8  : FPR< 8,  "f8">, DwarfRegNum<[40]>;
def F9  : FPR< 9,  "f9">, DwarfRegNum<[41]>;
def F10 : FPR<10, "f10">, DwarfRegNum<[42]>;
def F11 : FPR<11, "f11">, DwarfRegNum<[43]>;
def F12 : FPR<12, "f12">, DwarfRegNum<[44]>;
def F13 : FPR<13, "f13">, DwarfRegNum<[45]>;
def F14 : FPR<14, "f14">, DwarfRegNum<[46]>;
def F15 : FPR<15, "f15">, DwarfRegNum<[47]>;
def F16 : FPR<16, "f16">, DwarfRegNum<[48]>;
def F17 : FPR<17, "f17">, DwarfRegNum<[49]>;
def F18 : FPR<18, "f18">, DwarfRegNum<[50]>;
def F19 : FPR<19, "f19">, DwarfRegNum<[51]>;
def F20 : FPR<20, "f20">, DwarfRegNum<[52]>;
def F21 : FPR<21, "f21">, DwarfRegNum<[53]>;
def F22 : FPR<22, "f22">, DwarfRegNum<[54]>;
def F23 : FPR<23, "f23">, DwarfRegNum<[55]>;
def F24 : FPR<24, "f24">, DwarfRegNum<[56]>;
def F25 : FPR<25, "f25">, DwarfRegNum<[57]>;
def F26 : FPR<26, "f26">, DwarfRegNum<[58]>;
def F27 : FPR<27, "f27">, DwarfRegNum<[59]>;
def F28 : FPR<28, "f28">, DwarfRegNum<[60]>;
def F29 : FPR<29, "f29">, DwarfRegNum<[61]>;
def F30 : FPR<30, "f30">, DwarfRegNum<[62]>;
def F31 : FPR<31, "f31">, DwarfRegNum<[63]>;

// Vector registers
def V0  : VR< 0,  "v0">, DwarfRegNum<[77]>;
def V1  : VR< 1,  "v1">, DwarfRegNum<[78]>;
def V2  : VR< 2,  "v2">, DwarfRegNum<[79]>;
def V3  : VR< 3,  "v3">, DwarfRegNum<[80]>;
def V4  : VR< 4,  "v4">, DwarfRegNum<[81]>;
def V5  : VR< 5,  "v5">, DwarfRegNum<[82]>;
def V6  : VR< 6,  "v6">, DwarfRegNum<[83]>;
def V7  : VR< 7,  "v7">, DwarfRegNum<[84]>;
def V8  : VR< 8,  "v8">, DwarfRegNum<[85]>;
def V9  : VR< 9,  "v9">, DwarfRegNum<[86]>;
def V10 : VR<10, "v10">, DwarfRegNum<[87]>;
def V11 : VR<11, "v11">, DwarfRegNum<[88]>;
def V12 : VR<12, "v12">, DwarfRegNum<[89]>;
def V13 : VR<13, "v13">, DwarfRegNum<[90]>;
def V14 : VR<14, "v14">, DwarfRegNum<[91]>;
def V15 : VR<15, "v15">, DwarfRegNum<[92]>;
def V16 : VR<16, "v16">, DwarfRegNum<[93]>;
def V17 : VR<17, "v17">, DwarfRegNum<[94]>;
def V18 : VR<18, "v18">, DwarfRegNum<[95]>;
def V19 : VR<19, "v19">, DwarfRegNum<[96]>;
def V20 : VR<20, "v20">, DwarfRegNum<[97]>;
def V21 : VR<21, "v21">, DwarfRegNum<[98]>;
def V22 : VR<22, "v22">, DwarfRegNum<[99]>;
def V23 : VR<23, "v23">, DwarfRegNum<[100]>;
def V24 : VR<24, "v24">, DwarfRegNum<[101]>;
def V25 : VR<25, "v25">, DwarfRegNum<[102]>;
def V26 : VR<26, "v26">, DwarfRegNum<[103]>;
def V27 : VR<27, "v27">, DwarfRegNum<[104]>;
def V28 : VR<28, "v28">, DwarfRegNum<[105]>;
def V29 : VR<29, "v29">, DwarfRegNum<[106]>;
def V30 : VR<30, "v30">, DwarfRegNum<[107]>;
def V31 : VR<31, "v31">, DwarfRegNum<[108]>;

// Condition registers
def CR0 : CR<0, "cr0">, DwarfRegNum<[68]>;
def CR1 : CR<1, "cr1">, DwarfRegNum<[69]>;
def CR2 : CR<2, "cr2">, DwarfRegNum<[70]>;
def CR3 : CR<3, "cr3">, DwarfRegNum<[71]>;
def CR4 : CR<4, "cr4">, DwarfRegNum<[72]>;
def CR5 : CR<5, "cr5">, DwarfRegNum<[73]>;
def CR6 : CR<6, "cr6">, DwarfRegNum<[74]>;
def CR7 : CR<7, "cr7">, DwarfRegNum<[75]>;

// Condition register bits
def CR0LT : CRBIT< 0, "0">, DwarfRegNum<[0]>;
def CR0GT : CRBIT< 1, "1">, DwarfRegNum<[0]>;
def CR0EQ : CRBIT< 2, "2">, DwarfRegNum<[0]>;
def CR0UN : CRBIT< 3, "3">, DwarfRegNum<[0]>;
def CR1LT : CRBIT< 4, "4">, DwarfRegNum<[0]>;
def CR1GT : CRBIT< 5, "5">, DwarfRegNum<[0]>;
def CR1EQ : CRBIT< 6, "6">, DwarfRegNum<[0]>;
def CR1UN : CRBIT< 7, "7">, DwarfRegNum<[0]>;
def CR2LT : CRBIT< 8, "8">, DwarfRegNum<[0]>;
def CR2GT : CRBIT< 9, "9">, DwarfRegNum<[0]>;
def CR2EQ : CRBIT<10, "10">, DwarfRegNum<[0]>;
def CR2UN : CRBIT<11, "11">, DwarfRegNum<[0]>;
def CR3LT : CRBIT<12, "12">, DwarfRegNum<[0]>;
def CR3GT : CRBIT<13, "13">, DwarfRegNum<[0]>;
def CR3EQ : CRBIT<14, "14">, DwarfRegNum<[0]>;
def CR3UN : CRBIT<15, "15">, DwarfRegNum<[0]>;
def CR4LT : CRBIT<16, "16">, DwarfRegNum<[0]>;
def CR4GT : CRBIT<17, "17">, DwarfRegNum<[0]>;
def CR4EQ : CRBIT<18, "18">, DwarfRegNum<[0]>;
def CR4UN : CRBIT<19, "19">, DwarfRegNum<[0]>;
def CR5LT : CRBIT<20, "20">, DwarfRegNum<[0]>;
def CR5GT : CRBIT<21, "21">, DwarfRegNum<[0]>;
def CR5EQ : CRBIT<22, "22">, DwarfRegNum<[0]>;
def CR5UN : CRBIT<23, "23">, DwarfRegNum<[0]>;
def CR6LT : CRBIT<24, "24">, DwarfRegNum<[0]>;
def CR6GT : CRBIT<25, "25">, DwarfRegNum<[0]>;
def CR6EQ : CRBIT<26, "26">, DwarfRegNum<[0]>;
def CR6UN : CRBIT<27, "27">, DwarfRegNum<[0]>;
def CR7LT : CRBIT<28, "28">, DwarfRegNum<[0]>;
def CR7GT : CRBIT<29, "29">, DwarfRegNum<[0]>;
def CR7EQ : CRBIT<30, "30">, DwarfRegNum<[0]>;
def CR7UN : CRBIT<31, "31">, DwarfRegNum<[0]>;

def : SubRegSet<1, [CR0, CR1, CR2, CR3, CR4, CR5, CR6, CR7],
                   [CR0LT, CR1LT, CR2LT, CR3LT, CR4LT, CR5LT, CR6LT, CR7LT]>;
def : SubRegSet<2, [CR0, CR1, CR2, CR3, CR4, CR5, CR6, CR7],
                   [CR0GT, CR1GT, CR2GT, CR3GT, CR4GT, CR5GT, CR6GT, CR7GT]>;
def : SubRegSet<3, [CR0, CR1, CR2, CR3, CR4, CR5, CR6, CR7],
                   [CR0EQ, CR1EQ, CR2EQ, CR3EQ, CR4EQ, CR5EQ, CR6EQ, CR7EQ]>;
def : SubRegSet<4, [CR0, CR1, CR2, CR3, CR4, CR5, CR6, CR7],
                   [CR0UN, CR1UN, CR2UN, CR3UN, CR4UN, CR5UN, CR6UN, CR7UN]>;

// Link register
def LR  : SPR<8, "lr">, DwarfRegNum<[65]>;
//let Aliases = [LR] in
def LR8 : SPR<8, "lr">, DwarfRegNum<[65]>;

// Count register
def CTR  : SPR<9, "ctr">, DwarfRegNum<[66]>;
def CTR8 : SPR<9, "ctr">, DwarfRegNum<[66]>;

// VRsave register
def VRSAVE: SPR<256, "VRsave">, DwarfRegNum<[107]>;

// FP rounding mode:  bits 30 and 31 of the FP status and control register
// This is not allocated as a normal register; it appears only in
// Uses and Defs.  The ABI says it needs to be preserved by a function,
// but this is not achieved by saving and restoring it as with
// most registers, it has to be done in code; to make this work all the
// return and call instructions are described as Uses of RM, so instructions
// that do nothing but change RM will not get deleted.
// Also, in the architecture it is not really a SPR; 512 is arbitrary.
def RM: SPR<512, "**ROUNDING MODE**">, DwarfRegNum<[0]>;

/// Register classes
// Allocate volatiles first
// then nonvolatiles in reverse order since stmw/lmw save from rN to r31
def GPRC : RegisterClass<"PPC", [i32], 32,
     [R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12,
      R30, R29, R28, R27, R26, R25, R24, R23, R22, R21, R20, R19, R18, R17,
      R16, R15, R14, R13, R31, R0, R1, LR]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GPRCClass::iterator
    GPRCClass::allocation_order_begin(const MachineFunction &MF) const {
      // In Linux, r2 is reserved for the OS.
      if (!MF.getTarget().getSubtarget<PPCSubtarget>().isDarwin())
        return begin()+1;

      return begin();
    }
    GPRCClass::iterator
    GPRCClass::allocation_order_end(const MachineFunction &MF) const {
      // On PPC64, r13 is the thread pointer.  Never allocate this register.
      // Note that this is overconservative, as it also prevents allocation of
      // R31 when the FP is not needed.
      if (MF.getTarget().getSubtarget<PPCSubtarget>().isPPC64())
        return end()-5;  // don't allocate R13, R31, R0, R1, LR
        
      if (needsFP(MF))
        return end()-4;  // don't allocate R31, R0, R1, LR
      else
        return end()-3;  // don't allocate R0, R1, LR
    }
  }];
}
def G8RC : RegisterClass<"PPC", [i64], 64,
     [X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12,
      X30, X29, X28, X27, X26, X25, X24, X23, X22, X21, X20, X19, X18, X17,
      X16, X15, X14, X31, X13, X0, X1, LR8]>
{
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    G8RCClass::iterator
    G8RCClass::allocation_order_begin(const MachineFunction &MF) const {
      return begin();
    }
    G8RCClass::iterator
    G8RCClass::allocation_order_end(const MachineFunction &MF) const {
      if (needsFP(MF))
        return end()-5;
      else
        return end()-4;
    }
  }];
}



def F8RC : RegisterClass<"PPC", [f64], 64, [F0, F1, F2, F3, F4, F5, F6, F7,
  F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21,
  F22, F23, F24, F25, F26, F27, F28, F29, F30, F31]>;
def F4RC : RegisterClass<"PPC", [f32], 32, [F0, F1, F2, F3, F4, F5, F6, F7,
  F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21,
  F22, F23, F24, F25, F26, F27, F28, F29, F30, F31]>;

def VRRC : RegisterClass<"PPC", [v16i8,v8i16,v4i32,v4f32], 128,
 [V2, V3, V4, V5, V0, V1, 
  V6, V7, V8, V9, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21,
  V22, V23, V24, V25, V26, V27, V28, V29, V30, V31]>;

def CRRC : RegisterClass<"PPC", [i32], 32, [CR0, CR1, CR5, CR6, CR7, CR2, 
  CR3, CR4]>;

def CRBITRC : RegisterClass<"PPC", [i32], 32,
  [CR0LT, CR0GT, CR0EQ, CR0UN,
   CR1LT, CR1GT, CR1EQ, CR1UN,
   CR2LT, CR2GT, CR2EQ, CR2UN,
   CR3LT, CR3GT, CR3EQ, CR3UN,
   CR4LT, CR4GT, CR4EQ, CR4UN,
   CR5LT, CR5GT, CR5EQ, CR5UN,
   CR6LT, CR6GT, CR6EQ, CR6UN,
   CR7LT, CR7GT, CR7EQ, CR7UN
  ]>
{
  let CopyCost = -1;
}


def CTRRC : RegisterClass<"PPC", [i32], 32, [CTR]>;
def CTRRC8 : RegisterClass<"PPC", [i64], 64, [CTR8]>;
