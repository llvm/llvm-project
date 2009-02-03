//===- ARMRegisterInfo.td - ARM Register defs -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the ARM register file
//===----------------------------------------------------------------------===//

// Registers are identified with 4-bit ID numbers.
class ARMReg<bits<4> num, string n, list<Register> subregs = []> : Register<n> {
  field bits<4> Num;
  let Namespace = "ARM";
  let SubRegs = subregs;
}

class ARMFReg<bits<5> num, string n> : Register<n> {
  field bits<5> Num;
  let Namespace = "ARM";
}

// Integer registers
def R0  : ARMReg< 0, "r0">,  DwarfRegNum<[0]>;
def R1  : ARMReg< 1, "r1">,  DwarfRegNum<[1]>;
def R2  : ARMReg< 2, "r2">,  DwarfRegNum<[2]>;
def R3  : ARMReg< 3, "r3">,  DwarfRegNum<[3]>;
def R4  : ARMReg< 4, "r4">,  DwarfRegNum<[4]>;
def R5  : ARMReg< 5, "r5">,  DwarfRegNum<[5]>;
def R6  : ARMReg< 6, "r6">,  DwarfRegNum<[6]>;
def R7  : ARMReg< 7, "r7">,  DwarfRegNum<[7]>;
def R8  : ARMReg< 8, "r8">,  DwarfRegNum<[8]>;
def R9  : ARMReg< 9, "r9">,  DwarfRegNum<[9]>;
def R10 : ARMReg<10, "r10">, DwarfRegNum<[10]>;
def R11 : ARMReg<11, "r11">, DwarfRegNum<[11]>;
def R12 : ARMReg<12, "r12">, DwarfRegNum<[12]>;
def SP  : ARMReg<13, "sp">,  DwarfRegNum<[13]>;
def LR  : ARMReg<14, "lr">,  DwarfRegNum<[14]>;
def PC  : ARMReg<15, "pc">,  DwarfRegNum<[15]>;

// Float registers
def S0  : ARMFReg< 0, "s0">;  def S1  : ARMFReg< 1, "s1">;
def S2  : ARMFReg< 2, "s2">;  def S3  : ARMFReg< 3, "s3">;
def S4  : ARMFReg< 4, "s4">;  def S5  : ARMFReg< 5, "s5">;
def S6  : ARMFReg< 6, "s6">;  def S7  : ARMFReg< 7, "s7">;
def S8  : ARMFReg< 8, "s8">;  def S9  : ARMFReg< 9, "s9">;
def S10 : ARMFReg<10, "s10">; def S11 : ARMFReg<11, "s11">;
def S12 : ARMFReg<12, "s12">; def S13 : ARMFReg<13, "s13">;
def S14 : ARMFReg<14, "s14">; def S15 : ARMFReg<15, "s15">;
def S16 : ARMFReg<16, "s16">; def S17 : ARMFReg<17, "s17">;
def S18 : ARMFReg<18, "s18">; def S19 : ARMFReg<19, "s19">;
def S20 : ARMFReg<20, "s20">; def S21 : ARMFReg<21, "s21">;
def S22 : ARMFReg<22, "s22">; def S23 : ARMFReg<23, "s23">;
def S24 : ARMFReg<24, "s24">; def S25 : ARMFReg<25, "s25">;
def S26 : ARMFReg<26, "s26">; def S27 : ARMFReg<27, "s27">;
def S28 : ARMFReg<28, "s28">; def S29 : ARMFReg<29, "s29">;
def S30 : ARMFReg<30, "s30">; def S31 : ARMFReg<31, "s31">;

// Aliases of the F* registers used to hold 64-bit fp values (doubles)
def D0  : ARMReg< 0,  "d0", [S0,   S1]>;
def D1  : ARMReg< 1,  "d1", [S2,   S3]>; 
def D2  : ARMReg< 2,  "d2", [S4,   S5]>;
def D3  : ARMReg< 3,  "d3", [S6,   S7]>;
def D4  : ARMReg< 4,  "d4", [S8,   S9]>;
def D5  : ARMReg< 5,  "d5", [S10, S11]>;
def D6  : ARMReg< 6,  "d6", [S12, S13]>;
def D7  : ARMReg< 7,  "d7", [S14, S15]>;
def D8  : ARMReg< 8,  "d8", [S16, S17]>;
def D9  : ARMReg< 9,  "d9", [S18, S19]>;
def D10 : ARMReg<10, "d10", [S20, S21]>;
def D11 : ARMReg<11, "d11", [S22, S23]>;
def D12 : ARMReg<12, "d12", [S24, S25]>;
def D13 : ARMReg<13, "d13", [S26, S27]>;
def D14 : ARMReg<14, "d14", [S28, S29]>;
def D15 : ARMReg<15, "d15", [S30, S31]>;

// Current Program Status Register.
def CPSR : ARMReg<0, "cpsr">;

// Register classes.
//
// pc  == Program Counter
// lr  == Link Register
// sp  == Stack Pointer
// r12 == ip (scratch)
// r7  == Frame Pointer (thumb-style backtraces)
// r11 == Frame Pointer (arm-style backtraces)
// r10 == Stack Limit
//
def GPR : RegisterClass<"ARM", [i32], 32, [R0, R1, R2, R3, R4, R5, R6,
                                           R7, R8, R9, R10, R12, R11,
                                           LR, SP, PC]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  // FIXME: We are reserving r12 in case the PEI needs to use it to
  // generate large stack offset. Make it available once we have register
  // scavenging. Similarly r3 is reserved in Thumb mode for now.
  let MethodBodies = [{
    // FP is R11, R9 is available.
    static const unsigned ARM_GPR_AO_1[] = {
      ARM::R3, ARM::R2, ARM::R1, ARM::R0,
      ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6, ARM::R7,
      ARM::R8, ARM::R9, ARM::R10,
      ARM::R11 };
    // FP is R11, R9 is not available.
    static const unsigned ARM_GPR_AO_2[] = {
      ARM::R3, ARM::R2, ARM::R1, ARM::R0,
      ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6, ARM::R7,
      ARM::R8, ARM::R10,
      ARM::R11 };
    // FP is R7, R9 is available.
    static const unsigned ARM_GPR_AO_3[] = {
      ARM::R3, ARM::R2, ARM::R1, ARM::R0,
      ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6,
      ARM::R8, ARM::R9, ARM::R10,ARM::R11,
      ARM::R7 };
    // FP is R7, R9 is not available.
    static const unsigned ARM_GPR_AO_4[] = {
      ARM::R3, ARM::R2, ARM::R1, ARM::R0,
      ARM::R12,ARM::LR,
      ARM::R4, ARM::R5, ARM::R6,
      ARM::R8, ARM::R10,ARM::R11,
      ARM::R7 };

    // FP is R7, only low registers available.
    static const unsigned THUMB_GPR_AO[] = {
      ARM::R2, ARM::R1, ARM::R0,
      ARM::R4, ARM::R5, ARM::R6, ARM::R7 };

    GPRClass::iterator
    GPRClass::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      if (Subtarget.isThumb())
        return THUMB_GPR_AO;
      if (Subtarget.useThumbBacktraces()) {
        if (Subtarget.isR9Reserved())
          return ARM_GPR_AO_4;
        else
          return ARM_GPR_AO_3;
      } else {
        if (Subtarget.isR9Reserved())
          return ARM_GPR_AO_2;
        else
          return ARM_GPR_AO_1;
      }
    }

    GPRClass::iterator
    GPRClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const ARMSubtarget &Subtarget = TM.getSubtarget<ARMSubtarget>();
      GPRClass::iterator I;
      if (Subtarget.isThumb())
        I = THUMB_GPR_AO + (sizeof(THUMB_GPR_AO)/sizeof(unsigned));
      else if (Subtarget.useThumbBacktraces()) {
        if (Subtarget.isR9Reserved()) {
          I = ARM_GPR_AO_4 + (sizeof(ARM_GPR_AO_4)/sizeof(unsigned));
        } else {
          I = ARM_GPR_AO_3 + (sizeof(ARM_GPR_AO_3)/sizeof(unsigned));
        }
      } else {
        if (Subtarget.isR9Reserved()) {
          I = ARM_GPR_AO_2 + (sizeof(ARM_GPR_AO_2)/sizeof(unsigned));
        } else {
          I = ARM_GPR_AO_1 + (sizeof(ARM_GPR_AO_1)/sizeof(unsigned));
        }
      }

      // Mac OS X requires FP not to be clobbered for backtracing purpose.
      return (Subtarget.isTargetDarwin() || RI->hasFP(MF)) ? I-1 : I;
    }
  }];
}

def SPR : RegisterClass<"ARM", [f32], 32, [S0, S1, S2, S3, S4, S5, S6, S7, S8,
  S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22,
  S23, S24, S25, S26, S27, S28, S29, S30, S31]>;

// ARM requires only word alignment for double. It's more performant if it
// is double-word alignment though.
def DPR : RegisterClass<"ARM", [f64], 64, [D0, D1, D2, D3, D4, D5, D6, D7, D8,
  D9, D10, D11, D12, D13, D14, D15]>;

// Condition code registers.
def CCR : RegisterClass<"ARM", [i32], 32, [CPSR]>;
