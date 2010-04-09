//===- X86RegisterInfo.td - Describe the X86 Register File --*- tablegen -*-==//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 Register file, defining the registers themselves,
// aliases between the registers, and the register classes built out of the
// registers.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Register definitions...
//
let Namespace = "X86" in {

  // In the register alias definitions below, we define which registers alias
  // which others.  We only specify which registers the small registers alias,
  // because the register file generator is smart enough to figure out that
  // AL aliases AX if we tell it that AX aliased AL (for example).

  // Dwarf numbering is different for 32-bit and 64-bit, and there are 
  // variations by target as well. Currently the first entry is for X86-64, 
  // second - for EH on X86-32/Darwin and third is 'generic' one (X86-32/Linux
  // and debug information on X86-32/Darwin)

  // 8-bit registers
  // Low registers
  def AL : Register<"al">, DwarfRegNum<[0, 0, 0]>;
  def DL : Register<"dl">, DwarfRegNum<[1, 2, 2]>;
  def CL : Register<"cl">, DwarfRegNum<[2, 1, 1]>;
  def BL : Register<"bl">, DwarfRegNum<[3, 3, 3]>;

  // X86-64 only
  def SIL : Register<"sil">, DwarfRegNum<[4, 6, 6]>;
  def DIL : Register<"dil">, DwarfRegNum<[5, 7, 7]>;
  def BPL : Register<"bpl">, DwarfRegNum<[6, 4, 5]>;
  def SPL : Register<"spl">, DwarfRegNum<[7, 5, 4]>;
  def R8B  : Register<"r8b">,  DwarfRegNum<[8, -2, -2]>;
  def R9B  : Register<"r9b">,  DwarfRegNum<[9, -2, -2]>;
  def R10B : Register<"r10b">, DwarfRegNum<[10, -2, -2]>;
  def R11B : Register<"r11b">, DwarfRegNum<[11, -2, -2]>;
  def R12B : Register<"r12b">, DwarfRegNum<[12, -2, -2]>;
  def R13B : Register<"r13b">, DwarfRegNum<[13, -2, -2]>;
  def R14B : Register<"r14b">, DwarfRegNum<[14, -2, -2]>;
  def R15B : Register<"r15b">, DwarfRegNum<[15, -2, -2]>;

  // High registers. On x86-64, these cannot be used in any instruction
  // with a REX prefix.
  def AH : Register<"ah">, DwarfRegNum<[0, 0, 0]>;
  def DH : Register<"dh">, DwarfRegNum<[1, 2, 2]>;
  def CH : Register<"ch">, DwarfRegNum<[2, 1, 1]>;
  def BH : Register<"bh">, DwarfRegNum<[3, 3, 3]>;

  // 16-bit registers
  def AX : RegisterWithSubRegs<"ax", [AL,AH]>, DwarfRegNum<[0, 0, 0]>;
  def DX : RegisterWithSubRegs<"dx", [DL,DH]>, DwarfRegNum<[1, 2, 2]>;
  def CX : RegisterWithSubRegs<"cx", [CL,CH]>, DwarfRegNum<[2, 1, 1]>;
  def BX : RegisterWithSubRegs<"bx", [BL,BH]>, DwarfRegNum<[3, 3, 3]>;
  def SI : RegisterWithSubRegs<"si", [SIL]>, DwarfRegNum<[4, 6, 6]>;
  def DI : RegisterWithSubRegs<"di", [DIL]>, DwarfRegNum<[5, 7, 7]>;
  def BP : RegisterWithSubRegs<"bp", [BPL]>, DwarfRegNum<[6, 4, 5]>;
  def SP : RegisterWithSubRegs<"sp", [SPL]>, DwarfRegNum<[7, 5, 4]>;
  def IP : Register<"ip">, DwarfRegNum<[16]>;
  
  // X86-64 only
  def R8W  : RegisterWithSubRegs<"r8w", [R8B]>, DwarfRegNum<[8, -2, -2]>;
  def R9W  : RegisterWithSubRegs<"r9w", [R9B]>, DwarfRegNum<[9, -2, -2]>;
  def R10W : RegisterWithSubRegs<"r10w", [R10B]>, DwarfRegNum<[10, -2, -2]>;
  def R11W : RegisterWithSubRegs<"r11w", [R11B]>, DwarfRegNum<[11, -2, -2]>;
  def R12W : RegisterWithSubRegs<"r12w", [R12B]>, DwarfRegNum<[12, -2, -2]>;
  def R13W : RegisterWithSubRegs<"r13w", [R13B]>, DwarfRegNum<[13, -2, -2]>;
  def R14W : RegisterWithSubRegs<"r14w", [R14B]>, DwarfRegNum<[14, -2, -2]>;
  def R15W : RegisterWithSubRegs<"r15w", [R15B]>, DwarfRegNum<[15, -2, -2]>;

  // 32-bit registers
  def EAX : RegisterWithSubRegs<"eax", [AX]>, DwarfRegNum<[0, 0, 0]>;
  def EDX : RegisterWithSubRegs<"edx", [DX]>, DwarfRegNum<[1, 2, 2]>;
  def ECX : RegisterWithSubRegs<"ecx", [CX]>, DwarfRegNum<[2, 1, 1]>;
  def EBX : RegisterWithSubRegs<"ebx", [BX]>, DwarfRegNum<[3, 3, 3]>;
  def ESI : RegisterWithSubRegs<"esi", [SI]>, DwarfRegNum<[4, 6, 6]>;
  def EDI : RegisterWithSubRegs<"edi", [DI]>, DwarfRegNum<[5, 7, 7]>;
  def EBP : RegisterWithSubRegs<"ebp", [BP]>, DwarfRegNum<[6, 4, 5]>;
  def ESP : RegisterWithSubRegs<"esp", [SP]>, DwarfRegNum<[7, 5, 4]>;
  def EIP : RegisterWithSubRegs<"eip", [IP]>, DwarfRegNum<[16, 8, 8]>;  
  
  // X86-64 only
  def R8D  : RegisterWithSubRegs<"r8d", [R8W]>, DwarfRegNum<[8, -2, -2]>;
  def R9D  : RegisterWithSubRegs<"r9d", [R9W]>, DwarfRegNum<[9, -2, -2]>;
  def R10D : RegisterWithSubRegs<"r10d", [R10W]>, DwarfRegNum<[10, -2, -2]>;
  def R11D : RegisterWithSubRegs<"r11d", [R11W]>, DwarfRegNum<[11, -2, -2]>;
  def R12D : RegisterWithSubRegs<"r12d", [R12W]>, DwarfRegNum<[12, -2, -2]>;
  def R13D : RegisterWithSubRegs<"r13d", [R13W]>, DwarfRegNum<[13, -2, -2]>;
  def R14D : RegisterWithSubRegs<"r14d", [R14W]>, DwarfRegNum<[14, -2, -2]>;
  def R15D : RegisterWithSubRegs<"r15d", [R15W]>, DwarfRegNum<[15, -2, -2]>;

  // 64-bit registers, X86-64 only
  def RAX : RegisterWithSubRegs<"rax", [EAX]>, DwarfRegNum<[0, -2, -2]>;
  def RDX : RegisterWithSubRegs<"rdx", [EDX]>, DwarfRegNum<[1, -2, -2]>;
  def RCX : RegisterWithSubRegs<"rcx", [ECX]>, DwarfRegNum<[2, -2, -2]>;
  def RBX : RegisterWithSubRegs<"rbx", [EBX]>, DwarfRegNum<[3, -2, -2]>;
  def RSI : RegisterWithSubRegs<"rsi", [ESI]>, DwarfRegNum<[4, -2, -2]>;
  def RDI : RegisterWithSubRegs<"rdi", [EDI]>, DwarfRegNum<[5, -2, -2]>;
  def RBP : RegisterWithSubRegs<"rbp", [EBP]>, DwarfRegNum<[6, -2, -2]>;
  def RSP : RegisterWithSubRegs<"rsp", [ESP]>, DwarfRegNum<[7, -2, -2]>;

  def R8  : RegisterWithSubRegs<"r8", [R8D]>, DwarfRegNum<[8, -2, -2]>;
  def R9  : RegisterWithSubRegs<"r9", [R9D]>, DwarfRegNum<[9, -2, -2]>;
  def R10 : RegisterWithSubRegs<"r10", [R10D]>, DwarfRegNum<[10, -2, -2]>;
  def R11 : RegisterWithSubRegs<"r11", [R11D]>, DwarfRegNum<[11, -2, -2]>;
  def R12 : RegisterWithSubRegs<"r12", [R12D]>, DwarfRegNum<[12, -2, -2]>;
  def R13 : RegisterWithSubRegs<"r13", [R13D]>, DwarfRegNum<[13, -2, -2]>;
  def R14 : RegisterWithSubRegs<"r14", [R14D]>, DwarfRegNum<[14, -2, -2]>;
  def R15 : RegisterWithSubRegs<"r15", [R15D]>, DwarfRegNum<[15, -2, -2]>;
  def RIP : RegisterWithSubRegs<"rip", [EIP]>,  DwarfRegNum<[16, -2, -2]>;

  // MMX Registers. These are actually aliased to ST0 .. ST7
  def MM0 : Register<"mm0">, DwarfRegNum<[41, 29, 29]>;
  def MM1 : Register<"mm1">, DwarfRegNum<[42, 30, 30]>;
  def MM2 : Register<"mm2">, DwarfRegNum<[43, 31, 31]>;
  def MM3 : Register<"mm3">, DwarfRegNum<[44, 32, 32]>;
  def MM4 : Register<"mm4">, DwarfRegNum<[45, 33, 33]>;
  def MM5 : Register<"mm5">, DwarfRegNum<[46, 34, 34]>;
  def MM6 : Register<"mm6">, DwarfRegNum<[47, 35, 35]>;
  def MM7 : Register<"mm7">, DwarfRegNum<[48, 36, 36]>;
  
  // Pseudo Floating Point registers
  def FP0 : Register<"fp0">;
  def FP1 : Register<"fp1">;
  def FP2 : Register<"fp2">;
  def FP3 : Register<"fp3">;
  def FP4 : Register<"fp4">;
  def FP5 : Register<"fp5">;
  def FP6 : Register<"fp6">; 

  // XMM Registers, used by the various SSE instruction set extensions
  def XMM0: Register<"xmm0">, DwarfRegNum<[17, 21, 21]>;
  def XMM1: Register<"xmm1">, DwarfRegNum<[18, 22, 22]>;
  def XMM2: Register<"xmm2">, DwarfRegNum<[19, 23, 23]>;
  def XMM3: Register<"xmm3">, DwarfRegNum<[20, 24, 24]>;
  def XMM4: Register<"xmm4">, DwarfRegNum<[21, 25, 25]>;
  def XMM5: Register<"xmm5">, DwarfRegNum<[22, 26, 26]>;
  def XMM6: Register<"xmm6">, DwarfRegNum<[23, 27, 27]>;
  def XMM7: Register<"xmm7">, DwarfRegNum<[24, 28, 28]>;

  // X86-64 only
  def XMM8:  Register<"xmm8">,  DwarfRegNum<[25, -2, -2]>;
  def XMM9:  Register<"xmm9">,  DwarfRegNum<[26, -2, -2]>;
  def XMM10: Register<"xmm10">, DwarfRegNum<[27, -2, -2]>;
  def XMM11: Register<"xmm11">, DwarfRegNum<[28, -2, -2]>;
  def XMM12: Register<"xmm12">, DwarfRegNum<[29, -2, -2]>;
  def XMM13: Register<"xmm13">, DwarfRegNum<[30, -2, -2]>;
  def XMM14: Register<"xmm14">, DwarfRegNum<[31, -2, -2]>;
  def XMM15: Register<"xmm15">, DwarfRegNum<[32, -2, -2]>;

  // YMM Registers, used by AVX instructions
  def YMM0: RegisterWithSubRegs<"ymm0", [XMM0]>, DwarfRegNum<[17, 21, 21]>;
  def YMM1: RegisterWithSubRegs<"ymm1", [XMM1]>, DwarfRegNum<[18, 22, 22]>;
  def YMM2: RegisterWithSubRegs<"ymm2", [XMM2]>, DwarfRegNum<[19, 23, 23]>;
  def YMM3: RegisterWithSubRegs<"ymm3", [XMM3]>, DwarfRegNum<[20, 24, 24]>;
  def YMM4: RegisterWithSubRegs<"ymm4", [XMM4]>, DwarfRegNum<[21, 25, 25]>;
  def YMM5: RegisterWithSubRegs<"ymm5", [XMM5]>, DwarfRegNum<[22, 26, 26]>;
  def YMM6: RegisterWithSubRegs<"ymm6", [XMM6]>, DwarfRegNum<[23, 27, 27]>;
  def YMM7: RegisterWithSubRegs<"ymm7", [XMM7]>, DwarfRegNum<[24, 28, 28]>;
  def YMM8:  RegisterWithSubRegs<"ymm8", [XMM8]>,  DwarfRegNum<[25, -2, -2]>;
  def YMM9:  RegisterWithSubRegs<"ymm9", [XMM9]>,  DwarfRegNum<[26, -2, -2]>;
  def YMM10: RegisterWithSubRegs<"ymm10", [XMM10]>, DwarfRegNum<[27, -2, -2]>;
  def YMM11: RegisterWithSubRegs<"ymm11", [XMM11]>, DwarfRegNum<[28, -2, -2]>;
  def YMM12: RegisterWithSubRegs<"ymm12", [XMM12]>, DwarfRegNum<[29, -2, -2]>;
  def YMM13: RegisterWithSubRegs<"ymm13", [XMM13]>, DwarfRegNum<[30, -2, -2]>;
  def YMM14: RegisterWithSubRegs<"ymm14", [XMM14]>, DwarfRegNum<[31, -2, -2]>;
  def YMM15: RegisterWithSubRegs<"ymm15", [XMM15]>, DwarfRegNum<[32, -2, -2]>;

  // Floating point stack registers
  def ST0 : Register<"st(0)">, DwarfRegNum<[33, 12, 11]>;
  def ST1 : Register<"st(1)">, DwarfRegNum<[34, 13, 12]>;
  def ST2 : Register<"st(2)">, DwarfRegNum<[35, 14, 13]>;
  def ST3 : Register<"st(3)">, DwarfRegNum<[36, 15, 14]>;
  def ST4 : Register<"st(4)">, DwarfRegNum<[37, 16, 15]>;
  def ST5 : Register<"st(5)">, DwarfRegNum<[38, 17, 16]>;
  def ST6 : Register<"st(6)">, DwarfRegNum<[39, 18, 17]>;
  def ST7 : Register<"st(7)">, DwarfRegNum<[40, 19, 18]>; 

  // Status flags register
  def EFLAGS : Register<"flags">;

  // Segment registers
  def CS : Register<"cs">;
  def DS : Register<"ds">;
  def SS : Register<"ss">;
  def ES : Register<"es">;
  def FS : Register<"fs">;
  def GS : Register<"gs">;
  
  // Debug registers
  def DR0 : Register<"dr0">;
  def DR1 : Register<"dr1">;
  def DR2 : Register<"dr2">;
  def DR3 : Register<"dr3">;
  def DR4 : Register<"dr4">;
  def DR5 : Register<"dr5">;
  def DR6 : Register<"dr6">;
  def DR7 : Register<"dr7">;
  
  // Condition registers
  def ECR0 : Register<"ecr0">;
  def ECR1 : Register<"ecr1">;
  def ECR2 : Register<"ecr2">;
  def ECR3 : Register<"ecr3">;
  def ECR4 : Register<"ecr4">;
  def ECR5 : Register<"ecr5">;
  def ECR6 : Register<"ecr6">;
  def ECR7 : Register<"ecr7">;

  def RCR0 : Register<"rcr0">;
  def RCR1 : Register<"rcr1">;
  def RCR2 : Register<"rcr2">;
  def RCR3 : Register<"rcr3">;
  def RCR4 : Register<"rcr4">;
  def RCR5 : Register<"rcr5">;
  def RCR6 : Register<"rcr6">;
  def RCR7 : Register<"rcr7">;
  def RCR8 : Register<"rcr8">; 
}


//===----------------------------------------------------------------------===//
// Subregister Set Definitions... now that we have all of the pieces, define the
// sub registers for each register.
//

def x86_subreg_8bit    : PatLeaf<(i32 1)>;
def x86_subreg_8bit_hi : PatLeaf<(i32 2)>;
def x86_subreg_16bit   : PatLeaf<(i32 3)>;
def x86_subreg_32bit   : PatLeaf<(i32 4)>;

def x86_subreg_ss   : PatLeaf<(i32 1)>;
def x86_subreg_sd   : PatLeaf<(i32 2)>;
def x86_subreg_xmm  : PatLeaf<(i32 3)>;

def : SubRegSet<1, [AX, CX, DX, BX, SP,  BP,  SI,  DI,  
                    R8W, R9W, R10W, R11W, R12W, R13W, R14W, R15W],
                   [AL, CL, DL, BL, SPL, BPL, SIL, DIL, 
                    R8B, R9B, R10B, R11B, R12B, R13B, R14B, R15B]>;

def : SubRegSet<2, [AX, CX, DX, BX],
                   [AH, CH, DH, BH]>;

def : SubRegSet<1, [EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI,  
                    R8D, R9D, R10D, R11D, R12D, R13D, R14D, R15D],
                   [AL, CL, DL, BL, SPL, BPL, SIL, DIL, 
                    R8B, R9B, R10B, R11B, R12B, R13B, R14B, R15B]>;

def : SubRegSet<2, [EAX, ECX, EDX, EBX],
                   [AH, CH, DH, BH]>;

def : SubRegSet<3, [EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI,
                    R8D, R9D, R10D, R11D, R12D, R13D, R14D, R15D],
                   [AX,  CX,  DX,  BX,  SP,  BP,  SI,  DI, 
                    R8W, R9W, R10W, R11W, R12W, R13W, R14W, R15W]>;

def : SubRegSet<1, [RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,  
                    R8,  R9,  R10, R11, R12, R13, R14, R15],
                   [AL, CL, DL, BL, SPL, BPL, SIL, DIL, 
                    R8B, R9B, R10B, R11B, R12B, R13B, R14B, R15B]>;

def : SubRegSet<2, [RAX, RCX, RDX, RBX],
                   [AH, CH, DH, BH]>;

def : SubRegSet<3, [RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
                    R8,  R9,  R10, R11, R12, R13, R14, R15],
                   [AX,  CX,  DX,  BX,  SP,  BP,  SI,  DI, 
                    R8W, R9W, R10W, R11W, R12W, R13W, R14W, R15W]>;

def : SubRegSet<4, [RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
                    R8,  R9,  R10, R11, R12, R13, R14, R15],
                   [EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI, 
                    R8D, R9D, R10D, R11D, R12D, R13D, R14D, R15D]>;

def : SubRegSet<1, [YMM0, YMM1, YMM2, YMM3, YMM4, YMM5, YMM6, YMM7,
                    YMM8, YMM9, YMM10, YMM11, YMM12, YMM13, YMM14, YMM15],
                   [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                    XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15]>;

def : SubRegSet<2, [YMM0, YMM1, YMM2, YMM3, YMM4, YMM5, YMM6, YMM7,
                    YMM8, YMM9, YMM10, YMM11, YMM12, YMM13, YMM14, YMM15],
                   [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                    XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15]>;

def : SubRegSet<3, [YMM0, YMM1, YMM2, YMM3, YMM4, YMM5, YMM6, YMM7,  
                    YMM8, YMM9, YMM10, YMM11, YMM12, YMM13, YMM14, YMM15],
                   [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, 
                    XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15]>;

def : SubRegSet<1, [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                    XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15],
                   [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                    XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15]>;

def : SubRegSet<2, [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                    XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15],
                   [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                    XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15]>;

//===----------------------------------------------------------------------===//
// Register Class Definitions... now that we have all of the pieces, define the
// top-level register classes.  The order specified in the register list is
// implicitly defined to be the register allocation order.
//

// List call-clobbered registers before callee-save registers. RBX, RBP, (and 
// R12, R13, R14, and R15 for X86-64) are callee-save registers.
// In 64-mode, there are 12 additional i8 registers, SIL, DIL, BPL, SPL, and
// R8B, ... R15B. 
// Allocate R12 and R13 last, as these require an extra byte when
// encoded in x86_64 instructions.
// FIXME: Allow AH, CH, DH, BH to be used as general-purpose registers in
// 64-bit mode. The main complication is that they cannot be encoded in an
// instruction requiring a REX prefix, while SIL, DIL, BPL, R8D, etc.
// require a REX prefix. For example, "addb %ah, %dil" and "movzbl %ah, %r8d"
// cannot be encoded.
def GR8 : RegisterClass<"X86", [i8],  8,
                        [AL, CL, DL, AH, CH, DH, BL, BH, SIL, DIL, BPL, SPL,
                         R8B, R9B, R10B, R11B, R14B, R15B, R12B, R13B]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned X86_GR8_AO_64[] = {
      X86::AL,   X86::CL,   X86::DL,   X86::SIL, X86::DIL,
      X86::R8B,  X86::R9B,  X86::R10B, X86::R11B,
      X86::BL,   X86::R14B, X86::R15B, X86::R12B, X86::R13B, X86::BPL
    };

    GR8Class::iterator
    GR8Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (Subtarget.is64Bit())
        return X86_GR8_AO_64;
      else
        return begin();
    }

    GR8Class::iterator
    GR8Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      const X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
      // Does the function dedicate RBP / EBP to being a frame ptr?
      if (!Subtarget.is64Bit())
        // In 32-mode, none of the 8-bit registers aliases EBP or ESP.
        return begin() + 8;
      else if (RI->hasFP(MF) || MFI->getReserveFP())
        // If so, don't allocate SPL or BPL.
        return array_endof(X86_GR8_AO_64) - 1;
      else
        // If not, just don't allocate SPL.
        return array_endof(X86_GR8_AO_64);
    }
  }];
}

def GR16 : RegisterClass<"X86", [i16], 16,
                         [AX, CX, DX, SI, DI, BX, BP, SP,
                          R8W, R9W, R10W, R11W, R14W, R15W, R12W, R13W]> {
  let SubRegClassList = [GR8, GR8];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned X86_GR16_AO_64[] = {
      X86::AX,  X86::CX,   X86::DX,   X86::SI,   X86::DI,
      X86::R8W, X86::R9W,  X86::R10W, X86::R11W,
      X86::BX, X86::R14W, X86::R15W,  X86::R12W, X86::R13W, X86::BP
    };

    GR16Class::iterator
    GR16Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (Subtarget.is64Bit())
        return X86_GR16_AO_64;
      else
        return begin();
    }

    GR16Class::iterator
    GR16Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      const X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
      if (Subtarget.is64Bit()) {
        // Does the function dedicate RBP to being a frame ptr?
        if (RI->hasFP(MF) || MFI->getReserveFP())
          // If so, don't allocate SP or BP.
          return array_endof(X86_GR16_AO_64) - 1;
        else
          // If not, just don't allocate SP.
          return array_endof(X86_GR16_AO_64);
      } else {
        // Does the function dedicate EBP to being a frame ptr?
        if (RI->hasFP(MF) || MFI->getReserveFP())
          // If so, don't allocate SP or BP.
          return begin() + 6;
        else
          // If not, just don't allocate SP.
          return begin() + 7;
      }
    }
  }];
}

def GR32 : RegisterClass<"X86", [i32], 32, 
                         [EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP,
                          R8D, R9D, R10D, R11D, R14D, R15D, R12D, R13D]> {
  let SubRegClassList = [GR8, GR8, GR16];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned X86_GR32_AO_64[] = {
      X86::EAX, X86::ECX,  X86::EDX,  X86::ESI,  X86::EDI,
      X86::R8D, X86::R9D,  X86::R10D, X86::R11D,
      X86::EBX, X86::R14D, X86::R15D, X86::R12D, X86::R13D, X86::EBP
    };

    GR32Class::iterator
    GR32Class::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (Subtarget.is64Bit())
        return X86_GR32_AO_64;
      else
        return begin();
    }

    GR32Class::iterator
    GR32Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      const X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
      if (Subtarget.is64Bit()) {
        // Does the function dedicate RBP to being a frame ptr?
        if (RI->hasFP(MF) || MFI->getReserveFP())
          // If so, don't allocate ESP or EBP.
          return array_endof(X86_GR32_AO_64) - 1;
        else
          // If not, just don't allocate ESP.
          return array_endof(X86_GR32_AO_64);
      } else {
        // Does the function dedicate EBP to being a frame ptr?
        if (RI->hasFP(MF) || MFI->getReserveFP())
          // If so, don't allocate ESP or EBP.
          return begin() + 6;
        else
          // If not, just don't allocate ESP.
          return begin() + 7;
      }
    }
  }];
}

// GR64 - 64-bit GPRs. This oddly includes RIP, which isn't accurate, since
// RIP isn't really a register and it can't be used anywhere except in an
// address, but it doesn't cause trouble.
def GR64 : RegisterClass<"X86", [i64], 64, 
                         [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
                          RBX, R14, R15, R12, R13, RBP, RSP, RIP]> {
  let SubRegClassList = [GR8, GR8, GR16, GR32];
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR64Class::iterator
    GR64Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      const X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
      if (!Subtarget.is64Bit())
        return begin();  // None of these are allocatable in 32-bit.
      // Does the function dedicate RBP to being a frame ptr?
      if (RI->hasFP(MF) || MFI->getReserveFP())
        return end()-3;  // If so, don't allocate RIP, RSP or RBP
      else
        return end()-2;  // If not, just don't allocate RIP or RSP
    }
  }];
}

// Segment registers for use by MOV instructions (and others) that have a
//   segment register as one operand.  Always contain a 16-bit segment
//   descriptor.
def SEGMENT_REG : RegisterClass<"X86", [i16], 16, [CS, DS, SS, ES, FS, GS]> {
}

// Debug registers.
def DEBUG_REG : RegisterClass<"X86", [i32], 32, 
                              [DR0, DR1, DR2, DR3, DR4, DR5, DR6, DR7]> {
}

// Control registers.
def CONTROL_REG_32 : RegisterClass<"X86", [i32], 32,
                                   [ECR0, ECR1, ECR2, ECR3, ECR4, ECR5, ECR6,
                                    ECR7]> {
}

def CONTROL_REG_64 : RegisterClass<"X86", [i64], 64,
                                   [RCR0, RCR1, RCR2, RCR3, RCR4, RCR5, RCR6,
                                    RCR7, RCR8]> {
}

// GR8_ABCD_L, GR8_ABCD_H, GR16_ABCD, GR32_ABCD, GR64_ABCD - Subclasses of
// GR8, GR16, GR32, and GR64 which contain just the "a" "b", "c", and "d"
// registers. On x86-32, GR16_ABCD and GR32_ABCD are classes for registers
// that support 8-bit subreg operations. On x86-64, GR16_ABCD, GR32_ABCD,
// and GR64_ABCD are classes for registers that support 8-bit h-register
// operations.
def GR8_ABCD_L : RegisterClass<"X86", [i8], 8, [AL, CL, DL, BL]> {
}
def GR8_ABCD_H : RegisterClass<"X86", [i8], 8, [AH, CH, DH, BH]> {
}
def GR16_ABCD : RegisterClass<"X86", [i16], 16, [AX, CX, DX, BX]> {
  let SubRegClassList = [GR8_ABCD_L, GR8_ABCD_H];
}
def GR32_ABCD : RegisterClass<"X86", [i32], 32, [EAX, ECX, EDX, EBX]> {
  let SubRegClassList = [GR8_ABCD_L, GR8_ABCD_H, GR16_ABCD];
}
def GR64_ABCD : RegisterClass<"X86", [i64], 64, [RAX, RCX, RDX, RBX]> {
  let SubRegClassList = [GR8_ABCD_L, GR8_ABCD_H, GR16_ABCD, GR32_ABCD];
}

// GR8_NOREX - GR8 registers which do not require a REX prefix.
def GR8_NOREX : RegisterClass<"X86", [i8], 8,
                              [AL, CL, DL, AH, CH, DH, BL, BH]> {
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    // In 64-bit mode, it's not safe to blindly allocate H registers.
    static const unsigned X86_GR8_NOREX_AO_64[] = {
      X86::AL, X86::CL, X86::DL, X86::BL
    };

    GR8_NOREXClass::iterator
    GR8_NOREXClass::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (Subtarget.is64Bit())
        return X86_GR8_NOREX_AO_64;
      else
        return begin();
    }

    GR8_NOREXClass::iterator
    GR8_NOREXClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (Subtarget.is64Bit())
        return array_endof(X86_GR8_NOREX_AO_64);
      else
        return end();
    }
  }];
}
// GR16_NOREX - GR16 registers which do not require a REX prefix.
def GR16_NOREX : RegisterClass<"X86", [i16], 16,
                               [AX, CX, DX, SI, DI, BX, BP, SP]> {
  let SubRegClassList = [GR8_NOREX, GR8_NOREX];
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR16_NOREXClass::iterator
    GR16_NOREXClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
      // Does the function dedicate RBP / EBP to being a frame ptr?
      if (RI->hasFP(MF) || MFI->getReserveFP())
        // If so, don't allocate SP or BP.
        return end() - 2;
      else
        // If not, just don't allocate SP.
        return end() - 1;
    }
  }];
}
// GR32_NOREX - GR32 registers which do not require a REX prefix.
def GR32_NOREX : RegisterClass<"X86", [i32], 32,
                               [EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP]> {
  let SubRegClassList = [GR8_NOREX, GR8_NOREX, GR16_NOREX];
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR32_NOREXClass::iterator
    GR32_NOREXClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
      // Does the function dedicate RBP / EBP to being a frame ptr?
      if (RI->hasFP(MF) || MFI->getReserveFP())
        // If so, don't allocate ESP or EBP.
        return end() - 2;
      else
        // If not, just don't allocate ESP.
        return end() - 1;
    }
  }];
}
// GR64_NOREX - GR64 registers which do not require a REX prefix.
def GR64_NOREX : RegisterClass<"X86", [i64], 64,
                               [RAX, RCX, RDX, RSI, RDI, RBX, RBP, RSP, RIP]> {
  let SubRegClassList = [GR8_NOREX, GR8_NOREX, GR16_NOREX, GR32_NOREX];
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR64_NOREXClass::iterator
    GR64_NOREXClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
      // Does the function dedicate RBP to being a frame ptr?
      if (RI->hasFP(MF) || MFI->getReserveFP())
        // If so, don't allocate RIP, RSP or RBP.
        return end() - 3;
      else
        // If not, just don't allocate RIP or RSP.
        return end() - 2;
    }
  }];
}

// GR32_NOSP - GR32 registers except ESP.
def GR32_NOSP : RegisterClass<"X86", [i32], 32,
                              [EAX, ECX, EDX, ESI, EDI, EBX, EBP,
                               R8D, R9D, R10D, R11D, R14D, R15D, R12D, R13D]> {
  let SubRegClassList = [GR8, GR8, GR16];
  let MethodProtos = [{
    iterator allocation_order_begin(const MachineFunction &MF) const;
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    static const unsigned X86_GR32_NOSP_AO_64[] = {
      X86::EAX, X86::ECX,  X86::EDX,  X86::ESI,  X86::EDI,
      X86::R8D, X86::R9D,  X86::R10D, X86::R11D,
      X86::EBX, X86::R14D, X86::R15D, X86::R12D, X86::R13D, X86::EBP
    };

    GR32_NOSPClass::iterator
    GR32_NOSPClass::allocation_order_begin(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (Subtarget.is64Bit())
        return X86_GR32_NOSP_AO_64;
      else
        return begin();
    }

    GR32_NOSPClass::iterator
    GR32_NOSPClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      const X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
      if (Subtarget.is64Bit()) {
        // Does the function dedicate RBP to being a frame ptr?
        if (RI->hasFP(MF) || MFI->getReserveFP())
          // If so, don't allocate EBP.
          return array_endof(X86_GR32_NOSP_AO_64) - 1;
        else
          // If not, any reg in this class is ok.
          return array_endof(X86_GR32_NOSP_AO_64);
      } else {
        // Does the function dedicate EBP to being a frame ptr?
        if (RI->hasFP(MF) || MFI->getReserveFP())
          // If so, don't allocate EBP.
          return begin() + 6;
        else
          // If not, any reg in this class is ok.
          return begin() + 7;
      }
    }
  }];
}

// GR64_NOSP - GR64 registers except RSP (and RIP).
def GR64_NOSP : RegisterClass<"X86", [i64], 64,
                              [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
                               RBX, R14, R15, R12, R13, RBP]> {
  let SubRegClassList = [GR8, GR8, GR16, GR32_NOSP];
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR64_NOSPClass::iterator
    GR64_NOSPClass::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      const X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
      if (!Subtarget.is64Bit())
        return begin();  // None of these are allocatable in 32-bit.
      // Does the function dedicate RBP to being a frame ptr?
      if (RI->hasFP(MF) || MFI->getReserveFP())
        return end()-1;  // If so, don't allocate RBP
      else
        return end();  // If not, any reg in this class is ok.
    }
  }];
}

// GR64_NOREX_NOSP - GR64_NOREX registers except RSP.
def GR64_NOREX_NOSP : RegisterClass<"X86", [i64], 64,
                                    [RAX, RCX, RDX, RSI, RDI, RBX, RBP]> {
  let SubRegClassList = [GR8_NOREX, GR8_NOREX, GR16_NOREX, GR32_NOREX];
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    GR64_NOREX_NOSPClass::iterator
    GR64_NOREX_NOSPClass::allocation_order_end(const MachineFunction &MF) const
  {
      const TargetMachine &TM = MF.getTarget();
      const TargetRegisterInfo *RI = TM.getRegisterInfo();
      const X86MachineFunctionInfo *MFI = MF.getInfo<X86MachineFunctionInfo>();
      // Does the function dedicate RBP to being a frame ptr?
      if (RI->hasFP(MF) || MFI->getReserveFP())
        // If so, don't allocate RBP.
        return end() - 1;
      else
        // If not, any reg in this class is ok.
        return end();
    }
  }];
}

// A class to support the 'A' assembler constraint: EAX then EDX.
def GR32_AD : RegisterClass<"X86", [i32], 32, [EAX, EDX]> {
  let SubRegClassList = [GR8_ABCD_L, GR8_ABCD_H, GR16_ABCD];
}

// Scalar SSE2 floating point registers.
def FR32 : RegisterClass<"X86", [f32], 32,
                         [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                          XMM8, XMM9, XMM10, XMM11,
                          XMM12, XMM13, XMM14, XMM15]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    FR32Class::iterator
    FR32Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (!Subtarget.is64Bit())
        return end()-8; // Only XMM0 to XMM7 are available in 32-bit mode.
      else
        return end();
    }
  }];
}

def FR64 : RegisterClass<"X86", [f64], 64,
                         [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                          XMM8, XMM9, XMM10, XMM11,
                          XMM12, XMM13, XMM14, XMM15]> {
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    FR64Class::iterator
    FR64Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (!Subtarget.is64Bit())
        return end()-8; // Only XMM0 to XMM7 are available in 32-bit mode.
      else
        return end();
    }
  }];
}


// FIXME: This sets up the floating point register files as though they are f64
// values, though they really are f80 values.  This will cause us to spill
// values as 64-bit quantities instead of 80-bit quantities, which is much much
// faster on common hardware.  In reality, this should be controlled by a
// command line option or something.

def RFP32 : RegisterClass<"X86",[f32], 32, [FP0, FP1, FP2, FP3, FP4, FP5, FP6]>;
def RFP64 : RegisterClass<"X86",[f64], 32, [FP0, FP1, FP2, FP3, FP4, FP5, FP6]>;
def RFP80 : RegisterClass<"X86",[f80], 32, [FP0, FP1, FP2, FP3, FP4, FP5, FP6]>;

// Floating point stack registers (these are not allocatable by the
// register allocator - the floating point stackifier is responsible
// for transforming FPn allocations to STn registers)
def RST : RegisterClass<"X86", [f80, f64, f32], 32,
                        [ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7]> {
    let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    RSTClass::iterator
    RSTClass::allocation_order_end(const MachineFunction &MF) const {
      return begin();
    }
  }];
}

// Generic vector registers: VR64 and VR128.
def VR64  : RegisterClass<"X86", [v8i8, v4i16, v2i32, v1i64, v2f32], 64,
                          [MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7]>;
def VR128 : RegisterClass<"X86", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],128,
                          [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
                           XMM8, XMM9, XMM10, XMM11,
                           XMM12, XMM13, XMM14, XMM15]> {
  let SubRegClassList = [FR32, FR64];
  let MethodProtos = [{
    iterator allocation_order_end(const MachineFunction &MF) const;
  }];
  let MethodBodies = [{
    VR128Class::iterator
    VR128Class::allocation_order_end(const MachineFunction &MF) const {
      const TargetMachine &TM = MF.getTarget();
      const X86Subtarget &Subtarget = TM.getSubtarget<X86Subtarget>();
      if (!Subtarget.is64Bit())
        return end()-8; // Only XMM0 to XMM7 are available in 32-bit mode.
      else
        return end();
    }
  }];
}
def VR256 : RegisterClass<"X86", [ v8i32, v4i64, v8f32, v4f64],256,
                          [YMM0, YMM1, YMM2, YMM3, YMM4, YMM5, YMM6, YMM7,
                           YMM8, YMM9, YMM10, YMM11,
                           YMM12, YMM13, YMM14, YMM15]> {
  let SubRegClassList = [FR32, FR64, VR128];
}

// Status flags registers.
def CCR : RegisterClass<"X86", [i32], 32, [EFLAGS]> {
  let CopyCost = -1;  // Don't allow copying of status registers.
}
