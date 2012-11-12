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

class X86Reg<string n, bits<16> Enc, list<Register> subregs = []> : Register<n> {
  let Namespace = "X86";
  let HWEncoding = Enc;
  let SubRegs = subregs;
}

// Subregister indices.
let Namespace = "X86" in {
  def sub_8bit    : SubRegIndex;
  def sub_8bit_hi : SubRegIndex;
  def sub_16bit   : SubRegIndex;
  def sub_32bit   : SubRegIndex;
  def sub_xmm     : SubRegIndex;
}

//===----------------------------------------------------------------------===//
//  Register definitions...
//

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
def AL : X86Reg<"al", 0>;
def DL : X86Reg<"dl", 2>;
def CL : X86Reg<"cl", 1>;
def BL : X86Reg<"bl", 3>;

// High registers. On x86-64, these cannot be used in any instruction
// with a REX prefix.
def AH : X86Reg<"ah", 4>;
def DH : X86Reg<"dh", 6>;
def CH : X86Reg<"ch", 5>;
def BH : X86Reg<"bh", 7>;

// X86-64 only, requires REX.
let CostPerUse = 1 in {
def SIL  : X86Reg<"sil",   6>;
def DIL  : X86Reg<"dil",   7>;
def BPL  : X86Reg<"bpl",   5>;
def SPL  : X86Reg<"spl",   4>;
def R8B  : X86Reg<"r8b",   8>;
def R9B  : X86Reg<"r9b",   9>;
def R10B : X86Reg<"r10b", 10>;
def R11B : X86Reg<"r11b", 11>;
def R12B : X86Reg<"r12b", 12>;
def R13B : X86Reg<"r13b", 13>;
def R14B : X86Reg<"r14b", 14>;
def R15B : X86Reg<"r15b", 15>;
}

// 16-bit registers
let SubRegIndices = [sub_8bit, sub_8bit_hi], CoveredBySubRegs = 1 in {
def AX : X86Reg<"ax", 0, [AL,AH]>;
def DX : X86Reg<"dx", 2, [DL,DH]>;
def CX : X86Reg<"cx", 1, [CL,CH]>;
def BX : X86Reg<"bx", 3, [BL,BH]>;
}
let SubRegIndices = [sub_8bit] in {
def SI : X86Reg<"si", 6, [SIL]>;
def DI : X86Reg<"di", 7, [DIL]>;
def BP : X86Reg<"bp", 5, [BPL]>;
def SP : X86Reg<"sp", 4, [SPL]>;
}
def IP : X86Reg<"ip", 0>;

// X86-64 only, requires REX.
let SubRegIndices = [sub_8bit], CostPerUse = 1 in {
def R8W  : X86Reg<"r8w",   8, [R8B]>;
def R9W  : X86Reg<"r9w",   9, [R9B]>;
def R10W : X86Reg<"r10w", 10, [R10B]>;
def R11W : X86Reg<"r11w", 11, [R11B]>;
def R12W : X86Reg<"r12w", 12, [R12B]>;
def R13W : X86Reg<"r13w", 13, [R13B]>;
def R14W : X86Reg<"r14w", 14, [R14B]>;
def R15W : X86Reg<"r15w", 15, [R15B]>;
}

// 32-bit registers
let SubRegIndices = [sub_16bit] in {
def EAX : X86Reg<"eax", 0, [AX]>, DwarfRegNum<[-2, 0, 0]>;
def EDX : X86Reg<"edx", 2, [DX]>, DwarfRegNum<[-2, 2, 2]>;
def ECX : X86Reg<"ecx", 1, [CX]>, DwarfRegNum<[-2, 1, 1]>;
def EBX : X86Reg<"ebx", 3, [BX]>, DwarfRegNum<[-2, 3, 3]>;
def ESI : X86Reg<"esi", 6, [SI]>, DwarfRegNum<[-2, 6, 6]>;
def EDI : X86Reg<"edi", 7, [DI]>, DwarfRegNum<[-2, 7, 7]>;
def EBP : X86Reg<"ebp", 5, [BP]>, DwarfRegNum<[-2, 4, 5]>;
def ESP : X86Reg<"esp", 4, [SP]>, DwarfRegNum<[-2, 5, 4]>;
def EIP : X86Reg<"eip", 0, [IP]>, DwarfRegNum<[-2, 8, 8]>;

// X86-64 only, requires REX
let CostPerUse = 1 in {
def R8D  : X86Reg<"r8d",   8, [R8W]>;
def R9D  : X86Reg<"r9d",   9, [R9W]>;
def R10D : X86Reg<"r10d", 10, [R10W]>;
def R11D : X86Reg<"r11d", 11, [R11W]>;
def R12D : X86Reg<"r12d", 12, [R12W]>;
def R13D : X86Reg<"r13d", 13, [R13W]>;
def R14D : X86Reg<"r14d", 14, [R14W]>;
def R15D : X86Reg<"r15d", 15, [R15W]>;
}}

// 64-bit registers, X86-64 only
let SubRegIndices = [sub_32bit] in {
def RAX : X86Reg<"rax", 0, [EAX]>, DwarfRegNum<[0, -2, -2]>;
def RDX : X86Reg<"rdx", 2, [EDX]>, DwarfRegNum<[1, -2, -2]>;
def RCX : X86Reg<"rcx", 1, [ECX]>, DwarfRegNum<[2, -2, -2]>;
def RBX : X86Reg<"rbx", 3, [EBX]>, DwarfRegNum<[3, -2, -2]>;
def RSI : X86Reg<"rsi", 6, [ESI]>, DwarfRegNum<[4, -2, -2]>;
def RDI : X86Reg<"rdi", 7, [EDI]>, DwarfRegNum<[5, -2, -2]>;
def RBP : X86Reg<"rbp", 5, [EBP]>, DwarfRegNum<[6, -2, -2]>;
def RSP : X86Reg<"rsp", 4, [ESP]>, DwarfRegNum<[7, -2, -2]>;

// These also require REX.
let CostPerUse = 1 in {
def R8  : X86Reg<"r8",   8, [R8D]>,  DwarfRegNum<[ 8, -2, -2]>;
def R9  : X86Reg<"r9",   9, [R9D]>,  DwarfRegNum<[ 9, -2, -2]>;
def R10 : X86Reg<"r10", 10, [R10D]>, DwarfRegNum<[10, -2, -2]>;
def R11 : X86Reg<"r11", 11, [R11D]>, DwarfRegNum<[11, -2, -2]>;
def R12 : X86Reg<"r12", 12, [R12D]>, DwarfRegNum<[12, -2, -2]>;
def R13 : X86Reg<"r13", 13, [R13D]>, DwarfRegNum<[13, -2, -2]>;
def R14 : X86Reg<"r14", 14, [R14D]>, DwarfRegNum<[14, -2, -2]>;
def R15 : X86Reg<"r15", 15, [R15D]>, DwarfRegNum<[15, -2, -2]>;
def RIP : X86Reg<"rip",  0, [EIP]>,  DwarfRegNum<[16, -2, -2]>;
}}

// MMX Registers. These are actually aliased to ST0 .. ST7
def MM0 : X86Reg<"mm0", 0>, DwarfRegNum<[41, 29, 29]>;
def MM1 : X86Reg<"mm1", 1>, DwarfRegNum<[42, 30, 30]>;
def MM2 : X86Reg<"mm2", 2>, DwarfRegNum<[43, 31, 31]>;
def MM3 : X86Reg<"mm3", 3>, DwarfRegNum<[44, 32, 32]>;
def MM4 : X86Reg<"mm4", 4>, DwarfRegNum<[45, 33, 33]>;
def MM5 : X86Reg<"mm5", 5>, DwarfRegNum<[46, 34, 34]>;
def MM6 : X86Reg<"mm6", 6>, DwarfRegNum<[47, 35, 35]>;
def MM7 : X86Reg<"mm7", 7>, DwarfRegNum<[48, 36, 36]>;

// Pseudo Floating Point registers
def FP0 : X86Reg<"fp0", 0>;
def FP1 : X86Reg<"fp1", 0>;
def FP2 : X86Reg<"fp2", 0>;
def FP3 : X86Reg<"fp3", 0>;
def FP4 : X86Reg<"fp4", 0>;
def FP5 : X86Reg<"fp5", 0>;
def FP6 : X86Reg<"fp6", 0>;

// XMM Registers, used by the various SSE instruction set extensions.
def XMM0: X86Reg<"xmm0", 0>, DwarfRegNum<[17, 21, 21]>;
def XMM1: X86Reg<"xmm1", 1>, DwarfRegNum<[18, 22, 22]>;
def XMM2: X86Reg<"xmm2", 2>, DwarfRegNum<[19, 23, 23]>;
def XMM3: X86Reg<"xmm3", 3>, DwarfRegNum<[20, 24, 24]>;
def XMM4: X86Reg<"xmm4", 4>, DwarfRegNum<[21, 25, 25]>;
def XMM5: X86Reg<"xmm5", 5>, DwarfRegNum<[22, 26, 26]>;
def XMM6: X86Reg<"xmm6", 6>, DwarfRegNum<[23, 27, 27]>;
def XMM7: X86Reg<"xmm7", 7>, DwarfRegNum<[24, 28, 28]>;

// X86-64 only
let CostPerUse = 1 in {
def XMM8:  X86Reg<"xmm8",   8>, DwarfRegNum<[25, -2, -2]>;
def XMM9:  X86Reg<"xmm9",   9>, DwarfRegNum<[26, -2, -2]>;
def XMM10: X86Reg<"xmm10", 10>, DwarfRegNum<[27, -2, -2]>;
def XMM11: X86Reg<"xmm11", 11>, DwarfRegNum<[28, -2, -2]>;
def XMM12: X86Reg<"xmm12", 12>, DwarfRegNum<[29, -2, -2]>;
def XMM13: X86Reg<"xmm13", 13>, DwarfRegNum<[30, -2, -2]>;
def XMM14: X86Reg<"xmm14", 14>, DwarfRegNum<[31, -2, -2]>;
def XMM15: X86Reg<"xmm15", 15>, DwarfRegNum<[32, -2, -2]>;
} // CostPerUse

// YMM Registers, used by AVX instructions
let SubRegIndices = [sub_xmm] in {
def YMM0:  X86Reg<"ymm0",   0, [XMM0]>,  DwarfRegAlias<XMM0>;
def YMM1:  X86Reg<"ymm1",   1, [XMM1]>,  DwarfRegAlias<XMM1>;
def YMM2:  X86Reg<"ymm2",   2, [XMM2]>,  DwarfRegAlias<XMM2>;
def YMM3:  X86Reg<"ymm3",   3, [XMM3]>,  DwarfRegAlias<XMM3>;
def YMM4:  X86Reg<"ymm4",   4, [XMM4]>,  DwarfRegAlias<XMM4>;
def YMM5:  X86Reg<"ymm5",   5, [XMM5]>,  DwarfRegAlias<XMM5>;
def YMM6:  X86Reg<"ymm6",   6, [XMM6]>,  DwarfRegAlias<XMM6>;
def YMM7:  X86Reg<"ymm7",   7, [XMM7]>,  DwarfRegAlias<XMM7>;
def YMM8:  X86Reg<"ymm8",   8, [XMM8]>,  DwarfRegAlias<XMM8>;
def YMM9:  X86Reg<"ymm9",   9, [XMM9]>,  DwarfRegAlias<XMM9>;
def YMM10: X86Reg<"ymm10", 10, [XMM10]>, DwarfRegAlias<XMM10>;
def YMM11: X86Reg<"ymm11", 11, [XMM11]>, DwarfRegAlias<XMM11>;
def YMM12: X86Reg<"ymm12", 12, [XMM12]>, DwarfRegAlias<XMM12>;
def YMM13: X86Reg<"ymm13", 13, [XMM13]>, DwarfRegAlias<XMM13>;
def YMM14: X86Reg<"ymm14", 14, [XMM14]>, DwarfRegAlias<XMM14>;
def YMM15: X86Reg<"ymm15", 15, [XMM15]>, DwarfRegAlias<XMM15>;
}

class STRegister<string n, bits<16> Enc, list<Register> A> : X86Reg<n, Enc> {
  let Aliases = A;
}

// Floating point stack registers. These don't map one-to-one to the FP
// pseudo registers, but we still mark them as aliasing FP registers. That
// way both kinds can be live without exceeding the stack depth. ST registers
// are only live around inline assembly.
def ST0 : STRegister<"st(0)", 0, []>,    DwarfRegNum<[33, 12, 11]>;
def ST1 : STRegister<"st(1)", 1, [FP6]>, DwarfRegNum<[34, 13, 12]>;
def ST2 : STRegister<"st(2)", 2, [FP5]>, DwarfRegNum<[35, 14, 13]>;
def ST3 : STRegister<"st(3)", 3, [FP4]>, DwarfRegNum<[36, 15, 14]>;
def ST4 : STRegister<"st(4)", 4, [FP3]>, DwarfRegNum<[37, 16, 15]>;
def ST5 : STRegister<"st(5)", 5, [FP2]>, DwarfRegNum<[38, 17, 16]>;
def ST6 : STRegister<"st(6)", 6, [FP1]>, DwarfRegNum<[39, 18, 17]>;
def ST7 : STRegister<"st(7)", 7, [FP0]>, DwarfRegNum<[40, 19, 18]>;

// Floating-point status word
def FPSW : X86Reg<"fpsw", 0>;

// Status flags register
def EFLAGS : X86Reg<"flags", 0>;

// Segment registers
def CS : X86Reg<"cs", 1>;
def DS : X86Reg<"ds", 3>;
def SS : X86Reg<"ss", 2>;
def ES : X86Reg<"es", 0>;
def FS : X86Reg<"fs", 4>;
def GS : X86Reg<"gs", 5>;

// Debug registers
def DR0 : X86Reg<"dr0", 0>;
def DR1 : X86Reg<"dr1", 1>;
def DR2 : X86Reg<"dr2", 2>;
def DR3 : X86Reg<"dr3", 3>;
def DR4 : X86Reg<"dr4", 4>;
def DR5 : X86Reg<"dr5", 5>;
def DR6 : X86Reg<"dr6", 6>;
def DR7 : X86Reg<"dr7", 7>;

// Control registers
def CR0  : X86Reg<"cr0",   0>;
def CR1  : X86Reg<"cr1",   1>;
def CR2  : X86Reg<"cr2",   2>;
def CR3  : X86Reg<"cr3",   3>;
def CR4  : X86Reg<"cr4",   4>;
def CR5  : X86Reg<"cr5",   5>;
def CR6  : X86Reg<"cr6",   6>;
def CR7  : X86Reg<"cr7",   7>;
def CR8  : X86Reg<"cr8",   8>;
def CR9  : X86Reg<"cr9",   9>;
def CR10 : X86Reg<"cr10", 10>;
def CR11 : X86Reg<"cr11", 11>;
def CR12 : X86Reg<"cr12", 12>;
def CR13 : X86Reg<"cr13", 13>;
def CR14 : X86Reg<"cr14", 14>;
def CR15 : X86Reg<"cr15", 15>;

// Pseudo index registers
def EIZ : X86Reg<"eiz", 4>;
def RIZ : X86Reg<"riz", 4>;


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
                        (add AL, CL, DL, AH, CH, DH, BL, BH, SIL, DIL, BPL, SPL,
                             R8B, R9B, R10B, R11B, R14B, R15B, R12B, R13B)> {
  let AltOrders = [(sub GR8, AH, BH, CH, DH)];
  let AltOrderSelect = [{
    return MF.getTarget().getSubtarget<X86Subtarget>().is64Bit();
  }];
}

def GR16 : RegisterClass<"X86", [i16], 16,
                         (add AX, CX, DX, SI, DI, BX, BP, SP,
                              R8W, R9W, R10W, R11W, R14W, R15W, R12W, R13W)>;

def GR32 : RegisterClass<"X86", [i32], 32,
                         (add EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP,
                              R8D, R9D, R10D, R11D, R14D, R15D, R12D, R13D)>;

// GR64 - 64-bit GPRs. This oddly includes RIP, which isn't accurate, since
// RIP isn't really a register and it can't be used anywhere except in an
// address, but it doesn't cause trouble.
def GR64 : RegisterClass<"X86", [i64], 64,
                         (add RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
                              RBX, R14, R15, R12, R13, RBP, RSP, RIP)>;

// Segment registers for use by MOV instructions (and others) that have a
//   segment register as one operand.  Always contain a 16-bit segment
//   descriptor.
def SEGMENT_REG : RegisterClass<"X86", [i16], 16, (add CS, DS, SS, ES, FS, GS)>;

// Debug registers.
def DEBUG_REG : RegisterClass<"X86", [i32], 32, (sequence "DR%u", 0, 7)>;

// Control registers.
def CONTROL_REG : RegisterClass<"X86", [i64], 64, (sequence "CR%u", 0, 15)>;

// GR8_ABCD_L, GR8_ABCD_H, GR16_ABCD, GR32_ABCD, GR64_ABCD - Subclasses of
// GR8, GR16, GR32, and GR64 which contain just the "a" "b", "c", and "d"
// registers. On x86-32, GR16_ABCD and GR32_ABCD are classes for registers
// that support 8-bit subreg operations. On x86-64, GR16_ABCD, GR32_ABCD,
// and GR64_ABCD are classes for registers that support 8-bit h-register
// operations.
def GR8_ABCD_L : RegisterClass<"X86", [i8], 8, (add AL, CL, DL, BL)>;
def GR8_ABCD_H : RegisterClass<"X86", [i8], 8, (add AH, CH, DH, BH)>;
def GR16_ABCD : RegisterClass<"X86", [i16], 16, (add AX, CX, DX, BX)>;
def GR32_ABCD : RegisterClass<"X86", [i32], 32, (add EAX, ECX, EDX, EBX)>;
def GR64_ABCD : RegisterClass<"X86", [i64], 64, (add RAX, RCX, RDX, RBX)>;
def GR32_TC   : RegisterClass<"X86", [i32], 32, (add EAX, ECX, EDX)>;
def GR64_TC   : RegisterClass<"X86", [i64], 64, (add RAX, RCX, RDX, RSI, RDI,
                                                     R8, R9, R11, RIP)>;
def GR64_TCW64 : RegisterClass<"X86", [i64], 64, (add RAX, RCX, RDX,
                                                      R8, R9, R11)>;

// GR8_NOREX - GR8 registers which do not require a REX prefix.
def GR8_NOREX : RegisterClass<"X86", [i8], 8,
                              (add AL, CL, DL, AH, CH, DH, BL, BH)> {
  let AltOrders = [(sub GR8_NOREX, AH, BH, CH, DH)];
  let AltOrderSelect = [{
    return MF.getTarget().getSubtarget<X86Subtarget>().is64Bit();
  }];
}
// GR16_NOREX - GR16 registers which do not require a REX prefix.
def GR16_NOREX : RegisterClass<"X86", [i16], 16,
                               (add AX, CX, DX, SI, DI, BX, BP, SP)>;
// GR32_NOREX - GR32 registers which do not require a REX prefix.
def GR32_NOREX : RegisterClass<"X86", [i32], 32,
                               (add EAX, ECX, EDX, ESI, EDI, EBX, EBP, ESP)>;
// GR64_NOREX - GR64 registers which do not require a REX prefix.
def GR64_NOREX : RegisterClass<"X86", [i64], 64,
                            (add RAX, RCX, RDX, RSI, RDI, RBX, RBP, RSP, RIP)>;

// GR32_NOAX - GR32 registers except EAX. Used by AddRegFrm of XCHG32 in 64-bit
// mode to prevent encoding using the 0x90 NOP encoding. xchg %eax, %eax needs
// to clear upper 32-bits of RAX so is not a NOP.
def GR32_NOAX : RegisterClass<"X86", [i32], 32, (sub GR32, EAX)>;

// GR32_NOSP - GR32 registers except ESP.
def GR32_NOSP : RegisterClass<"X86", [i32], 32, (sub GR32, ESP)>;

// GR64_NOSP - GR64 registers except RSP (and RIP).
def GR64_NOSP : RegisterClass<"X86", [i64], 64, (sub GR64, RSP, RIP)>;

// GR32_NOREX_NOSP - GR32 registers which do not require a REX prefix except
// ESP.
def GR32_NOREX_NOSP : RegisterClass<"X86", [i32], 32,
                                    (and GR32_NOREX, GR32_NOSP)>;

// GR64_NOREX_NOSP - GR64_NOREX registers except RSP.
def GR64_NOREX_NOSP : RegisterClass<"X86", [i64], 64,
                                    (and GR64_NOREX, GR64_NOSP)>;

// A class to support the 'A' assembler constraint: EAX then EDX.
def GR32_AD : RegisterClass<"X86", [i32], 32, (add EAX, EDX)>;

// Scalar SSE2 floating point registers.
def FR32 : RegisterClass<"X86", [f32], 32, (sequence "XMM%u", 0, 15)>;

def FR64 : RegisterClass<"X86", [f64], 64, (add FR32)>;


// FIXME: This sets up the floating point register files as though they are f64
// values, though they really are f80 values.  This will cause us to spill
// values as 64-bit quantities instead of 80-bit quantities, which is much much
// faster on common hardware.  In reality, this should be controlled by a
// command line option or something.

def RFP32 : RegisterClass<"X86",[f32], 32, (sequence "FP%u", 0, 6)>;
def RFP64 : RegisterClass<"X86",[f64], 32, (add RFP32)>;
def RFP80 : RegisterClass<"X86",[f80], 32, (add RFP32)>;

// Floating point stack registers (these are not allocatable by the
// register allocator - the floating point stackifier is responsible
// for transforming FPn allocations to STn registers)
def RST : RegisterClass<"X86", [f80, f64, f32], 32, (sequence "ST%u", 0, 7)> {
  let isAllocatable = 0;
}

// Generic vector registers: VR64 and VR128.
def VR64: RegisterClass<"X86", [x86mmx], 64, (sequence "MM%u", 0, 7)>;
def VR128 : RegisterClass<"X86", [v16i8, v8i16, v4i32, v2i64, v4f32, v2f64],
                          128, (add FR32)>;
def VR256 : RegisterClass<"X86", [v32i8, v16i16, v8i32, v4i64, v8f32, v4f64],
                          256, (sequence "YMM%u", 0, 15)>;

// Status flags registers.
def CCR : RegisterClass<"X86", [i32], 32, (add EFLAGS)> {
  let CopyCost = -1;  // Don't allow copying of status registers.
  let isAllocatable = 0;
}
def FPCCR : RegisterClass<"X86", [i16], 16, (add FPSW)> {
  let CopyCost = -1;  // Don't allow copying of status registers.
  let isAllocatable = 0;
}
