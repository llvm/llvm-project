//===-- MipsInstrFPU.td - Mips FPU Instruction Information -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Mips FPU instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Floating Point Instructions
// ------------------------
// * 64bit fp:
//    - 32 64-bit registers (default mode)
//    - 16 even 32-bit registers (32-bit compatible mode) for
//      single and double access.
// * 32bit fp:
//    - 16 even 32-bit registers - single and double (aliased)
//    - 32 32-bit registers (within single-only mode)
//===----------------------------------------------------------------------===//

// Floating Point Compare and Branch
def SDT_MipsFPBrcond : SDTypeProfile<0, 2, [SDTCisInt<0>,
                                            SDTCisVT<1, OtherVT>]>;
def SDT_MipsFPCmp : SDTypeProfile<0, 3, [SDTCisSameAs<0, 1>, SDTCisFP<1>,
                                         SDTCisVT<2, i32>]>;
def SDT_MipsCMovFP : SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>,
                                          SDTCisSameAs<1, 2>]>;
def SDT_MipsBuildPairF64 : SDTypeProfile<1, 2, [SDTCisVT<0, f64>,
                                                SDTCisVT<1, i32>,
                                                SDTCisSameAs<1, 2>]>;
def SDT_MipsExtractElementF64 : SDTypeProfile<1, 2, [SDTCisVT<0, i32>,
                                                     SDTCisVT<1, f64>,
                                                     SDTCisVT<2, i32>]>;

def MipsFPCmp : SDNode<"MipsISD::FPCmp", SDT_MipsFPCmp, [SDNPOutGlue]>;
def MipsCMovFP_T : SDNode<"MipsISD::CMovFP_T", SDT_MipsCMovFP, [SDNPInGlue]>;
def MipsCMovFP_F : SDNode<"MipsISD::CMovFP_F", SDT_MipsCMovFP, [SDNPInGlue]>;
def MipsFPBrcond : SDNode<"MipsISD::FPBrcond", SDT_MipsFPBrcond,
                          [SDNPHasChain, SDNPOptInGlue]>;
def MipsBuildPairF64 : SDNode<"MipsISD::BuildPairF64", SDT_MipsBuildPairF64>;
def MipsExtractElementF64 : SDNode<"MipsISD::ExtractElementF64",
                                   SDT_MipsExtractElementF64>;

// Operand for printing out a condition code.
let PrintMethod = "printFCCOperand", DecoderMethod = "DecodeCondCode" in
  def condcode : Operand<i32>;

//===----------------------------------------------------------------------===//
// Feature predicates.
//===----------------------------------------------------------------------===//

def IsFP64bit        : Predicate<"Subtarget.isFP64bit()">,
                       AssemblerPredicate<"FeatureFP64Bit">;
def NotFP64bit       : Predicate<"!Subtarget.isFP64bit()">,
                       AssemblerPredicate<"!FeatureFP64Bit">;
def IsSingleFloat    : Predicate<"Subtarget.isSingleFloat()">,
                       AssemblerPredicate<"FeatureSingleFloat">;
def IsNotSingleFloat : Predicate<"!Subtarget.isSingleFloat()">,
                       AssemblerPredicate<"!FeatureSingleFloat">;

// FP immediate patterns.
def fpimm0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def fpimm0neg : PatLeaf<(fpimm), [{
  return N->isExactlyValue(-0.0);
}]>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//
// A set of multiclasses is used to address the register usage.
//
// S32 - single precision in 16 32bit even fp registers
//       single precision in 32 32bit fp registers in SingleOnly mode
// S64 - single precision in 32 64bit fp registers (In64BitMode)
// D32 - double precision in 16 32bit even fp registers
// D64 - double precision in 32 64bit fp registers (In64BitMode)
//
// Only S32 and D32 are supported right now.
//===----------------------------------------------------------------------===//

// FP load.
let DecoderMethod = "DecodeFMem" in {
class FPLoad<bits<6> op, string opstr, RegisterClass RC, Operand MemOpnd>:
  FMem<op, (outs RC:$ft), (ins MemOpnd:$addr),
      !strconcat(opstr, "\t$ft, $addr"), [(set RC:$ft, (load addr:$addr))],
      IILoad>;

// FP store.
class FPStore<bits<6> op, string opstr, RegisterClass RC, Operand MemOpnd>:
  FMem<op, (outs), (ins RC:$ft, MemOpnd:$addr),
      !strconcat(opstr, "\t$ft, $addr"), [(store RC:$ft, addr:$addr)],
      IIStore>;
}
// FP indexed load.
class FPIdxLoad<bits<6> funct, string opstr, RegisterClass DRC,
                RegisterClass PRC, SDPatternOperator FOp = null_frag>:
  FFMemIdx<funct, (outs DRC:$fd), (ins PRC:$base, PRC:$index),
           !strconcat(opstr, "\t$fd, ${index}(${base})"),
           [(set DRC:$fd, (FOp (add PRC:$base, PRC:$index)))]> {
  let fs = 0;
}

// FP indexed store.
class FPIdxStore<bits<6> funct, string opstr, RegisterClass DRC,
                 RegisterClass PRC, SDPatternOperator FOp= null_frag>:
  FFMemIdx<funct, (outs), (ins DRC:$fs, PRC:$base, PRC:$index),
           !strconcat(opstr, "\t$fs, ${index}(${base})"),
           [(FOp DRC:$fs, (add PRC:$base, PRC:$index))]> {
  let fd = 0;
}

// Instructions that convert an FP value to 32-bit fixed point.
multiclass FFR1_W_M<bits<6> funct, string opstr> {
  def _S   : FFR1<funct, 16, opstr, "w.s", FGR32, FGR32>;
  def _D32 : FFR1<funct, 17, opstr, "w.d", FGR32, AFGR64>,
             Requires<[NotFP64bit, HasStandardEncoding]>;
  def _D64 : FFR1<funct, 17, opstr, "w.d", FGR32, FGR64>,
             Requires<[IsFP64bit, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
  }
}

// Instructions that convert an FP value to 64-bit fixed point.
let Predicates = [IsFP64bit, HasStandardEncoding], DecoderNamespace = "Mips64" in
multiclass FFR1_L_M<bits<6> funct, string opstr> {
  def _S   : FFR1<funct, 16, opstr, "l.s", FGR64, FGR32>;
  def _D64 : FFR1<funct, 17, opstr, "l.d", FGR64, FGR64>;
}

// FP-to-FP conversion instructions.
multiclass FFR1P_M<bits<6> funct, string opstr, SDNode OpNode> {
  def _S   : FFR1P<funct, 16, opstr, "s", FGR32, FGR32, OpNode>;
  def _D32 : FFR1P<funct, 17, opstr, "d", AFGR64, AFGR64, OpNode>,
             Requires<[NotFP64bit, HasStandardEncoding]>;
  def _D64 : FFR1P<funct, 17, opstr, "d", FGR64, FGR64, OpNode>,
             Requires<[IsFP64bit, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
  }
}

multiclass FFR2P_M<bits<6> funct, string opstr, SDNode OpNode, bit isComm = 0> {
  let isCommutable = isComm in {
  def _S   : FFR2P<funct, 16, opstr, "s", FGR32, OpNode>;
  def _D32 : FFR2P<funct, 17, opstr, "d", AFGR64, OpNode>,
             Requires<[NotFP64bit, HasStandardEncoding]>;
  def _D64 : FFR2P<funct, 17, opstr, "d", FGR64, OpNode>,
             Requires<[IsFP64bit, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
  }
}
}

// FP madd/msub/nmadd/nmsub instruction classes.
class FMADDSUB<bits<3> funct, bits<3> fmt, string opstr, string fmtstr,
               SDNode OpNode, RegisterClass RC> :
  FFMADDSUB<funct, fmt, (outs RC:$fd), (ins RC:$fr, RC:$fs, RC:$ft),
            !strconcat(opstr, ".", fmtstr, "\t$fd, $fr, $fs, $ft"),
            [(set RC:$fd, (OpNode (fmul RC:$fs, RC:$ft), RC:$fr))]>;

class FNMADDSUB<bits<3> funct, bits<3> fmt, string opstr, string fmtstr,
                SDNode OpNode, RegisterClass RC> :
  FFMADDSUB<funct, fmt, (outs RC:$fd), (ins RC:$fr, RC:$fs, RC:$ft),
            !strconcat(opstr, ".", fmtstr, "\t$fd, $fr, $fs, $ft"),
            [(set RC:$fd, (fsub fpimm0, (OpNode (fmul RC:$fs, RC:$ft), RC:$fr)))]>;

//===----------------------------------------------------------------------===//
// Floating Point Instructions
//===----------------------------------------------------------------------===//
defm ROUND_W : FFR1_W_M<0xc, "round">;
defm ROUND_L : FFR1_L_M<0x8, "round">;
defm TRUNC_W : FFR1_W_M<0xd, "trunc">;
defm TRUNC_L : FFR1_L_M<0x9, "trunc">;
defm CEIL_W  : FFR1_W_M<0xe, "ceil">;
defm CEIL_L  : FFR1_L_M<0xa, "ceil">;
defm FLOOR_W : FFR1_W_M<0xf, "floor">;
defm FLOOR_L : FFR1_L_M<0xb, "floor">;
defm CVT_W   : FFR1_W_M<0x24, "cvt">, NeverHasSideEffects;
//defm CVT_L   : FFR1_L_M<0x25, "cvt">;

def CVT_S_W : FFR1<0x20, 20, "cvt", "s.w", FGR32, FGR32>, NeverHasSideEffects;
def CVT_L_S : FFR1<0x25, 16, "cvt", "l.s", FGR64, FGR32>, NeverHasSideEffects;
def CVT_L_D64: FFR1<0x25, 17, "cvt", "l.d", FGR64, FGR64>, NeverHasSideEffects;

let Predicates = [NotFP64bit, HasStandardEncoding], neverHasSideEffects = 1 in {
  def CVT_S_D32 : FFR1<0x20, 17, "cvt", "s.d", FGR32, AFGR64>;
  def CVT_D32_W : FFR1<0x21, 20, "cvt", "d.w", AFGR64, FGR32>;
  def CVT_D32_S : FFR1<0x21, 16, "cvt", "d.s", AFGR64, FGR32>;
}

let Predicates = [IsFP64bit, HasStandardEncoding], DecoderNamespace = "Mips64",
    neverHasSideEffects = 1 in {
 def CVT_S_D64 : FFR1<0x20, 17, "cvt", "s.d", FGR32, FGR64>;
 def CVT_S_L   : FFR1<0x20, 21, "cvt", "s.l", FGR32, FGR64>;
 def CVT_D64_W : FFR1<0x21, 20, "cvt", "d.w", FGR64, FGR32>;
 def CVT_D64_S : FFR1<0x21, 16, "cvt", "d.s", FGR64, FGR32>;
 def CVT_D64_L : FFR1<0x21, 21, "cvt", "d.l", FGR64, FGR64>;
}

let Predicates = [NoNaNsFPMath, HasStandardEncoding] in {
  defm FABS    : FFR1P_M<0x5, "abs",  fabs>;
  defm FNEG    : FFR1P_M<0x7, "neg",  fneg>;
}
defm FSQRT   : FFR1P_M<0x4, "sqrt", fsqrt>;

// The odd-numbered registers are only referenced when doing loads,
// stores, and moves between floating-point and integer registers.
// When defining instructions, we reference all 32-bit registers,
// regardless of register aliasing.

class FFRGPR<bits<5> _fmt, dag outs, dag ins, string asmstr, list<dag> pattern>:
             FFR<0x11, 0x0, _fmt, outs, ins, asmstr, pattern> {
  bits<5> rt;
  let ft = rt;
  let fd = 0;
}

/// Move Control Registers From/To CPU Registers
def CFC1  : FFRGPR<0x2, (outs CPURegs:$rt), (ins CCR:$fs),
                  "cfc1\t$rt, $fs", []>;

def CTC1  : FFRGPR<0x6, (outs CCR:$fs), (ins CPURegs:$rt),
                  "ctc1\t$rt, $fs", []>;

def MFC1  : FFRGPR<0x00, (outs CPURegs:$rt), (ins FGR32:$fs),
                  "mfc1\t$rt, $fs",
                  [(set CPURegs:$rt, (bitconvert FGR32:$fs))]>;

def MTC1  : FFRGPR<0x04, (outs FGR32:$fs), (ins CPURegs:$rt),
                  "mtc1\t$rt, $fs",
                  [(set FGR32:$fs, (bitconvert CPURegs:$rt))]>;

def DMFC1 : FFRGPR<0x01, (outs CPU64Regs:$rt), (ins FGR64:$fs),
                  "dmfc1\t$rt, $fs",
                  [(set CPU64Regs:$rt, (bitconvert FGR64:$fs))]>;

def DMTC1 : FFRGPR<0x05, (outs FGR64:$fs), (ins CPU64Regs:$rt),
                  "dmtc1\t$rt, $fs",
                  [(set FGR64:$fs, (bitconvert CPU64Regs:$rt))]>;

def FMOV_S   : FFR1<0x6, 16, "mov", "s", FGR32, FGR32>;
def FMOV_D32 : FFR1<0x6, 17, "mov", "d", AFGR64, AFGR64>,
               Requires<[NotFP64bit, HasStandardEncoding]>;
def FMOV_D64 : FFR1<0x6, 17, "mov", "d", FGR64, FGR64>,
               Requires<[IsFP64bit, HasStandardEncoding]> {
  let DecoderNamespace = "Mips64";
}

/// Floating Point Memory Instructions
let Predicates = [IsN64, HasStandardEncoding], DecoderNamespace = "Mips64" in {
  def LWC1_P8   : FPLoad<0x31, "lwc1", FGR32, mem64>;
  def SWC1_P8   : FPStore<0x39, "swc1", FGR32, mem64>;
  def LDC164_P8 : FPLoad<0x35, "ldc1", FGR64, mem64> {
    let isCodeGenOnly =1;
  }
  def SDC164_P8 : FPStore<0x3d, "sdc1", FGR64, mem64> {
    let isCodeGenOnly =1;
  }
}

let Predicates = [NotN64, HasStandardEncoding] in {
  def LWC1   : FPLoad<0x31, "lwc1", FGR32, mem>;
  def SWC1   : FPStore<0x39, "swc1", FGR32, mem>;
}

let Predicates = [NotN64, HasMips64, HasStandardEncoding],
  DecoderNamespace = "Mips64" in {
  def LDC164 : FPLoad<0x35, "ldc1", FGR64, mem>;
  def SDC164 : FPStore<0x3d, "sdc1", FGR64, mem>;
}

let Predicates = [NotN64, NotMips64, HasStandardEncoding] in {
  def LDC1   : FPLoad<0x35, "ldc1", AFGR64, mem>;
  def SDC1   : FPStore<0x3d, "sdc1", AFGR64, mem>;
}

// Indexed loads and stores.
let Predicates = [HasMips32r2Or64, HasStandardEncoding] in {
  def LWXC1 : FPIdxLoad<0x0, "lwxc1", FGR32, CPURegs, load>;
  def SWXC1 : FPIdxStore<0x8, "swxc1", FGR32, CPURegs, store>;
}

let Predicates = [HasMips32r2, NotMips64, HasStandardEncoding] in {
  def LDXC1 : FPIdxLoad<0x1, "ldxc1", AFGR64, CPURegs, load>;
  def SDXC1 : FPIdxStore<0x9, "sdxc1", AFGR64, CPURegs, store>;
}

let Predicates = [HasMips64, NotN64, HasStandardEncoding], DecoderNamespace="Mips64" in {
  def LDXC164 : FPIdxLoad<0x1, "ldxc1", FGR64, CPURegs, load>;
  def SDXC164 : FPIdxStore<0x9, "sdxc1", FGR64, CPURegs, store>;
}

// n64
let Predicates = [IsN64, HasStandardEncoding], isCodeGenOnly=1 in {
  def LWXC1_P8   : FPIdxLoad<0x0, "lwxc1", FGR32, CPU64Regs, load>;
  def LDXC164_P8 : FPIdxLoad<0x1, "ldxc1", FGR64, CPU64Regs, load>;
  def SWXC1_P8   : FPIdxStore<0x8, "swxc1", FGR32, CPU64Regs, store>;
  def SDXC164_P8 : FPIdxStore<0x9, "sdxc1", FGR64, CPU64Regs, store>;
}

// Load/store doubleword indexed unaligned.
let Predicates = [NotMips64, HasStandardEncoding] in {
  def LUXC1 : FPIdxLoad<0x5, "luxc1", AFGR64, CPURegs>;
  def SUXC1 : FPIdxStore<0xd, "suxc1", AFGR64, CPURegs>;
}

let Predicates = [HasMips64, HasStandardEncoding],
  DecoderNamespace="Mips64" in {
  def LUXC164 : FPIdxLoad<0x5, "luxc1", FGR64, CPURegs>;
  def SUXC164 : FPIdxStore<0xd, "suxc1", FGR64, CPURegs>;
}

/// Floating-point Aritmetic
defm FADD : FFR2P_M<0x00, "add", fadd, 1>;
defm FDIV : FFR2P_M<0x03, "div", fdiv>;
defm FMUL : FFR2P_M<0x02, "mul", fmul, 1>;
defm FSUB : FFR2P_M<0x01, "sub", fsub>;

let Predicates = [HasMips32r2, HasStandardEncoding] in {
  def MADD_S : FMADDSUB<0x4, 0, "madd", "s", fadd, FGR32>;
  def MSUB_S : FMADDSUB<0x5, 0, "msub", "s", fsub, FGR32>;
}

let Predicates = [HasMips32r2, NoNaNsFPMath, HasStandardEncoding] in {
  def NMADD_S : FNMADDSUB<0x6, 0, "nmadd", "s", fadd, FGR32>;
  def NMSUB_S : FNMADDSUB<0x7, 0, "nmsub", "s", fsub, FGR32>;
}

let Predicates = [HasMips32r2, NotFP64bit, HasStandardEncoding] in {
  def MADD_D32 : FMADDSUB<0x4, 1, "madd", "d", fadd, AFGR64>;
  def MSUB_D32 : FMADDSUB<0x5, 1, "msub", "d", fsub, AFGR64>;
}

let Predicates = [HasMips32r2, NotFP64bit, NoNaNsFPMath, HasStandardEncoding] in {
  def NMADD_D32 : FNMADDSUB<0x6, 1, "nmadd", "d", fadd, AFGR64>;
  def NMSUB_D32 : FNMADDSUB<0x7, 1, "nmsub", "d", fsub, AFGR64>;
}

let Predicates = [HasMips32r2, IsFP64bit, HasStandardEncoding], isCodeGenOnly=1 in {
  def MADD_D64 : FMADDSUB<0x4, 1, "madd", "d", fadd, FGR64>;
  def MSUB_D64 : FMADDSUB<0x5, 1, "msub", "d", fsub, FGR64>;
}

let Predicates = [HasMips32r2, IsFP64bit, NoNaNsFPMath, HasStandardEncoding],
    isCodeGenOnly=1 in {
  def NMADD_D64 : FNMADDSUB<0x6, 1, "nmadd", "d", fadd, FGR64>;
  def NMSUB_D64 : FNMADDSUB<0x7, 1, "nmsub", "d", fsub, FGR64>;
}

//===----------------------------------------------------------------------===//
// Floating Point Branch Codes
//===----------------------------------------------------------------------===//
// Mips branch codes. These correspond to condcode in MipsInstrInfo.h.
// They must be kept in synch.
def MIPS_BRANCH_F  : PatLeaf<(i32 0)>;
def MIPS_BRANCH_T  : PatLeaf<(i32 1)>;

/// Floating Point Branch of False/True (Likely)
let isBranch=1, isTerminator=1, hasDelaySlot=1, base=0x8, Uses=[FCR31] in
  class FBRANCH<bits<1> nd, bits<1> tf, PatLeaf op, string asmstr> :
      FFI<0x11, (outs), (ins brtarget:$dst), !strconcat(asmstr, "\t$dst"),
        [(MipsFPBrcond op, bb:$dst)]> {
  let Inst{20-18} = 0;
  let Inst{17} = nd;
  let Inst{16} = tf;
}

let DecoderMethod = "DecodeBC1" in {
def BC1F  : FBRANCH<0, 0, MIPS_BRANCH_F,  "bc1f">;
def BC1T  : FBRANCH<0, 1, MIPS_BRANCH_T,  "bc1t">;
}
//===----------------------------------------------------------------------===//
// Floating Point Flag Conditions
//===----------------------------------------------------------------------===//
// Mips condition codes. They must correspond to condcode in MipsInstrInfo.h.
// They must be kept in synch.
def MIPS_FCOND_F    : PatLeaf<(i32 0)>;
def MIPS_FCOND_UN   : PatLeaf<(i32 1)>;
def MIPS_FCOND_OEQ  : PatLeaf<(i32 2)>;
def MIPS_FCOND_UEQ  : PatLeaf<(i32 3)>;
def MIPS_FCOND_OLT  : PatLeaf<(i32 4)>;
def MIPS_FCOND_ULT  : PatLeaf<(i32 5)>;
def MIPS_FCOND_OLE  : PatLeaf<(i32 6)>;
def MIPS_FCOND_ULE  : PatLeaf<(i32 7)>;
def MIPS_FCOND_SF   : PatLeaf<(i32 8)>;
def MIPS_FCOND_NGLE : PatLeaf<(i32 9)>;
def MIPS_FCOND_SEQ  : PatLeaf<(i32 10)>;
def MIPS_FCOND_NGL  : PatLeaf<(i32 11)>;
def MIPS_FCOND_LT   : PatLeaf<(i32 12)>;
def MIPS_FCOND_NGE  : PatLeaf<(i32 13)>;
def MIPS_FCOND_LE   : PatLeaf<(i32 14)>;
def MIPS_FCOND_NGT  : PatLeaf<(i32 15)>;

class FCMP<bits<5> fmt, RegisterClass RC, string typestr> :
  FCC<fmt, (outs), (ins RC:$fs, RC:$ft, condcode:$cc),
      !strconcat("c.$cc.", typestr, "\t$fs, $ft"),
      [(MipsFPCmp RC:$fs, RC:$ft, imm:$cc)]>;

/// Floating Point Compare
let Defs=[FCR31] in {
  def FCMP_S32 : FCMP<0x10, FGR32, "s">;
  def FCMP_D32 : FCMP<0x11, AFGR64, "d">,
      Requires<[NotFP64bit, HasStandardEncoding]>;
  def FCMP_D64 : FCMP<0x11, FGR64, "d">,
      Requires<[IsFP64bit, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
  }
}

//===----------------------------------------------------------------------===//
// Floating Point Pseudo-Instructions
//===----------------------------------------------------------------------===//
def MOVCCRToCCR : PseudoSE<(outs CCR:$dst), (ins CCR:$src),
                           "# MOVCCRToCCR", []>;

// This pseudo instr gets expanded into 2 mtc1 instrs after register
// allocation.
def BuildPairF64 :
  PseudoSE<(outs AFGR64:$dst),
           (ins CPURegs:$lo, CPURegs:$hi), "",
           [(set AFGR64:$dst, (MipsBuildPairF64 CPURegs:$lo, CPURegs:$hi))]>;

// This pseudo instr gets expanded into 2 mfc1 instrs after register
// allocation.
// if n is 0, lower part of src is extracted.
// if n is 1, higher part of src is extracted.
def ExtractElementF64 :
  PseudoSE<(outs CPURegs:$dst), (ins AFGR64:$src, i32imm:$n), "",
           [(set CPURegs:$dst, (MipsExtractElementF64 AFGR64:$src, imm:$n))]>;

//===----------------------------------------------------------------------===//
// Floating Point Patterns
//===----------------------------------------------------------------------===//
def : MipsPat<(f32 fpimm0), (MTC1 ZERO)>;
def : MipsPat<(f32 fpimm0neg), (FNEG_S (MTC1 ZERO))>;

def : MipsPat<(f32 (sint_to_fp CPURegs:$src)), (CVT_S_W (MTC1 CPURegs:$src))>;
def : MipsPat<(i32 (fp_to_sint FGR32:$src)), (MFC1 (TRUNC_W_S FGR32:$src))>;

let Predicates = [NotFP64bit, HasStandardEncoding] in {
  def : MipsPat<(f64 (sint_to_fp CPURegs:$src)),
                (CVT_D32_W (MTC1 CPURegs:$src))>;
  def : MipsPat<(i32 (fp_to_sint AFGR64:$src)),
                (MFC1 (TRUNC_W_D32 AFGR64:$src))>;
  def : MipsPat<(f32 (fround AFGR64:$src)), (CVT_S_D32 AFGR64:$src)>;
  def : MipsPat<(f64 (fextend FGR32:$src)), (CVT_D32_S FGR32:$src)>;
}

let Predicates = [IsFP64bit, HasStandardEncoding] in {
  def : MipsPat<(f64 fpimm0), (DMTC1 ZERO_64)>;
  def : MipsPat<(f64 fpimm0neg), (FNEG_D64 (DMTC1 ZERO_64))>;

  def : MipsPat<(f64 (sint_to_fp CPURegs:$src)),
                (CVT_D64_W (MTC1 CPURegs:$src))>;
  def : MipsPat<(f32 (sint_to_fp CPU64Regs:$src)),
                (CVT_S_L (DMTC1 CPU64Regs:$src))>;
  def : MipsPat<(f64 (sint_to_fp CPU64Regs:$src)),
                (CVT_D64_L (DMTC1 CPU64Regs:$src))>;

  def : MipsPat<(i32 (fp_to_sint FGR64:$src)),
                (MFC1 (TRUNC_W_D64 FGR64:$src))>;
  def : MipsPat<(i64 (fp_to_sint FGR32:$src)), (DMFC1 (TRUNC_L_S FGR32:$src))>;
  def : MipsPat<(i64 (fp_to_sint FGR64:$src)),
                (DMFC1 (TRUNC_L_D64 FGR64:$src))>;

  def : MipsPat<(f32 (fround FGR64:$src)), (CVT_S_D64 FGR64:$src)>;
  def : MipsPat<(f64 (fextend FGR32:$src)), (CVT_D64_S FGR32:$src)>;
}
