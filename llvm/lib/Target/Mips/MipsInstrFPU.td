//===- MipsInstrFPU.td - Mips FPU Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
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
def SDT_MipsFPBrcond : SDTypeProfile<0, 3, [SDTCisSameAs<0, 2>, SDTCisInt<0>,
                                     SDTCisVT<1, OtherVT>]>;
def SDT_MipsFPCmp : SDTypeProfile<0, 3, [SDTCisSameAs<0, 1>, SDTCisFP<0>, 
                                  SDTCisInt<2>]>;
def SDT_MipsFPSelectCC : SDTypeProfile<1, 4, [SDTCisInt<1>, SDTCisInt<4>,
                                  SDTCisSameAs<0, 2>, SDTCisSameAs<2, 3>]>;
def MipsFPBrcond : SDNode<"MipsISD::FPBrcond", SDT_MipsFPBrcond, 
                          [SDNPHasChain]>; 
def MipsFPCmp : SDNode<"MipsISD::FPCmp", SDT_MipsFPCmp>;
def MipsFPSelectCC : SDNode<"MipsISD::FPSelectCC", SDT_MipsFPSelectCC>;

// Operand for printing out a condition code.
let PrintMethod = "printFCCOperand" in
  def condcode : Operand<i32>;

//===----------------------------------------------------------------------===//
// Feature predicates.
//===----------------------------------------------------------------------===//

def In32BitMode      : Predicate<"!Subtarget.isFP64bit()">;
def In64BitMode      : Predicate<"Subtarget.isFP64bit()">;
def IsSingleFloat    : Predicate<"Subtarget.isSingleFloat()">;
def IsNotSingleFloat : Predicate<"!Subtarget.isSingleFloat()">;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//
// A set of multiclasses is used to address this in one shot. 
// SO32 - single precision only, uses all 32 32-bit fp registers
//        require FGR32 Register Class and IsSingleFloat
// AS32 - 16 even fp registers are used for single precision
//        require AFGR32 Register Class and In32BitMode
// S64  - 32 64 bit registers are used to hold 32-bit single precision values.
//        require FGR64 Register Class and In64BitMode
// D32  - 16 even fp registers are used for double precision
//        require AFGR64 Register Class and In32BitMode
// D64  - 32 64 bit registers are used to hold 64-bit double precision values.
//        require FGR64 Register Class and In64BitMode
//
// Only SO32, AS32 and D32 are supported right now.
//
//===----------------------------------------------------------------------===//

multiclass FFR1_1<bits<6> funct, string asmstr> 
{
  def _SO32 : FFR<0x11, funct, 0x0, (outs FGR32:$fd), (ins FGR32:$fs),
      !strconcat(asmstr, ".s $fd, $fs"), []>, Requires<[IsSingleFloat]>;

  def _AS32 : FFR<0x11, funct, 0x0, (outs AFGR32:$fd), (ins AFGR32:$fs),
      !strconcat(asmstr, ".s $fd, $fs"), []>, Requires<[In32BitMode]>;

  def _D32  : FFR<0x11, funct, 0x1, (outs AFGR64:$fd), (ins AFGR64:$fs),
      !strconcat(asmstr, ".d $fd, $fs"), []>, Requires<[In32BitMode]>;
}

multiclass FFR1_2<bits<6> funct, string asmstr, SDNode FOp> 
{
  def _SO32 : FFR<0x11, funct, 0x0, (outs FGR32:$fd), (ins FGR32:$fs),
                 !strconcat(asmstr, ".s $fd, $fs"), 
                 [(set FGR32:$fd, (FOp FGR32:$fs))]>, Requires<[IsSingleFloat]>;

  def _AS32 : FFR<0x11, funct, 0x0, (outs AFGR32:$fd), (ins AFGR32:$fs),
                 !strconcat(asmstr, ".s $fd, $fs"), 
                 [(set AFGR32:$fd, (FOp AFGR32:$fs))]>, Requires<[In32BitMode]>;

  def _D32  : FFR<0x11, funct, 0x1, (outs AFGR64:$fd), (ins AFGR64:$fs),
                 !strconcat(asmstr, ".d $fd, $fs"), 
                 [(set AFGR64:$fd, (FOp AFGR64:$fs))]>, Requires<[In32BitMode]>;
}

class FFR1_3<bits<6> funct, bits<5> fmt, RegisterClass RcSrc, 
              RegisterClass RcDst, string asmstr>: 
  FFR<0x11, funct, fmt, (outs RcSrc:$fd), (ins RcDst:$fs), 
      !strconcat(asmstr, " $fd, $fs"), []>; 


multiclass FFR1_4<bits<6> funct, string asmstr, SDNode FOp> {
  def _SO32 : FFR<0x11, funct, 0x0, (outs FGR32:$fd), 
                 (ins FGR32:$fs, FGR32:$ft), 
                 !strconcat(asmstr, ".s $fd, $fs, $ft"),
                 [(set FGR32:$fd, (FOp FGR32:$fs, FGR32:$ft))]>,
                 Requires<[IsSingleFloat]>;

  def _AS32 : FFR<0x11, funct, 0x0, (outs AFGR32:$fd), 
                 (ins AFGR32:$fs, AFGR32:$ft), 
                 !strconcat(asmstr, ".s $fd, $fs, $ft"),
                 [(set AFGR32:$fd, (FOp AFGR32:$fs, AFGR32:$ft))]>,
                 Requires<[In32BitMode]>;

  def _D32  : FFR<0x11, funct, 0x1, (outs AFGR64:$fd), 
                 (ins AFGR64:$fs, AFGR64:$ft), 
                 !strconcat(asmstr, ".d $fd, $fs, $ft"),
                 [(set AFGR64:$fd, (FOp AFGR64:$fs, AFGR64:$ft))]>,
                 Requires<[In32BitMode]>;
}

//===----------------------------------------------------------------------===//
// Floating Point Instructions
//===----------------------------------------------------------------------===//

let ft = 0 in {
  defm FLOOR_W : FFR1_1<0b001111, "floor.w">;
  defm CEIL_W  : FFR1_1<0b001110, "ceil.w">;
  defm ROUND_W : FFR1_1<0b001100, "round.w">;
  defm TRUNC_W : FFR1_1<0b001101, "trunc.w">;
  defm CVTW    : FFR1_1<0b100100, "cvt.w">;
  defm FMOV    : FFR1_1<0b000110, "mov">;

  defm FABS    : FFR1_2<0b000101, "abs",  fabs>; 
  defm FNEG    : FFR1_2<0b000111, "neg",  fneg>; 
  defm FSQRT   : FFR1_2<0b000100, "sqrt", fsqrt>;

  let Predicates = [IsNotSingleFloat] in {
    /// Ceil to long signed integer
    def CEIL_LS   : FFR1_3<0b001010, 0x0, AFGR32, AFGR32, "ceil.l">;
    def CEIL_LD   : FFR1_3<0b001010, 0x1, AFGR64, AFGR64, "ceil.l">;

    /// Round to long signed integer
    def ROUND_LS  : FFR1_3<0b001000, 0x0, AFGR32, AFGR32, "round.l">;
    def ROUND_LD  : FFR1_3<0b001000, 0x1, AFGR64, AFGR64, "round.l">;

    /// Floor to long signed integer
    def FLOOR_LS  : FFR1_3<0b001011, 0x0, AFGR32, AFGR32, "floor.l">;
    def FLOOR_LD  : FFR1_3<0b001011, 0x1, AFGR64, AFGR64, "floor.l">;

    /// Trunc to long signed integer
    def TRUNC_LS  : FFR1_3<0b001001, 0x0, AFGR32, AFGR32, "trunc.l">;
    def TRUNC_LD  : FFR1_3<0b001001, 0x1, AFGR64, AFGR64, "trunc.l">;

    /// Convert to long signed integer
    def CVTL_S    : FFR1_3<0b100101, 0x0, AFGR32, AFGR32, "cvt.l">; 
    def CVTL_D    : FFR1_3<0b100101, 0x1, AFGR64, AFGR64, "cvt.l">; 

    /// Convert to Double Precison 
    def CVTD_S32 : FFR1_3<0b100001, 0x0, AFGR64, FGR32, "cvt.d.s">; 
    def CVTD_W32 : FFR1_3<0b100001, 0x2, AFGR64, FGR32, "cvt.d.w">; 
    def CVTD_L32 : FFR1_3<0b100001, 0x3, AFGR64, AFGR64, "cvt.d.l">; 
                   
    /// Convert to Single Precison
    def CVTS_D32 : FFR1_3<0b100000, 0x1, FGR32, AFGR64, "cvt.s.d">;
    def CVTS_L32 : FFR1_3<0b100000, 0x3, FGR32, AFGR64, "cvt.s.l">; 
  }

  /// Convert to Single Precison
  def CVTS_W32 : FFR1_3<0b100000, 0x2, FGR32,  FGR32,  "cvt.s.w">, 
                 Requires<[IsSingleFloat]>;
}

// The odd-numbered registers are only referenced when doing loads,
// stores, and moves between floating-point and integer registers.
// When defining instructions, we reference all 32-bit registers, 
// regardless of register aliasing.
let fd = 0 in {
  /// Move Control Registers From/To CPU Registers
  ///def CFC1  : FFR<0x11, 0x0, 0x2, (outs CPURegs:$rt), (ins FGR32:$fs),
  ///                "cfc1 $rt, $fs", []>;

  ///def CTC1  : FFR<0x11, 0x0, 0x6, (outs CPURegs:$rt), (ins FGR32:$fs),
  ///                "ctc1 $rt, $fs", []>;
  ///                
  ///def CFC1A : FFR<0x11, 0x0, 0x2, (outs CPURegs:$rt), (ins AFGR32:$fs),
  ///                "cfc1 $rt, $fs", []>;

  ///def CTC1A : FFR<0x11, 0x0, 0x6, (outs CPURegs:$rt), (ins AFGR32:$fs),
  ///                "ctc1 $rt, $fs", []>;

  def MFC1  : FFR<0x11, 0x00, 0x00, (outs CPURegs:$rt), (ins FGR32:$fs),
                  "mfc1 $rt, $fs", []>;

  def MTC1  : FFR<0x11, 0x00, 0x04, (outs FGR32:$fs), (ins CPURegs:$rt),
                  "mtc1 $rt, $fs", []>;

  def MFC1A : FFR<0x11, 0x00, 0x00, (outs CPURegs:$rt), (ins AFGR32:$fs),
                  "mfc1 $rt, $fs", []>;

  def MTC1A : FFR<0x11, 0x00, 0x04, (outs AFGR32:$fs), (ins CPURegs:$rt),
                  "mtc1 $rt, $fs", []>;
}

/// Floating Point Memory Instructions
let Predicates = [IsNotSingleFloat] in {
  def LDC1 : FFI<0b110101, (outs AFGR64:$ft), (ins mem:$addr), 
                 "ldc1 $ft, $addr", [(set AFGR64:$ft, (load addr:$addr))]>;

  def SDC1 : FFI<0b111101, (outs), (ins AFGR64:$ft, mem:$addr), 
                 "sdc1 $ft, $addr", [(store AFGR64:$ft, addr:$addr)]>;
}

// LWC1 and SWC1 can always be emited with odd registers.
def LWC1  : FFI<0b110001, (outs FGR32:$ft), (ins mem:$addr), "lwc1 $ft, $addr",
               [(set FGR32:$ft, (load addr:$addr))]>; 
def SWC1  : FFI<0b111001, (outs), (ins FGR32:$ft, mem:$addr), "swc1 $ft, $addr",
               [(store FGR32:$ft, addr:$addr)]>; 

def LWC1A : FFI<0b110001, (outs AFGR32:$ft), (ins mem:$addr), "lwc1 $ft, $addr",
               [(set AFGR32:$ft, (load addr:$addr))]>; 
def SWC1A : FFI<0b111001, (outs), (ins AFGR32:$ft, mem:$addr), 
                "swc1 $ft, $addr", [(store AFGR32:$ft, addr:$addr)]>; 

/// Floating-point Aritmetic
defm FADD : FFR1_4<0x10, "add", fadd>;
defm FDIV : FFR1_4<0x03, "div", fdiv>;
defm FMUL : FFR1_4<0x02, "mul", fmul>;
defm FSUB : FFR1_4<0x01, "sub", fsub>;

//===----------------------------------------------------------------------===//
// Floating Point Branch Codes
//===----------------------------------------------------------------------===//
// Mips branch codes. These correspond to condcode in MipsInstrInfo.h. 
// They must be kept in synch.
def MIPS_BRANCH_F  : PatLeaf<(i32 0)>;
def MIPS_BRANCH_T  : PatLeaf<(i32 1)>;
def MIPS_BRANCH_FL : PatLeaf<(i32 2)>;
def MIPS_BRANCH_TL : PatLeaf<(i32 3)>;

/// Floating Point Branch of False/True (Likely)
let isBranch=1, isTerminator=1, hasDelaySlot=1, base=0x8, Uses=[FCR31] in {
  class FBRANCH<PatLeaf op, string asmstr> : FFI<0x11, (outs), 
        (ins brtarget:$dst), !strconcat(asmstr, " $dst"),
        [(MipsFPBrcond op, bb:$dst, FCR31)]>;
}
def BC1F  : FBRANCH<MIPS_BRANCH_F,  "bc1f">;
def BC1T  : FBRANCH<MIPS_BRANCH_T,  "bc1t">;
def BC1FL : FBRANCH<MIPS_BRANCH_FL, "bc1fl">;
def BC1TL : FBRANCH<MIPS_BRANCH_TL, "bc1tl">;

//===----------------------------------------------------------------------===//
// Floating Point Flag Conditions
//===----------------------------------------------------------------------===//
// Mips condition codes. They must correspond to condcode in MipsInstrInfo.h. 
// They must be kept in synch.
def MIPS_FCOND_F    : PatLeaf<(i32 0)>;
def MIPS_FCOND_UN   : PatLeaf<(i32 1)>;
def MIPS_FCOND_EQ   : PatLeaf<(i32 2)>;
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

/// Floating Point Compare
let hasDelaySlot = 1, Defs=[FCR31] in {
  def FCMP_SO32 : FCC<0x0, (outs), (ins FGR32:$fs, FGR32:$ft, condcode:$cc),
      "c.$cc.s $fs, $ft", [(MipsFPCmp FGR32:$fs, FGR32:$ft, imm:$cc), 
      (implicit FCR31)]>, Requires<[IsSingleFloat]>;
  
  def FCMP_AS32 : FCC<0x0, (outs), (ins AFGR32:$fs, AFGR32:$ft, condcode:$cc),
      "c.$cc.s $fs, $ft", [(MipsFPCmp AFGR32:$fs, AFGR32:$ft, imm:$cc), 
      (implicit FCR31)]>, Requires<[In32BitMode]>;

  def FCMP_D32 : FCC<0x1, (outs), (ins AFGR64:$fs, AFGR64:$ft, condcode:$cc),
      "c.$cc.d $fs, $ft", [(MipsFPCmp AFGR64:$fs, AFGR64:$ft, imm:$cc),
      (implicit FCR31)]>, Requires<[In32BitMode]>;
}

//===----------------------------------------------------------------------===//
// Floating Point Pseudo-Instructions
//===----------------------------------------------------------------------===//

// For some explanation, see Select_CC at MipsInstrInfo.td. We also embedd a 
// condiciton code to enable easy handling by the Custom Inserter.
let usesCustomDAGSchedInserter = 1, Uses=[FCR31] in {
  class PseudoFPSelCC<RegisterClass RC, string asmstr> : 
    MipsPseudo<(outs RC:$dst), 
               (ins CPURegs:$CmpRes, RC:$T, RC:$F, condcode:$cc), asmstr, 
               [(set RC:$dst, (MipsFPSelectCC CPURegs:$CmpRes, RC:$T, RC:$F,
                 imm:$cc))]>;
}

// The values to be selected are fp but the condition test is with integers.
def Select_CC_SO32  : PseudoSelCC<FGR32, "# MipsSelect_CC_SO32_f32">,
                      Requires<[IsSingleFloat]>;
def Select_CC_AS32  : PseudoSelCC<AFGR32, "# MipsSelect_CC_AS32_f32">,
                      Requires<[In32BitMode]>;
def Select_CC_D32   : PseudoSelCC<AFGR64, "# MipsSelect_CC_D32_f32">,
                      Requires<[In32BitMode]>;

// The values to be selected are int but the condition test is done with fp.
def Select_FCC      : PseudoFPSelCC<CPURegs, "# MipsSelect_FCC">;

// The values to be selected and the condition test is done with fp.
def Select_FCC_SO32 : PseudoFPSelCC<FGR32, "# MipsSelect_FCC_SO32_f32">, 
                      Requires<[IsSingleFloat]>;
def Select_FCC_AS32 : PseudoFPSelCC<AFGR32, "# MipsSelect_FCC_AS32_f32">, 
                      Requires<[In32BitMode]>;
def Select_FCC_D32  : PseudoFPSelCC<AFGR64, "# MipsSelect_FCC_D32_f32">, 
                      Requires<[In32BitMode]>;


//===----------------------------------------------------------------------===//
// Floating Point Patterns
//===----------------------------------------------------------------------===//
def fpimm0 : PatLeaf<(fpimm), [{
  return N->isExactlyValue(+0.0);
}]>;

def : Pat<(f32 fpimm0), (MTC1 ZERO)>, Requires<[IsSingleFloat]>;
def : Pat<(f32 fpimm0), (MTC1A ZERO)>, Requires<[In32BitMode]>;

def : Pat<(f32 (sint_to_fp CPURegs:$src)), (CVTS_W32 (MTC1 CPURegs:$src))>;
def : Pat<(f64 (sint_to_fp CPURegs:$src)), (CVTD_W32 (MTC1 CPURegs:$src))>;

def : Pat<(i32 (fp_to_sint FGR32:$src)), (MFC1 (TRUNC_W_SO32 FGR32:$src))>;
def : Pat<(i32 (fp_to_sint AFGR32:$src)), (MFC1A (TRUNC_W_AS32 AFGR32:$src))>;

def : Pat<(i32 (bitconvert FGR32:$src)),  (MFC1 FGR32:$src)>;
def : Pat<(i32 (bitconvert AFGR32:$src)), (MFC1A AFGR32:$src)>;

def : Pat<(f32 (bitconvert CPURegs:$src)), (MTC1 CPURegs:$src)>,
      Requires<[IsSingleFloat]>;
def : Pat<(f32 (bitconvert CPURegs:$src)), (MTC1A CPURegs:$src)>,
      Requires<[In32BitMode]>;
