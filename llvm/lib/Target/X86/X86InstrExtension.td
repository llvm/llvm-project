//===-- X86InstrExtension.td - Sign and Zero Extensions ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the sign and zero extension operations.
//
//===----------------------------------------------------------------------===//

let hasSideEffects = 0 in {
  let Defs = [AX], Uses = [AL] in
  def CBW : I<0x98, RawFrm, (outs), (ins),
              "{cbtw|cbw}", [], IIC_CBW>, OpSize16;  // AX = signext(AL)
  let Defs = [EAX], Uses = [AX] in
  def CWDE : I<0x98, RawFrm, (outs), (ins),
              "{cwtl|cwde}", [], IIC_CBW>, OpSize32; // EAX = signext(AX)

  let Defs = [AX,DX], Uses = [AX] in
  def CWD : I<0x99, RawFrm, (outs), (ins),
              "{cwtd|cwd}", [], IIC_CBW>, OpSize16; // DX:AX = signext(AX)
  let Defs = [EAX,EDX], Uses = [EAX] in
  def CDQ : I<0x99, RawFrm, (outs), (ins),
              "{cltd|cdq}", [], IIC_CBW>, OpSize32; // EDX:EAX = signext(EAX)


  let Defs = [RAX], Uses = [EAX] in
  def CDQE : RI<0x98, RawFrm, (outs), (ins),
               "{cltq|cdqe}", [], IIC_CBW>;     // RAX = signext(EAX)

  let Defs = [RAX,RDX], Uses = [RAX] in
  def CQO  : RI<0x99, RawFrm, (outs), (ins),
                "{cqto|cqo}", [], IIC_CBW>; // RDX:RAX = signext(RAX)
}



// Sign/Zero extenders
let hasSideEffects = 0 in {
def MOVSX16rr8 : I<0xBE, MRMSrcReg, (outs GR16:$dst), (ins GR8:$src),
                   "movs{bw|x}\t{$src, $dst|$dst, $src}", [], IIC_MOVSX_R16_R8>,
                   TB, OpSize16, Sched<[WriteALU]>;
let mayLoad = 1 in
def MOVSX16rm8 : I<0xBE, MRMSrcMem, (outs GR16:$dst), (ins i8mem:$src),
                   "movs{bw|x}\t{$src, $dst|$dst, $src}", [], IIC_MOVSX_R16_M8>,
                   TB, OpSize16, Sched<[WriteALULd]>;
} // hasSideEffects = 0
def MOVSX32rr8 : I<0xBE, MRMSrcReg, (outs GR32:$dst), (ins GR8:$src),
                   "movs{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sext GR8:$src))], IIC_MOVSX>, TB,
                   OpSize32, Sched<[WriteALU]>;
def MOVSX32rm8 : I<0xBE, MRMSrcMem, (outs GR32:$dst), (ins i8mem :$src),
                   "movs{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sextloadi32i8 addr:$src))], IIC_MOVSX>, TB,
                   OpSize32, Sched<[WriteALULd]>;
def MOVSX32rr16: I<0xBF, MRMSrcReg, (outs GR32:$dst), (ins GR16:$src),
                   "movs{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sext GR16:$src))], IIC_MOVSX>, TB,
                   OpSize32, Sched<[WriteALU]>;
def MOVSX32rm16: I<0xBF, MRMSrcMem, (outs GR32:$dst), (ins i16mem:$src),
                   "movs{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sextloadi32i16 addr:$src))], IIC_MOVSX>,
                   OpSize32, TB, Sched<[WriteALULd]>;

let hasSideEffects = 0 in {
def MOVZX16rr8 : I<0xB6, MRMSrcReg, (outs GR16:$dst), (ins GR8:$src),
                   "movz{bw|x}\t{$src, $dst|$dst, $src}", [], IIC_MOVZX_R16_R8>,
                   TB, OpSize16, Sched<[WriteALU]>;
let mayLoad = 1 in
def MOVZX16rm8 : I<0xB6, MRMSrcMem, (outs GR16:$dst), (ins i8mem:$src),
                   "movz{bw|x}\t{$src, $dst|$dst, $src}", [], IIC_MOVZX_R16_M8>,
                   TB, OpSize16, Sched<[WriteALULd]>;
} // hasSideEffects = 0
def MOVZX32rr8 : I<0xB6, MRMSrcReg, (outs GR32:$dst), (ins GR8 :$src),
                   "movz{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zext GR8:$src))], IIC_MOVZX>, TB,
                   OpSize32, Sched<[WriteALU]>;
def MOVZX32rm8 : I<0xB6, MRMSrcMem, (outs GR32:$dst), (ins i8mem :$src),
                   "movz{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zextloadi32i8 addr:$src))], IIC_MOVZX>, TB,
                   OpSize32, Sched<[WriteALULd]>;
def MOVZX32rr16: I<0xB7, MRMSrcReg, (outs GR32:$dst), (ins GR16:$src),
                   "movz{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zext GR16:$src))], IIC_MOVZX>, TB,
                   OpSize32, Sched<[WriteALU]>;
def MOVZX32rm16: I<0xB7, MRMSrcMem, (outs GR32:$dst), (ins i16mem:$src),
                   "movz{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zextloadi32i16 addr:$src))], IIC_MOVZX>,
                   TB, OpSize32, Sched<[WriteALULd]>;

// These are the same as the regular MOVZX32rr8 and MOVZX32rm8
// except that they use GR32_NOREX for the output operand register class
// instead of GR32. This allows them to operate on h registers on x86-64.
let hasSideEffects = 0, isCodeGenOnly = 1 in {
def MOVZX32_NOREXrr8 : I<0xB6, MRMSrcReg,
                         (outs GR32_NOREX:$dst), (ins GR8_NOREX:$src),
                         "movz{bl|x}\t{$src, $dst|$dst, $src}  # NOREX",
                         [], IIC_MOVZX>, TB, Sched<[WriteALU]>;
let mayLoad = 1 in
def MOVZX32_NOREXrm8 : I<0xB6, MRMSrcMem,
                         (outs GR32_NOREX:$dst), (ins i8mem_NOREX:$src),
                         "movz{bl|x}\t{$src, $dst|$dst, $src}  # NOREX",
                         [], IIC_MOVZX>, TB, Sched<[WriteALULd]>;

def MOVSX32_NOREXrr8 : I<0xBE, MRMSrcReg,
                         (outs GR32_NOREX:$dst), (ins GR8_NOREX:$src),
                         "movs{bl|x}\t{$src, $dst|$dst, $src}  # NOREX",
                         [], IIC_MOVSX>, TB, Sched<[WriteALU]>;
let mayLoad = 1 in
def MOVSX32_NOREXrm8 : I<0xBE, MRMSrcMem,
                         (outs GR32_NOREX:$dst), (ins i8mem_NOREX:$src),
                         "movs{bl|x}\t{$src, $dst|$dst, $src}  # NOREX",
                         [], IIC_MOVSX>, TB, Sched<[WriteALULd]>;
}

// MOVSX64rr8 always has a REX prefix and it has an 8-bit register
// operand, which makes it a rare instruction with an 8-bit register
// operand that can never access an h register. If support for h registers
// were generalized, this would require a special register class.
def MOVSX64rr8 : RI<0xBE, MRMSrcReg, (outs GR64:$dst), (ins GR8 :$src),
                    "movs{bq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR8:$src))], IIC_MOVSX>, TB,
                    Sched<[WriteALU]>;
def MOVSX64rm8 : RI<0xBE, MRMSrcMem, (outs GR64:$dst), (ins i8mem :$src),
                    "movs{bq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i8 addr:$src))], IIC_MOVSX>,
                    TB, Sched<[WriteALULd]>;
def MOVSX64rr16: RI<0xBF, MRMSrcReg, (outs GR64:$dst), (ins GR16:$src),
                    "movs{wq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR16:$src))], IIC_MOVSX>, TB,
                    Sched<[WriteALU]>;
def MOVSX64rm16: RI<0xBF, MRMSrcMem, (outs GR64:$dst), (ins i16mem:$src),
                    "movs{wq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i16 addr:$src))], IIC_MOVSX>,
                    TB, Sched<[WriteALULd]>;
def MOVSX64rr32: RI<0x63, MRMSrcReg, (outs GR64:$dst), (ins GR32:$src),
                    "movs{lq|xd}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR32:$src))], IIC_MOVSX>,
                    Sched<[WriteALU]>, Requires<[In64BitMode]>;
def MOVSX64rm32: RI<0x63, MRMSrcMem, (outs GR64:$dst), (ins i32mem:$src),
                    "movs{lq|xd}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i32 addr:$src))], IIC_MOVSX>,
                    Sched<[WriteALULd]>, Requires<[In64BitMode]>;

// movzbq and movzwq encodings for the disassembler
def MOVZX64rr8_Q : RI<0xB6, MRMSrcReg, (outs GR64:$dst), (ins GR8:$src),
                       "movz{bq|x}\t{$src, $dst|$dst, $src}", [], IIC_MOVZX>,
                       TB, Sched<[WriteALU]>;
def MOVZX64rm8_Q : RI<0xB6, MRMSrcMem, (outs GR64:$dst), (ins i8mem:$src),
                       "movz{bq|x}\t{$src, $dst|$dst, $src}", [], IIC_MOVZX>,
                       TB, Sched<[WriteALULd]>;
def MOVZX64rr16_Q : RI<0xB7, MRMSrcReg, (outs GR64:$dst), (ins GR16:$src),
                       "movz{wq|x}\t{$src, $dst|$dst, $src}", [], IIC_MOVZX>,
                       TB, Sched<[WriteALU]>;
def MOVZX64rm16_Q : RI<0xB7, MRMSrcMem, (outs GR64:$dst), (ins i16mem:$src),
                       "movz{wq|x}\t{$src, $dst|$dst, $src}", [], IIC_MOVZX>,
                       TB, Sched<[WriteALULd]>;

// 64-bit zero-extension patterns use SUBREG_TO_REG and an operation writing a
// 32-bit register.
def : Pat<(i64 (zext GR8:$src)),
          (SUBREG_TO_REG (i64 0), (MOVZX32rr8 GR8:$src), sub_32bit)>;
def : Pat<(zextloadi64i8 addr:$src),
          (SUBREG_TO_REG (i64 0), (MOVZX32rm8 addr:$src), sub_32bit)>;

def : Pat<(i64 (zext GR16:$src)),
          (SUBREG_TO_REG (i64 0), (MOVZX32rr16 GR16:$src), sub_32bit)>;
def : Pat<(zextloadi64i16 addr:$src),
          (SUBREG_TO_REG (i64 0), (MOVZX32rm16 addr:$src), sub_32bit)>;

// The preferred way to do 32-bit-to-64-bit zero extension on x86-64 is to use a
// SUBREG_TO_REG to utilize implicit zero-extension, however this isn't possible
// when the 32-bit value is defined by a truncate or is copied from something
// where the high bits aren't necessarily all zero. In such cases, we fall back
// to these explicit zext instructions.
def : Pat<(i64 (zext GR32:$src)),
          (SUBREG_TO_REG (i64 0), (MOV32rr GR32:$src), sub_32bit)>;
def : Pat<(i64 (zextloadi64i32 addr:$src)),
          (SUBREG_TO_REG (i64 0), (MOV32rm addr:$src), sub_32bit)>;
