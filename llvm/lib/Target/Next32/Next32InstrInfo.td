//===-- Next32InstrInfo.td - Next32 Instruction Definition -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Next32 pseudo instruction set. This instruction set
// will be used by Next32 runtime library and simulator to build HW DAG.
//
//===----------------------------------------------------------------------===//

include "Next32InstrFormats.td"

def HasPrefetch : Predicate<"Subtarget->hasPrefetch()">;
def HasVectorInst : Predicate<"Subtarget->hasVectorInst()">;

def DUP : Next32RRBinaryInstruction<0x00, "dup", next32dup> {
  let DecoderMethod = "DecodeDUPInstruction";
}
let hasSideEffects = 1 in {
  def SETFRAME : Next32RRRRFrameInstruction<0x40, "setframe", next32setframe>;
  def RESETFRAME
      : Next32RRRRTernaryInstruction<0x41, "resetframe", next32resetframe>;
  def ALLOCA : Next32RRRRFrameInstruction<0x42, "alloca", next32alloca>;

  def PREFETCH : Next32RRRInstruction<0x61, (outs),
                                      (ins GPR32
                                       : $dst, GPR32
                                       : $src, GPR32
                                       : $src2),
                                      "prefetch", "\t$dst, $src, $src2",
                                      [(next32prefetch GPR32
                                        : $dst, GPR32
                                        : $src, GPR32
                                        : $src2)]> {
    let Predicates = [HasPrefetch];
  }

  def VMEMREAD : Next32VMemInstruction<0x5F, "vmemread">;
  def VMEMWRITE : Next32VMemInstruction<0x60, "vmemwrite">;

  def MEMREAD : Next32MemInstruction<0x48, "memread">;
  def MEMREAD_ONCE : Next32MemInstruction<0x49, "memread.once">;
  def MEMREAD_ATOMIC : Next32MemInstruction<0x4C, "memread.atomic">;
  def MEMWRITE : Next32MemInstruction<0x4A, "memwrite">;
  def MEMWRITE_ONCE : Next32MemInstruction<0x4B, "memwrite.once">;
  def MEMWRITE_ATOMIC : Next32MemInstruction<0x4D, "memwrite.atomic">;

  def MEMCAS : Next32MemInstruction<0x4E, "memcas">;
  def MEMFA_SET : Next32MemInstruction<0x51, "memfa.set">;
  def MEMFA_ADD : Next32MemInstruction<0x52, "memfa.add">;
  def MEMFA_SUB : Next32MemInstruction<0x53, "memfa.sub">;
  def MEMFA_AND : Next32MemInstruction<0x54, "memfa.and">;
  def MEMFA_OR : Next32MemInstruction<0x55, "memfa.or">;
  def MEMFA_XOR : Next32MemInstruction<0x56, "memfa.xor">;
  def MEMFA_NAND : Next32MemInstruction<0x57, "memfa.nand">;
  def MEMFA_SMIN : Next32MemInstruction<0x58, "memfa.smin">;
  def MEMFA_SMAX : Next32MemInstruction<0x59, "memfa.smax">;
  def MEMFA_UMIN : Next32MemInstruction<0x5a, "memfa.umin">;
  def MEMFA_UMAX : Next32MemInstruction<0x5b, "memfa.umax">;
  def MEMFA_FADD : Next32MemInstruction<0x63, "memfa.fadd">;
  def MEMFA_FSUB : Next32MemInstruction<0x64, "memfa.fsub">;
}

defm CHAIN : Next32RRIInstruction<0x28, "chain", next32chain, next32condchain>;
defm CHAINP
    : Next32RRIInstruction<0x29, "chain.p", next32chainp, next32condchainp>;

let hasSideEffects = 1, Constraints = "$dst = $src" in {
  def MEMDATA : Next32MemDataInstruction<0x50,
                                         (outs GPR32
                                          : $dst),
                                         (ins GPR32
                                          : $src),
                                         "memdata", "\t$dst", []>;
}

let hasSideEffects = 1 in {
  def LEADISP : Next32RRInstruction<0x5c, (outs),
                                    (ins GPR32
                                     : $dst, GPR32
                                     : $src),
                                    "leadisp", "\t$dst, $src", []>;
  def LEAINDEX : Next32RRInstruction<0x5d, (outs),
                                     (ins GPR32
                                      : $dst, GPR32
                                      : $src),
                                     "leaindex", "\t$dst, $src", []>;
  def LEABASE : Next32RRLoHiInstruction<0x5e, "leabase", null_frag>;
  def LEASCALE : Next32RRInstruction<0x62, (outs),
                                     (ins GPR32
                                      : $dst, GPR32
                                      : $src),
                                     "leascale", "\t$dst, $src", []>;
}

let isPseudo = 1,
    Constraints = "$dst_high = $base_high, $dst_low = $base_low" in {
  def PSEUDOLEA
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst_high, GPR32
                           : $dst_low),
                          (ins GPR32
                           : $base_high, GPR32
                           : $base_low, GPR32
                           : $index_high, GPR32
                           : $index_low, GPR32
                           : $scale_high, GPR32
                           : $scale_low, GPR32
                           : $disp_hi, GPR32
                           : $disp_low),
                          "pseudolea", "", [(set GPR32
                                             : $dst_high, GPR32
                                             : $dst_low, (next32pseudolea GPR32
                                                          : $base_high, GPR32
                                                          : $base_low, GPR32
                                                          : $index_high, GPR32
                                                          : $index_low, GPR32
                                                          : $scale_high, GPR32
                                                          : $scale_low, GPR32
                                                          : $disp_hi, GPR32
                                                          : $disp_low))]>;
}

let isPseudo = 1, hasSideEffects = 1 in {
  def STORE_REG : Next32Instruction<0x00, (outs),
                                    (ins GPR32
                                     : $src, i32imm
                                     : $frame_index),
                                    "store_reg", "", []>;
  def LOAD_REG : Next32Instruction<0x00,
                                   (outs GPR32
                                    : $dst),
                                   (ins i32imm
                                    : $frame_index),
                                   "load_reg", "", []>;
  def GMEMREAD_1
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst),
                          (ins i64imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, variable_ops),
                          "gmemread_1", "", [(set GPR32
                                              : $dst, (next32gmemread1 i64
                                                       : $params, GPR32
                                                       : $addr_high, GPR32
                                                       : $addr_low, GPR32
                                                       : $tid))]>;
  def GMEMREAD_2
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst1, GPR32
                           : $dst2),
                          (ins i64imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, variable_ops),
                          "gmemread_2", "", [(set GPR32
                                              : $dst1, GPR32
                                              : $dst2, (next32gmemread2 i64
                                                        : $params, GPR32
                                                        : $addr_high, GPR32
                                                        : $addr_low, GPR32
                                                        : $tid))]>;
  def GMEMREAD_4
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst1, GPR32
                           : $dst2, GPR32
                           : $dst3, GPR32
                           : $dst4),
                          (ins i64imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, variable_ops),
                          "gmemread_4", "", [(set GPR32
                                              : $dst1, GPR32
                                              : $dst2, GPR32
                                              : $dst3, GPR32
                                              : $dst4, (next32gmemread4 i64
                                                        : $params, GPR32
                                                        : $addr_high, GPR32
                                                        : $addr_low, GPR32
                                                        : $tid))]>;
  def GMEMREAD_8
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst1, GPR32
                           : $dst2, GPR32
                           : $dst3, GPR32
                           : $dst4, GPR32
                           : $dst5, GPR32
                           : $dst6, GPR32
                           : $dst7, GPR32
                           : $dst8),
                          (ins i64imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, variable_ops),
                          "gmemread_8", "", [(set GPR32
                                              : $dst1, GPR32
                                              : $dst2, GPR32
                                              : $dst3, GPR32
                                              : $dst4, GPR32
                                              : $dst5, GPR32
                                              : $dst6, GPR32
                                              : $dst7, GPR32
                                              : $dst8, (next32gmemread8 i64
                                                        : $params, GPR32
                                                        : $addr_high, GPR32
                                                        : $addr_low, GPR32
                                                        : $tid))]>;
  def GMEMREAD_16
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst1, GPR32
                           : $dst2, GPR32
                           : $dst3, GPR32
                           : $dst4, GPR32
                           : $dst5, GPR32
                           : $dst6, GPR32
                           : $dst7, GPR32
                           : $dst8, GPR32
                           : $dst9, GPR32
                           : $dst10, GPR32
                           : $dst11, GPR32
                           : $dst12, GPR32
                           : $dst13, GPR32
                           : $dst14, GPR32
                           : $dst15, GPR32
                           : $dst16),
                          (ins i64imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, variable_ops),
                          "gmemread_16", "", [(set GPR32
                                               : $dst1, GPR32
                                               : $dst2, GPR32
                                               : $dst3, GPR32
                                               : $dst4, GPR32
                                               : $dst5, GPR32
                                               : $dst6, GPR32
                                               : $dst7, GPR32
                                               : $dst8, GPR32
                                               : $dst9, GPR32
                                               : $dst10, GPR32
                                               : $dst11, GPR32
                                               : $dst12, GPR32
                                               : $dst13, GPR32
                                               : $dst14, GPR32
                                               : $dst15, GPR32
                                               : $dst16, (next32gmemread16 i64
                                                          : $params, GPR32
                                                          : $addr_high, GPR32
                                                          : $addr_low, GPR32
                                                          : $tid))]>;
  let Predicates = [HasVectorInst] in {
    def GVMEMREAD_1
        : Next32Instruction<0x00,
                            (outs GPR32
                             : $dst1),
                            (ins i32imm
                             : $params, GPR32
                             : $addr_high, GPR32
                             : $addr_low, GPR32
                             : $tid, GPR32
                             : $pred, variable_ops),
                            "gvmemread_1", "", [(set GPR32
                                                 : $dst1, (next32gvmemread1 i32
                                                           : $params, GPR32
                                                           : $addr_high, GPR32
                                                           : $addr_low, GPR32
                                                           : $tid, GPR32
                                                           : $pred))]>;
    def GVMEMREAD_2
        : Next32Instruction<0x00,
                            (outs GPR32
                             : $dst1, GPR32
                             : $dst2),
                            (ins i32imm
                             : $params, GPR32
                             : $addr_high, GPR32
                             : $addr_low, GPR32
                             : $tid, GPR32
                             : $pred, variable_ops),
                            "gvmemread_2", "", [(set GPR32
                                                 : $dst1, GPR32
                                                 : $dst2, (next32gvmemread2 i32
                                                           : $params, GPR32
                                                           : $addr_high, GPR32
                                                           : $addr_low, GPR32
                                                           : $tid, GPR32
                                                           : $pred))]>;
    def GVMEMREAD_4
        : Next32Instruction<0x00,
                            (outs GPR32
                             : $dst1, GPR32
                             : $dst2, GPR32
                             : $dst3, GPR32
                             : $dst4),
                            (ins i32imm
                             : $params, GPR32
                             : $addr_high, GPR32
                             : $addr_low, GPR32
                             : $tid, GPR32
                             : $pred, variable_ops),
                            "gvmemread_4", "", [(set GPR32
                                                 : $dst1, GPR32
                                                 : $dst2, GPR32
                                                 : $dst3, GPR32
                                                 : $dst4, (next32gvmemread4 i32
                                                           : $params, GPR32
                                                           : $addr_high, GPR32
                                                           : $addr_low, GPR32
                                                           : $tid, GPR32
                                                           : $pred))]>;
    def GVMEMREAD_8
        : Next32Instruction<0x00,
                            (outs GPR32
                             : $dst1, GPR32
                             : $dst2, GPR32
                             : $dst3, GPR32
                             : $dst4, GPR32
                             : $dst5, GPR32
                             : $dst6, GPR32
                             : $dst7, GPR32
                             : $dst8),
                            (ins i32imm
                             : $params, GPR32
                             : $addr_high, GPR32
                             : $addr_low, GPR32
                             : $tid, GPR32
                             : $pred, variable_ops),
                            "gvmemread_8", "", [(set GPR32
                                                 : $dst1, GPR32
                                                 : $dst2, GPR32
                                                 : $dst3, GPR32
                                                 : $dst4, GPR32
                                                 : $dst5, GPR32
                                                 : $dst6, GPR32
                                                 : $dst7, GPR32
                                                 : $dst8, (next32gvmemread8 i32
                                                           : $params, GPR32
                                                           : $addr_high, GPR32
                                                           : $addr_low, GPR32
                                                           : $tid, GPR32
                                                           : $pred))]>;
    def GVMEMREAD_16 : Next32Instruction<0x00,
                                         (outs GPR32
                                          : $dst1, GPR32
                                          : $dst2, GPR32
                                          : $dst3, GPR32
                                          : $dst4, GPR32
                                          : $dst5, GPR32
                                          : $dst6, GPR32
                                          : $dst7, GPR32
                                          : $dst8, GPR32
                                          : $dst9, GPR32
                                          : $dst10, GPR32
                                          : $dst11, GPR32
                                          : $dst12, GPR32
                                          : $dst13, GPR32
                                          : $dst14, GPR32
                                          : $dst15, GPR32
                                          : $dst16),
                                         (ins i32imm
                                          : $params, GPR32
                                          : $addr_high, GPR32
                                          : $addr_low, GPR32
                                          : $tid, GPR32
                                          : $pred, variable_ops),
                                         "gvmemread_16", "",
                                         [(set GPR32
                                           : $dst1, GPR32
                                           : $dst2, GPR32
                                           : $dst3, GPR32
                                           : $dst4, GPR32
                                           : $dst5, GPR32
                                           : $dst6, GPR32
                                           : $dst7, GPR32
                                           : $dst8, GPR32
                                           : $dst9, GPR32
                                           : $dst10, GPR32
                                           : $dst11, GPR32
                                           : $dst12, GPR32
                                           : $dst13, GPR32
                                           : $dst14, GPR32
                                           : $dst15, GPR32
                                           : $dst16, (next32gvmemread16 i32
                                                      : $params, GPR32
                                                      : $addr_high, GPR32
                                                      : $addr_low, GPR32
                                                      : $tid, GPR32
                                                      : $pred))]>;
  } // Predicates = [HasVectorInst]
}

let Constraints = "$dst = $src", isPseudo = 1, hasSideEffects = 1 in {
  def GMEMWRITE
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst),
                          (ins i64imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, GPR32
                           : $src, variable_ops),
                          "gmemwrite", "", [(set GPR32
                                             : $dst, (next32gmemwrite i64
                                                      : $params, GPR32
                                                      : $addr_high, GPR32
                                                      : $addr_low, GPR32
                                                      : $tid, GPR32
                                                      : $src))]>;
  def GVMEMWRITE
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst),
                          (ins i32imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, GPR32
                           : $pred, GPR32
                           : $src, variable_ops),
                          "gvmemwrite", "", [(set GPR32
                                              : $dst, (next32gvmemwrite i32
                                                       : $params, GPR32
                                                       : $addr_high, GPR32
                                                       : $addr_low, GPR32
                                                       : $tid, GPR32
                                                       : $pred, GPR32
                                                       : $src))]> {
    let Predicates = [HasVectorInst];
  }
}

let isPseudo = 1, hasSideEffects = 1 in {
  def GMEMFAOP_S
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst),
                          (ins i64imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, GPR32
                           : $src),
                          "gmemfaop_s", "", [(set GPR32
                                              : $dst, (next32gmemfaops i64
                                                       : $params, i32
                                                       : $addr_high, i32
                                                       : $addr_low, i32
                                                       : $tid, i32
                                                       : $src))]> {
    let Constraints = "$dst = $src";
  }
  def GMEMFAOP_D
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst_high, GPR32
                           : $dst_low),
                          (ins i64imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, GPR32
                           : $src_high, GPR32
                           : $src_low),
                          "gmemfaop_d", "", [(set GPR32
                                              : $dst_high, GPR32
                                              : $dst_low, (next32gmemfaopd i64
                                                           : $params, i32
                                                           : $addr_high, i32
                                                           : $addr_low, i32
                                                           : $tid, i32
                                                           : $src_high, i32
                                                           : $src_low))]> {
    let Constraints = "$dst_high = $src_high, $dst_low = $src_low";
  }
  def GMEMCAS_S
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst),
                          (ins i64imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, GPR32
                           : $cmp, GPR32
                           : $src),
                          "gmemcas_s", "", [(set GPR32
                                             : $dst, (next32gmemcass i64
                                                      : $params, i32
                                                      : $addr_high, i32
                                                      : $addr_low, i32
                                                      : $tid, i32
                                                      : $cmp, i32
                                                      : $src))]> {
    let Constraints = "$dst = $src";
  }
  def GMEMCAS_D
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $dst_high, GPR32
                           : $dst_low),
                          (ins i64imm
                           : $params, GPR32
                           : $addr_high, GPR32
                           : $addr_low, GPR32
                           : $tid, GPR32
                           : $cmp_high, GPR32
                           : $cmp_low, GPR32
                           : $swap_high, GPR32
                           : $swap_low),
                          "gmemcas_d", "", [(set GPR32
                                             : $dst_high, GPR32
                                             : $dst_low, (next32gmemcasd i64
                                                          : $params, i32
                                                          : $addr_high, i32
                                                          : $addr_low, i32
                                                          : $tid, i32
                                                          : $cmp_high, i32
                                                          : $cmp_low, i32
                                                          : $swap_high, i32
                                                          : $swap_low))]> {
    let Constraints = "$dst_high = $swap_high, $dst_low = $swap_low";
  }
}

let isAsCheapAsAMove = 1 in {
  def MUL : Next32RRBinaryInstruction<0x18, "mul", mul>;
  def UMUL : Next32RRLoHiInstruction<0x19, "umul", umullohi>;
  def SMUL : Next32RRLoHiInstruction<0x1A, "smul", smullohi>;

  defm ADD : Next32RRBinaryInstruction<0x10, "add", add, next32condadd>;
  defm XOR : Next32RRBinaryInstruction<0x09, "xor", xor, next32condxor>;
  defm OR : Next32RRBinaryInstruction<0x0A, "or", or, next32condor>;
  defm AND : Next32RRBinaryInstruction<0x08, "and", and, next32condand>;
  defm SUB : Next32RRBinaryInstruction<0x11, "sub", sub, next32condsub>;
  defm SHL : Next32RRBinaryInstruction<0x0B, "shl", shl, next32condshl>;
  defm SHR : Next32RRBinaryInstruction<0x0C, "shr", srl,
                                       next32condshr>; // = llvm lshr
  defm SHRI : Next32RRBinaryInstruction<0x0D, "shri", sra,
                                        next32condshri>; // = llvm ashr
  defm SELECT : Next32RRBinaryInstruction<0x1E, "select", next32select,
                                          next32condselect>;
  defm SMIN : Next32RRBinaryInstruction<0x32, "smin", smin, null_frag>;
  defm SMAX : Next32RRBinaryInstruction<0x33, "smax", smax, null_frag>;
  defm UMIN : Next32RRBinaryInstruction<0x34, "umin", umin, null_frag>;
  defm UMAX : Next32RRBinaryInstruction<0x35, "umax", umax, null_frag>;

  let isTerminator = 1 in {
    def WRITER : Next32RRSizedBinaryInstruction<0x20, "writer", next32writer>;
  }

  def SHL64 : Next32RRRRShiftInstruction<0x65, "shl64", next32shl64>;
  def SHR64 : Next32RRRRShiftInstruction<0x66, "shr64", next32shr64>;
  def SHRI64 : Next32RRRRShiftInstruction<0x67, "shri64", next32shri64>;

  defm NOT : Next32RRUnaryInstruction<0x04, "not", not, next32condnot>;
  defm NEG : Next32RRUnaryInstruction<0x01, "neg", ineg, next32condneg>;
  defm INC : Next32RRUnaryInstruction<0x14, "inc", next32inc, next32condinc>;
  defm DEC : Next32RRUnaryInstruction<0x15, "dec", next32dec, next32conddec>;
  defm CLZ : Next32RRUnaryInstruction<0x16, "clz", ctlz, next32condctlz>;
  defm CTZ : Next32RRUnaryInstruction<0x17, "ctz", cttz, next32condcttz>;

  let hasSideEffects = 1 in {
    def FEEDER : Next32RRSizedUnaryInstruction<0x02, "feeder">;
    def FEEDERP : Next32RRSizedUnaryInstruction<0x03, "feeder.p">;
    def CALLPTR : Next32RRBinaryInstruction<0x2A, "callptr", Next32CallPtr>;
  }

  defm SEXT : Next32Size16RRUnaryInstruction<0x30, "sext">;
  defm ZEXT : Next32Size16RRUnaryInstruction<0x31, "zext">;

  // Use null_frag because we currently create these only in FLAGS
  // pseudo-opcode expansion and they shouldn't come from SDAG patterns.
  // TODO: Add null_frag as default argument in the appropriate class
  // parameters.
  def FLAGS : Next32RRUnaryInstruction<0x1F, "flags", null_frag>;
  def BARRIER : Next32RRBinaryInstruction<0x1D, "barrier", Next32Barrier>;

  let Constraints = "$dst = $tempsrc" in {
    def ADC : Next32RRRInstruction<0x12,
                                   (outs GPR32
                                    : $dst),
                                   (ins GPR32
                                    : $tempsrc, GPR32
                                    : $src, GPR32
                                    : $src2),
                                   "adc", "\t$dst, $src [$src2]", []>;
    def SBB : Next32RRRInstruction<0x13,
                                   (outs GPR32
                                    : $dst),
                                   (ins GPR32
                                    : $tempsrc, GPR32
                                    : $src, GPR32
                                    : $src2),
                                   "sbb", "\t$dst, $src [$src2]", []>;
  }
}

def : Pat<(and i32 : $dst, 0xFF), (ZEXT_8 i32 : $dst)>;
def : Pat<(and i32 : $dst, 0xFFFF), (ZEXT_16 i32 : $dst)>;
def : Pat<(sext_inreg i32 : $dst, i8), (SEXT_8 i32 : $dst)>;
def : Pat<(sext_inreg i32 : $dst, i16), (SEXT_16 i32 : $dst)>;

let isAsCheapAsAMove = 1 in { def MOVL : Next32MOVInstruction; }

let isPseudo = 1, Constraints = "$value = $lhs" in {
  def ADDFLAGS
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $value, GPR32
                           : $flags),
                          (ins GPR32
                           : $lhs, GPR32
                           : $rhs),
                          "addflags", "", [(set GPR32
                                            : $value, GPR32
                                            : $flags, (next32addflags i32
                                                       : $lhs, i32
                                                       : $rhs))]>;
  def SUBFLAGS
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $value, GPR32
                           : $flags),
                          (ins GPR32
                           : $lhs, GPR32
                           : $rhs),
                          "subflags", "", [(set GPR32
                                            : $value, GPR32
                                            : $flags, (next32subflags i32
                                                       : $lhs, i32
                                                       : $rhs))]>;
  def ADCFLAGS
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $value, GPR32
                           : $flags),
                          (ins GPR32
                           : $lhs, GPR32
                           : $rhs, GPR32
                           : $carry),
                          "adcflags", "", [(set GPR32
                                            : $value, GPR32
                                            : $flags, (next32adcflags i32
                                                       : $lhs, i32
                                                       : $rhs, i32
                                                       : $carry))]>;
  def SBBFLAGS
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $value, GPR32
                           : $flags),
                          (ins GPR32
                           : $lhs, GPR32
                           : $rhs, GPR32
                           : $carry),
                          "sbbflags", "", [(set GPR32
                                            : $value, GPR32
                                            : $flags, (next32sbbflags i32
                                                       : $lhs, i32
                                                       : $rhs, i32
                                                       : $carry))]>;
}

let isPseudo = 1, hasSideEffects = 1 in {
  def SET_TID : Next32Instruction<0x0, (outs),
                                  (ins GPR32
                                   : $src),
                                  "set_tid", "\t$src", [(Next32SetTID i32
                                                         : $src)]>;
  def SYM_INSTR : Next32Instruction<0x0, (outs),
                                    (ins next32mcsymbol
                                     : $sym),
                                    "sym", "", [(Next32Sym mcsym
                                                 : $sym)]>;

  // CALL_TERMINATOR should take both variadic arguments and produce variadic
  // results, but this is not possible to model directly. Instead, we select
  // this to CALL_TERMINATOR_PARAMS taking variadic arguments linked with a
  // CALL_TERMINATOR_RESULTS that produces the variadic results. We recombine
  // the two in a custom inserter hook after DAG ISel.
  def CALL_TERMINATOR_PARAMS
      : Next32Instruction<0x0, (outs), (ins variable_ops),
                          "call_terminator_params", "", []>;
  let variadicOpsAreDefs = 1, usesCustomInserter =
                                  1 in def CALL_TERMINATOR_RESULTS
      : Next32Instruction<0x0, (outs), (ins variable_ops),
                          "call_terminator_results", "", []>;
  def CALL_TERMINATOR : Next32Instruction<0x0, (outs), (ins variable_ops),
                                          "call_terminator", "", []>;

  def CALL_TERMINATOR_TID
      : Next32Instruction<0x0, (outs), (ins variable_ops), "call_terminator",
                          "", [(Next32CallTerminatorTID)]>;
  def FEEDER_ARGS
      : Next32Instruction<0x0, (outs), (ins variable_ops), "feeder_pseudo", "",
                          [(Next32FeederArgs)]>;
}

let isPseudo = 1, isWriterChain = 1 in {
  let isTerminator = 1, isCall = 1 in {
    def CALLPTRWRAPPER
        : Next32Instruction<0x0, (outs), (ins variable_ops), "callptrwrapper",
                            "", [(Next32CallPtrWrapper)]>;
    def CALL : Next32Instruction<0x0, (outs), (ins variable_ops), "call", "",
                                 [(Next32Call)]>;
    def CALLc : Next32Instruction<0x0, (outs), (ins variable_ops), "call", "",
                                  [(Next32Call)]>;
  }

  let isTerminator = 1, isBarrier = 1 in {
    let isBranch = 1 in {
      def BR : Next32Instruction<0x0, (outs),
                                 (ins next32branchtarget
                                  : $dstbb),
                                 "br", "\t$dstbb", [(br bb
                                                     : $dstbb)]>;
      def BR_CC : Next32Instruction<0x0, (outs),
                                    (ins i32imm
                                     : $brcc_cond, GPR32
                                     : $brcc_value, next32branchtarget
                                     : $dstbb),
                                    "br_cc", "\t$dstbb", [(next32brcc i32
                                                           : $brcc_cond, i32
                                                           : $brcc_value, bb
                                                           : $dstbb)]>;
    }

    let isReturn = 1 in {
      def RET : Next32Instruction<0x0, (outs), (ins variable_ops), "ret", "",
                                  [(Next32RetFlag)]>;
      def RETc : Next32Instruction<0x0, (outs), (ins variable_ops), "ret", "",
                                   [(Next32RetFlag)]>;
    }
  }
}

let isPseudo = 1, isAsCheapAsAMove = 1 in {
  // This is a pesudo instruction that will be expanded to movl.
  def FRAME_OFFSET_WRAPPER
      : Next32Instruction<0x00,
                          (outs GPR32
                           : $OFFSET_HIGH, GPR32
                           : $OFFSET_LOW),
                          (ins i32imm
                           : $base, i64imm
                           : $offset),
                          "frame_offset_wrapper", "\t$base,$offset",
                          [(set GPR32
                            : $OFFSET_HIGH, GPR32
                            : $OFFSET_LOW, (next32frameoffsetwrapper i32
                                            : $base, i64
                                            : $offset))]>;
}
