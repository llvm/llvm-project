//===-- X86Schedule.td - X86 Scheduling Definitions --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// InstrSchedModel annotations for out-of-order CPUs.
//
// These annotations are independent of the itinerary classes defined below.

// Instructions with folded loads need to read the memory operand immediately,
// but other register operands don't have to be read until the load is ready.
// These operands are marked with ReadAfterLd.
def ReadAfterLd : SchedRead;

// Instructions with both a load and a store folded are modeled as a folded
// load + WriteRMW.
def WriteRMW : SchedWrite;

// Most instructions can fold loads, so almost every SchedWrite comes in two
// variants: With and without a folded load.
// An X86FoldableSchedWrite holds a reference to the corresponding SchedWrite
// with a folded load.
class X86FoldableSchedWrite : SchedWrite {
  // The SchedWrite to use when a load is folded into the instruction.
  SchedWrite Folded;
}

// Multiclass that produces a linked pair of SchedWrites.
multiclass X86SchedWritePair {
  // Register-Memory operation.
  def Ld : SchedWrite;
  // Register-Register operation.
  def NAME : X86FoldableSchedWrite {
    let Folded = !cast<SchedWrite>(NAME#"Ld");
  }
}

// Loads, stores, and moves, not folded with other operations.
def WriteLoad  : SchedWrite;
def WriteStore : SchedWrite;
def WriteMove  : SchedWrite;

// Arithmetic.
defm WriteALU  : X86SchedWritePair; // Simple integer ALU op.
def WriteALURMW : WriteSequence<[WriteALULd, WriteStore]>;
defm WriteIMul : X86SchedWritePair; // Integer multiplication.
def  WriteIMulH : SchedWrite;       // Integer multiplication, high part.
defm WriteIDiv : X86SchedWritePair; // Integer division.
def  WriteLEA  : SchedWrite;        // LEA instructions can't fold loads.

defm WriteBitScan : X86SchedWritePair; // Bit scan forward/reverse.
defm WritePOPCNT : X86SchedWritePair; // Bit population count.
defm WriteLZCNT : X86SchedWritePair; // Leading zero count.
defm WriteTZCNT : X86SchedWritePair; // Trailing zero count.
defm WriteCMOV : X86SchedWritePair; // Conditional move.
def  WriteSETCC : SchedWrite; // Set register based on condition code.
def  WriteSETCCStore : SchedWrite;

// Integer shifts and rotates.
defm WriteShift : X86SchedWritePair;

// BMI1 BEXTR, BMI2 BZHI
defm WriteBEXTR : X86SchedWritePair;
defm WriteBZHI  : X86SchedWritePair;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
def WriteZero : SchedWrite;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm WriteJump : X86SchedWritePair;

// Floating point. This covers both scalar and vector operations.
def  WriteFLoad  : SchedWrite;
def  WriteFStore : SchedWrite;
def  WriteFMove  : SchedWrite;
defm WriteFAdd   : X86SchedWritePair; // Floating point add/sub/compare.
defm WriteFMul   : X86SchedWritePair; // Floating point multiplication.
defm WriteFDiv   : X86SchedWritePair; // Floating point division.
defm WriteFSqrt  : X86SchedWritePair; // Floating point square root.
defm WriteFRcp   : X86SchedWritePair; // Floating point reciprocal estimate.
defm WriteFRsqrt : X86SchedWritePair; // Floating point reciprocal square root estimate.
defm WriteFMA    : X86SchedWritePair; // Fused Multiply Add.
defm WriteFShuffle  : X86SchedWritePair; // Floating point vector shuffles.
defm WriteFVarShuffle  : X86SchedWritePair; // Floating point vector variable shuffles.
defm WriteFBlend  : X86SchedWritePair; // Floating point vector blends.
defm WriteFVarBlend  : X86SchedWritePair; // Fp vector variable blends.

// FMA Scheduling helper class.
class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Horizontal Add/Sub (float and integer)
defm WriteFHAdd  : X86SchedWritePair;
defm WritePHAdd : X86SchedWritePair;

// Vector integer operations.
def  WriteVecLoad  : SchedWrite;
def  WriteVecStore : SchedWrite;
def  WriteVecMove  : SchedWrite;
defm WriteVecALU   : X86SchedWritePair; // Vector integer ALU op, no logicals.
defm WriteVecShift : X86SchedWritePair; // Vector integer shifts.
defm WriteVecIMul  : X86SchedWritePair; // Vector integer multiply.
defm WritePMULLD : X86SchedWritePair; // PMULLD
defm WriteShuffle  : X86SchedWritePair; // Vector shuffles.
defm WriteVarShuffle  : X86SchedWritePair; // Vector variable shuffles.
defm WriteBlend  : X86SchedWritePair; // Vector blends.
defm WriteVarBlend  : X86SchedWritePair; // Vector variable blends.
defm WriteMPSAD : X86SchedWritePair; // Vector MPSAD.

// Vector bitwise operations.
// These are often used on both floating point and integer vectors.
defm WriteVecLogic : X86SchedWritePair; // Vector and/or/xor.

// MOVMSK operations.
def WriteFMOVMSK : SchedWrite;
def WriteVecMOVMSK : SchedWrite;
def WriteMMXMOVMSK : SchedWrite;

// Conversion between integer and float.
defm WriteCvtF2I : X86SchedWritePair; // Float -> Integer.
defm WriteCvtI2F : X86SchedWritePair; // Integer -> Float.
defm WriteCvtF2F : X86SchedWritePair; // Float -> Float size conversion.

// CRC32 instruction.
defm WriteCRC32 : X86SchedWritePair;

// Strings instructions.
// Packed Compare Implicit Length Strings, Return Mask
defm WritePCmpIStrM : X86SchedWritePair;
// Packed Compare Explicit Length Strings, Return Mask
defm WritePCmpEStrM : X86SchedWritePair;
// Packed Compare Implicit Length Strings, Return Index
defm WritePCmpIStrI : X86SchedWritePair;
// Packed Compare Explicit Length Strings, Return Index
defm WritePCmpEStrI : X86SchedWritePair;

// AES instructions.
defm WriteAESDecEnc : X86SchedWritePair; // Decryption, encryption.
defm WriteAESIMC : X86SchedWritePair; // InvMixColumn.
defm WriteAESKeyGen : X86SchedWritePair; // Key Generation.

// Carry-less multiplication instructions.
defm WriteCLMul : X86SchedWritePair;

// Catch-all for expensive system instructions.
def WriteSystem : SchedWrite;

// AVX2.
defm WriteFShuffle256 : X86SchedWritePair; // Fp 256-bit width vector shuffles.
defm WriteFVarShuffle256 : X86SchedWritePair; // Fp 256-bit width variable shuffles.
defm WriteShuffle256 : X86SchedWritePair; // 256-bit width vector shuffles.
defm WriteVarShuffle256 : X86SchedWritePair; // 256-bit width vector variable shuffles.
defm WriteVarVecShift : X86SchedWritePair; // Variable vector shifts.

// Old microcoded instructions that nobody use.
def WriteMicrocoded : SchedWrite;

// Fence instructions.
def WriteFence : SchedWrite;

// Nop, not very useful expect it provides a model for nops!
def WriteNop : SchedWrite;

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for X86
def IIC_ALU_MEM     : InstrItinClass;
def IIC_ALU_NONMEM  : InstrItinClass;
def IIC_LEA         : InstrItinClass;
def IIC_LEA_16      : InstrItinClass;
def IIC_MUL8_MEM    : InstrItinClass;
def IIC_MUL8_REG    : InstrItinClass;
def IIC_MUL16_MEM   : InstrItinClass;
def IIC_MUL16_REG   : InstrItinClass;
def IIC_MUL32_MEM   : InstrItinClass;
def IIC_MUL32_REG   : InstrItinClass;
def IIC_MUL64_MEM   : InstrItinClass;
def IIC_MUL64_REG   : InstrItinClass;
// imul by al, ax, eax, tax
def IIC_IMUL8_MEM   : InstrItinClass;
def IIC_IMUL8_REG   : InstrItinClass;
def IIC_IMUL16_MEM  : InstrItinClass;
def IIC_IMUL16_REG  : InstrItinClass;
def IIC_IMUL32_MEM  : InstrItinClass;
def IIC_IMUL32_REG  : InstrItinClass;
def IIC_IMUL64_MEM  : InstrItinClass;
def IIC_IMUL64_REG  : InstrItinClass;
// imul reg by reg|mem
def IIC_IMUL16_RM   : InstrItinClass;
def IIC_IMUL16_RR   : InstrItinClass;
def IIC_IMUL32_RM   : InstrItinClass;
def IIC_IMUL32_RR   : InstrItinClass;
def IIC_IMUL64_RM   : InstrItinClass;
def IIC_IMUL64_RR   : InstrItinClass;
// imul reg = reg/mem * imm
def IIC_IMUL16_RMI  : InstrItinClass;
def IIC_IMUL16_RRI  : InstrItinClass;
def IIC_IMUL32_RMI  : InstrItinClass;
def IIC_IMUL32_RRI  : InstrItinClass;
def IIC_IMUL64_RMI  : InstrItinClass;
def IIC_IMUL64_RRI  : InstrItinClass;
// div
def IIC_DIV8_MEM    : InstrItinClass;
def IIC_DIV8_REG    : InstrItinClass;
def IIC_DIV16_MEM   : InstrItinClass;
def IIC_DIV16_REG   : InstrItinClass;
def IIC_DIV32_MEM   : InstrItinClass;
def IIC_DIV32_REG   : InstrItinClass;
def IIC_DIV64_MEM   : InstrItinClass;
def IIC_DIV64_REG   : InstrItinClass;
// idiv
def IIC_IDIV8_MEM   : InstrItinClass;
def IIC_IDIV8_REG   : InstrItinClass;
def IIC_IDIV16_MEM  : InstrItinClass;
def IIC_IDIV16_REG  : InstrItinClass;
def IIC_IDIV32_MEM  : InstrItinClass;
def IIC_IDIV32_REG  : InstrItinClass;
def IIC_IDIV64_MEM  : InstrItinClass;
def IIC_IDIV64_REG  : InstrItinClass;
// neg/not/inc/dec
def IIC_UNARY_REG   : InstrItinClass;
def IIC_UNARY_MEM   : InstrItinClass;
// add/sub/and/or/xor/sbc/cmp/test
def IIC_BIN_MEM     : InstrItinClass;
def IIC_BIN_NONMEM  : InstrItinClass;
// adc/sbc
def IIC_BIN_CARRY_MEM     : InstrItinClass;
def IIC_BIN_CARRY_NONMEM  : InstrItinClass;
// shift/rotate
def IIC_SR          : InstrItinClass;
// shift double
def IIC_SHD16_REG_IM : InstrItinClass;
def IIC_SHD16_REG_CL : InstrItinClass;
def IIC_SHD16_MEM_IM : InstrItinClass;
def IIC_SHD16_MEM_CL : InstrItinClass;
def IIC_SHD32_REG_IM : InstrItinClass;
def IIC_SHD32_REG_CL : InstrItinClass;
def IIC_SHD32_MEM_IM : InstrItinClass;
def IIC_SHD32_MEM_CL : InstrItinClass;
def IIC_SHD64_REG_IM : InstrItinClass;
def IIC_SHD64_REG_CL : InstrItinClass;
def IIC_SHD64_MEM_IM : InstrItinClass;
def IIC_SHD64_MEM_CL : InstrItinClass;
//sign extension movs
def IIC_MOVSX : InstrItinClass;
def IIC_MOVSX_R16_R8 : InstrItinClass;
def IIC_MOVSX_R16_M8 : InstrItinClass;
//zero extension movs
def IIC_MOVZX : InstrItinClass;
def IIC_MOVZX_R16_R8 : InstrItinClass;
def IIC_MOVZX_R16_M8 : InstrItinClass;

def IIC_REP_MOVS : InstrItinClass;
def IIC_REP_STOS : InstrItinClass;

// SSE scalar/parallel binary operations
def IIC_SSE_ALU_F32S_RR : InstrItinClass;
def IIC_SSE_ALU_F32S_RM : InstrItinClass;
def IIC_SSE_ALU_F64S_RR : InstrItinClass;
def IIC_SSE_ALU_F64S_RM : InstrItinClass;
def IIC_SSE_MUL_F32S_RR : InstrItinClass;
def IIC_SSE_MUL_F32S_RM : InstrItinClass;
def IIC_SSE_MUL_F64S_RR : InstrItinClass;
def IIC_SSE_MUL_F64S_RM : InstrItinClass;
def IIC_SSE_DIV_F32S_RR : InstrItinClass;
def IIC_SSE_DIV_F32S_RM : InstrItinClass;
def IIC_SSE_DIV_F64S_RR : InstrItinClass;
def IIC_SSE_DIV_F64S_RM : InstrItinClass;
def IIC_SSE_ALU_F32P_RR : InstrItinClass;
def IIC_SSE_ALU_F32P_RM : InstrItinClass;
def IIC_SSE_ALU_F64P_RR : InstrItinClass;
def IIC_SSE_ALU_F64P_RM : InstrItinClass;
def IIC_SSE_MUL_F32P_RR : InstrItinClass;
def IIC_SSE_MUL_F32P_RM : InstrItinClass;
def IIC_SSE_MUL_F64P_RR : InstrItinClass;
def IIC_SSE_MUL_F64P_RM : InstrItinClass;
def IIC_SSE_DIV_F32P_RR : InstrItinClass;
def IIC_SSE_DIV_F32P_RM : InstrItinClass;
def IIC_SSE_DIV_F64P_RR : InstrItinClass;
def IIC_SSE_DIV_F64P_RM : InstrItinClass;

def IIC_SSE_COMIS_RR : InstrItinClass;
def IIC_SSE_COMIS_RM : InstrItinClass;

def IIC_SSE_HADDSUB_RR : InstrItinClass;
def IIC_SSE_HADDSUB_RM : InstrItinClass;

def IIC_SSE_BIT_P_RR  : InstrItinClass;
def IIC_SSE_BIT_P_RM  : InstrItinClass;

def IIC_SSE_INTALU_P_RR  : InstrItinClass;
def IIC_SSE_INTALU_P_RM  : InstrItinClass;
def IIC_SSE_INTALUQ_P_RR  : InstrItinClass;
def IIC_SSE_INTALUQ_P_RM  : InstrItinClass;

def IIC_SSE_INTMUL_P_RR : InstrItinClass;
def IIC_SSE_INTMUL_P_RM : InstrItinClass;

def IIC_SSE_INTSH_P_RR : InstrItinClass;
def IIC_SSE_INTSH_P_RM : InstrItinClass;
def IIC_SSE_INTSH_P_RI : InstrItinClass;

def IIC_SSE_INTSHDQ_P_RI : InstrItinClass;

def IIC_SSE_SHUFP : InstrItinClass;
def IIC_SSE_PSHUF_RI : InstrItinClass;
def IIC_SSE_PSHUF_MI : InstrItinClass;

def IIC_SSE_PACK : InstrItinClass;
def IIC_SSE_UNPCK : InstrItinClass;

def IIC_SSE_MOVMSK : InstrItinClass;
def IIC_SSE_MASKMOV : InstrItinClass;

def IIC_SSE_PEXTRW : InstrItinClass;
def IIC_SSE_PINSRW : InstrItinClass;

def IIC_SSE_PABS_RR : InstrItinClass;
def IIC_SSE_PABS_RM : InstrItinClass;

def IIC_SSE_SQRTPS_RR : InstrItinClass;
def IIC_SSE_SQRTPS_RM : InstrItinClass;
def IIC_SSE_SQRTSS_RR : InstrItinClass;
def IIC_SSE_SQRTSS_RM : InstrItinClass;
def IIC_SSE_SQRTPD_RR : InstrItinClass;
def IIC_SSE_SQRTPD_RM : InstrItinClass;
def IIC_SSE_SQRTSD_RR : InstrItinClass;
def IIC_SSE_SQRTSD_RM : InstrItinClass;

def IIC_SSE_RSQRTPS_RR : InstrItinClass;
def IIC_SSE_RSQRTPS_RM : InstrItinClass;
def IIC_SSE_RSQRTSS_RR : InstrItinClass;
def IIC_SSE_RSQRTSS_RM : InstrItinClass;

def IIC_SSE_RCPP_RR : InstrItinClass;
def IIC_SSE_RCPP_RM : InstrItinClass;
def IIC_SSE_RCPS_RR : InstrItinClass;
def IIC_SSE_RCPS_RM : InstrItinClass;

def IIC_SSE_MOV_S_RR : InstrItinClass;
def IIC_SSE_MOV_S_RM : InstrItinClass;
def IIC_SSE_MOV_S_MR : InstrItinClass;

def IIC_SSE_MOVA_P_RR : InstrItinClass;
def IIC_SSE_MOVA_P_RM : InstrItinClass;
def IIC_SSE_MOVA_P_MR : InstrItinClass;

def IIC_SSE_MOVU_P_RR : InstrItinClass;
def IIC_SSE_MOVU_P_RM : InstrItinClass;
def IIC_SSE_MOVU_P_MR : InstrItinClass;

def IIC_SSE_MOVDQ : InstrItinClass;
def IIC_SSE_MOVD_ToGP : InstrItinClass;
def IIC_SSE_MOVQ_RR : InstrItinClass;

def IIC_SSE_MOV_LH : InstrItinClass;

def IIC_SSE_LDDQU : InstrItinClass;

def IIC_SSE_MOVNT : InstrItinClass;

def IIC_SSE_PHADDSUBD_RR : InstrItinClass;
def IIC_SSE_PHADDSUBD_RM : InstrItinClass;
def IIC_SSE_PHADDSUBSW_RR : InstrItinClass;
def IIC_SSE_PHADDSUBSW_RM : InstrItinClass;
def IIC_SSE_PHADDSUBW_RR : InstrItinClass;
def IIC_SSE_PHADDSUBW_RM : InstrItinClass;
def IIC_SSE_PSHUFB_RR : InstrItinClass;
def IIC_SSE_PSHUFB_RM : InstrItinClass;
def IIC_SSE_PSIGN_RR : InstrItinClass;
def IIC_SSE_PSIGN_RM : InstrItinClass;

def IIC_SSE_PMADD : InstrItinClass;
def IIC_SSE_PMULHRSW : InstrItinClass;
def IIC_SSE_PALIGNRR : InstrItinClass;
def IIC_SSE_PALIGNRM : InstrItinClass;
def IIC_SSE_MWAIT : InstrItinClass;
def IIC_SSE_MONITOR : InstrItinClass;
def IIC_SSE_MWAITX : InstrItinClass;
def IIC_SSE_MONITORX : InstrItinClass;
def IIC_SSE_CLZERO : InstrItinClass;

def IIC_SSE_PREFETCH : InstrItinClass;
def IIC_SSE_PAUSE : InstrItinClass;
def IIC_SSE_LFENCE : InstrItinClass;
def IIC_SSE_MFENCE : InstrItinClass;
def IIC_SSE_SFENCE : InstrItinClass;
def IIC_SSE_LDMXCSR : InstrItinClass;
def IIC_SSE_STMXCSR : InstrItinClass;

def IIC_SSE_CVT_PD_RR : InstrItinClass;
def IIC_SSE_CVT_PD_RM : InstrItinClass;
def IIC_SSE_CVT_PS_RR : InstrItinClass;
def IIC_SSE_CVT_PS_RM : InstrItinClass;
def IIC_SSE_CVT_Scalar_RR : InstrItinClass;
def IIC_SSE_CVT_Scalar_RM : InstrItinClass;
def IIC_SSE_CVT_SS2SI32_RM : InstrItinClass;
def IIC_SSE_CVT_SS2SI32_RR : InstrItinClass;
def IIC_SSE_CVT_SS2SI64_RM : InstrItinClass;
def IIC_SSE_CVT_SS2SI64_RR : InstrItinClass;
def IIC_SSE_CVT_SD2SI_RM : InstrItinClass;
def IIC_SSE_CVT_SD2SI_RR : InstrItinClass;

def IIC_AVX_ZERO : InstrItinClass;

def IIC_CMPX_LOCK : InstrItinClass;
def IIC_CMPX_LOCK_8 : InstrItinClass;
def IIC_CMPX_LOCK_8B : InstrItinClass;
def IIC_CMPX_LOCK_16B : InstrItinClass;

def IIC_XADD_LOCK_MEM : InstrItinClass;
def IIC_XADD_LOCK_MEM8 : InstrItinClass;

// System instructions
def IIC_RDRAND : InstrItinClass;
def IIC_RDSEED : InstrItinClass;
def IIC_INS : InstrItinClass;
def IIC_LWP : InstrItinClass;
def IIC_ENTER : InstrItinClass;
def IIC_LEAVE : InstrItinClass;
def IIC_POP_MEM : InstrItinClass;
def IIC_POP_REG16 : InstrItinClass;
def IIC_POP_REG : InstrItinClass;
def IIC_POP_F : InstrItinClass;
def IIC_POP_FD : InstrItinClass;
def IIC_POP_A : InstrItinClass;
def IIC_PUSH_IMM : InstrItinClass;
def IIC_PUSH_MEM : InstrItinClass;
def IIC_PUSH_REG : InstrItinClass;
def IIC_PUSH_F : InstrItinClass;
def IIC_PUSH_A : InstrItinClass;
def IIC_BSWAP : InstrItinClass;
def IIC_BIT_SCAN_MEM : InstrItinClass;
def IIC_BIT_SCAN_REG : InstrItinClass;
def IIC_LZCNT_RR : InstrItinClass;
def IIC_LZCNT_RM : InstrItinClass;
def IIC_TZCNT_RR : InstrItinClass;
def IIC_TZCNT_RM : InstrItinClass;
def IIC_MOVS : InstrItinClass;
def IIC_STOS : InstrItinClass;
def IIC_SCAS : InstrItinClass;
def IIC_CMPS : InstrItinClass;
def IIC_MOV : InstrItinClass;
def IIC_MOV_MEM : InstrItinClass;
def IIC_AHF : InstrItinClass;
def IIC_BT_MI : InstrItinClass;
def IIC_BT_MR : InstrItinClass;
def IIC_BT_RI : InstrItinClass;
def IIC_BT_RR : InstrItinClass;
def IIC_BTX_MI : InstrItinClass;
def IIC_BTX_MR : InstrItinClass;
def IIC_BTX_RI : InstrItinClass;
def IIC_BTX_RR : InstrItinClass;
def IIC_XCHG_REG : InstrItinClass;
def IIC_XCHG_MEM : InstrItinClass;
def IIC_XADD_REG : InstrItinClass;
def IIC_XADD_MEM : InstrItinClass;
def IIC_CMPXCHG_MEM : InstrItinClass;
def IIC_CMPXCHG_REG : InstrItinClass;
def IIC_CMPXCHG_MEM8 : InstrItinClass;
def IIC_CMPXCHG_REG8 : InstrItinClass;
def IIC_CMPXCHG_8B : InstrItinClass;
def IIC_CMPXCHG_16B : InstrItinClass;
def IIC_LODS : InstrItinClass;
def IIC_OUTS : InstrItinClass;
def IIC_CLC_CMC_STC : InstrItinClass;
def IIC_CLD : InstrItinClass;
def IIC_STD : InstrItinClass;
def IIC_XLAT : InstrItinClass;
def IIC_AAA : InstrItinClass;
def IIC_AAD : InstrItinClass;
def IIC_AAM : InstrItinClass;
def IIC_AAS : InstrItinClass;
def IIC_DAA : InstrItinClass;
def IIC_DAS : InstrItinClass;
def IIC_BOUND : InstrItinClass;
def IIC_ARPL_REG : InstrItinClass;
def IIC_ARPL_MEM : InstrItinClass;
def IIC_MOVBE : InstrItinClass;
def IIC_AES   : InstrItinClass;
def IIC_BLEND_MEM : InstrItinClass;
def IIC_BLEND_NOMEM : InstrItinClass;
def IIC_CBW   : InstrItinClass;
def IIC_CRC32_REG : InstrItinClass;
def IIC_CRC32_MEM : InstrItinClass;
def IIC_SSE_DPPD_RR : InstrItinClass;
def IIC_SSE_DPPD_RM : InstrItinClass;
def IIC_SSE_DPPS_RR : InstrItinClass;
def IIC_SSE_DPPS_RM : InstrItinClass;
def IIC_SSE_EXTRACTPS_RR : InstrItinClass;
def IIC_SSE_EXTRACTPS_RM : InstrItinClass;
def IIC_SSE_INSERTPS_RR : InstrItinClass;
def IIC_SSE_INSERTPS_RM : InstrItinClass;
def IIC_SSE_MPSADBW_RR : InstrItinClass;
def IIC_SSE_MPSADBW_RM : InstrItinClass;
def IIC_SSE_PMULLD_RR : InstrItinClass;
def IIC_SSE_PMULLD_RM : InstrItinClass;
def IIC_SSE_ROUNDPS_REG : InstrItinClass;
def IIC_SSE_ROUNDPS_MEM : InstrItinClass;
def IIC_SSE_ROUNDPD_REG : InstrItinClass;
def IIC_SSE_ROUNDPD_MEM : InstrItinClass;
def IIC_SSE_POPCNT_RR : InstrItinClass;
def IIC_SSE_POPCNT_RM : InstrItinClass;
def IIC_SSE_PCLMULQDQ_RR : InstrItinClass;
def IIC_SSE_PCLMULQDQ_RM : InstrItinClass;

//===----------------------------------------------------------------------===//
// Processor instruction itineraries.

// IssueWidth is analogous to the number of decode units. Core and its
// descendents, including Nehalem and SandyBridge have 4 decoders.
// Resources beyond the decoder operate on micro-ops and are bufferred
// so adjacent micro-ops don't directly compete.
//
// MicroOpBufferSize > 1 indicates that RAW dependencies can be
// decoded in the same cycle. The value 32 is a reasonably arbitrary
// number of in-flight instructions.
//
// HighLatency=10 is optimistic. X86InstrInfo::isHighLatencyDef
// indicates high latency opcodes. Alternatively, InstrItinData
// entries may be included here to define specific operand
// latencies. Since these latencies are not used for pipeline hazards,
// they do not need to be exact.
//
// The GenericX86Model contains no instruction itineraries
// and disables PostRAScheduler.
class GenericX86Model : SchedMachineModel {
  let IssueWidth = 4;
  let MicroOpBufferSize = 32;
  let LoadLatency = 4;
  let HighLatency = 10;
  let PostRAScheduler = 0;
  let CompleteModel = 0;
}

def GenericModel : GenericX86Model;

// Define a model with the PostRAScheduler enabled.
def GenericPostRAModel : GenericX86Model {
  let PostRAScheduler = 1;
}

