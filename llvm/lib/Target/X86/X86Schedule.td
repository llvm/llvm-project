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

def IIC_SSE_MOV_LH : InstrItinClass;

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

//===----------------------------------------------------------------------===//
// Generic Processor Scheduler Models.

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

