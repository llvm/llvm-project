//===-- X86Schedule.td - X86 Scheduling Definitions --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// InstrSchedModel annotations for out-of-order CPUs.

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

// Multiclass that wraps X86FoldableSchedWrite for each vector width.
class X86SchedWriteWidths<X86FoldableSchedWrite sScl,
                          X86FoldableSchedWrite s128,
                          X86FoldableSchedWrite s256,
                          X86FoldableSchedWrite s512> {
  X86FoldableSchedWrite Scl = sScl; // Scalar float/double operations.
  X86FoldableSchedWrite MMX = sScl; // MMX operations.
  X86FoldableSchedWrite XMM = s128; // XMM operations.
  X86FoldableSchedWrite YMM = s256; // YMM operations.
  X86FoldableSchedWrite ZMM = s512; // ZMM operations.
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
defm WriteFAdd   : X86SchedWritePair; // Floating point add/sub.
defm WriteFAddY  : X86SchedWritePair; // Floating point add/sub (YMM/ZMM).
defm WriteFCmp   : X86SchedWritePair; // Floating point compare.
defm WriteFCmpY  : X86SchedWritePair; // Floating point compare (YMM/ZMM).
defm WriteFCom   : X86SchedWritePair; // Floating point compare to flags.
defm WriteFMul   : X86SchedWritePair; // Floating point multiplication.
defm WriteFMulY  : X86SchedWritePair; // Floating point multiplication (YMM/ZMM).
defm WriteFDiv   : X86SchedWritePair; // Floating point division.
defm WriteFDivY  : X86SchedWritePair; // Floating point division (YMM/ZMM).
defm WriteFSqrt  : X86SchedWritePair; // Floating point square root.
defm WriteFSqrtY : X86SchedWritePair; // Floating point square root (YMM/ZMM).
defm WriteFRcp   : X86SchedWritePair; // Floating point reciprocal estimate.
defm WriteFRcpY  : X86SchedWritePair; // Floating point reciprocal estimate (YMM/ZMM).
defm WriteFRsqrt : X86SchedWritePair; // Floating point reciprocal square root estimate.
defm WriteFRsqrtY: X86SchedWritePair; // Floating point reciprocal square root estimate (YMM/ZMM).
defm WriteFMA    : X86SchedWritePair; // Fused Multiply Add.
defm WriteFMAS   : X86SchedWritePair; // Fused Multiply Add (Scalar).
defm WriteFMAY   : X86SchedWritePair; // Fused Multiply Add (YMM/ZMM).
defm WriteFSign  : X86SchedWritePair; // Floating point fabs/fchs.
defm WriteFLogic  : X86SchedWritePair; // Floating point and/or/xor logicals.
defm WriteFLogicY : X86SchedWritePair; // Floating point and/or/xor logicals (YMM/ZMM).
defm WriteFShuffle  : X86SchedWritePair; // Floating point vector shuffles.
defm WriteFShuffleY : X86SchedWritePair; // Floating point vector shuffles (YMM/ZMM).
defm WriteFVarShuffle  : X86SchedWritePair; // Floating point vector variable shuffles.
defm WriteFVarShuffleY : X86SchedWritePair; // Floating point vector variable shuffles (YMM/ZMM).
defm WriteFBlend  : X86SchedWritePair; // Floating point vector blends.
defm WriteFBlendY : X86SchedWritePair; // Floating point vector blends (YMM/ZMM).
defm WriteFVarBlend  : X86SchedWritePair; // Fp vector variable blends.
defm WriteFVarBlendY : X86SchedWritePair; // Fp vector variable blends (YMM/ZMM).

// FMA Scheduling helper class.
class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Horizontal Add/Sub (float and integer)
defm WriteFHAdd  : X86SchedWritePair;
defm WriteFHAddY : X86SchedWritePair; // YMM/ZMM.
defm WritePHAdd  : X86SchedWritePair;

// Vector integer operations.
def  WriteVecLoad  : SchedWrite;
def  WriteVecStore : SchedWrite;
def  WriteVecMove  : SchedWrite;
defm WriteVecALU   : X86SchedWritePair; // Vector integer ALU op, no logicals.
defm WriteVecLogic : X86SchedWritePair; // Vector integer and/or/xor logicals.
defm WriteVecLogicY: X86SchedWritePair; // Vector integer and/or/xor logicals (YMM/ZMM).
defm WriteVecShift : X86SchedWritePair; // Vector integer shifts.
defm WriteVecIMul  : X86SchedWritePair; // Vector integer multiply.
defm WritePMULLD : X86SchedWritePair; // PMULLD
defm WriteShuffle  : X86SchedWritePair; // Vector shuffles.
defm WriteVarShuffle  : X86SchedWritePair; // Vector variable shuffles.
defm WriteBlend  : X86SchedWritePair; // Vector blends.
defm WriteVarBlend  : X86SchedWritePair; // Vector variable blends.
defm WritePSADBW : X86SchedWritePair; // Vector PSADBW.
defm WriteMPSAD : X86SchedWritePair; // Vector MPSAD.
defm WritePHMINPOS : X86SchedWritePair; // Vector PHMINPOS.

// Vector insert/extract operations.
defm WriteVecInsert : X86SchedWritePair; // Insert gpr to vector element.
def  WriteVecExtract : SchedWrite; // Extract vector element to gpr.
def  WriteVecExtractSt : SchedWrite; // Extract vector element and store.

// MOVMSK operations.
def WriteFMOVMSK : SchedWrite;
def WriteVecMOVMSK : SchedWrite;
def WriteMMXMOVMSK : SchedWrite;

// Conversion between integer and float.
defm WriteCvtF2I : X86SchedWritePair; // Float -> Integer.
defm WriteCvtI2F : X86SchedWritePair; // Integer -> Float.
defm WriteCvtF2F : X86SchedWritePair; // Float -> Float size conversion.
def  WriteCvtF2FSt : SchedWrite; // // Float -> Float + store size conversion.

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

// Load/store MXCSR
def WriteLDMXCSR : SchedWrite;
def WriteSTMXCSR : SchedWrite;

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

// Vector width wrappers.
def SchedWriteFAdd
 : X86SchedWriteWidths<WriteFAdd, WriteFAdd, WriteFAddY, WriteFAddY>;
def SchedWriteFCmp
 : X86SchedWriteWidths<WriteFCmp, WriteFCmp, WriteFCmpY, WriteFCmpY>;
def SchedWriteFMul
 : X86SchedWriteWidths<WriteFMul, WriteFMul, WriteFMulY, WriteFMulY>;
def SchedWriteFDiv
 : X86SchedWriteWidths<WriteFDiv, WriteFDiv, WriteFDivY, WriteFDivY>;
def SchedWriteFSqrt
 : X86SchedWriteWidths<WriteFSqrt, WriteFSqrt, WriteFSqrtY, WriteFSqrtY>;
def SchedWriteFRcp
 : X86SchedWriteWidths<WriteFRcp, WriteFRcp, WriteFRcpY, WriteFRcpY>;
def SchedWriteFRsqrt
 : X86SchedWriteWidths<WriteFRsqrt, WriteFRsqrt, WriteFRsqrtY, WriteFRsqrtY>;
def SchedWriteFLogic
 : X86SchedWriteWidths<WriteFLogic, WriteFLogic, WriteFLogicY, WriteFLogicY>;

def SchedWriteFShuffle
 : X86SchedWriteWidths<WriteFShuffle, WriteFShuffle,
                       WriteFShuffleY, WriteFShuffleY>;
def SchedWriteFVarShuffle
 : X86SchedWriteWidths<WriteFVarShuffle, WriteFVarShuffle,
                       WriteFVarShuffleY, WriteFVarShuffleY>;
def SchedWriteFBlend
 : X86SchedWriteWidths<WriteFBlend, WriteFBlend, WriteFBlendY, WriteFBlendY>;
def SchedWriteFVarBlend
 : X86SchedWriteWidths<WriteFVarBlend, WriteFVarBlend,
                       WriteFVarBlendY, WriteFVarBlendY>;

def SchedWriteVecALU
 : X86SchedWriteWidths<WriteVecALU, WriteVecALU, WriteVecALU, WriteVecALU>;
def SchedWriteVecLogic
 : X86SchedWriteWidths<WriteVecLogic, WriteVecLogic,
                       WriteVecLogicY, WriteVecLogicY>;
def SchedWriteVecShift
 : X86SchedWriteWidths<WriteVecShift, WriteVecShift,
                       WriteVecShift, WriteVecShift>;
def SchedWriteVecIMul
 : X86SchedWriteWidths<WriteVecIMul, WriteVecIMul,
                       WriteVecIMul, WriteVecIMul>;
def SchedWritePMULLD
 : X86SchedWriteWidths<WritePMULLD, WritePMULLD,
                       WritePMULLD, WritePMULLD>;

def SchedWriteShuffle
 : X86SchedWriteWidths<WriteShuffle, WriteShuffle,
                       WriteShuffle, WriteShuffle>;
def SchedWriteVarShuffle
 : X86SchedWriteWidths<WriteVarShuffle, WriteVarShuffle,
                       WriteVarShuffle, WriteVarShuffle>;
def SchedWriteBlend
 : X86SchedWriteWidths<WriteBlend, WriteBlend, WriteBlend, WriteBlend>;
def SchedWriteVarBlend
 : X86SchedWriteWidths<WriteVarBlend, WriteVarBlend,
                       WriteVarBlend, WriteVarBlend>;

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
// The GenericX86Model contains no instruction schedules
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

