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

// Helper to set SchedWrite ExePorts/Latency/ResourceCycles/NumMicroOps.
multiclass X86WriteRes<SchedWrite SchedRW,
                       list<ProcResourceKind> ExePorts,
                       int Lat, list<int> Res, int UOps> {
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }
}

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
defm WriteFMAX   : X86SchedWritePair; // Fused Multiply Add (XMM).
defm WriteFMAY   : X86SchedWritePair; // Fused Multiply Add (YMM/ZMM).
defm WriteDPPD   : X86SchedWritePair; // Floating point double dot product.
defm WriteDPPS   : X86SchedWritePair; // Floating point single dot product.
defm WriteDPPSY  : X86SchedWritePair; // Floating point single dot product (YMM).
defm WriteFSign  : X86SchedWritePair; // Floating point fabs/fchs.
defm WriteFRnd   : X86SchedWritePair; // Floating point rounding.
defm WriteFRndY  : X86SchedWritePair; // Floating point rounding (YMM/ZMM).
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
defm WritePHAddY : X86SchedWritePair; // YMM/ZMM.

// Vector integer operations.
def  WriteVecLoad  : SchedWrite;
def  WriteVecStore : SchedWrite;
def  WriteVecMove  : SchedWrite;
defm WriteVecALU   : X86SchedWritePair; // Vector integer ALU op, no logicals.
defm WriteVecALUY  : X86SchedWritePair; // Vector integer ALU op, no logicals (YMM/ZMM).
defm WriteVecLogic : X86SchedWritePair; // Vector integer and/or/xor logicals.
defm WriteVecLogicY: X86SchedWritePair; // Vector integer and/or/xor logicals (YMM/ZMM).
defm WriteVecShift  : X86SchedWritePair; // Vector integer shifts (default).
defm WriteVecShiftX : X86SchedWritePair; // Vector integer shifts (XMM).
defm WriteVecShiftY : X86SchedWritePair; // Vector integer shifts (YMM/ZMM).
defm WriteVecShiftImm : X86SchedWritePair; // Vector integer immediate shifts (default).
defm WriteVecShiftImmX: X86SchedWritePair; // Vector integer immediate shifts (XMM).
defm WriteVecShiftImmY: X86SchedWritePair; // Vector integer immediate shifts (YMM/ZMM).
defm WriteVecIMul  : X86SchedWritePair; // Vector integer multiply (default).
defm WriteVecIMulX : X86SchedWritePair; // Vector integer multiply (XMM).
defm WriteVecIMulY : X86SchedWritePair; // Vector integer multiply (YMM/ZMM).
defm WritePMULLD   : X86SchedWritePair; // Vector PMULLD.
defm WritePMULLDY   : X86SchedWritePair; // Vector PMULLD (YMM/ZMM).
defm WriteShuffle  : X86SchedWritePair; // Vector shuffles.
defm WriteShuffleY : X86SchedWritePair; // Vector shuffles (YMM/ZMM).
defm WriteVarShuffle  : X86SchedWritePair; // Vector variable shuffles.
defm WriteVarShuffleY : X86SchedWritePair; // Vector variable shuffles (YMM/ZMM).
defm WriteBlend  : X86SchedWritePair; // Vector blends.
defm WriteBlendY : X86SchedWritePair; // Vector blends (YMM/ZMM).
defm WriteVarBlend  : X86SchedWritePair; // Vector variable blends.
defm WriteVarBlendY : X86SchedWritePair; // Vector variable blends (YMM/ZMM).
defm WritePSADBW  : X86SchedWritePair; // Vector PSADBW.
defm WritePSADBWY : X86SchedWritePair; // Vector PSADBW (YMM/ZMM).
defm WriteMPSAD  : X86SchedWritePair; // Vector MPSAD.
defm WriteMPSADY : X86SchedWritePair; // Vector MPSAD (YMM/ZMM).
defm WritePHMINPOS : X86SchedWritePair; // Vector PHMINPOS.

// Vector insert/extract operations.
defm WriteVecInsert : X86SchedWritePair; // Insert gpr to vector element.
def  WriteVecExtract : SchedWrite; // Extract vector element to gpr.
def  WriteVecExtractSt : SchedWrite; // Extract vector element and store.

// MOVMSK operations.
def WriteFMOVMSK    : SchedWrite;
def WriteVecMOVMSK  : SchedWrite;
def WriteVecMOVMSKY : SchedWrite;
def WriteMMXMOVMSK  : SchedWrite;

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

// EMMS/FEMMS
def WriteEMMS : SchedWrite;

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
defm WriteVarVecShift  : X86SchedWritePair; // Variable vector shifts.
defm WriteVarVecShiftY : X86SchedWritePair; // Variable vector shifts (YMM/ZMM).

// Old microcoded instructions that nobody use.
def WriteMicrocoded : SchedWrite;

// Fence instructions.
def WriteFence : SchedWrite;

// Nop, not very useful expect it provides a model for nops!
def WriteNop : SchedWrite;

// Vector width wrappers.
def SchedWriteFAdd
 : X86SchedWriteWidths<WriteFAdd, WriteFAdd, WriteFAddY, WriteFAddY>;
def SchedWriteFHAdd
 : X86SchedWriteWidths<WriteFHAdd, WriteFHAdd, WriteFHAddY, WriteFHAddY>;
def SchedWriteFCmp
 : X86SchedWriteWidths<WriteFCmp, WriteFCmp, WriteFCmpY, WriteFCmpY>;
def SchedWriteFMul
 : X86SchedWriteWidths<WriteFMul, WriteFMul, WriteFMulY, WriteFMulY>;
def SchedWriteFMA
 : X86SchedWriteWidths<WriteFMA, WriteFMAX, WriteFMAY, WriteFMAY>;
def SchedWriteDPPD
 : X86SchedWriteWidths<WriteDPPD, WriteDPPD, WriteDPPD, WriteDPPD>;
def SchedWriteDPPS
 : X86SchedWriteWidths<WriteDPPS, WriteDPPS, WriteDPPSY, WriteDPPSY>;
def SchedWriteFDiv
 : X86SchedWriteWidths<WriteFDiv, WriteFDiv, WriteFDivY, WriteFDivY>;
def SchedWriteFSqrt
 : X86SchedWriteWidths<WriteFSqrt, WriteFSqrt, WriteFSqrtY, WriteFSqrtY>;
def SchedWriteFRcp
 : X86SchedWriteWidths<WriteFRcp, WriteFRcp, WriteFRcpY, WriteFRcpY>;
def SchedWriteFRsqrt
 : X86SchedWriteWidths<WriteFRsqrt, WriteFRsqrt, WriteFRsqrtY, WriteFRsqrtY>;
def SchedWriteFRnd
 : X86SchedWriteWidths<WriteFRnd, WriteFRnd, WriteFRndY, WriteFRndY>;
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
 : X86SchedWriteWidths<WriteVecALU, WriteVecALU, WriteVecALUY, WriteVecALUY>;
def SchedWritePHAdd
 : X86SchedWriteWidths<WritePHAdd, WritePHAdd, WritePHAddY, WritePHAddY>;
def SchedWriteVecLogic
 : X86SchedWriteWidths<WriteVecLogic, WriteVecLogic,
                       WriteVecLogicY, WriteVecLogicY>;
def SchedWriteVecShift
 : X86SchedWriteWidths<WriteVecShift, WriteVecShiftX,
                       WriteVecShiftY, WriteVecShiftY>;
def SchedWriteVecShiftImm
 : X86SchedWriteWidths<WriteVecShiftImm, WriteVecShiftImmX,
                       WriteVecShiftImmY, WriteVecShiftImmY>;
def SchedWriteVarVecShift
 : X86SchedWriteWidths<WriteVarVecShift, WriteVarVecShift,
                       WriteVarVecShiftY, WriteVarVecShiftY>;
def SchedWriteVecIMul
 : X86SchedWriteWidths<WriteVecIMul, WriteVecIMulX,
                       WriteVecIMulY, WriteVecIMulY>;
def SchedWritePMULLD
 : X86SchedWriteWidths<WritePMULLD, WritePMULLD,
                       WritePMULLDY, WritePMULLDY>;
def SchedWriteMPSAD
 : X86SchedWriteWidths<WriteMPSAD, WriteMPSAD,
                       WriteMPSADY, WriteMPSADY>;
def SchedWritePSADBW
 : X86SchedWriteWidths<WritePSADBW, WritePSADBW,
                       WritePSADBWY, WritePSADBWY>;

def SchedWriteShuffle
 : X86SchedWriteWidths<WriteShuffle, WriteShuffle,
                       WriteShuffleY, WriteShuffleY>;
def SchedWriteVarShuffle
 : X86SchedWriteWidths<WriteVarShuffle, WriteVarShuffle,
                       WriteVarShuffleY, WriteVarShuffleY>;
def SchedWriteBlend
 : X86SchedWriteWidths<WriteBlend, WriteBlend, WriteBlendY, WriteBlendY>;
def SchedWriteVarBlend
 : X86SchedWriteWidths<WriteVarBlend, WriteVarBlend,
                       WriteVarBlendY, WriteVarBlendY>;

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

