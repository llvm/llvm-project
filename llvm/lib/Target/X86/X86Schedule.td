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

// Helpers to mark SchedWrites as unsupported.
multiclass X86WriteResUnsupported<SchedWrite SchedRW> {
  let Unsupported = 1 in {
    def : WriteRes<SchedRW, []>;
  }
}
multiclass X86WriteResPairUnsupported<X86FoldableSchedWrite SchedRW> {
  let Unsupported = 1 in {
    def : WriteRes<SchedRW, []>;
    def : WriteRes<SchedRW.Folded, []>;
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

// Multiclass that wraps X86SchedWriteWidths for each fp vector type.
class X86SchedWriteSizes<X86SchedWriteWidths sPS,
                         X86SchedWriteWidths sPD> {
  X86SchedWriteWidths PS = sPS;
  X86SchedWriteWidths PD = sPD;
}

// Multiclass that wraps move/load/store triple for a vector width.
class X86SchedWriteMoveLS<SchedWrite MoveRR,
                          SchedWrite LoadRM,
                          SchedWrite StoreMR> {
  SchedWrite RR = MoveRR;
  SchedWrite RM = LoadRM;
  SchedWrite MR = StoreMR;
}

// Multiclass that wraps X86SchedWriteMoveLS for each vector width.
class X86SchedWriteMoveLSWidths<X86SchedWriteMoveLS sScl,
                                X86SchedWriteMoveLS s128,
                                X86SchedWriteMoveLS s256,
                                X86SchedWriteMoveLS s512> {
  X86SchedWriteMoveLS Scl = sScl; // Scalar float/double operations.
  X86SchedWriteMoveLS MMX = sScl; // MMX operations.
  X86SchedWriteMoveLS XMM = s128; // XMM operations.
  X86SchedWriteMoveLS YMM = s256; // YMM operations.
  X86SchedWriteMoveLS ZMM = s512; // ZMM operations.
}

// Loads, stores, and moves, not folded with other operations.
def WriteLoad    : SchedWrite;
def WriteStore   : SchedWrite;
def WriteStoreNT : SchedWrite;
def WriteMove    : SchedWrite;

// Arithmetic.
defm WriteALU    : X86SchedWritePair; // Simple integer ALU op.
defm WriteADC    : X86SchedWritePair; // Integer ALU + flags op.
def  WriteALURMW : WriteSequence<[WriteALULd, WriteStore]>;
def  WriteADCRMW : WriteSequence<[WriteADCLd, WriteStore]>;
defm WriteIMul   : X86SchedWritePair; // Integer multiplication.
defm WriteIMul64 : X86SchedWritePair; // Integer 64-bit multiplication.
def  WriteIMulH  : SchedWrite;        // Integer multiplication, high part.
def  WriteLEA    : SchedWrite;        // LEA instructions can't fold loads.

def  WriteBSWAP32 : SchedWrite; // Byte Order (Endianness) 32-bit Swap.
def  WriteBSWAP64 : SchedWrite; // Byte Order (Endianness) 64-bit Swap.

// Integer division.
defm WriteDiv8   : X86SchedWritePair;
defm WriteDiv16  : X86SchedWritePair;
defm WriteDiv32  : X86SchedWritePair;
defm WriteDiv64  : X86SchedWritePair;
defm WriteIDiv8  : X86SchedWritePair;
defm WriteIDiv16 : X86SchedWritePair;
defm WriteIDiv32 : X86SchedWritePair;
defm WriteIDiv64 : X86SchedWritePair;

defm WriteBSF : X86SchedWritePair; // Bit scan forward.
defm WriteBSR : X86SchedWritePair; // Bit scan reverse.
defm WritePOPCNT : X86SchedWritePair; // Bit population count.
defm WriteLZCNT : X86SchedWritePair; // Leading zero count.
defm WriteTZCNT : X86SchedWritePair; // Trailing zero count.
defm WriteCMOV  : X86SchedWritePair; // Conditional move.
defm WriteCMOV2 : X86SchedWritePair; // Conditional (CF + ZF flag) move.
def  WriteFCMOV : SchedWrite; // X87 conditional move.
def  WriteSETCC : SchedWrite; // Set register based on condition code.
def  WriteSETCCStore : SchedWrite;
def  WriteLAHFSAHF : SchedWrite; // Load/Store flags in AH.
def  WriteBitTest  : SchedWrite; // Bit Test - TODO add memory folding support

// Integer shifts and rotates.
defm WriteShift : X86SchedWritePair;
// Double shift instructions.
def  WriteSHDrri  : SchedWrite;
def  WriteSHDrrcl : SchedWrite;
def  WriteSHDmri  : SchedWrite;
def  WriteSHDmrcl : SchedWrite;

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
def  WriteFLD0          : SchedWrite;
def  WriteFLD1          : SchedWrite;
def  WriteFLDC          : SchedWrite;
def  WriteFLoad         : SchedWrite;
def  WriteFLoadX        : SchedWrite;
def  WriteFLoadY        : SchedWrite;
def  WriteFMaskedLoad   : SchedWrite;
def  WriteFMaskedLoadY  : SchedWrite;
def  WriteFStore        : SchedWrite;
def  WriteFStoreX       : SchedWrite;
def  WriteFStoreY       : SchedWrite;
def  WriteFStoreNT      : SchedWrite;
def  WriteFStoreNTX     : SchedWrite;
def  WriteFStoreNTY     : SchedWrite;
def  WriteFMaskedStore  : SchedWrite;
def  WriteFMaskedStoreY : SchedWrite;
def  WriteFMove         : SchedWrite;
def  WriteFMoveX        : SchedWrite;
def  WriteFMoveY        : SchedWrite;

defm WriteFAdd    : X86SchedWritePair; // Floating point add/sub.
defm WriteFAddX   : X86SchedWritePair; // Floating point add/sub (XMM).
defm WriteFAddY   : X86SchedWritePair; // Floating point add/sub (YMM).
defm WriteFAddZ   : X86SchedWritePair; // Floating point add/sub (ZMM).
defm WriteFAdd64  : X86SchedWritePair; // Floating point double add/sub.
defm WriteFAdd64X : X86SchedWritePair; // Floating point double add/sub (XMM).
defm WriteFAdd64Y : X86SchedWritePair; // Floating point double add/sub (YMM).
defm WriteFAdd64Z : X86SchedWritePair; // Floating point double add/sub (ZMM).
defm WriteFCmp    : X86SchedWritePair; // Floating point compare.
defm WriteFCmpX   : X86SchedWritePair; // Floating point compare (XMM).
defm WriteFCmpY   : X86SchedWritePair; // Floating point compare (YMM).
defm WriteFCmpZ   : X86SchedWritePair; // Floating point compare (ZMM).
defm WriteFCmp64  : X86SchedWritePair; // Floating point double compare.
defm WriteFCmp64X : X86SchedWritePair; // Floating point double compare (XMM).
defm WriteFCmp64Y : X86SchedWritePair; // Floating point double compare (YMM).
defm WriteFCmp64Z : X86SchedWritePair; // Floating point double compare (ZMM).
defm WriteFCom    : X86SchedWritePair; // Floating point compare to flags.
defm WriteFMul    : X86SchedWritePair; // Floating point multiplication.
defm WriteFMulX   : X86SchedWritePair; // Floating point multiplication (XMM).
defm WriteFMulY   : X86SchedWritePair; // Floating point multiplication (YMM).
defm WriteFMulZ   : X86SchedWritePair; // Floating point multiplication (YMM).
defm WriteFMul64  : X86SchedWritePair; // Floating point double multiplication.
defm WriteFMul64X : X86SchedWritePair; // Floating point double multiplication (XMM).
defm WriteFMul64Y : X86SchedWritePair; // Floating point double multiplication (YMM).
defm WriteFMul64Z : X86SchedWritePair; // Floating point double multiplication (ZMM).
defm WriteFDiv    : X86SchedWritePair; // Floating point division.
defm WriteFDivX   : X86SchedWritePair; // Floating point division (XMM).
defm WriteFDivY   : X86SchedWritePair; // Floating point division (YMM).
defm WriteFDivZ   : X86SchedWritePair; // Floating point division (ZMM).
defm WriteFDiv64  : X86SchedWritePair; // Floating point double division.
defm WriteFDiv64X : X86SchedWritePair; // Floating point double division (XMM).
defm WriteFDiv64Y : X86SchedWritePair; // Floating point double division (YMM).
defm WriteFDiv64Z : X86SchedWritePair; // Floating point double division (ZMM).
defm WriteFSqrt  : X86SchedWritePair; // Floating point square root.
defm WriteFSqrtX : X86SchedWritePair; // Floating point square root (XMM).
defm WriteFSqrtY : X86SchedWritePair; // Floating point square root (YMM).
defm WriteFSqrtZ : X86SchedWritePair; // Floating point square root (ZMM).
defm WriteFSqrt64  : X86SchedWritePair; // Floating point double square root.
defm WriteFSqrt64X : X86SchedWritePair; // Floating point double square root (XMM).
defm WriteFSqrt64Y : X86SchedWritePair; // Floating point double square root (YMM).
defm WriteFSqrt64Z : X86SchedWritePair; // Floating point double square root (ZMM).
defm WriteFSqrt80  : X86SchedWritePair; // Floating point long double square root.
defm WriteFRcp   : X86SchedWritePair; // Floating point reciprocal estimate.
defm WriteFRcpX  : X86SchedWritePair; // Floating point reciprocal estimate (XMM).
defm WriteFRcpY  : X86SchedWritePair; // Floating point reciprocal estimate (YMM).
defm WriteFRcpZ  : X86SchedWritePair; // Floating point reciprocal estimate (ZMM).
defm WriteFRsqrt : X86SchedWritePair; // Floating point reciprocal square root estimate.
defm WriteFRsqrtX: X86SchedWritePair; // Floating point reciprocal square root estimate (XMM).
defm WriteFRsqrtY: X86SchedWritePair; // Floating point reciprocal square root estimate (YMM).
defm WriteFRsqrtZ: X86SchedWritePair; // Floating point reciprocal square root estimate (ZMM).
defm WriteFMA    : X86SchedWritePair; // Fused Multiply Add.
defm WriteFMAX   : X86SchedWritePair; // Fused Multiply Add (XMM).
defm WriteFMAY   : X86SchedWritePair; // Fused Multiply Add (YMM).
defm WriteFMAZ   : X86SchedWritePair; // Fused Multiply Add (ZMM).
defm WriteDPPD   : X86SchedWritePair; // Floating point double dot product.
defm WriteDPPS   : X86SchedWritePair; // Floating point single dot product.
defm WriteDPPSY  : X86SchedWritePair; // Floating point single dot product (YMM).
defm WriteDPPSZ  : X86SchedWritePair; // Floating point single dot product (ZMM).
defm WriteFSign  : X86SchedWritePair; // Floating point fabs/fchs.
defm WriteFRnd   : X86SchedWritePair; // Floating point rounding.
defm WriteFRndY  : X86SchedWritePair; // Floating point rounding (YMM).
defm WriteFRndZ  : X86SchedWritePair; // Floating point rounding (ZMM).
defm WriteFLogic  : X86SchedWritePair; // Floating point and/or/xor logicals.
defm WriteFLogicY : X86SchedWritePair; // Floating point and/or/xor logicals (YMM).
defm WriteFLogicZ : X86SchedWritePair; // Floating point and/or/xor logicals (ZMM).
defm WriteFTest   : X86SchedWritePair; // Floating point TEST instructions.
defm WriteFTestY  : X86SchedWritePair; // Floating point TEST instructions (YMM).
defm WriteFTestZ  : X86SchedWritePair; // Floating point TEST instructions (ZMM).
defm WriteFShuffle  : X86SchedWritePair; // Floating point vector shuffles.
defm WriteFShuffleY : X86SchedWritePair; // Floating point vector shuffles (YMM).
defm WriteFShuffleZ : X86SchedWritePair; // Floating point vector shuffles (ZMM).
defm WriteFVarShuffle  : X86SchedWritePair; // Floating point vector variable shuffles.
defm WriteFVarShuffleY : X86SchedWritePair; // Floating point vector variable shuffles (YMM).
defm WriteFVarShuffleZ : X86SchedWritePair; // Floating point vector variable shuffles (ZMM).
defm WriteFBlend  : X86SchedWritePair; // Floating point vector blends.
defm WriteFBlendY : X86SchedWritePair; // Floating point vector blends (YMM).
defm WriteFBlendZ : X86SchedWritePair; // Floating point vector blends (ZMM).
defm WriteFVarBlend  : X86SchedWritePair; // Fp vector variable blends.
defm WriteFVarBlendY : X86SchedWritePair; // Fp vector variable blends (YMM).
defm WriteFVarBlendZ : X86SchedWritePair; // Fp vector variable blends (YMZMM).

// FMA Scheduling helper class.
class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Horizontal Add/Sub (float and integer)
defm WriteFHAdd  : X86SchedWritePair;
defm WriteFHAddY : X86SchedWritePair;
defm WriteFHAddZ : X86SchedWritePair;
defm WritePHAdd  : X86SchedWritePair;
defm WritePHAddX : X86SchedWritePair;
defm WritePHAddY : X86SchedWritePair;
defm WritePHAddZ : X86SchedWritePair;

// Vector integer operations.
def  WriteVecLoad         : SchedWrite;
def  WriteVecLoadX        : SchedWrite;
def  WriteVecLoadY        : SchedWrite;
def  WriteVecLoadNT       : SchedWrite;
def  WriteVecLoadNTY      : SchedWrite;
def  WriteVecMaskedLoad   : SchedWrite;
def  WriteVecMaskedLoadY  : SchedWrite;
def  WriteVecStore        : SchedWrite;
def  WriteVecStoreX       : SchedWrite;
def  WriteVecStoreY       : SchedWrite;
def  WriteVecStoreNT      : SchedWrite;
def  WriteVecStoreNTY     : SchedWrite;
def  WriteVecMaskedStore  : SchedWrite;
def  WriteVecMaskedStoreY : SchedWrite;
def  WriteVecMove         : SchedWrite;
def  WriteVecMoveX        : SchedWrite;
def  WriteVecMoveY        : SchedWrite;
def  WriteVecMoveToGpr    : SchedWrite;
def  WriteVecMoveFromGpr  : SchedWrite;

defm WriteVecALU    : X86SchedWritePair; // Vector integer ALU op, no logicals.
defm WriteVecALUX   : X86SchedWritePair; // Vector integer ALU op, no logicals (XMM).
defm WriteVecALUY   : X86SchedWritePair; // Vector integer ALU op, no logicals (YMM).
defm WriteVecALUZ   : X86SchedWritePair; // Vector integer ALU op, no logicals (ZMM).
defm WriteVecLogic  : X86SchedWritePair; // Vector integer and/or/xor logicals.
defm WriteVecLogicX : X86SchedWritePair; // Vector integer and/or/xor logicals (XMM).
defm WriteVecLogicY : X86SchedWritePair; // Vector integer and/or/xor logicals (YMM).
defm WriteVecLogicZ : X86SchedWritePair; // Vector integer and/or/xor logicals (ZMM).
defm WriteVecTest  : X86SchedWritePair; // Vector integer TEST instructions.
defm WriteVecTestY : X86SchedWritePair; // Vector integer TEST instructions (YMM).
defm WriteVecTestZ : X86SchedWritePair; // Vector integer TEST instructions (ZMM).
defm WriteVecShift  : X86SchedWritePair; // Vector integer shifts (default).
defm WriteVecShiftX : X86SchedWritePair; // Vector integer shifts (XMM).
defm WriteVecShiftY : X86SchedWritePair; // Vector integer shifts (YMM).
defm WriteVecShiftZ : X86SchedWritePair; // Vector integer shifts (ZMM).
defm WriteVecShiftImm : X86SchedWritePair; // Vector integer immediate shifts (default).
defm WriteVecShiftImmX: X86SchedWritePair; // Vector integer immediate shifts (XMM).
defm WriteVecShiftImmY: X86SchedWritePair; // Vector integer immediate shifts (YMM).
defm WriteVecShiftImmZ: X86SchedWritePair; // Vector integer immediate shifts (ZMM).
defm WriteVecIMul  : X86SchedWritePair; // Vector integer multiply (default).
defm WriteVecIMulX : X86SchedWritePair; // Vector integer multiply (XMM).
defm WriteVecIMulY : X86SchedWritePair; // Vector integer multiply (YMM).
defm WriteVecIMulZ : X86SchedWritePair; // Vector integer multiply (ZMM).
defm WritePMULLD   : X86SchedWritePair; // Vector PMULLD.
defm WritePMULLDY   : X86SchedWritePair; // Vector PMULLD (YMM).
defm WritePMULLDZ   : X86SchedWritePair; // Vector PMULLD (ZMM).
defm WriteShuffle  : X86SchedWritePair; // Vector shuffles.
defm WriteShuffleX : X86SchedWritePair; // Vector shuffles (XMM).
defm WriteShuffleY : X86SchedWritePair; // Vector shuffles (YMM).
defm WriteShuffleZ : X86SchedWritePair; // Vector shuffles (ZMM).
defm WriteVarShuffle  : X86SchedWritePair; // Vector variable shuffles.
defm WriteVarShuffleX : X86SchedWritePair; // Vector variable shuffles (XMM).
defm WriteVarShuffleY : X86SchedWritePair; // Vector variable shuffles (YMM).
defm WriteVarShuffleZ : X86SchedWritePair; // Vector variable shuffles (ZMM).
defm WriteBlend  : X86SchedWritePair; // Vector blends.
defm WriteBlendY : X86SchedWritePair; // Vector blends (YMM).
defm WriteBlendZ : X86SchedWritePair; // Vector blends (ZMM).
defm WriteVarBlend  : X86SchedWritePair; // Vector variable blends.
defm WriteVarBlendY : X86SchedWritePair; // Vector variable blends (YMM).
defm WriteVarBlendZ : X86SchedWritePair; // Vector variable blends (ZMM).
defm WritePSADBW  : X86SchedWritePair; // Vector PSADBW.
defm WritePSADBWX : X86SchedWritePair; // Vector PSADBW (XMM).
defm WritePSADBWY : X86SchedWritePair; // Vector PSADBW (YMM).
defm WritePSADBWZ : X86SchedWritePair; // Vector PSADBW (ZMM).
defm WriteMPSAD  : X86SchedWritePair; // Vector MPSAD.
defm WriteMPSADY : X86SchedWritePair; // Vector MPSAD (YMM).
defm WriteMPSADZ : X86SchedWritePair; // Vector MPSAD (ZMM).
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
defm WriteCvtSD2I  : X86SchedWritePair; // Double -> Integer.
defm WriteCvtPD2I  : X86SchedWritePair; // Double -> Integer (XMM).
defm WriteCvtPD2IY : X86SchedWritePair; // Double -> Integer (YMM).
defm WriteCvtPD2IZ : X86SchedWritePair; // Double -> Integer (ZMM).

defm WriteCvtSS2I  : X86SchedWritePair; // Float -> Integer.
defm WriteCvtPS2I  : X86SchedWritePair; // Float -> Integer (XMM).
defm WriteCvtPS2IY : X86SchedWritePair; // Float -> Integer (YMM).
defm WriteCvtPS2IZ : X86SchedWritePair; // Float -> Integer (ZMM).

defm WriteCvtI2SD  : X86SchedWritePair; // Integer -> Double.
defm WriteCvtI2PD  : X86SchedWritePair; // Integer -> Double (XMM).
defm WriteCvtI2PDY : X86SchedWritePair; // Integer -> Double (YMM).
defm WriteCvtI2PDZ : X86SchedWritePair; // Integer -> Double (ZMM).

defm WriteCvtI2SS  : X86SchedWritePair; // Integer -> Float.
defm WriteCvtI2PS  : X86SchedWritePair; // Integer -> Float (XMM).
defm WriteCvtI2PSY : X86SchedWritePair; // Integer -> Float (YMM).
defm WriteCvtI2PSZ : X86SchedWritePair; // Integer -> Float (ZMM).

defm WriteCvtSS2SD  : X86SchedWritePair; // Float -> Double size conversion.
defm WriteCvtPS2PD  : X86SchedWritePair; // Float -> Double size conversion (XMM).
defm WriteCvtPS2PDY : X86SchedWritePair; // Float -> Double size conversion (YMM).
defm WriteCvtPS2PDZ : X86SchedWritePair; // Float -> Double size conversion (ZMM).

defm WriteCvtSD2SS  : X86SchedWritePair; // Double -> Float size conversion.
defm WriteCvtPD2PS  : X86SchedWritePair; // Double -> Float size conversion (XMM).
defm WriteCvtPD2PSY : X86SchedWritePair; // Double -> Float size conversion (YMM).
defm WriteCvtPD2PSZ : X86SchedWritePair; // Double -> Float size conversion (ZMM).

defm WriteCvtPH2PS    : X86SchedWritePair; // Half -> Float size conversion.
defm WriteCvtPH2PSY   : X86SchedWritePair; // Half -> Float size conversion (YMM).
defm WriteCvtPH2PSZ   : X86SchedWritePair; // Half -> Float size conversion (ZMM).

def  WriteCvtPS2PH    : SchedWrite; // // Float -> Half size conversion.
def  WriteCvtPS2PHY   : SchedWrite; // // Float -> Half size conversion (YMM).
def  WriteCvtPS2PHZ   : SchedWrite; // // Float -> Half size conversion (ZMM).
def  WriteCvtPS2PHSt  : SchedWrite; // // Float -> Half + store size conversion.
def  WriteCvtPS2PHYSt : SchedWrite; // // Float -> Half + store size conversion (YMM).
def  WriteCvtPS2PHZSt : SchedWrite; // // Float -> Half + store size conversion (ZMM).

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
defm WriteVarVecShiftY : X86SchedWritePair; // Variable vector shifts (YMM).
defm WriteVarVecShiftZ : X86SchedWritePair; // Variable vector shifts (ZMM).

// Old microcoded instructions that nobody use.
def WriteMicrocoded : SchedWrite;

// Fence instructions.
def WriteFence : SchedWrite;

// Nop, not very useful expect it provides a model for nops!
def WriteNop : SchedWrite;

// Move/Load/Store wrappers.
def WriteFMoveLS
 : X86SchedWriteMoveLS<WriteFMove, WriteFLoad, WriteFStore>;
def WriteFMoveLSX
 : X86SchedWriteMoveLS<WriteFMoveX, WriteFLoadX, WriteFStoreX>;
def WriteFMoveLSY
 : X86SchedWriteMoveLS<WriteFMoveY, WriteFLoadY, WriteFStoreY>;
def SchedWriteFMoveLS
  : X86SchedWriteMoveLSWidths<WriteFMoveLS, WriteFMoveLSX,
                              WriteFMoveLSY, WriteFMoveLSY>;

def WriteFMoveLSNT
 : X86SchedWriteMoveLS<WriteFMove, WriteFLoad, WriteFStoreNT>;
def WriteFMoveLSNTX
 : X86SchedWriteMoveLS<WriteFMove, WriteFLoad, WriteFStoreNTX>;
def WriteFMoveLSNTY
 : X86SchedWriteMoveLS<WriteFMoveY, WriteFLoadY, WriteFStoreNTY>;
def SchedWriteFMoveLSNT
  : X86SchedWriteMoveLSWidths<WriteFMoveLSNT, WriteFMoveLSNTX,
                              WriteFMoveLSNTY, WriteFMoveLSNTY>;

def WriteVecMoveLS
 : X86SchedWriteMoveLS<WriteVecMove, WriteVecLoad, WriteVecStore>;
def WriteVecMoveLSX
 : X86SchedWriteMoveLS<WriteVecMoveX, WriteVecLoadX, WriteVecStoreX>;
def WriteVecMoveLSY
 : X86SchedWriteMoveLS<WriteVecMoveY, WriteVecLoadY, WriteVecStoreY>;
def SchedWriteVecMoveLS
  : X86SchedWriteMoveLSWidths<WriteVecMoveLS, WriteVecMoveLSX,
                              WriteVecMoveLSY, WriteVecMoveLSY>;

def WriteVecMoveLSNT
 : X86SchedWriteMoveLS<WriteVecMove, WriteVecLoadNT, WriteVecStoreNT>;
def WriteVecMoveLSNTX
 : X86SchedWriteMoveLS<WriteVecMoveX, WriteVecLoadNT, WriteVecStoreNT>;
def WriteVecMoveLSNTY
 : X86SchedWriteMoveLS<WriteVecMoveY, WriteVecLoadNTY, WriteVecStoreNTY>;
def SchedWriteVecMoveLSNT
  : X86SchedWriteMoveLSWidths<WriteVecMoveLSNT, WriteVecMoveLSNTX,
                              WriteVecMoveLSNTY, WriteVecMoveLSNTY>;

// Vector width wrappers.
def SchedWriteFAdd
 : X86SchedWriteWidths<WriteFAdd, WriteFAddX, WriteFAddY, WriteFAddZ>;
def SchedWriteFAdd64
 : X86SchedWriteWidths<WriteFAdd64, WriteFAdd64X, WriteFAdd64Y, WriteFAdd64Z>;
def SchedWriteFHAdd
 : X86SchedWriteWidths<WriteFHAdd, WriteFHAdd, WriteFHAddY, WriteFHAddZ>;
def SchedWriteFCmp
 : X86SchedWriteWidths<WriteFCmp, WriteFCmpX, WriteFCmpY, WriteFCmpZ>;
def SchedWriteFCmp64
 : X86SchedWriteWidths<WriteFCmp64, WriteFCmp64X, WriteFCmp64Y, WriteFCmp64Z>;
def SchedWriteFMul
 : X86SchedWriteWidths<WriteFMul, WriteFMulX, WriteFMulY, WriteFMulZ>;
def SchedWriteFMul64
 : X86SchedWriteWidths<WriteFMul64, WriteFMul64X, WriteFMul64Y, WriteFMul64Z>;
def SchedWriteFMA
 : X86SchedWriteWidths<WriteFMA, WriteFMAX, WriteFMAY, WriteFMAZ>;
def SchedWriteDPPD
 : X86SchedWriteWidths<WriteDPPD, WriteDPPD, WriteDPPD, WriteDPPD>;
def SchedWriteDPPS
 : X86SchedWriteWidths<WriteDPPS, WriteDPPS, WriteDPPSY, WriteDPPSZ>;
def SchedWriteFDiv
 : X86SchedWriteWidths<WriteFDiv, WriteFDivX, WriteFDivY, WriteFDivZ>;
def SchedWriteFDiv64
 : X86SchedWriteWidths<WriteFDiv64, WriteFDiv64X, WriteFDiv64Y, WriteFDiv64Z>;
def SchedWriteFSqrt
 : X86SchedWriteWidths<WriteFSqrt, WriteFSqrtX,
                       WriteFSqrtY, WriteFSqrtZ>;
def SchedWriteFSqrt64
 : X86SchedWriteWidths<WriteFSqrt64, WriteFSqrt64X,
                       WriteFSqrt64Y, WriteFSqrt64Z>;
def SchedWriteFRcp
 : X86SchedWriteWidths<WriteFRcp, WriteFRcpX, WriteFRcpY, WriteFRcpZ>;
def SchedWriteFRsqrt
 : X86SchedWriteWidths<WriteFRsqrt, WriteFRsqrtX, WriteFRsqrtY, WriteFRsqrtZ>;
def SchedWriteFRnd
 : X86SchedWriteWidths<WriteFRnd, WriteFRnd, WriteFRndY, WriteFRndZ>;
def SchedWriteFLogic
 : X86SchedWriteWidths<WriteFLogic, WriteFLogic, WriteFLogicY, WriteFLogicZ>;
def SchedWriteFTest
 : X86SchedWriteWidths<WriteFTest, WriteFTest, WriteFTestY, WriteFTestZ>;

def SchedWriteFShuffle
 : X86SchedWriteWidths<WriteFShuffle, WriteFShuffle,
                       WriteFShuffleY, WriteFShuffleZ>;
def SchedWriteFVarShuffle
 : X86SchedWriteWidths<WriteFVarShuffle, WriteFVarShuffle,
                       WriteFVarShuffleY, WriteFVarShuffleZ>;
def SchedWriteFBlend
 : X86SchedWriteWidths<WriteFBlend, WriteFBlend, WriteFBlendY, WriteFBlendZ>;
def SchedWriteFVarBlend
 : X86SchedWriteWidths<WriteFVarBlend, WriteFVarBlend,
                       WriteFVarBlendY, WriteFVarBlendZ>;

def SchedWriteCvtDQ2PD
 : X86SchedWriteWidths<WriteCvtI2SD, WriteCvtI2PD,
                       WriteCvtI2PDY, WriteCvtI2PDZ>;
def SchedWriteCvtDQ2PS
 : X86SchedWriteWidths<WriteCvtI2SS, WriteCvtI2PS,
                       WriteCvtI2PSY, WriteCvtI2PSZ>;
def SchedWriteCvtPD2DQ
 : X86SchedWriteWidths<WriteCvtSD2I, WriteCvtPD2I,
                       WriteCvtPD2IY, WriteCvtPD2IZ>;
def SchedWriteCvtPS2DQ
 : X86SchedWriteWidths<WriteCvtSS2I, WriteCvtPS2I,
                       WriteCvtPS2IY, WriteCvtPS2IZ>;
def SchedWriteCvtPS2PD
 : X86SchedWriteWidths<WriteCvtSS2SD, WriteCvtPS2PD,
                       WriteCvtPS2PDY, WriteCvtPS2PDZ>;
def SchedWriteCvtPD2PS
 : X86SchedWriteWidths<WriteCvtSD2SS, WriteCvtPD2PS,
                       WriteCvtPD2PSY, WriteCvtPD2PSZ>;

def SchedWriteVecALU
 : X86SchedWriteWidths<WriteVecALU, WriteVecALUX, WriteVecALUY, WriteVecALUZ>;
def SchedWritePHAdd
 : X86SchedWriteWidths<WritePHAdd, WritePHAddX, WritePHAddY, WritePHAddZ>;
def SchedWriteVecLogic
 : X86SchedWriteWidths<WriteVecLogic, WriteVecLogicX,
                       WriteVecLogicY, WriteVecLogicZ>;
def SchedWriteVecTest
 : X86SchedWriteWidths<WriteVecTest, WriteVecTest,
                       WriteVecTestY, WriteVecTestZ>;
def SchedWriteVecShift
 : X86SchedWriteWidths<WriteVecShift, WriteVecShiftX,
                       WriteVecShiftY, WriteVecShiftZ>;
def SchedWriteVecShiftImm
 : X86SchedWriteWidths<WriteVecShiftImm, WriteVecShiftImmX,
                       WriteVecShiftImmY, WriteVecShiftImmZ>;
def SchedWriteVarVecShift
 : X86SchedWriteWidths<WriteVarVecShift, WriteVarVecShift,
                       WriteVarVecShiftY, WriteVarVecShiftZ>;
def SchedWriteVecIMul
 : X86SchedWriteWidths<WriteVecIMul, WriteVecIMulX,
                       WriteVecIMulY, WriteVecIMulZ>;
def SchedWritePMULLD
 : X86SchedWriteWidths<WritePMULLD, WritePMULLD,
                       WritePMULLDY, WritePMULLDZ>;
def SchedWriteMPSAD
 : X86SchedWriteWidths<WriteMPSAD, WriteMPSAD,
                       WriteMPSADY, WriteMPSADZ>;
def SchedWritePSADBW
 : X86SchedWriteWidths<WritePSADBW, WritePSADBWX,
                       WritePSADBWY, WritePSADBWZ>;

def SchedWriteShuffle
 : X86SchedWriteWidths<WriteShuffle, WriteShuffleX,
                       WriteShuffleY, WriteShuffleZ>;
def SchedWriteVarShuffle
 : X86SchedWriteWidths<WriteVarShuffle, WriteVarShuffleX,
                       WriteVarShuffleY, WriteVarShuffleZ>;
def SchedWriteBlend
 : X86SchedWriteWidths<WriteBlend, WriteBlend, WriteBlendY, WriteBlendZ>;
def SchedWriteVarBlend
 : X86SchedWriteWidths<WriteVarBlend, WriteVarBlend,
                       WriteVarBlendY, WriteVarBlendZ>;

// Vector size wrappers.
def SchedWriteFAddSizes
 : X86SchedWriteSizes<SchedWriteFAdd, SchedWriteFAdd64>;
def SchedWriteFCmpSizes
 : X86SchedWriteSizes<SchedWriteFCmp, SchedWriteFCmp64>;
def SchedWriteFMulSizes
 : X86SchedWriteSizes<SchedWriteFMul, SchedWriteFMul64>;
def SchedWriteFDivSizes
 : X86SchedWriteSizes<SchedWriteFDiv, SchedWriteFDiv64>;
def SchedWriteFSqrtSizes
 : X86SchedWriteSizes<SchedWriteFSqrt, SchedWriteFSqrt64>;
def SchedWriteFLogicSizes
 : X86SchedWriteSizes<SchedWriteFLogic, SchedWriteFLogic>;
def SchedWriteFShuffleSizes
 : X86SchedWriteSizes<SchedWriteFShuffle, SchedWriteFShuffle>;

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
