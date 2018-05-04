//=- X86SchedSkylake.td - X86 Skylake Client Scheduling ------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Skylake Client to support
// instruction scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def SkylakeClientModel : SchedMachineModel {
  // All x86 instructions are modeled as a single micro-op, and SKylake can
  // decode 6 instructions per cycle.
  let IssueWidth = 6;
  let MicroOpBufferSize = 224; // Based on the reorder buffer.
  let LoadLatency = 5;
  let MispredictPenalty = 14;

  // Based on the LSD (loop-stream detector) queue size and benchmarking data.
  let LoopMicroOpBufferSize = 50;

  // This flag is set to allow the scheduler to assign a default model to
  // unrecognized opcodes.
  let CompleteModel = 0;
}

let SchedModel = SkylakeClientModel in {

// Skylake Client can issue micro-ops to 8 different ports in one cycle.

// Ports 0, 1, 5, and 6 handle all computation.
// Port 4 gets the data half of stores. Store data can be available later than
// the store address, but since we don't model the latency of stores, we can
// ignore that.
// Ports 2 and 3 are identical. They handle loads and the address half of
// stores. Port 7 can handle address calculations.
def SKLPort0 : ProcResource<1>;
def SKLPort1 : ProcResource<1>;
def SKLPort2 : ProcResource<1>;
def SKLPort3 : ProcResource<1>;
def SKLPort4 : ProcResource<1>;
def SKLPort5 : ProcResource<1>;
def SKLPort6 : ProcResource<1>;
def SKLPort7 : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def SKLPort01  : ProcResGroup<[SKLPort0, SKLPort1]>;
def SKLPort23  : ProcResGroup<[SKLPort2, SKLPort3]>;
def SKLPort237 : ProcResGroup<[SKLPort2, SKLPort3, SKLPort7]>;
def SKLPort04  : ProcResGroup<[SKLPort0, SKLPort4]>;
def SKLPort05  : ProcResGroup<[SKLPort0, SKLPort5]>;
def SKLPort06  : ProcResGroup<[SKLPort0, SKLPort6]>;
def SKLPort15  : ProcResGroup<[SKLPort1, SKLPort5]>;
def SKLPort16  : ProcResGroup<[SKLPort1, SKLPort6]>;
def SKLPort56  : ProcResGroup<[SKLPort5, SKLPort6]>;
def SKLPort015 : ProcResGroup<[SKLPort0, SKLPort1, SKLPort5]>;
def SKLPort056 : ProcResGroup<[SKLPort0, SKLPort5, SKLPort6]>;
def SKLPort0156: ProcResGroup<[SKLPort0, SKLPort1, SKLPort5, SKLPort6]>;

def SKLDivider : ProcResource<1>; // Integer division issued on port 0.
// FP division and sqrt on port 0.
def SKLFPDivider : ProcResource<1>;

// 60 Entry Unified Scheduler
def SKLPortAny : ProcResGroup<[SKLPort0, SKLPort1, SKLPort2, SKLPort3, SKLPort4,
                              SKLPort5, SKLPort6, SKLPort7]> {
  let BufferSize=60;
}

// Loads are 5 cycles, so ReadAfterLd registers needn't be available until 5
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 5>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass SKLWriteResPair<X86FoldableSchedWrite SchedRW,
                          list<ProcResourceKind> ExePorts,
                          int Lat, list<int> Res = [1], int UOps = 1,
                          int LoadLat = 5> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on port 2/3 and adds LoadLat cycles to
  // the latency (default = 5).
  def : WriteRes<SchedRW.Folded, !listconcat([SKLPort23], ExePorts)> {
    let Latency = !add(Lat, LoadLat);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = !add(UOps, 1);
  }
}

// A folded store needs a cycle on port 4 for the store data, and an extra port
// 2/3/7 cycle to recompute the address.
def : WriteRes<WriteRMW, [SKLPort237,SKLPort4]>;

// Arithmetic.
defm : SKLWriteResPair<WriteALU,   [SKLPort0156], 1>; // Simple integer ALU op.
defm : SKLWriteResPair<WriteIMul,  [SKLPort1],   3>; // Integer multiplication.
defm : SKLWriteResPair<WriteIDiv, [SKLPort0, SKLDivider], 25, [1,10], 1, 4>; // Integer division.
defm : SKLWriteResPair<WriteCRC32, [SKLPort1], 3>;

def : WriteRes<WriteIMulH, []> { let Latency = 3; } // Integer multiplication, high part.
def : WriteRes<WriteLEA, [SKLPort15]>; // LEA instructions can't fold loads.

defm : SKLWriteResPair<WriteCMOV,  [SKLPort06], 1>; // Conditional move.
def  : WriteRes<WriteSETCC, [SKLPort06]>; // Setcc.
def  : WriteRes<WriteSETCCStore, [SKLPort06,SKLPort4,SKLPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

// Bit counts.
defm : SKLWriteResPair<WriteBitScan, [SKLPort1], 3>;
defm : SKLWriteResPair<WriteLZCNT,   [SKLPort1], 3>;
defm : SKLWriteResPair<WriteTZCNT,   [SKLPort1], 3>;
defm : SKLWriteResPair<WritePOPCNT,  [SKLPort1], 3>;

// Integer shifts and rotates.
defm : SKLWriteResPair<WriteShift, [SKLPort06],  1>;

// BMI1 BEXTR, BMI2 BZHI
defm : SKLWriteResPair<WriteBEXTR, [SKLPort06,SKLPort15], 2, [1,1], 2>;
defm : SKLWriteResPair<WriteBZHI, [SKLPort15], 1>;

// Loads, stores, and moves, not folded with other operations.
def : WriteRes<WriteLoad,  [SKLPort23]> { let Latency = 5; }
def : WriteRes<WriteStore, [SKLPort237, SKLPort4]>;
def : WriteRes<WriteMove,  [SKLPort0156]>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
def : WriteRes<WriteZero,  []>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : SKLWriteResPair<WriteJump,  [SKLPort06],   1>;

// Floating point. This covers both scalar and vector operations.
def  : WriteRes<WriteFLoad,         [SKLPort23]> { let Latency = 6; }
def  : WriteRes<WriteFStore,        [SKLPort237, SKLPort4]>;
def  : WriteRes<WriteFMove,         [SKLPort015]>;
defm : X86WriteRes<WriteEMMS,       [SKLPort05,SKLPort0156], 10, [9,1], 10>;

defm : SKLWriteResPair<WriteFAdd,  [SKLPort01],  4, [1], 1, 6>; // Floating point add/sub.
defm : SKLWriteResPair<WriteFAddY, [SKLPort01],  4, [1], 1, 7>; // Floating point add/sub (YMM/ZMM).
defm : SKLWriteResPair<WriteFCmp,  [SKLPort01],  4, [1], 1, 6>; // Floating point compare.
defm : SKLWriteResPair<WriteFCmpY, [SKLPort01],  4, [1], 1, 7>; // Floating point compare (YMM/ZMM).
defm : SKLWriteResPair<WriteFCom,   [SKLPort0],  2>; // Floating point compare to flags.
defm : SKLWriteResPair<WriteFMul,  [SKLPort01],  4, [1], 1, 6>; // Floating point multiplication.
defm : SKLWriteResPair<WriteFMulY, [SKLPort01],  4, [1], 1, 7>; // Floating point multiplication (YMM/ZMM).
defm : SKLWriteResPair<WriteFDiv,   [SKLPort0], 12, [1], 1, 5>; // 10-14 cycles. // Floating point division.
defm : SKLWriteResPair<WriteFDivY,  [SKLPort0], 12, [1], 1, 7>; // 10-14 cycles. // Floating point division (YMM/ZMM).
defm : SKLWriteResPair<WriteFSqrt,  [SKLPort0], 15, [1], 1, 5>; // Floating point square root.
defm : SKLWriteResPair<WriteFSqrtY, [SKLPort0], 15, [1], 1, 7>; // Floating point square root (YMM/ZMM).
defm : SKLWriteResPair<WriteFRcp,   [SKLPort0], 4, [1], 1, 5>; // Floating point reciprocal estimate.
defm : SKLWriteResPair<WriteFRcpY,  [SKLPort0], 4, [1], 1, 5>; // Floating point reciprocal estimate (YMM/ZMM).
defm : SKLWriteResPair<WriteFRsqrt, [SKLPort0], 4, [1], 1, 5>; // Floating point reciprocal square root estimate.
defm : SKLWriteResPair<WriteFRsqrtY,[SKLPort0], 4, [1], 1, 5>; // Floating point reciprocal square root estimate (YMM/ZMM).
defm : SKLWriteResPair<WriteFMA,    [SKLPort01], 4, [1], 1, 5>; // Fused Multiply Add.
defm : SKLWriteResPair<WriteFMAX,   [SKLPort01], 4, [1], 1, 6>; // Fused Multiply Add (XMM).
defm : SKLWriteResPair<WriteFMAY,   [SKLPort01], 4, [1], 1, 7>; // Fused Multiply Add (YMM/ZMM).
defm : SKLWriteResPair<WriteDPPD,   [SKLPort5,SKLPort01],  9, [1,2], 3, 6>; // Floating point double dot product.
defm : SKLWriteResPair<WriteDPPS,   [SKLPort5,SKLPort01], 13, [1,3], 4, 6>; // Floating point single dot product.
defm : SKLWriteResPair<WriteDPPSY,  [SKLPort5,SKLPort01], 13, [1,3], 4, 7>; // Floating point single dot product (YMM).
defm : SKLWriteResPair<WriteFSign,   [SKLPort0], 1>; // Floating point fabs/fchs.
defm : SKLWriteResPair<WriteFRnd,     [SKLPort01], 8, [2], 2, 6>; // Floating point rounding.
defm : SKLWriteResPair<WriteFRndY,    [SKLPort01], 8, [2], 2, 7>; // Floating point rounding (YMM/ZMM).
defm : SKLWriteResPair<WriteFLogic,  [SKLPort015], 1, [1], 1, 6>; // Floating point and/or/xor logicals.
defm : SKLWriteResPair<WriteFLogicY, [SKLPort015], 1, [1], 1, 7>; // Floating point and/or/xor logicals (YMM/ZMM).
defm : SKLWriteResPair<WriteFShuffle,  [SKLPort5], 1, [1], 1, 6>; // Floating point vector shuffles.
defm : SKLWriteResPair<WriteFShuffleY, [SKLPort5], 1, [1], 1, 7>; // Floating point vector shuffles (YMM/ZMM).
defm : SKLWriteResPair<WriteFVarShuffle,  [SKLPort5], 1, [1], 1, 6>; // Floating point vector shuffles.
defm : SKLWriteResPair<WriteFVarShuffleY, [SKLPort5], 1, [1], 1, 7>; // Floating point vector shuffles.
defm : SKLWriteResPair<WriteFBlend,  [SKLPort015], 1, [1], 1, 6>; // Floating point vector blends.
defm : SKLWriteResPair<WriteFBlendY, [SKLPort015], 1, [1], 1, 7>; // Floating point vector blends.
defm : SKLWriteResPair<WriteFVarBlend, [SKLPort015], 2, [2], 2, 6>; // Fp vector variable blends.
defm : SKLWriteResPair<WriteFVarBlendY,[SKLPort015], 2, [2], 2, 7>; // Fp vector variable blends.

def  : WriteRes<WriteCvtF2FSt, [SKLPort4,SKLPort5,SKLPort237,SKLPort01]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}

// FMA Scheduling helper class.
// class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Vector integer operations.
def  : WriteRes<WriteVecLoad,         [SKLPort23]> { let Latency = 6; }
def  : WriteRes<WriteVecStore,        [SKLPort237, SKLPort4]>;
def  : WriteRes<WriteVecMove,         [SKLPort015]>;

defm : SKLWriteResPair<WriteVecALU,   [SKLPort01], 1, [1], 1, 6>; // Vector integer ALU op, no logicals.
defm : SKLWriteResPair<WriteVecALUY,  [SKLPort01], 1, [1], 1, 7>; // Vector integer ALU op, no logicals (YMM/ZMM).
defm : SKLWriteResPair<WriteVecLogic, [SKLPort015], 1, [1], 1, 6>; // Vector integer and/or/xor.
defm : SKLWriteResPair<WriteVecLogicY,[SKLPort015], 1, [1], 1, 7>; // Vector integer and/or/xor (YMM/ZMM).
defm : SKLWriteResPair<WriteVecIMul,  [SKLPort0] ,  4, [1], 1, 5>; // Vector integer multiply.
defm : SKLWriteResPair<WriteVecIMulX, [SKLPort01],  4, [1], 1, 6>; // Vector integer multiply (XMM).
defm : SKLWriteResPair<WriteVecIMulY, [SKLPort01],  4, [1], 1, 7>; // Vector integer multiply (YMM/ZMM).
defm : SKLWriteResPair<WritePMULLD,   [SKLPort01], 10, [2], 2, 6>; // Vector PMULLD.
defm : SKLWriteResPair<WritePMULLDY,  [SKLPort01], 10, [2], 2, 7>; // Vector PMULLD (YMM/ZMM).
defm : SKLWriteResPair<WriteShuffle,  [SKLPort5], 1, [1], 1, 5>; // Vector shuffles.
defm : SKLWriteResPair<WriteShuffleY, [SKLPort5], 1, [1], 1, 7>; // Vector shuffles (YMM/ZMM).
defm : SKLWriteResPair<WriteVarShuffle,  [SKLPort5], 1, [1], 1, 6>; // Vector shuffles.
defm : SKLWriteResPair<WriteVarShuffleY, [SKLPort5], 1, [1], 1, 7>; // Vector shuffles (YMM/ZMM).
defm : SKLWriteResPair<WriteBlend,  [SKLPort5], 1, [1], 1, 6>; // Vector blends.
defm : SKLWriteResPair<WriteBlendY, [SKLPort5], 1, [1], 1, 7>; // Vector blends (YMM/ZMM).
defm : SKLWriteResPair<WriteVarBlend,  [SKLPort015], 2, [2], 2, 6>; // Vector variable blends.
defm : SKLWriteResPair<WriteVarBlendY, [SKLPort015], 2, [2], 2, 6>; // Vector variable blends (YMM/ZMM).
defm : SKLWriteResPair<WriteMPSAD,  [SKLPort5], 4, [2], 2, 6>; // Vector MPSAD.
defm : SKLWriteResPair<WriteMPSADY, [SKLPort5], 4, [2], 2, 7>; // Vector MPSAD.
defm : SKLWriteResPair<WritePSADBW,  [SKLPort5], 3, [1], 1, 6>; // Vector PSADBW.
defm : SKLWriteResPair<WritePSADBWY, [SKLPort5], 3, [1], 1, 7>; // Vector PSADBW.
defm : SKLWriteResPair<WritePHMINPOS, [SKLPort01], 4, [1], 1, 6>; // Vector PHMINPOS.

// Vector integer shifts.
defm : SKLWriteResPair<WriteVecShift,     [SKLPort0], 1, [1], 1, 5>;
defm : X86WriteRes<WriteVecShiftX,        [SKLPort5,SKLPort01],  2, [1,1], 2>;
defm : X86WriteRes<WriteVecShiftY,        [SKLPort5,SKLPort01],  4, [1,1], 2>;
defm : X86WriteRes<WriteVecShiftXLd,      [SKLPort01,SKLPort23], 7, [1,1], 2>;
defm : X86WriteRes<WriteVecShiftYLd,      [SKLPort01,SKLPort23], 8, [1,1], 2>;

defm : SKLWriteResPair<WriteVecShiftImm,  [SKLPort0],  1, [1], 1, 5>;
defm : SKLWriteResPair<WriteVecShiftImmX, [SKLPort01], 1, [1], 1, 6>; // Vector integer immediate shifts (XMM).
defm : SKLWriteResPair<WriteVecShiftImmY, [SKLPort01], 1, [1], 1, 7>; // Vector integer immediate shifts (YMM/ZMM).
defm : SKLWriteResPair<WriteVarVecShift,  [SKLPort01], 1, [1], 1, 6>; // Variable vector shifts.
defm : SKLWriteResPair<WriteVarVecShiftY, [SKLPort01], 1, [1], 1, 7>; // Variable vector shifts (YMM/ZMM).

// Vector insert/extract operations.
def : WriteRes<WriteVecInsert, [SKLPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteVecInsertLd, [SKLPort5,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def: InstRW<[WriteVecInsertLd], (instregex "(V?)MOV(H|L)(PD|PS)rm")>;

def : WriteRes<WriteVecExtract, [SKLPort0,SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : WriteRes<WriteVecExtractSt, [SKLPort4,SKLPort5,SKLPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

// Conversion between integer and float.
defm : SKLWriteResPair<WriteCvtF2I, [SKLPort1], 3>; // Float -> Integer.
defm : SKLWriteResPair<WriteCvtI2F, [SKLPort1], 4>; // Integer -> Float.
defm : SKLWriteResPair<WriteCvtF2F, [SKLPort1], 3>; // Float -> Float size conversion.

// Strings instructions.

// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [SKLPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [SKLPort0, SKLPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [SKLPort0, SKLPort5, SKLPort015, SKLPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def : WriteRes<WritePCmpEStrMLd, [SKLPort0, SKLPort5,SKLPort23, SKLPort015, SKLPort0156]> {
  let Latency = 25;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [SKLPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [SKLPort0, SKLPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [SKLPort0, SKLPort5, SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def : WriteRes<WritePCmpEStrILd, [SKLPort0, SKLPort5, SKLPort23, SKLPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}

// MOVMSK Instructions.
def : WriteRes<WriteFMOVMSK,    [SKLPort0]> { let Latency = 2; }
def : WriteRes<WriteVecMOVMSK,  [SKLPort0]> { let Latency = 2; }
def : WriteRes<WriteVecMOVMSKY, [SKLPort0]> { let Latency = 2; }
def : WriteRes<WriteMMXMOVMSK,  [SKLPort0]> { let Latency = 2; }

// AES instructions.
def : WriteRes<WriteAESDecEnc, [SKLPort0]> { // Decryption, encryption.
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [SKLPort0, SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

def : WriteRes<WriteAESIMC, [SKLPort0]> { // InvMixColumn.
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [SKLPort0, SKLPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}

def : WriteRes<WriteAESKeyGen, [SKLPort0, SKLPort5, SKLPort015]> { // Key Generation.
  let Latency = 20;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,2];
}
def : WriteRes<WriteAESKeyGenLd, [SKLPort0, SKLPort5, SKLPort23, SKLPort015]> {
  let Latency = 25;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,1,1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [SKLPort5]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteCLMulLd, [SKLPort5, SKLPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

// Catch-all for expensive system instructions.
def : WriteRes<WriteSystem,     [SKLPort0156]> { let Latency = 100; } // def WriteSystem : SchedWrite;

// AVX2.
defm : SKLWriteResPair<WriteFShuffle256, [SKLPort5], 3, [1], 1, 7>; // Fp 256-bit width vector shuffles.
defm : SKLWriteResPair<WriteFVarShuffle256, [SKLPort5], 3, [1], 1, 7>; // Fp 256-bit width vector variable shuffles.
defm : SKLWriteResPair<WriteShuffle256, [SKLPort5], 3, [1], 1, 7>;  // 256-bit width vector shuffles.
defm : SKLWriteResPair<WriteVarShuffle256, [SKLPort5], 3, [1], 1, 7>;  // 256-bit width vector variable shuffles.

// Old microcoded instructions that nobody use.
def : WriteRes<WriteMicrocoded, [SKLPort0156]> { let Latency = 100; } // def WriteMicrocoded : SchedWrite;

// Fence instructions.
def : WriteRes<WriteFence,  [SKLPort23, SKLPort4]>;

// Load/store MXCSR.
def : WriteRes<WriteLDMXCSR, [SKLPort0,SKLPort23,SKLPort0156]> { let Latency = 7; let NumMicroOps = 3; let ResourceCycles = [1,1,1]; }
def : WriteRes<WriteSTMXCSR, [SKLPort4,SKLPort5,SKLPort237]> { let Latency = 2; let NumMicroOps = 3; let ResourceCycles = [1,1,1]; }

// Nop, not very useful expect it provides a model for nops!
def : WriteRes<WriteNop, []>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : SKLWriteResPair<WriteFHAdd,  [SKLPort5,SKLPort01], 6, [2,1], 3, 6>;
defm : SKLWriteResPair<WriteFHAddY, [SKLPort5,SKLPort01], 6, [2,1], 3, 7>;
defm : SKLWriteResPair<WritePHAdd,  [SKLPort5,SKLPort015], 3, [2,1], 3, 6>;
defm : SKLWriteResPair<WritePHAddY, [SKLPort5,SKLPort015], 3, [2,1], 3, 7>;

// Remaining instrs.

def SKLWriteResGroup1 : SchedWriteRes<[SKLPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PADDSBirr",
                                            "MMX_PADDSWirr",
                                            "MMX_PADDUSBirr",
                                            "MMX_PADDUSWirr",
                                            "MMX_PAVGBirr",
                                            "MMX_PAVGWirr",
                                            "MMX_PCMPEQBirr",
                                            "MMX_PCMPEQDirr",
                                            "MMX_PCMPEQWirr",
                                            "MMX_PCMPGTBirr",
                                            "MMX_PCMPGTDirr",
                                            "MMX_PCMPGTWirr",
                                            "MMX_PMAXSWirr",
                                            "MMX_PMAXUBirr",
                                            "MMX_PMINSWirr",
                                            "MMX_PMINUBirr",
                                            "MMX_PSUBSBirr",
                                            "MMX_PSUBSWirr",
                                            "MMX_PSUBUSBirr",
                                            "MMX_PSUBUSWirr")>;

def SKLWriteResGroup3 : SchedWriteRes<[SKLPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup3], (instregex "COMP_FST0r",
                                            "COM_FST0r",
                                            "MMX_MOVD64rr",
                                            "MMX_MOVD64to64rr",
                                            "UCOM_FPr",
                                            "UCOM_Fr",
                                            "(V?)MOV64toPQIrr",
                                            "(V?)MOVDI2PDIrr")>;

def SKLWriteResGroup4 : SchedWriteRes<[SKLPort6]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup4], (instregex "JMP(16|32|64)r")>;

def SKLWriteResGroup6 : SchedWriteRes<[SKLPort05]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup6], (instrs FINCSTP, FNOP)>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_MOVQ64rr",
                                            "MMX_PABS(B|D|W)rr",
                                            "MMX_PADD(B|D|Q|W)irr",
                                            "MMX_PANDNirr",
                                            "MMX_PANDirr",
                                            "MMX_PORirr",
                                            "MMX_PSIGN(B|D|W)rr",
                                            "MMX_PSUB(B|D|Q|W)irr",
                                            "MMX_PXORirr")>;

def SKLWriteResGroup7 : SchedWriteRes<[SKLPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup7], (instrs CDQ, CQO, CLAC, STAC)>;
def: InstRW<[SKLWriteResGroup7], (instregex "ADC(16|32|64)ri",
                                            "ADC(16|32|64)i",
                                            "ADC(8|16|32|64)rr",
                                            "ADCX(32|64)rr",
                                            "ADOX(32|64)rr",
                                            "BT(16|32|64)ri8",
                                            "BT(16|32|64)rr",
                                            "BTC(16|32|64)ri8",
                                            "BTC(16|32|64)rr",
                                            "BTR(16|32|64)ri8",
                                            "BTR(16|32|64)rr",
                                            "BTS(16|32|64)ri8",
                                            "BTS(16|32|64)rr",
                                            "SBB(16|32|64)ri",
                                            "SBB(16|32|64)i",
                                            "SBB(8|16|32|64)rr")>;

def SKLWriteResGroup8 : SchedWriteRes<[SKLPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup8], (instregex "ANDN(32|64)rr",
                                            "BLSI(32|64)rr",
                                            "BLSMSK(32|64)rr",
                                            "BLSR(32|64)rr")>;

def SKLWriteResGroup9 : SchedWriteRes<[SKLPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup9], (instregex "(V?)PADDB(Y?)rr",
                                            "(V?)PADDD(Y?)rr",
                                            "(V?)PADDQ(Y?)rr",
                                            "(V?)PADDW(Y?)rr",
                                            "VPBLENDD(Y?)rri",
                                            "(V?)PSUBB(Y?)rr",
                                            "(V?)PSUBD(Y?)rr",
                                            "(V?)PSUBQ(Y?)rr",
                                            "(V?)PSUBW(Y?)rr")>;

def SKLWriteResGroup10 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup10], (instrs CBW, CWDE, CDQE)>;
def: InstRW<[SKLWriteResGroup10], (instrs LAHF, SAHF)>; // TODO: This doesn't match Agner's data
def: InstRW<[SKLWriteResGroup10], (instregex "CLC",
                                             "CMC",
                                             "NOOP",
                                             "SGDT64m",
                                             "SIDT64m",
                                             "SMSW16m",
                                             "STC",
                                             "STRm",
                                             "SYSCALL")>;

def SKLWriteResGroup11 : SchedWriteRes<[SKLPort4,SKLPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup11], (instregex "FBSTPm",
                                             "MMX_MOVD64from64rm",
                                             "MMX_MOVD64mr",
                                             "MMX_MOVNTQmr",
                                             "MMX_MOVQ64mr",
                                             "MOVNTI_64mr",
                                             "MOVNTImr",
                                             "ST_FP(32|64|80)m",
                                             "VEXTRACTF128mr",
                                             "VEXTRACTI128mr",
                                             "(V?)MOVAPDYmr",
                                             "(V?)MOVAPS(Y?)mr",
                                             "(V?)MOVDQA(Y?)mr",
                                             "(V?)MOVDQU(Y?)mr",
                                             "(V?)MOVHPDmr",
                                             "(V?)MOVHPSmr",
                                             "(V?)MOVLPDmr",
                                             "(V?)MOVLPSmr",
                                             "(V?)MOVNTDQ(Y?)mr",
                                             "(V?)MOVNTPD(Y?)mr",
                                             "(V?)MOVNTPS(Y?)mr",
                                             "(V?)MOVPDI2DImr",
                                             "(V?)MOVPQI2QImr",
                                             "(V?)MOVPQIto64mr",
                                             "(V?)MOVSDmr",
                                             "(V?)MOVSSmr",
                                             "(V?)MOVUPD(Y?)mr",
                                             "(V?)MOVUPS(Y?)mr",
                                             "VMPTRSTm")>;

def SKLWriteResGroup12 : SchedWriteRes<[SKLPort0]> {
  let Latency = 2;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_MOVD64from64rr",
                                             "MMX_MOVD64grr",
                                             "(V?)MOVPDI2DIrr",
                                             "(V?)MOVPQIto64rr",
                                             "VTESTPD(Y?)rr",
                                             "VTESTPS(Y?)rr")>;

def SKLWriteResGroup13 : SchedWriteRes<[SKLPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup13], (instregex "MMX_MOVQ2DQrr")>;

def SKLWriteResGroup14 : SchedWriteRes<[SKLPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup14], (instrs FDECSTP)>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_MOVDQ2Qrr")>;

def SKLWriteResGroup15 : SchedWriteRes<[SKLPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup15], (instregex "CMOV(A|BE)(16|32|64)rr",
                                             "ROL(8|16|32|64)r1",
                                             "ROL(8|16|32|64)ri",
                                             "ROR(8|16|32|64)r1",
                                             "ROR(8|16|32|64)ri",
                                             "SET(A|BE)r")>;

def SKLWriteResGroup17 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup17], (instrs LFENCE,
                                          WAIT,
                                          XGETBV)>;

def SKLWriteResGroup18 : SchedWriteRes<[SKLPort0,SKLPort237]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup18], (instregex "VMASKMOVPD(Y?)mr",
                                             "VMASKMOVPS(Y?)mr",
                                             "VPMASKMOVD(Y?)mr",
                                             "VPMASKMOVQ(Y?)mr")>;

def SKLWriteResGroup20 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup20], (instregex "CLFLUSH")>;

def SKLWriteResGroup21 : SchedWriteRes<[SKLPort237,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup21], (instregex "SFENCE")>;

def SKLWriteResGroup22 : SchedWriteRes<[SKLPort06,SKLPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup22], (instrs BSWAP64r)>;

def SKLWriteResGroup22_1 : SchedWriteRes<[SKLPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup22_1], (instrs BSWAP32r)>;

def SKLWriteResGroup23 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup23], (instrs CWD)>;
def: InstRW<[SKLWriteResGroup23], (instrs JCXZ, JECXZ, JRCXZ)>;
def: InstRW<[SKLWriteResGroup23], (instregex "ADC8i8",
                                             "ADC8ri",
                                             "SBB8i8",
                                             "SBB8ri")>;

def SKLWriteResGroup25 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup25], (instregex "FNSTCW16m")>;

def SKLWriteResGroup27 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort15]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup27], (instregex "MOVBE(16|32|64)mr")>;

def SKLWriteResGroup28 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup28], (instrs PUSH16r, PUSH32r, PUSH64r,
                                          STOSB, STOSL, STOSQ, STOSW)>;
def: InstRW<[SKLWriteResGroup28], (instregex "PUSH(16|32|64)rmr",
                                             "PUSH64i8")>;

def SKLWriteResGroup29 : SchedWriteRes<[SKLPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup29], (instregex "CMOV(N?)(B|BE|E|P)_F",
                                             "PDEP(32|64)rr",
                                             "PEXT(32|64)rr",
                                             "SHLD(16|32|64)rri8",
                                             "SHRD(16|32|64)rri8")>;

def SKLWriteResGroup29_16i : SchedWriteRes<[SKLPort1, SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup29_16i], (instrs IMUL16rri, IMUL16rri8)>;

def SKLWriteResGroup30 : SchedWriteRes<[SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup30], (instregex "(ADD|SUB|SUBR)_FPrST0",
                                             "(ADD|SUB|SUBR)_FST0r",
                                             "(ADD|SUB|SUBR)_FrST0",
                                             "VPBROADCASTBrr",
                                             "VPBROADCASTWrr",
                                             "(V?)PCMPGTQ(Y?)rr",
                                             "VPMOVSXBDYrr",
                                             "VPMOVSXBQYrr",
                                             "VPMOVSXBWYrr",
                                             "VPMOVSXDQYrr",
                                             "VPMOVSXWDYrr",
                                             "VPMOVSXWQYrr",
                                             "VPMOVZXBDYrr",
                                             "VPMOVZXBQYrr",
                                             "VPMOVZXBWYrr",
                                             "VPMOVZXDQYrr",
                                             "VPMOVZXWDYrr",
                                             "VPMOVZXWQYrr")>;

def SKLWriteResGroup31 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup31], (instregex "(V?)PTEST(Y?)rr")>;

def SKLWriteResGroup32 : SchedWriteRes<[SKLPort0,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup32], (instregex "FNSTSW16r")>;

def SKLWriteResGroup33 : SchedWriteRes<[SKLPort06]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKLWriteResGroup33], (instregex "ROL(8|16|32|64)rCL",
                                             "ROR(8|16|32|64)rCL",
                                             "SAR(8|16|32|64)rCL",
                                             "SHL(8|16|32|64)rCL",
                                             "SHR(8|16|32|64)rCL")>;

def SKLWriteResGroup34 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKLWriteResGroup34], (instrs XADD8rr, XADD16rr, XADD32rr, XADD64rr,
                                          XCHG8rr, XCHG16rr, XCHG32rr, XCHG64rr,
                                          XCHG16ar, XCHG32ar, XCHG64ar)>;

def SKLWriteResGroup35 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup35], (instregex "MMX_PH(ADD|SUB)SWrr")>;

def SKLWriteResGroup36 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup36], (instregex "(V?)PHADDSW(Y?)rr",
                                             "(V?)PHSUBSW(Y?)rr")>;

def SKLWriteResGroup37 : SchedWriteRes<[SKLPort5,SKLPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup37], (instregex "MMX_PH(ADD|SUB)(D|W)rr")>;

def SKLWriteResGroup39 : SchedWriteRes<[SKLPort5,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup39], (instregex "MMX_PACKSSDWirr",
                                             "MMX_PACKSSWBirr",
                                             "MMX_PACKUSWBirr")>;

def SKLWriteResGroup40 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup40], (instregex "CLD")>;

def SKLWriteResGroup41 : SchedWriteRes<[SKLPort237,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup41], (instrs MFENCE)>;

def SKLWriteResGroup42 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup42], (instregex "RCL(8|16|32|64)r1",
                                             "RCL(8|16|32|64)ri",
                                             "RCR(8|16|32|64)r1",
                                             "RCR(8|16|32|64)ri")>;

def SKLWriteResGroup43 : SchedWriteRes<[SKLPort0,SKLPort4,SKLPort237]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup43], (instregex "FNSTSWm")>;

def SKLWriteResGroup44 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort06]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKLWriteResGroup44], (instregex "SET(A|BE)m")>;

def SKLWriteResGroup45 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort237,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup45], (instregex "CALL(16|32|64)r")>;

def SKLWriteResGroup46 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup46], (instregex "CALL64pcrel32")>;

def SKLWriteResGroup47 : SchedWriteRes<[SKLPort0]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup47], (instregex "MUL_FPrST0",
                                             "MUL_FST0r",
                                             "MUL_FrST0")>;

def SKLWriteResGroup48 : SchedWriteRes<[SKLPort01]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup48], (instregex "(V?)CVTDQ2PS(Y?)rr",
                                             "(V?)CVTPS2DQ(Y?)rr",
                                             "(V?)CVTTPS2DQ(Y?)rr")>;

def SKLWriteResGroup51 : SchedWriteRes<[SKLPort1,SKLPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup51], (instrs IMUL64r, MUL64r, MULX64rr)>;

def SKLWriteResGroup51_16 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKLWriteResGroup51_16], (instrs IMUL16r, MUL16r)>;

def SKLWriteResGroup53 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup53], (instregex "IST(T?)_FP(16|32|64)m",
                                             "IST_F(16|32)m")>;

def SKLWriteResGroup54 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[SKLWriteResGroup54], (instrs FNCLEX)>;

def SKLWriteResGroup55 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup55], (instrs PAUSE)>;

def SKLWriteResGroup56 : SchedWriteRes<[SKLPort015,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup56], (instrs VZEROUPPER)>;

def SKLWriteResGroup57 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKLWriteResGroup57], (instregex "LAR(16|32|64)rr")>;

def SKLWriteResGroup58 : SchedWriteRes<[SKLPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup58], (instregex "MOVSX(16|32|64)rm16",
                                             "MOVSX(16|32|64)rm32",
                                             "MOVSX(16|32|64)rm8",
                                             "MOVZX(16|32|64)rm16",
                                             "MOVZX(16|32|64)rm8",
                                             "(V?)MOVDDUPrm")>; // TODO: Should this be SKLWriteResGroup67?

def SKLWriteResGroup59 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup59], (instregex "MMX_CVTPI2PDirr",
                                             "(V?)CVTDQ2PDrr")>;

def SKLWriteResGroup60 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup60], (instregex "MMX_CVTPD2PIirr",
                                             "MMX_CVTPS2PIirr",
                                             "MMX_CVTTPD2PIirr",
                                             "MMX_CVTTPS2PIirr",
                                             "(V?)CVTPD2DQrr",
                                             "(V?)CVTPD2PSrr",
                                             "VCVTPH2PSrr",
                                             "(V?)CVTPS2PDrr",
                                             "VCVTPS2PHrr",
                                             "(V?)CVTSD2SSrr",
                                             "(V?)CVTSI642SDrr",
                                             "(V?)CVTSI2SDrr",
                                             "(V?)CVTSI2SSrr",
                                             "(V?)CVTSS2SDrr",
                                             "(V?)CVTTPD2DQrr")>;

def SKLWriteResGroup61 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup61], (instregex "STR(16|32|64)r")>;

def SKLWriteResGroup62 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup62], (instrs IMUL32r, MUL32r, MULX32rr)>;

def SKLWriteResGroup63 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[SKLWriteResGroup63], (instregex "XSETBV")>;

def SKLWriteResGroup64 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[SKLWriteResGroup64], (instregex "CMPXCHG(8|16|32|64)rr")>;

def SKLWriteResGroup65 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKLWriteResGroup65], (instregex "PUSHF16",
                                             "PUSHF64")>;

def SKLWriteResGroup67 : SchedWriteRes<[SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup67], (instregex "VBROADCASTSSrm",
                                             "(V?)MOVSHDUPrm",
                                             "(V?)MOVSLDUPrm",
                                             "VPBROADCASTDrm",
                                             "VPBROADCASTQrm")>;

def SKLWriteResGroup68 : SchedWriteRes<[SKLPort0]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup68], (instregex "MMX_CVTPI2PSirr")>;

def SKLWriteResGroup69 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PADDSBirm",
                                             "MMX_PADDSWirm",
                                             "MMX_PADDUSBirm",
                                             "MMX_PADDUSWirm",
                                             "MMX_PAVGBirm",
                                             "MMX_PAVGWirm",
                                             "MMX_PCMPEQBirm",
                                             "MMX_PCMPEQDirm",
                                             "MMX_PCMPEQWirm",
                                             "MMX_PCMPGTBirm",
                                             "MMX_PCMPGTDirm",
                                             "MMX_PCMPGTWirm",
                                             "MMX_PMAXSWirm",
                                             "MMX_PMAXUBirm",
                                             "MMX_PMINSWirm",
                                             "MMX_PMINUBirm",
                                             "MMX_PSUBSBirm",
                                             "MMX_PSUBSWirm",
                                             "MMX_PSUBUSBirm",
                                             "MMX_PSUBUSWirm")>;

def SKLWriteResGroup70 : SchedWriteRes<[SKLPort0,SKLPort01]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup70], (instregex "(V?)CVTSD2SI64rr",
                                             "(V?)CVTSD2SIrr",
                                             "(V?)CVTSS2SI64rr",
                                             "(V?)CVTSS2SIrr",
                                             "(V?)CVTTSD2SI64rr",
                                             "(V?)CVTTSD2SIrr")>;

def SKLWriteResGroup72 : SchedWriteRes<[SKLPort6,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup72], (instregex "FARJMP64",
                                             "JMP(16|32|64)m")>;

def SKLWriteResGroup73 : SchedWriteRes<[SKLPort23,SKLPort05]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PABS(B|D|W)rm",
                                             "MMX_PADD(B|D|Q|W)irm",
                                             "MMX_PANDNirm",
                                             "MMX_PANDirm",
                                             "MMX_PORirm",
                                             "MMX_PSIGN(B|D|W)rm",
                                             "MMX_PSUB(B|D|Q|W)irm",
                                             "MMX_PXORirm")>;

def SKLWriteResGroup74 : SchedWriteRes<[SKLPort23,SKLPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup74], (instregex "BT(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup74, ReadAfterLd], (instrs ADC8rm, ADC16rm, ADC32rm, ADC64rm,
                                                       ADCX32rm, ADCX64rm,
                                                       ADOX32rm, ADOX64rm,
                                                       SBB8rm, SBB16rm, SBB32rm, SBB64rm)>;

def SKLWriteResGroup75 : SchedWriteRes<[SKLPort23,SKLPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup75], (instregex "ANDN(32|64)rm",
                                             "BLSI(32|64)rm",
                                             "BLSMSK(32|64)rm",
                                             "BLSR(32|64)rm",
                                             "MOVBE(16|32|64)rm")>;

def SKLWriteResGroup76 : SchedWriteRes<[SKLPort23,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup76], (instrs POP16r, POP32r, POP64r)>;
def: InstRW<[SKLWriteResGroup76], (instregex "POP(16|32|64)rmr")>;

def SKLWriteResGroup78 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup78], (instregex "(V?)CVTSI642SSrr")>;

def SKLWriteResGroup79 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKLWriteResGroup79], (instregex "SHLD(16|32|64)rrCL",
                                             "SHRD(16|32|64)rrCL")>;

def SKLWriteResGroup80 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup80], (instregex "SLDT(16|32|64)r")>;

def SKLWriteResGroup82 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup82], (instregex "BTC(16|32|64)mi8",
                                             "BTR(16|32|64)mi8",
                                             "BTS(16|32|64)mi8",
                                             "SAR(8|16|32|64)m1",
                                             "SAR(8|16|32|64)mi",
                                             "SHL(8|16|32|64)m1",
                                             "SHL(8|16|32|64)mi",
                                             "SHR(8|16|32|64)m1",
                                             "SHR(8|16|32|64)mi")>;

def SKLWriteResGroup83 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup83], (instregex "POP(16|32|64)rmm",
                                             "PUSH(16|32|64)rmm")>;

def SKLWriteResGroup84 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKLWriteResGroup84], (instregex "STD")>;

def SKLWriteResGroup85 : SchedWriteRes<[SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup85], (instregex "LD_F(32|64|80)m",
                                             "VBROADCASTF128",
                                             "VBROADCASTI128",
                                             "VBROADCASTSDYrm",
                                             "VBROADCASTSSYrm",
                                             "VLDDQUYrm",
                                             "VMOVAPDYrm",
                                             "VMOVAPSYrm",
                                             "VMOVDDUPYrm",
                                             "VMOVDQAYrm",
                                             "VMOVDQUYrm",
                                             "VMOVNTDQAYrm",
                                             "VMOVSHDUPYrm",
                                             "VMOVSLDUPYrm",
                                             "VMOVUPDYrm",
                                             "VMOVUPSYrm",
                                             "VPBROADCASTDYrm",
                                             "VPBROADCASTQYrm")>;

def SKLWriteResGroup86 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup86], (instregex "VCVTDQ2PDYrr")>;

def SKLWriteResGroup88 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup88], (instregex "(V?)PACKSSDWrm",
                                             "(V?)PACKSSWBrm",
                                             "(V?)PACKUSDWrm",
                                             "(V?)PACKUSWBrm",
                                             "(V?)PALIGNRrmi",
                                             "VPBROADCASTBrm",
                                             "VPBROADCASTWrm",
                                             "(V?)PSHUFDmi",
                                             "(V?)PSHUFHWmi",
                                             "(V?)PSHUFLWmi",
                                             "(V?)PUNPCKHBWrm",
                                             "(V?)PUNPCKHDQrm",
                                             "(V?)PUNPCKHQDQrm",
                                             "(V?)PUNPCKHWDrm",
                                             "(V?)PUNPCKLBWrm",
                                             "(V?)PUNPCKLDQrm",
                                             "(V?)PUNPCKLQDQrm",
                                             "(V?)PUNPCKLWDrm")>;

def SKLWriteResGroup88a : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup88a], (instregex "MMX_PSHUFBrm")>;

def SKLWriteResGroup89 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTPD2DQYrr",
                                             "VCVTPD2PSYrr",
                                             "VCVTPH2PSYrr",
                                             "VCVTPS2PDYrr",
                                             "VCVTPS2PHYrr",
                                             "VCVTTPD2DQYrr")>;

def SKLWriteResGroup91 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup91], (instregex "(V?)INSERTF128rm",
                                             "(V?)INSERTI128rm",
                                             "(V?)MASKMOVPDrm",
                                             "(V?)MASKMOVPSrm",
                                             "(V?)PADDBrm",
                                             "(V?)PADDDrm",
                                             "(V?)PADDQrm",
                                             "(V?)PADDWrm",
                                             "(V?)PBLENDDrmi",
                                             "(V?)PMASKMOVDrm",
                                             "(V?)PMASKMOVQrm",
                                             "(V?)PSUBBrm",
                                             "(V?)PSUBDrm",
                                             "(V?)PSUBQrm",
                                             "(V?)PSUBWrm")>;

def SKLWriteResGroup92 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup92], (instregex "MMX_PACKSSDWirm",
                                             "MMX_PACKSSWBirm",
                                             "MMX_PACKUSWBirm")>;

def SKLWriteResGroup93 : SchedWriteRes<[SKLPort23,SKLPort06]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup93], (instregex "CMOV(A|BE)(16|32|64)rm")>;

def SKLWriteResGroup94 : SchedWriteRes<[SKLPort23,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup94], (instrs LEAVE, LEAVE64,
                                          SCASB, SCASL, SCASQ, SCASW)>;

def SKLWriteResGroup95 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort01]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup95], (instregex "(V?)CVTTSS2SI64rr",
                                             "(V?)CVTTSS2SIrr")>;

def SKLWriteResGroup96 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort05]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup96], (instregex "FLDCW16m")>;

def SKLWriteResGroup98 : SchedWriteRes<[SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup98], (instregex "LRETQ",
                                             "RETQ")>;

def SKLWriteResGroup100 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup100], (instregex "ROL(8|16|32|64)m1",
                                              "ROL(8|16|32|64)mi",
                                              "ROR(8|16|32|64)m1",
                                              "ROR(8|16|32|64)mi")>;

def SKLWriteResGroup101 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup101], (instregex "XADD(8|16|32|64)rm")>;

def SKLWriteResGroup102 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[SKLWriteResGroup102], (instregex "CALL(16|32|64)m",
                                              "FARCALL64")>;

def SKLWriteResGroup103 : SchedWriteRes<[SKLPort6,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,1,2];
}
def: InstRW<[SKLWriteResGroup103], (instrs LOOP)>;

def SKLWriteResGroup106 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup106], (instregex "VTESTPDrm",
                                              "VTESTPSrm")>;

def SKLWriteResGroup107 : SchedWriteRes<[SKLPort1,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup107], (instregex "PDEP(32|64)rm",
                                              "PEXT(32|64)rm")>;

def SKLWriteResGroup107_16 : SchedWriteRes<[SKLPort1, SKLPort0156, SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup107_16], (instrs IMUL16rmi, IMUL16rmi8)>;

def SKLWriteResGroup107_16_2 : SchedWriteRes<[SKLPort1, SKLPort06, SKLPort0156, SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,2,1];
}
def: InstRW<[SKLWriteResGroup107_16_2], (instrs IMUL16m, MUL16m)>;

def SKLWriteResGroup108 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup108], (instregex "FCOM32m",
                                              "FCOM64m",
                                              "FCOMP32m",
                                              "FCOMP64m",
                                              "MMX_PSADBWirm", // TODO - SKLWriteResGroup120??
                                              "VPBROADCASTBYrm",
                                              "VPBROADCASTWYrm",
                                              "VPMOVSXBDYrm",
                                              "VPMOVSXBQYrm",
                                              "VPMOVSXWQYrm")>;

def SKLWriteResGroup110 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup110], (instregex "VMASKMOVPDYrm",
                                              "VMASKMOVPSYrm",
                                              "VPADDBYrm",
                                              "VPADDDYrm",
                                              "VPADDQYrm",
                                              "VPADDWYrm",
                                              "VPBLENDDYrmi",
                                              "VPMASKMOVDYrm",
                                              "VPMASKMOVQYrm",
                                              "VPSUBBYrm",
                                              "VPSUBDYrm",
                                              "VPSUBQYrm",
                                              "VPSUBWYrm")>;

def SKLWriteResGroup112 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKLWriteResGroup112], (instregex "MMX_PH(ADD|SUB)SWrm")>;

def SKLWriteResGroup113 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort05]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup113], (instregex "MMX_PH(ADD|SUB)(D|W)rm")>;

def SKLWriteResGroup114 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237,SKLPort01]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup114], (instregex "VCVTPS2PHYmr")>;

def SKLWriteResGroup115 : SchedWriteRes<[SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKLWriteResGroup115], (instregex "ROR(8|16|32|64)mCL")>;

def SKLWriteResGroup116 : SchedWriteRes<[SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup116], (instregex "RCL(8|16|32|64)m1",
                                              "RCL(8|16|32|64)mi",
                                              "RCR(8|16|32|64)m1",
                                              "RCR(8|16|32|64)mi")>;

def SKLWriteResGroup117 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKLWriteResGroup117], (instregex "ROL(8|16|32|64)mCL",
                                              "SAR(8|16|32|64)mCL",
                                              "SHL(8|16|32|64)mCL",
                                              "SHR(8|16|32|64)mCL")>;

def SKLWriteResGroup119 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKLWriteResGroup119], (instregex "ADC(8|16|32|64)mi",
                                              "CMPXCHG(8|16|32|64)rm",
                                              "SBB(8|16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup119, ReadAfterLd], (instrs ADC8mr, ADC16mr, ADC32mr, ADC64mr,
                                                        SBB8mr, SBB16mr, SBB32mr, SBB64mr)>;

def SKLWriteResGroup120 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup120], (instregex "MMX_CVTPI2PSirm",
                                              "VTESTPDYrm",
                                              "VTESTPSYrm")>;

def SKLWriteResGroup121 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup121], (instregex "(V?)PCMPGTQrm",
                                              "VPMOVSXBWYrm",
                                              "VPMOVSXDQYrm",
                                              "VPMOVSXWDYrm",
                                              "VPMOVZXWDYrm")>;

def SKLWriteResGroup122 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup122], (instregex "(V?)ADDSDrm",
                                              "(V?)ADDSSrm",
                                              "(V?)CMPSDrm",
                                              "(V?)CMPSSrm",
                                              "(V?)MAX(C?)SDrm",
                                              "(V?)MAX(C?)SSrm",
                                              "(V?)MIN(C?)SDrm",
                                              "(V?)MIN(C?)SSrm",
                                              "(V?)MULSDrm",
                                              "(V?)MULSSrm",
                                              "(V?)SUBSDrm",
                                              "(V?)SUBSSrm")>;

def SKLWriteResGroup123 : SchedWriteRes<[SKLPort23,SKLPort01]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup123], (instregex "MMX_CVTPS2PIirm",
                                              "MMX_CVTTPS2PIirm",
                                              "VCVTPH2PSrm",
                                              "(V?)CVTPS2PDrm")>;

def SKLWriteResGroup126 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup126], (instregex "(V?)PTESTrm")>;

def SKLWriteResGroup127 : SchedWriteRes<[SKLPort1,SKLPort5,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup127], (instrs IMUL64m, MUL64m, MULX64rm)>;

def SKLWriteResGroup128 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup128], (instregex "(V?)PHADDSWrm",
                                              "(V?)PHSUBSWrm")>;

def SKLWriteResGroup130 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup130], (instregex "SHLD(16|32|64)mri8",
                                              "SHRD(16|32|64)mri8")>;

def SKLWriteResGroup131 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKLWriteResGroup131], (instregex "LAR(16|32|64)rm",
                                              "LSL(16|32|64)rm")>;

def SKLWriteResGroup132 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup132], (instregex "(V?)RCPPSm",
                                              "(V?)RSQRTPSm")>;

def SKLWriteResGroup133 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup133], (instregex "(ADD|SUB|SUBR)_F(32|64)m",
                                              "ILD_F(16|32|64)m",
                                              "VPCMPGTQYrm",
                                              "VPMOVZXBDYrm",
                                              "VPMOVZXBQYrm",
                                              "VPMOVZXBWYrm",
                                              "VPMOVZXDQYrm",
                                              "VPMOVZXWQYrm")>;

def SKLWriteResGroup134 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup134], (instregex "(V?)CVTDQ2PSrm",
                                              "(V?)CVTPH2PSYrm",
                                              "(V?)CVTPS2DQrm",
                                              "(V?)CVTSS2SDrm",
                                              "(V?)CVTTPS2DQrm")>;

def SKLWriteResGroup138 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup138], (instregex "MMX_CVTPI2PDirm",
                                              "VPTESTYrm")>;

def SKLWriteResGroup139 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort01]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup139], (instregex "(V?)CVTSD2SSrm")>;

def SKLWriteResGroup140 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup140], (instregex "VPHADDSWYrm",
                                              "VPHSUBSWYrm")>;

def SKLWriteResGroup142 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort06,SKLPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup142], (instrs IMUL32m, MUL32m, MULX32rm)>;

def SKLWriteResGroup143 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 10;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,3];
}
def: InstRW<[SKLWriteResGroup143], (instregex "XCHG(8|16|32|64)rm")>;

def SKLWriteResGroup145 : SchedWriteRes<[SKLPort0,SKLFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup145], (instregex "(V?)DIVPSrr",
                                              "(V?)DIVSSrr")>;

def SKLWriteResGroup145_1 : SchedWriteRes<[SKLPort0,SKLFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKLWriteResGroup145_1], (instregex "VDIVPSYrr")>;

def SKLWriteResGroup146 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup146], (instregex "MUL_F(32|64)m",
                                              "VRCPPSYm",
                                              "VRSQRTPSYm")>;

def SKLWriteResGroup147 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup147], (instregex "VCVTDQ2PSYrm",
                                              "VCVTPS2DQYrm",
                                              "VCVTPS2PDYrm",
                                              "VCVTTPS2DQYrm")>;

def SKLWriteResGroup149 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup149], (instregex "FICOM16m",
                                              "FICOM32m",
                                              "FICOMP16m",
                                              "FICOMP32m")>;

def SKLWriteResGroup150 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup150], (instregex "(V?)CVTDQ2PDrm")>;

def SKLWriteResGroup151 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort01]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup151], (instregex "(V?)CVTSD2SI64rm",
                                              "(V?)CVTSD2SIrm",
                                              "(V?)CVTSS2SI64rm",
                                              "(V?)CVTSS2SIrm",
                                              "(V?)CVTTSD2SI64rm",
                                              "(V?)CVTTSD2SIrm",
                                              "VCVTTSS2SI64rm",
                                              "(V?)CVTTSS2SIrm")>;

def SKLWriteResGroup152 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort01]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup152], (instregex "CVTPD2DQrm",
                                              "CVTPD2PSrm",
                                              "CVTTPD2DQrm",
                                              "MMX_CVTPD2PIirm",
                                              "MMX_CVTTPD2PIirm")>;

def SKLWriteResGroup153 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKLWriteResGroup153], (instregex "SHLD(16|32|64)mrCL",
                                              "SHRD(16|32|64)mrCL")>;

def SKLWriteResGroup154 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,3,2];
}
def: InstRW<[SKLWriteResGroup154], (instregex "RCL(16|32|64)rCL",
                                              "RCR(16|32|64)rCL")>;

def SKLWriteResGroup155 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,1,2];
}
def: InstRW<[SKLWriteResGroup155], (instregex "RCL8rCL")>;

def SKLWriteResGroup156 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 11;
  let ResourceCycles = [2,9];
}
def: InstRW<[SKLWriteResGroup156], (instrs LOOPE, LOOPNE)>;

def SKLWriteResGroup157 : SchedWriteRes<[SKLPort0,SKLFPDivider]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup157], (instregex "(V?)SQRTPSr",
                                              "(V?)SQRTSSr")>;

def SKLWriteResGroup158 : SchedWriteRes<[SKLPort0,SKLFPDivider]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1,6];
}
def: InstRW<[SKLWriteResGroup158], (instregex "VSQRTPSYr")>;

def SKLWriteResGroup160 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort01]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup160], (instregex "CVTTSS2SI64rm")>;

def SKLWriteResGroup162 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup162], (instregex "(ADD|SUB|SUBR)_FI(16|32)m")>;

def SKLWriteResGroup163 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup163], (instregex "VCVTDQ2PDYrm")>;

def SKLWriteResGroup166 : SchedWriteRes<[SKLPort0,SKLFPDivider]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup166], (instregex "(V?)DIVPDrr",
                                              "(V?)DIVSDrr")>;

def SKLWriteResGroup166_1 : SchedWriteRes<[SKLPort0,SKLFPDivider]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKLWriteResGroup166_1], (instregex "VDIVPDYrr")>;

def SKLWriteResGroup169 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup169], (instregex "MUL_FI(16|32)m")>;

def SKLWriteResGroup170 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 14;
  let NumMicroOps = 10;
  let ResourceCycles = [2,4,1,3];
}
def: InstRW<[SKLWriteResGroup170], (instregex "RCR8rCL")>;

def SKLWriteResGroup171 : SchedWriteRes<[SKLPort0]> {
  let Latency = 15;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup171], (instregex "DIVR_FPrST0",
                                              "DIVR_FST0r",
                                              "DIVR_FrST0")>;

def SKLWriteResGroup174 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 15;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,5,1,1];
}
def: InstRW<[SKLWriteResGroup174], (instregex "RCL(8|16|32|64)mCL")>;

def SKLWriteResGroup175 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKLWriteResGroup175], (instregex "(V?)DIVSSrm")>;

def SKLWriteResGroup177 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 16;
  let NumMicroOps = 14;
  let ResourceCycles = [1,1,1,4,2,5];
}
def: InstRW<[SKLWriteResGroup177], (instregex "CMPXCHG8B")>;

def SKLWriteResGroup178 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 16;
  let NumMicroOps = 16;
  let ResourceCycles = [16];
}
def: InstRW<[SKLWriteResGroup178], (instrs VZEROALL)>;

def SKLWriteResGroup179 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 17;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,5];
}
def: InstRW<[SKLWriteResGroup179], (instregex "(V?)DIVPSrm")>;

def SKLWriteResGroup179_1 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 17;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKLWriteResGroup179_1], (instregex "(V?)SQRTSSm")>;

def SKLWriteResGroup180 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort5,SKLPort6,SKLPort05,SKLPort0156]> {
  let Latency = 17;
  let NumMicroOps = 15;
  let ResourceCycles = [2,1,2,4,2,4];
}
def: InstRW<[SKLWriteResGroup180], (instrs XCH_F)>;

def SKLWriteResGroup181 : SchedWriteRes<[SKLPort0,SKLFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 1;
  let ResourceCycles = [1,6];
}
def: InstRW<[SKLWriteResGroup181], (instregex "(V?)SQRTPDr",
                                              "(V?)SQRTSDr")>;

def SKLWriteResGroup181_1 : SchedWriteRes<[SKLPort0,SKLFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 1;
  let ResourceCycles = [1,12];
}
def: InstRW<[SKLWriteResGroup181_1], (instregex "VSQRTPDYr")>;

def SKLWriteResGroup182 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,5];
}
def: InstRW<[SKLWriteResGroup182], (instregex "VDIVPSYrm")>;

def SKLWriteResGroup183 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKLWriteResGroup183], (instregex "(V?)SQRTPSm")>;

def SKLWriteResGroup184 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[SKLWriteResGroup184], (instrs CPUID, RDTSC)>;

def SKLWriteResGroup185 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,4,1,2];
}
def: InstRW<[SKLWriteResGroup185], (instregex "RCR(8|16|32|64)mCL")>;

def SKLWriteResGroup186 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKLWriteResGroup186], (instregex "(V?)DIVSDrm")>;

def SKLWriteResGroup186_1 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,6];
}
def: InstRW<[SKLWriteResGroup186_1], (instregex "VSQRTPSYm")>;

def SKLWriteResGroup189 : SchedWriteRes<[SKLPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup189], (instregex "DIV_FPrST0",
                                              "DIV_FST0r",
                                              "DIV_FrST0")>;

def SKLWriteResGroup190 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKLWriteResGroup190], (instregex "(V?)DIVPDrm")>;

def SKLWriteResGroup192 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 20;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,1,2];
}
def: InstRW<[SKLWriteResGroup192], (instrs INSB, INSL, INSW)>;

def SKLWriteResGroup193 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort0156]> {
  let Latency = 20;
  let NumMicroOps = 10;
  let ResourceCycles = [1,2,7];
}
def: InstRW<[SKLWriteResGroup193], (instregex "MWAITrr")>;

def SKLWriteResGroup195 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,8];
}
def: InstRW<[SKLWriteResGroup195], (instregex "VDIVPDYrm")>;

def SKLWriteResGroup196 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 22;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup196], (instregex "DIV_F(32|64)m")>;

def SKLWriteResGroup196_1 : SchedWriteRes<[SKLPort0, SKLPort23, SKLPort5, SKLPort015]> {
  let Latency = 22;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKLWriteResGroup196_1], (instrs VGATHERDPSrm,
                                             VGATHERDPDrm,
                                             VGATHERQPDrm,
                                             VGATHERQPSrm,
                                             VPGATHERDDrm,
                                             VPGATHERDQrm,
                                             VPGATHERQDrm,
                                             VPGATHERQQrm)>;

def SKLWriteResGroup196_2 : SchedWriteRes<[SKLPort0, SKLPort23, SKLPort5, SKLPort015]> {
  let Latency = 25;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKLWriteResGroup196_2], (instrs VGATHERDPSYrm,
                                             VGATHERQPDYrm,
                                             VGATHERQPSYrm,
                                             VPGATHERDDYrm,
                                             VPGATHERDQYrm,
                                             VPGATHERQDYrm,
                                             VPGATHERQQYrm,
                                             VGATHERDPDYrm)>;

def SKLWriteResGroup197 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 23;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,6];
}
def: InstRW<[SKLWriteResGroup197], (instregex "(V?)SQRTSDm")>;

def SKLWriteResGroup198 : SchedWriteRes<[SKLPort0,SKLPort4,SKLPort5,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 23;
  let NumMicroOps = 19;
  let ResourceCycles = [2,1,4,1,1,4,6];
}
def: InstRW<[SKLWriteResGroup198], (instregex "CMPXCHG16B")>;

def SKLWriteResGroup199 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 24;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,6];
}
def: InstRW<[SKLWriteResGroup199], (instregex "(V?)SQRTPDm")>;

def SKLWriteResGroup201 : SchedWriteRes<[SKLPort0,SKLPort23,SKLFPDivider]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,12];
}
def: InstRW<[SKLWriteResGroup201], (instregex "VSQRTPDYm")>;

def SKLWriteResGroup202 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 25;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup202], (instregex "DIV_FI(16|32)m")>;

def SKLWriteResGroup206 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 27;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup206], (instregex "DIVR_F(32|64)m")>;

def SKLWriteResGroup207 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort0156]> {
  let Latency = 28;
  let NumMicroOps = 8;
  let ResourceCycles = [2,4,1,1];
}
def: InstRW<[SKLWriteResGroup207], (instregex "IDIV(8|16|32|64)m")>;

def SKLWriteResGroup208 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 30;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup208], (instregex "DIVR_FI(16|32)m")>;

def SKLWriteResGroup209 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort23,SKLPort06,SKLPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[SKLWriteResGroup209], (instregex "IN(8|16|32)ri",
                                              "IN(8|16|32)rr")>;

def SKLWriteResGroup210 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[SKLWriteResGroup210], (instregex "OUT(8|16|32)ir",
                                              "OUT(8|16|32)rr")>;

def SKLWriteResGroup211 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 37;
  let NumMicroOps = 31;
  let ResourceCycles = [1,8,1,21];
}
def: InstRW<[SKLWriteResGroup211], (instregex "XRSTOR(64)?")>;

def SKLWriteResGroup212 : SchedWriteRes<[SKLPort1,SKLPort4,SKLPort5,SKLPort6,SKLPort23,SKLPort237,SKLPort15,SKLPort0156]> {
  let Latency = 40;
  let NumMicroOps = 18;
  let ResourceCycles = [1,1,2,3,1,1,1,8];
}
def: InstRW<[SKLWriteResGroup212], (instregex "VMCLEARm")>;

def SKLWriteResGroup213 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 41;
  let NumMicroOps = 39;
  let ResourceCycles = [1,10,1,1,26];
}
def: InstRW<[SKLWriteResGroup213], (instregex "XSAVE64")>;

def SKLWriteResGroup214 : SchedWriteRes<[SKLPort5,SKLPort0156]> {
  let Latency = 42;
  let NumMicroOps = 22;
  let ResourceCycles = [2,20];
}
def: InstRW<[SKLWriteResGroup214], (instrs RDTSCP)>;

def SKLWriteResGroup215 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 42;
  let NumMicroOps = 40;
  let ResourceCycles = [1,11,1,1,26];
}
def: InstRW<[SKLWriteResGroup215], (instregex "^XSAVE$", "XSAVEC", "XSAVES")>;

def SKLWriteResGroup216 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 46;
  let NumMicroOps = 44;
  let ResourceCycles = [1,11,1,1,30];
}
def: InstRW<[SKLWriteResGroup216], (instregex "XSAVEOPT")>;

def SKLWriteResGroup217 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort05,SKLPort06,SKLPort0156]> {
  let Latency = 62;
  let NumMicroOps = 64;
  let ResourceCycles = [2,8,5,10,39];
}
def: InstRW<[SKLWriteResGroup217], (instregex "FLDENVm")>;

def SKLWriteResGroup218 : SchedWriteRes<[SKLPort0,SKLPort6,SKLPort23,SKLPort05,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 63;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[SKLWriteResGroup218], (instrs FXRSTOR64)>;

def SKLWriteResGroup219 : SchedWriteRes<[SKLPort0,SKLPort6,SKLPort23,SKLPort05,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 63;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[SKLWriteResGroup219], (instrs FXRSTOR)>;

def SKLWriteResGroup220 : SchedWriteRes<[SKLPort5,SKLPort05,SKLPort0156]> {
  let Latency = 75;
  let NumMicroOps = 15;
  let ResourceCycles = [6,3,6];
}
def: InstRW<[SKLWriteResGroup220], (instrs FNINIT)>;

def SKLWriteResGroup221 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort5,SKLPort6,SKLPort05,SKLPort0156]> {
  let Latency = 76;
  let NumMicroOps = 32;
  let ResourceCycles = [7,2,8,3,1,11];
}
def: InstRW<[SKLWriteResGroup221], (instregex "DIV(16|32|64)r")>;

def SKLWriteResGroup222 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort5,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 102;
  let NumMicroOps = 66;
  let ResourceCycles = [4,2,4,8,14,34];
}
def: InstRW<[SKLWriteResGroup222], (instregex "IDIV(16|32|64)r")>;

def SKLWriteResGroup223 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort4,SKLPort5,SKLPort6,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 106;
  let NumMicroOps = 100;
  let ResourceCycles = [9,1,11,16,1,11,21,30];
}
def: InstRW<[SKLWriteResGroup223], (instregex "FSTENVm")>;

} // SchedModel
