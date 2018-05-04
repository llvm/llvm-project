//=- X86SchedSkylake.td - X86 Skylake Server Scheduling ------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Skylake Server to support
// instruction scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def SkylakeServerModel : SchedMachineModel {
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

let SchedModel = SkylakeServerModel in {

// Skylake Server can issue micro-ops to 8 different ports in one cycle.

// Ports 0, 1, 5, and 6 handle all computation.
// Port 4 gets the data half of stores. Store data can be available later than
// the store address, but since we don't model the latency of stores, we can
// ignore that.
// Ports 2 and 3 are identical. They handle loads and the address half of
// stores. Port 7 can handle address calculations.
def SKXPort0 : ProcResource<1>;
def SKXPort1 : ProcResource<1>;
def SKXPort2 : ProcResource<1>;
def SKXPort3 : ProcResource<1>;
def SKXPort4 : ProcResource<1>;
def SKXPort5 : ProcResource<1>;
def SKXPort6 : ProcResource<1>;
def SKXPort7 : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def SKXPort01  : ProcResGroup<[SKXPort0, SKXPort1]>;
def SKXPort23  : ProcResGroup<[SKXPort2, SKXPort3]>;
def SKXPort237 : ProcResGroup<[SKXPort2, SKXPort3, SKXPort7]>;
def SKXPort04  : ProcResGroup<[SKXPort0, SKXPort4]>;
def SKXPort05  : ProcResGroup<[SKXPort0, SKXPort5]>;
def SKXPort06  : ProcResGroup<[SKXPort0, SKXPort6]>;
def SKXPort15  : ProcResGroup<[SKXPort1, SKXPort5]>;
def SKXPort16  : ProcResGroup<[SKXPort1, SKXPort6]>;
def SKXPort56  : ProcResGroup<[SKXPort5, SKXPort6]>;
def SKXPort015 : ProcResGroup<[SKXPort0, SKXPort1, SKXPort5]>;
def SKXPort056 : ProcResGroup<[SKXPort0, SKXPort5, SKXPort6]>;
def SKXPort0156: ProcResGroup<[SKXPort0, SKXPort1, SKXPort5, SKXPort6]>;

def SKXDivider : ProcResource<1>; // Integer division issued on port 0.
// FP division and sqrt on port 0.
def SKXFPDivider : ProcResource<1>;

// 60 Entry Unified Scheduler
def SKXPortAny : ProcResGroup<[SKXPort0, SKXPort1, SKXPort2, SKXPort3, SKXPort4,
                              SKXPort5, SKXPort6, SKXPort7]> {
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
multiclass SKXWriteResPair<X86FoldableSchedWrite SchedRW,
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
  def : WriteRes<SchedRW.Folded, !listconcat([SKXPort23], ExePorts)> {
    let Latency = !add(Lat, LoadLat);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = !add(UOps, 1);
  }
}

// A folded store needs a cycle on port 4 for the store data, and an extra port
// 2/3/7 cycle to recompute the address.
def : WriteRes<WriteRMW, [SKXPort237,SKXPort4]>;

// Arithmetic.
defm : SKXWriteResPair<WriteALU,   [SKXPort0156], 1>; // Simple integer ALU op.
defm : SKXWriteResPair<WriteIMul,  [SKXPort1],   3>; // Integer multiplication.
defm : SKXWriteResPair<WriteIDiv, [SKXPort0, SKXDivider], 25, [1,10], 1, 4>; // Integer division.
defm : SKXWriteResPair<WriteCRC32, [SKXPort1], 3>;

def : WriteRes<WriteIMulH, []> { let Latency = 3; } // Integer multiplication, high part.
def : WriteRes<WriteLEA, [SKXPort15]>; // LEA instructions can't fold loads.

defm : SKXWriteResPair<WriteCMOV,  [SKXPort06], 1>; // Conditional move.
def  : WriteRes<WriteSETCC, [SKXPort06]>; // Setcc.
def  : WriteRes<WriteSETCCStore, [SKXPort06,SKXPort4,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

// Integer shifts and rotates.
defm : SKXWriteResPair<WriteShift, [SKXPort06],  1>;

// Bit counts.
defm : SKXWriteResPair<WriteBitScan, [SKXPort1], 3>;
defm : SKXWriteResPair<WriteLZCNT,   [SKXPort1], 3>;
defm : SKXWriteResPair<WriteTZCNT,   [SKXPort1], 3>;
defm : SKXWriteResPair<WritePOPCNT,  [SKXPort1], 3>;

// BMI1 BEXTR, BMI2 BZHI
defm : SKXWriteResPair<WriteBEXTR, [SKXPort06,SKXPort15], 2, [1,1], 2>;
defm : SKXWriteResPair<WriteBZHI, [SKXPort15], 1>;

// Loads, stores, and moves, not folded with other operations.
def : WriteRes<WriteLoad,  [SKXPort23]> { let Latency = 5; }
def : WriteRes<WriteStore, [SKXPort237, SKXPort4]>;
def : WriteRes<WriteMove,  [SKXPort0156]>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
def : WriteRes<WriteZero,  []>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : SKXWriteResPair<WriteJump,  [SKXPort06],   1>;

// Floating point. This covers both scalar and vector operations.
def  : WriteRes<WriteFLoad,         [SKXPort23]> { let Latency = 5; }
def  : WriteRes<WriteFStore,        [SKXPort237, SKXPort4]>;
def  : WriteRes<WriteFMove,         [SKXPort015]>;
defm : X86WriteRes<WriteEMMS,       [SKXPort05,SKXPort0156], 10, [9,1], 10>;

defm : SKXWriteResPair<WriteFAdd, [SKXPort015],  4, [1], 1, 6>; // Floating point add/sub.
defm : SKXWriteResPair<WriteFAddY,[SKXPort015],  4, [1], 1, 7>; // Floating point add/sub (YMM/ZMM).
defm : SKXWriteResPair<WriteFCmp, [SKXPort015],  4, [1], 1, 6>; // Floating point compare.
defm : SKXWriteResPair<WriteFCmpY,[SKXPort015],  4, [1], 1, 7>; // Floating point compare (YMM/ZMM).
defm : SKXWriteResPair<WriteFCom,   [SKXPort0],  2>; // Floating point compare to flags.
defm : SKXWriteResPair<WriteFMul, [SKXPort015],  4, [1], 1, 6>; // Floating point multiplication.
defm : SKXWriteResPair<WriteFMulY,[SKXPort015],  4, [1], 1, 7>; // Floating point multiplication (YMM/ZMM).
defm : SKXWriteResPair<WriteFDiv,   [SKXPort0], 12, [1], 1, 5>; // 10-14 cycles. // Floating point division.
defm : SKXWriteResPair<WriteFDivY,  [SKXPort0], 12, [1], 1, 7>; // 10-14 cycles. // Floating point division (YMM/ZMM).
defm : SKXWriteResPair<WriteFSqrt,  [SKXPort0], 15, [1], 1, 5>; // Floating point square root.
defm : SKXWriteResPair<WriteFSqrtY, [SKXPort0], 15, [1], 1, 7>; // Floating point square root (YMM/ZMM).
defm : SKXWriteResPair<WriteFRcp,   [SKXPort0],  4, [1], 1, 6>; // Floating point reciprocal estimate.
defm : SKXWriteResPair<WriteFRcpY,  [SKXPort0],  4, [1], 1, 7>; // Floating point reciprocal estimate (YMM/ZMM).
defm : SKXWriteResPair<WriteFRsqrt, [SKXPort0],  4, [1], 1, 6>; // Floating point reciprocal square root estimate.
defm : SKXWriteResPair<WriteFRsqrtY,[SKXPort0],  4, [1], 1, 7>; // Floating point reciprocal square root estimate (YMM/ZMM).
defm : SKXWriteResPair<WriteFMA,  [SKXPort015],  4, [1], 1, 5>; // Fused Multiply Add.
defm : SKXWriteResPair<WriteFMAX, [SKXPort015],  4, [1], 1, 6>; // Fused Multiply Add (XMM).
defm : SKXWriteResPair<WriteFMAY, [SKXPort015],  4, [1], 1, 7>; // Fused Multiply Add (YMM/ZMM).
defm : SKXWriteResPair<WriteDPPD, [SKXPort5,SKXPort015],  9, [1,2], 3, 6>; // Floating point double dot product.
defm : SKXWriteResPair<WriteDPPS, [SKXPort5,SKXPort015], 13, [1,3], 4, 6>; // Floating point single dot product.
defm : SKXWriteResPair<WriteDPPSY,[SKXPort5,SKXPort015], 13, [1,3], 4, 7>; // Floating point single dot product (YMM).
defm : SKXWriteResPair<WriteFSign,  [SKXPort0],  1>; // Floating point fabs/fchs.
defm : SKXWriteResPair<WriteFRnd,   [SKXPort015], 8, [2], 2, 6>; // Floating point rounding.
defm : SKXWriteResPair<WriteFRndY,  [SKXPort015], 8, [2], 2, 7>; // Floating point rounding (YMM/ZMM).
defm : SKXWriteResPair<WriteFLogic, [SKXPort015], 1, [1], 1, 6>; // Floating point and/or/xor logicals.
defm : SKXWriteResPair<WriteFLogicY, [SKXPort015], 1, [1], 1, 7>; // Floating point and/or/xor logicals (YMM/ZMM).
defm : SKXWriteResPair<WriteFShuffle,  [SKXPort5], 1, [1], 1, 6>; // Floating point vector shuffles.
defm : SKXWriteResPair<WriteFShuffleY, [SKXPort5], 1, [1], 1, 7>; // Floating point vector shuffles (YMM/ZMM).
defm : SKXWriteResPair<WriteFVarShuffle,  [SKXPort5], 1, [1], 1, 6>; // Floating point vector variable shuffles.
defm : SKXWriteResPair<WriteFVarShuffleY, [SKXPort5], 1, [1], 1, 7>; // Floating point vector variable shuffles.
defm : SKXWriteResPair<WriteFBlend, [SKXPort015], 1, [1], 1, 6>; // Floating point vector blends.
defm : SKXWriteResPair<WriteFBlendY,[SKXPort015], 1, [1], 1, 7>; // Floating point vector blends.
defm : SKXWriteResPair<WriteFVarBlend, [SKXPort015], 2, [2], 2, 6>; // Fp vector variable blends.
defm : SKXWriteResPair<WriteFVarBlendY,[SKXPort015], 2, [2], 2, 7>; // Fp vector variable blends.

def  : WriteRes<WriteCvtF2FSt, [SKXPort4,SKXPort5,SKXPort237,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}

// FMA Scheduling helper class.
// class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Vector integer operations.
def  : WriteRes<WriteVecLoad,         [SKXPort23]> { let Latency = 5; }
def  : WriteRes<WriteVecStore,        [SKXPort237, SKXPort4]>;
def  : WriteRes<WriteVecMove,         [SKXPort015]>;

defm : SKXWriteResPair<WriteVecALU,   [SKXPort01], 1, [1], 1, 6>; // Vector integer ALU op, no logicals.
defm : SKXWriteResPair<WriteVecALUY,  [SKXPort01], 1, [1], 1, 7>; // Vector integer ALU op, no logicals (YMM/ZMM).
defm : SKXWriteResPair<WriteVecLogic, [SKXPort015], 1, [1], 1, 6>; // Vector integer and/or/xor.
defm : SKXWriteResPair<WriteVecLogicY,[SKXPort015], 1, [1], 1, 7>; // Vector integer and/or/xor (YMM/ZMM).
defm : SKXWriteResPair<WriteVecIMul,  [SKXPort0],    4, [1], 1, 5>; // Vector integer multiply.
defm : SKXWriteResPair<WriteVecIMulX, [SKXPort015],  4, [1], 1, 6>; // Vector integer multiply (XMM).
defm : SKXWriteResPair<WriteVecIMulY, [SKXPort015],  4, [1], 1, 7>; // Vector integer multiply (YMM/ZMM).
defm : SKXWriteResPair<WritePMULLD,   [SKXPort015], 10, [2], 2, 6>; // Vector PMULLD.
defm : SKXWriteResPair<WritePMULLDY,  [SKXPort015], 10, [2], 2, 7>; // Vector PMULLD (YMM/ZMM).
defm : SKXWriteResPair<WriteShuffle,  [SKXPort5], 1, [1], 1, 5>; // Vector shuffles.
defm : SKXWriteResPair<WriteShuffleY, [SKXPort5], 1, [1], 1, 7>; // Vector shuffles (YMM/ZMM).
defm : SKXWriteResPair<WriteVarShuffle,  [SKXPort5], 1, [1], 1, 6>; // Vector variable shuffles.
defm : SKXWriteResPair<WriteVarShuffleY, [SKXPort5], 1, [1], 1, 7>; // Vector variable shuffles (YMM/ZMM).
defm : SKXWriteResPair<WriteBlend, [SKXPort5], 1, [1], 1, 6>; // Vector blends.
defm : SKXWriteResPair<WriteBlendY,[SKXPort5], 1, [1], 1, 7>; // Vector blends (YMM/ZMM).
defm : SKXWriteResPair<WriteVarBlend, [SKXPort015], 2, [2], 2, 6>; // Vector variable blends.
defm : SKXWriteResPair<WriteVarBlendY,[SKXPort015], 2, [2], 2, 6>; // Vector variable blends (YMM/ZMM).
defm : SKXWriteResPair<WriteMPSAD,   [SKXPort5], 4, [2], 2, 6>; // Vector MPSAD.
defm : SKXWriteResPair<WriteMPSADY,  [SKXPort5], 4, [2], 2, 7>; // Vector MPSAD.
defm : SKXWriteResPair<WritePSADBW,  [SKXPort5], 3, [1], 1, 6>; // Vector PSADBW.
defm : SKXWriteResPair<WritePSADBWY, [SKXPort5], 3, [1], 1, 7>; // Vector PSADBW.
defm : SKXWriteResPair<WritePHMINPOS, [SKXPort015], 4, [1], 1, 6>; // Vector PHMINPOS.

// Vector integer shifts.
defm : SKXWriteResPair<WriteVecShift, [SKXPort0], 1, [1], 1, 5>;
defm : X86WriteRes<WriteVecShiftX,    [SKXPort5,SKXPort01],  2, [1,1], 2>;
defm : X86WriteRes<WriteVecShiftY,    [SKXPort5,SKXPort01],  4, [1,1], 2>;
defm : X86WriteRes<WriteVecShiftXLd,  [SKXPort01,SKXPort23], 7, [1,1], 2>;
defm : X86WriteRes<WriteVecShiftYLd,  [SKXPort01,SKXPort23], 8, [1,1], 2>;

defm : SKXWriteResPair<WriteVecShiftImm,  [SKXPort0],  1, [1], 1, 5>;
defm : SKXWriteResPair<WriteVecShiftImmX, [SKXPort01], 1, [1], 1, 6>; // Vector integer immediate shifts (XMM).
defm : SKXWriteResPair<WriteVecShiftImmY, [SKXPort01], 1, [1], 1, 7>; // Vector integer immediate shifts (YMM/ZMM).
defm : SKXWriteResPair<WriteVarVecShift,  [SKXPort01], 1, [1], 1, 6>; // Variable vector shifts.
defm : SKXWriteResPair<WriteVarVecShiftY, [SKXPort01], 1, [1], 1, 7>; // Variable vector shifts (YMM/ZMM).

// Vector insert/extract operations.
def : WriteRes<WriteVecInsert, [SKXPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteVecInsertLd, [SKXPort5,SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def: InstRW<[WriteVecInsertLd], (instregex "(V?)MOV(H|L)(PD|PS)rm")>;

def : WriteRes<WriteVecExtract, [SKXPort0,SKXPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : WriteRes<WriteVecExtractSt, [SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

// Conversion between integer and float.
defm : SKXWriteResPair<WriteCvtF2I, [SKXPort1], 3>; // Float -> Integer.
defm : SKXWriteResPair<WriteCvtI2F, [SKXPort1], 4>; // Integer -> Float.
defm : SKXWriteResPair<WriteCvtF2F, [SKXPort1], 3>; // Float -> Float size conversion.

// Strings instructions.

// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [SKXPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [SKXPort0, SKXPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [SKXPort0, SKXPort5, SKXPort015, SKXPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def : WriteRes<WritePCmpEStrMLd, [SKXPort0, SKXPort5, SKXPort23, SKXPort015, SKXPort0156]> {
  let Latency = 25;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [SKXPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [SKXPort0, SKXPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [SKXPort0,SKXPort5,SKXPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def : WriteRes<WritePCmpEStrILd, [SKXPort0, SKXPort5, SKXPort23, SKXPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}

// MOVMSK Instructions.
def : WriteRes<WriteFMOVMSK,    [SKXPort0]> { let Latency = 2; }
def : WriteRes<WriteVecMOVMSK,  [SKXPort0]> { let Latency = 2; }
def : WriteRes<WriteVecMOVMSKY, [SKXPort0]> { let Latency = 2; }
def : WriteRes<WriteMMXMOVMSK,  [SKXPort0]> { let Latency = 2; }

// AES instructions.
def : WriteRes<WriteAESDecEnc, [SKXPort0]> { // Decryption, encryption.
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [SKXPort0, SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

def : WriteRes<WriteAESIMC, [SKXPort0]> { // InvMixColumn.
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [SKXPort0, SKXPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}

def : WriteRes<WriteAESKeyGen, [SKXPort0,SKXPort5,SKXPort015]> { // Key Generation.
  let Latency = 20;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,2];
}
def : WriteRes<WriteAESKeyGenLd, [SKXPort0,SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 25;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,1,1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [SKXPort5]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteCLMulLd, [SKXPort5, SKXPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

// Catch-all for expensive system instructions.
def : WriteRes<WriteSystem,     [SKXPort0156]> { let Latency = 100; } // def WriteSystem : SchedWrite;

// AVX2.
defm : SKXWriteResPair<WriteFShuffle256, [SKXPort5], 3, [1], 1, 7>; // Fp 256-bit width vector shuffles.
defm : SKXWriteResPair<WriteFVarShuffle256, [SKXPort5], 3, [1], 1, 7>; // Fp 256-bit width vector variable shuffles.
defm : SKXWriteResPair<WriteShuffle256, [SKXPort5], 3, [1], 1, 7>;  // 256-bit width vector shuffles.
defm : SKXWriteResPair<WriteVarShuffle256, [SKXPort5], 3, [1], 1, 7>;  // 256-bit width vector variable shuffles.

// Old microcoded instructions that nobody use.
def : WriteRes<WriteMicrocoded, [SKXPort0156]> { let Latency = 100; } // def WriteMicrocoded : SchedWrite;

// Fence instructions.
def : WriteRes<WriteFence,  [SKXPort23, SKXPort4]>;

// Load/store MXCSR.
def : WriteRes<WriteLDMXCSR, [SKXPort0,SKXPort23,SKXPort0156]> { let Latency = 7; let NumMicroOps = 3; let ResourceCycles = [1,1,1]; }
def : WriteRes<WriteSTMXCSR, [SKXPort4,SKXPort5,SKXPort237]> { let Latency = 2; let NumMicroOps = 3; let ResourceCycles = [1,1,1]; }

// Nop, not very useful expect it provides a model for nops!
def : WriteRes<WriteNop, []>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : SKXWriteResPair<WriteFHAdd,  [SKXPort5,SKXPort015], 6, [2,1], 3, 6>;
defm : SKXWriteResPair<WriteFHAddY, [SKXPort5,SKXPort015], 6, [2,1], 3, 7>;
defm : SKXWriteResPair<WritePHAdd,  [SKXPort5,SKXPort015], 3, [2,1], 3, 6>;
defm : SKXWriteResPair<WritePHAddY, [SKXPort5,SKXPort015], 3, [2,1], 3, 7>;

// Remaining instrs.

def SKXWriteResGroup1 : SchedWriteRes<[SKXPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup1], (instregex "KANDBrr",
                                            "KANDDrr",
                                            "KANDNBrr",
                                            "KANDNDrr",
                                            "KANDNQrr",
                                            "KANDNWrr",
                                            "KANDQrr",
                                            "KANDWrr",
                                            "KMOVBkk",
                                            "KMOVDkk",
                                            "KMOVQkk",
                                            "KMOVWkk",
                                            "KNOTBrr",
                                            "KNOTDrr",
                                            "KNOTQrr",
                                            "KNOTWrr",
                                            "KORBrr",
                                            "KORDrr",
                                            "KORQrr",
                                            "KORWrr",
                                            "KXNORBrr",
                                            "KXNORDrr",
                                            "KXNORQrr",
                                            "KXNORWrr",
                                            "KXORBrr",
                                            "KXORDrr",
                                            "KXORQrr",
                                            "KXORWrr",
                                            "MMX_PADDSBirr",
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
                                            "MMX_PSUBUSWirr",
                                            "VPMOVB2MZ128rr",
                                            "VPMOVB2MZ256rr",
                                            "VPMOVB2MZrr",
                                            "VPMOVD2MZ128rr",
                                            "VPMOVD2MZ256rr",
                                            "VPMOVD2MZrr",
                                            "VPMOVQ2MZ128rr",
                                            "VPMOVQ2MZ256rr",
                                            "VPMOVQ2MZrr",
                                            "VPMOVW2MZ128rr",
                                            "VPMOVW2MZ256rr",
                                            "VPMOVW2MZrr")>;

def SKXWriteResGroup3 : SchedWriteRes<[SKXPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup3], (instregex "COMP_FST0r",
                                            "COM_FST0r",
                                            "KMOVBkr",
                                            "KMOVDkr",
                                            "KMOVQkr",
                                            "KMOVWkr",
                                            "MMX_MOVD64rr",
                                            "MMX_MOVD64to64rr",
                                            "MOV64toPQIrr",
                                            "MOVDI2PDIrr",
                                            "UCOM_FPr",
                                            "UCOM_Fr",
                                            "VMOV64toPQIZrr",
                                            "VMOV64toPQIrr",
                                            "VMOVDI2PDIZrr",
                                            "VMOVDI2PDIrr")>;

def SKXWriteResGroup4 : SchedWriteRes<[SKXPort6]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup4], (instregex "JMP(16|32|64)r")>;

def SKXWriteResGroup6 : SchedWriteRes<[SKXPort05]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup6], (instrs FINCSTP, FNOP)>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_MOVQ64rr",
                                            "MMX_PABS(B|D|W)rr",
                                            "MMX_PADD(B|D|Q|W)irr",
                                            "MMX_PANDNirr",
                                            "MMX_PANDirr",
                                            "MMX_PORirr",
                                            "MMX_PSIGN(B|D|W)rr",
                                            "MMX_PSUB(B|D|Q|W)irr",
                                            "MMX_PXORirr")>;

def SKXWriteResGroup7 : SchedWriteRes<[SKXPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup7], (instrs CDQ, CQO, CLAC, STAC)>;
def: InstRW<[SKXWriteResGroup7], (instregex "ADC(16|32|64)ri",
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

def SKXWriteResGroup8 : SchedWriteRes<[SKXPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup8], (instregex "ANDN(32|64)rr",
                                            "BLSI(32|64)rr",
                                            "BLSMSK(32|64)rr",
                                            "BLSR(32|64)rr")>;

def SKXWriteResGroup9 : SchedWriteRes<[SKXPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDMPDZ128rr",
                                            "VBLENDMPDZ256rr",
                                            "VBLENDMPDZrr",
                                            "VBLENDMPSZ128rr",
                                            "VBLENDMPSZ256rr",
                                            "VBLENDMPSZrr",
                                            "VPADDBYrr",
                                            "VPADDBZ128rr",
                                            "VPADDBZ256rr",
                                            "VPADDBZrr",
                                            "(V?)PADDBrr",
                                            "VPADDDYrr",
                                            "VPADDDZ128rr",
                                            "VPADDDZ256rr",
                                            "VPADDDZrr",
                                            "(V?)PADDDrr",
                                            "VPADDQYrr",
                                            "VPADDQZ128rr",
                                            "VPADDQZ256rr",
                                            "VPADDQZrr",
                                            "(V?)PADDQrr",
                                            "VPADDWYrr",
                                            "VPADDWZ128rr",
                                            "VPADDWZ256rr",
                                            "VPADDWZrr",
                                            "(V?)PADDWrr",
                                            "VPBLENDDYrri",
                                            "VPBLENDDrri",
                                            "VPBLENDMBZ128rr",
                                            "VPBLENDMBZ256rr",
                                            "VPBLENDMBZrr",
                                            "VPBLENDMDZ128rr",
                                            "VPBLENDMDZ256rr",
                                            "VPBLENDMDZrr",
                                            "VPBLENDMQZ128rr",
                                            "VPBLENDMQZ256rr",
                                            "VPBLENDMQZrr",
                                            "VPBLENDMWZ128rr",
                                            "VPBLENDMWZ256rr",
                                            "VPBLENDMWZrr",
                                            "VPSUBBYrr",
                                            "VPSUBBZ128rr",
                                            "VPSUBBZ256rr",
                                            "VPSUBBZrr",
                                            "(V?)PSUBBrr",
                                            "VPSUBDYrr",
                                            "VPSUBDZ128rr",
                                            "VPSUBDZ256rr",
                                            "VPSUBDZrr",
                                            "(V?)PSUBDrr",
                                            "VPSUBQYrr",
                                            "VPSUBQZ128rr",
                                            "VPSUBQZ256rr",
                                            "VPSUBQZrr",
                                            "(V?)PSUBQrr",
                                            "VPSUBWYrr",
                                            "VPSUBWZ128rr",
                                            "VPSUBWZrr",
                                            "(V?)PSUBWrr",
                                            "VPTERNLOGDZ128rri",
                                            "VPTERNLOGDZ256rri",
                                            "VPTERNLOGDZrri",
                                            "VPTERNLOGQZ128rri",
                                            "VPTERNLOGQZ256rri",
                                            "VPTERNLOGQZrri")>;

def SKXWriteResGroup10 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup10], (instrs CBW, CWDE, CDQE)>;
def: InstRW<[SKXWriteResGroup10], (instrs LAHF, SAHF)>; // TODO: This doesn't match Agner's data
def: InstRW<[SKXWriteResGroup10], (instregex "CLC",
                                             "CMC",
                                             "NOOP",
                                             "SGDT64m",
                                             "SIDT64m",
                                             "SMSW16m",
                                             "STC",
                                             "STRm",
                                             "SYSCALL")>;

def SKXWriteResGroup11 : SchedWriteRes<[SKXPort4,SKXPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup11], (instregex "FBSTPm",
                                             "KMOVBmk",
                                             "KMOVDmk",
                                             "KMOVQmk",
                                             "KMOVWmk",
                                             "MMX_MOVD64from64rm",
                                             "MMX_MOVD64mr",
                                             "MMX_MOVNTQmr",
                                             "MMX_MOVQ64mr",
                                             "MOVAPDmr",
                                             "MOVAPSmr",
                                             "MOVDQAmr",
                                             "MOVDQUmr",
                                             "MOVHPDmr",
                                             "MOVHPSmr",
                                             "MOVLPDmr",
                                             "MOVLPSmr",
                                             "MOVNTDQmr",
                                             "MOVNTI_64mr",
                                             "MOVNTImr",
                                             "MOVNTPDmr",
                                             "MOVNTPSmr",
                                             "MOVPDI2DImr",
                                             "MOVPQI2QImr",
                                             "MOVPQIto64mr",
                                             "MOVSDmr",
                                             "MOVSSmr",
                                             "MOVUPDmr",
                                             "MOVUPSmr",
                                             "ST_FP32m",
                                             "ST_FP64m",
                                             "ST_FP80m",
                                             "VEXTRACTF128mr",
                                             "VEXTRACTF32x4Z256mr(b?)",
                                             "VEXTRACTF32x4Zmr(b?)",
                                             "VEXTRACTF32x8Zmr(b?)",
                                             "VEXTRACTF64x2Z256mr(b?)",
                                             "VEXTRACTF64x2Zmr(b?)",
                                             "VEXTRACTF64x4Zmr(b?)",
                                             "VEXTRACTI128mr",
                                             "VEXTRACTI32x4Z256mr(b?)",
                                             "VEXTRACTI32x4Zmr(b?)",
                                             "VEXTRACTI32x8Zmr(b?)",
                                             "VEXTRACTI64x2Z256mr(b?)",
                                             "VEXTRACTI64x2Zmr(b?)",
                                             "VEXTRACTI64x4Zmr(b?)",
                                             "VMOVAPDYmr",
                                             "VMOVAPDZ128mr(b?)",
                                             "VMOVAPDZ256mr(b?)",
                                             "VMOVAPDZmr(b?)",
                                             "VMOVAPDmr",
                                             "VMOVAPSYmr",
                                             "VMOVAPSZ128mr(b?)",
                                             "VMOVAPSZ256mr(b?)",
                                             "VMOVAPSZmr(b?)",
                                             "VMOVAPSmr",
                                             "VMOVDQA32Z128mr(b?)",
                                             "VMOVDQA32Z256mr(b?)",
                                             "VMOVDQA32Zmr(b?)",
                                             "VMOVDQA64Z128mr(b?)",
                                             "VMOVDQA64Z256mr(b?)",
                                             "VMOVDQA64Zmr(b?)",
                                             "VMOVDQAYmr",
                                             "VMOVDQAmr",
                                             "VMOVDQU16Z128mr(b?)",
                                             "VMOVDQU16Z256mr(b?)",
                                             "VMOVDQU16Zmr(b?)",
                                             "VMOVDQU32Z128mr(b?)",
                                             "VMOVDQU32Z256mr(b?)",
                                             "VMOVDQU32Zmr(b?)",
                                             "VMOVDQU64Z128mr(b?)",
                                             "VMOVDQU64Z256mr(b?)",
                                             "VMOVDQU64Zmr(b?)",
                                             "VMOVDQU8Z128mr(b?)",
                                             "VMOVDQU8Z256mr(b?)",
                                             "VMOVDQUYmr",
                                             "VMOVDQUmr",
                                             "VMOVHPDZ128mr(b?)",
                                             "VMOVHPDmr",
                                             "VMOVHPSZ128mr(b?)",
                                             "VMOVHPSmr",
                                             "VMOVLPDZ128mr(b?)",
                                             "VMOVLPDmr",
                                             "VMOVLPSZ128mr(b?)",
                                             "VMOVLPSmr",
                                             "VMOVNTDQYmr",
                                             "VMOVNTDQZ128mr(b?)",
                                             "VMOVNTDQZ256mr(b?)",
                                             "VMOVNTDQZmr(b?)",
                                             "VMOVNTDQmr",
                                             "VMOVNTPDYmr",
                                             "VMOVNTPDZ128mr(b?)",
                                             "VMOVNTPDZ256mr(b?)",
                                             "VMOVNTPDZmr(b?)",
                                             "VMOVNTPDmr",
                                             "VMOVNTPSYmr",
                                             "VMOVNTPSZ128mr(b?)",
                                             "VMOVNTPSZ256mr(b?)",
                                             "VMOVNTPSZmr(b?)",
                                             "VMOVNTPSmr",
                                             "VMOVPDI2DIZmr(b?)",
                                             "VMOVPDI2DImr",
                                             "VMOVPQI(2QI|to64)Zmr(b?)",
                                             "VMOVPQI2QImr",
                                             "VMOVPQIto64mr",
                                             "VMOVSDZmr(b?)",
                                             "VMOVSDmr",
                                             "VMOVSSZmr(b?)",
                                             "VMOVSSmr",
                                             "VMOVUPDYmr",
                                             "VMOVUPDZ128mr(b?)",
                                             "VMOVUPDZ256mr(b?)",
                                             "VMOVUPDZmr(b?)",
                                             "VMOVUPDmr",
                                             "VMOVUPSYmr",
                                             "VMOVUPSZ128mr(b?)",
                                             "VMOVUPSZ256mr(b?)",
                                             "VMOVUPSZmr(b?)",
                                             "VMOVUPSmr",
                                             "VMPTRSTm")>;

def SKXWriteResGroup12 : SchedWriteRes<[SKXPort0]> {
  let Latency = 2;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup12], (instregex "MMX_MOVD64from64rr",
                                             "MMX_MOVD64grr",
                                             "MOVPDI2DIrr",
                                             "MOVPQIto64rr",
                                             "VMOVPDI2DIZrr",
                                             "VMOVPDI2DIrr",
                                             "VMOVPQIto64Zrr",
                                             "VMOVPQIto64rr",
                                             "VTESTPDYrr",
                                             "VTESTPDrr",
                                             "VTESTPSYrr",
                                             "VTESTPSrr")>;

def SKXWriteResGroup13 : SchedWriteRes<[SKXPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup13], (instregex "MMX_MOVQ2DQrr")>;

def SKXWriteResGroup14 : SchedWriteRes<[SKXPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup14], (instrs FDECSTP)>;
def: InstRW<[SKXWriteResGroup14], (instregex "MMX_MOVDQ2Qrr")>;

def SKXWriteResGroup15 : SchedWriteRes<[SKXPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup15], (instregex "CMOV(A|BE)(16|32|64)rr",
                                             "ROL(8|16|32|64)r1",
                                             "ROL(8|16|32|64)ri",
                                             "ROR(8|16|32|64)r1",
                                             "ROR(8|16|32|64)ri",
                                             "SET(A|BE)r")>;

def SKXWriteResGroup17 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup17], (instrs LFENCE,
                                          WAIT,
                                          XGETBV)>;

def SKXWriteResGroup18 : SchedWriteRes<[SKXPort0,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup18], (instregex "VMASKMOVPDYmr",
                                             "VMASKMOVPDmr",
                                             "VMASKMOVPSYmr",
                                             "VMASKMOVPSmr",
                                             "VPMASKMOVDYmr",
                                             "VPMASKMOVDmr",
                                             "VPMASKMOVQYmr",
                                             "VPMASKMOVQmr")>;

def SKXWriteResGroup20 : SchedWriteRes<[SKXPort6,SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup20], (instregex "CLFLUSH")>;

def SKXWriteResGroup21 : SchedWriteRes<[SKXPort237,SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup21], (instregex "SFENCE")>;

def SKXWriteResGroup22 : SchedWriteRes<[SKXPort06,SKXPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup22], (instrs BSWAP64r)>;

def SKXWriteResGroup22_1 : SchedWriteRes<[SKXPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup22_1], (instrs BSWAP32r)>;

def SKXWriteResGroup23 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup23], (instrs CWD)>;
def: InstRW<[SKXWriteResGroup23], (instrs JCXZ, JECXZ, JRCXZ)>;
def: InstRW<[SKXWriteResGroup23], (instregex "ADC8i8",
                                             "ADC8ri",
                                             "SBB8i8",
                                             "SBB8ri")>;

def SKXWriteResGroup25 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup25], (instregex "FNSTCW16m")>;

def SKXWriteResGroup27 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort15]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup27], (instregex "MOVBE(16|32|64)mr")>;

def SKXWriteResGroup28 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup28], (instrs PUSH16r, PUSH32r, PUSH64r,
                                          STOSB, STOSL, STOSQ, STOSW)>;
def: InstRW<[SKXWriteResGroup28], (instregex "PUSH(16|32|64)rmr",
                                             "PUSH64i8")>;

def SKXWriteResGroup29 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort15]> {
  let Latency = 2;
  let NumMicroOps = 5;
  let ResourceCycles = [2,2,1];
}
def: InstRW<[SKXWriteResGroup29], (instregex "VMOVDQU8Zmr(b?)")>;

def SKXWriteResGroup30 : SchedWriteRes<[SKXPort0]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup30], (instregex "KADDBrr",
                                             "KADDDrr",
                                             "KADDQrr",
                                             "KADDWrr",
                                             "KMOVBrk",
                                             "KMOVDrk",
                                             "KMOVQrk",
                                             "KMOVWrk",
                                             "KORTESTBrr",
                                             "KORTESTDrr",
                                             "KORTESTQrr",
                                             "KORTESTWrr",
                                             "KTESTBrr",
                                             "KTESTDrr",
                                             "KTESTQrr",
                                             "KTESTWrr")>;

def SKXWriteResGroup31 : SchedWriteRes<[SKXPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup31], (instregex "CMOV(N?)(B|BE|E|P)_F",
                                             "PDEP(32|64)rr",
                                             "PEXT(32|64)rr",
                                             "SHLD(16|32|64)rri8",
                                             "SHRD(16|32|64)rri8")>;

def SKXWriteResGroup31_16i : SchedWriteRes<[SKXPort1, SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup31_16i], (instrs IMUL16rri, IMUL16rri8)>;


def SKXWriteResGroup32 : SchedWriteRes<[SKXPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup32], (instregex "(ADD|SUB|SUBR)_FPrST0",
                                             "(ADD|SUB|SUBR)_FST0r",
                                             "(ADD|SUB|SUBR)_FrST0",
                                             "KSHIFTLBri",
                                             "KSHIFTLDri",
                                             "KSHIFTLQri",
                                             "KSHIFTLWri",
                                             "KSHIFTRBri",
                                             "KSHIFTRDri",
                                             "KSHIFTRQri",
                                             "KSHIFTRWri",
                                             "KUNPCKBWrr",
                                             "KUNPCKDQrr",
                                             "KUNPCKWDrr",
                                             "VALIGNDZ128rri",
                                             "VALIGNDZ256rri",
                                             "VALIGNDZrri",
                                             "VALIGNQZ128rri",
                                             "VALIGNQZ256rri",
                                             "VALIGNQZrri",
                                             "VCMPPDZ128rri",
                                             "VCMPPDZ256rri",
                                             "VCMPPDZrri",
                                             "VCMPPSZ128rri",
                                             "VCMPPSZ256rri",
                                             "VCMPPSZrri",
                                             "VCMPSDZrr",
                                             "VCMPSSZrr",
                                             "VDBPSADBWZrri", // TODO: 512-bit ops require ports 0/1 to be joined.
                                             "VFPCLASSPDZ128rr",
                                             "VFPCLASSPDZ256rr",
                                             "VFPCLASSPDZrr",
                                             "VFPCLASSPSZ128rr",
                                             "VFPCLASSPSZ256rr",
                                             "VFPCLASSPSZrr",
                                             "VFPCLASSSDrr",
                                             "VFPCLASSSSrr",
                                             "VPBROADCASTBrr",
                                             "VPBROADCASTWrr",
                                             "VPCMPBZ128rri",
                                             "VPCMPBZ256rri",
                                             "VPCMPBZrri",
                                             "VPCMPDZ128rri",
                                             "VPCMPDZ256rri",
                                             "VPCMPDZrri",
                                             "VPCMPEQBZ128rr",
                                             "VPCMPEQBZ256rr",
                                             "VPCMPEQBZrr",
                                             "VPCMPEQDZ128rr",
                                             "VPCMPEQDZ256rr",
                                             "VPCMPEQDZrr",
                                             "VPCMPEQQZ128rr",
                                             "VPCMPEQQZ256rr",
                                             "VPCMPEQQZrr",
                                             "VPCMPEQWZ128rr",
                                             "VPCMPEQWZ256rr",
                                             "VPCMPEQWZrr",
                                             "VPCMPGTBZ128rr",
                                             "VPCMPGTBZ256rr",
                                             "VPCMPGTBZrr",
                                             "VPCMPGTDZ128rr",
                                             "VPCMPGTDZ256rr",
                                             "VPCMPGTDZrr",
                                             "VPCMPGTQYrr",
                                             "VPCMPGTQZ128rr",
                                             "VPCMPGTQZ256rr",
                                             "VPCMPGTQZrr",
                                             "(V?)PCMPGTQrr",
                                             "VPCMPGTWZ128rr",
                                             "VPCMPGTWZ256rr",
                                             "VPCMPGTWZrr",
                                             "VPCMPQZ128rri",
                                             "VPCMPQZ256rri",
                                             "VPCMPQZrri",
                                             "VPCMPUBZ128rri",
                                             "VPCMPUBZ256rri",
                                             "VPCMPUBZrri",
                                             "VPCMPUDZ128rri",
                                             "VPCMPUDZ256rri",
                                             "VPCMPUDZrri",
                                             "VPCMPUQZ128rri",
                                             "VPCMPUQZ256rri",
                                             "VPCMPUQZrri",
                                             "VPCMPUWZ128rri",
                                             "VPCMPUWZ256rri",
                                             "VPCMPUWZrri",
                                             "VPCMPWZ128rri",
                                             "VPCMPWZ256rri",
                                             "VPCMPWZrri",
                                             "VPMAXSQZ128rr",
                                             "VPMAXSQZ256rr",
                                             "VPMAXSQZrr",
                                             "VPMAXUQZ128rr",
                                             "VPMAXUQZ256rr",
                                             "VPMAXUQZrr",
                                             "VPMINSQZ128rr",
                                             "VPMINSQZ256rr",
                                             "VPMINSQZrr",
                                             "VPMINUQZ128rr",
                                             "VPMINUQZ256rr",
                                             "VPMINUQZrr",
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
                                             "VPMOVZXWQYrr",
                                             "VPSADBWZrr", // TODO: 512-bit ops require ports 0/1 to be joined.
                                             "VPTESTMBZ128rr",
                                             "VPTESTMBZ256rr",
                                             "VPTESTMBZrr",
                                             "VPTESTMDZ128rr",
                                             "VPTESTMDZ256rr",
                                             "VPTESTMDZrr",
                                             "VPTESTMQZ128rr",
                                             "VPTESTMQZ256rr",
                                             "VPTESTMQZrr",
                                             "VPTESTMWZ128rr",
                                             "VPTESTMWZ256rr",
                                             "VPTESTMWZrr",
                                             "VPTESTNMBZ128rr",
                                             "VPTESTNMBZ256rr",
                                             "VPTESTNMBZrr",
                                             "VPTESTNMDZ128rr",
                                             "VPTESTNMDZ256rr",
                                             "VPTESTNMDZrr",
                                             "VPTESTNMQZ128rr",
                                             "VPTESTNMQZ256rr",
                                             "VPTESTNMQZrr",
                                             "VPTESTNMWZ128rr",
                                             "VPTESTNMWZ256rr",
                                             "VPTESTNMWZrr")>;

def SKXWriteResGroup33 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup33], (instregex "(V?)PTEST(Y?)rr")>;

def SKXWriteResGroup34 : SchedWriteRes<[SKXPort0,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup34], (instregex "FNSTSW16r")>;

def SKXWriteResGroup35 : SchedWriteRes<[SKXPort06]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKXWriteResGroup35], (instregex "ROL(8|16|32|64)rCL",
                                             "ROR(8|16|32|64)rCL",
                                             "SAR(8|16|32|64)rCL",
                                             "SHL(8|16|32|64)rCL",
                                             "SHR(8|16|32|64)rCL")>;

def SKXWriteResGroup36 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKXWriteResGroup36], (instrs XADD8rr, XADD16rr, XADD32rr, XADD64rr,
                                          XCHG8rr, XCHG16rr, XCHG32rr, XCHG64rr,
                                          XCHG16ar, XCHG32ar, XCHG64ar)>;

def SKXWriteResGroup37 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup37], (instregex "MMX_PH(ADD|SUB)SWrr")>;

def SKXWriteResGroup38 : SchedWriteRes<[SKXPort5,SKXPort01]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup38], (instregex "(V?)PH(ADD|SUB)SW(Y?)rr")>;

def SKXWriteResGroup39 : SchedWriteRes<[SKXPort5,SKXPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup39], (instregex "MMX_PH(ADD|SUB)(D|W)rr")>;

def SKXWriteResGroup41 : SchedWriteRes<[SKXPort5,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup41], (instregex "MMX_PACKSSDWirr",
                                             "MMX_PACKSSWBirr",
                                             "MMX_PACKUSWBirr")>;

def SKXWriteResGroup42 : SchedWriteRes<[SKXPort6,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup42], (instregex "CLD")>;

def SKXWriteResGroup43 : SchedWriteRes<[SKXPort237,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup43], (instrs MFENCE)>;

def SKXWriteResGroup44 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup44], (instregex "RCL(8|16|32|64)r1",
                                             "RCL(8|16|32|64)ri",
                                             "RCR(8|16|32|64)r1",
                                             "RCR(8|16|32|64)ri")>;

def SKXWriteResGroup45 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup45], (instregex "FNSTSWm")>;

def SKXWriteResGroup46 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort06]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKXWriteResGroup46], (instregex "SET(A|BE)m")>;

def SKXWriteResGroup47 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort237,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup47], (instregex "CALL(16|32|64)r")>;

def SKXWriteResGroup48 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup48], (instregex "CALL64pcrel32")>;

def SKXWriteResGroup49 : SchedWriteRes<[SKXPort0]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup49], (instregex "MUL_FPrST0",
                                             "MUL_FST0r",
                                             "MUL_FrST0")>;

def SKXWriteResGroup50 : SchedWriteRes<[SKXPort015]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTDQ2PSYrr",
                                             "VCVTDQ2PSZ128rr",
                                             "VCVTDQ2PSZ256rr",
                                             "VCVTDQ2PSZrr",
                                             "(V?)CVTDQ2PSrr",
                                             "VCVTPD2QQZ128rr",
                                             "VCVTPD2QQZ256rr",
                                             "VCVTPD2QQZrr",
                                             "VCVTPD2UQQZ128rr",
                                             "VCVTPD2UQQZ256rr",
                                             "VCVTPD2UQQZrr",
                                             "VCVTPS2DQYrr",
                                             "VCVTPS2DQZ128rr",
                                             "VCVTPS2DQZ256rr",
                                             "VCVTPS2DQZrr",
                                             "(V?)CVTPS2DQrr",
                                             "VCVTPS2UDQZ128rr",
                                             "VCVTPS2UDQZ256rr",
                                             "VCVTPS2UDQZrr",
                                             "VCVTQQ2PDZ128rr",
                                             "VCVTQQ2PDZ256rr",
                                             "VCVTQQ2PDZrr",
                                             "VCVTTPD2QQZ128rr",
                                             "VCVTTPD2QQZ256rr",
                                             "VCVTTPD2QQZrr",
                                             "VCVTTPD2UQQZ128rr",
                                             "VCVTTPD2UQQZ256rr",
                                             "VCVTTPD2UQQZrr",
                                             "VCVTTPS2DQYrr",
                                             "VCVTTPS2DQZ128rr",
                                             "VCVTTPS2DQZ256rr",
                                             "VCVTTPS2DQZrr",
                                             "(V?)CVTTPS2DQrr",
                                             "VCVTTPS2UDQZ128rr",
                                             "VCVTTPS2UDQZ256rr",
                                             "VCVTTPS2UDQZrr",
                                             "VCVTUDQ2PSZ128rr",
                                             "VCVTUDQ2PSZ256rr",
                                             "VCVTUDQ2PSZrr",
                                             "VCVTUQQ2PDZ128rr",
                                             "VCVTUQQ2PDZ256rr",
                                             "VCVTUQQ2PDZrr")>;

def SKXWriteResGroup51 : SchedWriteRes<[SKXPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup51], (instregex "VEXPANDPDZ128rr",
                                             "VEXPANDPDZ256rr",
                                             "VEXPANDPDZrr",
                                             "VEXPANDPSZ128rr",
                                             "VEXPANDPSZ256rr",
                                             "VEXPANDPSZrr",
                                             "VPEXPANDDZ128rr",
                                             "VPEXPANDDZ256rr",
                                             "VPEXPANDDZrr",
                                             "VPEXPANDQZ128rr",
                                             "VPEXPANDQZ256rr",
                                             "VPEXPANDQZrr",
                                             "VPMOVDBZ128rr",
                                             "VPMOVDBZ256rr",
                                             "VPMOVDBZrr",
                                             "VPMOVDWZ128rr",
                                             "VPMOVDWZ256rr",
                                             "VPMOVDWZrr",
                                             "VPMOVQBZ128rr",
                                             "VPMOVQBZ256rr",
                                             "VPMOVQBZrr",
                                             "VPMOVQWZ128rr",
                                             "VPMOVQWZ256rr",
                                             "VPMOVQWZrr",
                                             "VPMOVSDBZ128rr",
                                             "VPMOVSDBZ256rr",
                                             "VPMOVSDBZrr",
                                             "VPMOVSDWZ128rr",
                                             "VPMOVSDWZ256rr",
                                             "VPMOVSDWZrr",
                                             "VPMOVSQBZ128rr",
                                             "VPMOVSQBZ256rr",
                                             "VPMOVSQBZrr",
                                             "VPMOVSQDZ128rr",
                                             "VPMOVSQDZ256rr",
                                             "VPMOVSQDZrr",
                                             "VPMOVSQWZ128rr",
                                             "VPMOVSQWZ256rr",
                                             "VPMOVSQWZrr",
                                             "VPMOVSWBZ128rr",
                                             "VPMOVSWBZ256rr",
                                             "VPMOVSWBZrr",
                                             "VPMOVUSDBZ128rr",
                                             "VPMOVUSDBZ256rr",
                                             "VPMOVUSDBZrr",
                                             "VPMOVUSDWZ128rr",
                                             "VPMOVUSDWZ256rr",
                                             "VPMOVUSDWZrr",
                                             "VPMOVUSQBZ128rr",
                                             "VPMOVUSQBZ256rr",
                                             "VPMOVUSQBZrr",
                                             "VPMOVUSQDZ128rr",
                                             "VPMOVUSQDZ256rr",
                                             "VPMOVUSQDZrr",
                                             "VPMOVUSQWZ128rr",
                                             "VPMOVUSQWZ256rr",
                                             "VPMOVUSQWZrr",
                                             "VPMOVUSWBZ128rr",
                                             "VPMOVUSWBZ256rr",
                                             "VPMOVUSWBZrr",
                                             "VPMOVWBZ128rr",
                                             "VPMOVWBZ256rr",
                                             "VPMOVWBZrr")>;

def SKXWriteResGroup52 : SchedWriteRes<[SKXPort1,SKXPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup52], (instrs IMUL64r, MUL64r, MULX64rr)>;

def SKXWriteResGroup52_16 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKXWriteResGroup52_16], (instrs IMUL16r, MUL16r)>;

def SKXWriteResGroup54 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup54], (instregex "IST(T?)_FP(16|32|64)m",
                                             "IST_F(16|32)m",
                                             "VPMOVQDZ128mr(b?)",
                                             "VPMOVQDZ256mr(b?)",
                                             "VPMOVQDZmr(b?)")>;

def SKXWriteResGroup55 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[SKXWriteResGroup55], (instrs FNCLEX)>;

def SKXWriteResGroup56 : SchedWriteRes<[SKXPort015,SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup56], (instrs VZEROUPPER)>;

def SKXWriteResGroup57 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKXWriteResGroup57], (instregex "LAR(16|32|64)rr")>;

def SKXWriteResGroup58 : SchedWriteRes<[SKXPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup58], (instregex "MOVSX(16|32|64)rm16",
                                             "MOVSX(16|32|64)rm32",
                                             "MOVSX(16|32|64)rm8",
                                             "MOVZX(16|32|64)rm16",
                                             "MOVZX(16|32|64)rm8",
                                             "(V?)MOVDDUPrm")>;  // TODO: Should this be SKXWriteResGroup71?

def SKXWriteResGroup59 : SchedWriteRes<[SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup59], (instregex "VCVTSD2SSZrr")>;

def SKXWriteResGroup60 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup60], (instregex "(V?)CVTDQ2PDrr",
                                             "MMX_CVTPI2PDirr")>;

def SKXWriteResGroup61 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup61], (instregex "CVTPD2DQrr",
                                             "CVTPD2PSrr",
                                             "CVTPS2PDrr",
                                             "CVTSD2SSrr",
                                             "CVTSI642SDrr",
                                             "CVTSI2SDrr",
                                             "CVTSI2SSrr",
                                             "CVTSS2SDrr",
                                             "CVTTPD2DQrr",
                                             "MMX_CVTPD2PIirr",
                                             "MMX_CVTPS2PIirr",
                                             "MMX_CVTTPD2PIirr",
                                             "MMX_CVTTPS2PIirr",
                                             "VCVTDQ2PDZ128rr",
                                             "VCVTPD2DQZ128rr",
                                             "VCVTPD2DQrr",
                                             "VCVTPD2PSZ128rr",
                                             "VCVTPD2PSrr",
                                             "VCVTPD2UDQZ128rr",
                                             "VCVTPH2PSZ128rr",
                                             "VCVTPH2PSrr",
                                             "VCVTPS2PDZ128rr",
                                             "VCVTPS2PDrr",
                                             "VCVTPS2PHZ128rr",
                                             "VCVTPS2PHrr",
                                             "VCVTPS2QQZ128rr",
                                             "VCVTPS2UQQZ128rr",
                                             "VCVTQQ2PSZ128rr",
                                             "VCVTSD2SSrr",
                                             "VCVTSI642SDrr",
                                             "VCVTSI2SDZrr",
                                             "VCVTSI2SDrr",
                                             "VCVTSI2SSZrr",
                                             "VCVTSI2SSrr",
                                             "VCVTSI642SDZrr",
                                             "VCVTSS2SDZrr",
                                             "VCVTSS2SDrr",
                                             "VCVTTPD2DQZ128rr",
                                             "VCVTTPD2DQrr",
                                             "VCVTTPD2UDQZ128rr",
                                             "VCVTTPS2QQZ128rr",
                                             "VCVTTPS2UQQZ128rr",
                                             "VCVTUDQ2PDZ128rr",
                                             "VCVTUQQ2PSZ128rr",
                                             "VCVTUSI2SDZrr",
                                             "VCVTUSI2SSZrr",
                                             "VCVTUSI642SDZrr")>;

def SKXWriteResGroup62 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup62], (instregex "VPCONFLICTQZ128rr")>;

def SKXWriteResGroup63 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup63], (instregex "STR(16|32|64)r")>;

def SKXWriteResGroup64 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup64], (instrs IMUL32r, MUL32r, MULX32rr)>;

def SKXWriteResGroup65 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup65], (instregex "VCVTPS2PHZ128mr(b?)",
                                             "VCVTPS2PHZ256mr(b?)",
                                             "VCVTPS2PHZmr(b?)")>;

def SKXWriteResGroup66 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVDBZ128mr(b?)",
                                             "VPMOVDBZ256mr(b?)",
                                             "VPMOVDBZmr(b?)",
                                             "VPMOVDWZ128mr(b?)",
                                             "VPMOVDWZ256mr(b?)",
                                             "VPMOVDWZmr(b?)",
                                             "VPMOVQBZ128mr(b?)",
                                             "VPMOVQBZ256mr(b?)",
                                             "VPMOVQBZmr(b?)",
                                             "VPMOVQWZ128mr(b?)",
                                             "VPMOVQWZ256mr(b?)",
                                             "VPMOVQWZmr(b?)",
                                             "VPMOVSDBZ128mr(b?)",
                                             "VPMOVSDBZ256mr(b?)",
                                             "VPMOVSDBZmr(b?)",
                                             "VPMOVSDWZ128mr(b?)",
                                             "VPMOVSDWZ256mr(b?)",
                                             "VPMOVSDWZmr(b?)",
                                             "VPMOVSQBZ128mr(b?)",
                                             "VPMOVSQBZ256mr(b?)",
                                             "VPMOVSQBZmr(b?)",
                                             "VPMOVSQDZ128mr(b?)",
                                             "VPMOVSQDZ256mr(b?)",
                                             "VPMOVSQDZmr(b?)",
                                             "VPMOVSQWZ128mr(b?)",
                                             "VPMOVSQWZ256mr(b?)",
                                             "VPMOVSQWZmr(b?)",
                                             "VPMOVSWBZ128mr(b?)",
                                             "VPMOVSWBZ256mr(b?)",
                                             "VPMOVSWBZmr(b?)",
                                             "VPMOVUSDBZ128mr(b?)",
                                             "VPMOVUSDBZ256mr(b?)",
                                             "VPMOVUSDBZmr(b?)",
                                             "VPMOVUSDWZ128mr(b?)",
                                             "VPMOVUSDWZ256mr(b?)",
                                             "VPMOVUSDWZmr(b?)",
                                             "VPMOVUSQBZ128mr(b?)",
                                             "VPMOVUSQBZ256mr(b?)",
                                             "VPMOVUSQBZmr(b?)",
                                             "VPMOVUSQDZ128mr(b?)",
                                             "VPMOVUSQDZ256mr(b?)",
                                             "VPMOVUSQDZmr(b?)",
                                             "VPMOVUSQWZ128mr(b?)",
                                             "VPMOVUSQWZ256mr(b?)",
                                             "VPMOVUSQWZmr(b?)",
                                             "VPMOVUSWBZ128mr(b?)",
                                             "VPMOVUSWBZ256mr(b?)",
                                             "VPMOVUSWBZmr(b?)",
                                             "VPMOVWBZ128mr(b?)",
                                             "VPMOVWBZ256mr(b?)",
                                             "VPMOVWBZmr(b?)")>;

def SKXWriteResGroup67 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[SKXWriteResGroup67], (instregex "XSETBV")>;

def SKXWriteResGroup68 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[SKXWriteResGroup68], (instregex "CMPXCHG(8|16|32|64)rr")>;

def SKXWriteResGroup69 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKXWriteResGroup69], (instregex "PUSHF16",
                                             "PUSHF64")>;

def SKXWriteResGroup71 : SchedWriteRes<[SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup71], (instregex "LDDQUrm",
                                             "MOVAPDrm",
                                             "MOVAPSrm",
                                             "MOVDQArm",
                                             "MOVDQUrm",
                                             "MOVNTDQArm",
                                             "MOVSHDUPrm",
                                             "MOVSLDUPrm",
                                             "MOVUPDrm",
                                             "MOVUPSrm",
                                             "VBROADCASTSSrm",
                                             "VLDDQUrm",
                                             "VMOVAPDrm",
                                             "VMOVAPSrm",
                                             "VMOVDQArm",
                                             "VMOVDQUrm",
                                             "VMOVNTDQArm",
                                             "VMOVSHDUPrm",
                                             "VMOVSLDUPrm",
                                             "VMOVUPDrm",
                                             "VMOVUPSrm",
                                             "VPBROADCASTDrm",
                                             "VPBROADCASTQrm")>;

def SKXWriteResGroup72 : SchedWriteRes<[SKXPort0]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup72], (instregex "MMX_CVTPI2PSirr",
                                             "VCOMPRESSPDZ128rr",
                                             "VCOMPRESSPDZ256rr",
                                             "VCOMPRESSPDZrr",
                                             "VCOMPRESSPSZ128rr",
                                             "VCOMPRESSPSZ256rr",
                                             "VCOMPRESSPSZrr",
                                             "VPCOMPRESSDZ128rr",
                                             "VPCOMPRESSDZ256rr",
                                             "VPCOMPRESSDZrr",
                                             "VPCOMPRESSQZ128rr",
                                             "VPCOMPRESSQZ256rr",
                                             "VPCOMPRESSQZrr",
                                             "VPERMWZ128rr",
                                             "VPERMWZ256rr",
                                             "VPERMWZrr")>;

def SKXWriteResGroup73 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PADDSBirm",
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

def SKXWriteResGroup74 : SchedWriteRes<[SKXPort0,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup74], (instregex "CVTSD2SI64rr",
                                             "CVTSD2SIrr",
                                             "CVTSS2SI64rr",
                                             "CVTSS2SIrr",
                                             "CVTTSD2SI64rr",
                                             "CVTTSD2SIrr",
                                             "VCVTSD2SI64Zrr",
                                             "VCVTSD2SI64rr",
                                             "VCVTSD2SIZrr",
                                             "VCVTSD2SIrr",
                                             "VCVTSD2USI64Zrr",
                                             "VCVTSD2USIZrr",
                                             "VCVTSS2SI64Zrr",
                                             "VCVTSS2SI64rr",
                                             "VCVTSS2SIZrr",
                                             "VCVTSS2SIrr",
                                             "VCVTSS2USIZrr",
                                             "VCVTTSD2SI64Zrr(b?)",
                                             "VCVTTSD2SI64rr",
                                             "VCVTTSD2SIZrr(b?)",
                                             "VCVTTSD2SIrr",
                                             "VCVTTSD2USI64Zrr(b?)",
                                             "VCVTTSD2USIZrr(b?)",
                                             "VCVTTSS2USIZrr(b?)")>;

def SKXWriteResGroup76 : SchedWriteRes<[SKXPort6,SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup76], (instregex "FARJMP64",
                                             "JMP(16|32|64)m")>;

def SKXWriteResGroup77 : SchedWriteRes<[SKXPort23,SKXPort05]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PABS(B|D|W)rm",
                                             "MMX_PADD(B|D|Q|W)irm",
                                             "MMX_PANDNirm",
                                             "MMX_PANDirm",
                                             "MMX_PORirm",
                                             "MMX_PSIGN(B|D|W)rm",
                                             "MMX_PSUB(B|D|Q|W)irm",
                                             "MMX_PXORirm")>;

def SKXWriteResGroup78 : SchedWriteRes<[SKXPort23,SKXPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup78], (instregex "BT(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup78, ReadAfterLd], (instrs ADC8rm, ADC16rm, ADC32rm, ADC64rm,
                                                       ADCX32rm, ADCX64rm,
                                                       ADOX32rm, ADOX64rm,
                                                       SBB8rm, SBB16rm, SBB32rm, SBB64rm)>;

def SKXWriteResGroup79 : SchedWriteRes<[SKXPort23,SKXPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup79], (instregex "ANDN(32|64)rm",
                                             "BLSI(32|64)rm",
                                             "BLSMSK(32|64)rm",
                                             "BLSR(32|64)rm",
                                             "MOVBE(16|32|64)rm")>;

def SKXWriteResGroup80 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup80], (instregex "VMOV(64to|QI2)PQIZrm(b?)",
                                             "VMOVDI2PDIZrm(b?)")>;

def SKXWriteResGroup81 : SchedWriteRes<[SKXPort23,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup81], (instrs POP16r, POP32r, POP64r)>;
def: InstRW<[SKXWriteResGroup81], (instregex "POP(16|32|64)rmr")>;

def SKXWriteResGroup82 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup82], (instregex "CVTSI642SSrr",
                                             "VCVTSI642SSrr",
                                             "VCVTSI642SSZrr",
                                             "VCVTUSI642SSZrr")>;

def SKXWriteResGroup83 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup83], (instregex "SHLD(16|32|64)rrCL",
                                             "SHRD(16|32|64)rrCL")>;

def SKXWriteResGroup84 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort06,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup84], (instregex "SLDT(16|32|64)r")>;

def SKXWriteResGroup86 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup86], (instregex "BTC(16|32|64)mi8",
                                             "BTR(16|32|64)mi8",
                                             "BTS(16|32|64)mi8",
                                             "SAR(8|16|32|64)m1",
                                             "SAR(8|16|32|64)mi",
                                             "SHL(8|16|32|64)m1",
                                             "SHL(8|16|32|64)mi",
                                             "SHR(8|16|32|64)m1",
                                             "SHR(8|16|32|64)mi")>;

def SKXWriteResGroup87 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup87], (instregex "POP(16|32|64)rmm",
                                             "PUSH(16|32|64)rmm")>;

def SKXWriteResGroup88 : SchedWriteRes<[SKXPort6,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKXWriteResGroup88], (instregex "STD")>;

def SKXWriteResGroup89 : SchedWriteRes<[SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup89], (instregex "LD_F(32|64|80)m",
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
                                             "VMOVNTDQAZrm(b?)",
                                             "VMOVSHDUPYrm",
                                             "VMOVSLDUPYrm",
                                             "VMOVUPDYrm",
                                             "VMOVUPSYrm",
                                             "VPBROADCASTDYrm",
                                             "VPBROADCASTQYrm")>;

def SKXWriteResGroup90 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup90], (instregex "VCVTDQ2PDYrr")>;

def SKXWriteResGroup92 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup92], (instregex "VMOVSDZrm(b?)",
                                             "VMOVSSZrm(b?)",
                                             "VPACKSSDWZ128rm(b?)",
                                             "(V?)PACKSSDWrm",
                                             "VPACKSSWBZ128rm(b?)",
                                             "(V?)PACKSSWBrm",
                                             "VPACKUSDWZ128rm(b?)",
                                             "(V?)PACKUSDWrm",
                                             "VPACKUSWBZ128rm(b?)",
                                             "(V?)PACKUSWBrm",
                                             "VPALIGNRZ128rmi(b?)",
                                             "(V?)PALIGNRrmi",
                                             "VPBROADCASTBZ128m(b?)",
                                             "VPBROADCASTBrm",
                                             "VPBROADCASTWZ128m(b?)",
                                             "VPBROADCASTWrm",
                                             "VPSHUFDZ128m(b?)i",
                                             "(V?)PSHUFDmi",
                                             "VPSHUFHWZ128mi(b?)",
                                             "(V?)PSHUFHWmi",
                                             "VPSHUFLWZ128mi(b?)",
                                             "(V?)PSHUFLWmi",
                                             "VPSLLDQZ128rm(b?)",
                                             "VPSRLDQZ128rm(b?)",
                                             "VPUNPCKHBWZ128rm(b?)",
                                             "(V?)PUNPCKHBWrm",
                                             "VPUNPCKHDQZ128rm(b?)",
                                             "(V?)PUNPCKHDQrm",
                                             "VPUNPCKHQDQZ128rm(b?)",
                                             "(V?)PUNPCKHQDQrm",
                                             "VPUNPCKHWDZ128rm(b?)",
                                             "(V?)PUNPCKHWDrm",
                                             "VPUNPCKLBWZ128rm(b?)",
                                             "(V?)PUNPCKLBWrm",
                                             "VPUNPCKLDQZ128rm(b?)",
                                             "(V?)PUNPCKLDQrm",
                                             "VPUNPCKLQDQZ128rm(b?)",
                                             "(V?)PUNPCKLQDQrm",
                                             "VPUNPCKLWDZ128rm(b?)",
                                             "(V?)PUNPCKLWDrm")>;

def SKXWriteResGroup92a : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup92a], (instregex "MMX_PSHUFBrm")>;

def SKXWriteResGroup93 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTDQ2PDZ256rr",
                                             "VCVTDQ2PDZrr",
                                             "VCVTPD2DQYrr",
                                             "VCVTPD2DQZ256rr",
                                             "VCVTPD2DQZrr",
                                             "VCVTPD2PSYrr",
                                             "VCVTPD2PSZ256rr",
                                             "VCVTPD2PSZrr",
                                             "VCVTPD2UDQZ256rr",
                                             "VCVTPD2UDQZrr",
                                             "VCVTPH2PSYrr",
                                             "VCVTPH2PSZ256rr",
                                             "VCVTPH2PSZrr",
                                             "VCVTPS2PDYrr",
                                             "VCVTPS2PDZ256rr",
                                             "VCVTPS2PDZrr",
                                             "VCVTPS2PHYrr",
                                             "VCVTPS2PHZ256rr",
                                             "VCVTPS2PHZrr",
                                             "VCVTPS2QQZ256rr",
                                             "VCVTPS2QQZrr",
                                             "VCVTPS2UQQZ256rr",
                                             "VCVTPS2UQQZrr",
                                             "VCVTQQ2PSZ256rr",
                                             "VCVTQQ2PSZrr",
                                             "VCVTTPD2DQYrr",
                                             "VCVTTPD2DQZ256rr",
                                             "VCVTTPD2DQZrr",
                                             "VCVTTPD2UDQZ256rr",
                                             "VCVTTPD2UDQZrr",
                                             "VCVTTPS2QQZ256rr",
                                             "VCVTTPS2QQZrr",
                                             "VCVTTPS2UQQZ256rr",
                                             "VCVTTPS2UQQZrr",
                                             "VCVTUDQ2PDZ256rr",
                                             "VCVTUDQ2PDZrr",
                                             "VCVTUQQ2PSZ256rr",
                                             "VCVTUQQ2PSZrr")>;


def SKXWriteResGroup95 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup95], (instregex "VBLENDMPDZ128rm(b?)",
                                             "VBLENDMPSZ128rm(b?)",
                                             "VBROADCASTI32X2Z128m(b?)",
                                             "VBROADCASTSSZ128m(b?)",
                                             "VINSERTF128rm",
                                             "VINSERTI128rm",
                                             "VMASKMOVPDrm",
                                             "VMASKMOVPSrm",
                                             "VMOVAPDZ128rm(b?)",
                                             "VMOVAPSZ128rm(b?)",
                                             "VMOVDDUPZ128rm(b?)",
                                             "VMOVDQA32Z128rm(b?)",
                                             "VMOVDQA64Z128rm(b?)",
                                             "VMOVDQU16Z128rm(b?)",
                                             "VMOVDQU32Z128rm(b?)",
                                             "VMOVDQU64Z128rm(b?)",
                                             "VMOVDQU8Z128rm(b?)",
                                             "VMOVNTDQAZ128rm(b?)",
                                             "VMOVSHDUPZ128rm(b?)",
                                             "VMOVSLDUPZ128rm(b?)",
                                             "VMOVUPDZ128rm(b?)",
                                             "VMOVUPSZ128rm(b?)",
                                             "VPADDBZ128rm(b?)",
                                             "(V?)PADDBrm",
                                             "VPADDDZ128rm(b?)",
                                             "(V?)PADDDrm",
                                             "VPADDQZ128rm(b?)",
                                             "(V?)PADDQrm",
                                             "VPADDWZ128rm(b?)",
                                             "(V?)PADDWrm",
                                             "VPBLENDDrmi",
                                             "VPBLENDMBZ128rm(b?)",
                                             "VPBLENDMDZ128rm(b?)",
                                             "VPBLENDMQZ128rm(b?)",
                                             "VPBLENDMWZ128rm(b?)",
                                             "VPBROADCASTDZ128m(b?)",
                                             "VPBROADCASTQZ128m(b?)",
                                             "VPMASKMOVDrm",
                                             "VPMASKMOVQrm",
                                             "VPSUBBZ128rm(b?)",
                                             "(V?)PSUBBrm",
                                             "VPSUBDZ128rm(b?)",
                                             "(V?)PSUBDrm",
                                             "VPSUBQZ128rm(b?)",
                                             "(V?)PSUBQrm",
                                             "VPSUBWZ128rm(b?)",
                                             "(V?)PSUBWrm",
                                             "VPTERNLOGDZ128rm(b?)i",
                                             "VPTERNLOGQZ128rm(b?)i")>;

def SKXWriteResGroup96 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup96], (instregex "MMX_PACKSSDWirm",
                                             "MMX_PACKSSWBirm",
                                             "MMX_PACKUSWBirm")>;

def SKXWriteResGroup97 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup97], (instregex "VPERMI2W128rr",
                                             "VPERMI2W256rr",
                                             "VPERMI2Wrr",
                                             "VPERMT2W128rr",
                                             "VPERMT2W256rr",
                                             "VPERMT2Wrr")>;

def SKXWriteResGroup98 : SchedWriteRes<[SKXPort23,SKXPort06]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup98], (instregex "CMOV(A|BE)(16|32|64)rm")>;

def SKXWriteResGroup99 : SchedWriteRes<[SKXPort23,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup99], (instrs LEAVE, LEAVE64,
                                          SCASB, SCASL, SCASQ, SCASW)>;

def SKXWriteResGroup100 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup100], (instregex "CVTTSS2SI64rr",
                                              "CVTTSS2SIrr",
                                              "VCVTSS2USI64Zrr",
                                              "VCVTTSS2SI64Zrr",
                                              "VCVTTSS2SI64rr",
                                              "VCVTTSS2SIZrr",
                                              "VCVTTSS2SIrr",
                                              "VCVTTSS2USI64Zrr")>;

def SKXWriteResGroup101 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort05]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup101], (instregex "FLDCW16m")>;

def SKXWriteResGroup103 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup103], (instregex "KMOVBkm",
                                              "KMOVDkm",
                                              "KMOVQkm",
                                              "KMOVWkm")>;

def SKXWriteResGroup104 : SchedWriteRes<[SKXPort6,SKXPort23,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup104], (instregex "LRETQ",
                                              "RETQ")>;

def SKXWriteResGroup106 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup106], (instregex "VCOMPRESSPDZ128mr(b?)",
                                              "VCOMPRESSPDZ256mr(b?)",
                                              "VCOMPRESSPDZmr(b?)",
                                              "VCOMPRESSPSZ128mr(b?)",
                                              "VCOMPRESSPSZ256mr(b?)",
                                              "VCOMPRESSPSZmr(b?)",
                                              "VPCOMPRESSDZ128mr(b?)",
                                              "VPCOMPRESSDZ256mr(b?)",
                                              "VPCOMPRESSDZmr(b?)",
                                              "VPCOMPRESSQZ128mr(b?)",
                                              "VPCOMPRESSQZ256mr(b?)",
                                              "VPCOMPRESSQZmr(b?)")>;

def SKXWriteResGroup107 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKXWriteResGroup107], (instregex "ROL(8|16|32|64)m1",
                                              "ROL(8|16|32|64)mi",
                                              "ROR(8|16|32|64)m1",
                                              "ROR(8|16|32|64)mi")>;

def SKXWriteResGroup108 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKXWriteResGroup108], (instregex "XADD(8|16|32|64)rm")>;

def SKXWriteResGroup109 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[SKXWriteResGroup109], (instregex "CALL(16|32|64)m",
                                              "FARCALL64")>;

def SKXWriteResGroup110 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [1,2,2,2];
}
def: InstRW<[SKXWriteResGroup110], (instrs VPSCATTERDQZ128mr,
                                           VPSCATTERQQZ128mr,
                                           VSCATTERDPDZ128mr,
                                           VSCATTERQPDZ128mr)>;

def SKXWriteResGroup111 : SchedWriteRes<[SKXPort6,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,1,2];
}
def: InstRW<[SKXWriteResGroup111], (instrs LOOP)>;

def SKXWriteResGroup112 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 11;
  let ResourceCycles = [1,4,4,2];
}
def: InstRW<[SKXWriteResGroup112], (instrs VPSCATTERDQZ256mr,
                                           VPSCATTERQQZ256mr,
                                           VSCATTERDPDZ256mr,
                                           VSCATTERQPDZ256mr)>;

def SKXWriteResGroup113 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 19;
  let ResourceCycles = [1,8,8,2];
}
def: InstRW<[SKXWriteResGroup113], (instrs VPSCATTERDQZmr,
                                           VPSCATTERQQZmr,
                                           VSCATTERDPDZmr,
                                           VSCATTERQPDZmr)>;

def SKXWriteResGroup114 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 36;
  let ResourceCycles = [1,16,1,16,2];
}
def: InstRW<[SKXWriteResGroup114], (instrs VSCATTERDPSZmr)>;

def SKXWriteResGroup117 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup117], (instregex "VTESTPDrm",
                                              "VTESTPSrm")>;

def SKXWriteResGroup118 : SchedWriteRes<[SKXPort1,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup118], (instregex "PDEP(32|64)rm",
                                              "PEXT(32|64)rm")>;

def SKXWriteResGroup118_16_1 : SchedWriteRes<[SKXPort1, SKXPort0156, SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup118_16_1], (instrs IMUL16rm, IMUL16rmi, IMUL16rmi8)>;

def SKXWriteResGroup118_16_2 : SchedWriteRes<[SKXPort1, SKXPort06, SKXPort0156, SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,2,1];
}
def: InstRW<[SKXWriteResGroup118_16_2], (instrs IMUL16m, MUL16m)>;

def SKXWriteResGroup119 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup119], (instregex "FCOM32m",
                                              "FCOM64m",
                                              "FCOMP32m",
                                              "FCOMP64m",
                                              "MMX_PSADBWirm",
                                              "VFPCLASSSDrm(b?)",
                                              "VPBROADCASTBYrm",
                                              "VPBROADCASTBZ256m(b?)",
                                              "VPBROADCASTBZm(b?)",
                                              "VPBROADCASTWYrm",
                                              "VPBROADCASTWZ256m(b?)",
                                              "VPBROADCASTWZm(b?)",
                                              "VPMOVSXBDYrm",
                                              "VPMOVSXBQYrm",
                                              "VPMOVSXWQYrm")>;

def SKXWriteResGroup121 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup121], (instregex "VBLENDMPDZ256rm(b?)",
                                              "VBLENDMPDZrm(b?)",
                                              "VBLENDMPSZ256rm(b?)",
                                              "VBLENDMPSZrm(b?)",
                                              "VBROADCASTF32X2Z256m(b?)",
                                              "VBROADCASTF32X2Zm(b?)",
                                              "VBROADCASTF32X4Z256rm(b?)",
                                              "VBROADCASTF32X4rm(b?)",
                                              "VBROADCASTF32X8rm(b?)",
                                              "VBROADCASTF64X2Z128rm(b?)",
                                              "VBROADCASTF64X2rm(b?)",
                                              "VBROADCASTF64X4rm(b?)",
                                              "VBROADCASTI32X2Z256m(b?)",
                                              "VBROADCASTI32X2Zm(b?)",
                                              "VBROADCASTI32X4Z256rm(b?)",
                                              "VBROADCASTI32X4rm(b?)",
                                              "VBROADCASTI32X8rm(b?)",
                                              "VBROADCASTI64X2Z128rm(b?)",
                                              "VBROADCASTI64X2rm(b?)",
                                              "VBROADCASTI64X4rm(b?)",
                                              "VBROADCASTSDZ256m(b?)",
                                              "VBROADCASTSDZm(b?)",
                                              "VBROADCASTSSZ256m(b?)",
                                              "VBROADCASTSSZm(b?)",
                                              "VINSERTF32x4Z256rm(b?)",
                                              "VINSERTF32x4Zrm(b?)",
                                              "VINSERTF32x8Zrm(b?)",
                                              "VINSERTF64x2Z256rm(b?)",
                                              "VINSERTF64x2Zrm(b?)",
                                              "VINSERTF64x4Zrm(b?)",
                                              "VINSERTI32x4Z256rm(b?)",
                                              "VINSERTI32x4Zrm(b?)",
                                              "VINSERTI32x8Zrm(b?)",
                                              "VINSERTI64x2Z256rm(b?)",
                                              "VINSERTI64x2Zrm(b?)",
                                              "VINSERTI64x4Zrm(b?)",
                                              "VMASKMOVPDYrm",
                                              "VMASKMOVPSYrm",
                                              "VMOVAPDZ256rm(b?)",
                                              "VMOVAPDZrm(b?)",
                                              "VMOVAPSZ256rm(b?)",
                                              "VMOVAPSZrm(b?)",
                                              "VMOVDDUPZ256rm(b?)",
                                              "VMOVDDUPZrm(b?)",
                                              "VMOVDQA32Z256rm(b?)",
                                              "VMOVDQA32Zrm(b?)",
                                              "VMOVDQA64Z256rm(b?)",
                                              "VMOVDQA64Zrm(b?)",
                                              "VMOVDQU16Z256rm(b?)",
                                              "VMOVDQU16Zrm(b?)",
                                              "VMOVDQU32Z256rm(b?)",
                                              "VMOVDQU32Zrm(b?)",
                                              "VMOVDQU64Z256rm(b?)",
                                              "VMOVDQU64Zrm(b?)",
                                              "VMOVDQU8Z256rm(b?)",
                                              "VMOVDQU8Zrm(b?)",
                                              "VMOVNTDQAZ256rm(b?)",
                                              "VMOVSHDUPZ256rm(b?)",
                                              "VMOVSHDUPZrm(b?)",
                                              "VMOVSLDUPZ256rm(b?)",
                                              "VMOVSLDUPZrm(b?)",
                                              "VMOVUPDZ256rm(b?)",
                                              "VMOVUPDZrm(b?)",
                                              "VMOVUPSZ256rm(b?)",
                                              "VMOVUPSZrm(b?)",
                                              "VPADDBYrm",
                                              "VPADDBZ256rm(b?)",
                                              "VPADDBZrm(b?)",
                                              "VPADDDYrm",
                                              "VPADDDZ256rm(b?)",
                                              "VPADDDZrm(b?)",
                                              "VPADDQYrm",
                                              "VPADDQZ256rm(b?)",
                                              "VPADDQZrm(b?)",
                                              "VPADDWYrm",
                                              "VPADDWZ256rm(b?)",
                                              "VPADDWZrm(b?)",
                                              "VPBLENDDYrmi",
                                              "VPBLENDMBZ256rm(b?)",
                                              "VPBLENDMBZrm(b?)",
                                              "VPBLENDMDZ256rm(b?)",
                                              "VPBLENDMDZrm(b?)",
                                              "VPBLENDMQZ256rm(b?)",
                                              "VPBLENDMQZrm(b?)",
                                              "VPBLENDMWZ256rm(b?)",
                                              "VPBLENDMWZrm(b?)",
                                              "VPBROADCASTDZ256m(b?)",
                                              "VPBROADCASTDZm(b?)",
                                              "VPBROADCASTQZ256m(b?)",
                                              "VPBROADCASTQZm(b?)",
                                              "VPMASKMOVDYrm",
                                              "VPMASKMOVQYrm",
                                              "VPSUBBYrm",
                                              "VPSUBBZ256rm(b?)",
                                              "VPSUBBZrm(b?)",
                                              "VPSUBDYrm",
                                              "VPSUBDZ256rm(b?)",
                                              "VPSUBDZrm(b?)",
                                              "VPSUBQYrm",
                                              "VPSUBQZ256rm(b?)",
                                              "VPSUBQZrm(b?)",
                                              "VPSUBWYrm",
                                              "VPSUBWZrm(b?)",
                                              "VPTERNLOGDZ256rm(b?)i",
                                              "VPTERNLOGDZrm(b?)i",
                                              "VPTERNLOGQZ256rm(b?)i",
                                              "VPTERNLOGQZrm(b?)i")>;

def SKXWriteResGroup123 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup123], (instregex "MMX_PH(ADD|SUB)SWrm")>;

def SKXWriteResGroup124 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort05]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup124], (instregex "MMX_PH(ADD|SUB)(D|W)rm")>;

def SKXWriteResGroup125 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237,SKXPort015]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup125], (instregex "VCVTPS2PHYmr")>;

def SKXWriteResGroup126 : SchedWriteRes<[SKXPort23,SKXPort237,SKXPort06]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup126], (instregex "ROR(8|16|32|64)mCL")>;

def SKXWriteResGroup127 : SchedWriteRes<[SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKXWriteResGroup127], (instregex "RCL(8|16|32|64)m1",
                                              "RCL(8|16|32|64)mi",
                                              "RCR(8|16|32|64)m1",
                                              "RCR(8|16|32|64)mi")>;

def SKXWriteResGroup128 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKXWriteResGroup128], (instregex "ROL(8|16|32|64)mCL",
                                              "SAR(8|16|32|64)mCL",
                                              "SHL(8|16|32|64)mCL",
                                              "SHR(8|16|32|64)mCL")>;

def SKXWriteResGroup130 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKXWriteResGroup130], (instregex "ADC(8|16|32|64)mi",
                                              "CMPXCHG(8|16|32|64)rm",
                                              "SBB(8|16|32|64)mi")>;
def: InstRW<[SKXWriteResGroup130, ReadAfterLd], (instrs ADC8mr, ADC16mr, ADC32mr, ADC64mr,
                                                        SBB8mr, SBB16mr, SBB32mr, SBB64mr)>;

def SKXWriteResGroup131 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 8;
  let ResourceCycles = [1,2,1,2,2];
}
def: InstRW<[SKXWriteResGroup131], (instrs VPSCATTERQDZ128mr,
                                           VPSCATTERQDZ256mr,
                                           VSCATTERQPSZ128mr,
                                           VSCATTERQPSZ256mr)>;

def SKXWriteResGroup132 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 12;
  let ResourceCycles = [1,4,1,4,2];
}
def: InstRW<[SKXWriteResGroup132], (instrs VPSCATTERDDZ128mr,
                                           VSCATTERDPSZ128mr)>;

def SKXWriteResGroup133 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 20;
  let ResourceCycles = [1,8,1,8,2];
}
def: InstRW<[SKXWriteResGroup133], (instrs VPSCATTERDDZ256mr,
                                           VSCATTERDPSZ256mr)>;

def SKXWriteResGroup134 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 36;
  let ResourceCycles = [1,16,1,16,2];
}
def: InstRW<[SKXWriteResGroup134], (instrs VPSCATTERDDZmr)>;

def SKXWriteResGroup135 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup135], (instregex "MMX_CVTPI2PSirm",
                                              "RCPSSm",
                                              "RSQRTSSm",
                                              "VRCPSSm",
                                              "VRSQRTSSm",
                                              "VTESTPDYrm",
                                              "VTESTPSYrm")>;

def SKXWriteResGroup136 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup136], (instregex "VALIGNDZ128rm(b?)i",
                                              "VALIGNQZ128rm(b?)i",
                                              "VCMPPDZ128rm(b?)i",
                                              "VCMPPSZ128rm(b?)i",
                                              "VCMPSDZrm",
                                              "VCMPSSZrm",
                                              "VFPCLASSSSrm(b?)",
                                              "VPCMPBZ128rmi(b?)",
                                              "VPCMPDZ128rmi(b?)",
                                              "VPCMPEQBZ128rm(b?)",
                                              "VPCMPEQDZ128rm(b?)",
                                              "VPCMPEQQZ128rm(b?)",
                                              "VPCMPEQWZ128rm(b?)",
                                              "VPCMPGTBZ128rm(b?)",
                                              "VPCMPGTDZ128rm(b?)",
                                              "VPCMPGTQZ128rm(b?)",
                                              "(V?)PCMPGTQrm",
                                              "VPCMPGTWZ128rm(b?)",
                                              "VPCMPQZ128rmi(b?)",
                                              "VPCMPUBZ128rmi(b?)",
                                              "VPCMPUDZ128rmi(b?)",
                                              "VPCMPUQZ128rmi(b?)",
                                              "VPCMPUWZ128rmi(b?)",
                                              "VPCMPWZ128rmi(b?)",
                                              "VPERMI2D128rm(b?)",
                                              "VPERMI2PD128rm(b?)",
                                              "VPERMI2PS128rm(b?)",
                                              "VPERMI2Q128rm(b?)",
                                              "VPERMT2D128rm(b?)",
                                              "VPERMT2PD128rm(b?)",
                                              "VPERMT2PS128rm(b?)",
                                              "VPERMT2Q128rm(b?)",
                                              "VPMAXSQZ128rm(b?)",
                                              "VPMAXUQZ128rm(b?)",
                                              "VPMINSQZ128rm(b?)",
                                              "VPMINUQZ128rm(b?)",
                                              "VPMOVSXBDZ128rm(b?)",
                                              "VPMOVSXBQZ128rm(b?)",
                                              "VPMOVSXBWYrm",
                                              "VPMOVSXBWZ128rm(b?)",
                                              "VPMOVSXDQYrm",
                                              "VPMOVSXDQZ128rm(b?)",
                                              "VPMOVSXWDYrm",
                                              "VPMOVSXWDZ128rm(b?)",
                                              "VPMOVSXWQZ128rm(b?)",
                                              "VPMOVZXBDZ128rm(b?)",
                                              "VPMOVZXBQZ128rm(b?)",
                                              "VPMOVZXBWZ128rm(b?)",
                                              "VPMOVZXDQZ128rm(b?)",
                                              "VPMOVZXWDYrm",
                                              "VPMOVZXWDZ128rm(b?)",
                                              "VPMOVZXWQZ128rm(b?)",
                                              "VPTESTMBZ128rm(b?)",
                                              "VPTESTMDZ128rm(b?)",
                                              "VPTESTMQZ128rm(b?)",
                                              "VPTESTMWZ128rm(b?)",
                                              "VPTESTNMBZ128rm(b?)",
                                              "VPTESTNMDZ128rm(b?)",
                                              "VPTESTNMQZ128rm(b?)",
                                              "VPTESTNMWZ128rm(b?)")>;

def SKXWriteResGroup137 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup137], (instregex "MMX_CVTPS2PIirm",
                                              "MMX_CVTTPS2PIirm",
                                              "(V?)ADDSDrm",
                                              "(V?)ADDSSrm",
                                              "(V?)CMPSDrm",
                                              "(V?)CMPSSrm",
                                              "VCVTPH2PSrm",
                                              "(V?)CVTPS2PDrm",
                                              "(V?)MAX(C?)SDrm",
                                              "(V?)MAX(C?)SSrm",
                                              "(V?)MIN(C?)SDrm",
                                              "(V?)MIN(C?)SSrm",
                                              "(V?)MULSDrm",
                                              "(V?)MULSSrm",
                                              "(V?)SUBSDrm",
                                              "(V?)SUBSSrm")>;

def SKXWriteResGroup138 : SchedWriteRes<[SKXPort0,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup138], (instregex "VRCP14PDZr(b?)",
                                              "VRCP14PSZr(b?)",
                                              "VRSQRT14PDZr(b?)",
                                              "VRSQRT14PSZr(b?)")>;

def SKXWriteResGroup141 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup141], (instregex "(V?)PTESTrm")>;

def SKXWriteResGroup142 : SchedWriteRes<[SKXPort1,SKXPort5,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup142], (instrs IMUL64m, MUL64m, MULX64rm)>;

def SKXWriteResGroup143 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup143], (instregex "(V?)PHADDSWrm",
                                              "(V?)PHSUBSWrm")>;

def SKXWriteResGroup145 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup145], (instregex "SHLD(16|32|64)mri8",
                                              "SHRD(16|32|64)mri8")>;

def SKXWriteResGroup146 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort23,SKXPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup146], (instregex "LAR(16|32|64)rm",
                                              "LSL(16|32|64)rm")>;

def SKXWriteResGroup148 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup148], (instregex "(ADD|SUB|SUBR)_F(32|64)m",
                                              "ILD_F(16|32|64)m",
                                              "VALIGNDZ256rm(b?)i",
                                              "VALIGNDZrm(b?)i",
                                              "VALIGNQZ256rm(b?)i",
                                              "VALIGNQZrm(b?)i",
                                              "VCMPPDZ256rm(b?)i",
                                              "VCMPPDZrm(b?)i",
                                              "VCMPPSZ256rm(b?)i",
                                              "VCMPPSZrm(b?)i",
                                              "VPCMPBZ256rmi(b?)",
                                              "VPCMPBZrmi(b?)",
                                              "VPCMPDZ256rmi(b?)",
                                              "VPCMPDZrmi(b?)",
                                              "VPCMPEQBZ256rm(b?)",
                                              "VPCMPEQBZrm(b?)",
                                              "VPCMPEQDZ256rm(b?)",
                                              "VPCMPEQDZrm(b?)",
                                              "VPCMPEQQZ256rm(b?)",
                                              "VPCMPEQQZrm(b?)",
                                              "VPCMPEQWZ256rm(b?)",
                                              "VPCMPEQWZrm(b?)",
                                              "VPCMPGTBZ256rm(b?)",
                                              "VPCMPGTBZrm(b?)",
                                              "VPCMPGTDZ256rm(b?)",
                                              "VPCMPGTDZrm(b?)",
                                              "VPCMPGTQYrm",
                                              "VPCMPGTQZ256rm(b?)",
                                              "VPCMPGTQZrm(b?)",
                                              "VPCMPGTWZ256rm(b?)",
                                              "VPCMPGTWZrm(b?)",
                                              "VPCMPQZ256rmi(b?)",
                                              "VPCMPQZrmi(b?)",
                                              "VPCMPUBZ256rmi(b?)",
                                              "VPCMPUBZrmi(b?)",
                                              "VPCMPUDZ256rmi(b?)",
                                              "VPCMPUDZrmi(b?)",
                                              "VPCMPUQZ256rmi(b?)",
                                              "VPCMPUQZrmi(b?)",
                                              "VPCMPUWZ256rmi(b?)",
                                              "VPCMPUWZrmi(b?)",
                                              "VPCMPWZ256rmi(b?)",
                                              "VPCMPWZrmi(b?)",
                                              "VPMAXSQZ256rm(b?)",
                                              "VPMAXSQZrm(b?)",
                                              "VPMAXUQZ256rm(b?)",
                                              "VPMAXUQZrm(b?)",
                                              "VPMINSQZ256rm(b?)",
                                              "VPMINSQZrm(b?)",
                                              "VPMINUQZ256rm(b?)",
                                              "VPMINUQZrm(b?)",
                                              "VPMOVZXBDYrm",
                                              "VPMOVZXBQYrm",
                                              "VPMOVZXBWYrm",
                                              "VPMOVZXDQYrm",
                                              "VPMOVZXWQYrm",
                                              "VPTESTMBZ256rm(b?)",
                                              "VPTESTMBZrm(b?)",
                                              "VPTESTMDZ256rm(b?)",
                                              "VPTESTMDZrm(b?)",
                                              "VPTESTMQZ256rm(b?)",
                                              "VPTESTMQZrm(b?)",
                                              "VPTESTMWZ256rm(b?)",
                                              "VPTESTMWZrm(b?)",
                                              "VPTESTNMBZ256rm(b?)",
                                              "VPTESTNMBZrm(b?)",
                                              "VPTESTNMDZ256rm(b?)",
                                              "VPTESTNMDZrm(b?)",
                                              "VPTESTNMQZ256rm(b?)",
                                              "VPTESTNMQZrm(b?)",
                                              "VPTESTNMWZ256rm(b?)",
                                              "VPTESTNMWZrm(b?)")>;

def SKXWriteResGroup149 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup149], (instregex "CVTDQ2PSrm",
                                              "CVTPS2DQrm",
                                              "CVTSS2SDrm",
                                              "CVTTPS2DQrm",
                                              "VCVTDQ2PDZ128rm(b?)",
                                              "VCVTDQ2PSZ128rm(b?)",
                                              "VCVTDQ2PSrm",
                                              "VCVTPD2QQZ128rm(b?)",
                                              "VCVTPD2UQQZ128rm(b?)",
                                              "VCVTPH2PSYrm",
                                              "VCVTPH2PSZ128rm(b?)",
                                              "VCVTPS2DQZ128rm(b?)",
                                              "VCVTPS2DQrm",
                                              "VCVTPS2PDZ128rm(b?)",
                                              "VCVTPS2QQZ128rm(b?)",
                                              "VCVTPS2UDQZ128rm(b?)",
                                              "VCVTPS2UQQZ128rm(b?)",
                                              "VCVTQQ2PDZ128rm(b?)",
                                              "VCVTQQ2PSZ128rm(b?)",
                                              "VCVTSS2SDZrm",
                                              "VCVTSS2SDrm",
                                              "VCVTTPD2QQZ128rm(b?)",
                                              "VCVTTPD2UQQZ128rm(b?)",
                                              "VCVTTPS2DQZ128rm(b?)",
                                              "VCVTTPS2DQrm",
                                              "VCVTTPS2QQZ128rm(b?)",
                                              "VCVTTPS2UDQZ128rm(b?)",
                                              "VCVTTPS2UQQZ128rm(b?)",
                                              "VCVTUDQ2PDZ128rm(b?)",
                                              "VCVTUDQ2PSZ128rm(b?)",
                                              "VCVTUQQ2PDZ128rm(b?)",
                                              "VCVTUQQ2PSZ128rm(b?)")>;

def SKXWriteResGroup151 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup151], (instregex "VEXPANDPDZ128rm(b?)",
                                              "VEXPANDPSZ128rm(b?)",
                                              "VPEXPANDDZ128rm(b?)",
                                              "VPEXPANDQZ128rm(b?)")>;

def SKXWriteResGroup152 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup152], (instregex "MMX_CVTPI2PDirm",
                                              "VPTESTYrm")>;

def SKXWriteResGroup153 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup153], (instregex "(V?)CVTSD2SSrm")>;

def SKXWriteResGroup154 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup154], (instregex "VPHADDSWYrm",
                                              "VPHSUBSWYrm")>;

def SKXWriteResGroup156 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort06,SKXPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup156], (instrs IMUL32m, MUL32m, MULX32rm)>;

def SKXWriteResGroup157 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 10;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,3];
}
def: InstRW<[SKXWriteResGroup157], (instregex "XCHG(8|16|32|64)rm")>;

def SKXWriteResGroup159 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup159], (instregex "(V?)DIVPS(Z128)?rr",
                                              "(V?)DIVSS(Z?)rr")>;

def SKXWriteResGroup159_1 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKXWriteResGroup159_1], (instregex "VDIVPS(Y|Z256)rr")>;

def SKXWriteResGroup160 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup160], (instregex "MUL_F(32|64)m")>;

def SKXWriteResGroup161 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTDQ2PDZ256rm(b?)",
                                              "VCVTDQ2PDZrm(b?)",
                                              "VCVTDQ2PSYrm",
                                              "VCVTDQ2PSZ256rm(b?)",
                                              "VCVTDQ2PSZrm(b?)",
                                              "VCVTPD2QQZ256rm(b?)",
                                              "VCVTPD2QQZrm(b?)",
                                              "VCVTPD2UQQZ256rm(b?)",
                                              "VCVTPD2UQQZrm(b?)",
                                              "VCVTPH2PSZ256rm(b?)",
                                              "VCVTPH2PSZrm(b?)",
                                              "VCVTPS2DQYrm",
                                              "VCVTPS2DQZ256rm(b?)",
                                              "VCVTPS2DQZrm(b?)",
                                              "VCVTPS2PDYrm",
                                              "VCVTPS2PDZ256rm(b?)",
                                              "VCVTPS2PDZrm(b?)",
                                              "VCVTPS2QQZ256rm(b?)",
                                              "VCVTPS2UDQZ256rm(b?)",
                                              "VCVTPS2UDQZrm(b?)",
                                              "VCVTPS2UQQZ256rm(b?)",
                                              "VCVTQQ2PDZ256rm(b?)",
                                              "VCVTQQ2PDZrm(b?)",
                                              "VCVTQQ2PSZ256rm(b?)",
                                              "VCVTTPD2QQZ256rm(b?)",
                                              "VCVTTPD2QQZrm(b?)",
                                              "VCVTTPD2UQQZ256rm(b?)",
                                              "VCVTTPD2UQQZrm(b?)",
                                              "VCVTTPS2DQYrm",
                                              "VCVTTPS2DQZ256rm(b?)",
                                              "VCVTTPS2DQZrm(b?)",
                                              "VCVTTPS2QQZ256rm(b?)",
                                              "VCVTTPS2UDQZ256rm(b?)",
                                              "VCVTTPS2UDQZrm(b?)",
                                              "VCVTTPS2UQQZ256rm(b?)",
                                              "VCVTUDQ2PDZ256rm(b?)",
                                              "VCVTUDQ2PDZrm(b?)",
                                              "VCVTUDQ2PSZ256rm(b?)",
                                              "VCVTUDQ2PSZrm(b?)",
                                              "VCVTUQQ2PDZ256rm(b?)",
                                              "VCVTUQQ2PDZrm(b?)",
                                              "VCVTUQQ2PSZ256rm(b?)")>;

def SKXWriteResGroup162 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup162], (instregex "FICOM16m",
                                              "FICOM32m",
                                              "FICOMP16m",
                                              "FICOMP32m",
                                              "VEXPANDPDZ256rm(b?)",
                                              "VEXPANDPDZrm(b?)",
                                              "VEXPANDPSZ256rm(b?)",
                                              "VEXPANDPSZrm(b?)",
                                              "VPEXPANDDZ256rm(b?)",
                                              "VPEXPANDDZrm(b?)",
                                              "VPEXPANDQZ256rm(b?)",
                                              "VPEXPANDQZrm(b?)")>;

def SKXWriteResGroup163 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup163], (instregex "VCVTSD2SSZrm")>;

def SKXWriteResGroup164 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup164], (instregex "(V?)CVTDQ2PDrm")>;

def SKXWriteResGroup165 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup165], (instregex "CVTSD2SI64rm",
                                              "CVTSD2SIrm",
                                              "CVTSS2SI64rm",
                                              "CVTSS2SIrm",
                                              "CVTTSD2SI64rm",
                                              "CVTTSD2SIrm",
                                              "CVTTSS2SIrm",
                                              "VCVTSD2SI64Zrm(b?)",
                                              "VCVTSD2SI64rm",
                                              "VCVTSD2SIZrm(b?)",
                                              "VCVTSD2SIrm",
                                              "VCVTSD2USI64Zrm(b?)",
                                              "VCVTSS2SI64Zrm(b?)",
                                              "VCVTSS2SI64rm",
                                              "VCVTSS2SIZrm(b?)",
                                              "VCVTSS2SIrm",
                                              "VCVTSS2USIZrm(b?)",
                                              "VCVTTSD2SI64Zrm(b?)",
                                              "VCVTTSD2SI64rm",
                                              "VCVTTSD2SIZrm(b?)",
                                              "VCVTTSD2SIrm",
                                              "VCVTTSD2USI64Zrm(b?)",
                                              "VCVTTSS2SI64Zrm(b?)",
                                              "VCVTTSS2SI64rm",
                                              "VCVTTSS2SIZrm(b?)",
                                              "VCVTTSS2SIrm",
                                              "VCVTTSS2USIZrm(b?)")>;

def SKXWriteResGroup166 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup166], (instregex "CVTPD2DQrm",
                                              "CVTPD2PSrm",
                                              "CVTTPD2DQrm",
                                              "MMX_CVTPD2PIirm",
                                              "MMX_CVTTPD2PIirm")>;

def SKXWriteResGroup167 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup167], (instregex "VPCONFLICTQZ128rm(b?)")>;

def SKXWriteResGroup168 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 11;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKXWriteResGroup168], (instregex "SHLD(16|32|64)mrCL",
                                              "SHRD(16|32|64)mrCL")>;

def SKXWriteResGroup169 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,3,2];
}
def: InstRW<[SKXWriteResGroup169], (instregex "RCL(16|32|64)rCL",
                                              "RCR(16|32|64)rCL")>;

def SKXWriteResGroup170 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 11;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,1,2];
}
def: InstRW<[SKXWriteResGroup170], (instregex "RCL8rCL")>;

def SKXWriteResGroup171 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 11;
  let NumMicroOps = 11;
  let ResourceCycles = [2,9];
}
def: InstRW<[SKXWriteResGroup171], (instrs LOOPE, LOOPNE)>;

def SKXWriteResGroup172 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup172], (instregex "(V?)SQRTPS(Z128)?r",
                                              "(V?)SQRTSS(Z?)r")>;

def SKXWriteResGroup173 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1,6];
}
def: InstRW<[SKXWriteResGroup173], (instregex "VSQRTPS(Y|Z256)r")>;

def SKXWriteResGroup174 : SchedWriteRes<[SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKXWriteResGroup174], (instregex "VPMULLQZ128rr",
                                              "VPMULLQZ256rr",
                                              "VPMULLQZrr")>;

def SKXWriteResGroup175 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup175], (instregex "VPERMWZ128rm(b?)")>;

def SKXWriteResGroup176 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup176], (instregex "VCVTSD2USIZrm(b?)",
                                              "VCVTSS2USI64Zrm(b?)",
                                              "VCVTTSD2USIZrm(b?)",
                                              "VCVTTSS2USI64Zrm(b?)")>;

def SKXWriteResGroup177 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup177], (instregex "VCVTPS2QQZrm(b?)",
                                              "VCVTPS2UQQZrm(b?)",
                                              "VCVTTPS2QQZrm(b?)",
                                              "VCVTTPS2UQQZrm(b?)")>;

def SKXWriteResGroup179 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup179], (instregex "CVTTSS2SI64rm")>;

def SKXWriteResGroup180 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup180], (instregex "(ADD|SUB|SUBR)_FI(16|32)m",
                                              "VPERMWZ256rm(b?)",
                                              "VPERMWZrm(b?)")>;

def SKXWriteResGroup181 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup181], (instregex "VCVTDQ2PDYrm")>;

def SKXWriteResGroup183 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup183], (instregex "VPERMI2W128rm(b?)",
                                              "VPERMT2W128rm(b?)")>;

def SKXWriteResGroup184 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup184], (instregex "(V?)DIVPDrr",
                                              "(V?)DIVSD(Z?)rr")>;

def SKXWriteResGroup184_1 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKXWriteResGroup184_1], (instregex "VDIVPD(Y|Z256)rr")>;

def SKXWriteResGroup187 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup187], (instregex "MUL_FI(16|32)m")>;

def SKXWriteResGroup188 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup188], (instregex "VCVTPD2DQZrm(b?)",
                                              "VCVTPD2PSZrm(b?)",
                                              "VCVTPD2UDQZrm(b?)",
                                              "VCVTQQ2PSZrm(b?)",
                                              "VCVTTPD2DQZrm(b?)",
                                              "VCVTTPD2UDQZrm(b?)",
                                              "VCVTUQQ2PSZrm(b?)")>;

def SKXWriteResGroup189 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup189], (instregex "VPERMI2W256rm(b?)",
                                              "VPERMI2Wrm(b?)",
                                              "VPERMT2W256rm(b?)",
                                              "VPERMT2Wrm(b?)")>;

def SKXWriteResGroup190 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 14;
  let NumMicroOps = 10;
  let ResourceCycles = [2,4,1,3];
}
def: InstRW<[SKXWriteResGroup190], (instregex "RCR8rCL")>;

def SKXWriteResGroup191 : SchedWriteRes<[SKXPort0]> {
  let Latency = 15;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup191], (instregex "DIVR_FPrST0",
                                              "DIVR_FST0r",
                                              "DIVR_FrST0")>;

def SKXWriteResGroup194 : SchedWriteRes<[SKXPort1,SKXPort5,SKXPort01,SKXPort23,SKXPort015]> {
  let Latency = 15;
  let NumMicroOps = 8;
  let ResourceCycles = [1,2,2,1,2];
}
def: InstRW<[SKXWriteResGroup194], (instregex "VPCONFLICTDZ128rm(b?)")>;

def SKXWriteResGroup195 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 15;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,5,1,1];
}
def: InstRW<[SKXWriteResGroup195], (instregex "RCL(8|16|32|64)mCL")>;

def SKXWriteResGroup196 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup196], (instregex "(V?)DIVSS(Z?)rm")>;

def SKXWriteResGroup198 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup198], (instregex "VRCP14PDZm(b?)",
                                              "VRCP14PSZm(b?)",
                                              "VRSQRT14PDZm(b?)",
                                              "VRSQRT14PSZm(b?)")>;

def SKXWriteResGroup199 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 16;
  let NumMicroOps = 14;
  let ResourceCycles = [1,1,1,4,2,5];
}
def: InstRW<[SKXWriteResGroup199], (instregex "CMPXCHG8B")>;

def SKXWriteResGroup200 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 16;
  let NumMicroOps = 16;
  let ResourceCycles = [16];
}
def: InstRW<[SKXWriteResGroup200], (instrs VZEROALL)>;

def SKXWriteResGroup201 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 17;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,5];
}
def: InstRW<[SKXWriteResGroup201], (instregex "(V?)DIVPS(Z128)?rm")>;

def SKXWriteResGroup201_1 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 17;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup201_1], (instregex "(V?)SQRTSS(Z?)m")>;

def SKXWriteResGroup202 : SchedWriteRes<[SKXPort0,SKXPort1,SKXPort5,SKXPort6,SKXPort05,SKXPort0156]> {
  let Latency = 17;
  let NumMicroOps = 15;
  let ResourceCycles = [2,1,2,4,2,4];
}
def: InstRW<[SKXWriteResGroup202], (instrs XCH_F)>;

def SKXWriteResGroup203 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 1;
  let ResourceCycles = [1,6];
}
def: InstRW<[SKXWriteResGroup203], (instregex "(V?)SQRTPD(Z128)?r",
                                              "(V?)SQRTSD(Z?)r")>;

def SKXWriteResGroup203_1 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 1;
  let ResourceCycles = [1,12];
}
def: InstRW<[SKXWriteResGroup203_1], (instregex "VSQRTPD(Y|Z256)r")>;

def SKXWriteResGroup204 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,5];
}
def: InstRW<[SKXWriteResGroup204], (instregex "VDIVPS(Y|Z256)rm")>;

def SKXWriteResGroup204_1 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup204_1], (instregex "(V?)SQRTPS(Z128)?m")>;

def SKXWriteResGroup205 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 18;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup205], (instregex "VPMULLQZ128rm(b?)")>;

def SKXWriteResGroup207 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort06,SKXPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[SKXWriteResGroup207], (instrs CPUID, RDTSC)>;

def SKXWriteResGroup208 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 18;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,4,1,2];
}
def: InstRW<[SKXWriteResGroup208], (instregex "RCR(8|16|32|64)mCL")>;

def SKXWriteResGroup209 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKXWriteResGroup209], (instregex "(V?)DIVSD(Z?)rm")>;

def SKXWriteResGroup209_1 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,6];
}
def: InstRW<[SKXWriteResGroup209_1], (instregex "VSQRTPS(Y|Z256)m")>;

def SKXWriteResGroup210 : SchedWriteRes<[SKXPort0,SKXPort5,SKXFPDivider]> {
  let Latency = 20;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,12];
}
def: InstRW<[SKXWriteResGroup210], (instregex "VSQRTPSZr")>;

def SKXWriteResGroup211 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 19;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup211], (instregex "VPMULLQZ256rm(b?)",
                                              "VPMULLQZrm(b?)")>;

def SKXWriteResGroup214 : SchedWriteRes<[]> {
  let Latency = 20;
  let NumMicroOps = 0;
}
def: InstRW<[SKXWriteResGroup214], (instrs VGATHERDPSZ128rm,
                                           VGATHERQPSZrm,
                                           VPGATHERDDZ128rm)>;

def SKXWriteResGroup215 : SchedWriteRes<[SKXPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup215], (instregex "DIV_FPrST0",
                                              "DIV_FST0r",
                                              "DIV_FrST0")>;

def SKXWriteResGroup216 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKXWriteResGroup216], (instregex "(V?)DIVPD(Z128)?rm")>;

def SKXWriteResGroup218 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 20;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup218], (instrs VGATHERQPSZ128rm,
                                           VGATHERQPSZ256rm,
                                           VPGATHERQDZ128rm,
                                           VPGATHERQDZ256rm)>;

def SKXWriteResGroup219 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort6,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 20;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,1,2];
}
def: InstRW<[SKXWriteResGroup219], (instrs INSB, INSL, INSW)>;

def SKXWriteResGroup220 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort0156]> {
  let Latency = 20;
  let NumMicroOps = 10;
  let ResourceCycles = [1,2,7];
}
def: InstRW<[SKXWriteResGroup220], (instregex "MWAITrr")>;

def SKXWriteResGroup222 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,8];
}
def: InstRW<[SKXWriteResGroup222], (instregex "VDIVPD(Y|Z256)rm")>;

def SKXWriteResGroup223 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 22;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup223], (instregex "DIV_F(32|64)m")>;

def SKXWriteResGroup224 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 22;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup224], (instrs VGATHERDPDZ128rm,
                                           VGATHERQPDZ128rm,
                                           VPGATHERDQZ128rm,
                                           VPGATHERQQZ128rm)>;

def SKXWriteResGroup224_2 : SchedWriteRes<[SKXPort0, SKXPort23, SKXPort5, SKXPort015]> {
  let Latency = 22;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup224_2], (instrs VGATHERDPSrm,
                                             VGATHERDPDrm,
                                             VGATHERQPDrm,
                                             VGATHERQPSrm,
                                             VPGATHERDDrm,
                                             VPGATHERDQrm,
                                             VPGATHERQDrm,
                                             VPGATHERQQrm,
                                             VPGATHERDDrm,
                                             VPGATHERQDrm,
                                             VPGATHERDQrm,
                                             VPGATHERQQrm,
                                             VGATHERDPSrm,
                                             VGATHERQPSrm,
                                             VGATHERDPDrm,
                                             VGATHERQPDrm)>;

def SKXWriteResGroup224_3 : SchedWriteRes<[SKXPort0, SKXPort23, SKXPort5, SKXPort015]> {
  let Latency = 25;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup224_3], (instrs VGATHERDPSYrm,
                                             VGATHERQPDYrm,
                                             VGATHERQPSYrm,
                                             VPGATHERDDYrm,
                                             VPGATHERDQYrm,
                                             VPGATHERQDYrm,
                                             VPGATHERQQYrm,
                                             VPGATHERDDYrm,
                                             VPGATHERQDYrm,
                                             VPGATHERDQYrm,
                                             VPGATHERQQYrm,
                                             VGATHERDPSYrm,
                                             VGATHERQPSYrm,
                                             VGATHERDPDYrm)>;

def SKXWriteResGroup225 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort015]> {
  let Latency = 22;
  let NumMicroOps = 14;
  let ResourceCycles = [5,5,4];
}
def: InstRW<[SKXWriteResGroup225], (instregex "VPCONFLICTDZ128rr",
                                              "VPCONFLICTQZ256rr")>;

def SKXWriteResGroup226 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 23;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,6];
}
def: InstRW<[SKXWriteResGroup226], (instregex "(V?)SQRTSD(Z?)m")>;

def SKXWriteResGroup227 : SchedWriteRes<[SKXPort0,SKXPort5,SKXFPDivider]> {
  let Latency = 23;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,16];
}
def: InstRW<[SKXWriteResGroup227], (instregex "VDIVPDZrr")>;

def SKXWriteResGroup227_1 : SchedWriteRes<[SKXPort0,SKXPort5,SKXFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,10];
}
def: InstRW<[SKXWriteResGroup227_1], (instregex "VDIVPSZrr")>;

def SKXWriteResGroup228 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 23;
  let NumMicroOps = 19;
  let ResourceCycles = [2,1,4,1,1,4,6];
}
def: InstRW<[SKXWriteResGroup228], (instregex "CMPXCHG16B")>;

def SKXWriteResGroup229 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 24;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,6];
}
def: InstRW<[SKXWriteResGroup229], (instregex "(V?)SQRTPD(Z128)?m")>;

def SKXWriteResGroup230 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort5,SKXFPDivider]> {
  let Latency = 25;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,10];
}
def: InstRW<[SKXWriteResGroup230], (instregex "VDIVPSZrm(b?)")>;

def SKXWriteResGroup232 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,12];
}
def: InstRW<[SKXWriteResGroup232], (instregex "VSQRTPD(Y|Z256)m")>;

def SKXWriteResGroup233 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 25;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup233], (instregex "DIV_FI(16|32)m")>;

def SKXWriteResGroup234 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 25;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup234], (instrs VGATHERDPDZ256rm,
                                           VGATHERQPDZ256rm,
                                           VPGATHERDQZ256rm,
                                           VPGATHERQDZrm,
                                           VPGATHERQQZ256rm)>;

def SKXWriteResGroup237 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort5,SKXFPDivider]> {
  let Latency = 27;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,12];
}
def: InstRW<[SKXWriteResGroup237], (instregex "VSQRTPSZm(b?)")>;

def SKXWriteResGroup238 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 26;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup238], (instrs VGATHERDPDZrm,
                                           VGATHERQPDZrm,
                                           VPGATHERDQZrm,
                                           VPGATHERQQZrm)>;

def SKXWriteResGroup239 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 27;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup239], (instregex "DIVR_F(32|64)m")>;

def SKXWriteResGroup240 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 27;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup240], (instrs VGATHERDPSZ256rm,
                                           VPGATHERDDZ256rm)>;

def SKXWriteResGroup241 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23,SKXPort0156]> {
  let Latency = 28;
  let NumMicroOps = 8;
  let ResourceCycles = [2,4,1,1];
}
def: InstRW<[SKXWriteResGroup241], (instregex "IDIV(8|16|32|64)m")>;

def SKXWriteResGroup242 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23,SKXPort015]> {
  let Latency = 29;
  let NumMicroOps = 15;
  let ResourceCycles = [5,5,1,4];
}
def: InstRW<[SKXWriteResGroup242], (instregex "VPCONFLICTQZ256rm(b?)")>;

def SKXWriteResGroup243 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 30;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup243], (instregex "DIVR_FI(16|32)m")>;

def SKXWriteResGroup244 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort5,SKXFPDivider]> {
  let Latency = 30;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,16];
}
def: InstRW<[SKXWriteResGroup244], (instregex "VDIVPDZrm(b?)")>;

def SKXWriteResGroup245 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 30;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup245], (instrs VGATHERDPSZrm,
                                           VPGATHERDDZrm)>;

def SKXWriteResGroup246 : SchedWriteRes<[SKXPort0,SKXPort5,SKXFPDivider]> {
  let Latency = 32;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,24];
}
def: InstRW<[SKXWriteResGroup246], (instregex "VSQRTPDZr")>;

def SKXWriteResGroup247 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort23,SKXPort06,SKXPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[SKXWriteResGroup247], (instregex "IN(8|16|32)ri",
                                              "IN(8|16|32)rr")>;

def SKXWriteResGroup248 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[SKXWriteResGroup248], (instregex "OUT(8|16|32)ir",
                                              "OUT(8|16|32)rr")>;

def SKXWriteResGroup249 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort015]> {
  let Latency = 37;
  let NumMicroOps = 21;
  let ResourceCycles = [9,7,5];
}
def: InstRW<[SKXWriteResGroup249], (instregex "VPCONFLICTDZ256rr",
                                              "VPCONFLICTQZrr")>;

def SKXWriteResGroup250 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort23,SKXPort0156]> {
  let Latency = 37;
  let NumMicroOps = 31;
  let ResourceCycles = [1,8,1,21];
}
def: InstRW<[SKXWriteResGroup250], (instregex "XRSTOR(64)?")>;

def SKXWriteResGroup251 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort5,SKXFPDivider]> {
  let Latency = 39;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,24];
}
def: InstRW<[SKXWriteResGroup251], (instregex "VSQRTPDZm(b?)")>;

def SKXWriteResGroup252 : SchedWriteRes<[SKXPort1,SKXPort4,SKXPort5,SKXPort6,SKXPort23,SKXPort237,SKXPort15,SKXPort0156]> {
  let Latency = 40;
  let NumMicroOps = 18;
  let ResourceCycles = [1,1,2,3,1,1,1,8];
}
def: InstRW<[SKXWriteResGroup252], (instregex "VMCLEARm")>;

def SKXWriteResGroup253 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 41;
  let NumMicroOps = 39;
  let ResourceCycles = [1,10,1,1,26];
}
def: InstRW<[SKXWriteResGroup253], (instregex "XSAVE64")>;

def SKXWriteResGroup254 : SchedWriteRes<[SKXPort5,SKXPort0156]> {
  let Latency = 42;
  let NumMicroOps = 22;
  let ResourceCycles = [2,20];
}
def: InstRW<[SKXWriteResGroup254], (instrs RDTSCP)>;

def SKXWriteResGroup255 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 42;
  let NumMicroOps = 40;
  let ResourceCycles = [1,11,1,1,26];
}
def: InstRW<[SKXWriteResGroup255], (instregex "^XSAVE$", "XSAVEC", "XSAVES", "XSAVEOPT")>;

def SKXWriteResGroup256 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23,SKXPort015]> {
  let Latency = 44;
  let NumMicroOps = 22;
  let ResourceCycles = [9,7,1,5];
}
def: InstRW<[SKXWriteResGroup256], (instregex "VPCONFLICTDZ256rm(b?)",
                                              "VPCONFLICTQZrm(b?)")>;

def SKXWriteResGroup258 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort05,SKXPort06,SKXPort0156]> {
  let Latency = 62;
  let NumMicroOps = 64;
  let ResourceCycles = [2,8,5,10,39];
}
def: InstRW<[SKXWriteResGroup258], (instregex "FLDENVm")>;

def SKXWriteResGroup259 : SchedWriteRes<[SKXPort0,SKXPort6,SKXPort23,SKXPort05,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 63;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[SKXWriteResGroup259], (instrs FXRSTOR64)>;

def SKXWriteResGroup260 : SchedWriteRes<[SKXPort0,SKXPort6,SKXPort23,SKXPort05,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 63;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[SKXWriteResGroup260], (instrs FXRSTOR)>;

def SKXWriteResGroup261 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort015]> {
  let Latency = 67;
  let NumMicroOps = 35;
  let ResourceCycles = [17,11,7];
}
def: InstRW<[SKXWriteResGroup261], (instregex "VPCONFLICTDZrr")>;

def SKXWriteResGroup262 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23,SKXPort015]> {
  let Latency = 74;
  let NumMicroOps = 36;
  let ResourceCycles = [17,11,1,7];
}
def: InstRW<[SKXWriteResGroup262], (instregex "VPCONFLICTDZrm(b?)")>;

def SKXWriteResGroup263 : SchedWriteRes<[SKXPort5,SKXPort05,SKXPort0156]> {
  let Latency = 75;
  let NumMicroOps = 15;
  let ResourceCycles = [6,3,6];
}
def: InstRW<[SKXWriteResGroup263], (instrs FNINIT)>;

def SKXWriteResGroup264 : SchedWriteRes<[SKXPort0,SKXPort1,SKXPort5,SKXPort6,SKXPort05,SKXPort0156]> {
  let Latency = 76;
  let NumMicroOps = 32;
  let ResourceCycles = [7,2,8,3,1,11];
}
def: InstRW<[SKXWriteResGroup264], (instregex "DIV(16|32|64)r")>;

def SKXWriteResGroup265 : SchedWriteRes<[SKXPort0,SKXPort1,SKXPort5,SKXPort6,SKXPort06,SKXPort0156]> {
  let Latency = 102;
  let NumMicroOps = 66;
  let ResourceCycles = [4,2,4,8,14,34];
}
def: InstRW<[SKXWriteResGroup265], (instregex "IDIV(16|32|64)r")>;

def SKXWriteResGroup266 : SchedWriteRes<[SKXPort0,SKXPort1,SKXPort4,SKXPort5,SKXPort6,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 106;
  let NumMicroOps = 100;
  let ResourceCycles = [9,1,11,16,1,11,21,30];
}
def: InstRW<[SKXWriteResGroup266], (instregex "FSTENVm")>;

def SKXWriteResGroup267 : SchedWriteRes<[SKXPort6,SKXPort0156]> {
  let Latency = 140;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup267], (instrs PAUSE)>;
} // SchedModel
