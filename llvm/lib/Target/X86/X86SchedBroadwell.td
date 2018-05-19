//=- X86SchedBroadwell.td - X86 Broadwell Scheduling ---------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Broadwell to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def BroadwellModel : SchedMachineModel {
  // All x86 instructions are modeled as a single micro-op, and BW can decode 4
  // instructions per cycle.
  let IssueWidth = 4;
  let MicroOpBufferSize = 192; // Based on the reorder buffer.
  let LoadLatency = 5;
  let MispredictPenalty = 16;

  // Based on the LSD (loop-stream detector) queue size and benchmarking data.
  let LoopMicroOpBufferSize = 50;

  // This flag is set to allow the scheduler to assign a default model to
  // unrecognized opcodes.
  let CompleteModel = 0;
}

let SchedModel = BroadwellModel in {

// Broadwell can issue micro-ops to 8 different ports in one cycle.

// Ports 0, 1, 5, and 6 handle all computation.
// Port 4 gets the data half of stores. Store data can be available later than
// the store address, but since we don't model the latency of stores, we can
// ignore that.
// Ports 2 and 3 are identical. They handle loads and the address half of
// stores. Port 7 can handle address calculations.
def BWPort0 : ProcResource<1>;
def BWPort1 : ProcResource<1>;
def BWPort2 : ProcResource<1>;
def BWPort3 : ProcResource<1>;
def BWPort4 : ProcResource<1>;
def BWPort5 : ProcResource<1>;
def BWPort6 : ProcResource<1>;
def BWPort7 : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def BWPort01  : ProcResGroup<[BWPort0, BWPort1]>;
def BWPort23  : ProcResGroup<[BWPort2, BWPort3]>;
def BWPort237 : ProcResGroup<[BWPort2, BWPort3, BWPort7]>;
def BWPort04  : ProcResGroup<[BWPort0, BWPort4]>;
def BWPort05  : ProcResGroup<[BWPort0, BWPort5]>;
def BWPort06  : ProcResGroup<[BWPort0, BWPort6]>;
def BWPort15  : ProcResGroup<[BWPort1, BWPort5]>;
def BWPort16  : ProcResGroup<[BWPort1, BWPort6]>;
def BWPort56  : ProcResGroup<[BWPort5, BWPort6]>;
def BWPort015 : ProcResGroup<[BWPort0, BWPort1, BWPort5]>;
def BWPort056 : ProcResGroup<[BWPort0, BWPort5, BWPort6]>;
def BWPort0156: ProcResGroup<[BWPort0, BWPort1, BWPort5, BWPort6]>;

// 60 Entry Unified Scheduler
def BWPortAny : ProcResGroup<[BWPort0, BWPort1, BWPort2, BWPort3, BWPort4,
                              BWPort5, BWPort6, BWPort7]> {
  let BufferSize=60;
}

// Integer division issued on port 0.
def BWDivider : ProcResource<1>;
// FP division and sqrt on port 0.
def BWFPDivider : ProcResource<1>;

// Loads are 5 cycles, so ReadAfterLd registers needn't be available until 5
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 5>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass BWWriteResPair<X86FoldableSchedWrite SchedRW,
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
  def : WriteRes<SchedRW.Folded, !listconcat([BWPort23], ExePorts)> {
    let Latency = !add(Lat, LoadLat);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = !add(UOps, 1);
  }
}

// A folded store needs a cycle on port 4 for the store data, and an extra port
// 2/3/7 cycle to recompute the address.
def : WriteRes<WriteRMW, [BWPort237,BWPort4]>;

// Arithmetic.
defm : BWWriteResPair<WriteALU,    [BWPort0156], 1>; // Simple integer ALU op.
defm : BWWriteResPair<WriteADC,    [BWPort06], 1>; // Integer ALU + flags op.
defm : BWWriteResPair<WriteIMul,   [BWPort1], 3>; // Integer multiplication.
defm : BWWriteResPair<WriteIMul64, [BWPort1], 3>; // Integer 64-bit multiplication.

defm : BWWriteResPair<WriteDiv8,   [BWPort0, BWDivider], 25, [1, 10]>;
defm : BWWriteResPair<WriteDiv16,  [BWPort0, BWDivider], 25, [1, 10]>;
defm : BWWriteResPair<WriteDiv32,  [BWPort0, BWDivider], 25, [1, 10]>;
defm : BWWriteResPair<WriteDiv64,  [BWPort0, BWDivider], 25, [1, 10]>;
defm : BWWriteResPair<WriteIDiv8,  [BWPort0, BWDivider], 25, [1, 10]>;
defm : BWWriteResPair<WriteIDiv16, [BWPort0, BWDivider], 25, [1, 10]>;
defm : BWWriteResPair<WriteIDiv32, [BWPort0, BWDivider], 25, [1, 10]>;
defm : BWWriteResPair<WriteIDiv64, [BWPort0, BWDivider], 25, [1, 10]>;

defm : BWWriteResPair<WriteCRC32, [BWPort1],   3>;
def : WriteRes<WriteIMulH, []> { let Latency = 3; } // Integer multiplication, high part.

def : WriteRes<WriteLEA, [BWPort15]>; // LEA instructions can't fold loads.

defm : BWWriteResPair<WriteCMOV,  [BWPort06], 1>; // Conditional move.
defm : BWWriteResPair<WriteCMOV2, [BWPort06,BWPort0156], 2, [1,1], 2>; // // Conditional (CF + ZF flag) move.
defm : X86WriteRes<WriteFCMOV, [BWPort1], 3, [1], 1>; // x87 conditional move.

def  : WriteRes<WriteSETCC, [BWPort06]>; // Setcc.
def  : WriteRes<WriteSETCCStore, [BWPort06,BWPort4,BWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

// Bit counts.
defm : BWWriteResPair<WriteBitScan, [BWPort1], 3>;
defm : BWWriteResPair<WriteLZCNT,   [BWPort1], 3>;
defm : BWWriteResPair<WriteTZCNT,   [BWPort1], 3>;
defm : BWWriteResPair<WritePOPCNT,  [BWPort1], 3>;

// Integer shifts and rotates.
defm : BWWriteResPair<WriteShift, [BWPort06],  1>;

// BMI1 BEXTR, BMI2 BZHI
defm : BWWriteResPair<WriteBEXTR, [BWPort06,BWPort15], 2, [1,1], 2>;
defm : BWWriteResPair<WriteBZHI, [BWPort15], 1>;

// Loads, stores, and moves, not folded with other operations.
defm : X86WriteRes<WriteLoad,    [BWPort23], 5, [1], 1>;
defm : X86WriteRes<WriteStore,   [BWPort237, BWPort4], 1, [1,1], 1>;
defm : X86WriteRes<WriteStoreNT, [BWPort237, BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteMove,    [BWPort0156], 1, [1,1], 1>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
def : WriteRes<WriteZero,  []>;

// Treat misc copies as a move.
def : InstRW<[WriteMove], (instrs COPY)>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : BWWriteResPair<WriteJump,  [BWPort06],   1>;

// Floating point. This covers both scalar and vector operations.
defm : X86WriteRes<WriteFLoad,         [BWPort23], 5, [1], 1>;
defm : X86WriteRes<WriteFLoadX,        [BWPort23], 5, [1], 1>;
defm : X86WriteRes<WriteFLoadY,        [BWPort23], 6, [1], 1>;
defm : X86WriteRes<WriteFMaskedLoad,   [BWPort23,BWPort5], 7, [1,2], 3>;
defm : X86WriteRes<WriteFMaskedLoadY,  [BWPort23,BWPort5], 8, [1,2], 3>;
defm : X86WriteRes<WriteFStore,        [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteFStoreX,       [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteFStoreY,       [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteFStoreNT,      [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteFStoreNTX,     [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteFStoreNTY,     [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteFMaskedStore,  [BWPort0,BWPort4,BWPort237,BWPort15], 5, [1,1,1,1], 4>;
defm : X86WriteRes<WriteFMaskedStoreY, [BWPort0,BWPort4,BWPort237,BWPort15], 5, [1,1,1,1], 4>;
defm : X86WriteRes<WriteFMove,         [BWPort5], 1, [1], 1>;
defm : X86WriteRes<WriteFMoveX,        [BWPort5], 1, [1], 1>;
defm : X86WriteRes<WriteFMoveY,        [BWPort5], 1, [1], 1>;

defm : BWWriteResPair<WriteFAdd,    [BWPort1],  3, [1], 1, 5>; // Floating point add/sub.
defm : BWWriteResPair<WriteFAddX,   [BWPort1],  3, [1], 1, 5>; // Floating point add/sub (XMM).
defm : BWWriteResPair<WriteFAddY,   [BWPort1],  3, [1], 1, 6>; // Floating point add/sub (YMM/ZMM).
defm : BWWriteResPair<WriteFAdd64,  [BWPort1],  3, [1], 1, 5>; // Floating point double add/sub.
defm : BWWriteResPair<WriteFAdd64X, [BWPort1],  3, [1], 1, 5>; // Floating point double add/sub (XMM).
defm : BWWriteResPair<WriteFAdd64Y, [BWPort1],  3, [1], 1, 6>; // Floating point double add/sub (YMM/ZMM).

defm : BWWriteResPair<WriteFCmp,    [BWPort1],  3, [1], 1, 5>; // Floating point compare.
defm : BWWriteResPair<WriteFCmpX,   [BWPort1],  3, [1], 1, 5>; // Floating point compare (XMM).
defm : BWWriteResPair<WriteFCmpY,   [BWPort1],  3, [1], 1, 6>; // Floating point compare (YMM/ZMM).
defm : BWWriteResPair<WriteFCmp64,  [BWPort1],  3, [1], 1, 5>; // Floating point double compare.
defm : BWWriteResPair<WriteFCmp64X, [BWPort1],  3, [1], 1, 5>; // Floating point double compare (XMM).
defm : BWWriteResPair<WriteFCmp64Y, [BWPort1],  3, [1], 1, 6>; // Floating point double compare (YMM/ZMM).

defm : BWWriteResPair<WriteFCom,    [BWPort1],  3>; // Floating point compare to flags.

defm : BWWriteResPair<WriteFMul,    [BWPort01], 3, [1], 1, 5>; // Floating point multiplication.
defm : BWWriteResPair<WriteFMulX,   [BWPort01], 3, [1], 1, 5>; // Floating point multiplication (XMM).
defm : BWWriteResPair<WriteFMulY,   [BWPort01], 3, [1], 1, 6>; // Floating point multiplication (YMM/ZMM).
defm : BWWriteResPair<WriteFMul64,  [BWPort01], 3, [1], 1, 5>; // Floating point double multiplication.
defm : BWWriteResPair<WriteFMul64X, [BWPort01], 3, [1], 1, 5>; // Floating point double multiplication (XMM).
defm : BWWriteResPair<WriteFMul64Y, [BWPort01], 3, [1], 1, 6>; // Floating point double multiplication (YMM/ZMM).

//defm : BWWriteResPair<WriteFDiv,     [BWPort0,BWFPDivider], 11, [1,3], 1, 5>; // Floating point division.
defm : BWWriteResPair<WriteFDivX,    [BWPort0,BWFPDivider], 11, [1,5], 1, 5>; // Floating point division (XMM).
defm : BWWriteResPair<WriteFDivY,    [BWPort0,BWPort015,BWFPDivider], 17, [2,1,10], 3, 6>; // Floating point division (YMM).
defm : BWWriteResPair<WriteFDivZ,    [BWPort0,BWPort015,BWFPDivider], 17, [2,1,10], 3, 6>; // Floating point division (ZMM).
//defm : BWWriteResPair<WriteFDiv64,   [BWPort0,BWFPDivider], 14, [1,8], 1, 5>; // Floating point division.
defm : BWWriteResPair<WriteFDiv64X,  [BWPort0,BWFPDivider], 14, [1,8], 1, 5>; // Floating point division (XMM).
defm : BWWriteResPair<WriteFDiv64Y,  [BWPort0,BWPort015,BWFPDivider], 23, [2,1,16], 3, 6>; // Floating point division (YMM).
defm : BWWriteResPair<WriteFDiv64Z,  [BWPort0,BWPort015,BWFPDivider], 23, [2,1,16], 3, 6>; // Floating point division (ZMM).

defm : X86WriteRes<WriteFSqrt,       [BWPort0,BWFPDivider], 11, [1,4], 1>; // Floating point square root.
defm : X86WriteRes<WriteFSqrtLd,     [BWPort0,BWPort23,BWFPDivider], 16, [1,1,7], 2>;
defm : BWWriteResPair<WriteFSqrtX,   [BWPort0,BWFPDivider], 11, [1,7], 1, 5>; // Floating point square root (XMM).
defm : BWWriteResPair<WriteFSqrtY,   [BWPort0,BWPort015,BWFPDivider], 21, [2,1,14], 3, 6>; // Floating point square root (YMM).
defm : BWWriteResPair<WriteFSqrtZ,   [BWPort0,BWPort015,BWFPDivider], 21, [2,1,14], 3, 6>; // Floating point square root (ZMM).
defm : X86WriteRes<WriteFSqrt64,     [BWPort0,BWFPDivider], 16, [1,8], 1>; // Floating point double square root.
defm : X86WriteRes<WriteFSqrt64Ld,   [BWPort0,BWPort23,BWFPDivider], 21, [1,1,14], 2>;
defm : BWWriteResPair<WriteFSqrt64X, [BWPort0,BWFPDivider], 16, [1,14],1, 5>; // Floating point double square root (XMM).
defm : BWWriteResPair<WriteFSqrt64Y, [BWPort0,BWPort015,BWFPDivider], 29, [2,1,28], 3, 6>; // Floating point double square root (YMM).
defm : BWWriteResPair<WriteFSqrt64Z, [BWPort0,BWPort015,BWFPDivider], 29, [2,1,28], 3, 6>; // Floating point double square root (ZMM).
defm : BWWriteResPair<WriteFSqrt80,  [BWPort0,BWFPDivider], 23, [1,9]>; // Floating point long double square root.

defm : BWWriteResPair<WriteFRcp,   [BWPort0],  5, [1], 1, 5>; // Floating point reciprocal estimate.
defm : BWWriteResPair<WriteFRcpX,  [BWPort0],  5, [1], 1, 5>; // Floating point reciprocal estimate (XMM).
defm : BWWriteResPair<WriteFRcpY,  [BWPort0,BWPort015], 11, [2,1], 3, 6>; // Floating point reciprocal estimate (YMM/ZMM).

defm : BWWriteResPair<WriteFRsqrt, [BWPort0],  5, [1], 1, 5>; // Floating point reciprocal square root estimate.
defm : BWWriteResPair<WriteFRsqrtX,[BWPort0],  5, [1], 1, 5>; // Floating point reciprocal square root estimate (XMM).
defm : BWWriteResPair<WriteFRsqrtY,[BWPort0,BWPort015], 11, [2,1], 3, 6>; // Floating point reciprocal square root estimate (YMM/ZMM).

defm : BWWriteResPair<WriteFMA,    [BWPort01], 5, [1], 1, 5>; // Fused Multiply Add.
defm : BWWriteResPair<WriteFMAX,   [BWPort01], 5, [1], 1, 5>; // Fused Multiply Add (XMM).
defm : BWWriteResPair<WriteFMAY,   [BWPort01], 5, [1], 1, 6>; // Fused Multiply Add (YMM/ZMM).
defm : BWWriteResPair<WriteDPPD,   [BWPort0,BWPort1,BWPort5],  9, [1,1,1], 3, 5>; // Floating point double dot product.
defm : BWWriteResPair<WriteDPPS,   [BWPort0,BWPort1,BWPort5], 14, [2,1,1], 4, 5>; // Floating point single dot product.
defm : BWWriteResPair<WriteDPPSY,  [BWPort0,BWPort1,BWPort5], 14, [2,1,1], 4, 6>; // Floating point single dot product (YMM).
defm : BWWriteResPair<WriteFSign,     [BWPort5], 1>; // Floating point fabs/fchs.
defm : X86WriteRes<WriteFRnd,            [BWPort23],  6, [1],   1>; // Floating point rounding.
defm : X86WriteRes<WriteFRndY,           [BWPort23],  6, [1],   1>; // Floating point rounding (YMM/ZMM).
defm : X86WriteRes<WriteFRndLd,  [BWPort1,BWPort23], 11, [2,1], 3>;
defm : X86WriteRes<WriteFRndYLd, [BWPort1,BWPort23], 12, [2,1], 3>;
defm : BWWriteResPair<WriteFLogic,    [BWPort5], 1, [1], 1, 5>; // Floating point and/or/xor logicals.
defm : BWWriteResPair<WriteFLogicY,   [BWPort5], 1, [1], 1, 6>; // Floating point and/or/xor logicals (YMM/ZMM).
defm : BWWriteResPair<WriteFTest,     [BWPort0], 1, [1], 1, 5>; // Floating point TEST instructions.
defm : BWWriteResPair<WriteFTestY,    [BWPort0], 1, [1], 1, 6>; // Floating point TEST instructions (YMM/ZMM).
defm : BWWriteResPair<WriteFShuffle,  [BWPort5], 1, [1], 1, 5>; // Floating point vector shuffles.
defm : BWWriteResPair<WriteFShuffleY, [BWPort5], 1, [1], 1, 6>; // Floating point vector shuffles (YMM/ZMM).
defm : BWWriteResPair<WriteFVarShuffle,  [BWPort5], 1, [1], 1, 5>; // Floating point vector variable shuffles.
defm : BWWriteResPair<WriteFVarShuffleY, [BWPort5], 1, [1], 1, 6>; // Floating point vector variable shuffles.
defm : BWWriteResPair<WriteFBlend,  [BWPort015], 1, [1], 1, 5>; // Floating point vector blends.
defm : BWWriteResPair<WriteFBlendY, [BWPort015], 1, [1], 1, 6>; // Floating point vector blends.
defm : BWWriteResPair<WriteFVarBlend,  [BWPort5], 2, [2], 2, 5>; // Fp vector variable blends.
defm : BWWriteResPair<WriteFVarBlendY, [BWPort5], 2, [2], 2, 6>; // Fp vector variable blends.

// FMA Scheduling helper class.
// class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Vector integer operations.
defm : X86WriteRes<WriteVecLoad,         [BWPort23], 5, [1], 1>;
defm : X86WriteRes<WriteVecLoadX,        [BWPort23], 5, [1], 1>;
defm : X86WriteRes<WriteVecLoadY,        [BWPort23], 6, [1], 1>;
defm : X86WriteRes<WriteVecLoadNT,       [BWPort23], 5, [1], 1>;
defm : X86WriteRes<WriteVecLoadNTY,      [BWPort23], 6, [1], 1>;
defm : X86WriteRes<WriteVecMaskedLoad,   [BWPort23,BWPort5], 7, [1,2], 3>;
defm : X86WriteRes<WriteVecMaskedLoadY,  [BWPort23,BWPort5], 8, [1,2], 3>;
defm : X86WriteRes<WriteVecStore,        [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteVecStoreX,       [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteVecStoreY,       [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteVecStoreNT,      [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteVecStoreNTY,     [BWPort237,BWPort4], 1, [1,1], 2>;
defm : X86WriteRes<WriteVecMaskedStore,  [BWPort0,BWPort4,BWPort237,BWPort15], 5, [1,1,1,1], 4>;
defm : X86WriteRes<WriteVecMaskedStoreY, [BWPort0,BWPort4,BWPort237,BWPort15], 5, [1,1,1,1], 4>;
defm : X86WriteRes<WriteVecMove,         [BWPort015], 1, [1], 1>;
defm : X86WriteRes<WriteVecMoveX,        [BWPort015], 1, [1], 1>;
defm : X86WriteRes<WriteVecMoveY,        [BWPort015], 1, [1], 1>;
defm : X86WriteRes<WriteVecMoveToGpr,    [BWPort0], 1, [1], 1>;
defm : X86WriteRes<WriteVecMoveFromGpr,  [BWPort5], 1, [1], 1>;

defm : X86WriteRes<WriteEMMS,            [BWPort01,BWPort15,BWPort015,BWPort0156], 31, [8,1,21,1], 31>;

defm : BWWriteResPair<WriteVecALU,   [BWPort15], 1, [1], 1, 5>; // Vector integer ALU op, no logicals.
defm : BWWriteResPair<WriteVecALUX,  [BWPort15], 1, [1], 1, 5>; // Vector integer ALU op, no logicals.
defm : BWWriteResPair<WriteVecALUY,  [BWPort15], 1, [1], 1, 6>; // Vector integer ALU op, no logicals (YMM/ZMM).
defm : BWWriteResPair<WriteVecLogic, [BWPort015], 1, [1], 1, 5>; // Vector integer and/or/xor.
defm : BWWriteResPair<WriteVecLogicX,[BWPort015], 1, [1], 1, 5>; // Vector integer and/or/xor.
defm : BWWriteResPair<WriteVecLogicY,[BWPort015], 1, [1], 1, 6>; // Vector integer and/or/xor (YMM/ZMM).
defm : BWWriteResPair<WriteVecTest,  [BWPort0,BWPort5], 2, [1,1], 2, 5>; // Vector integer TEST instructions.
defm : BWWriteResPair<WriteVecTestY, [BWPort0,BWPort5], 4, [1,1], 2, 6>; // Vector integer TEST instructions (YMM/ZMM).
defm : BWWriteResPair<WriteVecIMul,  [BWPort0],  5, [1], 1, 5>; // Vector integer multiply.
defm : BWWriteResPair<WriteVecIMulX, [BWPort0],  5, [1], 1, 5>; // Vector integer multiply.
defm : BWWriteResPair<WriteVecIMulY, [BWPort0],  5, [1], 1, 6>; // Vector integer multiply.
defm : BWWriteResPair<WritePMULLD,   [BWPort0], 10, [2], 2, 5>; // Vector PMULLD.
defm : BWWriteResPair<WritePMULLDY,  [BWPort0], 10, [2], 2, 6>; // Vector PMULLD (YMM/ZMM).
defm : BWWriteResPair<WriteShuffle,  [BWPort5], 1, [1], 1, 5>; // Vector shuffles.
defm : BWWriteResPair<WriteShuffleX, [BWPort5], 1, [1], 1, 5>; // Vector shuffles.
defm : BWWriteResPair<WriteShuffleY, [BWPort5], 1, [1], 1, 6>; // Vector shuffles (YMM/ZMM).
defm : BWWriteResPair<WriteVarShuffle, [BWPort5], 1, [1], 1, 5>; // Vector variable shuffles.
defm : BWWriteResPair<WriteVarShuffleX,[BWPort5], 1, [1], 1, 5>; // Vector variable shuffles.
defm : BWWriteResPair<WriteVarShuffleY,[BWPort5], 1, [1], 1, 6>; // Vector variable shuffles (YMM/ZMM).
defm : BWWriteResPair<WriteBlend,  [BWPort5], 1, [1], 1, 5>; // Vector blends.
defm : BWWriteResPair<WriteBlendY, [BWPort5], 1, [1], 1, 6>; // Vector blends (YMM/ZMM).
defm : BWWriteResPair<WriteVarBlend,  [BWPort5], 2, [2], 2, 5>; // Vector variable blends.
defm : BWWriteResPair<WriteVarBlendY, [BWPort5], 2, [2], 2, 6>; // Vector variable blends (YMM/ZMM).
defm : BWWriteResPair<WriteMPSAD,  [BWPort0, BWPort5], 7, [1, 2], 3, 5>; // Vector MPSAD.
defm : BWWriteResPair<WriteMPSADY, [BWPort0, BWPort5], 7, [1, 2], 3, 6>; // Vector MPSAD.
defm : BWWriteResPair<WritePSADBW,   [BWPort0], 5, [1], 1, 5>; // Vector PSADBW.
defm : BWWriteResPair<WritePSADBWX,  [BWPort0], 5, [1], 1, 5>; // Vector PSADBW.
defm : BWWriteResPair<WritePSADBWY,  [BWPort0], 5, [1], 1, 6>; // Vector PSADBW (YMM/ZMM).
defm : BWWriteResPair<WritePHMINPOS, [BWPort0], 5>; // Vector PHMINPOS.

// Vector integer shifts.
defm : BWWriteResPair<WriteVecShift,     [BWPort0], 1, [1], 1, 5>;
defm : BWWriteResPair<WriteVecShiftX,    [BWPort0,BWPort5],  2, [1,1], 2, 5>;
defm : X86WriteRes<WriteVecShiftY,       [BWPort0,BWPort5],  4, [1,1], 2>;
defm : X86WriteRes<WriteVecShiftYLd,     [BWPort0,BWPort23], 7, [1,1], 2>;

defm : BWWriteResPair<WriteVecShiftImm,  [BWPort0],  1, [1], 1, 5>;
defm : BWWriteResPair<WriteVecShiftImmX, [BWPort0],  1, [1], 1, 5>; // Vector integer immediate shifts (XMM).
defm : BWWriteResPair<WriteVecShiftImmY, [BWPort0],  1, [1], 1, 6>; // Vector integer immediate shifts (YMM/ZMM).
defm : BWWriteResPair<WriteVarVecShift,  [BWPort0, BWPort5], 3, [2,1], 3, 5>; // Variable vector shifts.
defm : BWWriteResPair<WriteVarVecShiftY, [BWPort0, BWPort5], 3, [2,1], 3, 6>; // Variable vector shifts (YMM/ZMM).

// Vector insert/extract operations.
def : WriteRes<WriteVecInsert, [BWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteVecInsertLd, [BWPort5,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
}

def : WriteRes<WriteVecExtract, [BWPort0,BWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : WriteRes<WriteVecExtractSt, [BWPort4,BWPort5,BWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

// Conversion between integer and float.
defm : BWWriteResPair<WriteCvtSS2I,   [BWPort1], 3>;
defm : BWWriteResPair<WriteCvtPS2I,   [BWPort1], 3>;
defm : BWWriteResPair<WriteCvtPS2IY,  [BWPort1], 3>;
defm : BWWriteResPair<WriteCvtSD2I,   [BWPort1], 3>;
defm : BWWriteResPair<WriteCvtPD2I,   [BWPort1], 3>;
defm : BWWriteResPair<WriteCvtPD2IY,  [BWPort1], 3>;

defm : BWWriteResPair<WriteCvtI2SS,   [BWPort1], 4>;
defm : BWWriteResPair<WriteCvtI2PS,   [BWPort1], 4>;
defm : BWWriteResPair<WriteCvtI2PSY,  [BWPort1], 4>;
defm : BWWriteResPair<WriteCvtI2SD,   [BWPort1], 4>;
defm : BWWriteResPair<WriteCvtI2PD,   [BWPort1], 4>;
defm : BWWriteResPair<WriteCvtI2PDY,  [BWPort1], 4>;

defm : BWWriteResPair<WriteCvtSS2SD,  [BWPort1], 3>;
defm : BWWriteResPair<WriteCvtPS2PD,  [BWPort1], 3>;
defm : BWWriteResPair<WriteCvtPS2PDY, [BWPort1], 3>;
defm : BWWriteResPair<WriteCvtSD2SS,  [BWPort1], 3>;
defm : BWWriteResPair<WriteCvtPD2PS,  [BWPort1], 3>;
defm : BWWriteResPair<WriteCvtPD2PSY, [BWPort1], 3>;

defm : X86WriteRes<WriteCvtPH2PS,     [BWPort0,BWPort5], 2, [1,1], 2>;
defm : X86WriteRes<WriteCvtPH2PSY,    [BWPort0,BWPort5], 2, [1,1], 2>;
defm : X86WriteRes<WriteCvtPH2PSLd,  [BWPort0,BWPort23], 6, [1,1], 2>;
defm : X86WriteRes<WriteCvtPH2PSYLd, [BWPort0,BWPort23], 6, [1,1], 2>;

defm : X86WriteRes<WriteCvtPS2PH,    [BWPort1,BWPort5], 4, [1,1], 2>;
defm : X86WriteRes<WriteCvtPS2PHY,   [BWPort1,BWPort5], 6, [1,1], 2>;
defm : X86WriteRes<WriteCvtPS2PHSt,  [BWPort1,BWPort4,BWPort237], 5, [1,1,1], 3>;
defm : X86WriteRes<WriteCvtPS2PHYSt, [BWPort1,BWPort4,BWPort237], 7, [1,1,1], 3>;

// Strings instructions.

// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [BWPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [BWPort0, BWPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [BWPort0, BWPort5, BWPort015, BWPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def : WriteRes<WritePCmpEStrMLd, [BWPort0, BWPort5, BWPort23, BWPort015, BWPort0156]> {
  let Latency = 24;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [BWPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [BWPort0, BWPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [BWPort0, BWPort5, BWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def : WriteRes<WritePCmpEStrILd, [BWPort0, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}

// MOVMSK Instructions.
def : WriteRes<WriteFMOVMSK,    [BWPort0]> { let Latency = 3; }
def : WriteRes<WriteVecMOVMSK,  [BWPort0]> { let Latency = 3; }
def : WriteRes<WriteVecMOVMSKY, [BWPort0]> { let Latency = 3; }
def : WriteRes<WriteMMXMOVMSK,  [BWPort0]> { let Latency = 1; }

// AES instructions.
def : WriteRes<WriteAESDecEnc, [BWPort5]> { // Decryption, encryption.
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [BWPort5, BWPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

def : WriteRes<WriteAESIMC, [BWPort5]> { // InvMixColumn.
  let Latency = 14;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [BWPort5, BWPort23]> {
  let Latency = 19;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}

def : WriteRes<WriteAESKeyGen, [BWPort0, BWPort5, BWPort015]> { // Key Generation.
  let Latency = 29;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,2];
}
def : WriteRes<WriteAESKeyGenLd, [BWPort0, BWPort5, BWPort23, BWPort015]> {
  let Latency = 33;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,1,1];
}

// Carry-less multiplication instructions.
defm : BWWriteResPair<WriteCLMul,  [BWPort0], 5>;

// Catch-all for expensive system instructions.
def : WriteRes<WriteSystem,     [BWPort0156]> { let Latency = 100; } // def WriteSystem : SchedWrite;

// AVX2.
defm : BWWriteResPair<WriteFShuffle256, [BWPort5], 3, [1], 1, 6>; // Fp 256-bit width vector shuffles.
defm : BWWriteResPair<WriteFVarShuffle256, [BWPort5], 3, [1], 1, 6>; // Fp 256-bit width vector variable shuffles.
defm : BWWriteResPair<WriteShuffle256, [BWPort5], 3, [1], 1, 6>;  // 256-bit width vector shuffles.
defm : BWWriteResPair<WriteVarShuffle256, [BWPort5], 3, [1], 1, 6>;  // 256-bit width vector variable shuffles.

// Old microcoded instructions that nobody use.
def : WriteRes<WriteMicrocoded, [BWPort0156]> { let Latency = 100; } // def WriteMicrocoded : SchedWrite;

// Fence instructions.
def : WriteRes<WriteFence,  [BWPort23, BWPort4]>;

// Load/store MXCSR.
def : WriteRes<WriteLDMXCSR, [BWPort0,BWPort23,BWPort0156]> { let Latency = 7; let NumMicroOps = 3; let ResourceCycles = [1,1,1]; }
def : WriteRes<WriteSTMXCSR, [BWPort4,BWPort5,BWPort237]> { let Latency = 2; let NumMicroOps = 3; let ResourceCycles = [1,1,1]; }

// Nop, not very useful expect it provides a model for nops!
def : WriteRes<WriteNop, []>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : BWWriteResPair<WriteFHAdd,   [BWPort1,BWPort5], 5, [1,2], 3, 5>;
defm : BWWriteResPair<WriteFHAddY,  [BWPort1,BWPort5], 5, [1,2], 3, 6>;
defm : BWWriteResPair<WritePHAdd,  [BWPort5,BWPort15], 3, [2,1], 3, 5>;
defm : BWWriteResPair<WritePHAddX, [BWPort5,BWPort15], 3, [2,1], 3, 5>;
defm : BWWriteResPair<WritePHAddY, [BWPort5,BWPort15], 3, [2,1], 3, 6>;

// Remaining instrs.

def BWWriteResGroup1 : SchedWriteRes<[BWPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup1], (instregex "VPSLLVQ(Y?)rr",
                                           "VPSRLVQ(Y?)rr")>;

def BWWriteResGroup2 : SchedWriteRes<[BWPort1]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup2], (instregex "COM(P?)_FST0r",
                                           "UCOM_F(P?)r")>;

def BWWriteResGroup3 : SchedWriteRes<[BWPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup3], (instregex "MMX_MOVQ2DQrr")>;

def BWWriteResGroup4 : SchedWriteRes<[BWPort6]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup4], (instregex "JMP(16|32|64)r")>;

def BWWriteResGroup5 : SchedWriteRes<[BWPort01]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup5], (instrs FINCSTP, FNOP)>;

def BWWriteResGroup6 : SchedWriteRes<[BWPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup6], (instrs CDQ, CQO)>;
def: InstRW<[BWWriteResGroup6], (instregex "BT(16|32|64)ri8",
                                           "BT(16|32|64)rr",
                                           "BTC(16|32|64)ri8",
                                           "BTC(16|32|64)rr",
                                           "BTR(16|32|64)ri8",
                                           "BTR(16|32|64)rr",
                                           "BTS(16|32|64)ri8",
                                           "BTS(16|32|64)rr")>;

def BWWriteResGroup7 : SchedWriteRes<[BWPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup7], (instregex "ANDN(32|64)rr",
                                           "BLSI(32|64)rr",
                                           "BLSMSK(32|64)rr",
                                           "BLSR(32|64)rr")>;

def BWWriteResGroup8 : SchedWriteRes<[BWPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup8], (instregex "VPBLENDD(Y?)rri")>;

def BWWriteResGroup9 : SchedWriteRes<[BWPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup9], (instrs LAHF, SAHF)>; // TODO: This doesnt match Agner's data
def: InstRW<[BWWriteResGroup9], (instregex "NOOP",
                                           "SGDT64m",
                                           "SIDT64m",
                                           "SMSW16m",
                                           "STRm",
                                           "SYSCALL")>;

def BWWriteResGroup10 : SchedWriteRes<[BWPort4,BWPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup10], (instregex "FBSTPm",
                                            "ST_FP(32|64|80)m")>;

def BWWriteResGroup12 : SchedWriteRes<[BWPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup12], (instrs FDECSTP)>;

def BWWriteResGroup13 : SchedWriteRes<[BWPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup13], (instregex "ROL(8|16|32|64)r1",
                                            "ROL(8|16|32|64)ri",
                                            "ROR(8|16|32|64)r1",
                                            "ROR(8|16|32|64)ri")>;

def BWWriteResGroup14 : SchedWriteRes<[BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup14], (instrs LFENCE,
                                         MFENCE,
                                         WAIT,
                                         XGETBV)>;

def BWWriteResGroup15 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup15], (instregex "(V?)CVTPS2PDrr",
                                            "(V?)CVTSS2SDrr")>;

def BWWriteResGroup16 : SchedWriteRes<[BWPort6,BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup16], (instregex "CLFLUSH")>;

def BWWriteResGroup17 : SchedWriteRes<[BWPort01,BWPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup17], (instregex "MMX_MOVDQ2Qrr")>;

def BWWriteResGroup18 : SchedWriteRes<[BWPort237,BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup18], (instrs SFENCE)>;

def BWWriteResGroup19 : SchedWriteRes<[BWPort06,BWPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup19], (instrs BSWAP64r)>;

def BWWriteResGroup19_1 : SchedWriteRes<[BWPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup19_1], (instrs BSWAP32r)>;

def BWWriteResGroup20 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup20], (instrs CWD)>;
def: InstRW<[BWWriteResGroup20], (instrs JCXZ, JECXZ, JRCXZ)>;
def: InstRW<[BWWriteResGroup20], (instregex "ADC8i8",
                                            "ADC8ri",
                                            "SBB8i8",
                                            "SBB8ri",
                                            "SET(A|BE)r")>;

def BWWriteResGroup22 : SchedWriteRes<[BWPort4,BWPort6,BWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup22], (instrs FNSTCW16m)>;

def BWWriteResGroup24 : SchedWriteRes<[BWPort4,BWPort237,BWPort15]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup24], (instregex "MOVBE(16|32|64)mr")>;

def BWWriteResGroup25 : SchedWriteRes<[BWPort4,BWPort237,BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup25], (instrs PUSH16r, PUSH32r, PUSH64r,
                                         STOSB, STOSL, STOSQ, STOSW)>;
def: InstRW<[BWWriteResGroup25], (instregex "PUSH(16|32|64)rmr",
                                            "PUSH64i8")>;

def BWWriteResGroup27 : SchedWriteRes<[BWPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup27], (instregex "MMX_CVTPI2PSirr",
                                            "PDEP(32|64)rr",
                                            "PEXT(32|64)rr",
                                            "SHLD(16|32|64)rri8",
                                            "SHRD(16|32|64)rri8",
                                            "(V?)CVTDQ2PS(Y?)rr")>;

def BWWriteResGroup27_16 : SchedWriteRes<[BWPort1, BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup27_16], (instrs IMUL16rri, IMUL16rri8)>;

def BWWriteResGroup28 : SchedWriteRes<[BWPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup28], (instregex "VPBROADCASTBrr",
                                            "VPBROADCASTWrr")>;

def BWWriteResGroup30 : SchedWriteRes<[BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[BWWriteResGroup30], (instrs XADD8rr, XADD16rr, XADD32rr, XADD64rr,
                                         XCHG8rr, XCHG16rr, XCHG32rr, XCHG64rr,
                                         XCHG16ar, XCHG32ar, XCHG64ar)>;

def BWWriteResGroup33 : SchedWriteRes<[BWPort5,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup33], (instregex "MMX_PACKSSDWirr",
                                            "MMX_PACKSSWBirr",
                                            "MMX_PACKUSWBirr")>;

def BWWriteResGroup34 : SchedWriteRes<[BWPort6,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup34], (instregex "CLD")>;

def BWWriteResGroup35 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup35], (instregex "RCL(8|16|32|64)r1",
                                            "RCL(8|16|32|64)ri",
                                            "RCR(8|16|32|64)r1",
                                            "RCR(8|16|32|64)ri")>;

def BWWriteResGroup36 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup36], (instregex "ROL(8|16|32|64)rCL",
                                            "ROR(8|16|32|64)rCL",
                                            "SAR(8|16|32|64)rCL",
                                            "SHL(8|16|32|64)rCL",
                                            "SHR(8|16|32|64)rCL")>;

def BWWriteResGroup37 : SchedWriteRes<[BWPort4,BWPort6,BWPort237,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup37], (instregex "CALL(16|32|64)r")>;

def BWWriteResGroup38 : SchedWriteRes<[BWPort4,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup38], (instrs CALL64pcrel32)>;
def: InstRW<[BWWriteResGroup38], (instregex "SET(A|BE)m")>;

def BWWriteResGroup39 : SchedWriteRes<[BWPort0,BWPort1]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup39], (instregex "(V?)CVT(T?)SD2SI64rr",
                                            "(V?)CVT(T?)SD2SIrr",
                                            "(V?)CVT(T?)SS2SI64rr",
                                            "(V?)CVT(T?)SS2SIrr")>;

def BWWriteResGroup40 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup40], (instregex "VCVTPS2PDYrr")>;

def BWWriteResGroup41 : SchedWriteRes<[BWPort0,BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup41], (instrs FNSTSW16r)>;

def BWWriteResGroup42 : SchedWriteRes<[BWPort1,BWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup42], (instrs IMUL64r, MUL64r, MULX64rr)>;
def: InstRW<[BWWriteResGroup42], (instregex "MMX_CVTPI2PDirr",
                                            "MMX_CVT(T?)PD2PIirr",
                                            "MMX_CVT(T?)PS2PIirr",
                                            "(V?)CVTDQ2PDrr",
                                            "(V?)CVTPD2PSrr",
                                            "(V?)CVTSD2SSrr",
                                            "(V?)CVTSI642SDrr",
                                            "(V?)CVTSI2SDrr",
                                            "(V?)CVTSI2SSrr",
                                            "(V?)CVT(T?)PD2DQrr")>;

def BWWriteResGroup42_16 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[BWWriteResGroup42_16], (instrs IMUL16r, MUL16r)>;

def BWWriteResGroup43 : SchedWriteRes<[BWPort0,BWPort4,BWPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup43], (instrs FNSTSWm)>;

def BWWriteResGroup44 : SchedWriteRes<[BWPort1,BWPort4,BWPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup44], (instregex "IST(T?)_FP(16|32|64)m",
                                            "IST_F(16|32)m")>;

def BWWriteResGroup45 : SchedWriteRes<[BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[BWWriteResGroup45], (instrs FNCLEX)>;

def BWWriteResGroup46 : SchedWriteRes<[BWPort015,BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[BWWriteResGroup46], (instrs VZEROUPPER)>;

def BWWriteResGroup47 : SchedWriteRes<[BWPort0]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup47], (instregex "(V?)PCMPGTQ(Y?)rr",
                                            "MUL_(FPrST0|FST0r|FrST0)")>;

def BWWriteResGroup49 : SchedWriteRes<[BWPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup49], (instregex "MOVSX(16|32|64)rm16",
                                            "MOVSX(16|32|64)rm32",
                                            "MOVSX(16|32|64)rm8",
                                            "MOVZX(16|32|64)rm16",
                                            "MOVZX(16|32|64)rm8",
                                            "VBROADCASTSSrm",
                                            "(V?)MOVDDUPrm",
                                            "(V?)MOVSHDUPrm",
                                            "(V?)MOVSLDUPrm",
                                            "VPBROADCASTDrm",
                                            "VPBROADCASTQrm")>;

def BWWriteResGroup50 : SchedWriteRes<[BWPort1,BWPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup50], (instregex "(V?)CVTSI642SSrr")>;

def BWWriteResGroup51 : SchedWriteRes<[BWPort1,BWPort6,BWPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup51], (instregex "STR(16|32|64)r")>;

def BWWriteResGroup52 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup52], (instrs IMUL32r, MUL32r, MULX32rr)>;

def BWWriteResGroup54 : SchedWriteRes<[BWPort6,BWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[BWWriteResGroup54], (instrs PAUSE)>;

def BWWriteResGroup55 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[BWWriteResGroup55], (instrs XSETBV)>;

def BWWriteResGroup56 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[BWWriteResGroup56], (instregex "CMPXCHG(8|16|32|64)rr")>;

def BWWriteResGroup57 : SchedWriteRes<[BWPort4,BWPort237,BWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[BWWriteResGroup57], (instregex "PUSHF(16|64)")>;

def BWWriteResGroup58 : SchedWriteRes<[BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup58], (instregex "LD_F(32|64|80)m",
                                            "VBROADCASTF128",
                                            "VBROADCASTI128",
                                            "VBROADCASTSDYrm",
                                            "VBROADCASTSSYrm",
                                            "VMOVDDUPYrm",
                                            "VMOVSHDUPYrm",
                                            "VMOVSLDUPYrm",
                                            "VPBROADCASTDYrm",
                                            "VPBROADCASTQYrm")>;

def BWWriteResGroup59 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup59], (instregex "(V?)CVTPS2PDrm",
                                            "(V?)CVTSS2SDrm",
                                            "VPSLLVQrm",
                                            "VPSRLVQrm")>;

def BWWriteResGroup60 : SchedWriteRes<[BWPort1,BWPort5]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup60], (instregex "VCVTDQ2PDYrr",
                                            "VCVTPD2PSYrr",
                                            "VCVT(T?)PD2DQYrr")>;

def BWWriteResGroup62 : SchedWriteRes<[BWPort6,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup62], (instregex "FARJMP64",
                                            "JMP(16|32|64)m")>;

def BWWriteResGroup63 : SchedWriteRes<[BWPort23,BWPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup63], (instregex "BT(16|32|64)mi8")>;

def BWWriteResGroup64 : SchedWriteRes<[BWPort23,BWPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup64], (instregex "ANDN(32|64)rm",
                                            "BLSI(32|64)rm",
                                            "BLSMSK(32|64)rm",
                                            "BLSR(32|64)rm",
                                            "MOVBE(16|32|64)rm")>;

def BWWriteResGroup65 : SchedWriteRes<[BWPort23,BWPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup65], (instregex "VINSERTF128rm",
                                            "VINSERTI128rm",
                                            "VPBLENDDrmi")>;

def BWWriteResGroup66 : SchedWriteRes<[BWPort23,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup66], (instrs POP16r, POP32r, POP64r)>;
def: InstRW<[BWWriteResGroup66], (instregex "POP(16|32|64)rmr")>;

def BWWriteResGroup67 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[BWWriteResGroup67], (instregex "SHLD(16|32|64)rrCL",
                                            "SHRD(16|32|64)rrCL")>;

def BWWriteResGroup68 : SchedWriteRes<[BWPort1,BWPort6,BWPort06,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup68], (instregex "SLDT(16|32|64)r")>;

def BWWriteResGroup69 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup69], (instregex "BTC(16|32|64)mi8",
                                            "BTR(16|32|64)mi8",
                                            "BTS(16|32|64)mi8",
                                            "SAR(8|16|32|64)m1",
                                            "SAR(8|16|32|64)mi",
                                            "SHL(8|16|32|64)m1",
                                            "SHL(8|16|32|64)mi",
                                            "SHR(8|16|32|64)m1",
                                            "SHR(8|16|32|64)mi")>;

def BWWriteResGroup70 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup70], (instregex "POP(16|32|64)rmm",
                                            "PUSH(16|32|64)rmm")>;

def BWWriteResGroup71 : SchedWriteRes<[BWPort6,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,5];
}
def: InstRW<[BWWriteResGroup71], (instrs STD)>;

def BWWriteResGroup73 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup73], (instregex "VPSLLVQYrm",
                                            "VPSRLVQYrm")>;

def BWWriteResGroup74 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup74], (instregex "FCOM(P?)(32|64)m")>;

def BWWriteResGroup77 : SchedWriteRes<[BWPort23,BWPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup77], (instregex "VPBLENDDYrmi")>;

def BWWriteResGroup79 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup79], (instregex "MMX_PACKSSDWirm",
                                            "MMX_PACKSSWBirm",
                                            "MMX_PACKUSWBirm")>;

def BWWriteResGroup80 : SchedWriteRes<[BWPort23,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup80], (instrs LEAVE, LEAVE64,
                                         SCASB, SCASL, SCASQ, SCASW)>;

def BWWriteResGroup82 : SchedWriteRes<[BWPort0,BWPort01,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup82], (instrs FLDCW16m)>;

def BWWriteResGroup84 : SchedWriteRes<[BWPort6,BWPort23,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup84], (instrs LRETQ, RETQ)>;

def BWWriteResGroup87 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[BWWriteResGroup87], (instregex "ROL(8|16|32|64)m1",
                                            "ROL(8|16|32|64)mi",
                                            "ROR(8|16|32|64)m1",
                                            "ROR(8|16|32|64)mi")>;

def BWWriteResGroup88 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[BWWriteResGroup88], (instregex "XADD(8|16|32|64)rm")>;

def BWWriteResGroup89 : SchedWriteRes<[BWPort4,BWPort6,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[BWWriteResGroup89], (instregex "CALL(16|32|64)m",
                                            "FARCALL64")>;

def BWWriteResGroup90 : SchedWriteRes<[BWPort6,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [2,2,1,2];
}
def: InstRW<[BWWriteResGroup90], (instrs LOOP)>;

def BWWriteResGroup91 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup91], (instregex "MMX_CVTPI2PSirm",
                                            "PDEP(32|64)rm",
                                            "PEXT(32|64)rm",
                                            "(V?)CVTDQ2PSrm")>;

def BWWriteResGroup91_16 : SchedWriteRes<[BWPort1, BWPort0156, BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup91_16], (instrs IMUL16rmi, IMUL16rmi8)>;

def BWWriteResGroup91_16_2 : SchedWriteRes<[BWPort1, BWPort06, BWPort0156, BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,2,1];
}
def: InstRW<[BWWriteResGroup91_16_2], (instrs IMUL16m, MUL16m)>;

def BWWriteResGroup92 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup92], (instregex "VPMOVSXBDYrm",
                                            "VPMOVSXBQYrm",
                                            "VPMOVSXBWYrm",
                                            "VPMOVSXDQYrm",
                                            "VPMOVSXWDYrm",
                                            "VPMOVSXWQYrm",
                                            "VPMOVZXWDYrm")>;

def BWWriteResGroup97 : SchedWriteRes<[BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[BWWriteResGroup97], (instregex "RCL(8|16|32|64)m1",
                                            "RCL(8|16|32|64)mi",
                                            "RCR(8|16|32|64)m1",
                                            "RCR(8|16|32|64)mi")>;

def BWWriteResGroup98 : SchedWriteRes<[BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,2,1];
}
def: InstRW<[BWWriteResGroup98], (instregex "ROR(8|16|32|64)mCL")>;

def BWWriteResGroup99 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[BWWriteResGroup99], (instregex "XCHG(8|16|32|64)rm")>;

def BWWriteResGroup100 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def : SchedAlias<WriteADCRMW, BWWriteResGroup100>;
def: InstRW<[BWWriteResGroup100], (instregex "CMPXCHG(8|16|32|64)rm",
                                             "ROL(8|16|32|64)mCL",
                                             "SAR(8|16|32|64)mCL",
                                             "SHL(8|16|32|64)mCL",
                                             "SHR(8|16|32|64)mCL")>;

def BWWriteResGroup101 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup101], (instregex "(ADD|SUB|SUBR)_F(32|64)m",
                                             "ILD_F(16|32|64)m",
                                             "VCVTPS2DQYrm",
                                             "VCVTTPS2DQYrm")>;

def BWWriteResGroup105 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup105], (instregex "(V?)CVTSS2SI(64)?rm",
                                             "(V?)CVT(T?)SD2SI64rm",
                                             "(V?)CVT(T?)SD2SIrm",
                                             "VCVTTSS2SI64rm",
                                             "(V?)CVTTSS2SIrm")>;

def BWWriteResGroup106 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup106], (instregex "VCVTPS2PDYrm")>;

def BWWriteResGroup107 : SchedWriteRes<[BWPort1,BWPort5,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup107], (instrs IMUL64m, MUL64m, MULX64rm)>;
def: InstRW<[BWWriteResGroup107], (instregex "CVTPD2PSrm",
                                             "CVT(T?)PD2DQrm",
                                             "MMX_CVTPI2PDirm",
                                             "MMX_CVT(T?)PD2PIirm",
                                             "(V?)CVTDQ2PDrm",
                                             "(V?)CVTSD2SSrm")>;

def BWWriteResGroup108 : SchedWriteRes<[BWPort5,BWPort23,BWPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup108], (instregex "VPBROADCASTB(Y?)rm",
                                             "VPBROADCASTW(Y?)rm")>;

def BWWriteResGroup111 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup111], (instregex "SHLD(16|32|64)mri8",
                                             "SHRD(16|32|64)mri8")>;

def BWWriteResGroup112 : SchedWriteRes<[BWPort23,BWPort06,BWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[BWWriteResGroup112], (instregex "RDRAND(16|32|64)r")>;

def BWWriteResGroup113 : SchedWriteRes<[BWPort1,BWPort6,BWPort23,BWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[BWWriteResGroup113], (instregex "LAR(16|32|64)rm",
                                             "LSL(16|32|64)rm")>;

def BWWriteResGroup115 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup115], (instregex "(V?)PCMPGTQrm")>;

def BWWriteResGroup117 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup117], (instregex "FICOM(P?)(16|32)m")>;

def BWWriteResGroup120 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup120], (instregex "CVTTSS2SI64rm")>;

def BWWriteResGroup121 : SchedWriteRes<[BWPort1,BWPort23,BWPort06,BWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup121], (instrs IMUL32m, MUL32m, MULX32rm)>;

def BWWriteResGroup122_1 : SchedWriteRes<[BWPort0,BWFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3]; // Really 2.5 cycle throughput
}
def : SchedAlias<WriteFDiv, BWWriteResGroup122_1>; // TODO - convert to ZnWriteResFpuPair

def BWWriteResGroup123 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup123], (instregex "MUL_F(32|64)m",
                                             "VPCMPGTQYrm")>;

def BWWriteResGroup128 : SchedWriteRes<[BWPort1,BWPort5,BWPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup128], (instregex "VCVTDQ2PDYrm")>;

def BWWriteResGroup130 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,1,2];
}
def: InstRW<[BWWriteResGroup130], (instregex "SHLD(16|32|64)mrCL",
                                             "SHRD(16|32|64)mrCL")>;

def BWWriteResGroup131 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,2,3];
}
def: InstRW<[BWWriteResGroup131], (instregex "RCL(16|32|64)rCL",
                                             "RCR(16|32|64)rCL")>;

def BWWriteResGroup132 : SchedWriteRes<[BWPort1,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 9;
  let ResourceCycles = [1,4,1,3];
}
def: InstRW<[BWWriteResGroup132], (instregex "RCL8rCL")>;

def BWWriteResGroup133 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 11;
  let ResourceCycles = [2,9];
}
def: InstRW<[BWWriteResGroup133], (instrs LOOPE)>;
def: InstRW<[BWWriteResGroup133], (instrs LOOPNE)>;

def BWWriteResGroup135 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup135], (instregex "(ADD|SUB|SUBR)_FI(16|32)m")>;

def BWWriteResGroup139_1 : SchedWriteRes<[BWPort0,BWFPDivider]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1,4];
}
def : SchedAlias<WriteFDiv64, BWWriteResGroup139_1>; // TODO - convert to ZnWriteResFpuPair

def BWWriteResGroup141 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup141], (instregex "MUL_FI(16|32)m")>;

def BWWriteResGroup144 : SchedWriteRes<[BWPort1,BWPort6,BWPort23,BWPort0156]> {
  let Latency = 14;
  let NumMicroOps = 8;
  let ResourceCycles = [2,2,1,3];
}
def: InstRW<[BWWriteResGroup144], (instregex "LAR(16|32|64)rr")>;

def BWWriteResGroup145 : SchedWriteRes<[BWPort1,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 14;
  let NumMicroOps = 10;
  let ResourceCycles = [2,3,1,4];
}
def: InstRW<[BWWriteResGroup145], (instregex "RCR8rCL")>;

def BWWriteResGroup146 : SchedWriteRes<[BWPort0,BWPort1,BWPort6,BWPort0156]> {
  let Latency = 14;
  let NumMicroOps = 12;
  let ResourceCycles = [2,1,4,5];
}
def: InstRW<[BWWriteResGroup146], (instrs XCH_F)>;

def BWWriteResGroup147 : SchedWriteRes<[BWPort0]> {
  let Latency = 15;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup147], (instregex "DIVR_(FPrST0|FST0r|FrST0)")>;

def BWWriteResGroup149 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 15;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,4,1,2];
}
def: InstRW<[BWWriteResGroup149], (instregex "RCL(8|16|32|64)mCL")>;

def BWWriteResGroup150 : SchedWriteRes<[BWPort0,BWPort23,BWFPDivider]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,5];
}
def : SchedAlias<WriteFDivLd, BWWriteResGroup150>; // TODO - convert to ZnWriteResFpuPair

def BWWriteResGroup153 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 16;
  let NumMicroOps = 14;
  let ResourceCycles = [1,1,1,4,2,5];
}
def: InstRW<[BWWriteResGroup153], (instrs CMPXCHG8B)>;

def BWWriteResGroup154 : SchedWriteRes<[BWPort5]> {
  let Latency = 16;
  let NumMicroOps = 16;
  let ResourceCycles = [16];
}
def: InstRW<[BWWriteResGroup154], (instrs VZEROALL)>;

def BWWriteResGroup159 : SchedWriteRes<[BWPort5,BWPort6,BWPort06,BWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[BWWriteResGroup159], (instrs CPUID)>;
def: InstRW<[BWWriteResGroup159], (instrs RDTSC)>;

def BWWriteResGroup160 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,3,1,3];
}
def: InstRW<[BWWriteResGroup160], (instregex "RCR(8|16|32|64)mCL")>;

def BWWriteResGroup161 : SchedWriteRes<[BWPort0,BWPort23,BWFPDivider]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,8];
}
def : SchedAlias<WriteFDiv64Ld, BWWriteResGroup161>; // TODO - convert to ZnWriteResFpuPair

def BWWriteResGroup165 : SchedWriteRes<[BWPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup165], (instregex "DIV_(FPrST0|FST0r|FrST0)")>;

def BWWriteResGroup167 : SchedWriteRes<[BWPort4,BWPort5,BWPort6,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 20;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,1,2];
}
def: InstRW<[BWWriteResGroup167], (instrs INSB, INSL, INSW)>;

def BWWriteResGroup169 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup169], (instregex "DIV_F(32|64)m")>;

def BWWriteResGroup171 : SchedWriteRes<[BWPort0,BWPort4,BWPort5,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 21;
  let NumMicroOps = 19;
  let ResourceCycles = [2,1,4,1,1,4,6];
}
def: InstRW<[BWWriteResGroup171], (instrs CMPXCHG16B)>;

def BWWriteResGroup172 : SchedWriteRes<[BWPort6,BWPort23,BWPort0156]> {
  let Latency = 22;
  let NumMicroOps = 18;
  let ResourceCycles = [1,1,16];
}
def: InstRW<[BWWriteResGroup172], (instregex "POPF64")>;

def BWWriteResGroup176 : SchedWriteRes<[BWPort6,BWPort23,BWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 19;
  let ResourceCycles = [3,1,15];
}
def: InstRW<[BWWriteResGroup176], (instregex "XRSTOR(64)?")>;

def BWWriteResGroup177 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 24;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup177], (instregex "DIV_FI(16|32)m")>;

def BWWriteResGroup180 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 26;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup180], (instregex "DIVR_F(32|64)m")>;

def BWWriteResGroup182 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 29;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup182], (instregex "DIVR_FI(16|32)m")>;

def BWWriteResGroup183_1 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 22;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,2,1];
}
def: InstRW<[BWWriteResGroup183_1], (instrs VGATHERQPDrm)>;

def BWWriteResGroup183_2 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 9;
  let ResourceCycles = [1,3,4,1];
}
def: InstRW<[BWWriteResGroup183_2], (instrs VGATHERQPDYrm)>;

def BWWriteResGroup183_3 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,2,1];
}
def: InstRW<[BWWriteResGroup183_3], (instrs VGATHERQPSYrm)>;

def BWWriteResGroup183_4 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 25;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,2,1];
}
def: InstRW<[BWWriteResGroup183_4], (instrs VGATHERDPDrm,
                                            VGATHERDPSrm)>;

def BWWriteResGroup183_5 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 26;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,2,1];
}
def: InstRW<[BWWriteResGroup183_5], (instrs VGATHERDPDYrm)>;

def BWWriteResGroup183_6 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 26;
  let NumMicroOps = 14;
  let ResourceCycles = [1,4,8,1];
}
def: InstRW<[BWWriteResGroup183_6], (instrs VGATHERDPSYrm)>;

def BWWriteResGroup183_7 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 27;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,2,1];
}
def: InstRW<[BWWriteResGroup183_7], (instrs VGATHERQPSrm)>;

def BWWriteResGroup185 : SchedWriteRes<[BWPort4,BWPort6,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 29;
  let NumMicroOps = 27;
  let ResourceCycles = [1,5,1,1,19];
}
def: InstRW<[BWWriteResGroup185], (instrs XSAVE64)>;

def BWWriteResGroup186 : SchedWriteRes<[BWPort4,BWPort6,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 30;
  let NumMicroOps = 28;
  let ResourceCycles = [1,6,1,1,19];
}
def: InstRW<[BWWriteResGroup186], (instrs XSAVE)>;
def: InstRW<[BWWriteResGroup186], (instregex "XSAVEC", "XSAVES", "XSAVEOPT")>;

def BWWriteResGroup190 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23,BWPort0156]> {
  let Latency = 34;
  let NumMicroOps = 8;
  let ResourceCycles = [2,2,2,1,1];
}
def: InstRW<[BWWriteResGroup190], (instregex "DIV(8|16|32|64)m")>;

def BWWriteResGroup191 : SchedWriteRes<[BWPort5,BWPort6,BWPort23,BWPort06,BWPort0156]> {
  let Latency = 34;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[BWWriteResGroup191], (instregex "IN(8|16|32)ri",
                                             "IN(8|16|32)rr")>;

def BWWriteResGroup193 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23,BWPort0156]> {
  let Latency = 35;
  let NumMicroOps = 8;
  let ResourceCycles = [2,2,2,1,1];
}
def: InstRW<[BWWriteResGroup193], (instregex "IDIV(8|16|32|64)m")>;

def BWWriteResGroup194 : SchedWriteRes<[BWPort5,BWPort6,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[BWWriteResGroup194], (instregex "OUT(8|16|32)ir",
                                             "OUT(8|16|32)rr")>;

def BWWriteResGroup196 : SchedWriteRes<[BWPort5,BWPort0156]> {
  let Latency = 42;
  let NumMicroOps = 22;
  let ResourceCycles = [2,20];
}
def: InstRW<[BWWriteResGroup196], (instrs RDTSCP)>;

def BWWriteResGroup197 : SchedWriteRes<[BWPort0,BWPort01,BWPort23,BWPort05,BWPort06,BWPort015,BWPort0156]> {
  let Latency = 60;
  let NumMicroOps = 64;
  let ResourceCycles = [2,2,8,1,10,2,39];
}
def: InstRW<[BWWriteResGroup197], (instrs FLDENVm)>;

def BWWriteResGroup198 : SchedWriteRes<[BWPort0,BWPort6,BWPort23,BWPort05,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 63;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[BWWriteResGroup198], (instrs FXRSTOR64)>;

def BWWriteResGroup199 : SchedWriteRes<[BWPort0,BWPort6,BWPort23,BWPort05,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 63;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[BWWriteResGroup199], (instrs FXRSTOR)>;

def BWWriteResGroup200 : SchedWriteRes<[BWPort5,BWPort01,BWPort0156]> {
  let Latency = 75;
  let NumMicroOps = 15;
  let ResourceCycles = [6,3,6];
}
def: InstRW<[BWWriteResGroup200], (instrs FNINIT)>;

def BWWriteResGroup201 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort6,BWPort01,BWPort0156]> {
  let Latency = 80;
  let NumMicroOps = 32;
  let ResourceCycles = [7,7,3,3,1,11];
}
def: InstRW<[BWWriteResGroup201], (instregex "DIV(16|32|64)r")>;

def BWWriteResGroup202 : SchedWriteRes<[BWPort0,BWPort1,BWPort4,BWPort5,BWPort6,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 115;
  let NumMicroOps = 100;
  let ResourceCycles = [9,9,11,8,1,11,21,30];
}
def: InstRW<[BWWriteResGroup202], (instrs FSTENVm)>;

} // SchedModel

