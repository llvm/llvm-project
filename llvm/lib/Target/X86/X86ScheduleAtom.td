//===- X86ScheduleAtom.td - X86 Atom Scheduling Definitions -*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the schedule class data for the Intel Atom
// in order (Saltwell-32nm/Bonnell-45nm) processors.
//
//===----------------------------------------------------------------------===//

//
// Scheduling information derived from the "Intel 64 and IA32 Architectures
// Optimization Reference Manual", Chapter 13, Section 4.

// Atom machine model.
def AtomModel : SchedMachineModel {
  let IssueWidth = 2;  // Allows 2 instructions per scheduling group.
  let MicroOpBufferSize = 0; // In-order execution, always hide latency.
  let LoadLatency = 3; // Expected cycles, may be overriden.
  let HighLatency = 30;// Expected, may be overriden.

  // On the Atom, the throughput for taken branches is 2 cycles. For small
  // simple loops, expand by a small factor to hide the backedge cost.
  let LoopMicroOpBufferSize = 10;
  let PostRAScheduler = 1;
  let CompleteModel = 0;
}

let SchedModel = AtomModel in {

// Functional Units
def AtomPort0 : ProcResource<1>; // ALU: ALU0, shift/rotate, load/store
                                 // SIMD/FP: SIMD ALU, Shuffle,SIMD/FP multiply, divide
def AtomPort1 : ProcResource<1>; // ALU: ALU1, bit processing, jump, and LEA
                                 // SIMD/FP: SIMD ALU, FP Adder

def AtomPort01 : ProcResGroup<[AtomPort0, AtomPort1]>;

// Loads are 3 cycles, so ReadAfterLd registers needn't be available until 3
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 3>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when dispatched by the schedulers.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass AtomWriteResPair<X86FoldableSchedWrite SchedRW,
                            list<ProcResourceKind> RRPorts,
                            list<ProcResourceKind> RMPorts,
                            int RRLat = 1, int RMLat = 1,
                            list<int> RRRes = [1],
                            list<int> RMRes = [1]> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, RRPorts> {
    let Latency = RRLat;
    let ResourceCycles = RRRes;
  }

  // Memory variant also uses a cycle on JLAGU and adds 3 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, RMPorts> {
    let Latency = RMLat;
    let ResourceCycles = RMRes;
  }
}

// A folded store needs a cycle on Port0 for the store data.
def : WriteRes<WriteRMW, [AtomPort0]>;

////////////////////////////////////////////////////////////////////////////////
// Arithmetic.
////////////////////////////////////////////////////////////////////////////////

defm : AtomWriteResPair<WriteALU,    [AtomPort01], [AtomPort0]>;
defm : AtomWriteResPair<WriteIMul,   [AtomPort01], [AtomPort01],  7,  7,  [7],  [7]>;
defm : AtomWriteResPair<WriteIMul64, [AtomPort01], [AtomPort01], 12, 12, [12], [12]>;

defm : AtomWriteResPair<WriteDiv8,   [AtomPort01], [AtomPort01], 50, 68, [50], [68]>;
defm : AtomWriteResPair<WriteDiv16,  [AtomPort01], [AtomPort01], 50, 50, [50], [50]>;
defm : AtomWriteResPair<WriteDiv32,  [AtomPort01], [AtomPort01], 50, 50, [50], [50]>;
defm : AtomWriteResPair<WriteDiv64,  [AtomPort01], [AtomPort01],130,130,[130],[130]>;
defm : AtomWriteResPair<WriteIDiv8,  [AtomPort01], [AtomPort01], 62, 62, [62], [62]>;
defm : AtomWriteResPair<WriteIDiv16, [AtomPort01], [AtomPort01], 62, 62, [62], [62]>;
defm : AtomWriteResPair<WriteIDiv32, [AtomPort01], [AtomPort01], 62, 62, [62], [62]>;
defm : AtomWriteResPair<WriteIDiv64, [AtomPort01], [AtomPort01],130,130,[130],[130]>;

defm : AtomWriteResPair<WriteCRC32, [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.

defm : AtomWriteResPair<WriteCMOV,  [AtomPort01], [AtomPort0]>;

def  : WriteRes<WriteSETCC, [AtomPort01]>;
def  : WriteRes<WriteSETCCStore, [AtomPort01]> {
  let Latency = 2;
  let ResourceCycles = [2];
}

def : WriteRes<WriteIMulH, [AtomPort01]>; // NOTE: Doesn't exist on Atom.

// This is for simple LEAs with one or two input operands.
def : WriteRes<WriteLEA, [AtomPort1]>;

def AtomWriteIMul16Ld : SchedWriteRes<[AtomPort01]> {
  let Latency = 8;
  let ResourceCycles = [8];
}
def : InstRW<[AtomWriteIMul16Ld], (instrs MUL16m, IMUL16m)>;

def AtomWriteIMul32 : SchedWriteRes<[AtomPort01]> {
  let Latency = 6;
  let ResourceCycles = [6];
}
def : InstRW<[AtomWriteIMul32], (instrs MUL32r, IMUL32r)>;

def AtomWriteIMul64I : SchedWriteRes<[AtomPort01]> {
  let Latency = 14;
  let ResourceCycles = [14];
}
def : InstRW<[AtomWriteIMul64I], (instrs IMUL64rri8, IMUL64rri32,
                                         IMUL64rmi8, IMUL64rmi32)>;

// Bit counts.
defm : AtomWriteResPair<WriteBitScan, [AtomPort01], [AtomPort01], 16, 16, [16], [16]>;
defm : AtomWriteResPair<WritePOPCNT,  [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteLZCNT,   [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteTZCNT,   [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.

// BMI1 BEXTR, BMI2 BZHI
defm : AtomWriteResPair<WriteBEXTR, [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteBZHI,  [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.

////////////////////////////////////////////////////////////////////////////////
// Integer shifts and rotates.
////////////////////////////////////////////////////////////////////////////////

defm : AtomWriteResPair<WriteShift, [AtomPort0], [AtomPort0]>;

////////////////////////////////////////////////////////////////////////////////
// Loads, stores, and moves, not folded with other operations.
////////////////////////////////////////////////////////////////////////////////

def : WriteRes<WriteLoad,  [AtomPort0]>;
def : WriteRes<WriteStore, [AtomPort0]>;
def : WriteRes<WriteMove,  [AtomPort01]>;

// Treat misc copies as a move.
def : InstRW<[WriteMove], (instrs COPY)>;

////////////////////////////////////////////////////////////////////////////////
// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
////////////////////////////////////////////////////////////////////////////////

def : WriteRes<WriteZero,  []>;

////////////////////////////////////////////////////////////////////////////////
// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
////////////////////////////////////////////////////////////////////////////////

defm : AtomWriteResPair<WriteJump, [AtomPort1], [AtomPort1]>;

////////////////////////////////////////////////////////////////////////////////
// Special case scheduling classes.
////////////////////////////////////////////////////////////////////////////////

def : WriteRes<WriteSystem,     [AtomPort01]> { let Latency = 100; }
def : WriteRes<WriteMicrocoded, [AtomPort01]> { let Latency = 100; }
def : WriteRes<WriteFence,      [AtomPort0]>;

// Nops don't have dependencies, so there's no actual latency, but we set this
// to '1' to tell the scheduler that the nop uses an ALU slot for a cycle.
def : WriteRes<WriteNop, [AtomPort01]>;

////////////////////////////////////////////////////////////////////////////////
// Floating point. This covers both scalar and vector operations.
////////////////////////////////////////////////////////////////////////////////

def  : WriteRes<WriteFLoad,         [AtomPort0]>;
def  : WriteRes<WriteFMaskedLoad,   [AtomPort0]>;
def  : WriteRes<WriteFMaskedLoadY,  [AtomPort0]>;

def  : WriteRes<WriteFStore,        [AtomPort0]>;
def  : WriteRes<WriteFMaskedStore,  [AtomPort0]>;
def  : WriteRes<WriteFMaskedStoreY, [AtomPort0]>;

def  : WriteRes<WriteFMove,  [AtomPort01]>;
defm : X86WriteRes<WriteEMMS,[AtomPort01], 5, [5], 1>;

defm : AtomWriteResPair<WriteFAdd,           [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFAddX,          [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFAddY,          [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFAdd64,         [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFAdd64X,       [AtomPort01], [AtomPort01],  6,  7,  [6],  [7]>;
defm : AtomWriteResPair<WriteFAdd64Y,       [AtomPort01], [AtomPort01],  6,  7,  [6],  [7]>;
defm : AtomWriteResPair<WriteFCmp,           [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFCmpX,          [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFCmpY,          [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFCmp64,         [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFCmp64X,       [AtomPort01], [AtomPort01],  6,  7,  [6],  [7]>;
defm : AtomWriteResPair<WriteFCmp64Y,       [AtomPort01], [AtomPort01],  6,  7,  [6],  [7]>;
defm : AtomWriteResPair<WriteFCom,           [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFMul,           [AtomPort0],  [AtomPort0],  4,  4,  [4],  [4]>;
defm : AtomWriteResPair<WriteFMulX,          [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFMulY,          [AtomPort0],  [AtomPort0],  4,  4,  [4],  [4]>;
defm : AtomWriteResPair<WriteFMul64,         [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFMul64X,       [AtomPort01], [AtomPort01],  9, 10,  [9], [10]>;
defm : AtomWriteResPair<WriteFMul64Y,       [AtomPort01], [AtomPort01],  9, 10,  [9], [10]>;
defm : AtomWriteResPair<WriteFRcp,           [AtomPort0],  [AtomPort0],  4,  4,  [4],  [4]>;
defm : AtomWriteResPair<WriteFRcpX,         [AtomPort01], [AtomPort01],  9, 10,  [9], [10]>;
defm : AtomWriteResPair<WriteFRcpY,         [AtomPort01], [AtomPort01],  9, 10,  [9], [10]>;
defm : AtomWriteResPair<WriteFRsqrt,         [AtomPort0],  [AtomPort0],  4,  4,  [4],  [4]>;
defm : AtomWriteResPair<WriteFRsqrtX,       [AtomPort01], [AtomPort01],  9, 10,  [9], [10]>;
defm : AtomWriteResPair<WriteFRsqrtY,       [AtomPort01], [AtomPort01],  9, 10,  [9], [10]>;
defm : AtomWriteResPair<WriteFDiv,          [AtomPort01], [AtomPort01], 34, 34, [34], [34]>;
defm : AtomWriteResPair<WriteFDivX,         [AtomPort01], [AtomPort01], 70, 70, [70], [70]>;
defm : AtomWriteResPair<WriteFDivY,         [AtomPort01], [AtomPort01], 70, 70, [70], [70]>;
defm : AtomWriteResPair<WriteFDivZ,         [AtomPort01], [AtomPort01], 70, 70, [70], [70]>;
defm : AtomWriteResPair<WriteFDiv64,        [AtomPort01], [AtomPort01], 62, 62, [62], [62]>;
defm : AtomWriteResPair<WriteFDiv64X,       [AtomPort01], [AtomPort01],125,125,[125],[125]>;
defm : AtomWriteResPair<WriteFDiv64Y,       [AtomPort01], [AtomPort01],125,125,[125],[125]>;
defm : AtomWriteResPair<WriteFDiv64Z,       [AtomPort01], [AtomPort01],125,125,[125],[125]>;
defm : AtomWriteResPair<WriteFSqrt,         [AtomPort01], [AtomPort01], 34, 34, [34], [34]>;
defm : AtomWriteResPair<WriteFSqrtX,        [AtomPort01], [AtomPort01], 70, 70, [70], [70]>;
defm : AtomWriteResPair<WriteFSqrtY,        [AtomPort01], [AtomPort01], 70, 70, [70], [70]>;
defm : AtomWriteResPair<WriteFSqrtZ,        [AtomPort01], [AtomPort01], 70, 70, [70], [70]>;
defm : AtomWriteResPair<WriteFSqrt64,       [AtomPort01], [AtomPort01], 62, 62, [62], [62]>;
defm : AtomWriteResPair<WriteFSqrt64X,      [AtomPort01], [AtomPort01],125,125,[125],[125]>;
defm : AtomWriteResPair<WriteFSqrt64Y,      [AtomPort01], [AtomPort01],125,125,[125],[125]>;
defm : AtomWriteResPair<WriteFSqrt64Z,      [AtomPort01], [AtomPort01],125,125,[125],[125]>;
defm : AtomWriteResPair<WriteFSqrt80,       [AtomPort01], [AtomPort01], 71, 71, [71], [71]>;
defm : AtomWriteResPair<WriteFSign,          [AtomPort1],  [AtomPort1]>;
defm : AtomWriteResPair<WriteFRnd,           [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFRndY,          [AtomPort0],  [AtomPort0],  5,  5,  [5],  [5]>;
defm : AtomWriteResPair<WriteFLogic,        [AtomPort01],  [AtomPort0]>;
defm : AtomWriteResPair<WriteFLogicY,       [AtomPort01],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFTest,         [AtomPort01],  [AtomPort0]>;
defm : AtomWriteResPair<WriteFTestY ,       [AtomPort01],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFShuffle,       [AtomPort0],  [AtomPort0]>;
defm : AtomWriteResPair<WriteFShuffleY,      [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFVarShuffle,    [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFVarShuffleY,   [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFMA,            [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFMAX,           [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFMAY,           [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteDPPD,           [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteDPPS,           [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteDPPSY,          [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFBlend,         [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFBlendY,        [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFVarBlend,      [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFVarBlendY,     [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFShuffle256,    [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteFVarShuffle256, [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.

////////////////////////////////////////////////////////////////////////////////
// Conversions.
////////////////////////////////////////////////////////////////////////////////

defm : AtomWriteResPair<WriteCvtF2I, [AtomPort01], [AtomPort01], 8, 9, [8], [9]>; // Float -> Integer.
defm : AtomWriteResPair<WriteCvtI2F, [AtomPort01], [AtomPort01], 6, 7, [6], [7]>; // Integer -> Float.
defm : AtomWriteResPair<WriteCvtF2F, [AtomPort01], [AtomPort01], 6, 7, [6], [7]>; // Float -> Float size conversion.
def  : WriteRes<WriteCvtF2FSt, [AtomPort0]>; // NOTE: Doesn't exist on Atom.

////////////////////////////////////////////////////////////////////////////////
// Vector integer operations.
////////////////////////////////////////////////////////////////////////////////

def  : WriteRes<WriteVecLoad,         [AtomPort0]>;
def  : WriteRes<WriteVecMaskedLoad,   [AtomPort0]>;
def  : WriteRes<WriteVecMaskedLoadY,  [AtomPort0]>;

def  : WriteRes<WriteVecStore,        [AtomPort0]>;
def  : WriteRes<WriteVecMaskedStore,  [AtomPort0]>;
def  : WriteRes<WriteVecMaskedStoreY, [AtomPort0]>;

def  : WriteRes<WriteVecMove,  [AtomPort01]>;

defm : AtomWriteResPair<WriteVecALU,       [AtomPort01],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WriteVecALUY,      [AtomPort01],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WriteVecLogic,     [AtomPort01],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WriteVecLogicY,    [AtomPort01],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WriteVecTest,      [AtomPort01],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WriteVecTestY,     [AtomPort01],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WriteVecShift,     [AtomPort01], [AtomPort01], 2, 3, [2], [3]>;
defm : AtomWriteResPair<WriteVecShiftX,    [AtomPort01], [AtomPort01], 2, 3, [2], [3]>;
defm : AtomWriteResPair<WriteVecShiftY,    [AtomPort01], [AtomPort01], 2, 3, [2], [3]>;
defm : AtomWriteResPair<WriteVecShiftImm,  [AtomPort01], [AtomPort01], 1, 1, [1], [1]>;
defm : AtomWriteResPair<WriteVecShiftImmX, [AtomPort01], [AtomPort01], 1, 1, [1], [1]>;
defm : AtomWriteResPair<WriteVecShiftImmY, [AtomPort01], [AtomPort01], 1, 1, [1], [1]>;
defm : AtomWriteResPair<WriteVecIMul,       [AtomPort0],  [AtomPort0], 4, 4, [4], [4]>;
defm : AtomWriteResPair<WriteVecIMulX,      [AtomPort0],  [AtomPort0], 5, 5, [5], [5]>;
defm : AtomWriteResPair<WriteVecIMulY,      [AtomPort0],  [AtomPort0], 5, 5, [5], [5]>;
defm : AtomWriteResPair<WritePMULLD,       [AtomPort01],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WritePMULLDY,      [AtomPort01],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WritePHMINPOS,      [AtomPort0],  [AtomPort0], 5, 5, [5], [5]>;
defm : AtomWriteResPair<WriteMPSAD,        [AtomPort01],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WriteMPSADY,       [AtomPort01],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WritePSADBW,        [AtomPort0],  [AtomPort0], 5, 5, [5], [5]>;
defm : AtomWriteResPair<WritePSADBWY,       [AtomPort0],  [AtomPort0], 5, 5, [5], [5]>;
defm : AtomWriteResPair<WriteShuffle,       [AtomPort0],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WriteShuffleY,      [AtomPort0],  [AtomPort0], 1, 1>;
defm : AtomWriteResPair<WriteVarShuffle,   [AtomPort01], [AtomPort01], 4, 5, [4], [5]>;
defm : AtomWriteResPair<WriteVarShuffleY,  [AtomPort01], [AtomPort01], 4, 5, [4], [5]>;
defm : AtomWriteResPair<WriteBlend,         [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteBlendY,        [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteVarBlend,      [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteVarBlendY,     [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteShuffle256,    [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteVarShuffle256, [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteVarVecShift,   [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteVarVecShiftY,  [AtomPort0],  [AtomPort0]>; // NOTE: Doesn't exist on Atom.

////////////////////////////////////////////////////////////////////////////////
// Vector insert/extract operations.
////////////////////////////////////////////////////////////////////////////////

defm : AtomWriteResPair<WriteVecInsert,     [AtomPort0],  [AtomPort0], 1, 1>;
def  : WriteRes<WriteVecExtract,   [AtomPort0]>;
def  : WriteRes<WriteVecExtractSt, [AtomPort0]>;

////////////////////////////////////////////////////////////////////////////////
// SSE42 String instructions.
////////////////////////////////////////////////////////////////////////////////

defm : AtomWriteResPair<WritePCmpIStrI, [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WritePCmpIStrM, [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WritePCmpEStrI, [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WritePCmpEStrM, [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.

////////////////////////////////////////////////////////////////////////////////
// MOVMSK Instructions.
////////////////////////////////////////////////////////////////////////////////

def : WriteRes<WriteFMOVMSK,    [AtomPort0]> { let Latency = 3; let ResourceCycles = [3]; }
def : WriteRes<WriteVecMOVMSK,  [AtomPort0]> { let Latency = 3; let ResourceCycles = [3]; }
def : WriteRes<WriteVecMOVMSKY, [AtomPort0]> { let Latency = 3; let ResourceCycles = [3]; }
def : WriteRes<WriteMMXMOVMSK,  [AtomPort0]> { let Latency = 3; let ResourceCycles = [3]; }

////////////////////////////////////////////////////////////////////////////////
// AES Instructions.
////////////////////////////////////////////////////////////////////////////////

defm : AtomWriteResPair<WriteAESIMC,    [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteAESKeyGen, [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.
defm : AtomWriteResPair<WriteAESDecEnc, [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : AtomWriteResPair<WriteFHAdd,  [AtomPort01], [AtomPort01], 8, 9, [8], [9]>;
defm : AtomWriteResPair<WriteFHAddY, [AtomPort01], [AtomPort01], 8, 9, [8], [9]>;
defm : AtomWriteResPair<WritePHAdd,  [AtomPort01], [AtomPort01], 7, 8, [7], [8]>;
defm : AtomWriteResPair<WritePHAddY, [AtomPort01], [AtomPort01], 7, 8, [7], [8]>;

////////////////////////////////////////////////////////////////////////////////
// Carry-less multiplication instructions.
////////////////////////////////////////////////////////////////////////////////

defm : AtomWriteResPair<WriteCLMul, [AtomPort01], [AtomPort01]>; // NOTE: Doesn't exist on Atom.

////////////////////////////////////////////////////////////////////////////////
// Load/store MXCSR.
////////////////////////////////////////////////////////////////////////////////

def : WriteRes<WriteLDMXCSR, [AtomPort01]> { let Latency = 5; let ResourceCycles = [5]; }
def : WriteRes<WriteSTMXCSR, [AtomPort01]> { let Latency = 15; let ResourceCycles = [15]; }

////////////////////////////////////////////////////////////////////////////////
// Special Cases.
////////////////////////////////////////////////////////////////////////////////

// Port0
def AtomWrite0_1 : SchedWriteRes<[AtomPort0]> {
  let Latency = 1;
  let ResourceCycles = [1];
}
def : InstRW<[AtomWrite0_1], (instrs FXAM, LD_Frr,
                                     BSWAP32r, BSWAP64r,
                                     DEC8m, DEC16m, DEC32m, DEC64m,
                                     INC8m, INC16m, INC32m, INC64m,
                                     MOVSX64rr32,
                                     MMX_MOVD64rr,
                                     MMX_MOVD64to64rr,
                                     MMX_PSHUFBrr, MMX_PSHUFBrm,
                                     MOVDI2PDIrr,
                                     MOVDI2SSrr,
                                     MOV64toPQIrr,
                                     MOV64toSDrr)>;
def : InstRW<[AtomWrite0_1], (instregex "(ADC|ADD|AND|NEG|NOT|OR|SBB|SUB|XOR)(8|16|32|64)m",
                                        "(RCL|RCR|ROL|ROR|SAR|SHL|SHR)(8|16|32|64)m",
                                        "MOV(S|Z)X(32|64)rr(8|8_NOREX|16)")>;

def AtomWrite0_3 : SchedWriteRes<[AtomPort0]> {
  let Latency = 3;
  let ResourceCycles = [3];
}
def : InstRW<[AtomWrite0_3], (instrs MMX_MOVD64from64rr, MMX_MOVD64grr,
                                     MOVPDI2DIrr, MOVPQIto64rr,
                                     MOVSDto64rr, MOVSS2DIrr)>;

def AtomWrite0_5 : SchedWriteRes<[AtomPort0]> {
  let Latency = 5;
  let ResourceCycles = [5];
}
def : InstRW<[AtomWrite0_5], (instregex "IMUL32(rm|rr)")>;

// Port1
def AtomWrite1_1 : SchedWriteRes<[AtomPort1]> {
  let Latency = 1;
  let ResourceCycles = [1];
}
def : InstRW<[AtomWrite1_1], (instrs FCOMPP)>;
def : InstRW<[AtomWrite1_1], (instregex "UCOM_F(P|PP)?r",
                                        "BT(C|R|S)?(16|32|64)(rr|ri8)")>;

def AtomWrite1_5 : SchedWriteRes<[AtomPort1]> {
  let Latency = 5;
  let ResourceCycles = [5];
}
def : InstRW<[AtomWrite1_5], (instrs MMX_CVTPI2PSirr, MMX_CVTPI2PSirm,
                                     MMX_CVTPS2PIirr, MMX_CVTTPS2PIirr)>;

// Port0 and Port1
def AtomWrite0_1_1 : SchedWriteRes<[AtomPort0, AtomPort1]> {
  let Latency = 1;
  let ResourceCycles = [1, 1];
}
def : InstRW<[AtomWrite0_1_1], (instrs POP32r, POP64r,
                                       POP16rmr, POP32rmr, POP64rmr,
                                       PUSH16r, PUSH32r, PUSH64r,
                                       PUSHi16, PUSHi32,
                                       PUSH16rmr, PUSH32rmr, PUSH64rmr,
                                       PUSH16i8, PUSH32i8, PUSH64i8, PUSH64i32,
                                       XCH_F)>;
def : InstRW<[AtomWrite0_1_1], (instregex "RETI(L|Q|W)$",
                                          "IRET(16|32|64)?")>;

def AtomWrite0_1_5 : SchedWriteRes<[AtomPort0, AtomPort1]> {
  let Latency = 5;
  let ResourceCycles = [5, 5];
}
def : InstRW<[AtomWrite0_1_5], (instrs MMX_CVTPS2PIirm, MMX_CVTTPS2PIirm)>;
def : InstRW<[AtomWrite0_1_5], (instregex "ILD_F(16|32|64)")>;

// Port0 or Port1
def AtomWrite01_1 : SchedWriteRes<[AtomPort01]> {
  let Latency = 1;
  let ResourceCycles = [1];
}
def : InstRW<[AtomWrite01_1], (instrs FDECSTP, FFREE, FFREEP, FINCSTP, LD_F0, WAIT,
                                      LFENCE,
                                      STOSB, STOSL, STOSQ, STOSW,
                                      MOVSSrr, MOVSSrr_REV,
                                      PSLLDQri, PSRLDQri)>;
def : InstRW<[AtomWrite01_1], (instregex "MMX_PACK(SSDW|SSWB|USWB)irr",
                                         "MMX_PUNPCKH(BW|DQ|WD)irr")>;

def AtomWrite01_2 : SchedWriteRes<[AtomPort01]> {
  let Latency = 2;
  let ResourceCycles = [2];
}
def : InstRW<[AtomWrite01_2], (instrs LEAVE, LEAVE64, POP16r,
                                      PUSH16rmm, PUSH32rmm, PUSH64rmm,
                                      LODSB, LODSL, LODSQ, LODSW,
                                      SCASB, SCASL, SCASQ, SCASW,
                                      SHLD32rrCL, SHRD32rrCL,
                                      SHLD32rri8, SHRD32rri8)>;
def : InstRW<[AtomWrite01_2], (instregex "BT(C|R|S)(16|32|64)mi8",
                                         "PUSH(CS|DS|ES|FS|GS|SS)(16|32|64)",
                                         "XADD(8|16|32|64)rr",
                                         "XCHG(8|16|32|64)(ar|rr)",
                                         "(ST|ISTT)_F(P)?(16|32|64)?(m|rr)",
                                         "MMX_P(ADD|SUB)Qirr",
                                         "MOV(S|Z)X16rr8",
                                         "MOV(UPS|UPD|DQU)mr",
                                         "MASKMOVDQU(64)?",
                                         "P(ADD|SUB)Qrr")>;

def AtomWrite01_3 : SchedWriteRes<[AtomPort01]> {
  let Latency = 3;
  let ResourceCycles = [3];
}
def : InstRW<[AtomWrite01_3], (instrs CLD, LDDQUrm,
                                      CMPSB, CMPSL, CMPSQ, CMPSW,
                                      MOVSB, MOVSL, MOVSQ, MOVSW,
                                      POP16rmm, POP32rmm, POP64rmm)>;
def : InstRW<[AtomWrite01_3], (instregex "XADD(8|16|32|64)rm",
                                         "XCHG(8|16|32|64)rm",
                                         "(MMX_)?PH(ADD|SUB)Drr",
                                         "MOV(S|Z)X16rm8",
                                         "MMX_P(ADD|SUB)Qirm",
                                         "MOV(UPS|UPD|DQU)rm",
                                         "P(ADD|SUB)Qrm")>;

def AtomWrite01_4 : SchedWriteRes<[AtomPort01]> {
  let Latency = 4;
  let ResourceCycles = [4];
}
def : InstRW<[AtomWrite01_4], (instrs CBW, CWD, CWDE, CDQ, CDQE, CQO,
                                      JCXZ, JECXZ, JRCXZ,
                                      SHLD32mrCL, SHRD32mrCL,
                                      SHLD32mri8, SHRD32mri8,
                                      LD_F80m,
                                      MMX_PSADBWirr, MMX_PSADBWirm)>;
def : InstRW<[AtomWrite01_4], (instregex "(MMX_)?PH(ADD|SUB)Drm",
                                         "(MMX_)?PEXTRWrr(_REV)?")>;

def AtomWrite01_5 : SchedWriteRes<[AtomPort01]> {
  let Latency = 5;
  let ResourceCycles = [5];
}
def : InstRW<[AtomWrite01_5], (instrs FLDCW16m, ST_FP80m)>;
def : InstRW<[AtomWrite01_5], (instregex "MMX_PH(ADD|SUB)S?Wrr")>;

def AtomWrite01_6 : SchedWriteRes<[AtomPort01]> {
  let Latency = 6;
  let ResourceCycles = [6];
}
def : InstRW<[AtomWrite01_6], (instrs LD_F1, CMPXCHG8rm, INTO, XLAT,
                                      SHLD16rrCL, SHRD16rrCL,
                                      SHLD16rri8, SHRD16rri8,
                                      SHLD16mrCL, SHRD16mrCL,
                                      SHLD16mri8, SHRD16mri8,
                                      CVTPS2DQrr, CVTTPS2DQrr)>;
def : InstRW<[AtomWrite01_6], (instregex "IMUL16rr",
                                         "IST_F(P)?(16|32|64)?m",
                                         "MMX_PH(ADD|SUB)S?Wrm")>;

def AtomWrite01_7 : SchedWriteRes<[AtomPort01]> {
  let Latency = 7;
  let ResourceCycles = [7];
}
def : InstRW<[AtomWrite01_7], (instrs AAD8i8,
                                      CVTDQ2PDrr,
                                      CVTPD2DQrr,
                                      CVTPD2PSrr,
                                      CVTPS2DQrm,
                                      CVTPS2PDrr,
                                      CVTTPD2DQrr,
                                      CVTTPS2DQrm,
                                      MMX_CVTPD2PIirr,
                                      MMX_CVTPI2PDirr,
                                      MMX_CVTTPD2PIirr)>;

def AtomWrite01_8 : SchedWriteRes<[AtomPort01]> {
  let Latency = 8;
  let ResourceCycles = [8];
}
def : InstRW<[AtomWrite01_8], (instrs LOOPE,
                                      PUSHA16, PUSHA32,
                                      SHLD64rrCL, SHRD64rrCL,
                                      FNSTCW16m,
                                      CVTDQ2PDrm,
                                      CVTPD2DQrm,
                                      CVTPD2PSrm,
                                      CVTPS2PDrm,
                                      CVTTPD2DQrm,
                                      MMX_CVTPD2PIirm,
                                      MMX_CVTPI2PDirm,
                                      MMX_CVTTPD2PIirm)>;

def AtomWrite01_9 : SchedWriteRes<[AtomPort01]> {
  let Latency = 9;
  let ResourceCycles = [9];
}
def : InstRW<[AtomWrite01_9], (instrs BT16mr, BT32mr, BT64mr,
                                      POPA16, POPA32,
                                      PUSHF16, PUSHF32, PUSHF64,
                                      SHLD64mrCL, SHRD64mrCL,
                                      SHLD64mri8, SHRD64mri8,
                                      SHLD64rri8, SHRD64rri8,
                                      CMPXCHG8rr)>;
def : InstRW<[AtomWrite01_9], (instregex "CMOV(B|BE|E|P|NB|NBE|NE|NP)_F",
                                         "(U)?COM_FI", "TST_F",
                                         "(U)?COMIS(D|S)rr",
                                         "CVT(T)?SS2SI64rr(_Int)?")>;

def AtomWrite01_10 : SchedWriteRes<[AtomPort01]> {
  let Latency = 10;
  let ResourceCycles = [10];
}
def : InstRW<[AtomWrite01_10], (instrs FLDL2E, FLDL2T, FLDLG2, FLDLN2, FLDPI)>;
def : InstRW<[AtomWrite01_10], (instregex "(U)?COMIS(D|S)rm",
                                          "CVT(T)?SS2SI64rm(_Int)?")>;

def AtomWrite01_11 : SchedWriteRes<[AtomPort01]> {
  let Latency = 11;
  let ResourceCycles = [11];
}
def : InstRW<[AtomWrite01_11], (instrs BOUNDS16rm, BOUNDS32rm)>;
def : InstRW<[AtomWrite01_11], (instregex "BT(C|R|S)(16|32|64)mr")>;

def AtomWrite01_13 : SchedWriteRes<[AtomPort01]> {
  let Latency = 13;
  let ResourceCycles = [13];
}
def : InstRW<[AtomWrite01_13], (instrs AAA, AAS)>;

def AtomWrite01_14 : SchedWriteRes<[AtomPort01]> {
  let Latency = 14;
  let ResourceCycles = [14];
}
def : InstRW<[AtomWrite01_14], (instrs CMPXCHG16rm, CMPXCHG32rm, CMPXCHG64rm)>;

def AtomWrite01_15 : SchedWriteRes<[AtomPort01]> {
  let Latency = 15;
  let ResourceCycles = [15];
}
def : InstRW<[AtomWrite01_15], (instrs CMPXCHG16rr, CMPXCHG32rr, CMPXCHG64rr)>;

def AtomWrite01_17 : SchedWriteRes<[AtomPort01]> {
  let Latency = 17;
  let ResourceCycles = [17];
}
def : InstRW<[AtomWrite01_17], (instrs LOOPNE, PAUSE)>;

def AtomWrite01_18 : SchedWriteRes<[AtomPort01]> {
  let Latency = 18;
  let ResourceCycles = [18];
}
def : InstRW<[AtomWrite01_18], (instrs CMPXCHG8B, DAA, LOOP)>;

def AtomWrite01_20 : SchedWriteRes<[AtomPort01]> {
  let Latency = 20;
  let ResourceCycles = [20];
}
def : InstRW<[AtomWrite01_20], (instrs DAS)>;

def AtomWrite01_21 : SchedWriteRes<[AtomPort01]> {
  let Latency = 21;
  let ResourceCycles = [21];
}
def : InstRW<[AtomWrite01_21], (instrs AAM8i8, STD)>;

def AtomWrite01_22 : SchedWriteRes<[AtomPort01]> {
  let Latency = 22;
  let ResourceCycles = [22];
}
def : InstRW<[AtomWrite01_22], (instrs CMPXCHG16B)>;

def AtomWrite01_23 : SchedWriteRes<[AtomPort01]> {
  let Latency = 23;
  let ResourceCycles = [23];
}
def : InstRW<[AtomWrite01_23], (instrs ARPL16mr, ARPL16rr)>;

def AtomWrite01_25 : SchedWriteRes<[AtomPort01]> {
  let Latency = 25;
  let ResourceCycles = [25];
}
def : InstRW<[AtomWrite01_25], (instrs FNCLEX, FXTRACT)>;

def AtomWrite01_26 : SchedWriteRes<[AtomPort01]> {
  let Latency = 26;
  let ResourceCycles = [26];
}
def : InstRW<[AtomWrite01_26], (instrs POPF32, POPF64)>;

def AtomWrite01_29 : SchedWriteRes<[AtomPort01]> {
  let Latency = 29;
  let ResourceCycles = [29];
}
def : InstRW<[AtomWrite01_29], (instregex "POP(DS|ES|FS|GS)(16|32|64)")>;

def AtomWrite01_30 : SchedWriteRes<[AtomPort01]> {
  let Latency = 30;
  let ResourceCycles = [30];
}
def : InstRW<[AtomWrite01_30], (instrs RDTSC, RDTSCP)>;

def AtomWrite01_32 : SchedWriteRes<[AtomPort01]> {
  let Latency = 32;
  let ResourceCycles = [32];
}
def : InstRW<[AtomWrite01_32], (instrs ENTER, POPF16)>;

def AtomWrite01_45 : SchedWriteRes<[AtomPort01]> {
  let Latency = 45;
  let ResourceCycles = [45];
}
def : InstRW<[AtomWrite01_45], (instrs MONITORrrr)>;

def AtomWrite01_46 : SchedWriteRes<[AtomPort01]> {
  let Latency = 46;
  let ResourceCycles = [46];
}
def : InstRW<[AtomWrite01_46], (instrs FRNDINT, MWAITrr, RDPMC)>;

def AtomWrite01_48 : SchedWriteRes<[AtomPort01]> {
  let Latency = 48;
  let ResourceCycles = [48];
}
def : InstRW<[AtomWrite01_48], (instrs POPSS16, POPSS32)>;

def AtomWrite01_55 : SchedWriteRes<[AtomPort01]> {
  let Latency = 55;
  let ResourceCycles = [55];
}
def : InstRW<[AtomWrite01_55], (instrs FPREM)>;

def AtomWrite01_59 : SchedWriteRes<[AtomPort01]> {
  let Latency = 59;
  let ResourceCycles = [59];
}
def : InstRW<[AtomWrite01_59], (instrs INSB, INSL, INSW)>;

def AtomWrite01_63 : SchedWriteRes<[AtomPort01]> {
  let Latency = 63;
  let ResourceCycles = [63];
}
def : InstRW<[AtomWrite01_63], (instrs FNINIT)>;

def AtomWrite01_68 : SchedWriteRes<[AtomPort01]> {
  let Latency = 68;
  let ResourceCycles = [68];
}
def : InstRW<[AtomWrite01_68], (instrs OUT8rr, OUT16rr, OUT32rr)>;

def AtomWrite01_71 : SchedWriteRes<[AtomPort01]> {
  let Latency = 71;
  let ResourceCycles = [71];
}
def : InstRW<[AtomWrite01_71], (instrs FPREM1,
                                       INVLPG, INVLPGA32, INVLPGA64)>;

def AtomWrite01_72 : SchedWriteRes<[AtomPort01]> {
  let Latency = 72;
  let ResourceCycles = [72];
}
def : InstRW<[AtomWrite01_72], (instrs OUT8ir, OUT16ir, OUT32ir)>;

def AtomWrite01_74 : SchedWriteRes<[AtomPort01]> {
  let Latency = 74;
  let ResourceCycles = [74];
}
def : InstRW<[AtomWrite01_74], (instrs OUTSB, OUTSL, OUTSW)>;

def AtomWrite01_77 : SchedWriteRes<[AtomPort01]> {
  let Latency = 77;
  let ResourceCycles = [77];
}
def : InstRW<[AtomWrite01_77], (instrs FSCALE)>;

def AtomWrite01_78 : SchedWriteRes<[AtomPort01]> {
  let Latency = 78;
  let ResourceCycles = [78];
}
def : InstRW<[AtomWrite01_78], (instrs RDMSR)>;

def AtomWrite01_79 : SchedWriteRes<[AtomPort01]> {
  let Latency = 79;
  let ResourceCycles = [79];
}
def : InstRW<[AtomWrite01_79], (instregex "RET(L|Q|W)?$",
                                          "LRETI?(L|Q|W)")>;

def AtomWrite01_92 : SchedWriteRes<[AtomPort01]> {
  let Latency = 92;
  let ResourceCycles = [92];
}
def : InstRW<[AtomWrite01_92], (instrs IN8ri, IN16ri, IN32ri)>;

def AtomWrite01_94 : SchedWriteRes<[AtomPort01]> {
  let Latency = 94;
  let ResourceCycles = [94];
}
def : InstRW<[AtomWrite01_94], (instrs IN8rr, IN16rr, IN32rr)>;

def AtomWrite01_99 : SchedWriteRes<[AtomPort01]> {
  let Latency = 99;
  let ResourceCycles = [99];
}
def : InstRW<[AtomWrite01_99], (instrs F2XM1)>;

def AtomWrite01_121 : SchedWriteRes<[AtomPort01]> {
  let Latency = 121;
  let ResourceCycles = [121];
}
def : InstRW<[AtomWrite01_121], (instrs CPUID)>;

def AtomWrite01_127 : SchedWriteRes<[AtomPort01]> {
  let Latency = 127;
  let ResourceCycles = [127];
}
def : InstRW<[AtomWrite01_127], (instrs INT)>;

def AtomWrite01_130 : SchedWriteRes<[AtomPort01]> {
  let Latency = 130;
  let ResourceCycles = [130];
}
def : InstRW<[AtomWrite01_130], (instrs INT3)>;

def AtomWrite01_140 : SchedWriteRes<[AtomPort01]> {
  let Latency = 140;
  let ResourceCycles = [140];
}
def : InstRW<[AtomWrite01_140], (instrs FXSAVE, FXSAVE64)>;

def AtomWrite01_141 : SchedWriteRes<[AtomPort01]> {
  let Latency = 141;
  let ResourceCycles = [141];
}
def : InstRW<[AtomWrite01_141], (instrs FXRSTOR, FXRSTOR64)>;

def AtomWrite01_146 : SchedWriteRes<[AtomPort01]> {
  let Latency = 146;
  let ResourceCycles = [146];
}
def : InstRW<[AtomWrite01_146], (instrs FYL2X)>;

def AtomWrite01_147 : SchedWriteRes<[AtomPort01]> {
  let Latency = 147;
  let ResourceCycles = [147];
}
def : InstRW<[AtomWrite01_147], (instrs FYL2XP1)>;

def AtomWrite01_168 : SchedWriteRes<[AtomPort01]> {
  let Latency = 168;
  let ResourceCycles = [168];
}
def : InstRW<[AtomWrite01_168], (instrs FPTAN)>;

def AtomWrite01_174 : SchedWriteRes<[AtomPort01]> {
  let Latency = 174;
  let ResourceCycles = [174];
}
def : InstRW<[AtomWrite01_174], (instrs FSINCOS)>;
def : InstRW<[AtomWrite01_174], (instregex "(COS|SIN)_F")>;

def AtomWrite01_183 : SchedWriteRes<[AtomPort01]> {
  let Latency = 183;
  let ResourceCycles = [183];
}
def : InstRW<[AtomWrite01_183], (instrs FPATAN)>;

def AtomWrite01_202 : SchedWriteRes<[AtomPort01]> {
  let Latency = 202;
  let ResourceCycles = [202];
}
def : InstRW<[AtomWrite01_202], (instrs WRMSR)>;

} // SchedModel
