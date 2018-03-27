//=- X86ScheduleSLM.td - X86 Silvermont Scheduling -----------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Intel Silvermont to support
// instruction scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def SLMModel : SchedMachineModel {
  // All x86 instructions are modeled as a single micro-op, and SLM can decode 2
  // instructions per cycle.
  let IssueWidth = 2;
  let MicroOpBufferSize = 32; // Based on the reorder buffer.
  let LoadLatency = 3;
  let MispredictPenalty = 10;
  let PostRAScheduler = 1;

  // For small loops, expand by a small factor to hide the backedge cost.
  let LoopMicroOpBufferSize = 10;

  // FIXME: SSE4 is unimplemented. This flag is set to allow
  // the scheduler to assign a default model to unrecognized opcodes.
  let CompleteModel = 0;
}

let SchedModel = SLMModel in {

// Silvermont has 5 reservation stations for micro-ops
def SLM_IEC_RSV0 : ProcResource<1>;
def SLM_IEC_RSV1 : ProcResource<1>;
def SLM_FPC_RSV0 : ProcResource<1> { let BufferSize = 1; }
def SLM_FPC_RSV1 : ProcResource<1> { let BufferSize = 1; }
def SLM_MEC_RSV  : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def SLM_IEC_RSV01  : ProcResGroup<[SLM_IEC_RSV0, SLM_IEC_RSV1]>;
def SLM_FPC_RSV01  : ProcResGroup<[SLM_FPC_RSV0, SLM_FPC_RSV1]>;

def SLMDivider      : ProcResource<1>;
def SLMFPMultiplier : ProcResource<1>;
def SLMFPDivider    : ProcResource<1>;

// Loads are 3 cycles, so ReadAfterLd registers needn't be available until 3
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 3>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass SLMWriteResPair<X86FoldableSchedWrite SchedRW,
                           list<ProcResourceKind> ExePorts,
                           int Lat, list<int> Res = [1], int UOps = 1,
                           int LoadLat = 3> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on MEC_RSV and adds LoadLat cycles to
  // the latency (default = 3).
  def : WriteRes<SchedRW.Folded, !listconcat([SLM_MEC_RSV], ExePorts)> {
    let Latency = !add(Lat, LoadLat);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = UOps;
  }
}

// A folded store needs a cycle on MEC_RSV for the store data, but it does not
// need an extra port cycle to recompute the address.
def : WriteRes<WriteRMW, [SLM_MEC_RSV]>;

def : WriteRes<WriteStore, [SLM_IEC_RSV01, SLM_MEC_RSV]>;
def : WriteRes<WriteLoad,  [SLM_MEC_RSV]> { let Latency = 3; }
def : WriteRes<WriteMove,  [SLM_IEC_RSV01]>;
def : WriteRes<WriteZero,  []>;

// Treat misc copies as a move.
def : InstRW<[WriteMove], (instrs COPY)>;

defm : SLMWriteResPair<WriteALU,   [SLM_IEC_RSV01], 1>;
defm : SLMWriteResPair<WriteIMul,  [SLM_IEC_RSV1],  3>;
defm : SLMWriteResPair<WriteShift, [SLM_IEC_RSV0],  1>;
defm : SLMWriteResPair<WriteJump,  [SLM_IEC_RSV1],  1>;
defm : SLMWriteResPair<WriteCRC32, [SLM_IEC_RSV1],  3>;

// This is for simple LEAs with one or two input operands.
// The complex ones can only execute on port 1, and they require two cycles on
// the port to read all inputs. We don't model that.
def : WriteRes<WriteLEA, [SLM_IEC_RSV1]>;

// Bit counts.
defm : SLMWriteResPair<WriteBitScan, [SLM_IEC_RSV01], 10, [20], 10>;
defm : SLMWriteResPair<WriteLZCNT,   [SLM_IEC_RSV0], 3>;
defm : SLMWriteResPair<WriteTZCNT,   [SLM_IEC_RSV0], 3>;
defm : SLMWriteResPair<WritePOPCNT,  [SLM_IEC_RSV0], 3>;

// This is quite rough, latency depends on the dividend.
defm : SLMWriteResPair<WriteIDiv, [SLM_IEC_RSV01, SLMDivider], 25, [1,25], 1, 4>;

// Scalar and vector floating point.
def  : WriteRes<WriteFStore,       [SLM_FPC_RSV01, SLM_MEC_RSV]>;
def  : WriteRes<WriteFLoad,        [SLM_MEC_RSV]> { let Latency = 3; }
def  : WriteRes<WriteFMove,        [SLM_FPC_RSV01]>;

defm : SLMWriteResPair<WriteFAdd,   [SLM_FPC_RSV1], 3>;
defm : SLMWriteResPair<WriteFMul,   [SLM_FPC_RSV0, SLMFPMultiplier], 5, [1,2]>;
defm : SLMWriteResPair<WriteFDiv,   [SLM_FPC_RSV0, SLMFPDivider], 34, [1,34]>;
defm : SLMWriteResPair<WriteFRcp,   [SLM_FPC_RSV0], 5>;
defm : SLMWriteResPair<WriteFRsqrt, [SLM_FPC_RSV0], 5>;
defm : SLMWriteResPair<WriteFSqrt,  [SLM_FPC_RSV0], 15>;
defm : SLMWriteResPair<WriteCvtF2I, [SLM_FPC_RSV01], 4>;
defm : SLMWriteResPair<WriteCvtI2F, [SLM_FPC_RSV01], 4>;
defm : SLMWriteResPair<WriteCvtF2F, [SLM_FPC_RSV01], 4>;
defm : SLMWriteResPair<WriteFShuffle, [SLM_FPC_RSV0],  1>;
defm : SLMWriteResPair<WriteFBlend,  [SLM_FPC_RSV0],  1>;

// Vector integer operations.
def  : WriteRes<WriteVecStore,       [SLM_FPC_RSV01, SLM_MEC_RSV]>;
def  : WriteRes<WriteVecLoad,        [SLM_MEC_RSV]> { let Latency = 3; }
def  : WriteRes<WriteVecMove,        [SLM_FPC_RSV01]>;

defm : SLMWriteResPair<WriteVecShift, [SLM_FPC_RSV0],  1>;
defm : SLMWriteResPair<WriteVecLogic, [SLM_FPC_RSV01], 1>;
defm : SLMWriteResPair<WriteVecALU,   [SLM_FPC_RSV01],  1>;
defm : SLMWriteResPair<WriteVecIMul,  [SLM_FPC_RSV0],   4>;
defm : SLMWriteResPair<WriteShuffle,  [SLM_FPC_RSV0],  1>;
defm : SLMWriteResPair<WriteBlend,  [SLM_FPC_RSV0],  1>;
defm : SLMWriteResPair<WriteMPSAD,  [SLM_FPC_RSV0],  7>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : SLMWriteResPair<WriteFHAdd,   [SLM_FPC_RSV01], 3, [2]>;
defm : SLMWriteResPair<WritePHAdd,   [SLM_FPC_RSV01], 1>;

// String instructions.
// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [SLM_FPC_RSV0]> {
  let Latency = 13;
  let ResourceCycles = [13];
}
def : WriteRes<WritePCmpIStrMLd, [SLM_FPC_RSV0, SLM_MEC_RSV]> {
  let Latency = 13;
  let ResourceCycles = [13, 1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [SLM_FPC_RSV0]> {
  let Latency = 17;
  let ResourceCycles = [17];
}
def : WriteRes<WritePCmpEStrMLd, [SLM_FPC_RSV0, SLM_MEC_RSV]> {
  let Latency = 17;
  let ResourceCycles = [17, 1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [SLM_FPC_RSV0]> {
  let Latency = 17;
  let ResourceCycles = [17];
}
def : WriteRes<WritePCmpIStrILd, [SLM_FPC_RSV0, SLM_MEC_RSV]> {
  let Latency = 17;
  let ResourceCycles = [17, 1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [SLM_FPC_RSV0]> {
  let Latency = 21;
  let ResourceCycles = [21];
}
def : WriteRes<WritePCmpEStrILd, [SLM_FPC_RSV0, SLM_MEC_RSV]> {
  let Latency = 21;
  let ResourceCycles = [21, 1];
}

// AES Instructions.
def : WriteRes<WriteAESDecEnc, [SLM_FPC_RSV0]> {
  let Latency = 8;
  let ResourceCycles = [5];
}
def : WriteRes<WriteAESDecEncLd, [SLM_FPC_RSV0, SLM_MEC_RSV]> {
  let Latency = 8;
  let ResourceCycles = [5, 1];
}

def : WriteRes<WriteAESIMC, [SLM_FPC_RSV0]> {
  let Latency = 8;
  let ResourceCycles = [5];
}
def : WriteRes<WriteAESIMCLd, [SLM_FPC_RSV0, SLM_MEC_RSV]> {
  let Latency = 8;
  let ResourceCycles = [5, 1];
}

def : WriteRes<WriteAESKeyGen, [SLM_FPC_RSV0]> {
  let Latency = 8;
  let ResourceCycles = [5];
}
def : WriteRes<WriteAESKeyGenLd, [SLM_FPC_RSV0, SLM_MEC_RSV]> {
  let Latency = 8;
  let ResourceCycles = [5, 1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [SLM_FPC_RSV0]> {
  let Latency = 10;
  let ResourceCycles = [10];
}
def : WriteRes<WriteCLMulLd, [SLM_FPC_RSV0, SLM_MEC_RSV]> {
  let Latency = 10;
  let ResourceCycles = [10, 1];
}


def : WriteRes<WriteSystem,     [SLM_FPC_RSV0]> { let Latency = 100; }
def : WriteRes<WriteMicrocoded, [SLM_FPC_RSV0]> { let Latency = 100; }
def : WriteRes<WriteFence, [SLM_MEC_RSV]>;
def : WriteRes<WriteNop, []>;

// AVX/FMA is not supported on that architecture, but we should define the basic
// scheduling resources anyway.
def  : WriteRes<WriteIMulH, [SLM_FPC_RSV0]>;
defm : SLMWriteResPair<WriteVarBlend, [SLM_FPC_RSV0], 1>;
defm : SLMWriteResPair<WriteFVarBlend, [SLM_FPC_RSV0], 1>;
defm : SLMWriteResPair<WriteFShuffle256, [SLM_FPC_RSV0],  1>;
defm : SLMWriteResPair<WriteShuffle256, [SLM_FPC_RSV0],  1>;
defm : SLMWriteResPair<WriteVarVecShift, [SLM_FPC_RSV0],  1>;
defm : SLMWriteResPair<WriteFMA, [SLM_FPC_RSV0],  1>;
} // SchedModel
