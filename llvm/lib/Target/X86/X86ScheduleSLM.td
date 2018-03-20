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
def IEC_RSV0 : ProcResource<1>;
def IEC_RSV1 : ProcResource<1>;
def FPC_RSV0 : ProcResource<1> { let BufferSize = 1; }
def FPC_RSV1 : ProcResource<1> { let BufferSize = 1; }
def MEC_RSV  : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def IEC_RSV01  : ProcResGroup<[IEC_RSV0, IEC_RSV1]>;
def FPC_RSV01  : ProcResGroup<[FPC_RSV0, FPC_RSV1]>;

def SMDivider      : ProcResource<1>;
def SMFPMultiplier : ProcResource<1>;
def SMFPDivider    : ProcResource<1>;

// Loads are 3 cycles, so ReadAfterLd registers needn't be available until 3
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 3>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass SMWriteResPair<X86FoldableSchedWrite SchedRW,
                          list<ProcResourceKind> ExePorts,
                          int Lat, list<int> Res = [1], int UOps = 1> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on MEC_RSV and adds 3 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, !listconcat([MEC_RSV], ExePorts)> {
    let Latency = !add(Lat, 3);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = UOps;
  }
}

// A folded store needs a cycle on MEC_RSV for the store data, but it does not
// need an extra port cycle to recompute the address.
def : WriteRes<WriteRMW, [MEC_RSV]>;

def : WriteRes<WriteStore, [IEC_RSV01, MEC_RSV]>;
def : WriteRes<WriteLoad,  [MEC_RSV]> { let Latency = 3; }
def : WriteRes<WriteMove,  [IEC_RSV01]>;
def : WriteRes<WriteZero,  []>;

// Treat misc copies as a move.
def : InstRW<[WriteMove], (instrs COPY)>;

defm : SMWriteResPair<WriteALU,   [IEC_RSV01], 1>;
defm : SMWriteResPair<WriteIMul,  [IEC_RSV1],  3>;
defm : SMWriteResPair<WriteShift, [IEC_RSV0],  1>;
defm : SMWriteResPair<WriteJump,  [IEC_RSV1],   1>;

// This is for simple LEAs with one or two input operands.
// The complex ones can only execute on port 1, and they require two cycles on
// the port to read all inputs. We don't model that.
def : WriteRes<WriteLEA, [IEC_RSV1]>;

// This is quite rough, latency depends on the dividend.
def : WriteRes<WriteIDiv, [IEC_RSV01, SMDivider]> {
  let Latency = 25;
  let ResourceCycles = [1, 25];
}
def : WriteRes<WriteIDivLd, [MEC_RSV, IEC_RSV01, SMDivider]> {
  let Latency = 29;
  let ResourceCycles = [1, 1, 25];
}

// Scalar and vector floating point.
def  : WriteRes<WriteFStore,       [FPC_RSV01, MEC_RSV]>;
def  : WriteRes<WriteFLoad,        [MEC_RSV]> { let Latency = 3; }
def  : WriteRes<WriteFMove,        [FPC_RSV01]>;

defm : SMWriteResPair<WriteFAdd,   [FPC_RSV1], 3>;
defm : SMWriteResPair<WriteFMul,   [FPC_RSV0, SMFPMultiplier], 5, [1,2]>;
defm : SMWriteResPair<WriteFDiv,   [FPC_RSV0, SMFPDivider], 34, [1,34]>;
defm : SMWriteResPair<WriteFRcp,   [FPC_RSV0], 5>;
defm : SMWriteResPair<WriteFRsqrt, [FPC_RSV0], 5>;
defm : SMWriteResPair<WriteFSqrt,  [FPC_RSV0], 15>;
defm : SMWriteResPair<WriteCvtF2I, [FPC_RSV01], 4>;
defm : SMWriteResPair<WriteCvtI2F, [FPC_RSV01], 4>;
defm : SMWriteResPair<WriteCvtF2F, [FPC_RSV01], 4>;
defm : SMWriteResPair<WriteFShuffle, [FPC_RSV0],  1>;
defm : SMWriteResPair<WriteFBlend,  [FPC_RSV0],  1>;

// Vector integer operations.
def  : WriteRes<WriteVecStore,       [FPC_RSV01, MEC_RSV]>;
def  : WriteRes<WriteVecLoad,        [MEC_RSV]> { let Latency = 3; }
def  : WriteRes<WriteVecMove,        [FPC_RSV01]>;

defm : SMWriteResPair<WriteVecShift, [FPC_RSV0],  1>;
defm : SMWriteResPair<WriteVecLogic, [FPC_RSV01], 1>;
defm : SMWriteResPair<WriteVecALU,   [FPC_RSV01],  1>;
defm : SMWriteResPair<WriteVecIMul,  [FPC_RSV0],   4>;
defm : SMWriteResPair<WriteShuffle,  [FPC_RSV0],  1>;
defm : SMWriteResPair<WriteBlend,  [FPC_RSV0],  1>;
defm : SMWriteResPair<WriteMPSAD,  [FPC_RSV0],  7>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : SMWriteResPair<WriteFHAdd,   [FPC_RSV01], 3, [2]>;
defm : SMWriteResPair<WritePHAdd,   [FPC_RSV01], 1>;

// String instructions.
// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [FPC_RSV0]> {
  let Latency = 13;
  let ResourceCycles = [13];
}
def : WriteRes<WritePCmpIStrMLd, [FPC_RSV0, MEC_RSV]> {
  let Latency = 13;
  let ResourceCycles = [13, 1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [FPC_RSV0]> {
  let Latency = 17;
  let ResourceCycles = [17];
}
def : WriteRes<WritePCmpEStrMLd, [FPC_RSV0, MEC_RSV]> {
  let Latency = 17;
  let ResourceCycles = [17, 1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [FPC_RSV0]> {
  let Latency = 17;
  let ResourceCycles = [17];
}
def : WriteRes<WritePCmpIStrILd, [FPC_RSV0, MEC_RSV]> {
  let Latency = 17;
  let ResourceCycles = [17, 1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [FPC_RSV0]> {
  let Latency = 21;
  let ResourceCycles = [21];
}
def : WriteRes<WritePCmpEStrILd, [FPC_RSV0, MEC_RSV]> {
  let Latency = 21;
  let ResourceCycles = [21, 1];
}

// AES Instructions.
def : WriteRes<WriteAESDecEnc, [FPC_RSV0]> {
  let Latency = 8;
  let ResourceCycles = [5];
}
def : WriteRes<WriteAESDecEncLd, [FPC_RSV0, MEC_RSV]> {
  let Latency = 8;
  let ResourceCycles = [5, 1];
}

def : WriteRes<WriteAESIMC, [FPC_RSV0]> {
  let Latency = 8;
  let ResourceCycles = [5];
}
def : WriteRes<WriteAESIMCLd, [FPC_RSV0, MEC_RSV]> {
  let Latency = 8;
  let ResourceCycles = [5, 1];
}

def : WriteRes<WriteAESKeyGen, [FPC_RSV0]> {
  let Latency = 8;
  let ResourceCycles = [5];
}
def : WriteRes<WriteAESKeyGenLd, [FPC_RSV0, MEC_RSV]> {
  let Latency = 8;
  let ResourceCycles = [5, 1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [FPC_RSV0]> {
  let Latency = 10;
  let ResourceCycles = [10];
}
def : WriteRes<WriteCLMulLd, [FPC_RSV0, MEC_RSV]> {
  let Latency = 10;
  let ResourceCycles = [10, 1];
}


def : WriteRes<WriteSystem,     [FPC_RSV0]> { let Latency = 100; }
def : WriteRes<WriteMicrocoded, [FPC_RSV0]> { let Latency = 100; }
def : WriteRes<WriteFence, [MEC_RSV]>;
def : WriteRes<WriteNop, []>;

// AVX/FMA is not supported on that architecture, but we should define the basic
// scheduling resources anyway.
def  : WriteRes<WriteIMulH, [FPC_RSV0]>;
defm : SMWriteResPair<WriteVarBlend, [FPC_RSV0], 1>;
defm : SMWriteResPair<WriteFVarBlend, [FPC_RSV0], 1>;
defm : SMWriteResPair<WriteFShuffle256, [FPC_RSV0],  1>;
defm : SMWriteResPair<WriteShuffle256, [FPC_RSV0],  1>;
defm : SMWriteResPair<WriteVarVecShift, [FPC_RSV0],  1>;
defm : SMWriteResPair<WriteFMA, [FPC_RSV0],  1>;
} // SchedModel
