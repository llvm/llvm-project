//=- X86ScheduleBtVer2.td - X86 BtVer2 (Jaguar) Scheduling ---*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for AMD btver2 (Jaguar) to support
// instruction scheduling and other instruction cost heuristics. Based off AMD Software
// Optimization Guide for AMD Family 16h Processors & Instruction Latency appendix.
//
//===----------------------------------------------------------------------===//

def BtVer2Model : SchedMachineModel {
  // All x86 instructions are modeled as a single micro-op, and btver2 can
  // decode 2 instructions per cycle.
  let IssueWidth = 2;
  let MicroOpBufferSize = 64; // Retire Control Unit
  let LoadLatency = 5; // FPU latency (worse case cf Integer 3 cycle latency)
  let HighLatency = 25;
  let MispredictPenalty = 14; // Minimum branch misdirection penalty
  let PostRAScheduler = 1;

  // FIXME: SSE4/AVX is unimplemented. This flag is set to allow
  // the scheduler to assign a default model to unrecognized opcodes.
  let CompleteModel = 0;
}

let SchedModel = BtVer2Model in {

// Jaguar can issue up to 6 micro-ops in one cycle
def JALU0 : ProcResource<1>; // Integer Pipe0: integer ALU0 (also handle FP->INT jam)
def JALU1 : ProcResource<1>; // Integer Pipe1: integer ALU1/MUL/DIV
def JLAGU : ProcResource<1>; // Integer Pipe2: LAGU
def JSAGU : ProcResource<1>; // Integer Pipe3: SAGU (also handles 3-operand LEA)
def JFPU0 : ProcResource<1>; // Vector/FPU Pipe0: VALU0/VIMUL/FPA
def JFPU1 : ProcResource<1>; // Vector/FPU Pipe1: VALU1/STC/FPM

// Integer Pipe Scheduler
def JALU01 : ProcResGroup<[JALU0, JALU1]> {
  let BufferSize=20;
}

// AGU Pipe Scheduler
def JLSAGU : ProcResGroup<[JLAGU, JSAGU]> {
  let BufferSize=12;
}

// Fpu Pipe Scheduler
def JFPU01 : ProcResGroup<[JFPU0, JFPU1]> {
  let BufferSize=18;
}

// Functional units
def JDiv    : ProcResource<1>; // integer division
def JMul    : ProcResource<1>; // integer multiplication
def JVALU0  : ProcResource<1>; // vector integer
def JVALU1  : ProcResource<1>; // vector integer
def JVIMUL  : ProcResource<1>; // vector integer multiplication
def JSTC    : ProcResource<1>; // vector store/convert
def JFPM    : ProcResource<1>; // FP multiplication
def JFPA    : ProcResource<1>; // FP addition

// Functional unit groups
def JFPX  : ProcResGroup<[JFPA, JFPM]>;
def JVALU : ProcResGroup<[JVALU0, JVALU1]>;

// Integer loads are 3 cycles, so ReadAfterLd registers needn't be available until 3
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 3>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when dispatched by the schedulers.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass JWriteResIntPair<X86FoldableSchedWrite SchedRW,
                            list<ProcResourceKind> ExePorts,
                            int Lat, list<int> Res = [1], int UOps = 1> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on JLAGU and adds 3 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, !listconcat([JLAGU], ExePorts)> {
    let Latency = !add(Lat, 3);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = UOps;
  }
}

multiclass JWriteResFpuPair<X86FoldableSchedWrite SchedRW,
                            list<ProcResourceKind> ExePorts,
                            int Lat, list<int> Res = [1], int UOps = 1> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on JLAGU and adds 5 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, !listconcat([JLAGU], ExePorts)> {
    let Latency = !add(Lat, 5);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = UOps;
  }
}

// A folded store needs a cycle on the SAGU for the store data.
def : WriteRes<WriteRMW, [JSAGU]>;

////////////////////////////////////////////////////////////////////////////////
// Arithmetic.
////////////////////////////////////////////////////////////////////////////////

defm : JWriteResIntPair<WriteALU,   [JALU01], 1>;
defm : JWriteResIntPair<WriteIMul,  [JALU1, JMul], 3, [1, 1], 2>; // i8/i16/i32 multiplication
defm : JWriteResIntPair<WriteIDiv,  [JALU1, JDiv], 41, [1, 41], 2>; // Worst case (i64 division)
defm : JWriteResIntPair<WriteCRC32, [JALU01], 3, [4], 3>;

def  : WriteRes<WriteIMulH, [JALU1]> {
  let Latency = 6;
  let ResourceCycles = [4];
}

// This is for simple LEAs with one or two input operands.
// FIXME: SAGU 3-operand LEA
def : WriteRes<WriteLEA, [JALU01]>;

// Bit counts.
defm : JWriteResIntPair<WriteBitScan, [JALU01], 5, [4], 8>;
defm : JWriteResIntPair<WritePOPCNT,  [JALU01], 1>;
defm : JWriteResIntPair<WriteLZCNT,   [JALU01], 1>;
defm : JWriteResIntPair<WriteTZCNT,   [JALU01], 2, [2]>;

def JWriteIMul64 : SchedWriteRes<[JALU1, JMul]> {
  let Latency = 6;
  let ResourceCycles = [1, 4];
  let NumMicroOps = 2;
}
def JWriteIMul64Ld : SchedWriteRes<[JLAGU, JALU1, JMul]> {
  let Latency = 9;
  let ResourceCycles = [1, 1, 4];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteIMul64], (instrs MUL64r, IMUL64r)>;
def : InstRW<[JWriteIMul64Ld], (instrs MUL64m, IMUL64m)>;

def JWriteIDiv8 : SchedWriteRes<[JALU1, JDiv]> {
  let Latency = 12;
  let ResourceCycles = [1, 12];
}
def JWriteIDiv8Ld : SchedWriteRes<[JLAGU, JALU1, JDiv]> {
  let Latency = 15;
  let ResourceCycles = [1, 1, 12];
}
def : InstRW<[JWriteIDiv8], (instrs DIV8r, IDIV8r)>;
def : InstRW<[JWriteIDiv8Ld], (instrs DIV8m, IDIV8m)>;

def JWriteIDiv16 : SchedWriteRes<[JALU1, JDiv]> {
  let Latency = 17;
  let ResourceCycles = [1, 17];
  let NumMicroOps = 2;
}
def JWriteIDiv16Ld : SchedWriteRes<[JLAGU, JALU1, JDiv]> {
  let Latency = 20;
  let ResourceCycles = [1, 1, 17];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteIDiv16], (instrs DIV16r, IDIV16r)>;
def : InstRW<[JWriteIDiv16Ld], (instrs DIV16m, IDIV16m)>;

def JWriteIDiv32 : SchedWriteRes<[JALU1, JDiv]> {
  let Latency = 25;
  let ResourceCycles = [1, 25];
  let NumMicroOps = 2;
}
def JWriteIDiv32Ld : SchedWriteRes<[JLAGU, JALU1, JDiv]> {
  let Latency = 28;
  let ResourceCycles = [1, 1, 25];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteIDiv32], (instrs DIV32r, IDIV32r)>;
def : InstRW<[JWriteIDiv32Ld], (instrs DIV32m, IDIV32m)>;

////////////////////////////////////////////////////////////////////////////////
// Integer shifts and rotates.
////////////////////////////////////////////////////////////////////////////////

defm : JWriteResIntPair<WriteShift, [JALU01], 1>;

def JWriteSHLDrri : SchedWriteRes<[JALU01]> {
  let Latency = 3;
  let ResourceCycles = [6];
  let NumMicroOps = 6;
}
def: InstRW<[JWriteSHLDrri], (instrs SHLD16rri8, SHLD32rri8, SHLD64rri8,
                                     SHRD16rri8, SHRD32rri8, SHRD64rri8)>;

def JWriteSHLDrrCL : SchedWriteRes<[JALU01]> {
  let Latency = 4;
  let ResourceCycles = [8];
  let NumMicroOps = 7;
}
def: InstRW<[JWriteSHLDrrCL], (instrs SHLD16rrCL, SHLD32rrCL, SHLD64rrCL,
                                      SHRD16rrCL, SHRD32rrCL, SHRD64rrCL)>;

def JWriteSHLDm : SchedWriteRes<[JLAGU, JALU01]> {
  let Latency = 9;
  let ResourceCycles = [1, 22];
  let NumMicroOps = 8;
}
def: InstRW<[JWriteSHLDm],(instrs SHLD16mri8, SHLD32mri8, SHLD64mri8,
                                  SHLD16mrCL, SHLD32mrCL, SHLD64mrCL,
                                  SHRD16mri8, SHRD32mri8, SHRD64mri8,
                                  SHRD16mrCL, SHRD32mrCL, SHRD64mrCL)>;

////////////////////////////////////////////////////////////////////////////////
// Loads, stores, and moves, not folded with other operations.
////////////////////////////////////////////////////////////////////////////////

def : WriteRes<WriteLoad,  [JLAGU]> { let Latency = 5; }
def : WriteRes<WriteStore, [JSAGU]>;
def : WriteRes<WriteMove,  [JALU01]>;

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

defm : JWriteResIntPair<WriteJump,  [JALU01], 1>;

////////////////////////////////////////////////////////////////////////////////
// Special case scheduling classes.
////////////////////////////////////////////////////////////////////////////////

def : WriteRes<WriteSystem,     [JALU01]> { let Latency = 100; }
def : WriteRes<WriteMicrocoded, [JALU01]> { let Latency = 100; }
def : WriteRes<WriteFence,  [JSAGU]>;
// Nops don't have dependencies, so there's no actual latency, but we set this
// to '1' to tell the scheduler that the nop uses an ALU slot for a cycle.
def : WriteRes<WriteNop, [JALU01]> { let Latency = 1; }

////////////////////////////////////////////////////////////////////////////////
// Floating point. This covers both scalar and vector operations.
// FIXME: should we bother splitting JFPU pipe + unit stages for fast instructions?
// FIXME: Double precision latencies
// FIXME: SS vs PS latencies
////////////////////////////////////////////////////////////////////////////////

def  : WriteRes<WriteFLoad,        [JLAGU, JFPU01, JFPX]> { let Latency = 5; }
def  : WriteRes<WriteFStore,        [JSAGU, JFPU1, JSTC]>;
def  : WriteRes<WriteFMove,               [JFPU01, JFPX]>;

defm : JWriteResFpuPair<WriteFAdd,         [JFPU0, JFPA],  3>;
defm : JWriteResFpuPair<WriteFMul,         [JFPU1, JFPM],  2>;
defm : JWriteResFpuPair<WriteFMA,          [JFPU1, JFPM],  2>; // NOTE: Doesn't exist on Jaguar.
defm : JWriteResFpuPair<WriteFRcp,         [JFPU1, JFPM],  2>;
defm : JWriteResFpuPair<WriteFRsqrt,       [JFPU1, JFPM],  2>;
defm : JWriteResFpuPair<WriteFDiv,         [JFPU1, JFPM], 19, [1, 19]>;
defm : JWriteResFpuPair<WriteFSqrt,        [JFPU1, JFPM], 21, [1, 21]>;
defm : JWriteResFpuPair<WriteFShuffle,    [JFPU01, JFPX],  1>;
defm : JWriteResFpuPair<WriteFBlend,      [JFPU01, JFPX],  1>;
defm : JWriteResFpuPair<WriteFVarBlend,   [JFPU01, JFPX],  2, [1, 4], 3>;
defm : JWriteResFpuPair<WriteFShuffle256, [JFPU01, JFPX],  1>;

////////////////////////////////////////////////////////////////////////////////
// Conversions.
// FIXME: integer pipes
////////////////////////////////////////////////////////////////////////////////

defm : JWriteResFpuPair<WriteCvtF2I,       [JFPU1, JSTC], 3>; // Float -> Integer.
defm : JWriteResFpuPair<WriteCvtI2F,       [JFPU1, JSTC], 3>; // Integer -> Float.
defm : JWriteResFpuPair<WriteCvtF2F,       [JFPU1, JSTC], 3>; // Float -> Float size conversion.

def JWriteCVTF2F : SchedWriteRes<[JFPU1, JSTC]> {
  let Latency = 7;
  let ResourceCycles = [1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteCVTF2F], (instregex "(V)?CVTS(D|S)2S(D|S)rr")>;

def JWriteCVTF2FLd : SchedWriteRes<[JLAGU, JFPU1, JSTC]> {
  let Latency = 12;
  let ResourceCycles = [1, 1, 2];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteCVTF2FLd], (instregex "(V)?CVTS(D|S)2S(D|S)rm")>;

def JWriteCVTF2SI : SchedWriteRes<[JFPU1, JSTC, JFPA, JALU0]> {
  let Latency = 7;
  let NumMicroOps = 2;
}
def : InstRW<[JWriteCVTF2SI], (instregex "(V)?CVT(T?)S(D|S)2SI(64)?rr")>;

def JWriteCVTF2SILd : SchedWriteRes<[JLAGU, JFPU1, JSTC, JFPA, JALU0]> {
  let Latency = 12;
  let NumMicroOps = 2;
}
def : InstRW<[JWriteCVTF2SILd], (instregex "(V)?CVT(T?)S(D|S)2SI(64)?rm")>;

// FIXME: f+3 ST,LD+STC latency
def JWriteCVTSI2F : SchedWriteRes<[JFPU1, JSTC]> {
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[JWriteCVTSI2F], (instregex "(V)?CVTSI(64)?2S(D|S)rr")>;

def JWriteCVTSI2FLd : SchedWriteRes<[JLAGU, JFPU1, JSTC]> {
  let Latency = 14;
  let NumMicroOps = 2;
}
def : InstRW<[JWriteCVTSI2FLd], (instregex "(V)?CVTSI(64)?2S(D|S)rm")>;

////////////////////////////////////////////////////////////////////////////////
// Vector integer operations.
////////////////////////////////////////////////////////////////////////////////

def  : WriteRes<WriteVecLoad,      [JLAGU, JFPU01, JVALU]> { let Latency = 5; }
def  : WriteRes<WriteVecStore,       [JSAGU, JFPU1, JSTC]>;
def  : WriteRes<WriteVecMove,             [JFPU01, JVALU]>;

defm : JWriteResFpuPair<WriteVecALU,      [JFPU01, JVALU], 1>;
defm : JWriteResFpuPair<WriteVecShift,    [JFPU01, JVALU], 1>;
defm : JWriteResFpuPair<WriteVecIMul,     [JFPU0, JVIMUL], 2>;
defm : JWriteResFpuPair<WriteMPSAD,       [JFPU0, JVIMUL], 3, [1, 2]>;
defm : JWriteResFpuPair<WriteShuffle,     [JFPU01, JVALU], 1>;
defm : JWriteResFpuPair<WriteBlend,       [JFPU01, JVALU], 1>;
defm : JWriteResFpuPair<WriteVarBlend,    [JFPU01, JVALU], 2, [1, 4], 3>;
defm : JWriteResFpuPair<WriteVecLogic,    [JFPU01, JVALU], 1>;
defm : JWriteResFpuPair<WriteShuffle256,  [JFPU01, JVALU], 1>;
defm : JWriteResFpuPair<WriteVarVecShift, [JFPU01, JVALU], 1>; // NOTE: Doesn't exist on Jaguar.

////////////////////////////////////////////////////////////////////////////////
// SSE42 String instructions.
////////////////////////////////////////////////////////////////////////////////

defm : JWriteResFpuPair<WritePCmpIStrI, [JFPU1, JVALU1, JFPA, JALU0], 7, [1, 2, 1, 1], 3>;
defm : JWriteResFpuPair<WritePCmpIStrM, [JFPU1, JVALU1, JFPA, JALU0], 8, [1, 2, 1, 1], 3>;
defm : JWriteResFpuPair<WritePCmpEStrI, [JFPU1, JSAGU, JLAGU, JVALU, JVALU1, JFPA, JALU0], 14, [1, 2, 2, 6, 4, 1, 1], 9>;
defm : JWriteResFpuPair<WritePCmpEStrM, [JFPU1, JSAGU, JLAGU, JVALU, JVALU1, JFPA, JALU0], 14, [1, 2, 2, 6, 4, 1, 1], 9>;

////////////////////////////////////////////////////////////////////////////////
// AES Instructions.
////////////////////////////////////////////////////////////////////////////////

defm : JWriteResFpuPair<WriteAESIMC,      [JFPU0, JVIMUL], 2>;
defm : JWriteResFpuPair<WriteAESKeyGen,   [JFPU0, JVIMUL], 2>;
defm : JWriteResFpuPair<WriteAESDecEnc,   [JFPU0, JVIMUL], 3>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : JWriteResFpuPair<WriteFHAdd,         [JFPU0, JFPA], 3>;
defm : JWriteResFpuPair<WritePHAdd,       [JFPU01, JVALU], 1>;

def JWriteFHAddY: SchedWriteRes<[JFPU0, JFPA]> {
  let Latency = 3;
  let ResourceCycles = [2, 2];
}
def : InstRW<[JWriteFHAddY], (instrs VHADDPDYrr, VHADDPSYrr, VHSUBPDYrr, VHSUBPSYrr)>;

def JWriteFHAddYLd: SchedWriteRes<[JLAGU, JFPU0, JFPA]> {
  let Latency = 8;
  let ResourceCycles = [2, 2, 2];
}
def : InstRW<[JWriteFHAddYLd], (instrs VHADDPDYrm, VHADDPSYrm, VHSUBPDYrm, VHSUBPSYrm)>;

////////////////////////////////////////////////////////////////////////////////
// Carry-less multiplication instructions.
////////////////////////////////////////////////////////////////////////////////

defm : JWriteResFpuPair<WriteCLMul,       [JFPU0, JVIMUL], 2>;

////////////////////////////////////////////////////////////////////////////////
// SSE4.1 instructions.
////////////////////////////////////////////////////////////////////////////////

def JWriteDPPS: SchedWriteRes<[JFPU1, JFPM, JFPA]> {
  let Latency = 11;
  let ResourceCycles = [1, 3, 3];
  let NumMicroOps = 5;
}
def : InstRW<[JWriteDPPS], (instrs DPPSrri, VDPPSrri)>;

def JWriteDPPSLd: SchedWriteRes<[JLAGU, JFPU1, JFPM, JFPA]> {
  let Latency = 16;
  let ResourceCycles = [1, 1, 3, 3];
  let NumMicroOps = 5;
}
def : InstRW<[JWriteDPPSLd], (instrs DPPSrmi, VDPPSrmi)>;

def JWriteDPPD: SchedWriteRes<[JFPU1, JFPM, JFPA]> {
  let Latency = 9;
  let ResourceCycles = [1, 3, 3];
  let NumMicroOps = 3;
}
def : InstRW<[JWriteDPPD], (instrs DPPDrri, VDPPDrri)>;

def JWriteDPPDLd: SchedWriteRes<[JLAGU, JFPU1, JFPM, JFPA]> {
  let Latency = 14;
  let ResourceCycles = [1, 1, 3, 3];
  let NumMicroOps = 3;
}
def : InstRW<[JWriteDPPDLd], (instrs DPPDrmi, VDPPDrmi)>;

////////////////////////////////////////////////////////////////////////////////
// SSE4A instructions.
////////////////////////////////////////////////////////////////////////////////

def JWriteEXTRQ: SchedWriteRes<[JFPU01, JVALU]> {
}
def : InstRW<[JWriteEXTRQ], (instrs EXTRQ, EXTRQI)>;

def JWriteINSERTQ: SchedWriteRes<[JFPU01, JVALU]> {
  let Latency = 2;
  let ResourceCycles = [1, 4];
}
def : InstRW<[JWriteINSERTQ], (instrs INSERTQ, INSERTQI)>;

////////////////////////////////////////////////////////////////////////////////
// F16C instructions.
////////////////////////////////////////////////////////////////////////////////

def JWriteCVT3: SchedWriteRes<[JFPU1, JSTC]> {
  let Latency = 3;
}
def : InstRW<[JWriteCVT3], (instrs VCVTPS2PHrr, VCVTPH2PSrr)>;

def JWriteCVT3St: SchedWriteRes<[JFPU1, JSTC, JSAGU]> {
  let Latency = 3;
}
def : InstRW<[JWriteCVT3St], (instrs VCVTPS2PHmr)>;

def JWriteCVT3Ld: SchedWriteRes<[JLAGU, JFPU1, JSTC]> {
  let Latency = 8;
}
def : InstRW<[JWriteCVT3Ld], (instrs VCVTPH2PSrm)>;

def JWriteCVTPS2PHY: SchedWriteRes<[JFPU1, JSTC, JFPX]> {
  let Latency = 6;
  let ResourceCycles = [2, 2, 2];
  let NumMicroOps = 3;
}
def : InstRW<[JWriteCVTPS2PHY], (instrs VCVTPS2PHYrr)>;

def JWriteCVTPS2PHYSt: SchedWriteRes<[JFPU1, JSTC, JFPX, JSAGU]> {
  let Latency = 11;
  let ResourceCycles = [2, 2, 2, 1];
  let NumMicroOps = 3;
}
def : InstRW<[JWriteCVTPS2PHYSt], (instrs VCVTPS2PHYmr)>;

def JWriteCVTPH2PSY: SchedWriteRes<[JFPU1, JSTC]> {
  let Latency = 3;
  let ResourceCycles = [2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteCVTPH2PSY], (instrs VCVTPH2PSYrr)>;

def JWriteCVTPH2PSYLd: SchedWriteRes<[JLAGU, JFPU1, JSTC]> {
  let Latency = 8;
  let ResourceCycles = [1, 2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteCVTPH2PSYLd], (instrs VCVTPH2PSYrm)>;

////////////////////////////////////////////////////////////////////////////////
// AVX instructions.
////////////////////////////////////////////////////////////////////////////////

def JWriteFLogic: SchedWriteRes<[JFPU01, JFPX]> {
}
def : InstRW<[JWriteFLogic], (instrs ORPDrr, ORPSrr, VORPDrr, VORPSrr,
                                     XORPDrr, XORPSrr, VXORPDrr, VXORPSrr,
                                     ANDPDrr, ANDPSrr, VANDPDrr, VANDPSrr,
                                     ANDNPDrr, ANDNPSrr, VANDNPDrr, VANDNPSrr)>;

def JWriteFLogicLd: SchedWriteRes<[JLAGU, JFPU01, JFPX]> {
  let Latency = 6;
}
def : InstRW<[JWriteFLogicLd], (instrs ORPDrm, ORPSrm, VORPDrm, VORPSrm,
                                       XORPDrm, XORPSrm, VXORPDrm, VXORPSrm,
                                       ANDPDrm, ANDPSrm, VANDPDrm, VANDPSrm,
                                       ANDNPDrm, ANDNPSrm, VANDNPDrm, VANDNPSrm)>;

def JWriteFLogicY: SchedWriteRes<[JFPU01, JFPX]> {
  let ResourceCycles = [2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteFLogicY], (instrs VORPDYrr, VORPSYrr,
                                      VXORPDYrr, VXORPSYrr,
                                      VANDPDYrr, VANDPSYrr,
                                      VANDNPDYrr, VANDNPSYrr)>;

def JWriteFLogicYLd: SchedWriteRes<[JLAGU, JFPU01, JFPX]> {
  let Latency = 6;
  let ResourceCycles = [2, 2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteFLogicYLd], (instrs VORPDYrm, VORPSYrm,
                                        VXORPDYrm, VXORPSYrm,
                                        VANDPDYrm, VANDPSYrm,
                                        VANDNPDYrm, VANDNPSYrm)>;

def JWriteVDPPSY: SchedWriteRes<[JFPU1, JFPM, JFPA]> {
  let Latency = 12;
  let ResourceCycles = [2, 6, 6];
  let NumMicroOps = 10;
}
def : InstRW<[JWriteVDPPSY], (instrs VDPPSYrri)>;

def JWriteVDPPSYLd: SchedWriteRes<[JLAGU, JFPU1, JFPM, JFPA]> {
  let Latency = 17;
  let ResourceCycles = [2, 2, 6, 6];
  let NumMicroOps = 10;
}
def : InstRW<[JWriteVDPPSYLd, ReadAfterLd], (instrs VDPPSYrmi)>;

def JWriteFAddY: SchedWriteRes<[JFPU0, JFPA]> {
  let Latency = 3;
  let ResourceCycles = [2, 2];
}
def : InstRW<[JWriteFAddY], (instrs VADDPDYrr, VADDPSYrr,
                                    VSUBPDYrr, VSUBPSYrr,
                                    VADDSUBPDYrr, VADDSUBPSYrr)>;

def JWriteFAddYLd: SchedWriteRes<[JLAGU, JFPU0, JFPA]> {
  let Latency = 8;
  let ResourceCycles = [2, 2, 2];
}
def : InstRW<[JWriteFAddYLd, ReadAfterLd], (instrs VADDPDYrm, VADDPSYrm,
                                                   VSUBPDYrm, VSUBPSYrm,
                                                   VADDSUBPDYrm, VADDSUBPSYrm)>;

def JWriteFDivY: SchedWriteRes<[JFPU1, JFPM]> {
  let Latency = 38;
  let ResourceCycles = [2, 38];
}
def : InstRW<[JWriteFDivY], (instrs VDIVPDYrr, VDIVPSYrr)>;

def JWriteFDivYLd: SchedWriteRes<[JLAGU, JFPU1, JFPM]> {
  let Latency = 43;
  let ResourceCycles = [2, 2, 38];
}
def : InstRW<[JWriteFDivYLd, ReadAfterLd], (instrs VDIVPDYrm, VDIVPSYrm)>;

def JWriteVMULYPD: SchedWriteRes<[JFPU1, JFPM]> {
  let Latency = 4;
  let ResourceCycles = [2, 4];
}
def : InstRW<[JWriteVMULYPD], (instrs VMULPDYrr)>;

def JWriteVMULYPDLd: SchedWriteRes<[JLAGU, JFPU1, JFPM]> {
  let Latency = 9;
  let ResourceCycles = [2, 2, 4];
}
def : InstRW<[JWriteVMULYPDLd, ReadAfterLd], (instrs VMULPDYrm)>;

def JWriteVMULYPS: SchedWriteRes<[JFPU1, JFPM]> {
  let Latency = 2;
  let ResourceCycles = [2, 2];
}
def : InstRW<[JWriteVMULYPS], (instrs VMULPSYrr, VRCPPSYr, VRSQRTPSYr)>;

def JWriteVMULYPSLd: SchedWriteRes<[JLAGU, JFPU1, JFPM]> {
  let Latency = 7;
  let ResourceCycles = [2, 2, 2];
}
def : InstRW<[JWriteVMULYPSLd, ReadAfterLd], (instrs VMULPSYrm, VRCPPSYm, VRSQRTPSYm)>;

def JWriteVMULPD: SchedWriteRes<[JFPU1, JFPM]> {
  let Latency = 4;
  let ResourceCycles = [1, 2];
}
def : InstRW<[JWriteVMULPD], (instrs MULPDrr, MULSDrr, VMULPDrr, VMULSDrr)>;

def JWriteVMULPDLd: SchedWriteRes<[JLAGU, JFPU1, JFPM]> {
  let Latency = 9;
  let ResourceCycles = [1, 1, 2];
}
def : InstRW<[JWriteVMULPDLd], (instrs MULPDrm, MULSDrm, VMULPDrm, VMULSDrm)>;

def JWriteVCVTY: SchedWriteRes<[JFPU1, JSTC]> {
  let Latency = 3;
  let ResourceCycles = [2, 2];
}
def : InstRW<[JWriteVCVTY], (instrs VCVTDQ2PDYrr, VCVTDQ2PSYrr,
                                    VCVTPS2DQYrr, VCVTTPS2DQYrr,
                                    VROUNDPDYr,   VROUNDPSYr)>;

def JWriteVCVTYLd: SchedWriteRes<[JLAGU, JFPU1, JSTC]> {
  let Latency = 8;
  let ResourceCycles = [2, 2, 2];
}
def : InstRW<[JWriteVCVTYLd, ReadAfterLd], (instrs VCVTDQ2PDYrm, VCVTDQ2PSYrm,
                                                   VCVTPS2DQYrm, VCVTTPS2DQYrm,
                                                   VROUNDPDYm,   VROUNDPSYm)>;

def JWriteVMOVNTDQSt: SchedWriteRes<[JFPU1, JSTC, JSAGU]> {
  let Latency = 2;
}
def : InstRW<[JWriteVMOVNTDQSt], (instrs MOVNTDQmr, VMOVNTDQmr)>;

def JWriteMOVNTSt: SchedWriteRes<[JFPU1, JSTC, JSAGU]> {
  let Latency = 3;
}
def : InstRW<[JWriteMOVNTSt], (instrs MOVNTPDmr, MOVNTPSmr, MOVNTSD, MOVNTSS, VMOVNTPDmr, VMOVNTPSmr)>;

def JWriteVMOVNTPYSt: SchedWriteRes<[JFPU1, JSTC, JSAGU]> {
  let Latency = 3;
  let ResourceCycles = [2, 2, 2];
}
def : InstRW<[JWriteVMOVNTPYSt], (instrs VMOVNTDQYmr, VMOVNTPDYmr, VMOVNTPSYmr)>;

def JWriteFComi : SchedWriteRes<[JFPU0, JFPA, JALU0]> {
  let Latency = 3;
}
def : InstRW<[JWriteFComi], (instregex "(V)?(U)?COMIS(D|S)rr")>;

def JWriteFComiLd : SchedWriteRes<[JLAGU, JFPU0, JFPA, JALU0]> {
  let Latency = 8;
}
def : InstRW<[JWriteFComiLd], (instregex "(V)?(U)?COMIS(D|S)rm")>;

def JWriteFCmp: SchedWriteRes<[JFPU0, JFPA]> {
  let Latency = 2;
}
def : InstRW<[JWriteFCmp], (instregex "(V)?M(AX|IN)(P|S)(D|S)rr",
                                      "(V)?CMPP(S|D)rri", "(V)?CMPS(S|D)rr")>;

def JWriteFCmpLd: SchedWriteRes<[JLAGU, JFPU0, JFPA]> {
  let Latency = 7;
}
def : InstRW<[JWriteFCmpLd], (instregex "(V)?M(AX|IN)(P|S)(D|S)rm",
                                        "(V)?CMPP(S|D)rmi", "(V)?CMPS(S|D)rm")>;

def JWriteVCVTPDY: SchedWriteRes<[JFPU1, JSTC, JFPX]> {
  let Latency = 6;
  let ResourceCycles = [2, 2, 4];
}
def : InstRW<[JWriteVCVTPDY], (instrs VCVTPD2DQYrr, VCVTTPD2DQYrr, VCVTPD2PSYrr)>;

def JWriteVCVTPDYLd: SchedWriteRes<[JLAGU, JFPU1, JSTC, JFPX]> {
  let Latency = 11;
  let ResourceCycles = [2, 2, 2, 4];
}
def : InstRW<[JWriteVCVTPDYLd, ReadAfterLd], (instrs VCVTPD2DQYrm, VCVTTPD2DQYrm, VCVTPD2PSYrm)>;

def JWritePSHUFB: SchedWriteRes<[JFPU01, JVALU]> {
  let Latency = 2;
  let ResourceCycles = [1, 4];
  let NumMicroOps = 3;
}
def : InstRW<[JWritePSHUFB], (instrs PSHUFBrr, VPSHUFBrr)>;

def JWritePSHUFBLd: SchedWriteRes<[JLAGU, JFPU01, JVALU]> {
  let Latency = 7;
  let ResourceCycles = [1, 1, 4];
  let NumMicroOps = 3;
}
def : InstRW<[JWritePSHUFBLd, ReadAfterLd], (instrs PSHUFBrm, VPSHUFBrm)>;

def JWriteVPERM: SchedWriteRes<[JFPU01, JFPX]> {
  let Latency = 2;
  let ResourceCycles = [1, 4];
  let NumMicroOps = 3;
}
def : InstRW<[JWriteVPERM], (instrs VPERMILPDrr, VPERMILPSrr)>;

def JWriteVPERMLd: SchedWriteRes<[JLAGU, JFPU01, JFPX]> {
  let Latency = 7;
  let ResourceCycles = [1, 1, 4];
  let NumMicroOps = 3;
}
def : InstRW<[JWriteVPERMLd, ReadAfterLd], (instrs VPERMILPDrm, VPERMILPSrm)>;

def JWriteVPERMY: SchedWriteRes<[JFPU01, JFPX]> {
  let Latency = 3;
  let ResourceCycles = [2, 6];
  let NumMicroOps = 6;
}
def : InstRW<[JWriteVPERMY], (instrs VBLENDVPDYrr, VBLENDVPSYrr, VPERMILPDYrr, VPERMILPSYrr)>;

def JWriteVPERMYLd: SchedWriteRes<[JLAGU, JFPU01, JFPX]> {
  let Latency = 8;
  let ResourceCycles = [2, 2, 6];
  let NumMicroOps = 6;
}
def : InstRW<[JWriteVPERMYLd, ReadAfterLd], (instrs VBLENDVPDYrm, VBLENDVPSYrm, VPERMILPDYrm, VPERMILPSYrm)>;

def JWriteShuffleY: SchedWriteRes<[JFPU01, JFPX]> {
  let ResourceCycles = [2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteShuffleY], (instrs VBLENDPDYrri, VBLENDPSYrri,
                                       VMOVDDUPYrr, VMOVSHDUPYrr, VMOVSLDUPYrr,
                                       VPERMILPDYri, VPERMILPSYri, VSHUFPDYrri, VSHUFPSYrri,
                                       VUNPCKHPDYrr, VUNPCKHPSYrr, VUNPCKLPDYrr, VUNPCKLPSYrr)>;

def JWriteShuffleYLd: SchedWriteRes<[JLAGU, JFPU01, JFPX]> {
  let Latency = 6;
  let ResourceCycles = [2, 2, 2];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteShuffleYLd, ReadAfterLd], (instrs VBLENDPDYrmi, VBLENDPSYrmi,
                                                      VMOVDDUPYrm, VMOVSHDUPYrm, VMOVSLDUPYrm,
                                                      VPERMILPDYmi, VPERMILPSYmi, VSHUFPDYrmi, VSHUFPSYrmi,
                                                      VUNPCKHPDYrm, VUNPCKHPSYrm, VUNPCKLPDYrm, VUNPCKLPSYrm)>;

def JWriteVBROADCASTYLd: SchedWriteRes<[JLAGU, JFPU01, JFPX]> {
  let Latency = 6;
  let ResourceCycles = [1, 2, 4];
}
def : InstRW<[JWriteVBROADCASTYLd, ReadAfterLd], (instrs VBROADCASTSDYrm, VBROADCASTSSYrm)>;

def JWriteFPAY22: SchedWriteRes<[JFPU0, JFPA]> {
  let Latency = 2;
  let ResourceCycles = [2, 2];
}
def : InstRW<[JWriteFPAY22], (instregex "VCMPP(S|D)Yrri", "VM(AX|IN)P(D|S)Yrr")>;

def JWriteFPAY22Ld: SchedWriteRes<[JLAGU, JFPU0, JFPA]> {
  let Latency = 7;
  let ResourceCycles = [2, 2, 2];
}
def : InstRW<[JWriteFPAY22Ld, ReadAfterLd], (instregex "VCMPP(S|D)Yrmi", "VM(AX|IN)P(D|S)Yrm")>;

def JWriteVMaskMovLd: SchedWriteRes<[JLAGU, JFPU01, JFPX]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 2];
}
def : InstRW<[JWriteVMaskMovLd], (instrs VMASKMOVPDrm, VMASKMOVPSrm)>;

def JWriteVMaskMovYLd: SchedWriteRes<[JLAGU, JFPU01, JFPX]> {
  let Latency = 6;
  let ResourceCycles = [2, 2, 4];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteVMaskMovYLd], (instrs VMASKMOVPDYrm, VMASKMOVPSYrm)>;

def JWriteVMaskMovSt: SchedWriteRes<[JFPU01, JFPX, JSAGU]> {
  let Latency = 6;
  let ResourceCycles = [1, 4, 1];
}
def : InstRW<[JWriteVMaskMovSt], (instrs VMASKMOVPDmr, VMASKMOVPSmr)>;

def JWriteVMaskMovYSt: SchedWriteRes<[JFPU01, JFPX, JSAGU]> {
  let Latency = 6;
  let ResourceCycles = [2, 4, 2];
  let NumMicroOps = 2;
}
def : InstRW<[JWriteVMaskMovYSt], (instrs VMASKMOVPDYmr, VMASKMOVPSYmr)>;

def JWriteVMOVMSK: SchedWriteRes<[JFPU0, JFPA, JALU0]> {
  let Latency = 3;
}
def : InstRW<[JWriteVMOVMSK], (instrs MOVMSKPDrr, VMOVMSKPDrr, VMOVMSKPDYrr,
                                      MOVMSKPSrr, VMOVMSKPSrr, VMOVMSKPSYrr,
                                      PMOVMSKBrr, VPMOVMSKBrr)>;

def JWriteVTESTY: SchedWriteRes<[JFPU01, JFPX, JFPA, JALU0]> {
  let Latency = 4;
  let ResourceCycles = [2, 2, 2, 1];
  let NumMicroOps = 3;
}
def : InstRW<[JWriteVTESTY], (instrs VPTESTYrr, VTESTPDYrr, VTESTPSYrr)>;

def JWriteVTESTYLd: SchedWriteRes<[JLAGU, JFPU01, JFPX, JFPA, JALU0]> {
  let Latency = 9;
  let ResourceCycles = [2, 2, 2, 2, 1];
  let NumMicroOps = 3;
}
def : InstRW<[JWriteVTESTYLd], (instrs VPTESTYrm, VTESTPDYrm, VTESTPSYrm)>;

def JWriteVTEST: SchedWriteRes<[JFPU0, JFPA, JALU0]> {
  let Latency = 3;
}
def : InstRW<[JWriteVTEST], (instrs PTESTrr, VPTESTrr, VTESTPDrr, VTESTPSrr)>;

def JWriteVTESTLd: SchedWriteRes<[JLAGU, JFPU0, JFPA, JALU0]> {
  let Latency = 8;
}
def : InstRW<[JWriteVTESTLd], (instrs PTESTrm, VPTESTrm, VTESTPDrm, VTESTPSrm)>;

def JWriteVSQRTPD: SchedWriteRes<[JFPU1, JFPM]> {
  let Latency = 27;
  let ResourceCycles = [1, 27];
}
def : InstRW<[JWriteVSQRTPD], (instrs SQRTPDr, VSQRTPDr,
                                      SQRTSDr, VSQRTSDr,
                                      SQRTSDr_Int, VSQRTSDr_Int)>;

def JWriteVSQRTPDLd: SchedWriteRes<[JLAGU, JFPU1, JFPM]> {
  let Latency = 32;
  let ResourceCycles = [1, 1, 27];
}
def : InstRW<[JWriteVSQRTPDLd], (instrs SQRTPDm, VSQRTPDm,
                                        SQRTSDm, VSQRTSDm,
                                        SQRTSDm_Int, VSQRTSDm_Int)>;

def JWriteVSQRTYPD: SchedWriteRes<[JFPU1, JFPM]> {
  let Latency = 54;
  let ResourceCycles = [2, 54];
}
def : InstRW<[JWriteVSQRTYPD], (instrs VSQRTPDYr)>;

def JWriteVSQRTYPDLd: SchedWriteRes<[JLAGU, JFPU1, JFPM]> {
  let Latency = 59;
  let ResourceCycles = [2, 2, 54];
}
def : InstRW<[JWriteVSQRTYPDLd], (instrs VSQRTPDYm)>;

def JWriteVSQRTYPS: SchedWriteRes<[JFPU1, JFPM]> {
  let Latency = 42;
  let ResourceCycles = [2, 42];
}
def : InstRW<[JWriteVSQRTYPS], (instrs VSQRTPSYr)>;

def JWriteVSQRTYPSLd: SchedWriteRes<[JLAGU, JFPU1, JFPM]> {
  let Latency = 47;
  let ResourceCycles = [2, 2, 42];
}
def : InstRW<[JWriteVSQRTYPSLd], (instrs VSQRTPSYm)>;

def JWriteJVZEROALL: SchedWriteRes<[]> {
  let Latency = 90;
  let NumMicroOps = 73;
}
def : InstRW<[JWriteJVZEROALL], (instrs VZEROALL)>;

def JWriteJVZEROUPPER: SchedWriteRes<[]> {
  let Latency = 46;
  let NumMicroOps = 37;
}
def : InstRW<[JWriteJVZEROUPPER], (instrs VZEROUPPER)>;
} // SchedModel

