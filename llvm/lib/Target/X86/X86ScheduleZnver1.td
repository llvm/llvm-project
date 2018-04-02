//=- X86ScheduleZnver1.td - X86 Znver1 Scheduling -------------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Znver1 to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def Znver1Model : SchedMachineModel {
  // Zen can decode 4 instructions per cycle.
  let IssueWidth = 4;
  // Based on the reorder buffer we define MicroOpBufferSize
  let MicroOpBufferSize = 192;
  let LoadLatency = 4;
  let MispredictPenalty = 17;
  let HighLatency = 25;
  let PostRAScheduler = 1;

  // FIXME: This variable is required for incomplete model.
  // We haven't catered all instructions.
  // So, we reset the value of this variable so as to
  // say that the model is incomplete.
  let CompleteModel = 0;
}

let SchedModel = Znver1Model in {

// Zen can issue micro-ops to 10 different units in one cycle.
// These are
//  * Four integer ALU units (ZALU0, ZALU1, ZALU2, ZALU3)
//  * Two AGU units (ZAGU0, ZAGU1)
//  * Four FPU units (ZFPU0, ZFPU1, ZFPU2, ZFPU3)
// AGUs feed load store queues @two loads and 1 store per cycle.

// Four ALU units are defined below
def ZnALU0 : ProcResource<1>;
def ZnALU1 : ProcResource<1>;
def ZnALU2 : ProcResource<1>;
def ZnALU3 : ProcResource<1>;

// Two AGU units are defined below
def ZnAGU0 : ProcResource<1>;
def ZnAGU1 : ProcResource<1>;

// Four FPU units are defined below
def ZnFPU0 : ProcResource<1>;
def ZnFPU1 : ProcResource<1>;
def ZnFPU2 : ProcResource<1>;
def ZnFPU3 : ProcResource<1>;

// FPU grouping
def ZnFPU     : ProcResGroup<[ZnFPU0, ZnFPU1, ZnFPU2, ZnFPU3]>;
def ZnFPU013  : ProcResGroup<[ZnFPU0, ZnFPU1, ZnFPU3]>;
def ZnFPU01   : ProcResGroup<[ZnFPU0, ZnFPU1]>;
def ZnFPU12   : ProcResGroup<[ZnFPU1, ZnFPU2]>;
def ZnFPU13   : ProcResGroup<[ZnFPU1, ZnFPU3]>;
def ZnFPU23   : ProcResGroup<[ZnFPU2, ZnFPU3]>;
def ZnFPU02   : ProcResGroup<[ZnFPU0, ZnFPU2]>;
def ZnFPU03   : ProcResGroup<[ZnFPU0, ZnFPU3]>;

// Below are the grouping of the units.
// Micro-ops to be issued to multiple units are tackled this way.

// ALU grouping
// ZnALU03 - 0,3 grouping
def ZnALU03: ProcResGroup<[ZnALU0, ZnALU3]>;

// 56 Entry (14x4 entries) Int Scheduler
def ZnALU : ProcResGroup<[ZnALU0, ZnALU1, ZnALU2, ZnALU3]> {
  let BufferSize=56;
}

// 28 Entry (14x2) AGU group. AGUs can't be used for all ALU operations
// but are relevant for some instructions
def ZnAGU : ProcResGroup<[ZnAGU0, ZnAGU1]> {
  let BufferSize=28;
}

// Integer Multiplication issued on ALU1.
def ZnMultiplier : ProcResource<1>;

// Integer division issued on ALU2.
def ZnDivider : ProcResource<1>;

// 4 Cycles load-to use Latency is captured
def : ReadAdvance<ReadAfterLd, 4>;

// (a folded load is an instruction that loads and does some operation)
// Ex: ADDPD xmm,[mem]-> This instruction has two micro-ops
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops.
//      a. load and
//      b. addpd
// This multiclass is for folded loads for integer units.
multiclass ZnWriteResPair<X86FoldableSchedWrite SchedRW,
                          list<ProcResourceKind> ExePorts,
                          int Lat, list<int> Res = [1], int UOps = 1> {
  // Register variant takes 1-cycle on Execution Port.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on ZnAGU
  // adds 4 cycles to the latency.
  def : WriteRes<SchedRW.Folded, !listconcat([ZnAGU], ExePorts)> {
    let Latency = !add(Lat, 4);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = !add(UOps, 1);
  }
}

// This multiclass is for folded loads for floating point units.
multiclass ZnWriteResFpuPair<X86FoldableSchedWrite SchedRW,
                          list<ProcResourceKind> ExePorts,
                          int Lat, list<int> Res = [1], int UOps = 1> {
  // Register variant takes 1-cycle on Execution Port.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on ZnAGU
  // adds 7 cycles to the latency.
  def : WriteRes<SchedRW.Folded, !listconcat([ZnAGU], ExePorts)> {
    let Latency = !add(Lat, 7);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = UOps;
  }
}

// WriteRMW is set for instructions with Memory write
// operation in codegen
def : WriteRes<WriteRMW, [ZnAGU]>;

def : WriteRes<WriteStore, [ZnAGU]>;
def : WriteRes<WriteMove,  [ZnALU]>;
def : WriteRes<WriteLoad,  [ZnAGU]> { let Latency = 8; }

def : WriteRes<WriteZero,  []>;
def : WriteRes<WriteLEA, [ZnALU]>;
defm : ZnWriteResPair<WriteALU,   [ZnALU], 1>;
defm : ZnWriteResPair<WriteIMul,   [ZnALU1, ZnMultiplier], 4>;
defm : ZnWriteResPair<WriteShift, [ZnALU], 1>;
defm : ZnWriteResPair<WriteJump,  [ZnALU], 1>;
defm : ZnWriteResFpuPair<WriteCRC32, [ZnFPU0], 3>;

// Bit counts.
defm : ZnWriteResPair<WriteBitScan, [ZnALU], 3>;
defm : ZnWriteResPair<WriteLZCNT,   [ZnALU], 2>;
defm : ZnWriteResPair<WriteTZCNT,   [ZnALU], 2>;
defm : ZnWriteResPair<WritePOPCNT,  [ZnALU], 1>;

// Treat misc copies as a move.
def : InstRW<[WriteMove], (instrs COPY)>;

// BMI1 BEXTR, BMI2 BZHI
defm : ZnWriteResPair<WriteBEXTR, [ZnALU], 1>;
defm : ZnWriteResPair<WriteBZHI, [ZnALU], 1>;

// IDIV
def : WriteRes<WriteIDiv, [ZnALU2, ZnDivider]> {
  let Latency = 41;
  let ResourceCycles = [1, 41];
}

def : WriteRes<WriteIDivLd, [ZnALU2, ZnAGU, ZnDivider]> {
  let Latency = 45;
  let ResourceCycles = [1, 4, 41];
}

// IMULH
def  : WriteRes<WriteIMulH, [ZnALU1, ZnMultiplier]>{
  let Latency = 4;
}

// Floating point operations
def : WriteRes<WriteFStore,             [ZnAGU]>;
def : WriteRes<WriteFMove,              [ZnFPU]>;
def : WriteRes<WriteFLoad,              [ZnAGU]> { let Latency = 8; }

defm : ZnWriteResFpuPair<WriteFHAdd,     [ZnFPU0],  3>;
defm : ZnWriteResFpuPair<WriteFAdd,      [ZnFPU0],  3>;
defm : ZnWriteResFpuPair<WriteFBlend,    [ZnFPU01], 1>;
defm : ZnWriteResFpuPair<WriteFVarBlend, [ZnFPU01], 1>;
defm : ZnWriteResFpuPair<WriteVarBlend,  [ZnFPU0],  1>;
defm : ZnWriteResFpuPair<WriteCvtI2F,    [ZnFPU3],  5>;
defm : ZnWriteResFpuPair<WriteCvtF2F,    [ZnFPU3],  5>;
defm : ZnWriteResFpuPair<WriteCvtF2I,    [ZnFPU3],  5>;
defm : ZnWriteResFpuPair<WriteFDiv,      [ZnFPU3], 15>;
defm : ZnWriteResFpuPair<WriteFShuffle,  [ZnFPU12], 1>;
defm : ZnWriteResFpuPair<WriteFMul,      [ZnFPU0],  5>;
defm : ZnWriteResFpuPair<WriteFMA,       [ZnFPU03], 5>;
defm : ZnWriteResFpuPair<WriteFRcp,      [ZnFPU01], 5>;
defm : ZnWriteResFpuPair<WriteFRsqrt,    [ZnFPU01], 5>;
defm : ZnWriteResFpuPair<WriteFSqrt,     [ZnFPU3], 20>;

// Vector integer operations which uses FPU units
def  : WriteRes<WriteVecStore,            [ZnAGU]>;
def  : WriteRes<WriteVecMove,             [ZnFPU]>;
def  : WriteRes<WriteVecLoad,             [ZnAGU]> { let Latency = 8; }

defm : ZnWriteResFpuPair<WriteVecShift,   [ZnFPU],   1>;
defm : ZnWriteResFpuPair<WriteVecLogic,   [ZnFPU],   1>;
defm : ZnWriteResFpuPair<WritePHAdd,      [ZnFPU],   1>;
defm : ZnWriteResFpuPair<WriteVecALU,     [ZnFPU],   1>;
defm : ZnWriteResFpuPair<WriteVecIMul,    [ZnFPU0],  4>;
defm : ZnWriteResFpuPair<WritePMULLD,     [ZnFPU0],  4>; // FIXME
defm : ZnWriteResFpuPair<WriteShuffle,    [ZnFPU],   1>;
defm : ZnWriteResFpuPair<WriteBlend,      [ZnFPU01], 1>;
defm : ZnWriteResFpuPair<WriteShuffle256, [ZnFPU],   2>;

// Vector Shift Operations
defm : ZnWriteResFpuPair<WriteVarVecShift, [ZnFPU12], 1>;

// MOVMSK Instructions.
def : WriteRes<WriteFMOVMSK, [ZnFPU2]>;
def : WriteRes<WriteVecMOVMSK, [ZnFPU2]>;
def : WriteRes<WriteMMXMOVMSK, [ZnFPU2]>;

// AES Instructions.
defm : ZnWriteResFpuPair<WriteAESDecEnc, [ZnFPU01], 4>;
defm : ZnWriteResFpuPair<WriteAESIMC,    [ZnFPU01], 4>;
defm : ZnWriteResFpuPair<WriteAESKeyGen, [ZnFPU01], 4>;

def : WriteRes<WriteFence,  [ZnAGU]>;
def : WriteRes<WriteNop, []>;

// Following instructions with latency=100 are microcoded.
// We set long latency so as to block the entire pipeline.
defm : ZnWriteResFpuPair<WriteFShuffle256, [ZnFPU], 100>;

//Microcoded Instructions
let Latency = 100 in {
  def : WriteRes<WriteMicrocoded, []>;
  def : WriteRes<WriteSystem, []>;
  def : WriteRes<WriteMPSAD, []>;
  def : WriteRes<WriteMPSADLd, []>;
  def : WriteRes<WriteCLMul, []>;
  def : WriteRes<WriteCLMulLd, []>;
  def : WriteRes<WritePCmpIStrM, []>;
  def : WriteRes<WritePCmpIStrMLd, []>;
  def : WriteRes<WritePCmpEStrI, []>;
  def : WriteRes<WritePCmpEStrILd, []>;
  def : WriteRes<WritePCmpEStrM, []>;
  def : WriteRes<WritePCmpEStrMLd, []>;
  def : WriteRes<WritePCmpIStrI, []>;
  def : WriteRes<WritePCmpIStrILd, []>;
  }

//=== Regex based itineraries ===//
// Notation:
// - r: register.
// - m = memory.
// - i = immediate
// - mm: 64 bit mmx register.
// - x = 128 bit xmm register.
// - (x)mm = mmx or xmm register.
// - y = 256 bit ymm register.
// - v = any vector register.

//=== Integer Instructions ===//
//-- Move instructions --//
// MOV.
// r16,m.
def : InstRW<[WriteALULd, ReadAfterLd], (instregex "MOV16rm")>;

// MOVSX, MOVZX.
// r,m.
def : InstRW<[WriteLoad], (instregex "MOV(S|Z)X32rm(8|16)")>;

// CMOVcc.
// r,r.
def : InstRW<[WriteALU],
      (instregex "CMOV(O|NO|B|AE|E|NE|BE|A|S|NS|P|NP|L|GE|LE|G)(16|32|64)rr")>;
// r,m.
def : InstRW<[WriteALULd, ReadAfterLd],
      (instregex "CMOV(O|NO|B|AE|E|NE|BE|A|S|NS|P|NP|L|GE|LE|G)(16|32|64)rm")>;

// XCHG.
// r,r.
def ZnWriteXCHG : SchedWriteRes<[ZnALU]> {
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}

def : InstRW<[ZnWriteXCHG], (instregex "XCHG(8|16|32|64)rr", "XCHG(16|32|64)ar")>;

// r,m.
def ZnWriteXCHGrm : SchedWriteRes<[ZnAGU, ZnALU]> {
  let Latency = 5;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteXCHGrm, ReadAfterLd], (instregex "XCHG(8|16|32|64)rm")>;

def : InstRW<[WriteMicrocoded], (instregex "XLAT")>;

// POP16.
// r.
def ZnWritePop16r : SchedWriteRes<[ZnAGU]>{
  let Latency = 5;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWritePop16r], (instregex "POP16rmm")>;
def : InstRW<[WriteMicrocoded], (instregex "POPF(16|32)")>;
def : InstRW<[WriteMicrocoded], (instregex "POPA(16|32)")>;


// PUSH.
// r. Has default values.
// m.
def ZnWritePUSH : SchedWriteRes<[ZnAGU]>{
  let Latency = 4;
}
def : InstRW<[ZnWritePUSH], (instregex "PUSH(16|32)rmm")>;

//PUSHF
def : InstRW<[WriteMicrocoded], (instregex "PUSHF(16|32)")>;

// PUSHA.
def ZnWritePushA : SchedWriteRes<[ZnAGU]> {
  let Latency = 8;
}
def : InstRW<[ZnWritePushA], (instregex "PUSHA(16|32)")>;

//LAHF
def : InstRW<[WriteMicrocoded], (instregex "LAHF")>;

// SAHF.
def ZnWriteSAHF : SchedWriteRes<[ZnALU]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteSAHF], (instregex "SAHF")>;

// BSWAP.
def ZnWriteBSwap : SchedWriteRes<[ZnALU]> {
  let ResourceCycles = [4];
}
def : InstRW<[ZnWriteBSwap], (instregex "BSWAP")>;

// MOVBE.
// r,m.
def ZnWriteMOVBE : SchedWriteRes<[ZnAGU, ZnALU]> {
  let Latency = 5;
}
def : InstRW<[ZnWriteMOVBE, ReadAfterLd], (instregex "MOVBE(16|32|64)rm")>;

// m16,r16.
def : InstRW<[ZnWriteMOVBE], (instregex "MOVBE(16|32|64)mr")>;

//-- Arithmetic instructions --//

// ADD SUB.
// m,r/i.
def : InstRW<[WriteALULd], (instregex "(ADD|SUB)(8|16|32|64)m(r|i)",
                          "(ADD|SUB)(8|16|32|64)mi8",
                          "(ADD|SUB)64mi32")>;

// ADC SBB.
// r,r/i.
def : InstRW<[WriteALU], (instregex "(ADC|SBB)(8|16|32|64)r(r|i)",
                          "(ADC|SBB)(16|32|64)ri8",
                          "(ADC|SBB)64ri32")>;

// r,m.
def : InstRW<[WriteALULd, ReadAfterLd],
            (instregex "(ADC|SBB)(8|16|32|64)rm")>;

// m,r/i.
def : InstRW<[WriteALULd],
             (instregex "(ADC|SBB)(8|16|32|64)m(r|i)",
              "(ADC|SBB)(16|32|64)mi8",
              "(ADC|SBB)64mi32")>;

// INC DEC NOT NEG.
// m.
def : InstRW<[WriteALULd],
             (instregex "(INC|DEC|NOT|NEG)(8|16|32|64)m")>;

// MUL IMUL.
// r16.
def ZnWriteMul16 : SchedWriteRes<[ZnALU1, ZnMultiplier]> {
  let Latency = 3;
}
def : InstRW<[ZnWriteMul16], (instrs IMUL16r, MUL16r)>;
def : InstRW<[ZnWriteMul16], (instrs IMUL16rr, IMUL16rri, IMUL16rri8)>; // TODO: is this right?
def : InstRW<[ZnWriteMul16], (instrs IMUL16rm, IMUL16rmi, IMUL16rmi8)>; // TODO: this is definitely wrong but matches what the instregex did.

// m16.
def ZnWriteMul16Ld : SchedWriteRes<[ZnAGU, ZnALU1, ZnMultiplier]> {
  let Latency = 8;
}
def : InstRW<[ZnWriteMul16Ld, ReadAfterLd], (instrs IMUL16m, MUL16m)>;

// r32.
def ZnWriteMul32 : SchedWriteRes<[ZnALU1, ZnMultiplier]> {
  let Latency = 3;
}
def : InstRW<[ZnWriteMul32], (instrs IMUL32r, MUL32r)>;
def : InstRW<[ZnWriteMul32], (instrs IMUL32rr, IMUL32rri, IMUL32rri8)>; // TODO: is this right?
def : InstRW<[ZnWriteMul32], (instrs IMUL32rm, IMUL32rmi, IMUL32rmi8)>; // TODO: this is definitely wrong but matches what the instregex did.

// m32.
def ZnWriteMul32Ld : SchedWriteRes<[ZnAGU, ZnALU1, ZnMultiplier]> {
  let Latency = 8;
}
def : InstRW<[ZnWriteMul32Ld, ReadAfterLd], (instrs IMUL32m, MUL32m)>;

// r64.
def ZnWriteMul64 : SchedWriteRes<[ZnALU1, ZnMultiplier]> {
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteMul64], (instrs IMUL64r, MUL64r)>;
def : InstRW<[ZnWriteMul64], (instrs IMUL64rr, IMUL64rri8, IMUL64rri32)>; // TODO: is this right?
def : InstRW<[ZnWriteMul64], (instrs IMUL64rm, IMUL64rmi32, IMUL64rmi8)>; // TODO: this is definitely wrong but matches what the instregex did.

// m64.
def ZnWriteMul64Ld : SchedWriteRes<[ZnAGU, ZnALU1, ZnMultiplier]> {
  let Latency = 9;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteMul64Ld, ReadAfterLd], (instrs IMUL64m, MUL64m)>;

// MULX.
// r32,r32,r32.
def ZnWriteMulX32 : SchedWriteRes<[ZnALU1, ZnMultiplier]> {
  let Latency = 3;
  let ResourceCycles = [1, 2];
}
def : InstRW<[ZnWriteMulX32], (instrs MULX32rr)>;

// r32,r32,m32.
def ZnWriteMulX32Ld : SchedWriteRes<[ZnAGU, ZnALU1, ZnMultiplier]> {
  let Latency = 8;
  let ResourceCycles = [1, 2, 2];
}
def : InstRW<[ZnWriteMulX32Ld, ReadAfterLd], (instrs MULX32rm)>;

// r64,r64,r64.
def ZnWriteMulX64 : SchedWriteRes<[ZnALU1]> {
  let Latency = 3;
}
def : InstRW<[ZnWriteMulX64], (instrs MULX64rr)>;

// r64,r64,m64.
def ZnWriteMulX64Ld : SchedWriteRes<[ZnAGU, ZnALU1, ZnMultiplier]> {
  let Latency = 8;
}
def : InstRW<[ZnWriteMulX64Ld, ReadAfterLd], (instrs MULX64rm)>;

// DIV, IDIV.
// r8.
def ZnWriteDiv8 : SchedWriteRes<[ZnALU2, ZnDivider]> {
  let Latency = 15;
}
def : InstRW<[ZnWriteDiv8], (instregex "DIV8r", "IDIV8r")>;

// r16.
def ZnWriteDiv16 : SchedWriteRes<[ZnALU2, ZnDivider]> {
  let Latency = 17;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteDiv16], (instregex "DIV16r", "IDIV16r")>;

// r32.
def ZnWriteDiv32 : SchedWriteRes<[ZnALU2, ZnDivider]> {
  let Latency = 25;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteDiv32], (instregex "DIV32r", "IDIV32r")>;

// r64.
def ZnWriteDiv64 : SchedWriteRes<[ZnALU2, ZnDivider]> {
  let Latency = 41;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteDiv64], (instregex "DIV64r", "IDIV64r")>;

//-- Control transfer instructions --//

// J(E|R)CXZ.
def ZnWriteJCXZ : SchedWriteRes<[ZnALU03]>;
def : InstRW<[ZnWriteJCXZ], (instrs JCXZ, JECXZ, JRCXZ)>;

// INTO
def : InstRW<[WriteMicrocoded], (instregex "INTO")>;

// LOOP.
def ZnWriteLOOP : SchedWriteRes<[ZnALU03]>;
def : InstRW<[ZnWriteLOOP], (instrs LOOP)>;

// LOOP(N)E, LOOP(N)Z
def ZnWriteLOOPE : SchedWriteRes<[ZnALU03]>;
def : InstRW<[ZnWriteLOOPE], (instrs LOOPE, LOOPNE)>;

// CALL.
// r.
def ZnWriteCALLr : SchedWriteRes<[ZnAGU, ZnALU03]>;
def : InstRW<[ZnWriteCALLr], (instregex "CALL(16|32)r")>;

def : InstRW<[WriteMicrocoded], (instregex "CALL(16|32)m")>;

// RET.
def ZnWriteRET : SchedWriteRes<[ZnALU03]> {
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteRET], (instregex "RET(L|Q|W)", "LRET(L|Q|W)",
                            "IRET(16|32|64)")>;

//-- Logic instructions --//

// AND OR XOR.
// m,r/i.
def : InstRW<[WriteALULd],
             (instregex "(AND|OR|XOR)(8|16|32|64)m(r|i)",
              "(AND|OR|XOR)(8|16|32|64)mi8", "(AND|OR|XOR)64mi32")>;

// ANDN.
// r,r.
def : InstRW<[WriteALU], (instregex "ANDN(32|64)rr")>;
// r,m.
def : InstRW<[WriteALULd, ReadAfterLd], (instregex "ANDN(32|64)rm")>;

// Define ALU latency variants
def ZnWriteALULat2 : SchedWriteRes<[ZnALU]> {
  let Latency = 2;
}
def ZnWriteALULat2Ld : SchedWriteRes<[ZnAGU, ZnALU]> {
  let Latency = 6;
}

// BT.
// r,r/i.
def : InstRW<[WriteShift], (instregex "BT(16|32|64)r(r|i8)")>;

def : InstRW<[WriteShiftLd], (instregex "BT(16|32|64)mr")>;
def : InstRW<[WriteShiftLd], (instregex "BT(16|32|64)mi8")>;

// BTR BTS BTC.
// r,r,i.
def ZnWriteBTRSC : SchedWriteRes<[ZnALU]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteBTRSC], (instregex "BT(R|S|C)(16|32|64)r(r|i8)")>;


// m,r,i.
def ZnWriteBTRSCm : SchedWriteRes<[ZnAGU, ZnALU]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
// m,r,i.
def : InstRW<[ZnWriteBTRSCm], (instregex "BT(R|S|C)(16|32|64)m(r|i8)")>;

// BLSI BLSMSK BLSR.
// r,r.
def : InstRW<[ZnWriteALULat2], (instregex "BLS(I|MSK|R)(32|64)rr")>;
// r,m.
def : InstRW<[ZnWriteALULat2Ld], (instregex "BLS(I|MSK|R)(32|64)rm")>;

// CLD STD.
def : InstRW<[WriteALU], (instregex "STD", "CLD")>;

// PDEP PEXT.
// r,r,r.
def : InstRW<[WriteMicrocoded], (instregex "PDEP(32|64)rr", "PEXT(32|64)rr")>;
// r,r,m.
def : InstRW<[WriteMicrocoded], (instregex "PDEP(32|64)rm", "PEXT(32|64)rm")>;

// ROR ROL.
def : InstRW<[WriteShift], (instregex "RO(R|L)(8|16|32|64)r1")>;

// RCR RCL.
// r,1.
def : InstRW<[WriteShift], (instregex "RC(R|L)(8|16|32|64)r1")>;

// m,1.
def : InstRW<[WriteMicrocoded], (instregex "RC(R|L)(8|16|32|64)m1")>;

// i.
def : InstRW<[WriteShift], (instregex "RC(R|L)(8|16|32|64)r(i|CL)")>;

// m,i.
def : InstRW<[WriteMicrocoded], (instregex "RC(R|L)(8|16|32|64)m(i|CL)")>;

// SHR SHL SAR.
// m,i.
def : InstRW<[WriteShiftLd], (instregex "S(A|H)(R|L)(8|16|32|64)m(i|1)")>;

// SHRD SHLD.
// r,r
def : InstRW<[WriteShift], (instregex "SH(R|L)D(16|32|64)rri8")>;

// m,r
def : InstRW<[WriteShiftLd], (instregex "SH(R|L)D(16|32|64)mri8")>;

// r,r,cl.
def : InstRW<[WriteMicrocoded], (instregex "SHLD(16|32|64)rrCL")>;

// r,r,cl.
def : InstRW<[WriteMicrocoded], (instregex "SHRD(16|32|64)rrCL")>;

// m,r,cl.
def : InstRW<[WriteMicrocoded], (instregex "SH(R|L)D(16|32|64)mrCL")>;

// SETcc.
// r.
def : InstRW<[WriteShift],
             (instregex "SET(O|NO|B|AE|E|NE|BE|A|S|NS|P|NP|L|GE|LE|G)r")>;
// m.
def : InstRW<[WriteShift],
             (instregex "SET(O|NO|B|AE|E|NE|BE|A|S|NS|P|NP|L|GE|LE|G)m")>;

//-- Misc instructions --//
// CMPXCHG.
def ZnWriteCMPXCHG : SchedWriteRes<[ZnAGU, ZnALU]> {
  let Latency = 8;
  let NumMicroOps = 5;
}
def : InstRW<[ZnWriteCMPXCHG], (instregex "CMPXCHG(8|16|32|64)rm")>;

// CMPXCHG8B.
def ZnWriteCMPXCHG8B : SchedWriteRes<[ZnAGU, ZnALU]> {
  let NumMicroOps = 18;
}
def : InstRW<[ZnWriteCMPXCHG8B], (instregex "CMPXCHG8B")>;

def : InstRW<[WriteMicrocoded], (instregex "CMPXCHG16B")>;

// LEAVE
def ZnWriteLEAVE : SchedWriteRes<[ZnALU, ZnAGU]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteLEAVE], (instregex "LEAVE")>;

// PAUSE.
def : InstRW<[WriteMicrocoded], (instregex "PAUSE")>;

// RDTSC.
def : InstRW<[WriteMicrocoded], (instregex "RDTSC")>;

// RDPMC.
def : InstRW<[WriteMicrocoded], (instregex "RDPMC")>;

// RDRAND.
def : InstRW<[WriteMicrocoded], (instregex "RDRAND(16|32|64)r")>;

// XGETBV.
def : InstRW<[WriteMicrocoded], (instregex "XGETBV")>;

//-- String instructions --//
// CMPS.
def : InstRW<[WriteMicrocoded], (instregex "CMPS(B|L|Q|W)")>;

// LODSB/W.
def : InstRW<[WriteMicrocoded], (instregex "LODS(B|W)")>;

// LODSD/Q.
def : InstRW<[WriteMicrocoded], (instregex "LODS(L|Q)")>;

// MOVS.
def : InstRW<[WriteMicrocoded], (instregex "MOVS(B|L|Q|W)")>;

// SCAS.
def : InstRW<[WriteMicrocoded], (instregex "SCAS(B|W|L|Q)")>;

// STOS
def : InstRW<[WriteMicrocoded], (instregex "STOS(B|L|Q|W)")>;

// XADD.
def : InstRW<[WriteMicrocoded], (instregex "XADD(8|16|32|64)rm")>;

//=== Floating Point x87 Instructions ===//
//-- Move instructions --//

def ZnWriteFLDr : SchedWriteRes<[ZnFPU13]> ;

def ZnWriteSTr: SchedWriteRes<[ZnFPU23]> {
  let Latency = 5;
  let NumMicroOps = 2;
}

// LD_F.
// r.
def : InstRW<[ZnWriteFLDr], (instregex "LD_Frr")>;

// m.
def ZnWriteLD_F80m : SchedWriteRes<[ZnAGU, ZnFPU13]> {
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteLD_F80m], (instregex "LD_F80m")>;

// FBLD.
def : InstRW<[WriteMicrocoded], (instregex "FBLDm")>;

// FST(P).
// r.
def : InstRW<[ZnWriteSTr], (instregex "ST_(F|FP)rr")>;

// m80.
def ZnWriteST_FP80m : SchedWriteRes<[ZnAGU, ZnFPU23]> {
  let Latency = 5;
}
def : InstRW<[ZnWriteST_FP80m], (instregex "ST_FP80m")>;

// FBSTP.
// m80.
def : InstRW<[WriteMicrocoded], (instregex "FBSTPm")>;

def ZnWriteFXCH : SchedWriteRes<[ZnFPU]>;

// FXCHG.
def : InstRW<[ZnWriteFXCH], (instregex "XCH_F")>;

// FILD.
def ZnWriteFILD : SchedWriteRes<[ZnAGU, ZnFPU3]> {
  let Latency = 11;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteFILD], (instregex "ILD_F(16|32|64)m")>;

// FIST(P) FISTTP.
def ZnWriteFIST : SchedWriteRes<[ZnAGU, ZnFPU23]> {
  let Latency = 12;
}
def : InstRW<[ZnWriteFIST], (instregex "IS(T|TT)_(F|FP)(16|32|64)m")>;

def ZnWriteFPU13 : SchedWriteRes<[ZnAGU, ZnFPU13]> {
  let Latency = 8;
}

def ZnWriteFPU3 : SchedWriteRes<[ZnAGU, ZnFPU3]> {
  let Latency = 11;
}

// FLDZ.
def : InstRW<[ZnWriteFPU13], (instregex "LD_F0")>;

// FLD1.
def : InstRW<[ZnWriteFPU3], (instregex "LD_F1")>;

// FLDPI FLDL2E etc.
def : InstRW<[ZnWriteFPU3], (instregex "FLDPI", "FLDL2(T|E)", "FLDL(G|N)2")>;

def : InstRW<[WriteMicrocoded], (instregex "CMOV(B|BE|E|P|NB|NBE|NE|NP)_F")>;

// FNSTSW.
// AX.
def : InstRW<[WriteMicrocoded], (instregex "FNSTSW16r")>;

// m16.
def : InstRW<[WriteMicrocoded], (instregex "FNSTSWm")>;

// FLDCW.
def : InstRW<[WriteMicrocoded], (instregex "FLDCW16m")>;

// FNSTCW.
def : InstRW<[WriteMicrocoded], (instregex "FNSTCW16m")>;

// FINCSTP FDECSTP.
def : InstRW<[ZnWriteFPU3], (instregex "FINCSTP", "FDECSTP")>;

// FFREE.
def : InstRW<[ZnWriteFPU3], (instregex "FFREE")>;

// FNSAVE.
def : InstRW<[WriteMicrocoded], (instregex "FSAVEm")>;

// FRSTOR.
def : InstRW<[WriteMicrocoded], (instregex "FRSTORm")>;

//-- Arithmetic instructions --//

def ZnWriteFPU3Lat2 : SchedWriteRes<[ZnFPU3]> {
  let Latency = 2;
}

def ZnWriteFPU3Lat1 : SchedWriteRes<[ZnFPU3]> ;

def ZnWriteFPU0Lat1 : SchedWriteRes<[ZnFPU0]> ;

def ZnWriteFPU0Lat1Ld : SchedWriteRes<[ZnAGU, ZnFPU0]> {
  let Latency = 8;
}

// FABS.
def : InstRW<[ZnWriteFPU3Lat2], (instregex "ABS_F")>;

// FCHS.
def : InstRW<[ZnWriteFPU3Lat1], (instregex "CHS_F")>;

// FCOM(P) FUCOM(P).
// r.
def : InstRW<[ZnWriteFPU0Lat1], (instregex "COM_FST0r", "COMP_FST0r", "UCOM_Fr",
                         "UCOM_FPr")>;
// m.
def : InstRW<[ZnWriteFPU0Lat1Ld], (instregex "FCOM(32|64)m", "FCOMP(32|64)m")>;

// FCOMPP FUCOMPP.
// r.
def : InstRW<[ZnWriteFPU0Lat1], (instregex "FCOMPP", "UCOM_FPPr")>;

def ZnWriteFPU02 : SchedWriteRes<[ZnAGU, ZnFPU02]>
{
  let Latency = 9;
}

// FCOMI(P) FUCOMI(P).
// m.
def : InstRW<[ZnWriteFPU02], (instregex "COM_FIr", "COM_FIPr", "UCOM_FIr",
                           "UCOM_FIPr")>;

def ZnWriteFPU03 : SchedWriteRes<[ZnAGU, ZnFPU03]>
{
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,3];
}

// FICOM(P).
def : InstRW<[ZnWriteFPU03], (instregex "FICOM(16|32)m", "FICOMP(16|32)m")>;

// FTST.
def : InstRW<[ZnWriteFPU0Lat1], (instregex "TST_F")>;

// FXAM.
def : InstRW<[ZnWriteFPU3Lat1], (instregex "FXAM")>;

// FPREM.
def : InstRW<[WriteMicrocoded], (instrs FPREM)>;

// FPREM1.
def : InstRW<[WriteMicrocoded], (instrs FPREM1)>;

// FRNDINT.
def : InstRW<[WriteMicrocoded], (instregex "FRNDINT")>;

// FSCALE.
def : InstRW<[WriteMicrocoded], (instregex "FSCALE")>;

// FXTRACT.
def : InstRW<[WriteMicrocoded], (instregex "FXTRACT")>;

// FNOP.
def : InstRW<[ZnWriteFPU0Lat1], (instregex "FNOP")>;

// WAIT.
def : InstRW<[ZnWriteFPU0Lat1], (instregex "WAIT")>;

// FNCLEX.
def : InstRW<[WriteMicrocoded], (instregex "FNCLEX")>;

// FNINIT.
def : InstRW<[WriteMicrocoded], (instregex "FNINIT")>;

//=== Integer MMX and XMM Instructions ===//
//-- Move instructions --//

// Moves from GPR to FPR incurs a penalty
def ZnWriteFPU2 : SchedWriteRes<[ZnFPU2]> {
  let Latency = 3;
}

// Move to ALU doesn't incur penalty
def ZnWriteToALU2 : SchedWriteRes<[ZnFPU2]> {
  let Latency = 2;
}

def ZnWriteFPU : SchedWriteRes<[ZnFPU]>;
def ZnWriteFPUY : SchedWriteRes<[ZnFPU]> {
  let NumMicroOps = 2;
  let Latency=2;
}

// MOVD.
// r32/64 <- (x)mm.
def : InstRW<[ZnWriteToALU2], (instregex "MMX_MOVD64grr", "MMX_MOVD64from64rr",
                         "VMOVPDI2DIrr", "MOVPDI2DIrr")>;

// (x)mm <- r32/64.
def : InstRW<[ZnWriteFPU2], (instregex "MMX_MOVD64rr", "MMX_MOVD64to64rr",
                         "VMOVDI2PDIrr", "MOVDI2PDIrr")>;

// MOVQ.
// r64 <- (x)mm.
def : InstRW<[ZnWriteToALU2], (instregex "VMOVPQIto64rr")>;

// (x)mm <- r64.
def : InstRW<[ZnWriteFPU2], (instregex "VMOV64toPQIrr")>;

// (x)mm <- (x)mm.
def : InstRW<[ZnWriteFPU], (instregex "MMX_MOVQ64rr")>;

// (V)MOVDQA/U.
// x <- x.
def : InstRW<[ZnWriteFPU], (instregex "MOVDQ(A|U)rr", "VMOVDQ(A|U)rr")>;

// y <- y.
def : InstRW<[ZnWriteFPUY], (instregex "VMOVDQ(A|U)Yrr")>;

// MOVDQ2Q.
def : InstRW<[ZnWriteFPU], (instregex "MMX_MOVDQ2Qrr")>;

// MOVQ2DQ.
def : InstRW<[ZnWriteFPU], (instregex "MMX_MOVQ2DQrr")>;

// PACKSSWB/DW.
// mm <- mm.
def ZnWriteFPU12 : SchedWriteRes<[ZnFPU12]> ;
def ZnWriteFPU12Y : SchedWriteRes<[ZnFPU12]> {
  let NumMicroOps = 2;
}
def ZnWriteFPU12m : SchedWriteRes<[ZnAGU, ZnFPU12]> ;

def : InstRW<[ZnWriteFPU12], (instregex "MMX_PACKSSDWirr",
                                  "MMX_PACKSSWBirr", "MMX_PACKUSWBirr")>;
def : InstRW<[ZnWriteFPU12m], (instregex "MMX_PACKSSDWirm",
                                  "MMX_PACKSSWBirm", "MMX_PACKUSWBirm")>;

// VPMOVSX/ZX BW BD BQ DW DQ.
// y <- x.
def : InstRW<[ZnWriteFPU12Y], (instregex "VPMOV(SX|ZX)(BW|BQ|DW|DQ)Yrr")>;

def ZnWriteFPU013 : SchedWriteRes<[ZnFPU013]> ;
def ZnWriteFPU013Y : SchedWriteRes<[ZnFPU013]> {
  let Latency = 2;
}
def ZnWriteFPU013m : SchedWriteRes<[ZnAGU, ZnFPU013]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def ZnWriteFPU013Ld : SchedWriteRes<[ZnAGU, ZnFPU013]> {
  let Latency = 8;
  let NumMicroOps = 2;
}
def ZnWriteFPU013LdY : SchedWriteRes<[ZnAGU, ZnFPU013]> {
  let Latency = 9;
  let NumMicroOps = 2;
}

// PBLENDW.
// x,x,i / v,v,v,i
def : InstRW<[ZnWriteFPU013], (instregex "(V?)PBLENDWrri")>;
// ymm
def : InstRW<[ZnWriteFPU013Y], (instregex "(V?)PBLENDWYrri")>;

// x,m,i / v,v,m,i
def : InstRW<[ZnWriteFPU013Ld], (instregex "(V?)PBLENDWrmi")>;
// y,m,i
def : InstRW<[ZnWriteFPU013LdY], (instregex "(V?)PBLENDWYrmi")>;

def ZnWriteFPU01 : SchedWriteRes<[ZnFPU01]> ;
def ZnWriteFPU01Y : SchedWriteRes<[ZnFPU01]> {
  let NumMicroOps = 2;
}

// VPBLENDD.
// v,v,v,i.
def : InstRW<[ZnWriteFPU01], (instregex "VPBLENDDrri")>;
// ymm
def : InstRW<[ZnWriteFPU01Y], (instregex "VPBLENDDYrri")>;

// v,v,m,i
def ZnWriteFPU01Op2 : SchedWriteRes<[ZnAGU, ZnFPU01]> {
  let NumMicroOps = 2;
  let Latency = 8;
  let ResourceCycles = [1, 2];
}
def ZnWriteFPU01Op2Y : SchedWriteRes<[ZnAGU, ZnFPU01]> {
  let NumMicroOps = 2;
  let Latency = 9;
  let ResourceCycles = [1, 3];
}
def : InstRW<[ZnWriteFPU01Op2], (instregex "VPBLENDDrmi")>;
def : InstRW<[ZnWriteFPU01Op2Y], (instregex "VPBLENDDYrmi")>;

// MASKMOVQ.
def : InstRW<[WriteMicrocoded], (instregex "MMX_MASKMOVQ(64)?")>;

// MASKMOVDQU.
def : InstRW<[WriteMicrocoded], (instregex "(V?)MASKMOVDQU(64)?")>;

// VPMASKMOVQ.
// ymm
def : InstRW<[ZnWriteFPU01Op2],(instregex "VPMASKMOVQrm")>;
def : InstRW<[ZnWriteFPU01Op2Y],(instregex "VPMASKMOVQYrm")>;

def : InstRW<[WriteMicrocoded],
                               (instregex "VPMASKMOVD(Y?)rm")>;
// m, v,v.
def : InstRW<[WriteMicrocoded], (instregex "VPMASKMOV(D|Q)(Y?)mr")>;

// PMOVMSKBY.
def ZnWritePMOVMSKBY : SchedWriteRes<[ZnFPU2]> {
  let NumMicroOps = 2;
  let Latency = 2;
  let ResourceCycles = [2];
}
def : InstRW<[ZnWritePMOVMSKBY], (instregex "(V|MMX_)?PMOVMSKBYrr")>;

// PEXTR B/W/D/Q.
// r32,x,i.
def ZnWritePEXTRr : SchedWriteRes<[ZnFPU12, ZnFPU2]> {
  let Latency = 2;
  let ResourceCycles = [1, 2];
}
def : InstRW<[ZnWritePEXTRr], (instregex "(V?)PEXTR(B|W|D|Q)rr", "MMX_PEXTRWrr")>;

def ZnWritePEXTRm : SchedWriteRes<[ZnAGU, ZnFPU12, ZnFPU2]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2, 3];
}
// m8,x,i.
def : InstRW<[ZnWritePEXTRm], (instregex "(V?)PEXTR(B|W|D|Q)mr")>;

// VPBROADCAST B/W.
// x, m8/16.
def ZnWriteVPBROADCAST128Ld : SchedWriteRes<[ZnAGU, ZnFPU12]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2];
}
def : InstRW<[ZnWriteVPBROADCAST128Ld],
                                     (instregex "VPBROADCAST(B|W)rm")>;

// y, m8/16
def ZnWriteVPBROADCAST256Ld : SchedWriteRes<[ZnAGU, ZnFPU1]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2];
}
def : InstRW<[ZnWriteVPBROADCAST256Ld],
                                     (instregex "VPBROADCAST(B|W)Yrm")>;

// VPGATHER.
def : InstRW<[WriteMicrocoded], (instregex "VPGATHER(Q|D)(Q|D)(Y?)rm")>;

//-- Arithmetic instructions --//

// HADD, HSUB PS/PD
// PHADD|PHSUB (S) W/D.
def : InstRW<[WriteMicrocoded], (instregex "MMX_PHADD(W|D)r(r|m)",
                               "MMX_PHADDSWr(r|m)",
                               "MMX_PHSUB(W|D)r(r|m)",
                               "MMX_PHSUBSWrr",
                               "(V?)PH(ADD|SUB)(W|D)(Y?)r(r|m)",
                               "(V?)PH(ADD|SUB)SW(Y?)r(r|m)")>;


// PCMPGTQ.
def ZnWritePCMPGTQr : SchedWriteRes<[ZnFPU03]>;
def : InstRW<[ZnWritePCMPGTQr], (instregex "(V?)PCMPGTQ(Y?)rr")>;

// x <- x,m.
def ZnWritePCMPGTQm : SchedWriteRes<[ZnAGU, ZnFPU03]> {
  let Latency = 8;
}
// ymm.
def ZnWritePCMPGTQYm : SchedWriteRes<[ZnAGU, ZnFPU03]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,2];
}
def : InstRW<[ZnWritePCMPGTQm], (instregex "(V?)PCMPGTQrm")>;
def : InstRW<[ZnWritePCMPGTQYm], (instregex "(V?)PCMPGTQYrm")>;

// PMULLD.
// x,x.
def ZnWritePMULLDr : SchedWriteRes<[ZnFPU0]> {
  let Latency = 4;
}
// ymm.
def ZnWritePMULLDYr : SchedWriteRes<[ZnFPU0]> {
  let Latency = 5;
  let ResourceCycles = [2];
}
def : InstRW<[ZnWritePMULLDr], (instregex "(V?)PMULLDrr")>;
def : InstRW<[ZnWritePMULLDYr], (instregex "(V?)PMULLDYrr")>;

// x,m.
def ZnWritePMULLDm : SchedWriteRes<[ZnAGU, ZnFPU0]> {
  let Latency = 11;
  let NumMicroOps = 2;
}
// y,m.
def ZnWritePMULLDYm : SchedWriteRes<[ZnAGU, ZnFPU0]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2];
}
def : InstRW<[ZnWritePMULLDm], (instregex "(V?)PMULLDrm")>;
def : InstRW<[ZnWritePMULLDYm], (instregex "(V?)PMULLDYrm")>;

//-- Logic instructions --//

// PTEST.
// v,v.
def ZnWritePTESTr : SchedWriteRes<[ZnFPU12]> {
  let ResourceCycles = [2];
}
def : InstRW<[ZnWritePTESTr], (instregex "(V?)PTEST(Y?)rr")>;

// v,m.
def ZnWritePTESTm : SchedWriteRes<[ZnAGU, ZnFPU12]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2];
}
def : InstRW<[ZnWritePTESTm], (instregex "(V?)PTEST(Y?)rm")>;

// PSLL,PSRL,PSRA W/D/Q.
// x,x / v,v,x.
def ZnWritePShift  : SchedWriteRes<[ZnFPU2]> ;
def ZnWritePShiftY : SchedWriteRes<[ZnFPU2]> {
  let Latency = 2;
}
def ZnWritePShiftLd  : SchedWriteRes<[ZnAGU,ZnFPU2]> {
  let Latency = 8;
}
def ZnWritePShiftYLd : SchedWriteRes<[ZnAGU, ZnFPU2]> {
  let Latency = 9;
}
def : InstRW<[ZnWritePShift], (instregex "(V?)PS(LL|RL|RA)(W|D|Q)rr")>;
def : InstRW<[ZnWritePShiftY], (instregex "(V?)PS(LL|RL|RA)(W|D|Q)Yrr")>;

def : InstRW<[ZnWritePShiftLd], (instregex "(V?)PS(LL|RL|RA)(W|D|Q)rm")>;
def : InstRW<[ZnWritePShiftYLd], (instregex "(V?)PS(LL|RL|RA)(W|D|Q)Yrm")>;

// PSLL,PSRL DQ.
def : InstRW<[ZnWritePShift], (instregex "(V?)PS(R|L)LDQri")>;
def : InstRW<[ZnWritePShiftY], (instregex "(V?)PS(R|L)LDQYri")>;

//=== Floating Point XMM and YMM Instructions ===//
//-- Move instructions --//

// VPERM2F128.
def : InstRW<[WriteMicrocoded], (instregex "VPERM2F128rr")>;
def : InstRW<[WriteMicrocoded], (instregex "VPERM2F128rm")>;

// BLENDVP S/D.
def ZnWriteFPU01Lat3 : SchedWriteRes<[ZnFPU013]> {
  let Latency = 3;
}
def ZnWriteFPU01Lat3Ld : SchedWriteRes<[ZnAGU, ZnFPU013]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2];
}
def : InstRW<[ZnWriteFPU01Lat3], (instregex "BLENDVP(S|D)rr0")>;
def : InstRW<[ZnWriteFPU01Lat3Ld, ReadAfterLd], (instregex "BLENDVP(S|D)rm0")>;

def ZnWriteBROADCAST : SchedWriteRes<[ZnAGU, ZnFPU13]> {
  let NumMicroOps = 2;
  let Latency = 8;
}
// VBROADCASTF128.
def : InstRW<[ZnWriteBROADCAST], (instregex "VBROADCASTF128")>;

// EXTRACTPS.
// r32,x,i.
def ZnWriteEXTRACTPSr : SchedWriteRes<[ZnFPU12, ZnFPU2]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2];
}
def : InstRW<[ZnWriteEXTRACTPSr], (instregex "(V?)EXTRACTPSrr")>;

def ZnWriteEXTRACTPSm : SchedWriteRes<[ZnAGU,ZnFPU12, ZnFPU2]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [5, 1, 2];
}
// m32,x,i.
def : InstRW<[ZnWriteEXTRACTPSm], (instregex "(V?)EXTRACTPSmr")>;

// VEXTRACTF128.
// x,y,i.
def : InstRW<[ZnWriteFPU013], (instregex "VEXTRACTF128rr")>;

// m128,y,i.
def : InstRW<[ZnWriteFPU013m], (instregex "VEXTRACTF128mr")>;

def ZnWriteVINSERT128r: SchedWriteRes<[ZnFPU013]> {
  let Latency = 2;
  let ResourceCycles = [2];
}
def ZnWriteVINSERT128Ld: SchedWriteRes<[ZnAGU,ZnFPU013]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2];
}
// VINSERTF128.
// y,y,x,i.
def : InstRW<[ZnWriteVINSERT128r], (instregex "VINSERTF128rr")>;
def : InstRW<[ZnWriteVINSERT128Ld], (instregex "VINSERTF128rm")>;

// VMASKMOVP S/D.
// x,x,m.
def ZnWriteVMASKMOVPLd : SchedWriteRes<[ZnAGU, ZnFPU01]> {
  let Latency = 8;
}
// y,y,m.
def ZnWriteVMASKMOVPLdY : SchedWriteRes<[ZnAGU, ZnFPU01]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2];
}
def ZnWriteVMASKMOVPm : SchedWriteRes<[ZnAGU, ZnFPU01]> {
  let Latency = 4;
}
def : InstRW<[ZnWriteVMASKMOVPLd], (instregex "VMASKMOVP(S|D)rm")>;
def : InstRW<[ZnWriteVMASKMOVPLdY], (instregex "VMASKMOVP(S|D)Yrm")>;
def : InstRW<[ZnWriteVMASKMOVPm], (instregex "VMASKMOVP(S|D)mr")>;

// m256,y,y.
def ZnWriteVMASKMOVPYmr : SchedWriteRes<[ZnAGU,ZnFPU01]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2];
}
def : InstRW<[ZnWriteVMASKMOVPYmr], (instregex "VMASKMOVP(S|D)Ymr")>;

// VGATHERDPS.
// x.
def : InstRW<[WriteMicrocoded], (instregex "VGATHERDPSrm")>;
// y.
def : InstRW<[WriteMicrocoded], (instregex "VGATHERDPSYrm")>;

// VGATHERQPS.
// x.
def : InstRW<[WriteMicrocoded], (instregex "VGATHERQPSrm")>;

// y.
def : InstRW<[WriteMicrocoded], (instregex "VGATHERQPSYrm")>;

// VGATHERDPD.
// x.
def : InstRW<[WriteMicrocoded], (instregex "VGATHERDPDrm")>;

// y.
def : InstRW<[WriteMicrocoded], (instregex "VGATHERDPDYrm")>;

// VGATHERQPD.
// x.
def : InstRW<[WriteMicrocoded], (instregex "VGATHERQPDrm")>;

// y.
def : InstRW<[WriteMicrocoded], (instregex "VGATHERQPDYrm")>;

//-- Conversion instructions --//
def ZnWriteCVTPD2PSr: SchedWriteRes<[ZnFPU3]> {
  let Latency = 4;
}
// CVTPD2PS.
// x,x.
def : InstRW<[ZnWriteCVTPD2PSr], (instregex "(V?)CVTPD2PSrr")>;

def ZnWriteCVTPD2PSLd: SchedWriteRes<[ZnAGU,ZnFPU03]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,2];
}
// x,m128.
def : InstRW<[ZnWriteCVTPD2PSLd], (instregex "(V?)CVTPD2PS(X?)rm")>;

// x,y.
def ZnWriteCVTPD2PSYr : SchedWriteRes<[ZnFPU3]> {
  let Latency = 5;
}
def : InstRW<[ZnWriteCVTPD2PSYr], (instregex "(V?)CVTPD2PSYrr")>;

// x,m256.
def ZnWriteCVTPD2PSYLd : SchedWriteRes<[ZnAGU, ZnFPU3]> {
  let Latency = 11;
}
def : InstRW<[ZnWriteCVTPD2PSYLd], (instregex "(V?)CVTPD2PSYrm")>;

// CVTSD2SS.
// x,x.
// Same as WriteCVTPD2PSr
def : InstRW<[ZnWriteCVTPD2PSr], (instregex "(V)?CVTSD2SSrr")>;

// x,m64.
def : InstRW<[ZnWriteCVTPD2PSLd], (instregex "(V)?CVTSD2SSrm")>;

// CVTPS2PD.
// x,x.
def ZnWriteCVTPS2PDr : SchedWriteRes<[ZnFPU3]> {
  let Latency = 3;
}
def : InstRW<[ZnWriteCVTPS2PDr], (instregex "(V?)CVTPS2PDrr")>;

// x,m64.
// y,m128.
def ZnWriteCVTPS2PDLd : SchedWriteRes<[ZnAGU, ZnFPU3]> {
  let Latency = 10;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteCVTPS2PDLd], (instregex "(V?)CVTPS2PD(Y?)rm")>;

// y,x.
def ZnWriteVCVTPS2PDY : SchedWriteRes<[ZnFPU3]> {
  let Latency = 3;
}
def : InstRW<[ZnWriteVCVTPS2PDY], (instregex "VCVTPS2PDYrr")>;

// CVTSS2SD.
// x,x.
def ZnWriteCVTSS2SDr : SchedWriteRes<[ZnFPU3]> {
  let Latency = 4;
}
def : InstRW<[ZnWriteCVTSS2SDr], (instregex "(V?)CVTSS2SDrr")>;

// x,m32.
def ZnWriteCVTSS2SDLd : SchedWriteRes<[ZnAGU, ZnFPU3]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 2];
}
def : InstRW<[ZnWriteCVTSS2SDLd], (instregex "(V?)CVTSS2SDrm")>;

def ZnWriteCVTDQ2PDr: SchedWriteRes<[ZnFPU12,ZnFPU3]> {
  let Latency = 5;
}
// CVTDQ2PD.
// x,x.
def : InstRW<[ZnWriteCVTDQ2PDr], (instregex "(V)?CVTDQ2PDrr")>;

// Same as xmm
// y,x.
def : InstRW<[ZnWriteCVTDQ2PDr], (instregex "VCVTDQ2PDYrr")>;

def ZnWriteCVTPD2DQr: SchedWriteRes<[ZnFPU12, ZnFPU3]> {
  let Latency = 5;
}
// CVT(T)PD2DQ.
// x,x.
def : InstRW<[ZnWriteCVTDQ2PDr], (instregex "(V?)CVT(T?)PD2DQrr")>;

def ZnWriteCVTPD2DQLd: SchedWriteRes<[ZnAGU,ZnFPU12,ZnFPU3]> {
  let Latency = 12;
  let NumMicroOps = 2;
}
// x,m128.
def : InstRW<[ZnWriteCVTPD2DQLd], (instregex "(V?)CVT(T?)PD2DQrm")>;
// same as xmm handling
// x,y.
def : InstRW<[ZnWriteCVTPD2DQr], (instregex "VCVT(T?)PD2DQYrr")>;
// x,m256.
def : InstRW<[ZnWriteCVTPD2DQLd], (instregex "VCVT(T?)PD2DQYrm")>;

def ZnWriteCVTPS2PIr: SchedWriteRes<[ZnFPU3]> {
  let Latency = 4;
}
// CVT(T)PS2PI.
// mm,x.
def : InstRW<[ZnWriteCVTPS2PIr], (instregex "MMX_CVT(T?)PS2PIirr")>;

// CVTPI2PD.
// x,mm.
def : InstRW<[ZnWriteCVTPS2PDr], (instregex "MMX_CVT(T?)PI2PDirr")>;

// CVT(T)PD2PI.
// mm,x.
def : InstRW<[ZnWriteCVTPS2PIr], (instregex "MMX_CVT(T?)PD2PIirr")>;

def ZnWriteCVSTSI2SSr: SchedWriteRes<[ZnFPU3]> {
  let Latency = 5;
}
// CVSTSI2SS.
// x,r32.
def : InstRW<[ZnWriteCVSTSI2SSr], (instregex "(V?)CVTSI2SS(64)?rr")>;

// same as CVTPD2DQr
// CVT(T)SS2SI.
// r32,x.
def : InstRW<[ZnWriteCVTPD2DQr], (instregex "(V?)CVT(T?)SS2SI(64)?rr")>;
// same as CVTPD2DQm
// r32,m32.
def : InstRW<[ZnWriteCVTPD2DQLd], (instregex "(V?)CVT(T?)SS2SI(64)?rm")>;

def ZnWriteCVSTSI2SDr: SchedWriteRes<[ZnFPU013, ZnFPU3]> {
  let Latency = 5;
}
// CVTSI2SD.
// x,r32/64.
def : InstRW<[ZnWriteCVSTSI2SDr], (instregex "(V?)CVTSI2SD(64)?rr")>;


def ZnWriteCVSTSI2SIr: SchedWriteRes<[ZnFPU3, ZnFPU2]> {
  let Latency = 5;
}
def ZnWriteCVSTSI2SILd: SchedWriteRes<[ZnAGU, ZnFPU3, ZnFPU2]> {
  let Latency = 12;
}
// CVTSD2SI.
// r32/64
def : InstRW<[ZnWriteCVSTSI2SIr], (instregex "(V?)CVT(T?)SD2SI(64)?rr")>;
// r32,m32.
def : InstRW<[ZnWriteCVSTSI2SILd], (instregex "(V?)CVT(T?)SD2SI(64)?rm")>;


// VCVTPS2PH.
// x,v,i.
def : InstRW<[WriteMicrocoded], (instregex "VCVTPS2PH(Y?)rr")>;
// m,v,i.
def : InstRW<[WriteMicrocoded], (instregex "VCVTPS2PH(Y?)mr")>;

// VCVTPH2PS.
// v,x.
def : InstRW<[WriteMicrocoded], (instregex "VCVTPH2PS(Y?)rr")>;
// v,m.
def : InstRW<[WriteMicrocoded], (instregex "VCVTPH2PS(Y?)rm")>;

//-- SSE4A instructions --//
// EXTRQ
def ZnWriteEXTRQ: SchedWriteRes<[ZnFPU12, ZnFPU2]> {
  let Latency = 2;
}
def : InstRW<[ZnWriteEXTRQ], (instregex "EXTRQ")>;

// INSERTQ
def ZnWriteINSERTQ: SchedWriteRes<[ZnFPU03,ZnFPU1]> {
  let Latency = 4;
}
def : InstRW<[ZnWriteINSERTQ], (instregex "INSERTQ")>;

// MOVNTSS/MOVNTSD
def ZnWriteMOVNT: SchedWriteRes<[ZnAGU,ZnFPU2]> {
  let Latency = 8;
}
def : InstRW<[ZnWriteMOVNT], (instregex "MOVNTS(S|D)")>;

//-- SHA instructions --//
// SHA256MSG2
def : InstRW<[WriteMicrocoded], (instregex "SHA256MSG2(Y?)r(r|m)")>;

// SHA1MSG1, SHA256MSG1
// x,x.
def ZnWriteSHA1MSG1r : SchedWriteRes<[ZnFPU12]> {
  let Latency = 2;
  let ResourceCycles = [2];
}
def : InstRW<[ZnWriteSHA1MSG1r], (instregex "SHA(1|256)MSG1rr")>;
// x,m.
def ZnWriteSHA1MSG1Ld : SchedWriteRes<[ZnAGU, ZnFPU12]> {
  let Latency = 9;
  let ResourceCycles = [1,2];
}
def : InstRW<[ZnWriteSHA1MSG1Ld], (instregex "SHA(1|256)MSG1rm")>;

// SHA1MSG2
// x,x.
def ZnWriteSHA1MSG2r : SchedWriteRes<[ZnFPU12]> ;
def : InstRW<[ZnWriteSHA1MSG2r], (instregex "SHA1MSG2rr")>;
// x,m.
def ZnWriteSHA1MSG2Ld : SchedWriteRes<[ZnAGU, ZnFPU12]> {
  let Latency = 8;
}
def : InstRW<[ZnWriteSHA1MSG2Ld], (instregex "SHA1MSG2rm")>;

// SHA1NEXTE
// x,x.
def ZnWriteSHA1NEXTEr : SchedWriteRes<[ZnFPU1]> ;
def : InstRW<[ZnWriteSHA1NEXTEr], (instregex "SHA1NEXTErr")>;
// x,m.
def ZnWriteSHA1NEXTELd : SchedWriteRes<[ZnAGU, ZnFPU1]> {
  let Latency = 8;
}
def : InstRW<[ZnWriteSHA1NEXTELd], (instregex "SHA1NEXTErm")>;

// SHA1RNDS4
// x,x.
def ZnWriteSHA1RNDS4r : SchedWriteRes<[ZnFPU1]> {
  let Latency = 6;
}
def : InstRW<[ZnWriteSHA1RNDS4r], (instregex "SHA1RNDS4rr")>;
// x,m.
def ZnWriteSHA1RNDS4Ld : SchedWriteRes<[ZnAGU, ZnFPU1]> {
  let Latency = 13;
}
def : InstRW<[ZnWriteSHA1RNDS4Ld], (instregex "SHA1RNDS4rm")>;

// SHA256RNDS2
// x,x.
def ZnWriteSHA256RNDS2r : SchedWriteRes<[ZnFPU1]> {
  let Latency = 4;
}
def : InstRW<[ZnWriteSHA256RNDS2r], (instregex "SHA256RNDS2rr")>;
// x,m.
def ZnWriteSHA256RNDS2Ld : SchedWriteRes<[ZnAGU, ZnFPU1]> {
  let Latency = 11;
}
def : InstRW<[ZnWriteSHA256RNDS2Ld], (instregex "SHA256RNDS2rm")>;

//-- Arithmetic instructions --//

// HADD, HSUB PS/PD
def : InstRW<[WriteMicrocoded], (instregex "(V?)H(ADD|SUB)P(S|D)(Y?)r(r|m)")>;

// MULL SS/SD PS/PD.
// x,x / v,v,v.
def ZnWriteMULr : SchedWriteRes<[ZnFPU01]> {
  let Latency = 3;
}
// ymm.
def ZnWriteMULYr : SchedWriteRes<[ZnFPU01]> {
  let Latency = 4;
}
def : InstRW<[ZnWriteMULr], (instregex "(V?)MUL(P|S)(S|D)rr")>;
def : InstRW<[ZnWriteMULYr], (instregex "(V?)MUL(P|S)(S|D)Yrr")>;

// x,m / v,v,m.
def ZnWriteMULLd : SchedWriteRes<[ZnAGU, ZnFPU01]> {
  let Latency = 10;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteMULLd], (instregex "(V?)MUL(P|S)(S|D)rm")>;

// ymm
def ZnWriteMULYLd : SchedWriteRes<[ZnAGU, ZnFPU01]> {
  let Latency = 11;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteMULYLd], (instregex "(V?)MUL(P|S)(S|D)Yrm")>;

// VDIVPS.
// y,y,y.
def ZnWriteVDIVPSYr : SchedWriteRes<[ZnFPU3]> {
  let Latency = 12;
  let ResourceCycles = [12];
}
def : InstRW<[ZnWriteVDIVPSYr], (instregex "VDIVPSYrr")>;

// y,y,m256.
def ZnWriteVDIVPSYLd : SchedWriteRes<[ZnAGU, ZnFPU3]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 19];
}
def : InstRW<[ZnWriteVDIVPSYLd], (instregex "VDIVPSYrm")>;

// VDIVPD.
// y,y,y.
def ZnWriteVDIVPDY : SchedWriteRes<[ZnFPU3]> {
  let Latency = 15;
  let ResourceCycles = [15];
}
def : InstRW<[ZnWriteVDIVPDY], (instregex "VDIVPDYrr")>;

// y,y,m256.
def ZnWriteVDIVPDYLd : SchedWriteRes<[ZnAGU, ZnFPU3]> {
  let Latency = 22;
  let NumMicroOps = 2;
  let ResourceCycles = [1,22];
}
def : InstRW<[ZnWriteVDIVPDYLd], (instregex "VDIVPDYrm")>;

// VRCPPS.
// y,y.
def ZnWriteVRCPPSr : SchedWriteRes<[ZnFPU01]> {
  let Latency = 5;
}
def : InstRW<[ZnWriteVRCPPSr], (instregex "VRCPPSYr")>;

// y,m256.
def ZnWriteVRCPPSLd : SchedWriteRes<[ZnAGU, ZnFPU01]> {
  let Latency = 12;
  let NumMicroOps = 3;
}
def : InstRW<[ZnWriteVRCPPSLd], (instregex "VRCPPSYm")>;

// ROUND SS/SD PS/PD.
// v,v,i.
def ZnWriteROUNDr : SchedWriteRes<[ZnFPU3]> {
  let Latency = 4;
}
def : InstRW<[ZnWriteROUNDr], (instregex "(V?)ROUND(S|P)(S|D)(Y?)r")>;

// v,m,i.
def ZnWriteROUNDm : SchedWriteRes<[ZnAGU, ZnFPU3]> {
  let Latency = 11;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteROUNDm], (instregex "(V?)ROUND(S|P)(S|D)(Y?)m")>;

// DPPS.
// x,x,i / v,v,v,i.
def : InstRW<[WriteMicrocoded], (instregex "(V?)DPPS(Y?)rri")>;

// x,m,i / v,v,m,i.
def : InstRW<[WriteMicrocoded], (instregex "(V?)DPPS(Y?)rmi")>;

// DPPD.
// x,x,i.
def : InstRW<[WriteMicrocoded], (instregex "(V?)DPPDrri")>;

// x,m,i.
def : InstRW<[WriteMicrocoded], (instregex "(V?)DPPDrmi")>;

// VSQRTPS.
// y,y.
def ZnWriteVSQRTPSYr : SchedWriteRes<[ZnFPU3]> {
  let Latency = 28;
  let ResourceCycles = [28];
}
def : InstRW<[ZnWriteVSQRTPSYr], (instregex "VSQRTPSYr")>;

// y,m256.
def ZnWriteVSQRTPSYLd : SchedWriteRes<[ZnAGU, ZnFPU3]> {
  let Latency = 35;
  let ResourceCycles = [1,35];
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteVSQRTPSYLd], (instregex "VSQRTPSYm")>;

// VSQRTPD.
// y,y.
def ZnWriteVSQRTPDYr : SchedWriteRes<[ZnFPU3]> {
  let Latency = 40;
  let ResourceCycles = [40];
}
def : InstRW<[ZnWriteVSQRTPDYr], (instregex "VSQRTPDYr")>;

// y,m256.
def ZnWriteVSQRTPDYLd : SchedWriteRes<[ZnAGU, ZnFPU3]> {
  let Latency = 47;
  let NumMicroOps = 2;
  let ResourceCycles = [1,47];
}
def : InstRW<[ZnWriteVSQRTPDYLd], (instregex "VSQRTPDYm")>;

// RSQRTSS
// x,x.
def ZnWriteRSQRTSSr : SchedWriteRes<[ZnFPU02]> {
  let Latency = 5;
}
def : InstRW<[ZnWriteRSQRTSSr], (instregex "(V?)RSQRTSS(Y?)r")>;

// RSQRTPS
// x,x.
def ZnWriteRSQRTPSr : SchedWriteRes<[ZnFPU01]> {
  let Latency = 5;
}
def : InstRW<[ZnWriteRSQRTPSr], (instregex "(V?)RSQRTPSr")>;

// RSQRTSSm
// x,m128.
def ZnWriteRSQRTSSLd: SchedWriteRes<[ZnAGU, ZnFPU02]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,2];
}
def : InstRW<[ZnWriteRSQRTSSLd], (instregex "(V?)RSQRTSSm")>;

// RSQRTPSm
def ZnWriteRSQRTPSLd : SchedWriteRes<[ZnAGU, ZnFPU01]> {
  let Latency = 12;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteRSQRTPSLd], (instregex "(V?)RSQRTPSm")>;

// RSQRTPS 256.
// y,y.
def ZnWriteRSQRTPSYr : SchedWriteRes<[ZnFPU01]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : InstRW<[ZnWriteRSQRTPSYr], (instregex "VRSQRTPSYr")>;

// y,m256.
def ZnWriteRSQRTPSYLd : SchedWriteRes<[ZnAGU, ZnFPU01]> {
  let Latency = 12;
  let NumMicroOps = 2;
}
def : InstRW<[ZnWriteRSQRTPSYLd], (instregex "VRSQRTPSYm")>;

//-- Logic instructions --//

// AND, ANDN, OR, XOR PS/PD.
// x,x / v,v,v.
def : InstRW<[WriteVecLogic], (instregex "(V?)(AND|ANDN|OR|XOR)P(S|D)(Y?)rr")>;
// x,m / v,v,m.
def : InstRW<[WriteVecLogicLd],
                         (instregex "(V?)(AND|ANDN|OR|XOR)P(S|D)(Y?)rm")>;

//-- Other instructions --//

// VZEROUPPER.
def : InstRW<[WriteMicrocoded], (instregex "VZEROUPPER")>;

// VZEROALL.
def : InstRW<[WriteMicrocoded], (instregex "VZEROALL")>;

// LDMXCSR.
def : InstRW<[WriteMicrocoded], (instregex "(V)?LDMXCSR")>;

// STMXCSR.
def : InstRW<[WriteMicrocoded], (instregex "(V)?STMXCSR")>;

} // SchedModel
