//=- X86SchedHaswell.td - X86 Haswell Scheduling -------------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Haswell to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def HaswellModel : SchedMachineModel {
  // All x86 instructions are modeled as a single micro-op, and HW can decode 4
  // instructions per cycle.
  let IssueWidth = 4;
  let MicroOpBufferSize = 192; // Based on the reorder buffer.
  let LoadLatency = 4;
  let MispredictPenalty = 16;

  // Based on the LSD (loop-stream detector) queue size and benchmarking data.
  let LoopMicroOpBufferSize = 50;

  // FIXME: SSE4 and AVX are unimplemented. This flag is set to allow
  // the scheduler to assign a default model to unrecognized opcodes.
  let CompleteModel = 0;
}

let SchedModel = HaswellModel in {

// Haswell can issue micro-ops to 8 different ports in one cycle.

// Ports 0, 1, 5, and 6 handle all computation.
// Port 4 gets the data half of stores. Store data can be available later than
// the store address, but since we don't model the latency of stores, we can
// ignore that.
// Ports 2 and 3 are identical. They handle loads and the address half of
// stores. Port 7 can handle address calculations.
def HWPort0 : ProcResource<1>;
def HWPort1 : ProcResource<1>;
def HWPort2 : ProcResource<1>;
def HWPort3 : ProcResource<1>;
def HWPort4 : ProcResource<1>;
def HWPort5 : ProcResource<1>;
def HWPort6 : ProcResource<1>;
def HWPort7 : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def HWPort01  : ProcResGroup<[HWPort0, HWPort1]>;
def HWPort23  : ProcResGroup<[HWPort2, HWPort3]>;
def HWPort237 : ProcResGroup<[HWPort2, HWPort3, HWPort7]>;
def HWPort04  : ProcResGroup<[HWPort0, HWPort4]>;
def HWPort05  : ProcResGroup<[HWPort0, HWPort5]>;
def HWPort06  : ProcResGroup<[HWPort0, HWPort6]>;
def HWPort15  : ProcResGroup<[HWPort1, HWPort5]>;
def HWPort16  : ProcResGroup<[HWPort1, HWPort6]>;
def HWPort56  : ProcResGroup<[HWPort5, HWPort6]>;
def HWPort015 : ProcResGroup<[HWPort0, HWPort1, HWPort5]>;
def HWPort056 : ProcResGroup<[HWPort0, HWPort5, HWPort6]>;
def HWPort0156: ProcResGroup<[HWPort0, HWPort1, HWPort5, HWPort6]>;

// 60 Entry Unified Scheduler
def HWPortAny : ProcResGroup<[HWPort0, HWPort1, HWPort2, HWPort3, HWPort4,
                              HWPort5, HWPort6, HWPort7]> {
  let BufferSize=60;
}

// Integer division issued on port 0.
def HWDivider : ProcResource<1>;

// Loads are 4 cycles, so ReadAfterLd registers needn't be available until 4
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 4>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass HWWriteResPair<X86FoldableSchedWrite SchedRW,
                          ProcResourceKind ExePort,
                          int Lat> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, [ExePort]> { let Latency = Lat; }

  // Memory variant also uses a cycle on port 2/3 and adds 4 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, [HWPort23, ExePort]> {
     let Latency = !add(Lat, 4);
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [HWPort4]>;

// Store_addr on 237.
// Store_data on 4.
def : WriteRes<WriteStore, [HWPort237, HWPort4]>;
def : WriteRes<WriteLoad,  [HWPort23]> { let Latency = 4; }
def : WriteRes<WriteMove,  [HWPort0156]>;
def : WriteRes<WriteZero,  []>;

defm : HWWriteResPair<WriteALU,   HWPort0156, 1>;
defm : HWWriteResPair<WriteIMul,  HWPort1,   3>;
def  : WriteRes<WriteIMulH, []> { let Latency = 3; }
defm : HWWriteResPair<WriteShift, HWPort06,  1>;
defm : HWWriteResPair<WriteJump,  HWPort06,   1>;

// This is for simple LEAs with one or two input operands.
// The complex ones can only execute on port 1, and they require two cycles on
// the port to read all inputs. We don't model that.
def : WriteRes<WriteLEA, [HWPort15]>;

// This is quite rough, latency depends on the dividend.
def : WriteRes<WriteIDiv, [HWPort0, HWDivider]> {
  let Latency = 25;
  let ResourceCycles = [1, 10];
}
def : WriteRes<WriteIDivLd, [HWPort23, HWPort0, HWDivider]> {
  let Latency = 29;
  let ResourceCycles = [1, 1, 10];
}

// Scalar and vector floating point.
defm : HWWriteResPair<WriteFAdd,   HWPort1, 3>;
defm : HWWriteResPair<WriteFMul,   HWPort0, 5>;
defm : HWWriteResPair<WriteFDiv,   HWPort0, 12>; // 10-14 cycles.
defm : HWWriteResPair<WriteFRcp,   HWPort0, 5>;
defm : HWWriteResPair<WriteFRsqrt, HWPort0, 5>;
defm : HWWriteResPair<WriteFSqrt,  HWPort0, 15>;
defm : HWWriteResPair<WriteCvtF2I, HWPort1, 3>;
defm : HWWriteResPair<WriteCvtI2F, HWPort1, 4>;
defm : HWWriteResPair<WriteCvtF2F, HWPort1, 3>;
defm : HWWriteResPair<WriteFShuffle,  HWPort5,  1>;
defm : HWWriteResPair<WriteFBlend,  HWPort015,  1>;
defm : HWWriteResPair<WriteFShuffle256,  HWPort5,  3>;

def : WriteRes<WriteFVarBlend, [HWPort5]> {
  let Latency = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteFVarBlendLd, [HWPort5, HWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

// Vector integer operations.
defm : HWWriteResPair<WriteVecShift, HWPort0,  1>;
defm : HWWriteResPair<WriteVecLogic, HWPort015, 1>;
defm : HWWriteResPair<WriteVecALU,   HWPort15,  1>;
defm : HWWriteResPair<WriteVecIMul,  HWPort0,   5>;
defm : HWWriteResPair<WriteShuffle,  HWPort5,  1>;
defm : HWWriteResPair<WriteBlend,  HWPort15,  1>;
defm : HWWriteResPair<WriteShuffle256,  HWPort5,  3>;

def : WriteRes<WriteVarBlend, [HWPort5]> {
  let Latency = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteVarBlendLd, [HWPort5, HWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

def : WriteRes<WriteVarVecShift, [HWPort0, HWPort5]> {
  let Latency = 2;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteVarVecShiftLd, [HWPort0, HWPort5, HWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1, 1];
}

def : WriteRes<WriteMPSAD, [HWPort0, HWPort5]> {
  let Latency = 6;
  let ResourceCycles = [1, 2];
}
def : WriteRes<WriteMPSADLd, [HWPort23, HWPort0, HWPort5]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 2];
}

// String instructions.
// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [HWPort0]> {
  let Latency = 10;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [HWPort0, HWPort23]> {
  let Latency = 10;
  let ResourceCycles = [3, 1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [HWPort0, HWPort16, HWPort5]> {
  let Latency = 10;
  let ResourceCycles = [3, 2, 4];
}
def : WriteRes<WritePCmpEStrMLd, [HWPort05, HWPort16, HWPort23]> {
  let Latency = 10;
  let ResourceCycles = [6, 2, 1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [HWPort0]> {
  let Latency = 11;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [HWPort0, HWPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [HWPort05, HWPort16]> {
  let Latency = 11;
  let ResourceCycles = [6, 2];
}
def : WriteRes<WritePCmpEStrILd, [HWPort0, HWPort16, HWPort5, HWPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 2, 2, 1];
}

// AES Instructions.
def : WriteRes<WriteAESDecEnc, [HWPort5]> {
  let Latency = 7;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [HWPort5, HWPort23]> {
  let Latency = 7;
  let ResourceCycles = [1, 1];
}

def : WriteRes<WriteAESIMC, [HWPort5]> {
  let Latency = 14;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [HWPort5, HWPort23]> {
  let Latency = 14;
  let ResourceCycles = [2, 1];
}

def : WriteRes<WriteAESKeyGen, [HWPort0, HWPort5]> {
  let Latency = 10;
  let ResourceCycles = [2, 8];
}
def : WriteRes<WriteAESKeyGenLd, [HWPort0, HWPort5, HWPort23]> {
  let Latency = 10;
  let ResourceCycles = [2, 7, 1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [HWPort0, HWPort5]> {
  let Latency = 7;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteCLMulLd, [HWPort0, HWPort5, HWPort23]> {
  let Latency = 7;
  let ResourceCycles = [2, 1, 1];
}

def : WriteRes<WriteSystem,     [HWPort0156]> { let Latency = 100; }
def : WriteRes<WriteMicrocoded, [HWPort0156]> { let Latency = 100; }
def : WriteRes<WriteFence,  [HWPort23, HWPort4]>;
def : WriteRes<WriteNop, []>;

//================ Exceptions ================//

//-- Specific Scheduling Models --//

// Starting with P0.
def WriteP0 : SchedWriteRes<[HWPort0]>;

def WriteP0_P1_Lat4 : SchedWriteRes<[HWPort0, HWPort1]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}

def WriteP0_P1_Lat4Ld : SchedWriteRes<[HWPort0, HWPort1, HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}

def WriteP01 : SchedWriteRes<[HWPort01]>;

def Write2P01 : SchedWriteRes<[HWPort01]> {
  let NumMicroOps = 2;
}
def Write3P01 : SchedWriteRes<[HWPort01]> {
  let NumMicroOps = 3;
}

def WriteP015 : SchedWriteRes<[HWPort015]>;

def WriteP01_P5 : SchedWriteRes<[HWPort01, HWPort5]> {
  let NumMicroOps = 2;
}
def WriteP06 : SchedWriteRes<[HWPort06]>;

def Write2P06 : SchedWriteRes<[HWPort06]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}

def Write3P06_Lat2 : SchedWriteRes<[HWPort06]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}

def WriteP0156_P23 : SchedWriteRes<[HWPort0156, HWPort23]> {
  let NumMicroOps = 2;
}

def Write2P0156_P23 : SchedWriteRes<[HWPort0156, HWPort23]> {
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}

def Write2P0156_Lat2 : SchedWriteRes<[HWPort0156]> {
  let Latency = 2;
  let ResourceCycles = [2];
}
def Write2P0156_Lat2Ld : SchedWriteRes<[HWPort0156, HWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

def Write5P0156 : SchedWriteRes<[HWPort0156]> {
  let NumMicroOps = 5;
  let ResourceCycles = [5];
}

def WriteP0156_2P237_P4 : SchedWriteRes<[HWPort0156, HWPort237, HWPort4]> {
  let Latency = 1;
  let ResourceCycles = [1, 2, 1];
}

def Write2P0156_2P237_P4 : SchedWriteRes<[HWPort0156, HWPort237, HWPort4]> {
  let Latency = 1;
  let ResourceCycles = [2, 2, 1];
}

def Write3P0156_2P237_P4 : SchedWriteRes<[HWPort0156, HWPort237, HWPort4]> {
  let Latency = 1;
  let ResourceCycles = [3, 2, 1];
}

// Starting with P1.
def WriteP1 : SchedWriteRes<[HWPort1]>;

def WriteP1_P23 : SchedWriteRes<[HWPort1, HWPort23]> {
  let NumMicroOps = 2;
}
def WriteP1_Lat3 : SchedWriteRes<[HWPort1]> {
  let Latency = 3;
}
def WriteP1_Lat3Ld : SchedWriteRes<[HWPort1, HWPort23]> {
  let Latency = 7;
}

def Write2P1 : SchedWriteRes<[HWPort1]> {
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def Write2P1_P23 : SchedWriteRes<[HWPort1, HWPort23]> {
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def WriteP15 : SchedWriteRes<[HWPort15]>;
def WriteP15Ld : SchedWriteRes<[HWPort15, HWPort23]> {
  let Latency = 4;
}

def WriteP1_P5_Lat4 : SchedWriteRes<[HWPort1, HWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}

def WriteP1_P5_Lat4Ld : SchedWriteRes<[HWPort1, HWPort5, HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}

def WriteP1_P5_Lat6 : SchedWriteRes<[HWPort1, HWPort5]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}

def WriteP1_P5_Lat6Ld : SchedWriteRes<[HWPort1, HWPort5, HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}

// Starting with P2.
def Write2P237_P4 : SchedWriteRes<[HWPort237, HWPort4]> {
  let Latency = 1;
  let ResourceCycles = [2, 1];
}

// Starting with P5.
def WriteP5 : SchedWriteRes<[HWPort5]>;
def WriteP5Ld : SchedWriteRes<[HWPort5, HWPort23]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}

// Notation:
// - r: register.
// - mm: 64 bit mmx register.
// - x = 128 bit xmm register.
// - (x)mm = mmx or xmm register.
// - y = 256 bit ymm register.
// - v = any vector register.
// - m = memory.

//=== Integer Instructions ===//
//-- Move instructions --//

// MOV.
// r16,m.
def : InstRW<[WriteALULd], (instregex "MOV16rm")>;

// MOVSX, MOVZX.
// r,m.
def : InstRW<[WriteLoad], (instregex "MOV(S|Z)X32rm(8|16)")>;

// CMOVcc.
// r,r.
def : InstRW<[Write2P0156_Lat2],
      (instregex "CMOV(O|NO|B|AE|E|NE|BE|A|S|NS|P|NP|L|GE|LE|G)(16|32|64)rr")>;
// r,m.
def : InstRW<[Write2P0156_Lat2Ld, ReadAfterLd],
      (instregex "CMOV(O|NO|B|AE|E|NE|BE|A|S|NS|P|NP|L|GE|LE|G)(16|32|64)rm")>;

// XCHG.
// r,r.
def WriteXCHG : SchedWriteRes<[HWPort0156]> {
  let Latency = 2;
  let ResourceCycles = [3];
}

def : InstRW<[WriteXCHG], (instregex "XCHG(8|16|32|64)rr", "XCHG(16|32|64)ar")>;

// r,m.
def WriteXCHGrm : SchedWriteRes<[]> {
  let Latency = 21;
  let NumMicroOps = 8;
}
def : InstRW<[WriteXCHGrm], (instregex "XCHG(8|16|32|64)rm")>;

// XLAT.
def WriteXLAT : SchedWriteRes<[]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[WriteXLAT], (instregex "XLAT")>;

// PUSH.
// m.
def : InstRW<[Write2P237_P4], (instregex "PUSH(16|32)rmm")>;

// PUSHF.
def WritePushF : SchedWriteRes<[HWPort1, HWPort4, HWPort237, HWPort06]> {
  let NumMicroOps = 4;
}
def : InstRW<[WritePushF], (instregex "PUSHF(16|32)")>;

// PUSHA.
def WritePushA : SchedWriteRes<[]> {
  let NumMicroOps = 19;
}
def : InstRW<[WritePushA], (instregex "PUSHA(16|32)")>;

// POP.
// m.
def : InstRW<[Write2P237_P4], (instregex "POP(16|32)rmm")>;

// POPF.
def WritePopF : SchedWriteRes<[]> {
  let NumMicroOps = 9;
}
def : InstRW<[WritePopF], (instregex "POPF(16|32)")>;

// POPA.
def WritePopA : SchedWriteRes<[]> {
  let NumMicroOps = 18;
}
def : InstRW<[WritePopA], (instregex "POPA(16|32)")>;

// LAHF SAHF.
def : InstRW<[WriteP06], (instregex "(S|L)AHF")>;

// BSWAP.
// r32.
def WriteBSwap32 : SchedWriteRes<[HWPort15]>;
def : InstRW<[WriteBSwap32], (instregex "BSWAP32r")>;

// r64.
def WriteBSwap64 : SchedWriteRes<[HWPort06, HWPort15]> {
  let NumMicroOps = 2;
}
def : InstRW<[WriteBSwap64], (instregex "BSWAP64r")>;

// MOVBE.
// r16,m16 / r64,m64.
def : InstRW<[Write2P0156_Lat2Ld], (instregex "MOVBE(16|64)rm")>;

// r32, m32.
def WriteMoveBE32rm : SchedWriteRes<[HWPort15, HWPort23]> {
  let NumMicroOps = 2;
}
def : InstRW<[WriteMoveBE32rm], (instregex "MOVBE32rm")>;

// m16,r16.
def WriteMoveBE16mr : SchedWriteRes<[HWPort06, HWPort237, HWPort4]> {
  let NumMicroOps = 3;
}
def : InstRW<[WriteMoveBE16mr], (instregex "MOVBE16mr")>;

// m32,r32.
def WriteMoveBE32mr : SchedWriteRes<[HWPort15, HWPort237, HWPort4]> {
  let NumMicroOps = 3;
}
def : InstRW<[WriteMoveBE32mr], (instregex "MOVBE32mr")>;

// m64,r64.
def WriteMoveBE64mr : SchedWriteRes<[HWPort06, HWPort15, HWPort237, HWPort4]> {
  let NumMicroOps = 4;
}
def : InstRW<[WriteMoveBE64mr], (instregex "MOVBE64mr")>;

//-- Arithmetic instructions --//

// ADD SUB.
// m,r/i.
def : InstRW<[Write2P0156_2P237_P4],
              (instregex "(ADD|SUB)(8|16|32|64)m(r|i)",
              "(ADD|SUB)(8|16|32|64)mi8", "(ADD|SUB)64mi32")>;

// ADC SBB.
// r,r/i.
def : InstRW<[Write2P0156_Lat2], (instregex "(ADC|SBB)(8|16|32|64)r(r|i)",
                           "(ADC|SBB)(16|32|64)ri8",
                           "(ADC|SBB)64ri32",
                           "(ADC|SBB)(8|16|32|64)rr_REV")>;

// r,m.
def : InstRW<[Write2P0156_Lat2Ld, ReadAfterLd], (instregex "(ADC|SBB)(8|16|32|64)rm")>;

// m,r/i.
def : InstRW<[Write3P0156_2P237_P4],
             (instregex "(ADC|SBB)(8|16|32|64)m(r|i)",
              "(ADC|SBB)(16|32|64)mi8",
              "(ADC|SBB)64mi32")>;

// INC DEC NOT NEG.
// m.
def : InstRW<[WriteP0156_2P237_P4],
             (instregex "(INC|DEC|NOT|NEG)(8|16|32|64)m",
              "(INC|DEC)64(16|32)m")>;

// MUL IMUL.
// r16.
def WriteMul16 : SchedWriteRes<[HWPort1, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def : InstRW<[WriteMul16], (instregex "IMUL16r", "MUL16r")>;

// m16.
def WriteMul16Ld : SchedWriteRes<[HWPort1, HWPort0156, HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 5;
}
def : InstRW<[WriteMul16Ld], (instregex "IMUL16m", "MUL16m")>;

// r32.
def WriteMul32 : SchedWriteRes<[HWPort1, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 3;
}
def : InstRW<[WriteMul32], (instregex "IMUL32r", "MUL32r")>;

// m32.
def WriteMul32Ld : SchedWriteRes<[HWPort1, HWPort0156, HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 4;
}
def : InstRW<[WriteMul32Ld], (instregex "IMUL32m", "MUL32m")>;

// r64.
def WriteMul64 : SchedWriteRes<[HWPort1, HWPort6]> {
  let Latency = 3;
  let NumMicroOps = 2;
}
def : InstRW<[WriteMul64], (instregex "IMUL64r", "MUL64r")>;

// m64.
def WriteMul64Ld : SchedWriteRes<[HWPort1, HWPort6, HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[WriteMul64Ld], (instregex "IMUL64m", "MUL64m")>;

// r16,r16.
def WriteMul16rri : SchedWriteRes<[HWPort1, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[WriteMul16rri], (instregex "IMUL16rri", "IMUL16rri8")>;

// r16,m16.
def WriteMul16rmi : SchedWriteRes<[HWPort1, HWPort0156, HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
}
def : InstRW<[WriteMul16rmi], (instregex "IMUL16rmi", "IMUL16rmi8")>;

// MULX.
// r32,r32,r32.
def WriteMulX32 : SchedWriteRes<[HWPort1, HWPort056]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 2];
}
def : InstRW<[WriteMulX32], (instregex "MULX32rr")>;

// r32,r32,m32.
def WriteMulX32Ld : SchedWriteRes<[HWPort1, HWPort056, HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1, 2, 1];
}
def : InstRW<[WriteMulX32Ld], (instregex "MULX32rm")>;

// r64,r64,r64.
def WriteMulX64 : SchedWriteRes<[HWPort1, HWPort6]> {
  let Latency = 4;
  let NumMicroOps = 2;
}
def : InstRW<[WriteMulX64], (instregex "MULX64rr")>;

// r64,r64,m64.
def WriteMulX64Ld : SchedWriteRes<[HWPort1, HWPort6, HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
}
def : InstRW<[WriteMulX64Ld], (instregex "MULX64rm")>;

// DIV.
// r8.
def WriteDiv8 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 22;
  let NumMicroOps = 9;
}
def : InstRW<[WriteDiv8], (instregex "DIV8r")>;

// r16.
def WriteDiv16 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 23;
  let NumMicroOps = 10;
}
def : InstRW<[WriteDiv16], (instregex "DIV16r")>;

// r32.
def WriteDiv32 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 22;
  let NumMicroOps = 10;
}
def : InstRW<[WriteDiv32], (instregex "DIV32r")>;

// r64.
def WriteDiv64 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 32;
  let NumMicroOps = 36;
}
def : InstRW<[WriteDiv64], (instregex "DIV64r")>;

// IDIV.
// r8.
def WriteIDiv8 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 23;
  let NumMicroOps = 9;
}
def : InstRW<[WriteIDiv8], (instregex "IDIV8r")>;

// r16.
def WriteIDiv16 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 23;
  let NumMicroOps = 10;
}
def : InstRW<[WriteIDiv16], (instregex "IDIV16r")>;

// r32.
def WriteIDiv32 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 22;
  let NumMicroOps = 9;
}
def : InstRW<[WriteIDiv32], (instregex "IDIV32r")>;

// r64.
def WriteIDiv64 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 39;
  let NumMicroOps = 59;
}
def : InstRW<[WriteIDiv64], (instregex "IDIV64r")>;

//-- Logic instructions --//

// AND OR XOR.
// m,r/i.
def : InstRW<[Write2P0156_2P237_P4],
             (instregex "(AND|OR|XOR)(8|16|32|64)m(r|i)",
              "(AND|OR|XOR)(8|16|32|64)mi8", "(AND|OR|XOR)64mi32")>;

// SHR SHL SAR.
// m,i.
def WriteShiftRMW : SchedWriteRes<[HWPort06, HWPort237, HWPort4]> {
  let NumMicroOps = 4;
  let ResourceCycles = [2, 1, 1];
}
def : InstRW<[WriteShiftRMW], (instregex "S(A|H)(R|L)(8|16|32|64)m(i|1)")>;

// r,cl.
def : InstRW<[Write3P06_Lat2], (instregex "S(A|H)(R|L)(8|16|32|64)rCL")>;

// m,cl.
def WriteShiftClLdRMW : SchedWriteRes<[HWPort06, HWPort23, HWPort4]> {
  let NumMicroOps = 6;
  let ResourceCycles = [3, 2, 1];
}
def : InstRW<[WriteShiftClLdRMW], (instregex "S(A|H)(R|L)(8|16|32|64)mCL")>;

// ROR ROL.
// r,1.
def : InstRW<[Write2P06], (instregex "RO(R|L)(8|16|32|64)r1")>;

// m,i.
def WriteRotateRMW : SchedWriteRes<[HWPort06, HWPort237, HWPort4]> {
  let NumMicroOps = 5;
  let ResourceCycles = [2, 2, 1];
}
def : InstRW<[WriteRotateRMW], (instregex "RO(R|L)(8|16|32|64)mi")>;

// r,cl.
def : InstRW<[Write3P06_Lat2], (instregex "RO(R|L)(8|16|32|64)rCL")>;

// m,cl.
def WriteRotateRMWCL : SchedWriteRes<[]> {
  let NumMicroOps = 6;
}
def : InstRW<[WriteRotateRMWCL], (instregex "RO(R|L)(8|16|32|64)mCL")>;

// RCR RCL.
// r,1.
def WriteRCr1 : SchedWriteRes<[HWPort06, HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteRCr1], (instregex "RC(R|L)(8|16|32|64)r1")>;

// m,1.
def WriteRCm1 : SchedWriteRes<[]> {
  let NumMicroOps = 6;
}
def : InstRW<[WriteRCm1], (instregex "RC(R|L)(8|16|32|64)m1")>;

// r,i.
def WriteRCri : SchedWriteRes<[HWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 8;
}
def : InstRW<[WriteRCri], (instregex "RC(R|L)(8|16|32|64)r(i|CL)")>;

// m,i.
def WriteRCmi : SchedWriteRes<[]> {
  let NumMicroOps = 11;
}
def : InstRW<[WriteRCmi], (instregex "RC(R|L)(8|16|32|64)m(i|CL)")>;

// SHRD SHLD.
// r,r,i.
def WriteShDrr : SchedWriteRes<[HWPort1]> {
  let Latency = 3;
}
def : InstRW<[WriteShDrr], (instregex "SH(R|L)D(16|32|64)rri8")>;

// m,r,i.
def WriteShDmr : SchedWriteRes<[]> {
  let NumMicroOps = 5;
}
def : InstRW<[WriteShDmr], (instregex "SH(R|L)D(16|32|64)mri8")>;

// r,r,cl.
def WriteShlDCL : SchedWriteRes<[HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
}
def : InstRW<[WriteShlDCL], (instregex "SHLD(16|32|64)rrCL")>;

// r,r,cl.
def WriteShrDCL : SchedWriteRes<[HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def : InstRW<[WriteShrDCL], (instregex "SHRD(16|32|64)rrCL")>;

// m,r,cl.
def WriteShDmrCL : SchedWriteRes<[]> {
  let NumMicroOps = 7;
}
def : InstRW<[WriteShDmrCL], (instregex "SH(R|L)D(16|32|64)mrCL")>;

// BT.
// r,r/i.
def : InstRW<[WriteShift], (instregex "BT(16|32|64)r(r|i8)")>;

// m,r.
def WriteBTmr : SchedWriteRes<[]> {
  let NumMicroOps = 10;
}
def : InstRW<[WriteBTmr], (instregex "BT(16|32|64)mr")>;

// m,i.
def : InstRW<[WriteShiftLd], (instregex "BT(16|32|64)mi8")>;

// BTR BTS BTC.
// r,r,i.
def : InstRW<[WriteShift], (instregex "BT(R|S|C)(16|32|64)r(r|i8)")>;

// m,r.
def WriteBTRSCmr : SchedWriteRes<[]> {
  let NumMicroOps = 11;
}
def : InstRW<[WriteBTRSCmr], (instregex "BT(R|S|C)(16|32|64)mr")>;

// m,i.
def : InstRW<[WriteShiftLd], (instregex "BT(R|S|C)(16|32|64)mi8")>;

// BSF BSR.
// r,r.
def : InstRW<[WriteP1_Lat3], (instregex "BS(R|F)(16|32|64)rr")>;
// r,m.
def : InstRW<[WriteP1_Lat3Ld], (instregex "BS(R|F)(16|32|64)rm")>;

// SETcc.
// r.
def : InstRW<[WriteShift],
             (instregex "SET(O|NO|B|AE|E|NE|BE|A|S|NS|P|NP|L|GE|LE|G)r")>;
// m.
def WriteSetCCm : SchedWriteRes<[HWPort06, HWPort237, HWPort4]> {
  let NumMicroOps = 3;
}
def : InstRW<[WriteSetCCm],
             (instregex "SET(O|NO|B|AE|E|NE|BE|A|S|NS|P|NP|L|GE|LE|G)m")>;

// CLD STD.
def WriteCldStd : SchedWriteRes<[HWPort15, HWPort6]> {
  let NumMicroOps = 3;
}
def : InstRW<[WriteCldStd], (instregex "STD", "CLD")>;

// LZCNT TZCNT.
// r,r.
def : InstRW<[WriteP1_Lat3], (instregex "(L|TZCNT)(16|32|64)rr")>;
// r,m.
def : InstRW<[WriteP1_Lat3Ld], (instregex "(L|TZCNT)(16|32|64)rm")>;

// ANDN.
// r,r.
def : InstRW<[WriteP15], (instregex "ANDN(32|64)rr")>;
// r,m.
def : InstRW<[WriteP15Ld], (instregex "ANDN(32|64)rm")>;

// BLSI BLSMSK BLSR.
// r,r.
def : InstRW<[WriteP15], (instregex "BLS(I|MSK|R)(32|64)rr")>;
// r,m.
def : InstRW<[WriteP15Ld], (instregex "BLS(I|MSK|R)(32|64)rm")>;

// BEXTR.
// r,r,r.
def : InstRW<[Write2P0156_Lat2], (instregex "BEXTR(32|64)rr")>;
// r,m,r.
def : InstRW<[Write2P0156_Lat2Ld], (instregex "BEXTR(32|64)rm")>;

// BZHI.
// r,r,r.
def : InstRW<[WriteP15], (instregex "BZHI(32|64)rr")>;
// r,m,r.
def : InstRW<[WriteP15Ld], (instregex "BZHI(32|64)rm")>;

// PDEP PEXT.
// r,r,r.
def : InstRW<[WriteP1_Lat3], (instregex "PDEP(32|64)rr", "PEXT(32|64)rr")>;
// r,m,r.
def : InstRW<[WriteP1_Lat3Ld], (instregex "PDEP(32|64)rm", "PEXT(32|64)rm")>;

//-- Control transfer instructions --//

// J(E|R)CXZ.
def WriteJCXZ : SchedWriteRes<[HWPort0156, HWPort6]> {
  let NumMicroOps = 2;
}
def : InstRW<[WriteJCXZ], (instregex "JCXZ", "JECXZ_(32|64)", "JRCXZ")>;

// LOOP.
def WriteLOOP : SchedWriteRes<[]> {
  let NumMicroOps = 7;
}
def : InstRW<[WriteLOOP], (instregex "LOOP")>;

// LOOP(N)E
def WriteLOOPE : SchedWriteRes<[]> {
  let NumMicroOps = 11;
}
def : InstRW<[WriteLOOPE], (instregex "LOOPE", "LOOPNE")>;

// CALL.
// r.
def WriteCALLr : SchedWriteRes<[HWPort237, HWPort4, HWPort6]> {
  let NumMicroOps = 3;
}
def : InstRW<[WriteCALLr], (instregex "CALL(16|32)r")>;

// m.
def WriteCALLm : SchedWriteRes<[HWPort237, HWPort4, HWPort6]> {
  let NumMicroOps = 4;
  let ResourceCycles = [2, 1, 1];
}
def : InstRW<[WriteCALLm], (instregex "CALL(16|32)m")>;

// RET.
def WriteRET : SchedWriteRes<[HWPort237, HWPort6]> {
  let NumMicroOps = 2;
}
def : InstRW<[WriteRET], (instregex "RET(L|Q|W)", "LRET(L|Q|W)")>;

// i.
def WriteRETI : SchedWriteRes<[HWPort23, HWPort6, HWPort015]> {
  let NumMicroOps = 4;
  let ResourceCycles = [1, 2, 1];
}
def : InstRW<[WriteRETI], (instregex "RETI(L|Q|W)", "LRETI(L|Q|W)")>;

// BOUND.
// r,m.
def WriteBOUND : SchedWriteRes<[]> {
  let NumMicroOps = 15;
}
def : InstRW<[WriteBOUND], (instregex "BOUNDS(16|32)rm")>;

// INTO.
def WriteINTO : SchedWriteRes<[]> {
  let NumMicroOps = 4;
}
def : InstRW<[WriteINTO], (instregex "INTO")>;

//-- String instructions --//

// LODSB/W.
def : InstRW<[Write2P0156_P23], (instregex "LODS(B|W)")>;

// LODSD/Q.
def : InstRW<[WriteP0156_P23], (instregex "LODS(L|Q)")>;

// STOS.
def WriteSTOS : SchedWriteRes<[HWPort23, HWPort0156, HWPort4]> {
  let NumMicroOps = 3;
}
def : InstRW<[WriteSTOS], (instregex "STOS(B|L|Q|W)")>;

// MOVS.
def WriteMOVS : SchedWriteRes<[HWPort23, HWPort4, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 5;
  let ResourceCycles = [2, 1, 2];
}
def : InstRW<[WriteMOVS], (instregex "MOVS(B|L|Q|W)")>;

// SCAS.
def : InstRW<[Write2P0156_P23], (instregex "SCAS(B|W|L|Q)")>;

// CMPS.
def WriteCMPS : SchedWriteRes<[HWPort23, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 5;
  let ResourceCycles = [2, 3];
}
def : InstRW<[WriteCMPS], (instregex "CMPS(B|L|Q|W)")>;

//-- Synchronization instructions --//

// XADD.
def WriteXADD : SchedWriteRes<[]> {
  let NumMicroOps = 5;
}
def : InstRW<[WriteXADD], (instregex "XADD(8|16|32|64)rm")>;

// CMPXCHG.
def WriteCMPXCHG : SchedWriteRes<[]> {
  let NumMicroOps = 6;
}
def : InstRW<[WriteCMPXCHG], (instregex "CMPXCHG(8|16|32|64)rm")>;

// CMPXCHG8B.
def WriteCMPXCHG8B : SchedWriteRes<[]> {
  let NumMicroOps = 15;
}
def : InstRW<[WriteCMPXCHG8B], (instregex "CMPXCHG8B")>;

// CMPXCHG16B.
def WriteCMPXCHG16B : SchedWriteRes<[]> {
  let NumMicroOps = 22;
}
def : InstRW<[WriteCMPXCHG16B], (instregex "CMPXCHG16B")>;

//-- Other --//

// PAUSE.
def WritePAUSE : SchedWriteRes<[HWPort05, HWPort6]> {
  let NumMicroOps = 5;
  let ResourceCycles = [1, 3];
}
def : InstRW<[WritePAUSE], (instregex "PAUSE")>;

// LEAVE.
def : InstRW<[Write2P0156_P23], (instregex "LEAVE")>;

// XGETBV.
def WriteXGETBV : SchedWriteRes<[]> {
  let NumMicroOps = 8;
}
def : InstRW<[WriteXGETBV], (instregex "XGETBV")>;

// RDTSC.
def WriteRDTSC : SchedWriteRes<[]> {
  let NumMicroOps = 15;
}
def : InstRW<[WriteRDTSC], (instregex "RDTSC")>;

// RDPMC.
def WriteRDPMC : SchedWriteRes<[]> {
  let NumMicroOps = 34;
}
def : InstRW<[WriteRDPMC], (instregex "RDPMC")>;

// RDRAND.
def WriteRDRAND : SchedWriteRes<[HWPort23, HWPort015]> {
  let NumMicroOps = 17;
  let ResourceCycles = [1, 16];
}
def : InstRW<[WriteRDRAND], (instregex "RDRAND(16|32|64)r")>;

//=== Floating Point x87 Instructions ===//
//-- Move instructions --//

// FLD.
// m80.
def : InstRW<[WriteP01], (instregex "LD_Frr")>;

def WriteLD_F80m : SchedWriteRes<[HWPort01, HWPort23]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [2, 2];
}
def : InstRW<[WriteLD_F80m], (instregex "LD_F80m")>;

// FBLD.
// m80.
def WriteFBLD : SchedWriteRes<[]> {
  let Latency = 47;
  let NumMicroOps = 43;
}
def : InstRW<[WriteFBLD], (instregex "FBLDm")>;

// FST(P).
// r.
def : InstRW<[WriteP01], (instregex "ST_(F|FP)rr")>;

// m80.
def WriteST_FP80m : SchedWriteRes<[HWPort0156, HWPort23, HWPort4]> {
  let NumMicroOps = 7;
  let ResourceCycles = [3, 2, 2];
}
def : InstRW<[WriteST_FP80m], (instregex "ST_FP80m")>;

// FBSTP.
// m80.
def WriteFBSTP : SchedWriteRes<[]> {
  let NumMicroOps = 226;
}
def : InstRW<[WriteFBSTP], (instregex "FBSTPm")>;

// FXCHG.
def : InstRW<[WriteNop], (instregex "XCH_F")>;

// FILD.
def WriteFILD : SchedWriteRes<[HWPort01, HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
}
def : InstRW<[WriteFILD], (instregex "ILD_F(16|32|64)m")>;

// FIST(P) FISTTP.
def WriteFIST : SchedWriteRes<[HWPort1, HWPort23, HWPort4]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[WriteFIST], (instregex "IST_(F|FP)(16|32)m")>;

// FLDZ.
def : InstRW<[WriteP01], (instregex "LD_F0")>;

// FLD1.
def : InstRW<[Write2P01], (instregex "LD_F1")>;

// FLDPI FLDL2E etc.
def : InstRW<[Write2P01], (instregex "FLDPI", "FLDL2(T|E)" "FLDL(G|N)2")>;

// FCMOVcc.
def WriteFCMOVcc : SchedWriteRes<[HWPort0, HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteFCMOVcc], (instregex "CMOV(B|BE|P|NB|NBE|NE|NP)_F")>;

// FNSTSW.
// AX.
def WriteFNSTSW : SchedWriteRes<[HWPort0, HWPort0156]> {
  let NumMicroOps = 2;
}
def : InstRW<[WriteFNSTSW], (instregex "FNSTSW16r")>;

// m16.
def WriteFNSTSWm : SchedWriteRes<[HWPort0, HWPort4, HWPort237]> {
  let Latency = 6;
  let NumMicroOps = 3;
}
def : InstRW<[WriteFNSTSWm], (instregex "FNSTSWm")>;

// FLDCW.
def WriteFLDCW : SchedWriteRes<[HWPort01, HWPort23, HWPort6]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[WriteFLDCW], (instregex "FLDCW16m")>;

// FNSTCW.
def WriteFNSTCW : SchedWriteRes<[HWPort237, HWPort4, HWPort6]> {
  let NumMicroOps = 3;
}
def : InstRW<[WriteFNSTCW], (instregex "FNSTCW16m")>;

// FINCSTP FDECSTP.
def : InstRW<[WriteP01], (instregex "FINCSTP", "FDECSTP")>;

// FFREE.
def : InstRW<[WriteP01], (instregex "FFREE")>;

// FNSAVE.
def WriteFNSAVE : SchedWriteRes<[]> {
  let NumMicroOps = 147;
}
def : InstRW<[WriteFNSAVE], (instregex "FSAVEm")>;

// FRSTOR.
def WriteFRSTOR : SchedWriteRes<[]> {
  let NumMicroOps = 90;
}
def : InstRW<[WriteFRSTOR], (instregex "FRSTORm")>;

//-- Arithmetic instructions --//

// FABS.
def : InstRW<[WriteP0], (instregex "ABS_F")>;

// FCHS.
def : InstRW<[WriteP0], (instregex "CHS_F")>;

// FCOM(P) FUCOM(P).
// r.
def : InstRW<[WriteP1], (instregex "COM_FST0r", "COMP_FST0r", "UCOM_Fr",
                         "UCOM_FPr")>;
// m.
def : InstRW<[WriteP1_P23], (instregex "FCOM(32|64)m", "FCOMP(32|64)m")>;

// FCOMPP FUCOMPP.
// r.
def : InstRW<[Write2P01], (instregex "FCOMPP", "UCOM_FPPr")>;

// FCOMI(P) FUCOMI(P).
// m.
def : InstRW<[Write3P01], (instregex "COM_FIr", "COM_FIPr", "UCOM_FIr",
                           "UCOM_FIPr")>;

// FICOM(P).
def : InstRW<[Write2P1_P23], (instregex "FICOM(16|32)m", "FICOMP(16|32)m")>;

// FTST.
def : InstRW<[WriteP1], (instregex "TST_F")>;

// FXAM.
def : InstRW<[Write2P1], (instregex "FXAM")>;

// FPREM.
def WriteFPREM : SchedWriteRes<[]> {
  let Latency = 19;
  let NumMicroOps = 28;
}
def : InstRW<[WriteFPREM], (instregex "FPREM")>;

// FPREM1.
def WriteFPREM1 : SchedWriteRes<[]> {
  let Latency = 27;
  let NumMicroOps = 41;
}
def : InstRW<[WriteFPREM1], (instregex "FPREM1")>;

// FRNDINT.
def WriteFRNDINT : SchedWriteRes<[]> {
  let Latency = 11;
  let NumMicroOps = 17;
}
def : InstRW<[WriteFRNDINT], (instregex "FRNDINT")>;

//-- Math instructions --//

// FSCALE.
def WriteFSCALE : SchedWriteRes<[]> {
  let Latency = 75; // 49-125
  let NumMicroOps = 50; // 25-75
}
def : InstRW<[WriteFSCALE], (instregex "FSCALE")>;

// FXTRACT.
def WriteFXTRACT : SchedWriteRes<[]> {
  let Latency = 15;
  let NumMicroOps = 17;
}
def : InstRW<[WriteFXTRACT], (instregex "FXTRACT")>;

//-- Other instructions --//

// FNOP.
def : InstRW<[WriteP01], (instregex "FNOP")>;

// WAIT.
def : InstRW<[Write2P01], (instregex "WAIT")>;

// FNCLEX.
def : InstRW<[Write5P0156], (instregex "FNCLEX")>;

// FNINIT.
def WriteFNINIT : SchedWriteRes<[]> {
  let NumMicroOps = 26;
}
def : InstRW<[WriteFNINIT], (instregex "FNINIT")>;

//=== Integer MMX and XMM Instructions ===//
//-- Move instructions --//

// MOVD.
// r32/64 <- (x)mm.
def : InstRW<[WriteP0], (instregex "MMX_MOVD64grr", "MMX_MOVD64from64rr",
                         "VMOVPDI2DIrr", "MOVPDI2DIrr")>;

// (x)mm <- r32/64.
def : InstRW<[WriteP5], (instregex "MMX_MOVD64rr", "MMX_MOVD64to64rr",
                         "VMOVDI2PDIrr", "MOVDI2PDIrr")>;

// MOVQ.
// r64 <- (x)mm.
def : InstRW<[WriteP0], (instregex "VMOVPQIto64rr")>;

// (x)mm <- r64.
def : InstRW<[WriteP5], (instregex "VMOV64toPQIrr", "VMOVZQI2PQIrr")>;

// (x)mm <- (x)mm.
def : InstRW<[WriteP015], (instregex "MMX_MOVQ64rr")>;

// (V)MOVDQA/U.
// x <- x.
def : InstRW<[WriteP015], (instregex "MOVDQ(A|U)rr", "VMOVDQ(A|U)rr",
                           "MOVDQ(A|U)rr_REV", "VMOVDQ(A|U)rr_REV",
                           "VMOVDQ(A|U)Yrr", "VMOVDQ(A|U)Yrr_REV")>;

// MOVDQ2Q.
def : InstRW<[WriteP01_P5], (instregex "MMX_MOVDQ2Qrr")>;

// MOVQ2DQ.
def : InstRW<[WriteP015], (instregex "MMX_MOVQ2DQrr")>;


// PACKSSWB/DW.
// mm <- mm.
def WriteMMXPACKSSrr : SchedWriteRes<[HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : InstRW<[WriteMMXPACKSSrr], (instregex "MMX_PACKSSDWirr",
                                  "MMX_PACKSSWBirr", "MMX_PACKUSWBirr")>;

// mm <- m64.
def WriteMMXPACKSSrm : SchedWriteRes<[HWPort23, HWPort5]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 3];
}
def : InstRW<[WriteMMXPACKSSrm], (instregex "MMX_PACKSSDWirm",
                                  "MMX_PACKSSWBirm", "MMX_PACKUSWBirm")>;

// VPMOVSX/ZX BW BD BQ DW DQ.
// y <- x.
def WriteVPMOVSX : SchedWriteRes<[HWPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
}
def : InstRW<[WriteVPMOVSX], (instregex "VPMOV(SX|ZX)(BW|BQ|DW|DQ)Yrr")>;

// PBLENDW.
// x,x,i / v,v,v,i
def WritePBLENDWr : SchedWriteRes<[HWPort5]>;
def : InstRW<[WritePBLENDWr], (instregex "(V?)PBLENDW(Y?)rri")>;

// x,m,i / v,v,m,i
def WritePBLENDWm : SchedWriteRes<[HWPort5, HWPort23]> {
  let NumMicroOps = 2;
  let Latency = 4;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WritePBLENDWm, ReadAfterLd], (instregex "(V?)PBLENDW(Y?)rmi")>;

// VPBLENDD.
// v,v,v,i.
def WriteVPBLENDDr : SchedWriteRes<[HWPort015]>;
def : InstRW<[WriteVPBLENDDr], (instregex "VPBLENDD(Y?)rri")>;

// v,v,m,i
def WriteVPBLENDDm : SchedWriteRes<[HWPort015, HWPort23]> {
  let NumMicroOps = 2;
  let Latency = 4;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteVPBLENDDm, ReadAfterLd], (instregex "VPBLENDD(Y?)rmi")>;

// MASKMOVQ.
def WriteMASKMOVQ : SchedWriteRes<[HWPort0, HWPort4, HWPort23]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1, 1, 2];
}
def : InstRW<[WriteMASKMOVQ], (instregex "MMX_MASKMOVQ(64)?")>;

// MASKMOVDQU.
def WriteMASKMOVDQU : SchedWriteRes<[HWPort04, HWPort56, HWPort23]> {
  let Latency = 14;
  let NumMicroOps = 10;
  let ResourceCycles = [4, 2, 4];
}
def : InstRW<[WriteMASKMOVDQU], (instregex "(V?)MASKMOVDQU(64)?")>;

// VPMASKMOV D/Q.
// v,v,m.
def WriteVPMASKMOVr : SchedWriteRes<[HWPort5, HWPort23]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteVPMASKMOVr, ReadAfterLd],
                               (instregex "VPMASKMOV(D|Q)(Y?)rm")>;

// m, v,v.
def WriteVPMASKMOVm : SchedWriteRes<[HWPort0, HWPort1, HWPort4, HWPort23]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1, 1, 1, 1];
}
def : InstRW<[WriteVPMASKMOVm], (instregex "VPMASKMOV(D|Q)(Y?)mr")>;

// PMOVMSKB.
def WritePMOVMSKB : SchedWriteRes<[HWPort0]> {
  let Latency = 3;
}
def : InstRW<[WritePMOVMSKB], (instregex "(V|MMX_)?PMOVMSKB(Y?)rr")>;

// PEXTR B/W/D/Q.
// r32,x,i.
def WritePEXTRr : SchedWriteRes<[HWPort0, HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WritePEXTRr], (instregex "PEXTR(B|W|D|Q)rr", "MMX_PEXTRWirri")>;

// m8,x,i.
def WritePEXTRm : SchedWriteRes<[HWPort23, HWPort4, HWPort5]> {
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}
def : InstRW<[WritePEXTRm], (instregex "PEXTR(B|W|D|Q)mr")>;

// VPBROADCAST B/W.
// x, m8/16.
def WriteVPBROADCAST128Ld : SchedWriteRes<[HWPort01, HWPort23, HWPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}
def : InstRW<[WriteVPBROADCAST128Ld, ReadAfterLd],
                                     (instregex "VPBROADCAST(B|W)rm")>;

// y, m8/16
def WriteVPBROADCAST256Ld : SchedWriteRes<[HWPort01, HWPort23, HWPort5]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}
def : InstRW<[WriteVPBROADCAST256Ld, ReadAfterLd],
                                     (instregex "VPBROADCAST(B|W)Yrm")>;

// VPGATHERDD.
// x.
def WriteVPGATHERDD128 : SchedWriteRes<[]> {
  let NumMicroOps = 20;
}
def : InstRW<[WriteVPGATHERDD128, ReadAfterLd], (instregex "VPGATHERDDrm")>;

// y.
def WriteVPGATHERDD256 : SchedWriteRes<[]> {
  let NumMicroOps = 34;
}
def : InstRW<[WriteVPGATHERDD256, ReadAfterLd], (instregex "VPGATHERDDYrm")>;

// VPGATHERQD.
// x.
def WriteVPGATHERQD128 : SchedWriteRes<[]> {
  let NumMicroOps = 15;
}
def : InstRW<[WriteVPGATHERQD128, ReadAfterLd], (instregex "VPGATHERQDrm")>;

// y.
def WriteVPGATHERQD256 : SchedWriteRes<[]> {
  let NumMicroOps = 22;
}
def : InstRW<[WriteVPGATHERQD256, ReadAfterLd], (instregex "VPGATHERQDYrm")>;

// VPGATHERDQ.
// x.
def WriteVPGATHERDQ128 : SchedWriteRes<[]> {
  let NumMicroOps = 12;
}
def : InstRW<[WriteVPGATHERDQ128, ReadAfterLd], (instregex "VPGATHERDQrm")>;

// y.
def WriteVPGATHERDQ256 : SchedWriteRes<[]> {
  let NumMicroOps = 20;
}
def : InstRW<[WriteVPGATHERDQ256, ReadAfterLd], (instregex "VPGATHERDQYrm")>;

// VPGATHERQQ.
// x.
def WriteVPGATHERQQ128 : SchedWriteRes<[]> {
  let NumMicroOps = 14;
}
def : InstRW<[WriteVPGATHERQQ128, ReadAfterLd], (instregex "VPGATHERQQrm")>;

// y.
def WriteVPGATHERQQ256 : SchedWriteRes<[]> {
  let NumMicroOps = 22;
}
def : InstRW<[WriteVPGATHERQQ256, ReadAfterLd], (instregex "VPGATHERQQYrm")>;

//-- Arithmetic instructions --//

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

// HADD, HSUB PS/PD
// x,x / v,v,v.
def : WriteRes<WriteFHAdd, [HWPort1, HWPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 2];
}

// x,m / v,v,m.
def : WriteRes<WriteFHAddLd, [HWPort1, HWPort5, HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1, 2, 1];
}

// PHADD|PHSUB (S) W/D.
// v <- v,v.
def : WriteRes<WritePHAdd, [HWPort1, HWPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 2];
}
// v <- v,m.
def : WriteRes<WritePHAddLd, [HWPort1, HWPort5, HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 2, 1];
}

// PHADD|PHSUB (S) W/D.
// v <- v,v.
def WritePHADDSUBr : SchedWriteRes<[HWPort1, HWPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 2];
}
def : InstRW<[WritePHADDSUBr], (instregex "MMX_PHADD(W?)rr64",
                               "MMX_PHADDSWrr64",
                               "MMX_PHSUB(W|D)rr64",
                               "MMX_PHSUBSWrr64",
                               "(V?)PH(ADD|SUB)(W|D)(Y?)rr",
                               "(V?)PH(ADD|SUB)SWrr(256)?")>;

// v <- v,m.
def WritePHADDSUBm : SchedWriteRes<[HWPort1, HWPort5, HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 2, 1];
}
def : InstRW<[WritePHADDSUBm, ReadAfterLd],
                              (instregex "MMX_PHADD(W?)rm64",
                               "MMX_PHADDSWrm64",
                               "MMX_PHSUB(W|D)rm64",
                               "MMX_PHSUBSWrm64",
                               "(V?)PH(ADD|SUB)(W|D)(Y?)rm",
                               "(V?)PH(ADD|SUB)SWrm(128|256)?")>;

// PCMPGTQ.
// v <- v,v.
def WritePCMPGTQr : SchedWriteRes<[HWPort0]> {
  let Latency = 5;
  let NumMicroOps = 1;
}
def : InstRW<[WritePCMPGTQr], (instregex "(V?)PCMPGTQ(Y?)rr")>;

// v <- v,m.
def WritePCMPGTQm : SchedWriteRes<[HWPort0, HWPort23]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WritePCMPGTQm, ReadAfterLd], (instregex "(V?)PCMPGTQ(Y?)rm")>;

// PMULLD.
// x,x / y,y,y.
def WritePMULLDr : SchedWriteRes<[HWPort0]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : InstRW<[WritePMULLDr], (instregex "(V?)PMULLD(Y?)rr")>;

// x,m / y,y,m.
def WritePMULLDm : SchedWriteRes<[HWPort0, HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WritePMULLDm, ReadAfterLd], (instregex "(V?)PMULLD(Y?)rm")>;

//-- Logic instructions --//

// PTEST.
// v,v.
def WritePTESTr : SchedWriteRes<[HWPort0, HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WritePTESTr], (instregex "(V?)PTEST(Y?)rr")>;

// v,m.
def WritePTESTm : SchedWriteRes<[HWPort0, HWPort5, HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}
def : InstRW<[WritePTESTr], (instregex "(V?)PTEST(Y?)rm")>;

// PSLL,PSRL,PSRA W/D/Q.
// x,x / v,v,x.
def WritePShift : SchedWriteRes<[HWPort0, HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WritePShift], (instregex "(V?)PS(LL|RL|RA)(W|D|Q)(Y?)rr")>;

// PSLL,PSRL DQ.
def : InstRW<[WriteP5], (instregex "(V?)PS(R|L)LDQ(Y?)ri")>;

//-- Other --//

// EMMS.
def WriteEMMS : SchedWriteRes<[]> {
  let Latency = 13;
  let NumMicroOps = 31;
}
def : InstRW<[WriteEMMS], (instregex "MMX_EMMS")>;

//=== Floating Point XMM and YMM Instructions ===//
//-- Move instructions --//

// MOVMSKP S/D.
// r32 <- x.
def WriteMOVMSKPr : SchedWriteRes<[HWPort0]> {
  let Latency = 3;
}
def : InstRW<[WriteMOVMSKPr], (instregex "(V?)MOVMSKP(S|D)rr")>;

// r32 <- y.
def WriteVMOVMSKPYr : SchedWriteRes<[HWPort0]> {
  let Latency = 2;
}
def : InstRW<[WriteVMOVMSKPYr], (instregex "VMOVMSKP(S|D)Yrr")>;

// VPERM2F128.
def : InstRW<[WriteFShuffle256], (instregex "VPERM2F128rr")>;
def : InstRW<[WriteFShuffle256Ld, ReadAfterLd], (instregex "VPERM2F128rm")>;

// BLENDVP S/D.
def : InstRW<[WriteFVarBlend], (instregex "BLENDVP(S|D)rr0")>;
def : InstRW<[WriteFVarBlendLd, ReadAfterLd], (instregex "BLENDVP(S|D)rm0")>;

// VBROADCASTF128.
def : InstRW<[WriteLoad], (instregex "VBROADCASTF128")>;

// EXTRACTPS.
// r32,x,i.
def WriteEXTRACTPSr : SchedWriteRes<[HWPort0, HWPort5]> {
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteEXTRACTPSr], (instregex "(V?)EXTRACTPSrr")>;

// m32,x,i.
def WriteEXTRACTPSm : SchedWriteRes<[HWPort0, HWPort5, HWPort23]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}
def : InstRW<[WriteEXTRACTPSm], (instregex "(V?)EXTRACTPSmr")>;

// VEXTRACTF128.
// x,y,i.
def : InstRW<[WriteFShuffle256], (instregex "VEXTRACTF128rr")>;

// m128,y,i.
def WriteVEXTRACTF128m : SchedWriteRes<[HWPort23, HWPort4]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteVEXTRACTF128m], (instregex "VEXTRACTF128mr")>;

// VINSERTF128.
// y,y,x,i.
def : InstRW<[WriteFShuffle256], (instregex "VINSERTF128rr")>;

// y,y,m128,i.
def WriteVINSERTF128m : SchedWriteRes<[HWPort015, HWPort23]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteFShuffle256, ReadAfterLd], (instregex "VINSERTF128rm")>;

// VMASKMOVP S/D.
// v,v,m.
def WriteVMASKMOVPrm : SchedWriteRes<[HWPort5, HWPort23]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteVMASKMOVPrm], (instregex "VMASKMOVP(S|D)(Y?)rm")>;

// m128,x,x.
def WriteVMASKMOVPmr : SchedWriteRes<[HWPort0, HWPort1, HWPort4, HWPort23]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1, 1, 1, 1];
}
def : InstRW<[WriteVMASKMOVPmr], (instregex "VMASKMOVP(S|D)mr")>;

// m256,y,y.
def WriteVMASKMOVPYmr : SchedWriteRes<[HWPort0, HWPort1, HWPort4, HWPort23]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [1, 1, 1, 1];
}
def : InstRW<[WriteVMASKMOVPYmr], (instregex "VMASKMOVP(S|D)Ymr")>;

// VGATHERDPS.
// x.
def WriteVGATHERDPS128 : SchedWriteRes<[]> {
  let NumMicroOps = 20;
}
def : InstRW<[WriteVGATHERDPS128, ReadAfterLd], (instregex "VGATHERDPSrm")>;

// y.
def WriteVGATHERDPS256 : SchedWriteRes<[]> {
  let NumMicroOps = 34;
}
def : InstRW<[WriteVGATHERDPS256, ReadAfterLd], (instregex "VGATHERDPSYrm")>;

// VGATHERQPS.
// x.
def WriteVGATHERQPS128 : SchedWriteRes<[]> {
  let NumMicroOps = 15;
}
def : InstRW<[WriteVGATHERQPS128, ReadAfterLd], (instregex "VGATHERQPSrm")>;

// y.
def WriteVGATHERQPS256 : SchedWriteRes<[]> {
  let NumMicroOps = 22;
}
def : InstRW<[WriteVGATHERQPS256, ReadAfterLd], (instregex "VGATHERQPSYrm")>;

// VGATHERDPD.
// x.
def WriteVGATHERDPD128 : SchedWriteRes<[]> {
  let NumMicroOps = 12;
}
def : InstRW<[WriteVGATHERDPD128, ReadAfterLd], (instregex "VGATHERDPDrm")>;

// y.
def WriteVGATHERDPD256 : SchedWriteRes<[]> {
  let NumMicroOps = 20;
}
def : InstRW<[WriteVGATHERDPD256, ReadAfterLd], (instregex "VGATHERDPDYrm")>;

// VGATHERQPD.
// x.
def WriteVGATHERQPD128 : SchedWriteRes<[]> {
  let NumMicroOps = 14;
}
def : InstRW<[WriteVGATHERQPD128, ReadAfterLd], (instregex "VGATHERQPDrm")>;

// y.
def WriteVGATHERQPD256 : SchedWriteRes<[]> {
  let NumMicroOps = 22;
}
def : InstRW<[WriteVGATHERQPD256, ReadAfterLd], (instregex "VGATHERQPDYrm")>;

//-- Conversion instructions --//

// CVTPD2PS.
// x,x.
def : InstRW<[WriteP1_P5_Lat4], (instregex "(V?)CVTPD2PSrr")>;

// x,m128.
def : InstRW<[WriteP1_P5_Lat4Ld], (instregex "(V?)CVTPD2PS(X?)rm")>;

// x,y.
def WriteCVTPD2PSYrr : SchedWriteRes<[HWPort1, HWPort5]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteCVTPD2PSYrr], (instregex "(V?)CVTPD2PSYrr")>;

// x,m256.
def WriteCVTPD2PSYrm : SchedWriteRes<[HWPort1, HWPort5, HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}
def : InstRW<[WriteCVTPD2PSYrm], (instregex "(V?)CVTPD2PSYrm")>;

// CVTSD2SS.
// x,x.
def : InstRW<[WriteP1_P5_Lat4], (instregex "(Int_)?(V)?CVTSD2SSrr")>;

// x,m64.
def : InstRW<[WriteP1_P5_Lat4Ld], (instregex "(Int_)?(V)?CVTSD2SSrm")>;

// CVTPS2PD.
// x,x.
def WriteCVTPS2PDrr : SchedWriteRes<[HWPort0, HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteCVTPS2PDrr], (instregex "(V?)CVTPS2PDrr")>;

// x,m64.
// y,m128.
def WriteCVTPS2PDrm : SchedWriteRes<[HWPort0, HWPort23]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteCVTPS2PDrm], (instregex "(V?)CVTPS2PD(Y?)rm")>;

// y,x.
def WriteVCVTPS2PDYrr : SchedWriteRes<[HWPort0, HWPort5]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteVCVTPS2PDYrr], (instregex "VCVTPS2PDYrr")>;

// CVTSS2SD.
// x,x.
def WriteCVTSS2SDrr : SchedWriteRes<[HWPort0, HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteCVTSS2SDrr], (instregex "(Int_)?(V?)CVTSS2SDrr")>;

// x,m32.
def WriteCVTSS2SDrm : SchedWriteRes<[HWPort0, HWPort23]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteCVTSS2SDrm], (instregex "(Int_)?(V?)CVTSS2SDrm")>;

// CVTDQ2PD.
// x,x.
def : InstRW<[WriteP1_P5_Lat4], (instregex "(V)?CVTDQ2PDrr")>;

// y,x.
def : InstRW<[WriteP1_P5_Lat6], (instregex "VCVTDQ2PDYrr")>;

// CVT(T)PD2DQ.
// x,x.
def : InstRW<[WriteP1_P5_Lat4], (instregex "(V?)CVT(T?)PD2DQrr")>;
// x,m128.
def : InstRW<[WriteP1_P5_Lat4Ld], (instregex "(V?)CVT(T?)PD2DQrm")>;
// x,y.
def : InstRW<[WriteP1_P5_Lat6], (instregex "VCVT(T?)PD2DQYrr")>;
// x,m256.
def : InstRW<[WriteP1_P5_Lat6Ld], (instregex "VCVT(T?)PD2DQYrm")>;

// CVT(T)PS2PI.
// mm,x.
def : InstRW<[WriteP1_P5_Lat4], (instregex "MMX_CVT(T?)PS2PIirr")>;

// CVTPI2PD.
// x,mm.
def : InstRW<[WriteP1_P5_Lat4], (instregex "MMX_CVT(T?)PI2PDirr")>;

// CVT(T)PD2PI.
// mm,x.
def : InstRW<[WriteP1_P5_Lat4], (instregex "MMX_CVT(T?)PD2PIirr")>;

// CVSTSI2SS.
// x,r32.
def : InstRW<[WriteP1_P5_Lat4], (instregex "(Int_)?(V?)CVT(T?)SI2SS(64)?rr")>;

// CVT(T)SS2SI.
// r32,x.
def : InstRW<[WriteP0_P1_Lat4], (instregex "(Int_)?(V?)CVT(T?)SS2SI(64)?rr")>;
// r32,m32.
def : InstRW<[WriteP0_P1_Lat4Ld], (instregex "(Int_)?(V?)CVT(T?)SS2SI(64)?rm")>;

// CVTSI2SD.
// x,r32/64.
def : InstRW<[WriteP0_P1_Lat4], (instregex "(Int_)?(V?)CVTSI2SS(64)?rr")>;

// CVTSD2SI.
// r32/64
def : InstRW<[WriteP0_P1_Lat4], (instregex "(Int_)?(V?)CVT(T?)SD2SI(64)?rr")>;
// r32,m32.
def : InstRW<[WriteP0_P1_Lat4Ld], (instregex "(Int_)?(V?)CVT(T?)SD2SI(64)?rm")>;

// VCVTPS2PH.
// x,v,i.
def : InstRW<[WriteP1_P5_Lat4], (instregex "VCVTPS2PH(Y?)rr")>;
// m,v,i.
def : InstRW<[WriteP1_P5_Lat4Ld, WriteRMW], (instregex "VCVTPS2PH(Y?)mr")>;

// VCVTPH2PS.
// v,x.
def : InstRW<[WriteP1_P5_Lat4], (instregex "VCVTPH2PS(Y?)rr")>;

//-- Arithmetic instructions --//

// HADD, HSUB PS/PD
// x,x / v,v,v.
def WriteHADDSUBPr : SchedWriteRes<[HWPort1, HWPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 2];
}
def : InstRW<[WriteHADDSUBPr], (instregex "(V?)H(ADD|SUB)P(S|D)(Y?)rr")>;

// x,m / v,v,m.
def WriteHADDSUBPm : SchedWriteRes<[HWPort1, HWPort5, HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1, 2, 1];
}
def : InstRW<[WriteHADDSUBPm], (instregex "(V?)H(ADD|SUB)P(S|D)(Y?)rm")>;

// MULL SS/SD PS/PD.
// x,x / v,v,v.
def WriteMULr : SchedWriteRes<[HWPort01]> {
  let Latency = 5;
}
def : InstRW<[WriteMULr], (instregex "(V?)MUL(P|S)(S|D)rr")>;

// x,m / v,v,m.
def WriteMULm : SchedWriteRes<[HWPort01, HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteMULm], (instregex "(V?)MUL(P|S)(S|D)rm")>;

// VDIVPS.
// y,y,y.
def WriteVDIVPSYrr : SchedWriteRes<[HWPort0, HWPort15]> {
  let Latency = 19; // 18-21 cycles.
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteVDIVPSYrr], (instregex "VDIVPSYrr")>;

// y,y,m256.
def WriteVDIVPSYrm : SchedWriteRes<[HWPort0, HWPort15, HWPort23]> {
  let Latency = 23; // 18-21 + 4 cycles.
  let NumMicroOps = 4;
  let ResourceCycles = [2, 1, 1];
}
def : InstRW<[WriteVDIVPSYrm, ReadAfterLd], (instregex "VDIVPSYrm")>;

// VDIVPD.
// y,y,y.
def WriteVDIVPDYrr : SchedWriteRes<[HWPort0, HWPort15]> {
  let Latency = 27; // 19-35 cycles.
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteVDIVPDYrr], (instregex "VDIVPDYrr")>;

// y,y,m256.
def WriteVDIVPDYrm : SchedWriteRes<[HWPort0, HWPort15, HWPort23]> {
  let Latency = 31; // 19-35 + 4 cycles.
  let NumMicroOps = 4;
  let ResourceCycles = [2, 1, 1];
}
def : InstRW<[WriteVDIVPDYrm, ReadAfterLd], (instregex "VDIVPDYrm")>;

// VRCPPS.
// y,y.
def WriteVRCPPSr : SchedWriteRes<[HWPort0, HWPort15]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteVRCPPSr], (instregex "VRCPPSYr(_Int)?")>;

// y,m256.
def WriteVRCPPSm : SchedWriteRes<[HWPort0, HWPort15, HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [2, 1, 1];
}
def : InstRW<[WriteVRCPPSm], (instregex "VRCPPSYm(_Int)?")>;

// ROUND SS/SD PS/PD.
// v,v,i.
def WriteROUNDr : SchedWriteRes<[HWPort1]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : InstRW<[WriteROUNDr], (instregex "(V?)ROUND(Y?)(S|P)(S|D)r(_Int)?")>;

// v,m,i.
def WriteROUNDm : SchedWriteRes<[HWPort1, HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteROUNDm], (instregex "(V?)ROUND(Y?)(S|P)(S|D)m(_Int)?")>;

// DPPS.
// x,x,i / v,v,v,i.
def WriteDPPSr : SchedWriteRes<[HWPort0, HWPort1, HWPort5]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2, 1, 1];
}
def : InstRW<[WriteDPPSr], (instregex "(V?)DPPS(Y?)rri")>;

// x,m,i / v,v,m,i.
def WriteDPPSm : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort23, HWPort6]> {
  let Latency = 18;
  let NumMicroOps = 6;
  let ResourceCycles = [2, 1, 1, 1, 1];
}
def : InstRW<[WriteDPPSm, ReadAfterLd], (instregex "(V?)DPPS(Y?)rmi")>;

// DPPD.
// x,x,i.
def WriteDPPDr : SchedWriteRes<[HWPort0, HWPort1, HWPort5]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}
def : InstRW<[WriteDPPDr], (instregex "(V?)DPPDrri")>;

// x,m,i.
def WriteDPPDm : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort23]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1, 1, 1, 1];
}
def : InstRW<[WriteDPPDm], (instregex "(V?)DPPDrmi")>;

// VFMADD.
// v,v,v.
def WriteFMADDr : SchedWriteRes<[HWPort01]> {
  let Latency = 5;
  let NumMicroOps = 1;
}
def : InstRW<[WriteFMADDr],
    (instregex
    // 3p forms.
    "VF(N?)M(ADD|SUB|ADDSUB|SUBADD)P(S|D)(r213|r132|r231)r(Y)?",
    // 3s forms.
    "VF(N?)M(ADD|SUB)S(S|D)(r132|r231|r213)r",
    // 4s/4s_int forms.
    "VF(N?)M(ADD|SUB)S(S|D)4rr(_REV|_Int)?",
    // 4p forms.
    "VF(N?)M(ADD|SUB)P(S|D)4rr(Y)?(_REV)?")>;

// v,v,m.
def WriteFMADDm : SchedWriteRes<[HWPort01, HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteFMADDm],
    (instregex
    // 3p forms.
    "VF(N?)M(ADD|SUB|ADDSUB|SUBADD)P(S|D)(r213|r132|r231)m(Y)?",
    // 3s forms.
    "VF(N?)M(ADD|SUB)S(S|D)(r132|r231|r213)m",
    // 4s/4s_int forms.
    "VF(N?)M(ADD|SUB)S(S|D)4(rm|mr)(_Int)?",
    // 4p forms.
    "VF(N?)M(ADD|SUB)P(S|D)4(rm|mr)(Y)?")>;

//-- Math instructions --//

// VSQRTPS.
// y,y.
def WriteVSQRTPSYr : SchedWriteRes<[HWPort0, HWPort15]> {
  let Latency = 19;
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteVSQRTPSYr], (instregex "VSQRTPSYr")>;

// y,m256.
def WriteVSQRTPSYm : SchedWriteRes<[HWPort0, HWPort15, HWPort23]> {
  let Latency = 23;
  let NumMicroOps = 4;
  let ResourceCycles = [2, 1, 1];
}
def : InstRW<[WriteVSQRTPSYm], (instregex "VSQRTPSYm")>;

// VSQRTPD.
// y,y.
def WriteVSQRTPDYr : SchedWriteRes<[HWPort0, HWPort15]> {
  let Latency = 28;
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteVSQRTPDYr], (instregex "VSQRTPDYr")>;

// y,m256.
def WriteVSQRTPDYm : SchedWriteRes<[HWPort0, HWPort15, HWPort23]> {
  let Latency = 32;
  let NumMicroOps = 4;
  let ResourceCycles = [2, 1, 1];
}
def : InstRW<[WriteVSQRTPDYm], (instregex "VSQRTPDYm")>;

// RSQRT SS/PS.
// x,x.
def WriteRSQRTr : SchedWriteRes<[HWPort0]> {
  let Latency = 5;
}
def : InstRW<[WriteRSQRTr], (instregex "(V?)RSQRT(SS|PS)r(_Int)?")>;

// x,m128.
def WriteRSQRTm : SchedWriteRes<[HWPort0, HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1, 1];
}
def : InstRW<[WriteRSQRTm], (instregex "(V?)RSQRT(SS|PS)m(_Int)?")>;

// RSQRTPS 256.
// y,y.
def WriteRSQRTPSYr : SchedWriteRes<[HWPort0, HWPort15]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}
def : InstRW<[WriteRSQRTPSYr], (instregex "VRSQRTPSYr(_Int)?")>;

// y,m256.
def WriteRSQRTPSYm : SchedWriteRes<[HWPort0, HWPort15, HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [2, 1, 1];
}
def : InstRW<[WriteRSQRTPSYm], (instregex "VRSQRTPSYm(_Int)?")>;

//-- Logic instructions --//

// AND, ANDN, OR, XOR PS/PD.
// x,x / v,v,v.
def : InstRW<[WriteP5], (instregex "(V?)(AND|ANDN|OR|XOR)P(S|D)(Y?)rr")>;
// x,m / v,v,m.
def : InstRW<[WriteP5Ld, ReadAfterLd],
                         (instregex "(V?)(AND|ANDN|OR|XOR)P(S|D)(Y?)rm")>;

//-- Other instructions --//

// VZEROUPPER.
def WriteVZEROUPPER : SchedWriteRes<[]> {
  let NumMicroOps = 4;
}
def : InstRW<[WriteVZEROUPPER], (instregex "VZEROUPPER")>;

// VZEROALL.
def WriteVZEROALL : SchedWriteRes<[]> {
  let NumMicroOps = 12;
}
def : InstRW<[WriteVZEROALL], (instregex "VZEROALL")>;

// LDMXCSR.
def WriteLDMXCSR : SchedWriteRes<[HWPort0, HWPort6, HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [1, 1, 1];
}
def : InstRW<[WriteLDMXCSR], (instregex "(V)?LDMXCSR")>;

// STMXCSR.
def WriteSTMXCSR : SchedWriteRes<[HWPort0, HWPort4, HWPort6, HWPort237]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1, 1, 1, 1];
}
def : InstRW<[WriteSTMXCSR], (instregex "(V)?STMXCSR")>;

} // SchedModel
