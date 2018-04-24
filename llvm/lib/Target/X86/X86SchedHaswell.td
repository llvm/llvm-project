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
  let LoadLatency = 5;
  let MispredictPenalty = 16;

  // Based on the LSD (loop-stream detector) queue size and benchmarking data.
  let LoopMicroOpBufferSize = 50;

  // This flag is set to allow the scheduler to assign a default model to
  // unrecognized opcodes.
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
// FP division and sqrt on port 0.
def HWFPDivider : ProcResource<1>;

// Loads are 5 cycles, so ReadAfterLd registers needn't be available until 5
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 5>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass HWWriteResPair<X86FoldableSchedWrite SchedRW,
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
  def : WriteRes<SchedRW.Folded, !listconcat([HWPort23], ExePorts)> {
    let Latency = !add(Lat, LoadLat);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = !add(UOps, 1);
  }
}

// A folded store needs a cycle on port 4 for the store data, and an extra port
// 2/3/7 cycle to recompute the address.
def : WriteRes<WriteRMW, [HWPort237,HWPort4]>;

// Store_addr on 237.
// Store_data on 4.
def : WriteRes<WriteStore, [HWPort237, HWPort4]>;
def : WriteRes<WriteLoad,  [HWPort23]> { let Latency = 5; }
def : WriteRes<WriteMove,  [HWPort0156]>;
def : WriteRes<WriteZero,  []>;

defm : HWWriteResPair<WriteALU,   [HWPort0156], 1>;
defm : HWWriteResPair<WriteIMul,  [HWPort1],   3>;
def  : WriteRes<WriteIMulH, []> { let Latency = 3; }
defm : HWWriteResPair<WriteShift, [HWPort06],  1>;
defm : HWWriteResPair<WriteJump,  [HWPort06],   1>;
defm : HWWriteResPair<WriteCRC32, [HWPort1],   3>;

defm : HWWriteResPair<WriteCMOV,  [HWPort06,HWPort0156], 2, [1,1], 2>; // Conditional move.
def  : WriteRes<WriteSETCC, [HWPort06]>; // Setcc.
def  : WriteRes<WriteSETCCStore, [HWPort06,HWPort4,HWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

// This is for simple LEAs with one or two input operands.
// The complex ones can only execute on port 1, and they require two cycles on
// the port to read all inputs. We don't model that.
def : WriteRes<WriteLEA, [HWPort15]>;

// Bit counts.
defm : HWWriteResPair<WriteBitScan, [HWPort1], 3>;
defm : HWWriteResPair<WriteLZCNT,   [HWPort1], 3>;
defm : HWWriteResPair<WriteTZCNT,   [HWPort1], 3>;
defm : HWWriteResPair<WritePOPCNT,  [HWPort1], 3>;

// BMI1 BEXTR, BMI2 BZHI
defm : HWWriteResPair<WriteBEXTR, [HWPort06,HWPort15], 2, [1,1], 2>;
defm : HWWriteResPair<WriteBZHI, [HWPort15], 1>;

// This is quite rough, latency depends on the dividend.
defm : HWWriteResPair<WriteIDiv,  [HWPort0, HWDivider], 25, [1,10], 1, 4>;
// Scalar and vector floating point.
def  : WriteRes<WriteFStore,       [HWPort237, HWPort4]>;
def  : WriteRes<WriteFLoad,        [HWPort23]> { let Latency = 5; }
def  : WriteRes<WriteFMove,        [HWPort5]>;

defm : HWWriteResPair<WriteFAdd,   [HWPort1], 3>;
defm : HWWriteResPair<WriteFCmp,   [HWPort1], 3, [1], 1, 6>;
defm : HWWriteResPair<WriteFCom,   [HWPort1], 3>;
defm : HWWriteResPair<WriteFMul,   [HWPort0], 5>;
defm : HWWriteResPair<WriteFDiv,   [HWPort0], 12>; // 10-14 cycles.
defm : HWWriteResPair<WriteFRcp,   [HWPort0], 5>;
defm : HWWriteResPair<WriteFRsqrt, [HWPort0], 5>;
defm : HWWriteResPair<WriteFSqrt,  [HWPort0], 15>;
defm : HWWriteResPair<WriteCvtF2I, [HWPort1], 3>;
defm : HWWriteResPair<WriteCvtI2F, [HWPort1], 4>;
defm : HWWriteResPair<WriteCvtF2F, [HWPort1], 3>;
defm : HWWriteResPair<WriteFMA,   [HWPort01], 5>;
defm : HWWriteResPair<WriteFSign,  [HWPort0], 1>;
defm : HWWriteResPair<WriteFLogic, [HWPort5], 1, [1], 1, 6>;
defm : HWWriteResPair<WriteFShuffle,  [HWPort5],  1>;
defm : HWWriteResPair<WriteFVarShuffle,  [HWPort5],  1>;
defm : HWWriteResPair<WriteFBlend,  [HWPort015], 1, [1], 1, 6>;
defm : HWWriteResPair<WriteFShuffle256,  [HWPort5],  3>;
defm : HWWriteResPair<WriteFVarShuffle256,  [HWPort5],  3>;
defm : HWWriteResPair<WriteFVarBlend,  [HWPort5], 2, [2], 2, 6>;

def  : WriteRes<WriteCvtF2FSt, [HWPort1,HWPort4,HWPort5,HWPort237]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}

// Vector integer operations.
def  : WriteRes<WriteVecStore,       [HWPort237, HWPort4]>;
def  : WriteRes<WriteVecLoad,        [HWPort23]> { let Latency = 5; }
def  : WriteRes<WriteVecMove,        [HWPort015]>;

defm : HWWriteResPair<WriteVecShift, [HWPort0],  1>;
defm : HWWriteResPair<WriteVecLogic, [HWPort015], 1, [1], 1, 6>;
defm : HWWriteResPair<WriteVecALU,   [HWPort15],  1>;
defm : HWWriteResPair<WriteVecIMul,  [HWPort0],   5>;
defm : HWWriteResPair<WritePMULLD,   [HWPort0], 10, [2], 2, 6>;
defm : HWWriteResPair<WriteShuffle,  [HWPort5],  1>;
defm : HWWriteResPair<WriteVarShuffle,  [HWPort5],  1>;
defm : HWWriteResPair<WriteBlend,  [HWPort5], 1, [1], 1, 6>;
defm : HWWriteResPair<WriteShuffle256,  [HWPort5],  3>;
defm : HWWriteResPair<WriteVarShuffle256,  [HWPort5],  3>;
defm : HWWriteResPair<WriteVarBlend,  [HWPort5], 2, [2], 2, 6>;
defm : HWWriteResPair<WriteVarVecShift,  [HWPort0, HWPort5], 2, [2, 1]>;
defm : HWWriteResPair<WriteMPSAD,  [HWPort0, HWPort5], 7, [1, 2], 3, 6>;
defm : HWWriteResPair<WritePSADBW, [HWPort0], 5>;

// Vector insert/extract operations.
def : WriteRes<WriteVecInsert, [HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteVecInsertLd, [HWPort5,HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
}

def : WriteRes<WriteVecExtract, [HWPort0,HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : WriteRes<WriteVecExtractSt, [HWPort4,HWPort5,HWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

// String instructions.

// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [HWPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [HWPort0, HWPort23]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [HWPort0, HWPort5, HWPort015, HWPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def : WriteRes<WritePCmpEStrMLd, [HWPort0, HWPort5, HWPort23, HWPort015, HWPort0156]> {
  let Latency = 25;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [HWPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [HWPort0, HWPort23]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [HWPort0, HWPort5, HWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def : WriteRes<WritePCmpEStrILd, [HWPort0, HWPort5, HWPort23, HWPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}

// MOVMSK Instructions.
def : WriteRes<WriteFMOVMSK, [HWPort0]> { let Latency = 3; }
def : WriteRes<WriteVecMOVMSK, [HWPort0]> { let Latency = 3; }
def : WriteRes<WriteMMXMOVMSK, [HWPort0]> { let Latency = 1; }

// AES Instructions.
def : WriteRes<WriteAESDecEnc, [HWPort5]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [HWPort5, HWPort23]> {
  let Latency = 13;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

def : WriteRes<WriteAESIMC, [HWPort5]> {
  let Latency = 14;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [HWPort5, HWPort23]> {
  let Latency = 20;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}

def : WriteRes<WriteAESKeyGen, [HWPort0,HWPort5,HWPort015]> {
  let Latency = 29;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,2];
}
def : WriteRes<WriteAESKeyGenLd, [HWPort0,HWPort5,HWPort23,HWPort015]> {
  let Latency = 34;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,1,1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [HWPort0, HWPort5]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def : WriteRes<WriteCLMulLd, [HWPort0, HWPort5, HWPort23]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}

// Load/store MXCSR.
def : WriteRes<WriteLDMXCSR, [HWPort0,HWPort23,HWPort0156]> { let Latency = 7; let NumMicroOps = 3; let ResourceCycles = [1,1,1]; }
def : WriteRes<WriteSTMXCSR, [HWPort4,HWPort5,HWPort237]> { let Latency = 2; let NumMicroOps = 3; let ResourceCycles = [1,1,1]; }

def : WriteRes<WriteSystem,     [HWPort0156]> { let Latency = 100; }
def : WriteRes<WriteMicrocoded, [HWPort0156]> { let Latency = 100; }
def : WriteRes<WriteFence,  [HWPort23, HWPort4]>;
def : WriteRes<WriteNop, []>;

//================ Exceptions ================//

//-- Specific Scheduling Models --//

// Starting with P0.
def HWWriteP0 : SchedWriteRes<[HWPort0]>;

def HWWriteP01 : SchedWriteRes<[HWPort01]>;

def HWWrite2P01 : SchedWriteRes<[HWPort01]> {
  let NumMicroOps = 2;
}
def HWWrite3P01 : SchedWriteRes<[HWPort01]> {
  let NumMicroOps = 3;
}

def HWWriteP0156_P23 : SchedWriteRes<[HWPort0156, HWPort23]> {
  let NumMicroOps = 2;
}

def HWWrite2P0156_P23 : SchedWriteRes<[HWPort0156, HWPort23]> {
  let NumMicroOps = 3;
  let ResourceCycles = [2, 1];
}

// Starting with P1.
def HWWriteP1 : SchedWriteRes<[HWPort1]>;


def HWWrite2P1 : SchedWriteRes<[HWPort1]> {
  let NumMicroOps = 2;
  let ResourceCycles = [2];
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

// XLAT.
def HWWriteXLAT : SchedWriteRes<[]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[HWWriteXLAT], (instregex "XLAT")>;

// PUSHA.
def HWWritePushA : SchedWriteRes<[]> {
  let NumMicroOps = 19;
}
def : InstRW<[HWWritePushA], (instregex "PUSHA(16|32)")>;

// POPA.
def HWWritePopA : SchedWriteRes<[]> {
  let NumMicroOps = 18;
}
def : InstRW<[HWWritePopA], (instregex "POPA(16|32)")>;

//-- Arithmetic instructions --//

// DIV.
// r8.
def HWWriteDiv8 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 22;
  let NumMicroOps = 9;
}
def : InstRW<[HWWriteDiv8], (instregex "DIV8r")>;

// IDIV.
// r8.
def HWWriteIDiv8 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 23;
  let NumMicroOps = 9;
}
def : InstRW<[HWWriteIDiv8], (instregex "IDIV8r")>;

// BT.
// m,r.
def HWWriteBTmr : SchedWriteRes<[]> {
  let NumMicroOps = 10;
}
def : InstRW<[HWWriteBTmr], (instregex "BT(16|32|64)mr")>;

// BTR BTS BTC.
// m,r.
def HWWriteBTRSCmr : SchedWriteRes<[]> {
  let NumMicroOps = 11;
}
def : InstRW<[HWWriteBTRSCmr], (instregex "BT(R|S|C)(16|32|64)mr")>;

//-- Control transfer instructions --//

// CALL.
// i.
def HWWriteRETI : SchedWriteRes<[HWPort23, HWPort6, HWPort015]> {
  let NumMicroOps = 4;
  let ResourceCycles = [1, 2, 1];
}
def : InstRW<[HWWriteRETI], (instregex "RETI(L|Q|W)", "LRETI(L|Q|W)")>;

// BOUND.
// r,m.
def HWWriteBOUND : SchedWriteRes<[]> {
  let NumMicroOps = 15;
}
def : InstRW<[HWWriteBOUND], (instregex "BOUNDS(16|32)rm")>;

// INTO.
def HWWriteINTO : SchedWriteRes<[]> {
  let NumMicroOps = 4;
}
def : InstRW<[HWWriteINTO], (instregex "INTO")>;

//-- String instructions --//

// LODSB/W.
def : InstRW<[HWWrite2P0156_P23], (instregex "LODS(B|W)")>;

// LODSD/Q.
def : InstRW<[HWWriteP0156_P23], (instregex "LODS(L|Q)")>;

// MOVS.
def HWWriteMOVS : SchedWriteRes<[HWPort23, HWPort4, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 5;
  let ResourceCycles = [2, 1, 2];
}
def : InstRW<[HWWriteMOVS], (instrs MOVSB, MOVSL, MOVSQ, MOVSW)>;

// CMPS.
def HWWriteCMPS : SchedWriteRes<[HWPort23, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 5;
  let ResourceCycles = [2, 3];
}
def : InstRW<[HWWriteCMPS], (instregex "CMPS(B|L|Q|W)")>;

//-- Other --//

// RDPMC.f
def HWWriteRDPMC : SchedWriteRes<[]> {
  let NumMicroOps = 34;
}
def : InstRW<[HWWriteRDPMC], (instregex "RDPMC")>;

// RDRAND.
def HWWriteRDRAND : SchedWriteRes<[HWPort23, HWPort015]> {
  let NumMicroOps = 17;
  let ResourceCycles = [1, 16];
}
def : InstRW<[HWWriteRDRAND], (instregex "RDRAND(16|32|64)r")>;

//=== Floating Point x87 Instructions ===//
//-- Move instructions --//

// FLD.
// m80.
def : InstRW<[HWWriteP01], (instregex "LD_Frr")>;

// FBLD.
// m80.
def HWWriteFBLD : SchedWriteRes<[]> {
  let Latency = 47;
  let NumMicroOps = 43;
}
def : InstRW<[HWWriteFBLD], (instregex "FBLDm")>;

// FST(P).
// r.
def : InstRW<[HWWriteP01], (instregex "ST_(F|FP)rr")>;

// FLDZ.
def : InstRW<[HWWriteP01], (instregex "LD_F0")>;

// FLDPI FLDL2E etc.
def : InstRW<[HWWrite2P01], (instregex "FLDPI", "FLDL2(T|E)", "FLDL(G|N)2")>;

// FFREE.
def : InstRW<[HWWriteP01], (instregex "FFREE")>;

// FNSAVE.
def HWWriteFNSAVE : SchedWriteRes<[]> {
  let NumMicroOps = 147;
}
def : InstRW<[HWWriteFNSAVE], (instregex "FSAVEm")>;

// FRSTOR.
def HWWriteFRSTOR : SchedWriteRes<[]> {
  let NumMicroOps = 90;
}
def : InstRW<[HWWriteFRSTOR], (instregex "FRSTORm")>;

//-- Arithmetic instructions --//

// FCOMPP FUCOMPP.
// r.
def : InstRW<[HWWrite2P01], (instregex "FCOMPP", "UCOM_FPPr")>;

// FCOMI(P) FUCOMI(P).
// m.
def : InstRW<[HWWrite3P01], (instregex "COM_FIr", "COM_FIPr", "UCOM_FIr",
                             "UCOM_FIPr")>;

// FTST.
def : InstRW<[HWWriteP1], (instregex "TST_F")>;

// FXAM.
def : InstRW<[HWWrite2P1], (instrs FXAM)>;

// FPREM.
def HWWriteFPREM : SchedWriteRes<[]> {
  let Latency = 19;
  let NumMicroOps = 28;
}
def : InstRW<[HWWriteFPREM], (instrs FPREM)>;

// FPREM1.
def HWWriteFPREM1 : SchedWriteRes<[]> {
  let Latency = 27;
  let NumMicroOps = 41;
}
def : InstRW<[HWWriteFPREM1], (instrs FPREM1)>;

// FRNDINT.
def HWWriteFRNDINT : SchedWriteRes<[]> {
  let Latency = 11;
  let NumMicroOps = 17;
}
def : InstRW<[HWWriteFRNDINT], (instrs FRNDINT)>;

//-- Math instructions --//

// FSCALE.
def HWWriteFSCALE : SchedWriteRes<[]> {
  let Latency = 75; // 49-125
  let NumMicroOps = 50; // 25-75
}
def : InstRW<[HWWriteFSCALE], (instrs FSCALE)>;

// FXTRACT.
def HWWriteFXTRACT : SchedWriteRes<[]> {
  let Latency = 15;
  let NumMicroOps = 17;
}
def : InstRW<[HWWriteFXTRACT], (instrs FXTRACT)>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : HWWriteResPair<WriteFHAdd,  [HWPort1, HWPort5], 5, [1,2], 3, 6>;
defm : HWWriteResPair<WritePHAdd, [HWPort5, HWPort15], 3, [2,1], 3, 6>;

//=== Floating Point XMM and YMM Instructions ===//

// Remaining instrs.

def HWWriteResGroup0 : SchedWriteRes<[HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup0], (instregex "VBROADCASTSSrm",
                                           "(V?)LDDQUrm",
                                           "(V?)MOVAPDrm",
                                           "(V?)MOVAPSrm",
                                           "(V?)MOVDQArm",
                                           "(V?)MOVDQUrm",
                                           "(V?)MOVNTDQArm",
                                           "(V?)MOVSHDUPrm",
                                           "(V?)MOVSLDUPrm",
                                           "(V?)MOVUPDrm",
                                           "(V?)MOVUPSrm",
                                           "VPBROADCASTDrm",
                                           "VPBROADCASTQrm",
                                           "(V?)ROUNDPD(Y?)r",
                                           "(V?)ROUNDPS(Y?)r",
                                           "(V?)ROUNDSDr",
                                           "(V?)ROUNDSSr")>;

def HWWriteResGroup0_1 : SchedWriteRes<[HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup0_1], (instregex "LD_F32m",
                                             "LD_F64m",
                                             "LD_F80m",
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

def HWWriteResGroup0_2 : SchedWriteRes<[HWPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVSX(16|32|64)rm16",
                                             "MOVSX(16|32|64)rm32",
                                             "MOVSX(16|32|64)rm8",
                                             "MOVZX(16|32|64)rm16",
                                             "MOVZX(16|32|64)rm8",
                                             "(V?)MOVDDUPrm")>;

def HWWriteResGroup1 : SchedWriteRes<[HWPort4,HWPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup1], (instregex "FBSTPm",
                                           "MMX_MOVD64from64rm",
                                           "MMX_MOVD64mr",
                                           "MMX_MOVNTQmr",
                                           "MMX_MOVQ64mr",
                                           "MOVNTI_64mr",
                                           "MOVNTImr",
                                           "ST_FP32m",
                                           "ST_FP64m",
                                           "ST_FP80m",
                                           "VEXTRACTF128mr",
                                           "VEXTRACTI128mr",
                                           "(V?)MOVAPD(Y?)mr",
                                           "(V?)MOVAPS(V?)mr",
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

def HWWriteResGroup2 : SchedWriteRes<[HWPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup2], (instregex "MMX_MOVD64from64rr",
                                           "MMX_MOVD64grr",
                                           "(V?)MOVPDI2DIrr",
                                           "(V?)MOVPQIto64rr",
                                           "VPSLLVQ(Y?)rr",
                                           "VPSRLVQ(Y?)rr",
                                           "VTESTPD(Y?)rr",
                                           "VTESTPS(Y?)rr")>;

def HWWriteResGroup3 : SchedWriteRes<[HWPort1]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup3], (instregex "COMP_FST0r",
                                           "COM_FST0r",
                                           "UCOM_FPr",
                                           "UCOM_Fr")>;

def HWWriteResGroup4 : SchedWriteRes<[HWPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup4], (instregex "MMX_MOVD64rr",
                                           "MMX_MOVD64to64rr",
                                           "MMX_MOVQ2DQrr",
                                           "(V?)MOV64toPQIrr",
                                           "(V?)MOVDI2PDIrr",
                                           "(V?)PSLLDQ(Y?)ri",
                                           "(V?)PSRLDQ(Y?)ri")>;

def HWWriteResGroup5 : SchedWriteRes<[HWPort6]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup5], (instregex "JMP(16|32|64)r")>;

def HWWriteResGroup6 : SchedWriteRes<[HWPort01]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup6], (instrs FINCSTP, FNOP)>;

def HWWriteResGroup7 : SchedWriteRes<[HWPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup7], (instrs CDQ, CQO)>;
def: InstRW<[HWWriteResGroup7], (instregex "BT(16|32|64)ri8",
                                           "BT(16|32|64)rr",
                                           "BTC(16|32|64)ri8",
                                           "BTC(16|32|64)rr",
                                           "BTR(16|32|64)ri8",
                                           "BTR(16|32|64)rr",
                                           "BTS(16|32|64)ri8",
                                           "BTS(16|32|64)rr",
                                           "SAR(8|16|32|64)r1",
                                           "SAR(8|16|32|64)ri",
                                           "SHL(8|16|32|64)r1",
                                           "SHL(8|16|32|64)ri",
                                           "SHR(8|16|32|64)r1",
                                           "SHR(8|16|32|64)ri")>;

def HWWriteResGroup8 : SchedWriteRes<[HWPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup8], (instregex "ANDN(32|64)rr",
                                           "BLSI(32|64)rr",
                                           "BLSMSK(32|64)rr",
                                           "BLSR(32|64)rr")>;

def HWWriteResGroup9 : SchedWriteRes<[HWPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup9], (instregex "MMX_MOVQ64rr",
                                           "VPBLENDD(Y?)rri")>;

def HWWriteResGroup10 : SchedWriteRes<[HWPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup10], (instrs CBW, CWDE, CDQE)>;
def: InstRW<[HWWriteResGroup10], (instregex "CLC",
                                            "CMC",
                                            "LAHF", // TODO: This doesn't match Agner's data
                                            "NOOP",
                                            "SAHF", // TODO: This doesn't match Agner's data
                                            "SGDT64m",
                                            "SIDT64m",
                                            "SLDT64m",
                                            "SMSW16m",
                                            "STC",
                                            "STRm",
                                            "SYSCALL")>;

def HWWriteResGroup11 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup11], (instregex "VCVTPH2PSrm",
                                            "(V?)CVTPS2PDrm")>;

def HWWriteResGroup11_1 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup11_1], (instregex "VCVTPH2PSYrm",
                                              "(V?)CVTSS2SDrm",
                                              "VPSLLVQrm",
                                              "VPSRLVQrm",
                                              "VTESTPDrm",
                                              "VTESTPSrm")>;

def HWWriteResGroup11_2 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSLLDYrm",
                                              "VPSLLQYrm",
                                              "VPSLLVQYrm",
                                              "VPSLLWYrm",
                                              "VPSRADYrm",
                                              "VPSRAWYrm",
                                              "VPSRLDYrm",
                                              "VPSRLQYrm",
                                              "VPSRLVQYrm",
                                              "VPSRLWYrm",
                                              "VTESTPDYrm",
                                              "VTESTPSYrm")>;

def HWWriteResGroup12 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup12], (instregex "FCOM32m",
                                            "FCOM64m",
                                            "FCOMP32m",
                                            "FCOMP64m",
                                            "MMX_CVTPI2PSirm",
                                            "PDEP(32|64)rm",
                                            "PEXT(32|64)rm",
                                            "(V?)CMPSDrm",
                                            "(V?)CMPSSrm",
                                            "(V?)MAX(C?)SDrm",
                                            "(V?)MAX(C?)SSrm",
                                            "(V?)MIN(C?)SDrm",
                                            "(V?)MIN(C?)SSrm")>;

def HWWriteResGroup12_1 : SchedWriteRes<[HWPort1,HWPort0156,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup12_1], (instrs IMUL16rmi, IMUL16rmi8)>;

def HWWriteResGroup12_2 : SchedWriteRes<[HWPort1,HWPort06,HWPort0156,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,2,1];
}
def: InstRW<[HWWriteResGroup12_2], (instrs IMUL16m, MUL16m)>;

def HWWriteResGroup13 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup13], (instregex "PUNPCKLWDrm",
                                            "(V?)INSERTPSrm",
                                            "(V?)PACKSSDWrm",
                                            "(V?)PACKSSWBrm",
                                            "(V?)PACKUSDWrm",
                                            "(V?)PACKUSWBrm",
                                            "(V?)PALIGNRrmi",
                                            "VPERMILPDmi",
                                            "VPERMILPDrm",
                                            "VPERMILPSmi",
                                            "VPERMILPSrm",
                                            "(V?)PSHUFBrm",
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
                                            "(V?)PUNPCKLWDrm",
                                            "(V?)SHUFPDrmi",
                                            "(V?)SHUFPSrmi",
                                            "(V?)UNPCKHPDrm",
                                            "(V?)UNPCKHPSrm",
                                            "(V?)UNPCKLPDrm",
                                            "(V?)UNPCKLPSrm")>;

def HWWriteResGroup13_1 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup13_1], (instregex "VANDNPDYrm",
                                              "VANDNPSYrm",
                                              "VANDPDYrm",
                                              "VANDPSYrm",
                                              "VORPDYrm",
                                              "VORPSYrm",
                                              "VPACKSSDWYrm",
                                              "VPACKSSWBYrm",
                                              "VPACKUSDWYrm",
                                              "VPACKUSWBYrm",
                                              "VPALIGNRYrmi",
                                              "VPBLENDWYrmi",
                                              "VPERMILPDYmi",
                                              "VPERMILPDYrm",
                                              "VPERMILPSYmi",
                                              "VPERMILPSYrm",
                                              "VPMOVSXBDYrm",
                                              "VPMOVSXBQYrm",
                                              "VPMOVSXWQYrm",
                                              "VPSHUFBYrm",
                                              "VPSHUFDYmi",
                                              "VPSHUFHWYmi",
                                              "VPSHUFLWYmi",
                                              "VPUNPCKHBWYrm",
                                              "VPUNPCKHDQYrm",
                                              "VPUNPCKHQDQYrm",
                                              "VPUNPCKHWDYrm",
                                              "VPUNPCKLBWYrm",
                                              "VPUNPCKLDQYrm",
                                              "VPUNPCKLQDQYrm",
                                              "VPUNPCKLWDYrm",
                                              "VSHUFPDYrmi",
                                              "VSHUFPSYrmi",
                                              "VUNPCKHPDYrm",
                                              "VUNPCKHPSYrm",
                                              "VUNPCKLPDYrm",
                                              "VUNPCKLPSYrm",
                                              "VXORPDYrm",
                                              "VXORPSYrm")>;

def HWWriteResGroup14 : SchedWriteRes<[HWPort6,HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup14], (instregex "FARJMP64",
                                            "JMP(16|32|64)m")>;

def HWWriteResGroup15 : SchedWriteRes<[HWPort23,HWPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup15], (instregex "BT(16|32|64)mi8")>;

def HWWriteResGroup16 : SchedWriteRes<[HWPort23,HWPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup16], (instregex "ANDN(32|64)rm",
                                            "BLSI(32|64)rm",
                                            "BLSMSK(32|64)rm",
                                            "BLSR(32|64)rm",
                                            "MOVBE(16|32|64)rm")>;

def HWWriteResGroup16_1 : SchedWriteRes<[HWPort23,HWPort15]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup16_1], (instregex "(V?)PABSBrm",
                                              "(V?)PABSDrm",
                                              "(V?)PABSWrm",
                                              "(V?)PADDBrm",
                                              "(V?)PADDDrm",
                                              "(V?)PADDQrm",
                                              "(V?)PADDSBrm",
                                              "(V?)PADDSWrm",
                                              "(V?)PADDUSBrm",
                                              "(V?)PADDUSWrm",
                                              "(V?)PADDWrm",
                                              "(V?)PAVGBrm",
                                              "(V?)PAVGWrm",
                                              "(V?)PCMPEQBrm",
                                              "(V?)PCMPEQDrm",
                                              "(V?)PCMPEQQrm",
                                              "(V?)PCMPEQWrm",
                                              "(V?)PCMPGTBrm",
                                              "(V?)PCMPGTDrm",
                                              "(V?)PCMPGTWrm",
                                              "(V?)PMAXSBrm",
                                              "(V?)PMAXSDrm",
                                              "(V?)PMAXSWrm",
                                              "(V?)PMAXUBrm",
                                              "(V?)PMAXUDrm",
                                              "(V?)PMAXUWrm",
                                              "(V?)PMINSBrm",
                                              "(V?)PMINSDrm",
                                              "(V?)PMINSWrm",
                                              "(V?)PMINUBrm",
                                              "(V?)PMINUDrm",
                                              "(V?)PMINUWrm",
                                              "(V?)PSIGNBrm",
                                              "(V?)PSIGNDrm",
                                              "(V?)PSIGNWrm",
                                              "(V?)PSUBBrm",
                                              "(V?)PSUBDrm",
                                              "(V?)PSUBQrm",
                                              "(V?)PSUBSBrm",
                                              "(V?)PSUBSWrm",
                                              "(V?)PSUBUSBrm",
                                              "(V?)PSUBUSWrm",
                                              "(V?)PSUBWrm")>;

def HWWriteResGroup16_2 : SchedWriteRes<[HWPort23,HWPort15]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup16_2], (instregex "VPABSBYrm",
                                              "VPABSDYrm",
                                              "VPABSWYrm",
                                              "VPADDBYrm",
                                              "VPADDDYrm",
                                              "VPADDQYrm",
                                              "VPADDSBYrm",
                                              "VPADDSWYrm",
                                              "VPADDUSBYrm",
                                              "VPADDUSWYrm",
                                              "VPADDWYrm",
                                              "VPAVGBYrm",
                                              "VPAVGWYrm",
                                              "VPCMPEQBYrm",
                                              "VPCMPEQDYrm",
                                              "VPCMPEQQYrm",
                                              "VPCMPEQWYrm",
                                              "VPCMPGTBYrm",
                                              "VPCMPGTDYrm",
                                              "VPCMPGTWYrm",
                                              "VPMAXSBYrm",
                                              "VPMAXSDYrm",
                                              "VPMAXSWYrm",
                                              "VPMAXUBYrm",
                                              "VPMAXUDYrm",
                                              "VPMAXUWYrm",
                                              "VPMINSBYrm",
                                              "VPMINSDYrm",
                                              "VPMINSWYrm",
                                              "VPMINUBYrm",
                                              "VPMINUDYrm",
                                              "VPMINUWYrm",
                                              "VPSIGNBYrm",
                                              "VPSIGNDYrm",
                                              "VPSIGNWYrm",
                                              "VPSUBBYrm",
                                              "VPSUBDYrm",
                                              "VPSUBQYrm",
                                              "VPSUBSBYrm",
                                              "VPSUBSWYrm",
                                              "VPSUBUSBYrm",
                                              "VPSUBUSWYrm",
                                              "VPSUBWYrm")>;

def HWWriteResGroup17 : SchedWriteRes<[HWPort23,HWPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup17], (instregex "VINSERTF128rm",
                                            "VINSERTI128rm",
                                            "VPBLENDDrmi")>;

def HWWriteResGroup17_1 : SchedWriteRes<[HWPort23,HWPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup17_1], (instregex "MMX_PANDNirm",
                                              "MMX_PANDirm",
                                              "MMX_PORirm",
                                              "MMX_PXORirm")>;

def HWWriteResGroup17_2 : SchedWriteRes<[HWPort23,HWPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup17_2], (instregex "VBLENDPDYrmi",
                                              "VBLENDPSYrmi",
                                              "VPANDNYrm",
                                              "VPANDYrm",
                                              "VPBLENDDYrmi",
                                              "VPORYrm",
                                              "VPXORYrm")>;

def HWWriteResGroup18 : SchedWriteRes<[HWPort23,HWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup18], (instrs POP16r, POP32r, POP64r)>;
def: InstRW<[HWWriteResGroup18], (instregex "POP(16|32|64)rmr")>;

def HWWriteResGroup19 : SchedWriteRes<[HWPort237,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup19], (instregex "SFENCE")>;

def HWWriteResGroup21 : SchedWriteRes<[HWPort4,HWPort6,HWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup21], (instregex "FNSTCW16m")>;

def HWWriteResGroup23 : SchedWriteRes<[HWPort4,HWPort237,HWPort15]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup23], (instregex "MOVBE(32|64)mr")>;

def HWWriteResGroup23_16 : SchedWriteRes<[HWPort06, HWPort237, HWPort4]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup23_16], (instregex "MOVBE16mr")>;

def HWWriteResGroup24 : SchedWriteRes<[HWPort4,HWPort237,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup24], (instrs PUSH16r, PUSH32r, PUSH64r)>;
def: InstRW<[HWWriteResGroup24], (instregex "PUSH(16|32|64)rmr",
                                            "PUSH64i8",
                                            "STOSB",
                                            "STOSL",
                                            "STOSQ",
                                            "STOSW")>;

def HWWriteResGroup25 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort06]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup25], (instregex "BTC(16|32|64)mi8",
                                            "BTR(16|32|64)mi8",
                                            "BTS(16|32|64)mi8",
                                            "SAR(8|16|32|64)m1",
                                            "SAR(8|16|32|64)mi",
                                            "SHL(8|16|32|64)m1",
                                            "SHL(8|16|32|64)mi",
                                            "SHR(8|16|32|64)m1",
                                            "SHR(8|16|32|64)mi")>;

def HWWriteResGroup26 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup26], (instregex "POP(16|32|64)rmm",
                                            "PUSH(16|32|64)rmm")>;

def HWWriteResGroup28 : SchedWriteRes<[HWPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup28], (instrs FDECSTP)>;

def HWWriteResGroup29 : SchedWriteRes<[HWPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup29], (instregex "ROL(8|16|32|64)r1",
                                            "ROL(8|16|32|64)ri",
                                            "ROR(8|16|32|64)r1",
                                            "ROR(8|16|32|64)ri")>;

def HWWriteResGroup30 : SchedWriteRes<[HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup30], (instregex "LFENCE")>;
def: InstRW<[HWWriteResGroup30], (instregex "MFENCE")>;
def: InstRW<[HWWriteResGroup30], (instrs WAIT)>;
def: InstRW<[HWWriteResGroup30], (instregex "XGETBV")>;

def HWWriteResGroup31 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup31], (instregex "VCVTPH2PSYrr",
                                            "VCVTPH2PSrr",
                                            "(V?)CVTPS2PDrr",
                                            "(V?)CVTSS2SDrr",
                                            "(V?)PSLLDrr",
                                            "(V?)PSLLQrr",
                                            "(V?)PSLLWrr",
                                            "(V?)PSRADrr",
                                            "(V?)PSRAWrr",
                                            "(V?)PSRLDrr",
                                            "(V?)PSRLQrr",
                                            "(V?)PSRLWrr",
                                            "(V?)PTESTrr")>;

def HWWriteResGroup32 : SchedWriteRes<[HWPort6,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup32], (instregex "CLFLUSH")>;

def HWWriteResGroup33 : SchedWriteRes<[HWPort01,HWPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup33], (instregex "MMX_MOVDQ2Qrr")>;

def HWWriteResGroup34 : SchedWriteRes<[HWPort06,HWPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup34], (instrs BSWAP64r)>;

def HWWriteResGroup34_1 : SchedWriteRes<[HWPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup34_1], (instrs BSWAP32r)>;

def HWWriteResGroup35 : SchedWriteRes<[HWPort06,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup35], (instrs CWD, JCXZ, JECXZ, JRCXZ)>;
def: InstRW<[HWWriteResGroup35], (instregex "ADC(8|16|32|64)ri",
                                            "ADC(8|16|32|64)rr",
                                            "ADC(8|16|32|64)i",
                                            "SBB(8|16|32|64)ri",
                                            "SBB(8|16|32|64)rr",
                                            "SBB(8|16|32|64)i",
                                            "SET(A|BE)r")>;

def HWWriteResGroup36 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup36], (instregex "VMASKMOVPDrm",
                                            "VMASKMOVPSrm",
                                            "VPMASKMOVDrm",
                                            "VPMASKMOVQrm")>;

def HWWriteResGroup36_1 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup36_1], (instregex "VBLENDVPDYrm",
                                              "VBLENDVPSYrm",
                                              "VMASKMOVPDYrm",
                                              "VMASKMOVPSYrm",
                                              "VPBLENDVBYrm",
                                              "VPMASKMOVDYrm",
                                              "VPMASKMOVQYrm")>;

def HWWriteResGroup36_2 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup36_2], (instregex "MMX_PACKSSDWirm",
                                              "MMX_PACKSSWBirm",
                                              "MMX_PACKUSWBirm")>;

def HWWriteResGroup37 : SchedWriteRes<[HWPort23,HWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup37], (instrs LEAVE, LEAVE64,
                                         SCASB, SCASL, SCASQ, SCASW)>;

def HWWriteResGroup38 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup38], (instregex "(V?)PSLLDrm",
                                            "(V?)PSLLQrm",
                                            "(V?)PSLLWrm",
                                            "(V?)PSRADrm",
                                            "(V?)PSRAWrm",
                                            "(V?)PSRLDrm",
                                            "(V?)PSRLQrm",
                                            "(V?)PSRLWrm",
                                            "(V?)PTESTrm")>;

def HWWriteResGroup39 : SchedWriteRes<[HWPort0,HWPort01,HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup39], (instregex "FLDCW16m")>;

def HWWriteResGroup41 : SchedWriteRes<[HWPort6,HWPort23,HWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup41], (instregex "LRETQ",
                                            "RETL",
                                            "RETQ")>;

def HWWriteResGroup43 : SchedWriteRes<[HWPort23,HWPort06,HWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup43, ReadAfterLd], (instrs ADC8rm, ADC16rm, ADC32rm, ADC64rm,
                                                      SBB8rm, SBB16rm, SBB32rm, SBB64rm)>;

def HWWriteResGroup44 : SchedWriteRes<[HWPort4,HWPort6,HWPort237,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup44], (instregex "CALL(16|32|64)r")>;

def HWWriteResGroup45 : SchedWriteRes<[HWPort4,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup45], (instregex "CALL64pcrel32",
                                            "SET(A|BE)m")>;

def HWWriteResGroup46 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort06]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[HWWriteResGroup46], (instregex "ROL(8|16|32|64)m1",
                                            "ROL(8|16|32|64)mi",
                                            "ROR(8|16|32|64)m1",
                                            "ROR(8|16|32|64)mi")>;

def HWWriteResGroup47 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[HWWriteResGroup47], (instregex "XADD(8|16|32|64)rm")>;

def HWWriteResGroup48 : SchedWriteRes<[HWPort4,HWPort6,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[HWWriteResGroup48], (instregex "CALL(16|32|64)m",
                                            "FARCALL64")>;

def HWWriteResGroup50 : SchedWriteRes<[HWPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup50], (instregex "MMX_CVTPI2PSirr",
                                            "PDEP(32|64)rr",
                                            "PEXT(32|64)rr",
                                            "SHLD(16|32|64)rri8",
                                            "SHRD(16|32|64)rri8",
                                            "(V?)CVTDQ2PS(Y?)rr")>;

def HWWriteResGroup50_16i : SchedWriteRes<[HWPort1, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup50_16i], (instrs IMUL16rri, IMUL16rri8)>;

def HWWriteResGroup51 : SchedWriteRes<[HWPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup51], (instregex "VPBROADCASTBrr",
                                            "VPBROADCASTWrr",
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

def HWWriteResGroup52 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup52], (instregex "(V?)ADDPDrm",
                                            "(V?)ADDPSrm",
                                            "(V?)ADDSUBPDrm",
                                            "(V?)ADDSUBPSrm",
                                            "(V?)CVTPS2DQrm",
                                            "(V?)CVTTPS2DQrm",
                                            "(V?)SUBPDrm",
                                            "(V?)SUBPSrm")>;

def HWWriteResGroup52_1 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup52_1], (instregex "ADD_F32m",
                                              "ADD_F64m",
                                              "ILD_F16m",
                                              "ILD_F32m",
                                              "ILD_F64m",
                                              "SUBR_F32m",
                                              "SUBR_F64m",
                                              "SUB_F32m",
                                              "SUB_F64m",
                                              "VADDPDYrm",
                                              "VADDPSYrm",
                                              "VADDSUBPDYrm",
                                              "VADDSUBPSYrm",
                                              "VCMPPDYrmi",
                                              "VCMPPSYrmi",
                                              "VCVTDQ2PSYrm",
                                              "VCVTPS2DQYrm",
                                              "VCVTTPS2DQYrm",
                                              "VMAX(C?)PDYrm",
                                              "VMAX(C?)PSYrm",
                                              "VMIN(C?)PDYrm",
                                              "VMIN(C?)PSYrm",
                                              "VSUBPDYrm",
                                              "VSUBPSYrm")>;

def HWWriteResGroup53 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup53], (instregex "VPERM2F128rm",
                                            "VPERM2I128rm",
                                            "VPERMDYrm",
                                            "VPERMPDYmi",
                                            "VPERMPSYrm",
                                            "VPERMQYmi",
                                            "VPMOVZXBDYrm",
                                            "VPMOVZXBQYrm",
                                            "VPMOVZXBWYrm",
                                            "VPMOVZXDQYrm",
                                            "VPMOVZXWQYrm")>;

def HWWriteResGroup53_1 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup53_1], (instregex "VPMOVSXBWYrm",
                                              "VPMOVSXDQYrm",
                                              "VPMOVSXWDYrm",
                                              "VPMOVZXWDYrm")>;

def HWWriteResGroup54 : SchedWriteRes<[HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[HWWriteResGroup54], (instrs XADD8rr, XADD16rr, XADD32rr, XADD64rr,
                                         XCHG8rr, XCHG16rr, XCHG32rr, XCHG64rr,
                                         XCHG16ar, XCHG32ar, XCHG64ar)>;

def HWWriteResGroup55 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup55], (instregex "VPSLLVD(Y?)rr",
                                            "VPSRAVD(Y?)rr",
                                            "VPSRLVD(Y?)rr")>;

def HWWriteResGroup57 : SchedWriteRes<[HWPort5,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup57], (instregex "MMX_PACKSSDWirr",
                                            "MMX_PACKSSWBirr",
                                            "MMX_PACKUSWBirr")>;

def HWWriteResGroup58 : SchedWriteRes<[HWPort6,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup58], (instregex "CLD")>;

def HWWriteResGroup59 : SchedWriteRes<[HWPort06,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup59], (instregex "CMOV(A|BE)(16|32|64)rr",
                                            "RCL(8|16|32|64)r1",
                                            "RCL(8|16|32|64)ri",
                                            "RCR(8|16|32|64)r1",
                                            "RCR(8|16|32|64)ri")>;

def HWWriteResGroup60 : SchedWriteRes<[HWPort06,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup60], (instregex "ROL(8|16|32|64)rCL",
                                            "ROR(8|16|32|64)rCL",
                                            "SAR(8|16|32|64)rCL",
                                            "SHL(8|16|32|64)rCL",
                                            "SHR(8|16|32|64)rCL")>;

def HWWriteResGroup61 : SchedWriteRes<[HWPort0,HWPort4,HWPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup61], (instregex "FNSTSWm")>;

def HWWriteResGroup62 : SchedWriteRes<[HWPort1,HWPort4,HWPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup62], (instregex "ISTT_FP16m",
                                            "ISTT_FP32m",
                                            "ISTT_FP64m",
                                            "IST_F16m",
                                            "IST_F32m",
                                            "IST_FP16m",
                                            "IST_FP32m",
                                            "IST_FP64m")>;

def HWWriteResGroup63 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup63], (instregex "VPSLLVDYrm",
                                            "VPSRAVDYrm",
                                            "VPSRLVDYrm")>;

def HWWriteResGroup63_1 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup63_1], (instregex "VPSLLVDrm",
                                              "VPSRAVDrm",
                                              "VPSRLVDrm")>;

def HWWriteResGroup64 : SchedWriteRes<[HWPort5,HWPort23,HWPort15]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup64], (instregex "MMX_PH(ADD|SUB)(D|SW|W)rm")>;

def HWWriteResGroup64_1 : SchedWriteRes<[HWPort5,HWPort23,HWPort15]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup64_1], (instregex "VPHADDDYrm",
                                              "VPHADDSWYrm",
                                              "VPHADDWYrm",
                                              "VPHSUBDYrm",
                                              "VPHSUBSWYrm",
                                              "VPHSUBWYrm")>;

def HWWriteResGroup65 : SchedWriteRes<[HWPort23,HWPort06,HWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[HWWriteResGroup65], (instregex "CMOV(A|BE)(16|32|64)rm")>;

def HWWriteResGroup66 : SchedWriteRes<[HWPort23,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[HWWriteResGroup66], (instregex "RCL(8|16|32|64)m1",
                                            "RCL(8|16|32|64)mi",
                                            "RCR(8|16|32|64)m1",
                                            "RCR(8|16|32|64)mi")>;

def HWWriteResGroup67 : SchedWriteRes<[HWPort23,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,2,1];
}
def: InstRW<[HWWriteResGroup67], (instregex "ROR(8|16|32|64)mCL")>;

def HWWriteResGroup68 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[HWWriteResGroup68], (instregex "XCHG(8|16|32|64)rm")>;

def HWWriteResGroup69 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[HWWriteResGroup69], (instregex "ADC(8|16|32|64)mi",
                                            "CMPXCHG(8|16|32|64)rm",
                                            "ROL(8|16|32|64)mCL",
                                            "SAR(8|16|32|64)mCL",
                                            "SBB(8|16|32|64)mi",
                                            "SHL(8|16|32|64)mCL",
                                            "SHR(8|16|32|64)mCL")>;
def: InstRW<[HWWriteResGroup69, ReadAfterLd], (instrs ADC8mr, ADC16mr, ADC32mr, ADC64mr,
                                                      SBB8mr, SBB16mr, SBB32mr, SBB64mr)>;

def HWWriteResGroup70 : SchedWriteRes<[HWPort0,HWPort1]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup70], (instregex "(V?)CVTSD2SI64rr",
                                            "(V?)CVTSD2SIrr",
                                            "(V?)CVTSS2SI64rr",
                                            "(V?)CVTSS2SIrr",
                                            "(V?)CVTTSD2SI64rr",
                                            "(V?)CVTTSD2SIrr",
                                            "(V?)CVTTSS2SI64rr",
                                            "(V?)CVTTSS2SIrr")>;

def HWWriteResGroup71 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup71], (instregex "VCVTPS2PDYrr",
                                            "VPSLLDYrr",
                                            "VPSLLQYrr",
                                            "VPSLLWYrr",
                                            "VPSRADYrr",
                                            "VPSRAWYrr",
                                            "VPSRLDYrr",
                                            "VPSRLQYrr",
                                            "VPSRLWYrr",
                                            "VPTESTYrr")>;

def HWWriteResGroup72 : SchedWriteRes<[HWPort0,HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup72], (instregex "FNSTSW16r")>;

def HWWriteResGroup73 : SchedWriteRes<[HWPort1,HWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup73], (instregex "MMX_CVTPD2PIirr",
                                            "MMX_CVTPI2PDirr",
                                            "MMX_CVTPS2PIirr",
                                            "MMX_CVTTPD2PIirr",
                                            "MMX_CVTTPS2PIirr",
                                            "(V?)CVTDQ2PDrr",
                                            "(V?)CVTPD2DQrr",
                                            "(V?)CVTPD2PSrr",
                                            "VCVTPS2PHrr",
                                            "(V?)CVTSD2SSrr",
                                            "(V?)CVTSI642SDrr",
                                            "(V?)CVTSI2SDrr",
                                            "(V?)CVTSI2SSrr",
                                            "(V?)CVTTPD2DQrr")>;

def HWWriteResGroup74 : SchedWriteRes<[HWPort1,HWPort6]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup74], (instrs IMUL64r, MUL64r, MULX64rr)>;

def HWWriteResGroup74_16 : SchedWriteRes<[HWPort1, HWPort06, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[HWWriteResGroup74_16], (instrs IMUL16r, MUL16r)>;

def HWWriteResGroup75 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup75], (instregex "FICOM16m",
                                            "FICOM32m",
                                            "FICOMP16m",
                                            "FICOMP32m")>;

def HWWriteResGroup76 : SchedWriteRes<[HWPort0,HWPort1,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup76], (instregex "(V?)CVTSD2SI64rm",
                                            "(V?)CVTSD2SIrm",
                                            "(V?)CVTSS2SI64rm",
                                            "(V?)CVTSS2SIrm",
                                            "(V?)CVTTSD2SI64rm",
                                            "(V?)CVTTSD2SIrm",
                                            "VCVTTSS2SI64rm",
                                            "(V?)CVTTSS2SIrm")>;

def HWWriteResGroup77 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup77], (instregex "VCVTPS2PDYrm")>;

def HWWriteResGroup77_1 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup77_1], (instregex "VPTESTYrm")>;

def HWWriteResGroup78 : SchedWriteRes<[HWPort1,HWPort5,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup78], (instregex "CVTPD2DQrm",
                                            "CVTPD2PSrm",
                                            "CVTTPD2DQrm",
                                            "MMX_CVTPD2PIirm",
                                            "MMX_CVTTPD2PIirm",
                                            "(V?)CVTDQ2PDrm")>;

def HWWriteResGroup78_1 : SchedWriteRes<[HWPort1,HWPort5,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup78_1], (instregex "MMX_CVTPI2PDirm",
                                              "(V?)CVTSD2SSrm")>;

def HWWriteResGroup79 : SchedWriteRes<[HWPort1,HWPort6,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup79], (instrs IMUL64m, MUL64m, MULX64rm)>;

def HWWriteResGroup80 : SchedWriteRes<[HWPort5,HWPort23,HWPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup80], (instregex "VPBROADCASTBYrm",
                                            "VPBROADCASTBrm",
                                            "VPBROADCASTWYrm",
                                            "VPBROADCASTWrm")>;

def HWWriteResGroup81 : SchedWriteRes<[HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[HWWriteResGroup81], (instrs FNCLEX)>;

def HWWriteResGroup82 : SchedWriteRes<[HWPort015,HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[HWWriteResGroup82], (instregex "VZEROUPPER")>;

def HWWriteResGroup83 : SchedWriteRes<[HWPort1,HWPort6,HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[HWWriteResGroup83], (instregex "LAR(16|32|64)rr")>;

def HWWriteResGroup84 : SchedWriteRes<[HWPort0,HWPort4,HWPort237,HWPort15]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup84], (instregex "VMASKMOVPD(Y?)mr",
                                            "VMASKMOVPS(Y?)mr",
                                            "VPMASKMOVD(Y?)mr",
                                            "VPMASKMOVQ(Y?)mr")>;

def HWWriteResGroup86 : SchedWriteRes<[HWPort1,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup86], (instregex "SHLD(16|32|64)mri8",
                                            "SHRD(16|32|64)mri8")>;

def HWWriteResGroup87 : SchedWriteRes<[HWPort1,HWPort6,HWPort23,HWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[HWWriteResGroup87], (instregex "LAR(16|32|64)rm",
                                            "LSL(16|32|64)rm")>;

def HWWriteResGroup88 : SchedWriteRes<[HWPort4,HWPort237,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[HWWriteResGroup88], (instregex "PUSHF16",
                                            "PUSHF64")>;

def HWWriteResGroup89 : SchedWriteRes<[HWPort0]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup89], (instregex "(V?)PCMPGTQ(Y?)rr")>;

def HWWriteResGroup90 : SchedWriteRes<[HWPort01]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup90], (instregex "(V?)MULPD(Y?)rr",
                                            "(V?)MULPS(Y?)rr",
                                            "(V?)MULSDrr",
                                            "(V?)MULSSrr")>;

def HWWriteResGroup91_1 : SchedWriteRes<[HWPort0,HWPort23,HWFPDivider]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,7];
}
def: InstRW<[HWWriteResGroup91_1], (instregex "(V?)SQRTSSm")>;

def HWWriteResGroup91_4 : SchedWriteRes<[HWPort0,HWPort23,HWFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,7];
}
def: InstRW<[HWWriteResGroup91_4], (instregex "(V?)DIVSSrm")>;

def HWWriteResGroup91_2 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup91_2], (instregex "(V?)PCMPGTQrm",
                                              "(V?)PHMINPOSUWrm",
                                              "(V?)PMADDUBSWrm",
                                              "(V?)PMADDWDrm",
                                              "(V?)PMULDQrm",
                                              "(V?)PMULHRSWrm",
                                              "(V?)PMULHUWrm",
                                              "(V?)PMULHWrm",
                                              "(V?)PMULLWrm",
                                              "(V?)PMULUDQrm",
                                              "(V?)PSADBWrm",
                                              "(V?)RCPPSm",
                                              "(V?)RSQRTPSm")>;

def HWWriteResGroup91_3 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup91_3], (instregex "MUL_F32m",
                                            "MUL_F64m",
                                            "VPCMPGTQYrm",
                                            "VPMADDUBSWYrm",
                                            "VPMADDWDYrm",
                                            "VPMULDQYrm",
                                            "VPMULHRSWYrm",
                                            "VPMULHUWYrm",
                                            "VPMULHWYrm",
                                            "VPMULLWYrm",
                                            "VPMULUDQYrm",
                                            "VPSADBWYrm")>;

def HWWriteResGroup92 : SchedWriteRes<[HWPort01,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup92], (instregex "(V?)MULPDrm",
                                            "(V?)MULPSrm",
                                            "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)m")>;

def HWWriteResGroup92_1 : SchedWriteRes<[HWPort01,HWPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup92_1], (instregex "VMULPDYrm",
                                              "VMULPSYrm",
                                              "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Ym")>;

def HWWriteResGroup92_2 : SchedWriteRes<[HWPort01,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup92_2], (instregex "(V?)MULSDrm",
                                              "(V?)MULSSrm")>;

def HWWriteResGroup93 : SchedWriteRes<[HWPort1,HWPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup93], (instregex "(V?)CVTSI642SSrr")>;

def HWWriteResGroup94 : SchedWriteRes<[HWPort1,HWPort6,HWPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup94], (instregex "STR(16|32|64)r")>;

def HWWriteResGroup95 : SchedWriteRes<[HWPort1,HWPort06,HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup95], (instrs IMUL32r, MUL32r, MULX32rr)>;

def HWWriteResGroup96_1 : SchedWriteRes<[HWPort1,HWPort5,HWPort23]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[HWWriteResGroup96_1], (instregex "VHADDPDYrm",
                                              "VHADDPSYrm",
                                              "VHSUBPDYrm",
                                              "VHSUBPSYrm")>;

def HWWriteResGroup97 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup97], (instregex "CVTTSS2SI64rm")>;

def HWWriteResGroup98 : SchedWriteRes<[HWPort1,HWPort23,HWPort06,HWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup98], (instrs IMUL32m, MUL32m, MULX32rm)>;

def HWWriteResGroup99 : SchedWriteRes<[HWPort6,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[HWWriteResGroup99], (instregex "PAUSE")>;

def HWWriteResGroup100 : SchedWriteRes<[HWPort06,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[HWWriteResGroup100], (instregex "XSETBV")>;

def HWWriteResGroup101 : SchedWriteRes<[HWPort06,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[HWWriteResGroup101], (instregex "CMPXCHG(8|16|32|64)rr")>;

def HWWriteResGroup102 : SchedWriteRes<[HWPort1,HWPort5]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup102], (instregex "VCVTDQ2PDYrr",
                                             "VCVTPD2DQYrr",
                                             "VCVTPD2PSYrr",
                                             "VCVTPS2PHYrr",
                                             "VCVTTPD2DQYrr")>;

def HWWriteResGroup103 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup103], (instregex "ADD_FI16m",
                                             "ADD_FI32m",
                                             "SUBR_FI16m",
                                             "SUBR_FI32m",
                                             "SUB_FI16m",
                                             "SUB_FI32m",
                                             "VROUNDPDYm",
                                             "VROUNDPSYm")>;

def HWWriteResGroup103_1 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup103_1], (instregex "(V?)ROUNDPDm",
                                               "(V?)ROUNDPSm",
                                               "(V?)ROUNDSDm",
                                               "(V?)ROUNDSSm")>;

def HWWriteResGroup104 : SchedWriteRes<[HWPort1,HWPort5,HWPort23]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup104], (instregex "VCVTDQ2PDYrm")>;

def HWWriteResGroup105 : SchedWriteRes<[HWPort1,HWPort06,HWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[HWWriteResGroup105], (instregex "SHLD(16|32|64)rrCL",
                                             "SHRD(16|32|64)rrCL")>;

def HWWriteResGroup106 : SchedWriteRes<[HWPort1,HWPort4,HWPort5,HWPort237]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup106], (instregex "VCVTPS2PHYmr")>;

def HWWriteResGroup107 : SchedWriteRes<[HWPort1,HWPort6,HWPort06,HWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup107], (instregex "SLDT(16|32|64)r")>;

def HWWriteResGroup108 : SchedWriteRes<[HWPort6,HWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,5];
}
def: InstRW<[HWWriteResGroup108], (instregex "STD")>;

def HWWriteResGroup109 : SchedWriteRes<[HWPort1,HWPort23,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 12;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,1,2];
}
def: InstRW<[HWWriteResGroup109], (instregex "SHLD(16|32|64)mrCL",
                                             "SHRD(16|32|64)mrCL")>;

def HWWriteResGroup113_1 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[HWWriteResGroup113_1], (instregex "VMPSADBWYrmi")>;

def HWWriteResGroup114 : SchedWriteRes<[HWPort6,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [2,2,1,2];
}
def: InstRW<[HWWriteResGroup114], (instrs LOOP)>;

def HWWriteResGroup115 : SchedWriteRes<[HWPort0,HWPort1,HWPort23]> {
  let Latency = 15;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup115], (instregex "MUL_FI16m",
                                             "MUL_FI32m")>;

def HWWriteResGroup116 : SchedWriteRes<[HWPort0,HWPort1,HWPort5]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup116], (instregex "(V?)DPPDrri")>;

def HWWriteResGroup117 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort23]> {
  let Latency = 15;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup117], (instregex "(V?)DPPDrmi")>;

def HWWriteResGroup119_1 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 17;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup119_1], (instregex "VPMULLDYrm")>;

def HWWriteResGroup120 : SchedWriteRes<[HWPort1,HWPort23,HWPort237,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 16;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,4,1,2];
}
def: InstRW<[HWWriteResGroup120], (instregex "RCL(8|16|32|64)mCL")>;

def HWWriteResGroup121 : SchedWriteRes<[HWPort0,HWFPDivider]> {
  let Latency = 13;
  let NumMicroOps = 1;
  let ResourceCycles = [1,7];
}
def: InstRW<[HWWriteResGroup121], (instregex "(V?)DIVPSrr",
                                             "(V?)DIVSSrr")>;

def HWWriteResGroup125 : SchedWriteRes<[HWPort0,HWPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup125], (instregex "VRCPPSYr",
                                             "VRSQRTPSYr")>;

def HWWriteResGroup128 : SchedWriteRes<[HWPort0,HWPort23,HWPort015]> {
  let Latency = 18;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup128], (instregex "VRCPPSYm",
                                             "VRSQRTPSYm")>;

def HWWriteResGroup129 : SchedWriteRes<[HWPort1,HWPort06,HWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,2,3];
}
def: InstRW<[HWWriteResGroup129], (instregex "RCL(16|32|64)rCL",
                                             "RCR(16|32|64)rCL")>;

def HWWriteResGroup130 : SchedWriteRes<[HWPort1,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 9;
  let ResourceCycles = [1,4,1,3];
}
def: InstRW<[HWWriteResGroup130], (instregex "RCL8rCL")>;

def HWWriteResGroup131 : SchedWriteRes<[HWPort06,HWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 11;
  let ResourceCycles = [2,9];
}
def: InstRW<[HWWriteResGroup131], (instrs LOOPE, LOOPNE)>;

def HWWriteResGroup132 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 17;
  let NumMicroOps = 14;
  let ResourceCycles = [1,1,1,4,2,5];
}
def: InstRW<[HWWriteResGroup132], (instregex "CMPXCHG8B")>;

def HWWriteResGroup133 : SchedWriteRes<[HWPort0,HWFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,7];
}
def: InstRW<[HWWriteResGroup133], (instregex "(V?)SQRTPSr",
                                             "(V?)SQRTSSr")>;

def HWWriteResGroup134 : SchedWriteRes<[HWPort0,HWPort23,HWFPDivider]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,7];
}
def: InstRW<[HWWriteResGroup134], (instregex "(V?)DIVPSrm")>;

def HWWriteResGroup135 : SchedWriteRes<[HWPort1,HWPort23,HWPort237,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 19;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,3,1,3];
}
def: InstRW<[HWWriteResGroup135], (instregex "RCR(8|16|32|64)mCL")>;

def HWWriteResGroup138 : SchedWriteRes<[HWPort0,HWPort23,HWFPDivider]> {
  let Latency = 17;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,7];
}
def: InstRW<[HWWriteResGroup138], (instregex "(V?)SQRTPSm")>;

def HWWriteResGroup140 : SchedWriteRes<[HWPort0,HWPort1,HWPort5]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup140], (instregex "(V?)DPPS(Y?)rri")>;

def HWWriteResGroup141 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort23]> {
  let Latency = 20;
  let NumMicroOps = 5;
  let ResourceCycles = [2,1,1,1];
}
def: InstRW<[HWWriteResGroup141], (instregex "(V?)DPPSrmi")>;

def HWWriteResGroup141_1 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort23]> {
  let Latency = 21;
  let NumMicroOps = 5;
  let ResourceCycles = [2,1,1,1];
}
def: InstRW<[HWWriteResGroup141_1], (instregex "VDPPSYrmi")>;

def HWWriteResGroup142 : SchedWriteRes<[HWPort1,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 14;
  let NumMicroOps = 10;
  let ResourceCycles = [2,3,1,4];
}
def: InstRW<[HWWriteResGroup142], (instregex "RCR8rCL")>;

def HWWriteResGroup143 : SchedWriteRes<[HWPort23,HWPort0156]> {
  let Latency = 19;
  let NumMicroOps = 15;
  let ResourceCycles = [1,14];
}
def: InstRW<[HWWriteResGroup143], (instregex "POPF16")>;

def HWWriteResGroup144 : SchedWriteRes<[HWPort4,HWPort5,HWPort6,HWPort23,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 21;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,1,2];
}
def: InstRW<[HWWriteResGroup144], (instregex "INSB",
                                             "INSL",
                                             "INSW")>;

def HWWriteResGroup145 : SchedWriteRes<[HWPort5]> {
  let Latency = 16;
  let NumMicroOps = 16;
  let ResourceCycles = [16];
}
def: InstRW<[HWWriteResGroup145], (instregex "VZEROALL")>;

def HWWriteResGroup146 : SchedWriteRes<[HWPort0,HWPort4,HWPort5,HWPort23,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 22;
  let NumMicroOps = 19;
  let ResourceCycles = [2,1,4,1,1,4,6];
}
def: InstRW<[HWWriteResGroup146], (instregex "CMPXCHG16B")>;

def HWWriteResGroup147 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort6,HWPort01,HWPort0156]> {
  let Latency = 17;
  let NumMicroOps = 15;
  let ResourceCycles = [2,1,2,4,2,4];
}
def: InstRW<[HWWriteResGroup147], (instregex "XCH_F")>;

def HWWriteResGroup149 : SchedWriteRes<[HWPort5,HWPort6,HWPort06,HWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[HWWriteResGroup149], (instregex "CPUID")>;
def: InstRW<[HWWriteResGroup149], (instrs RDTSC)>;

def HWWriteResGroup151 : SchedWriteRes<[HWPort6,HWPort23,HWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 19;
  let ResourceCycles = [3,1,15];
}
def: InstRW<[HWWriteResGroup151], (instregex "XRSTOR(64)?")>;

def HWWriteResGroup154 : SchedWriteRes<[HWPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup154], (instregex "DIV_FPrST0",
                                             "DIV_FST0r",
                                             "DIV_FrST0")>;

def HWWriteResGroup154_1 : SchedWriteRes<[HWPort0,HWFPDivider]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1,14];
}
def: InstRW<[HWWriteResGroup154_1], (instregex "(V?)DIVPDrr",
                                               "(V?)DIVSDrr")>;

def HWWriteResGroup155 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 27;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup155], (instregex "DIVR_F32m",
                                             "DIVR_F64m")>;

def HWWriteResGroup155_1 : SchedWriteRes<[HWPort0,HWPort23,HWFPDivider]> {
  let Latency = 26;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,14];
}
def: InstRW<[HWWriteResGroup155_1], (instregex "(V?)DIVPDrm")>;

def HWWriteResGroup155_2 : SchedWriteRes<[HWPort0,HWPort23,HWFPDivider]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,14];
}
def: InstRW<[HWWriteResGroup155_2], (instregex "(V?)SQRTSDm")>;

def HWWriteResGroup155_3 : SchedWriteRes<[HWPort0,HWPort23,HWFPDivider]> {
  let Latency = 22;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,14];
}
def: InstRW<[HWWriteResGroup155_3], (instregex "(V?)SQRTPDm")>;

def HWWriteResGroup155_4 : SchedWriteRes<[HWPort0,HWPort23,HWFPDivider]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,14];
}
def: InstRW<[HWWriteResGroup155_4], (instregex "(V?)DIVSDrm")>;

def HWWriteResGroup156 : SchedWriteRes<[HWPort5,HWPort6,HWPort0156]> {
  let Latency = 20;
  let NumMicroOps = 10;
  let ResourceCycles = [1,2,7];
}
def: InstRW<[HWWriteResGroup156], (instregex "MWAITrr")>;

def HWWriteResGroup157 : SchedWriteRes<[HWPort0,HWFPDivider]> {
  let Latency = 16;
  let NumMicroOps = 1;
  let ResourceCycles = [1,14];
}
def: InstRW<[HWWriteResGroup157], (instregex "(V?)SQRTPDr",
                                             "(V?)SQRTSDr")>;

def HWWriteResGroup159 : SchedWriteRes<[HWPort0,HWPort15,HWFPDivider]> {
  let Latency = 21;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,14];
}
def: InstRW<[HWWriteResGroup159], (instregex "VDIVPSYrr",
                                             "VSQRTPSYr")>;

def HWWriteResGroup160 : SchedWriteRes<[HWPort0,HWPort23,HWPort15,HWFPDivider]> {
  let Latency = 28;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,14];
}
def: InstRW<[HWWriteResGroup160], (instregex "VDIVPSYrm",
                                             "VSQRTPSYm")>;

def HWWriteResGroup161 : SchedWriteRes<[HWPort0,HWPort1,HWPort23]> {
  let Latency = 30;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup161], (instregex "DIVR_FI16m",
                                             "DIVR_FI32m")>;

def HWWriteResGroup162 : SchedWriteRes<[HWPort0]> {
  let Latency = 24;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup162], (instregex "DIVR_FPrST0",
                                             "DIVR_FST0r",
                                             "DIVR_FrST0")>;

def HWWriteResGroup163 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 31;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup163], (instregex "DIV_F32m",
                                             "DIV_F64m")>;

def HWWriteResGroup164 : SchedWriteRes<[HWPort4,HWPort6,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 30;
  let NumMicroOps = 27;
  let ResourceCycles = [1,5,1,1,19];
}
def: InstRW<[HWWriteResGroup164], (instregex "XSAVE64")>;

def HWWriteResGroup165 : SchedWriteRes<[HWPort4,HWPort6,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 31;
  let NumMicroOps = 28;
  let ResourceCycles = [1,6,1,1,19];
}
def: InstRW<[HWWriteResGroup165], (instregex "^XSAVE$", "XSAVEC", "XSAVES", "XSAVEOPT")>;

def HWWriteResGroup166 : SchedWriteRes<[HWPort0,HWPort1,HWPort23]> {
  let Latency = 34;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup166], (instregex "DIV_FI16m",
                                             "DIV_FI32m")>;

def HWWriteResGroup170 : SchedWriteRes<[HWPort5,HWPort6,HWPort23,HWPort06,HWPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[HWWriteResGroup170], (instregex "IN(8|16|32)ri",
                                             "IN(8|16|32)rr")>;

def HWWriteResGroup171 : SchedWriteRes<[HWPort5,HWPort6,HWPort23,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 36;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[HWWriteResGroup171], (instregex "OUT(8|16|32)ir",
                                             "OUT(8|16|32)rr")>;

def HWWriteResGroup172 : SchedWriteRes<[HWPort01,HWPort15,HWPort015,HWPort0156]> {
  let Latency = 31;
  let NumMicroOps = 31;
  let ResourceCycles = [8,1,21,1];
}
def: InstRW<[HWWriteResGroup172], (instregex "MMX_EMMS")>;

def HWWriteResGroup173 : SchedWriteRes<[HWPort0,HWPort15,HWFPDivider]> {
  let Latency = 35;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,28];
}
def: InstRW<[HWWriteResGroup173], (instregex "VDIVPDYrr",
                                             "VSQRTPDYr")>;

def HWWriteResGroup174 : SchedWriteRes<[HWPort0,HWPort23,HWPort15,HWFPDivider]> {
  let Latency = 42;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,28];
}
def: InstRW<[HWWriteResGroup174], (instregex "VDIVPDYrm",
                                             "VSQRTPDYm")>;

def HWWriteResGroup175 : SchedWriteRes<[HWPort1,HWPort4,HWPort5,HWPort6,HWPort23,HWPort237,HWPort15,HWPort0156]> {
  let Latency = 41;
  let NumMicroOps = 18;
  let ResourceCycles = [1,1,2,3,1,1,1,8];
}
def: InstRW<[HWWriteResGroup175], (instregex "VMCLEARm")>;

def HWWriteResGroup176 : SchedWriteRes<[HWPort5,HWPort0156]> {
  let Latency = 42;
  let NumMicroOps = 22;
  let ResourceCycles = [2,20];
}
def: InstRW<[HWWriteResGroup176], (instrs RDTSCP)>;

def HWWriteResGroup177 : SchedWriteRes<[HWPort0,HWPort01,HWPort23,HWPort05,HWPort06,HWPort015,HWPort0156]> {
  let Latency = 61;
  let NumMicroOps = 64;
  let ResourceCycles = [2,2,8,1,10,2,39];
}
def: InstRW<[HWWriteResGroup177], (instregex "FLDENVm")>;

def HWWriteResGroup178 : SchedWriteRes<[HWPort0,HWPort6,HWPort23,HWPort05,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 64;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[HWWriteResGroup178], (instrs FXRSTOR64)>;

def HWWriteResGroup179 : SchedWriteRes<[HWPort0,HWPort6,HWPort23,HWPort05,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 64;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[HWWriteResGroup179], (instrs FXRSTOR)>;

def HWWriteResGroup180 : SchedWriteRes<[HWPort5,HWPort01,HWPort0156]> {
  let Latency = 75;
  let NumMicroOps = 15;
  let ResourceCycles = [6,3,6];
}
def: InstRW<[HWWriteResGroup180], (instrs FNINIT)>;

def HWWriteResGroup181 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort6,HWPort01,HWPort0156]> {
  let Latency = 98;
  let NumMicroOps = 32;
  let ResourceCycles = [7,7,3,3,1,11];
}
def: InstRW<[HWWriteResGroup181], (instregex "DIV(16|32|64)r")>;

def HWWriteResGroup182 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort6,HWPort06,HWPort0156]> {
  let Latency = 112;
  let NumMicroOps = 66;
  let ResourceCycles = [4,2,4,8,14,34];
}
def: InstRW<[HWWriteResGroup182], (instregex "IDIV(16|32|64)r")>;

def HWWriteResGroup183 : SchedWriteRes<[HWPort0,HWPort1,HWPort4,HWPort5,HWPort6,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 115;
  let NumMicroOps = 100;
  let ResourceCycles = [9,9,11,8,1,11,21,30];
}
def: InstRW<[HWWriteResGroup183], (instregex "FSTENVm")>;

def HWWriteResGroup184 : SchedWriteRes<[HWPort0, HWPort5, HWPort15, HWPort015, HWPort06, HWPort23]> {
  let Latency = 26;
  let NumMicroOps = 12;
  let ResourceCycles = [2,2,1,3,2,2];
}
def: InstRW<[HWWriteResGroup184], (instrs VGATHERDPDrm,
                                          VPGATHERDQrm,
                                          VPGATHERDDrm)>;

def HWWriteResGroup185 : SchedWriteRes<[HWPort0, HWPort5, HWPort06, HWPort15, HWPort015, HWPort23]> {
  let Latency = 24;
  let NumMicroOps = 22;
  let ResourceCycles = [5,3,4,1,5,4];
}
def: InstRW<[HWWriteResGroup185], (instrs VGATHERQPDYrm,
                                          VPGATHERQQYrm)>;

def HWWriteResGroup186 : SchedWriteRes<[HWPort0, HWPort5, HWPort06, HWPort15, HWPort015, HWPort23]> {
  let Latency = 28;
  let NumMicroOps = 22;
  let ResourceCycles = [5,3,4,1,5,4];
}
def: InstRW<[HWWriteResGroup186], (instrs VPGATHERQDYrm)>;

def HWWriteResGroup187 : SchedWriteRes<[HWPort0, HWPort5, HWPort06, HWPort15, HWPort015, HWPort23]> {
  let Latency = 25;
  let NumMicroOps = 22;
  let ResourceCycles = [5,3,4,1,5,4];
}
def: InstRW<[HWWriteResGroup187], (instrs VPGATHERQDrm)>;

def HWWriteResGroup188 : SchedWriteRes<[HWPort0, HWPort5, HWPort06, HWPort15, HWPort015, HWPort23]> {
  let Latency = 27;
  let NumMicroOps = 20;
  let ResourceCycles = [3,3,4,1,5,4];
}
def: InstRW<[HWWriteResGroup188], (instrs VGATHERDPDYrm,
                                          VPGATHERDQYrm)>;

def HWWriteResGroup189 : SchedWriteRes<[HWPort0, HWPort5, HWPort06, HWPort15, HWPort015, HWPort23]> {
  let Latency = 27;
  let NumMicroOps = 34;
  let ResourceCycles = [5,3,8,1,9,8];
}
def: InstRW<[HWWriteResGroup189], (instrs VGATHERDPSYrm,
                                          VPGATHERDDYrm)>;

def HWWriteResGroup190 : SchedWriteRes<[HWPort0, HWPort5, HWPort06, HWPort15, HWPort015, HWPort23]> {
  let Latency = 23;
  let NumMicroOps = 14;
  let ResourceCycles = [3,3,2,1,3,2];
}
def: InstRW<[HWWriteResGroup190], (instrs VGATHERQPDrm,
                                          VPGATHERQQrm)>;

def HWWriteResGroup191 : SchedWriteRes<[HWPort0, HWPort5, HWPort06, HWPort15, HWPort015, HWPort23]> {
  let Latency = 28;
  let NumMicroOps = 15;
  let ResourceCycles = [3,3,2,1,4,2];
}
def: InstRW<[HWWriteResGroup191], (instrs VGATHERQPSYrm)>;

def HWWriteResGroup192 : SchedWriteRes<[HWPort0, HWPort5, HWPort06, HWPort15, HWPort015, HWPort23]> {
  let Latency = 25;
  let NumMicroOps = 15;
  let ResourceCycles = [3,3,2,1,4,2];
}
def: InstRW<[HWWriteResGroup192], (instrs VGATHERQPSrm,
                                          VGATHERDPSrm)>;

} // SchedModel
