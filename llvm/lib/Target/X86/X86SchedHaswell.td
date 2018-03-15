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

// Loads are 5 cycles, so ReadAfterLd registers needn't be available until 5
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 5>;

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

  // Memory variant also uses a cycle on port 2/3 and adds 5 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, [HWPort23, ExePort]> {
     let Latency = !add(Lat, 5);
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [HWPort4]>;

// Store_addr on 237.
// Store_data on 4.
def : WriteRes<WriteStore, [HWPort237, HWPort4]>;
def : WriteRes<WriteLoad,  [HWPort23]> { let Latency = 5; }
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
def  : WriteRes<WriteFStore,       [HWPort237, HWPort4]>;
def  : WriteRes<WriteFLoad,        [HWPort23]> { let Latency = 5; }
def  : WriteRes<WriteFMove,        [HWPort5]>;

defm : HWWriteResPair<WriteFAdd,   HWPort1, 3>;
defm : HWWriteResPair<WriteFMul,   HWPort0, 5>;
defm : HWWriteResPair<WriteFDiv,   HWPort0, 12>; // 10-14 cycles.
defm : HWWriteResPair<WriteFRcp,   HWPort0, 5>;
defm : HWWriteResPair<WriteFRsqrt, HWPort0, 5>;
defm : HWWriteResPair<WriteFSqrt,  HWPort0, 15>;
defm : HWWriteResPair<WriteCvtF2I, HWPort1, 3>;
defm : HWWriteResPair<WriteCvtI2F, HWPort1, 4>;
defm : HWWriteResPair<WriteCvtF2F, HWPort1, 3>;
defm : HWWriteResPair<WriteFMA,   HWPort01, 5>;
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
def  : WriteRes<WriteVecStore,       [HWPort237, HWPort4]>;
def  : WriteRes<WriteVecLoad,        [HWPort23]> { let Latency = 5; }
def  : WriteRes<WriteVecMove,        [HWPort015]>;

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
def : InstRW<[WriteLoad], (instregex "MOV(S|Z)X32rm8")>;

// XLAT.
def WriteXLAT : SchedWriteRes<[]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[WriteXLAT], (instregex "XLAT")>;

// PUSH.
// m.
def : InstRW<[Write2P237_P4], (instregex "PUSH(16|32)rmm")>;

// PUSHA.
def WritePushA : SchedWriteRes<[]> {
  let NumMicroOps = 19;
}
def : InstRW<[WritePushA], (instregex "PUSHA(16|32)")>;

// POP.
// m.
def : InstRW<[Write2P237_P4], (instregex "POP(16|32)rmm")>;

// POPA.
def WritePopA : SchedWriteRes<[]> {
  let NumMicroOps = 18;
}
def : InstRW<[WritePopA], (instregex "POPA(16|32)")>;

//-- Arithmetic instructions --//

// DIV.
// r8.
def WriteDiv8 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 22;
  let NumMicroOps = 9;
}
def : InstRW<[WriteDiv8], (instregex "DIV8r")>;

// IDIV.
// r8.
def WriteIDiv8 : SchedWriteRes<[HWPort0, HWPort1, HWPort5, HWPort6]> {
  let Latency = 23;
  let NumMicroOps = 9;
}
def : InstRW<[WriteIDiv8], (instregex "IDIV8r")>;

// BT.
// m,r.
def WriteBTmr : SchedWriteRes<[]> {
  let NumMicroOps = 10;
}
def : InstRW<[WriteBTmr], (instregex "BT(16|32|64)mr")>;

// BTR BTS BTC.
// m,r.
def WriteBTRSCmr : SchedWriteRes<[]> {
  let NumMicroOps = 11;
}
def : InstRW<[WriteBTRSCmr], (instregex "BT(R|S|C)(16|32|64)mr")>;

//-- Control transfer instructions --//

// CALL.
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

// MOVS.
def WriteMOVS : SchedWriteRes<[HWPort23, HWPort4, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 5;
  let ResourceCycles = [2, 1, 2];
}
def : InstRW<[WriteMOVS], (instregex "MOVS(B|L|Q|W)")>;

// CMPS.
def WriteCMPS : SchedWriteRes<[HWPort23, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 5;
  let ResourceCycles = [2, 3];
}
def : InstRW<[WriteCMPS], (instregex "CMPS(B|L|Q|W)")>;

//-- Other --//

// RDPMC.f
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

// FLDZ.
def : InstRW<[WriteP01], (instregex "LD_F0")>;

// FLDPI FLDL2E etc.
def : InstRW<[Write2P01], (instregex "FLDPI", "FLDL2(T|E)" "FLDL(G|N)2")>;

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

// FCOMPP FUCOMPP.
// r.
def : InstRW<[Write2P01], (instregex "FCOMPP", "UCOM_FPPr")>;

// FCOMI(P) FUCOMI(P).
// m.
def : InstRW<[Write3P01], (instregex "COM_FIr", "COM_FIPr", "UCOM_FIr",
                           "UCOM_FIPr")>;

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

//=== Floating Point XMM and YMM Instructions ===//

// Remaining instrs.

def HWWriteResGroup0 : SchedWriteRes<[HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup0], (instregex "LDDQUrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVAPDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVAPSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVDQArm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVDQUrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVNTDQArm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVSHDUPrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVSLDUPrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVUPDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVUPSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VBROADCASTSSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VLDDQUrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVAPDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVAPSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVDQArm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVDQUrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVNTDQArm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVSHDUPrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVSLDUPrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVUPDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVUPSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VPBROADCASTDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VPBROADCASTQrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "ROUNDPDr")>;
def: InstRW<[HWWriteResGroup0], (instregex "ROUNDPSr")>;
def: InstRW<[HWWriteResGroup0], (instregex "ROUNDSDr")>;
def: InstRW<[HWWriteResGroup0], (instregex "ROUNDSSr")>;
def: InstRW<[HWWriteResGroup0], (instregex "VROUNDPDr")>;
def: InstRW<[HWWriteResGroup0], (instregex "VROUNDPSr")>;
def: InstRW<[HWWriteResGroup0], (instregex "VROUNDSDr")>;
def: InstRW<[HWWriteResGroup0], (instregex "VROUNDSSr")>;
def: InstRW<[HWWriteResGroup0], (instregex "VROUNDYPDr")>;
def: InstRW<[HWWriteResGroup0], (instregex "VROUNDYPSr")>;

def HWWriteResGroup0_1 : SchedWriteRes<[HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup0_1], (instregex "LD_F32m")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "LD_F64m")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "LD_F80m")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VBROADCASTF128")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VBROADCASTI128")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VBROADCASTSDYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VBROADCASTSSYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VLDDQUYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VMOVAPDYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VMOVAPSYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VMOVDDUPYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VMOVDQAYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VMOVDQUYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VMOVNTDQAYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VMOVSHDUPYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VMOVSLDUPYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VMOVUPDYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VMOVUPSYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VPBROADCASTDYrm")>;
def: InstRW<[HWWriteResGroup0_1], (instregex "VPBROADCASTQYrm")>;

def HWWriteResGroup0_2 : SchedWriteRes<[HWPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup0_2], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MMX_MOVD64rm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MMX_MOVD64to64rm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MMX_MOVQ64rm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOV(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOV64toPQIrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOV8rm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVDDUPrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVDI2PDIrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVQI2PQIrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVSDrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVSSrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVSX(16|32|64)rm16")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVSX(16|32|64)rm32")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVSX(16|32|64)rm8")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVZX(16|32|64)rm16")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "MOVZX(16|32|64)rm8")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "PREFETCHNTA")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "PREFETCHT0")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "PREFETCHT1")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "PREFETCHT2")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "VMOV64toPQIrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "VMOVDDUPrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "VMOVDI2PDIrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "VMOVQI2PQIrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "VMOVSDrm")>;
def: InstRW<[HWWriteResGroup0_2], (instregex "VMOVSSrm")>;

def HWWriteResGroup1 : SchedWriteRes<[HWPort4,HWPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup1], (instregex "FBSTPm")>;
def: InstRW<[HWWriteResGroup1], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[HWWriteResGroup1], (instregex "MMX_MOVD64mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MMX_MOVNTQmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MMX_MOVQ64mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOV(16|32|64)mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOV8mi")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOV8mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVAPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVAPSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVDQAmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVDQUmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVHPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVHPSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVLPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVLPSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVNTDQmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVNTI_64mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVNTImr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVNTPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVNTPSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVPDI2DImr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVPQI2QImr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVPQIto64mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVSDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVSSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVUPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVUPSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "ST_FP32m")>;
def: InstRW<[HWWriteResGroup1], (instregex "ST_FP64m")>;
def: InstRW<[HWWriteResGroup1], (instregex "ST_FP80m")>;
def: InstRW<[HWWriteResGroup1], (instregex "VEXTRACTF128mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VEXTRACTI128mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVAPDYmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVAPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVAPSYmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVAPSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVDQAYmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVDQAmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVDQUYmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVDQUmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVHPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVHPSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVLPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVLPSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVNTDQYmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVNTDQmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVNTPDYmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVNTPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVNTPSYmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVNTPSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVPDI2DImr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVPQI2QImr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVPQIto64mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVSDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVSSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVUPDYmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVUPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVUPSYmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMOVUPSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "VMPTRSTm")>;

def HWWriteResGroup2 : SchedWriteRes<[HWPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup2], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_MOVD64grr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PMOVMSKBrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSLLDri")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSLLDrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSLLQri")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSLLQrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSLLWri")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSLLWrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSRADri")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSRADrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSRAWri")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSRAWrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSRLDri")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSRLDrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSRLQri")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSRLQrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSRLWri")>;
def: InstRW<[HWWriteResGroup2], (instregex "MMX_PSRLWrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MOVPDI2DIrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "MOVPQIto64rr")>;
def: InstRW<[HWWriteResGroup2], (instregex "PSLLDri")>;
def: InstRW<[HWWriteResGroup2], (instregex "PSLLQri")>;
def: InstRW<[HWWriteResGroup2], (instregex "PSLLWri")>;
def: InstRW<[HWWriteResGroup2], (instregex "PSRADri")>;
def: InstRW<[HWWriteResGroup2], (instregex "PSRAWri")>;
def: InstRW<[HWWriteResGroup2], (instregex "PSRLDri")>;
def: InstRW<[HWWriteResGroup2], (instregex "PSRLQri")>;
def: InstRW<[HWWriteResGroup2], (instregex "PSRLWri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VMOVPDI2DIrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VMOVPQIto64rr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLDYri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLDri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLQYri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLQri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLVQYrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLVQrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLWYri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLWri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRADYri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRADri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRAWYri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRAWri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRLDYri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRLDri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRLQYri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRLQri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRLVQYrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRLVQrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRLWYri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSRLWri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VTESTPDYrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VTESTPDrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VTESTPSYrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VTESTPSrr")>;

def HWWriteResGroup3 : SchedWriteRes<[HWPort1]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup3], (instregex "COMP_FST0r")>;
def: InstRW<[HWWriteResGroup3], (instregex "COM_FST0r")>;
def: InstRW<[HWWriteResGroup3], (instregex "MMX_MASKMOVQ64")>;
def: InstRW<[HWWriteResGroup3], (instregex "MMX_MASKMOVQ64")>;
def: InstRW<[HWWriteResGroup3], (instregex "UCOM_FPr")>;
def: InstRW<[HWWriteResGroup3], (instregex "UCOM_Fr")>;
def: InstRW<[HWWriteResGroup3], (instregex "VMASKMOVDQU")>;

def HWWriteResGroup4 : SchedWriteRes<[HWPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup4], (instregex "ANDNPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "ANDNPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "ANDPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "ANDPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "INSERTPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_MOVD64rr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_MOVD64to64rr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_MOVQ2DQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PALIGNRrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PSHUFBrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PSHUFWri")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PUNPCKHBWirr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PUNPCKHDQirr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PUNPCKHWDirr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PUNPCKLBWirr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PUNPCKLDQirr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PUNPCKLWDirr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOV64toPQIrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVAPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVAPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVDDUPrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVDI2PDIrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVHLPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVLHPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVSDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVSHDUPrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVSLDUPrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVSSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVUPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MOVUPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "ORPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "ORPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PACKSSDWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PACKSSWBrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PACKUSDWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PACKUSWBrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PALIGNRrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "PBLENDWrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVSXBDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVSXBQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVSXBWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVSXDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVSXWDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVSXWQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVZXBDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVZXBQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVZXBWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVZXDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVZXWDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PMOVZXWQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PSHUFBrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PSHUFDri")>;
def: InstRW<[HWWriteResGroup4], (instregex "PSHUFHWri")>;
def: InstRW<[HWWriteResGroup4], (instregex "PSHUFLWri")>;
def: InstRW<[HWWriteResGroup4], (instregex "PSLLDQri")>;
def: InstRW<[HWWriteResGroup4], (instregex "PSRLDQri")>;
def: InstRW<[HWWriteResGroup4], (instregex "PUNPCKHBWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PUNPCKHDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PUNPCKHQDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PUNPCKHWDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PUNPCKLBWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PUNPCKLDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PUNPCKLQDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "PUNPCKLWDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "SHUFPDrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "SHUFPSrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "UNPCKHPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "UNPCKHPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "UNPCKLPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "UNPCKLPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VANDNPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VANDNPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VANDNPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VANDNPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VANDPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VANDPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VANDPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VANDPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VBROADCASTSSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VINSERTPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOV64toPQIrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVAPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVAPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVAPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVAPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVDDUPYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVDDUPrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVDI2PDIrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVHLPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVLHPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVSDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVSHDUPYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVSHDUPrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVSLDUPYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVSLDUPrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVSSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVUPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVUPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVUPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVUPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VORPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VORPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VORPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VORPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPACKSSDWYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPACKSSDWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPACKSSWBYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPACKSSWBrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPACKUSDWYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPACKUSDWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPACKUSWBYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPACKUSWBrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPALIGNRYrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPALIGNRrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPBLENDWYrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPBLENDWrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPBROADCASTDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPBROADCASTQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPDYri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPDri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPSYri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPSri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVSXBDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVSXBQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVSXBWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVSXDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVSXWDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVSXWQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVZXBDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVZXBQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVZXBWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVZXDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVZXWDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPMOVZXWQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFBYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFBrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFDYri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFDri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFHWYri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFHWri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFLWYri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFLWri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSLLDQYri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSLLDQri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSRLDQYri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSRLDQri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHBWYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHBWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHDQYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHQDQYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHQDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHWDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHWDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKLBWYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKLBWrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKLDQYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKLDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKLQDQYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKLQDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKLWDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKLWDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VSHUFPDYrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VSHUFPDrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VSHUFPSYrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VSHUFPSrri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKHPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKHPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKHPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKHPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKLPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKLPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKLPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKLPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VXORPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VXORPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VXORPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VXORPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "XORPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "XORPSrr")>;

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
def: InstRW<[HWWriteResGroup6], (instregex "FINCSTP")>;
def: InstRW<[HWWriteResGroup6], (instregex "FNOP")>;

def HWWriteResGroup7 : SchedWriteRes<[HWPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup7], (instregex "BT(16|32|64)ri8")>;
def: InstRW<[HWWriteResGroup7], (instregex "BT(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTC(16|32|64)ri8")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTC(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTR(16|32|64)ri8")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTR(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTS(16|32|64)ri8")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTS(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "CDQ")>;
def: InstRW<[HWWriteResGroup7], (instregex "CQO")>;
def: InstRW<[HWWriteResGroup7], (instregex "J(A|AE|B|BE|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)_1")>;
def: InstRW<[HWWriteResGroup7], (instregex "J(A|AE|B|BE|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)_4")>;
def: InstRW<[HWWriteResGroup7], (instregex "JMP_1")>;
def: InstRW<[HWWriteResGroup7], (instregex "JMP_4")>;
def: InstRW<[HWWriteResGroup7], (instregex "RORX(32|64)ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SAR(16|32|64)r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SAR(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SAR8r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SAR8ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SARX(32|64)rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SET(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)r")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHL(16|32|64)r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHL(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHL8r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHL8ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHLX(32|64)rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHR(16|32|64)r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHR(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHR8r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHR8ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHRX(32|64)rr")>;

def HWWriteResGroup8 : SchedWriteRes<[HWPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup8], (instregex "ANDN(32|64)rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BLSI(32|64)rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BLSMSK(32|64)rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BLSR(32|64)rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BZHI(32|64)rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "LEA(16|32|64)(_32)?r")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PABSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PABSDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PABSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PADDBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PADDDirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PADDQirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PADDSBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PADDSWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PADDUSBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PADDUSWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PADDWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PAVGBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PAVGWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PCMPEQBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PCMPEQDirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PCMPEQWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PCMPGTBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PCMPGTDirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PCMPGTWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PMAXSWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PMAXUBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PMINSWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PMINUBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSIGNBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSIGNDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSIGNWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSUBBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSUBDirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSUBQirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSUBSBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSUBSWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSUBUSBirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSUBUSWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSUBWirr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PABSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PABSDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PABSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PADDBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PADDDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PADDQrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PADDSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PADDSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PADDUSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PADDUSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PADDWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PAVGBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PAVGWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PCMPEQBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PCMPEQDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PCMPEQQrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PCMPEQWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PCMPGTBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PCMPGTDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PCMPGTWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMAXSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMAXSDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMAXSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMAXUBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMAXUDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMAXUWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMINSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMINSDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMINSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMINUBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMINUDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PMINUWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSIGNBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSIGNDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSIGNWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBQrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBUSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBUSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDQYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDQrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDSBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDSWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDUSBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDUSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDUSWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDUSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPAVGBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPAVGBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPAVGWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPAVGWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQQYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQQrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPGTBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPGTBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPGTDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPGTDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPGTWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPGTWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXSBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXSDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXSDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXSWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXUBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXUBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXUDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXUDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXUWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMAXUWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINSBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINSDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINSDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINSWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINUBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINUBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINUDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINUDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINUWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPMINUWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBQYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBQrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBSBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBSWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBUSBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBUSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBUSWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBUSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSUBWrr")>;

def HWWriteResGroup9 : SchedWriteRes<[HWPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup9], (instregex "BLENDPDrri")>;
def: InstRW<[HWWriteResGroup9], (instregex "BLENDPSrri")>;
def: InstRW<[HWWriteResGroup9], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[HWWriteResGroup9], (instregex "MMX_MOVQ64rr")>;
def: InstRW<[HWWriteResGroup9], (instregex "MMX_PANDNirr")>;
def: InstRW<[HWWriteResGroup9], (instregex "MMX_PANDirr")>;
def: InstRW<[HWWriteResGroup9], (instregex "MMX_PORirr")>;
def: InstRW<[HWWriteResGroup9], (instregex "MMX_PXORirr")>;
def: InstRW<[HWWriteResGroup9], (instregex "MOVDQArr")>;
def: InstRW<[HWWriteResGroup9], (instregex "MOVDQUrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "MOVPQI2QIrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "PANDNrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "PANDrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "PORrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "PXORrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VBLENDPDYrri")>;
def: InstRW<[HWWriteResGroup9], (instregex "VBLENDPDrri")>;
def: InstRW<[HWWriteResGroup9], (instregex "VBLENDPSYrri")>;
def: InstRW<[HWWriteResGroup9], (instregex "VBLENDPSrri")>;
def: InstRW<[HWWriteResGroup9], (instregex "VMOVDQAYrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VMOVDQArr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VMOVDQUYrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VMOVDQUrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VMOVPQI2QIrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VMOVZPQILo2PQIrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VPANDNYrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VPANDNrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VPANDYrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VPANDrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VPBLENDDYrri")>;
def: InstRW<[HWWriteResGroup9], (instregex "VPBLENDDrri")>;
def: InstRW<[HWWriteResGroup9], (instregex "VPORYrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VPORrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VPXORYrr")>;
def: InstRW<[HWWriteResGroup9], (instregex "VPXORrr")>;

def HWWriteResGroup10 : SchedWriteRes<[HWPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup10], (instregex "ADD(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "ADD(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "ADD8i8")>;
def: InstRW<[HWWriteResGroup10], (instregex "ADD8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "ADD8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "AND(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "AND(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "AND8i8")>;
def: InstRW<[HWWriteResGroup10], (instregex "AND8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "AND8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "CBW")>;
def: InstRW<[HWWriteResGroup10], (instregex "CLC")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMC")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMP(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMP(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMP8i8")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMP8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMP8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "CWDE")>;
def: InstRW<[HWWriteResGroup10], (instregex "DEC(16|32|64)r")>;
def: InstRW<[HWWriteResGroup10], (instregex "DEC8r")>;
def: InstRW<[HWWriteResGroup10], (instregex "INC(16|32|64)r")>;
def: InstRW<[HWWriteResGroup10], (instregex "INC8r")>;
def: InstRW<[HWWriteResGroup10], (instregex "LAHF")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOV(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOV8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOV8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOVSX(16|32|64)rr16")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOVSX(16|32|64)rr32")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOVSX(16|32|64)rr8")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOVZX(16|32|64)rr16")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOVZX(16|32|64)rr8")>;
def: InstRW<[HWWriteResGroup10], (instregex "NEG(16|32|64)r")>;
def: InstRW<[HWWriteResGroup10], (instregex "NEG8r")>;
def: InstRW<[HWWriteResGroup10], (instregex "NOOP")>;
def: InstRW<[HWWriteResGroup10], (instregex "NOT(16|32|64)r")>;
def: InstRW<[HWWriteResGroup10], (instregex "NOT8r")>;
def: InstRW<[HWWriteResGroup10], (instregex "OR(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "OR(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "OR8i8")>;
def: InstRW<[HWWriteResGroup10], (instregex "OR8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "OR8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "SAHF")>;
def: InstRW<[HWWriteResGroup10], (instregex "SGDT64m")>;
def: InstRW<[HWWriteResGroup10], (instregex "SIDT64m")>;
def: InstRW<[HWWriteResGroup10], (instregex "SLDT64m")>;
def: InstRW<[HWWriteResGroup10], (instregex "SMSW16m")>;
def: InstRW<[HWWriteResGroup10], (instregex "STC")>;
def: InstRW<[HWWriteResGroup10], (instregex "STRm")>;
def: InstRW<[HWWriteResGroup10], (instregex "SUB(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "SUB(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "SUB8i8")>;
def: InstRW<[HWWriteResGroup10], (instregex "SUB8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "SUB8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "SYSCALL")>;
def: InstRW<[HWWriteResGroup10], (instregex "TEST(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "TEST8i8")>;
def: InstRW<[HWWriteResGroup10], (instregex "TEST8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "TEST8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "XCHG(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "XOR(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "XOR(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "XOR8i8")>;
def: InstRW<[HWWriteResGroup10], (instregex "XOR8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "XOR8rr")>;

def HWWriteResGroup11 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup11], (instregex "CVTPS2PDrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSLLDrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSLLQrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSLLWrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSRADrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSRAWrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSRLDrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSRLQrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSRLWrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VCVTPH2PSrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VCVTPS2PDrm")>;

def HWWriteResGroup11_1 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup11_1], (instregex "CVTSS2SDrm")>;
def: InstRW<[HWWriteResGroup11_1], (instregex "VCVTPH2PSYrm")>;
def: InstRW<[HWWriteResGroup11_1], (instregex "VCVTSS2SDrm")>;
def: InstRW<[HWWriteResGroup11_1], (instregex "VPSLLVQrm")>;
def: InstRW<[HWWriteResGroup11_1], (instregex "VPSRLVQrm")>;
def: InstRW<[HWWriteResGroup11_1], (instregex "VTESTPDrm")>;
def: InstRW<[HWWriteResGroup11_1], (instregex "VTESTPSrm")>;

def HWWriteResGroup11_2 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSLLDYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSLLQYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSLLVQYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSLLWYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSRADYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSRAWYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSRLDYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSRLQYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSRLVQYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VPSRLWYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VTESTPDYrm")>;
def: InstRW<[HWWriteResGroup11_2], (instregex "VTESTPSYrm")>;

def HWWriteResGroup12 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup12], (instregex "ADDSDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "ADDSSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "BSF(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup12], (instregex "BSR(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup12], (instregex "CMPSDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "CMPSSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "COMISDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "COMISSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "FCOM32m")>;
def: InstRW<[HWWriteResGroup12], (instregex "FCOM64m")>;
def: InstRW<[HWWriteResGroup12], (instregex "FCOMP32m")>;
def: InstRW<[HWWriteResGroup12], (instregex "FCOMP64m")>;
def: InstRW<[HWWriteResGroup12], (instrs IMUL16m, IMUL32m, IMUL64m)>;
def: InstRW<[HWWriteResGroup12], (instrs IMUL16rm, IMUL16rmi, IMUL16rmi8, IMUL32rm, IMUL32rmi, IMUL32rmi8, IMUL64rm, IMUL64rmi32, IMUL64rmi8)>;
def: InstRW<[HWWriteResGroup12], (instrs IMUL8m)>;
def: InstRW<[HWWriteResGroup12], (instregex "LZCNT(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MAX(C?)SDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MAX(C?)SSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MIN(C?)SDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MIN(C?)SSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_CVTPI2PSirm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_CVTPS2PIirm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_CVTTPS2PIirm")>;
def: InstRW<[HWWriteResGroup12], (instrs MUL16m, MUL32m, MUL64m)>;
def: InstRW<[HWWriteResGroup12], (instrs MUL8m)>;
def: InstRW<[HWWriteResGroup12], (instregex "PDEP(32|64)rm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PEXT(32|64)rm")>;
def: InstRW<[HWWriteResGroup12], (instregex "POPCNT(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup12], (instregex "SUBSDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "SUBSSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "TZCNT(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup12], (instregex "UCOMISDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "UCOMISSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VADDSDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VADDSSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VCMPSDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VCMPSSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VCOMISDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VCOMISSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VMAX(C?)SDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VMAX(C?)SSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VMIN(C?)SDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VMIN(C?)SSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VSUBSDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VSUBSSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VUCOMISDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VUCOMISSrm")>;

def HWWriteResGroup13 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup13], (instregex "ANDNPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "ANDNPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "ANDPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "ANDPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "INSERTPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "ORPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "ORPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PACKSSDWrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PACKSSWBrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PACKUSDWrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PACKUSWBrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PALIGNRrmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "PBLENDWrmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "PSHUFBrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PSHUFDmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "PSHUFHWmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "PSHUFLWmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "PUNPCKHBWrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PUNPCKHDQrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PUNPCKHQDQrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PUNPCKHWDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PUNPCKLBWrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PUNPCKLDQrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PUNPCKLQDQrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "PUNPCKLWDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "SHUFPDrmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "SHUFPSrmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "UNPCKHPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "UNPCKHPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "UNPCKLPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "UNPCKLPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VANDNPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VANDNPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VANDPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VANDPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VINSERTPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VORPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VORPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPACKSSDWrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPACKSSWBrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPACKUSDWrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPACKUSWBrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPALIGNRrmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPBLENDWrmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPERMILPDmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPERMILPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPERMILPSmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPERMILPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPSHUFBrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPSHUFDmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPSHUFHWmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPSHUFLWmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPUNPCKHBWrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPUNPCKHDQrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPUNPCKHQDQrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPUNPCKHWDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPUNPCKLBWrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPUNPCKLDQrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPUNPCKLQDQrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VPUNPCKLWDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VSHUFPDrmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "VSHUFPSrmi")>;
def: InstRW<[HWWriteResGroup13], (instregex "VUNPCKHPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VUNPCKHPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VUNPCKLPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VUNPCKLPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VXORPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "VXORPSrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "XORPDrm")>;
def: InstRW<[HWWriteResGroup13], (instregex "XORPSrm")>;

def HWWriteResGroup13_1 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup13_1], (instregex "VANDNPDYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VANDNPSYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VANDPDYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VANDPSYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VORPDYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VORPSYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPACKSSDWYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPACKSSWBYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPACKUSDWYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPACKUSWBYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPALIGNRYrmi")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPBLENDWYrmi")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPERMILPDYmi")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPERMILPDYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPERMILPSYmi")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPERMILPSYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPMOVSXBDYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPMOVSXBQYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPMOVSXWQYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPSHUFBYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPSHUFDYmi")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPSHUFHWYmi")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPSHUFLWYmi")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPUNPCKHBWYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPUNPCKHDQYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPUNPCKHQDQYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPUNPCKHWDYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPUNPCKLBWYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPUNPCKLDQYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPUNPCKLQDQYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VPUNPCKLWDYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VSHUFPDYrmi")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VSHUFPSYrmi")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VUNPCKHPDYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VUNPCKHPSYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VUNPCKLPDYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VUNPCKLPSYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VXORPDYrm")>;
def: InstRW<[HWWriteResGroup13_1], (instregex "VXORPSYrm")>;

def HWWriteResGroup13_2 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup13_2], (instregex "MMX_PALIGNRrmi")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MMX_PINSRWrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MMX_PSHUFBrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MMX_PSHUFWmi")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MMX_PUNPCKHBWirm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MMX_PUNPCKHDQirm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MMX_PUNPCKHWDirm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MMX_PUNPCKLBWirm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MMX_PUNPCKLDQirm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MMX_PUNPCKLWDirm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MOVHPDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MOVHPSrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MOVLPDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "MOVLPSrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PINSRBrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PINSRDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PINSRQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PINSRWrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVSXBDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVSXBQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVSXBWrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVSXDQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVSXWDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVSXWQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVZXBDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVZXBQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVZXBWrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVZXDQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVZXWDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "PMOVZXWQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VMOVHPDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VMOVHPSrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VMOVLPDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VMOVLPSrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPINSRBrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPINSRDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPINSRQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPINSRWrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVSXBDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVSXBQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVSXBWrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVSXDQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVSXWDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVSXWQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVZXBDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVZXBQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVZXBWrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVZXDQrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVZXWDrm")>;
def: InstRW<[HWWriteResGroup13_2], (instregex "VPMOVZXWQrm")>;

def HWWriteResGroup14 : SchedWriteRes<[HWPort6,HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup14], (instregex "FARJMP64")>;
def: InstRW<[HWWriteResGroup14], (instregex "JMP(16|32|64)m")>;

def HWWriteResGroup15 : SchedWriteRes<[HWPort23,HWPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup15], (instregex "BT(16|32|64)mi8")>;
def: InstRW<[HWWriteResGroup15], (instregex "RORX32mi")>;
def: InstRW<[HWWriteResGroup15], (instregex "RORX64mi")>;
def: InstRW<[HWWriteResGroup15], (instregex "SARX32rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "SARX64rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "SHLX32rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "SHLX64rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "SHRX32rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "SHRX64rm")>;

def HWWriteResGroup16 : SchedWriteRes<[HWPort23,HWPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup16], (instregex "ANDN(32|64)rm")>;
def: InstRW<[HWWriteResGroup16], (instregex "BLSI(32|64)rm")>;
def: InstRW<[HWWriteResGroup16], (instregex "BLSMSK(32|64)rm")>;
def: InstRW<[HWWriteResGroup16], (instregex "BLSR(32|64)rm")>;
def: InstRW<[HWWriteResGroup16], (instregex "BZHI(32|64)rm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PABSBrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PABSDrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PABSWrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PADDBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PADDDirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PADDQirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PADDSBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PADDSWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PADDUSBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PADDUSWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PADDWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PAVGBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PAVGWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PCMPEQBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PCMPEQDirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PCMPEQWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PCMPGTBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PCMPGTDirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PCMPGTWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PMAXSWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PMAXUBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PMINSWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PMINUBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSIGNBrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSIGNDrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSIGNWrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSUBBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSUBDirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSUBQirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSUBSBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSUBSWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSUBUSBirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSUBUSWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PSUBWirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MOVBE(16|32|64)rm")>;

def HWWriteResGroup16_1 : SchedWriteRes<[HWPort23,HWPort15]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup16_1], (instregex "PABSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PABSDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PABSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PADDBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PADDDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PADDQrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PADDSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PADDSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PADDUSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PADDUSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PADDWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PAVGBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PAVGWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PCMPEQBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PCMPEQDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PCMPEQQrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PCMPEQWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PCMPGTBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PCMPGTDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PCMPGTWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMAXSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMAXSDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMAXSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMAXUBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMAXUDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMAXUWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMINSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMINSDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMINSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMINUBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMINUDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PMINUWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSIGNBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSIGNDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSIGNWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSUBBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSUBDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSUBQrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSUBSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSUBSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSUBUSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSUBUSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "PSUBWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPABSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPABSDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPABSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPADDBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPADDDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPADDQrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPADDSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPADDSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPADDUSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPADDUSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPADDWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPAVGBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPAVGWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPCMPEQBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPCMPEQDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPCMPEQQrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPCMPEQWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPCMPGTBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPCMPGTDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPCMPGTWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMAXSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMAXSDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMAXSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMAXUBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMAXUDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMAXUWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMINSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMINSDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMINSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMINUBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMINUDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPMINUWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSIGNBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSIGNDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSIGNWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSUBBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSUBDrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSUBQrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSUBSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSUBSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSUBUSBrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSUBUSWrm")>;
def: InstRW<[HWWriteResGroup16_1], (instregex "VPSUBWrm")>;

def HWWriteResGroup16_2 : SchedWriteRes<[HWPort23,HWPort15]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup16_2], (instregex "VPABSBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPABSDYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPABSWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPADDBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPADDDYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPADDQYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPADDSBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPADDSWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPADDUSBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPADDUSWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPADDWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPAVGBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPAVGWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPCMPEQBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPCMPEQDYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPCMPEQQYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPCMPEQWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPCMPGTBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPCMPGTDYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPCMPGTWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMAXSBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMAXSDYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMAXSWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMAXUBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMAXUDYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMAXUWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMINSBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMINSDYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMINSWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMINUBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMINUDYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPMINUWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSIGNBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSIGNDYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSIGNWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSUBBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSUBDYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSUBQYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSUBSBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSUBSWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSUBUSBYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSUBUSWYrm")>;
def: InstRW<[HWWriteResGroup16_2], (instregex "VPSUBWYrm")>;

def HWWriteResGroup17 : SchedWriteRes<[HWPort23,HWPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup17], (instregex "BLENDPDrmi")>;
def: InstRW<[HWWriteResGroup17], (instregex "BLENDPSrmi")>;
def: InstRW<[HWWriteResGroup17], (instregex "PANDNrm")>;
def: InstRW<[HWWriteResGroup17], (instregex "PANDrm")>;
def: InstRW<[HWWriteResGroup17], (instregex "PORrm")>;
def: InstRW<[HWWriteResGroup17], (instregex "PXORrm")>;
def: InstRW<[HWWriteResGroup17], (instregex "VBLENDPDrmi")>;
def: InstRW<[HWWriteResGroup17], (instregex "VBLENDPSrmi")>;
def: InstRW<[HWWriteResGroup17], (instregex "VINSERTF128rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "VINSERTI128rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "VPANDNrm")>;
def: InstRW<[HWWriteResGroup17], (instregex "VPANDrm")>;
def: InstRW<[HWWriteResGroup17], (instregex "VPBLENDDrmi")>;
def: InstRW<[HWWriteResGroup17], (instregex "VPORrm")>;
def: InstRW<[HWWriteResGroup17], (instregex "VPXORrm")>;

def HWWriteResGroup17_1 : SchedWriteRes<[HWPort23,HWPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup17_1], (instregex "MMX_PANDNirm")>;
def: InstRW<[HWWriteResGroup17_1], (instregex "MMX_PANDirm")>;
def: InstRW<[HWWriteResGroup17_1], (instregex "MMX_PORirm")>;
def: InstRW<[HWWriteResGroup17_1], (instregex "MMX_PXORirm")>;

def HWWriteResGroup17_2 : SchedWriteRes<[HWPort23,HWPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup17_2], (instregex "VBLENDPDYrmi")>;
def: InstRW<[HWWriteResGroup17_2], (instregex "VBLENDPSYrmi")>;
def: InstRW<[HWWriteResGroup17_2], (instregex "VPANDNYrm")>;
def: InstRW<[HWWriteResGroup17_2], (instregex "VPANDYrm")>;
def: InstRW<[HWWriteResGroup17_2], (instregex "VPBLENDDYrmi")>;
def: InstRW<[HWWriteResGroup17_2], (instregex "VPORYrm")>;
def: InstRW<[HWWriteResGroup17_2], (instregex "VPXORYrm")>;

def HWWriteResGroup18 : SchedWriteRes<[HWPort23,HWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup18], (instregex "ADD(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "ADD8rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "AND(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "AND8rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "CMP(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup18], (instregex "CMP(16|32|64)mr")>;
def: InstRW<[HWWriteResGroup18], (instregex "CMP(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "CMP8mi")>;
def: InstRW<[HWWriteResGroup18], (instregex "CMP8mr")>;
def: InstRW<[HWWriteResGroup18], (instregex "CMP8rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "OR(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "OR8rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "POP(16|32|64)r(mr)?")>;
def: InstRW<[HWWriteResGroup18], (instregex "SUB(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "SUB8rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "TEST(16|32|64)mr")>;
def: InstRW<[HWWriteResGroup18], (instregex "TEST8mi")>;
def: InstRW<[HWWriteResGroup18], (instregex "TEST8mr")>;
def: InstRW<[HWWriteResGroup18], (instregex "XOR(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup18], (instregex "XOR8rm")>;

def HWWriteResGroup19 : SchedWriteRes<[HWPort237,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup19], (instregex "SFENCE")>;

def HWWriteResGroup20 : SchedWriteRes<[HWPort4,HWPort5,HWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup20], (instregex "EXTRACTPSmr")>;
def: InstRW<[HWWriteResGroup20], (instregex "PEXTRBmr")>;
def: InstRW<[HWWriteResGroup20], (instregex "PEXTRDmr")>;
def: InstRW<[HWWriteResGroup20], (instregex "PEXTRQmr")>;
def: InstRW<[HWWriteResGroup20], (instregex "PEXTRWmr")>;
def: InstRW<[HWWriteResGroup20], (instregex "STMXCSR")>;
def: InstRW<[HWWriteResGroup20], (instregex "VEXTRACTPSmr")>;
def: InstRW<[HWWriteResGroup20], (instregex "VPEXTRBmr")>;
def: InstRW<[HWWriteResGroup20], (instregex "VPEXTRDmr")>;
def: InstRW<[HWWriteResGroup20], (instregex "VPEXTRQmr")>;
def: InstRW<[HWWriteResGroup20], (instregex "VPEXTRWmr")>;
def: InstRW<[HWWriteResGroup20], (instregex "VSTMXCSR")>;

def HWWriteResGroup21 : SchedWriteRes<[HWPort4,HWPort6,HWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup21], (instregex "FNSTCW16m")>;

def HWWriteResGroup22 : SchedWriteRes<[HWPort4,HWPort237,HWPort06]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup22], (instregex "SET(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)m")>;

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
def: InstRW<[HWWriteResGroup24], (instregex "PUSH(16|32|64)r(mr)?")>;
def: InstRW<[HWWriteResGroup24], (instregex "PUSH64i8")>;
def: InstRW<[HWWriteResGroup24], (instregex "STOSB")>;
def: InstRW<[HWWriteResGroup24], (instregex "STOSL")>;
def: InstRW<[HWWriteResGroup24], (instregex "STOSQ")>;
def: InstRW<[HWWriteResGroup24], (instregex "STOSW")>;

def HWWriteResGroup25 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort06]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup25], (instregex "BTC(16|32|64)mi8")>;
def: InstRW<[HWWriteResGroup25], (instregex "BTR(16|32|64)mi8")>;
def: InstRW<[HWWriteResGroup25], (instregex "BTS(16|32|64)mi8")>;
def: InstRW<[HWWriteResGroup25], (instregex "SAR(16|32|64)m1")>;
def: InstRW<[HWWriteResGroup25], (instregex "SAR(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup25], (instregex "SAR8m1")>;
def: InstRW<[HWWriteResGroup25], (instregex "SAR8mi")>;
def: InstRW<[HWWriteResGroup25], (instregex "SHL(16|32|64)m1")>;
def: InstRW<[HWWriteResGroup25], (instregex "SHL(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup25], (instregex "SHL8m1")>;
def: InstRW<[HWWriteResGroup25], (instregex "SHL8mi")>;
def: InstRW<[HWWriteResGroup25], (instregex "SHR(16|32|64)m1")>;
def: InstRW<[HWWriteResGroup25], (instregex "SHR(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup25], (instregex "SHR8m1")>;
def: InstRW<[HWWriteResGroup25], (instregex "SHR8mi")>;

def HWWriteResGroup26 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup26], (instregex "ADD(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup26], (instregex "ADD(16|32|64)mr")>;
def: InstRW<[HWWriteResGroup26], (instregex "ADD8mi")>;
def: InstRW<[HWWriteResGroup26], (instregex "ADD8mr")>;
def: InstRW<[HWWriteResGroup26], (instregex "AND(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup26], (instregex "AND(16|32|64)mr")>;
def: InstRW<[HWWriteResGroup26], (instregex "AND8mi")>;
def: InstRW<[HWWriteResGroup26], (instregex "AND8mr")>;
def: InstRW<[HWWriteResGroup26], (instregex "DEC(16|32|64)m")>;
def: InstRW<[HWWriteResGroup26], (instregex "DEC8m")>;
def: InstRW<[HWWriteResGroup26], (instregex "INC(16|32|64)m")>;
def: InstRW<[HWWriteResGroup26], (instregex "INC8m")>;
def: InstRW<[HWWriteResGroup26], (instregex "NEG(16|32|64)m")>;
def: InstRW<[HWWriteResGroup26], (instregex "NEG8m")>;
def: InstRW<[HWWriteResGroup26], (instregex "NOT(16|32|64)m")>;
def: InstRW<[HWWriteResGroup26], (instregex "NOT8m")>;
def: InstRW<[HWWriteResGroup26], (instregex "OR(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup26], (instregex "OR(16|32|64)mr")>;
def: InstRW<[HWWriteResGroup26], (instregex "OR8mi")>;
def: InstRW<[HWWriteResGroup26], (instregex "OR8mr")>;
def: InstRW<[HWWriteResGroup26], (instregex "POP(16|32|64)rmm")>;
def: InstRW<[HWWriteResGroup26], (instregex "PUSH(16|32|64)rmm")>;
def: InstRW<[HWWriteResGroup26], (instregex "SUB(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup26], (instregex "SUB(16|32|64)mr")>;
def: InstRW<[HWWriteResGroup26], (instregex "SUB8mi")>;
def: InstRW<[HWWriteResGroup26], (instregex "SUB8mr")>;
def: InstRW<[HWWriteResGroup26], (instregex "XOR(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup26], (instregex "XOR(16|32|64)mr")>;
def: InstRW<[HWWriteResGroup26], (instregex "XOR8mi")>;
def: InstRW<[HWWriteResGroup26], (instregex "XOR8mr")>;

def HWWriteResGroup27 : SchedWriteRes<[HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup27], (instregex "BLENDVPDrr0")>;
def: InstRW<[HWWriteResGroup27], (instregex "BLENDVPSrr0")>;
def: InstRW<[HWWriteResGroup27], (instregex "MMX_PINSRWrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "PBLENDVBrr0")>;
def: InstRW<[HWWriteResGroup27], (instregex "PINSRBrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "PINSRDrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "PINSRQrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "PINSRWrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "VBLENDVPDYrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "VBLENDVPDrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "VBLENDVPSYrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "VBLENDVPSrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "VPBLENDVBYrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "VPBLENDVBrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "VPINSRBrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "VPINSRDrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "VPINSRQrr")>;
def: InstRW<[HWWriteResGroup27], (instregex "VPINSRWrr")>;

def HWWriteResGroup28 : SchedWriteRes<[HWPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup28], (instregex "FDECSTP")>;

def HWWriteResGroup29 : SchedWriteRes<[HWPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup29], (instregex "ROL(16|32|64)r1")>;
def: InstRW<[HWWriteResGroup29], (instregex "ROL(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup29], (instregex "ROL8r1")>;
def: InstRW<[HWWriteResGroup29], (instregex "ROL8ri")>;
def: InstRW<[HWWriteResGroup29], (instregex "ROR(16|32|64)r1")>;
def: InstRW<[HWWriteResGroup29], (instregex "ROR(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup29], (instregex "ROR8r1")>;
def: InstRW<[HWWriteResGroup29], (instregex "ROR8ri")>;

def HWWriteResGroup30 : SchedWriteRes<[HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup30], (instregex "LFENCE")>;
def: InstRW<[HWWriteResGroup30], (instregex "MFENCE")>;
def: InstRW<[HWWriteResGroup30], (instregex "WAIT")>;
def: InstRW<[HWWriteResGroup30], (instregex "XGETBV")>;

def HWWriteResGroup31 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup31], (instregex "CVTPS2PDrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "CVTSS2SDrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "EXTRACTPSrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "MMX_PEXTRWrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PEXTRBrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PEXTRDrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PEXTRQrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PEXTRWrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PSLLDrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PSLLQrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PSLLWrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PSRADrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PSRAWrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PSRLDrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PSRLQrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PSRLWrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "PTESTrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VCVTPH2PSYrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VCVTPH2PSrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VCVTPS2PDrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VCVTSS2SDrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VEXTRACTPSrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPEXTRBrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPEXTRDrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPEXTRQrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPEXTRWrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPSLLDrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPSLLQrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPSLLWrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPSRADrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPSRAWrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPSRLDrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPSRLQrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPSRLWrr")>;
def: InstRW<[HWWriteResGroup31], (instregex "VPTESTrr")>;

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
def: InstRW<[HWWriteResGroup34], (instregex "BEXTR(32|64)rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "BSWAP(16|32|64)r")>;

def HWWriteResGroup35 : SchedWriteRes<[HWPort06,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup35], (instregex "ADC(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup35], (instregex "ADC(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup35], (instregex "ADC8i8")>;
def: InstRW<[HWWriteResGroup35], (instregex "ADC8ri")>;
def: InstRW<[HWWriteResGroup35], (instregex "ADC8rr")>;
def: InstRW<[HWWriteResGroup35], (instregex "CMOV(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup35], (instregex "CWD")>;
def: InstRW<[HWWriteResGroup35], (instregex "JRCXZ")>;
def: InstRW<[HWWriteResGroup35], (instregex "SBB(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup35], (instregex "SBB(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup35], (instregex "SBB8i8")>;
def: InstRW<[HWWriteResGroup35], (instregex "SBB8ri")>;
def: InstRW<[HWWriteResGroup35], (instregex "SBB8rr")>;
def: InstRW<[HWWriteResGroup35], (instregex "SET(A|BE)r")>;

def HWWriteResGroup36 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup36], (instregex "BLENDVPDrm0")>;
def: InstRW<[HWWriteResGroup36], (instregex "BLENDVPSrm0")>;
def: InstRW<[HWWriteResGroup36], (instregex "PBLENDVBrm0")>;
def: InstRW<[HWWriteResGroup36], (instregex "VBLENDVPDrm")>;
def: InstRW<[HWWriteResGroup36], (instregex "VBLENDVPSrm")>;
def: InstRW<[HWWriteResGroup36], (instregex "VMASKMOVPDrm")>;
def: InstRW<[HWWriteResGroup36], (instregex "VMASKMOVPSrm")>;
def: InstRW<[HWWriteResGroup36], (instregex "VPBLENDVBrm")>;
def: InstRW<[HWWriteResGroup36], (instregex "VPMASKMOVDrm")>;
def: InstRW<[HWWriteResGroup36], (instregex "VPMASKMOVQrm")>;

def HWWriteResGroup36_1 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup36_1], (instregex "VBLENDVPDYrm")>;
def: InstRW<[HWWriteResGroup36_1], (instregex "VBLENDVPSYrm")>;
def: InstRW<[HWWriteResGroup36_1], (instregex "VMASKMOVPDYrm")>;
def: InstRW<[HWWriteResGroup36_1], (instregex "VMASKMOVPSYrm")>;
def: InstRW<[HWWriteResGroup36_1], (instregex "VPBLENDVBYrm")>;
def: InstRW<[HWWriteResGroup36_1], (instregex "VPMASKMOVDYrm")>;
def: InstRW<[HWWriteResGroup36_1], (instregex "VPMASKMOVQYrm")>;

def HWWriteResGroup36_2 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup36_2], (instregex "MMX_PACKSSDWirm")>;
def: InstRW<[HWWriteResGroup36_2], (instregex "MMX_PACKSSWBirm")>;
def: InstRW<[HWWriteResGroup36_2], (instregex "MMX_PACKUSWBirm")>;

def HWWriteResGroup37 : SchedWriteRes<[HWPort23,HWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup37], (instregex "LEAVE64")>;
def: InstRW<[HWWriteResGroup37], (instregex "SCASB")>;
def: InstRW<[HWWriteResGroup37], (instregex "SCASL")>;
def: InstRW<[HWWriteResGroup37], (instregex "SCASQ")>;
def: InstRW<[HWWriteResGroup37], (instregex "SCASW")>;

def HWWriteResGroup38 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup38], (instregex "PSLLDrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "PSLLQrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "PSLLWrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "PSRADrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "PSRAWrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "PSRLDrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "PSRLQrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "PSRLWrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "PTESTrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "VPSLLDrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "VPSLLQrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "VPSLLWrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "VPSRADrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "VPSRAWrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "VPSRLDrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "VPSRLQrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "VPSRLWrm")>;
def: InstRW<[HWWriteResGroup38], (instregex "VPTESTrm")>;

def HWWriteResGroup39 : SchedWriteRes<[HWPort0,HWPort01,HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup39], (instregex "FLDCW16m")>;

def HWWriteResGroup40 : SchedWriteRes<[HWPort0,HWPort23,HWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup40], (instregex "LDMXCSR")>;
def: InstRW<[HWWriteResGroup40], (instregex "VLDMXCSR")>;

def HWWriteResGroup41 : SchedWriteRes<[HWPort6,HWPort23,HWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup41], (instregex "LRETQ")>;
def: InstRW<[HWWriteResGroup41], (instregex "RETL")>;
def: InstRW<[HWWriteResGroup41], (instregex "RETQ")>;

def HWWriteResGroup42 : SchedWriteRes<[HWPort23,HWPort06,HWPort15]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup42], (instregex "BEXTR(32|64)rm")>;

def HWWriteResGroup43 : SchedWriteRes<[HWPort23,HWPort06,HWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup43], (instregex "ADC(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup43], (instregex "ADC8rm")>;
def: InstRW<[HWWriteResGroup43], (instregex "CMOV(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup43], (instregex "SBB(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup43], (instregex "SBB8rm")>;

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
def: InstRW<[HWWriteResGroup45], (instregex "CALL64pcrel32")>;
def: InstRW<[HWWriteResGroup45], (instregex "SET(A|BE)m")>;

def HWWriteResGroup46 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort06]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[HWWriteResGroup46], (instregex "ROL(16|32|64)m1")>;
def: InstRW<[HWWriteResGroup46], (instregex "ROL(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup46], (instregex "ROL8m1")>;
def: InstRW<[HWWriteResGroup46], (instregex "ROL8mi")>;
def: InstRW<[HWWriteResGroup46], (instregex "ROR(16|32|64)m1")>;
def: InstRW<[HWWriteResGroup46], (instregex "ROR(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup46], (instregex "ROR8m1")>;
def: InstRW<[HWWriteResGroup46], (instregex "ROR8mi")>;

def HWWriteResGroup47 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[HWWriteResGroup47], (instregex "XADD(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup47], (instregex "XADD8rm")>;

def HWWriteResGroup48 : SchedWriteRes<[HWPort4,HWPort6,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[HWWriteResGroup48], (instregex "CALL(16|32|64)m")>;
def: InstRW<[HWWriteResGroup48], (instregex "FARCALL64")>;

def HWWriteResGroup49 : SchedWriteRes<[HWPort0]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup49], (instregex "MOVMSKPDrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "MOVMSKPSrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "PMOVMSKBrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VMOVMSKPDYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VMOVMSKPDrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VMOVMSKPSYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VMOVMSKPSrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVMSKBYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVMSKBrr")>;

def HWWriteResGroup50 : SchedWriteRes<[HWPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup50], (instregex "ADDPDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADDPSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADDSDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADDSSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADDSUBPDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADDSUBPSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADD_FPrST0")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADD_FST0r")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADD_FrST0")>;
def: InstRW<[HWWriteResGroup50], (instregex "BSF(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup50], (instregex "BSR(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup50], (instregex "CMPPDrri")>;
def: InstRW<[HWWriteResGroup50], (instregex "CMPPSrri")>;
def: InstRW<[HWWriteResGroup50], (instregex "CMPSDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "CMPSSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "COMISDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "COMISSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "CVTDQ2PSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "CVTPS2DQrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "CVTTPS2DQrr")>;
def: InstRW<[HWWriteResGroup50], (instrs IMUL16rr, IMUL32rr, IMUL32rri, IMUL32rri8, IMUL64rr, IMUL64rri32, IMUL64rri8)>;
def: InstRW<[HWWriteResGroup50], (instrs IMUL8r)>;
def: InstRW<[HWWriteResGroup50], (instregex "LZCNT(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup50], (instregex "MAX(C?)PDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "MAX(C?)PSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "MAX(C?)SDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "MAX(C?)SSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "MIN(C?)PDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "MIN(C?)PSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "MIN(C?)SDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "MIN(C?)SSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "MMX_CVTPI2PSirr")>;
def: InstRW<[HWWriteResGroup50], (instrs MUL8r)>;
def: InstRW<[HWWriteResGroup50], (instregex "PDEP(32|64)rr")>;
def: InstRW<[HWWriteResGroup50], (instregex "PEXT(32|64)rr")>;
def: InstRW<[HWWriteResGroup50], (instregex "POPCNT(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup50], (instregex "SHLD(16|32|64)rri8")>;
def: InstRW<[HWWriteResGroup50], (instregex "SHRD(16|32|64)rri8")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBPDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBPSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBR_FPrST0")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBR_FST0r")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBR_FrST0")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBSDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBSSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUB_FPrST0")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUB_FST0r")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUB_FrST0")>;
def: InstRW<[HWWriteResGroup50], (instregex "TZCNT(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup50], (instregex "UCOMISDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "UCOMISSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDPDYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDPDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDPSYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDPSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSUBPDYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSUBPDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSUBPSYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSUBPSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPPDYrri")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPPDrri")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPPSYrri")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPPSrri")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPSDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPSSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCOMISDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCOMISSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTDQ2PSYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTDQ2PSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTPS2DQYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTPS2DQrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTTPS2DQYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTTPS2DQrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAX(C?)PDYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAX(C?)PDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAX(C?)PSYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAX(C?)PSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAX(C?)SDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAX(C?)SSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMIN(C?)PDYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMIN(C?)PDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMIN(C?)PSYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMIN(C?)PSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMIN(C?)SDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMIN(C?)SSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBPDYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBPDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBPSYrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBPSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBSDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBSSrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VUCOMISDrr")>;
def: InstRW<[HWWriteResGroup50], (instregex "VUCOMISSrr")>;

def HWWriteResGroup50_16i : SchedWriteRes<[HWPort1, HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup50_16i], (instrs IMUL16rri, IMUL16rri8)>;

def HWWriteResGroup51 : SchedWriteRes<[HWPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup51], (instregex "VBROADCASTSDYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VBROADCASTSSYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VEXTRACTF128rr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VEXTRACTI128rr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VINSERTF128rr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VINSERTI128rr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPBROADCASTBYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPBROADCASTBrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPBROADCASTDYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPBROADCASTQYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPBROADCASTWYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPBROADCASTWrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERM2F128rr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERM2I128rr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERMDYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERMPDYri")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERMPSYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERMQYri")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXBDYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXBQYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXBWYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXDQYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXWDYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXWQYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXBDYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXBQYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXBWYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXDQYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXWDYrr")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXWQYrr")>;

def HWWriteResGroup52 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup52], (instregex "ADDPDrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "ADDPSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "ADDSUBPDrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "ADDSUBPSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "CMPPDrmi")>;
def: InstRW<[HWWriteResGroup52], (instregex "CMPPSrmi")>;
def: InstRW<[HWWriteResGroup52], (instregex "CVTDQ2PSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "CVTPS2DQrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "CVTTPS2DQrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "MAX(C?)PDrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "MAX(C?)PSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "MIN(C?)PDrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "MIN(C?)PSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "SUBPDrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "SUBPSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VADDPDrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VADDPSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VADDSUBPDrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VADDSUBPSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VCMPPDrmi")>;
def: InstRW<[HWWriteResGroup52], (instregex "VCMPPSrmi")>;
def: InstRW<[HWWriteResGroup52], (instregex "VCVTDQ2PSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VCVTPS2DQrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VCVTTPS2DQrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VMAX(C?)PDrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VMAX(C?)PSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VMIN(C?)PDrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VMIN(C?)PSrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VSUBPDrm")>;
def: InstRW<[HWWriteResGroup52], (instregex "VSUBPSrm")>;

def HWWriteResGroup52_1 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup52_1], (instregex "ADD_F32m")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "ADD_F64m")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "ILD_F16m")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "ILD_F32m")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "ILD_F64m")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "SUBR_F32m")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "SUBR_F64m")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "SUB_F32m")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "SUB_F64m")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VADDPDYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VADDPSYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VADDSUBPDYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VADDSUBPSYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VCMPPDYrmi")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VCMPPSYrmi")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VCVTDQ2PSYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VCVTPS2DQYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VCVTTPS2DQYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VMAX(C?)PDYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VMAX(C?)PSYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VMIN(C?)PDYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VMIN(C?)PSYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VSUBPDYrm")>;
def: InstRW<[HWWriteResGroup52_1], (instregex "VSUBPSYrm")>;

def HWWriteResGroup53 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup53], (instregex "VPERM2F128rm")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPERM2I128rm")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPERMDYrm")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPERMPDYmi")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPERMPSYrm")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPERMQYmi")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPMOVZXBDYrm")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPMOVZXBQYrm")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPMOVZXBWYrm")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPMOVZXDQYrm")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPMOVZXWQYrm")>;

def HWWriteResGroup53_1 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup53_1], (instregex "VPMOVSXBWYrm")>;
def: InstRW<[HWWriteResGroup53_1], (instregex "VPMOVSXDQYrm")>;
def: InstRW<[HWWriteResGroup53_1], (instregex "VPMOVSXWDYrm")>;
def: InstRW<[HWWriteResGroup53_1], (instregex "VPMOVZXWDYrm")>;

def HWWriteResGroup54 : SchedWriteRes<[HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[HWWriteResGroup54], (instregex "XADD(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup54], (instregex "XADD8rr")>;
def: InstRW<[HWWriteResGroup54], (instregex "XCHG8rr")>;

def HWWriteResGroup55 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup55], (instregex "VPSLLVDYrr")>;
def: InstRW<[HWWriteResGroup55], (instregex "VPSLLVDrr")>;
def: InstRW<[HWWriteResGroup55], (instregex "VPSRAVDYrr")>;
def: InstRW<[HWWriteResGroup55], (instregex "VPSRAVDrr")>;
def: InstRW<[HWWriteResGroup55], (instregex "VPSRLVDYrr")>;
def: InstRW<[HWWriteResGroup55], (instregex "VPSRLVDrr")>;

def HWWriteResGroup56 : SchedWriteRes<[HWPort5,HWPort15]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup56], (instregex "MMX_PHADDDrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "MMX_PHADDSWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "MMX_PHADDWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "MMX_PHSUBDrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "MMX_PHSUBSWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "MMX_PHSUBWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "PHADDDrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "PHADDSWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "PHADDWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "PHSUBDrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "PHSUBSWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "PHSUBWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHADDDYrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHADDDrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHADDSWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHADDSWYrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHADDWYrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHADDWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHSUBDYrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHSUBDrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHSUBSWrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHSUBSWYrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHSUBWYrr")>;
def: InstRW<[HWWriteResGroup56], (instregex "VPHSUBWrr")>;

def HWWriteResGroup57 : SchedWriteRes<[HWPort5,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup57], (instregex "MMX_PACKSSDWirr")>;
def: InstRW<[HWWriteResGroup57], (instregex "MMX_PACKSSWBirr")>;
def: InstRW<[HWWriteResGroup57], (instregex "MMX_PACKUSWBirr")>;

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
def: InstRW<[HWWriteResGroup59], (instregex "CMOV(A|BE)(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup59], (instregex "RCL(16|32|64)r1")>;
def: InstRW<[HWWriteResGroup59], (instregex "RCL(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup59], (instregex "RCL8r1")>;
def: InstRW<[HWWriteResGroup59], (instregex "RCL8ri")>;
def: InstRW<[HWWriteResGroup59], (instregex "RCR(16|32|64)r1")>;
def: InstRW<[HWWriteResGroup59], (instregex "RCR(16|32|64)ri")>;
def: InstRW<[HWWriteResGroup59], (instregex "RCR8r1")>;
def: InstRW<[HWWriteResGroup59], (instregex "RCR8ri")>;

def HWWriteResGroup60 : SchedWriteRes<[HWPort06,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup60], (instregex "ROL(16|32|64)rCL")>;
def: InstRW<[HWWriteResGroup60], (instregex "ROL8rCL")>;
def: InstRW<[HWWriteResGroup60], (instregex "ROR(16|32|64)rCL")>;
def: InstRW<[HWWriteResGroup60], (instregex "ROR8rCL")>;
def: InstRW<[HWWriteResGroup60], (instregex "SAR(16|32|64)rCL")>;
def: InstRW<[HWWriteResGroup60], (instregex "SAR8rCL")>;
def: InstRW<[HWWriteResGroup60], (instregex "SHL(16|32|64)rCL")>;
def: InstRW<[HWWriteResGroup60], (instregex "SHL8rCL")>;
def: InstRW<[HWWriteResGroup60], (instregex "SHR(16|32|64)rCL")>;
def: InstRW<[HWWriteResGroup60], (instregex "SHR8rCL")>;

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
def: InstRW<[HWWriteResGroup62], (instregex "ISTT_FP16m")>;
def: InstRW<[HWWriteResGroup62], (instregex "ISTT_FP32m")>;
def: InstRW<[HWWriteResGroup62], (instregex "ISTT_FP64m")>;
def: InstRW<[HWWriteResGroup62], (instregex "IST_F16m")>;
def: InstRW<[HWWriteResGroup62], (instregex "IST_F32m")>;
def: InstRW<[HWWriteResGroup62], (instregex "IST_FP16m")>;
def: InstRW<[HWWriteResGroup62], (instregex "IST_FP32m")>;
def: InstRW<[HWWriteResGroup62], (instregex "IST_FP64m")>;

def HWWriteResGroup63 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup63], (instregex "VPSLLVDYrm")>;
def: InstRW<[HWWriteResGroup63], (instregex "VPSRAVDYrm")>;
def: InstRW<[HWWriteResGroup63], (instregex "VPSRLVDYrm")>;

def HWWriteResGroup63_1 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup63_1], (instregex "VPSLLVDrm")>;
def: InstRW<[HWWriteResGroup63_1], (instregex "VPSRAVDrm")>;
def: InstRW<[HWWriteResGroup63_1], (instregex "VPSRLVDrm")>;

def HWWriteResGroup64 : SchedWriteRes<[HWPort5,HWPort23,HWPort15]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup64], (instregex "MMX_PHADDDrm")>;
def: InstRW<[HWWriteResGroup64], (instregex "MMX_PHADDSWrm")>;
def: InstRW<[HWWriteResGroup64], (instregex "MMX_PHADDWrm")>;
def: InstRW<[HWWriteResGroup64], (instregex "MMX_PHSUBDrm")>;
def: InstRW<[HWWriteResGroup64], (instregex "MMX_PHSUBSWrm")>;
def: InstRW<[HWWriteResGroup64], (instregex "MMX_PHSUBWrm")>;

def HWWriteResGroup64_1 : SchedWriteRes<[HWPort5,HWPort23,HWPort15]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup64_1], (instregex "VPHADDDYrm")>;
def: InstRW<[HWWriteResGroup64_1], (instregex "VPHADDSWYrm")>;
def: InstRW<[HWWriteResGroup64_1], (instregex "VPHADDWYrm")>;
def: InstRW<[HWWriteResGroup64_1], (instregex "VPHSUBDYrm")>;
def: InstRW<[HWWriteResGroup64_1], (instregex "VPHSUBSWYrm")>;
def: InstRW<[HWWriteResGroup64_1], (instregex "VPHSUBWYrm")>;

def HWWriteResGroup64_2 : SchedWriteRes<[HWPort5,HWPort23,HWPort15]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup64_2], (instregex "PHADDDrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "PHADDSWrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "PHADDWrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "PHSUBDrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "PHSUBSWrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "PHSUBWrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "VPHADDDrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "VPHADDSWrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "VPHADDWrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "VPHSUBDrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "VPHSUBSWrm")>;
def: InstRW<[HWWriteResGroup64_2], (instregex "VPHSUBWrm")>;

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
def: InstRW<[HWWriteResGroup66], (instregex "RCL(16|32|64)m1")>;
def: InstRW<[HWWriteResGroup66], (instregex "RCL(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup66], (instregex "RCL8m1")>;
def: InstRW<[HWWriteResGroup66], (instregex "RCL8mi")>;
def: InstRW<[HWWriteResGroup66], (instregex "RCR(16|32|64)m1")>;
def: InstRW<[HWWriteResGroup66], (instregex "RCR(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup66], (instregex "RCR8m1")>;
def: InstRW<[HWWriteResGroup66], (instregex "RCR8mi")>;

def HWWriteResGroup67 : SchedWriteRes<[HWPort23,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,2,1];
}
def: InstRW<[HWWriteResGroup67], (instregex "ROR(16|32|64)mCL")>;
def: InstRW<[HWWriteResGroup67], (instregex "ROR8mCL")>;

def HWWriteResGroup68 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[HWWriteResGroup68], (instregex "ADC(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup68], (instregex "ADC8mi")>;
def: InstRW<[HWWriteResGroup68], (instregex "ADD8mi")>;
def: InstRW<[HWWriteResGroup68], (instregex "AND8mi")>;
def: InstRW<[HWWriteResGroup68], (instregex "OR8mi")>;
def: InstRW<[HWWriteResGroup68], (instregex "SUB8mi")>;
def: InstRW<[HWWriteResGroup68], (instregex "XCHG(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup68], (instregex "XCHG8rm")>;
def: InstRW<[HWWriteResGroup68], (instregex "XOR8mi")>;

def HWWriteResGroup69 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[HWWriteResGroup69], (instregex "ADC(16|32|64)mr")>;
def: InstRW<[HWWriteResGroup69], (instregex "ADC8mr")>;
def: InstRW<[HWWriteResGroup69], (instregex "CMPXCHG(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup69], (instregex "CMPXCHG8rm")>;
def: InstRW<[HWWriteResGroup69], (instregex "ROL(16|32|64)mCL")>;
def: InstRW<[HWWriteResGroup69], (instregex "ROL8mCL")>;
def: InstRW<[HWWriteResGroup69], (instregex "SAR(16|32|64)mCL")>;
def: InstRW<[HWWriteResGroup69], (instregex "SAR8mCL")>;
def: InstRW<[HWWriteResGroup69], (instregex "SBB(16|32|64)mi")>;
def: InstRW<[HWWriteResGroup69], (instregex "SBB(16|32|64)mr")>;
def: InstRW<[HWWriteResGroup69], (instregex "SBB8mi")>;
def: InstRW<[HWWriteResGroup69], (instregex "SBB8mr")>;
def: InstRW<[HWWriteResGroup69], (instregex "SHL(16|32|64)mCL")>;
def: InstRW<[HWWriteResGroup69], (instregex "SHL8mCL")>;
def: InstRW<[HWWriteResGroup69], (instregex "SHR(16|32|64)mCL")>;
def: InstRW<[HWWriteResGroup69], (instregex "SHR8mCL")>;

def HWWriteResGroup70 : SchedWriteRes<[HWPort0,HWPort1]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup70], (instregex "CVTSD2SI64rr")>;
def: InstRW<[HWWriteResGroup70], (instregex "CVTSD2SIrr")>;
def: InstRW<[HWWriteResGroup70], (instregex "CVTSS2SI64rr")>;
def: InstRW<[HWWriteResGroup70], (instregex "CVTSS2SIrr")>;
def: InstRW<[HWWriteResGroup70], (instregex "CVTTSD2SI64rr")>;
def: InstRW<[HWWriteResGroup70], (instregex "CVTTSD2SIrr")>;
def: InstRW<[HWWriteResGroup70], (instregex "CVTTSS2SI64rr")>;
def: InstRW<[HWWriteResGroup70], (instregex "CVTTSS2SIrr")>;
def: InstRW<[HWWriteResGroup70], (instregex "VCVTSD2SI64rr")>;
def: InstRW<[HWWriteResGroup70], (instregex "VCVTSD2SIrr")>;
def: InstRW<[HWWriteResGroup70], (instregex "VCVTSS2SI64rr")>;
def: InstRW<[HWWriteResGroup70], (instregex "VCVTSS2SIrr")>;
def: InstRW<[HWWriteResGroup70], (instregex "VCVTTSD2SI64rr")>;
def: InstRW<[HWWriteResGroup70], (instregex "VCVTTSD2SIrr")>;
def: InstRW<[HWWriteResGroup70], (instregex "VCVTTSS2SI64rr")>;
def: InstRW<[HWWriteResGroup70], (instregex "VCVTTSS2SIrr")>;

def HWWriteResGroup71 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup71], (instregex "VCVTPS2PDYrr")>;
def: InstRW<[HWWriteResGroup71], (instregex "VPSLLDYrr")>;
def: InstRW<[HWWriteResGroup71], (instregex "VPSLLQYrr")>;
def: InstRW<[HWWriteResGroup71], (instregex "VPSLLWYrr")>;
def: InstRW<[HWWriteResGroup71], (instregex "VPSRADYrr")>;
def: InstRW<[HWWriteResGroup71], (instregex "VPSRAWYrr")>;
def: InstRW<[HWWriteResGroup71], (instregex "VPSRLDYrr")>;
def: InstRW<[HWWriteResGroup71], (instregex "VPSRLQYrr")>;
def: InstRW<[HWWriteResGroup71], (instregex "VPSRLWYrr")>;
def: InstRW<[HWWriteResGroup71], (instregex "VPTESTYrr")>;

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
def: InstRW<[HWWriteResGroup73], (instregex "CVTDQ2PDrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "CVTPD2DQrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "CVTPD2PSrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "CVTSD2SSrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "CVTSI642SDrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "CVTSI2SDrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "CVTSI2SSrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "CVTTPD2DQrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "MMX_CVTPD2PIirr")>;
def: InstRW<[HWWriteResGroup73], (instregex "MMX_CVTPI2PDirr")>;
def: InstRW<[HWWriteResGroup73], (instregex "MMX_CVTPS2PIirr")>;
def: InstRW<[HWWriteResGroup73], (instregex "MMX_CVTTPD2PIirr")>;
def: InstRW<[HWWriteResGroup73], (instregex "MMX_CVTTPS2PIirr")>;
def: InstRW<[HWWriteResGroup73], (instregex "VCVTDQ2PDrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "VCVTPD2DQrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "VCVTPD2PSrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "VCVTPS2PHrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "VCVTSD2SSrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "VCVTSI642SDrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "VCVTSI2SDrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "VCVTSI2SSrr")>;
def: InstRW<[HWWriteResGroup73], (instregex "VCVTTPD2DQrr")>;

def HWWriteResGroup74 : SchedWriteRes<[HWPort1,HWPort6]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup74], (instrs IMUL64r)>;
def: InstRW<[HWWriteResGroup74], (instrs MUL64r)>;
def: InstRW<[HWWriteResGroup74], (instrs MULX64rr)>;

def HWWriteResGroup74_16 : SchedWriteRes<[HWPort1, HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def: InstRW<[HWWriteResGroup74_16], (instrs IMUL16r)>;
def: InstRW<[HWWriteResGroup74_16], (instrs MUL16r)>;

def HWWriteResGroup74_32 : SchedWriteRes<[HWPort1,HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 3;
}
def: InstRW<[HWWriteResGroup74_32], (instrs IMUL32r)>;
def: InstRW<[HWWriteResGroup74_32], (instrs MUL32r)>;

def HWWriteResGroup75 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup75], (instregex "FICOM16m")>;
def: InstRW<[HWWriteResGroup75], (instregex "FICOM32m")>;
def: InstRW<[HWWriteResGroup75], (instregex "FICOMP16m")>;
def: InstRW<[HWWriteResGroup75], (instregex "FICOMP32m")>;

def HWWriteResGroup76 : SchedWriteRes<[HWPort0,HWPort1,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup76], (instregex "CVTSD2SI64rm")>;
def: InstRW<[HWWriteResGroup76], (instregex "CVTSD2SIrm")>;
def: InstRW<[HWWriteResGroup76], (instregex "CVTSS2SI64rm")>;
def: InstRW<[HWWriteResGroup76], (instregex "CVTSS2SIrm")>;
def: InstRW<[HWWriteResGroup76], (instregex "CVTTSD2SI64rm")>;
def: InstRW<[HWWriteResGroup76], (instregex "CVTTSD2SIrm")>;
def: InstRW<[HWWriteResGroup76], (instregex "CVTTSS2SIrm")>;
def: InstRW<[HWWriteResGroup76], (instregex "VCVTSD2SI64rm")>;
def: InstRW<[HWWriteResGroup76], (instregex "VCVTSD2SIrm")>;
def: InstRW<[HWWriteResGroup76], (instregex "VCVTSS2SI64rm")>;
def: InstRW<[HWWriteResGroup76], (instregex "VCVTSS2SIrm")>;
def: InstRW<[HWWriteResGroup76], (instregex "VCVTTSD2SI64rm")>;
def: InstRW<[HWWriteResGroup76], (instregex "VCVTTSD2SIrm")>;
def: InstRW<[HWWriteResGroup76], (instregex "VCVTTSS2SI64rm")>;
def: InstRW<[HWWriteResGroup76], (instregex "VCVTTSS2SIrm")>;

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
def: InstRW<[HWWriteResGroup78], (instregex "CVTDQ2PDrm")>;
def: InstRW<[HWWriteResGroup78], (instregex "CVTPD2DQrm")>;
def: InstRW<[HWWriteResGroup78], (instregex "CVTPD2PSrm")>;
def: InstRW<[HWWriteResGroup78], (instregex "CVTTPD2DQrm")>;
def: InstRW<[HWWriteResGroup78], (instregex "MMX_CVTPD2PIirm")>;
def: InstRW<[HWWriteResGroup78], (instregex "MMX_CVTTPD2PIirm")>;
def: InstRW<[HWWriteResGroup78], (instregex "VCVTDQ2PDrm")>;

def HWWriteResGroup78_1 : SchedWriteRes<[HWPort1,HWPort5,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup78_1], (instregex "CVTSD2SSrm")>;
def: InstRW<[HWWriteResGroup78_1], (instregex "MMX_CVTPI2PDirm")>;
def: InstRW<[HWWriteResGroup78_1], (instregex "VCVTSD2SSrm")>;

def HWWriteResGroup79 : SchedWriteRes<[HWPort1,HWPort6,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup79], (instrs MULX64rm)>;

def HWWriteResGroup80 : SchedWriteRes<[HWPort5,HWPort23,HWPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup80], (instregex "VPBROADCASTBYrm")>;
def: InstRW<[HWWriteResGroup80], (instregex "VPBROADCASTBrm")>;
def: InstRW<[HWWriteResGroup80], (instregex "VPBROADCASTWYrm")>;
def: InstRW<[HWWriteResGroup80], (instregex "VPBROADCASTWrm")>;

def HWWriteResGroup81 : SchedWriteRes<[HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[HWWriteResGroup81], (instregex "FNCLEX")>;

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
def: InstRW<[HWWriteResGroup84], (instregex "VMASKMOVPDYmr")>;
def: InstRW<[HWWriteResGroup84], (instregex "VMASKMOVPDmr")>;
def: InstRW<[HWWriteResGroup84], (instregex "VMASKMOVPSYmr")>;
def: InstRW<[HWWriteResGroup84], (instregex "VMASKMOVPSmr")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMASKMOVDYmr")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMASKMOVDmr")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMASKMOVQYmr")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMASKMOVQmr")>;

def HWWriteResGroup85 : SchedWriteRes<[HWPort1,HWPort4,HWPort5,HWPort237]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup85], (instregex "VCVTPS2PHmr")>;

def HWWriteResGroup86 : SchedWriteRes<[HWPort1,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup86], (instregex "SHLD(16|32|64)mri8")>;
def: InstRW<[HWWriteResGroup86], (instregex "SHRD(16|32|64)mri8")>;

def HWWriteResGroup87 : SchedWriteRes<[HWPort1,HWPort6,HWPort23,HWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[HWWriteResGroup87], (instregex "LAR(16|32|64)rm")>;
def: InstRW<[HWWriteResGroup87], (instregex "LSL(16|32|64)rm")>;

def HWWriteResGroup88 : SchedWriteRes<[HWPort4,HWPort237,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[HWWriteResGroup88], (instregex "PUSHF16")>;
def: InstRW<[HWWriteResGroup88], (instregex "PUSHF64")>;

def HWWriteResGroup89 : SchedWriteRes<[HWPort0]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup89], (instregex "MMX_PMADDUBSWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "MMX_PMADDWDirr")>;
def: InstRW<[HWWriteResGroup89], (instregex "MMX_PMULHRSWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "MMX_PMULHUWirr")>;
def: InstRW<[HWWriteResGroup89], (instregex "MMX_PMULHWirr")>;
def: InstRW<[HWWriteResGroup89], (instregex "MMX_PMULLWirr")>;
def: InstRW<[HWWriteResGroup89], (instregex "MMX_PMULUDQirr")>;
def: InstRW<[HWWriteResGroup89], (instregex "MMX_PSADBWirr")>;
def: InstRW<[HWWriteResGroup89], (instregex "MUL_FPrST0")>;
def: InstRW<[HWWriteResGroup89], (instregex "MUL_FST0r")>;
def: InstRW<[HWWriteResGroup89], (instregex "MUL_FrST0")>;
def: InstRW<[HWWriteResGroup89], (instregex "PCMPGTQrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "PHMINPOSUWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "PMADDUBSWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "PMADDWDrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "PMULDQrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "PMULHRSWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "PMULHUWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "PMULHWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "PMULLWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "PMULUDQrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "PSADBWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "RCPPSr")>;
def: InstRW<[HWWriteResGroup89], (instregex "RCPSSr")>;
def: InstRW<[HWWriteResGroup89], (instregex "RSQRTPSr")>;
def: InstRW<[HWWriteResGroup89], (instregex "RSQRTSSr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPCMPGTQYrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPCMPGTQrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPHMINPOSUWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMADDUBSWYrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMADDUBSWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMADDWDYrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMADDWDrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULDQYrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULDQrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULHRSWYrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULHRSWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULHUWYrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULHUWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULHWYrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULHWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULLWYrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULLWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULUDQYrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPMULUDQrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPSADBWYrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VPSADBWrr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VRCPPSr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VRCPSSr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VRSQRTPSr")>;
def: InstRW<[HWWriteResGroup89], (instregex "VRSQRTSSr")>;

def HWWriteResGroup90 : SchedWriteRes<[HWPort01]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup90], (instregex "MULPDrr")>;
def: InstRW<[HWWriteResGroup90], (instregex "MULPSrr")>;
def: InstRW<[HWWriteResGroup90], (instregex "MULSDrr")>;
def: InstRW<[HWWriteResGroup90], (instregex "MULSSrr")>;
def: InstRW<[HWWriteResGroup90], (instregex "VMULPDYrr")>;
def: InstRW<[HWWriteResGroup90], (instregex "VMULPDrr")>;
def: InstRW<[HWWriteResGroup90], (instregex "VMULPSYrr")>;
def: InstRW<[HWWriteResGroup90], (instregex "VMULPSrr")>;
def: InstRW<[HWWriteResGroup90], (instregex "VMULSDrr")>;
def: InstRW<[HWWriteResGroup90], (instregex "VMULSSrr")>;
def: InstRW<[HWWriteResGroup90],
            (instregex "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)(Y)?r",
                       "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)r")>;

def HWWriteResGroup91 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup91], (instregex "MMX_PMADDUBSWrm")>;
def: InstRW<[HWWriteResGroup91], (instregex "MMX_PMADDWDirm")>;
def: InstRW<[HWWriteResGroup91], (instregex "MMX_PMULHRSWrm")>;
def: InstRW<[HWWriteResGroup91], (instregex "MMX_PMULHUWirm")>;
def: InstRW<[HWWriteResGroup91], (instregex "MMX_PMULHWirm")>;
def: InstRW<[HWWriteResGroup91], (instregex "MMX_PMULLWirm")>;
def: InstRW<[HWWriteResGroup91], (instregex "MMX_PMULUDQirm")>;
def: InstRW<[HWWriteResGroup91], (instregex "MMX_PSADBWirm")>;
def: InstRW<[HWWriteResGroup91], (instregex "RCPSSm")>;
def: InstRW<[HWWriteResGroup91], (instregex "RSQRTSSm")>;
def: InstRW<[HWWriteResGroup91], (instregex "VRCPSSm")>;
def: InstRW<[HWWriteResGroup91], (instregex "VRSQRTSSm")>;

def HWWriteResGroup91_1 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup91_1], (instregex "SQRTSSm")>;
def: InstRW<[HWWriteResGroup91_1], (instregex "VDIVSSrm")>;

def HWWriteResGroup91_2 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup91_2], (instregex "PCMPGTQrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "PHMINPOSUWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "PMADDUBSWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "PMADDWDrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "PMULDQrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "PMULHRSWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "PMULHUWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "PMULHWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "PMULLWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "PMULUDQrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "PSADBWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "RCPPSm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "RSQRTPSm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPCMPGTQrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPHMINPOSUWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPMADDUBSWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPMADDWDrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPMULDQrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPMULHRSWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPMULHUWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPMULHWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPMULLWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPMULUDQrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VPSADBWrm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VRCPPSm")>;
def: InstRW<[HWWriteResGroup91_2], (instregex "VRSQRTPSm")>;

def HWWriteResGroup91_3 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup91_3], (instregex "MUL_F32m")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "MUL_F64m")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "VPCMPGTQYrm")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "VPMADDUBSWYrm")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "VPMADDWDYrm")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "VPMULDQYrm")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "VPMULHRSWYrm")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "VPMULHUWYrm")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "VPMULHWYrm")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "VPMULLWYrm")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "VPMULUDQYrm")>;
def: InstRW<[HWWriteResGroup91_3], (instregex "VPSADBWYrm")>;

def HWWriteResGroup92 : SchedWriteRes<[HWPort01,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup92], (instregex "MULPDrm")>;
def: InstRW<[HWWriteResGroup92], (instregex "MULPSrm")>;
def: InstRW<[HWWriteResGroup92], (instregex "VMULPDrm")>;
def: InstRW<[HWWriteResGroup92], (instregex "VMULPSrm")>;
def: InstRW<[HWWriteResGroup92],
            (instregex "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)m")>;

def HWWriteResGroup92_1 : SchedWriteRes<[HWPort01,HWPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup92_1], (instregex "VMULPDYrm")>;
def: InstRW<[HWWriteResGroup92_1], (instregex "VMULPSYrm")>;
def: InstRW<[HWWriteResGroup92_1],
            (instregex "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Ym")>;

def HWWriteResGroup92_2 : SchedWriteRes<[HWPort01,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup92_2], (instregex "MULSDrm")>;
def: InstRW<[HWWriteResGroup92_2], (instregex "MULSSrm")>;
def: InstRW<[HWWriteResGroup92_2], (instregex "VMULSDrm")>;
def: InstRW<[HWWriteResGroup92_2], (instregex "VMULSSrm")>;
def: InstRW<[HWWriteResGroup92_2],
            (instregex "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)m")>;

def HWWriteResGroup93 : SchedWriteRes<[HWPort1,HWPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup93], (instregex "CVTSI642SSrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "HADDPDrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "HADDPSrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "HSUBPDrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "HSUBPSrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "VCVTSI642SSrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "VHADDPDYrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "VHADDPDrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "VHADDPSYrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "VHADDPSrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "VHSUBPDYrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "VHSUBPDrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "VHSUBPSYrr")>;
def: InstRW<[HWWriteResGroup93], (instregex "VHSUBPSrr")>;

def HWWriteResGroup94 : SchedWriteRes<[HWPort1,HWPort6,HWPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup94], (instregex "STR(16|32|64)r")>;

def HWWriteResGroup95 : SchedWriteRes<[HWPort1,HWPort06,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup95], (instrs MULX32rr)>;

def HWWriteResGroup96 : SchedWriteRes<[HWPort1,HWPort5,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[HWWriteResGroup96], (instregex "HADDPDrm")>;
def: InstRW<[HWWriteResGroup96], (instregex "HADDPSrm")>;
def: InstRW<[HWWriteResGroup96], (instregex "HSUBPDrm")>;
def: InstRW<[HWWriteResGroup96], (instregex "HSUBPSrm")>;
def: InstRW<[HWWriteResGroup96], (instregex "VHADDPDrm")>;
def: InstRW<[HWWriteResGroup96], (instregex "VHADDPSrm")>;
def: InstRW<[HWWriteResGroup96], (instregex "VHSUBPDrm")>;
def: InstRW<[HWWriteResGroup96], (instregex "VHSUBPSrm")>;

def HWWriteResGroup96_1 : SchedWriteRes<[HWPort1,HWPort5,HWPort23]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[HWWriteResGroup96_1], (instregex "VHADDPDYrm")>;
def: InstRW<[HWWriteResGroup96_1], (instregex "VHADDPSYrm")>;
def: InstRW<[HWWriteResGroup96_1], (instregex "VHSUBPDYrm")>;
def: InstRW<[HWWriteResGroup96_1], (instregex "VHSUBPSYrm")>;

def HWWriteResGroup97 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup97], (instregex "CVTTSS2SI64rm")>;

def HWWriteResGroup98 : SchedWriteRes<[HWPort1,HWPort23,HWPort06,HWPort0156]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup98], (instrs MULX32rm)>;

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
def: InstRW<[HWWriteResGroup101], (instregex "CMPXCHG(16|32|64)rr")>;
def: InstRW<[HWWriteResGroup101], (instregex "CMPXCHG8rr")>;

def HWWriteResGroup102 : SchedWriteRes<[HWPort1,HWPort5]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup102], (instregex "VCVTDQ2PDYrr")>;
def: InstRW<[HWWriteResGroup102], (instregex "VCVTPD2DQYrr")>;
def: InstRW<[HWWriteResGroup102], (instregex "VCVTPD2PSYrr")>;
def: InstRW<[HWWriteResGroup102], (instregex "VCVTPS2PHYrr")>;
def: InstRW<[HWWriteResGroup102], (instregex "VCVTTPD2DQYrr")>;

def HWWriteResGroup103 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup103], (instregex "ADD_FI16m")>;
def: InstRW<[HWWriteResGroup103], (instregex "ADD_FI32m")>;
def: InstRW<[HWWriteResGroup103], (instregex "SUBR_FI16m")>;
def: InstRW<[HWWriteResGroup103], (instregex "SUBR_FI32m")>;
def: InstRW<[HWWriteResGroup103], (instregex "SUB_FI16m")>;
def: InstRW<[HWWriteResGroup103], (instregex "SUB_FI32m")>;
def: InstRW<[HWWriteResGroup103], (instregex "VROUNDYPDm")>;
def: InstRW<[HWWriteResGroup103], (instregex "VROUNDYPSm")>;

def HWWriteResGroup103_1 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup103_1], (instregex "ROUNDPDm")>;
def: InstRW<[HWWriteResGroup103_1], (instregex "ROUNDPSm")>;
def: InstRW<[HWWriteResGroup103_1], (instregex "ROUNDSDm")>;
def: InstRW<[HWWriteResGroup103_1], (instregex "ROUNDSSm")>;
def: InstRW<[HWWriteResGroup103_1], (instregex "VROUNDPDm")>;
def: InstRW<[HWWriteResGroup103_1], (instregex "VROUNDPSm")>;
def: InstRW<[HWWriteResGroup103_1], (instregex "VROUNDSDm")>;
def: InstRW<[HWWriteResGroup103_1], (instregex "VROUNDSSm")>;

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
def: InstRW<[HWWriteResGroup105], (instregex "SHLD(16|32|64)rrCL")>;
def: InstRW<[HWWriteResGroup105], (instregex "SHRD(16|32|64)rrCL")>;

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
def: InstRW<[HWWriteResGroup109], (instregex "SHLD(16|32|64)mrCL")>;
def: InstRW<[HWWriteResGroup109], (instregex "SHRD(16|32|64)mrCL")>;

def HWWriteResGroup110 : SchedWriteRes<[HWPort5]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup110], (instregex "AESDECLASTrr")>;
def: InstRW<[HWWriteResGroup110], (instregex "AESDECrr")>;
def: InstRW<[HWWriteResGroup110], (instregex "AESENCLASTrr")>;
def: InstRW<[HWWriteResGroup110], (instregex "AESENCrr")>;
def: InstRW<[HWWriteResGroup110], (instregex "VAESDECLASTrr")>;
def: InstRW<[HWWriteResGroup110], (instregex "VAESDECrr")>;
def: InstRW<[HWWriteResGroup110], (instregex "VAESENCLASTrr")>;
def: InstRW<[HWWriteResGroup110], (instregex "VAESENCrr")>;

def HWWriteResGroup111 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 13;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup111], (instregex "AESDECLASTrm")>;
def: InstRW<[HWWriteResGroup111], (instregex "AESDECrm")>;
def: InstRW<[HWWriteResGroup111], (instregex "AESENCLASTrm")>;
def: InstRW<[HWWriteResGroup111], (instregex "AESENCrm")>;
def: InstRW<[HWWriteResGroup111], (instregex "VAESDECLASTrm")>;
def: InstRW<[HWWriteResGroup111], (instregex "VAESDECrm")>;
def: InstRW<[HWWriteResGroup111], (instregex "VAESENCLASTrm")>;
def: InstRW<[HWWriteResGroup111], (instregex "VAESENCrm")>;

def HWWriteResGroup112 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup112], (instregex "MPSADBWrri")>;
def: InstRW<[HWWriteResGroup112], (instregex "VMPSADBWYrri")>;
def: InstRW<[HWWriteResGroup112], (instregex "VMPSADBWrri")>;

def HWWriteResGroup113 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[HWWriteResGroup113], (instregex "MPSADBWrmi")>;
def: InstRW<[HWWriteResGroup113], (instregex "VMPSADBWrmi")>;

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
def: InstRW<[HWWriteResGroup114], (instregex "LOOP")>;

def HWWriteResGroup115 : SchedWriteRes<[HWPort0,HWPort1,HWPort23]> {
  let Latency = 15;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup115], (instregex "MUL_FI16m")>;
def: InstRW<[HWWriteResGroup115], (instregex "MUL_FI32m")>;

def HWWriteResGroup116 : SchedWriteRes<[HWPort0,HWPort1,HWPort5]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup116], (instregex "DPPDrri")>;
def: InstRW<[HWWriteResGroup116], (instregex "VDPPDrri")>;

def HWWriteResGroup117 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort23]> {
  let Latency = 15;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup117], (instregex "DPPDrmi")>;
def: InstRW<[HWWriteResGroup117], (instregex "VDPPDrmi")>;

def HWWriteResGroup118 : SchedWriteRes<[HWPort0]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup118], (instregex "PMULLDrr")>;
def: InstRW<[HWWriteResGroup118], (instregex "VPMULLDYrr")>;
def: InstRW<[HWWriteResGroup118], (instregex "VPMULLDrr")>;

def HWWriteResGroup119 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 16;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup119], (instregex "PMULLDrm")>;
def: InstRW<[HWWriteResGroup119], (instregex "VPMULLDrm")>;

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
def: InstRW<[HWWriteResGroup120], (instregex "RCL(16|32|64)mCL")>;
def: InstRW<[HWWriteResGroup120], (instregex "RCL8mCL")>;

def HWWriteResGroup121 : SchedWriteRes<[HWPort0]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup121], (instregex "DIVPSrr")>;
def: InstRW<[HWWriteResGroup121], (instregex "DIVSSrr")>;

def HWWriteResGroup122 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 17;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup122], (instregex "DIVPSrm")>;

def HWWriteResGroup122_1 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup122_1], (instregex "DIVSSrm")>;

def HWWriteResGroup123 : SchedWriteRes<[HWPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[HWWriteResGroup123], (instregex "PCMPISTRIrr")>;
def: InstRW<[HWWriteResGroup123], (instregex "PCMPISTRM128rr")>;
def: InstRW<[HWWriteResGroup123], (instregex "VPCMPISTRIrr")>;
def: InstRW<[HWWriteResGroup123], (instregex "VPCMPISTRM128rr")>;

def HWWriteResGroup124 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup124], (instregex "PCLMULQDQrr")>;
def: InstRW<[HWWriteResGroup124], (instregex "VPCLMULQDQrr")>;

def HWWriteResGroup125 : SchedWriteRes<[HWPort0,HWPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup125], (instregex "VRCPPSYr")>;
def: InstRW<[HWWriteResGroup125], (instregex "VRSQRTPSYr")>;

def HWWriteResGroup126 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}
def: InstRW<[HWWriteResGroup126], (instregex "PCMPISTRIrm")>;
def: InstRW<[HWWriteResGroup126], (instregex "PCMPISTRM128rm")>;
def: InstRW<[HWWriteResGroup126], (instregex "VPCMPISTRIrm")>;
def: InstRW<[HWWriteResGroup126], (instregex "VPCMPISTRM128rm")>;

def HWWriteResGroup127 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup127], (instregex "PCLMULQDQrm")>;
def: InstRW<[HWWriteResGroup127], (instregex "VPCLMULQDQrm")>;

def HWWriteResGroup128 : SchedWriteRes<[HWPort0,HWPort23,HWPort015]> {
  let Latency = 18;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup128], (instregex "VRCPPSYm")>;
def: InstRW<[HWWriteResGroup128], (instregex "VRSQRTPSYm")>;

def HWWriteResGroup129 : SchedWriteRes<[HWPort1,HWPort06,HWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,2,3];
}
def: InstRW<[HWWriteResGroup129], (instregex "RCL(16|32|64)rCL")>;
def: InstRW<[HWWriteResGroup129], (instregex "RCR(16|32|64)rCL")>;

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
def: InstRW<[HWWriteResGroup131], (instregex "LOOPE")>;
def: InstRW<[HWWriteResGroup131], (instregex "LOOPNE")>;

def HWWriteResGroup132 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 17;
  let NumMicroOps = 14;
  let ResourceCycles = [1,1,1,4,2,5];
}
def: InstRW<[HWWriteResGroup132], (instregex "CMPXCHG8B")>;

def HWWriteResGroup133 : SchedWriteRes<[HWPort0]> {
  let Latency = 13;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup133], (instregex "SQRTPSr")>;
def: InstRW<[HWWriteResGroup133], (instregex "SQRTSSr")>;
def: InstRW<[HWWriteResGroup133], (instregex "VDIVPSrr")>;
def: InstRW<[HWWriteResGroup133], (instregex "VDIVSSrr")>;

def HWWriteResGroup134 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup134], (instregex "DIVSDrm")>;
def: InstRW<[HWWriteResGroup134], (instregex "SQRTPSm")>;
def: InstRW<[HWWriteResGroup134], (instregex "VDIVPSrm")>;
def: InstRW<[HWWriteResGroup134], (instregex "VSQRTSSm")>;

def HWWriteResGroup135 : SchedWriteRes<[HWPort1,HWPort23,HWPort237,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 19;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,3,1,3];
}
def: InstRW<[HWWriteResGroup135], (instregex "RCR(16|32|64)mCL")>;
def: InstRW<[HWWriteResGroup135], (instregex "RCR8mCL")>;

def HWWriteResGroup136 : SchedWriteRes<[HWPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup136], (instregex "DIVPDrr")>;
def: InstRW<[HWWriteResGroup136], (instregex "DIVSDrr")>;
def: InstRW<[HWWriteResGroup136], (instregex "VSQRTPSr")>;
def: InstRW<[HWWriteResGroup136], (instregex "VSQRTSSr")>;

def HWWriteResGroup137 : SchedWriteRes<[HWPort5]> {
  let Latency = 14;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup137], (instregex "AESIMCrr")>;
def: InstRW<[HWWriteResGroup137], (instregex "VAESIMCrr")>;

def HWWriteResGroup138 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup138], (instregex "DIVPDrm")>;
def: InstRW<[HWWriteResGroup138], (instregex "VSQRTPSm")>;

def HWWriteResGroup139 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 20;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup139], (instregex "AESIMCrm")>;
def: InstRW<[HWWriteResGroup139], (instregex "VAESIMCrm")>;

def HWWriteResGroup140 : SchedWriteRes<[HWPort0,HWPort1,HWPort5]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup140], (instregex "DPPSrri")>;
def: InstRW<[HWWriteResGroup140], (instregex "VDPPSYrri")>;
def: InstRW<[HWWriteResGroup140], (instregex "VDPPSrri")>;

def HWWriteResGroup141 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort23]> {
  let Latency = 20;
  let NumMicroOps = 5;
  let ResourceCycles = [2,1,1,1];
}
def: InstRW<[HWWriteResGroup141], (instregex "DPPSrmi")>;
def: InstRW<[HWWriteResGroup141], (instregex "VDPPSrmi")>;

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
def: InstRW<[HWWriteResGroup144], (instregex "INSB")>;
def: InstRW<[HWWriteResGroup144], (instregex "INSL")>;
def: InstRW<[HWWriteResGroup144], (instregex "INSW")>;

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

def HWWriteResGroup148 : SchedWriteRes<[HWPort0,HWPort5,HWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def: InstRW<[HWWriteResGroup148], (instregex "PCMPESTRIrr")>;
def: InstRW<[HWWriteResGroup148], (instregex "VPCMPESTRIrr")>;

def HWWriteResGroup149 : SchedWriteRes<[HWPort5,HWPort6,HWPort06,HWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[HWWriteResGroup149], (instregex "CPUID")>;
def: InstRW<[HWWriteResGroup149], (instregex "RDTSC")>;

def HWWriteResGroup150 : SchedWriteRes<[HWPort0,HWPort5,HWPort23,HWPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[HWWriteResGroup150], (instregex "PCMPESTRIrm")>;
def: InstRW<[HWWriteResGroup150], (instregex "VPCMPESTRIrm")>;

def HWWriteResGroup151 : SchedWriteRes<[HWPort6,HWPort23,HWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 19;
  let ResourceCycles = [3,1,15];
}
def: InstRW<[HWWriteResGroup151], (instregex "XRSTOR(64)?")>;

def HWWriteResGroup152 : SchedWriteRes<[HWPort0,HWPort5,HWPort015,HWPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[HWWriteResGroup152], (instregex "PCMPESTRM128rr")>;
def: InstRW<[HWWriteResGroup152], (instregex "VPCMPESTRM128rr")>;

def HWWriteResGroup153 : SchedWriteRes<[HWPort0,HWPort5,HWPort23,HWPort015,HWPort0156]> {
  let Latency = 25;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}
def: InstRW<[HWWriteResGroup153], (instregex "PCMPESTRM128rm")>;
def: InstRW<[HWWriteResGroup153], (instregex "VPCMPESTRM128rm")>;

def HWWriteResGroup154 : SchedWriteRes<[HWPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup154], (instregex "DIV_FPrST0")>;
def: InstRW<[HWWriteResGroup154], (instregex "DIV_FST0r")>;
def: InstRW<[HWWriteResGroup154], (instregex "DIV_FrST0")>;
def: InstRW<[HWWriteResGroup154], (instregex "SQRTPDr")>;
def: InstRW<[HWWriteResGroup154], (instregex "SQRTSDr")>;
def: InstRW<[HWWriteResGroup154], (instregex "VDIVPDrr")>;
def: InstRW<[HWWriteResGroup154], (instregex "VDIVSDrr")>;

def HWWriteResGroup155 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 27;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup155], (instregex "DIVR_F32m")>;
def: InstRW<[HWWriteResGroup155], (instregex "DIVR_F64m")>;
def: InstRW<[HWWriteResGroup155], (instregex "VSQRTPDm")>;

def HWWriteResGroup155_1 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 26;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup155_1], (instregex "SQRTPDm")>;
def: InstRW<[HWWriteResGroup155_1], (instregex "VDIVPDrm")>;
def: InstRW<[HWWriteResGroup155_1], (instregex "VSQRTSDm")>;

def HWWriteResGroup155_2 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup155_2], (instregex "SQRTSDm")>;
def: InstRW<[HWWriteResGroup155_2], (instregex "VDIVSDrm")>;

def HWWriteResGroup156 : SchedWriteRes<[HWPort5,HWPort6,HWPort0156]> {
  let Latency = 20;
  let NumMicroOps = 10;
  let ResourceCycles = [1,2,7];
}
def: InstRW<[HWWriteResGroup156], (instregex "MWAITrr")>;

def HWWriteResGroup157 : SchedWriteRes<[HWPort0]> {
  let Latency = 21;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup157], (instregex "VSQRTPDr")>;
def: InstRW<[HWWriteResGroup157], (instregex "VSQRTSDr")>;

def HWWriteResGroup159 : SchedWriteRes<[HWPort0,HWPort015]> {
  let Latency = 21;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup159], (instregex "VDIVPSYrr")>;
def: InstRW<[HWWriteResGroup159], (instregex "VSQRTPSYr")>;

def HWWriteResGroup160 : SchedWriteRes<[HWPort0,HWPort23,HWPort015]> {
  let Latency = 28;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup160], (instregex "VDIVPSYrm")>;
def: InstRW<[HWWriteResGroup160], (instregex "VSQRTPSYm")>;

def HWWriteResGroup161 : SchedWriteRes<[HWPort0,HWPort1,HWPort23]> {
  let Latency = 30;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup161], (instregex "DIVR_FI16m")>;
def: InstRW<[HWWriteResGroup161], (instregex "DIVR_FI32m")>;

def HWWriteResGroup162 : SchedWriteRes<[HWPort0]> {
  let Latency = 24;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup162], (instregex "DIVR_FPrST0")>;
def: InstRW<[HWWriteResGroup162], (instregex "DIVR_FST0r")>;
def: InstRW<[HWWriteResGroup162], (instregex "DIVR_FrST0")>;

def HWWriteResGroup163 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 31;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup163], (instregex "DIV_F32m")>;
def: InstRW<[HWWriteResGroup163], (instregex "DIV_F64m")>;

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
def: InstRW<[HWWriteResGroup165], (instregex "XSAVE(OPT)?")>;

def HWWriteResGroup166 : SchedWriteRes<[HWPort0,HWPort1,HWPort23]> {
  let Latency = 34;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup166], (instregex "DIV_FI16m")>;
def: InstRW<[HWWriteResGroup166], (instregex "DIV_FI32m")>;

def HWWriteResGroup167 : SchedWriteRes<[HWPort0,HWPort5,HWPort23,HWPort015]> {
  let Latency = 34;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,1,1];
}
def: InstRW<[HWWriteResGroup167], (instregex "AESKEYGENASSIST128rm")>;
def: InstRW<[HWWriteResGroup167], (instregex "VAESKEYGENASSIST128rm")>;

def HWWriteResGroup168 : SchedWriteRes<[HWPort0,HWPort5,HWPort015]> {
  let Latency = 29;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,2];
}
def: InstRW<[HWWriteResGroup168], (instregex "AESKEYGENASSIST128rr")>;
def: InstRW<[HWWriteResGroup168], (instregex "VAESKEYGENASSIST128rr")>;

def HWWriteResGroup170 : SchedWriteRes<[HWPort5,HWPort6,HWPort23,HWPort06,HWPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[HWWriteResGroup170], (instregex "IN(16|32)ri")>;
def: InstRW<[HWWriteResGroup170], (instregex "IN(16|32)rr")>;
def: InstRW<[HWWriteResGroup170], (instregex "IN8ri")>;
def: InstRW<[HWWriteResGroup170], (instregex "IN8rr")>;

def HWWriteResGroup171 : SchedWriteRes<[HWPort5,HWPort6,HWPort23,HWPort237,HWPort06,HWPort0156]> {
  let Latency = 36;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[HWWriteResGroup171], (instregex "OUT(16|32)ir")>;
def: InstRW<[HWWriteResGroup171], (instregex "OUT(16|32)rr")>;
def: InstRW<[HWWriteResGroup171], (instregex "OUT8ir")>;
def: InstRW<[HWWriteResGroup171], (instregex "OUT8rr")>;

def HWWriteResGroup172 : SchedWriteRes<[HWPort01,HWPort15,HWPort015,HWPort0156]> {
  let Latency = 31;
  let NumMicroOps = 31;
  let ResourceCycles = [8,1,21,1];
}
def: InstRW<[HWWriteResGroup172], (instregex "MMX_EMMS")>;

def HWWriteResGroup173 : SchedWriteRes<[HWPort0,HWPort015]> {
  let Latency = 35;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup173], (instregex "VDIVPDYrr")>;
def: InstRW<[HWWriteResGroup173], (instregex "VSQRTPDYr")>;

def HWWriteResGroup174 : SchedWriteRes<[HWPort0,HWPort23,HWPort015]> {
  let Latency = 42;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup174], (instregex "VDIVPDYrm")>;
def: InstRW<[HWWriteResGroup174], (instregex "VSQRTPDYm")>;

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
def: InstRW<[HWWriteResGroup176], (instregex "RDTSCP")>;

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
def: InstRW<[HWWriteResGroup178], (instregex "FXRSTOR64")>;

def HWWriteResGroup179 : SchedWriteRes<[HWPort0,HWPort6,HWPort23,HWPort05,HWPort06,HWPort15,HWPort0156]> {
  let Latency = 64;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[HWWriteResGroup179], (instregex "FXRSTOR")>;

def HWWriteResGroup180 : SchedWriteRes<[HWPort5,HWPort01,HWPort0156]> {
  let Latency = 75;
  let NumMicroOps = 15;
  let ResourceCycles = [6,3,6];
}
def: InstRW<[HWWriteResGroup180], (instregex "FNINIT")>;

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
