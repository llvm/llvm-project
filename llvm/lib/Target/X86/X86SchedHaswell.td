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

// Remaining instrs.

def HWWriteResGroup0 : SchedWriteRes<[HWPort23]> {
  let Latency = 0;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup0], (instregex "LDDQUrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MMX_MOVD64rm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MMX_MOVD64to64rm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MMX_MOVQ64rm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOV32rm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOV64toPQIrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOV8rm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVAPDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVAPSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVDDUPrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVDI2PDIrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVDQArm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVDQUrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVNTDQArm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVSHDUPrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVSLDUPrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVSSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVSX32rm16")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVSX32rm8")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVUPDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVUPSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVZX32rm16")>;
def: InstRW<[HWWriteResGroup0], (instregex "MOVZX32rm8")>;
def: InstRW<[HWWriteResGroup0], (instregex "PREFETCHNTA")>;
def: InstRW<[HWWriteResGroup0], (instregex "PREFETCHT0")>;
def: InstRW<[HWWriteResGroup0], (instregex "PREFETCHT1")>;
def: InstRW<[HWWriteResGroup0], (instregex "PREFETCHT2")>;
def: InstRW<[HWWriteResGroup0], (instregex "VBROADCASTF128")>;
def: InstRW<[HWWriteResGroup0], (instregex "VBROADCASTI128")>;
def: InstRW<[HWWriteResGroup0], (instregex "VBROADCASTSDYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VBROADCASTSSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VBROADCASTSSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VLDDQUYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VLDDQUrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOV64toPQIrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVAPDYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVAPDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVAPSYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVAPSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVDDUPYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVDDUPrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVDI2PDIrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVDQAYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVDQArm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVDQUYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVDQUrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVNTDQAYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVNTDQArm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVQI2PQIrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVSDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVSHDUPYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVSHDUPrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVSLDUPYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVSLDUPrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVSSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVUPDYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVUPDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVUPSYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VMOVUPSrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VPBROADCASTDYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VPBROADCASTDrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VPBROADCASTQYrm")>;
def: InstRW<[HWWriteResGroup0], (instregex "VPBROADCASTQrm")>;

def HWWriteResGroup1 : SchedWriteRes<[HWPort4,HWPort237]> {
  let Latency = 0;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup1], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[HWWriteResGroup1], (instregex "MMX_MOVD64mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MMX_MOVNTQmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MMX_MOVQ64mr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOV64mr")>;
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
def: InstRW<[HWWriteResGroup1], (instregex "MOVSSmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVUPDmr")>;
def: InstRW<[HWWriteResGroup1], (instregex "MOVUPSmr")>;
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
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLDri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLDrm")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLQri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLQrm")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLVQYrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLVQrr")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLWri")>;
def: InstRW<[HWWriteResGroup2], (instregex "VPSLLWrm")>;
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
def: InstRW<[HWWriteResGroup3], (instregex "MASKMOVDQU64")>;
def: InstRW<[HWWriteResGroup3], (instregex "MMX_MASKMOVQ64")>;

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
def: InstRW<[HWWriteResGroup4], (instregex "KORTESTBrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_MOVD64rr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_MOVD64to64rr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_MOVQ2DQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PALIGNR64irr")>;
def: InstRW<[HWWriteResGroup4], (instregex "MMX_PSHUFBrr64")>;
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
def: InstRW<[HWWriteResGroup4], (instregex "VANDPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VANDPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VANDPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VBROADCASTSSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VGATHERQPSZrm")>;
def: InstRW<[HWWriteResGroup4], (instregex "VINSERTPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOV64toPQIrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOV64toPQIrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVAPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVAPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVAPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVAPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVDDUPYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VMOVDDUPrr")>;
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
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPDri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPDrm")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPSri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPSrm")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPERMILPSrr")>;
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
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFBrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFBrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFDYri")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPSHUFDri")>;
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
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHQDQrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHWDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VPUNPCKHWDrr")>;
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
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKHPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKHPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKLPDYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKLPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKLPSYrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VUNPCKLPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VXORPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "VXORPSrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "XORPDrr")>;
def: InstRW<[HWWriteResGroup4], (instregex "XORPSrr")>;

def HWWriteResGroup5 : SchedWriteRes<[HWPort6]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup5], (instregex "JMP64r")>;

def HWWriteResGroup6 : SchedWriteRes<[HWPort01]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup6], (instregex "FINCSTP")>;
def: InstRW<[HWWriteResGroup6], (instregex "FNOP")>;

def HWWriteResGroup7 : SchedWriteRes<[HWPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup7], (instregex "BT32ri8")>;
def: InstRW<[HWWriteResGroup7], (instregex "BT32rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTC32ri8")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTC32rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTR32ri8")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTR32rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTS32ri8")>;
def: InstRW<[HWWriteResGroup7], (instregex "BTS32rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "CDQ")>;
def: InstRW<[HWWriteResGroup7], (instregex "CQO")>;
def: InstRW<[HWWriteResGroup7], (instregex "RORX32ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "RORX64ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SAR32ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SAR64r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SAR8r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SAR8ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SARX32rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SARX64rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETAEr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETBr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETEr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETGEr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETGr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETLEr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETLr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETNEr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETNOr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETNPr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETNSr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETOr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETPr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SETSr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHL32ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHL64r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHL8r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHL8ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHLX32rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHLX64rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHR32ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHR64r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHR8r1")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHR8ri")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHRX32rr")>;
def: InstRW<[HWWriteResGroup7], (instregex "SHRX64rr")>;

def HWWriteResGroup8 : SchedWriteRes<[HWPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup8], (instregex "ANDN32rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "ANDN64rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BLSI32rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BLSI64rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BLSMSK32rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BLSMSK64rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BLSR32rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BLSR64rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BZHI32rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "BZHI64rr")>;
def: InstRW<[HWWriteResGroup8], (instregex "LEA64_32r")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PABSBrr64")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PABSDrr64")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PABSWrr64")>;
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
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSIGNBrr64")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSIGNDrr64")>;
def: InstRW<[HWWriteResGroup8], (instregex "MMX_PSIGNWrr64")>;
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
def: InstRW<[HWWriteResGroup8], (instregex "PSIGNBrr128")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSIGNDrr128")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSIGNWrr128")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBQrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBUSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBUSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "PSUBWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VMASKMOVPSYrm")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSDYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPABSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDQYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDQrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDUSBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDUSBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDUSWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDUSWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPADDWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPAVGBYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPAVGBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPAVGWYrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPAVGWrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQBrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQDrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQQrr")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPCMPEQWrr")>;
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
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNBYrr256")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNBrr128")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNDYrr256")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNDrr128")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNWYrr256")>;
def: InstRW<[HWWriteResGroup8], (instregex "VPSIGNWrr128")>;
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
def: InstRW<[HWWriteResGroup10], (instregex "ADD32ri8")>;
def: InstRW<[HWWriteResGroup10], (instregex "ADD32rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "ADD8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "ADD8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "AND32ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "AND64ri8")>;
def: InstRW<[HWWriteResGroup10], (instregex "AND64rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "AND8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "AND8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "CBW")>;
def: InstRW<[HWWriteResGroup10], (instregex "CLC")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMC")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMP16ri8")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMP32i32")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMP64rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMP8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "CMP8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "CWDE")>;
def: InstRW<[HWWriteResGroup10], (instregex "DEC64r")>;
def: InstRW<[HWWriteResGroup10], (instregex "DEC8r")>;
def: InstRW<[HWWriteResGroup10], (instregex "INC64r")>;
def: InstRW<[HWWriteResGroup10], (instregex "INC8r")>;
def: InstRW<[HWWriteResGroup10], (instregex "LAHF")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOV32rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOV8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOV8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOVSX32rr16")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOVSX32rr8")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOVZX32rr16")>;
def: InstRW<[HWWriteResGroup10], (instregex "MOVZX32rr8")>;
def: InstRW<[HWWriteResGroup10], (instregex "NEG64r")>;
def: InstRW<[HWWriteResGroup10], (instregex "NEG8r")>;
def: InstRW<[HWWriteResGroup10], (instregex "NOOP")>;
def: InstRW<[HWWriteResGroup10], (instregex "NOT64r")>;
def: InstRW<[HWWriteResGroup10], (instregex "NOT8r")>;
def: InstRW<[HWWriteResGroup10], (instregex "OR64ri8")>;
def: InstRW<[HWWriteResGroup10], (instregex "OR64rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "OR8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "OR8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "SAHF")>;
def: InstRW<[HWWriteResGroup10], (instregex "SGDT64m")>;
def: InstRW<[HWWriteResGroup10], (instregex "SIDT64m")>;
def: InstRW<[HWWriteResGroup10], (instregex "SLDT16m")>;
def: InstRW<[HWWriteResGroup10], (instregex "SMSW16m")>;
def: InstRW<[HWWriteResGroup10], (instregex "STC")>;
def: InstRW<[HWWriteResGroup10], (instregex "STRm")>;
def: InstRW<[HWWriteResGroup10], (instregex "SUB64ri8")>;
def: InstRW<[HWWriteResGroup10], (instregex "SUB64rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "SUB8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "SUB8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "SYSCALL")>;
def: InstRW<[HWWriteResGroup10], (instregex "TEST64rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "TEST8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "TEST8rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "XCHG64rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "XOR32rr")>;
def: InstRW<[HWWriteResGroup10], (instregex "XOR64ri8")>;
def: InstRW<[HWWriteResGroup10], (instregex "XOR8ri")>;
def: InstRW<[HWWriteResGroup10], (instregex "XOR8rr")>;

def HWWriteResGroup11 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup11], (instregex "CVTPS2PDrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "CVTSS2SDrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSLLDrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSLLQrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSLLWrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSRADrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSRAWrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSRLDrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSRLQrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "MMX_PSRLWrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VCVTPH2PSYrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VCVTPH2PSrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VCVTPS2PDrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VCVTSS2SDrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSLLDYri")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSLLQYri")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSLLVQYrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSLLVQrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSLLWYri")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSRADYrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSRAWYrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSRLDYrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSRLQYrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSRLVQYrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSRLVQrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VPSRLWYrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VTESTPDYrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VTESTPDrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VTESTPSYrm")>;
def: InstRW<[HWWriteResGroup11], (instregex "VTESTPSrm")>;

def HWWriteResGroup12 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup12], (instregex "ANDNPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "ANDNPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "ANDPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "ANDPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "INSERTPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_PALIGNR64irm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_PINSRWirmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_PSHUFBrm64")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_PSHUFWmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_PUNPCKHBWirm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_PUNPCKHDQirm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_PUNPCKHWDirm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_PUNPCKLBWirm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_PUNPCKLDQirm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MMX_PUNPCKLWDirm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MOVHPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MOVHPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MOVLPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "MOVLPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "ORPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "ORPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PACKSSDWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PACKSSWBrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PACKUSDWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PACKUSWBrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PALIGNRrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "PBLENDWrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "PINSRBrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PINSRDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PINSRQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PINSRWrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVSXBDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVSXBQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVSXBWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVSXDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVSXWDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVSXWQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVZXBDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVZXBQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVZXBWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVZXDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVZXWDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PMOVZXWQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PSHUFBrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PSHUFDmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "PSHUFHWmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "PSHUFLWmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "PUNPCKHBWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PUNPCKHDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PUNPCKHQDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PUNPCKHWDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PUNPCKLBWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PUNPCKLDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PUNPCKLQDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "PUNPCKLWDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "SHUFPDrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "SHUFPSrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "UNPCKHPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "UNPCKHPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "UNPCKLPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "UNPCKLPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VANDNPDYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VANDNPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VANDNPSYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VANDNPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VANDPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VANDPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VANDPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VANDPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VINSERTPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VMOVHPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VMOVHPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VMOVLPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VMOVLPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VORPDYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VORPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VORPSYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VORPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPACKSSDWYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPACKSSDWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPACKSSWBYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPACKSSWBrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPACKUSDWYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPACKUSDWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPACKUSWBYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPACKUSWBrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPALIGNRYrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPALIGNRrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPBLENDWYrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPBLENDWrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPERMILPDYri")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPERMILPDmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPERMILPDmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPERMILPDri")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPERMILPSYri")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPERMILPSmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPERMILPSmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPERMILPSri")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPINSRBrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPINSRDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPINSRQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPINSRWrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVSXBDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVSXBQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVSXBWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVSXDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVSXWDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVSXWQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVZXBDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVZXBQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVZXBWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVZXDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVZXWDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPMOVZXWQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPSHUFBrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPSHUFBrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPSHUFDYmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPSHUFDmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPSHUFHWmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPSHUFHWmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPSHUFLWYmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPSHUFLWmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKHBWYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKHBWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKHDQYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKHDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKHQDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKHQDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKHWDYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKHWDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKLBWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKLBWrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKLDQYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKLDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKLQDQYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKLQDQrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKLWDYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VPUNPCKLWDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VSHUFPDYrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VSHUFPDrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VSHUFPSYrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VSHUFPSrmi")>;
def: InstRW<[HWWriteResGroup12], (instregex "VUNPCKHPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VUNPCKHPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VUNPCKHPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VUNPCKHPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VUNPCKLPDYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VUNPCKLPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VUNPCKLPSYrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VUNPCKLPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VXORPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VXORPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VXORPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "VXORPSrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "XORPDrm")>;
def: InstRW<[HWWriteResGroup12], (instregex "XORPSrm")>;

def HWWriteResGroup13 : SchedWriteRes<[HWPort6,HWPort23]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup13], (instregex "FARJMP64")>;
def: InstRW<[HWWriteResGroup13], (instregex "JMP64m")>;

def HWWriteResGroup14 : SchedWriteRes<[HWPort23,HWPort0]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup14], (instregex "BT64mi8")>;
def: InstRW<[HWWriteResGroup14], (instregex "RORX32mi")>;
def: InstRW<[HWWriteResGroup14], (instregex "RORX64mi")>;
def: InstRW<[HWWriteResGroup14], (instregex "SARX32rm")>;
def: InstRW<[HWWriteResGroup14], (instregex "SARX64rm")>;
def: InstRW<[HWWriteResGroup14], (instregex "SHLX32rm")>;
def: InstRW<[HWWriteResGroup14], (instregex "SHLX64rm")>;
def: InstRW<[HWWriteResGroup14], (instregex "SHRX32rm")>;
def: InstRW<[HWWriteResGroup14], (instregex "SHRX64rm")>;

def HWWriteResGroup15 : SchedWriteRes<[HWPort23,HWPort15]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup15], (instregex "ANDN32rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "ANDN64rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "BLSI32rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "BLSI64rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "BLSMSK32rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "BLSMSK64rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "BLSR32rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "BLSR64rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "BZHI32rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "BZHI64rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PABSBrm64")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PABSDrm64")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PABSWrm64")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PADDBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PADDDirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PADDQirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PADDSBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PADDSWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PADDUSBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PADDUSWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PADDWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PAVGBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PAVGWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PCMPEQBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PCMPEQDirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PCMPEQWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PCMPGTBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PCMPGTDirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PCMPGTWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PMAXSWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PMAXUBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PMINSWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PMINUBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSIGNBrm64")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSIGNDrm64")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSIGNWrm64")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSUBBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSUBDirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSUBQirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSUBSBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSUBSWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSUBUSBirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSUBUSWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MMX_PSUBWirm")>;
def: InstRW<[HWWriteResGroup15], (instregex "MOVBE64rm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PABSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PABSDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PABSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PADDBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PADDDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PADDQrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PADDSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PADDSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PADDUSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PADDUSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PADDWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PAVGBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PAVGWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PCMPEQBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PCMPEQDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PCMPEQQrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PCMPEQWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PCMPGTBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PCMPGTDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PCMPGTWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMAXSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMAXSDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMAXSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMAXUBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMAXUDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMAXUWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMINSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMINSDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMINSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMINUBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMINUDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PMINUWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSIGNBrm128")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSIGNDrm128")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSIGNWrm128")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSUBBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSUBDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSUBQrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSUBSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSUBSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSUBUSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSUBUSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "PSUBWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPABSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPABSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPABSDYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPABSDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPABSWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPABSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDQYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDQrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDUSBYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDUSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDUSWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDUSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPADDWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPAVGBYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPAVGBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPAVGWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPAVGWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPEQBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPEQBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPEQDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPEQDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPEQQrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPEQQrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPEQWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPEQWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPGTBYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPGTBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPGTDYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPGTDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPGTWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPCMPGTWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXSBYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXSDYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXSDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXSWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXUBYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXUBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXUDYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXUDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXUWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMAXUWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINSBYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINSDYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINSDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINSWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINUBYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINUBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINUDYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINUDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINUWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPMINUWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSIGNBYrm256")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSIGNBrm128")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSIGNDYrm256")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSIGNDrm128")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSIGNWYrm256")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSIGNWrm128")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBBYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBDYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBDrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBQYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBQrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBSBYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBSWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBUSBYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBUSBrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBUSWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBUSWrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBWYrm")>;
def: InstRW<[HWWriteResGroup15], (instregex "VPSUBWrm")>;

def HWWriteResGroup16 : SchedWriteRes<[HWPort23,HWPort015]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup16], (instregex "BLENDPDrmi")>;
def: InstRW<[HWWriteResGroup16], (instregex "BLENDPSrmi")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PANDNirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PANDirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PORirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "MMX_PXORirm")>;
def: InstRW<[HWWriteResGroup16], (instregex "PANDNrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "PANDrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "PORrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "PXORrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "VBLENDPDYrmi")>;
def: InstRW<[HWWriteResGroup16], (instregex "VBLENDPDrmi")>;
def: InstRW<[HWWriteResGroup16], (instregex "VBLENDPSYrmi")>;
def: InstRW<[HWWriteResGroup16], (instregex "VBLENDPSrmi")>;
def: InstRW<[HWWriteResGroup16], (instregex "VINSERTF128rm")>;
def: InstRW<[HWWriteResGroup16], (instregex "VINSERTI128rm")>;
def: InstRW<[HWWriteResGroup16], (instregex "VPANDNYrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "VPANDNrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "VPANDYrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "VPANDrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "VPBLENDDYrmi")>;
def: InstRW<[HWWriteResGroup16], (instregex "VPBLENDDrmi")>;
def: InstRW<[HWWriteResGroup16], (instregex "VPORYrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "VPORrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "VPXORYrm")>;
def: InstRW<[HWWriteResGroup16], (instregex "VPXORrm")>;

def HWWriteResGroup17 : SchedWriteRes<[HWPort23,HWPort0156]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup17], (instregex "ADD64rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "ADD8rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "AND64rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "AND8rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "CMP64mi8")>;
def: InstRW<[HWWriteResGroup17], (instregex "CMP64mr")>;
def: InstRW<[HWWriteResGroup17], (instregex "CMP64rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "CMP8mi")>;
def: InstRW<[HWWriteResGroup17], (instregex "CMP8mr")>;
def: InstRW<[HWWriteResGroup17], (instregex "CMP8rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "OR64rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "OR8rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "POP64r")>;
def: InstRW<[HWWriteResGroup17], (instregex "SUB64rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "SUB8rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "TEST64rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "TEST8mi")>;
def: InstRW<[HWWriteResGroup17], (instregex "TEST8rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "XOR64rm")>;
def: InstRW<[HWWriteResGroup17], (instregex "XOR8rm")>;

def HWWriteResGroup18 : SchedWriteRes<[HWPort237,HWPort0156]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup18], (instregex "SFENCE")>;

def HWWriteResGroup19 : SchedWriteRes<[HWPort4,HWPort5,HWPort237]> {
  let Latency = 1;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup19], (instregex "EXTRACTPSmr")>;
def: InstRW<[HWWriteResGroup19], (instregex "PEXTRBmr")>;
def: InstRW<[HWWriteResGroup19], (instregex "PEXTRDmr")>;
def: InstRW<[HWWriteResGroup19], (instregex "PEXTRQmr")>;
def: InstRW<[HWWriteResGroup19], (instregex "PEXTRWmr")>;
def: InstRW<[HWWriteResGroup19], (instregex "STMXCSR")>;
def: InstRW<[HWWriteResGroup19], (instregex "VEXTRACTPSmr")>;
def: InstRW<[HWWriteResGroup19], (instregex "VPEXTRBmr")>;
def: InstRW<[HWWriteResGroup19], (instregex "VPEXTRDmr")>;
def: InstRW<[HWWriteResGroup19], (instregex "VPEXTRQmr")>;
def: InstRW<[HWWriteResGroup19], (instregex "VPEXTRWmr")>;
def: InstRW<[HWWriteResGroup19], (instregex "VSTMXCSR")>;

def HWWriteResGroup20 : SchedWriteRes<[HWPort4,HWPort6,HWPort237]> {
  let Latency = 1;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup20], (instregex "FNSTCW16m")>;

def HWWriteResGroup21 : SchedWriteRes<[HWPort4,HWPort237,HWPort0]> {
  let Latency = 1;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup21], (instregex "SETAEm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETBm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETEm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETGEm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETGm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETLEm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETLm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETNEm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETNOm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETNPm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETNSm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETOm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETPm")>;
def: InstRW<[HWWriteResGroup21], (instregex "SETSm")>;

def HWWriteResGroup22 : SchedWriteRes<[HWPort4,HWPort237,HWPort15]> {
  let Latency = 1;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup22], (instregex "MOVBE64mr")>;

def HWWriteResGroup23 : SchedWriteRes<[HWPort4,HWPort237,HWPort0156]> {
  let Latency = 1;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup23], (instregex "PUSH64i8")>;
def: InstRW<[HWWriteResGroup23], (instregex "PUSH64r")>;
def: InstRW<[HWWriteResGroup23], (instregex "STOSB")>;
def: InstRW<[HWWriteResGroup23], (instregex "STOSL")>;
def: InstRW<[HWWriteResGroup23], (instregex "STOSQ")>;
def: InstRW<[HWWriteResGroup23], (instregex "STOSW")>;

def HWWriteResGroup24 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0]> {
  let Latency = 1;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup24], (instregex "BTC64mi8")>;
def: InstRW<[HWWriteResGroup24], (instregex "BTR64mi8")>;
def: InstRW<[HWWriteResGroup24], (instregex "BTS64mi8")>;
def: InstRW<[HWWriteResGroup24], (instregex "SAR64m1")>;
def: InstRW<[HWWriteResGroup24], (instregex "SAR64mi")>;
def: InstRW<[HWWriteResGroup24], (instregex "SAR8m1")>;
def: InstRW<[HWWriteResGroup24], (instregex "SAR8mi")>;
def: InstRW<[HWWriteResGroup24], (instregex "SHL64m1")>;
def: InstRW<[HWWriteResGroup24], (instregex "SHL64mi")>;
def: InstRW<[HWWriteResGroup24], (instregex "SHL8m1")>;
def: InstRW<[HWWriteResGroup24], (instregex "SHL8mi")>;
def: InstRW<[HWWriteResGroup24], (instregex "SHR64m1")>;
def: InstRW<[HWWriteResGroup24], (instregex "SHR64mi")>;
def: InstRW<[HWWriteResGroup24], (instregex "SHR8m1")>;
def: InstRW<[HWWriteResGroup24], (instregex "SHR8mi")>;

def HWWriteResGroup25 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 1;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup25], (instregex "ADD64mi8")>;
def: InstRW<[HWWriteResGroup25], (instregex "ADD64mr")>;
def: InstRW<[HWWriteResGroup25], (instregex "ADD8mi")>;
def: InstRW<[HWWriteResGroup25], (instregex "ADD8mr")>;
def: InstRW<[HWWriteResGroup25], (instregex "AND64mi8")>;
def: InstRW<[HWWriteResGroup25], (instregex "AND64mr")>;
def: InstRW<[HWWriteResGroup25], (instregex "AND8mi")>;
def: InstRW<[HWWriteResGroup25], (instregex "AND8mr")>;
def: InstRW<[HWWriteResGroup25], (instregex "DEC64m")>;
def: InstRW<[HWWriteResGroup25], (instregex "DEC8m")>;
def: InstRW<[HWWriteResGroup25], (instregex "INC64m")>;
def: InstRW<[HWWriteResGroup25], (instregex "INC8m")>;
def: InstRW<[HWWriteResGroup25], (instregex "NEG64m")>;
def: InstRW<[HWWriteResGroup25], (instregex "NEG8m")>;
def: InstRW<[HWWriteResGroup25], (instregex "NOT64m")>;
def: InstRW<[HWWriteResGroup25], (instregex "NOT8m")>;
def: InstRW<[HWWriteResGroup25], (instregex "OR64mi8")>;
def: InstRW<[HWWriteResGroup25], (instregex "OR64mr")>;
def: InstRW<[HWWriteResGroup25], (instregex "OR8mi")>;
def: InstRW<[HWWriteResGroup25], (instregex "OR8mr")>;
def: InstRW<[HWWriteResGroup25], (instregex "POP64rmm")>;
def: InstRW<[HWWriteResGroup25], (instregex "PUSH64rmm")>;
def: InstRW<[HWWriteResGroup25], (instregex "SUB64mi8")>;
def: InstRW<[HWWriteResGroup25], (instregex "SUB64mr")>;
def: InstRW<[HWWriteResGroup25], (instregex "SUB8mi")>;
def: InstRW<[HWWriteResGroup25], (instregex "SUB8mr")>;
def: InstRW<[HWWriteResGroup25], (instregex "XOR64mi8")>;
def: InstRW<[HWWriteResGroup25], (instregex "XOR64mr")>;
def: InstRW<[HWWriteResGroup25], (instregex "XOR8mi")>;
def: InstRW<[HWWriteResGroup25], (instregex "XOR8mr")>;

def HWWriteResGroup26 : SchedWriteRes<[HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup26], (instregex "BLENDVPDrr0")>;
def: InstRW<[HWWriteResGroup26], (instregex "BLENDVPSrr0")>;
def: InstRW<[HWWriteResGroup26], (instregex "MMX_PINSRWirri")>;
def: InstRW<[HWWriteResGroup26], (instregex "PBLENDVBrr0")>;
def: InstRW<[HWWriteResGroup26], (instregex "PINSRBrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "PINSRDrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "PINSRQrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "PINSRWrri")>;
def: InstRW<[HWWriteResGroup26], (instregex "VBLENDVPDYrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "VBLENDVPDrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "VBLENDVPSYrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "VBLENDVPSrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "VPBLENDVBYrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "VPBLENDVBrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "VPINSRBrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "VPINSRDrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "VPINSRQrr")>;
def: InstRW<[HWWriteResGroup26], (instregex "VPINSRWrri")>;

def HWWriteResGroup27 : SchedWriteRes<[HWPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup27], (instregex "FDECSTP")>;

def HWWriteResGroup28 : SchedWriteRes<[HWPort0]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup28], (instregex "ROL32ri")>;
def: InstRW<[HWWriteResGroup28], (instregex "ROL64r1")>;
def: InstRW<[HWWriteResGroup28], (instregex "ROL8r1")>;
def: InstRW<[HWWriteResGroup28], (instregex "ROL8ri")>;
def: InstRW<[HWWriteResGroup28], (instregex "ROR32ri")>;
def: InstRW<[HWWriteResGroup28], (instregex "ROR64r1")>;
def: InstRW<[HWWriteResGroup28], (instregex "ROR8r1")>;
def: InstRW<[HWWriteResGroup28], (instregex "ROR8ri")>;

def HWWriteResGroup29 : SchedWriteRes<[HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup29], (instregex "LFENCE")>;
def: InstRW<[HWWriteResGroup29], (instregex "MFENCE")>;
def: InstRW<[HWWriteResGroup29], (instregex "WAIT")>;
def: InstRW<[HWWriteResGroup29], (instregex "XGETBV")>;

def HWWriteResGroup30 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup30], (instregex "CVTPS2PDrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "CVTSS2SDrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "EXTRACTPSrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "MMX_PEXTRWirri")>;
def: InstRW<[HWWriteResGroup30], (instregex "PEXTRBrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PEXTRDrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PEXTRQrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PEXTRWri")>;
def: InstRW<[HWWriteResGroup30], (instregex "PSLLDrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PSLLQrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PSLLWrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PSRADrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PSRAWrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PSRLDrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PSRLQrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PSRLWrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "PTESTrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VCVTPH2PSYrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VCVTPH2PSrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VCVTPS2PDrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VCVTSS2SDrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VEXTRACTPSrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VPEXTRBrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VPEXTRDrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VPEXTRQrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VPEXTRWri")>;
def: InstRW<[HWWriteResGroup30], (instregex "VPSRADrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VPSRAWrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VPSRLDrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VPSRLQrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VPSRLWrr")>;
def: InstRW<[HWWriteResGroup30], (instregex "VPTESTrr")>;

def HWWriteResGroup31 : SchedWriteRes<[HWPort6,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup31], (instregex "CLFLUSH")>;

def HWWriteResGroup32 : SchedWriteRes<[HWPort01,HWPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup32], (instregex "MMX_MOVDQ2Qrr")>;

def HWWriteResGroup33 : SchedWriteRes<[HWPort0,HWPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup33], (instregex "BEXTR32rr")>;
def: InstRW<[HWWriteResGroup33], (instregex "BEXTR64rr")>;
def: InstRW<[HWWriteResGroup33], (instregex "BSWAP32r")>;

def HWWriteResGroup34 : SchedWriteRes<[HWPort0,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup34], (instregex "ADC64ri8")>;
def: InstRW<[HWWriteResGroup34], (instregex "ADC64rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "ADC8ri")>;
def: InstRW<[HWWriteResGroup34], (instregex "ADC8rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVAE32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVB32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVE32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVG32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVGE32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVL32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVLE32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVNE32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVNO32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVNP32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVNS32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVO32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVP32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CMOVS32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "CWD")>;
def: InstRW<[HWWriteResGroup34], (instregex "SBB32rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "SBB64ri8")>;
def: InstRW<[HWWriteResGroup34], (instregex "SBB8ri")>;
def: InstRW<[HWWriteResGroup34], (instregex "SBB8rr")>;
def: InstRW<[HWWriteResGroup34], (instregex "SETAr")>;
def: InstRW<[HWWriteResGroup34], (instregex "SETBEr")>;

def HWWriteResGroup35 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup35], (instregex "BLENDVPDrm0")>;
def: InstRW<[HWWriteResGroup35], (instregex "BLENDVPSrm0")>;
def: InstRW<[HWWriteResGroup35], (instregex "MMX_PACKSSDWirm")>;
def: InstRW<[HWWriteResGroup35], (instregex "MMX_PACKSSWBirm")>;
def: InstRW<[HWWriteResGroup35], (instregex "MMX_PACKUSWBirm")>;
def: InstRW<[HWWriteResGroup35], (instregex "PBLENDVBrm0")>;
def: InstRW<[HWWriteResGroup35], (instregex "VBLENDVPDYrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VBLENDVPDrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VBLENDVPSYrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VBLENDVPSrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VMASKMOVPDrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VMASKMOVPDrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VMASKMOVPSrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VMASKMOVPSrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VPBLENDVBYrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VPBLENDVBrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VPMASKMOVDYrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VPMASKMOVDrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VPMASKMOVQYrm")>;
def: InstRW<[HWWriteResGroup35], (instregex "VPMASKMOVQrm")>;

def HWWriteResGroup36 : SchedWriteRes<[HWPort23,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup36], (instregex "LEAVE64")>;
def: InstRW<[HWWriteResGroup36], (instregex "SCASB")>;
def: InstRW<[HWWriteResGroup36], (instregex "SCASL")>;
def: InstRW<[HWWriteResGroup36], (instregex "SCASQ")>;
def: InstRW<[HWWriteResGroup36], (instregex "SCASW")>;

def HWWriteResGroup37 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup37], (instregex "PSLLDrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "PSLLQrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "PSLLWrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "PSRADrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "PSRAWrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "PSRLDrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "PSRLQrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "PSRLWrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "PTESTrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "VPSLLDri")>;
def: InstRW<[HWWriteResGroup37], (instregex "VPSLLQri")>;
def: InstRW<[HWWriteResGroup37], (instregex "VPSLLWri")>;
def: InstRW<[HWWriteResGroup37], (instregex "VPSRADrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "VPSRAWrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "VPSRLDrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "VPSRLQrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "VPSRLWrm")>;
def: InstRW<[HWWriteResGroup37], (instregex "VPTESTrm")>;

def HWWriteResGroup38 : SchedWriteRes<[HWPort0,HWPort01,HWPort23]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup38], (instregex "FLDCW16m")>;

def HWWriteResGroup39 : SchedWriteRes<[HWPort0,HWPort23,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup39], (instregex "LDMXCSR")>;
def: InstRW<[HWWriteResGroup39], (instregex "VLDMXCSR")>;

def HWWriteResGroup40 : SchedWriteRes<[HWPort6,HWPort23,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup40], (instregex "LRETQ")>;
def: InstRW<[HWWriteResGroup40], (instregex "RETQ")>;

def HWWriteResGroup41 : SchedWriteRes<[HWPort23,HWPort0,HWPort15]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup41], (instregex "BEXTR32rm")>;
def: InstRW<[HWWriteResGroup41], (instregex "BEXTR64rm")>;

def HWWriteResGroup42 : SchedWriteRes<[HWPort23,HWPort0,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup42], (instregex "ADC64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "ADC8rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVAE64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVB64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVE64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVG64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVGE64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVL64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVLE64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVNE64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVNO64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVNP64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVNS64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVO64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVP64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "CMOVS64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "SBB64rm")>;
def: InstRW<[HWWriteResGroup42], (instregex "SBB8rm")>;

def HWWriteResGroup43 : SchedWriteRes<[HWPort4,HWPort6,HWPort237,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup43], (instregex "CALL64r")>;
def: InstRW<[HWWriteResGroup43], (instregex "SETAm")>;
def: InstRW<[HWWriteResGroup43], (instregex "SETBEm")>;

def HWWriteResGroup44 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0]> {
  let Latency = 2;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[HWWriteResGroup44], (instregex "ROL64m1")>;
def: InstRW<[HWWriteResGroup44], (instregex "ROL64mi")>;
def: InstRW<[HWWriteResGroup44], (instregex "ROL8m1")>;
def: InstRW<[HWWriteResGroup44], (instregex "ROL8mi")>;
def: InstRW<[HWWriteResGroup44], (instregex "ROR64m1")>;
def: InstRW<[HWWriteResGroup44], (instregex "ROR64mi")>;
def: InstRW<[HWWriteResGroup44], (instregex "ROR8m1")>;
def: InstRW<[HWWriteResGroup44], (instregex "ROR8mi")>;

def HWWriteResGroup45 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[HWWriteResGroup45], (instregex "XADD64rm")>;
def: InstRW<[HWWriteResGroup45], (instregex "XADD8rm")>;

def HWWriteResGroup46 : SchedWriteRes<[HWPort4,HWPort6,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[HWWriteResGroup46], (instregex "CALL64m")>;
def: InstRW<[HWWriteResGroup46], (instregex "FARCALL64")>;

def HWWriteResGroup47 : SchedWriteRes<[HWPort0]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup47], (instregex "MOVMSKPDrr")>;
def: InstRW<[HWWriteResGroup47], (instregex "MOVMSKPSrr")>;
def: InstRW<[HWWriteResGroup47], (instregex "PMOVMSKBrr")>;
def: InstRW<[HWWriteResGroup47], (instregex "VMOVMSKPDYrr")>;
def: InstRW<[HWWriteResGroup47], (instregex "VMOVMSKPDrr")>;
def: InstRW<[HWWriteResGroup47], (instregex "VMOVMSKPSrr")>;
def: InstRW<[HWWriteResGroup47], (instregex "VPMOVMSKBYrr")>;
def: InstRW<[HWWriteResGroup47], (instregex "VPMOVMSKBrr")>;

def HWWriteResGroup48 : SchedWriteRes<[HWPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup48], (instregex "ADDPDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "ADDPSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "ADDSDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "ADDSSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "ADDSUBPDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "ADDSUBPSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "BSF32rr")>;
def: InstRW<[HWWriteResGroup48], (instregex "BSR32rr")>;
def: InstRW<[HWWriteResGroup48], (instregex "CMPPDrri")>;
def: InstRW<[HWWriteResGroup48], (instregex "CMPPSrri")>;
def: InstRW<[HWWriteResGroup48], (instregex "CMPSDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "CMPSSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "COMISDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "COMISSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "CVTDQ2PSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "CVTPS2DQrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "CVTTPS2DQrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "IMUL32rri8")>;
def: InstRW<[HWWriteResGroup48], (instregex "IMUL64rr")>;
def: InstRW<[HWWriteResGroup48], (instregex "IMUL8r")>;
def: InstRW<[HWWriteResGroup48], (instregex "LZCNT32rr")>;
def: InstRW<[HWWriteResGroup48], (instregex "MAXPDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "MAXPSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "MAXSDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "MAXSSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "MINPDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "MINPSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "MINSDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "MINSSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "MMX_CVTPI2PSirr")>;
def: InstRW<[HWWriteResGroup48], (instregex "MUL8r")>;
def: InstRW<[HWWriteResGroup48], (instregex "PDEP32rr")>;
def: InstRW<[HWWriteResGroup48], (instregex "PDEP64rr")>;
def: InstRW<[HWWriteResGroup48], (instregex "PEXT32rr")>;
def: InstRW<[HWWriteResGroup48], (instregex "PEXT64rr")>;
def: InstRW<[HWWriteResGroup48], (instregex "POPCNT32rr")>;
def: InstRW<[HWWriteResGroup48], (instregex "SHLD32rri8")>;
def: InstRW<[HWWriteResGroup48], (instregex "SHRD32rri8")>;
def: InstRW<[HWWriteResGroup48], (instregex "SUBPDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "SUBPSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "SUBSDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "SUBSSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "TZCNT32rr")>;
def: InstRW<[HWWriteResGroup48], (instregex "UCOMISDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "UCOMISSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VADDPDYrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VADDPDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VADDPSYrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VADDPSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VADDSDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VADDSSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VADDSUBPDYrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VADDSUBPDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VADDSUBPSYrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VADDSUBPSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCMPPDYrri")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCMPPDrri")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCMPPSYrri")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCMPPSrri")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCMPSDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCMPSSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCOMISDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCOMISSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCVTDQ2PSYrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCVTDQ2PSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCVTPS2DQYrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCVTPS2DQrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VCVTTPS2DQrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VMAXPDYrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VMAXPDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VMAXPSYrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VMAXPSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VMAXSDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VMAXSSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VMINPDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VMINPSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VMINSDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VMINSSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VSUBPDYrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VSUBPDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VSUBPSYrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VSUBPSrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VUCOMISDrr")>;
def: InstRW<[HWWriteResGroup48], (instregex "VUCOMISSrr")>;

def HWWriteResGroup49 : SchedWriteRes<[HWPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup49], (instregex "KSHIFTRDri")>;
def: InstRW<[HWWriteResGroup49], (instregex "KSHIFTRWri")>;
def: InstRW<[HWWriteResGroup49], (instregex "VBROADCASTSDYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VBROADCASTSSrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VEXTRACTF128rr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VEXTRACTI128rr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VINSERTF128rr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VINSERTI128rr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPBROADCASTBYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPBROADCASTBrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPBROADCASTDYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPBROADCASTQYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPBROADCASTWYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPBROADCASTWrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPERM2I128rr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPERMDYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPERMQYri")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVSXBDYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVSXBQYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVSXBWYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVSXDQYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVSXWDYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVSXWQYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVZXBDYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVZXBQYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVZXBWYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVZXDQYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVZXWDYrr")>;
def: InstRW<[HWWriteResGroup49], (instregex "VPMOVZXWQYrr")>;

def HWWriteResGroup50 : SchedWriteRes<[HWPort1,HWPort23]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup50], (instregex "ADDPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADDPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADDSDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADDSSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADDSUBPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "ADDSUBPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "BSF64rm")>;
def: InstRW<[HWWriteResGroup50], (instregex "BSR64rm")>;
def: InstRW<[HWWriteResGroup50], (instregex "CMPPDrmi")>;
def: InstRW<[HWWriteResGroup50], (instregex "CMPPSrmi")>;
def: InstRW<[HWWriteResGroup50], (instregex "CMPSSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "COMISDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "COMISSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "CVTDQ2PSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "CVTPS2DQrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "CVTTPS2DQrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "IMUL64m")>;
def: InstRW<[HWWriteResGroup50], (instregex "IMUL64rm")>;
def: InstRW<[HWWriteResGroup50], (instregex "IMUL8m")>;
def: InstRW<[HWWriteResGroup50], (instregex "LZCNT64rm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MAXPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MAXPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MAXSDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MAXSSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MINPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MINPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MINSDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MINSSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MMX_CVTPI2PSirm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MMX_CVTPS2PIirm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MMX_CVTTPS2PIirm")>;
def: InstRW<[HWWriteResGroup50], (instregex "MUL64m")>;
def: InstRW<[HWWriteResGroup50], (instregex "MUL8m")>;
def: InstRW<[HWWriteResGroup50], (instregex "PDEP32rm")>;
def: InstRW<[HWWriteResGroup50], (instregex "PDEP64rm")>;
def: InstRW<[HWWriteResGroup50], (instregex "PEXT32rm")>;
def: InstRW<[HWWriteResGroup50], (instregex "PEXT64rm")>;
def: InstRW<[HWWriteResGroup50], (instregex "POPCNT64rm")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBSDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "SUBSSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "TZCNT64rm")>;
def: InstRW<[HWWriteResGroup50], (instregex "UCOMISDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "UCOMISSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDPDYrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDPSYrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSUBPDYrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSUBPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSUBPSYrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VADDSUBPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPPDYrmi")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPPDrmi")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPPSYrmi")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPPSrmi")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPSDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCMPSSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCOMISDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCOMISSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTDQ2PSYrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTDQ2PSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTPS2DQYrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTPS2DQrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTTPS2DQrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VCVTTPS2DQrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAXPDYrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAXPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAXPSYrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAXPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAXSDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMAXSSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMINPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMINPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMINPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMINPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMINSDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VMINSSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBPDYrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBPDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBPSYrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBPSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBSDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VSUBSSrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VUCOMISDrm")>;
def: InstRW<[HWWriteResGroup50], (instregex "VUCOMISSrm")>;

def HWWriteResGroup51 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup51], (instregex "VPERM2F128rm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERM2I128rm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERMDYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERMPDYmi")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERMPSYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPERMQYmi")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXBDYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXBQYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXBWYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXDQYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXWDYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVSXWQYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXBDYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXBQYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXBWYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXDQYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXWDYrm")>;
def: InstRW<[HWWriteResGroup51], (instregex "VPMOVZXWQYrm")>;

def HWWriteResGroup52 : SchedWriteRes<[HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[HWWriteResGroup52], (instregex "XADD32rr")>;
def: InstRW<[HWWriteResGroup52], (instregex "XADD8rr")>;
def: InstRW<[HWWriteResGroup52], (instregex "XCHG8rr")>;

def HWWriteResGroup53 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup53], (instregex "VPSLLVDYrr")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPSLLVDrr")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPSRAVDYrr")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPSRAVDrr")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPSRLVDYrr")>;
def: InstRW<[HWWriteResGroup53], (instregex "VPSRLVDrr")>;

def HWWriteResGroup54 : SchedWriteRes<[HWPort5,HWPort15]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup54], (instregex "MMX_PHADDSWrr64")>;
def: InstRW<[HWWriteResGroup54], (instregex "MMX_PHADDWrr64")>;
def: InstRW<[HWWriteResGroup54], (instregex "MMX_PHADDrr64")>;
def: InstRW<[HWWriteResGroup54], (instregex "MMX_PHSUBDrr64")>;
def: InstRW<[HWWriteResGroup54], (instregex "MMX_PHSUBSWrr64")>;
def: InstRW<[HWWriteResGroup54], (instregex "MMX_PHSUBWrr64")>;
def: InstRW<[HWWriteResGroup54], (instregex "PHADDDrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "PHADDSWrr128")>;
def: InstRW<[HWWriteResGroup54], (instregex "PHADDWrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "PHSUBDrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "PHSUBSWrr128")>;
def: InstRW<[HWWriteResGroup54], (instregex "PHSUBWrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHADDDYrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHADDDrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHADDSWrr128")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHADDSWrr256")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHADDWYrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHADDWrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHSUBDYrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHSUBDrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHSUBSWrr128")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHSUBSWrr256")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHSUBWYrr")>;
def: InstRW<[HWWriteResGroup54], (instregex "VPHSUBWrr")>;

def HWWriteResGroup55 : SchedWriteRes<[HWPort5,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup55], (instregex "MMX_PACKSSDWirr")>;
def: InstRW<[HWWriteResGroup55], (instregex "MMX_PACKSSWBirr")>;
def: InstRW<[HWWriteResGroup55], (instregex "MMX_PACKUSWBirr")>;

def HWWriteResGroup56 : SchedWriteRes<[HWPort6,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup56], (instregex "CLD")>;

def HWWriteResGroup57 : SchedWriteRes<[HWPort0,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup57], (instregex "CMOVA32rr")>;
def: InstRW<[HWWriteResGroup57], (instregex "CMOVBE32rr")>;
def: InstRW<[HWWriteResGroup57], (instregex "RCL32ri")>;
def: InstRW<[HWWriteResGroup57], (instregex "RCL64r1")>;
def: InstRW<[HWWriteResGroup57], (instregex "RCL8r1")>;
def: InstRW<[HWWriteResGroup57], (instregex "RCL8ri")>;
def: InstRW<[HWWriteResGroup57], (instregex "RCR32ri")>;
def: InstRW<[HWWriteResGroup57], (instregex "RCR64r1")>;
def: InstRW<[HWWriteResGroup57], (instregex "RCR8r1")>;
def: InstRW<[HWWriteResGroup57], (instregex "RCR8ri")>;
def: InstRW<[HWWriteResGroup57], (instregex "SHL64rCL")>;
def: InstRW<[HWWriteResGroup57], (instregex "SHL8rCL")>;

def HWWriteResGroup58 : SchedWriteRes<[HWPort0,HWPort4,HWPort237]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup58], (instregex "FNSTSWm")>;

def HWWriteResGroup59 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup59], (instregex "VPSLLVDYrm")>;
def: InstRW<[HWWriteResGroup59], (instregex "VPSLLVDrm")>;
def: InstRW<[HWWriteResGroup59], (instregex "VPSRAVDYrm")>;
def: InstRW<[HWWriteResGroup59], (instregex "VPSRAVDrm")>;
def: InstRW<[HWWriteResGroup59], (instregex "VPSRLVDYrm")>;
def: InstRW<[HWWriteResGroup59], (instregex "VPSRLVDrm")>;

def HWWriteResGroup60 : SchedWriteRes<[HWPort5,HWPort23,HWPort15]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup60], (instregex "MMX_PHADDSWrm64")>;
def: InstRW<[HWWriteResGroup60], (instregex "MMX_PHADDWrm64")>;
def: InstRW<[HWWriteResGroup60], (instregex "MMX_PHADDrm64")>;
def: InstRW<[HWWriteResGroup60], (instregex "MMX_PHSUBDrm64")>;
def: InstRW<[HWWriteResGroup60], (instregex "MMX_PHSUBSWrm64")>;
def: InstRW<[HWWriteResGroup60], (instregex "MMX_PHSUBWrm64")>;
def: InstRW<[HWWriteResGroup60], (instregex "PHADDDrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "PHADDSWrm128")>;
def: InstRW<[HWWriteResGroup60], (instregex "PHADDWrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "PHSUBDrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "PHSUBSWrm128")>;
def: InstRW<[HWWriteResGroup60], (instregex "PHSUBWrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHADDDYrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHADDDrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHADDSWrm128")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHADDSWrm256")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHADDWYrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHADDWrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHSUBDYrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHSUBDrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHSUBSWrm128")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHSUBSWrm256")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHSUBWYrm")>;
def: InstRW<[HWWriteResGroup60], (instregex "VPHSUBWrm")>;

def HWWriteResGroup61 : SchedWriteRes<[HWPort23,HWPort0,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[HWWriteResGroup61], (instregex "CMOVA64rm")>;
def: InstRW<[HWWriteResGroup61], (instregex "CMOVBE64rm")>;

def HWWriteResGroup62 : SchedWriteRes<[HWPort23,HWPort237,HWPort0,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[HWWriteResGroup62], (instregex "RCL64m1")>;
def: InstRW<[HWWriteResGroup62], (instregex "RCL64mi")>;
def: InstRW<[HWWriteResGroup62], (instregex "RCL8m1")>;
def: InstRW<[HWWriteResGroup62], (instregex "RCL8mi")>;
def: InstRW<[HWWriteResGroup62], (instregex "RCR64m1")>;
def: InstRW<[HWWriteResGroup62], (instregex "RCR64mi")>;
def: InstRW<[HWWriteResGroup62], (instregex "RCR8m1")>;
def: InstRW<[HWWriteResGroup62], (instregex "RCR8mi")>;

def HWWriteResGroup63 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[HWWriteResGroup63], (instregex "ADC64mi8")>;
def: InstRW<[HWWriteResGroup63], (instregex "ADC8mi")>;
def: InstRW<[HWWriteResGroup63], (instregex "ADD8mi")>;
def: InstRW<[HWWriteResGroup63], (instregex "AND8mi")>;
def: InstRW<[HWWriteResGroup63], (instregex "OR8mi")>;
def: InstRW<[HWWriteResGroup63], (instregex "SUB8mi")>;
def: InstRW<[HWWriteResGroup63], (instregex "XCHG64rm")>;
def: InstRW<[HWWriteResGroup63], (instregex "XCHG8rm")>;
def: InstRW<[HWWriteResGroup63], (instregex "XOR8mi")>;

def HWWriteResGroup64 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0,HWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[HWWriteResGroup64], (instregex "ADC64mr")>;
def: InstRW<[HWWriteResGroup64], (instregex "ADC8mr")>;
def: InstRW<[HWWriteResGroup64], (instregex "CMPXCHG64rm")>;
def: InstRW<[HWWriteResGroup64], (instregex "CMPXCHG8rm")>;
def: InstRW<[HWWriteResGroup64], (instregex "SBB64mi8")>;
def: InstRW<[HWWriteResGroup64], (instregex "SBB64mr")>;
def: InstRW<[HWWriteResGroup64], (instregex "SBB8mi")>;
def: InstRW<[HWWriteResGroup64], (instregex "SBB8mr")>;
def: InstRW<[HWWriteResGroup64], (instregex "SHL64mCL")>;
def: InstRW<[HWWriteResGroup64], (instregex "SHL8mCL")>;

def HWWriteResGroup65 : SchedWriteRes<[HWPort0,HWPort1]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup65], (instregex "CVTSD2SI64rr")>;
def: InstRW<[HWWriteResGroup65], (instregex "CVTSD2SIrr")>;
def: InstRW<[HWWriteResGroup65], (instregex "CVTSS2SI64rr")>;
def: InstRW<[HWWriteResGroup65], (instregex "CVTSS2SIrr")>;
def: InstRW<[HWWriteResGroup65], (instregex "CVTTSD2SI64rr")>;
def: InstRW<[HWWriteResGroup65], (instregex "CVTTSD2SIrr")>;
def: InstRW<[HWWriteResGroup65], (instregex "CVTTSS2SI64rr")>;
def: InstRW<[HWWriteResGroup65], (instregex "CVTTSS2SIrr")>;
def: InstRW<[HWWriteResGroup65], (instregex "VCVTSD2SI64rr")>;
def: InstRW<[HWWriteResGroup65], (instregex "VCVTSS2SI64rr")>;
def: InstRW<[HWWriteResGroup65], (instregex "VCVTSS2SIrr")>;
def: InstRW<[HWWriteResGroup65], (instregex "VCVTTSD2SI64rr")>;
def: InstRW<[HWWriteResGroup65], (instregex "VCVTTSD2SIrr")>;
def: InstRW<[HWWriteResGroup65], (instregex "VCVTTSS2SI64rr")>;
def: InstRW<[HWWriteResGroup65], (instregex "VCVTTSS2SIrr")>;

def HWWriteResGroup66 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup66], (instregex "VCVTPS2PDYrr")>;
def: InstRW<[HWWriteResGroup66], (instregex "VPSLLDrr")>;
def: InstRW<[HWWriteResGroup66], (instregex "VPSLLQrr")>;
def: InstRW<[HWWriteResGroup66], (instregex "VPSLLWrr")>;
def: InstRW<[HWWriteResGroup66], (instregex "VPSRADYrr")>;
def: InstRW<[HWWriteResGroup66], (instregex "VPSRAWYrr")>;
def: InstRW<[HWWriteResGroup66], (instregex "VPSRLDYrr")>;
def: InstRW<[HWWriteResGroup66], (instregex "VPSRLQYrr")>;
def: InstRW<[HWWriteResGroup66], (instregex "VPSRLWYrr")>;
def: InstRW<[HWWriteResGroup66], (instregex "VPTESTYrr")>;

def HWWriteResGroup67 : SchedWriteRes<[HWPort1,HWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup67], (instregex "CVTDQ2PDrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "CVTPD2DQrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "CVTPD2PSrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "CVTSD2SSrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "CVTSI2SD64rr")>;
def: InstRW<[HWWriteResGroup67], (instregex "CVTSI2SDrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "CVTSI2SSrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "CVTTPD2DQrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "MMX_CVTPD2PIirr")>;
def: InstRW<[HWWriteResGroup67], (instregex "MMX_CVTPI2PDirr")>;
def: InstRW<[HWWriteResGroup67], (instregex "MMX_CVTPS2PIirr")>;
def: InstRW<[HWWriteResGroup67], (instregex "MMX_CVTTPD2PIirr")>;
def: InstRW<[HWWriteResGroup67], (instregex "MMX_CVTTPS2PIirr")>;
def: InstRW<[HWWriteResGroup67], (instregex "VCVTDQ2PDrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "VCVTPD2DQrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "VCVTPD2PSrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "VCVTPS2PHrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "VCVTSI2SD64rr")>;
def: InstRW<[HWWriteResGroup67], (instregex "VCVTSI2SDrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "VCVTSI2SSrr")>;
def: InstRW<[HWWriteResGroup67], (instregex "VCVTTPD2DQrr")>;

def HWWriteResGroup68 : SchedWriteRes<[HWPort1,HWPort6]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup68], (instregex "IMUL64r")>;
def: InstRW<[HWWriteResGroup68], (instregex "MUL64r")>;
def: InstRW<[HWWriteResGroup68], (instregex "MULX64rr")>;

def HWWriteResGroup69 : SchedWriteRes<[HWPort0,HWPort1,HWPort23]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup69], (instregex "CVTSD2SI64rm")>;
def: InstRW<[HWWriteResGroup69], (instregex "CVTSD2SIrm")>;
def: InstRW<[HWWriteResGroup69], (instregex "CVTSS2SI64rm")>;
def: InstRW<[HWWriteResGroup69], (instregex "CVTSS2SIrm")>;
def: InstRW<[HWWriteResGroup69], (instregex "CVTTSD2SI64rm")>;
def: InstRW<[HWWriteResGroup69], (instregex "CVTTSD2SIrm")>;
def: InstRW<[HWWriteResGroup69], (instregex "CVTTSS2SIrm")>;
def: InstRW<[HWWriteResGroup69], (instregex "VCVTSD2SI64rm")>;
def: InstRW<[HWWriteResGroup69], (instregex "VCVTSD2SI64rr")>;
def: InstRW<[HWWriteResGroup69], (instregex "VCVTSS2SI64rm")>;
def: InstRW<[HWWriteResGroup69], (instregex "VCVTSS2SIrm")>;
def: InstRW<[HWWriteResGroup69], (instregex "VCVTTSD2SI64rm")>;
def: InstRW<[HWWriteResGroup69], (instregex "VCVTTSD2SI64rr")>;
def: InstRW<[HWWriteResGroup69], (instregex "VCVTTSS2SI64rm")>;
def: InstRW<[HWWriteResGroup69], (instregex "VCVTTSS2SIrm")>;

def HWWriteResGroup70 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup70], (instregex "VCVTPS2PDYrm")>;
def: InstRW<[HWWriteResGroup70], (instregex "VPTESTYrm")>;

def HWWriteResGroup71 : SchedWriteRes<[HWPort1,HWPort5,HWPort23]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup71], (instregex "CVTDQ2PDrm")>;
def: InstRW<[HWWriteResGroup71], (instregex "CVTPD2DQrm")>;
def: InstRW<[HWWriteResGroup71], (instregex "CVTPD2PSrm")>;
def: InstRW<[HWWriteResGroup71], (instregex "CVTSD2SSrm")>;
def: InstRW<[HWWriteResGroup71], (instregex "CVTTPD2DQrm")>;
def: InstRW<[HWWriteResGroup71], (instregex "MMX_CVTPD2PIirm")>;
def: InstRW<[HWWriteResGroup71], (instregex "MMX_CVTPI2PDirm")>;
def: InstRW<[HWWriteResGroup71], (instregex "MMX_CVTTPD2PIirm")>;
def: InstRW<[HWWriteResGroup71], (instregex "VCVTDQ2PDrm")>;
def: InstRW<[HWWriteResGroup71], (instregex "VCVTSD2SSrm")>;

def HWWriteResGroup72 : SchedWriteRes<[HWPort1,HWPort6,HWPort23]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup72], (instregex "MULX64rm")>;

def HWWriteResGroup73 : SchedWriteRes<[HWPort5,HWPort23,HWPort015]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup73], (instregex "VPBROADCASTBYrm")>;
def: InstRW<[HWWriteResGroup73], (instregex "VPBROADCASTBrm")>;
def: InstRW<[HWWriteResGroup73], (instregex "VPBROADCASTWYrm")>;
def: InstRW<[HWWriteResGroup73], (instregex "VPBROADCASTWrm")>;

def HWWriteResGroup74 : SchedWriteRes<[HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[HWWriteResGroup74], (instregex "FNCLEX")>;

def HWWriteResGroup75 : SchedWriteRes<[HWPort015,HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[HWWriteResGroup75], (instregex "VZEROUPPER")>;

def HWWriteResGroup76 : SchedWriteRes<[HWPort1,HWPort6,HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[HWWriteResGroup76], (instregex "LAR32rr")>;

def HWWriteResGroup77 : SchedWriteRes<[HWPort0,HWPort4,HWPort237,HWPort15]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup77], (instregex "VMASKMOVPDYrm")>;
def: InstRW<[HWWriteResGroup77], (instregex "VMASKMOVPDmr")>;
def: InstRW<[HWWriteResGroup77], (instregex "VMASKMOVPSmr")>;
def: InstRW<[HWWriteResGroup77], (instregex "VPMASKMOVDYmr")>;
def: InstRW<[HWWriteResGroup77], (instregex "VPMASKMOVDmr")>;
def: InstRW<[HWWriteResGroup77], (instregex "VPMASKMOVQYmr")>;
def: InstRW<[HWWriteResGroup77], (instregex "VPMASKMOVQmr")>;

def HWWriteResGroup78 : SchedWriteRes<[HWPort1,HWPort4,HWPort5,HWPort237]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup78], (instregex "VCVTPS2PHmr")>;

def HWWriteResGroup79 : SchedWriteRes<[HWPort1,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup79], (instregex "SHLD64mri8")>;
def: InstRW<[HWWriteResGroup79], (instregex "SHRD64mri8")>;

def HWWriteResGroup80 : SchedWriteRes<[HWPort1,HWPort6,HWPort23,HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[HWWriteResGroup80], (instregex "LAR32rm")>;
def: InstRW<[HWWriteResGroup80], (instregex "LSL32rm")>;

def HWWriteResGroup81 : SchedWriteRes<[HWPort4,HWPort237,HWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[HWWriteResGroup81], (instregex "PUSHF16")>;
def: InstRW<[HWWriteResGroup81], (instregex "PUSHF64")>;

def HWWriteResGroup82 : SchedWriteRes<[HWPort0]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup82], (instregex "MMX_PMADDUBSWrr64")>;
def: InstRW<[HWWriteResGroup82], (instregex "MMX_PMADDWDirr")>;
def: InstRW<[HWWriteResGroup82], (instregex "MMX_PMULHRSWrr64")>;
def: InstRW<[HWWriteResGroup82], (instregex "MMX_PMULHUWirr")>;
def: InstRW<[HWWriteResGroup82], (instregex "MMX_PMULHWirr")>;
def: InstRW<[HWWriteResGroup82], (instregex "MMX_PMULLWirr")>;
def: InstRW<[HWWriteResGroup82], (instregex "MMX_PMULUDQirr")>;
def: InstRW<[HWWriteResGroup82], (instregex "MMX_PSADBWirr")>;
def: InstRW<[HWWriteResGroup82], (instregex "PCMPGTQrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "PHMINPOSUWrr128")>;
def: InstRW<[HWWriteResGroup82], (instregex "PMADDUBSWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "PMADDWDrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "PMULDQrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "PMULHRSWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "PMULHUWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "PMULHWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "PMULLWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "PMULUDQrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "PSADBWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "RCPPSr")>;
def: InstRW<[HWWriteResGroup82], (instregex "RCPSSr")>;
def: InstRW<[HWWriteResGroup82], (instregex "RSQRTPSr")>;
def: InstRW<[HWWriteResGroup82], (instregex "RSQRTSSr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VMOVMSKPSYrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPCMPGTQYrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPCMPGTQrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPHMINPOSUWrr128")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMADDUBSWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMADDUBSWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMADDWDYrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMADDWDrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMULDQYrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMULDQrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMULHRSWYrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMULHRSWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMULHUWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMULHWYrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMULHWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMULLWYrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMULLWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPMULUDQrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPSADBWYrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VPSADBWrr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VRSQRTPSr")>;
def: InstRW<[HWWriteResGroup82], (instregex "VRSQRTSSr")>;

def HWWriteResGroup83 : SchedWriteRes<[HWPort01]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup83], (instregex "MULPDrr")>;
def: InstRW<[HWWriteResGroup83], (instregex "MULPSrr")>;
def: InstRW<[HWWriteResGroup83], (instregex "MULSDrr")>;
def: InstRW<[HWWriteResGroup83], (instregex "MULSSrr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD132PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD132PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD132PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD132PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD132SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD132SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD213PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD213PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD213PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD213PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD213SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD213SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD231PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD231PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD231PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD231PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD231SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADD231SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB132PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB132PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB132PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB132PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB213PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB213PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB213PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB213PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB231PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB231PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB231PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMADDSUB231PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB132PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB132PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB132PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB132PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB132SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB132SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB213PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB213PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB213PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB213PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB213SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB213SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB231PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB231PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB231PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB231PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB231SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUB231SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD132PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD132PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD132PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD132PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD213PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD213PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD213PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD213PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD231PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD231PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD231PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFMSUBADD231PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD132PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD132PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD132PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD132PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD132SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD132SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD213PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD213PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD213PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD213PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD213SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD213SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD231PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD231PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD231PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD231PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD231SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMADD231SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB132PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB132PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB132PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB132PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB132SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB132SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB213PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB213PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB213PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB213PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB213SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB213SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB231PDYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB231PDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB231PSYr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB231PSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB231SDr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VFNMSUB231SSr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VMULPDYrr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VMULPDrr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VMULPSYrr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VMULPSrr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VMULSDrr")>;
def: InstRW<[HWWriteResGroup83], (instregex "VMULSSrr")>;

def HWWriteResGroup84 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup84], (instregex "MMX_PMADDUBSWrm64")>;
def: InstRW<[HWWriteResGroup84], (instregex "MMX_PMADDWDirm")>;
def: InstRW<[HWWriteResGroup84], (instregex "MMX_PMULHRSWrm64")>;
def: InstRW<[HWWriteResGroup84], (instregex "MMX_PMULHUWirm")>;
def: InstRW<[HWWriteResGroup84], (instregex "MMX_PMULHWirm")>;
def: InstRW<[HWWriteResGroup84], (instregex "MMX_PMULLWirm")>;
def: InstRW<[HWWriteResGroup84], (instregex "MMX_PMULUDQirm")>;
def: InstRW<[HWWriteResGroup84], (instregex "MMX_PSADBWirm")>;
def: InstRW<[HWWriteResGroup84], (instregex "PCMPGTQrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "PHMINPOSUWrm128")>;
def: InstRW<[HWWriteResGroup84], (instregex "PMADDUBSWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "PMADDWDrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "PMULDQrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "PMULHRSWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "PMULHUWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "PMULHWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "PMULLWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "PMULUDQrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "PSADBWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "RCPPSm")>;
def: InstRW<[HWWriteResGroup84], (instregex "RCPSSm")>;
def: InstRW<[HWWriteResGroup84], (instregex "RSQRTPSm")>;
def: InstRW<[HWWriteResGroup84], (instregex "RSQRTSSm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPCMPGTQYrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPCMPGTQrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPHMINPOSUWrm128")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMADDUBSWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMADDUBSWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMADDWDYrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMADDWDrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULDQYrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULDQrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULHRSWYrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULHRSWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULHUWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULHUWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULHWYrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULHWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULLWYrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULLWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULUDQrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPMULUDQrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPSADBWYrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VPSADBWrm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VRCPPSm(_Int)?")>;
def: InstRW<[HWWriteResGroup84], (instregex "VRCPSSm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VRSQRTPSm")>;
def: InstRW<[HWWriteResGroup84], (instregex "VRSQRTSSm")>;

def HWWriteResGroup85 : SchedWriteRes<[HWPort01,HWPort23]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup85], (instregex "MULPDrm")>;
def: InstRW<[HWWriteResGroup85], (instregex "MULPSrm")>;
def: InstRW<[HWWriteResGroup85], (instregex "MULSDrm")>;
def: InstRW<[HWWriteResGroup85], (instregex "MULSSrm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD132PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD132PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD132PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD132PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD132SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD132SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD213PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD213PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD213PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD213PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD213SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD213SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD231PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD231PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD231PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD231PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD231SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADD231SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB132PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB132PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB132PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB132PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB213PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB213PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB213PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB213PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB231PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB231PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB231PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMADDSUB231PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB132PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB132PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB132PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB132PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB132SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB132SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB213PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB213PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB213PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB213PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB213SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB213SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB231PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB231PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB231PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB231PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB231SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUB231SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD132PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD132PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD132PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD132PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD213PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD213PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD213PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD213PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD231PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD231PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD231PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFMSUBADD231PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD132PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD132PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD132PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD132PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD132SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD132SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD213PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD213PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD213PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD213PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD213SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD213SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD231PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD231PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD231PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD231PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD231SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMADD231SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB132PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB132PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB132PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB132PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB132SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB132SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB213PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB213PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB213PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB213PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB213SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB213SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB231PDYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB231PDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB231PSYm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB231PSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB231SDm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VFNMSUB231SSm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VMULPDYrm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VMULPDrm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VMULPSYrm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VMULPSrm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VMULSDrm")>;
def: InstRW<[HWWriteResGroup85], (instregex "VMULSSrm")>;

def HWWriteResGroup86 : SchedWriteRes<[HWPort1,HWPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup86], (instregex "CVTSI2SS64rr")>;
def: InstRW<[HWWriteResGroup86], (instregex "HADDPDrr")>;
def: InstRW<[HWWriteResGroup86], (instregex "HADDPSrr")>;
def: InstRW<[HWWriteResGroup86], (instregex "HSUBPDrr")>;
def: InstRW<[HWWriteResGroup86], (instregex "HSUBPSrr")>;
def: InstRW<[HWWriteResGroup86], (instregex "VCVTSI2SS64rr")>;
def: InstRW<[HWWriteResGroup86], (instregex "VHADDPDrr")>;
def: InstRW<[HWWriteResGroup86], (instregex "VHADDPSYrr")>;
def: InstRW<[HWWriteResGroup86], (instregex "VHADDPSrr")>;
def: InstRW<[HWWriteResGroup86], (instregex "VHSUBPDYrr")>;
def: InstRW<[HWWriteResGroup86], (instregex "VHSUBPDrr")>;
def: InstRW<[HWWriteResGroup86], (instregex "VHSUBPSYrr")>;
def: InstRW<[HWWriteResGroup86], (instregex "VHSUBPSrr")>;

def HWWriteResGroup87 : SchedWriteRes<[HWPort1,HWPort6,HWPort0]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup87], (instregex "STR32r")>;

def HWWriteResGroup88 : SchedWriteRes<[HWPort1,HWPort0,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup88], (instregex "MULX32rr")>;

def HWWriteResGroup89 : SchedWriteRes<[HWPort1,HWPort5,HWPort23]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[HWWriteResGroup89], (instregex "HADDPDrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "HADDPSrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "HSUBPDrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "HSUBPSrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "VHADDPDrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "VHADDPDrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "VHADDPSYrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "VHADDPSrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "VHSUBPDYrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "VHSUBPDrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "VHSUBPSYrm")>;
def: InstRW<[HWWriteResGroup89], (instregex "VHSUBPSrm")>;

def HWWriteResGroup90 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort23]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup90], (instregex "CVTTSS2SI64rm")>;

def HWWriteResGroup91 : SchedWriteRes<[HWPort1,HWPort23,HWPort0,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup91], (instregex "MULX32rm")>;

def HWWriteResGroup92 : SchedWriteRes<[HWPort6,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[HWWriteResGroup92], (instregex "PAUSE")>;

def HWWriteResGroup93 : SchedWriteRes<[HWPort0,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[HWWriteResGroup93], (instregex "XSETBV")>;

def HWWriteResGroup94 : SchedWriteRes<[HWPort0,HWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[HWWriteResGroup94], (instregex "CMPXCHG32rr")>;
def: InstRW<[HWWriteResGroup94], (instregex "CMPXCHG8rr")>;
def: InstRW<[HWWriteResGroup94], (instregex "ROUNDPDr")>;
def: InstRW<[HWWriteResGroup94], (instregex "ROUNDPSr")>;
def: InstRW<[HWWriteResGroup94], (instregex "ROUNDSDr")>;
def: InstRW<[HWWriteResGroup94], (instregex "ROUNDSSr")>;
def: InstRW<[HWWriteResGroup94], (instregex "VBROADCASTF128")>;
def: InstRW<[HWWriteResGroup94], (instregex "VPBROADCASTMB2QZrr")>;
def: InstRW<[HWWriteResGroup94], (instregex "VROUNDPDr")>;
def: InstRW<[HWWriteResGroup94], (instregex "VROUNDPSr")>;
def: InstRW<[HWWriteResGroup94], (instregex "VROUNDSDr")>;

def HWWriteResGroup95 : SchedWriteRes<[HWPort1,HWPort5]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup95], (instregex "VCVTDQ2PDYrr")>;
def: InstRW<[HWWriteResGroup95], (instregex "VCVTPD2DQYrr")>;
def: InstRW<[HWWriteResGroup95], (instregex "VCVTPD2PSYrr")>;
def: InstRW<[HWWriteResGroup95], (instregex "VCVTPS2PHYrr")>;
def: InstRW<[HWWriteResGroup95], (instregex "VCVTTPD2DQYrr")>;
def: InstRW<[HWWriteResGroup95], (instregex "ROUNDPDm")>;
def: InstRW<[HWWriteResGroup95], (instregex "ROUNDPSm")>;
def: InstRW<[HWWriteResGroup95], (instregex "ROUNDSDm")>;
def: InstRW<[HWWriteResGroup95], (instregex "ROUNDSSm")>;
def: InstRW<[HWWriteResGroup95], (instregex "VROUNDPDm")>;
def: InstRW<[HWWriteResGroup95], (instregex "VROUNDPDm")>;
def: InstRW<[HWWriteResGroup95], (instregex "VROUNDPSm")>;
def: InstRW<[HWWriteResGroup95], (instregex "VROUNDPSm")>;
def: InstRW<[HWWriteResGroup95], (instregex "VROUNDSDm")>;
def: InstRW<[HWWriteResGroup95], (instregex "VROUNDSSm")>;

def HWWriteResGroup96 : SchedWriteRes<[HWPort1,HWPort5,HWPort23]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup96], (instregex "VCVTDQ2PDYrm")>;

def HWWriteResGroup97 : SchedWriteRes<[HWPort1,HWPort4,HWPort5,HWPort237]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup97], (instregex "VCVTPS2PHYmr")>;

def HWWriteResGroup98 : SchedWriteRes<[HWPort1,HWPort6,HWPort0,HWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup98], (instregex "SLDT32r")>;

def HWWriteResGroup99 : SchedWriteRes<[HWPort6,HWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,5];
}
def: InstRW<[HWWriteResGroup99], (instregex "STD")>;

def HWWriteResGroup100 : SchedWriteRes<[HWPort5]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup100], (instregex "AESDECLASTrr")>;
def: InstRW<[HWWriteResGroup100], (instregex "AESDECrr")>;
def: InstRW<[HWWriteResGroup100], (instregex "AESENCLASTrr")>;
def: InstRW<[HWWriteResGroup100], (instregex "AESENCrr")>;
def: InstRW<[HWWriteResGroup100], (instregex "KANDQrr")>;
def: InstRW<[HWWriteResGroup100], (instregex "VAESDECLASTrr")>;
def: InstRW<[HWWriteResGroup100], (instregex "VAESDECrr")>;
def: InstRW<[HWWriteResGroup100], (instregex "VAESENCrr")>;

def HWWriteResGroup101 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup101], (instregex "AESDECLASTrm")>;
def: InstRW<[HWWriteResGroup101], (instregex "AESDECrm")>;
def: InstRW<[HWWriteResGroup101], (instregex "AESENCLASTrm")>;
def: InstRW<[HWWriteResGroup101], (instregex "AESENCrm")>;
def: InstRW<[HWWriteResGroup101], (instregex "VAESDECLASTrm")>;
def: InstRW<[HWWriteResGroup101], (instregex "VAESDECrm")>;
def: InstRW<[HWWriteResGroup101], (instregex "VAESENCLASTrm")>;
def: InstRW<[HWWriteResGroup101], (instregex "VAESENCrm")>;

def HWWriteResGroup102 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[HWWriteResGroup102], (instregex "MPSADBWrri")>;
def: InstRW<[HWWriteResGroup102], (instregex "VMPSADBWYrri")>;
def: InstRW<[HWWriteResGroup102], (instregex "VMPSADBWrri")>;

def HWWriteResGroup103 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[HWWriteResGroup103], (instregex "MPSADBWrmi")>;
def: InstRW<[HWWriteResGroup103], (instregex "VMPSADBWYrmi")>;
def: InstRW<[HWWriteResGroup103], (instregex "VMPSADBWrmi")>;

def HWWriteResGroup104 : SchedWriteRes<[HWPort0,HWPort1,HWPort5]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[HWWriteResGroup104], (instregex "DPPDrri")>;
def: InstRW<[HWWriteResGroup104], (instregex "VDPPDrri")>;

def HWWriteResGroup105 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[HWWriteResGroup105], (instregex "DPPDrmi")>;
def: InstRW<[HWWriteResGroup105], (instregex "VDPPDrmi")>;

def HWWriteResGroup106 : SchedWriteRes<[HWPort0]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup106], (instregex "PMULLDrr")>;
def: InstRW<[HWWriteResGroup106], (instregex "VPMULLDYrr")>;
def: InstRW<[HWWriteResGroup106], (instregex "VPMULLDrr")>;

def HWWriteResGroup107 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup107], (instregex "PMULLDrm")>;
def: InstRW<[HWWriteResGroup107], (instregex "VPMULLDYrm")>;
def: InstRW<[HWWriteResGroup107], (instregex "VPMULLDrm")>;

def HWWriteResGroup108 : SchedWriteRes<[HWPort0]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup108], (instregex "DIVPSrr")>;
def: InstRW<[HWWriteResGroup108], (instregex "DIVSSrr")>;

def HWWriteResGroup109 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup109], (instregex "DIVPSrm")>;
def: InstRW<[HWWriteResGroup109], (instregex "DIVSSrm")>;

def HWWriteResGroup110 : SchedWriteRes<[HWPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[HWWriteResGroup110], (instregex "PCMPISTRIrr")>;
def: InstRW<[HWWriteResGroup110], (instregex "PCMPISTRM128rr")>;
def: InstRW<[HWWriteResGroup110], (instregex "VPCMPISTRIrr")>;
def: InstRW<[HWWriteResGroup110], (instregex "VPCMPISTRM128rr")>;

def HWWriteResGroup111 : SchedWriteRes<[HWPort0,HWPort5]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup111], (instregex "PCLMULQDQrr")>;
def: InstRW<[HWWriteResGroup111], (instregex "VPCLMULQDQrr")>;

def HWWriteResGroup112 : SchedWriteRes<[HWPort0,HWPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup112], (instregex "VRCPPSYr(_Int)?")>;
def: InstRW<[HWWriteResGroup112], (instregex "VRSQRTPSYr")>;

def HWWriteResGroup113 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}
def: InstRW<[HWWriteResGroup113], (instregex "PCMPISTRIrm")>;
def: InstRW<[HWWriteResGroup113], (instregex "PCMPISTRM128rm")>;
def: InstRW<[HWWriteResGroup113], (instregex "VPCMPISTRIrm")>;
def: InstRW<[HWWriteResGroup113], (instregex "VPCMPISTRM128rm")>;

def HWWriteResGroup114 : SchedWriteRes<[HWPort0,HWPort5,HWPort23]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup114], (instregex "PCLMULQDQrm")>;
def: InstRW<[HWWriteResGroup114], (instregex "VPCLMULQDQrm")>;
def: InstRW<[HWWriteResGroup114], (instregex "VRCPPSYm(_Int)?")>;

def HWWriteResGroup115 : SchedWriteRes<[HWPort0,HWPort23,HWPort015]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup115], (instregex "VRCPPSm")>;
def: InstRW<[HWWriteResGroup115], (instregex "VRSQRTPSYm")>;

def HWWriteResGroup116 : SchedWriteRes<[HWPort4,HWPort23,HWPort237,HWPort0,HWPort15,HWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 14;
  let ResourceCycles = [1,1,1,4,2,5];
}
def: InstRW<[HWWriteResGroup116], (instregex "CMPXCHG8B")>;

def HWWriteResGroup117 : SchedWriteRes<[HWPort0]> {
  let Latency = 13;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup117], (instregex "SQRTPSr")>;
def: InstRW<[HWWriteResGroup117], (instregex "SQRTSSr")>;
def: InstRW<[HWWriteResGroup117], (instregex "VDIVPSrr")>;
def: InstRW<[HWWriteResGroup117], (instregex "VDIVSSrr")>;

def HWWriteResGroup118 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 13;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup118], (instregex "SQRTPSm")>;
def: InstRW<[HWWriteResGroup118], (instregex "SQRTSSm")>;
def: InstRW<[HWWriteResGroup118], (instregex "VDIVPSrm")>;
def: InstRW<[HWWriteResGroup118], (instregex "VDIVSSrm")>;

def HWWriteResGroup119 : SchedWriteRes<[HWPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup119], (instregex "DIVPDrr")>;
def: InstRW<[HWWriteResGroup119], (instregex "DIVSDrr")>;
def: InstRW<[HWWriteResGroup119], (instregex "VSQRTPSr")>;
def: InstRW<[HWWriteResGroup119], (instregex "VSQRTSSr")>;

def HWWriteResGroup120 : SchedWriteRes<[HWPort5]> {
  let Latency = 14;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[HWWriteResGroup120], (instregex "AESIMCrr")>;
def: InstRW<[HWWriteResGroup120], (instregex "VAESIMCrr")>;

def HWWriteResGroup121 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 14;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup121], (instregex "DIVPDrm")>;
def: InstRW<[HWWriteResGroup121], (instregex "DIVSDrm")>;
def: InstRW<[HWWriteResGroup121], (instregex "VSQRTPSm")>;
def: InstRW<[HWWriteResGroup121], (instregex "VSQRTSSm")>;

def HWWriteResGroup122 : SchedWriteRes<[HWPort5,HWPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup122], (instregex "AESIMCrm")>;
def: InstRW<[HWWriteResGroup122], (instregex "VAESIMCrm")>;

def HWWriteResGroup123 : SchedWriteRes<[HWPort0,HWPort1,HWPort5]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup123], (instregex "DPPSrri")>;
def: InstRW<[HWWriteResGroup123], (instregex "VDPPSYrri")>;
def: InstRW<[HWWriteResGroup123], (instregex "VDPPSrri")>;

def HWWriteResGroup124 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort23]> {
  let Latency = 14;
  let NumMicroOps = 5;
  let ResourceCycles = [2,1,1,1];
}
def: InstRW<[HWWriteResGroup124], (instregex "DPPSrmi")>;
def: InstRW<[HWWriteResGroup124], (instregex "VDPPSYrmi")>;
def: InstRW<[HWWriteResGroup124], (instregex "VDPPSrmi")>;

def HWWriteResGroup125 : SchedWriteRes<[HWPort23,HWPort0156]> {
  let Latency = 14;
  let NumMicroOps = 15;
  let ResourceCycles = [1,14];
}
def: InstRW<[HWWriteResGroup125], (instregex "POPF16")>;

def HWWriteResGroup126 : SchedWriteRes<[HWPort4,HWPort5,HWPort6,HWPort23,HWPort237,HWPort0,HWPort0156]> {
  let Latency = 15;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,1,2];
}
def: InstRW<[HWWriteResGroup126], (instregex "INSB")>;
def: InstRW<[HWWriteResGroup126], (instregex "INSL")>;
def: InstRW<[HWWriteResGroup126], (instregex "INSW")>;

def HWWriteResGroup127 : SchedWriteRes<[HWPort5]> {
  let Latency = 16;
  let NumMicroOps = 16;
  let ResourceCycles = [16];
}
def: InstRW<[HWWriteResGroup127], (instregex "VZEROALL")>;

def HWWriteResGroup128 : SchedWriteRes<[HWPort0,HWPort4,HWPort5,HWPort23,HWPort237,HWPort0,HWPort0156]> {
  let Latency = 16;
  let NumMicroOps = 19;
  let ResourceCycles = [2,1,4,1,1,4,6];
}
def: InstRW<[HWWriteResGroup128], (instregex "CMPXCHG16B")>;

def HWWriteResGroup129 : SchedWriteRes<[HWPort0,HWPort5,HWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def: InstRW<[HWWriteResGroup129], (instregex "PCMPESTRIrr")>;
def: InstRW<[HWWriteResGroup129], (instregex "VPCMPESTRIrr")>;

def HWWriteResGroup130 : SchedWriteRes<[HWPort5,HWPort6,HWPort0,HWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[HWWriteResGroup130], (instregex "CPUID")>;
def: InstRW<[HWWriteResGroup130], (instregex "RDTSC")>;

def HWWriteResGroup131 : SchedWriteRes<[HWPort0,HWPort5,HWPort23,HWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[HWWriteResGroup131], (instregex "PCMPESTRIrm")>;
def: InstRW<[HWWriteResGroup131], (instregex "VPCMPESTRIrm")>;

def HWWriteResGroup132 : SchedWriteRes<[HWPort6,HWPort23,HWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 19;
  let ResourceCycles = [3,1,15];
}
def: InstRW<[HWWriteResGroup132], (instregex "XRSTOR")>;

def HWWriteResGroup133 : SchedWriteRes<[HWPort0,HWPort5,HWPort015,HWPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[HWWriteResGroup133], (instregex "PCMPESTRM128rr")>;
def: InstRW<[HWWriteResGroup133], (instregex "VPCMPESTRM128rr")>;

def HWWriteResGroup134 : SchedWriteRes<[HWPort0,HWPort5,HWPort23,HWPort015,HWPort0156]> {
  let Latency = 19;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}
def: InstRW<[HWWriteResGroup134], (instregex "PCMPESTRM128rm")>;
def: InstRW<[HWWriteResGroup134], (instregex "VPCMPESTRM128rm")>;
def: InstRW<[HWWriteResGroup134], (instregex "SQRTPDr")>;
def: InstRW<[HWWriteResGroup134], (instregex "SQRTSDr")>;
def: InstRW<[HWWriteResGroup134], (instregex "VDIVPDrr")>;
def: InstRW<[HWWriteResGroup134], (instregex "VDIVSDrr")>;
def: InstRW<[HWWriteResGroup134], (instregex "SQRTPDm")>;
def: InstRW<[HWWriteResGroup134], (instregex "SQRTSDm")>;
def: InstRW<[HWWriteResGroup134], (instregex "VDIVPDrm")>;
def: InstRW<[HWWriteResGroup134], (instregex "VDIVSDrm")>;

def HWWriteResGroup135 : SchedWriteRes<[HWPort5,HWPort6,HWPort0156]> {
  let Latency = 20;
  let NumMicroOps = 10;
  let ResourceCycles = [1,2,7];
}
def: InstRW<[HWWriteResGroup135], (instregex "MWAITrr")>;

def HWWriteResGroup136 : SchedWriteRes<[HWPort0]> {
  let Latency = 21;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[HWWriteResGroup136], (instregex "VSQRTPDr")>;
def: InstRW<[HWWriteResGroup136], (instregex "VSQRTSDr")>;

def HWWriteResGroup137 : SchedWriteRes<[HWPort0,HWPort23]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[HWWriteResGroup137], (instregex "VSQRTPDm")>;
def: InstRW<[HWWriteResGroup137], (instregex "VSQRTSDm")>;

def HWWriteResGroup138 : SchedWriteRes<[HWPort0,HWPort015]> {
  let Latency = 21;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup138], (instregex "VDIVPSYrr")>;
def: InstRW<[HWWriteResGroup138], (instregex "VSQRTPSYr")>;

def HWWriteResGroup139 : SchedWriteRes<[HWPort0,HWPort23,HWPort015]> {
  let Latency = 21;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup139], (instregex "VDIVPSYrm")>;
def: InstRW<[HWWriteResGroup139], (instregex "VSQRTPSYm")>;

def HWWriteResGroup140 : SchedWriteRes<[HWPort4,HWPort6,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 24;
  let NumMicroOps = 27;
  let ResourceCycles = [1,5,1,1,19];
}
def: InstRW<[HWWriteResGroup140], (instregex "XSAVE64")>;

def HWWriteResGroup141 : SchedWriteRes<[HWPort4,HWPort6,HWPort23,HWPort237,HWPort0156]> {
  let Latency = 25;
  let NumMicroOps = 28;
  let ResourceCycles = [1,6,1,1,19];
}
def: InstRW<[HWWriteResGroup141], (instregex "XSAVE")>;

def HWWriteResGroup142 : SchedWriteRes<[HWPort0,HWPort5,HWPort23,HWPort015]> {
  let Latency = 28;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,1,1];
}
def: InstRW<[HWWriteResGroup142], (instregex "AESKEYGENASSIST128rm")>;
def: InstRW<[HWWriteResGroup142], (instregex "VAESKEYGENASSIST128rm")>;

def HWWriteResGroup143 : SchedWriteRes<[HWPort0,HWPort5,HWPort015]> {
  let Latency = 29;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,2];
}
def: InstRW<[HWWriteResGroup143], (instregex "AESKEYGENASSIST128rr")>;
def: InstRW<[HWWriteResGroup143], (instregex "VAESKEYGENASSIST128rr")>;

def HWWriteResGroup145 : SchedWriteRes<[HWPort01,HWPort15,HWPort015,HWPort0156]> {
  let Latency = 31;
  let NumMicroOps = 31;
  let ResourceCycles = [8,1,21,1];
}
def: InstRW<[HWWriteResGroup145], (instregex "MMX_EMMS")>;

def HWWriteResGroup146 : SchedWriteRes<[HWPort0,HWPort015]> {
  let Latency = 35;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[HWWriteResGroup146], (instregex "VDIVPDYrr")>;
def: InstRW<[HWWriteResGroup146], (instregex "VSQRTPDYr")>;

def HWWriteResGroup147 : SchedWriteRes<[HWPort0,HWPort23,HWPort015]> {
  let Latency = 35;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[HWWriteResGroup147], (instregex "VDIVPDYrm")>;
def: InstRW<[HWWriteResGroup147], (instregex "VSQRTPDYm")>;

def HWWriteResGroup148 : SchedWriteRes<[HWPort1,HWPort4,HWPort5,HWPort6,HWPort23,HWPort237,HWPort15,HWPort0156]> {
  let Latency = 35;
  let NumMicroOps = 18;
  let ResourceCycles = [1,1,2,3,1,1,1,8];
}
def: InstRW<[HWWriteResGroup148], (instregex "VMCLEARm")>;

def HWWriteResGroup149 : SchedWriteRes<[HWPort5,HWPort0156]> {
  let Latency = 42;
  let NumMicroOps = 22;
  let ResourceCycles = [2,20];
}
def: InstRW<[HWWriteResGroup149], (instregex "RDTSCP")>;

def HWWriteResGroup150 : SchedWriteRes<[HWPort0,HWPort01,HWPort23,HWPort0,HWPort0,HWPort015,HWPort0156]> {
  let Latency = 56;
  let NumMicroOps = 64;
  let ResourceCycles = [2,2,8,1,10,2,39];
}
def: InstRW<[HWWriteResGroup150], (instregex "FLDENVm")>;
def: InstRW<[HWWriteResGroup150], (instregex "FLDENVm")>;

def HWWriteResGroup151 : SchedWriteRes<[HWPort0,HWPort6,HWPort23,HWPort0,HWPort0,HWPort15,HWPort0156]> {
  let Latency = 59;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[HWWriteResGroup151], (instregex "FXRSTOR64")>;

def HWWriteResGroup152 : SchedWriteRes<[HWPort0,HWPort6,HWPort23,HWPort0,HWPort0,HWPort15,HWPort0156]> {
  let Latency = 59;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[HWWriteResGroup152], (instregex "FXRSTOR")>;

def HWWriteResGroup153 : SchedWriteRes<[HWPort5,HWPort01,HWPort0156]> {
  let Latency = 75;
  let NumMicroOps = 15;
  let ResourceCycles = [6,3,6];
}
def: InstRW<[HWWriteResGroup153], (instregex "FNINIT")>;

def HWWriteResGroup154 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort6,HWPort01,HWPort0156]> {
  let Latency = 98;
  let NumMicroOps = 32;
  let ResourceCycles = [7,7,3,3,1,11];
}
def: InstRW<[HWWriteResGroup154], (instregex "DIV64r")>;

def HWWriteResGroup155 : SchedWriteRes<[HWPort0,HWPort1,HWPort5,HWPort6,HWPort0,HWPort0156]> {
  let Latency = 112;
  let NumMicroOps = 66;
  let ResourceCycles = [4,2,4,8,14,34];
}
def: InstRW<[HWWriteResGroup155], (instregex "IDIV64r")>;

def HWWriteResGroup156 : SchedWriteRes<[HWPort0,HWPort1,HWPort4,HWPort5,HWPort6,HWPort237,HWPort0,HWPort0156]> {
  let Latency = 114;
  let NumMicroOps = 100;
  let ResourceCycles = [9,9,11,8,1,11,21,30];
}
def: InstRW<[HWWriteResGroup156], (instregex "FSTENVm")>;
def: InstRW<[HWWriteResGroup156], (instregex "FSTENVm")>;

} // SchedModel
