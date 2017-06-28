//=- X86SchedSandyBridge.td - X86 Sandy Bridge Scheduling ----*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Sandy Bridge to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def SandyBridgeModel : SchedMachineModel {
  // All x86 instructions are modeled as a single micro-op, and SB can decode 4
  // instructions per cycle.
  // FIXME: Identify instructions that aren't a single fused micro-op.
  let IssueWidth = 4;
  let MicroOpBufferSize = 168; // Based on the reorder buffer.
  let LoadLatency = 4;
  let MispredictPenalty = 16;

  // Based on the LSD (loop-stream detector) queue size.
  let LoopMicroOpBufferSize = 28;

  // This flag is set to allow the scheduler to assign
  // a default model to unrecognized opcodes.
  let CompleteModel = 0;
}

let SchedModel = SandyBridgeModel in {

// Sandy Bridge can issue micro-ops to 6 different ports in one cycle.

// Ports 0, 1, and 5 handle all computation.
def SBPort0 : ProcResource<1>;
def SBPort1 : ProcResource<1>;
def SBPort5 : ProcResource<1>;

// Ports 2 and 3 are identical. They handle loads and the address half of
// stores.
def SBPort23 : ProcResource<2>;

// Port 4 gets the data half of stores. Store data can be available later than
// the store address, but since we don't model the latency of stores, we can
// ignore that.
def SBPort4 : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def SBPort01  : ProcResGroup<[SBPort0, SBPort1]>;
def SBPort05  : ProcResGroup<[SBPort0, SBPort5]>;
def SBPort15  : ProcResGroup<[SBPort1, SBPort5]>;
def SBPort015 : ProcResGroup<[SBPort0, SBPort1, SBPort5]>;

// 54 Entry Unified Scheduler
def SBPortAny : ProcResGroup<[SBPort0, SBPort1, SBPort23, SBPort4, SBPort5]> {
  let BufferSize=54;
}

// Integer division issued on port 0.
def SBDivider : ProcResource<1>;

// Loads are 4 cycles, so ReadAfterLd registers needn't be available until 4
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 4>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass SBWriteResPair<X86FoldableSchedWrite SchedRW,
                          ProcResourceKind ExePort,
                          int Lat> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, [ExePort]> { let Latency = Lat; }

  // Memory variant also uses a cycle on port 2/3 and adds 4 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, [SBPort23, ExePort]> {
     let Latency = !add(Lat, 4);
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [SBPort4]>;

def : WriteRes<WriteStore, [SBPort23, SBPort4]>;
def : WriteRes<WriteLoad,  [SBPort23]> { let Latency = 4; }
def : WriteRes<WriteMove,  [SBPort015]>;
def : WriteRes<WriteZero,  []>;

defm : SBWriteResPair<WriteALU,   SBPort015, 1>;
defm : SBWriteResPair<WriteIMul,  SBPort1,   3>;
def  : WriteRes<WriteIMulH, []> { let Latency = 3; }
defm : SBWriteResPair<WriteShift, SBPort05,  1>;
defm : SBWriteResPair<WriteJump,  SBPort5,   1>;

// This is for simple LEAs with one or two input operands.
// The complex ones can only execute on port 1, and they require two cycles on
// the port to read all inputs. We don't model that.
def : WriteRes<WriteLEA, [SBPort15]>;

// This is quite rough, latency depends on the dividend.
def : WriteRes<WriteIDiv, [SBPort0, SBDivider]> {
  let Latency = 25;
  let ResourceCycles = [1, 10];
}
def : WriteRes<WriteIDivLd, [SBPort23, SBPort0, SBDivider]> {
  let Latency = 29;
  let ResourceCycles = [1, 1, 10];
}

// Scalar and vector floating point.
defm : SBWriteResPair<WriteFAdd,   SBPort1, 3>;
defm : SBWriteResPair<WriteFMul,   SBPort0, 5>;
defm : SBWriteResPair<WriteFDiv,   SBPort0, 12>; // 10-14 cycles.
defm : SBWriteResPair<WriteFRcp,   SBPort0, 5>;
defm : SBWriteResPair<WriteFRsqrt, SBPort0, 5>;
defm : SBWriteResPair<WriteFSqrt,  SBPort0, 15>;
defm : SBWriteResPair<WriteCvtF2I, SBPort1, 3>;
defm : SBWriteResPair<WriteCvtI2F, SBPort1, 4>;
defm : SBWriteResPair<WriteCvtF2F, SBPort1, 3>;
defm : SBWriteResPair<WriteFShuffle,  SBPort5,  1>;
defm : SBWriteResPair<WriteFBlend,  SBPort05,  1>;
def : WriteRes<WriteFVarBlend, [SBPort0, SBPort5]> {
  let Latency = 2;
  let ResourceCycles = [1, 1];
}
def : WriteRes<WriteFVarBlendLd, [SBPort0, SBPort5, SBPort23]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 1];
}

// Vector integer operations.
defm : SBWriteResPair<WriteVecShift, SBPort05,  1>;
defm : SBWriteResPair<WriteVecLogic, SBPort015, 1>;
defm : SBWriteResPair<WriteVecALU,   SBPort15,  1>;
defm : SBWriteResPair<WriteVecIMul,  SBPort0,   5>;
defm : SBWriteResPair<WriteShuffle,  SBPort15,  1>;
defm : SBWriteResPair<WriteBlend,  SBPort15,  1>;
def : WriteRes<WriteVarBlend, [SBPort1, SBPort5]> {
  let Latency = 2;
  let ResourceCycles = [1, 1];
}
def : WriteRes<WriteVarBlendLd, [SBPort1, SBPort5, SBPort23]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 1];
}
def : WriteRes<WriteMPSAD, [SBPort0, SBPort1, SBPort5]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 1];
}
def : WriteRes<WriteMPSADLd, [SBPort0, SBPort1, SBPort5, SBPort23]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 1, 1];
}

// String instructions.
// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [SBPort015]> {
  let Latency = 11;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [SBPort015, SBPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [SBPort015]> {
  let Latency = 11;
  let ResourceCycles = [8];
}
def : WriteRes<WritePCmpEStrMLd, [SBPort015, SBPort23]> {
  let Latency = 11;
  let ResourceCycles = [7, 1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [SBPort015]> {
  let Latency = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [SBPort015, SBPort23]> {
  let Latency = 3;
  let ResourceCycles = [3, 1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [SBPort015]> {
  let Latency = 4;
  let ResourceCycles = [8];
}
def : WriteRes<WritePCmpEStrILd, [SBPort015, SBPort23]> {
  let Latency = 4;
  let ResourceCycles = [7, 1];
}

// AES Instructions.
def : WriteRes<WriteAESDecEnc, [SBPort015]> {
  let Latency = 8;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESDecEncLd, [SBPort015, SBPort23]> {
  let Latency = 8;
  let ResourceCycles = [2, 1];
}

def : WriteRes<WriteAESIMC, [SBPort015]> {
  let Latency = 8;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [SBPort015, SBPort23]> {
  let Latency = 8;
  let ResourceCycles = [2, 1];
}

def : WriteRes<WriteAESKeyGen, [SBPort015]> {
  let Latency = 8;
  let ResourceCycles = [11];
}
def : WriteRes<WriteAESKeyGenLd, [SBPort015, SBPort23]> {
  let Latency = 8;
  let ResourceCycles = [10, 1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [SBPort015]> {
  let Latency = 14;
  let ResourceCycles = [18];
}
def : WriteRes<WriteCLMulLd, [SBPort015, SBPort23]> {
  let Latency = 14;
  let ResourceCycles = [17, 1];
}


def : WriteRes<WriteSystem,     [SBPort015]> { let Latency = 100; }
def : WriteRes<WriteMicrocoded, [SBPort015]> { let Latency = 100; }
def : WriteRes<WriteFence, [SBPort23, SBPort4]>;
def : WriteRes<WriteNop, []>;

// AVX2 is not supported on that architecture, but we should define the basic
// scheduling resources anyway.
defm : SBWriteResPair<WriteFShuffle256, SBPort0,  1>;
defm : SBWriteResPair<WriteShuffle256, SBPort0,  1>;
defm : SBWriteResPair<WriteVarVecShift, SBPort0,  1>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////
// HADD, HSUB PS/PD
// x,x / v,v,v.
def : WriteRes<WriteFHAdd, [SBPort1]> {
  let Latency = 3;
}

// x,m / v,v,m.
def : WriteRes<WriteFHAddLd, [SBPort1, SBPort23]> {
  let Latency = 7;
  let ResourceCycles = [1, 1];
}

// PHADD|PHSUB (S) W/D.
// v <- v,v.
def : WriteRes<WritePHAdd, [SBPort15]>;

// v <- v,m.
def : WriteRes<WritePHAddLd, [SBPort15, SBPort23]> {
  let Latency = 5;
  let ResourceCycles = [1, 1];
}

// Remaining SNB instrs.

def SBWriteResGroup0 : SchedWriteRes<[SBPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup0], (instregex "CVTSS2SDrr")>;
def: InstRW<[SBWriteResGroup0], (instregex "PSLLDri")>;
def: InstRW<[SBWriteResGroup0], (instregex "PSLLQri")>;
def: InstRW<[SBWriteResGroup0], (instregex "PSLLWri")>;
def: InstRW<[SBWriteResGroup0], (instregex "PSRADri")>;
def: InstRW<[SBWriteResGroup0], (instregex "PSRAWri")>;
def: InstRW<[SBWriteResGroup0], (instregex "PSRLDri")>;
def: InstRW<[SBWriteResGroup0], (instregex "PSRLQri")>;
def: InstRW<[SBWriteResGroup0], (instregex "PSRLWri")>;
def: InstRW<[SBWriteResGroup0], (instregex "VCVTSS2SDrr")>;
def: InstRW<[SBWriteResGroup0], (instregex "VPMOVMSKBrr")>;
def: InstRW<[SBWriteResGroup0], (instregex "VPSLLDri")>;
def: InstRW<[SBWriteResGroup0], (instregex "VPSLLQri")>;
def: InstRW<[SBWriteResGroup0], (instregex "VPSLLWri")>;
def: InstRW<[SBWriteResGroup0], (instregex "VPSRADri")>;
def: InstRW<[SBWriteResGroup0], (instregex "VPSRAWri")>;
def: InstRW<[SBWriteResGroup0], (instregex "VPSRLDri")>;
def: InstRW<[SBWriteResGroup0], (instregex "VPSRLQri")>;
def: InstRW<[SBWriteResGroup0], (instregex "VPSRLWri")>;
def: InstRW<[SBWriteResGroup0], (instregex "VTESTPDYrr")>;
def: InstRW<[SBWriteResGroup0], (instregex "VTESTPDrr")>;
def: InstRW<[SBWriteResGroup0], (instregex "VTESTPSYrr")>;
def: InstRW<[SBWriteResGroup0], (instregex "VTESTPSrr")>;

def SBWriteResGroup1 : SchedWriteRes<[SBPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup1], (instregex "ANDNPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "ANDNPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "ANDPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "ANDPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "FDECSTP")>;
def: InstRW<[SBWriteResGroup1], (instregex "FFREE")>;
def: InstRW<[SBWriteResGroup1], (instregex "FINCSTP")>;
def: InstRW<[SBWriteResGroup1], (instregex "FNOP")>;
def: InstRW<[SBWriteResGroup1], (instregex "INSERTPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "JMP64r")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOV64toPQIrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVAPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVAPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVDDUPrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVDI2PDIrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVHLPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVLHPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVSDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVSHDUPrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVSLDUPrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVSSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVUPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "MOVUPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "ORPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "ORPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "RETQ")>;
def: InstRW<[SBWriteResGroup1], (instregex "SHUFPDrri")>;
def: InstRW<[SBWriteResGroup1], (instregex "SHUFPSrri")>;
def: InstRW<[SBWriteResGroup1], (instregex "UNPCKHPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "UNPCKHPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "UNPCKLPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "UNPCKLPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VANDNPDYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VANDNPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VANDNPSYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VANDNPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VANDPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VANDPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VANDPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VEXTRACTF128rr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VGATHERQPSZrm")>;
def: InstRW<[SBWriteResGroup1], (instregex "VINSERTF128rr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VINSERTPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOV64toPQIrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOV64toPQIrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVAPDYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVAPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVAPSYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVAPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVDDUPYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVDDUPrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVHLPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVHLPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVSDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVSHDUPYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVSHDUPrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVSLDUPYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVSLDUPrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVSSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVUPDYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVUPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVUPSYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VMOVUPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VORPDYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VORPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VORPSYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VORPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VPERMILPDri")>;
def: InstRW<[SBWriteResGroup1], (instregex "VPERMILPDrm")>;
def: InstRW<[SBWriteResGroup1], (instregex "VPERMILPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VPERMILPSri")>;
def: InstRW<[SBWriteResGroup1], (instregex "VPERMILPSrm")>;
def: InstRW<[SBWriteResGroup1], (instregex "VPERMILPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VPERMILPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VSHUFPDYrri")>;
def: InstRW<[SBWriteResGroup1], (instregex "VSHUFPDrri")>;
def: InstRW<[SBWriteResGroup1], (instregex "VSHUFPSYrri")>;
def: InstRW<[SBWriteResGroup1], (instregex "VSHUFPSrri")>;
def: InstRW<[SBWriteResGroup1], (instregex "VUNPCKHPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VUNPCKHPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VUNPCKLPDYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VUNPCKLPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VUNPCKLPSYrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VUNPCKLPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VXORPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "VXORPSrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "XORPDrr")>;
def: InstRW<[SBWriteResGroup1], (instregex "XORPSrr")>;

def SBWriteResGroup2 : SchedWriteRes<[SBPort01]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup2], (instregex "LEA64_32r")>;

def SBWriteResGroup3 : SchedWriteRes<[SBPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup3], (instregex "BLENDPDrri")>;
def: InstRW<[SBWriteResGroup3], (instregex "BLENDPSrri")>;
def: InstRW<[SBWriteResGroup3], (instregex "BT32ri8")>;
def: InstRW<[SBWriteResGroup3], (instregex "BT32rr")>;
def: InstRW<[SBWriteResGroup3], (instregex "BTC32ri8")>;
def: InstRW<[SBWriteResGroup3], (instregex "BTC32rr")>;
def: InstRW<[SBWriteResGroup3], (instregex "BTR32ri8")>;
def: InstRW<[SBWriteResGroup3], (instregex "BTR32rr")>;
def: InstRW<[SBWriteResGroup3], (instregex "BTS32ri8")>;
def: InstRW<[SBWriteResGroup3], (instregex "BTS32rr")>;
def: InstRW<[SBWriteResGroup3], (instregex "CDQ")>;
def: InstRW<[SBWriteResGroup3], (instregex "CQO")>;
def: InstRW<[SBWriteResGroup3], (instregex "LAHF")>;
def: InstRW<[SBWriteResGroup3], (instregex "SAHF")>;
def: InstRW<[SBWriteResGroup3], (instregex "SAR32ri")>;
def: InstRW<[SBWriteResGroup3], (instregex "SAR8ri")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETAEr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETBr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETEr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETGEr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETGr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETLEr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETLr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETNEr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETNOr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETNPr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETNSr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETOr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETPr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SETSr")>;
def: InstRW<[SBWriteResGroup3], (instregex "SHL32ri")>;
def: InstRW<[SBWriteResGroup3], (instregex "SHL64r1")>;
def: InstRW<[SBWriteResGroup3], (instregex "SHL8r1")>;
def: InstRW<[SBWriteResGroup3], (instregex "SHL8ri")>;
def: InstRW<[SBWriteResGroup3], (instregex "SHR32ri")>;
def: InstRW<[SBWriteResGroup3], (instregex "SHR8ri")>;
def: InstRW<[SBWriteResGroup3], (instregex "VBLENDPDYrri")>;
def: InstRW<[SBWriteResGroup3], (instregex "VBLENDPDrri")>;
def: InstRW<[SBWriteResGroup3], (instregex "VBLENDPSYrri")>;
def: InstRW<[SBWriteResGroup3], (instregex "VBLENDPSrri")>;
def: InstRW<[SBWriteResGroup3], (instregex "VMOVDQAYrr")>;
def: InstRW<[SBWriteResGroup3], (instregex "VMOVDQArr")>;
def: InstRW<[SBWriteResGroup3], (instregex "VMOVDQUYrr")>;
def: InstRW<[SBWriteResGroup3], (instregex "VMOVDQUrr")>;

def SBWriteResGroup4 : SchedWriteRes<[SBPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup4], (instregex "KORTESTBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "MMX_PABSBrr64")>;
def: InstRW<[SBWriteResGroup4], (instregex "MMX_PABSDrr64")>;
def: InstRW<[SBWriteResGroup4], (instregex "MMX_PABSWrr64")>;
def: InstRW<[SBWriteResGroup4], (instregex "MMX_PADDQirr")>;
def: InstRW<[SBWriteResGroup4], (instregex "MMX_PALIGNR64irr")>;
def: InstRW<[SBWriteResGroup4], (instregex "MMX_PSHUFBrr64")>;
def: InstRW<[SBWriteResGroup4], (instregex "MMX_PSIGNBrr64")>;
def: InstRW<[SBWriteResGroup4], (instregex "MMX_PSIGNDrr64")>;
def: InstRW<[SBWriteResGroup4], (instregex "MMX_PSIGNWrr64")>;
def: InstRW<[SBWriteResGroup4], (instregex "PABSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PABSDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PABSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PACKSSDWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PACKSSWBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PACKUSDWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PACKUSWBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PADDBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PADDDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PADDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PADDSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PADDSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PADDUSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PADDUSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PADDWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PALIGNRrri")>;
def: InstRW<[SBWriteResGroup4], (instregex "PAVGBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PAVGWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PBLENDWrri")>;
def: InstRW<[SBWriteResGroup4], (instregex "PCMPEQBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PCMPEQDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PCMPEQQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PCMPEQWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PCMPGTBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PCMPGTDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PCMPGTWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMAXSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMAXSDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMAXSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMAXUBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMAXUDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMAXUWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMINSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMINSDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMINSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMINUBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMINUDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMINUWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVSXBDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVSXBQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVSXBWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVSXDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVSXWDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVSXWQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVZXBDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVZXBQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVZXBWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVZXDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVZXWDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PMOVZXWQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSHUFBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSHUFDri")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSHUFHWri")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSHUFLWri")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSIGNBrr128")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSIGNDrr128")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSIGNWrr128")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSLLDQri")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSRLDQri")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSUBBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSUBDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSUBQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSUBSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSUBSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSUBUSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSUBUSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PSUBWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PUNPCKHBWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PUNPCKHDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PUNPCKHQDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PUNPCKHWDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PUNPCKLBWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PUNPCKLDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PUNPCKLQDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "PUNPCKLWDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VMASKMOVPSYrm")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPABSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPABSDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPABSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPACKSSDWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPACKSSWBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPACKUSDWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPACKUSWBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPADDBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPADDDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPADDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPADDUSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPADDUSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPALIGNRrri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPAVGBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPAVGWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPBLENDWrri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPCMPEQBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPCMPEQDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPCMPEQWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPCMPGTBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPCMPGTDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPCMPGTWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMAXSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMAXSDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMAXSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMAXUBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMAXUDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMAXUWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMINSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMINSDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMINSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMINUBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMINUDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMINUWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVSXBDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVSXBQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVSXBWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVSXDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVSXWDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVSXWQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVZXBDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVZXBQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVZXBWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVZXDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVZXWDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPMOVZXWQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSHUFBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSHUFDri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSHUFLWri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSIGNBrr128")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSIGNDrr128")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSIGNWrr128")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSLLDQri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSRLDQri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSUBBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSUBDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSUBQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSUBSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSUBSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSUBUSBrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSUBUSWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPSUBWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPUNPCKHBWrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPUNPCKHDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPUNPCKHWDrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPUNPCKLDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPUNPCKLQDQrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VPUNPCKLWDrr")>;

def SBWriteResGroup5 : SchedWriteRes<[SBPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup5], (instregex "ADD32ri8")>;
def: InstRW<[SBWriteResGroup5], (instregex "ADD32rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "ADD8ri")>;
def: InstRW<[SBWriteResGroup5], (instregex "ADD8rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "AND32ri")>;
def: InstRW<[SBWriteResGroup5], (instregex "AND64ri8")>;
def: InstRW<[SBWriteResGroup5], (instregex "AND64rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "AND8ri")>;
def: InstRW<[SBWriteResGroup5], (instregex "AND8rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "CBW")>;
def: InstRW<[SBWriteResGroup5], (instregex "CMC")>;
def: InstRW<[SBWriteResGroup5], (instregex "CMP16ri8")>;
def: InstRW<[SBWriteResGroup5], (instregex "CMP32i32")>;
def: InstRW<[SBWriteResGroup5], (instregex "CMP64rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "CMP8ri")>;
def: InstRW<[SBWriteResGroup5], (instregex "CMP8rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "CWDE")>;
def: InstRW<[SBWriteResGroup5], (instregex "DEC64r")>;
def: InstRW<[SBWriteResGroup5], (instregex "DEC8r")>;
def: InstRW<[SBWriteResGroup5], (instregex "INC64r")>;
def: InstRW<[SBWriteResGroup5], (instregex "INC8r")>;
def: InstRW<[SBWriteResGroup5], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "MMX_MOVQ2DQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "MOV32rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "MOV8ri")>;
def: InstRW<[SBWriteResGroup5], (instregex "MOV8rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "MOVDQArr")>;
def: InstRW<[SBWriteResGroup5], (instregex "MOVDQUrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "MOVPQI2QIrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "MOVSX32rr16")>;
def: InstRW<[SBWriteResGroup5], (instregex "MOVSX32rr8")>;
def: InstRW<[SBWriteResGroup5], (instregex "MOVZX32rr16")>;
def: InstRW<[SBWriteResGroup5], (instregex "MOVZX32rr8")>;
def: InstRW<[SBWriteResGroup5], (instregex "NEG64r")>;
def: InstRW<[SBWriteResGroup5], (instregex "NEG8r")>;
def: InstRW<[SBWriteResGroup5], (instregex "NOT64r")>;
def: InstRW<[SBWriteResGroup5], (instregex "NOT8r")>;
def: InstRW<[SBWriteResGroup5], (instregex "OR64ri8")>;
def: InstRW<[SBWriteResGroup5], (instregex "OR64rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "OR8ri")>;
def: InstRW<[SBWriteResGroup5], (instregex "OR8rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PANDNrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PANDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PORrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PXORrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "STC")>;
def: InstRW<[SBWriteResGroup5], (instregex "SUB64ri8")>;
def: InstRW<[SBWriteResGroup5], (instregex "SUB64rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "SUB8ri")>;
def: InstRW<[SBWriteResGroup5], (instregex "SUB8rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "TEST64rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "TEST8ri")>;
def: InstRW<[SBWriteResGroup5], (instregex "TEST8rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VMOVPQI2QIrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VMOVZPQILo2PQIrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPANDNrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPANDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPORrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPXORrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "XOR32rr")>;
def: InstRW<[SBWriteResGroup5], (instregex "XOR64ri8")>;
def: InstRW<[SBWriteResGroup5], (instregex "XOR8ri")>;
def: InstRW<[SBWriteResGroup5], (instregex "XOR8rr")>;

def SBWriteResGroup6 : SchedWriteRes<[SBPort0]> {
  let Latency = 2;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup6], (instregex "MOVMSKPDrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVMSKPSrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVPDI2DIrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVPQIto64rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "PMOVMSKBrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VMOVMSKPDYrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VMOVMSKPDrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VMOVMSKPSrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VMOVPDI2DIrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VMOVPQIto64rr")>;

def SBWriteResGroup8 : SchedWriteRes<[SBPort0]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup8], (instregex "BLENDVPDrr0")>;
def: InstRW<[SBWriteResGroup8], (instregex "BLENDVPSrr0")>;
def: InstRW<[SBWriteResGroup8], (instregex "ROL32ri")>;
def: InstRW<[SBWriteResGroup8], (instregex "ROL8ri")>;
def: InstRW<[SBWriteResGroup8], (instregex "ROR32ri")>;
def: InstRW<[SBWriteResGroup8], (instregex "ROR8ri")>;
def: InstRW<[SBWriteResGroup8], (instregex "SETAr")>;
def: InstRW<[SBWriteResGroup8], (instregex "SETBEr")>;
def: InstRW<[SBWriteResGroup8], (instregex "VBLENDVPDYrr")>;
def: InstRW<[SBWriteResGroup8], (instregex "VBLENDVPDrr")>;
def: InstRW<[SBWriteResGroup8], (instregex "VBLENDVPSYrr")>;
def: InstRW<[SBWriteResGroup8], (instregex "VBLENDVPSrr")>;

def SBWriteResGroup9 : SchedWriteRes<[SBPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup9], (instregex "VPBLENDVBrr")>;

def SBWriteResGroup10 : SchedWriteRes<[SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup10], (instregex "SCASB")>;
def: InstRW<[SBWriteResGroup10], (instregex "SCASL")>;
def: InstRW<[SBWriteResGroup10], (instregex "SCASQ")>;
def: InstRW<[SBWriteResGroup10], (instregex "SCASW")>;

def SBWriteResGroup11 : SchedWriteRes<[SBPort0,SBPort1]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup11], (instregex "COMISDrr")>;
def: InstRW<[SBWriteResGroup11], (instregex "COMISSrr")>;
def: InstRW<[SBWriteResGroup11], (instregex "UCOMISDrr")>;
def: InstRW<[SBWriteResGroup11], (instregex "UCOMISSrr")>;
def: InstRW<[SBWriteResGroup11], (instregex "VCOMISDrr")>;
def: InstRW<[SBWriteResGroup11], (instregex "VCOMISSrr")>;
def: InstRW<[SBWriteResGroup11], (instregex "VUCOMISDrr")>;
def: InstRW<[SBWriteResGroup11], (instregex "VUCOMISSrr")>;

def SBWriteResGroup12 : SchedWriteRes<[SBPort0,SBPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup12], (instregex "CVTPS2PDrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "PTESTrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "VCVTPS2PDYrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "VCVTPS2PDrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "VPTESTYrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "VPTESTrr")>;

def SBWriteResGroup13 : SchedWriteRes<[SBPort0,SBPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup13], (instregex "PSLLDrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "PSLLQrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "PSLLWrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "PSRADrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "PSRAWrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "PSRLDrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "PSRLQrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "PSRLWrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "VPSRADrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "VPSRAWrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "VPSRLDrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "VPSRLQrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "VPSRLWrr")>;

def SBWriteResGroup14 : SchedWriteRes<[SBPort1,SBPort0]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup14], (instregex "BSWAP32r")>;

def SBWriteResGroup15 : SchedWriteRes<[SBPort5,SBPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup15], (instregex "PINSRBrr")>;
def: InstRW<[SBWriteResGroup15], (instregex "PINSRDrr")>;
def: InstRW<[SBWriteResGroup15], (instregex "PINSRQrr")>;
def: InstRW<[SBWriteResGroup15], (instregex "PINSRWrri")>;
def: InstRW<[SBWriteResGroup15], (instregex "VPINSRBrr")>;
def: InstRW<[SBWriteResGroup15], (instregex "VPINSRDrr")>;
def: InstRW<[SBWriteResGroup15], (instregex "VPINSRQrr")>;
def: InstRW<[SBWriteResGroup15], (instregex "VPINSRWrri")>;

def SBWriteResGroup16 : SchedWriteRes<[SBPort5,SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup16], (instregex "MMX_MOVDQ2Qrr")>;

def SBWriteResGroup17 : SchedWriteRes<[SBPort0,SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup17], (instregex "ADC64ri8")>;
def: InstRW<[SBWriteResGroup17], (instregex "ADC64rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "ADC8ri")>;
def: InstRW<[SBWriteResGroup17], (instregex "ADC8rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVAE32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVB32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVE32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVG32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVGE32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVL32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVLE32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVNE32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVNO32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVNP32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVNS32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVO32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVP32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "CMOVS32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "SBB32rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "SBB64ri8")>;
def: InstRW<[SBWriteResGroup17], (instregex "SBB8ri")>;
def: InstRW<[SBWriteResGroup17], (instregex "SBB8rr")>;
def: InstRW<[SBWriteResGroup17], (instregex "SHLD32rri8")>;
def: InstRW<[SBWriteResGroup17], (instregex "SHRD32rri8")>;

def SBWriteResGroup18 : SchedWriteRes<[SBPort0]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup18], (instregex "MMX_PMADDUBSWrr64")>;
def: InstRW<[SBWriteResGroup18], (instregex "MMX_PMULHRSWrr64")>;
def: InstRW<[SBWriteResGroup18], (instregex "MMX_PMULUDQirr")>;
def: InstRW<[SBWriteResGroup18], (instregex "PMADDUBSWrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "PMADDWDrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "PMULDQrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "PMULHRSWrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "PMULHUWrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "PMULHWrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "PMULLDrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "PMULLWrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "PMULUDQrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "PSADBWrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "VMOVMSKPSYrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "VPMADDUBSWrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "VPMADDWDrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "VPMULDQrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "VPMULHRSWrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "VPMULHWrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "VPMULLDrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "VPMULLWrr")>;
def: InstRW<[SBWriteResGroup18], (instregex "VPSADBWrr")>;

def SBWriteResGroup19 : SchedWriteRes<[SBPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup19], (instregex "ADDPDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "ADDPSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "ADDSDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "ADDSSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "ADDSUBPDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "ADDSUBPSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "BSF32rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "BSR32rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMPPDrri")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMPPSrri")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMPSDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMPSSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CRC32r32r32")>;
def: InstRW<[SBWriteResGroup19], (instregex "CRC32r32r8")>;
def: InstRW<[SBWriteResGroup19], (instregex "CVTDQ2PSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CVTPS2DQrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CVTTPS2DQrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MAXPDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MAXPSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MAXSDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MAXSSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MINPDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MINPSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MINSDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MINSSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MMX_CVTPI2PSirr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MMX_CVTPS2PIirr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MMX_CVTTPS2PIirr")>;
def: InstRW<[SBWriteResGroup19], (instregex "MUL8r")>;
def: InstRW<[SBWriteResGroup19], (instregex "POPCNT32rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "ROUNDPDr")>;
def: InstRW<[SBWriteResGroup19], (instregex "ROUNDPSr")>;
def: InstRW<[SBWriteResGroup19], (instregex "ROUNDSDr")>;
def: InstRW<[SBWriteResGroup19], (instregex "ROUNDSSr")>;
def: InstRW<[SBWriteResGroup19], (instregex "SUBPDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "SUBPSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "SUBSDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "SUBSSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VADDPDYrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VADDPDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VADDPSYrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VADDPSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VADDSDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VADDSSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VADDSUBPDYrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VADDSUBPDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VADDSUBPSYrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VADDSUBPSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VBROADCASTF128")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCMPPDYrri")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCMPPDrri")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCMPPSYrri")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCMPPSrri")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCMPSDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCMPSSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCVTDQ2PSYrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCVTDQ2PSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCVTPS2DQYrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCVTPS2DQrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VCVTTPS2DQrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VMAXPDYrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VMAXPDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VMAXPSYrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VMAXPSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VMAXSDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VMAXSSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VMINPDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VMINPSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VMINSDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VMINSSrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VPBROADCASTMB2QZrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VROUNDPDr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VROUNDPSr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VROUNDSDr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VSUBPDYrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VSUBPDrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VSUBPSYrr")>;
def: InstRW<[SBWriteResGroup19], (instregex "VSUBPSrr")>;

def SBWriteResGroup20 : SchedWriteRes<[SBPort0,SBPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup20], (instregex "EXTRACTPSrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VEXTRACTPSrr")>;

def SBWriteResGroup21 : SchedWriteRes<[SBPort0,SBPort15]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup21], (instregex "PEXTRBrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "PEXTRDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "PEXTRQrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "PEXTRWri")>;
def: InstRW<[SBWriteResGroup21], (instregex "VPEXTRBrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VPEXTRDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VPEXTRQrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VPEXTRWri")>;
def: InstRW<[SBWriteResGroup21], (instregex "SHL64rCL")>;
def: InstRW<[SBWriteResGroup21], (instregex "SHL8rCL")>;

def SBWriteResGroup22 : SchedWriteRes<[SBPort15]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SBWriteResGroup22], (instregex "MMX_PHADDSWrr64")>;
def: InstRW<[SBWriteResGroup22], (instregex "MMX_PHADDWrr64")>;
def: InstRW<[SBWriteResGroup22], (instregex "MMX_PHADDrr64")>;
def: InstRW<[SBWriteResGroup22], (instregex "MMX_PHSUBDrr64")>;
def: InstRW<[SBWriteResGroup22], (instregex "MMX_PHSUBSWrr64")>;
def: InstRW<[SBWriteResGroup22], (instregex "MMX_PHSUBWrr64")>;
def: InstRW<[SBWriteResGroup22], (instregex "PHADDDrr")>;
def: InstRW<[SBWriteResGroup22], (instregex "PHADDSWrr128")>;
def: InstRW<[SBWriteResGroup22], (instregex "PHADDWrr")>;
def: InstRW<[SBWriteResGroup22], (instregex "PHSUBDrr")>;
def: InstRW<[SBWriteResGroup22], (instregex "PHSUBSWrr128")>;
def: InstRW<[SBWriteResGroup22], (instregex "PHSUBWrr")>;
def: InstRW<[SBWriteResGroup22], (instregex "VPHADDDrr")>;
def: InstRW<[SBWriteResGroup22], (instregex "VPHADDSWrr128")>;
def: InstRW<[SBWriteResGroup22], (instregex "VPHADDWrr")>;
def: InstRW<[SBWriteResGroup22], (instregex "VPHSUBDrr")>;
def: InstRW<[SBWriteResGroup22], (instregex "VPHSUBSWrr128")>;
def: InstRW<[SBWriteResGroup22], (instregex "VPHSUBWrr")>;

def SBWriteResGroup23 : SchedWriteRes<[SBPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SBWriteResGroup23], (instregex "LEAVE64")>;
def: InstRW<[SBWriteResGroup23], (instregex "XADD32rr")>;
def: InstRW<[SBWriteResGroup23], (instregex "XADD8rr")>;

def SBWriteResGroup24 : SchedWriteRes<[SBPort0,SBPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup24], (instregex "CMOVA32rr")>;
def: InstRW<[SBWriteResGroup24], (instregex "CMOVBE32rr")>;

def SBWriteResGroup25 : SchedWriteRes<[SBPort0,SBPort1]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup25], (instregex "MUL64r")>;

def SBWriteResGroup26 : SchedWriteRes<[SBPort1,SBPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup26], (instregex "CVTDQ2PDrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "CVTPD2DQrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "CVTPD2PSrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "CVTSD2SSrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "CVTSI2SD64rr")>;
def: InstRW<[SBWriteResGroup26], (instregex "CVTSI2SDrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "CVTTPD2DQrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "MMX_CVTPD2PIirr")>;
def: InstRW<[SBWriteResGroup26], (instregex "MMX_CVTPI2PDirr")>;
def: InstRW<[SBWriteResGroup26], (instregex "MMX_CVTTPD2PIirr")>;
def: InstRW<[SBWriteResGroup26], (instregex "VCVTDQ2PDYrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "VCVTDQ2PDrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "VCVTPD2DQYrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "VCVTPD2DQrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "VCVTPD2PSYrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "VCVTPD2PSrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "VCVTSI2SD64rr")>;
def: InstRW<[SBWriteResGroup26], (instregex "VCVTSI2SDrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "VCVTTPD2DQYrr")>;
def: InstRW<[SBWriteResGroup26], (instregex "VCVTTPD2DQrr")>;

def SBWriteResGroup27 : SchedWriteRes<[SBPort1,SBPort015]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup27], (instregex "MOV64sr")>;
def: InstRW<[SBWriteResGroup27], (instregex "PAUSE")>;

def SBWriteResGroup28 : SchedWriteRes<[SBPort0]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup28], (instregex "MULPDrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "MULPSrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "MULSDrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "MULSSrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "PCMPGTQrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "PHMINPOSUWrr128")>;
def: InstRW<[SBWriteResGroup28], (instregex "RCPPSr")>;
def: InstRW<[SBWriteResGroup28], (instregex "RCPSSr")>;
def: InstRW<[SBWriteResGroup28], (instregex "RSQRTPSr")>;
def: InstRW<[SBWriteResGroup28], (instregex "RSQRTSSr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VMULPDYrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VMULPDrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VMULPSYrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VMULPSrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VMULSDrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VMULSSrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VPCMPGTQrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VPHMINPOSUWrr128")>;
def: InstRW<[SBWriteResGroup28], (instregex "VRSQRTPSr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VRSQRTSSr")>;

def SBWriteResGroup29 : SchedWriteRes<[SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup29], (instregex "MOV32rm")>;
def: InstRW<[SBWriteResGroup29], (instregex "MOV8rm")>;
def: InstRW<[SBWriteResGroup29], (instregex "MOVSX32rm16")>;
def: InstRW<[SBWriteResGroup29], (instregex "MOVSX32rm8")>;
def: InstRW<[SBWriteResGroup29], (instregex "MOVZX32rm16")>;
def: InstRW<[SBWriteResGroup29], (instregex "MOVZX32rm8")>;
def: InstRW<[SBWriteResGroup29], (instregex "PREFETCH")>;

def SBWriteResGroup30 : SchedWriteRes<[SBPort0,SBPort1]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup30], (instregex "CVTSD2SI64rr")>;
def: InstRW<[SBWriteResGroup30], (instregex "CVTSD2SIrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "CVTSS2SI64rr")>;
def: InstRW<[SBWriteResGroup30], (instregex "CVTSS2SIrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "CVTTSD2SI64rr")>;
def: InstRW<[SBWriteResGroup30], (instregex "CVTTSD2SIrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "CVTTSS2SI64rr")>;
def: InstRW<[SBWriteResGroup30], (instregex "CVTTSS2SIrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VCVTSD2SI64rr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VCVTSS2SI64rr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VCVTSS2SIrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VCVTTSD2SI64rr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VCVTTSD2SIrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VCVTTSS2SI64rr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VCVTTSS2SIrr")>;

def SBWriteResGroup31 : SchedWriteRes<[SBPort4,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup31], (instregex "MOV64mr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOV8mr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVAPDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVAPSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVDQAmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVDQUmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVHPDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVHPSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVLPDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVLPSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVNTDQmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVNTI_64mr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVNTImr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVNTPDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVNTPSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVPDI2DImr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVPQI2QImr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVPQIto64mr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVSSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVUPDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVUPSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "PUSH64i8")>;
def: InstRW<[SBWriteResGroup31], (instregex "PUSH64r")>;
def: InstRW<[SBWriteResGroup31], (instregex "VEXTRACTF128mr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVAPDYmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVAPDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVAPSYmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVAPSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVDQAYmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVDQAmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVDQUYmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVDQUmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVHPDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVHPSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVLPDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVLPSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVNTDQYmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVNTDQmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVNTPDYmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVNTPDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVNTPSYmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVNTPSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVPDI2DImr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVPQI2QImr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVPQIto64mr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVSDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVSSmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVUPDYmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVUPDmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVUPSYmr")>;
def: InstRW<[SBWriteResGroup31], (instregex "VMOVUPSmr")>;

def SBWriteResGroup32 : SchedWriteRes<[SBPort0,SBPort15]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup32], (instregex "MPSADBWrri")>;
def: InstRW<[SBWriteResGroup32], (instregex "VMPSADBWrri")>;

def SBWriteResGroup33 : SchedWriteRes<[SBPort1,SBPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup33], (instregex "CLI")>;
def: InstRW<[SBWriteResGroup33], (instregex "CVTSI2SS64rr")>;
def: InstRW<[SBWriteResGroup33], (instregex "CVTSI2SSrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "HADDPDrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "HADDPSrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "HSUBPDrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "HSUBPSrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VCVTSI2SS64rr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VCVTSI2SSrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VHADDPDrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VHADDPSYrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VHADDPSrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VHSUBPDYrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VHSUBPDrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VHSUBPSYrr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VHSUBPSrr")>;

def SBWriteResGroup34 : SchedWriteRes<[SBPort4,SBPort5,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup34], (instregex "CALL64r")>;
def: InstRW<[SBWriteResGroup34], (instregex "EXTRACTPSmr")>;
def: InstRW<[SBWriteResGroup34], (instregex "VEXTRACTPSmr")>;

def SBWriteResGroup35 : SchedWriteRes<[SBPort4,SBPort01,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup35], (instregex "VMASKMOVPDYrm")>;
def: InstRW<[SBWriteResGroup35], (instregex "VMASKMOVPDmr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VMASKMOVPSmr")>;

def SBWriteResGroup36 : SchedWriteRes<[SBPort4,SBPort23,SBPort0]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup36], (instregex "SETAEm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETBm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETEm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETGEm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETGm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETLEm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETLm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETNEm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETNOm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETNPm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETNSm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETOm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETPm")>;
def: InstRW<[SBWriteResGroup36], (instregex "SETSm")>;

def SBWriteResGroup37 : SchedWriteRes<[SBPort4,SBPort23,SBPort15]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup37], (instregex "PEXTRBmr")>;
def: InstRW<[SBWriteResGroup37], (instregex "VPEXTRBmr")>;
def: InstRW<[SBWriteResGroup37], (instregex "VPEXTRDmr")>;
def: InstRW<[SBWriteResGroup37], (instregex "VPEXTRWmr")>;

def SBWriteResGroup38 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup38], (instregex "MOV8mi")>;
def: InstRW<[SBWriteResGroup38], (instregex "STOSB")>;
def: InstRW<[SBWriteResGroup38], (instregex "STOSL")>;
def: InstRW<[SBWriteResGroup38], (instregex "STOSQ")>;
def: InstRW<[SBWriteResGroup38], (instregex "STOSW")>;

def SBWriteResGroup39 : SchedWriteRes<[SBPort5,SBPort015]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup39], (instregex "FNINIT")>;

def SBWriteResGroup40 : SchedWriteRes<[SBPort0,SBPort015]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup40], (instregex "CMPXCHG32rr")>;
def: InstRW<[SBWriteResGroup40], (instregex "CMPXCHG8rr")>;

def SBWriteResGroup41 : SchedWriteRes<[SBPort4,SBPort23,SBPort0]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup41], (instregex "SETAm")>;
def: InstRW<[SBWriteResGroup41], (instregex "SETBEm")>;

def SBWriteResGroup42 : SchedWriteRes<[SBPort0,SBPort4,SBPort5,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup42], (instregex "LDMXCSR")>;
def: InstRW<[SBWriteResGroup42], (instregex "STMXCSR")>;
def: InstRW<[SBWriteResGroup42], (instregex "VLDMXCSR")>;
def: InstRW<[SBWriteResGroup42], (instregex "VSTMXCSR")>;

def SBWriteResGroup43 : SchedWriteRes<[SBPort0,SBPort4,SBPort23,SBPort15]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup43], (instregex "PEXTRDmr")>;
def: InstRW<[SBWriteResGroup43], (instregex "PEXTRQmr")>;
def: InstRW<[SBWriteResGroup43], (instregex "VPEXTRQmr")>;
def: InstRW<[SBWriteResGroup43], (instregex "PUSHF16")>;
def: InstRW<[SBWriteResGroup43], (instregex "PUSHF64")>;

def SBWriteResGroup44 : SchedWriteRes<[SBPort4,SBPort5,SBPort01,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup44], (instregex "CLFLUSH")>;

def SBWriteResGroup45 : SchedWriteRes<[SBPort4,SBPort5,SBPort01,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SBWriteResGroup45], (instregex "FXRSTOR")>;

def SBWriteResGroup46 : SchedWriteRes<[SBPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup46], (instregex "LDDQUrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOV64toPQIrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVAPDrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVAPSrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVDDUPrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVDI2PDIrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVDQArm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVDQUrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVNTDQArm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVSHDUPrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVSLDUPrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVSSrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVUPDrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "MOVUPSrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "POP64r")>;
def: InstRW<[SBWriteResGroup46], (instregex "VBROADCASTSSrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VLDDQUYrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VLDDQUrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOV64toPQIrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVAPDrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVAPSrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVDDUPrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVDI2PDIrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVDQArm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVDQUrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVNTDQArm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVQI2PQIrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVSDrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVSHDUPrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVSLDUPrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVSSrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVUPDrm")>;
def: InstRW<[SBWriteResGroup46], (instregex "VMOVUPSrm")>;

def SBWriteResGroup47 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup47], (instregex "JMP64m")>;
def: InstRW<[SBWriteResGroup47], (instregex "MOV64sm")>;

def SBWriteResGroup48 : SchedWriteRes<[SBPort23,SBPort0]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup48], (instregex "BT64mi8")>;

def SBWriteResGroup49 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup49], (instregex "MMX_PABSBrm64")>;
def: InstRW<[SBWriteResGroup49], (instregex "MMX_PABSDrm64")>;
def: InstRW<[SBWriteResGroup49], (instregex "MMX_PABSWrm64")>;
def: InstRW<[SBWriteResGroup49], (instregex "MMX_PALIGNR64irm")>;
def: InstRW<[SBWriteResGroup49], (instregex "MMX_PSHUFBrm64")>;
def: InstRW<[SBWriteResGroup49], (instregex "MMX_PSIGNBrm64")>;
def: InstRW<[SBWriteResGroup49], (instregex "MMX_PSIGNDrm64")>;
def: InstRW<[SBWriteResGroup49], (instregex "MMX_PSIGNWrm64")>;

def SBWriteResGroup50 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup50], (instregex "ADD64rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "ADD8rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "AND64rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "AND8rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "CMP64mi8")>;
def: InstRW<[SBWriteResGroup50], (instregex "CMP64mr")>;
def: InstRW<[SBWriteResGroup50], (instregex "CMP64rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "CMP8mi")>;
def: InstRW<[SBWriteResGroup50], (instregex "CMP8mr")>;
def: InstRW<[SBWriteResGroup50], (instregex "CMP8rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "LODSL")>;
def: InstRW<[SBWriteResGroup50], (instregex "LODSQ")>;
def: InstRW<[SBWriteResGroup50], (instregex "OR64rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "OR8rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "SUB64rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "SUB8rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "XOR64rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "XOR8rm")>;
def: InstRW<[SBWriteResGroup50], (instregex "POP64rmm")>;
def: InstRW<[SBWriteResGroup50], (instregex "PUSH64rmm")>;

def SBWriteResGroup51 : SchedWriteRes<[SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup51], (instregex "VBROADCASTSDYrm")>;
def: InstRW<[SBWriteResGroup51], (instregex "VBROADCASTSSrm")>;
def: InstRW<[SBWriteResGroup51], (instregex "VMOVAPDYrm")>;
def: InstRW<[SBWriteResGroup51], (instregex "VMOVAPSYrm")>;
def: InstRW<[SBWriteResGroup51], (instregex "VMOVDDUPYrm")>;
def: InstRW<[SBWriteResGroup51], (instregex "VMOVDQAYrm")>;
def: InstRW<[SBWriteResGroup51], (instregex "VMOVDQUYrm")>;
def: InstRW<[SBWriteResGroup51], (instregex "VMOVSHDUPYrm")>;
def: InstRW<[SBWriteResGroup51], (instregex "VMOVSLDUPYrm")>;
def: InstRW<[SBWriteResGroup51], (instregex "VMOVUPDYrm")>;
def: InstRW<[SBWriteResGroup51], (instregex "VMOVUPSYrm")>;

def SBWriteResGroup52 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup52], (instregex "CVTPS2PDrm")>;
def: InstRW<[SBWriteResGroup52], (instregex "CVTSS2SDrm")>;
def: InstRW<[SBWriteResGroup52], (instregex "VCVTPS2PDYrm")>;
def: InstRW<[SBWriteResGroup52], (instregex "VCVTPS2PDrm")>;
def: InstRW<[SBWriteResGroup52], (instregex "VCVTSS2SDrm")>;
def: InstRW<[SBWriteResGroup52], (instregex "VTESTPDrm")>;
def: InstRW<[SBWriteResGroup52], (instregex "VTESTPSrm")>;

def SBWriteResGroup53 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup53], (instregex "ANDNPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "ANDNPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "ANDPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "ANDPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "INSERTPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "MOVHPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "MOVHPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "MOVLPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "MOVLPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "ORPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "ORPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "SHUFPDrmi")>;
def: InstRW<[SBWriteResGroup53], (instregex "SHUFPSrmi")>;
def: InstRW<[SBWriteResGroup53], (instregex "UNPCKHPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "UNPCKHPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "UNPCKLPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "UNPCKLPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VANDNPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VANDNPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VANDPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VANDPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VBROADCASTF128")>;
def: InstRW<[SBWriteResGroup53], (instregex "VINSERTPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VMOVHPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VMOVHPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VMOVLPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VMOVLPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VORPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VORPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VPERMILPDmi")>;
def: InstRW<[SBWriteResGroup53], (instregex "VPERMILPDri")>;
def: InstRW<[SBWriteResGroup53], (instregex "VPERMILPSmi")>;
def: InstRW<[SBWriteResGroup53], (instregex "VPERMILPSri")>;
def: InstRW<[SBWriteResGroup53], (instregex "VSHUFPDrmi")>;
def: InstRW<[SBWriteResGroup53], (instregex "VSHUFPSrmi")>;
def: InstRW<[SBWriteResGroup53], (instregex "VUNPCKHPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VUNPCKHPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VUNPCKLPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VUNPCKLPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VXORPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "VXORPSrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "XORPDrm")>;
def: InstRW<[SBWriteResGroup53], (instregex "XORPSrm")>;

def SBWriteResGroup54 : SchedWriteRes<[SBPort5,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup54], (instregex "AESDECLASTrr")>;
def: InstRW<[SBWriteResGroup54], (instregex "AESDECrr")>;
def: InstRW<[SBWriteResGroup54], (instregex "AESENCLASTrr")>;
def: InstRW<[SBWriteResGroup54], (instregex "AESENCrr")>;
def: InstRW<[SBWriteResGroup54], (instregex "KANDQrr")>;
def: InstRW<[SBWriteResGroup54], (instregex "VAESDECLASTrr")>;
def: InstRW<[SBWriteResGroup54], (instregex "VAESDECrr")>;
def: InstRW<[SBWriteResGroup54], (instregex "VAESENCrr")>;

def SBWriteResGroup55 : SchedWriteRes<[SBPort23,SBPort0]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup55], (instregex "BLENDPDrmi")>;
def: InstRW<[SBWriteResGroup55], (instregex "BLENDPSrmi")>;
def: InstRW<[SBWriteResGroup55], (instregex "VBLENDPDrmi")>;
def: InstRW<[SBWriteResGroup55], (instregex "VBLENDPSrmi")>;
def: InstRW<[SBWriteResGroup55], (instregex "VINSERTF128rm")>;

def SBWriteResGroup56 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup56], (instregex "MMX_PADDQirm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PABSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PABSDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PABSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PACKSSDWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PACKSSWBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PACKUSDWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PACKUSWBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PADDBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PADDDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PADDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PADDSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PADDSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PADDUSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PADDUSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PADDWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PALIGNRrmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "PAVGBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PAVGWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PBLENDWrmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "PCMPEQBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PCMPEQDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PCMPEQQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PCMPEQWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PCMPGTBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PCMPGTDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PCMPGTWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PINSRBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PINSRDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PINSRQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PINSRWrmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMAXSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMAXSDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMAXSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMAXUBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMAXUDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMAXUWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMINSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMINSDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMINSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMINUBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMINUDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMINUWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVSXBDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVSXBQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVSXBWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVSXDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVSXWDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVSXWQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVZXBDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVZXBQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVZXBWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVZXDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVZXWDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PMOVZXWQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSHUFBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSHUFDmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSHUFHWmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSHUFLWmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSIGNBrm128")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSIGNDrm128")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSIGNWrm128")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSUBBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSUBDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSUBQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSUBSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSUBSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSUBUSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSUBUSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PSUBWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PUNPCKHBWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PUNPCKHDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PUNPCKHQDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PUNPCKHWDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PUNPCKLBWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PUNPCKLDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PUNPCKLQDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "PUNPCKLWDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPABSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPABSDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPABSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPACKSSDWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPACKSSWBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPACKUSDWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPACKUSWBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPADDBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPADDDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPADDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPADDSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPADDSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPADDUSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPADDUSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPADDWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPALIGNRrmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPAVGBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPAVGWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPBLENDWrmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPCMPEQBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPCMPEQDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPCMPEQQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPCMPEQWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPCMPGTBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPCMPGTDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPCMPGTWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPINSRBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPINSRDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPINSRQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPINSRWrmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMAXSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMAXSDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMAXSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMAXUBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMAXUDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMAXUWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMINSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMINSDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMINSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMINUBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMINUDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMINUWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVSXBDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVSXBQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVSXBWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVSXDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVSXWDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVSXWQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVZXBDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVZXBQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVZXBWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVZXDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVZXWDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPMOVZXWQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSHUFBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSHUFDmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSHUFHWmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSHUFLWmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSIGNBrm128")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSIGNDrm128")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSIGNWrm128")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSUBBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSUBDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSUBQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSUBSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSUBSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSUBUSBrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSUBUSWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPSUBWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPUNPCKHBWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPUNPCKHDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPUNPCKHQDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPUNPCKHWDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPUNPCKLBWrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPUNPCKLDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPUNPCKLQDQrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPUNPCKLWDrm")>;

def SBWriteResGroup57 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup57], (instregex "PANDNrm")>;
def: InstRW<[SBWriteResGroup57], (instregex "PANDrm")>;
def: InstRW<[SBWriteResGroup57], (instregex "PORrm")>;
def: InstRW<[SBWriteResGroup57], (instregex "PXORrm")>;
def: InstRW<[SBWriteResGroup57], (instregex "VPANDNrm")>;
def: InstRW<[SBWriteResGroup57], (instregex "VPANDrm")>;
def: InstRW<[SBWriteResGroup57], (instregex "VPORrm")>;
def: InstRW<[SBWriteResGroup57], (instregex "VPXORrm")>;

def SBWriteResGroup58 : SchedWriteRes<[SBPort0,SBPort0]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup58], (instregex "VRCPPSr")>;
def: InstRW<[SBWriteResGroup58], (instregex "VRSQRTPSYr")>;

def SBWriteResGroup59 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup59], (instregex "VERRm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VERWm")>;

def SBWriteResGroup60 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup60], (instregex "LODSB")>;
def: InstRW<[SBWriteResGroup60], (instregex "LODSW")>;

def SBWriteResGroup61 : SchedWriteRes<[SBPort5,SBPort01,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup61], (instregex "FARJMP64")>;

def SBWriteResGroup62 : SchedWriteRes<[SBPort23,SBPort0,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup62], (instregex "ADC64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "ADC8rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVAE64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVB64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVE64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVG64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVGE64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVL64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVLE64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVNE64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVNO64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVNP64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVNS64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVO64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVP64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "CMOVS64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "SBB64rm")>;
def: InstRW<[SBWriteResGroup62], (instregex "SBB8rm")>;

def SBWriteResGroup63 : SchedWriteRes<[SBPort0,SBPort4,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup63], (instregex "FNSTSWm")>;

def SBWriteResGroup64 : SchedWriteRes<[SBPort1,SBPort5,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup64], (instregex "SLDT32r")>;
def: InstRW<[SBWriteResGroup64], (instregex "STR32r")>;

def SBWriteResGroup65 : SchedWriteRes<[SBPort4,SBPort5,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup65], (instregex "CALL64m")>;
def: InstRW<[SBWriteResGroup65], (instregex "FNSTCW16m")>;

def SBWriteResGroup66 : SchedWriteRes<[SBPort4,SBPort23,SBPort0]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup66], (instregex "BTC64mi8")>;
def: InstRW<[SBWriteResGroup66], (instregex "BTR64mi8")>;
def: InstRW<[SBWriteResGroup66], (instregex "BTS64mi8")>;
def: InstRW<[SBWriteResGroup66], (instregex "SAR64mi")>;
def: InstRW<[SBWriteResGroup66], (instregex "SAR8mi")>;
def: InstRW<[SBWriteResGroup66], (instregex "SHL64m1")>;
def: InstRW<[SBWriteResGroup66], (instregex "SHL64mi")>;
def: InstRW<[SBWriteResGroup66], (instregex "SHL8m1")>;
def: InstRW<[SBWriteResGroup66], (instregex "SHL8mi")>;
def: InstRW<[SBWriteResGroup66], (instregex "SHR64mi")>;
def: InstRW<[SBWriteResGroup66], (instregex "SHR8mi")>;

def SBWriteResGroup67 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup67], (instregex "ADD64mi8")>;
def: InstRW<[SBWriteResGroup67], (instregex "ADD64mr")>;
def: InstRW<[SBWriteResGroup67], (instregex "ADD8mi")>;
def: InstRW<[SBWriteResGroup67], (instregex "ADD8mr")>;
def: InstRW<[SBWriteResGroup67], (instregex "AND64mi8")>;
def: InstRW<[SBWriteResGroup67], (instregex "AND64mr")>;
def: InstRW<[SBWriteResGroup67], (instregex "AND8mi")>;
def: InstRW<[SBWriteResGroup67], (instregex "AND8mr")>;
def: InstRW<[SBWriteResGroup67], (instregex "DEC64m")>;
def: InstRW<[SBWriteResGroup67], (instregex "DEC8m")>;
def: InstRW<[SBWriteResGroup67], (instregex "INC64m")>;
def: InstRW<[SBWriteResGroup67], (instregex "INC8m")>;
def: InstRW<[SBWriteResGroup67], (instregex "NEG64m")>;
def: InstRW<[SBWriteResGroup67], (instregex "NEG8m")>;
def: InstRW<[SBWriteResGroup67], (instregex "NOT64m")>;
def: InstRW<[SBWriteResGroup67], (instregex "NOT8m")>;
def: InstRW<[SBWriteResGroup67], (instregex "OR64mi8")>;
def: InstRW<[SBWriteResGroup67], (instregex "OR64mr")>;
def: InstRW<[SBWriteResGroup67], (instregex "OR8mi")>;
def: InstRW<[SBWriteResGroup67], (instregex "OR8mr")>;
def: InstRW<[SBWriteResGroup67], (instregex "SUB64mi8")>;
def: InstRW<[SBWriteResGroup67], (instregex "SUB64mr")>;
def: InstRW<[SBWriteResGroup67], (instregex "SUB8mi")>;
def: InstRW<[SBWriteResGroup67], (instregex "SUB8mr")>;
def: InstRW<[SBWriteResGroup67], (instregex "TEST64rm")>;
def: InstRW<[SBWriteResGroup67], (instregex "TEST8mi")>;
def: InstRW<[SBWriteResGroup67], (instregex "TEST8rm")>;
def: InstRW<[SBWriteResGroup67], (instregex "XOR64mi8")>;
def: InstRW<[SBWriteResGroup67], (instregex "XOR64mr")>;
def: InstRW<[SBWriteResGroup67], (instregex "XOR8mi")>;
def: InstRW<[SBWriteResGroup67], (instregex "XOR8mr")>;

def SBWriteResGroup68 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup68], (instregex "MMX_PMADDUBSWrm64")>;
def: InstRW<[SBWriteResGroup68], (instregex "MMX_PMULHRSWrm64")>;
def: InstRW<[SBWriteResGroup68], (instregex "VTESTPDYrm")>;
def: InstRW<[SBWriteResGroup68], (instregex "VTESTPSYrm")>;

def SBWriteResGroup69 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup69], (instregex "BSF64rm")>;
def: InstRW<[SBWriteResGroup69], (instregex "BSR64rm")>;
def: InstRW<[SBWriteResGroup69], (instregex "CRC32r32m16")>;
def: InstRW<[SBWriteResGroup69], (instregex "CRC32r32m8")>;
def: InstRW<[SBWriteResGroup69], (instregex "MUL8m")>;

def SBWriteResGroup70 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup70], (instregex "VANDNPDYrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VANDNPSYrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VANDPDrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VANDPSrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VORPDYrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VORPSYrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VPERM2F128rm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VPERMILPDYri")>;
def: InstRW<[SBWriteResGroup70], (instregex "VPERMILPDmi")>;
def: InstRW<[SBWriteResGroup70], (instregex "VPERMILPSYri")>;
def: InstRW<[SBWriteResGroup70], (instregex "VPERMILPSmi")>;
def: InstRW<[SBWriteResGroup70], (instregex "VSHUFPDYrmi")>;
def: InstRW<[SBWriteResGroup70], (instregex "VSHUFPSYrmi")>;
def: InstRW<[SBWriteResGroup70], (instregex "VUNPCKHPDrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VUNPCKHPSrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VUNPCKLPDYrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VUNPCKLPSYrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VXORPDrm")>;
def: InstRW<[SBWriteResGroup70], (instregex "VXORPSrm")>;

def SBWriteResGroup71 : SchedWriteRes<[SBPort23,SBPort0]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup71], (instregex "VBLENDPDYrmi")>;
def: InstRW<[SBWriteResGroup71], (instregex "VBLENDPSYrmi")>;

def SBWriteResGroup72 : SchedWriteRes<[SBPort23,SBPort0]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup72], (instregex "BLENDVPDrm0")>;
def: InstRW<[SBWriteResGroup72], (instregex "BLENDVPSrm0")>;
def: InstRW<[SBWriteResGroup72], (instregex "VBLENDVPDrm")>;
def: InstRW<[SBWriteResGroup72], (instregex "VBLENDVPSrm")>;
def: InstRW<[SBWriteResGroup72], (instregex "VMASKMOVPDrm")>;
def: InstRW<[SBWriteResGroup72], (instregex "VMASKMOVPSrm")>;

def SBWriteResGroup73 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup73], (instregex "PBLENDVBrr0")>;
def: InstRW<[SBWriteResGroup73], (instregex "VPBLENDVBrm")>;

def SBWriteResGroup74 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup74], (instregex "COMISDrm")>;
def: InstRW<[SBWriteResGroup74], (instregex "COMISSrm")>;
def: InstRW<[SBWriteResGroup74], (instregex "UCOMISDrm")>;
def: InstRW<[SBWriteResGroup74], (instregex "UCOMISSrm")>;
def: InstRW<[SBWriteResGroup74], (instregex "VCOMISDrm")>;
def: InstRW<[SBWriteResGroup74], (instregex "VCOMISSrm")>;
def: InstRW<[SBWriteResGroup74], (instregex "VUCOMISDrm")>;
def: InstRW<[SBWriteResGroup74], (instregex "VUCOMISSrm")>;

def SBWriteResGroup75 : SchedWriteRes<[SBPort0,SBPort5,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup75], (instregex "PTESTrm")>;
def: InstRW<[SBWriteResGroup75], (instregex "VPTESTrm")>;

def SBWriteResGroup76 : SchedWriteRes<[SBPort0,SBPort23,SBPort15]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup76], (instregex "PSLLDrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "PSLLQrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "PSLLWrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "PSRADrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "PSRAWrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "PSRLDrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "PSRLQrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "PSRLWrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "VPSLLDri")>;
def: InstRW<[SBWriteResGroup76], (instregex "VPSLLQri")>;
def: InstRW<[SBWriteResGroup76], (instregex "VPSLLWri")>;
def: InstRW<[SBWriteResGroup76], (instregex "VPSRADrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "VPSRAWrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "VPSRLDrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "VPSRLQrm")>;
def: InstRW<[SBWriteResGroup76], (instregex "VPSRLWrm")>;

def SBWriteResGroup77 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup77], (instregex "MMX_PHADDSWrm64")>;
def: InstRW<[SBWriteResGroup77], (instregex "MMX_PHADDWrm64")>;
def: InstRW<[SBWriteResGroup77], (instregex "MMX_PHADDrm64")>;
def: InstRW<[SBWriteResGroup77], (instregex "MMX_PHSUBDrm64")>;
def: InstRW<[SBWriteResGroup77], (instregex "MMX_PHSUBSWrm64")>;
def: InstRW<[SBWriteResGroup77], (instregex "MMX_PHSUBWrm64")>;

def SBWriteResGroup78 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup78], (instregex "CMPXCHG64rm")>;
def: InstRW<[SBWriteResGroup78], (instregex "CMPXCHG8rm")>;

def SBWriteResGroup79 : SchedWriteRes<[SBPort23,SBPort0,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup79], (instregex "CMOVA64rm")>;
def: InstRW<[SBWriteResGroup79], (instregex "CMOVBE64rm")>;

def SBWriteResGroup80 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[SBWriteResGroup80], (instregex "CMPSB")>;
def: InstRW<[SBWriteResGroup80], (instregex "CMPSL")>;
def: InstRW<[SBWriteResGroup80], (instregex "CMPSQ")>;
def: InstRW<[SBWriteResGroup80], (instregex "CMPSW")>;

def SBWriteResGroup81 : SchedWriteRes<[SBPort4,SBPort5,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,2];
}
def: InstRW<[SBWriteResGroup81], (instregex "FLDCW16m")>;

def SBWriteResGroup82 : SchedWriteRes<[SBPort4,SBPort23,SBPort0]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,2];
}
def: InstRW<[SBWriteResGroup82], (instregex "ROL64mi")>;
def: InstRW<[SBWriteResGroup82], (instregex "ROL8mi")>;
def: InstRW<[SBWriteResGroup82], (instregex "ROR64mi")>;
def: InstRW<[SBWriteResGroup82], (instregex "ROR8mi")>;

def SBWriteResGroup83 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,2];
}
def: InstRW<[SBWriteResGroup83], (instregex "MOVSB")>;
def: InstRW<[SBWriteResGroup83], (instregex "MOVSL")>;
def: InstRW<[SBWriteResGroup83], (instregex "MOVSQ")>;
def: InstRW<[SBWriteResGroup83], (instregex "MOVSW")>;
def: InstRW<[SBWriteResGroup83], (instregex "XADD64rm")>;
def: InstRW<[SBWriteResGroup83], (instregex "XADD8rm")>;

def SBWriteResGroup84 : SchedWriteRes<[SBPort4,SBPort5,SBPort01,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SBWriteResGroup84], (instregex "FARCALL64")>;

def SBWriteResGroup85 : SchedWriteRes<[SBPort4,SBPort23,SBPort0,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SBWriteResGroup85], (instregex "SHLD64mri8")>;
def: InstRW<[SBWriteResGroup85], (instregex "SHRD64mri8")>;

def SBWriteResGroup86 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup86], (instregex "MMX_PMULUDQirm")>;
def: InstRW<[SBWriteResGroup86], (instregex "PMADDUBSWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "PMADDWDrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "PMULDQrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "PMULHRSWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "PMULHUWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "PMULHWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "PMULLDrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "PMULLWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "PMULUDQrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "PSADBWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "VPMADDUBSWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "VPMADDWDrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "VPMULDQrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "VPMULHRSWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "VPMULHUWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "VPMULHWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "VPMULLDrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "VPMULLWrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "VPMULUDQrm")>;
def: InstRW<[SBWriteResGroup86], (instregex "VPSADBWrm")>;

def SBWriteResGroup87 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup87], (instregex "ADDPDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "ADDPSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "ADDSDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "ADDSSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "ADDSUBPDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "ADDSUBPSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "CMPPDrmi")>;
def: InstRW<[SBWriteResGroup87], (instregex "CMPPSrmi")>;
def: InstRW<[SBWriteResGroup87], (instregex "CMPSSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "CVTDQ2PSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "CVTPS2DQrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "CVTSI2SD64rm")>;
def: InstRW<[SBWriteResGroup87], (instregex "CVTSI2SDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "CVTTPS2DQrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MAXPDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MAXPSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MAXSDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MAXSSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MINPDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MINPSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MINSDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MINSSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MMX_CVTPI2PSirm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MMX_CVTPS2PIirm")>;
def: InstRW<[SBWriteResGroup87], (instregex "MMX_CVTTPS2PIirm")>;
def: InstRW<[SBWriteResGroup87], (instregex "POPCNT64rm")>;
def: InstRW<[SBWriteResGroup87], (instregex "ROUNDPDm")>;
def: InstRW<[SBWriteResGroup87], (instregex "ROUNDPSm")>;
def: InstRW<[SBWriteResGroup87], (instregex "ROUNDSDm")>;
def: InstRW<[SBWriteResGroup87], (instregex "ROUNDSSm")>;
def: InstRW<[SBWriteResGroup87], (instregex "SUBPDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "SUBPSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "SUBSDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "SUBSSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VADDPDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VADDPSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VADDSDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VADDSSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VADDSUBPDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VADDSUBPSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VCMPPDrmi")>;
def: InstRW<[SBWriteResGroup87], (instregex "VCMPPSrmi")>;
def: InstRW<[SBWriteResGroup87], (instregex "VCMPSDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VCMPSSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VCVTDQ2PSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VCVTPS2DQrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VCVTSI2SD64rm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VCVTSI2SDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VCVTTPS2DQrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VMAXPDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VMAXPSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VMAXSDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VMAXSSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VMINPDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VMINPSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VMINSDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VMINSSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VROUNDPDm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VROUNDPSm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VROUNDSDm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VROUNDSSm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VSUBPDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VSUBPSrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VSUBSDrm")>;
def: InstRW<[SBWriteResGroup87], (instregex "VSUBSSrm")>;

def SBWriteResGroup88 : SchedWriteRes<[SBPort23,SBPort0]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup88], (instregex "VBLENDVPDYrm")>;
def: InstRW<[SBWriteResGroup88], (instregex "VBLENDVPSYrm")>;
def: InstRW<[SBWriteResGroup88], (instregex "VMASKMOVPDrm")>;
def: InstRW<[SBWriteResGroup88], (instregex "VMASKMOVPSrm")>;

def SBWriteResGroup89 : SchedWriteRes<[SBPort0,SBPort1,SBPort5]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup89], (instregex "DPPDrri")>;
def: InstRW<[SBWriteResGroup89], (instregex "VDPPDrri")>;

def SBWriteResGroup90 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup90], (instregex "CVTSD2SI64rm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTSD2SIrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTSS2SI64rm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTSS2SIrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTTSD2SI64rm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTTSD2SIrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTTSS2SI64rm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTTSS2SIrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MUL64m")>;

def SBWriteResGroup91 : SchedWriteRes<[SBPort0,SBPort5,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup91], (instregex "VPTESTYrm")>;

def SBWriteResGroup92 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup92], (instregex "PHADDDrm")>;
def: InstRW<[SBWriteResGroup92], (instregex "PHADDSWrm128")>;
def: InstRW<[SBWriteResGroup92], (instregex "PHADDWrm")>;
def: InstRW<[SBWriteResGroup92], (instregex "PHSUBDrm")>;
def: InstRW<[SBWriteResGroup92], (instregex "PHSUBSWrm128")>;
def: InstRW<[SBWriteResGroup92], (instregex "PHSUBWrm")>;
def: InstRW<[SBWriteResGroup92], (instregex "VPHADDDrm")>;
def: InstRW<[SBWriteResGroup92], (instregex "VPHADDSWrm128")>;
def: InstRW<[SBWriteResGroup92], (instregex "VPHADDWrm")>;
def: InstRW<[SBWriteResGroup92], (instregex "VPHSUBDrm")>;
def: InstRW<[SBWriteResGroup92], (instregex "VPHSUBSWrm128")>;
def: InstRW<[SBWriteResGroup92], (instregex "VPHSUBWrm")>;
def: InstRW<[SBWriteResGroup92], (instregex "SHL64mCL")>;
def: InstRW<[SBWriteResGroup92], (instregex "SHL8mCL")>;

def SBWriteResGroup93 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,2,3];
}
def: InstRW<[SBWriteResGroup93], (instregex "ADC64mi8")>;
def: InstRW<[SBWriteResGroup93], (instregex "ADC8mi")>;
def: InstRW<[SBWriteResGroup93], (instregex "SBB64mi8")>;
def: InstRW<[SBWriteResGroup93], (instregex "SBB8mi")>;

def SBWriteResGroup94 : SchedWriteRes<[SBPort4,SBPort23,SBPort0,SBPort015]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,2,2,1];
}
def: InstRW<[SBWriteResGroup94], (instregex "ADC64mr")>;
def: InstRW<[SBWriteResGroup94], (instregex "ADC8mr")>;
def: InstRW<[SBWriteResGroup94], (instregex "SBB64mr")>;
def: InstRW<[SBWriteResGroup94], (instregex "SBB8mr")>;

def SBWriteResGroup95 : SchedWriteRes<[SBPort4,SBPort5,SBPort23,SBPort0,SBPort015]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,2,1,1];
}
def: InstRW<[SBWriteResGroup95], (instregex "BT64mr")>;
def: InstRW<[SBWriteResGroup95], (instregex "BTC64mr")>;
def: InstRW<[SBWriteResGroup95], (instregex "BTR64mr")>;
def: InstRW<[SBWriteResGroup95], (instregex "BTS64mr")>;
def: InstRW<[SBWriteResGroup95], (instregex "VADDPDYrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VADDPSYrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VADDSUBPDYrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VADDSUBPSYrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VCMPPDYrmi")>;
def: InstRW<[SBWriteResGroup95], (instregex "VCMPPSYrmi")>;
def: InstRW<[SBWriteResGroup95], (instregex "VCVTDQ2PSYrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VCVTPS2DQYrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VCVTTPS2DQrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VMAXPDYrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VMAXPSYrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VMINPDrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VMINPSrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VROUNDPDm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VROUNDPSm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VSUBPDYrm")>;
def: InstRW<[SBWriteResGroup95], (instregex "VSUBPSYrm")>;

def SBWriteResGroup96 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup96], (instregex "VCVTSD2SI64rm")>;
def: InstRW<[SBWriteResGroup96], (instregex "VCVTSD2SI64rr")>;
def: InstRW<[SBWriteResGroup96], (instregex "VCVTSS2SI64rm")>;
def: InstRW<[SBWriteResGroup96], (instregex "VCVTSS2SIrm")>;
def: InstRW<[SBWriteResGroup96], (instregex "VCVTTSD2SI64rm")>;
def: InstRW<[SBWriteResGroup96], (instregex "VCVTTSD2SI64rr")>;
def: InstRW<[SBWriteResGroup96], (instregex "VCVTTSS2SI64rm")>;
def: InstRW<[SBWriteResGroup96], (instregex "VCVTTSS2SIrm")>;

def SBWriteResGroup97 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup97], (instregex "CVTDQ2PDrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "CVTPD2DQrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "CVTPD2PSrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "CVTSD2SSrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "CVTSI2SS64rm")>;
def: InstRW<[SBWriteResGroup97], (instregex "CVTSI2SSrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "CVTTPD2DQrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "MMX_CVTPD2PIirm")>;
def: InstRW<[SBWriteResGroup97], (instregex "MMX_CVTPI2PDirm")>;
def: InstRW<[SBWriteResGroup97], (instregex "MMX_CVTTPD2PIirm")>;
def: InstRW<[SBWriteResGroup97], (instregex "VCVTDQ2PDYrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "VCVTDQ2PDrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "VCVTPD2DQrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "VCVTPD2PSrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "VCVTSD2SSrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "VCVTSI2SS64rm")>;
def: InstRW<[SBWriteResGroup97], (instregex "VCVTSI2SSrm")>;
def: InstRW<[SBWriteResGroup97], (instregex "VCVTTPD2DQrm")>;

def SBWriteResGroup98 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup98], (instregex "MULPDrm")>;
def: InstRW<[SBWriteResGroup98], (instregex "MULPSrm")>;
def: InstRW<[SBWriteResGroup98], (instregex "MULSDrm")>;
def: InstRW<[SBWriteResGroup98], (instregex "MULSSrm")>;
def: InstRW<[SBWriteResGroup98], (instregex "PCMPGTQrm")>;
def: InstRW<[SBWriteResGroup98], (instregex "PHMINPOSUWrm128")>;
def: InstRW<[SBWriteResGroup98], (instregex "RCPPSm")>;
def: InstRW<[SBWriteResGroup98], (instregex "RCPSSm")>;
def: InstRW<[SBWriteResGroup98], (instregex "RSQRTPSm")>;
def: InstRW<[SBWriteResGroup98], (instregex "RSQRTSSm")>;
def: InstRW<[SBWriteResGroup98], (instregex "VMULPDrm")>;
def: InstRW<[SBWriteResGroup98], (instregex "VMULPSrm")>;
def: InstRW<[SBWriteResGroup98], (instregex "VMULSDrm")>;
def: InstRW<[SBWriteResGroup98], (instregex "VMULSSrm")>;
def: InstRW<[SBWriteResGroup98], (instregex "VPCMPGTQrm")>;
def: InstRW<[SBWriteResGroup98], (instregex "VPHMINPOSUWrm128")>;
def: InstRW<[SBWriteResGroup98], (instregex "VRCPPSm")>;
def: InstRW<[SBWriteResGroup98], (instregex "VRCPSSm")>;
def: InstRW<[SBWriteResGroup98], (instregex "VRSQRTPSm")>;
def: InstRW<[SBWriteResGroup98], (instregex "VRSQRTSSm")>;

def SBWriteResGroup99 : SchedWriteRes<[SBPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SBWriteResGroup99], (instregex "PCMPISTRIrr")>;
def: InstRW<[SBWriteResGroup99], (instregex "PCMPISTRM128rr")>;
def: InstRW<[SBWriteResGroup99], (instregex "VPCMPISTRIrr")>;
def: InstRW<[SBWriteResGroup99], (instregex "VPCMPISTRM128rr")>;

def SBWriteResGroup100 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup100], (instregex "VCVTPD2DQYrm")>;
def: InstRW<[SBWriteResGroup100], (instregex "VCVTPD2PSYrm")>;
def: InstRW<[SBWriteResGroup100], (instregex "VCVTTPD2DQYrm")>;

def SBWriteResGroup101 : SchedWriteRes<[SBPort0,SBPort23,SBPort15]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup101], (instregex "MPSADBWrmi")>;
def: InstRW<[SBWriteResGroup101], (instregex "VMPSADBWrmi")>;

def SBWriteResGroup102 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup102], (instregex "HADDPDrm")>;
def: InstRW<[SBWriteResGroup102], (instregex "HADDPSrm")>;
def: InstRW<[SBWriteResGroup102], (instregex "HSUBPDrm")>;
def: InstRW<[SBWriteResGroup102], (instregex "HSUBPSrm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VHADDPDrm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VHADDPSrm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VHSUBPDrm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VHSUBPSrm")>;

def SBWriteResGroup103 : SchedWriteRes<[SBPort5]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup103], (instregex "AESIMCrr")>;
def: InstRW<[SBWriteResGroup103], (instregex "VAESIMCrr")>;
def: InstRW<[SBWriteResGroup103], (instregex "VMULPDYrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "VMULPSYrm")>;

def SBWriteResGroup104 : SchedWriteRes<[SBPort0,SBPort1,SBPort5]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup104], (instregex "DPPSrri")>;
def: InstRW<[SBWriteResGroup104], (instregex "VDPPSYrri")>;
def: InstRW<[SBWriteResGroup104], (instregex "VDPPSrri")>;

def SBWriteResGroup105 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup105], (instregex "VHADDPDrm")>;
def: InstRW<[SBWriteResGroup105], (instregex "VHADDPSYrm")>;
def: InstRW<[SBWriteResGroup105], (instregex "VHSUBPDYrm")>;
def: InstRW<[SBWriteResGroup105], (instregex "VHSUBPSYrm")>;

def SBWriteResGroup106 : SchedWriteRes<[SBPort5,SBPort23,SBPort015]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup106], (instregex "AESDECLASTrm")>;
def: InstRW<[SBWriteResGroup106], (instregex "AESDECrm")>;
def: InstRW<[SBWriteResGroup106], (instregex "AESENCLASTrm")>;
def: InstRW<[SBWriteResGroup106], (instregex "AESENCrm")>;
def: InstRW<[SBWriteResGroup106], (instregex "VAESDECLASTrm")>;
def: InstRW<[SBWriteResGroup106], (instregex "VAESDECrm")>;
def: InstRW<[SBWriteResGroup106], (instregex "VAESENCLASTrm")>;
def: InstRW<[SBWriteResGroup106], (instregex "VAESENCrm")>;

def SBWriteResGroup107 : SchedWriteRes<[SBPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup107], (instregex "DIVPSrr")>;
def: InstRW<[SBWriteResGroup107], (instregex "DIVSSrr")>;
def: InstRW<[SBWriteResGroup107], (instregex "SQRTPSr")>;
def: InstRW<[SBWriteResGroup107], (instregex "SQRTSSr")>;
def: InstRW<[SBWriteResGroup107], (instregex "VDIVPSrr")>;
def: InstRW<[SBWriteResGroup107], (instregex "VDIVSSrr")>;
def: InstRW<[SBWriteResGroup107], (instregex "VSQRTPSr")>;

def SBWriteResGroup108 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 14;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup108], (instregex "VSQRTSSm")>;

def SBWriteResGroup109 : SchedWriteRes<[SBPort0,SBPort23,SBPort0]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SBWriteResGroup109], (instregex "VRCPPSm")>;
def: InstRW<[SBWriteResGroup109], (instregex "VRSQRTPSYm")>;

def SBWriteResGroup110 : SchedWriteRes<[SBPort0,SBPort1,SBPort5,SBPort23]> {
  let Latency = 15;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup110], (instregex "DPPDrmi")>;
def: InstRW<[SBWriteResGroup110], (instregex "VDPPDrmi")>;

def SBWriteResGroup111 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}
def: InstRW<[SBWriteResGroup111], (instregex "PCMPISTRIrm")>;
def: InstRW<[SBWriteResGroup111], (instregex "PCMPISTRM128rm")>;
def: InstRW<[SBWriteResGroup111], (instregex "VPCMPISTRIrm")>;
def: InstRW<[SBWriteResGroup111], (instregex "VPCMPISTRM128rm")>;

def SBWriteResGroup112 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 18;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup112], (instregex "AESIMCrm")>;
def: InstRW<[SBWriteResGroup112], (instregex "VAESIMCrm")>;

def SBWriteResGroup113 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup113], (instregex "DIVPSrm")>;
def: InstRW<[SBWriteResGroup113], (instregex "DIVSSrm")>;
def: InstRW<[SBWriteResGroup113], (instregex "SQRTPSm")>;
def: InstRW<[SBWriteResGroup113], (instregex "SQRTSSm")>;
def: InstRW<[SBWriteResGroup113], (instregex "VDIVPSrm")>;
def: InstRW<[SBWriteResGroup113], (instregex "VDIVSSrm")>;
def: InstRW<[SBWriteResGroup113], (instregex "VSQRTPSm")>;

def SBWriteResGroup114 : SchedWriteRes<[SBPort0]> {
  let Latency = 21;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup114], (instregex "VSQRTSDr")>;

def SBWriteResGroup115 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup115], (instregex "VSQRTSDm")>;

def SBWriteResGroup116 : SchedWriteRes<[SBPort0]> {
  let Latency = 22;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup116], (instregex "DIVPDrr")>;
def: InstRW<[SBWriteResGroup116], (instregex "DIVSDrr")>;
def: InstRW<[SBWriteResGroup116], (instregex "SQRTPDr")>;
def: InstRW<[SBWriteResGroup116], (instregex "SQRTSDr")>;
def: InstRW<[SBWriteResGroup116], (instregex "VDIVPDrr")>;
def: InstRW<[SBWriteResGroup116], (instregex "VDIVSDrr")>;
def: InstRW<[SBWriteResGroup116], (instregex "VSQRTPDr")>;

def SBWriteResGroup117 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 28;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup117], (instregex "DIVPDrm")>;
def: InstRW<[SBWriteResGroup117], (instregex "DIVSDrm")>;
def: InstRW<[SBWriteResGroup117], (instregex "SQRTPDm")>;
def: InstRW<[SBWriteResGroup117], (instregex "SQRTSDm")>;
def: InstRW<[SBWriteResGroup117], (instregex "VDIVPDrm")>;
def: InstRW<[SBWriteResGroup117], (instregex "VDIVSDrm")>;
def: InstRW<[SBWriteResGroup117], (instregex "VSQRTPDm")>;

def SBWriteResGroup118 : SchedWriteRes<[SBPort0,SBPort0]> {
  let Latency = 29;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup118], (instregex "VDIVPSYrr")>;
def: InstRW<[SBWriteResGroup118], (instregex "VSQRTPSYr")>;

def SBWriteResGroup119 : SchedWriteRes<[SBPort0,SBPort23,SBPort0]> {
  let Latency = 36;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SBWriteResGroup119], (instregex "VDIVPSYrm")>;
def: InstRW<[SBWriteResGroup119], (instregex "VSQRTPSYm")>;

def SBWriteResGroup120 : SchedWriteRes<[SBPort0,SBPort0]> {
  let Latency = 45;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup120], (instregex "VDIVPDYrr")>;
def: InstRW<[SBWriteResGroup120], (instregex "VSQRTPDYr")>;

def SBWriteResGroup121 : SchedWriteRes<[SBPort0,SBPort23,SBPort0]> {
  let Latency = 52;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SBWriteResGroup121], (instregex "VDIVPDYrm")>;
def: InstRW<[SBWriteResGroup121], (instregex "VSQRTPDYm")>;

def SBWriteResGroup122 : SchedWriteRes<[SBPort0]> {
  let Latency = 114;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup122], (instregex "VSQRTSSr")>;

} // SchedModel
