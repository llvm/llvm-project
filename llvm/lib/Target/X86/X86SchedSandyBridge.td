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
defm : SBWriteResPair<WriteFDiv,   SBPort0, 24>;
defm : SBWriteResPair<WriteFRcp,   SBPort0, 5>;
defm : SBWriteResPair<WriteFRsqrt, SBPort0, 5>;
defm : SBWriteResPair<WriteFSqrt,  SBPort0, 14>;
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
defm : SBWriteResPair<WriteVecShift, SBPort5,  1>;
defm : SBWriteResPair<WriteVecLogic, SBPort5, 1>;
defm : SBWriteResPair<WriteVecALU,   SBPort1,  3>;
defm : SBWriteResPair<WriteVecIMul,  SBPort0,   5>;
defm : SBWriteResPair<WriteShuffle,  SBPort5,  1>;
defm : SBWriteResPair<WriteBlend,  SBPort15,  1>;
def : WriteRes<WriteVarBlend, [SBPort1, SBPort5]> {
  let Latency = 2;
  let ResourceCycles = [1, 1];
}
def : WriteRes<WriteVarBlendLd, [SBPort1, SBPort5, SBPort23]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 1];
}
def : WriteRes<WriteMPSAD, [SBPort0,SBPort15]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def : WriteRes<WriteMPSADLd, [SBPort0,SBPort23,SBPort15]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}

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
def : WriteRes<WritePCmpIStrI, [SBPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [SBPort0,SBPort23]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
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
def : WriteRes<WriteAESDecEnc, [SBPort5,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def : WriteRes<WriteAESDecEncLd, [SBPort5,SBPort23,SBPort015]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}

def : WriteRes<WriteAESIMC, [SBPort5]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [SBPort5,SBPort23]> {
  let Latency = 18;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
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

// AVX2/FMA is not supported on that architecture, but we should define the basic
// scheduling resources anyway.
defm : SBWriteResPair<WriteFShuffle256, SBPort0,  1>;
defm : SBWriteResPair<WriteShuffle256, SBPort0,  1>;
defm : SBWriteResPair<WriteVarVecShift, SBPort0,  1>;
defm : SBWriteResPair<WriteFMA, SBPort01,  5>;

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

def SBWriteResGroup1 : SchedWriteRes<[SBPort1]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup1], (instregex "COMP_FST0r")>;
def: InstRW<[SBWriteResGroup1], (instregex "COM_FST0r")>;
def: InstRW<[SBWriteResGroup1], (instregex "UCOM_FPr")>;
def: InstRW<[SBWriteResGroup1], (instregex "UCOM_Fr")>;

def SBWriteResGroup2 : SchedWriteRes<[SBPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup2], (instregex "ANDNPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "ANDNPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "ANDPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "ANDPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "FDECSTP")>;
def: InstRW<[SBWriteResGroup2], (instregex "FFREE")>;
def: InstRW<[SBWriteResGroup2], (instregex "FINCSTP")>;
def: InstRW<[SBWriteResGroup2], (instregex "FNOP")>;
def: InstRW<[SBWriteResGroup2], (instregex "INSERTPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "JAE_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JAE_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JA_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JA_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JBE_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JBE_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JB_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JB_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JE_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JE_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JGE_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JGE_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JG_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JG_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JLE_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JLE_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JL_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JL_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JMP64r")>;
def: InstRW<[SBWriteResGroup2], (instregex "JMP_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JMP_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JNE_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JNE_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JNO_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JNO_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JNP_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JNP_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JNS_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JNS_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JO_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JO_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JP_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JP_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "JS_1")>;
def: InstRW<[SBWriteResGroup2], (instregex "JS_4")>;
def: InstRW<[SBWriteResGroup2], (instregex "LD_Frr")>;
def: InstRW<[SBWriteResGroup2], (instregex "LOOP")>;
def: InstRW<[SBWriteResGroup2], (instregex "LOOPE")>;
def: InstRW<[SBWriteResGroup2], (instregex "LOOPNE")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOV64toPQIrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVAPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVAPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVDDUPrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVDI2PDIrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVHLPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVLHPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVSDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVSHDUPrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVSLDUPrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVSSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVUPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "MOVUPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "ORPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "ORPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "RETQ")>;
def: InstRW<[SBWriteResGroup2], (instregex "SHUFPDrri")>;
def: InstRW<[SBWriteResGroup2], (instregex "SHUFPSrri")>;
def: InstRW<[SBWriteResGroup2], (instregex "ST_FPrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "ST_Frr")>;
def: InstRW<[SBWriteResGroup2], (instregex "UNPCKHPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "UNPCKHPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "UNPCKLPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "UNPCKLPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VANDNPDYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VANDNPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VANDNPSYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VANDNPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VANDPDYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VANDPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VANDPSYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VANDPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VEXTRACTF128rr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VINSERTF128rr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VINSERTPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOV64toPQIrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVAPDYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVAPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVAPSYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVAPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVDDUPYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVDDUPrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVDI2PDIrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVHLPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVHLPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVSDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVSHDUPYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVSHDUPrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVSLDUPYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVSLDUPrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVSSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVUPDYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVUPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVUPSYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VMOVUPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VORPDYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VORPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VORPSYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VORPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VPERM2F128rr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VPERMILPDYri")>;
def: InstRW<[SBWriteResGroup2], (instregex "VPERMILPDYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VPERMILPDri")>;
def: InstRW<[SBWriteResGroup2], (instregex "VPERMILPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VPERMILPSYri")>;
def: InstRW<[SBWriteResGroup2], (instregex "VPERMILPSYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VPERMILPSri")>;
def: InstRW<[SBWriteResGroup2], (instregex "VPERMILPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VSHUFPDYrri")>;
def: InstRW<[SBWriteResGroup2], (instregex "VSHUFPDrri")>;
def: InstRW<[SBWriteResGroup2], (instregex "VSHUFPSYrri")>;
def: InstRW<[SBWriteResGroup2], (instregex "VSHUFPSrri")>;
def: InstRW<[SBWriteResGroup2], (instregex "VUNPCKHPDYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VUNPCKHPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VUNPCKHPSYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VUNPCKHPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VUNPCKLPDYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VUNPCKLPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VUNPCKLPSYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VUNPCKLPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VXORPDYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VXORPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VXORPSYrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "VXORPSrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "XORPDrr")>;
def: InstRW<[SBWriteResGroup2], (instregex "XORPSrr")>;

def SBWriteResGroup3 : SchedWriteRes<[SBPort01]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup3], (instregex "LEA(16|32|64)r")>;

def SBWriteResGroup4 : SchedWriteRes<[SBPort05]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup4], (instregex "BLENDPDrri")>;
def: InstRW<[SBWriteResGroup4], (instregex "BLENDPSrri")>;
def: InstRW<[SBWriteResGroup4], (instregex "BT(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup4], (instregex "BT(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup4], (instregex "BTC(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup4], (instregex "BTC(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup4], (instregex "BTR(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup4], (instregex "BTR(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup4], (instregex "BTS(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup4], (instregex "BTS(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup4], (instregex "CDQ")>;
def: InstRW<[SBWriteResGroup4], (instregex "CQO")>;
def: InstRW<[SBWriteResGroup4], (instregex "LAHF")>;
def: InstRW<[SBWriteResGroup4], (instregex "SAHF")>;
def: InstRW<[SBWriteResGroup4], (instregex "SAR(16|32|64)ri")>;
def: InstRW<[SBWriteResGroup4], (instregex "SAR8ri")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETAEr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETBr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETEr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETGEr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETGr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETLEr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETLr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETNEr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETNOr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETNPr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETNSr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETOr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETPr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SETSr")>;
def: InstRW<[SBWriteResGroup4], (instregex "SHL(16|32|64)ri")>;
def: InstRW<[SBWriteResGroup4], (instregex "SHL(16|32|64)r1")>;
def: InstRW<[SBWriteResGroup4], (instregex "SHL8r1")>;
def: InstRW<[SBWriteResGroup4], (instregex "SHL8ri")>;
def: InstRW<[SBWriteResGroup4], (instregex "SHR(16|32|64)ri")>;
def: InstRW<[SBWriteResGroup4], (instregex "SHR8ri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VBLENDPDYrri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VBLENDPDrri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VBLENDPSYrri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VBLENDPSrri")>;
def: InstRW<[SBWriteResGroup4], (instregex "VMOVDQAYrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VMOVDQArr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VMOVDQUYrr")>;
def: InstRW<[SBWriteResGroup4], (instregex "VMOVDQUrr")>;

def SBWriteResGroup5 : SchedWriteRes<[SBPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup5], (instregex "MMX_PABSBrr64")>;
def: InstRW<[SBWriteResGroup5], (instregex "MMX_PABSDrr64")>;
def: InstRW<[SBWriteResGroup5], (instregex "MMX_PABSWrr64")>;
def: InstRW<[SBWriteResGroup5], (instregex "MMX_PADDQirr")>;
def: InstRW<[SBWriteResGroup5], (instregex "MMX_PALIGNR64irr")>;
def: InstRW<[SBWriteResGroup5], (instregex "MMX_PSHUFBrr64")>;
def: InstRW<[SBWriteResGroup5], (instregex "MMX_PSIGNBrr64")>;
def: InstRW<[SBWriteResGroup5], (instregex "MMX_PSIGNDrr64")>;
def: InstRW<[SBWriteResGroup5], (instregex "MMX_PSIGNWrr64")>;
def: InstRW<[SBWriteResGroup5], (instregex "PABSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PABSDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PABSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PACKSSDWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PACKSSWBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PACKUSDWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PACKUSWBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PADDBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PADDDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PADDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PADDSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PADDSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PADDUSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PADDUSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PADDWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PALIGNRrri")>;
def: InstRW<[SBWriteResGroup5], (instregex "PAVGBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PAVGWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PBLENDWrri")>;
def: InstRW<[SBWriteResGroup5], (instregex "PCMPEQBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PCMPEQDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PCMPEQQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PCMPEQWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PCMPGTBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PCMPGTDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PCMPGTWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMAXSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMAXSDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMAXSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMAXUBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMAXUDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMAXUWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMINSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMINSDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMINSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMINUBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMINUDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMINUWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVSXBDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVSXBQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVSXBWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVSXDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVSXWDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVSXWQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVZXBDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVZXBQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVZXBWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVZXDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVZXWDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PMOVZXWQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSHUFBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSHUFDri")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSHUFHWri")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSHUFLWri")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSIGNBrr128")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSIGNDrr128")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSIGNWrr128")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSLLDQri")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSRLDQri")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSUBBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSUBDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSUBQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSUBSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSUBSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSUBUSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSUBUSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PSUBWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PUNPCKHBWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PUNPCKHDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PUNPCKHQDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PUNPCKHWDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PUNPCKLBWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PUNPCKLDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PUNPCKLQDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "PUNPCKLWDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPABSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPABSDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPABSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPACKSSDWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPACKSSWBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPACKUSDWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPACKUSWBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPADDBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPADDDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPADDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPADDSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPADDSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPADDUSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPADDUSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPADDWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPALIGNRrri")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPAVGBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPAVGWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPBLENDWrri")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPCMPEQBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPCMPEQDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPCMPEQQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPCMPEQWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPCMPGTBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPCMPGTDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPCMPGTWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMAXSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMAXSDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMAXSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMAXUBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMAXUDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMAXUWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMINSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMINSDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMINSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMINUBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMINUDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMINUWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVSXBDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVSXBQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVSXBWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVSXDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVSXWDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVSXWQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVZXBDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVZXBQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVZXBWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVZXDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVZXWDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPMOVZXWQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSHUFBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSHUFDri")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSHUFHWri")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSHUFLWri")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSIGNBrr128")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSIGNDrr128")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSIGNWrr128")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSLLDQri")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSRLDQri")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSUBBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSUBDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSUBQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSUBSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSUBSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSUBUSBrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSUBUSWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPSUBWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPUNPCKHBWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPUNPCKHDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPUNPCKHQDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPUNPCKHWDrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPUNPCKLBWrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPUNPCKLDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPUNPCKLQDQrr")>;
def: InstRW<[SBWriteResGroup5], (instregex "VPUNPCKLWDrr")>;

def SBWriteResGroup6 : SchedWriteRes<[SBPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup6], (instregex "ADD(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup6], (instregex "ADD(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "ADD8i8")>;
def: InstRW<[SBWriteResGroup6], (instregex "ADD8ri")>;
def: InstRW<[SBWriteResGroup6], (instregex "ADD8rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "AND(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup6], (instregex "AND(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "AND8i8")>;
def: InstRW<[SBWriteResGroup6], (instregex "AND8ri")>;
def: InstRW<[SBWriteResGroup6], (instregex "AND8rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "CBW")>;
def: InstRW<[SBWriteResGroup6], (instregex "CMC")>;
def: InstRW<[SBWriteResGroup6], (instregex "CMP(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup6], (instregex "CMP(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "CMP8i8")>;
def: InstRW<[SBWriteResGroup6], (instregex "CMP8ri")>;
def: InstRW<[SBWriteResGroup6], (instregex "CMP8rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "CWDE")>;
def: InstRW<[SBWriteResGroup6], (instregex "DEC(16|32|64)r")>;
def: InstRW<[SBWriteResGroup6], (instregex "DEC8r")>;
def: InstRW<[SBWriteResGroup6], (instregex "INC(16|32|64)r")>;
def: InstRW<[SBWriteResGroup6], (instregex "INC8r")>;
def: InstRW<[SBWriteResGroup6], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "MMX_MOVQ2DQrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOV(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOV8ri")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOV8rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVDQArr")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVDQUrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVPQI2QIrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVSX(16|32|64)rr16")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVSX(16|32|64)rr32")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVSX(16|32|64)rr8")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVZX(16|32|64)rr16")>;
def: InstRW<[SBWriteResGroup6], (instregex "MOVZX(16|32|64)rr8")>;
def: InstRW<[SBWriteResGroup6], (instregex "NEG(16|32|64)r")>;
def: InstRW<[SBWriteResGroup6], (instregex "NEG8r")>;
def: InstRW<[SBWriteResGroup6], (instregex "NOT(16|32|64)r")>;
def: InstRW<[SBWriteResGroup6], (instregex "NOT8r")>;
def: InstRW<[SBWriteResGroup6], (instregex "OR(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup6], (instregex "OR(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "OR8i8")>;
def: InstRW<[SBWriteResGroup6], (instregex "OR8ri")>;
def: InstRW<[SBWriteResGroup6], (instregex "OR8rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "PANDNrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "PANDrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "PORrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "PXORrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "STC")>;
def: InstRW<[SBWriteResGroup6], (instregex "SUB(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup6], (instregex "SUB(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "SUB8i8")>;
def: InstRW<[SBWriteResGroup6], (instregex "SUB8ri")>;
def: InstRW<[SBWriteResGroup6], (instregex "SUB8rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "TEST(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "TEST8i8")>;
def: InstRW<[SBWriteResGroup6], (instregex "TEST8ri")>;
def: InstRW<[SBWriteResGroup6], (instregex "TEST8rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VMOVPQI2QIrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VMOVZPQILo2PQIrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VPANDNrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VPANDrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VPORrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "VPXORrr")>;
def: InstRW<[SBWriteResGroup6], (instregex "XOR(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup6], (instregex "XOR(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup6], (instregex "XOR8i8")>;
def: InstRW<[SBWriteResGroup6], (instregex "XOR8ri")>;
def: InstRW<[SBWriteResGroup6], (instregex "XOR8rr")>;

def SBWriteResGroup7 : SchedWriteRes<[SBPort0]> {
  let Latency = 2;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup7], (instregex "MOVMSKPDrr")>;
def: InstRW<[SBWriteResGroup7], (instregex "MOVMSKPSrr")>;
def: InstRW<[SBWriteResGroup7], (instregex "MOVPDI2DIrr")>;
def: InstRW<[SBWriteResGroup7], (instregex "MOVPQIto64rr")>;
def: InstRW<[SBWriteResGroup7], (instregex "PMOVMSKBrr")>;
def: InstRW<[SBWriteResGroup7], (instregex "VMOVMSKPDYrr")>;
def: InstRW<[SBWriteResGroup7], (instregex "VMOVMSKPDrr")>;
def: InstRW<[SBWriteResGroup7], (instregex "VMOVMSKPSYrr")>;
def: InstRW<[SBWriteResGroup7], (instregex "VMOVMSKPSrr")>;
def: InstRW<[SBWriteResGroup7], (instregex "VMOVPDI2DIrr")>;
def: InstRW<[SBWriteResGroup7], (instregex "VMOVPQIto64rr")>;

def SBWriteResGroup9 : SchedWriteRes<[SBPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup9], (instregex "BLENDVPDrr0")>;
def: InstRW<[SBWriteResGroup9], (instregex "BLENDVPSrr0")>;
def: InstRW<[SBWriteResGroup9], (instregex "ROL(16|32|64)ri")>;
def: InstRW<[SBWriteResGroup9], (instregex "ROL8ri")>;
def: InstRW<[SBWriteResGroup9], (instregex "ROR(16|32|64)ri")>;
def: InstRW<[SBWriteResGroup9], (instregex "ROR8ri")>;
def: InstRW<[SBWriteResGroup9], (instregex "SETAr")>;
def: InstRW<[SBWriteResGroup9], (instregex "SETBEr")>;
def: InstRW<[SBWriteResGroup9], (instregex "VBLENDVPDYrr")>;
def: InstRW<[SBWriteResGroup9], (instregex "VBLENDVPDrr")>;
def: InstRW<[SBWriteResGroup9], (instregex "VBLENDVPSYrr")>;
def: InstRW<[SBWriteResGroup9], (instregex "VBLENDVPSrr")>;

def SBWriteResGroup10 : SchedWriteRes<[SBPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup10], (instregex "VPBLENDVBrr")>;

def SBWriteResGroup11 : SchedWriteRes<[SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup11], (instregex "SCASB")>;
def: InstRW<[SBWriteResGroup11], (instregex "SCASL")>;
def: InstRW<[SBWriteResGroup11], (instregex "SCASQ")>;
def: InstRW<[SBWriteResGroup11], (instregex "SCASW")>;

def SBWriteResGroup12 : SchedWriteRes<[SBPort0,SBPort1]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup12], (instregex "COMISDrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "COMISSrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "UCOMISDrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "UCOMISSrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "VCOMISDrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "VCOMISSrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "VUCOMISDrr")>;
def: InstRW<[SBWriteResGroup12], (instregex "VUCOMISSrr")>;

def SBWriteResGroup13 : SchedWriteRes<[SBPort0,SBPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup13], (instregex "CVTPS2PDrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "PTESTrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "VCVTPS2PDYrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "VCVTPS2PDrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "VPTESTYrr")>;
def: InstRW<[SBWriteResGroup13], (instregex "VPTESTrr")>;

def SBWriteResGroup14 : SchedWriteRes<[SBPort0,SBPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup14], (instregex "PSLLDrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "PSLLQrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "PSLLWrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "PSRADrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "PSRAWrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "PSRLDrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "PSRLQrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "PSRLWrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "VPSLLDrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "VPSLLQrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "VPSLLWrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "VPSRADrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "VPSRAWrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "VPSRLDrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "VPSRLQrr")>;
def: InstRW<[SBWriteResGroup14], (instregex "VPSRLWrr")>;

def SBWriteResGroup15 : SchedWriteRes<[SBPort0,SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup15], (instregex "CWD")>;
def: InstRW<[SBWriteResGroup15], (instregex "FNSTSW16r")>;

def SBWriteResGroup16 : SchedWriteRes<[SBPort1,SBPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup16], (instregex "BSWAP(16|32|64)r")>;

def SBWriteResGroup17 : SchedWriteRes<[SBPort5,SBPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup17], (instregex "PINSRBrr")>;
def: InstRW<[SBWriteResGroup17], (instregex "PINSRDrr")>;
def: InstRW<[SBWriteResGroup17], (instregex "PINSRQrr")>;
def: InstRW<[SBWriteResGroup17], (instregex "PINSRWrri")>;
def: InstRW<[SBWriteResGroup17], (instregex "VPINSRBrr")>;
def: InstRW<[SBWriteResGroup17], (instregex "VPINSRDrr")>;
def: InstRW<[SBWriteResGroup17], (instregex "VPINSRQrr")>;
def: InstRW<[SBWriteResGroup17], (instregex "VPINSRWrri")>;

def SBWriteResGroup18 : SchedWriteRes<[SBPort5,SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup18], (instregex "JRCXZ")>;
def: InstRW<[SBWriteResGroup18], (instregex "MMX_MOVDQ2Qrr")>;

def SBWriteResGroup19 : SchedWriteRes<[SBPort05,SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup19], (instregex "ADC(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup19], (instregex "ADC(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "ADC8ri")>;
def: InstRW<[SBWriteResGroup19], (instregex "ADC8rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVAE(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVB(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVE(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVG(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVGE(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVL(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVLE(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVNE(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVNO(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVNP(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVNS(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVO(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVP(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "CMOVS(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "SBB(16|32|64)ri8")>;
def: InstRW<[SBWriteResGroup19], (instregex "SBB(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "SBB8ri")>;
def: InstRW<[SBWriteResGroup19], (instregex "SBB8rr")>;
def: InstRW<[SBWriteResGroup19], (instregex "SHLD(16|32|64)rri8")>;
def: InstRW<[SBWriteResGroup19], (instregex "SHRD(16|32|64)rri8")>;

def SBWriteResGroup20 : SchedWriteRes<[SBPort0]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup20], (instregex "MMX_PMADDUBSWrr64")>;
def: InstRW<[SBWriteResGroup20], (instregex "MMX_PMULHRSWrr64")>;
def: InstRW<[SBWriteResGroup20], (instregex "MMX_PMULUDQirr")>;
def: InstRW<[SBWriteResGroup20], (instregex "PMADDUBSWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "PMADDWDrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "PMULDQrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "PMULHRSWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "PMULHUWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "PMULHWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "PMULLDrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "PMULLWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "PMULUDQrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "PSADBWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VPMADDUBSWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VPMADDWDrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VPMULDQrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VPMULHRSWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VPMULHUWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VPMULHWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VPMULLDrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VPMULLWrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VPMULUDQrr")>;
def: InstRW<[SBWriteResGroup20], (instregex "VPSADBWrr")>;

def SBWriteResGroup21 : SchedWriteRes<[SBPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup21], (instregex "ADDPDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "ADDPSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "ADDSDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "ADDSSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "ADDSUBPDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "ADDSUBPSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "ADD_FPrST0")>;
def: InstRW<[SBWriteResGroup21], (instregex "ADD_FST0r")>;
def: InstRW<[SBWriteResGroup21], (instregex "ADD_FrST0")>;
def: InstRW<[SBWriteResGroup21], (instregex "BSF(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup21], (instregex "BSR(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup21], (instregex "CMPPDrri")>;
def: InstRW<[SBWriteResGroup21], (instregex "CMPPSrri")>;
def: InstRW<[SBWriteResGroup21], (instregex "CMPSSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "CRC32r(16|32|64)r8")>;
def: InstRW<[SBWriteResGroup21], (instregex "CRC32r(16|32|64)r64")>;
def: InstRW<[SBWriteResGroup21], (instregex "CVTDQ2PSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "CVTPS2DQrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "CVTTPS2DQrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MAXPDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MAXPSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MAXSDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MAXSSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MINPDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MINPSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MINSDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MINSSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MMX_CVTPI2PSirr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MMX_CVTPS2PIirr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MMX_CVTTPS2PIirr")>;
def: InstRW<[SBWriteResGroup21], (instregex "MUL8r")>;
def: InstRW<[SBWriteResGroup21], (instregex "POPCNT(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup21], (instregex "PUSHFS64")>;
def: InstRW<[SBWriteResGroup21], (instregex "ROUNDPDr")>;
def: InstRW<[SBWriteResGroup21], (instregex "ROUNDPSr")>;
def: InstRW<[SBWriteResGroup21], (instregex "ROUNDSDr")>;
def: InstRW<[SBWriteResGroup21], (instregex "ROUNDSSr")>;
def: InstRW<[SBWriteResGroup21], (instregex "SUBPDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "SUBPSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "SUBR_FPrST0")>;
def: InstRW<[SBWriteResGroup21], (instregex "SUBR_FST0r")>;
def: InstRW<[SBWriteResGroup21], (instregex "SUBR_FrST0")>;
def: InstRW<[SBWriteResGroup21], (instregex "SUBSDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "SUBSSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "SUB_FPrST0")>;
def: InstRW<[SBWriteResGroup21], (instregex "SUB_FST0r")>;
def: InstRW<[SBWriteResGroup21], (instregex "SUB_FrST0")>;
def: InstRW<[SBWriteResGroup21], (instregex "VADDPDYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VADDPDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VADDPSYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VADDPSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VADDSDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VADDSSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VADDSUBPDYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VADDSUBPDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VADDSUBPSYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VADDSUBPSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCMPPDYrri")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCMPPDrri")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCMPPSYrri")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCMPPSrri")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCMPSDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCMPSSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCVTDQ2PSYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCVTDQ2PSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCVTPS2DQYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCVTPS2DQrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCVTTPS2DQYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VCVTTPS2DQrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMAXPDYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMAXPDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMAXPSYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMAXPSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMAXSDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMAXSSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMINPDYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMINPDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMINPSYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMINPSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMINSDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VMINSSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VROUNDPDr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VROUNDPSr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VROUNDSDr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VROUNDSSr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VROUNDYPDr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VROUNDYPSr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VSUBPDYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VSUBPDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VSUBPSYrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VSUBPSrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VSUBSDrr")>;
def: InstRW<[SBWriteResGroup21], (instregex "VSUBSSrr")>;

def SBWriteResGroup22 : SchedWriteRes<[SBPort0,SBPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup22], (instregex "EXTRACTPSrr")>;
def: InstRW<[SBWriteResGroup22], (instregex "VEXTRACTPSrr")>;

def SBWriteResGroup23 : SchedWriteRes<[SBPort0,SBPort15]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup23], (instregex "PEXTRBrr")>;
def: InstRW<[SBWriteResGroup23], (instregex "PEXTRDrr")>;
def: InstRW<[SBWriteResGroup23], (instregex "PEXTRQrr")>;
def: InstRW<[SBWriteResGroup23], (instregex "PEXTRWri")>;
def: InstRW<[SBWriteResGroup23], (instregex "VPEXTRBrr")>;
def: InstRW<[SBWriteResGroup23], (instregex "VPEXTRDrr")>;
def: InstRW<[SBWriteResGroup23], (instregex "VPEXTRQrr")>;
def: InstRW<[SBWriteResGroup23], (instregex "VPEXTRWri")>;

def SBWriteResGroup23_2 : SchedWriteRes<[SBPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SBWriteResGroup23_2], (instregex "ROL(16|32|64)rCL")>;
def: InstRW<[SBWriteResGroup23_2], (instregex "ROL8rCL")>;
def: InstRW<[SBWriteResGroup23_2], (instregex "ROR(16|32|64)rCL")>;
def: InstRW<[SBWriteResGroup23_2], (instregex "ROR8rCL")>;
def: InstRW<[SBWriteResGroup23_2], (instregex "SAR(16|32|64)rCL")>;
def: InstRW<[SBWriteResGroup23_2], (instregex "SAR8rCL")>;
def: InstRW<[SBWriteResGroup23_2], (instregex "SHL(16|32|64)rCL")>;
def: InstRW<[SBWriteResGroup23_2], (instregex "SHL8rCL")>;
def: InstRW<[SBWriteResGroup23_2], (instregex "SHR(16|32|64)rCL")>;
def: InstRW<[SBWriteResGroup23_2], (instregex "SHR8rCL")>;

def SBWriteResGroup24 : SchedWriteRes<[SBPort15]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SBWriteResGroup24], (instregex "MMX_PHADDSWrr64")>;
def: InstRW<[SBWriteResGroup24], (instregex "MMX_PHADDWrr64")>;
def: InstRW<[SBWriteResGroup24], (instregex "MMX_PHADDrr64")>;
def: InstRW<[SBWriteResGroup24], (instregex "MMX_PHSUBDrr64")>;
def: InstRW<[SBWriteResGroup24], (instregex "MMX_PHSUBSWrr64")>;
def: InstRW<[SBWriteResGroup24], (instregex "MMX_PHSUBWrr64")>;
def: InstRW<[SBWriteResGroup24], (instregex "PHADDDrr")>;
def: InstRW<[SBWriteResGroup24], (instregex "PHADDSWrr128")>;
def: InstRW<[SBWriteResGroup24], (instregex "PHADDWrr")>;
def: InstRW<[SBWriteResGroup24], (instregex "PHSUBDrr")>;
def: InstRW<[SBWriteResGroup24], (instregex "PHSUBSWrr128")>;
def: InstRW<[SBWriteResGroup24], (instregex "PHSUBWrr")>;
def: InstRW<[SBWriteResGroup24], (instregex "VPHADDDrr")>;
def: InstRW<[SBWriteResGroup24], (instregex "VPHADDSWrr128")>;
def: InstRW<[SBWriteResGroup24], (instregex "VPHADDWrr")>;
def: InstRW<[SBWriteResGroup24], (instregex "VPHSUBDrr")>;
def: InstRW<[SBWriteResGroup24], (instregex "VPHSUBSWrr128")>;
def: InstRW<[SBWriteResGroup24], (instregex "VPHSUBWrr")>;

def SBWriteResGroup25 : SchedWriteRes<[SBPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SBWriteResGroup25], (instregex "ADC8i8")>;
def: InstRW<[SBWriteResGroup25], (instregex "LEAVE64")>;
def: InstRW<[SBWriteResGroup25], (instregex "OUT32rr")>;
def: InstRW<[SBWriteResGroup25], (instregex "OUT8rr")>;
def: InstRW<[SBWriteResGroup25], (instregex "SBB8i8")>;
def: InstRW<[SBWriteResGroup25], (instregex "XADD(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup25], (instregex "XADD8rr")>;

def SBWriteResGroup25_2 : SchedWriteRes<[SBPort5,SBPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup25_2], (instregex "CMOVBE_F")>;
def: InstRW<[SBWriteResGroup25_2], (instregex "CMOVB_F")>;
def: InstRW<[SBWriteResGroup25_2], (instregex "CMOVE_F")>;
def: InstRW<[SBWriteResGroup25_2], (instregex "CMOVNBE_F")>;
def: InstRW<[SBWriteResGroup25_2], (instregex "CMOVNB_F")>;
def: InstRW<[SBWriteResGroup25_2], (instregex "CMOVNE_F")>;
def: InstRW<[SBWriteResGroup25_2], (instregex "CMOVNP_F")>;
def: InstRW<[SBWriteResGroup25_2], (instregex "CMOVP_F")>;

def SBWriteResGroup26 : SchedWriteRes<[SBPort05,SBPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup26], (instregex "CMOVA(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup26], (instregex "CMOVBE(16|32|64)rr")>;

def SBWriteResGroup26_2 : SchedWriteRes<[SBPort0,SBPort1,SBPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup26_2], (instregex "COM_FIPr")>;
def: InstRW<[SBWriteResGroup26_2], (instregex "COM_FIr")>;
def: InstRW<[SBWriteResGroup26_2], (instregex "UCOM_FIPr")>;
def: InstRW<[SBWriteResGroup26_2], (instregex "UCOM_FIr")>;

def SBWriteResGroup27 : SchedWriteRes<[SBPort0,SBPort1]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup27], (instregex "MUL(16|32|64)r")>;

def SBWriteResGroup28 : SchedWriteRes<[SBPort1,SBPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup28], (instregex "CVTDQ2PDrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "CVTPD2DQrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "CVTPD2PSrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "CVTSD2SSrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "CVTSI2SD64rr")>;
def: InstRW<[SBWriteResGroup28], (instregex "CVTSI2SDrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "CVTTPD2DQrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "MMX_CVTPD2PIirr")>;
def: InstRW<[SBWriteResGroup28], (instregex "MMX_CVTPI2PDirr")>;
def: InstRW<[SBWriteResGroup28], (instregex "MMX_CVTTPD2PIirr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTDQ2PDYrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTDQ2PDrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTPD2DQYrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTPD2DQrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTPD2PSYrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTPD2PSrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTSD2SSrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTSI2SD64rr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTSI2SDrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTTPD2DQYrr")>;
def: InstRW<[SBWriteResGroup28], (instregex "VCVTTPD2DQrr")>;

def SBWriteResGroup29 : SchedWriteRes<[SBPort1,SBPort015]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup29], (instregex "MOV64sr")>;

def SBWriteResGroup29_2 : SchedWriteRes<[SBPort5,SBPort015]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup29_2], (instregex "OUT32ir")>;
def: InstRW<[SBWriteResGroup29_2], (instregex "OUT8ir")>;
def: InstRW<[SBWriteResGroup29_2], (instregex "PAUSE")>;

def SBWriteResGroup29_3 : SchedWriteRes<[SBPort05,SBPort015]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}
def: InstRW<[SBWriteResGroup29_3], (instregex "SHLD(16|32|64)rrCL")>;
def: InstRW<[SBWriteResGroup29_3], (instregex "SHRD(16|32|64)rrCL")>;

def SBWriteResGroup30 : SchedWriteRes<[SBPort0]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup30], (instregex "MULPDrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "MULPSrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "MULSDrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "MULSSrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "MUL_FPrST0")>;
def: InstRW<[SBWriteResGroup30], (instregex "MUL_FST0r")>;
def: InstRW<[SBWriteResGroup30], (instregex "MUL_FrST0")>;
def: InstRW<[SBWriteResGroup30], (instregex "PCMPGTQrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "PHMINPOSUWrr128")>;
def: InstRW<[SBWriteResGroup30], (instregex "RCPPSr")>;
def: InstRW<[SBWriteResGroup30], (instregex "RCPSSr")>;
def: InstRW<[SBWriteResGroup30], (instregex "RSQRTPSr")>;
def: InstRW<[SBWriteResGroup30], (instregex "RSQRTSSr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VMULPDYrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VMULPDrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VMULPSYrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VMULPSrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VMULSDrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VMULSSrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VPCMPGTQrr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VPHMINPOSUWrr128")>;
def: InstRW<[SBWriteResGroup30], (instregex "VRCPPSr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VRCPSSr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VRSQRTPSr")>;
def: InstRW<[SBWriteResGroup30], (instregex "VRSQRTSSr")>;

def SBWriteResGroup31 : SchedWriteRes<[SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup31], (instregex "MOV(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOV8rm")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVSX(16|32|64)rm16")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVSX(16|32|64)rm32")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVSX(16|32|64)rm8")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVZX(16|32|64)rm16")>;
def: InstRW<[SBWriteResGroup31], (instregex "MOVZX(16|32|64)rm8")>;
def: InstRW<[SBWriteResGroup31], (instregex "PREFETCH")>;

def SBWriteResGroup32 : SchedWriteRes<[SBPort0,SBPort1]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup32], (instregex "CVTSD2SI64rr")>;
def: InstRW<[SBWriteResGroup32], (instregex "CVTSD2SIrr")>;
def: InstRW<[SBWriteResGroup32], (instregex "CVTSS2SI64rr")>;
def: InstRW<[SBWriteResGroup32], (instregex "CVTSS2SIrr")>;
def: InstRW<[SBWriteResGroup32], (instregex "CVTTSD2SI64rr")>;
def: InstRW<[SBWriteResGroup32], (instregex "CVTTSD2SIrr")>;
def: InstRW<[SBWriteResGroup32], (instregex "CVTTSS2SI64rr")>;
def: InstRW<[SBWriteResGroup32], (instregex "CVTTSS2SIrr")>;
def: InstRW<[SBWriteResGroup32], (instregex "VCVTSD2SI64rr")>;
def: InstRW<[SBWriteResGroup32], (instregex "VCVTSD2SIrr")>;
def: InstRW<[SBWriteResGroup32], (instregex "VCVTSS2SI64rr")>;
def: InstRW<[SBWriteResGroup32], (instregex "VCVTSS2SIrr")>;
def: InstRW<[SBWriteResGroup32], (instregex "VCVTTSD2SI64rr")>;
def: InstRW<[SBWriteResGroup32], (instregex "VCVTTSD2SIrr")>;
def: InstRW<[SBWriteResGroup32], (instregex "VCVTTSS2SI64rr")>;
def: InstRW<[SBWriteResGroup32], (instregex "VCVTTSS2SIrr")>;

def SBWriteResGroup33 : SchedWriteRes<[SBPort4,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup33], (instregex "MOV(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOV8mr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVAPDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVAPSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVDQAmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVDQUmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVHPDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVHPSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVLPDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVLPSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVNTDQmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVNTI_64mr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVNTImr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVNTPDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVNTPSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVPDI2DImr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVPQI2QImr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVPQIto64mr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVSSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVUPDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "MOVUPSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "PUSH64i8")>;
def: InstRW<[SBWriteResGroup33], (instregex "PUSH(16|32|64)r")>;
def: InstRW<[SBWriteResGroup33], (instregex "VEXTRACTF128mr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVAPDYmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVAPDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVAPSYmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVAPSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVDQAYmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVDQAmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVDQUYmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVDQUmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVHPDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVHPSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVLPDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVLPSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVNTDQYmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVNTDQmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVNTPDYmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVNTPDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVNTPSYmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVNTPSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVPDI2DImr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVPQI2QImr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVPQIto64mr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVSDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVSSmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVUPDYmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVUPDmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVUPSYmr")>;
def: InstRW<[SBWriteResGroup33], (instregex "VMOVUPSmr")>;

def SBWriteResGroup34 : SchedWriteRes<[SBPort0,SBPort15]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup34], (instregex "MPSADBWrri")>;
def: InstRW<[SBWriteResGroup34], (instregex "VMPSADBWrri")>;

def SBWriteResGroup35 : SchedWriteRes<[SBPort1,SBPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup35], (instregex "CLI")>;
def: InstRW<[SBWriteResGroup35], (instregex "CVTSI2SS64rr")>;
def: InstRW<[SBWriteResGroup35], (instregex "CVTSI2SSrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "HADDPDrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "HADDPSrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "HSUBPDrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "HSUBPSrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VCVTSI2SS64rr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VCVTSI2SSrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VHADDPDYrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VHADDPDrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VHADDPSYrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VHADDPSrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VHSUBPDYrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VHSUBPDrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VHSUBPSYrr")>;
def: InstRW<[SBWriteResGroup35], (instregex "VHSUBPSrr")>;

def SBWriteResGroup35_2 : SchedWriteRes<[SBPort1,SBPort4,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup35_2], (instregex "ISTT_FP16m")>;
def: InstRW<[SBWriteResGroup35_2], (instregex "ISTT_FP32m")>;
def: InstRW<[SBWriteResGroup35_2], (instregex "ISTT_FP64m")>;
def: InstRW<[SBWriteResGroup35_2], (instregex "PUSHGS64")>;

def SBWriteResGroup36 : SchedWriteRes<[SBPort4,SBPort5,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup36], (instregex "CALL64pcrel32")>;
def: InstRW<[SBWriteResGroup36], (instregex "CALL(16|32|64)r")>;
def: InstRW<[SBWriteResGroup36], (instregex "EXTRACTPSmr")>;
def: InstRW<[SBWriteResGroup36], (instregex "VEXTRACTPSmr")>;

def SBWriteResGroup37 : SchedWriteRes<[SBPort4,SBPort01,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup37], (instregex "VMASKMOVPDYmr")>;
def: InstRW<[SBWriteResGroup37], (instregex "VMASKMOVPDmr")>;
def: InstRW<[SBWriteResGroup37], (instregex "VMASKMOVPSYmr")>;
def: InstRW<[SBWriteResGroup37], (instregex "VMASKMOVPSmr")>;

def SBWriteResGroup38 : SchedWriteRes<[SBPort4,SBPort23,SBPort05]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup38], (instregex "SETAEm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETBm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETEm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETGEm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETGm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETLEm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETLm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETNEm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETNOm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETNPm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETNSm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETOm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETPm")>;
def: InstRW<[SBWriteResGroup38], (instregex "SETSm")>;

def SBWriteResGroup39 : SchedWriteRes<[SBPort4,SBPort23,SBPort15]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup39], (instregex "PEXTRBmr")>;
def: InstRW<[SBWriteResGroup39], (instregex "VPEXTRBmr")>;
def: InstRW<[SBWriteResGroup39], (instregex "VPEXTRDmr")>;
def: InstRW<[SBWriteResGroup39], (instregex "VPEXTRWmr")>;

def SBWriteResGroup40 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup40], (instregex "MOV8mi")>;
def: InstRW<[SBWriteResGroup40], (instregex "STOSB")>;
def: InstRW<[SBWriteResGroup40], (instregex "STOSL")>;
def: InstRW<[SBWriteResGroup40], (instregex "STOSQ")>;
def: InstRW<[SBWriteResGroup40], (instregex "STOSW")>;

def SBWriteResGroup41 : SchedWriteRes<[SBPort5,SBPort015]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup41], (instregex "FNINIT")>;

def SBWriteResGroup42 : SchedWriteRes<[SBPort05,SBPort015]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup42], (instregex "CMPXCHG(16|32|64)rr")>;
def: InstRW<[SBWriteResGroup42], (instregex "CMPXCHG8rr")>;

def SBWriteResGroup43 : SchedWriteRes<[SBPort4,SBPort23,SBPort05]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup43], (instregex "SETAm")>;
def: InstRW<[SBWriteResGroup43], (instregex "SETBEm")>;

def SBWriteResGroup44 : SchedWriteRes<[SBPort0,SBPort4,SBPort5,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup44], (instregex "LDMXCSR")>;
def: InstRW<[SBWriteResGroup44], (instregex "STMXCSR")>;
def: InstRW<[SBWriteResGroup44], (instregex "VLDMXCSR")>;
def: InstRW<[SBWriteResGroup44], (instregex "VSTMXCSR")>;

def SBWriteResGroup45 : SchedWriteRes<[SBPort0,SBPort4,SBPort23,SBPort15]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup45], (instregex "PEXTRDmr")>;
def: InstRW<[SBWriteResGroup45], (instregex "PEXTRQmr")>;
def: InstRW<[SBWriteResGroup45], (instregex "VPEXTRQmr")>;
def: InstRW<[SBWriteResGroup45], (instregex "PUSHF16")>;
def: InstRW<[SBWriteResGroup45], (instregex "PUSHF64")>;

def SBWriteResGroup46 : SchedWriteRes<[SBPort4,SBPort5,SBPort01,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup46], (instregex "CLFLUSH")>;

def SBWriteResGroup47 : SchedWriteRes<[SBPort4,SBPort5,SBPort01,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SBWriteResGroup47], (instregex "FXRSTOR")>;

def SBWriteResGroup48 : SchedWriteRes<[SBPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup48], (instregex "LDDQUrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOV64toPQIrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVAPDrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVAPSrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVDDUPrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVDI2PDIrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVDQArm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVDQUrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVNTDQArm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVSHDUPrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVSLDUPrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVSSrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVUPDrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "MOVUPSrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "POP(16|32|64)r")>;
def: InstRW<[SBWriteResGroup48], (instregex "VBROADCASTSSrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VLDDQUYrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VLDDQUrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOV64toPQIrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVAPDrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVAPSrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVDDUPrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVDI2PDIrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVDQArm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVDQUrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVNTDQArm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVQI2PQIrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVSDrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVSHDUPrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVSLDUPrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVSSrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVUPDrm")>;
def: InstRW<[SBWriteResGroup48], (instregex "VMOVUPSrm")>;

def SBWriteResGroup49 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup49], (instregex "JMP(16|32|64)m")>;
def: InstRW<[SBWriteResGroup49], (instregex "MOV16sm")>;

def SBWriteResGroup50 : SchedWriteRes<[SBPort23,SBPort05]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup50], (instregex "BT(16|32|64)mi8")>;

def SBWriteResGroup51 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup51], (instregex "MMX_PABSBrm64")>;
def: InstRW<[SBWriteResGroup51], (instregex "MMX_PABSDrm64")>;
def: InstRW<[SBWriteResGroup51], (instregex "MMX_PABSWrm64")>;
def: InstRW<[SBWriteResGroup51], (instregex "MMX_PALIGNR64irm")>;
def: InstRW<[SBWriteResGroup51], (instregex "MMX_PSHUFBrm64")>;
def: InstRW<[SBWriteResGroup51], (instregex "MMX_PSIGNBrm64")>;
def: InstRW<[SBWriteResGroup51], (instregex "MMX_PSIGNDrm64")>;
def: InstRW<[SBWriteResGroup51], (instregex "MMX_PSIGNWrm64")>;

def SBWriteResGroup52 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup52], (instregex "ADD(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "ADD8rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "AND(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "AND8rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "CMP(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup52], (instregex "CMP(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup52], (instregex "CMP(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "CMP8mi")>;
def: InstRW<[SBWriteResGroup52], (instregex "CMP8mr")>;
def: InstRW<[SBWriteResGroup52], (instregex "CMP8rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "LODSL")>;
def: InstRW<[SBWriteResGroup52], (instregex "LODSQ")>;
def: InstRW<[SBWriteResGroup52], (instregex "OR(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "OR8rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "SUB(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "SUB8rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "XOR(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup52], (instregex "XOR8rm")>;

def SBWriteResGroup53 : SchedWriteRes<[SBPort4,SBPort23]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup53], (instregex "ST_F32m")>;
def: InstRW<[SBWriteResGroup53], (instregex "ST_F64m")>;
def: InstRW<[SBWriteResGroup53], (instregex "ST_FP32m")>;
def: InstRW<[SBWriteResGroup53], (instregex "ST_FP64m")>;
def: InstRW<[SBWriteResGroup53], (instregex "ST_FP80m")>;

def SBWriteResGroup54 : SchedWriteRes<[SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup54], (instregex "VBROADCASTSDYrm")>;
def: InstRW<[SBWriteResGroup54], (instregex "VBROADCASTSSYrm")>;
def: InstRW<[SBWriteResGroup54], (instregex "VMOVAPDYrm")>;
def: InstRW<[SBWriteResGroup54], (instregex "VMOVAPSYrm")>;
def: InstRW<[SBWriteResGroup54], (instregex "VMOVDDUPYrm")>;
def: InstRW<[SBWriteResGroup54], (instregex "VMOVDQAYrm")>;
def: InstRW<[SBWriteResGroup54], (instregex "VMOVDQUYrm")>;
def: InstRW<[SBWriteResGroup54], (instregex "VMOVSHDUPYrm")>;
def: InstRW<[SBWriteResGroup54], (instregex "VMOVSLDUPYrm")>;
def: InstRW<[SBWriteResGroup54], (instregex "VMOVUPDYrm")>;
def: InstRW<[SBWriteResGroup54], (instregex "VMOVUPSYrm")>;

def SBWriteResGroup55 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup55], (instregex "CVTPS2PDrm")>;
def: InstRW<[SBWriteResGroup55], (instregex "CVTSS2SDrm")>;
def: InstRW<[SBWriteResGroup55], (instregex "VCVTPS2PDYrm")>;
def: InstRW<[SBWriteResGroup55], (instregex "VCVTPS2PDrm")>;
def: InstRW<[SBWriteResGroup55], (instregex "VCVTSS2SDrm")>;
def: InstRW<[SBWriteResGroup55], (instregex "VTESTPDrm")>;
def: InstRW<[SBWriteResGroup55], (instregex "VTESTPSrm")>;

def SBWriteResGroup56 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup56], (instregex "ANDNPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "ANDNPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "ANDPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "ANDPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "INSERTPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "MOVHPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "MOVHPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "MOVLPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "MOVLPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "ORPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "ORPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "SHUFPDrmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "SHUFPSrmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "UNPCKHPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "UNPCKHPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "UNPCKLPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "UNPCKLPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VANDNPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VANDNPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VANDPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VANDPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VBROADCASTF128")>;
def: InstRW<[SBWriteResGroup56], (instregex "VINSERTPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VMOVHPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VMOVHPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VMOVLPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VMOVLPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VORPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VORPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPERMILPDmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPERMILPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPERMILPSmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "VPERMILPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VSHUFPDrmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "VSHUFPSrmi")>;
def: InstRW<[SBWriteResGroup56], (instregex "VUNPCKHPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VUNPCKHPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VUNPCKLPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VUNPCKLPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VXORPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "VXORPSrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "XORPDrm")>;
def: InstRW<[SBWriteResGroup56], (instregex "XORPSrm")>;

def SBWriteResGroup57 : SchedWriteRes<[SBPort5,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup57], (instregex "AESDECLASTrr")>;
def: InstRW<[SBWriteResGroup57], (instregex "AESDECrr")>;
def: InstRW<[SBWriteResGroup57], (instregex "AESENCLASTrr")>;
def: InstRW<[SBWriteResGroup57], (instregex "AESENCrr")>;
def: InstRW<[SBWriteResGroup57], (instregex "VAESDECLASTrr")>;
def: InstRW<[SBWriteResGroup57], (instregex "VAESDECrr")>;
def: InstRW<[SBWriteResGroup57], (instregex "VAESENCLASTrr")>;
def: InstRW<[SBWriteResGroup57], (instregex "VAESENCrr")>;

def SBWriteResGroup58 : SchedWriteRes<[SBPort23,SBPort05]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup58], (instregex "BLENDPDrmi")>;
def: InstRW<[SBWriteResGroup58], (instregex "BLENDPSrmi")>;
def: InstRW<[SBWriteResGroup58], (instregex "VBLENDPDrmi")>;
def: InstRW<[SBWriteResGroup58], (instregex "VBLENDPSrmi")>;
def: InstRW<[SBWriteResGroup58], (instregex "VINSERTF128rm")>;

def SBWriteResGroup59 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup59], (instregex "MMX_PADDQirm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PABSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PABSDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PABSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PACKSSDWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PACKSSWBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PACKUSDWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PACKUSWBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PADDBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PADDDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PADDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PADDSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PADDSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PADDUSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PADDUSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PADDWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PALIGNRrmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "PAVGBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PAVGWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PBLENDWrmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "PCMPEQBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PCMPEQDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PCMPEQQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PCMPEQWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PCMPGTBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PCMPGTDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PCMPGTWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PINSRBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PINSRDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PINSRQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PINSRWrmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMAXSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMAXSDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMAXSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMAXUBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMAXUDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMAXUWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMINSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMINSDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMINSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMINUBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMINUDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMINUWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVSXBDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVSXBQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVSXBWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVSXDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVSXWDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVSXWQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVZXBDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVZXBQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVZXBWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVZXDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVZXWDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PMOVZXWQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSHUFBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSHUFDmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSHUFHWmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSHUFLWmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSIGNBrm128")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSIGNDrm128")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSIGNWrm128")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSUBBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSUBDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSUBQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSUBSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSUBSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSUBUSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSUBUSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PSUBWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PUNPCKHBWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PUNPCKHDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PUNPCKHQDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PUNPCKHWDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PUNPCKLBWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PUNPCKLDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PUNPCKLQDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "PUNPCKLWDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPABSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPABSDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPABSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPACKSSDWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPACKSSWBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPACKUSDWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPACKUSWBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPADDBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPADDDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPADDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPADDSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPADDSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPADDUSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPADDUSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPADDWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPALIGNRrmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPAVGBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPAVGWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPBLENDWrmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPCMPEQBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPCMPEQDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPCMPEQQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPCMPEQWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPCMPGTBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPCMPGTDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPCMPGTWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPINSRBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPINSRDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPINSRQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPINSRWrmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMAXSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMAXSDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMAXSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMAXUBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMAXUDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMAXUWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMINSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMINSDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMINSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMINUBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMINUDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMINUWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVSXBDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVSXBQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVSXBWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVSXDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVSXWDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVSXWQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVZXBDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVZXBQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVZXBWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVZXDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVZXWDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPMOVZXWQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSHUFBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSHUFDmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSHUFHWmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSHUFLWmi")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSIGNBrm128")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSIGNDrm128")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSIGNWrm128")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSUBBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSUBDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSUBQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSUBSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSUBSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSUBUSBrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSUBUSWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPSUBWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPUNPCKHBWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPUNPCKHDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPUNPCKHQDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPUNPCKHWDrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPUNPCKLBWrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPUNPCKLDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPUNPCKLQDQrm")>;
def: InstRW<[SBWriteResGroup59], (instregex "VPUNPCKLWDrm")>;

def SBWriteResGroup60 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup60], (instregex "PANDNrm")>;
def: InstRW<[SBWriteResGroup60], (instregex "PANDrm")>;
def: InstRW<[SBWriteResGroup60], (instregex "PORrm")>;
def: InstRW<[SBWriteResGroup60], (instregex "PXORrm")>;
def: InstRW<[SBWriteResGroup60], (instregex "VPANDNrm")>;
def: InstRW<[SBWriteResGroup60], (instregex "VPANDrm")>;
def: InstRW<[SBWriteResGroup60], (instregex "VPORrm")>;
def: InstRW<[SBWriteResGroup60], (instregex "VPXORrm")>;

def SBWriteResGroup61 : SchedWriteRes<[SBPort0,SBPort05]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup61], (instregex "VRCPPSYr")>;
def: InstRW<[SBWriteResGroup61], (instregex "VRSQRTPSYr")>;

def SBWriteResGroup62 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup62], (instregex "VERRm")>;
def: InstRW<[SBWriteResGroup62], (instregex "VERWm")>;

def SBWriteResGroup63 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup63], (instregex "LODSB")>;
def: InstRW<[SBWriteResGroup63], (instregex "LODSW")>;

def SBWriteResGroup64 : SchedWriteRes<[SBPort5,SBPort01,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup64], (instregex "FARJMP64")>;

def SBWriteResGroup65 : SchedWriteRes<[SBPort23,SBPort05,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup65], (instregex "ADC(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "ADC8rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVAE(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVB(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVE(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVG(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVGE(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVL(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVLE(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVNE(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVNO(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVNP(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVNS(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVO(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVP(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "CMOVS(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "SBB(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup65], (instregex "SBB8rm")>;

def SBWriteResGroup66 : SchedWriteRes<[SBPort0,SBPort4,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup66], (instregex "FNSTSWm")>;

def SBWriteResGroup67 : SchedWriteRes<[SBPort1,SBPort5,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup67], (instregex "SLDT(16|32|64)r")>;
def: InstRW<[SBWriteResGroup67], (instregex "STR(16|32|64)r")>;

def SBWriteResGroup68 : SchedWriteRes<[SBPort4,SBPort5,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup68], (instregex "CALL(16|32|64)m")>;
def: InstRW<[SBWriteResGroup68], (instregex "FNSTCW16m")>;

def SBWriteResGroup69 : SchedWriteRes<[SBPort4,SBPort23,SBPort05]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup69], (instregex "BTC(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup69], (instregex "BTR(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup69], (instregex "BTS(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup69], (instregex "SAR(16|32|64)mi")>;
def: InstRW<[SBWriteResGroup69], (instregex "SAR8mi")>;
def: InstRW<[SBWriteResGroup69], (instregex "SHL(16|32|64)m1")>;
def: InstRW<[SBWriteResGroup69], (instregex "SHL(16|32|64)mi")>;
def: InstRW<[SBWriteResGroup69], (instregex "SHL8m1")>;
def: InstRW<[SBWriteResGroup69], (instregex "SHL8mi")>;
def: InstRW<[SBWriteResGroup69], (instregex "SHR(16|32|64)mi")>;
def: InstRW<[SBWriteResGroup69], (instregex "SHR8mi")>;

def SBWriteResGroup70 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup70], (instregex "ADD(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup70], (instregex "ADD(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "ADD8mi")>;
def: InstRW<[SBWriteResGroup70], (instregex "ADD8mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "AND(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup70], (instregex "AND(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "AND8mi")>;
def: InstRW<[SBWriteResGroup70], (instregex "AND8mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "DEC(16|32|64)m")>;
def: InstRW<[SBWriteResGroup70], (instregex "DEC8m")>;
def: InstRW<[SBWriteResGroup70], (instregex "INC(16|32|64)m")>;
def: InstRW<[SBWriteResGroup70], (instregex "INC8m")>;
def: InstRW<[SBWriteResGroup70], (instregex "NEG(16|32|64)m")>;
def: InstRW<[SBWriteResGroup70], (instregex "NEG8m")>;
def: InstRW<[SBWriteResGroup70], (instregex "NOT(16|32|64)m")>;
def: InstRW<[SBWriteResGroup70], (instregex "NOT8m")>;
def: InstRW<[SBWriteResGroup70], (instregex "OR(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup70], (instregex "OR(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "OR8mi")>;
def: InstRW<[SBWriteResGroup70], (instregex "OR8mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "SUB(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup70], (instregex "SUB(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "SUB8mi")>;
def: InstRW<[SBWriteResGroup70], (instregex "SUB8mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "TEST(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "TEST8mi")>;
def: InstRW<[SBWriteResGroup70], (instregex "TEST8mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "XOR(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup70], (instregex "XOR(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup70], (instregex "XOR8mi")>;
def: InstRW<[SBWriteResGroup70], (instregex "XOR8mr")>;

def SBWriteResGroup71 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup71], (instregex "MMX_PMADDUBSWrm64")>;
def: InstRW<[SBWriteResGroup71], (instregex "MMX_PMULHRSWrm64")>;
def: InstRW<[SBWriteResGroup71], (instregex "VTESTPDYrm")>;
def: InstRW<[SBWriteResGroup71], (instregex "VTESTPSYrm")>;

def SBWriteResGroup72 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup72], (instregex "BSF(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup72], (instregex "BSR(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup72], (instregex "CRC32r(16|32|64)m64")>;
def: InstRW<[SBWriteResGroup72], (instregex "CRC32r(16|32|64)m8")>;
def: InstRW<[SBWriteResGroup72], (instregex "FCOM32m")>;
def: InstRW<[SBWriteResGroup72], (instregex "FCOM64m")>;
def: InstRW<[SBWriteResGroup72], (instregex "FCOMP32m")>;
def: InstRW<[SBWriteResGroup72], (instregex "FCOMP64m")>;
def: InstRW<[SBWriteResGroup72], (instregex "MUL8m")>;

def SBWriteResGroup73 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup73], (instregex "VANDNPDYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VANDNPSYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VANDPDYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VANDPSYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VORPDYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VORPSYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VPERM2F128rm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VPERMILPDYmi")>;
def: InstRW<[SBWriteResGroup73], (instregex "VPERMILPDYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VPERMILPSYmi")>;
def: InstRW<[SBWriteResGroup73], (instregex "VPERMILPSYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VSHUFPDYrmi")>;
def: InstRW<[SBWriteResGroup73], (instregex "VSHUFPSYrmi")>;
def: InstRW<[SBWriteResGroup73], (instregex "VUNPCKHPDYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VUNPCKHPSYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VUNPCKLPDYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VUNPCKLPSYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VXORPDYrm")>;
def: InstRW<[SBWriteResGroup73], (instregex "VXORPSYrm")>;

def SBWriteResGroup74 : SchedWriteRes<[SBPort23,SBPort05]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup74], (instregex "VBLENDPDYrmi")>;
def: InstRW<[SBWriteResGroup74], (instregex "VBLENDPSYrmi")>;

def SBWriteResGroup75 : SchedWriteRes<[SBPort23,SBPort05]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup75], (instregex "BLENDVPDrm0")>;
def: InstRW<[SBWriteResGroup75], (instregex "BLENDVPSrm0")>;
def: InstRW<[SBWriteResGroup75], (instregex "VBLENDVPDrm")>;
def: InstRW<[SBWriteResGroup75], (instregex "VBLENDVPSrm")>;
def: InstRW<[SBWriteResGroup75], (instregex "VMASKMOVPDrm")>;
def: InstRW<[SBWriteResGroup75], (instregex "VMASKMOVPSrm")>;

def SBWriteResGroup76 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup76], (instregex "PBLENDVBrr0")>;
def: InstRW<[SBWriteResGroup76], (instregex "VPBLENDVBrm")>;

def SBWriteResGroup77 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup77], (instregex "COMISDrm")>;
def: InstRW<[SBWriteResGroup77], (instregex "COMISSrm")>;
def: InstRW<[SBWriteResGroup77], (instregex "UCOMISDrm")>;
def: InstRW<[SBWriteResGroup77], (instregex "UCOMISSrm")>;
def: InstRW<[SBWriteResGroup77], (instregex "VCOMISDrm")>;
def: InstRW<[SBWriteResGroup77], (instregex "VCOMISSrm")>;
def: InstRW<[SBWriteResGroup77], (instregex "VUCOMISDrm")>;
def: InstRW<[SBWriteResGroup77], (instregex "VUCOMISSrm")>;

def SBWriteResGroup78 : SchedWriteRes<[SBPort0,SBPort5,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup78], (instregex "PTESTrm")>;
def: InstRW<[SBWriteResGroup78], (instregex "VPTESTrm")>;

def SBWriteResGroup79 : SchedWriteRes<[SBPort0,SBPort23,SBPort15]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup79], (instregex "PSLLDrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "PSLLQrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "PSLLWrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "PSRADrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "PSRAWrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "PSRLDrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "PSRLQrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "PSRLWrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "VPSLLDrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "VPSLLQrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "VPSLLWrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "VPSRADrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "VPSRAWrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "VPSRLDrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "VPSRLQrm")>;
def: InstRW<[SBWriteResGroup79], (instregex "VPSRLWrm")>;

def SBWriteResGroup80 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup80], (instregex "MMX_PHADDSWrm64")>;
def: InstRW<[SBWriteResGroup80], (instregex "MMX_PHADDWrm64")>;
def: InstRW<[SBWriteResGroup80], (instregex "MMX_PHADDrm64")>;
def: InstRW<[SBWriteResGroup80], (instregex "MMX_PHSUBDrm64")>;
def: InstRW<[SBWriteResGroup80], (instregex "MMX_PHSUBSWrm64")>;
def: InstRW<[SBWriteResGroup80], (instregex "MMX_PHSUBWrm64")>;

def SBWriteResGroup81 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup81], (instregex "CMPXCHG(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup81], (instregex "CMPXCHG8rm")>;

def SBWriteResGroup82 : SchedWriteRes<[SBPort23,SBPort05,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup82], (instregex "CMOVA(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup82], (instregex "CMOVBE(16|32|64)rm")>;

def SBWriteResGroup83 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[SBWriteResGroup83], (instregex "CMPSB")>;
def: InstRW<[SBWriteResGroup83], (instregex "CMPSL")>;
def: InstRW<[SBWriteResGroup83], (instregex "CMPSQ")>;
def: InstRW<[SBWriteResGroup83], (instregex "CMPSW")>;

def SBWriteResGroup84 : SchedWriteRes<[SBPort4,SBPort5,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,2];
}
def: InstRW<[SBWriteResGroup84], (instregex "FLDCW16m")>;

def SBWriteResGroup85 : SchedWriteRes<[SBPort4,SBPort23,SBPort05]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,2];
}
def: InstRW<[SBWriteResGroup85], (instregex "ROL(16|32|64)mi")>;
def: InstRW<[SBWriteResGroup85], (instregex "ROL8mi")>;
def: InstRW<[SBWriteResGroup85], (instregex "ROR(16|32|64)mi")>;
def: InstRW<[SBWriteResGroup85], (instregex "ROR8mi")>;

def SBWriteResGroup86 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,2];
}
def: InstRW<[SBWriteResGroup86], (instregex "MOVSB")>;
def: InstRW<[SBWriteResGroup86], (instregex "MOVSL")>;
def: InstRW<[SBWriteResGroup86], (instregex "MOVSQ")>;
def: InstRW<[SBWriteResGroup86], (instregex "MOVSW")>;
def: InstRW<[SBWriteResGroup86], (instregex "XADD(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup86], (instregex "XADD8rm")>;

def SBWriteResGroup87 : SchedWriteRes<[SBPort4,SBPort5,SBPort01,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SBWriteResGroup87], (instregex "FARCALL64")>;

def SBWriteResGroup88 : SchedWriteRes<[SBPort4,SBPort23,SBPort05,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SBWriteResGroup88], (instregex "SHLD(16|32|64)mri8")>;
def: InstRW<[SBWriteResGroup88], (instregex "SHRD(16|32|64)mri8")>;

def SBWriteResGroup89 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup89], (instregex "MMX_PMULUDQirm")>;
def: InstRW<[SBWriteResGroup89], (instregex "PMADDUBSWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "PMADDWDrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "PMULDQrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "PMULHRSWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "PMULHUWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "PMULHWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "PMULLDrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "PMULLWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "PMULUDQrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "PSADBWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "VPMADDUBSWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "VPMADDWDrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "VPMULDQrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "VPMULHRSWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "VPMULHUWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "VPMULHWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "VPMULLDrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "VPMULLWrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "VPMULUDQrm")>;
def: InstRW<[SBWriteResGroup89], (instregex "VPSADBWrm")>;

def SBWriteResGroup90 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup90], (instregex "ADDPDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "ADDPSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "ADDSDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "ADDSSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "ADDSUBPDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "ADDSUBPSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CMPPDrmi")>;
def: InstRW<[SBWriteResGroup90], (instregex "CMPPSrmi")>;
def: InstRW<[SBWriteResGroup90], (instregex "CMPSSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTDQ2PSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTPS2DQrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTSI2SD64rm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTSI2SDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "CVTTPS2DQrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MAXPDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MAXPSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MAXSDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MAXSSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MINPDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MINPSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MINSDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MINSSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MMX_CVTPI2PSirm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MMX_CVTPS2PIirm")>;
def: InstRW<[SBWriteResGroup90], (instregex "MMX_CVTTPS2PIirm")>;
def: InstRW<[SBWriteResGroup90], (instregex "POPCNT(16|32|64)rm")>;
def: InstRW<[SBWriteResGroup90], (instregex "ROUNDPDm")>;
def: InstRW<[SBWriteResGroup90], (instregex "ROUNDPSm")>;
def: InstRW<[SBWriteResGroup90], (instregex "ROUNDSDm")>;
def: InstRW<[SBWriteResGroup90], (instregex "ROUNDSSm")>;
def: InstRW<[SBWriteResGroup90], (instregex "SUBPDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "SUBPSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "SUBSDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "SUBSSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VADDPDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VADDPSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VADDSDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VADDSSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VADDSUBPDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VADDSUBPSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VCMPPDrmi")>;
def: InstRW<[SBWriteResGroup90], (instregex "VCMPPSrmi")>;
def: InstRW<[SBWriteResGroup90], (instregex "VCMPSDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VCMPSSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VCVTDQ2PSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VCVTPS2DQrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VCVTSI2SD64rm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VCVTSI2SDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VCVTTPS2DQrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VMAXPDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VMAXPSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VMAXSDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VMAXSSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VMINPDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VMINPSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VMINSDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VMINSSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VROUNDPDm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VROUNDPSm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VROUNDSDm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VROUNDSSm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VSUBPDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VSUBPSrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VSUBSDrm")>;
def: InstRW<[SBWriteResGroup90], (instregex "VSUBSSrm")>;

def SBWriteResGroup91 : SchedWriteRes<[SBPort23,SBPort05]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup91], (instregex "VBLENDVPDYrm")>;
def: InstRW<[SBWriteResGroup91], (instregex "VBLENDVPSYrm")>;
def: InstRW<[SBWriteResGroup91], (instregex "VMASKMOVPDYrm")>;
def: InstRW<[SBWriteResGroup91], (instregex "VMASKMOVPSYrm")>;

def SBWriteResGroup92 : SchedWriteRes<[SBPort0,SBPort1,SBPort5]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup92], (instregex "DPPDrri")>;
def: InstRW<[SBWriteResGroup92], (instregex "VDPPDrri")>;

def SBWriteResGroup93 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup93], (instregex "CVTSD2SI64rm")>;
def: InstRW<[SBWriteResGroup93], (instregex "CVTSD2SIrm")>;
def: InstRW<[SBWriteResGroup93], (instregex "CVTSS2SI64rm")>;
def: InstRW<[SBWriteResGroup93], (instregex "CVTSS2SIrm")>;
def: InstRW<[SBWriteResGroup93], (instregex "CVTTSD2SI64rm")>;
def: InstRW<[SBWriteResGroup93], (instregex "CVTTSD2SIrm")>;
def: InstRW<[SBWriteResGroup93], (instregex "CVTTSS2SI64rm")>;
def: InstRW<[SBWriteResGroup93], (instregex "CVTTSS2SIrm")>;
def: InstRW<[SBWriteResGroup93], (instregex "MUL(16|32|64)m")>;

def SBWriteResGroup94 : SchedWriteRes<[SBPort0,SBPort5,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup94], (instregex "VPTESTYrm")>;

def SBWriteResGroup95 : SchedWriteRes<[SBPort5,SBPort01,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup95], (instregex "LD_F32m")>;
def: InstRW<[SBWriteResGroup95], (instregex "LD_F64m")>;
def: InstRW<[SBWriteResGroup95], (instregex "LD_F80m")>;

def SBWriteResGroup96 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup96], (instregex "PHADDDrm")>;
def: InstRW<[SBWriteResGroup96], (instregex "PHADDSWrm128")>;
def: InstRW<[SBWriteResGroup96], (instregex "PHADDWrm")>;
def: InstRW<[SBWriteResGroup96], (instregex "PHSUBDrm")>;
def: InstRW<[SBWriteResGroup96], (instregex "PHSUBSWrm128")>;
def: InstRW<[SBWriteResGroup96], (instregex "PHSUBWrm")>;
def: InstRW<[SBWriteResGroup96], (instregex "VPHADDDrm")>;
def: InstRW<[SBWriteResGroup96], (instregex "VPHADDSWrm128")>;
def: InstRW<[SBWriteResGroup96], (instregex "VPHADDWrm")>;
def: InstRW<[SBWriteResGroup96], (instregex "VPHSUBDrm")>;
def: InstRW<[SBWriteResGroup96], (instregex "VPHSUBSWrm128")>;
def: InstRW<[SBWriteResGroup96], (instregex "VPHSUBWrm")>;

def SBWriteResGroup97 : SchedWriteRes<[SBPort1,SBPort4,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup97], (instregex "IST_F16m")>;
def: InstRW<[SBWriteResGroup97], (instregex "IST_F32m")>;
def: InstRW<[SBWriteResGroup97], (instregex "IST_FP16m")>;
def: InstRW<[SBWriteResGroup97], (instregex "IST_FP32m")>;
def: InstRW<[SBWriteResGroup97], (instregex "IST_FP64m")>;

def SBWriteResGroup97_2 : SchedWriteRes<[SBPort4,SBPort23,SBPort05]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,2,3];
}
def: InstRW<[SBWriteResGroup97_2], (instregex "ROL(16|32|64)mCL")>;
def: InstRW<[SBWriteResGroup97_2], (instregex "ROL8mCL")>;
def: InstRW<[SBWriteResGroup97_2], (instregex "ROR(16|32|64)mCL")>;
def: InstRW<[SBWriteResGroup97_2], (instregex "ROR8mCL")>;
def: InstRW<[SBWriteResGroup97_2], (instregex "SAR(16|32|64)mCL")>;
def: InstRW<[SBWriteResGroup97_2], (instregex "SAR8mCL")>;
def: InstRW<[SBWriteResGroup97_2], (instregex "SHL(16|32|64)mCL")>;
def: InstRW<[SBWriteResGroup97_2], (instregex "SHL8mCL")>;
def: InstRW<[SBWriteResGroup97_2], (instregex "SHR(16|32|64)mCL")>;
def: InstRW<[SBWriteResGroup97_2], (instregex "SHR8mCL")>;

def SBWriteResGroup98 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,2,3];
}
def: InstRW<[SBWriteResGroup98], (instregex "ADC(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup98], (instregex "ADC8mi")>;
def: InstRW<[SBWriteResGroup98], (instregex "SBB(16|32|64)mi8")>;
def: InstRW<[SBWriteResGroup98], (instregex "SBB8mi")>;

def SBWriteResGroup99 : SchedWriteRes<[SBPort4,SBPort23,SBPort05,SBPort015]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,2,2,1];
}
def: InstRW<[SBWriteResGroup99], (instregex "ADC(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup99], (instregex "ADC8mr")>;
def: InstRW<[SBWriteResGroup99], (instregex "SBB(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup99], (instregex "SBB8mr")>;

def SBWriteResGroup100 : SchedWriteRes<[SBPort4,SBPort5,SBPort23,SBPort05,SBPort015]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,2,1,1];
}
def: InstRW<[SBWriteResGroup100], (instregex "BT(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup100], (instregex "BTC(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup100], (instregex "BTR(16|32|64)mr")>;
def: InstRW<[SBWriteResGroup100], (instregex "BTS(16|32|64)mr")>;

def SBWriteResGroup101 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup101], (instregex "ADD_F32m")>;
def: InstRW<[SBWriteResGroup101], (instregex "ADD_F64m")>;
def: InstRW<[SBWriteResGroup101], (instregex "ILD_F16m")>;
def: InstRW<[SBWriteResGroup101], (instregex "ILD_F32m")>;
def: InstRW<[SBWriteResGroup101], (instregex "ILD_F64m")>;
def: InstRW<[SBWriteResGroup101], (instregex "SUBR_F32m")>;
def: InstRW<[SBWriteResGroup101], (instregex "SUBR_F64m")>;
def: InstRW<[SBWriteResGroup101], (instregex "SUB_F32m")>;
def: InstRW<[SBWriteResGroup101], (instregex "SUB_F64m")>;
def: InstRW<[SBWriteResGroup101], (instregex "VADDPDYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VADDPSYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VADDSUBPDYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VADDSUBPSYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VCMPPDYrmi")>;
def: InstRW<[SBWriteResGroup101], (instregex "VCMPPSYrmi")>;
def: InstRW<[SBWriteResGroup101], (instregex "VCVTDQ2PSYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VCVTPS2DQYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VCVTTPS2DQYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VMAXPDYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VMAXPSYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VMINPDYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VMINPSYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VROUNDYPDm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VROUNDYPSm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VSUBPDYrm")>;
def: InstRW<[SBWriteResGroup101], (instregex "VSUBPSYrm")>;

def SBWriteResGroup102 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup102], (instregex "VCVTSD2SI64rm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VCVTSD2SIrm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VCVTSS2SI64rm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VCVTSS2SIrm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VCVTTSD2SI64rm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VCVTTSD2SIrm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VCVTTSS2SI64rm")>;
def: InstRW<[SBWriteResGroup102], (instregex "VCVTTSS2SIrm")>;

def SBWriteResGroup103 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup103], (instregex "CVTDQ2PDrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "CVTPD2DQrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "CVTPD2PSrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "CVTSD2SSrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "CVTSI2SS64rm")>;
def: InstRW<[SBWriteResGroup103], (instregex "CVTSI2SSrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "CVTTPD2DQrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "MMX_CVTPD2PIirm")>;
def: InstRW<[SBWriteResGroup103], (instregex "MMX_CVTPI2PDirm")>;
def: InstRW<[SBWriteResGroup103], (instregex "MMX_CVTTPD2PIirm")>;
def: InstRW<[SBWriteResGroup103], (instregex "VCVTDQ2PDYrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "VCVTDQ2PDrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "VCVTPD2DQrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "VCVTPD2PSrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "VCVTSD2SSrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "VCVTSI2SS64rm")>;
def: InstRW<[SBWriteResGroup103], (instregex "VCVTSI2SSrm")>;
def: InstRW<[SBWriteResGroup103], (instregex "VCVTTPD2DQrm")>;

def SBWriteResGroup103_2 : SchedWriteRes<[SBPort4,SBPort23,SBPort05,SBPort015]> {
  let Latency = 10;
  let NumMicroOps = 7;
  let ResourceCycles = [1,2,3,1];
}
def: InstRW<[SBWriteResGroup103_2], (instregex "SHLD(16|32|64)mrCL")>;
def: InstRW<[SBWriteResGroup103_2], (instregex "SHRD(16|32|64)mrCL")>;

def SBWriteResGroup104 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup104], (instregex "MULPDrm")>;
def: InstRW<[SBWriteResGroup104], (instregex "MULPSrm")>;
def: InstRW<[SBWriteResGroup104], (instregex "MULSDrm")>;
def: InstRW<[SBWriteResGroup104], (instregex "MULSSrm")>;
def: InstRW<[SBWriteResGroup104], (instregex "PCMPGTQrm")>;
def: InstRW<[SBWriteResGroup104], (instregex "PHMINPOSUWrm128")>;
def: InstRW<[SBWriteResGroup104], (instregex "RCPPSm")>;
def: InstRW<[SBWriteResGroup104], (instregex "RCPSSm")>;
def: InstRW<[SBWriteResGroup104], (instregex "RSQRTPSm")>;
def: InstRW<[SBWriteResGroup104], (instregex "RSQRTSSm")>;
def: InstRW<[SBWriteResGroup104], (instregex "VMULPDrm")>;
def: InstRW<[SBWriteResGroup104], (instregex "VMULPSrm")>;
def: InstRW<[SBWriteResGroup104], (instregex "VMULSDrm")>;
def: InstRW<[SBWriteResGroup104], (instregex "VMULSSrm")>;
def: InstRW<[SBWriteResGroup104], (instregex "VPCMPGTQrm")>;
def: InstRW<[SBWriteResGroup104], (instregex "VPHMINPOSUWrm128")>;
def: InstRW<[SBWriteResGroup104], (instregex "VRCPPSm")>;
def: InstRW<[SBWriteResGroup104], (instregex "VRCPSSm")>;
def: InstRW<[SBWriteResGroup104], (instregex "VRSQRTPSm")>;
def: InstRW<[SBWriteResGroup104], (instregex "VRSQRTSSm")>;

def SBWriteResGroup105 : SchedWriteRes<[SBPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SBWriteResGroup105], (instregex "PCMPISTRIrr")>;
def: InstRW<[SBWriteResGroup105], (instregex "PCMPISTRM128rr")>;
def: InstRW<[SBWriteResGroup105], (instregex "VPCMPISTRIrr")>;
def: InstRW<[SBWriteResGroup105], (instregex "VPCMPISTRM128rr")>;

def SBWriteResGroup106 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup106], (instregex "FICOM16m")>;
def: InstRW<[SBWriteResGroup106], (instregex "FICOM32m")>;
def: InstRW<[SBWriteResGroup106], (instregex "FICOMP16m")>;
def: InstRW<[SBWriteResGroup106], (instregex "FICOMP32m")>;

def SBWriteResGroup107 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup107], (instregex "VCVTPD2DQYrm")>;
def: InstRW<[SBWriteResGroup107], (instregex "VCVTPD2PSYrm")>;
def: InstRW<[SBWriteResGroup107], (instregex "VCVTTPD2DQYrm")>;

def SBWriteResGroup108 : SchedWriteRes<[SBPort0,SBPort23,SBPort15]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup108], (instregex "MPSADBWrmi")>;
def: InstRW<[SBWriteResGroup108], (instregex "VMPSADBWrmi")>;

def SBWriteResGroup109 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup109], (instregex "HADDPDrm")>;
def: InstRW<[SBWriteResGroup109], (instregex "HADDPSrm")>;
def: InstRW<[SBWriteResGroup109], (instregex "HSUBPDrm")>;
def: InstRW<[SBWriteResGroup109], (instregex "HSUBPSrm")>;
def: InstRW<[SBWriteResGroup109], (instregex "VHADDPDrm")>;
def: InstRW<[SBWriteResGroup109], (instregex "VHADDPSrm")>;
def: InstRW<[SBWriteResGroup109], (instregex "VHSUBPDrm")>;
def: InstRW<[SBWriteResGroup109], (instregex "VHSUBPSrm")>;

def SBWriteResGroup110 : SchedWriteRes<[SBPort5]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup110], (instregex "AESIMCrr")>;
def: InstRW<[SBWriteResGroup110], (instregex "VAESIMCrr")>;

def SBWriteResGroup111 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup111], (instregex "MUL_F32m")>;
def: InstRW<[SBWriteResGroup111], (instregex "MUL_F64m")>;
def: InstRW<[SBWriteResGroup111], (instregex "VMULPDYrm")>;
def: InstRW<[SBWriteResGroup111], (instregex "VMULPSYrm")>;

def SBWriteResGroup112 : SchedWriteRes<[SBPort0,SBPort1,SBPort5]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup112], (instregex "DPPSrri")>;
def: InstRW<[SBWriteResGroup112], (instregex "VDPPSYrri")>;
def: InstRW<[SBWriteResGroup112], (instregex "VDPPSrri")>;

def SBWriteResGroup113 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup113], (instregex "VHADDPDYrm")>;
def: InstRW<[SBWriteResGroup113], (instregex "VHADDPSYrm")>;
def: InstRW<[SBWriteResGroup113], (instregex "VHSUBPDYrm")>;
def: InstRW<[SBWriteResGroup113], (instregex "VHSUBPSYrm")>;

def SBWriteResGroup114 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup114], (instregex "ADD_FI16m")>;
def: InstRW<[SBWriteResGroup114], (instregex "ADD_FI32m")>;
def: InstRW<[SBWriteResGroup114], (instregex "SUBR_FI16m")>;
def: InstRW<[SBWriteResGroup114], (instregex "SUBR_FI32m")>;
def: InstRW<[SBWriteResGroup114], (instregex "SUB_FI16m")>;
def: InstRW<[SBWriteResGroup114], (instregex "SUB_FI32m")>;

def SBWriteResGroup115 : SchedWriteRes<[SBPort5,SBPort23,SBPort015]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup115], (instregex "AESDECLASTrm")>;
def: InstRW<[SBWriteResGroup115], (instregex "AESDECrm")>;
def: InstRW<[SBWriteResGroup115], (instregex "AESENCLASTrm")>;
def: InstRW<[SBWriteResGroup115], (instregex "AESENCrm")>;
def: InstRW<[SBWriteResGroup115], (instregex "VAESDECLASTrm")>;
def: InstRW<[SBWriteResGroup115], (instregex "VAESDECrm")>;
def: InstRW<[SBWriteResGroup115], (instregex "VAESENCLASTrm")>;
def: InstRW<[SBWriteResGroup115], (instregex "VAESENCrm")>;

def SBWriteResGroup116 : SchedWriteRes<[SBPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup116], (instregex "DIVPSrr")>;
def: InstRW<[SBWriteResGroup116], (instregex "DIVSSrr")>;
def: InstRW<[SBWriteResGroup116], (instregex "SQRTPSr")>;
def: InstRW<[SBWriteResGroup116], (instregex "SQRTSSr")>;
def: InstRW<[SBWriteResGroup116], (instregex "VDIVPSrr")>;
def: InstRW<[SBWriteResGroup116], (instregex "VDIVSSrr")>;
def: InstRW<[SBWriteResGroup116], (instregex "VSQRTPSr")>;

def SBWriteResGroup117 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 14;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup117], (instregex "VSQRTSSm")>;

def SBWriteResGroup118 : SchedWriteRes<[SBPort0,SBPort23,SBPort05]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SBWriteResGroup118], (instregex "VRCPPSYm")>;
def: InstRW<[SBWriteResGroup118], (instregex "VRSQRTPSYm")>;

def SBWriteResGroup119 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 15;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup119], (instregex "MUL_FI16m")>;
def: InstRW<[SBWriteResGroup119], (instregex "MUL_FI32m")>;

def SBWriteResGroup120 : SchedWriteRes<[SBPort0,SBPort1,SBPort5,SBPort23]> {
  let Latency = 15;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup120], (instregex "DPPDrmi")>;
def: InstRW<[SBWriteResGroup120], (instregex "VDPPDrmi")>;

def SBWriteResGroup121 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}
def: InstRW<[SBWriteResGroup121], (instregex "PCMPISTRIrm")>;
def: InstRW<[SBWriteResGroup121], (instregex "PCMPISTRM128rm")>;
def: InstRW<[SBWriteResGroup121], (instregex "VPCMPISTRIrm")>;
def: InstRW<[SBWriteResGroup121], (instregex "VPCMPISTRM128rm")>;

def SBWriteResGroup122 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 18;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup122], (instregex "AESIMCrm")>;
def: InstRW<[SBWriteResGroup122], (instregex "VAESIMCrm")>;

def SBWriteResGroup123 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup123], (instregex "DIVPSrm")>;
def: InstRW<[SBWriteResGroup123], (instregex "DIVSSrm")>;
def: InstRW<[SBWriteResGroup123], (instregex "SQRTPSm")>;
def: InstRW<[SBWriteResGroup123], (instregex "SQRTSSm")>;
def: InstRW<[SBWriteResGroup123], (instregex "VDIVPSrm")>;
def: InstRW<[SBWriteResGroup123], (instregex "VDIVSSrm")>;
def: InstRW<[SBWriteResGroup123], (instregex "VSQRTPSm")>;

def SBWriteResGroup124 : SchedWriteRes<[SBPort0]> {
  let Latency = 21;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup124], (instregex "VSQRTSDr")>;

def SBWriteResGroup125 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup125], (instregex "VSQRTSDm")>;

def SBWriteResGroup126 : SchedWriteRes<[SBPort0]> {
  let Latency = 22;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup126], (instregex "DIVPDrr")>;
def: InstRW<[SBWriteResGroup126], (instregex "DIVSDrr")>;
def: InstRW<[SBWriteResGroup126], (instregex "SQRTPDr")>;
def: InstRW<[SBWriteResGroup126], (instregex "SQRTSDr")>;
def: InstRW<[SBWriteResGroup126], (instregex "VDIVPDrr")>;
def: InstRW<[SBWriteResGroup126], (instregex "VDIVSDrr")>;
def: InstRW<[SBWriteResGroup126], (instregex "VSQRTPDr")>;

def SBWriteResGroup127 : SchedWriteRes<[SBPort0]> {
  let Latency = 24;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup127], (instregex "DIVR_FPrST0")>;
def: InstRW<[SBWriteResGroup127], (instregex "DIVR_FST0r")>;
def: InstRW<[SBWriteResGroup127], (instregex "DIVR_FrST0")>;
def: InstRW<[SBWriteResGroup127], (instregex "DIV_FPrST0")>;
def: InstRW<[SBWriteResGroup127], (instregex "DIV_FST0r")>;
def: InstRW<[SBWriteResGroup127], (instregex "DIV_FrST0")>;

def SBWriteResGroup128 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 28;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup128], (instregex "DIVPDrm")>;
def: InstRW<[SBWriteResGroup128], (instregex "DIVSDrm")>;
def: InstRW<[SBWriteResGroup128], (instregex "SQRTPDm")>;
def: InstRW<[SBWriteResGroup128], (instregex "SQRTSDm")>;
def: InstRW<[SBWriteResGroup128], (instregex "VDIVPDrm")>;
def: InstRW<[SBWriteResGroup128], (instregex "VDIVSDrm")>;
def: InstRW<[SBWriteResGroup128], (instregex "VSQRTPDm")>;

def SBWriteResGroup129 : SchedWriteRes<[SBPort0,SBPort05]> {
  let Latency = 29;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup129], (instregex "VDIVPSYrr")>;
def: InstRW<[SBWriteResGroup129], (instregex "VSQRTPSYr")>;

def SBWriteResGroup130 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 31;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup130], (instregex "DIVR_F32m")>;
def: InstRW<[SBWriteResGroup130], (instregex "DIVR_F64m")>;
def: InstRW<[SBWriteResGroup130], (instregex "DIV_F32m")>;
def: InstRW<[SBWriteResGroup130], (instregex "DIV_F64m")>;

def SBWriteResGroup131 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 34;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup131], (instregex "DIVR_FI16m")>;
def: InstRW<[SBWriteResGroup131], (instregex "DIVR_FI32m")>;
def: InstRW<[SBWriteResGroup131], (instregex "DIV_FI16m")>;
def: InstRW<[SBWriteResGroup131], (instregex "DIV_FI32m")>;

def SBWriteResGroup132 : SchedWriteRes<[SBPort0,SBPort23,SBPort05]> {
  let Latency = 36;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SBWriteResGroup132], (instregex "VDIVPSYrm")>;
def: InstRW<[SBWriteResGroup132], (instregex "VSQRTPSYm")>;

def SBWriteResGroup133 : SchedWriteRes<[SBPort0,SBPort05]> {
  let Latency = 45;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup133], (instregex "VDIVPDYrr")>;
def: InstRW<[SBWriteResGroup133], (instregex "VSQRTPDYr")>;

def SBWriteResGroup134 : SchedWriteRes<[SBPort0,SBPort23,SBPort05]> {
  let Latency = 52;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SBWriteResGroup134], (instregex "VDIVPDYrm")>;
def: InstRW<[SBWriteResGroup134], (instregex "VSQRTPDYm")>;

def SBWriteResGroup135 : SchedWriteRes<[SBPort0]> {
  let Latency = 114;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup135], (instregex "VSQRTSSr")>;

} // SchedModel
