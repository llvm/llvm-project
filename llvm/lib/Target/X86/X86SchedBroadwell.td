//=- X86SchedBroadwell.td - X86 Broadwell Scheduling ---------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Broadwell to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//
def BroadwellModel : SchedMachineModel {
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

let SchedModel = BroadwellModel in {

// Broadwell can issue micro-ops to 8 different ports in one cycle.

// Ports 0, 1, 5, and 6 handle all computation.
// Port 4 gets the data half of stores. Store data can be available later than
// the store address, but since we don't model the latency of stores, we can
// ignore that.
// Ports 2 and 3 are identical. They handle loads and the address half of
// stores. Port 7 can handle address calculations.
def BWPort0 : ProcResource<1>;
def BWPort1 : ProcResource<1>;
def BWPort2 : ProcResource<1>;
def BWPort3 : ProcResource<1>;
def BWPort4 : ProcResource<1>;
def BWPort5 : ProcResource<1>;
def BWPort6 : ProcResource<1>;
def BWPort7 : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def BWPort01  : ProcResGroup<[BWPort0, BWPort1]>;
def BWPort23  : ProcResGroup<[BWPort2, BWPort3]>;
def BWPort237 : ProcResGroup<[BWPort2, BWPort3, BWPort7]>;
def BWPort04  : ProcResGroup<[BWPort0, BWPort4]>;
def BWPort05  : ProcResGroup<[BWPort0, BWPort5]>;
def BWPort06  : ProcResGroup<[BWPort0, BWPort6]>;
def BWPort15  : ProcResGroup<[BWPort1, BWPort5]>;
def BWPort16  : ProcResGroup<[BWPort1, BWPort6]>;
def BWPort56  : ProcResGroup<[BWPort5, BWPort6]>;
def BWPort015 : ProcResGroup<[BWPort0, BWPort1, BWPort5]>;
def BWPort056 : ProcResGroup<[BWPort0, BWPort5, BWPort6]>;
def BWPort0156: ProcResGroup<[BWPort0, BWPort1, BWPort5, BWPort6]>;

// 60 Entry Unified Scheduler
def BWPortAny : ProcResGroup<[BWPort0, BWPort1, BWPort2, BWPort3, BWPort4,
                              BWPort5, BWPort6, BWPort7]> {
  let BufferSize=60;
}

// Integer division issued on port 0.
def BWDivider : ProcResource<1>; // Integer division issued on port 0.

// Loads are 5 cycles, so ReadAfterLd registers needn't be available until 5
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 5>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass BWWriteResPair<X86FoldableSchedWrite SchedRW,
                          list<ProcResourceKind> ExePorts,
                          int Lat, list<int> Res = [1], int UOps = 1> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on port 2/3 and adds 5 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, !listconcat([BWPort23], ExePorts)> {
    let Latency = !add(Lat, 5);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = UOps;
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [BWPort4]>;

// Arithmetic.
defm : BWWriteResPair<WriteALU,   [BWPort0156], 1>; // Simple integer ALU op.
defm : BWWriteResPair<WriteIMul,  [BWPort1],   3>; // Integer multiplication.
defm : BWWriteResPair<WriteIDiv, [BWPort0, BWDivider], 25, [1, 10]>;
def : WriteRes<WriteIMulH, []> { let Latency = 3; } // Integer multiplication, high part.

def : WriteRes<WriteLEA, [BWPort15]>; // LEA instructions can't fold loads.

// Integer shifts and rotates.
defm : BWWriteResPair<WriteShift, [BWPort06],  1>;

// Loads, stores, and moves, not folded with other operations.
def : WriteRes<WriteLoad,  [BWPort23]> { let Latency = 5; }
def : WriteRes<WriteStore, [BWPort237, BWPort4]>;
def : WriteRes<WriteMove,  [BWPort0156]>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
def : WriteRes<WriteZero,  []>;

// Treat misc copies as a move.
def : InstRW<[WriteMove], (instrs COPY)>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : BWWriteResPair<WriteJump,  [BWPort06],   1>;

// Floating point. This covers both scalar and vector operations.
def  : WriteRes<WriteFLoad,        [BWPort23]> { let Latency = 5; }
def  : WriteRes<WriteFStore,       [BWPort237, BWPort4]>;
def  : WriteRes<WriteFMove,        [BWPort5]>;

defm : BWWriteResPair<WriteFAdd,   [BWPort1], 3>; // Floating point add/sub/compare.
defm : BWWriteResPair<WriteFMul,   [BWPort0], 5>; // Floating point multiplication.
defm : BWWriteResPair<WriteFDiv,   [BWPort0], 12>; // 10-14 cycles. // Floating point division.
defm : BWWriteResPair<WriteFSqrt,  [BWPort0], 15>; // Floating point square root.
defm : BWWriteResPair<WriteFRcp,   [BWPort0], 5>; // Floating point reciprocal estimate.
defm : BWWriteResPair<WriteFRsqrt, [BWPort0], 5>; // Floating point reciprocal square root estimate.
defm : BWWriteResPair<WriteFMA,    [BWPort01], 5>; // Fused Multiply Add.
defm : BWWriteResPair<WriteFShuffle,  [BWPort5],  1>; // Floating point vector shuffles.
defm : BWWriteResPair<WriteFBlend,  [BWPort015],  1>; // Floating point vector blends.
defm : BWWriteResPair<WriteFVarBlend,  [HWPort5], 2, [2]>; // Fp vector variable blends.

// FMA Scheduling helper class.
// class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Vector integer operations.
def  : WriteRes<WriteVecLoad,        [BWPort23]> { let Latency = 5; }
def  : WriteRes<WriteVecStore,       [BWPort237, BWPort4]>;
def  : WriteRes<WriteVecMove,        [BWPort015]>;

defm : BWWriteResPair<WriteVecALU,   [BWPort15],  1>; // Vector integer ALU op, no logicals.
defm : BWWriteResPair<WriteVecShift, [BWPort0],  1>; // Vector integer shifts.
defm : BWWriteResPair<WriteVecIMul,  [BWPort0],   5>; // Vector integer multiply.
defm : BWWriteResPair<WriteShuffle,  [BWPort5],  1>; // Vector shuffles.
defm : BWWriteResPair<WriteBlend,  [BWPort15],  1>; // Vector blends.
defm : BWWriteResPair<WriteVarBlend,  [BWPort5], 2, [2]>; // Vector variable blends.
defm : BWWriteResPair<WriteMPSAD,  [BWPort0, BWPort5], 6, [1, 2]>; // Vector MPSAD.

// Vector bitwise operations.
// These are often used on both floating point and integer vectors.
defm : BWWriteResPair<WriteVecLogic, [BWPort015], 1>; // Vector and/or/xor.

// Conversion between integer and float.
defm : BWWriteResPair<WriteCvtF2I, [BWPort1], 3>; // Float -> Integer.
defm : BWWriteResPair<WriteCvtI2F, [BWPort1], 4>; // Integer -> Float.
defm : BWWriteResPair<WriteCvtF2F, [BWPort1], 3>; // Float -> Float size conversion.

// Strings instructions.

// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [BWPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [BWPort0, BWPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [BWPort0, BWPort5, BWPort015, BWPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def : WriteRes<WritePCmpEStrMLd, [BWPort0, BWPort5, BWPort23, BWPort015, BWPort0156]> {
  let Latency = 24;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [BWPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [BWPort0, BWPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [BWPort0, BWPort5, BWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def : WriteRes<WritePCmpEStrILd, [BWPort0, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}

// AES instructions.
def : WriteRes<WriteAESDecEnc, [BWPort5]> { // Decryption, encryption.
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [BWPort5, BWPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

def : WriteRes<WriteAESIMC, [BWPort5]> { // InvMixColumn.
  let Latency = 14;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [BWPort5, BWPort23]> {
  let Latency = 19;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}

def : WriteRes<WriteAESKeyGen, [BWPort0, BWPort5, BWPort015]> { // Key Generation.
  let Latency = 29;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,2];
}
def : WriteRes<WriteAESKeyGenLd, [BWPort0, BWPort5, BWPort23, BWPort015]> {
  let Latency = 33;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,1,1];
}

// Carry-less multiplication instructions.
defm : BWWriteResPair<WriteCLMul,  [BWPort0], 5>;

// Catch-all for expensive system instructions.
def : WriteRes<WriteSystem,     [BWPort0156]> { let Latency = 100; } // def WriteSystem : SchedWrite;

// AVX2.
defm : BWWriteResPair<WriteFShuffle256,  [BWPort5],  3>; // Fp 256-bit width vector shuffles.
defm : BWWriteResPair<WriteShuffle256,  [BWPort5],  3>;  // 256-bit width vector shuffles.
defm : BWWriteResPair<WriteVarVecShift,  [BWPort0, BWPort5], 2, [2, 1]>; // Variable vector shifts.

// Old microcoded instructions that nobody use.
def : WriteRes<WriteMicrocoded, [BWPort0156]> { let Latency = 100; } // def WriteMicrocoded : SchedWrite;

// Fence instructions.
def : WriteRes<WriteFence,  [BWPort23, BWPort4]>;

// Nop, not very useful expect it provides a model for nops!
def : WriteRes<WriteNop, []>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : BWWriteResPair<WriteFHAdd,   [BWPort1], 3>;
defm : BWWriteResPair<WritePHAdd,  [BWPort15], 1>;

// Remaining instrs.

def BWWriteResGroup1 : SchedWriteRes<[BWPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup1], (instregex "MMX_MOVD64from64rr",
                                           "MMX_MOVD64grr",
                                           "MMX_PMOVMSKBrr",
                                           "MMX_PSLLDri",
                                           "MMX_PSLLDrr",
                                           "MMX_PSLLQri",
                                           "MMX_PSLLQrr",
                                           "MMX_PSLLWri",
                                           "MMX_PSLLWrr",
                                           "MMX_PSRADri",
                                           "MMX_PSRADrr",
                                           "MMX_PSRAWri",
                                           "MMX_PSRAWrr",
                                           "MMX_PSRLDri",
                                           "MMX_PSRLDrr",
                                           "MMX_PSRLQri",
                                           "MMX_PSRLQrr",
                                           "MMX_PSRLWri",
                                           "MMX_PSRLWrr",
                                           "MOVPDI2DIrr",
                                           "MOVPQIto64rr",
                                           "PSLLDri",
                                           "PSLLQri",
                                           "PSLLWri",
                                           "PSRADri",
                                           "PSRAWri",
                                           "PSRLDri",
                                           "PSRLQri",
                                           "PSRLWri",
                                           "VMOVPDI2DIrr",
                                           "VMOVPQIto64rr",
                                           "VPSLLDYri",
                                           "VPSLLDri",
                                           "VPSLLQYri",
                                           "VPSLLQri",
                                           "VPSLLVQYrr",
                                           "VPSLLVQrr",
                                           "VPSLLWYri",
                                           "VPSLLWri",
                                           "VPSRADYri",
                                           "VPSRADri",
                                           "VPSRAWYri",
                                           "VPSRAWri",
                                           "VPSRLDYri",
                                           "VPSRLDri",
                                           "VPSRLQYri",
                                           "VPSRLQri",
                                           "VPSRLVQYrr",
                                           "VPSRLVQrr",
                                           "VPSRLWYri",
                                           "VPSRLWri",
                                           "VTESTPDYrr",
                                           "VTESTPDrr",
                                           "VTESTPSYrr",
                                           "VTESTPSrr")>;

def BWWriteResGroup2 : SchedWriteRes<[BWPort1]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup2], (instregex "COMP_FST0r",
                                           "COM_FST0r",
                                           "UCOM_FPr",
                                           "UCOM_Fr")>;

def BWWriteResGroup3 : SchedWriteRes<[BWPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup3], (instregex "ANDNPDrr",
                                           "ANDNPSrr",
                                           "ANDPDrr",
                                           "ANDPSrr",
                                           "INSERTPSrr",
                                           "MMX_MOVD64rr",
                                           "MMX_MOVD64to64rr",
                                           "MMX_MOVQ2DQrr",
                                           "MMX_PALIGNRrri",
                                           "MMX_PSHUFBrr",
                                           "MMX_PSHUFWri",
                                           "MMX_PUNPCKHBWirr",
                                           "MMX_PUNPCKHDQirr",
                                           "MMX_PUNPCKHWDirr",
                                           "MMX_PUNPCKLBWirr",
                                           "MMX_PUNPCKLDQirr",
                                           "MMX_PUNPCKLWDirr",
                                           "MOV64toPQIrr",
                                           "MOVAPDrr",
                                           "MOVAPSrr",
                                           "MOVDDUPrr",
                                           "MOVDI2PDIrr",
                                           "MOVHLPSrr",
                                           "MOVLHPSrr",
                                           "MOVSDrr",
                                           "MOVSHDUPrr",
                                           "MOVSLDUPrr",
                                           "MOVSSrr",
                                           "MOVUPDrr",
                                           "MOVUPSrr",
                                           "ORPDrr",
                                           "ORPSrr",
                                           "PACKSSDWrr",
                                           "PACKSSWBrr",
                                           "PACKUSDWrr",
                                           "PACKUSWBrr",
                                           "PALIGNRrri",
                                           "PBLENDWrri",
                                           "PMOVSXBDrr",
                                           "PMOVSXBQrr",
                                           "PMOVSXBWrr",
                                           "PMOVSXDQrr",
                                           "PMOVSXWDrr",
                                           "PMOVSXWQrr",
                                           "PMOVZXBDrr",
                                           "PMOVZXBQrr",
                                           "PMOVZXBWrr",
                                           "PMOVZXDQrr",
                                           "PMOVZXWDrr",
                                           "PMOVZXWQrr",
                                           "PSHUFBrr",
                                           "PSHUFDri",
                                           "PSHUFHWri",
                                           "PSHUFLWri",
                                           "PSLLDQri",
                                           "PSRLDQri",
                                           "PUNPCKHBWrr",
                                           "PUNPCKHDQrr",
                                           "PUNPCKHQDQrr",
                                           "PUNPCKHWDrr",
                                           "PUNPCKLBWrr",
                                           "PUNPCKLDQrr",
                                           "PUNPCKLQDQrr",
                                           "PUNPCKLWDrr",
                                           "SHUFPDrri",
                                           "SHUFPSrri",
                                           "UNPCKHPDrr",
                                           "UNPCKHPSrr",
                                           "UNPCKLPDrr",
                                           "UNPCKLPSrr",
                                           "VANDNPDYrr",
                                           "VANDNPDrr",
                                           "VANDNPSYrr",
                                           "VANDNPSrr",
                                           "VANDPDYrr",
                                           "VANDPDrr",
                                           "VANDPSYrr",
                                           "VANDPSrr",
                                           "VBROADCASTSSrr",
                                           "VINSERTPSrr",
                                           "VMOV64toPQIrr",
                                           "VMOVAPDYrr",
                                           "VMOVAPDrr",
                                           "VMOVAPSYrr",
                                           "VMOVAPSrr",
                                           "VMOVDDUPYrr",
                                           "VMOVDDUPrr",
                                           "VMOVDI2PDIrr",
                                           "VMOVHLPSrr",
                                           "VMOVLHPSrr",
                                           "VMOVSDrr",
                                           "VMOVSHDUPYrr",
                                           "VMOVSHDUPrr",
                                           "VMOVSLDUPYrr",
                                           "VMOVSLDUPrr",
                                           "VMOVSSrr",
                                           "VMOVUPDYrr",
                                           "VMOVUPDrr",
                                           "VMOVUPSYrr",
                                           "VMOVUPSrr",
                                           "VORPDYrr",
                                           "VORPDrr",
                                           "VORPSYrr",
                                           "VORPSrr",
                                           "VPACKSSDWYrr",
                                           "VPACKSSDWrr",
                                           "VPACKSSWBYrr",
                                           "VPACKSSWBrr",
                                           "VPACKUSDWYrr",
                                           "VPACKUSDWrr",
                                           "VPACKUSWBYrr",
                                           "VPACKUSWBrr",
                                           "VPALIGNRYrri",
                                           "VPALIGNRrri",
                                           "VPBLENDWYrri",
                                           "VPBLENDWrri",
                                           "VPBROADCASTDrr",
                                           "VPBROADCASTQrr",
                                           "VPERMILPDYri",
                                           "VPERMILPDYrr",
                                           "VPERMILPDri",
                                           "VPERMILPDrr",
                                           "VPERMILPSYri",
                                           "VPERMILPSYrr",
                                           "VPERMILPSri",
                                           "VPERMILPSrr",
                                           "VPMOVSXBDrr",
                                           "VPMOVSXBQrr",
                                           "VPMOVSXBWrr",
                                           "VPMOVSXDQrr",
                                           "VPMOVSXWDrr",
                                           "VPMOVSXWQrr",
                                           "VPMOVZXBDrr",
                                           "VPMOVZXBQrr",
                                           "VPMOVZXBWrr",
                                           "VPMOVZXDQrr",
                                           "VPMOVZXWDrr",
                                           "VPMOVZXWQrr",
                                           "VPSHUFBYrr",
                                           "VPSHUFBrr",
                                           "VPSHUFDYri",
                                           "VPSHUFDri",
                                           "VPSHUFHWYri",
                                           "VPSHUFHWri",
                                           "VPSHUFLWYri",
                                           "VPSHUFLWri",
                                           "VPSLLDQYri",
                                           "VPSLLDQri",
                                           "VPSRLDQYri",
                                           "VPSRLDQri",
                                           "VPUNPCKHBWYrr",
                                           "VPUNPCKHBWrr",
                                           "VPUNPCKHDQYrr",
                                           "VPUNPCKHDQrr",
                                           "VPUNPCKHQDQYrr",
                                           "VPUNPCKHQDQrr",
                                           "VPUNPCKHWDYrr",
                                           "VPUNPCKHWDrr",
                                           "VPUNPCKLBWYrr",
                                           "VPUNPCKLBWrr",
                                           "VPUNPCKLDQYrr",
                                           "VPUNPCKLDQrr",
                                           "VPUNPCKLQDQYrr",
                                           "VPUNPCKLQDQrr",
                                           "VPUNPCKLWDYrr",
                                           "VPUNPCKLWDrr",
                                           "VSHUFPDYrri",
                                           "VSHUFPDrri",
                                           "VSHUFPSYrri",
                                           "VSHUFPSrri",
                                           "VUNPCKHPDYrr",
                                           "VUNPCKHPDrr",
                                           "VUNPCKHPSYrr",
                                           "VUNPCKHPSrr",
                                           "VUNPCKLPDYrr",
                                           "VUNPCKLPDrr",
                                           "VUNPCKLPSYrr",
                                           "VUNPCKLPSrr",
                                           "VXORPDYrr",
                                           "VXORPDrr",
                                           "VXORPSYrr",
                                           "VXORPSrr",
                                           "XORPDrr",
                                           "XORPSrr")>;

def BWWriteResGroup4 : SchedWriteRes<[BWPort6]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup4], (instregex "JMP(16|32|64)r")>;

def BWWriteResGroup5 : SchedWriteRes<[BWPort01]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup5], (instregex "FINCSTP",
                                           "FNOP")>;

def BWWriteResGroup6 : SchedWriteRes<[BWPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup6], (instregex "ADC(16|32|64)ri",
                                           "ADC(16|32|64)i",
                                           "ADC(8|16|32|64)rr",
                                           "ADCX(32|64)rr",
                                           "ADOX(32|64)rr",
                                           "BT(16|32|64)ri8",
                                           "BT(16|32|64)rr",
                                           "BTC(16|32|64)ri8",
                                           "BTC(16|32|64)rr",
                                           "BTR(16|32|64)ri8",
                                           "BTR(16|32|64)rr",
                                           "BTS(16|32|64)ri8",
                                           "BTS(16|32|64)rr",
                                           "CDQ",
                                           "CMOV(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)(16|32|64)rr",
                                           "CQO",
                                           "J(A|AE|B|BE|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)_1",
                                           "J(A|AE|B|BE|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)_4",
                                           "JMP_1",
                                           "JMP_4",
                                           "RORX(32|64)ri",
                                           "SAR(8|16|32|64)r1",
                                           "SAR(8|16|32|64)ri",
                                           "SARX(32|64)rr",
                                           "SBB(16|32|64)ri",
                                           "SBB(16|32|64)i",
                                           "SBB(8|16|32|64)rr",
                                           "SET(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)r",
                                           "SHL(8|16|32|64)r1",
                                           "SHL(8|16|32|64)ri",
                                           "SHLX(32|64)rr",
                                           "SHR(8|16|32|64)r1",
                                           "SHR(8|16|32|64)ri",
                                           "SHRX(32|64)rr")>;

def BWWriteResGroup7 : SchedWriteRes<[BWPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup7], (instregex "ANDN(32|64)rr",
                                           "BLSI(32|64)rr",
                                           "BLSMSK(32|64)rr",
                                           "BLSR(32|64)rr",
                                           "BZHI(32|64)rr",
                                           "LEA(16|32|64)(_32)?r",
                                           "MMX_PABSBrr",
                                           "MMX_PABSDrr",
                                           "MMX_PABSWrr",
                                           "MMX_PADDBirr",
                                           "MMX_PADDDirr",
                                           "MMX_PADDQirr",
                                           "MMX_PADDSBirr",
                                           "MMX_PADDSWirr",
                                           "MMX_PADDUSBirr",
                                           "MMX_PADDUSWirr",
                                           "MMX_PADDWirr",
                                           "MMX_PAVGBirr",
                                           "MMX_PAVGWirr",
                                           "MMX_PCMPEQBirr",
                                           "MMX_PCMPEQDirr",
                                           "MMX_PCMPEQWirr",
                                           "MMX_PCMPGTBirr",
                                           "MMX_PCMPGTDirr",
                                           "MMX_PCMPGTWirr",
                                           "MMX_PMAXSWirr",
                                           "MMX_PMAXUBirr",
                                           "MMX_PMINSWirr",
                                           "MMX_PMINUBirr",
                                           "MMX_PSIGNBrr",
                                           "MMX_PSIGNDrr",
                                           "MMX_PSIGNWrr",
                                           "MMX_PSUBBirr",
                                           "MMX_PSUBDirr",
                                           "MMX_PSUBQirr",
                                           "MMX_PSUBSBirr",
                                           "MMX_PSUBSWirr",
                                           "MMX_PSUBUSBirr",
                                           "MMX_PSUBUSWirr",
                                           "MMX_PSUBWirr",
                                           "PABSBrr",
                                           "PABSDrr",
                                           "PABSWrr",
                                           "PADDBrr",
                                           "PADDDrr",
                                           "PADDQrr",
                                           "PADDSBrr",
                                           "PADDSWrr",
                                           "PADDUSBrr",
                                           "PADDUSWrr",
                                           "PADDWrr",
                                           "PAVGBrr",
                                           "PAVGWrr",
                                           "PCMPEQBrr",
                                           "PCMPEQDrr",
                                           "PCMPEQQrr",
                                           "PCMPEQWrr",
                                           "PCMPGTBrr",
                                           "PCMPGTDrr",
                                           "PCMPGTWrr",
                                           "PMAXSBrr",
                                           "PMAXSDrr",
                                           "PMAXSWrr",
                                           "PMAXUBrr",
                                           "PMAXUDrr",
                                           "PMAXUWrr",
                                           "PMINSBrr",
                                           "PMINSDrr",
                                           "PMINSWrr",
                                           "PMINUBrr",
                                           "PMINUDrr",
                                           "PMINUWrr",
                                           "PSIGNBrr",
                                           "PSIGNDrr",
                                           "PSIGNWrr",
                                           "PSUBBrr",
                                           "PSUBDrr",
                                           "PSUBQrr",
                                           "PSUBSBrr",
                                           "PSUBSWrr",
                                           "PSUBUSBrr",
                                           "PSUBUSWrr",
                                           "PSUBWrr",
                                           "VPABSBYrr",
                                           "VPABSBrr",
                                           "VPABSDYrr",
                                           "VPABSDrr",
                                           "VPABSWYrr",
                                           "VPABSWrr",
                                           "VPADDBYrr",
                                           "VPADDBrr",
                                           "VPADDDYrr",
                                           "VPADDDrr",
                                           "VPADDQYrr",
                                           "VPADDQrr",
                                           "VPADDSBYrr",
                                           "VPADDSBrr",
                                           "VPADDSWYrr",
                                           "VPADDSWrr",
                                           "VPADDUSBYrr",
                                           "VPADDUSBrr",
                                           "VPADDUSWYrr",
                                           "VPADDUSWrr",
                                           "VPADDWYrr",
                                           "VPADDWrr",
                                           "VPAVGBYrr",
                                           "VPAVGBrr",
                                           "VPAVGWYrr",
                                           "VPAVGWrr",
                                           "VPCMPEQBYrr",
                                           "VPCMPEQBrr",
                                           "VPCMPEQDYrr",
                                           "VPCMPEQDrr",
                                           "VPCMPEQQYrr",
                                           "VPCMPEQQrr",
                                           "VPCMPEQWYrr",
                                           "VPCMPEQWrr",
                                           "VPCMPGTBYrr",
                                           "VPCMPGTBrr",
                                           "VPCMPGTDYrr",
                                           "VPCMPGTDrr",
                                           "VPCMPGTWYrr",
                                           "VPCMPGTWrr",
                                           "VPMAXSBYrr",
                                           "VPMAXSBrr",
                                           "VPMAXSDYrr",
                                           "VPMAXSDrr",
                                           "VPMAXSWYrr",
                                           "VPMAXSWrr",
                                           "VPMAXUBYrr",
                                           "VPMAXUBrr",
                                           "VPMAXUDYrr",
                                           "VPMAXUDrr",
                                           "VPMAXUWYrr",
                                           "VPMAXUWrr",
                                           "VPMINSBYrr",
                                           "VPMINSBrr",
                                           "VPMINSDYrr",
                                           "VPMINSDrr",
                                           "VPMINSWYrr",
                                           "VPMINSWrr",
                                           "VPMINUBYrr",
                                           "VPMINUBrr",
                                           "VPMINUDYrr",
                                           "VPMINUDrr",
                                           "VPMINUWYrr",
                                           "VPMINUWrr",
                                           "VPSIGNBYrr",
                                           "VPSIGNBrr",
                                           "VPSIGNDYrr",
                                           "VPSIGNDrr",
                                           "VPSIGNWYrr",
                                           "VPSIGNWrr",
                                           "VPSUBBYrr",
                                           "VPSUBBrr",
                                           "VPSUBDYrr",
                                           "VPSUBDrr",
                                           "VPSUBQYrr",
                                           "VPSUBQrr",
                                           "VPSUBSBYrr",
                                           "VPSUBSBrr",
                                           "VPSUBSWYrr",
                                           "VPSUBSWrr",
                                           "VPSUBUSBYrr",
                                           "VPSUBUSBrr",
                                           "VPSUBUSWYrr",
                                           "VPSUBUSWrr",
                                           "VPSUBWYrr",
                                           "VPSUBWrr")>;

def BWWriteResGroup8 : SchedWriteRes<[BWPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup8], (instregex "BLENDPDrri",
                                           "BLENDPSrri",
                                           "MMX_MOVQ64rr",
                                           "MMX_PANDNirr",
                                           "MMX_PANDirr",
                                           "MMX_PORirr",
                                           "MMX_PXORirr",
                                           "MOVDQArr",
                                           "MOVDQUrr",
                                           "MOVPQI2QIrr",
                                           "PANDNrr",
                                           "PANDrr",
                                           "PORrr",
                                           "PXORrr",
                                           "VBLENDPDYrri",
                                           "VBLENDPDrri",
                                           "VBLENDPSYrri",
                                           "VBLENDPSrri",
                                           "VMOVDQAYrr",
                                           "VMOVDQArr",
                                           "VMOVDQUYrr",
                                           "VMOVDQUrr",
                                           "VMOVPQI2QIrr",
                                           "VMOVZPQILo2PQIrr",
                                           "VPANDNYrr",
                                           "VPANDNrr",
                                           "VPANDYrr",
                                           "VPANDrr",
                                           "VPBLENDDYrri",
                                           "VPBLENDDrri",
                                           "VPORYrr",
                                           "VPORrr",
                                           "VPXORYrr",
                                           "VPXORrr")>;

def BWWriteResGroup9 : SchedWriteRes<[BWPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup9], (instrs CWDE)>;
def: InstRW<[BWWriteResGroup9], (instregex "ADD(8|16|32|64)ri",
                                           "ADD(8|16|32|64)rr",
                                           "ADD(8|16|32|64)i",
                                           "AND(8|16|32|64)ri",
                                           "AND(8|16|32|64)rr",
                                           "AND(8|16|32|64)i",
                                           "CBW",
                                           "CLC",
                                           "CMC",
                                           "CMP(8|16|32|64)ri",
                                           "CMP(8|16|32|64)rr",
                                           "CMP(8|16|32|64)i",
                                           "DEC(8|16|32|64)r",
                                           "INC(8|16|32|64)r",
                                           "LAHF",
                                           "MOV(8|16|32|64)rr",
                                           "MOV(8|16|32|64)ri",
                                           "MOVSX(16|32|64)rr16",
                                           "MOVSX(16|32|64)rr32",
                                           "MOVSX(16|32|64)rr8",
                                           "MOVZX(16|32|64)rr16",
                                           "MOVZX(16|32|64)rr8",
                                           "NEG(8|16|32|64)r",
                                           "NOOP",
                                           "NOT(8|16|32|64)r",
                                           "OR(8|16|32|64)ri",
                                           "OR(8|16|32|64)rr",
                                           "OR(8|16|32|64)i",
                                           "SAHF",
                                           "SGDT64m",
                                           "SIDT64m",
                                           "SLDT64m",
                                           "SMSW16m",
                                           "STC",
                                           "STRm",
                                           "SUB(8|16|32|64)ri",
                                           "SUB(8|16|32|64)rr",
                                           "SUB(8|16|32|64)i",
                                           "SYSCALL",
                                           "TEST(8|16|32|64)rr",
                                           "TEST(8|16|32|64)i",
                                           "TEST(8|16|32|64)ri",
                                           "XCHG(16|32|64)rr",
                                           "XOR(8|16|32|64)ri",
                                           "XOR(8|16|32|64)rr",
                                           "XOR(8|16|32|64)i")>;

def BWWriteResGroup10 : SchedWriteRes<[BWPort4,BWPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup10], (instregex "FBSTPm",
                                            "MMX_MOVD64from64rm",
                                            "MMX_MOVD64mr",
                                            "MMX_MOVNTQmr",
                                            "MMX_MOVQ64mr",
                                            "MOV(16|32|64)mr",
                                            "MOV8mi",
                                            "MOV8mr",
                                            "MOVAPDmr",
                                            "MOVAPSmr",
                                            "MOVDQAmr",
                                            "MOVDQUmr",
                                            "MOVHPDmr",
                                            "MOVHPSmr",
                                            "MOVLPDmr",
                                            "MOVLPSmr",
                                            "MOVNTDQmr",
                                            "MOVNTI_64mr",
                                            "MOVNTImr",
                                            "MOVNTPDmr",
                                            "MOVNTPSmr",
                                            "MOVPDI2DImr",
                                            "MOVPQI2QImr",
                                            "MOVPQIto64mr",
                                            "MOVSDmr",
                                            "MOVSSmr",
                                            "MOVUPDmr",
                                            "MOVUPSmr",
                                            "ST_FP32m",
                                            "ST_FP64m",
                                            "ST_FP80m",
                                            "VEXTRACTF128mr",
                                            "VEXTRACTI128mr",
                                            "VMOVAPDYmr",
                                            "VMOVAPDmr",
                                            "VMOVAPSYmr",
                                            "VMOVAPSmr",
                                            "VMOVDQAYmr",
                                            "VMOVDQAmr",
                                            "VMOVDQUYmr",
                                            "VMOVDQUmr",
                                            "VMOVHPDmr",
                                            "VMOVHPSmr",
                                            "VMOVLPDmr",
                                            "VMOVLPSmr",
                                            "VMOVNTDQYmr",
                                            "VMOVNTDQmr",
                                            "VMOVNTPDYmr",
                                            "VMOVNTPDmr",
                                            "VMOVNTPSYmr",
                                            "VMOVNTPSmr",
                                            "VMOVPDI2DImr",
                                            "VMOVPQI2QImr",
                                            "VMOVPQIto64mr",
                                            "VMOVSDmr",
                                            "VMOVSSmr",
                                            "VMOVUPDYmr",
                                            "VMOVUPDmr",
                                            "VMOVUPSYmr",
                                            "VMOVUPSmr")>;

def BWWriteResGroup11 : SchedWriteRes<[BWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup11], (instregex "BLENDVPDrr0",
                                            "BLENDVPSrr0",
                                            "MMX_PINSRWrr",
                                            "PBLENDVBrr0",
                                            "PINSRBrr",
                                            "PINSRDrr",
                                            "PINSRQrr",
                                            "PINSRWrr",
                                            "VBLENDVPDYrr",
                                            "VBLENDVPDrr",
                                            "VBLENDVPSYrr",
                                            "VBLENDVPSrr",
                                            "VPBLENDVBYrr",
                                            "VPBLENDVBrr",
                                            "VPINSRBrr",
                                            "VPINSRDrr",
                                            "VPINSRQrr",
                                            "VPINSRWrr")>;

def BWWriteResGroup12 : SchedWriteRes<[BWPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup12], (instregex "FDECSTP")>;

def BWWriteResGroup13 : SchedWriteRes<[BWPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup13], (instregex "ROL(8|16|32|64)r1",
                                            "ROL(8|16|32|64)ri",
                                            "ROR(8|16|32|64)r1",
                                            "ROR(8|16|32|64)ri")>;

def BWWriteResGroup14 : SchedWriteRes<[BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup14], (instregex "LFENCE",
                                            "MFENCE",
                                            "WAIT",
                                            "XGETBV")>;

def BWWriteResGroup15 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup15], (instregex "CVTPS2PDrr",
                                            "CVTSS2SDrr",
                                            "EXTRACTPSrr",
                                            "MMX_PEXTRWrr",
                                            "PEXTRBrr",
                                            "PEXTRDrr",
                                            "PEXTRQrr",
                                            "PEXTRWrr",
                                            "PSLLDrr",
                                            "PSLLQrr",
                                            "PSLLWrr",
                                            "PSRADrr",
                                            "PSRAWrr",
                                            "PSRLDrr",
                                            "PSRLQrr",
                                            "PSRLWrr",
                                            "PTESTrr",
                                            "VCVTPH2PSYrr",
                                            "VCVTPH2PSrr",
                                            "VCVTPS2PDrr",
                                            "VCVTSS2SDrr",
                                            "VEXTRACTPSrr",
                                            "VPEXTRBrr",
                                            "VPEXTRDrr",
                                            "VPEXTRQrr",
                                            "VPEXTRWrr",
                                            "VPSLLDrr",
                                            "VPSLLQrr",
                                            "VPSLLWrr",
                                            "VPSRADrr",
                                            "VPSRAWrr",
                                            "VPSRLDrr",
                                            "VPSRLQrr",
                                            "VPSRLWrr",
                                            "VPTESTrr")>;

def BWWriteResGroup16 : SchedWriteRes<[BWPort6,BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup16], (instregex "CLFLUSH")>;

def BWWriteResGroup17 : SchedWriteRes<[BWPort01,BWPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup17], (instregex "MMX_MOVDQ2Qrr")>;

def BWWriteResGroup18 : SchedWriteRes<[BWPort237,BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup18], (instregex "SFENCE")>;

def BWWriteResGroup19 : SchedWriteRes<[BWPort06,BWPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup19], (instregex "BEXTR(32|64)rr",
                                            "BSWAP(16|32|64)r")>;

def BWWriteResGroup20 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup20], (instrs CWD)>;
def: InstRW<[BWWriteResGroup20], (instrs JCXZ, JECXZ, JRCXZ)>;
def: InstRW<[BWWriteResGroup20], (instregex "ADC8i8",
                                            "ADC8ri",
                                            "CMOV(A|BE)(16|32|64)rr",
                                            "SBB8i8",
                                            "SBB8ri",
                                            "SET(A|BE)r")>;

def BWWriteResGroup21 : SchedWriteRes<[BWPort4,BWPort5,BWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup21], (instregex "EXTRACTPSmr",
                                            "PEXTRBmr",
                                            "PEXTRDmr",
                                            "PEXTRQmr",
                                            "PEXTRWmr",
                                            "STMXCSR",
                                            "VEXTRACTPSmr",
                                            "VPEXTRBmr",
                                            "VPEXTRDmr",
                                            "VPEXTRQmr",
                                            "VPEXTRWmr",
                                            "VSTMXCSR")>;

def BWWriteResGroup22 : SchedWriteRes<[BWPort4,BWPort6,BWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup22], (instregex "FNSTCW16m")>;

def BWWriteResGroup23 : SchedWriteRes<[BWPort4,BWPort237,BWPort06]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup23], (instregex "SET(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)m")>;

def BWWriteResGroup24 : SchedWriteRes<[BWPort4,BWPort237,BWPort15]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup24], (instregex "MOVBE(16|32|64)mr")>;

def BWWriteResGroup25 : SchedWriteRes<[BWPort4,BWPort237,BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup25], (instrs PUSH16r, PUSH32r, PUSH64r)>;
def: InstRW<[BWWriteResGroup25], (instregex "PUSH(16|32|64)rmr",
                                            "PUSH64i8",
                                            "STOSB",
                                            "STOSL",
                                            "STOSQ",
                                            "STOSW")>;

def BWWriteResGroup26 : SchedWriteRes<[BWPort0]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup26], (instregex "MOVMSKPDrr",
                                            "MOVMSKPSrr",
                                            "PMOVMSKBrr",
                                            "VMOVMSKPDYrr",
                                            "VMOVMSKPDrr",
                                            "VMOVMSKPSYrr",
                                            "VMOVMSKPSrr",
                                            "VPMOVMSKBYrr",
                                            "VPMOVMSKBrr")>;

def BWWriteResGroup27 : SchedWriteRes<[BWPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup27], (instrs IMUL16rr, IMUL32rr, IMUL32rri, IMUL32rri8, IMUL64rr, IMUL64rri32, IMUL64rri8)>;
def: InstRW<[BWWriteResGroup27], (instrs IMUL8r, MUL8r)>;
def: InstRW<[BWWriteResGroup27], (instregex "ADDPDrr",
                                            "ADDPSrr",
                                            "ADDSDrr",
                                            "ADDSSrr",
                                            "ADDSUBPDrr",
                                            "ADDSUBPSrr",
                                            "ADD_FPrST0",
                                            "ADD_FST0r",
                                            "ADD_FrST0",
                                            "BSF(16|32|64)rr",
                                            "BSR(16|32|64)rr",
                                            "CMPPDrri",
                                            "CMPPSrri",
                                            "CMPSDrr",
                                            "CMPSSrr",
                                            "COMISDrr",
                                            "COMISSrr",
                                            "CVTDQ2PSrr",
                                            "CVTPS2DQrr",
                                            "CVTTPS2DQrr",
                                            "LZCNT(16|32|64)rr",
                                            "MAX(C?)PDrr",
                                            "MAX(C?)PSrr",
                                            "MAX(C?)SDrr",
                                            "MAX(C?)SSrr",
                                            "MIN(C?)PDrr",
                                            "MIN(C?)PSrr",
                                            "MIN(C?)SDrr",
                                            "MIN(C?)SSrr",
                                            "MMX_CVTPI2PSirr",
                                            "PDEP(32|64)rr",
                                            "PEXT(32|64)rr",
                                            "POPCNT(16|32|64)rr",
                                            "SHLD(16|32|64)rri8",
                                            "SHRD(16|32|64)rri8",
                                            "SUBPDrr",
                                            "SUBPSrr",
                                            "SUBR_FPrST0",
                                            "SUBR_FST0r",
                                            "SUBR_FrST0",
                                            "SUBSDrr",
                                            "SUBSSrr",
                                            "SUB_FPrST0",
                                            "SUB_FST0r",
                                            "SUB_FrST0",
                                            "TZCNT(16|32|64)rr",
                                            "UCOMISDrr",
                                            "UCOMISSrr",
                                            "VADDPDYrr",
                                            "VADDPDrr",
                                            "VADDPSYrr",
                                            "VADDPSrr",
                                            "VADDSDrr",
                                            "VADDSSrr",
                                            "VADDSUBPDYrr",
                                            "VADDSUBPDrr",
                                            "VADDSUBPSYrr",
                                            "VADDSUBPSrr",
                                            "VCMPPDYrri",
                                            "VCMPPDrri",
                                            "VCMPPSYrri",
                                            "VCMPPSrri",
                                            "VCMPSDrr",
                                            "VCMPSSrr",
                                            "VCOMISDrr",
                                            "VCOMISSrr",
                                            "VCVTDQ2PSYrr",
                                            "VCVTDQ2PSrr",
                                            "VCVTPS2DQYrr",
                                            "VCVTPS2DQrr",
                                            "VCVTTPS2DQYrr",
                                            "VCVTTPS2DQrr",
                                            "VMAX(C?)PDYrr",
                                            "VMAX(C?)PDrr",
                                            "VMAX(C?)PSYrr",
                                            "VMAX(C?)PSrr",
                                            "VMAX(C?)SDrr",
                                            "VMAX(C?)SSrr",
                                            "VMIN(C?)PDYrr",
                                            "VMIN(C?)PDrr",
                                            "VMIN(C?)PSYrr",
                                            "VMIN(C?)PSrr",
                                            "VMIN(C?)SDrr",
                                            "VMIN(C?)SSrr",
                                            "VSUBPDYrr",
                                            "VSUBPDrr",
                                            "VSUBPSYrr",
                                            "VSUBPSrr",
                                            "VSUBSDrr",
                                            "VSUBSSrr",
                                            "VUCOMISDrr",
                                            "VUCOMISSrr")>;

def BWWriteResGroup27_16 : SchedWriteRes<[BWPort1, BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup27_16], (instrs IMUL16rri, IMUL16rri8)>;

def BWWriteResGroup28 : SchedWriteRes<[BWPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup28], (instregex "VBROADCASTSDYrr",
                                            "VBROADCASTSSYrr",
                                            "VEXTRACTF128rr",
                                            "VEXTRACTI128rr",
                                            "VINSERTF128rr",
                                            "VINSERTI128rr",
                                            "VPBROADCASTBYrr",
                                            "VPBROADCASTBrr",
                                            "VPBROADCASTDYrr",
                                            "VPBROADCASTQYrr",
                                            "VPBROADCASTWYrr",
                                            "VPBROADCASTWrr",
                                            "VPERM2F128rr",
                                            "VPERM2I128rr",
                                            "VPERMDYrr",
                                            "VPERMPDYri",
                                            "VPERMPSYrr",
                                            "VPERMQYri",
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

def BWWriteResGroup29 : SchedWriteRes<[BWPort01]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup29], (instregex "MULPDrr",
                                            "MULPSrr",
                                            "MULSDrr",
                                            "MULSSrr",
                                            "VMULPDYrr",
                                            "VMULPDrr",
                                            "VMULPSYrr",
                                            "VMULPSrr",
                                            "VMULSDrr",
                                            "VMULSSrr")>;

def BWWriteResGroup30 : SchedWriteRes<[BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[BWWriteResGroup30], (instregex "XADD(8|16|32|64)rr",
                                            "XCHG8rr")>;

def BWWriteResGroup31 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup31], (instregex "VPSLLVDYrr",
                                            "VPSLLVDrr",
                                            "VPSRAVDYrr",
                                            "VPSRAVDrr",
                                            "VPSRLVDYrr",
                                            "VPSRLVDrr")>;

def BWWriteResGroup32 : SchedWriteRes<[BWPort5,BWPort15]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup32], (instregex "MMX_PHADDDrr",
                                            "MMX_PHADDSWrr",
                                            "MMX_PHADDWrr",
                                            "MMX_PHSUBDrr",
                                            "MMX_PHSUBSWrr",
                                            "MMX_PHSUBWrr",
                                            "PHADDDrr",
                                            "PHADDSWrr",
                                            "PHADDWrr",
                                            "PHSUBDrr",
                                            "PHSUBSWrr",
                                            "PHSUBWrr",
                                            "VPHADDDYrr",
                                            "VPHADDDrr",
                                            "VPHADDSWYrr",
                                            "VPHADDSWrr",
                                            "VPHADDWYrr",
                                            "VPHADDWrr",
                                            "VPHSUBDYrr",
                                            "VPHSUBDrr",
                                            "VPHSUBSWYrr",
                                            "VPHSUBSWrr",
                                            "VPHSUBWYrr",
                                            "VPHSUBWrr")>;

def BWWriteResGroup33 : SchedWriteRes<[BWPort5,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup33], (instregex "MMX_PACKSSDWirr",
                                            "MMX_PACKSSWBirr",
                                            "MMX_PACKUSWBirr")>;

def BWWriteResGroup34 : SchedWriteRes<[BWPort6,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup34], (instregex "CLD")>;

def BWWriteResGroup35 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup35], (instregex "RCL(8|16|32|64)r1",
                                            "RCL(8|16|32|64)ri",
                                            "RCR(8|16|32|64)r1",
                                            "RCR(8|16|32|64)ri")>;

def BWWriteResGroup36 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup36], (instregex "ROL(8|16|32|64)rCL",
                                            "ROR(8|16|32|64)rCL",
                                            "SAR(8|16|32|64)rCL",
                                            "SHL(8|16|32|64)rCL",
                                            "SHR(8|16|32|64)rCL")>;

def BWWriteResGroup37 : SchedWriteRes<[BWPort4,BWPort6,BWPort237,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup37], (instregex "CALL(16|32|64)r")>;

def BWWriteResGroup38 : SchedWriteRes<[BWPort4,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup38], (instregex "CALL64pcrel32",
                                            "SET(A|BE)m")>;

def BWWriteResGroup39 : SchedWriteRes<[BWPort0,BWPort1]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup39], (instregex "CVTSD2SI64rr",
                                            "CVTSD2SIrr",
                                            "CVTSS2SI64rr",
                                            "CVTSS2SIrr",
                                            "CVTTSD2SI64rr",
                                            "CVTTSD2SIrr",
                                            "CVTTSS2SI64rr",
                                            "CVTTSS2SIrr",
                                            "VCVTSD2SI64rr",
                                            "VCVTSD2SIrr",
                                            "VCVTSS2SI64rr",
                                            "VCVTSS2SIrr",
                                            "VCVTTSD2SI64rr",
                                            "VCVTTSD2SIrr",
                                            "VCVTTSS2SI64rr",
                                            "VCVTTSS2SIrr")>;

def BWWriteResGroup40 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup40], (instregex "VCVTPS2PDYrr",
                                            "VPSLLDYrr",
                                            "VPSLLQYrr",
                                            "VPSLLWYrr",
                                            "VPSRADYrr",
                                            "VPSRAWYrr",
                                            "VPSRLDYrr",
                                            "VPSRLQYrr",
                                            "VPSRLWYrr",
                                            "VPTESTYrr")>;

def BWWriteResGroup41 : SchedWriteRes<[BWPort0,BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup41], (instregex "FNSTSW16r")>;

def BWWriteResGroup42 : SchedWriteRes<[BWPort1,BWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup42], (instrs IMUL64r, MUL64r, MULX64rr)>;
def: InstRW<[BWWriteResGroup42], (instregex "CVTDQ2PDrr",
                                            "CVTPD2DQrr",
                                            "CVTPD2PSrr",
                                            "CVTSD2SSrr",
                                            "CVTSI642SDrr",
                                            "CVTSI2SDrr",
                                            "CVTSI2SSrr",
                                            "CVTTPD2DQrr",
                                            "MMX_CVTPD2PIirr",
                                            "MMX_CVTPI2PDirr",
                                            "MMX_CVTPS2PIirr",
                                            "MMX_CVTTPD2PIirr",
                                            "MMX_CVTTPS2PIirr",
                                            "VCVTDQ2PDrr",
                                            "VCVTPD2DQrr",
                                            "VCVTPD2PSrr",
                                            "VCVTPS2PHrr",
                                            "VCVTSD2SSrr",
                                            "VCVTSI642SDrr",
                                            "VCVTSI2SDrr",
                                            "VCVTSI2SSrr",
                                            "VCVTTPD2DQrr")>;

def BWWriteResGroup42_16 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def: InstRW<[BWWriteResGroup42_16], (instrs IMUL16r, MUL16r)>;

def BWWriteResGroup43 : SchedWriteRes<[BWPort0,BWPort4,BWPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup43], (instregex "FNSTSWm")>;

def BWWriteResGroup44 : SchedWriteRes<[BWPort1,BWPort4,BWPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup44], (instregex "ISTT_FP16m",
                                            "ISTT_FP32m",
                                            "ISTT_FP64m",
                                            "IST_F16m",
                                            "IST_F32m",
                                            "IST_FP16m",
                                            "IST_FP32m",
                                            "IST_FP64m",
                                            "VCVTPS2PHYmr",
                                            "VCVTPS2PHmr")>;

def BWWriteResGroup45 : SchedWriteRes<[BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[BWWriteResGroup45], (instregex "FNCLEX")>;

def BWWriteResGroup46 : SchedWriteRes<[BWPort015,BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[BWWriteResGroup46], (instregex "VZEROUPPER")>;

def BWWriteResGroup47 : SchedWriteRes<[BWPort0]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup47], (instregex "MMX_PMADDUBSWrr",
                                            "MMX_PMADDWDirr",
                                            "MMX_PMULHRSWrr",
                                            "MMX_PMULHUWirr",
                                            "MMX_PMULHWirr",
                                            "MMX_PMULLWirr",
                                            "MMX_PMULUDQirr",
                                            "MMX_PSADBWirr",
                                            "MUL_FPrST0",
                                            "MUL_FST0r",
                                            "MUL_FrST0",
                                            "PCMPGTQrr",
                                            "PHMINPOSUWrr",
                                            "PMADDUBSWrr",
                                            "PMADDWDrr",
                                            "PMULDQrr",
                                            "PMULHRSWrr",
                                            "PMULHUWrr",
                                            "PMULHWrr",
                                            "PMULLWrr",
                                            "PMULUDQrr",
                                            "PSADBWrr",
                                            "RCPPSr",
                                            "RCPSSr",
                                            "RSQRTPSr",
                                            "RSQRTSSr",
                                            "VPCMPGTQYrr",
                                            "VPCMPGTQrr",
                                            "VPHMINPOSUWrr",
                                            "VPMADDUBSWYrr",
                                            "VPMADDUBSWrr",
                                            "VPMADDWDYrr",
                                            "VPMADDWDrr",
                                            "VPMULDQYrr",
                                            "VPMULDQrr",
                                            "VPMULHRSWYrr",
                                            "VPMULHRSWrr",
                                            "VPMULHUWYrr",
                                            "VPMULHUWrr",
                                            "VPMULHWYrr",
                                            "VPMULHWrr",
                                            "VPMULLWYrr",
                                            "VPMULLWrr",
                                            "VPMULUDQYrr",
                                            "VPMULUDQrr",
                                            "VPSADBWYrr",
                                            "VPSADBWrr",
                                            "VRCPPSr",
                                            "VRCPSSr",
                                            "VRSQRTPSr",
                                            "VRSQRTSSr")>;

def BWWriteResGroup48 : SchedWriteRes<[BWPort01]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup48],
            (instregex "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)(Y)?r",
                       "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)r")>;

def BWWriteResGroup49 : SchedWriteRes<[BWPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup49], (instregex "LDDQUrm",
                                            "MMX_MOVD64rm",
                                            "MMX_MOVD64to64rm",
                                            "MMX_MOVQ64rm",
                                            "MOV(16|32|64)rm",
                                            "MOV64toPQIrm",
                                            "MOV8rm",
                                            "MOVAPDrm",
                                            "MOVAPSrm",
                                            "MOVDDUPrm",
                                            "MOVDI2PDIrm",
                                            "MOVDQArm",
                                            "MOVDQUrm",
                                            "MOVNTDQArm",
                                            "MOVQI2PQIrm",
                                            "MOVSDrm",
                                            "MOVSHDUPrm",
                                            "MOVSLDUPrm",
                                            "MOVSSrm",
                                            "MOVSX(16|32|64)rm16",
                                            "MOVSX(16|32|64)rm32",
                                            "MOVSX(16|32|64)rm8",
                                            "MOVUPDrm",
                                            "MOVUPSrm",
                                            "MOVZX(16|32|64)rm16",
                                            "MOVZX(16|32|64)rm8",
                                            "PREFETCHNTA",
                                            "PREFETCHT0",
                                            "PREFETCHT1",
                                            "PREFETCHT2",
                                            "VBROADCASTSSrm",
                                            "VLDDQUrm",
                                            "VMOV64toPQIrm",
                                            "VMOVAPDrm",
                                            "VMOVAPSrm",
                                            "VMOVDDUPrm",
                                            "VMOVDI2PDIrm",
                                            "VMOVDQArm",
                                            "VMOVDQUrm",
                                            "VMOVNTDQArm",
                                            "VMOVQI2PQIrm",
                                            "VMOVSDrm",
                                            "VMOVSHDUPrm",
                                            "VMOVSLDUPrm",
                                            "VMOVSSrm",
                                            "VMOVUPDrm",
                                            "VMOVUPSrm",
                                            "VPBROADCASTDrm",
                                            "VPBROADCASTQrm")>;

def BWWriteResGroup50 : SchedWriteRes<[BWPort1,BWPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup50], (instregex "CVTSI642SSrr",
                                            "HADDPDrr",
                                            "HADDPSrr",
                                            "HSUBPDrr",
                                            "HSUBPSrr",
                                            "VCVTSI642SSrr",
                                            "VHADDPDYrr",
                                            "VHADDPDrr",
                                            "VHADDPSYrr",
                                            "VHADDPSrr",
                                            "VHSUBPDYrr",
                                            "VHSUBPDrr",
                                            "VHSUBPSYrr",
                                            "VHSUBPSrr")>;

def BWWriteResGroup51 : SchedWriteRes<[BWPort1,BWPort6,BWPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup51], (instregex "STR(16|32|64)r")>;

def BWWriteResGroup52 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup52], (instrs IMUL32r, MUL32r, MULX32rr)>;

def BWWriteResGroup53 : SchedWriteRes<[BWPort0,BWPort4,BWPort237,BWPort15]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup53], (instregex "VMASKMOVPDYmr",
                                            "VMASKMOVPDmr",
                                            "VMASKMOVPSYmr",
                                            "VMASKMOVPSmr",
                                            "VPMASKMOVDYmr",
                                            "VPMASKMOVDmr",
                                            "VPMASKMOVQYmr",
                                            "VPMASKMOVQmr")>;

def BWWriteResGroup54 : SchedWriteRes<[BWPort6,BWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[BWWriteResGroup54], (instregex "PAUSE")>;

def BWWriteResGroup55 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[BWWriteResGroup55], (instregex "XSETBV")>;

def BWWriteResGroup56 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[BWWriteResGroup56], (instregex "CMPXCHG(8|16|32|64)rr")>;

def BWWriteResGroup57 : SchedWriteRes<[BWPort4,BWPort237,BWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[BWWriteResGroup57], (instregex "PUSHF16", "PUSHF64")>;

def BWWriteResGroup58 : SchedWriteRes<[BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup58], (instregex "LD_F32m",
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
                                            "VPBROADCASTQYrm",
                                            "ROUNDPDr",
                                            "ROUNDPSr",
                                            "ROUNDSDr",
                                            "ROUNDSSr",
                                            "VROUNDPDr",
                                            "VROUNDPSr",
                                            "VROUNDSDr",
                                            "VROUNDSSr",
                                            "VROUNDPDYr",
                                            "VROUNDPSYr")>;

def BWWriteResGroup59 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup59], (instregex "CVTPS2PDrm",
                                            "CVTSS2SDrm",
                                            "MMX_PSLLDrm",
                                            "MMX_PSLLQrm",
                                            "MMX_PSLLWrm",
                                            "MMX_PSRADrm",
                                            "MMX_PSRAWrm",
                                            "MMX_PSRLDrm",
                                            "MMX_PSRLQrm",
                                            "MMX_PSRLWrm",
                                            "VCVTPH2PSYrm",
                                            "VCVTPH2PSrm",
                                            "VCVTPS2PDrm",
                                            "VCVTSS2SDrm",
                                            "VPSLLVQrm",
                                            "VPSRLVQrm",
                                            "VTESTPDrm",
                                            "VTESTPSrm")>;

def BWWriteResGroup60 : SchedWriteRes<[BWPort1,BWPort5]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup60], (instregex "VCVTDQ2PDYrr",
                                            "VCVTPD2DQYrr",
                                            "VCVTPD2PSYrr",
                                            "VCVTPS2PHYrr",
                                            "VCVTTPD2DQYrr")>;

def BWWriteResGroup61 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup61], (instregex "ANDNPDrm",
                                            "ANDNPSrm",
                                            "ANDPDrm",
                                            "ANDPSrm",
                                            "INSERTPSrm",
                                            "MMX_PALIGNRrmi",
                                            "MMX_PINSRWrm",
                                            "MMX_PSHUFBrm",
                                            "MMX_PSHUFWmi",
                                            "MMX_PUNPCKHBWirm",
                                            "MMX_PUNPCKHDQirm",
                                            "MMX_PUNPCKHWDirm",
                                            "MMX_PUNPCKLBWirm",
                                            "MMX_PUNPCKLDQirm",
                                            "MMX_PUNPCKLWDirm",
                                            "MOVHPDrm",
                                            "MOVHPSrm",
                                            "MOVLPDrm",
                                            "MOVLPSrm",
                                            "ORPDrm",
                                            "ORPSrm",
                                            "PACKSSDWrm",
                                            "PACKSSWBrm",
                                            "PACKUSDWrm",
                                            "PACKUSWBrm",
                                            "PALIGNRrmi",
                                            "PBLENDWrmi",
                                            "PINSRBrm",
                                            "PINSRDrm",
                                            "PINSRQrm",
                                            "PINSRWrm",
                                            "PMOVSXBDrm",
                                            "PMOVSXBQrm",
                                            "PMOVSXBWrm",
                                            "PMOVSXDQrm",
                                            "PMOVSXWDrm",
                                            "PMOVSXWQrm",
                                            "PMOVZXBDrm",
                                            "PMOVZXBQrm",
                                            "PMOVZXBWrm",
                                            "PMOVZXDQrm",
                                            "PMOVZXWDrm",
                                            "PMOVZXWQrm",
                                            "PSHUFBrm",
                                            "PSHUFDmi",
                                            "PSHUFHWmi",
                                            "PSHUFLWmi",
                                            "PUNPCKHBWrm",
                                            "PUNPCKHDQrm",
                                            "PUNPCKHQDQrm",
                                            "PUNPCKHWDrm",
                                            "PUNPCKLBWrm",
                                            "PUNPCKLDQrm",
                                            "PUNPCKLQDQrm",
                                            "PUNPCKLWDrm",
                                            "SHUFPDrmi",
                                            "SHUFPSrmi",
                                            "UNPCKHPDrm",
                                            "UNPCKHPSrm",
                                            "UNPCKLPDrm",
                                            "UNPCKLPSrm",
                                            "VANDNPDrm",
                                            "VANDNPSrm",
                                            "VANDPDrm",
                                            "VANDPSrm",
                                            "VINSERTPSrm",
                                            "VMOVHPDrm",
                                            "VMOVHPSrm",
                                            "VMOVLPDrm",
                                            "VMOVLPSrm",
                                            "VORPDrm",
                                            "VORPSrm",
                                            "VPACKSSDWrm",
                                            "VPACKSSWBrm",
                                            "VPACKUSDWrm",
                                            "VPACKUSWBrm",
                                            "VPALIGNRrmi",
                                            "VPBLENDWrmi",
                                            "VPERMILPDmi",
                                            "VPERMILPDrm",
                                            "VPERMILPSmi",
                                            "VPERMILPSrm",
                                            "VPINSRBrm",
                                            "VPINSRDrm",
                                            "VPINSRQrm",
                                            "VPINSRWrm",
                                            "VPMOVSXBDrm",
                                            "VPMOVSXBQrm",
                                            "VPMOVSXBWrm",
                                            "VPMOVSXDQrm",
                                            "VPMOVSXWDrm",
                                            "VPMOVSXWQrm",
                                            "VPMOVZXBDrm",
                                            "VPMOVZXBQrm",
                                            "VPMOVZXBWrm",
                                            "VPMOVZXDQrm",
                                            "VPMOVZXWDrm",
                                            "VPMOVZXWQrm",
                                            "VPSHUFBrm",
                                            "VPSHUFDmi",
                                            "VPSHUFHWmi",
                                            "VPSHUFLWmi",
                                            "VPUNPCKHBWrm",
                                            "VPUNPCKHDQrm",
                                            "VPUNPCKHQDQrm",
                                            "VPUNPCKHWDrm",
                                            "VPUNPCKLBWrm",
                                            "VPUNPCKLDQrm",
                                            "VPUNPCKLQDQrm",
                                            "VPUNPCKLWDrm",
                                            "VSHUFPDrmi",
                                            "VSHUFPSrmi",
                                            "VUNPCKHPDrm",
                                            "VUNPCKHPSrm",
                                            "VUNPCKLPDrm",
                                            "VUNPCKLPSrm",
                                            "VXORPDrm",
                                            "VXORPSrm",
                                            "XORPDrm",
                                            "XORPSrm")>;

def BWWriteResGroup62 : SchedWriteRes<[BWPort6,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup62], (instregex "FARJMP64",
                                            "JMP(16|32|64)m")>;

def BWWriteResGroup63 : SchedWriteRes<[BWPort23,BWPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup63], (instregex "ADC(8|16|32|64)rm",
                                            "ADCX(32|64)rm",
                                            "ADOX(32|64)rm",
                                            "BT(16|32|64)mi8",
                                            "CMOV(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)(16|32|64)rm",
                                            "RORX(32|64)mi",
                                            "SARX(32|64)rm",
                                            "SBB(8|16|32|64)rm",
                                            "SHLX(32|64)rm",
                                            "SHRX(32|64)rm")>;

def BWWriteResGroup64 : SchedWriteRes<[BWPort23,BWPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup64], (instregex "ANDN(32|64)rm",
                                            "BLSI(32|64)rm",
                                            "BLSMSK(32|64)rm",
                                            "BLSR(32|64)rm",
                                            "BZHI(32|64)rm",
                                            "MMX_PABSBrm",
                                            "MMX_PABSDrm",
                                            "MMX_PABSWrm",
                                            "MMX_PADDBirm",
                                            "MMX_PADDDirm",
                                            "MMX_PADDQirm",
                                            "MMX_PADDSBirm",
                                            "MMX_PADDSWirm",
                                            "MMX_PADDUSBirm",
                                            "MMX_PADDUSWirm",
                                            "MMX_PADDWirm",
                                            "MMX_PAVGBirm",
                                            "MMX_PAVGWirm",
                                            "MMX_PCMPEQBirm",
                                            "MMX_PCMPEQDirm",
                                            "MMX_PCMPEQWirm",
                                            "MMX_PCMPGTBirm",
                                            "MMX_PCMPGTDirm",
                                            "MMX_PCMPGTWirm",
                                            "MMX_PMAXSWirm",
                                            "MMX_PMAXUBirm",
                                            "MMX_PMINSWirm",
                                            "MMX_PMINUBirm",
                                            "MMX_PSIGNBrm",
                                            "MMX_PSIGNDrm",
                                            "MMX_PSIGNWrm",
                                            "MMX_PSUBBirm",
                                            "MMX_PSUBDirm",
                                            "MMX_PSUBQirm",
                                            "MMX_PSUBSBirm",
                                            "MMX_PSUBSWirm",
                                            "MMX_PSUBUSBirm",
                                            "MMX_PSUBUSWirm",
                                            "MMX_PSUBWirm",
                                            "MOVBE(16|32|64)rm",
                                            "PABSBrm",
                                            "PABSDrm",
                                            "PABSWrm",
                                            "PADDBrm",
                                            "PADDDrm",
                                            "PADDQrm",
                                            "PADDSBrm",
                                            "PADDSWrm",
                                            "PADDUSBrm",
                                            "PADDUSWrm",
                                            "PADDWrm",
                                            "PAVGBrm",
                                            "PAVGWrm",
                                            "PCMPEQBrm",
                                            "PCMPEQDrm",
                                            "PCMPEQQrm",
                                            "PCMPEQWrm",
                                            "PCMPGTBrm",
                                            "PCMPGTDrm",
                                            "PCMPGTWrm",
                                            "PMAXSBrm",
                                            "PMAXSDrm",
                                            "PMAXSWrm",
                                            "PMAXUBrm",
                                            "PMAXUDrm",
                                            "PMAXUWrm",
                                            "PMINSBrm",
                                            "PMINSDrm",
                                            "PMINSWrm",
                                            "PMINUBrm",
                                            "PMINUDrm",
                                            "PMINUWrm",
                                            "PSIGNBrm",
                                            "PSIGNDrm",
                                            "PSIGNWrm",
                                            "PSUBBrm",
                                            "PSUBDrm",
                                            "PSUBQrm",
                                            "PSUBSBrm",
                                            "PSUBSWrm",
                                            "PSUBUSBrm",
                                            "PSUBUSWrm",
                                            "PSUBWrm",
                                            "VPABSBrm",
                                            "VPABSDrm",
                                            "VPABSWrm",
                                            "VPADDBrm",
                                            "VPADDDrm",
                                            "VPADDQrm",
                                            "VPADDSBrm",
                                            "VPADDSWrm",
                                            "VPADDUSBrm",
                                            "VPADDUSWrm",
                                            "VPADDWrm",
                                            "VPAVGBrm",
                                            "VPAVGWrm",
                                            "VPCMPEQBrm",
                                            "VPCMPEQDrm",
                                            "VPCMPEQQrm",
                                            "VPCMPEQWrm",
                                            "VPCMPGTBrm",
                                            "VPCMPGTDrm",
                                            "VPCMPGTWrm",
                                            "VPMAXSBrm",
                                            "VPMAXSDrm",
                                            "VPMAXSWrm",
                                            "VPMAXUBrm",
                                            "VPMAXUDrm",
                                            "VPMAXUWrm",
                                            "VPMINSBrm",
                                            "VPMINSDrm",
                                            "VPMINSWrm",
                                            "VPMINUBrm",
                                            "VPMINUDrm",
                                            "VPMINUWrm",
                                            "VPSIGNBrm",
                                            "VPSIGNDrm",
                                            "VPSIGNWrm",
                                            "VPSUBBrm",
                                            "VPSUBDrm",
                                            "VPSUBQrm",
                                            "VPSUBSBrm",
                                            "VPSUBSWrm",
                                            "VPSUBUSBrm",
                                            "VPSUBUSWrm",
                                            "VPSUBWrm")>;

def BWWriteResGroup65 : SchedWriteRes<[BWPort23,BWPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup65], (instregex "BLENDPDrmi",
                                            "BLENDPSrmi",
                                            "MMX_PANDNirm",
                                            "MMX_PANDirm",
                                            "MMX_PORirm",
                                            "MMX_PXORirm",
                                            "PANDNrm",
                                            "PANDrm",
                                            "PORrm",
                                            "PXORrm",
                                            "VBLENDPDrmi",
                                            "VBLENDPSrmi",
                                            "VINSERTF128rm",
                                            "VINSERTI128rm",
                                            "VPANDNrm",
                                            "VPANDrm",
                                            "VPBLENDDrmi",
                                            "VPORrm",
                                            "VPXORrm")>;

def BWWriteResGroup66 : SchedWriteRes<[BWPort23,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup66], (instrs POP16r, POP32r, POP64r)>;
def: InstRW<[BWWriteResGroup66], (instregex "ADD(8|16|32|64)rm",
                                            "AND(8|16|32|64)rm",
                                            "CMP(8|16|32|64)mi",
                                            "CMP(8|16|32|64)mr",
                                            "CMP(8|16|32|64)rm",
                                            "OR(8|16|32|64)rm",
                                            "POP(16|32|64)rmr",
                                            "SUB(8|16|32|64)rm",
                                            "TEST(8|16|32|64)mr",
                                            "TEST(8|16|32|64)mi",
                                            "XOR(8|16|32|64)rm")>;

def BWWriteResGroup67 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[BWWriteResGroup67], (instregex "SHLD(16|32|64)rrCL",
                                            "SHRD(16|32|64)rrCL")>;

def BWWriteResGroup68 : SchedWriteRes<[BWPort1,BWPort6,BWPort06,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup68], (instregex "SLDT(16|32|64)r")>;

def BWWriteResGroup69 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup69], (instregex "BTC(16|32|64)mi8",
                                            "BTR(16|32|64)mi8",
                                            "BTS(16|32|64)mi8",
                                            "SAR(8|16|32|64)m1",
                                            "SAR(8|16|32|64)mi",
                                            "SHL(8|16|32|64)m1",
                                            "SHL(8|16|32|64)mi",
                                            "SHR(8|16|32|64)m1",
                                            "SHR(8|16|32|64)mi")>;

def BWWriteResGroup70 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup70], (instregex "ADD(8|16|32|64)mi",
                                            "ADD(8|16|32|64)mr",
                                            "AND(8|16|32|64)mi",
                                            "AND(8|16|32|64)mr",
                                            "DEC(8|16|32|64)m",
                                            "INC(8|16|32|64)m",
                                            "NEG(8|16|32|64)m",
                                            "NOT(8|16|32|64)m",
                                            "OR(8|16|32|64)mi",
                                            "OR(8|16|32|64)mr",
                                            "POP(16|32|64)rmm",
                                            "PUSH(16|32|64)rmm",
                                            "SUB(8|16|32|64)mi",
                                            "SUB(8|16|32|64)mr",
                                            "XOR(8|16|32|64)mi",
                                            "XOR(8|16|32|64)mr")>;

def BWWriteResGroup71 : SchedWriteRes<[BWPort6,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,5];
}
def: InstRW<[BWWriteResGroup71], (instregex "STD")>;

def BWWriteResGroup73 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup73], (instregex "VPSLLDYrm",
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

def BWWriteResGroup74 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup74], (instregex "FCOM32m",
                                            "FCOM64m",
                                            "FCOMP32m",
                                            "FCOMP64m")>;

def BWWriteResGroup75 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup75], (instregex "VANDNPDYrm",
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

def BWWriteResGroup76 : SchedWriteRes<[BWPort23,BWPort15]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup76], (instregex "VPABSBYrm",
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

def BWWriteResGroup77 : SchedWriteRes<[BWPort23,BWPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup77], (instregex "VBLENDPDYrmi",
                                            "VBLENDPSYrmi",
                                            "VPANDNYrm",
                                            "VPANDYrm",
                                            "VPBLENDDYrmi",
                                            "VPORYrm",
                                            "VPXORYrm")>;

def BWWriteResGroup78 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup78], (instregex "MPSADBWrri",
                                            "VMPSADBWYrri",
                                            "VMPSADBWrri")>;

def BWWriteResGroup79 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup79], (instregex "BLENDVPDrm0",
                                            "BLENDVPSrm0",
                                            "MMX_PACKSSDWirm",
                                            "MMX_PACKSSWBirm",
                                            "MMX_PACKUSWBirm",
                                            "PBLENDVBrm0",
                                            "VBLENDVPDrm",
                                            "VBLENDVPSrm",
                                            "VMASKMOVPDrm",
                                            "VMASKMOVPSrm",
                                            "VPBLENDVBrm",
                                            "VPMASKMOVDrm",
                                            "VPMASKMOVQrm")>;

def BWWriteResGroup80 : SchedWriteRes<[BWPort23,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup80], (instregex "LEAVE64",
                                            "SCASB",
                                            "SCASL",
                                            "SCASQ",
                                            "SCASW")>;

def BWWriteResGroup81 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup81], (instregex "PSLLDrm",
                                            "PSLLQrm",
                                            "PSLLWrm",
                                            "PSRADrm",
                                            "PSRAWrm",
                                            "PSRLDrm",
                                            "PSRLQrm",
                                            "PSRLWrm",
                                            "PTESTrm",
                                            "VPSLLDrm",
                                            "VPSLLQrm",
                                            "VPSLLWrm",
                                            "VPSRADrm",
                                            "VPSRAWrm",
                                            "VPSRLDrm",
                                            "VPSRLQrm",
                                            "VPSRLWrm",
                                            "VPTESTrm")>;

def BWWriteResGroup82 : SchedWriteRes<[BWPort0,BWPort01,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup82], (instregex "FLDCW16m")>;

def BWWriteResGroup83 : SchedWriteRes<[BWPort0,BWPort23,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup83], (instregex "(V?)LDMXCSR")>;

def BWWriteResGroup84 : SchedWriteRes<[BWPort6,BWPort23,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup84], (instregex "LRETQ",
                                            "RETQ")>;

def BWWriteResGroup85 : SchedWriteRes<[BWPort23,BWPort06,BWPort15]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup85], (instregex "BEXTR(32|64)rm")>;

def BWWriteResGroup86 : SchedWriteRes<[BWPort23,BWPort06,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup86], (instregex "CMOV(A|BE)(16|32|64)rm")>;

def BWWriteResGroup87 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[BWWriteResGroup87], (instregex "ROL(8|16|32|64)m1",
                                            "ROL(8|16|32|64)mi",
                                            "ROR(8|16|32|64)m1",
                                            "ROR(8|16|32|64)mi")>;

def BWWriteResGroup88 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[BWWriteResGroup88], (instregex "XADD(8|16|32|64)rm")>;

def BWWriteResGroup89 : SchedWriteRes<[BWPort4,BWPort6,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[BWWriteResGroup89], (instregex "CALL(16|32|64)m",
                                            "FARCALL64")>;

def BWWriteResGroup90 : SchedWriteRes<[BWPort6,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [2,2,1,2];
}
def: InstRW<[BWWriteResGroup90], (instrs LOOP)>;

def BWWriteResGroup91 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup91], (instrs IMUL32rm, IMUL32rmi, IMUL32rmi8, IMUL64rm, IMUL64rmi8, IMUL64rmi32)>;
def: InstRW<[BWWriteResGroup91], (instrs IMUL8m, MUL8m)>;
def: InstRW<[BWWriteResGroup91], (instregex "ADDPDrm",
                                            "ADDPSrm",
                                            "ADDSDrm",
                                            "ADDSSrm",
                                            "ADDSUBPDrm",
                                            "ADDSUBPSrm",
                                            "BSF(16|32|64)rm",
                                            "BSR(16|32|64)rm",
                                            "CMPPDrmi",
                                            "CMPPSrmi",
                                            "CMPSDrm",
                                            "CMPSSrm",
                                            "COMISDrm",
                                            "COMISSrm",
                                            "CVTDQ2PSrm",
                                            "CVTPS2DQrm",
                                            "CVTTPS2DQrm",
                                            "LZCNT(16|32|64)rm",
                                            "MAX(C?)PDrm",
                                            "MAX(C?)PSrm",
                                            "MAX(C?)SDrm",
                                            "MAX(C?)SSrm",
                                            "MIN(C?)PDrm",
                                            "MIN(C?)PSrm",
                                            "MIN(C?)SDrm",
                                            "MIN(C?)SSrm",
                                            "MMX_CVTPI2PSirm",
                                            "MMX_CVTPS2PIirm",
                                            "MMX_CVTTPS2PIirm",
                                            "PDEP(32|64)rm",
                                            "PEXT(32|64)rm",
                                            "POPCNT(16|32|64)rm",
                                            "SUBPDrm",
                                            "SUBPSrm",
                                            "SUBSDrm",
                                            "SUBSSrm",
                                            "TZCNT(16|32|64)rm",
                                            "UCOMISDrm",
                                            "UCOMISSrm",
                                            "VADDPDrm",
                                            "VADDPSrm",
                                            "VADDSDrm",
                                            "VADDSSrm",
                                            "VADDSUBPDrm",
                                            "VADDSUBPSrm",
                                            "VCMPPDrmi",
                                            "VCMPPSrmi",
                                            "VCMPSDrm",
                                            "VCMPSSrm",
                                            "VCOMISDrm",
                                            "VCOMISSrm",
                                            "VCVTDQ2PSrm",
                                            "VCVTPS2DQrm",
                                            "VCVTTPS2DQrm",
                                            "VMAX(C?)PDrm",
                                            "VMAX(C?)PSrm",
                                            "VMAX(C?)SDrm",
                                            "VMAX(C?)SSrm",
                                            "VMIN(C?)PDrm",
                                            "VMIN(C?)PSrm",
                                            "VMIN(C?)SDrm",
                                            "VMIN(C?)SSrm",
                                            "VSUBPDrm",
                                            "VSUBPSrm",
                                            "VSUBSDrm",
                                            "VSUBSSrm",
                                            "VUCOMISDrm",
                                            "VUCOMISSrm")>;

def BWWriteResGroup91_16 : SchedWriteRes<[BWPort1, BWPort0156, BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1]; 
}
def: InstRW<[BWWriteResGroup91_16], (instrs IMUL16rm, IMUL16rmi, IMUL16rmi8)>;

def BWWriteResGroup91_16_2 : SchedWriteRes<[BWPort1, BWPort0156, BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 5;
}
def: InstRW<[BWWriteResGroup91_16_2], (instrs IMUL16m, MUL16m)>;

def BWWriteResGroup92 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup92], (instregex "VPMOVSXBDYrm",
                                            "VPMOVSXBQYrm",
                                            "VPMOVSXBWYrm",
                                            "VPMOVSXDQYrm",
                                            "VPMOVSXWDYrm",
                                            "VPMOVSXWQYrm",
                                            "VPMOVZXWDYrm")>;

def BWWriteResGroup93 : SchedWriteRes<[BWPort01,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup93], (instregex "MULPDrm",
                                            "MULPSrm",
                                            "MULSDrm",
                                            "MULSSrm",
                                            "VMULPDrm",
                                            "VMULPSrm",
                                            "VMULSDrm",
                                            "VMULSSrm")>;

def BWWriteResGroup94 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup94], (instregex "VBLENDVPDYrm",
                                            "VBLENDVPSYrm",
                                            "VMASKMOVPDYrm",
                                            "VMASKMOVPSYrm",
                                            "VPBLENDVBYrm",
                                            "VPMASKMOVDYrm",
                                            "VPMASKMOVQYrm")>;

def BWWriteResGroup95 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup95], (instregex "VPSLLVDrm",
                                            "VPSRAVDrm",
                                            "VPSRLVDrm")>;

def BWWriteResGroup96 : SchedWriteRes<[BWPort5,BWPort23,BWPort15]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup96], (instregex "MMX_PHADDDrm",
                                            "MMX_PHADDSWrm",
                                            "MMX_PHADDWrm",
                                            "MMX_PHSUBDrm",
                                            "MMX_PHSUBSWrm",
                                            "MMX_PHSUBWrm",
                                            "PHADDDrm",
                                            "PHADDSWrm",
                                            "PHADDWrm",
                                            "PHSUBDrm",
                                            "PHSUBSWrm",
                                            "PHSUBWrm",
                                            "VPHADDDrm",
                                            "VPHADDSWrm",
                                            "VPHADDWrm",
                                            "VPHSUBDrm",
                                            "VPHSUBSWrm",
                                            "VPHSUBWrm")>;

def BWWriteResGroup97 : SchedWriteRes<[BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[BWWriteResGroup97], (instregex "RCL(8|16|32|64)m1",
                                            "RCL(8|16|32|64)mi",
                                            "RCR(8|16|32|64)m1",
                                            "RCR(8|16|32|64)mi")>;

def BWWriteResGroup98 : SchedWriteRes<[BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,2,1];
}
def: InstRW<[BWWriteResGroup98], (instregex "ROR(8|16|32|64)mCL")>;

def BWWriteResGroup99 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[BWWriteResGroup99], (instregex "ADC(8|16|32|64)mi",
                                            "XCHG(8|16|32|64)rm")>;

def BWWriteResGroup100 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[BWWriteResGroup100], (instregex "ADC(8|16|32|64)mr",
                                             "CMPXCHG(8|16|32|64)rm",
                                             "ROL(8|16|32|64)mCL",
                                             "SAR(8|16|32|64)mCL",
                                             "SBB(8|16|32|64)mi",
                                             "SBB(8|16|32|64)mr",
                                             "SHL(8|16|32|64)mCL",
                                             "SHR(8|16|32|64)mCL")>;

def BWWriteResGroup101 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup101], (instregex "ADD_F32m",
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

def BWWriteResGroup102 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup102], (instregex "VPERM2F128rm",
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

def BWWriteResGroup103 : SchedWriteRes<[BWPort01,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup103], (instregex "VMULPDYrm",
                                             "VMULPSYrm")>;

def BWWriteResGroup104 : SchedWriteRes<[BWPort0,BWPort1,BWPort5]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup104], (instregex "DPPDrri",
                                             "VDPPDrri")>;

def BWWriteResGroup105 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup105], (instregex "CVTSD2SI64rm",
                                             "CVTSD2SIrm",
                                             "CVTSS2SI64rm",
                                             "CVTSS2SIrm",
                                             "CVTTSD2SI64rm",
                                             "CVTTSD2SIrm",
                                             "CVTTSS2SIrm",
                                             "VCVTSD2SI64rm",
                                             "VCVTSD2SIrm",
                                             "VCVTSS2SI64rm",
                                             "VCVTSS2SIrm",
                                             "VCVTTSD2SI64rm",
                                             "VCVTTSD2SIrm",
                                             "VCVTTSS2SI64rm",
                                             "VCVTTSS2SIrm")>;

def BWWriteResGroup106 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup106], (instregex "VCVTPS2PDYrm")>;

def BWWriteResGroup107 : SchedWriteRes<[BWPort1,BWPort5,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup107], (instrs IMUL64m, MUL64m, MULX64rm)>;
def: InstRW<[BWWriteResGroup107], (instregex "CVTDQ2PDrm",
                                             "CVTPD2DQrm",
                                             "CVTPD2PSrm",
                                             "CVTSD2SSrm",
                                             "CVTTPD2DQrm",
                                             "MMX_CVTPD2PIirm",
                                             "MMX_CVTPI2PDirm",
                                             "MMX_CVTTPD2PIirm",
                                             "VCVTDQ2PDrm",
                                             "VCVTSD2SSrm")>;

def BWWriteResGroup108 : SchedWriteRes<[BWPort5,BWPort23,BWPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup108], (instregex "VPBROADCASTBYrm",
                                             "VPBROADCASTBrm",
                                             "VPBROADCASTWYrm",
                                             "VPBROADCASTWrm")>;

def BWWriteResGroup109 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup109], (instregex "VPSLLVDYrm",
                                             "VPSRAVDYrm",
                                             "VPSRLVDYrm")>;

def BWWriteResGroup110 : SchedWriteRes<[BWPort5,BWPort23,BWPort15]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup110], (instregex "VPHADDDYrm",
                                             "VPHADDSWYrm",
                                             "VPHADDWYrm",
                                             "VPHSUBDYrm",
                                             "VPHSUBSWYrm",
                                             "VPHSUBWYrm")>;

def BWWriteResGroup111 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup111], (instregex "SHLD(16|32|64)mri8",
                                             "SHRD(16|32|64)mri8")>;

def BWWriteResGroup112 : SchedWriteRes<[BWPort23,BWPort06,BWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[BWWriteResGroup112], (instregex "RDRAND(16|32|64)r")>;

def BWWriteResGroup113 : SchedWriteRes<[BWPort1,BWPort6,BWPort23,BWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[BWWriteResGroup113], (instregex "LAR(16|32|64)rm",
                                             "LSL(16|32|64)rm")>;

def BWWriteResGroup114 : SchedWriteRes<[BWPort0]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup114], (instregex "PMULLDrr",
                                             "VPMULLDYrr",
                                             "VPMULLDrr")>;

def BWWriteResGroup115 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup115], (instregex "MMX_PMADDUBSWrm",
                                             "MMX_PMADDWDirm",
                                             "MMX_PMULHRSWrm",
                                             "MMX_PMULHUWirm",
                                             "MMX_PMULHWirm",
                                             "MMX_PMULLWirm",
                                             "MMX_PMULUDQirm",
                                             "MMX_PSADBWirm",
                                             "PCMPGTQrm",
                                             "PHMINPOSUWrm",
                                             "PMADDUBSWrm",
                                             "PMADDWDrm",
                                             "PMULDQrm",
                                             "PMULHRSWrm",
                                             "PMULHUWrm",
                                             "PMULHWrm",
                                             "PMULLWrm",
                                             "PMULUDQrm",
                                             "PSADBWrm",
                                             "RCPPSm",
                                             "RCPSSm",
                                             "RSQRTPSm",
                                             "RSQRTSSm",
                                             "VPCMPGTQrm",
                                             "VPHMINPOSUWrm",
                                             "VPMADDUBSWrm",
                                             "VPMADDWDrm",
                                             "VPMULDQrm",
                                             "VPMULHRSWrm",
                                             "VPMULHUWrm",
                                             "VPMULHWrm",
                                             "VPMULLWrm",
                                             "VPMULUDQrm",
                                             "VPSADBWrm",
                                             "VRCPPSm",
                                             "VRCPSSm",
                                             "VRSQRTPSm",
                                             "VRSQRTSSm")>;

def BWWriteResGroup116 : SchedWriteRes<[BWPort01,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup116],
            (instregex "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)m",
                       "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)m")>;

def BWWriteResGroup117 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup117], (instregex "FICOM16m",
                                             "FICOM32m",
                                             "FICOMP16m",
                                             "FICOMP32m")>;

def BWWriteResGroup118 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup118], (instregex "VPTESTYrm")>;

def BWWriteResGroup119 : SchedWriteRes<[BWPort1,BWPort5,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[BWWriteResGroup119], (instregex "HADDPDrm",
                                             "HADDPSrm",
                                             "HSUBPDrm",
                                             "HSUBPSrm",
                                             "VHADDPDrm",
                                             "VHADDPSrm",
                                             "VHSUBPDrm",
                                             "VHSUBPSrm")>;

def BWWriteResGroup120 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup120], (instregex "CVTTSS2SI64rm")>;

def BWWriteResGroup121 : SchedWriteRes<[BWPort1,BWPort23,BWPort06,BWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup121], (instrs IMUL32m, MUL32m, MULX32rm)>;

def BWWriteResGroup122 : SchedWriteRes<[BWPort0]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup122], (instregex "DIVPSrr",
                                             "DIVSSrr",
                                             "VDIVPSrr",
                                             "VDIVSSrr")>;

def BWWriteResGroup123 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup123], (instregex "MUL_F32m",
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

def BWWriteResGroup124 : SchedWriteRes<[BWPort01,BWPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup124],
            (instregex "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Ym")>;

def BWWriteResGroup126 : SchedWriteRes<[BWPort0,BWPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup126], (instregex "VRCPPSYr",
                                             "VRSQRTPSYr")>;

def BWWriteResGroup127 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup127], (instregex "ROUNDPDm",
                                             "ROUNDPSm",
                                             "ROUNDSDm",
                                             "ROUNDSSm",
                                             "VROUNDPDm",
                                             "VROUNDPSm",
                                             "VROUNDSDm",
                                             "VROUNDSSm")>;

def BWWriteResGroup128 : SchedWriteRes<[BWPort1,BWPort5,BWPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup128], (instregex "VCVTDQ2PDYrm")>;

def BWWriteResGroup129 : SchedWriteRes<[BWPort1,BWPort5,BWPort23]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[BWWriteResGroup129], (instregex "VHADDPDYrm",
                                             "VHADDPSYrm",
                                             "VHSUBPDYrm",
                                             "VHSUBPSYrm")>;

def BWWriteResGroup130 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,1,2];
}
def: InstRW<[BWWriteResGroup130], (instregex "SHLD(16|32|64)mrCL",
                                             "SHRD(16|32|64)mrCL")>;

def BWWriteResGroup131 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,2,3];
}
def: InstRW<[BWWriteResGroup131], (instregex "RCL(16|32|64)rCL",
                                             "RCR(16|32|64)rCL")>;

def BWWriteResGroup132 : SchedWriteRes<[BWPort1,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 9;
  let ResourceCycles = [1,4,1,3];
}
def: InstRW<[BWWriteResGroup132], (instregex "RCL8rCL")>;

def BWWriteResGroup133 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 11;
  let ResourceCycles = [2,9];
}
def: InstRW<[BWWriteResGroup133], (instrs LOOPE)>;
def: InstRW<[BWWriteResGroup133], (instrs LOOPNE)>;

def BWWriteResGroup135 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup135], (instregex "ADD_FI16m",
                                             "ADD_FI32m",
                                             "SUBR_FI16m",
                                             "SUBR_FI32m",
                                             "SUB_FI16m",
                                             "SUB_FI32m",
                                             "VROUNDPDYm",
                                             "VROUNDPSYm")>;

def BWWriteResGroup136 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[BWWriteResGroup136], (instregex "(V?)MPSADBWrmi")>;

def BWWriteResGroup137 : SchedWriteRes<[BWPort0]> {
  let Latency = 13;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup137], (instregex "SQRTPSr",
                                             "SQRTSSr")>;

def BWWriteResGroup138 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[BWWriteResGroup138], (instregex "VMPSADBWYrmi")>;

def BWWriteResGroup139 : SchedWriteRes<[BWPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup139], (instregex "DIVPDrr",
                                             "DIVSDrr",
                                             "VDIVPDrr",
                                             "VDIVSDrr",
                                             "VSQRTPSr",
                                             "VSQRTSSr")>;

def BWWriteResGroup141 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup141], (instregex "MUL_FI16m",
                                             "MUL_FI32m")>;

def BWWriteResGroup142 : SchedWriteRes<[BWPort0,BWPort1,BWPort5]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup142], (instregex "DPPSrri",
                                             "VDPPSYrri",
                                             "VDPPSrri")>;

def BWWriteResGroup143 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup143], (instregex "(V?)DPPDrmi")>;

def BWWriteResGroup144 : SchedWriteRes<[BWPort1,BWPort6,BWPort23,BWPort0156]> {
  let Latency = 14;
  let NumMicroOps = 8;
  let ResourceCycles = [2,2,1,3];
}
def: InstRW<[BWWriteResGroup144], (instregex "LAR(16|32|64)rr")>;

def BWWriteResGroup145 : SchedWriteRes<[BWPort1,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 14;
  let NumMicroOps = 10;
  let ResourceCycles = [2,3,1,4];
}
def: InstRW<[BWWriteResGroup145], (instregex "RCR8rCL")>;

def BWWriteResGroup146 : SchedWriteRes<[BWPort0,BWPort1,BWPort6,BWPort0156]> {
  let Latency = 14;
  let NumMicroOps = 12;
  let ResourceCycles = [2,1,4,5];
}
def: InstRW<[BWWriteResGroup146], (instregex "XCH_F")>;

def BWWriteResGroup147 : SchedWriteRes<[BWPort0]> {
  let Latency = 15;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup147], (instregex "DIVR_FPrST0",
                                             "DIVR_FST0r",
                                             "DIVR_FrST0")>;

def BWWriteResGroup148 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 15;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup148], (instregex "(V?)PMULLDrm")>;

def BWWriteResGroup149 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 15;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,4,1,2];
}
def: InstRW<[BWWriteResGroup149], (instregex "RCL(8|16|32|64)mCL")>;

def BWWriteResGroup150 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup150], (instregex "(V?)DIVPSrm",
                                             "(V?)DIVSSrm")>;

def BWWriteResGroup151 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 16;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup151], (instregex "VPMULLDYrm")>;

def BWWriteResGroup153 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 16;
  let NumMicroOps = 14;
  let ResourceCycles = [1,1,1,4,2,5];
}
def: InstRW<[BWWriteResGroup153], (instregex "CMPXCHG8B")>;

def BWWriteResGroup154 : SchedWriteRes<[BWPort5]> {
  let Latency = 16;
  let NumMicroOps = 16;
  let ResourceCycles = [16];
}
def: InstRW<[BWWriteResGroup154], (instrs VZEROALL)>;

def BWWriteResGroup155 : SchedWriteRes<[BWPort0,BWPort015]> {
  let Latency = 17;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup155], (instregex "VDIVPSYrr")>;

def BWWriteResGroup156 : SchedWriteRes<[BWPort0,BWPort23,BWPort015]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup156], (instregex "VRCPPSYm",
                                             "VRSQRTPSYm")>;

def BWWriteResGroup157 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup157], (instregex "SQRTPSm",
                                             "SQRTSSm")>;

def BWWriteResGroup159 : SchedWriteRes<[BWPort5,BWPort6,BWPort06,BWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[BWWriteResGroup159], (instrs CPUID)>;
def: InstRW<[BWWriteResGroup159], (instrs RDTSC)>;

def BWWriteResGroup160 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,3,1,3];
}
def: InstRW<[BWWriteResGroup160], (instregex "RCR(8|16|32|64)mCL")>;

def BWWriteResGroup161 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup161], (instregex "DIVPDrm",
                                             "DIVSDrm",
                                             "VDIVPDrm",
                                             "VDIVSDrm",
                                             "VSQRTPSm",
                                             "VSQRTSSm")>;

def BWWriteResGroup163 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23]> {
  let Latency = 19;
  let NumMicroOps = 5;
  let ResourceCycles = [2,1,1,1];
}
def: InstRW<[BWWriteResGroup163], (instregex "(V?)DPPSrmi")>;

def BWWriteResGroup165 : SchedWriteRes<[BWPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup165], (instregex "DIV_FPrST0",
                                             "DIV_FST0r",
                                             "DIV_FrST0",
                                             "SQRTPDr",
                                             "SQRTSDr")>;

def BWWriteResGroup166 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23]> {
  let Latency = 20;
  let NumMicroOps = 5;
  let ResourceCycles = [2,1,1,1];
}
def: InstRW<[BWWriteResGroup166], (instregex "VDPPSYrmi")>;

def BWWriteResGroup167 : SchedWriteRes<[BWPort4,BWPort5,BWPort6,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 20;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,1,2];
}
def: InstRW<[BWWriteResGroup167], (instregex "INSB",
                                             "INSL",
                                             "INSW")>;

def BWWriteResGroup168 : SchedWriteRes<[BWPort0]> {
  let Latency = 21;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup168], (instregex "VSQRTPDr",
                                             "VSQRTSDr")>;

def BWWriteResGroup169 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup169], (instregex "DIV_F32m",
                                             "DIV_F64m")>;

def BWWriteResGroup170 : SchedWriteRes<[BWPort0,BWPort015]> {
  let Latency = 21;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup170], (instregex "VSQRTPSYr")>;

def BWWriteResGroup171 : SchedWriteRes<[BWPort0,BWPort4,BWPort5,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 21;
  let NumMicroOps = 19;
  let ResourceCycles = [2,1,4,1,1,4,6];
}
def: InstRW<[BWWriteResGroup171], (instregex "CMPXCHG16B")>;

def BWWriteResGroup172 : SchedWriteRes<[BWPort6,BWPort23,BWPort0156]> {
  let Latency = 22;
  let NumMicroOps = 18;
  let ResourceCycles = [1,1,16];
}
def: InstRW<[BWWriteResGroup172], (instregex "POPF64")>;

def BWWriteResGroup173 : SchedWriteRes<[BWPort0,BWPort015]> {
  let Latency = 23;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup173], (instregex "VDIVPDYrr")>;

def BWWriteResGroup174 : SchedWriteRes<[BWPort0,BWPort23,BWPort015]> {
  let Latency = 23;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup174], (instregex "VDIVPSYrm")>;

def BWWriteResGroup176 : SchedWriteRes<[BWPort6,BWPort23,BWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 19;
  let ResourceCycles = [3,1,15];
}
def: InstRW<[BWWriteResGroup176], (instregex "XRSTOR(64)?")>;

def BWWriteResGroup177 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 24;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup177], (instregex "DIV_FI16m",
                                             "DIV_FI32m")>;

def BWWriteResGroup179 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup179], (instregex "SQRTPDm",
                                             "SQRTSDm")>;

def BWWriteResGroup180 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 26;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup180], (instregex "DIVR_F32m",
                                             "DIVR_F64m",
                                             "VSQRTPDm",
                                             "VSQRTSDm")>;

def BWWriteResGroup181 : SchedWriteRes<[BWPort0,BWPort23,BWPort015]> {
  let Latency = 27;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup181], (instregex "VSQRTPSYm")>;

def BWWriteResGroup182 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 29;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup182], (instregex "DIVR_FI16m",
                                             "DIVR_FI32m")>;

def BWWriteResGroup183 : SchedWriteRes<[BWPort0,BWPort23,BWPort015]> {
  let Latency = 29;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup183], (instregex "VDIVPDYrm")>;

def BWWriteResGroup183_1 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 22;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,2,1];
}
def: InstRW<[BWWriteResGroup183_1], (instrs VGATHERQPDrm)>;

def BWWriteResGroup183_2 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 9;
  let ResourceCycles = [1,3,4,1];
}
def: InstRW<[BWWriteResGroup183_2], (instrs VGATHERQPDYrm)>;

def BWWriteResGroup183_3 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,2,1];
}
def: InstRW<[BWWriteResGroup183_3], (instrs VGATHERQPSYrm)>;

def BWWriteResGroup183_4 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 25;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,2,1];
}
def: InstRW<[BWWriteResGroup183_4], (instrs VGATHERDPDrm,
                                            VGATHERDPSrm)>;

def BWWriteResGroup183_5 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 26;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,2,1];
}
def: InstRW<[BWWriteResGroup183_5], (instrs VGATHERDPDYrm)>;

def BWWriteResGroup183_6 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 26;
  let NumMicroOps = 14;
  let ResourceCycles = [1,4,8,1];  
}
def: InstRW<[BWWriteResGroup183_6], (instrs VGATHERDPSYrm)>;

def BWWriteResGroup183_7 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 27;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,2,1];
}
def: InstRW<[BWWriteResGroup183_7], (instrs VGATHERQPSrm)>;

def BWWriteResGroup185 : SchedWriteRes<[BWPort4,BWPort6,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 29;
  let NumMicroOps = 27;
  let ResourceCycles = [1,5,1,1,19];
}
def: InstRW<[BWWriteResGroup185], (instregex "XSAVE64")>;

def BWWriteResGroup186 : SchedWriteRes<[BWPort4,BWPort6,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 30;
  let NumMicroOps = 28;
  let ResourceCycles = [1,6,1,1,19];
}
def: InstRW<[BWWriteResGroup186], (instregex "^XSAVE$", "XSAVEC", "XSAVES", "XSAVEOPT")>;

def BWWriteResGroup187 : SchedWriteRes<[BWPort01,BWPort15,BWPort015,BWPort0156]> {
  let Latency = 31;
  let NumMicroOps = 31;
  let ResourceCycles = [8,1,21,1];
}
def: InstRW<[BWWriteResGroup187], (instregex "MMX_EMMS")>;

def BWWriteResGroup189 : SchedWriteRes<[BWPort0,BWPort015]> {
  let Latency = 34;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup189], (instregex "VSQRTPDYr")>;

def BWWriteResGroup190 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23,BWPort0156]> {
  let Latency = 34;
  let NumMicroOps = 8;
  let ResourceCycles = [2,2,2,1,1];
}
def: InstRW<[BWWriteResGroup190], (instregex "DIV(8|16|32|64)m")>;

def BWWriteResGroup191 : SchedWriteRes<[BWPort5,BWPort6,BWPort23,BWPort06,BWPort0156]> {
  let Latency = 34;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[BWWriteResGroup191], (instregex "IN(8|16|32)ri",
                                             "IN(8|16|32)rr")>;

def BWWriteResGroup193 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23,BWPort0156]> {
  let Latency = 35;
  let NumMicroOps = 8;
  let ResourceCycles = [2,2,2,1,1];
}
def: InstRW<[BWWriteResGroup193], (instregex "IDIV(8|16|32|64)m")>;

def BWWriteResGroup194 : SchedWriteRes<[BWPort5,BWPort6,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[BWWriteResGroup194], (instregex "OUT(8|16|32)ir",
                                             "OUT(8|16|32)rr")>;

def BWWriteResGroup195 : SchedWriteRes<[BWPort0,BWPort23,BWPort015]> {
  let Latency = 40;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup195], (instregex "VSQRTPDYm")>;

def BWWriteResGroup196 : SchedWriteRes<[BWPort5,BWPort0156]> {
  let Latency = 42;
  let NumMicroOps = 22;
  let ResourceCycles = [2,20];
}
def: InstRW<[BWWriteResGroup196], (instrs RDTSCP)>;

def BWWriteResGroup197 : SchedWriteRes<[BWPort0,BWPort01,BWPort23,BWPort05,BWPort06,BWPort015,BWPort0156]> {
  let Latency = 60;
  let NumMicroOps = 64;
  let ResourceCycles = [2,2,8,1,10,2,39];
}
def: InstRW<[BWWriteResGroup197], (instregex "FLDENVm")>;

def BWWriteResGroup198 : SchedWriteRes<[BWPort0,BWPort6,BWPort23,BWPort05,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 63;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[BWWriteResGroup198], (instrs FXRSTOR64)>;

def BWWriteResGroup199 : SchedWriteRes<[BWPort0,BWPort6,BWPort23,BWPort05,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 63;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[BWWriteResGroup199], (instrs FXRSTOR)>;

def BWWriteResGroup200 : SchedWriteRes<[BWPort5,BWPort01,BWPort0156]> {
  let Latency = 75;
  let NumMicroOps = 15;
  let ResourceCycles = [6,3,6];
}
def: InstRW<[BWWriteResGroup200], (instregex "FNINIT")>;

def BWWriteResGroup201 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort6,BWPort01,BWPort0156]> {
  let Latency = 80;
  let NumMicroOps = 32;
  let ResourceCycles = [7,7,3,3,1,11];
}
def: InstRW<[BWWriteResGroup201], (instregex "DIV(16|32|64)r")>;

def BWWriteResGroup202 : SchedWriteRes<[BWPort0,BWPort1,BWPort4,BWPort5,BWPort6,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 115;
  let NumMicroOps = 100;
  let ResourceCycles = [9,9,11,8,1,11,21,30];
}
def: InstRW<[BWWriteResGroup202], (instregex "FSTENVm")>;

} // SchedModel

