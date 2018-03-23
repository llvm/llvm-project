//=- X86SchedSkylake.td - X86 Skylake Server Scheduling ------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Skylake Server to support
// instruction scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def SkylakeServerModel : SchedMachineModel {
  // All x86 instructions are modeled as a single micro-op, and SKylake can
  // decode 6 instructions per cycle.
  let IssueWidth = 6;
  let MicroOpBufferSize = 224; // Based on the reorder buffer.
  let LoadLatency = 5;
  let MispredictPenalty = 14;
  
  // Based on the LSD (loop-stream detector) queue size and benchmarking data.
  let LoopMicroOpBufferSize = 50;

  // This flag is set to allow the scheduler to assign a default model to
  // unrecognized opcodes.
  let CompleteModel = 0;
}

let SchedModel = SkylakeServerModel in {

// Skylake Server can issue micro-ops to 8 different ports in one cycle.

// Ports 0, 1, 5, and 6 handle all computation.
// Port 4 gets the data half of stores. Store data can be available later than
// the store address, but since we don't model the latency of stores, we can
// ignore that.
// Ports 2 and 3 are identical. They handle loads and the address half of
// stores. Port 7 can handle address calculations.
def SKXPort0 : ProcResource<1>;
def SKXPort1 : ProcResource<1>;
def SKXPort2 : ProcResource<1>;
def SKXPort3 : ProcResource<1>;
def SKXPort4 : ProcResource<1>;
def SKXPort5 : ProcResource<1>;
def SKXPort6 : ProcResource<1>;
def SKXPort7 : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def SKXPort01  : ProcResGroup<[SKXPort0, SKXPort1]>;
def SKXPort23  : ProcResGroup<[SKXPort2, SKXPort3]>;
def SKXPort237 : ProcResGroup<[SKXPort2, SKXPort3, SKXPort7]>;
def SKXPort04  : ProcResGroup<[SKXPort0, SKXPort4]>;
def SKXPort05  : ProcResGroup<[SKXPort0, SKXPort5]>;
def SKXPort06  : ProcResGroup<[SKXPort0, SKXPort6]>;
def SKXPort15  : ProcResGroup<[SKXPort1, SKXPort5]>;
def SKXPort16  : ProcResGroup<[SKXPort1, SKXPort6]>;
def SKXPort56  : ProcResGroup<[SKXPort5, SKXPort6]>;
def SKXPort015 : ProcResGroup<[SKXPort0, SKXPort1, SKXPort5]>;
def SKXPort056 : ProcResGroup<[SKXPort0, SKXPort5, SKXPort6]>;
def SKXPort0156: ProcResGroup<[SKXPort0, SKXPort1, SKXPort5, SKXPort6]>;

// 60 Entry Unified Scheduler
def SKXPortAny : ProcResGroup<[SKXPort0, SKXPort1, SKXPort2, SKXPort3, SKXPort4,
                              SKXPort5, SKXPort6, SKXPort7]> {
  let BufferSize=60;
}

// Loads are 5 cycles, so ReadAfterLd registers needn't be available until 5
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 5>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass SKXWriteResPair<X86FoldableSchedWrite SchedRW,
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
  def : WriteRes<SchedRW.Folded, !listconcat([SKXPort23], ExePorts)> {
    let Latency = !add(Lat, 5);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = UOps;
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [SKXPort4]>;

// Arithmetic.
defm : SKXWriteResPair<WriteALU,   [SKXPort0156], 1>; // Simple integer ALU op.
defm : SKXWriteResPair<WriteIMul,  [SKXPort1],   3>; // Integer multiplication.
def : WriteRes<WriteIMulH, []> { let Latency = 3; } // Integer multiplication, high part.
def SKXDivider : ProcResource<1>; // Integer division issued on port 0.     
def : WriteRes<WriteIDiv, [SKXPort0, SKXDivider]> { // Integer division.
  let Latency = 25;
  let ResourceCycles = [1, 10];
}
def : WriteRes<WriteIDivLd, [SKXPort23, SKXPort0, SKXDivider]> {
  let Latency = 29;
  let ResourceCycles = [1, 1, 10];
}

def : WriteRes<WriteLEA, [SKXPort15]>; // LEA instructions can't fold loads.

// Integer shifts and rotates.
defm : SKXWriteResPair<WriteShift, [SKXPort06],  1>;

// Loads, stores, and moves, not folded with other operations.
def : WriteRes<WriteLoad,  [SKXPort23]> { let Latency = 5; }
def : WriteRes<WriteStore, [SKXPort237, SKXPort4]>;
def : WriteRes<WriteMove,  [SKXPort0156]>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
def : WriteRes<WriteZero,  []>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : SKXWriteResPair<WriteJump,  [SKXPort06],   1>;

// Floating point. This covers both scalar and vector operations.
def  : WriteRes<WriteFLoad,         [SKXPort23]> { let Latency = 5; }
def  : WriteRes<WriteFStore,        [SKXPort237, SKXPort4]>;
def  : WriteRes<WriteFMove,         [SKXPort015]>;

defm : SKXWriteResPair<WriteFAdd,   [SKXPort1], 3>; // Floating point add/sub/compare.
defm : SKXWriteResPair<WriteFMul,   [SKXPort0], 5>; // Floating point multiplication.
defm : SKXWriteResPair<WriteFDiv,   [SKXPort0], 12>; // 10-14 cycles. // Floating point division.
defm : SKXWriteResPair<WriteFSqrt,  [SKXPort0], 15>; // Floating point square root.
defm : SKXWriteResPair<WriteFRcp,   [SKXPort0], 5>; // Floating point reciprocal estimate.
defm : SKXWriteResPair<WriteFRsqrt, [SKXPort0], 5>; // Floating point reciprocal square root estimate.
defm : SKXWriteResPair<WriteFMA,  [SKXPort015], 4>; // Fused Multiply Add.
defm : SKXWriteResPair<WriteFShuffle,  [SKXPort5],  1>; // Floating point vector shuffles.
defm : SKXWriteResPair<WriteFBlend,  [SKXPort015],  1>; // Floating point vector blends.
defm : SKXWriteResPair<WriteFVarBlend,  [SKXPort5], 2, [2]>; // Fp vector variable blends.

// FMA Scheduling helper class.
// class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Vector integer operations.
def  : WriteRes<WriteVecLoad,         [SKXPort23]> { let Latency = 5; }
def  : WriteRes<WriteVecStore,        [SKXPort237, SKXPort4]>;
def  : WriteRes<WriteVecMove,         [SKXPort015]>;

defm : SKXWriteResPair<WriteVecALU,   [SKXPort15],  1>; // Vector integer ALU op, no logicals.
defm : SKXWriteResPair<WriteVecShift, [SKXPort0],  1>; // Vector integer shifts.
defm : SKXWriteResPair<WriteVecIMul,  [SKXPort0],   5>; // Vector integer multiply.
defm : SKXWriteResPair<WriteShuffle,  [SKXPort5],  1>; // Vector shuffles.
defm : SKXWriteResPair<WriteBlend,  [SKXPort15],  1>; // Vector blends.
defm : SKXWriteResPair<WriteVarBlend,  [SKXPort5], 2, [2]>; // Vector variable blends.
defm : SKXWriteResPair<WriteMPSAD,  [SKXPort0, SKXPort5], 6, [1, 2]>; // Vector MPSAD.

// Vector bitwise operations.
// These are often used on both floating point and integer vectors.
defm : SKXWriteResPair<WriteVecLogic, [SKXPort015], 1>; // Vector and/or/xor.

// Conversion between integer and float.
defm : SKXWriteResPair<WriteCvtF2I, [SKXPort1], 3>; // Float -> Integer.
defm : SKXWriteResPair<WriteCvtI2F, [SKXPort1], 4>; // Integer -> Float.
defm : SKXWriteResPair<WriteCvtF2F, [SKXPort1], 3>; // Float -> Float size conversion.

// Strings instructions.

// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [SKXPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [SKXPort0, SKXPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
} 

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [SKXPort0, SKXPort5, SKXPort015, SKXPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def : WriteRes<WritePCmpEStrMLd, [SKXPort0, SKXPort5, SKXPort23, SKXPort015, SKXPort0156]> {
  let Latency = 25;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [SKXPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [SKXPort0, SKXPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [SKXPort0,SKXPort5,SKXPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def : WriteRes<WritePCmpEStrILd, [SKXPort0, SKXPort5, SKXPort23, SKXPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}

// AES instructions.
def : WriteRes<WriteAESDecEnc, [SKXPort0]> { // Decryption, encryption.
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [SKXPort0, SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

def : WriteRes<WriteAESIMC, [SKXPort0]> { // InvMixColumn.
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [SKXPort0, SKXPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}

def : WriteRes<WriteAESKeyGen, [SKXPort0,SKXPort5,SKXPort015]> { // Key Generation.
  let Latency = 20;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,2];
}
def : WriteRes<WriteAESKeyGenLd, [SKXPort0,SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 25;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,1,1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [SKXPort5]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteCLMulLd, [SKXPort5, SKXPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

// Catch-all for expensive system instructions.
def : WriteRes<WriteSystem,     [SKXPort0156]> { let Latency = 100; } // def WriteSystem : SchedWrite;

// AVX2.
defm : SKXWriteResPair<WriteFShuffle256,  [SKXPort5],  3>; // Fp 256-bit width vector shuffles.
defm : SKXWriteResPair<WriteShuffle256,  [SKXPort5],  3>;  // 256-bit width vector shuffles.
defm : SKXWriteResPair<WriteVarVecShift,  [SKXPort0, SKXPort5], 2, [2, 1]>; // Variable vector shifts.

// Old microcoded instructions that nobody use.
def : WriteRes<WriteMicrocoded, [SKXPort0156]> { let Latency = 100; } // def WriteMicrocoded : SchedWrite;

// Fence instructions.
def : WriteRes<WriteFence,  [SKXPort23, SKXPort4]>;

// Nop, not very useful expect it provides a model for nops!
def : WriteRes<WriteNop, []>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : SKXWriteResPair<WriteFHAdd,   [SKXPort1], 3>;
defm : SKXWriteResPair<WritePHAdd,  [SKXPort15], 1>;

// Remaining instrs.

def SKXWriteResGroup1 : SchedWriteRes<[SKXPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup1], (instregex "KANDBrr",
                                            "KANDDrr",
                                            "KANDNBrr",
                                            "KANDNDrr",
                                            "KANDNQrr",
                                            "KANDNWrr",
                                            "KANDQrr",
                                            "KANDWrr",
                                            "KMOVBkk",
                                            "KMOVDkk",
                                            "KMOVQkk",
                                            "KMOVWkk",
                                            "KNOTBrr",
                                            "KNOTDrr",
                                            "KNOTQrr",
                                            "KNOTWrr",
                                            "KORBrr",
                                            "KORDrr",
                                            "KORQrr",
                                            "KORWrr",
                                            "KXNORBrr",
                                            "KXNORDrr",
                                            "KXNORQrr",
                                            "KXNORWrr",
                                            "KXORBrr",
                                            "KXORDrr",
                                            "KXORQrr",
                                            "KXORWrr",
                                            "MMX_PADDSBirr",
                                            "MMX_PADDSWirr",
                                            "MMX_PADDUSBirr",
                                            "MMX_PADDUSWirr",
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
                                            "MMX_PSUBSBirr",
                                            "MMX_PSUBSWirr",
                                            "MMX_PSUBUSBirr",
                                            "MMX_PSUBUSWirr",
                                            "VPMOVB2MZ128rr(b?)(k?)(z?)",
                                            "VPMOVB2MZ256rr(b?)(k?)(z?)",
                                            "VPMOVB2MZrr(b?)(k?)(z?)",
                                            "VPMOVD2MZ128rr(b?)(k?)(z?)",
                                            "VPMOVD2MZ256rr(b?)(k?)(z?)",
                                            "VPMOVD2MZrr(b?)(k?)(z?)",
                                            "VPMOVQ2MZ128rr(b?)(k?)(z?)",
                                            "VPMOVQ2MZ256rr(b?)(k?)(z?)",
                                            "VPMOVQ2MZrr(b?)(k?)(z?)",
                                            "VPMOVW2MZ128rr(b?)(k?)(z?)",
                                            "VPMOVW2MZ256rr(b?)(k?)(z?)",
                                            "VPMOVW2MZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup3 : SchedWriteRes<[SKXPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup3], (instregex "COMP_FST0r",
                                            "COM_FST0r",
                                            "INSERTPSrr",
                                            "KMOVBkr",
                                            "KMOVDkr",
                                            "KMOVQkr",
                                            "KMOVWkr",
                                            "MMX_MOVD64rr",
                                            "MMX_MOVD64to64rr",
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
                                            "MOVDDUPrr",
                                            "MOVDI2PDIrr",
                                            "MOVHLPSrr",
                                            "MOVLHPSrr",
                                            "MOVSDrr",
                                            "MOVSHDUPrr",
                                            "MOVSLDUPrr",
                                            "MOVUPDrr",
                                            "MOVUPSrr",
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
                                            "UCOM_FPr",
                                            "UCOM_Fr",
                                            "UNPCKHPDrr",
                                            "UNPCKHPSrr",
                                            "UNPCKLPDrr",
                                            "UNPCKLPSrr",
                                            "VBROADCASTI32X2Z128r(b?)(k?)(z?)",
                                            "VBROADCASTSSrr",
                                            "VINSERTPSZrr(b?)(k?)(z?)",
                                            "VINSERTPSrr",
                                            "VMOV64toPQIZrr(b?)(k?)(z?)",
                                            "VMOV64toPQIrr",
                                            "VMOVDDUPYrr",
                                            "VMOVDDUPZ128rr(b?)(k?)(z?)",
                                            "VMOVDDUPZ256rr(b?)(k?)(z?)",
                                            "VMOVDDUPZrr(b?)(k?)(z?)",
                                            "VMOVDDUPrr",
                                            "VMOVDI2PDIZrr(b?)(k?)(z?)",
                                            "VMOVDI2PDIrr",
                                            "VMOVHLPSZrr(b?)(k?)(z?)",
                                            "VMOVHLPSrr",
                                            "VMOVLHPSZrr(b?)(k?)(z?)",
                                            "VMOVLHPSrr",
                                            "VMOVSDZrr(b?)(k?)(z?)",
                                            "VMOVSDrr",
                                            "VMOVSHDUPYrr",
                                            "VMOVSHDUPZ128rr(b?)(k?)(z?)",
                                            "VMOVSHDUPZ256rr(b?)(k?)(z?)",
                                            "VMOVSHDUPZrr(b?)(k?)(z?)",
                                            "VMOVSHDUPrr",
                                            "VMOVSLDUPYrr",
                                            "VMOVSLDUPZ128rr(b?)(k?)(z?)",
                                            "VMOVSLDUPZ256rr(b?)(k?)(z?)",
                                            "VMOVSLDUPZrr(b?)(k?)(z?)",
                                            "VMOVSLDUPrr",
                                            "VMOVSSZrr(b?)(k?)(z?)",
                                            "VMOVUPDYrr",
                                            "VMOVUPDrr",
                                            "VMOVUPSYrr",
                                            "VMOVUPSrr",
                                            "VPACKSSDWYrr",
                                            "VPACKSSDWZ128rr(b?)(k?)(z?)",
                                            "VPACKSSDWZ256rr(b?)(k?)(z?)",
                                            "VPACKSSDWZrr(b?)(k?)(z?)",
                                            "VPACKSSDWrr",
                                            "VPACKSSWBYrr",
                                            "VPACKSSWBZ128rr(b?)(k?)(z?)",
                                            "VPACKSSWBZ256rr(b?)(k?)(z?)",
                                            "VPACKSSWBZrr(b?)(k?)(z?)",
                                            "VPACKSSWBrr",
                                            "VPACKUSDWYrr",
                                            "VPACKUSDWZ128rr(b?)(k?)(z?)",
                                            "VPACKUSDWZ256rr(b?)(k?)(z?)",
                                            "VPACKUSDWZrr(b?)(k?)(z?)",
                                            "VPACKUSDWrr",
                                            "VPACKUSWBYrr",
                                            "VPACKUSWBZ128rr(b?)(k?)(z?)",
                                            "VPACKUSWBZ256rr(b?)(k?)(z?)",
                                            "VPACKUSWBZrr(b?)(k?)(z?)",
                                            "VPACKUSWBrr",
                                            "VPALIGNRYrri",
                                            "VPALIGNRZ128rri(b?)(k?)(z?)",
                                            "VPALIGNRZ256rri(b?)(k?)(z?)",
                                            "VPALIGNRZrri(b?)(k?)(z?)",
                                            "VPALIGNRrri",
                                            "VPBLENDWYrri",
                                            "VPBLENDWrri",
                                            "VPBROADCASTDrr",
                                            "VPBROADCASTQrr",
                                            "VPERMILPDYri",
                                            "VPERMILPDYrr",
                                            "VPERMILPDZ128r(b?)i(k?)(z?)",
                                            "VPERMILPDZ128rr(b?)(k?)(z?)",
                                            "VPERMILPDZ256r(b?)i(k?)(z?)",
                                            "VPERMILPDZ256rr(b?)(k?)(z?)",
                                            "VPERMILPDZri(b?)(k?)(z?)",
                                            "VPERMILPDZrr(b?)(k?)(z?)",
                                            "VPERMILPDri",
                                            "VPERMILPDrr",
                                            "VPERMILPSYri",
                                            "VPERMILPSYrr",
                                            "VPERMILPSZ128r(b?)i(k?)(z?)",
                                            "VPERMILPSZ128rr(b?)(k?)(z?)",
                                            "VPERMILPSZ256r(b?)i(k?)(z?)",
                                            "VPERMILPSZ256rr(b?)(k?)(z?)",
                                            "VPERMILPSZri(b?)(k?)(z?)",
                                            "VPERMILPSZrr(b?)(k?)(z?)",
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
                                            "VPSHUFBZ128rr(b?)(k?)(z?)",
                                            "VPSHUFBZ256rr(b?)(k?)(z?)",
                                            "VPSHUFBZrr(b?)(k?)(z?)",
                                            "VPSHUFBrr",
                                            "VPSHUFDYri",
                                            "VPSHUFDZ128r(b?)i(k?)(z?)",
                                            "VPSHUFDZ256r(b?)i(k?)(z?)",
                                            "VPSHUFDZri(b?)(k?)(z?)",
                                            "VPSHUFDri",
                                            "VPSHUFHWYri",
                                            "VPSHUFHWZ128r(b?)i(k?)(z?)",
                                            "VPSHUFHWZ256r(b?)i(k?)(z?)",
                                            "VPSHUFHWZri(b?)(k?)(z?)",
                                            "VPSHUFHWri",
                                            "VPSHUFLWYri",
                                            "VPSHUFLWZ128r(b?)i(k?)(z?)",
                                            "VPSHUFLWZ256r(b?)i(k?)(z?)",
                                            "VPSHUFLWZri(b?)(k?)(z?)",
                                            "VPSHUFLWri",
                                            "VPSLLDQYri",
                                            "VPSLLDQZ128rr(b?)(k?)(z?)",
                                            "VPSLLDQZ256rr(b?)(k?)(z?)",
                                            "VPSLLDQZrr(b?)(k?)(z?)",
                                            "VPSLLDQri",
                                            "VPSRLDQYri",
                                            "VPSRLDQZ128rr(b?)(k?)(z?)",
                                            "VPSRLDQZ256rr(b?)(k?)(z?)",
                                            "VPSRLDQZrr(b?)(k?)(z?)",
                                            "VPSRLDQri",
                                            "VPUNPCKHBWYrr",
                                            "VPUNPCKHBWZ128rr(b?)(k?)(z?)",
                                            "VPUNPCKHBWZ256rr(b?)(k?)(z?)",
                                            "VPUNPCKHBWZrr(b?)(k?)(z?)",
                                            "VPUNPCKHBWrr",
                                            "VPUNPCKHDQYrr",
                                            "VPUNPCKHDQZ128rr(b?)(k?)(z?)",
                                            "VPUNPCKHDQZ256rr(b?)(k?)(z?)",
                                            "VPUNPCKHDQZrr(b?)(k?)(z?)",
                                            "VPUNPCKHDQrr",
                                            "VPUNPCKHQDQYrr",
                                            "VPUNPCKHQDQZ128rr(b?)(k?)(z?)",
                                            "VPUNPCKHQDQZ256rr(b?)(k?)(z?)",
                                            "VPUNPCKHQDQZrr(b?)(k?)(z?)",
                                            "VPUNPCKHQDQrr",
                                            "VPUNPCKHWDYrr",
                                            "VPUNPCKHWDZ128rr(b?)(k?)(z?)",
                                            "VPUNPCKHWDZ256rr(b?)(k?)(z?)",
                                            "VPUNPCKHWDZrr(b?)(k?)(z?)",
                                            "VPUNPCKHWDrr",
                                            "VPUNPCKLBWYrr",
                                            "VPUNPCKLBWZ128rr(b?)(k?)(z?)",
                                            "VPUNPCKLBWZ256rr(b?)(k?)(z?)",
                                            "VPUNPCKLBWZrr(b?)(k?)(z?)",
                                            "VPUNPCKLBWrr",
                                            "VPUNPCKLDQYrr",
                                            "VPUNPCKLDQZ128rr(b?)(k?)(z?)",
                                            "VPUNPCKLDQZ256rr(b?)(k?)(z?)",
                                            "VPUNPCKLDQZrr(b?)(k?)(z?)",
                                            "VPUNPCKLDQrr",
                                            "VPUNPCKLQDQYrr",
                                            "VPUNPCKLQDQZ128rr(b?)(k?)(z?)",
                                            "VPUNPCKLQDQZ256rr(b?)(k?)(z?)",
                                            "VPUNPCKLQDQZrr(b?)(k?)(z?)",
                                            "VPUNPCKLQDQrr",
                                            "VPUNPCKLWDYrr",
                                            "VPUNPCKLWDZ128rr(b?)(k?)(z?)",
                                            "VPUNPCKLWDZ256rr(b?)(k?)(z?)",
                                            "VPUNPCKLWDZrr(b?)(k?)(z?)",
                                            "VPUNPCKLWDrr",
                                            "VSHUFPDYrri",
                                            "VSHUFPDZ128rri(b?)(k?)(z?)",
                                            "VSHUFPDZ256rri(b?)(k?)(z?)",
                                            "VSHUFPDZrri(b?)(k?)(z?)",
                                            "VSHUFPDrri",
                                            "VSHUFPSYrri",
                                            "VSHUFPSZ128rri(b?)(k?)(z?)",
                                            "VSHUFPSZ256rri(b?)(k?)(z?)",
                                            "VSHUFPSZrri(b?)(k?)(z?)",
                                            "VSHUFPSrri",
                                            "VUNPCKHPDYrr",
                                            "VUNPCKHPDZ128rr(b?)(k?)(z?)",
                                            "VUNPCKHPDZ256rr(b?)(k?)(z?)",
                                            "VUNPCKHPDZrr(b?)(k?)(z?)",
                                            "VUNPCKHPDrr",
                                            "VUNPCKHPSYrr",
                                            "VUNPCKHPSZ128rr(b?)(k?)(z?)",
                                            "VUNPCKHPSZ256rr(b?)(k?)(z?)",
                                            "VUNPCKHPSZrr(b?)(k?)(z?)",
                                            "VUNPCKHPSrr",
                                            "VUNPCKLPDYrr",
                                            "VUNPCKLPDZ128rr(b?)(k?)(z?)",
                                            "VUNPCKLPDZ256rr(b?)(k?)(z?)",
                                            "VUNPCKLPDZrr(b?)(k?)(z?)",
                                            "VUNPCKLPDrr",
                                            "VUNPCKLPSYrr",
                                            "VUNPCKLPSZ128rr(b?)(k?)(z?)",
                                            "VUNPCKLPSZ256rr(b?)(k?)(z?)",
                                            "VUNPCKLPSZrr(b?)(k?)(z?)",
                                            "VUNPCKLPSrr")>;

def SKXWriteResGroup4 : SchedWriteRes<[SKXPort6]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup4], (instregex "JMP(16|32|64)r")>;

def SKXWriteResGroup5 : SchedWriteRes<[SKXPort01]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup5], (instregex "PABSBrr",
                                            "PABSDrr",
                                            "PABSWrr",
                                            "PADDSBrr",
                                            "PADDSWrr",
                                            "PADDUSBrr",
                                            "PADDUSWrr",
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
                                            "PMAX(C?)SDrr",
                                            "PMAXSWrr",
                                            "PMAXUBrr",
                                            "PMAXUDrr",
                                            "PMAXUWrr",
                                            "PMINSBrr",
                                            "PMIN(C?)SDrr",
                                            "PMINSWrr",
                                            "PMINUBrr",
                                            "PMINUDrr",
                                            "PMINUWrr",
                                            "PSIGNBrr",
                                            "PSIGNDrr",
                                            "PSIGNWrr",
                                            "PSLLDri",
                                            "PSLLQri",
                                            "PSLLWri",
                                            "PSRADri",
                                            "PSRAWri",
                                            "PSRLDri",
                                            "PSRLQri",
                                            "PSRLWri",
                                            "PSUBSBrr",
                                            "PSUBSWrr",
                                            "PSUBUSBrr",
                                            "PSUBUSWrr",
                                            "VPABSBYrr",
                                            "VPABSBZ128rr(b?)(k?)(z?)",
                                            "VPABSBZ256rr(b?)(k?)(z?)",
                                            "VPABSBZrr(b?)(k?)(z?)",
                                            "VPABSBrr",
                                            "VPABSDYrr",
                                            "VPABSDZ128rr(b?)(k?)(z?)",
                                            "VPABSDZ256rr(b?)(k?)(z?)",
                                            "VPABSDZrr(b?)(k?)(z?)",
                                            "VPABSDrr",
                                            "VPABSQZ128rr(b?)(k?)(z?)",
                                            "VPABSQZ256rr(b?)(k?)(z?)",
                                            "VPABSQZrr(b?)(k?)(z?)",
                                            "VPABSWYrr",
                                            "VPABSWZ128rr(b?)(k?)(z?)",
                                            "VPABSWZ256rr(b?)(k?)(z?)",
                                            "VPABSWZrr(b?)(k?)(z?)",
                                            "VPABSWrr",
                                            "VPADDSBYrr",
                                            "VPADDSBZ128rr(b?)(k?)(z?)",
                                            "VPADDSBZ256rr(b?)(k?)(z?)",
                                            "VPADDSBZrr(b?)(k?)(z?)",
                                            "VPADDSBrr",
                                            "VPADDSWYrr",
                                            "VPADDSWZ128rr(b?)(k?)(z?)",
                                            "VPADDSWZ256rr(b?)(k?)(z?)",
                                            "VPADDSWZrr(b?)(k?)(z?)",
                                            "VPADDSWrr",
                                            "VPADDUSBYrr",
                                            "VPADDUSBZ128rr(b?)(k?)(z?)",
                                            "VPADDUSBZ256rr(b?)(k?)(z?)",
                                            "VPADDUSBZrr(b?)(k?)(z?)",
                                            "VPADDUSBrr",
                                            "VPADDUSWYrr",
                                            "VPADDUSWZ128rr(b?)(k?)(z?)",
                                            "VPADDUSWZ256rr(b?)(k?)(z?)",
                                            "VPADDUSWZrr(b?)(k?)(z?)",
                                            "VPADDUSWrr",
                                            "VPAVGBYrr",
                                            "VPAVGBZ128rr(b?)(k?)(z?)",
                                            "VPAVGBZ256rr(b?)(k?)(z?)",
                                            "VPAVGBZrr(b?)(k?)(z?)",
                                            "VPAVGBrr",
                                            "VPAVGWYrr",
                                            "VPAVGWZ128rr(b?)(k?)(z?)",
                                            "VPAVGWZ256rr(b?)(k?)(z?)",
                                            "VPAVGWZrr(b?)(k?)(z?)",
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
                                            "VPMAXSBZ128rr(b?)(k?)(z?)",
                                            "VPMAXSBZ256rr(b?)(k?)(z?)",
                                            "VPMAXSBZrr(b?)(k?)(z?)",
                                            "VPMAXSBrr",
                                            "VPMAX(C?)SDYrr",
                                            "VPMAX(C?)SDZ128rr(b?)(k?)(z?)",
                                            "VPMAX(C?)SDZ256rr(b?)(k?)(z?)",
                                            "VPMAX(C?)SDZrr(b?)(k?)(z?)",
                                            "VPMAX(C?)SDrr",
                                            "VPMAXSWYrr",
                                            "VPMAXSWZ128rr(b?)(k?)(z?)",
                                            "VPMAXSWZ256rr(b?)(k?)(z?)",
                                            "VPMAXSWZrr(b?)(k?)(z?)",
                                            "VPMAXSWrr",
                                            "VPMAXUBYrr",
                                            "VPMAXUBZ128rr(b?)(k?)(z?)",
                                            "VPMAXUBZ256rr(b?)(k?)(z?)",
                                            "VPMAXUBZrr(b?)(k?)(z?)",
                                            "VPMAXUBrr",
                                            "VPMAXUDYrr",
                                            "VPMAXUDZ128rr(b?)(k?)(z?)",
                                            "VPMAXUDZ256rr(b?)(k?)(z?)",
                                            "VPMAXUDZrr(b?)(k?)(z?)",
                                            "VPMAXUDrr",
                                            "VPMAXUWYrr",
                                            "VPMAXUWZ128rr(b?)(k?)(z?)",
                                            "VPMAXUWZ256rr(b?)(k?)(z?)",
                                            "VPMAXUWZrr(b?)(k?)(z?)",
                                            "VPMAXUWrr",
                                            "VPMINSBYrr",
                                            "VPMINSBZ128rr(b?)(k?)(z?)",
                                            "VPMINSBZ256rr(b?)(k?)(z?)",
                                            "VPMINSBZrr(b?)(k?)(z?)",
                                            "VPMINSBrr",
                                            "VPMIN(C?)SDYrr",
                                            "VPMIN(C?)SDZ128rr(b?)(k?)(z?)",
                                            "VPMIN(C?)SDZ256rr(b?)(k?)(z?)",
                                            "VPMIN(C?)SDZrr(b?)(k?)(z?)",
                                            "VPMIN(C?)SDrr",
                                            "VPMINSWYrr",
                                            "VPMINSWZ128rr(b?)(k?)(z?)",
                                            "VPMINSWZ256rr(b?)(k?)(z?)",
                                            "VPMINSWZrr(b?)(k?)(z?)",
                                            "VPMINSWrr",
                                            "VPMINUBYrr",
                                            "VPMINUBZ128rr(b?)(k?)(z?)",
                                            "VPMINUBZ256rr(b?)(k?)(z?)",
                                            "VPMINUBZrr(b?)(k?)(z?)",
                                            "VPMINUBrr",
                                            "VPMINUDYrr",
                                            "VPMINUDZ128rr(b?)(k?)(z?)",
                                            "VPMINUDZ256rr(b?)(k?)(z?)",
                                            "VPMINUDZrr(b?)(k?)(z?)",
                                            "VPMINUDrr",
                                            "VPMINUWYrr",
                                            "VPMINUWZ128rr(b?)(k?)(z?)",
                                            "VPMINUWZ256rr(b?)(k?)(z?)",
                                            "VPMINUWZrr(b?)(k?)(z?)",
                                            "VPMINUWrr",
                                            "VPROLDZ128r(b?)i(k?)(z?)",
                                            "VPROLDZ256r(b?)i(k?)(z?)",
                                            "VPROLDZri(b?)(k?)(z?)",
                                            "VPROLQZ128r(b?)i(k?)(z?)",
                                            "VPROLQZ256r(b?)i(k?)(z?)",
                                            "VPROLQZri(b?)(k?)(z?)",
                                            "VPROLVDZ128rr(b?)(k?)(z?)",
                                            "VPROLVDZ256rr(b?)(k?)(z?)",
                                            "VPROLVDZrr(b?)(k?)(z?)",
                                            "VPROLVQZ128rr(b?)(k?)(z?)",
                                            "VPROLVQZ256rr(b?)(k?)(z?)",
                                            "VPROLVQZrr(b?)(k?)(z?)",
                                            "VPRORDZ128r(b?)i(k?)(z?)",
                                            "VPRORDZ256r(b?)i(k?)(z?)",
                                            "VPRORDZri(b?)(k?)(z?)",
                                            "VPRORQZ128r(b?)i(k?)(z?)",
                                            "VPRORQZ256r(b?)i(k?)(z?)",
                                            "VPRORQZri(b?)(k?)(z?)",
                                            "VPRORVDZ128rr(b?)(k?)(z?)",
                                            "VPRORVDZ256rr(b?)(k?)(z?)",
                                            "VPRORVDZrr(b?)(k?)(z?)",
                                            "VPRORVQZ128rr(b?)(k?)(z?)",
                                            "VPRORVQZ256rr(b?)(k?)(z?)",
                                            "VPRORVQZrr(b?)(k?)(z?)",
                                            "VPSIGNBYrr",
                                            "VPSIGNBrr",
                                            "VPSIGNDYrr",
                                            "VPSIGNDrr",
                                            "VPSIGNWYrr",
                                            "VPSIGNWrr",
                                            "VPSLLDYri",
                                            "VPSLLDZ128r(b?)i(k?)(z?)",
                                            "VPSLLDZ256r(b?)i(k?)(z?)",
                                            "VPSLLDZri(b?)(k?)(z?)",
                                            "VPSLLDri",
                                            "VPSLLQYri",
                                            "VPSLLQZ128r(b?)i(k?)(z?)",
                                            "VPSLLQZ256r(b?)i(k?)(z?)",
                                            "VPSLLQZri(b?)(k?)(z?)",
                                            "VPSLLQri",
                                            "VPSLLVDYrr",
                                            "VPSLLVDZ128rr(b?)(k?)(z?)",
                                            "VPSLLVDZ256rr(b?)(k?)(z?)",
                                            "VPSLLVDZrr(b?)(k?)(z?)",
                                            "VPSLLVDrr",
                                            "VPSLLVQYrr",
                                            "VPSLLVQZ128rr(b?)(k?)(z?)",
                                            "VPSLLVQZ256rr(b?)(k?)(z?)",
                                            "VPSLLVQZrr(b?)(k?)(z?)",
                                            "VPSLLVQrr",
                                            "VPSLLVWZ128rr(b?)(k?)(z?)",
                                            "VPSLLVWZ256rr(b?)(k?)(z?)",
                                            "VPSLLVWZrr(b?)(k?)(z?)",
                                            "VPSLLWYri",
                                            "VPSLLWZ128ri(b?)(k?)(z?)",
                                            "VPSLLWZ256ri(b?)(k?)(z?)",
                                            "VPSLLWZri(b?)(k?)(z?)",
                                            "VPSLLWri",
                                            "VPSRADYri",
                                            "VPSRADZ128r(b?)i(k?)(z?)",
                                            "VPSRADZ256r(b?)i(k?)(z?)",
                                            "VPSRADZri(b?)(k?)(z?)",
                                            "VPSRADri",
                                            "VPSRAQZ128r(b?)i(k?)(z?)",
                                            "VPSRAQZ256r(b?)i(k?)(z?)",
                                            "VPSRAQZri(b?)(k?)(z?)",
                                            "VPSRAVDYrr",
                                            "VPSRAVDZ128rr(b?)(k?)(z?)",
                                            "VPSRAVDZ256rr(b?)(k?)(z?)",
                                            "VPSRAVDZrr(b?)(k?)(z?)",
                                            "VPSRAVDrr",
                                            "VPSRAVQZ128rr(b?)(k?)(z?)",
                                            "VPSRAVQZ256rr(b?)(k?)(z?)",
                                            "VPSRAVQZrr(b?)(k?)(z?)",
                                            "VPSRAVWZ128rr(b?)(k?)(z?)",
                                            "VPSRAVWZ256rr(b?)(k?)(z?)",
                                            "VPSRAVWZrr(b?)(k?)(z?)",
                                            "VPSRAWYri",
                                            "VPSRAWZ128ri(b?)(k?)(z?)",
                                            "VPSRAWZ256ri(b?)(k?)(z?)",
                                            "VPSRAWZri(b?)(k?)(z?)",
                                            "VPSRAWri",
                                            "VPSRLDYri",
                                            "VPSRLDZ128r(b?)i(k?)(z?)",
                                            "VPSRLDZ256r(b?)i(k?)(z?)",
                                            "VPSRLDZri(b?)(k?)(z?)",
                                            "VPSRLDri",
                                            "VPSRLQYri",
                                            "VPSRLQZ128r(b?)i(k?)(z?)",
                                            "VPSRLQZ256r(b?)i(k?)(z?)",
                                            "VPSRLQZri(b?)(k?)(z?)",
                                            "VPSRLQri",
                                            "VPSRLVDYrr",
                                            "VPSRLVDZ128rr(b?)(k?)(z?)",
                                            "VPSRLVDZ256rr(b?)(k?)(z?)",
                                            "VPSRLVDZrr(b?)(k?)(z?)",
                                            "VPSRLVDrr",
                                            "VPSRLVQYrr",
                                            "VPSRLVQZ128rr(b?)(k?)(z?)",
                                            "VPSRLVQZ256rr(b?)(k?)(z?)",
                                            "VPSRLVQZrr(b?)(k?)(z?)",
                                            "VPSRLVQrr",
                                            "VPSRLVWZ128rr(b?)(k?)(z?)",
                                            "VPSRLVWZ256rr(b?)(k?)(z?)",
                                            "VPSRLVWZrr(b?)(k?)(z?)",
                                            "VPSRLWYri",
                                            "VPSRLWZ128ri(b?)(k?)(z?)",
                                            "VPSRLWZ256ri(b?)(k?)(z?)",
                                            "VPSRLWZri(b?)(k?)(z?)",
                                            "VPSRLWri",
                                            "VPSUBSBYrr",
                                            "VPSUBSBZ128rr(b?)(k?)(z?)",
                                            "VPSUBSBZ256rr(b?)(k?)(z?)",
                                            "VPSUBSBZrr(b?)(k?)(z?)",
                                            "VPSUBSBrr",
                                            "VPSUBSWYrr",
                                            "VPSUBSWZ128rr(b?)(k?)(z?)",
                                            "VPSUBSWZ256rr(b?)(k?)(z?)",
                                            "VPSUBSWZrr(b?)(k?)(z?)",
                                            "VPSUBSWrr",
                                            "VPSUBUSBYrr",
                                            "VPSUBUSBZ128rr(b?)(k?)(z?)",
                                            "VPSUBUSBZ256rr(b?)(k?)(z?)",
                                            "VPSUBUSBZrr(b?)(k?)(z?)",
                                            "VPSUBUSBrr",
                                            "VPSUBUSWYrr",
                                            "VPSUBUSWZ128rr(b?)(k?)(z?)",
                                            "VPSUBUSWZ256rr(b?)(k?)(z?)",
                                            "VPSUBUSWZrr(b?)(k?)(z?)",
                                            "VPSUBUSWrr")>;

def SKXWriteResGroup6 : SchedWriteRes<[SKXPort05]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup6], (instregex "FINCSTP",
                                            "FNOP",
                                            "MMX_MOVQ64rr",
                                            "MMX_PABSBrr",
                                            "MMX_PABSDrr",
                                            "MMX_PABSWrr",
                                            "MMX_PADDBirr",
                                            "MMX_PADDDirr",
                                            "MMX_PADDQirr",
                                            "MMX_PADDWirr",
                                            "MMX_PANDNirr",
                                            "MMX_PANDirr",
                                            "MMX_PORirr",
                                            "MMX_PSIGNBrr",
                                            "MMX_PSIGNDrr",
                                            "MMX_PSIGNWrr",
                                            "MMX_PSUBBirr",
                                            "MMX_PSUBDirr",
                                            "MMX_PSUBQirr",
                                            "MMX_PSUBWirr",
                                            "MMX_PXORirr")>;

def SKXWriteResGroup7 : SchedWriteRes<[SKXPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup7], (instregex "ADC(16|32|64)ri",
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
                                            "CLAC",
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
                                            "SHRX(32|64)rr",
                                            "STAC")>;

def SKXWriteResGroup8 : SchedWriteRes<[SKXPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup8], (instregex "ANDN(32|64)rr",
                                            "BLSI(32|64)rr",
                                            "BLSMSK(32|64)rr",
                                            "BLSR(32|64)rr",
                                            "BZHI(32|64)rr",
                                            "LEA(16|32|64)(_32)?r")>;

def SKXWriteResGroup9 : SchedWriteRes<[SKXPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup9], (instregex "ANDNPDrr",
                                            "ANDNPSrr",
                                            "ANDPDrr",
                                            "ANDPSrr",
                                            "BLENDPDrri",
                                            "BLENDPSrri",
                                            "MOVAPDrr",
                                            "MOVAPSrr",
                                            "MOVDQArr",
                                            "MOVDQUrr",
                                            "MOVPQI2QIrr",
                                            "MOVSSrr",
                                            "ORPDrr",
                                            "ORPSrr",
                                            "PADDBrr",
                                            "PADDDrr",
                                            "PADDQrr",
                                            "PADDWrr",
                                            "PANDNrr",
                                            "PANDrr",
                                            "PORrr",
                                            "PSUBBrr",
                                            "PSUBDrr",
                                            "PSUBQrr",
                                            "PSUBWrr",
                                            "PXORrr",
                                            "VANDNPDYrr",
                                            "VANDNPDZ128rr(b?)(k?)(z?)",
                                            "VANDNPDZ256rr(b?)(k?)(z?)",
                                            "VANDNPDZrr(b?)(k?)(z?)",
                                            "VANDNPDrr",
                                            "VANDNPSYrr",
                                            "VANDNPSZ128rr(b?)(k?)(z?)",
                                            "VANDNPSZ256rr(b?)(k?)(z?)",
                                            "VANDNPSZrr(b?)(k?)(z?)",
                                            "VANDNPSrr",
                                            "VANDPDYrr",
                                            "VANDPDZ128rr(b?)(k?)(z?)",
                                            "VANDPDZ256rr(b?)(k?)(z?)",
                                            "VANDPDZrr(b?)(k?)(z?)",
                                            "VANDPDrr",
                                            "VANDPSYrr",
                                            "VANDPSZ128rr(b?)(k?)(z?)",
                                            "VANDPSZ256rr(b?)(k?)(z?)",
                                            "VANDPSZrr(b?)(k?)(z?)",
                                            "VANDPSrr",
                                            "VBLENDMPDZ128rr(b?)(k?)(z?)",
                                            "VBLENDMPDZ256rr(b?)(k?)(z?)",
                                            "VBLENDMPDZrr(b?)(k?)(z?)",
                                            "VBLENDMPSZ128rr(b?)(k?)(z?)",
                                            "VBLENDMPSZ256rr(b?)(k?)(z?)",
                                            "VBLENDMPSZrr(b?)(k?)(z?)",
                                            "VBLENDPDYrri",
                                            "VBLENDPDrri",
                                            "VBLENDPSYrri",
                                            "VBLENDPSrri",
                                            "VMOVAPDYrr",
                                            "VMOVAPDZ128rr(b?)(k?)(z?)",
                                            "VMOVAPDZ256rr(b?)(k?)(z?)",
                                            "VMOVAPDZrr(b?)(k?)(z?)",
                                            "VMOVAPDrr",
                                            "VMOVAPSYrr",
                                            "VMOVAPSZ128rr(b?)(k?)(z?)",
                                            "VMOVAPSZ256rr(b?)(k?)(z?)",
                                            "VMOVAPSZrr(b?)(k?)(z?)",
                                            "VMOVAPSrr",
                                            "VMOVDQA32Z128rr(b?)(k?)(z?)",
                                            "VMOVDQA32Z256rr(b?)(k?)(z?)",
                                            "VMOVDQA32Zrr(b?)(k?)(z?)",
                                            "VMOVDQA64Z128rr(b?)(k?)(z?)",
                                            "VMOVDQA64Z256rr(b?)(k?)(z?)",
                                            "VMOVDQA64Zrr(b?)(k?)(z?)",
                                            "VMOVDQAYrr",
                                            "VMOVDQArr",
                                            "VMOVDQU16Z128rr(b?)(k?)(z?)",
                                            "VMOVDQU16Z256rr(b?)(k?)(z?)",
                                            "VMOVDQU16Zrr(b?)(k?)(z?)",
                                            "VMOVDQU32Z128rr(b?)(k?)(z?)",
                                            "VMOVDQU32Z256rr(b?)(k?)(z?)",
                                            "VMOVDQU32Zrr(b?)(k?)(z?)",
                                            "VMOVDQU64Z128rr(b?)(k?)(z?)",
                                            "VMOVDQU64Z256rr(b?)(k?)(z?)",
                                            "VMOVDQU64Zrr(b?)(k?)(z?)",
                                            "VMOVDQU8Z128rr(b?)(k?)(z?)",
                                            "VMOVDQU8Z256rr(b?)(k?)(z?)",
                                            "VMOVDQU8Zrr(b?)(k?)(z?)",
                                            "VMOVDQUYrr",
                                            "VMOVDQUrr",
                                            "VMOVPQI(2Q|Lo2PQ)IZrr(b?)(k?)(z?)",
                                            "VMOVPQI2QIrr",
                                            "VMOVSSrr",
                                            "VMOVUPDZ128rr(b?)(k?)(z?)",
                                            "VMOVUPDZ256rr(b?)(k?)(z?)",
                                            "VMOVUPDZrr(b?)(k?)(z?)",
                                            "VMOVUPSZ128rr(b?)(k?)(z?)",
                                            "VMOVUPSZ256rr(b?)(k?)(z?)",
                                            "VMOVUPSZrr(b?)(k?)(z?)",
                                            "VMOVZPQILo2PQIrr",
                                            "VORPDYrr",
                                            "VORPDZ128rr(b?)(k?)(z?)",
                                            "VORPDZ256rr(b?)(k?)(z?)",
                                            "VORPDZrr(b?)(k?)(z?)",
                                            "VORPDrr",
                                            "VORPSYrr",
                                            "VORPSZ128rr(b?)(k?)(z?)",
                                            "VORPSZ256rr(b?)(k?)(z?)",
                                            "VORPSZrr(b?)(k?)(z?)",
                                            "VORPSrr",
                                            "VPADDBYrr",
                                            "VPADDBZ128rr(b?)(k?)(z?)",
                                            "VPADDBZ256rr(b?)(k?)(z?)",
                                            "VPADDBZrr(b?)(k?)(z?)",
                                            "VPADDBrr",
                                            "VPADDDYrr",
                                            "VPADDDZ128rr(b?)(k?)(z?)",
                                            "VPADDDZ256rr(b?)(k?)(z?)",
                                            "VPADDDZrr(b?)(k?)(z?)",
                                            "VPADDDrr",
                                            "VPADDQYrr",
                                            "VPADDQZ128rr(b?)(k?)(z?)",
                                            "VPADDQZ256rr(b?)(k?)(z?)",
                                            "VPADDQZrr(b?)(k?)(z?)",
                                            "VPADDQrr",
                                            "VPADDWYrr",
                                            "VPADDWZ128rr(b?)(k?)(z?)",
                                            "VPADDWZ256rr(b?)(k?)(z?)",
                                            "VPADDWZrr(b?)(k?)(z?)",
                                            "VPADDWrr",
                                            "VPANDDZ128rr(b?)(k?)(z?)",
                                            "VPANDDZ256rr(b?)(k?)(z?)",
                                            "VPANDDZrr(b?)(k?)(z?)",
                                            "VPANDNDZ128rr(b?)(k?)(z?)",
                                            "VPANDNDZ256rr(b?)(k?)(z?)",
                                            "VPANDNDZrr(b?)(k?)(z?)",
                                            "VPANDNQZ128rr(b?)(k?)(z?)",
                                            "VPANDNQZ256rr(b?)(k?)(z?)",
                                            "VPANDNQZrr(b?)(k?)(z?)",
                                            "VPANDNYrr",
                                            "VPANDNrr",
                                            "VPANDQZ128rr(b?)(k?)(z?)",
                                            "VPANDQZ256rr(b?)(k?)(z?)",
                                            "VPANDQZrr(b?)(k?)(z?)",
                                            "VPANDYrr",
                                            "VPANDrr",
                                            "VPBLENDDYrri",
                                            "VPBLENDDrri",
                                            "VPBLENDMBZ128rr(b?)(k?)(z?)",
                                            "VPBLENDMBZ256rr(b?)(k?)(z?)",
                                            "VPBLENDMBZrr(b?)(k?)(z?)",
                                            "VPBLENDMDZ128rr(b?)(k?)(z?)",
                                            "VPBLENDMDZ256rr(b?)(k?)(z?)",
                                            "VPBLENDMDZrr(b?)(k?)(z?)",
                                            "VPBLENDMQZ128rr(b?)(k?)(z?)",
                                            "VPBLENDMQZ256rr(b?)(k?)(z?)",
                                            "VPBLENDMQZrr(b?)(k?)(z?)",
                                            "VPBLENDMWZ128rr(b?)(k?)(z?)",
                                            "VPBLENDMWZ256rr(b?)(k?)(z?)",
                                            "VPBLENDMWZrr(b?)(k?)(z?)",
                                            "VPORDZ128rr(b?)(k?)(z?)",
                                            "VPORDZ256rr(b?)(k?)(z?)",
                                            "VPORDZrr(b?)(k?)(z?)",
                                            "VPORQZ128rr(b?)(k?)(z?)",
                                            "VPORQZ256rr(b?)(k?)(z?)",
                                            "VPORQZrr(b?)(k?)(z?)",
                                            "VPORYrr",
                                            "VPORrr",
                                            "VPSUBBYrr",
                                            "VPSUBBZ128rr(b?)(k?)(z?)",
                                            "VPSUBBZ256rr(b?)(k?)(z?)",
                                            "VPSUBBZrr(b?)(k?)(z?)",
                                            "VPSUBBrr",
                                            "VPSUBDYrr",
                                            "VPSUBDZ128rr(b?)(k?)(z?)",
                                            "VPSUBDZ256rr(b?)(k?)(z?)",
                                            "VPSUBDZrr(b?)(k?)(z?)",
                                            "VPSUBDrr",
                                            "VPSUBQYrr",
                                            "VPSUBQZ128rr(b?)(k?)(z?)",
                                            "VPSUBQZ256rr(b?)(k?)(z?)",
                                            "VPSUBQZrr(b?)(k?)(z?)",
                                            "VPSUBQrr",
                                            "VPSUBWYrr",
                                            "VPSUBWZ128rr(b?)(k?)(z?)",
                                            "VPSUBWZrr(b?)(k?)(z?)",
                                            "VPSUBWrr",
                                            "VPTERNLOGDZ128rri(b?)(k?)(z?)",
                                            "VPTERNLOGDZ256rri(b?)(k?)(z?)",
                                            "VPTERNLOGDZrri(b?)(k?)(z?)",
                                            "VPTERNLOGQZ128rri(b?)(k?)(z?)",
                                            "VPTERNLOGQZ256rri(b?)(k?)(z?)",
                                            "VPTERNLOGQZrri(b?)(k?)(z?)",
                                            "VPXORDZ128rr(b?)(k?)(z?)",
                                            "VPXORDZ256rr(b?)(k?)(z?)",
                                            "VPXORDZrr(b?)(k?)(z?)",
                                            "VPXORQZ128rr(b?)(k?)(z?)",
                                            "VPXORQZ256rr(b?)(k?)(z?)",
                                            "VPXORQZrr(b?)(k?)(z?)",
                                            "VPXORYrr",
                                            "VPXORrr",
                                            "VXORPDYrr",
                                            "VXORPDZ128rr(b?)(k?)(z?)",
                                            "VXORPDZ256rr(b?)(k?)(z?)",
                                            "VXORPDZrr(b?)(k?)(z?)",
                                            "VXORPDrr",
                                            "VXORPSYrr",
                                            "VXORPSZ128rr(b?)(k?)(z?)",
                                            "VXORPSZ256rr(b?)(k?)(z?)",
                                            "VXORPSZrr(b?)(k?)(z?)",
                                            "VXORPSrr",
                                            "XORPDrr",
                                            "XORPSrr")>;

def SKXWriteResGroup10 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup10], (instrs CWDE)>;
def: InstRW<[SKXWriteResGroup10], (instregex "ADD(8|16|32|64)ri",
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

def SKXWriteResGroup11 : SchedWriteRes<[SKXPort4,SKXPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup11], (instregex "FBSTPm",
                                             "KMOVBmk",
                                             "KMOVDmk",
                                             "KMOVQmk",
                                             "KMOVWmk",
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
                                             "VEXTRACTF32x4Z256mr(b?)(k?)(z?)",
                                             "VEXTRACTF32x4Zmr(b?)(k?)(z?)",
                                             "VEXTRACTF32x8Zmr(b?)(k?)(z?)",
                                             "VEXTRACTF64x2Z256mr(b?)(k?)(z?)",
                                             "VEXTRACTF64x2Zmr(b?)(k?)(z?)",
                                             "VEXTRACTF64x4Zmr(b?)(k?)(z?)",
                                             "VEXTRACTI128mr",
                                             "VEXTRACTI32x4Z256mr(b?)(k?)(z?)",
                                             "VEXTRACTI32x4Zmr(b?)(k?)(z?)",
                                             "VEXTRACTI32x8Zmr(b?)(k?)(z?)",
                                             "VEXTRACTI64x2Z256mr(b?)(k?)(z?)",
                                             "VEXTRACTI64x2Zmr(b?)(k?)(z?)",
                                             "VEXTRACTI64x4Zmr(b?)(k?)(z?)",
                                             "VMOVAPDYmr",
                                             "VMOVAPDZ128mr(b?)(k?)(z?)",
                                             "VMOVAPDZ256mr(b?)(k?)(z?)",
                                             "VMOVAPDZmr(b?)(k?)(z?)",
                                             "VMOVAPDmr",
                                             "VMOVAPSYmr",
                                             "VMOVAPSZ128mr(b?)(k?)(z?)",
                                             "VMOVAPSZ256mr(b?)(k?)(z?)",
                                             "VMOVAPSZmr(b?)(k?)(z?)",
                                             "VMOVAPSmr",
                                             "VMOVDQA32Z128mr(b?)(k?)(z?)",
                                             "VMOVDQA32Z256mr(b?)(k?)(z?)",
                                             "VMOVDQA32Zmr(b?)(k?)(z?)",
                                             "VMOVDQA64Z128mr(b?)(k?)(z?)",
                                             "VMOVDQA64Z256mr(b?)(k?)(z?)",
                                             "VMOVDQA64Zmr(b?)(k?)(z?)",
                                             "VMOVDQAYmr",
                                             "VMOVDQAmr",
                                             "VMOVDQU16Z128mr(b?)(k?)(z?)",
                                             "VMOVDQU16Z256mr(b?)(k?)(z?)",
                                             "VMOVDQU16Zmr(b?)(k?)(z?)",
                                             "VMOVDQU32Z128mr(b?)(k?)(z?)",
                                             "VMOVDQU32Z256mr(b?)(k?)(z?)",
                                             "VMOVDQU32Zmr(b?)(k?)(z?)",
                                             "VMOVDQU64Z128mr(b?)(k?)(z?)",
                                             "VMOVDQU64Z256mr(b?)(k?)(z?)",
                                             "VMOVDQU64Zmr(b?)(k?)(z?)",
                                             "VMOVDQU8Z128mr(b?)(k?)(z?)",
                                             "VMOVDQU8Z256mr(b?)(k?)(z?)",
                                             "VMOVDQUYmr",
                                             "VMOVDQUmr",
                                             "VMOVHPDZ128mr(b?)(k?)(z?)",
                                             "VMOVHPDmr",
                                             "VMOVHPSZ128mr(b?)(k?)(z?)",
                                             "VMOVHPSmr",
                                             "VMOVLPDZ128mr(b?)(k?)(z?)",
                                             "VMOVLPDmr",
                                             "VMOVLPSZ128mr(b?)(k?)(z?)",
                                             "VMOVLPSmr",
                                             "VMOVNTDQYmr",
                                             "VMOVNTDQZ128mr(b?)(k?)(z?)",
                                             "VMOVNTDQZ256mr(b?)(k?)(z?)",
                                             "VMOVNTDQZmr(b?)(k?)(z?)",
                                             "VMOVNTDQmr",
                                             "VMOVNTPDYmr",
                                             "VMOVNTPDZ128mr(b?)(k?)(z?)",
                                             "VMOVNTPDZ256mr(b?)(k?)(z?)",
                                             "VMOVNTPDZmr(b?)(k?)(z?)",
                                             "VMOVNTPDmr",
                                             "VMOVNTPSYmr",
                                             "VMOVNTPSZ128mr(b?)(k?)(z?)",
                                             "VMOVNTPSZ256mr(b?)(k?)(z?)",
                                             "VMOVNTPSZmr(b?)(k?)(z?)",
                                             "VMOVNTPSmr",
                                             "VMOVPDI2DIZmr(b?)(k?)(z?)",
                                             "VMOVPDI2DImr",
                                             "VMOVPQI(2QI|to64)Zmr(b?)(k?)(z?)",
                                             "VMOVPQI2QImr",
                                             "VMOVPQIto64mr",
                                             "VMOVSDZmr(b?)(k?)(z?)",
                                             "VMOVSDmr",
                                             "VMOVSSZmr(b?)(k?)(z?)",
                                             "VMOVSSmr",
                                             "VMOVUPDYmr",
                                             "VMOVUPDZ128mr(b?)(k?)(z?)",
                                             "VMOVUPDZ256mr(b?)(k?)(z?)",
                                             "VMOVUPDZmr(b?)(k?)(z?)",
                                             "VMOVUPDmr",
                                             "VMOVUPSYmr",
                                             "VMOVUPSZ128mr(b?)(k?)(z?)",
                                             "VMOVUPSZ256mr(b?)(k?)(z?)",
                                             "VMOVUPSZmr(b?)(k?)(z?)",
                                             "VMOVUPSmr",
                                             "VMPTRSTm")>;

def SKXWriteResGroup12 : SchedWriteRes<[SKXPort0]> {
  let Latency = 2;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup12], (instregex "COMISDrr",
                                             "COMISSrr",
                                             "MMX_MOVD64from64rr",
                                             "MMX_MOVD64grr",
                                             "MMX_PMOVMSKBrr",
                                             "MOVMSKPDrr",
                                             "MOVMSKPSrr",
                                             "MOVPDI2DIrr",
                                             "MOVPQIto64rr",
                                             "PMOVMSKBrr",
                                             "UCOMISDrr",
                                             "UCOMISSrr",
                                             "VCOMISDZrr(b?)",
                                             "VCOMISDrr",
                                             "VCOMISSZrr(b?)",
                                             "VCOMISSrr",
                                             "VMOVMSKPDYrr",
                                             "VMOVMSKPDrr",
                                             "VMOVMSKPSYrr",
                                             "VMOVMSKPSrr",
                                             "VMOVPDI2DIZrr(b?)(k?)(z?)",
                                             "VMOVPDI2DIrr",
                                             "VMOVPQIto64Zrr(b?)(k?)(z?)",
                                             "VMOVPQIto64rr",
                                             "VPMOVMSKBYrr",
                                             "VPMOVMSKBrr",
                                             "VTESTPDYrr",
                                             "VTESTPDrr",
                                             "VTESTPSYrr",
                                             "VTESTPSrr",
                                             "VUCOMISDZrr(b?)",
                                             "VUCOMISDrr",
                                             "VUCOMISSZrr(b?)",
                                             "VUCOMISSrr")>;

def SKXWriteResGroup13 : SchedWriteRes<[SKXPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup13], (instregex "MMX_MOVQ2DQrr",
                                             "MMX_PINSRWrr",
                                             "PINSRBrr",
                                             "PINSRDrr",
                                             "PINSRQrr",
                                             "PINSRWrr",
                                             "VPINSRBZrr(b?)(k?)(z?)",
                                             "VPINSRBrr",
                                             "VPINSRDZrr(b?)(k?)(z?)",
                                             "VPINSRDrr",
                                             "VPINSRQZrr(b?)(k?)(z?)",
                                             "VPINSRQrr",
                                             "VPINSRWZrr(b?)(k?)(z?)",
                                             "VPINSRWrr")>;

def SKXWriteResGroup14 : SchedWriteRes<[SKXPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup14], (instregex "FDECSTP",
                                             "MMX_MOVDQ2Qrr")>;

def SKXWriteResGroup15 : SchedWriteRes<[SKXPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup15], (instregex "CMOV(A|BE)(16|32|64)rr",
                                             "ROL(8|16|32|64)r1",
                                             "ROL(8|16|32|64)ri",
                                             "ROR(8|16|32|64)r1",
                                             "ROR(8|16|32|64)ri",
                                             "SET(A|BE)r")>;

def SKXWriteResGroup16 : SchedWriteRes<[SKXPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup16], (instregex "BLENDVPDrr0",
                                             "BLENDVPSrr0",
                                             "PBLENDVBrr0",
                                             "VBLENDVPDYrr",
                                             "VBLENDVPDrr",
                                             "VBLENDVPSYrr",
                                             "VBLENDVPSrr",
                                             "VPBLENDVBYrr",
                                             "VPBLENDVBrr")>;

def SKXWriteResGroup17 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup17], (instregex "LFENCE",
                                             "WAIT",
                                             "XGETBV")>;

def SKXWriteResGroup18 : SchedWriteRes<[SKXPort0,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup18], (instregex "VMASKMOVPDYmr",
                                             "VMASKMOVPDmr",
                                             "VMASKMOVPSYmr",
                                             "VMASKMOVPSmr",
                                             "VPMASKMOVDYmr",
                                             "VPMASKMOVDmr",
                                             "VPMASKMOVQYmr",
                                             "VPMASKMOVQmr")>;

def SKXWriteResGroup19 : SchedWriteRes<[SKXPort5,SKXPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup19], (instregex "PSLLDrr",
                                             "PSLLQrr",
                                             "PSLLWrr",
                                             "PSRADrr",
                                             "PSRAWrr",
                                             "PSRLDrr",
                                             "PSRLQrr",
                                             "PSRLWrr",
                                             "VPSLLDZ128rr(b?)(k?)(z?)",
                                             "VPSLLDrr",
                                             "VPSLLQZ128rr(b?)(k?)(z?)",
                                             "VPSLLQrr",
                                             "VPSLLWZ128rr(b?)(k?)(z?)",
                                             "VPSLLWrr",
                                             "VPSRADZ128rr(b?)(k?)(z?)",
                                             "VPSRADrr",
                                             "VPSRAQZ128rr(b?)(k?)(z?)",
                                             "VPSRAWZ128rr(b?)(k?)(z?)",
                                             "VPSRAWrr",
                                             "VPSRLDZ128rr(b?)(k?)(z?)",
                                             "VPSRLDrr",
                                             "VPSRLQZ128rr(b?)(k?)(z?)",
                                             "VPSRLQrr",
                                             "VPSRLWrr")>;

def SKXWriteResGroup20 : SchedWriteRes<[SKXPort6,SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup20], (instregex "CLFLUSH")>;

def SKXWriteResGroup21 : SchedWriteRes<[SKXPort237,SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup21], (instregex "SFENCE")>;

def SKXWriteResGroup22 : SchedWriteRes<[SKXPort06,SKXPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup22], (instregex "BEXTR(32|64)rr",
                                             "BSWAP(16|32|64)r")>;

def SKXWriteResGroup23 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup23], (instrs CWD)>;
def: InstRW<[SKXWriteResGroup23], (instrs JCXZ, JECXZ, JRCXZ)>;
def: InstRW<[SKXWriteResGroup23], (instregex "ADC8i8",
                                             "ADC8ri",
                                             "SBB8i8",
                                             "SBB8ri")>;

def SKXWriteResGroup24 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup24], (instregex "EXTRACTPSmr",
                                             "PEXTRBmr",
                                             "PEXTRDmr",
                                             "PEXTRQmr",
                                             "PEXTRWmr",
                                             "STMXCSR",
                                             "VEXTRACTPSZmr(b?)(k?)(z?)",
                                             "VEXTRACTPSmr",
                                             "VPEXTRBZmr(b?)(k?)(z?)",
                                             "VPEXTRBmr",
                                             "VPEXTRDZmr(b?)(k?)(z?)",
                                             "VPEXTRDmr",
                                             "VPEXTRQZmr(b?)(k?)(z?)",
                                             "VPEXTRQmr",
                                             "VPEXTRWZmr(b?)(k?)(z?)",
                                             "VPEXTRWmr",
                                             "VSTMXCSR")>;

def SKXWriteResGroup25 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup25], (instregex "FNSTCW16m")>;

def SKXWriteResGroup26 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort06]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup26], (instregex "SET(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)m")>;

def SKXWriteResGroup27 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort15]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup27], (instregex "MOVBE(16|32|64)mr")>;

def SKXWriteResGroup28 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup28], (instrs PUSH16r, PUSH32r, PUSH64r)>;
def: InstRW<[SKXWriteResGroup28], (instregex "PUSH(16|32|64)rmr",
                                             "PUSH64i8",
                                             "STOSB",
                                             "STOSL",
                                             "STOSQ",
                                             "STOSW")>;

def SKXWriteResGroup29 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort15]> {
  let Latency = 2;
  let NumMicroOps = 5;
  let ResourceCycles = [2,2,1];
}
def: InstRW<[SKXWriteResGroup29], (instregex "VMOVDQU8Zmr(b?)(k?)(z?)")>;

def SKXWriteResGroup30 : SchedWriteRes<[SKXPort0]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup30], (instregex "KADDBrr",
                                             "KADDDrr",
                                             "KADDQrr",
                                             "KADDWrr",
                                             "KMOVBrk",
                                             "KMOVDrk",
                                             "KMOVQrk",
                                             "KMOVWrk",
                                             "KORTESTBrr",
                                             "KORTESTDrr",
                                             "KORTESTQrr",
                                             "KORTESTWrr",
                                             "KTESTBrr",
                                             "KTESTDrr",
                                             "KTESTQrr",
                                             "KTESTWrr")>;

def SKXWriteResGroup31 : SchedWriteRes<[SKXPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup31], (instrs IMUL16rr, IMUL32rr, IMUL32rri, IMUL32rri8, IMUL64rr, IMUL64rri32, IMUL64rri8)>;
def: InstRW<[SKXWriteResGroup31], (instrs IMUL8r, MUL8r)>;
def: InstRW<[SKXWriteResGroup31], (instregex "BSF(16|32|64)rr",
                                             "BSR(16|32|64)rr",
                                             "LZCNT(16|32|64)rr",
                                             "PDEP(32|64)rr",
                                             "PEXT(32|64)rr",
                                             "POPCNT(16|32|64)rr",
                                             "SHLD(16|32|64)rri8",
                                             "SHRD(16|32|64)rri8",
                                             "TZCNT(16|32|64)rr")>;

def SKXWriteResGroup31_16i : SchedWriteRes<[SKXPort1, SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup31_16i], (instrs IMUL16rri, IMUL16rri8)>;


def SKXWriteResGroup32 : SchedWriteRes<[SKXPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup32], (instregex "ADD_FPrST0",
                                             "ADD_FST0r",
                                             "ADD_FrST0",
                                             "KSHIFTLBri",
                                             "KSHIFTLDri",
                                             "KSHIFTLQri",
                                             "KSHIFTLWri",
                                             "KSHIFTRBri",
                                             "KSHIFTRDri",
                                             "KSHIFTRQri",
                                             "KSHIFTRWri",
                                             "KUNPCKBWrr",
                                             "KUNPCKDQrr",
                                             "KUNPCKWDrr",
                                             "MMX_PSADBWirr",
                                             "PCMPGTQrr",
                                             "PSADBWrr",
                                             "SUBR_FPrST0",
                                             "SUBR_FST0r",
                                             "SUBR_FrST0",
                                             "SUB_FPrST0",
                                             "SUB_FST0r",
                                             "SUB_FrST0",
                                             "VALIGNDZ128rri(b?)(k?)(z?)",
                                             "VALIGNDZ256rri(b?)(k?)(z?)",
                                             "VALIGNDZrri(b?)(k?)(z?)",
                                             "VALIGNQZ128rri(b?)(k?)(z?)",
                                             "VALIGNQZ256rri(b?)(k?)(z?)",
                                             "VALIGNQZrri(b?)(k?)(z?)",
                                             "VBROADCASTF32X2Z256r(b?)(k?)(z?)",
                                             "VBROADCASTF32X2Zr(b?)(k?)(z?)",
                                             "VBROADCASTI32X2Z256r(b?)(k?)(z?)",
                                             "VBROADCASTI32X2Zr(b?)(k?)(z?)",
                                             "VBROADCASTSDYrr",
                                             "VBROADCASTSDZ256r(b?)(k?)(z?)",
                                             "VBROADCASTSDZr(b?)(k?)(z?)",
                                             "VBROADCASTSSYrr",
                                             "VBROADCASTSSZ128r(b?)(k?)(z?)",
                                             "VBROADCASTSSZ256r(b?)(k?)(z?)",
                                             "VBROADCASTSSZr(b?)(k?)(z?)",
                                             "VCMPPDZ128rri(b?)(k?)(z?)",
                                             "VCMPPDZ256rri(b?)(k?)(z?)",
                                             "VCMPPDZrri(b?)(k?)(z?)",
                                             "VCMPPSZ128rri(b?)(k?)(z?)",
                                             "VCMPPSZ256rri(b?)(k?)(z?)",
                                             "VCMPPSZrri(b?)(k?)(z?)",
                                             "VCMPSDZrr(b?)(_Int)?(k?)(z?)",
                                             "VCMPSSZrr(b?)(_Int)?(k?)(z?)",
                                             "VDBPSADBWZ128rri(b?)(k?)(z?)",
                                             "VDBPSADBWZ256rri(b?)(k?)(z?)",
                                             "VDBPSADBWZrri(b?)(k?)(z?)",
                                             "VEXTRACTF128rr",
                                             "VEXTRACTF32x4Z256rr(b?)(k?)(z?)",
                                             "VEXTRACTF32x4Zrr(b?)(k?)(z?)",
                                             "VEXTRACTF32x8Zrr(b?)(k?)(z?)",
                                             "VEXTRACTF64x2Z256rr(b?)(k?)(z?)",
                                             "VEXTRACTF64x2Zrr(b?)(k?)(z?)",
                                             "VEXTRACTF64x4Zrr(b?)(k?)(z?)",
                                             "VEXTRACTI128rr",
                                             "VEXTRACTI32x4Z256rr(b?)(k?)(z?)",
                                             "VEXTRACTI32x4Zrr(b?)(k?)(z?)",
                                             "VEXTRACTI32x8Zrr(b?)(k?)(z?)",
                                             "VEXTRACTI64x2Z256rr(b?)(k?)(z?)",
                                             "VEXTRACTI64x2Zrr(b?)(k?)(z?)",
                                             "VEXTRACTI64x4Zrr(b?)(k?)(z?)",
                                             "VFPCLASSPDZ128rr(b?)(k?)(z?)",
                                             "VFPCLASSPDZ256rr(b?)(k?)(z?)",
                                             "VFPCLASSPDZrr(b?)(k?)(z?)",
                                             "VFPCLASSPSZ128rr(b?)(k?)(z?)",
                                             "VFPCLASSPSZ256rr(b?)(k?)(z?)",
                                             "VFPCLASSPSZrr(b?)(k?)(z?)",
                                             "VFPCLASSSDrr(b?)(k?)(z?)",
                                             "VFPCLASSSSrr(b?)(k?)(z?)",
                                             "VINSERTF128rr",
                                             "VINSERTF32x4Z256rr(b?)(k?)(z?)",
                                             "VINSERTF32x4Zrr(b?)(k?)(z?)",
                                             "VINSERTF32x8Zrr(b?)(k?)(z?)",
                                             "VINSERTF64x2Z256rr(b?)(k?)(z?)",
                                             "VINSERTF64x2Zrr(b?)(k?)(z?)",
                                             "VINSERTF64x4Zrr(b?)(k?)(z?)",
                                             "VINSERTI128rr",
                                             "VINSERTI32x4Z256rr(b?)(k?)(z?)",
                                             "VINSERTI32x4Zrr(b?)(k?)(z?)",
                                             "VINSERTI32x8Zrr(b?)(k?)(z?)",
                                             "VINSERTI64x2Z256rr(b?)(k?)(z?)",
                                             "VINSERTI64x2Zrr(b?)(k?)(z?)",
                                             "VINSERTI64x4Zrr(b?)(k?)(z?)",
                                             "VPBROADCASTBYrr",
                                             "VPBROADCASTBZ128r(b?)(k?)(z?)",
                                             "VPBROADCASTBZ256r(b?)(k?)(z?)",
                                             "VPBROADCASTBZr(b?)(k?)(z?)",
                                             "VPBROADCASTBrr",
                                             "VPBROADCASTDYrr",
                                             "VPBROADCASTDZ128r(b?)(k?)(z?)",
                                             "VPBROADCASTDZ256r(b?)(k?)(z?)",
                                             "VPBROADCASTDZr(b?)(k?)(z?)",
                                             "VPBROADCASTDrZ128r(b?)(k?)(z?)",
                                             "VPBROADCASTDrZ256r(b?)(k?)(z?)",
                                             "VPBROADCASTDrZr(b?)(k?)(z?)",
                                             "VPBROADCASTQYrr",
                                             "VPBROADCASTQZ128r(b?)(k?)(z?)",
                                             "VPBROADCASTQZ256r(b?)(k?)(z?)",
                                             "VPBROADCASTQZr(b?)(k?)(z?)",
                                             "VPBROADCASTQrZ128r(b?)(k?)(z?)",
                                             "VPBROADCASTQrZ256r(b?)(k?)(z?)",
                                             "VPBROADCASTQrZr(b?)(k?)(z?)",
                                             "VPBROADCASTWYrr",
                                             "VPBROADCASTWZ128r(b?)(k?)(z?)",
                                             "VPBROADCASTWZ256r(b?)(k?)(z?)",
                                             "VPBROADCASTWZr(b?)(k?)(z?)",
                                             "VPBROADCASTWrr",
                                             "VPCMPBZ128rri(b?)(k?)(z?)",
                                             "VPCMPBZ256rri(b?)(k?)(z?)",
                                             "VPCMPBZrri(b?)(k?)(z?)",
                                             "VPCMPDZ128rri(b?)(k?)(z?)",
                                             "VPCMPDZ256rri(b?)(k?)(z?)",
                                             "VPCMPDZrri(b?)(k?)(z?)",
                                             "VPCMPEQBZ128rr(b?)(k?)(z?)",
                                             "VPCMPEQBZ256rr(b?)(k?)(z?)",
                                             "VPCMPEQBZrr(b?)(k?)(z?)",
                                             "VPCMPEQDZ128rr(b?)(k?)(z?)",
                                             "VPCMPEQDZ256rr(b?)(k?)(z?)",
                                             "VPCMPEQDZrr(b?)(k?)(z?)",
                                             "VPCMPEQQZ128rr(b?)(k?)(z?)",
                                             "VPCMPEQQZ256rr(b?)(k?)(z?)",
                                             "VPCMPEQQZrr(b?)(k?)(z?)",
                                             "VPCMPEQWZ128rr(b?)(k?)(z?)",
                                             "VPCMPEQWZ256rr(b?)(k?)(z?)",
                                             "VPCMPEQWZrr(b?)(k?)(z?)",
                                             "VPCMPGTBZ128rr(b?)(k?)(z?)",
                                             "VPCMPGTBZ256rr(b?)(k?)(z?)",
                                             "VPCMPGTBZrr(b?)(k?)(z?)",
                                             "VPCMPGTDZ128rr(b?)(k?)(z?)",
                                             "VPCMPGTDZ256rr(b?)(k?)(z?)",
                                             "VPCMPGTDZrr(b?)(k?)(z?)",
                                             "VPCMPGTQYrr",
                                             "VPCMPGTQZ128rr(b?)(k?)(z?)",
                                             "VPCMPGTQZ256rr(b?)(k?)(z?)",
                                             "VPCMPGTQZrr(b?)(k?)(z?)",
                                             "VPCMPGTQrr",
                                             "VPCMPGTWZ128rr(b?)(k?)(z?)",
                                             "VPCMPGTWZ256rr(b?)(k?)(z?)",
                                             "VPCMPGTWZrr(b?)(k?)(z?)",
                                             "VPCMPQZ128rri(b?)(k?)(z?)",
                                             "VPCMPQZ256rri(b?)(k?)(z?)",
                                             "VPCMPQZrri(b?)(k?)(z?)",
                                             "VPCMPUBZ128rri(b?)(k?)(z?)",
                                             "VPCMPUBZ256rri(b?)(k?)(z?)",
                                             "VPCMPUBZrri(b?)(k?)(z?)",
                                             "VPCMPUDZ128rri(b?)(k?)(z?)",
                                             "VPCMPUDZ256rri(b?)(k?)(z?)",
                                             "VPCMPUDZrri(b?)(k?)(z?)",
                                             "VPCMPUQZ128rri(b?)(k?)(z?)",
                                             "VPCMPUQZ256rri(b?)(k?)(z?)",
                                             "VPCMPUQZrri(b?)(k?)(z?)",
                                             "VPCMPUWZ128rri(b?)(k?)(z?)",
                                             "VPCMPUWZ256rri(b?)(k?)(z?)",
                                             "VPCMPUWZrri(b?)(k?)(z?)",
                                             "VPCMPWZ128rri(b?)(k?)(z?)",
                                             "VPCMPWZ256rri(b?)(k?)(z?)",
                                             "VPCMPWZrri(b?)(k?)(z?)",
                                             "VPERM2F128rr",
                                             "VPERM2I128rr",
                                             "VPERMDYrr",
                                             "VPERMDZ256rr(b?)(k?)(z?)",
                                             "VPERMDZrr(b?)(k?)(z?)",
                                             "VPERMI2D128rr(b?)(k?)(z?)",
                                             "VPERMI2D256rr(b?)(k?)(z?)",
                                             "VPERMI2Drr(b?)(k?)(z?)",
                                             "VPERMI2PD128rr(b?)(k?)(z?)",
                                             "VPERMI2PD256rr(b?)(k?)(z?)",
                                             "VPERMI2PDrr(b?)(k?)(z?)",
                                             "VPERMI2PS128rr(b?)(k?)(z?)",
                                             "VPERMI2PS256rr(b?)(k?)(z?)",
                                             "VPERMI2PSrr(b?)(k?)(z?)",
                                             "VPERMI2Q128rr(b?)(k?)(z?)",
                                             "VPERMI2Q256rr(b?)(k?)(z?)",
                                             "VPERMI2Qrr(b?)(k?)(z?)",
                                             "VPERMPDYri",
                                             "VPERMPDZ256r(b?)i(k?)(z?)",
                                             "VPERMPDZ256rr(b?)(k?)(z?)",
                                             "VPERMPDZri(b?)(k?)(z?)",
                                             "VPERMPDZrr(b?)(k?)(z?)",
                                             "VPERMPSYrr",
                                             "VPERMPSZ256rr(b?)(k?)(z?)",
                                             "VPERMPSZrr(b?)(k?)(z?)",
                                             "VPERMQYri",
                                             "VPERMQZ256r(b?)i(k?)(z?)",
                                             "VPERMQZ256rr(b?)(k?)(z?)",
                                             "VPERMQZri(b?)(k?)(z?)",
                                             "VPERMQZrr(b?)(k?)(z?)",
                                             "VPERMT2D128rr(b?)(k?)(z?)",
                                             "VPERMT2D256rr(b?)(k?)(z?)",
                                             "VPERMT2Drr(b?)(k?)(z?)",
                                             "VPERMT2PD128rr(b?)(k?)(z?)",
                                             "VPERMT2PD256rr(b?)(k?)(z?)",
                                             "VPERMT2PDrr(b?)(k?)(z?)",
                                             "VPERMT2PS128rr(b?)(k?)(z?)",
                                             "VPERMT2PS256rr(b?)(k?)(z?)",
                                             "VPERMT2PSrr(b?)(k?)(z?)",
                                             "VPERMT2Q128rr(b?)(k?)(z?)",
                                             "VPERMT2Q256rr(b?)(k?)(z?)",
                                             "VPERMT2Qrr(b?)(k?)(z?)",
                                             "VPMAXSQZ128rr(b?)(k?)(z?)",
                                             "VPMAXSQZ256rr(b?)(k?)(z?)",
                                             "VPMAXSQZrr(b?)(k?)(z?)",
                                             "VPMAXUQZ128rr(b?)(k?)(z?)",
                                             "VPMAXUQZ256rr(b?)(k?)(z?)",
                                             "VPMAXUQZrr(b?)(k?)(z?)",
                                             "VPMINSQZ128rr(b?)(k?)(z?)",
                                             "VPMINSQZ256rr(b?)(k?)(z?)",
                                             "VPMINSQZrr(b?)(k?)(z?)",
                                             "VPMINUQZ128rr(b?)(k?)(z?)",
                                             "VPMINUQZ256rr(b?)(k?)(z?)",
                                             "VPMINUQZrr(b?)(k?)(z?)",
                                             "VPMOVQDZ128rr(b?)(k?)(z?)",
                                             "VPMOVQDZ256rr(b?)(k?)(z?)",
                                             "VPMOVQDZrr(b?)(k?)(z?)",
                                             "VPMOVSXBDYrr",
                                             "VPMOVSXBDZ128rr(b?)(k?)(z?)",
                                             "VPMOVSXBDZ256rr(b?)(k?)(z?)",
                                             "VPMOVSXBDZrr(b?)(k?)(z?)",
                                             "VPMOVSXBQYrr",
                                             "VPMOVSXBQZ128rr(b?)(k?)(z?)",
                                             "VPMOVSXBQZ256rr(b?)(k?)(z?)",
                                             "VPMOVSXBQZrr(b?)(k?)(z?)",
                                             "VPMOVSXBWYrr",
                                             "VPMOVSXBWZ128rr(b?)(k?)(z?)",
                                             "VPMOVSXBWZ256rr(b?)(k?)(z?)",
                                             "VPMOVSXBWZrr(b?)(k?)(z?)",
                                             "VPMOVSXDQYrr",
                                             "VPMOVSXDQZ128rr(b?)(k?)(z?)",
                                             "VPMOVSXDQZ256rr(b?)(k?)(z?)",
                                             "VPMOVSXDQZrr(b?)(k?)(z?)",
                                             "VPMOVSXWDYrr",
                                             "VPMOVSXWDZ128rr(b?)(k?)(z?)",
                                             "VPMOVSXWDZ256rr(b?)(k?)(z?)",
                                             "VPMOVSXWDZrr(b?)(k?)(z?)",
                                             "VPMOVSXWQYrr",
                                             "VPMOVSXWQZ128rr(b?)(k?)(z?)",
                                             "VPMOVSXWQZ256rr(b?)(k?)(z?)",
                                             "VPMOVSXWQZrr(b?)(k?)(z?)",
                                             "VPMOVZXBDYrr",
                                             "VPMOVZXBDZ128rr(b?)(k?)(z?)",
                                             "VPMOVZXBDZ256rr(b?)(k?)(z?)",
                                             "VPMOVZXBDZrr(b?)(k?)(z?)",
                                             "VPMOVZXBQYrr",
                                             "VPMOVZXBQZ128rr(b?)(k?)(z?)",
                                             "VPMOVZXBQZ256rr(b?)(k?)(z?)",
                                             "VPMOVZXBQZrr(b?)(k?)(z?)",
                                             "VPMOVZXBWYrr",
                                             "VPMOVZXBWZ128rr(b?)(k?)(z?)",
                                             "VPMOVZXBWZ256rr(b?)(k?)(z?)",
                                             "VPMOVZXBWZrr(b?)(k?)(z?)",
                                             "VPMOVZXDQYrr",
                                             "VPMOVZXDQZ128rr(b?)(k?)(z?)",
                                             "VPMOVZXDQZ256rr(b?)(k?)(z?)",
                                             "VPMOVZXDQZrr(b?)(k?)(z?)",
                                             "VPMOVZXWDYrr",
                                             "VPMOVZXWDZ128rr(b?)(k?)(z?)",
                                             "VPMOVZXWDZ256rr(b?)(k?)(z?)",
                                             "VPMOVZXWDZrr(b?)(k?)(z?)",
                                             "VPMOVZXWQYrr",
                                             "VPMOVZXWQZ128rr(b?)(k?)(z?)",
                                             "VPMOVZXWQZ256rr(b?)(k?)(z?)",
                                             "VPMOVZXWQZrr(b?)(k?)(z?)",
                                             "VPSADBWYrr",
                                             "VPSADBWZ128rr(b?)(k?)(z?)",
                                             "VPSADBWZ256rr(b?)(k?)(z?)",
                                             "VPSADBWZrr(b?)(k?)(z?)",
                                             "VPSADBWrr",
                                             "VPTESTMBZ128rr(b?)(k?)(z?)",
                                             "VPTESTMBZ256rr(b?)(k?)(z?)",
                                             "VPTESTMBZrr(b?)(k?)(z?)",
                                             "VPTESTMDZ128rr(b?)(k?)(z?)",
                                             "VPTESTMDZ256rr(b?)(k?)(z?)",
                                             "VPTESTMDZrr(b?)(k?)(z?)",
                                             "VPTESTMQZ128rr(b?)(k?)(z?)",
                                             "VPTESTMQZ256rr(b?)(k?)(z?)",
                                             "VPTESTMQZrr(b?)(k?)(z?)",
                                             "VPTESTMWZ128rr(b?)(k?)(z?)",
                                             "VPTESTMWZ256rr(b?)(k?)(z?)",
                                             "VPTESTMWZrr(b?)(k?)(z?)",
                                             "VPTESTNMBZ128rr(b?)(k?)(z?)",
                                             "VPTESTNMBZ256rr(b?)(k?)(z?)",
                                             "VPTESTNMBZrr(b?)(k?)(z?)",
                                             "VPTESTNMDZ128rr(b?)(k?)(z?)",
                                             "VPTESTNMDZ256rr(b?)(k?)(z?)",
                                             "VPTESTNMDZrr(b?)(k?)(z?)",
                                             "VPTESTNMQZ128rr(b?)(k?)(z?)",
                                             "VPTESTNMQZ256rr(b?)(k?)(z?)",
                                             "VPTESTNMQZrr(b?)(k?)(z?)",
                                             "VPTESTNMWZ128rr(b?)(k?)(z?)",
                                             "VPTESTNMWZ256rr(b?)(k?)(z?)",
                                             "VPTESTNMWZrr(b?)(k?)(z?)",
                                             "VSHUFF32X4Z256rri(b?)(k?)(z?)",
                                             "VSHUFF32X4Zrri(b?)(k?)(z?)",
                                             "VSHUFF64X2Z256rri(b?)(k?)(z?)",
                                             "VSHUFF64X2Zrri(b?)(k?)(z?)",
                                             "VSHUFI32X4Z256rri(b?)(k?)(z?)",
                                             "VSHUFI32X4Zrri(b?)(k?)(z?)",
                                             "VSHUFI64X2Z256rri(b?)(k?)(z?)",
                                             "VSHUFI64X2Zrri(b?)(k?)(z?)")>;

def SKXWriteResGroup33 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup33], (instregex "EXTRACTPSrr",
                                             "MMX_PEXTRWrr",
                                             "PEXTRBrr",
                                             "PEXTRDrr",
                                             "PEXTRQrr",
                                             "PEXTRWrr",
                                             "PTESTrr",
                                             "VEXTRACTPSZrr(b?)(k?)(z?)",
                                             "VEXTRACTPSrr",
                                             "VPEXTRBZrr(b?)(k?)(z?)",
                                             "VPEXTRBrr",
                                             "VPEXTRDZrr(b?)(k?)(z?)",
                                             "VPEXTRDrr",
                                             "VPEXTRQZrr(b?)(k?)(z?)",
                                             "VPEXTRQrr",
                                             "VPEXTRWZrr",
                                             "VPEXTRWrr",
                                             "VPTESTYrr",
                                             "VPTESTrr")>;

def SKXWriteResGroup34 : SchedWriteRes<[SKXPort0,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup34], (instregex "FNSTSW16r")>;

def SKXWriteResGroup35 : SchedWriteRes<[SKXPort06]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKXWriteResGroup35], (instregex "ROL(8|16|32|64)rCL",
                                             "ROR(8|16|32|64)rCL",
                                             "SAR(8|16|32|64)rCL",
                                             "SHL(8|16|32|64)rCL",
                                             "SHR(8|16|32|64)rCL")>;

def SKXWriteResGroup36 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKXWriteResGroup36], (instregex "XADD(8|16|32|64)rr",
                                             "XCHG8rr")>;

def SKXWriteResGroup37 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup37], (instregex "MMX_PHADDSWrr",
                                             "MMX_PHSUBSWrr")>;

def SKXWriteResGroup38 : SchedWriteRes<[SKXPort5,SKXPort01]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup38], (instregex "PHADDSWrr",
                                             "PHSUBSWrr",
                                             "VPHADDSWrr",
                                             "VPHADDSWYrr",
                                             "VPHSUBSWrr",
                                             "VPHSUBSWYrr")>;

def SKXWriteResGroup39 : SchedWriteRes<[SKXPort5,SKXPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup39], (instregex "MMX_PHADDDrr",
                                             "MMX_PHADDWrr",
                                             "MMX_PHSUBDrr",
                                             "MMX_PHSUBWrr")>;

def SKXWriteResGroup40 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup40], (instregex "PHADDDrr",
                                             "PHADDWrr",
                                             "PHSUBDrr",
                                             "PHSUBWrr",
                                             "VPHADDDYrr",
                                             "VPHADDDrr",
                                             "VPHADDWYrr",
                                             "VPHADDWrr",
                                             "VPHSUBDYrr",
                                             "VPHSUBDrr",
                                             "VPHSUBWYrr",
                                             "VPHSUBWrr")>;

def SKXWriteResGroup41 : SchedWriteRes<[SKXPort5,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup41], (instregex "MMX_PACKSSDWirr",
                                             "MMX_PACKSSWBirr",
                                             "MMX_PACKUSWBirr")>;

def SKXWriteResGroup42 : SchedWriteRes<[SKXPort6,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup42], (instregex "CLD")>;

def SKXWriteResGroup43 : SchedWriteRes<[SKXPort237,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup43], (instregex "MFENCE")>;

def SKXWriteResGroup44 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup44], (instregex "RCL(8|16|32|64)r1",
                                             "RCL(8|16|32|64)ri",
                                             "RCR(8|16|32|64)r1",
                                             "RCR(8|16|32|64)ri")>;

def SKXWriteResGroup45 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup45], (instregex "FNSTSWm")>;

def SKXWriteResGroup46 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort06]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKXWriteResGroup46], (instregex "SET(A|BE)m")>;

def SKXWriteResGroup47 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort237,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup47], (instregex "CALL(16|32|64)r")>;

def SKXWriteResGroup48 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup48], (instregex "CALL64pcrel32")>;

def SKXWriteResGroup49 : SchedWriteRes<[SKXPort0]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup49], (instregex "MMX_PMADDUBSWrr",
                                             "MMX_PMADDWDirr",
                                             "MMX_PMULHRSWrr",
                                             "MMX_PMULHUWirr",
                                             "MMX_PMULHWirr",
                                             "MMX_PMULLWirr",
                                             "MMX_PMULUDQirr",
                                             "MUL_FPrST0",
                                             "MUL_FST0r",
                                             "MUL_FrST0",
                                             "RCPPSr",
                                             "RCPSSr",
                                             "RSQRTPSr",
                                             "RSQRTSSr",
                                             "VRCP14PDZ128r(b?)(k?)(z?)",
                                             "VRCP14PDZ256r(b?)(k?)(z?)",
                                             "VRCP14PSZ128r(b?)(k?)(z?)",
                                             "VRCP14PSZ256r(b?)(k?)(z?)",
                                             "VRCP14SDrr(b?)(k?)(z?)",
                                             "VRCP14SSrr(b?)(k?)(z?)",
                                             "VRCPPSYr",
                                             "VRCPPSr",
                                             "VRCPSSr",
                                             "VRSQRT14PDZ128r(b?)(k?)(z?)",
                                             "VRSQRT14PDZ256r(b?)(k?)(z?)",
                                             "VRSQRT14PSZ128r(b?)(k?)(z?)",
                                             "VRSQRT14PSZ256r(b?)(k?)(z?)",
                                             "VRSQRT14SDrr(b?)(k?)(z?)",
                                             "VRSQRT14SSrr(b?)(k?)(z?)",
                                             "VRSQRTPSYr",
                                             "VRSQRTPSr",
                                             "VRSQRTSSr")>;

def SKXWriteResGroup50 : SchedWriteRes<[SKXPort015]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup50], (instregex "ADDPDrr",
                                             "ADDPSrr",
                                             "ADDSDrr",
                                             "ADDSSrr",
                                             "ADDSUBPDrr",
                                             "ADDSUBPSrr",
                                             "CMPPDrri",
                                             "CMPPSrri",
                                             "CMPSDrr",
                                             "CMPSSrr",
                                             "CVTDQ2PSrr",
                                             "CVTPS2DQrr",
                                             "CVTTPS2DQrr",
                                             "MAX(C?)PDrr",
                                             "MAX(C?)PSrr",
                                             "MAX(C?)SDrr",
                                             "MAX(C?)SSrr",
                                             "MIN(C?)PDrr",
                                             "MIN(C?)PSrr",
                                             "MIN(C?)SDrr",
                                             "MIN(C?)SSrr",
                                             "MULPDrr",
                                             "MULPSrr",
                                             "MULSDrr",
                                             "MULSSrr",
                                             "PHMINPOSUWrr",
                                             "PMADDUBSWrr",
                                             "PMADDWDrr",
                                             "PMULDQrr",
                                             "PMULHRSWrr",
                                             "PMULHUWrr",
                                             "PMULHWrr",
                                             "PMULLWrr",
                                             "PMULUDQrr",
                                             "SUBPDrr",
                                             "SUBPSrr",
                                             "SUBSDrr",
                                             "SUBSSrr",
                                             "VADDPDYrr",
                                             "VADDPDZ128rr(b?)(k?)(z?)",
                                             "VADDPDZ256rr(b?)(k?)(z?)",
                                             "VADDPDZrr(b?)(k?)(z?)",
                                             "VADDPDrr",
                                             "VADDPSYrr",
                                             "VADDPSZ128rr(b?)(k?)(z?)",
                                             "VADDPSZ256rr(b?)(k?)(z?)",
                                             "VADDPSZrr(b?)(k?)(z?)",
                                             "VADDPSrr",
                                             "VADDSDZrr(b?)(_Int)?(k?)(z?)",
                                             "VADDSDrr",
                                             "VADDSSZrr(b?)(_Int)?(k?)(z?)",
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
                                             "VCVTDQ2PSYrr",
                                             "VCVTDQ2PSZ128rr(b?)(k?)(z?)",
                                             "VCVTDQ2PSZ256rr(b?)(k?)(z?)",
                                             "VCVTDQ2PSZrr(b?)(k?)(z?)",
                                             "VCVTDQ2PSrr",
                                             "VCVTPD2QQZ128rr(b?)(k?)(z?)",
                                             "VCVTPD2QQZ256rr(b?)(k?)(z?)",
                                             "VCVTPD2QQZrr(b?)(k?)(z?)",
                                             "VCVTPD2UQQZ128rr(b?)(k?)(z?)",
                                             "VCVTPD2UQQZ256rr(b?)(k?)(z?)",
                                             "VCVTPD2UQQZrr(b?)(k?)(z?)",
                                             "VCVTPS2DQYrr",
                                             "VCVTPS2DQZ128rr(b?)(k?)(z?)",
                                             "VCVTPS2DQZ256rr(b?)(k?)(z?)",
                                             "VCVTPS2DQZrr(b?)(k?)(z?)",
                                             "VCVTPS2DQrr",
                                             "VCVTPS2UDQZ128rr(b?)(k?)(z?)",
                                             "VCVTPS2UDQZ256rr(b?)(k?)(z?)",
                                             "VCVTPS2UDQZrr(b?)(k?)(z?)",
                                             "VCVTQQ2PDZ128rr(b?)(k?)(z?)",
                                             "VCVTQQ2PDZ256rr(b?)(k?)(z?)",
                                             "VCVTQQ2PDZrr(b?)(k?)(z?)",
                                             "VCVTTPD2QQZ128rr(b?)(k?)(z?)",
                                             "VCVTTPD2QQZ256rr(b?)(k?)(z?)",
                                             "VCVTTPD2QQZrr(b?)(k?)(z?)",
                                             "VCVTTPD2UQQZ128rr(b?)(k?)(z?)",
                                             "VCVTTPD2UQQZ256rr(b?)(k?)(z?)",
                                             "VCVTTPD2UQQZrr(b?)(k?)(z?)",
                                             "VCVTTPS2DQYrr",
                                             "VCVTTPS2DQZ128rr(b?)(k?)(z?)",
                                             "VCVTTPS2DQZ256rr(b?)(k?)(z?)",
                                             "VCVTTPS2DQZrr(b?)(k?)(z?)",
                                             "VCVTTPS2DQrr",
                                             "VCVTTPS2UDQZ128rr(b?)(k?)(z?)",
                                             "VCVTTPS2UDQZ256rr(b?)(k?)(z?)",
                                             "VCVTTPS2UDQZrr(b?)(k?)(z?)",
                                             "VCVTUDQ2PSZ128rr(b?)(k?)(z?)",
                                             "VCVTUDQ2PSZ256rr(b?)(k?)(z?)",
                                             "VCVTUDQ2PSZrr(b?)(k?)(z?)",
                                             "VCVTUQQ2PDZ128rr(b?)(k?)(z?)",
                                             "VCVTUQQ2PDZ256rr(b?)(k?)(z?)",
                                             "VCVTUQQ2PDZrr(b?)(k?)(z?)",
                                             "VFIXUPIMMPDZ128rri(b?)(k?)(z?)",
                                             "VFIXUPIMMPDZ256rri(b?)(k?)(z?)",
                                             "VFIXUPIMMPDZrri(b?)(k?)(z?)",
                                             "VFIXUPIMMPSZ128rri(b?)(k?)(z?)",
                                             "VFIXUPIMMPSZ256rri(b?)(k?)(z?)",
                                             "VFIXUPIMMPSZrri(b?)(k?)(z?)",
                                             "VFIXUPIMMSDrri(b?)(k?)(z?)",
                                             "VFIXUPIMMSSrri(b?)(k?)(z?)",
                                             "VGETEXPPDZ128r(b?)(k?)(z?)",
                                             "VGETEXPPDZ256r(b?)(k?)(z?)",
                                             "VGETEXPPDr(b?)(k?)(z?)",
                                             "VGETEXPPSZ128r(b?)(k?)(z?)",
                                             "VGETEXPPSZ256r(b?)(k?)(z?)",
                                             "VGETEXPPSr(b?)(k?)(z?)",
                                             "VGETEXPSDr(b?)(k?)(z?)",
                                             "VGETEXPSSr(b?)(k?)(z?)",
                                             "VGETMANTPDZ128rri(b?)(k?)(z?)",
                                             "VGETMANTPDZ256rri(b?)(k?)(z?)",
                                             "VGETMANTPDZrri(b?)(k?)(z?)",
                                             "VGETMANTPSZ128rri(b?)(k?)(z?)",
                                             "VGETMANTPSZ256rri(b?)(k?)(z?)",
                                             "VGETMANTPSZrri(b?)(k?)(z?)",
                                             "VGETMANTSDZ128rri(b?)(k?)(z?)",
                                             "VGETMANTSSZ128rri(b?)(k?)(z?)",
                                             "VMAX(C?)PDYrr",
                                             "VMAX(C?)PDZ128rr(b?)(k?)(z?)",
                                             "VMAX(C?)PDZ256rr(b?)(k?)(z?)",
                                             "VMAX(C?)PDZrr(b?)(k?)(z?)",
                                             "VMAX(C?)PDrr",
                                             "VMAX(C?)PSYrr",
                                             "VMAX(C?)PSZ128rr(b?)(k?)(z?)",
                                             "VMAX(C?)PSZ256rr(b?)(k?)(z?)",
                                             "VMAX(C?)PSZrr(b?)(k?)(z?)",
                                             "VMAX(C?)PSrr",
                                             "VMAX(C?)SDZrr(b?)(_Int)?(k?)(z?)",
                                             "VMAX(C?)SDrr",
                                             "VMAX(C?)SSZrr(b?)(_Int)?(k?)(z?)",
                                             "VMAX(C?)SSrr",
                                             "VMIN(C?)PDYrr",
                                             "VMIN(C?)PDZ128rr(b?)(k?)(z?)",
                                             "VMIN(C?)PDZ256rr(b?)(k?)(z?)",
                                             "VMIN(C?)PDZrr(b?)(k?)(z?)",
                                             "VMIN(C?)PDrr",
                                             "VMIN(C?)PSYrr",
                                             "VMIN(C?)PSZ128rr(b?)(k?)(z?)",
                                             "VMIN(C?)PSZ256rr(b?)(k?)(z?)",
                                             "VMIN(C?)PSZrr(b?)(k?)(z?)",
                                             "VMIN(C?)PSrr",
                                             "VMIN(C?)SDZrr(b?)(_Int)?(k?)(z?)",
                                             "VMIN(C?)SDrr",
                                             "VMIN(C?)SSZrr(b?)(_Int)?(k?)(z?)",
                                             "VMIN(C?)SSrr",
                                             "VMULPDYrr",
                                             "VMULPDZ128rr(b?)(k?)(z?)",
                                             "VMULPDZ256rr(b?)(k?)(z?)",
                                             "VMULPDZrr(b?)(k?)(z?)",
                                             "VMULPDrr",
                                             "VMULPSYrr",
                                             "VMULPSZ128rr(b?)(k?)(z?)",
                                             "VMULPSZ256rr(b?)(k?)(z?)",
                                             "VMULPSZrr(b?)(k?)(z?)",
                                             "VMULPSrr",
                                             "VMULSDZrr(b?)(_Int)?(k?)(z?)",
                                             "VMULSDrr",
                                             "VMULSSZrr(b?)(_Int)?(k?)(z?)",
                                             "VMULSSrr",
                                             "VPHMINPOSUWrr",
                                             "VPLZCNTDZ128rr(b?)(k?)(z?)",
                                             "VPLZCNTDZ256rr(b?)(k?)(z?)",
                                             "VPLZCNTDZrr(b?)(k?)(z?)",
                                             "VPLZCNTQZ128rr(b?)(k?)(z?)",
                                             "VPLZCNTQZ256rr(b?)(k?)(z?)",
                                             "VPLZCNTQZrr(b?)(k?)(z?)",
                                             "VPMADDUBSWYrr",
                                             "VPMADDUBSWZ128rr(b?)(k?)(z?)",
                                             "VPMADDUBSWZ256rr(b?)(k?)(z?)",
                                             "VPMADDUBSWZrr(b?)(k?)(z?)",
                                             "VPMADDUBSWrr",
                                             "VPMADDWDYrr",
                                             "VPMADDWDZ128rr(b?)(k?)(z?)",
                                             "VPMADDWDZ256rr(b?)(k?)(z?)",
                                             "VPMADDWDZrr(b?)(k?)(z?)",
                                             "VPMADDWDrr",
                                             "VPMULDQYrr",
                                             "VPMULDQZ128rr(b?)(k?)(z?)",
                                             "VPMULDQZ256rr(b?)(k?)(z?)",
                                             "VPMULDQZrr(b?)(k?)(z?)",
                                             "VPMULDQrr",
                                             "VPMULHRSWYrr",
                                             "VPMULHRSWZ128rr(b?)(k?)(z?)",
                                             "VPMULHRSWZ256rr(b?)(k?)(z?)",
                                             "VPMULHRSWZrr(b?)(k?)(z?)",
                                             "VPMULHRSWrr",
                                             "VPMULHUWYrr",
                                             "VPMULHUWZ128rr(b?)(k?)(z?)",
                                             "VPMULHUWZ256rr(b?)(k?)(z?)",
                                             "VPMULHUWZrr(b?)(k?)(z?)",
                                             "VPMULHUWrr",
                                             "VPMULHWYrr",
                                             "VPMULHWZ128rr(b?)(k?)(z?)",
                                             "VPMULHWZ256rr(b?)(k?)(z?)",
                                             "VPMULHWZrr(b?)(k?)(z?)",
                                             "VPMULHWrr",
                                             "VPMULLWYrr",
                                             "VPMULLWZ128rr(b?)(k?)(z?)",
                                             "VPMULLWZ256rr(b?)(k?)(z?)",
                                             "VPMULLWZrr(b?)(k?)(z?)",
                                             "VPMULLWrr",
                                             "VPMULUDQYrr",
                                             "VPMULUDQZ128rr(b?)(k?)(z?)",
                                             "VPMULUDQZ256rr(b?)(k?)(z?)",
                                             "VPMULUDQZrr(b?)(k?)(z?)",
                                             "VPMULUDQrr",
                                             "VRANGEPDZ128rri(b?)(k?)(z?)",
                                             "VRANGEPDZ256rri(b?)(k?)(z?)",
                                             "VRANGEPDZrri(b?)(k?)(z?)",
                                             "VRANGEPSZ128rri(b?)(k?)(z?)",
                                             "VRANGEPSZ256rri(b?)(k?)(z?)",
                                             "VRANGEPSZrri(b?)(k?)(z?)",
                                             "VRANGESDZ128rri(b?)(k?)(z?)",
                                             "VRANGESSZ128rri(b?)(k?)(z?)",
                                             "VREDUCEPDZ128rri(b?)(k?)(z?)",
                                             "VREDUCEPDZ256rri(b?)(k?)(z?)",
                                             "VREDUCEPDZrri(b?)(k?)(z?)",
                                             "VREDUCEPSZ128rri(b?)(k?)(z?)",
                                             "VREDUCEPSZ256rri(b?)(k?)(z?)",
                                             "VREDUCEPSZrri(b?)(k?)(z?)",
                                             "VREDUCESDZ128rri(b?)(k?)(z?)",
                                             "VREDUCESSZ128rri(b?)(k?)(z?)",
                                             "VSCALEFPDZ128rr(b?)(k?)(z?)",
                                             "VSCALEFPDZ256rr(b?)(k?)(z?)",
                                             "VSCALEFPDZrr(b?)(k?)(z?)",
                                             "VSCALEFPSZ128rr(b?)(k?)(z?)",
                                             "VSCALEFPSZ256rr(b?)(k?)(z?)",
                                             "VSCALEFPSZrr(b?)(k?)(z?)",
                                             "VSCALEFSDZ128rr(b?)(k?)(z?)",
                                             "VSCALEFSSZ128rr(b?)(k?)(z?)",
                                             "VSUBPDYrr",
                                             "VSUBPDZ128rr(b?)(k?)(z?)",
                                             "VSUBPDZ256rr(b?)(k?)(z?)",
                                             "VSUBPDZrr(b?)(k?)(z?)",
                                             "VSUBPDrr",
                                             "VSUBPSYrr",
                                             "VSUBPSZ128rr(b?)(k?)(z?)",
                                             "VSUBPSZ256rr(b?)(k?)(z?)",
                                             "VSUBPSZrr(b?)(k?)(z?)",
                                             "VSUBPSrr",
                                             "VSUBSDZrr(b?)(_Int)?(k?)(z?)",
                                             "VSUBSDrr",
                                             "VSUBSSZrr(b?)(_Int)?(k?)(z?)",
                                             "VSUBSSrr")>;
def: InstRW<[SKXWriteResGroup50],
          (instregex
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Yr",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Z128r(b?)(k?)(z?)",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Z256r(b?)(k?)(z?)",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Zr(b?)(k?)(z?)",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)r",
           "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)Zr(b?)(_Int)?(k?)(z?)",
           "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)r")>;

def SKXWriteResGroup51 : SchedWriteRes<[SKXPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup51], (instregex "MPSADBWrri",
                                             "VEXPANDPDZ128rr(b?)(k?)(z?)",
                                             "VEXPANDPDZ256rr(b?)(k?)(z?)",
                                             "VEXPANDPDZrr(b?)(k?)(z?)",
                                             "VEXPANDPSZ128rr(b?)(k?)(z?)",
                                             "VEXPANDPSZ256rr(b?)(k?)(z?)",
                                             "VEXPANDPSZrr(b?)(k?)(z?)",
                                             "VMPSADBWYrri",
                                             "VMPSADBWrri",
                                             "VPEXPANDDZ128rr(b?)(k?)(z?)",
                                             "VPEXPANDDZ256rr(b?)(k?)(z?)",
                                             "VPEXPANDDZrr(b?)(k?)(z?)",
                                             "VPEXPANDQZ128rr(b?)(k?)(z?)",
                                             "VPEXPANDQZ256rr(b?)(k?)(z?)",
                                             "VPEXPANDQZrr(b?)(k?)(z?)",
                                             "VPMOVDBZ128rr(b?)(k?)(z?)",
                                             "VPMOVDBZ256rr(b?)(k?)(z?)",
                                             "VPMOVDBZrr(b?)(k?)(z?)",
                                             "VPMOVDWZ128rr(b?)(k?)(z?)",
                                             "VPMOVDWZ256rr(b?)(k?)(z?)",
                                             "VPMOVDWZrr(b?)(k?)(z?)",
                                             "VPMOVQBZ128rr(b?)(k?)(z?)",
                                             "VPMOVQBZ256rr(b?)(k?)(z?)",
                                             "VPMOVQBZrr(b?)(k?)(z?)",
                                             "VPMOVQWZ128rr(b?)(k?)(z?)",
                                             "VPMOVQWZ256rr(b?)(k?)(z?)",
                                             "VPMOVQWZrr(b?)(k?)(z?)",
                                             "VPMOVSDBZ128rr(b?)(k?)(z?)",
                                             "VPMOVSDBZ256rr(b?)(k?)(z?)",
                                             "VPMOVSDBZrr(b?)(k?)(z?)",
                                             "VPMOVSDWZ128rr(b?)(k?)(z?)",
                                             "VPMOVSDWZ256rr(b?)(k?)(z?)",
                                             "VPMOVSDWZrr(b?)(k?)(z?)",
                                             "VPMOVSQBZ128rr(b?)(k?)(z?)",
                                             "VPMOVSQBZ256rr(b?)(k?)(z?)",
                                             "VPMOVSQBZrr(b?)(k?)(z?)",
                                             "VPMOVSQDZ128rr(b?)(k?)(z?)",
                                             "VPMOVSQDZ256rr(b?)(k?)(z?)",
                                             "VPMOVSQDZrr(b?)(k?)(z?)",
                                             "VPMOVSQWZ128rr(b?)(k?)(z?)",
                                             "VPMOVSQWZ256rr(b?)(k?)(z?)",
                                             "VPMOVSQWZrr(b?)(k?)(z?)",
                                             "VPMOVSWBZ128rr(b?)(k?)(z?)",
                                             "VPMOVSWBZ256rr(b?)(k?)(z?)",
                                             "VPMOVSWBZrr(b?)(k?)(z?)",
                                             "VPMOVUSDBZ128rr(b?)(k?)(z?)",
                                             "VPMOVUSDBZ256rr(b?)(k?)(z?)",
                                             "VPMOVUSDBZrr(b?)(k?)(z?)",
                                             "VPMOVUSDWZ128rr(b?)(k?)(z?)",
                                             "VPMOVUSDWZ256rr(b?)(k?)(z?)",
                                             "VPMOVUSDWZrr(b?)(k?)(z?)",
                                             "VPMOVUSQBZ128rr(b?)(k?)(z?)",
                                             "VPMOVUSQBZ256rr(b?)(k?)(z?)",
                                             "VPMOVUSQBZrr(b?)(k?)(z?)",
                                             "VPMOVUSQDZ128rr(b?)(k?)(z?)",
                                             "VPMOVUSQDZ256rr(b?)(k?)(z?)",
                                             "VPMOVUSQDZrr(b?)(k?)(z?)",
                                             "VPMOVUSQWZ128rr(b?)(k?)(z?)",
                                             "VPMOVUSQWZ256rr(b?)(k?)(z?)",
                                             "VPMOVUSQWZrr(b?)(k?)(z?)",
                                             "VPMOVUSWBZ128rr(b?)(k?)(z?)",
                                             "VPMOVUSWBZ256rr(b?)(k?)(z?)",
                                             "VPMOVUSWBZrr(b?)(k?)(z?)",
                                             "VPMOVWBZ128rr(b?)(k?)(z?)",
                                             "VPMOVWBZ256rr(b?)(k?)(z?)",
                                             "VPMOVWBZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup52 : SchedWriteRes<[SKXPort1,SKXPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup52], (instrs IMUL64r, MUL64r, MULX64rr)>;

def SKXWriteResGroup52_16 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def: InstRW<[SKXWriteResGroup52_16], (instrs IMUL16r, MUL16r)>;

def SKXWriteResGroup53 : SchedWriteRes<[SKXPort5,SKXPort01]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup53], (instregex "VPSLLDYrr",
                                             "VPSLLDZ256rr(b?)(k?)(z?)",
                                             "VPSLLDZrr(b?)(k?)(z?)",
                                             "VPSLLQYrr",
                                             "VPSLLQZ256rr(b?)(k?)(z?)",
                                             "VPSLLQZrr(b?)(k?)(z?)",
                                             "VPSLLWYrr",
                                             "VPSLLWZ256rr(b?)(k?)(z?)",
                                             "VPSLLWZrr(b?)(k?)(z?)",
                                             "VPSRADYrr",
                                             "VPSRADZ256rr(b?)(k?)(z?)",
                                             "VPSRADZrr(b?)(k?)(z?)",
                                             "VPSRAQZ256rr(b?)(k?)(z?)",
                                             "VPSRAQZrr(b?)(k?)(z?)",
                                             "VPSRAWYrr",
                                             "VPSRAWZ256rr(b?)(k?)(z?)",
                                             "VPSRAWZrr(b?)(k?)(z?)",
                                             "VPSRLDYrr",
                                             "VPSRLDZ256rr(b?)(k?)(z?)",
                                             "VPSRLDZrr(b?)(k?)(z?)",
                                             "VPSRLQYrr",
                                             "VPSRLQZ256rr(b?)(k?)(z?)",
                                             "VPSRLQZrr(b?)(k?)(z?)",
                                             "VPSRLWYrr",
                                             "VPSRLWZ256rr(b?)(k?)(z?)",
                                             "VPSRLWZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup54 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup54], (instregex "ISTT_FP16m",
                                             "ISTT_FP32m",
                                             "ISTT_FP64m",
                                             "IST_F16m",
                                             "IST_F32m",
                                             "IST_FP16m",
                                             "IST_FP32m",
                                             "IST_FP64m",
                                             "VPMOVQDZ128mr(b?)(k?)(z?)",
                                             "VPMOVQDZ256mr(b?)(k?)(z?)",
                                             "VPMOVQDZmr(b?)(k?)(z?)")>;

def SKXWriteResGroup55 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[SKXWriteResGroup55], (instregex "FNCLEX")>;

def SKXWriteResGroup56 : SchedWriteRes<[SKXPort015,SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup56], (instregex "VZEROUPPER")>;

def SKXWriteResGroup57 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKXWriteResGroup57], (instregex "LAR(16|32|64)rr")>;

def SKXWriteResGroup58 : SchedWriteRes<[SKXPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup58], (instregex "MMX_MOVD64rm",
                                             "MMX_MOVD64to64rm",
                                             "MMX_MOVQ64rm",
                                             "MOV(8|16|32|64)rm",
                                             "MOV64toPQIrm",
                                             "MOVDDUPrm",
                                             "MOVDI2PDIrm",
                                             "MOVQI2PQIrm",
                                             "MOVSDrm",
                                             "MOVSSrm",
                                             "MOVSX(16|32|64)rm16",
                                             "MOVSX(16|32|64)rm32",
                                             "MOVSX(16|32|64)rm8",
                                             "MOVZX(16|32|64)rm16",
                                             "MOVZX(16|32|64)rm8",
                                             "PREFETCHNTA",
                                             "PREFETCHT0",
                                             "PREFETCHT1",
                                             "PREFETCHT2",
                                             "VMOV64toPQIrm",
                                             "VMOVDDUPrm",
                                             "VMOVDI2PDIrm",
                                             "VMOVQI2PQIrm",
                                             "VMOVSDrm",
                                             "VMOVSSrm")>;

def SKXWriteResGroup59 : SchedWriteRes<[SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup59], (instregex "VCVTSD2SSZrr(b?)(_Int)?(k?)(z?)")>;

def SKXWriteResGroup60 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup60], (instregex "(V?)CVTDQ2PDrr",
                                             "MMX_CVTPI2PDirr")>;

def SKXWriteResGroup61 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup61], (instregex "CVTPD2DQrr",
                                             "CVTPD2PSrr",
                                             "CVTPS2PDrr",
                                             "CVTSD2SSrr",
                                             "CVTSI642SDrr",
                                             "CVTSI2SDrr",
                                             "CVTSI2SSrr",
                                             "CVTSS2SDrr",
                                             "CVTTPD2DQrr",
                                             "MMX_CVTPD2PIirr",
                                             "MMX_CVTPS2PIirr",
                                             "MMX_CVTTPD2PIirr",
                                             "MMX_CVTTPS2PIirr",
                                             "VCVTDQ2PDZ128rr(b?)(k?)(z?)",
                                             "VCVTPD2DQZ128rr(b?)(k?)(z?)",
                                             "VCVTPD2DQrr",
                                             "VCVTPD2PSZ128rr(b?)(k?)(z?)",
                                             "VCVTPD2PSrr",
                                             "VCVTPD2UDQZ128rr(b?)(k?)(z?)",
                                             "VCVTPH2PSZ128rr(b?)(k?)(z?)",
                                             "VCVTPH2PSrr",
                                             "VCVTPS2PDZ128rr(b?)(k?)(z?)",
                                             "VCVTPS2PDrr",
                                             "VCVTPS2PHZ128rr(b?)(k?)(z?)",
                                             "VCVTPS2PHrr",
                                             "VCVTPS2QQZ128rr(b?)(k?)(z?)",
                                             "VCVTPS2UQQZ128rr(b?)(k?)(z?)",
                                             "VCVTQQ2PSZ128rr(b?)(k?)(z?)",
                                             "VCVTSD2SSrr",
                                             "VCVTSI642SDrr",
                                             "VCVTSI2SDZrr(b?)(k?)(z?)",
                                             "VCVTSI2SDrr",
                                             "VCVTSI2SSZrr(b?)(k?)(z?)",
                                             "VCVTSI2SSrr",
                                             "VCVTSI642SDZrr(b?)(k?)(z?)",
                                             "VCVTSS2SDZrr(b?)(_Int)?(k?)(z?)",
                                             "VCVTSS2SDrr",
                                             "VCVTTPD2DQZ128rr(b?)(k?)(z?)",
                                             "VCVTTPD2DQrr",
                                             "VCVTTPD2UDQZ128rr(b?)(k?)(z?)",
                                             "VCVTTPS2QQZ128rr(b?)(k?)(z?)",
                                             "VCVTTPS2UQQZ128rr(b?)(k?)(z?)",
                                             "VCVTUDQ2PDZ128rr(b?)(k?)(z?)",
                                             "VCVTUQQ2PSZ128rr(b?)(k?)(z?)",
                                             "VCVTUSI2SDZrr(b?)(k?)(z?)",
                                             "VCVTUSI2SSZrr(b?)(k?)(z?)",
                                             "VCVTUSI642SDZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup62 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup62], (instregex "VPCONFLICTQZ128rr(b?)(k?)(z?)")>;

def SKXWriteResGroup63 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup63], (instregex "STR(16|32|64)r")>;

def SKXWriteResGroup64 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup64], (instrs IMUL32r, MUL32r, MULX32rr)>;

def SKXWriteResGroup65 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup65], (instregex "VCVTPS2PHZ128mr(b?)(k?)(z?)",
                                             "VCVTPS2PHZ256mr(b?)(k?)(z?)",
                                             "VCVTPS2PHZmr(b?)(k?)(z?)")>;

def SKXWriteResGroup66 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVDBZ128mr(b?)(k?)(z?)",
                                             "VPMOVDBZ256mr(b?)(k?)(z?)",
                                             "VPMOVDBZmr(b?)(k?)(z?)",
                                             "VPMOVDWZ128mr(b?)(k?)(z?)",
                                             "VPMOVDWZ256mr(b?)(k?)(z?)",
                                             "VPMOVDWZmr(b?)(k?)(z?)",
                                             "VPMOVQBZ128mr(b?)(k?)(z?)",
                                             "VPMOVQBZ256mr(b?)(k?)(z?)",
                                             "VPMOVQBZmr(b?)(k?)(z?)",
                                             "VPMOVQWZ128mr(b?)(k?)(z?)",
                                             "VPMOVQWZ256mr(b?)(k?)(z?)",
                                             "VPMOVQWZmr(b?)(k?)(z?)",
                                             "VPMOVSDBZ128mr(b?)(k?)(z?)",
                                             "VPMOVSDBZ256mr(b?)(k?)(z?)",
                                             "VPMOVSDBZmr(b?)(k?)(z?)",
                                             "VPMOVSDWZ128mr(b?)(k?)(z?)",
                                             "VPMOVSDWZ256mr(b?)(k?)(z?)",
                                             "VPMOVSDWZmr(b?)(k?)(z?)",
                                             "VPMOVSQBZ128mr(b?)(k?)(z?)",
                                             "VPMOVSQBZ256mr(b?)(k?)(z?)",
                                             "VPMOVSQBZmr(b?)(k?)(z?)",
                                             "VPMOVSQDZ128mr(b?)(k?)(z?)",
                                             "VPMOVSQDZ256mr(b?)(k?)(z?)",
                                             "VPMOVSQDZmr(b?)(k?)(z?)",
                                             "VPMOVSQWZ128mr(b?)(k?)(z?)",
                                             "VPMOVSQWZ256mr(b?)(k?)(z?)",
                                             "VPMOVSQWZmr(b?)(k?)(z?)",
                                             "VPMOVSWBZ128mr(b?)(k?)(z?)",
                                             "VPMOVSWBZ256mr(b?)(k?)(z?)",
                                             "VPMOVSWBZmr(b?)(k?)(z?)",
                                             "VPMOVUSDBZ128mr(b?)(k?)(z?)",
                                             "VPMOVUSDBZ256mr(b?)(k?)(z?)",
                                             "VPMOVUSDBZmr(b?)(k?)(z?)",
                                             "VPMOVUSDWZ128mr(b?)(k?)(z?)",
                                             "VPMOVUSDWZ256mr(b?)(k?)(z?)",
                                             "VPMOVUSDWZmr(b?)(k?)(z?)",
                                             "VPMOVUSQBZ128mr(b?)(k?)(z?)",
                                             "VPMOVUSQBZ256mr(b?)(k?)(z?)",
                                             "VPMOVUSQBZmr(b?)(k?)(z?)",
                                             "VPMOVUSQDZ128mr(b?)(k?)(z?)",
                                             "VPMOVUSQDZ256mr(b?)(k?)(z?)",
                                             "VPMOVUSQDZmr(b?)(k?)(z?)",
                                             "VPMOVUSQWZ128mr(b?)(k?)(z?)",
                                             "VPMOVUSQWZ256mr(b?)(k?)(z?)",
                                             "VPMOVUSQWZmr(b?)(k?)(z?)",
                                             "VPMOVUSWBZ128mr(b?)(k?)(z?)",
                                             "VPMOVUSWBZ256mr(b?)(k?)(z?)",
                                             "VPMOVUSWBZmr(b?)(k?)(z?)",
                                             "VPMOVWBZ128mr(b?)(k?)(z?)",
                                             "VPMOVWBZ256mr(b?)(k?)(z?)",
                                             "VPMOVWBZmr(b?)(k?)(z?)")>;

def SKXWriteResGroup67 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[SKXWriteResGroup67], (instregex "XSETBV")>;

def SKXWriteResGroup68 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[SKXWriteResGroup68], (instregex "CMPXCHG(8|16|32|64)rr")>;

def SKXWriteResGroup69 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKXWriteResGroup69], (instregex "PUSHF16",
                                             "PUSHF64")>;

def SKXWriteResGroup71 : SchedWriteRes<[SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup71], (instregex "LDDQUrm",
                                             "MOVAPDrm",
                                             "MOVAPSrm",
                                             "MOVDQArm",
                                             "MOVDQUrm",
                                             "MOVNTDQArm",
                                             "MOVSHDUPrm",
                                             "MOVSLDUPrm",
                                             "MOVUPDrm",
                                             "MOVUPSrm",
                                             "VBROADCASTSSrm",
                                             "VLDDQUrm",
                                             "VMOVAPDrm",
                                             "VMOVAPSrm",
                                             "VMOVDQArm",
                                             "VMOVDQUrm",
                                             "VMOVNTDQArm",
                                             "VMOVSHDUPrm",
                                             "VMOVSLDUPrm",
                                             "VMOVUPDrm",
                                             "VMOVUPSrm",
                                             "VPBROADCASTDrm",
                                             "VPBROADCASTQrm")>;

def SKXWriteResGroup72 : SchedWriteRes<[SKXPort0]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup72], (instregex "MMX_CVTPI2PSirr",
                                             "VCOMPRESSPDZ128rr(b?)(k?)(z?)",
                                             "VCOMPRESSPDZ256rr(b?)(k?)(z?)",
                                             "VCOMPRESSPDZrr(b?)(k?)(z?)",
                                             "VCOMPRESSPSZ128rr(b?)(k?)(z?)",
                                             "VCOMPRESSPSZ256rr(b?)(k?)(z?)",
                                             "VCOMPRESSPSZrr(b?)(k?)(z?)",
                                             "VPCOMPRESSDZ128rr(b?)(k?)(z?)",
                                             "VPCOMPRESSDZ256rr(b?)(k?)(z?)",
                                             "VPCOMPRESSDZrr(b?)(k?)(z?)",
                                             "VPCOMPRESSQZ128rr(b?)(k?)(z?)",
                                             "VPCOMPRESSQZ256rr(b?)(k?)(z?)",
                                             "VPCOMPRESSQZrr(b?)(k?)(z?)",
                                             "VPERMWZ128rr(b?)(k?)(z?)",
                                             "VPERMWZ256rr(b?)(k?)(z?)",
                                             "VPERMWZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup73 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PADDSBirm",
                                             "MMX_PADDSWirm",
                                             "MMX_PADDUSBirm",
                                             "MMX_PADDUSWirm",
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
                                             "MMX_PSLLDrm",
                                             "MMX_PSLLQrm",
                                             "MMX_PSLLWrm",
                                             "MMX_PSRADrm",
                                             "MMX_PSRAWrm",
                                             "MMX_PSRLDrm",
                                             "MMX_PSRLQrm",
                                             "MMX_PSRLWrm",
                                             "MMX_PSUBSBirm",
                                             "MMX_PSUBSWirm",
                                             "MMX_PSUBUSBirm",
                                             "MMX_PSUBUSWirm")>;

def SKXWriteResGroup74 : SchedWriteRes<[SKXPort0,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup74], (instregex "CVTSD2SI64rr",
                                             "CVTSD2SIrr",
                                             "CVTSS2SI64rr",
                                             "CVTSS2SIrr",
                                             "CVTTSD2SI64rr",
                                             "CVTTSD2SIrr",
                                             "VCVTSD2SI64Zrr(b?)(k?)(z?)",
                                             "VCVTSD2SI64rr",
                                             "VCVTSD2SIZrr(b?)(k?)(z?)",
                                             "VCVTSD2SIrr",
                                             "VCVTSD2USI64Zrr(b?)(k?)(z?)",
                                             "VCVTSD2USIZrr(b?)(k?)(z?)",
                                             "VCVTSS2SI64Zrr(b?)(k?)(z?)",
                                             "VCVTSS2SI64rr",
                                             "VCVTSS2SIZrr(b?)(k?)(z?)",
                                             "VCVTSS2SIrr",
                                             "VCVTSS2USIZrr(b?)(k?)(z?)",
                                             "VCVTTSD2SI64Zrr(b?)",
                                             "VCVTTSD2SI64rr",
                                             "VCVTTSD2SIZrr(b?)",
                                             "VCVTTSD2SIrr",
                                             "VCVTTSD2USI64Zrr(b?)",
                                             "VCVTTSD2USIZrr(b?)",
                                             "VCVTTSS2USIZrr(b?)")>;

def SKXWriteResGroup75 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PALIGNRrmi",
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
                                             "VMOVHPDZ128rm(b?)(k?)(z?)",
                                             "VMOVHPDrm",
                                             "VMOVHPSZ128rm(b?)(k?)(z?)",
                                             "VMOVHPSrm",
                                             "VMOVLPDZ128rm(b?)(k?)(z?)",
                                             "VMOVLPDrm",
                                             "VMOVLPSZ128rm(b?)(k?)(z?)",
                                             "VMOVLPSrm",
                                             "VPINSRBZrm(b?)(k?)(z?)",
                                             "VPINSRBrm",
                                             "VPINSRDZrm(b?)(k?)(z?)",
                                             "VPINSRDrm",
                                             "VPINSRQZrm(b?)(k?)(z?)",
                                             "VPINSRQrm",
                                             "VPINSRWZrm(b?)(k?)(z?)",
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
                                             "VPMOVZXWQrm")>;

def SKXWriteResGroup76 : SchedWriteRes<[SKXPort6,SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup76], (instregex "FARJMP64",
                                             "JMP(16|32|64)m")>;

def SKXWriteResGroup77 : SchedWriteRes<[SKXPort23,SKXPort05]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PABSBrm",
                                             "MMX_PABSDrm",
                                             "MMX_PABSWrm",
                                             "MMX_PADDBirm",
                                             "MMX_PADDDirm",
                                             "MMX_PADDQirm",
                                             "MMX_PADDWirm",
                                             "MMX_PANDNirm",
                                             "MMX_PANDirm",
                                             "MMX_PORirm",
                                             "MMX_PSIGNBrm",
                                             "MMX_PSIGNDrm",
                                             "MMX_PSIGNWrm",
                                             "MMX_PSUBBirm",
                                             "MMX_PSUBDirm",
                                             "MMX_PSUBQirm",
                                             "MMX_PSUBWirm",
                                             "MMX_PXORirm")>;

def SKXWriteResGroup78 : SchedWriteRes<[SKXPort23,SKXPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup78], (instregex "ADC(8|16|32|64)rm",
                                             "ADCX(32|64)rm",
                                             "ADOX(32|64)rm",
                                             "BT(16|32|64)mi8",
                                             "CMOV(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)(16|32|64)rm",
                                             "RORX(32|64)mi",
                                             "SARX(32|64)rm",
                                             "SBB(8|16|32|64)rm",
                                             "SHLX(32|64)rm",
                                             "SHRX(32|64)rm")>;

def SKXWriteResGroup79 : SchedWriteRes<[SKXPort23,SKXPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup79], (instregex "ANDN(32|64)rm",
                                             "BLSI(32|64)rm",
                                             "BLSMSK(32|64)rm",
                                             "BLSR(32|64)rm",
                                             "BZHI(32|64)rm",
                                             "MOVBE(16|32|64)rm")>;

def SKXWriteResGroup80 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup80], (instregex "VMOV(64to|QI2)PQIZrm(b?)(k?)(z?)",
                                             "VMOVDI2PDIZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup81 : SchedWriteRes<[SKXPort23,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup81], (instrs POP16r, POP32r, POP64r)>;
def: InstRW<[SKXWriteResGroup81], (instregex "ADD(8|16|32|64)rm",
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

def SKXWriteResGroup82 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup82], (instregex "CVTSI642SSrr",
                                             "HADDPDrr",
                                             "HADDPSrr",
                                             "HSUBPDrr",
                                             "HSUBPSrr",
                                             "VCVTSI642SSrr",
                                             "VCVTSI642SSZrr(b?)(k?)(z?)",
                                             "VCVTUSI642SSZrr(b?)(k?)(z?)",
                                             "VHADDPDYrr",
                                             "VHADDPDrr",
                                             "VHADDPSYrr",
                                             "VHADDPSrr",
                                             "VHSUBPDYrr",
                                             "VHSUBPDrr",
                                             "VHSUBPSYrr",
                                             "VHSUBPSrr")>;

def SKXWriteResGroup83 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup83], (instregex "SHLD(16|32|64)rrCL",
                                             "SHRD(16|32|64)rrCL")>;

def SKXWriteResGroup84 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort06,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup84], (instregex "SLDT(16|32|64)r")>;

def SKXWriteResGroup85 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup85], (instregex "VCVTPS2PHmr")>;

def SKXWriteResGroup86 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup86], (instregex "BTC(16|32|64)mi8",
                                             "BTR(16|32|64)mi8",
                                             "BTS(16|32|64)mi8",
                                             "SAR(8|16|32|64)m1",
                                             "SAR(8|16|32|64)mi",
                                             "SHL(8|16|32|64)m1",
                                             "SHL(8|16|32|64)mi",
                                             "SHR(8|16|32|64)m1",
                                             "SHR(8|16|32|64)mi")>;

def SKXWriteResGroup87 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup87], (instregex "ADD(8|16|32|64)mi",
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

def SKXWriteResGroup88 : SchedWriteRes<[SKXPort6,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKXWriteResGroup88], (instregex "STD")>;

def SKXWriteResGroup89 : SchedWriteRes<[SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup89], (instregex "LD_F32m",
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
                                             "VMOVNTDQAZrm(b?)(k?)(z?)",
                                             "VMOVSHDUPYrm",
                                             "VMOVSLDUPYrm",
                                             "VMOVUPDYrm",
                                             "VMOVUPSYrm",
                                             "VPBROADCASTDYrm",
                                             "VPBROADCASTQYrm")>;

def SKXWriteResGroup90 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup90], (instregex "VCVTDQ2PDYrr")>;

def SKXWriteResGroup91 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup91], (instregex "COMISDrm",
                                             "COMISSrm",
                                             "UCOMISDrm",
                                             "UCOMISSrm",
                                             "VCOMISDZrm(b?)(k?)(z?)",
                                             "VCOMISDrm",
                                             "VCOMISSZrm(b?)(k?)(z?)",
                                             "VCOMISSrm",
                                             "VUCOMISDZrm(b?)(k?)(z?)",
                                             "VUCOMISDrm",
                                             "VUCOMISSZrm(b?)(k?)(z?)",
                                             "VUCOMISSrm")>;

def SKXWriteResGroup92 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup92], (instregex "INSERTPSrm",
                                             "PACKSSDWrm",
                                             "PACKSSWBrm",
                                             "PACKUSDWrm",
                                             "PACKUSWBrm",
                                             "PALIGNRrmi",
                                             "PBLENDWrmi",
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
                                             "VINSERTPSZrm(b?)(k?)(z?)",
                                             "VINSERTPSrm",
                                             "VMOVSDZrm(b?)(k?)(z?)",
                                             "VMOVSSZrm(b?)(k?)(z?)",
                                             "VPACKSSDWZ128rm(b?)(k?)(z?)",
                                             "VPACKSSDWrm",
                                             "VPACKSSWBZ128rm(b?)(k?)(z?)",
                                             "VPACKSSWBrm",
                                             "VPACKUSDWZ128rm(b?)(k?)(z?)",
                                             "VPACKUSDWrm",
                                             "VPACKUSWBZ128rm(b?)(k?)(z?)",
                                             "VPACKUSWBrm",
                                             "VPALIGNRZ128rmi(b?)(k?)(z?)",
                                             "VPALIGNRrmi",
                                             "VPBLENDWrmi",
                                             "VPBROADCASTBZ128m(b?)(k?)(z?)",
                                             "VPBROADCASTBrm",
                                             "VPBROADCASTWZ128m(b?)(k?)(z?)",
                                             "VPBROADCASTWrm",
                                             "VPERMILPDZ128m(b?)i(k?)(z?)",
                                             "VPERMILPDZ128rm(b?)(k?)(z?)",
                                             "VPERMILPDmi",
                                             "VPERMILPDrm",
                                             "VPERMILPSZ128m(b?)i(k?)(z?)",
                                             "VPERMILPSZ128rm(b?)(k?)(z?)",
                                             "VPERMILPSmi",
                                             "VPERMILPSrm",
                                             "VPSHUFBZ128rm(b?)(k?)(z?)",
                                             "VPSHUFBrm",
                                             "VPSHUFDZ128m(b?)i(k?)(z?)",
                                             "VPSHUFDmi",
                                             "VPSHUFHWZ128mi(b?)(k?)(z?)",
                                             "VPSHUFHWmi",
                                             "VPSHUFLWZ128mi(b?)(k?)(z?)",
                                             "VPSHUFLWmi",
                                             "VPSLLDQZ128rm(b?)(k?)(z?)",
                                             "VPSRLDQZ128rm(b?)(k?)(z?)",
                                             "VPUNPCKHBWZ128rm(b?)(k?)(z?)",
                                             "VPUNPCKHBWrm",
                                             "VPUNPCKHDQZ128rm(b?)(k?)(z?)",
                                             "VPUNPCKHDQrm",
                                             "VPUNPCKHQDQZ128rm(b?)(k?)(z?)",
                                             "VPUNPCKHQDQrm",
                                             "VPUNPCKHWDZ128rm(b?)(k?)(z?)",
                                             "VPUNPCKHWDrm",
                                             "VPUNPCKLBWZ128rm(b?)(k?)(z?)",
                                             "VPUNPCKLBWrm",
                                             "VPUNPCKLDQZ128rm(b?)(k?)(z?)",
                                             "VPUNPCKLDQrm",
                                             "VPUNPCKLQDQZ128rm(b?)(k?)(z?)",
                                             "VPUNPCKLQDQrm",
                                             "VPUNPCKLWDZ128rm(b?)(k?)(z?)",
                                             "VPUNPCKLWDrm",
                                             "VSHUFPDZ128rm(b?)i(k?)(z?)",
                                             "VSHUFPDrmi",
                                             "VSHUFPSZ128rm(b?)i(k?)(z?)",
                                             "VSHUFPSrmi",
                                             "VUNPCKHPDZ128rm(b?)(k?)(z?)",
                                             "VUNPCKHPDrm",
                                             "VUNPCKHPSZ128rm(b?)(k?)(z?)",
                                             "VUNPCKHPSrm",
                                             "VUNPCKLPDZ128rm(b?)(k?)(z?)",
                                             "VUNPCKLPDrm",
                                             "VUNPCKLPSZ128rm(b?)(k?)(z?)",
                                             "VUNPCKLPSrm")>;

def SKXWriteResGroup93 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTDQ2PDZ256rr(b?)(k?)(z?)",
                                             "VCVTDQ2PDZrr(b?)(k?)(z?)",
                                             "VCVTPD2DQYrr",
                                             "VCVTPD2DQZ256rr(b?)(k?)(z?)",
                                             "VCVTPD2DQZrr(b?)(k?)(z?)",
                                             "VCVTPD2PSYrr",
                                             "VCVTPD2PSZ256rr(b?)(k?)(z?)",
                                             "VCVTPD2PSZrr(b?)(k?)(z?)",
                                             "VCVTPD2UDQZ256rr(b?)(k?)(z?)",
                                             "VCVTPD2UDQZrr(b?)(k?)(z?)",
                                             "VCVTPH2PSYrr",
                                             "VCVTPH2PSZ256rr(b?)(k?)(z?)",
                                             "VCVTPH2PSZrr(b?)(k?)(z?)",
                                             "VCVTPS2PDYrr",
                                             "VCVTPS2PDZ256rr(b?)(k?)(z?)",
                                             "VCVTPS2PDZrr(b?)(k?)(z?)",
                                             "VCVTPS2PHYrr",
                                             "VCVTPS2PHZ256rr(b?)(k?)(z?)",
                                             "VCVTPS2PHZrr(b?)(k?)(z?)",
                                             "VCVTPS2QQZ256rr(b?)(k?)(z?)",
                                             "VCVTPS2QQZrr(b?)(k?)(z?)",
                                             "VCVTPS2UQQZ256rr(b?)(k?)(z?)",
                                             "VCVTPS2UQQZrr(b?)(k?)(z?)",
                                             "VCVTQQ2PSZ256rr(b?)(k?)(z?)",
                                             "VCVTQQ2PSZrr(b?)(k?)(z?)",
                                             "VCVTTPD2DQYrr",
                                             "VCVTTPD2DQZ256rr(b?)(k?)(z?)",
                                             "VCVTTPD2DQZrr(b?)(k?)(z?)",
                                             "VCVTTPD2UDQZ256rr(b?)(k?)(z?)",
                                             "VCVTTPD2UDQZrr(b?)(k?)(z?)",
                                             "VCVTTPS2QQZ256rr(b?)(k?)(z?)",
                                             "VCVTTPS2QQZrr(b?)(k?)(z?)",
                                             "VCVTTPS2UQQZ256rr(b?)(k?)(z?)",
                                             "VCVTTPS2UQQZrr(b?)(k?)(z?)",
                                             "VCVTUDQ2PDZ256rr(b?)(k?)(z?)",
                                             "VCVTUDQ2PDZrr(b?)(k?)(z?)",
                                             "VCVTUQQ2PSZ256rr(b?)(k?)(z?)",
                                             "VCVTUQQ2PSZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup94 : SchedWriteRes<[SKXPort01,SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup94], (instregex "PABSBrm",
                                             "PABSDrm",
                                             "PABSWrm",
                                             "PADDSBrm",
                                             "PADDSWrm",
                                             "PADDUSBrm",
                                             "PADDUSWrm",
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
                                             "PMAX(C?)SDrm",
                                             "PMAXSWrm",
                                             "PMAXUBrm",
                                             "PMAXUDrm",
                                             "PMAXUWrm",
                                             "PMINSBrm",
                                             "PMIN(C?)SDrm",
                                             "PMINSWrm",
                                             "PMINUBrm",
                                             "PMINUDrm",
                                             "PMINUWrm",
                                             "PSIGNBrm",
                                             "PSIGNDrm",
                                             "PSIGNWrm",
                                             "PSLLDrm",
                                             "PSLLQrm",
                                             "PSLLWrm",
                                             "PSRADrm",
                                             "PSRAWrm",
                                             "PSRLDrm",
                                             "PSRLQrm",
                                             "PSRLWrm",
                                             "PSUBSBrm",
                                             "PSUBSWrm",
                                             "PSUBUSBrm",
                                             "PSUBUSWrm",
                                             "VPABSBZ128rm(b?)(k?)(z?)",
                                             "VPABSBrm",
                                             "VPABSDZ128rm(b?)(k?)(z?)",
                                             "VPABSDrm",
                                             "VPABSQZ128rm(b?)(k?)(z?)",
                                             "VPABSWZ128rm(b?)(k?)(z?)",
                                             "VPABSWrm",
                                             "VPADDSBZ128rm(b?)(k?)(z?)",
                                             "VPADDSBrm",
                                             "VPADDSWZ128rm(b?)(k?)(z?)",
                                             "VPADDSWrm",
                                             "VPADDUSBZ128rm(b?)(k?)(z?)",
                                             "VPADDUSBrm",
                                             "VPADDUSWZ128rm(b?)(k?)(z?)",
                                             "VPADDUSWrm",
                                             "VPAVGBZ128rm(b?)(k?)(z?)",
                                             "VPAVGBrm",
                                             "VPAVGWZ128rm(b?)(k?)(z?)",
                                             "VPAVGWrm",
                                             "VPCMPEQBrm",
                                             "VPCMPEQDrm",
                                             "VPCMPEQQrm",
                                             "VPCMPEQWrm",
                                             "VPCMPGTBrm",
                                             "VPCMPGTDrm",
                                             "VPCMPGTWrm",
                                             "VPMAXSBZ128rm(b?)(k?)(z?)",
                                             "VPMAXSBrm",
                                             "VPMAX(C?)SDZ128rm(b?)(k?)(z?)",
                                             "VPMAX(C?)SDrm",
                                             "VPMAXSWZ128rm(b?)(k?)(z?)",
                                             "VPMAXSWrm",
                                             "VPMAXUBZ128rm(b?)(k?)(z?)",
                                             "VPMAXUBrm",
                                             "VPMAXUDZ128rm(b?)(k?)(z?)",
                                             "VPMAXUDrm",
                                             "VPMAXUWZ128rm(b?)(k?)(z?)",
                                             "VPMAXUWrm",
                                             "VPMINSBZ128rm(b?)(k?)(z?)",
                                             "VPMINSBrm",
                                             "VPMIN(C?)SDZ128rm(b?)(k?)(z?)",
                                             "VPMIN(C?)SDrm",
                                             "VPMINSWZ128rm(b?)(k?)(z?)",
                                             "VPMINSWrm",
                                             "VPMINUBZ128rm(b?)(k?)(z?)",
                                             "VPMINUBrm",
                                             "VPMINUDZ128rm(b?)(k?)(z?)",
                                             "VPMINUDrm",
                                             "VPMINUWZ128rm(b?)(k?)(z?)",
                                             "VPMINUWrm",
                                             "VPROLDZ128m(b?)i(k?)(z?)",
                                             "VPROLQZ128m(b?)i(k?)(z?)",
                                             "VPROLVDZ128rm(b?)(k?)(z?)",
                                             "VPROLVQZ128rm(b?)(k?)(z?)",
                                             "VPRORDZ128m(b?)i(k?)(z?)",
                                             "VPRORQZ128m(b?)i(k?)(z?)",
                                             "VPRORVDZ128rm(b?)(k?)(z?)",
                                             "VPRORVQZ128rm(b?)(k?)(z?)",
                                             "VPSIGNBrm",
                                             "VPSIGNDrm",
                                             "VPSIGNWrm",
                                             "VPSLLDZ128m(b?)i(k?)(z?)",
                                             "VPSLLDZ128rm(b?)(k?)(z?)",
                                             "VPSLLDrm",
                                             "VPSLLQZ128m(b?)i(k?)(z?)",
                                             "VPSLLQZ128rm(b?)(k?)(z?)",
                                             "VPSLLQrm",
                                             "VPSLLVDZ128rm(b?)(k?)(z?)",
                                             "VPSLLVDrm",
                                             "VPSLLVQZ128rm(b?)(k?)(z?)",
                                             "VPSLLVQrm",
                                             "VPSLLVWZ128rm(b?)(k?)(z?)",
                                             "VPSLLWZ128mi(b?)(k?)(z?)",
                                             "VPSLLWZ128rm(b?)(k?)(z?)",
                                             "VPSLLWrm",
                                             "VPSRADZ128m(b?)i(k?)(z?)",
                                             "VPSRADZ128rm(b?)(k?)(z?)",
                                             "VPSRADrm",
                                             "VPSRAQZ128m(b?)i(k?)(z?)",
                                             "VPSRAQZ128rm(b?)(k?)(z?)",
                                             "VPSRAVDZ128rm(b?)(k?)(z?)",
                                             "VPSRAVDrm",
                                             "VPSRAVQZ128rm(b?)(k?)(z?)",
                                             "VPSRAVWZ128rm(b?)(k?)(z?)",
                                             "VPSRAWZ128mi(b?)(k?)(z?)",
                                             "VPSRAWZ128rm(b?)(k?)(z?)",
                                             "VPSRAWrm",
                                             "VPSRLDZ128m(b?)i(k?)(z?)",
                                             "VPSRLDZ128rm(b?)(k?)(z?)",
                                             "VPSRLDrm",
                                             "VPSRLQZ128m(b?)i(k?)(z?)",
                                             "VPSRLQZ128rm(b?)(k?)(z?)",
                                             "VPSRLQrm",
                                             "VPSRLVDZ128rm(b?)(k?)(z?)",
                                             "VPSRLVDrm",
                                             "VPSRLVQZ128rm(b?)(k?)(z?)",
                                             "VPSRLVQrm",
                                             "VPSRLVWZ128rm(b?)(k?)(z?)",
                                             "VPSRLWZ128mi(b?)(k?)(z?)",
                                             "VPSRLWZ128rm(b?)(k?)(z?)",
                                             "VPSRLWrm",
                                             "VPSUBSBZ128rm(b?)(k?)(z?)",
                                             "VPSUBSBrm",
                                             "VPSUBSWZ128rm(b?)(k?)(z?)",
                                             "VPSUBSWrm",
                                             "VPSUBUSBZ128rm(b?)(k?)(z?)",
                                             "VPSUBUSBrm",
                                             "VPSUBUSWZ128rm(b?)(k?)(z?)",
                                             "VPSUBUSWrm")>;

def SKXWriteResGroup95 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup95], (instregex "ANDNPDrm",
                                             "ANDNPSrm",
                                             "ANDPDrm",
                                             "ANDPSrm",
                                             "BLENDPDrmi",
                                             "BLENDPSrmi",
                                             "ORPDrm",
                                             "ORPSrm",
                                             "PADDBrm",
                                             "PADDDrm",
                                             "PADDQrm",
                                             "PADDWrm",
                                             "PANDNrm",
                                             "PANDrm",
                                             "PORrm",
                                             "PSUBBrm",
                                             "PSUBDrm",
                                             "PSUBQrm",
                                             "PSUBWrm",
                                             "PXORrm",
                                             "VANDNPDZ128rm(b?)(k?)(z?)",
                                             "VANDNPDrm",
                                             "VANDNPSZ128rm(b?)(k?)(z?)",
                                             "VANDNPSrm",
                                             "VANDPDZ128rm(b?)(k?)(z?)",
                                             "VANDPDrm",
                                             "VANDPSZ128rm(b?)(k?)(z?)",
                                             "VANDPSrm",
                                             "VBLENDMPDZ128rm(b?)(k?)(z?)",
                                             "VBLENDMPSZ128rm(b?)(k?)(z?)",
                                             "VBLENDPDrmi",
                                             "VBLENDPSrmi",
                                             "VBROADCASTI32X2Z128m(b?)(k?)(z?)",
                                             "VBROADCASTSSZ128m(b?)(k?)(z?)",
                                             "VINSERTF128rm",
                                             "VINSERTI128rm",
                                             "VMASKMOVPDrm",
                                             "VMASKMOVPSrm",
                                             "VMOVAPDZ128rm(b?)(k?)(z?)",
                                             "VMOVAPSZ128rm(b?)(k?)(z?)",
                                             "VMOVDDUPZ128rm(b?)(k?)(z?)",
                                             "VMOVDQA32Z128rm(b?)(k?)(z?)",
                                             "VMOVDQA64Z128rm(b?)(k?)(z?)",
                                             "VMOVDQU16Z128rm(b?)(k?)(z?)",
                                             "VMOVDQU32Z128rm(b?)(k?)(z?)",
                                             "VMOVDQU64Z128rm(b?)(k?)(z?)",
                                             "VMOVDQU8Z128rm(b?)(k?)(z?)",
                                             "VMOVNTDQAZ128rm(b?)(k?)(z?)",
                                             "VMOVSHDUPZ128rm(b?)(k?)(z?)",
                                             "VMOVSLDUPZ128rm(b?)(k?)(z?)",
                                             "VMOVUPDZ128rm(b?)(k?)(z?)",
                                             "VMOVUPSZ128rm(b?)(k?)(z?)",
                                             "VORPDZ128rm(b?)(k?)(z?)",
                                             "VORPDrm",
                                             "VORPSZ128rm(b?)(k?)(z?)",
                                             "VORPSrm",
                                             "VPADDBZ128rm(b?)(k?)(z?)",
                                             "VPADDBrm",
                                             "VPADDDZ128rm(b?)(k?)(z?)",
                                             "VPADDDrm",
                                             "VPADDQZ128rm(b?)(k?)(z?)",
                                             "VPADDQrm",
                                             "VPADDWZ128rm(b?)(k?)(z?)",
                                             "VPADDWrm",
                                             "VPANDDZ128rm(b?)(k?)(z?)",
                                             "VPANDNDZ128rm(b?)(k?)(z?)",
                                             "VPANDNQZ128rm(b?)(k?)(z?)",
                                             "VPANDNrm",
                                             "VPANDQZ128rm(b?)(k?)(z?)",
                                             "VPANDrm",
                                             "VPBLENDDrmi",
                                             "VPBLENDMBZ128rm(b?)(k?)(z?)",
                                             "VPBLENDMDZ128rm(b?)(k?)(z?)",
                                             "VPBLENDMQZ128rm(b?)(k?)(z?)",
                                             "VPBLENDMWZ128rm(b?)(k?)(z?)",
                                             "VPBROADCASTDZ128m(b?)(k?)(z?)",
                                             "VPBROADCASTQZ128m(b?)(k?)(z?)",
                                             "VPMASKMOVDrm",
                                             "VPMASKMOVQrm",
                                             "VPORDZ128rm(b?)(k?)(z?)",
                                             "VPORQZ128rm(b?)(k?)(z?)",
                                             "VPORrm",
                                             "VPSUBBZ128rm(b?)(k?)(z?)",
                                             "VPSUBBrm",
                                             "VPSUBDZ128rm(b?)(k?)(z?)",
                                             "VPSUBDrm",
                                             "VPSUBQZ128rm(b?)(k?)(z?)",
                                             "VPSUBQrm",
                                             "VPSUBWZ128rm(b?)(k?)(z?)",
                                             "VPSUBWrm",
                                             "VPTERNLOGDZ128rm(b?)i(k?)(z?)",
                                             "VPTERNLOGQZ128rm(b?)i(k?)(z?)",
                                             "VPXORDZ128rm(b?)(k?)(z?)",
                                             "VPXORQZ128rm(b?)(k?)(z?)",
                                             "VPXORrm",
                                             "VXORPDZ128rm(b?)(k?)(z?)",
                                             "VXORPDrm",
                                             "VXORPSZ128rm(b?)(k?)(z?)",
                                             "VXORPSrm",
                                             "XORPDrm",
                                             "XORPSrm")>;

def SKXWriteResGroup96 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup96], (instregex "MMX_PACKSSDWirm",
                                             "MMX_PACKSSWBirm",
                                             "MMX_PACKUSWBirm")>;

def SKXWriteResGroup97 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup97], (instregex "VPERMI2W128rr(b?)(k?)(z?)",
                                             "VPERMI2W256rr(b?)(k?)(z?)",
                                             "VPERMI2Wrr(b?)(k?)(z?)",
                                             "VPERMT2W128rr(b?)(k?)(z?)",
                                             "VPERMT2W256rr(b?)(k?)(z?)",
                                             "VPERMT2Wrr(b?)(k?)(z?)")>;

def SKXWriteResGroup98 : SchedWriteRes<[SKXPort23,SKXPort06]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup98], (instregex "CMOV(A|BE)(16|32|64)rm")>;

def SKXWriteResGroup99 : SchedWriteRes<[SKXPort23,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup99], (instregex "LEAVE64",
                                             "SCASB",
                                             "SCASL",
                                             "SCASQ",
                                             "SCASW")>;

def SKXWriteResGroup100 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup100], (instregex "CVTTSS2SI64rr",
                                              "CVTTSS2SIrr",
                                              "VCVTSS2USI64Zrr(b?)(k?)(z?)",
                                              "VCVTTSS2SI64Zrr(b?)",
                                              "VCVTTSS2SI64rr",
                                              "VCVTTSS2SIZrr(b?)",
                                              "VCVTTSS2SIrr",
                                              "VCVTTSS2USI64Zrr(b?)")>;

def SKXWriteResGroup101 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort05]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup101], (instregex "FLDCW16m")>;

def SKXWriteResGroup102 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup102], (instregex "(V?)LDMXCSR")>;

def SKXWriteResGroup103 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup103], (instregex "KMOVBkm",
                                              "KMOVDkm",
                                              "KMOVQkm",
                                              "KMOVWkm")>;

def SKXWriteResGroup104 : SchedWriteRes<[SKXPort6,SKXPort23,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup104], (instregex "LRETQ",
                                              "RETQ")>;

def SKXWriteResGroup105 : SchedWriteRes<[SKXPort23,SKXPort06,SKXPort15]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup105], (instregex "BEXTR(32|64)rm")>;

def SKXWriteResGroup106 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup106], (instregex "VCOMPRESSPDZ128mr(b?)(k?)(z?)",
                                              "VCOMPRESSPDZ256mr(b?)(k?)(z?)",
                                              "VCOMPRESSPDZmr(b?)(k?)(z?)",
                                              "VCOMPRESSPSZ128mr(b?)(k?)(z?)",
                                              "VCOMPRESSPSZ256mr(b?)(k?)(z?)",
                                              "VCOMPRESSPSZmr(b?)(k?)(z?)",
                                              "VPCOMPRESSDZ128mr(b?)(k?)(z?)",
                                              "VPCOMPRESSDZ256mr(b?)(k?)(z?)",
                                              "VPCOMPRESSDZmr(b?)(k?)(z?)",
                                              "VPCOMPRESSQZ128mr(b?)(k?)(z?)",
                                              "VPCOMPRESSQZ256mr(b?)(k?)(z?)",
                                              "VPCOMPRESSQZmr(b?)(k?)(z?)")>;

def SKXWriteResGroup107 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKXWriteResGroup107], (instregex "ROL(8|16|32|64)m1",
                                              "ROL(8|16|32|64)mi",
                                              "ROR(8|16|32|64)m1",
                                              "ROR(8|16|32|64)mi")>;

def SKXWriteResGroup108 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKXWriteResGroup108], (instregex "XADD(8|16|32|64)rm")>;

def SKXWriteResGroup109 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[SKXWriteResGroup109], (instregex "CALL(16|32|64)m",
                                              "FARCALL64")>;

def SKXWriteResGroup110 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [1,2,2,2];
}
def: InstRW<[SKXWriteResGroup110], (instrs VPSCATTERDQZ128mr,
                                           VPSCATTERQQZ128mr,
                                           VSCATTERDPDZ128mr,
                                           VSCATTERQPDZ128mr)>;

def SKXWriteResGroup111 : SchedWriteRes<[SKXPort6,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,1,2];
}
def: InstRW<[SKXWriteResGroup111], (instrs LOOP)>;

def SKXWriteResGroup112 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 11;
  let ResourceCycles = [1,4,4,2];
}
def: InstRW<[SKXWriteResGroup112], (instrs VPSCATTERDQZ256mr,
                                           VPSCATTERQQZ256mr,
                                           VSCATTERDPDZ256mr,
                                           VSCATTERQPDZ256mr)>;

def SKXWriteResGroup113 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 19;
  let ResourceCycles = [1,8,8,2];
}
def: InstRW<[SKXWriteResGroup113], (instrs VPSCATTERDQZmr,
                                           VPSCATTERQQZmr,
                                           VSCATTERDPDZmr,
                                           VSCATTERQPDZmr)>;

def SKXWriteResGroup114 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 36;
  let ResourceCycles = [1,16,1,16,2];
}
def: InstRW<[SKXWriteResGroup114], (instrs VSCATTERDPSZmr)>;

def SKXWriteResGroup116 : SchedWriteRes<[SKXPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup116], (instregex "ROUNDPDr",
                                              "ROUNDPSr",
                                              "ROUNDSDr",
                                              "ROUNDSSr",
                                              "VRNDSCALEPDZ128rri(b?)(k?)(z?)",
                                              "VRNDSCALEPDZ256rri(b?)(k?)(z?)",
                                              "VRNDSCALEPDZrri(b?)(k?)(z?)",
                                              "VRNDSCALEPSZ128rri(b?)(k?)(z?)",
                                              "VRNDSCALEPSZ256rri(b?)(k?)(z?)",
                                              "VRNDSCALEPSZrri(b?)(k?)(z?)",
                                              "VRNDSCALESDr(b?)(k?)(z?)",
                                              "VRNDSCALESSr(b?)(k?)(z?)",
                                              "VROUNDPDr",
                                              "VROUNDPSr",
                                              "VROUNDSDr",
                                              "VROUNDSSr",
                                              "VROUNDPDYr",
                                              "VROUNDPSYr")>;

def SKXWriteResGroup116_2 : SchedWriteRes<[SKXPort015]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup116_2], (instregex "PMULLDrr",
                                                "VPMULLDYrr",
                                                "VPMULLDZ128rr(b?)(k?)(z?)",
                                                "VPMULLDZ256rr(b?)(k?)(z?)",
                                                "VPMULLDZrr(b?)(k?)(z?)",
                                                "VPMULLDrr")>;

def SKXWriteResGroup117 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup117], (instregex "VTESTPDrm",
                                              "VTESTPSrm")>;

def SKXWriteResGroup118 : SchedWriteRes<[SKXPort1,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup118], (instrs IMUL32rm, IMUL32rmi, IMUL32rmi8, IMUL64rm, IMUL64rmi32, IMUL64rmi8)>;
def: InstRW<[SKXWriteResGroup118], (instrs IMUL8m, MUL8m)>;
def: InstRW<[SKXWriteResGroup118], (instregex "BSF(16|32|64)rm",
                                              "BSR(16|32|64)rm",
                                              "LZCNT(16|32|64)rm",
                                              "PDEP(32|64)rm",
                                              "PEXT(32|64)rm",
                                              "POPCNT(16|32|64)rm",
                                              "TZCNT(16|32|64)rm")>;

def SKXWriteResGroup118_16_1 : SchedWriteRes<[SKXPort1, SKXPort0156, SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1]; 
}
def: InstRW<[SKXWriteResGroup118_16_1], (instrs IMUL16rm, IMUL16rmi, IMUL16rmi8)>;

def SKXWriteResGroup118_16_2 : SchedWriteRes<[SKXPort1, SKXPort0156, SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 5;
}
def: InstRW<[SKXWriteResGroup118_16_2], (instrs IMUL16m, MUL16m)>;

def SKXWriteResGroup119 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup119], (instregex "FCOM32m",
                                              "FCOM64m",
                                              "FCOMP32m",
                                              "FCOMP64m",
                                              "MMX_PSADBWirm",
                                              "VFPCLASSSDrm(b?)(k?)(z?)",
                                              "VPACKSSDWYrm",
                                              "VPACKSSDWZ256rm(b?)(k?)(z?)",
                                              "VPACKSSDWZrm(b?)(k?)(z?)",
                                              "VPACKSSWBYrm",
                                              "VPACKSSWBZ256rm(b?)(k?)(z?)",
                                              "VPACKSSWBZrm(b?)(k?)(z?)",
                                              "VPACKUSDWYrm",
                                              "VPACKUSDWZ256rm(b?)(k?)(z?)",
                                              "VPACKUSDWZrm(b?)(k?)(z?)",
                                              "VPACKUSWBYrm",
                                              "VPACKUSWBZ256rm(b?)(k?)(z?)",
                                              "VPACKUSWBZrm(b?)(k?)(z?)",
                                              "VPALIGNRYrmi",
                                              "VPALIGNRZ256rmi(b?)(k?)(z?)",
                                              "VPALIGNRZrmi(b?)(k?)(z?)",
                                              "VPBLENDWYrmi",
                                              "VPBROADCASTBYrm",
                                              "VPBROADCASTBZ256m(b?)(k?)(z?)",
                                              "VPBROADCASTBZm(b?)(k?)(z?)",
                                              "VPBROADCASTWYrm",
                                              "VPBROADCASTWZ256m(b?)(k?)(z?)",
                                              "VPBROADCASTWZm(b?)(k?)(z?)",
                                              "VPERMILPDYmi",
                                              "VPERMILPDYrm",
                                              "VPERMILPDZ256m(b?)i(k?)(z?)",
                                              "VPERMILPDZ256rm(b?)(k?)(z?)",
                                              "VPERMILPDZm(b?)i(k?)(z?)",
                                              "VPERMILPDZrm(b?)(k?)(z?)",
                                              "VPERMILPSYmi",
                                              "VPERMILPSYrm",
                                              "VPERMILPSZ256m(b?)i(k?)(z?)",
                                              "VPERMILPSZ256rm(b?)(k?)(z?)",
                                              "VPERMILPSZm(b?)i(k?)(z?)",
                                              "VPERMILPSZrm(b?)(k?)(z?)",
                                              "VPMOVSXBDYrm",
                                              "VPMOVSXBQYrm",
                                              "VPMOVSXWQYrm",
                                              "VPSHUFBYrm",
                                              "VPSHUFBZ256rm(b?)(k?)(z?)",
                                              "VPSHUFBZrm(b?)(k?)(z?)",
                                              "VPSHUFDYmi",
                                              "VPSHUFDZ256m(b?)i(k?)(z?)",
                                              "VPSHUFDZm(b?)i(k?)(z?)",
                                              "VPSHUFHWYmi",
                                              "VPSHUFHWZ256mi(b?)(k?)(z?)",
                                              "VPSHUFHWZmi(b?)(k?)(z?)",
                                              "VPSHUFLWYmi",
                                              "VPSHUFLWZ256mi(b?)(k?)(z?)",
                                              "VPSHUFLWZmi(b?)(k?)(z?)",
                                              "VPSLLDQZ256rm(b?)(k?)(z?)",
                                              "VPSLLDQZrm(b?)(k?)(z?)",
                                              "VPSRLDQZ256rm(b?)(k?)(z?)",
                                              "VPSRLDQZrm(b?)(k?)(z?)",
                                              "VPUNPCKHBWYrm",
                                              "VPUNPCKHBWZ256rm(b?)(k?)(z?)",
                                              "VPUNPCKHBWZrm(b?)(k?)(z?)",
                                              "VPUNPCKHDQYrm",
                                              "VPUNPCKHDQZ256rm(b?)(k?)(z?)",
                                              "VPUNPCKHDQZrm(b?)(k?)(z?)",
                                              "VPUNPCKHQDQYrm",
                                              "VPUNPCKHQDQZ256rm(b?)(k?)(z?)",
                                              "VPUNPCKHQDQZrm(b?)(k?)(z?)",
                                              "VPUNPCKHWDYrm",
                                              "VPUNPCKHWDZ256rm(b?)(k?)(z?)",
                                              "VPUNPCKHWDZrm(b?)(k?)(z?)",
                                              "VPUNPCKLBWYrm",
                                              "VPUNPCKLBWZ256rm(b?)(k?)(z?)",
                                              "VPUNPCKLBWZrm(b?)(k?)(z?)",
                                              "VPUNPCKLDQYrm",
                                              "VPUNPCKLDQZ256rm(b?)(k?)(z?)",
                                              "VPUNPCKLDQZrm(b?)(k?)(z?)",
                                              "VPUNPCKLQDQYrm",
                                              "VPUNPCKLQDQZ256rm(b?)(k?)(z?)",
                                              "VPUNPCKLQDQZrm(b?)(k?)(z?)",
                                              "VPUNPCKLWDYrm",
                                              "VPUNPCKLWDZ256rm(b?)(k?)(z?)",
                                              "VPUNPCKLWDZrm(b?)(k?)(z?)",
                                              "VSHUFPDYrmi",
                                              "VSHUFPDZ256rm(b?)i(k?)(z?)",
                                              "VSHUFPDZrm(b?)i(k?)(z?)",
                                              "VSHUFPSYrmi",
                                              "VSHUFPSZ256rm(b?)i(k?)(z?)",
                                              "VSHUFPSZrm(b?)i(k?)(z?)",
                                              "VUNPCKHPDYrm",
                                              "VUNPCKHPDZ256rm(b?)(k?)(z?)",
                                              "VUNPCKHPDZrm(b?)(k?)(z?)",
                                              "VUNPCKHPSYrm",
                                              "VUNPCKHPSZ256rm(b?)(k?)(z?)",
                                              "VUNPCKHPSZrm(b?)(k?)(z?)",
                                              "VUNPCKLPDYrm",
                                              "VUNPCKLPDZ256rm(b?)(k?)(z?)",
                                              "VUNPCKLPDZrm(b?)(k?)(z?)",
                                              "VUNPCKLPSYrm",
                                              "VUNPCKLPSZ256rm(b?)(k?)(z?)",
                                              "VUNPCKLPSZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup120 : SchedWriteRes<[SKXPort01,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSBYrm",
                                              "VPABSBZ256rm(b?)(k?)(z?)",
                                              "VPABSBZrm(b?)(k?)(z?)",
                                              "VPABSDYrm",
                                              "VPABSDZ256rm(b?)(k?)(z?)",
                                              "VPABSDZrm(b?)(k?)(z?)",
                                              "VPABSQZ256rm(b?)(k?)(z?)",
                                              "VPABSQZrm(b?)(k?)(z?)",
                                              "VPABSWYrm",
                                              "VPABSWZ256rm(b?)(k?)(z?)",
                                              "VPABSWZrm(b?)(k?)(z?)",
                                              "VPADDSBYrm",
                                              "VPADDSBZ256rm(b?)(k?)(z?)",
                                              "VPADDSBZrm(b?)(k?)(z?)",
                                              "VPADDSWYrm",
                                              "VPADDSWZ256rm(b?)(k?)(z?)",
                                              "VPADDSWZrm(b?)(k?)(z?)",
                                              "VPADDUSBYrm",
                                              "VPADDUSBZ256rm(b?)(k?)(z?)",
                                              "VPADDUSBZrm(b?)(k?)(z?)",
                                              "VPADDUSWYrm",
                                              "VPADDUSWZ256rm(b?)(k?)(z?)",
                                              "VPADDUSWZrm(b?)(k?)(z?)",
                                              "VPAVGBYrm",
                                              "VPAVGBZ256rm(b?)(k?)(z?)",
                                              "VPAVGBZrm(b?)(k?)(z?)",
                                              "VPAVGWYrm",
                                              "VPAVGWZ256rm(b?)(k?)(z?)",
                                              "VPAVGWZrm(b?)(k?)(z?)",
                                              "VPCMPEQBYrm",
                                              "VPCMPEQDYrm",
                                              "VPCMPEQQYrm",
                                              "VPCMPEQWYrm",
                                              "VPCMPGTBYrm",
                                              "VPCMPGTDYrm",
                                              "VPCMPGTWYrm",
                                              "VPMAXSBYrm",
                                              "VPMAXSBZ256rm(b?)(k?)(z?)",
                                              "VPMAXSBZrm(b?)(k?)(z?)",
                                              "VPMAX(C?)SDYrm",
                                              "VPMAX(C?)SDZ256rm(b?)(k?)(z?)",
                                              "VPMAX(C?)SDZrm(b?)(k?)(z?)",
                                              "VPMAXSWYrm",
                                              "VPMAXSWZ256rm(b?)(k?)(z?)",
                                              "VPMAXSWZrm(b?)(k?)(z?)",
                                              "VPMAXUBYrm",
                                              "VPMAXUBZ256rm(b?)(k?)(z?)",
                                              "VPMAXUBZrm(b?)(k?)(z?)",
                                              "VPMAXUDYrm",
                                              "VPMAXUDZ256rm(b?)(k?)(z?)",
                                              "VPMAXUDZrm(b?)(k?)(z?)",
                                              "VPMAXUWYrm",
                                              "VPMAXUWZ256rm(b?)(k?)(z?)",
                                              "VPMAXUWZrm(b?)(k?)(z?)",
                                              "VPMINSBYrm",
                                              "VPMINSBZ256rm(b?)(k?)(z?)",
                                              "VPMINSBZrm(b?)(k?)(z?)",
                                              "VPMIN(C?)SDYrm",
                                              "VPMIN(C?)SDZ256rm(b?)(k?)(z?)",
                                              "VPMIN(C?)SDZrm(b?)(k?)(z?)",
                                              "VPMINSWYrm",
                                              "VPMINSWZ256rm(b?)(k?)(z?)",
                                              "VPMINSWZrm(b?)(k?)(z?)",
                                              "VPMINUBYrm",
                                              "VPMINUBZ256rm(b?)(k?)(z?)",
                                              "VPMINUBZrm(b?)(k?)(z?)",
                                              "VPMINUDYrm",
                                              "VPMINUDZ256rm(b?)(k?)(z?)",
                                              "VPMINUDZrm(b?)(k?)(z?)",
                                              "VPMINUWYrm",
                                              "VPMINUWZ256rm(b?)(k?)(z?)",
                                              "VPMINUWZrm(b?)(k?)(z?)",
                                              "VPROLDZ256m(b?)i(k?)(z?)",
                                              "VPROLDZm(b?)i(k?)(z?)",
                                              "VPROLQZ256m(b?)i(k?)(z?)",
                                              "VPROLQZm(b?)i(k?)(z?)",
                                              "VPROLVDZ256rm(b?)(k?)(z?)",
                                              "VPROLVDZrm(b?)(k?)(z?)",
                                              "VPROLVQZ256rm(b?)(k?)(z?)",
                                              "VPROLVQZrm(b?)(k?)(z?)",
                                              "VPRORDZ256m(b?)i(k?)(z?)",
                                              "VPRORDZm(b?)i(k?)(z?)",
                                              "VPRORQZ256m(b?)i(k?)(z?)",
                                              "VPRORQZm(b?)i(k?)(z?)",
                                              "VPRORVDZ256rm(b?)(k?)(z?)",
                                              "VPRORVDZrm(b?)(k?)(z?)",
                                              "VPRORVQZ256rm(b?)(k?)(z?)",
                                              "VPRORVQZrm(b?)(k?)(z?)",
                                              "VPSIGNBYrm",
                                              "VPSIGNDYrm",
                                              "VPSIGNWYrm",
                                              "VPSLLDYrm",
                                              "VPSLLDZ256m(b?)i(k?)(z?)",
                                              "VPSLLDZ256rm(b?)(k?)(z?)",
                                              "VPSLLDZm(b?)i(k?)(z?)",
                                              "VPSLLDZrm(b?)(k?)(z?)",
                                              "VPSLLQYrm",
                                              "VPSLLQZ256m(b?)i(k?)(z?)",
                                              "VPSLLQZ256rm(b?)(k?)(z?)",
                                              "VPSLLQZm(b?)i(k?)(z?)",
                                              "VPSLLQZrm(b?)(k?)(z?)",
                                              "VPSLLVDYrm",
                                              "VPSLLVDZ256rm(b?)(k?)(z?)",
                                              "VPSLLVDZrm(b?)(k?)(z?)",
                                              "VPSLLVQYrm",
                                              "VPSLLVQZ256rm(b?)(k?)(z?)",
                                              "VPSLLVQZrm(b?)(k?)(z?)",
                                              "VPSLLVWZ256rm(b?)(k?)(z?)",
                                              "VPSLLVWZrm(b?)(k?)(z?)",
                                              "VPSLLWYrm",
                                              "VPSLLWZ256mi(b?)(k?)(z?)",
                                              "VPSLLWZ256rm(b?)(k?)(z?)",
                                              "VPSLLWZmi(b?)(k?)(z?)",
                                              "VPSLLWZrm(b?)(k?)(z?)",
                                              "VPSRADYrm",
                                              "VPSRADZ256m(b?)i(k?)(z?)",
                                              "VPSRADZ256rm(b?)(k?)(z?)",
                                              "VPSRADZm(b?)i(k?)(z?)",
                                              "VPSRADZrm(b?)(k?)(z?)",
                                              "VPSRAQZ256m(b?)i(k?)(z?)",
                                              "VPSRAQZ256rm(b?)(k?)(z?)",
                                              "VPSRAQZm(b?)i(k?)(z?)",
                                              "VPSRAQZrm(b?)(k?)(z?)",
                                              "VPSRAVDYrm",
                                              "VPSRAVDZ256rm(b?)(k?)(z?)",
                                              "VPSRAVDZrm(b?)(k?)(z?)",
                                              "VPSRAVQZ256rm(b?)(k?)(z?)",
                                              "VPSRAVQZrm(b?)(k?)(z?)",
                                              "VPSRAVWZ256rm(b?)(k?)(z?)",
                                              "VPSRAVWZrm(b?)(k?)(z?)",
                                              "VPSRAWYrm",
                                              "VPSRAWZ256mi(b?)(k?)(z?)",
                                              "VPSRAWZ256rm(b?)(k?)(z?)",
                                              "VPSRAWZmi(b?)(k?)(z?)",
                                              "VPSRAWZrm(b?)(k?)(z?)",
                                              "VPSRLDYrm",
                                              "VPSRLDZ256m(b?)i(k?)(z?)",
                                              "VPSRLDZ256rm(b?)(k?)(z?)",
                                              "VPSRLDZm(b?)i(k?)(z?)",
                                              "VPSRLDZrm(b?)(k?)(z?)",
                                              "VPSRLQYrm",
                                              "VPSRLQZ256m(b?)i(k?)(z?)",
                                              "VPSRLQZ256rm(b?)(k?)(z?)",
                                              "VPSRLQZm(b?)i(k?)(z?)",
                                              "VPSRLQZrm(b?)(k?)(z?)",
                                              "VPSRLVDYrm",
                                              "VPSRLVDZ256rm(b?)(k?)(z?)",
                                              "VPSRLVDZrm(b?)(k?)(z?)",
                                              "VPSRLVQYrm",
                                              "VPSRLVQZ256rm(b?)(k?)(z?)",
                                              "VPSRLVQZrm(b?)(k?)(z?)",
                                              "VPSRLVWZ256rm(b?)(k?)(z?)",
                                              "VPSRLVWZrm(b?)(k?)(z?)",
                                              "VPSRLWYrm",
                                              "VPSRLWZ256mi(b?)(k?)(z?)",
                                              "VPSRLWZ256rm(b?)(k?)(z?)",
                                              "VPSRLWZmi(b?)(k?)(z?)",
                                              "VPSRLWZrm(b?)(k?)(z?)",
                                              "VPSUBSBYrm",
                                              "VPSUBSBZ256rm(b?)(k?)(z?)",
                                              "VPSUBSBZrm(b?)(k?)(z?)",
                                              "VPSUBSWYrm",
                                              "VPSUBSWZ256rm(b?)(k?)(z?)",
                                              "VPSUBSWZrm(b?)(k?)(z?)",
                                              "VPSUBUSBYrm",
                                              "VPSUBUSBZ256rm(b?)(k?)(z?)",
                                              "VPSUBUSBZrm(b?)(k?)(z?)",
                                              "VPSUBUSWYrm",
                                              "VPSUBUSWZ256rm(b?)(k?)(z?)",
                                              "VPSUBUSWZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup121 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup121], (instregex "VANDNPDYrm",
                                              "VANDNPDZ256rm(b?)(k?)(z?)",
                                              "VANDNPDZrm(b?)(k?)(z?)",
                                              "VANDNPSYrm",
                                              "VANDNPSZ256rm(b?)(k?)(z?)",
                                              "VANDNPSZrm(b?)(k?)(z?)",
                                              "VANDPDYrm",
                                              "VANDPDZ256rm(b?)(k?)(z?)",
                                              "VANDPDZrm(b?)(k?)(z?)",
                                              "VANDPSYrm",
                                              "VANDPSZ256rm(b?)(k?)(z?)",
                                              "VANDPSZrm(b?)(k?)(z?)",
                                              "VBLENDMPDZ256rm(b?)(k?)(z?)",
                                              "VBLENDMPDZrm(b?)(k?)(z?)",
                                              "VBLENDMPSZ256rm(b?)(k?)(z?)",
                                              "VBLENDMPSZrm(b?)(k?)(z?)",
                                              "VBLENDPDYrmi",
                                              "VBLENDPSYrmi",
                                              "VBROADCASTF32X2Z256m(b?)(k?)(z?)",
                                              "VBROADCASTF32X2Zm(b?)(k?)(z?)",
                                              "VBROADCASTF32X4Z256rm(b?)(k?)(z?)",
                                              "VBROADCASTF32X4rm(b?)(k?)(z?)",
                                              "VBROADCASTF32X8rm(b?)(k?)(z?)",
                                              "VBROADCASTF64X2Z128rm(b?)(k?)(z?)",
                                              "VBROADCASTF64X2rm(b?)(k?)(z?)",
                                              "VBROADCASTF64X4rm(b?)(k?)(z?)",
                                              "VBROADCASTI32X2Z256m(b?)(k?)(z?)",
                                              "VBROADCASTI32X2Zm(b?)(k?)(z?)",
                                              "VBROADCASTI32X4Z256rm(b?)(k?)(z?)",
                                              "VBROADCASTI32X4rm(b?)(k?)(z?)",
                                              "VBROADCASTI32X8rm(b?)(k?)(z?)",
                                              "VBROADCASTI64X2Z128rm(b?)(k?)(z?)",
                                              "VBROADCASTI64X2rm(b?)(k?)(z?)",
                                              "VBROADCASTI64X4rm(b?)(k?)(z?)",
                                              "VBROADCASTSDZ256m(b?)(k?)(z?)",
                                              "VBROADCASTSDZm(b?)(k?)(z?)",
                                              "VBROADCASTSSZ256m(b?)(k?)(z?)",
                                              "VBROADCASTSSZm(b?)(k?)(z?)",
                                              "VINSERTF32x4Z256rm(b?)(k?)(z?)",
                                              "VINSERTF32x4Zrm(b?)(k?)(z?)",
                                              "VINSERTF32x8Zrm(b?)(k?)(z?)",
                                              "VINSERTF64x2Z256rm(b?)(k?)(z?)",
                                              "VINSERTF64x2Zrm(b?)(k?)(z?)",
                                              "VINSERTF64x4Zrm(b?)(k?)(z?)",
                                              "VINSERTI32x4Z256rm(b?)(k?)(z?)",
                                              "VINSERTI32x4Zrm(b?)(k?)(z?)",
                                              "VINSERTI32x8Zrm(b?)(k?)(z?)",
                                              "VINSERTI64x2Z256rm(b?)(k?)(z?)",
                                              "VINSERTI64x2Zrm(b?)(k?)(z?)",
                                              "VINSERTI64x4Zrm(b?)(k?)(z?)",
                                              "VMASKMOVPDYrm",
                                              "VMASKMOVPSYrm",
                                              "VMOVAPDZ256rm(b?)(k?)(z?)",
                                              "VMOVAPDZrm(b?)(k?)(z?)",
                                              "VMOVAPSZ256rm(b?)(k?)(z?)",
                                              "VMOVAPSZrm(b?)(k?)(z?)",
                                              "VMOVDDUPZ256rm(b?)(k?)(z?)",
                                              "VMOVDDUPZrm(b?)(k?)(z?)",
                                              "VMOVDQA32Z256rm(b?)(k?)(z?)",
                                              "VMOVDQA32Zrm(b?)(k?)(z?)",
                                              "VMOVDQA64Z256rm(b?)(k?)(z?)",
                                              "VMOVDQA64Zrm(b?)(k?)(z?)",
                                              "VMOVDQU16Z256rm(b?)(k?)(z?)",
                                              "VMOVDQU16Zrm(b?)(k?)(z?)",
                                              "VMOVDQU32Z256rm(b?)(k?)(z?)",
                                              "VMOVDQU32Zrm(b?)(k?)(z?)",
                                              "VMOVDQU64Z256rm(b?)(k?)(z?)",
                                              "VMOVDQU64Zrm(b?)(k?)(z?)",
                                              "VMOVDQU8Z256rm(b?)(k?)(z?)",
                                              "VMOVDQU8Zrm(b?)(k?)(z?)",
                                              "VMOVNTDQAZ256rm(b?)(k?)(z?)",
                                              "VMOVSHDUPZ256rm(b?)(k?)(z?)",
                                              "VMOVSHDUPZrm(b?)(k?)(z?)",
                                              "VMOVSLDUPZ256rm(b?)(k?)(z?)",
                                              "VMOVSLDUPZrm(b?)(k?)(z?)",
                                              "VMOVUPDZ256rm(b?)(k?)(z?)",
                                              "VMOVUPDZrm(b?)(k?)(z?)",
                                              "VMOVUPSZ256rm(b?)(k?)(z?)",
                                              "VMOVUPSZrm(b?)(k?)(z?)",
                                              "VORPDYrm",
                                              "VORPDZ256rm(b?)(k?)(z?)",
                                              "VORPDZrm(b?)(k?)(z?)",
                                              "VORPSYrm",
                                              "VORPSZ256rm(b?)(k?)(z?)",
                                              "VORPSZrm(b?)(k?)(z?)",
                                              "VPADDBYrm",
                                              "VPADDBZ256rm(b?)(k?)(z?)",
                                              "VPADDBZrm(b?)(k?)(z?)",
                                              "VPADDDYrm",
                                              "VPADDDZ256rm(b?)(k?)(z?)",
                                              "VPADDDZrm(b?)(k?)(z?)",
                                              "VPADDQYrm",
                                              "VPADDQZ256rm(b?)(k?)(z?)",
                                              "VPADDQZrm(b?)(k?)(z?)",
                                              "VPADDWYrm",
                                              "VPADDWZ256rm(b?)(k?)(z?)",
                                              "VPADDWZrm(b?)(k?)(z?)",
                                              "VPANDDZ256rm(b?)(k?)(z?)",
                                              "VPANDDZrm(b?)(k?)(z?)",
                                              "VPANDNDZ256rm(b?)(k?)(z?)",
                                              "VPANDNDZrm(b?)(k?)(z?)",
                                              "VPANDNQZ256rm(b?)(k?)(z?)",
                                              "VPANDNQZrm(b?)(k?)(z?)",
                                              "VPANDNYrm",
                                              "VPANDQZ256rm(b?)(k?)(z?)",
                                              "VPANDQZrm(b?)(k?)(z?)",
                                              "VPANDYrm",
                                              "VPBLENDDYrmi",
                                              "VPBLENDMBZ256rm(b?)(k?)(z?)",
                                              "VPBLENDMBZrm(b?)(k?)(z?)",
                                              "VPBLENDMDZ256rm(b?)(k?)(z?)",
                                              "VPBLENDMDZrm(b?)(k?)(z?)",
                                              "VPBLENDMQZ256rm(b?)(k?)(z?)",
                                              "VPBLENDMQZrm(b?)(k?)(z?)",
                                              "VPBLENDMWZ256rm(b?)(k?)(z?)",
                                              "VPBLENDMWZrm(b?)(k?)(z?)",
                                              "VPBROADCASTDZ256m(b?)(k?)(z?)",
                                              "VPBROADCASTDZm(b?)(k?)(z?)",
                                              "VPBROADCASTQZ256m(b?)(k?)(z?)",
                                              "VPBROADCASTQZm(b?)(k?)(z?)",
                                              "VPMASKMOVDYrm",
                                              "VPMASKMOVQYrm",
                                              "VPORDZ256rm(b?)(k?)(z?)",
                                              "VPORDZrm(b?)(k?)(z?)",
                                              "VPORQZ256rm(b?)(k?)(z?)",
                                              "VPORQZrm(b?)(k?)(z?)",
                                              "VPORYrm",
                                              "VPSUBBYrm",
                                              "VPSUBBZ256rm(b?)(k?)(z?)",
                                              "VPSUBBZrm(b?)(k?)(z?)",
                                              "VPSUBDYrm",
                                              "VPSUBDZ256rm(b?)(k?)(z?)",
                                              "VPSUBDZrm(b?)(k?)(z?)",
                                              "VPSUBQYrm",
                                              "VPSUBQZ256rm(b?)(k?)(z?)",
                                              "VPSUBQZrm(b?)(k?)(z?)",
                                              "VPSUBWYrm",
                                              "VPSUBWZrm(b?)(k?)(z?)",
                                              "VPTERNLOGDZ256rm(b?)i(k?)(z?)",
                                              "VPTERNLOGDZrm(b?)i(k?)(z?)",
                                              "VPTERNLOGQZ256rm(b?)i(k?)(z?)",
                                              "VPTERNLOGQZrm(b?)i(k?)(z?)",
                                              "VPXORDZ256rm(b?)(k?)(z?)",
                                              "VPXORDZrm(b?)(k?)(z?)",
                                              "VPXORQZ256rm(b?)(k?)(z?)",
                                              "VPXORQZrm(b?)(k?)(z?)",
                                              "VPXORYrm",
                                              "VXORPDYrm",
                                              "VXORPDZ256rm(b?)(k?)(z?)",
                                              "VXORPDZrm(b?)(k?)(z?)",
                                              "VXORPSYrm",
                                              "VXORPSZ256rm(b?)(k?)(z?)",
                                              "VXORPSZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup122 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup122], (instregex "BLENDVPDrm0",
                                              "BLENDVPSrm0",
                                              "PBLENDVBrm0",
                                              "VBLENDVPDrm",
                                              "VBLENDVPSrm",
                                              "VPBLENDVBYrm",
                                              "VPBLENDVBrm")>;

def SKXWriteResGroup123 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup123], (instregex "MMX_PHADDSWrm",
                                              "MMX_PHSUBSWrm")>;

def SKXWriteResGroup124 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort05]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup124], (instregex "MMX_PHADDDrm",
                                              "MMX_PHADDWrm",
                                              "MMX_PHSUBDrm",
                                              "MMX_PHSUBWrm")>;

def SKXWriteResGroup125 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237,SKXPort015]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup125], (instregex "VCVTPS2PHYmr")>;

def SKXWriteResGroup126 : SchedWriteRes<[SKXPort23,SKXPort237,SKXPort06]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup126], (instregex "ROR(8|16|32|64)mCL")>;

def SKXWriteResGroup127 : SchedWriteRes<[SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKXWriteResGroup127], (instregex "RCL(8|16|32|64)m1",
                                              "RCL(8|16|32|64)mi",
                                              "RCR(8|16|32|64)m1",
                                              "RCR(8|16|32|64)mi")>;

def SKXWriteResGroup128 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKXWriteResGroup128], (instregex "ROL(8|16|32|64)mCL",
                                              "SAR(8|16|32|64)mCL",
                                              "SHL(8|16|32|64)mCL",
                                              "SHR(8|16|32|64)mCL")>;

def SKXWriteResGroup129 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKXWriteResGroup129], (instregex "ADC(8|16|32|64)mi")>;

def SKXWriteResGroup130 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKXWriteResGroup130], (instregex "ADC(8|16|32|64)mr",
                                              "CMPXCHG(8|16|32|64)rm",
                                              "SBB(8|16|32|64)mi",
                                              "SBB(8|16|32|64)mr")>;

def SKXWriteResGroup131 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 8;
  let ResourceCycles = [1,2,1,2,2];
}
def: InstRW<[SKXWriteResGroup131], (instrs VPSCATTERQDZ128mr,
                                           VPSCATTERQDZ256mr,
                                           VSCATTERQPSZ128mr,
                                           VSCATTERQPSZ256mr)>;

def SKXWriteResGroup132 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 12;
  let ResourceCycles = [1,4,1,4,2];
}
def: InstRW<[SKXWriteResGroup132], (instrs VPSCATTERDDZ128mr,
                                           VSCATTERDPSZ128mr)>;

def SKXWriteResGroup133 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 20;
  let ResourceCycles = [1,8,1,8,2];
}
def: InstRW<[SKXWriteResGroup133], (instrs VPSCATTERDDZ256mr,
                                           VSCATTERDPSZ256mr)>;

def SKXWriteResGroup134 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 36;
  let ResourceCycles = [1,16,1,16,2];
}
def: InstRW<[SKXWriteResGroup134], (instrs VPSCATTERDDZmr)>;

def SKXWriteResGroup135 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup135], (instregex "MMX_CVTPI2PSirm",
                                              "MMX_PMADDUBSWrm",
                                              "MMX_PMADDWDirm",
                                              "MMX_PMULHRSWrm",
                                              "MMX_PMULHUWirm",
                                              "MMX_PMULHWirm",
                                              "MMX_PMULLWirm",
                                              "MMX_PMULUDQirm",
                                              "RCPSSm",
                                              "RSQRTSSm",
                                              "VRCPSSm",
                                              "VRSQRTSSm",
                                              "VTESTPDYrm",
                                              "VTESTPSYrm")>;

def SKXWriteResGroup136 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup136], (instregex "PCMPGTQrm",
                                              "PSADBWrm",
                                              "VALIGNDZ128rm(b?)i(k?)(z?)",
                                              "VALIGNQZ128rm(b?)i(k?)(z?)",
                                              "VCMPPDZ128rm(b?)i(k?)(z?)",
                                              "VCMPPSZ128rm(b?)i(k?)(z?)",
                                              "VCMPSDZrm(_Int)?(k?)(z?)",
                                              "VCMPSSZrm(_Int)?(k?)(z?)",
                                              "VDBPSADBWZ128rmi(b?)(k?)(z?)",
                                              "VFPCLASSSSrm(b?)(k?)(z?)",
                                              "VPCMPBZ128rmi(b?)(k?)(z?)",
                                              "VPCMPDZ128rmi(b?)(k?)(z?)",
                                              "VPCMPEQBZ128rm(b?)(k?)(z?)",
                                              "VPCMPEQDZ128rm(b?)(k?)(z?)",
                                              "VPCMPEQQZ128rm(b?)(k?)(z?)",
                                              "VPCMPEQWZ128rm(b?)(k?)(z?)",
                                              "VPCMPGTBZ128rm(b?)(k?)(z?)",
                                              "VPCMPGTDZ128rm(b?)(k?)(z?)",
                                              "VPCMPGTQZ128rm(b?)(k?)(z?)",
                                              "VPCMPGTQrm",
                                              "VPCMPGTWZ128rm(b?)(k?)(z?)",
                                              "VPCMPQZ128rmi(b?)(k?)(z?)",
                                              "VPCMPUBZ128rmi(b?)(k?)(z?)",
                                              "VPCMPUDZ128rmi(b?)(k?)(z?)",
                                              "VPCMPUQZ128rmi(b?)(k?)(z?)",
                                              "VPCMPUWZ128rmi(b?)(k?)(z?)",
                                              "VPCMPWZ128rmi(b?)(k?)(z?)",
                                              "VPERMI2D128rm(b?)(k?)(z?)",
                                              "VPERMI2PD128rm(b?)(k?)(z?)",
                                              "VPERMI2PS128rm(b?)(k?)(z?)",
                                              "VPERMI2Q128rm(b?)(k?)(z?)",
                                              "VPERMT2D128rm(b?)(k?)(z?)",
                                              "VPERMT2PD128rm(b?)(k?)(z?)",
                                              "VPERMT2PS128rm(b?)(k?)(z?)",
                                              "VPERMT2Q128rm(b?)(k?)(z?)",
                                              "VPMAXSQZ128rm(b?)(k?)(z?)",
                                              "VPMAXUQZ128rm(b?)(k?)(z?)",
                                              "VPMINSQZ128rm(b?)(k?)(z?)",
                                              "VPMINUQZ128rm(b?)(k?)(z?)",
                                              "VPMOVSXBDZ128rm(b?)(k?)(z?)",
                                              "VPMOVSXBQZ128rm(b?)(k?)(z?)",
                                              "VPMOVSXBWYrm",
                                              "VPMOVSXBWZ128rm(b?)(k?)(z?)",
                                              "VPMOVSXDQYrm",
                                              "VPMOVSXDQZ128rm(b?)(k?)(z?)",
                                              "VPMOVSXWDYrm",
                                              "VPMOVSXWDZ128rm(b?)(k?)(z?)",
                                              "VPMOVSXWQZ128rm(b?)(k?)(z?)",
                                              "VPMOVZXBDZ128rm(b?)(k?)(z?)",
                                              "VPMOVZXBQZ128rm(b?)(k?)(z?)",
                                              "VPMOVZXBWZ128rm(b?)(k?)(z?)",
                                              "VPMOVZXDQZ128rm(b?)(k?)(z?)",
                                              "VPMOVZXWDYrm",
                                              "VPMOVZXWDZ128rm(b?)(k?)(z?)",
                                              "VPMOVZXWQZ128rm(b?)(k?)(z?)",
                                              "VPSADBWZ128rm(b?)(k?)(z?)",
                                              "VPSADBWrm",
                                              "VPTESTMBZ128rm(b?)(k?)(z?)",
                                              "VPTESTMDZ128rm(b?)(k?)(z?)",
                                              "VPTESTMQZ128rm(b?)(k?)(z?)",
                                              "VPTESTMWZ128rm(b?)(k?)(z?)",
                                              "VPTESTNMBZ128rm(b?)(k?)(z?)",
                                              "VPTESTNMDZ128rm(b?)(k?)(z?)",
                                              "VPTESTNMQZ128rm(b?)(k?)(z?)",
                                              "VPTESTNMWZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup137 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup137],
            (instregex "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)m")>;
def: InstRW<[SKXWriteResGroup137], (instregex "ADDSDrm",
                                              "ADDSSrm",
                                              "CMPSDrm",
                                              "CMPSSrm",
                                              "CVTPS2PDrm",
                                              "MAX(C?)SDrm",
                                              "MAX(C?)SSrm",
                                              "MIN(C?)SDrm",
                                              "MIN(C?)SSrm",
                                              "MMX_CVTPS2PIirm",
                                              "MMX_CVTTPS2PIirm",
                                              "MULSDrm",
                                              "MULSSrm",
                                              "SUBSDrm",
                                              "SUBSSrm",
                                              "VADDSDrm",
                                              "VADDSSrm",
                                              "VCMPSDrm",
                                              "VCMPSSrm",
                                              "VCVTPH2PSrm",
                                              "VCVTPS2PDrm",
                                              "VMAX(C?)SDrm",
                                              "VMAX(C?)SSrm",
                                              "VMIN(C?)SDrm",
                                              "VMIN(C?)SSrm",
                                              "VMULSDrm",
                                              "VMULSSrm",
                                              "VSUBSDrm",
                                              "VSUBSSrm")>;

def SKXWriteResGroup138 : SchedWriteRes<[SKXPort0,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup138], (instregex "VRCP14PDZr(b?)(k?)(z?)",
                                              "VRCP14PSZr(b?)(k?)(z?)",
                                              "VRSQRT14PDZr(b?)(k?)(z?)",
                                              "VRSQRT14PSZr(b?)(k?)(z?)")>;

def SKXWriteResGroup139 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup139], (instregex "(V?)DPPDrri")>;

def SKXWriteResGroup140 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup140], (instregex "VBLENDVPDYrm",
                                              "VBLENDVPSYrm")>;

def SKXWriteResGroup141 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup141], (instregex "(V?)PTESTrm")>;

def SKXWriteResGroup142 : SchedWriteRes<[SKXPort1,SKXPort5,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup142], (instrs IMUL64m, MUL64m, MULX64rm)>;

def SKXWriteResGroup143 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup143], (instregex "(V?)PHADDSWrm",
                                              "(V?)PHSUBSWrm")>;

def SKXWriteResGroup144 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup144], (instregex "(V?)PHADDDrm",
                                              "(V?)PHADDWrm",
                                              "(V?)PHSUBDrm",
                                              "(V?)PHSUBWrm")>;

def SKXWriteResGroup145 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup145], (instregex "SHLD(16|32|64)mri8",
                                              "SHRD(16|32|64)mri8")>;

def SKXWriteResGroup146 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort23,SKXPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup146], (instregex "LAR(16|32|64)rm",
                                              "LSL(16|32|64)rm")>;

def SKXWriteResGroup147 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup147], (instregex "RCPPSm",
                                              "RSQRTPSm",
                                              "VRCP14PDZ128m(b?)(k?)(z?)",
                                              "VRCP14PSZ128m(b?)(k?)(z?)",
                                              "VRCP14SDrm(b?)(k?)(z?)",
                                              "VRCP14SSrm(b?)(k?)(z?)",
                                              "VRCPPSm",
                                              "VRSQRT14PDZ128m(b?)(k?)(z?)",
                                              "VRSQRT14PSZ128m(b?)(k?)(z?)",
                                              "VRSQRT14SDrm(b?)(k?)(z?)",
                                              "VRSQRT14SSrm(b?)(k?)(z?)",
                                              "VRSQRTPSm")>;

def SKXWriteResGroup148 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup148], (instregex "ADD_F32m",
                                              "ADD_F64m",
                                              "ILD_F16m",
                                              "ILD_F32m",
                                              "ILD_F64m",
                                              "SUBR_F32m",
                                              "SUBR_F64m",
                                              "SUB_F32m",
                                              "SUB_F64m",
                                              "VALIGNDZ256rm(b?)i(k?)(z?)",
                                              "VALIGNDZrm(b?)i(k?)(z?)",
                                              "VALIGNQZ256rm(b?)i(k?)(z?)",
                                              "VALIGNQZrm(b?)i(k?)(z?)",
                                              "VCMPPDZ256rm(b?)i(k?)(z?)",
                                              "VCMPPDZrm(b?)i(k?)(z?)",
                                              "VCMPPSZ256rm(b?)i(k?)(z?)",
                                              "VCMPPSZrm(b?)i(k?)(z?)",
                                              "VDBPSADBWZ256rmi(b?)(k?)(z?)",
                                              "VDBPSADBWZrmi(b?)(k?)(z?)",
                                              "VPCMPBZ256rmi(b?)(k?)(z?)",
                                              "VPCMPBZrmi(b?)(k?)(z?)",
                                              "VPCMPDZ256rmi(b?)(k?)(z?)",
                                              "VPCMPDZrmi(b?)(k?)(z?)",
                                              "VPCMPEQBZ256rm(b?)(k?)(z?)",
                                              "VPCMPEQBZrm(b?)(k?)(z?)",
                                              "VPCMPEQDZ256rm(b?)(k?)(z?)",
                                              "VPCMPEQDZrm(b?)(k?)(z?)",
                                              "VPCMPEQQZ256rm(b?)(k?)(z?)",
                                              "VPCMPEQQZrm(b?)(k?)(z?)",
                                              "VPCMPEQWZ256rm(b?)(k?)(z?)",
                                              "VPCMPEQWZrm(b?)(k?)(z?)",
                                              "VPCMPGTBZ256rm(b?)(k?)(z?)",
                                              "VPCMPGTBZrm(b?)(k?)(z?)",
                                              "VPCMPGTDZ256rm(b?)(k?)(z?)",
                                              "VPCMPGTDZrm(b?)(k?)(z?)",
                                              "VPCMPGTQYrm",
                                              "VPCMPGTQZ256rm(b?)(k?)(z?)",
                                              "VPCMPGTQZrm(b?)(k?)(z?)",
                                              "VPCMPGTWZ256rm(b?)(k?)(z?)",
                                              "VPCMPGTWZrm(b?)(k?)(z?)",
                                              "VPCMPQZ256rmi(b?)(k?)(z?)",
                                              "VPCMPQZrmi(b?)(k?)(z?)",
                                              "VPCMPUBZ256rmi(b?)(k?)(z?)",
                                              "VPCMPUBZrmi(b?)(k?)(z?)",
                                              "VPCMPUDZ256rmi(b?)(k?)(z?)",
                                              "VPCMPUDZrmi(b?)(k?)(z?)",
                                              "VPCMPUQZ256rmi(b?)(k?)(z?)",
                                              "VPCMPUQZrmi(b?)(k?)(z?)",
                                              "VPCMPUWZ256rmi(b?)(k?)(z?)",
                                              "VPCMPUWZrmi(b?)(k?)(z?)",
                                              "VPCMPWZ256rmi(b?)(k?)(z?)",
                                              "VPCMPWZrmi(b?)(k?)(z?)",
                                              "VPERM2F128rm",
                                              "VPERM2I128rm",
                                              "VPERMDYrm",
                                              "VPERMDZ256rm(b?)(k?)(z?)",
                                              "VPERMDZrm(b?)(k?)(z?)",
                                              "VPERMI2D256rm(b?)(k?)(z?)",
                                              "VPERMI2Drm(b?)(k?)(z?)",
                                              "VPERMI2PD256rm(b?)(k?)(z?)",
                                              "VPERMI2PDrm(b?)(k?)(z?)",
                                              "VPERMI2PS256rm(b?)(k?)(z?)",
                                              "VPERMI2PSrm(b?)(k?)(z?)",
                                              "VPERMI2Q256rm(b?)(k?)(z?)",
                                              "VPERMI2Qrm(b?)(k?)(z?)",
                                              "VPERMPDYmi",
                                              "VPERMPDZ256m(b?)i(k?)(z?)",
                                              "VPERMPDZ256rm(b?)(k?)(z?)",
                                              "VPERMPDZm(b?)i(k?)(z?)",
                                              "VPERMPDZrm(b?)(k?)(z?)",
                                              "VPERMPSYrm",
                                              "VPERMPSZ256rm(b?)(k?)(z?)",
                                              "VPERMPSZrm(b?)(k?)(z?)",
                                              "VPERMQYmi",
                                              "VPERMQZ256m(b?)i(k?)(z?)",
                                              "VPERMQZ256rm(b?)(k?)(z?)",
                                              "VPERMQZm(b?)i(k?)(z?)",
                                              "VPERMQZrm(b?)(k?)(z?)",
                                              "VPERMT2D256rm(b?)(k?)(z?)",
                                              "VPERMT2Drm(b?)(k?)(z?)",
                                              "VPERMT2PD256rm(b?)(k?)(z?)",
                                              "VPERMT2PDrm(b?)(k?)(z?)",
                                              "VPERMT2PS256rm(b?)(k?)(z?)",
                                              "VPERMT2PSrm(b?)(k?)(z?)",
                                              "VPERMT2Q256rm(b?)(k?)(z?)",
                                              "VPERMT2Qrm(b?)(k?)(z?)",
                                              "VPMAXSQZ256rm(b?)(k?)(z?)",
                                              "VPMAXSQZrm(b?)(k?)(z?)",
                                              "VPMAXUQZ256rm(b?)(k?)(z?)",
                                              "VPMAXUQZrm(b?)(k?)(z?)",
                                              "VPMINSQZ256rm(b?)(k?)(z?)",
                                              "VPMINSQZrm(b?)(k?)(z?)",
                                              "VPMINUQZ256rm(b?)(k?)(z?)",
                                              "VPMINUQZrm(b?)(k?)(z?)",
                                              "VPMOVSXBDZ256rm(b?)(k?)(z?)",
                                              "VPMOVSXBDZrm(b?)(k?)(z?)",
                                              "VPMOVSXBQZ256rm(b?)(k?)(z?)",
                                              "VPMOVSXBQZrm(b?)(k?)(z?)",
                                              "VPMOVSXBWZ256rm(b?)(k?)(z?)",
                                              "VPMOVSXBWZrm(b?)(k?)(z?)",
                                              "VPMOVSXDQZ256rm(b?)(k?)(z?)",
                                              "VPMOVSXDQZrm(b?)(k?)(z?)",
                                              "VPMOVSXWDZ256rm(b?)(k?)(z?)",
                                              "VPMOVSXWDZrm(b?)(k?)(z?)",
                                              "VPMOVSXWQZ256rm(b?)(k?)(z?)",
                                              "VPMOVSXWQZrm(b?)(k?)(z?)",
                                              "VPMOVZXBDYrm",
                                              "VPMOVZXBDZ256rm(b?)(k?)(z?)",
                                              "VPMOVZXBDZrm(b?)(k?)(z?)",
                                              "VPMOVZXBQYrm",
                                              "VPMOVZXBQZ256rm(b?)(k?)(z?)",
                                              "VPMOVZXBQZrm(b?)(k?)(z?)",
                                              "VPMOVZXBWYrm",
                                              "VPMOVZXBWZ256rm(b?)(k?)(z?)",
                                              "VPMOVZXBWZrm(b?)(k?)(z?)",
                                              "VPMOVZXDQYrm",
                                              "VPMOVZXDQZ256rm(b?)(k?)(z?)",
                                              "VPMOVZXDQZrm(b?)(k?)(z?)",
                                              "VPMOVZXWDZ256rm(b?)(k?)(z?)",
                                              "VPMOVZXWDZrm(b?)(k?)(z?)",
                                              "VPMOVZXWQYrm",
                                              "VPMOVZXWQZ256rm(b?)(k?)(z?)",
                                              "VPMOVZXWQZrm(b?)(k?)(z?)",
                                              "VPSADBWYrm",
                                              "VPSADBWZ256rm(b?)(k?)(z?)",
                                              "VPSADBWZrm(b?)(k?)(z?)",
                                              "VPTESTMBZ256rm(b?)(k?)(z?)",
                                              "VPTESTMBZrm(b?)(k?)(z?)",
                                              "VPTESTMDZ256rm(b?)(k?)(z?)",
                                              "VPTESTMDZrm(b?)(k?)(z?)",
                                              "VPTESTMQZ256rm(b?)(k?)(z?)",
                                              "VPTESTMQZrm(b?)(k?)(z?)",
                                              "VPTESTMWZ256rm(b?)(k?)(z?)",
                                              "VPTESTMWZrm(b?)(k?)(z?)",
                                              "VPTESTNMBZ256rm(b?)(k?)(z?)",
                                              "VPTESTNMBZrm(b?)(k?)(z?)",
                                              "VPTESTNMDZ256rm(b?)(k?)(z?)",
                                              "VPTESTNMDZrm(b?)(k?)(z?)",
                                              "VPTESTNMQZ256rm(b?)(k?)(z?)",
                                              "VPTESTNMQZrm(b?)(k?)(z?)",
                                              "VPTESTNMWZ256rm(b?)(k?)(z?)",
                                              "VPTESTNMWZrm(b?)(k?)(z?)",
                                              "VSHUFF32X4Z256rm(b?)i(k?)(z?)",
                                              "VSHUFF32X4Zrm(b?)i(k?)(z?)",
                                              "VSHUFF64X2Z256rm(b?)i(k?)(z?)",
                                              "VSHUFF64X2Zrm(b?)i(k?)(z?)",
                                              "VSHUFI32X4Z256rm(b?)i(k?)(z?)",
                                              "VSHUFI32X4Zrm(b?)i(k?)(z?)",
                                              "VSHUFI64X2Z256rm(b?)i(k?)(z?)",
                                              "VSHUFI64X2Zrm(b?)i(k?)(z?)")>;

def SKXWriteResGroup149 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup149],
          (instregex
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Z128m(b?)(k?)(z?)",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)m",
           "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)Zm(_Int)?(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "ADDPDrm",
                                              "ADDPSrm",
                                              "ADDSUBPDrm",
                                              "ADDSUBPSrm",
                                              "CMPPDrmi",
                                              "CMPPSrmi",
                                              "CVTDQ2PSrm",
                                              "CVTPS2DQrm",
                                              "CVTSS2SDrm",
                                              "CVTTPS2DQrm",
                                              "MAX(C?)PDrm",
                                              "MAX(C?)PSrm",
                                              "MIN(C?)PDrm",
                                              "MIN(C?)PSrm",
                                              "MULPDrm",
                                              "MULPSrm",
                                              "PHMINPOSUWrm",
                                              "PMADDUBSWrm",
                                              "PMADDWDrm",
                                              "PMULDQrm",
                                              "PMULHRSWrm",
                                              "PMULHUWrm",
                                              "PMULHWrm",
                                              "PMULLWrm",
                                              "PMULUDQrm",
                                              "SUBPDrm",
                                              "SUBPSrm",
                                              "VADDPDZ128rm(b?)(k?)(z?)",
                                              "VADDPDrm",
                                              "VADDPSZ128rm(b?)(k?)(z?)",
                                              "VADDPSrm",
                                              "VADDSDZrm(_Int)?(k?)(z?)",
                                              "VADDSSZrm(_Int)?(k?)(z?)",
                                              "VADDSUBPDrm",
                                              "VADDSUBPSrm",
                                              "VCMPPDrmi",
                                              "VCMPPSrmi",
                                              "VCVTDQ2PDZ128rm(b?)(k?)(z?)",
                                              "VCVTDQ2PSZ128rm(b?)(k?)(z?)",
                                              "VCVTDQ2PSrm",
                                              "VCVTPD2QQZ128rm(b?)(k?)(z?)",
                                              "VCVTPD2UQQZ128rm(b?)(k?)(z?)",
                                              "VCVTPH2PSYrm",
                                              "VCVTPH2PSZ128rm(b?)(k?)(z?)",
                                              "VCVTPS2DQZ128rm(b?)(k?)(z?)",
                                              "VCVTPS2DQrm",
                                              "VCVTPS2PDZ128rm(b?)(k?)(z?)",
                                              "VCVTPS2QQZ128rm(b?)(k?)(z?)",
                                              "VCVTPS2UDQZ128rm(b?)(k?)(z?)",
                                              "VCVTPS2UQQZ128rm(b?)(k?)(z?)",
                                              "VCVTQQ2PDZ128rm(b?)(k?)(z?)",
                                              "VCVTQQ2PSZ128rm(b?)(k?)(z?)",
                                              "VCVTSS2SDZrm(_Int)?(k?)(z?)",
                                              "VCVTSS2SDrm",
                                              "VCVTTPD2QQZ128rm(b?)(k?)(z?)",
                                              "VCVTTPD2UQQZ128rm(b?)(k?)(z?)",
                                              "VCVTTPS2DQZ128rm(b?)(k?)(z?)",
                                              "VCVTTPS2DQrm",
                                              "VCVTTPS2QQZ128rm(b?)(k?)(z?)",
                                              "VCVTTPS2UDQZ128rm(b?)(k?)(z?)",
                                              "VCVTTPS2UQQZ128rm(b?)(k?)(z?)",
                                              "VCVTUDQ2PDZ128rm(b?)(k?)(z?)",
                                              "VCVTUDQ2PSZ128rm(b?)(k?)(z?)",
                                              "VCVTUQQ2PDZ128rm(b?)(k?)(z?)",
                                              "VCVTUQQ2PSZ128rm(b?)(k?)(z?)",
                                              "VFIXUPIMMPDZ128rm(b?)i(k?)(z?)",
                                              "VFIXUPIMMPSZ128rm(b?)i(k?)(z?)",
                                              "VFIXUPIMMSDrmi(b?)(k?)(z?)",
                                              "VFIXUPIMMSSrmi(b?)(k?)(z?)",
                                              "VGETEXPPDZ128m(b?)(k?)(z?)",
                                              "VGETEXPPSZ128m(b?)(k?)(z?)",
                                              "VGETEXPSDm(b?)(k?)(z?)",
                                              "VGETEXPSSm(b?)(k?)(z?)",
                                              "VGETMANTPDZ128rm(b?)i(k?)(z?)",
                                              "VGETMANTPSZ128rm(b?)i(k?)(z?)",
                                              "VGETMANTSDZ128rmi(b?)(k?)(z?)",
                                              "VGETMANTSSZ128rmi(b?)(k?)(z?)",
                                              "VMAX(C?)PDZ128rm(b?)(k?)(z?)",
                                              "VMAX(C?)PDrm",
                                              "VMAX(C?)PSZ128rm(b?)(k?)(z?)",
                                              "VMAX(C?)PSrm",
                                              "VMAX(C?)SDZrm(_Int)?(k?)(z?)",
                                              "VMAX(C?)SSZrm(_Int)?(k?)(z?)",
                                              "VMIN(C?)PDZ128rm(b?)(k?)(z?)",
                                              "VMIN(C?)PDrm",
                                              "VMIN(C?)PSZ128rm(b?)(k?)(z?)",
                                              "VMIN(C?)PSrm",
                                              "VMIN(C?)SDZrm(_Int)?(k?)(z?)",
                                              "VMIN(C?)SSZrm(_Int)?(k?)(z?)",
                                              "VMULPDZ128rm(b?)(k?)(z?)",
                                              "VMULPDrm",
                                              "VMULPSZ128rm(b?)(k?)(z?)",
                                              "VMULPSrm",
                                              "VMULSDZrm(_Int)?(k?)(z?)",
                                              "VMULSSZrm(_Int)?(k?)(z?)",
                                              "VPHMINPOSUWrm",
                                              "VPLZCNTDZ128rm(b?)(k?)(z?)",
                                              "VPLZCNTQZ128rm(b?)(k?)(z?)",
                                              "VPMADDUBSWZ128rm(b?)(k?)(z?)",
                                              "VPMADDUBSWrm",
                                              "VPMADDWDZ128rm(b?)(k?)(z?)",
                                              "VPMADDWDrm",
                                              "VPMULDQZ128rm(b?)(k?)(z?)",
                                              "VPMULDQrm",
                                              "VPMULHRSWZ128rm(b?)(k?)(z?)",
                                              "VPMULHRSWrm",
                                              "VPMULHUWZ128rm(b?)(k?)(z?)",
                                              "VPMULHUWrm",
                                              "VPMULHWZ128rm(b?)(k?)(z?)",
                                              "VPMULHWrm",
                                              "VPMULLWZ128rm(b?)(k?)(z?)",
                                              "VPMULLWrm",
                                              "VPMULUDQZ128rm(b?)(k?)(z?)",
                                              "VPMULUDQrm",
                                              "VRANGEPDZ128rm(b?)i(k?)(z?)",
                                              "VRANGEPSZ128rm(b?)i(k?)(z?)",
                                              "VRANGESDZ128rmi(b?)(k?)(z?)",
                                              "VRANGESSZ128rmi(b?)(k?)(z?)",
                                              "VREDUCEPDZ128rm(b?)i(k?)(z?)",
                                              "VREDUCEPSZ128rm(b?)i(k?)(z?)",
                                              "VREDUCESDZ128rmi(b?)(k?)(z?)",
                                              "VREDUCESSZ128rmi(b?)(k?)(z?)",
                                              "VSCALEFPDZ128rm(b?)(k?)(z?)",
                                              "VSCALEFPSZ128rm(b?)(k?)(z?)",
                                              "VSCALEFSDZ128rm(b?)(k?)(z?)",
                                              "VSCALEFSSZ128rm(b?)(k?)(z?)",
                                              "VSUBPDZ128rm(b?)(k?)(z?)",
                                              "VSUBPDrm",
                                              "VSUBPSZ128rm(b?)(k?)(z?)",
                                              "VSUBPSrm",
                                              "VSUBSDZrm(_Int)?(k?)(z?)",
                                              "VSUBSSZrm(_Int)?(k?)(z?)")>;

def SKXWriteResGroup151 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup151], (instregex "MPSADBWrmi",
                                              "VEXPANDPDZ128rm(b?)(k?)(z?)",
                                              "VEXPANDPSZ128rm(b?)(k?)(z?)",
                                              "VMPSADBWrmi",
                                              "VPEXPANDDZ128rm(b?)(k?)(z?)",
                                              "VPEXPANDQZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup152 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup152], (instregex "MMX_CVTPI2PDirm",
                                              "VPTESTYrm")>;

def SKXWriteResGroup153 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup153], (instregex "(V?)CVTSD2SSrm")>;

def SKXWriteResGroup154 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup154], (instregex "VPHADDSWYrm",
                                              "VPHSUBSWYrm")>;

def SKXWriteResGroup155 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup155], (instregex "VPHADDDYrm",
                                              "VPHADDWYrm",
                                              "VPHSUBDYrm",
                                              "VPHSUBWYrm")>;

def SKXWriteResGroup156 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort06,SKXPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup156], (instrs IMUL32m, MUL32m, MULX32rm)>;

def SKXWriteResGroup157 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 10;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,3];
}
def: InstRW<[SKXWriteResGroup157], (instregex "XCHG(8|16|32|64)rm")>;

def SKXWriteResGroup158 : SchedWriteRes<[SKXPort05,SKXPort0156]> {
  let Latency = 10;
  let NumMicroOps = 10;
  let ResourceCycles = [9,1];
}
def: InstRW<[SKXWriteResGroup158], (instregex "MMX_EMMS")>;

def SKXWriteResGroup159 : SchedWriteRes<[SKXPort0]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup159], (instregex "DIVPSrr",
                                              "DIVSSrr",
                                              "VDIVPSYrr",
                                              "VDIVPSZ128rr(b?)(k?)(z?)",
                                              "VDIVPSZ256rr(b?)(k?)(z?)",
                                              "VDIVPSrr",
                                              "VDIVSSZrr(b?)(_Int)?(k?)(z?)",
                                              "VDIVSSrr")>;

def SKXWriteResGroup160 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup160], (instregex "MUL_F32m",
                                              "MUL_F64m",
                                              "VRCP14PDZ256m(b?)(k?)(z?)",
                                              "VRCP14PSZ256m(b?)(k?)(z?)",
                                              "VRCPPSYm",
                                              "VRSQRT14PDZ256m(b?)(k?)(z?)",
                                              "VRSQRT14PSZ256m(b?)(k?)(z?)",
                                              "VRSQRTPSYm")>;

def SKXWriteResGroup161 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup161],
          (instregex
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Ym",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Z256m(b?)(k?)(z?)",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Zm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VADDPDYrm",
                                              "VADDPDZ256rm(b?)(k?)(z?)",
                                              "VADDPDZrm(b?)(k?)(z?)",
                                              "VADDPSYrm",
                                              "VADDPSZ256rm(b?)(k?)(z?)",
                                              "VADDPSZrm(b?)(k?)(z?)",
                                              "VADDSUBPDYrm",
                                              "VADDSUBPSYrm",
                                              "VCMPPDYrmi",
                                              "VCMPPSYrmi",
                                              "VCVTDQ2PDZ256rm(b?)(k?)(z?)",
                                              "VCVTDQ2PDZrm(b?)(k?)(z?)",
                                              "VCVTDQ2PSYrm",
                                              "VCVTDQ2PSZ256rm(b?)(k?)(z?)",
                                              "VCVTDQ2PSZrm(b?)(k?)(z?)",
                                              "VCVTPD2QQZ256rm(b?)(k?)(z?)",
                                              "VCVTPD2QQZrm(b?)(k?)(z?)",
                                              "VCVTPD2UQQZ256rm(b?)(k?)(z?)",
                                              "VCVTPD2UQQZrm(b?)(k?)(z?)",
                                              "VCVTPH2PSZ256rm(b?)(k?)(z?)",
                                              "VCVTPH2PSZrm(b?)(k?)(z?)",
                                              "VCVTPS2DQYrm",
                                              "VCVTPS2DQZ256rm(b?)(k?)(z?)",
                                              "VCVTPS2DQZrm(b?)(k?)(z?)",
                                              "VCVTPS2PDYrm",
                                              "VCVTPS2PDZ256rm(b?)(k?)(z?)",
                                              "VCVTPS2PDZrm(b?)(k?)(z?)",
                                              "VCVTPS2QQZ256rm(b?)(k?)(z?)",
                                              "VCVTPS2UDQZ256rm(b?)(k?)(z?)",
                                              "VCVTPS2UDQZrm(b?)(k?)(z?)",
                                              "VCVTPS2UQQZ256rm(b?)(k?)(z?)",
                                              "VCVTQQ2PDZ256rm(b?)(k?)(z?)",
                                              "VCVTQQ2PDZrm(b?)(k?)(z?)",
                                              "VCVTQQ2PSZ256rm(b?)(k?)(z?)",
                                              "VCVTTPD2QQZ256rm(b?)(k?)(z?)",
                                              "VCVTTPD2QQZrm(b?)(k?)(z?)",
                                              "VCVTTPD2UQQZ256rm(b?)(k?)(z?)",
                                              "VCVTTPD2UQQZrm(b?)(k?)(z?)",
                                              "VCVTTPS2DQYrm",
                                              "VCVTTPS2DQZ256rm(b?)(k?)(z?)",
                                              "VCVTTPS2DQZrm(b?)(k?)(z?)",
                                              "VCVTTPS2QQZ256rm(b?)(k?)(z?)",
                                              "VCVTTPS2UDQZ256rm(b?)(k?)(z?)",
                                              "VCVTTPS2UDQZrm(b?)(k?)(z?)",
                                              "VCVTTPS2UQQZ256rm(b?)(k?)(z?)",
                                              "VCVTUDQ2PDZ256rm(b?)(k?)(z?)",
                                              "VCVTUDQ2PDZrm(b?)(k?)(z?)",
                                              "VCVTUDQ2PSZ256rm(b?)(k?)(z?)",
                                              "VCVTUDQ2PSZrm(b?)(k?)(z?)",
                                              "VCVTUQQ2PDZ256rm(b?)(k?)(z?)",
                                              "VCVTUQQ2PDZrm(b?)(k?)(z?)",
                                              "VCVTUQQ2PSZ256rm(b?)(k?)(z?)",
                                              "VFIXUPIMMPDZ256rm(b?)i(k?)(z?)",
                                              "VFIXUPIMMPDZrm(b?)i(k?)(z?)",
                                              "VFIXUPIMMPSZ256rm(b?)i(k?)(z?)",
                                              "VFIXUPIMMPSZrm(b?)i(k?)(z?)",
                                              "VGETEXPPDZ256m(b?)(k?)(z?)",
                                              "VGETEXPPDm(b?)(k?)(z?)",
                                              "VGETEXPPSZ256m(b?)(k?)(z?)",
                                              "VGETEXPPSm(b?)(k?)(z?)",
                                              "VGETMANTPDZ256rm(b?)i(k?)(z?)",
                                              "VGETMANTPDZrm(b?)i(k?)(z?)",
                                              "VGETMANTPSZ256rm(b?)i(k?)(z?)",
                                              "VGETMANTPSZrm(b?)i(k?)(z?)",
                                              "VMAX(C?)PDYrm",
                                              "VMAX(C?)PDZ256rm(b?)(k?)(z?)",
                                              "VMAX(C?)PDZrm(b?)(k?)(z?)",
                                              "VMAX(C?)PSYrm",
                                              "VMAX(C?)PSZ256rm(b?)(k?)(z?)",
                                              "VMAX(C?)PSZrm(b?)(k?)(z?)",
                                              "VMIN(C?)PDYrm",
                                              "VMIN(C?)PDZ256rm(b?)(k?)(z?)",
                                              "VMIN(C?)PDZrm(b?)(k?)(z?)",
                                              "VMIN(C?)PSYrm",
                                              "VMIN(C?)PSZ256rm(b?)(k?)(z?)",
                                              "VMIN(C?)PSZrm(b?)(k?)(z?)",
                                              "VMULPDYrm",
                                              "VMULPDZ256rm(b?)(k?)(z?)",
                                              "VMULPDZrm(b?)(k?)(z?)",
                                              "VMULPSYrm",
                                              "VMULPSZ256rm(b?)(k?)(z?)",
                                              "VMULPSZrm(b?)(k?)(z?)",
                                              "VPLZCNTDZ256rm(b?)(k?)(z?)",
                                              "VPLZCNTDZrm(b?)(k?)(z?)",
                                              "VPLZCNTQZ256rm(b?)(k?)(z?)",
                                              "VPLZCNTQZrm(b?)(k?)(z?)",
                                              "VPMADDUBSWYrm",
                                              "VPMADDUBSWZ256rm(b?)(k?)(z?)",
                                              "VPMADDUBSWZrm(b?)(k?)(z?)",
                                              "VPMADDWDYrm",
                                              "VPMADDWDZ256rm(b?)(k?)(z?)",
                                              "VPMADDWDZrm(b?)(k?)(z?)",
                                              "VPMULDQYrm",
                                              "VPMULDQZ256rm(b?)(k?)(z?)",
                                              "VPMULDQZrm(b?)(k?)(z?)",
                                              "VPMULHRSWYrm",
                                              "VPMULHRSWZ256rm(b?)(k?)(z?)",
                                              "VPMULHRSWZrm(b?)(k?)(z?)",
                                              "VPMULHUWYrm",
                                              "VPMULHUWZ256rm(b?)(k?)(z?)",
                                              "VPMULHUWZrm(b?)(k?)(z?)",
                                              "VPMULHWYrm",
                                              "VPMULHWZ256rm(b?)(k?)(z?)",
                                              "VPMULHWZrm(b?)(k?)(z?)",
                                              "VPMULLWYrm",
                                              "VPMULLWZ256rm(b?)(k?)(z?)",
                                              "VPMULLWZrm(b?)(k?)(z?)",
                                              "VPMULUDQYrm",
                                              "VPMULUDQZ256rm(b?)(k?)(z?)",
                                              "VPMULUDQZrm(b?)(k?)(z?)",
                                              "VRANGEPDZ256rm(b?)i(k?)(z?)",
                                              "VRANGEPDZrm(b?)i(k?)(z?)",
                                              "VRANGEPSZ256rm(b?)i(k?)(z?)",
                                              "VRANGEPSZrm(b?)i(k?)(z?)",
                                              "VREDUCEPDZ256rm(b?)i(k?)(z?)",
                                              "VREDUCEPDZrm(b?)i(k?)(z?)",
                                              "VREDUCEPSZ256rm(b?)i(k?)(z?)",
                                              "VREDUCEPSZrm(b?)i(k?)(z?)",
                                              "VSCALEFPDZ256rm(b?)(k?)(z?)",
                                              "VSCALEFPDZrm(b?)(k?)(z?)",
                                              "VSCALEFPSZ256rm(b?)(k?)(z?)",
                                              "VSCALEFPSZrm(b?)(k?)(z?)",
                                              "VSUBPDYrm",
                                              "VSUBPDZ256rm(b?)(k?)(z?)",
                                              "VSUBPDZrm(b?)(k?)(z?)",
                                              "VSUBPSYrm",
                                              "VSUBPSZ256rm(b?)(k?)(z?)",
                                              "VSUBPSZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup162 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup162], (instregex "FICOM16m",
                                              "FICOM32m",
                                              "FICOMP16m",
                                              "FICOMP32m",
                                              "VEXPANDPDZ256rm(b?)(k?)(z?)",
                                              "VEXPANDPDZrm(b?)(k?)(z?)",
                                              "VEXPANDPSZ256rm(b?)(k?)(z?)",
                                              "VEXPANDPSZrm(b?)(k?)(z?)",
                                              "VMPSADBWYrmi",
                                              "VPEXPANDDZ256rm(b?)(k?)(z?)",
                                              "VPEXPANDDZrm(b?)(k?)(z?)",
                                              "VPEXPANDQZ256rm(b?)(k?)(z?)",
                                              "VPEXPANDQZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup163 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup163], (instregex "VCVTSD2SSZrm(_Int)?(k?)(z?)")>;

def SKXWriteResGroup164 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup164], (instregex "(V?)CVTDQ2PDrm")>;

def SKXWriteResGroup165 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup165], (instregex "CVTSD2SI64rm",
                                              "CVTSD2SIrm",
                                              "CVTSS2SI64rm",
                                              "CVTSS2SIrm",
                                              "CVTTSD2SI64rm",
                                              "CVTTSD2SIrm",
                                              "CVTTSS2SIrm",
                                              "VCVTSD2SI64Zrm(b?)(k?)(z?)",
                                              "VCVTSD2SI64rm",
                                              "VCVTSD2SIZrm(b?)(k?)(z?)",
                                              "VCVTSD2SIrm",
                                              "VCVTSD2USI64Zrm(b?)(k?)(z?)",
                                              "VCVTSS2SI64Zrm(b?)(k?)(z?)",
                                              "VCVTSS2SI64rm",
                                              "VCVTSS2SIZrm(b?)(k?)(z?)",
                                              "VCVTSS2SIrm",
                                              "VCVTSS2USIZrm(b?)(k?)(z?)",
                                              "VCVTTSD2SI64Zrm(b?)(k?)(z?)",
                                              "VCVTTSD2SI64rm",
                                              "VCVTTSD2SIZrm(b?)(k?)(z?)",
                                              "VCVTTSD2SIrm",
                                              "VCVTTSD2USI64Zrm(b?)(k?)(z?)",
                                              "VCVTTSS2SI64Zrm(b?)(k?)(z?)",
                                              "VCVTTSS2SI64rm",
                                              "VCVTTSS2SIZrm(b?)(k?)(z?)",
                                              "VCVTTSS2SIrm",
                                              "VCVTTSS2USIZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup166 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup166], (instregex "CVTPD2DQrm",
                                              "CVTPD2PSrm",
                                              "CVTTPD2DQrm",
                                              "MMX_CVTPD2PIirm",
                                              "MMX_CVTTPD2PIirm")>;

def SKXWriteResGroup167 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup167], (instregex "VPCONFLICTQZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup168 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 11;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKXWriteResGroup168], (instregex "SHLD(16|32|64)mrCL",
                                              "SHRD(16|32|64)mrCL")>;

def SKXWriteResGroup169 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,3,2];
}
def: InstRW<[SKXWriteResGroup169], (instregex "RCL(16|32|64)rCL",
                                              "RCR(16|32|64)rCL")>;

def SKXWriteResGroup170 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 11;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,1,2];
}
def: InstRW<[SKXWriteResGroup170], (instregex "RCL8rCL")>;

def SKXWriteResGroup171 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 11;
  let NumMicroOps = 11;
  let ResourceCycles = [2,9];
}
def: InstRW<[SKXWriteResGroup171], (instrs LOOPE, LOOPNE)>;

def SKXWriteResGroup172 : SchedWriteRes<[SKXPort0]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup172], (instregex "SQRTPSr",
                                              "SQRTSSr",
                                              "VSQRTPSYr",
                                              "VSQRTPSZ128r(b?)(k?)(z?)",
                                              "VSQRTPSZ256r(b?)(k?)(z?)",
                                              "VSQRTPSr",
                                              "VSQRTSSZr(b?)(_Int)?(k?)(z?)",
                                              "VSQRTSSr")>;

def SKXWriteResGroup174 : SchedWriteRes<[SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKXWriteResGroup174], (instregex "VPMULLQZ128rr(b?)(k?)(z?)",
                                              "VPMULLQZ256rr(b?)(k?)(z?)",
                                              "VPMULLQZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup175 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup175], (instregex "VPERMWZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup176 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup176], (instregex "VCVTSD2USIZrm(b?)(k?)(z?)",
                                              "VCVTSS2USI64Zrm(b?)(k?)(z?)",
                                              "VCVTTSD2USIZrm(b?)(k?)(z?)",
                                              "VCVTTSS2USI64Zrm(b?)(k?)(z?)")>;

def SKXWriteResGroup177 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup177], (instregex "VCVTPS2QQZrm(b?)(k?)(z?)",
                                              "VCVTPS2UQQZrm(b?)(k?)(z?)",
                                              "VCVTTPS2QQZrm(b?)(k?)(z?)",
                                              "VCVTTPS2UQQZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup178 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup178], (instregex "(V?)HADDPDrm",
                                              "(V?)HADDPSrm",
                                              "(V?)HSUBPDrm",
                                              "(V?)HSUBPSrm")>;

def SKXWriteResGroup179 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup179], (instregex "CVTTSS2SI64rm")>;

def SKXWriteResGroup180 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup180], (instregex "ADD_FI16m",
                                              "ADD_FI32m",
                                              "SUBR_FI16m",
                                              "SUBR_FI32m",
                                              "SUB_FI16m",
                                              "SUB_FI32m",
                                              "VPERMWZ256rm(b?)(k?)(z?)",
                                              "VPERMWZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup181 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup181], (instregex "VCVTDQ2PDYrm")>;

def SKXWriteResGroup182 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup182], (instregex "DPPSrri",
                                              "VDPPSYrri",
                                              "VDPPSrri")>;

def SKXWriteResGroup183 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup183], (instregex "VHADDPDYrm",
                                              "VHADDPSYrm",
                                              "VHSUBPDYrm",
                                              "VHSUBPSYrm",
                                              "VPERMI2W128rm(b?)(k?)(z?)",
                                              "VPERMT2W128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup184 : SchedWriteRes<[SKXPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup184], (instregex "DIVPDrr",
                                              "DIVSDrr",
                                              "VDIVPDYrr",
                                              "VDIVPDZ128rr(b?)(k?)(z?)",
                                              "VDIVPDZ256rr(b?)(k?)(z?)",
                                              "VDIVPDrr",
                                              "VDIVSDZrr(b?)(_Int)?(k?)(z?)",
                                              "VDIVSDrr")>;

def SKXWriteResGroup186 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup186], (instregex "ROUNDPDm",
                                              "ROUNDPSm",
                                              "ROUNDSDm",
                                              "ROUNDSSm",
                                              "VRNDSCALEPDZ128rm(b?)i(k?)(z?)",
                                              "VRNDSCALEPSZ128rm(b?)i(k?)(z?)",
                                              "VRNDSCALESDm(b?)(k?)(z?)",
                                              "VRNDSCALESSm(b?)(k?)(z?)",
                                              "VROUNDPDm",
                                              "VROUNDPSm",
                                              "VROUNDSDm",
                                              "VROUNDSSm")>;

def SKXWriteResGroup186_2 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 16;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup186_2], (instregex "PMULLDrm",
                                                "VPMULLDZ128rm(b?)(k?)(z?)",
                                                "VPMULLDrm")>;

def SKXWriteResGroup187 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup187], (instregex "MUL_FI16m",
                                              "MUL_FI32m")>;

def SKXWriteResGroup188 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup188], (instregex "VCVTPD2DQZrm(b?)(k?)(z?)",
                                              "VCVTPD2PSZrm(b?)(k?)(z?)",
                                              "VCVTPD2UDQZrm(b?)(k?)(z?)",
                                              "VCVTQQ2PSZrm(b?)(k?)(z?)",
                                              "VCVTTPD2DQZrm(b?)(k?)(z?)",
                                              "VCVTTPD2UDQZrm(b?)(k?)(z?)",
                                              "VCVTUQQ2PSZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup189 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup189], (instregex "VPERMI2W256rm(b?)(k?)(z?)",
                                              "VPERMI2Wrm(b?)(k?)(z?)",
                                              "VPERMT2W256rm(b?)(k?)(z?)",
                                              "VPERMT2Wrm(b?)(k?)(z?)")>;

def SKXWriteResGroup190 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 14;
  let NumMicroOps = 10;
  let ResourceCycles = [2,4,1,3];
}
def: InstRW<[SKXWriteResGroup190], (instregex "RCR8rCL")>;

def SKXWriteResGroup191 : SchedWriteRes<[SKXPort0]> {
  let Latency = 15;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup191], (instregex "DIVR_FPrST0",
                                              "DIVR_FST0r",
                                              "DIVR_FrST0")>;

def SKXWriteResGroup192 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 15;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup192], (instregex "VRNDSCALEPDZ256rm(b?)i(k?)(z?)",
                                              "VRNDSCALEPDZrm(b?)i(k?)(z?)",
                                              "VRNDSCALEPSZ256rm(b?)i(k?)(z?)",
                                              "VRNDSCALEPSZrm(b?)i(k?)(z?)",
                                              "VROUNDPDYm",
                                              "VROUNDPSYm")>;

def SKXWriteResGroup192_2 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 17;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup192_2], (instregex "VPMULLDYrm")>;
def: InstRW<[SKXWriteResGroup192_2], (instregex "VPMULLDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup192_2], (instregex "VPMULLDZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup193 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 15;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKXWriteResGroup193], (instregex "(V?)DPPDrmi")>;

def SKXWriteResGroup194 : SchedWriteRes<[SKXPort1,SKXPort5,SKXPort01,SKXPort23,SKXPort015]> {
  let Latency = 15;
  let NumMicroOps = 8;
  let ResourceCycles = [1,2,2,1,2];
}
def: InstRW<[SKXWriteResGroup194], (instregex "VPCONFLICTDZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup195 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 15;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,5,1,1];
}
def: InstRW<[SKXWriteResGroup195], (instregex "RCL(8|16|32|64)mCL")>;

def SKXWriteResGroup196 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup196], (instregex "(V?)DIVSSrm")>;

def SKXWriteResGroup198 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup198], (instregex "VRCP14PDZm(b?)(k?)(z?)",
                                              "VRCP14PSZm(b?)(k?)(z?)",
                                              "VRSQRT14PDZm(b?)(k?)(z?)",
                                              "VRSQRT14PSZm(b?)(k?)(z?)")>;

def SKXWriteResGroup199 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 16;
  let NumMicroOps = 14;
  let ResourceCycles = [1,1,1,4,2,5];
}
def: InstRW<[SKXWriteResGroup199], (instregex "CMPXCHG8B")>;

def SKXWriteResGroup200 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 16;
  let NumMicroOps = 16;
  let ResourceCycles = [16];
}
def: InstRW<[SKXWriteResGroup200], (instregex "VZEROALL")>;

def SKXWriteResGroup201 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 17;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup201], (instregex "DIVPSrm",
                                              "SQRTSSm",
                                              "VDIVPSZ128rm(b?)(k?)(z?)",
                                              "VDIVPSrm",
                                              "VDIVSSZrm(_Int)?(k?)(z?)",
                                              "VSQRTSSm")>;

def SKXWriteResGroup202 : SchedWriteRes<[SKXPort0,SKXPort1,SKXPort5,SKXPort6,SKXPort05,SKXPort0156]> {
  let Latency = 17;
  let NumMicroOps = 15;
  let ResourceCycles = [2,1,2,4,2,4];
}
def: InstRW<[SKXWriteResGroup202], (instregex "XCH_F")>;

def SKXWriteResGroup203 : SchedWriteRes<[SKXPort0]> {
  let Latency = 18;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup203], (instregex "SQRTPDr",
                                              "SQRTSDr",
                                              "VSQRTPDYr",
                                              "VSQRTPDZ128r(b?)(k?)(z?)",
                                              "VSQRTPDZ256r(b?)(k?)(z?)",
                                              "VSQRTPDr",
                                              "VSQRTSDZr(b?)(_Int)?(k?)(z?)",
                                              "VSQRTSDr")>;

def SKXWriteResGroup204 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup204], (instregex "SQRTPSm",
                                              "VDIVPSYrm",
                                              "VDIVPSZ256rm(b?)(k?)(z?)",
                                              "VSQRTPSZ128m(b?)(k?)(z?)",
                                              "VSQRTPSm",
                                              "VSQRTSSZm(_Int)?(k?)(z?)")>;

def SKXWriteResGroup205 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 18;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup205], (instregex "VPMULLQZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup207 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort06,SKXPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[SKXWriteResGroup207], (instrs CPUID, RDTSC)>;

def SKXWriteResGroup208 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 18;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,4,1,2];
}
def: InstRW<[SKXWriteResGroup208], (instregex "RCR(8|16|32|64)mCL")>;

def SKXWriteResGroup209 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup209], (instregex "DIVSDrm",
                                              "VDIVSDrm",
                                              "VSQRTPSYm",
                                              "VSQRTPSZ256m(b?)(k?)(z?)")>;

def SKXWriteResGroup210 : SchedWriteRes<[SKXPort0,SKXPort015]> {
  let Latency = 19;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup210], (instregex "VSQRTPSZr(b?)(k?)(z?)")>;

def SKXWriteResGroup211 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 19;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup211], (instregex "VPMULLQZ256rm(b?)(k?)(z?)",
                                              "VPMULLQZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup212 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 19;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup212], (instregex "(V?)DPPSrmi")>;

def SKXWriteResGroup214 : SchedWriteRes<[]> {
  let Latency = 20;
  let NumMicroOps = 0;
}
def: InstRW<[SKXWriteResGroup214], (instrs VGATHERDPSZ128rm,
                                           VGATHERQPSZrm,
                                           VPGATHERDDZ128rm)>;

def SKXWriteResGroup215 : SchedWriteRes<[SKXPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup215], (instregex "DIV_FPrST0",
                                              "DIV_FST0r",
                                              "DIV_FrST0")>;

def SKXWriteResGroup216 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup216], (instregex "(V?)DIVPDrm",
                                              "VDIVPDZ128rm(b?)(k?)(z?)",
                                              "VDIVSDZrm(_Int)?(k?)(z?)")>;

def SKXWriteResGroup217 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 20;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup217], (instregex "VDPPSYrmi")>;

def SKXWriteResGroup218 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 20;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup218], (instrs VGATHERQPSZ128rm,
                                           VGATHERQPSZ256rm,
                                           VPGATHERQDZ128rm,
                                           VPGATHERQDZ256rm)>;

def SKXWriteResGroup219 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort6,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 20;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,1,2];
}
def: InstRW<[SKXWriteResGroup219], (instregex "INSB",
                                              "INSL",
                                              "INSW")>;

def SKXWriteResGroup220 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort0156]> {
  let Latency = 20;
  let NumMicroOps = 10;
  let ResourceCycles = [1,2,7];
}
def: InstRW<[SKXWriteResGroup220], (instregex "MWAITrr")>;

def SKXWriteResGroup222 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup222], (instregex "VDIVPDYrm",
                                              "VDIVPDZ256rm(b?)(k?)(z?)")>;

def SKXWriteResGroup223 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 22;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup223], (instregex "DIV_F32m",
                                              "DIV_F64m")>;

def SKXWriteResGroup224 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 22;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup224], (instrs VGATHERDPDZ128rm,
                                           VGATHERQPDZ128rm,
                                           VPGATHERDQZ128rm,
                                           VPGATHERQQZ128rm)>;

def SKXWriteResGroup224_2 : SchedWriteRes<[SKXPort0, SKXPort23, SKXPort5, SKXPort015]> {
  let Latency = 22;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup224_2], (instrs VGATHERDPSrm,
                                             VGATHERDPDrm,
                                             VGATHERQPDrm,
                                             VGATHERQPSrm,
                                             VPGATHERDDrm,
                                             VPGATHERDQrm,
                                             VPGATHERQDrm,
                                             VPGATHERQQrm,
                                             VPGATHERDDrm,
                                             VPGATHERQDrm,
                                             VPGATHERDQrm,
                                             VPGATHERQQrm,
                                             VGATHERDPSrm,
                                             VGATHERQPSrm,
                                             VGATHERDPDrm,
                                             VGATHERQPDrm)>;

def SKXWriteResGroup224_3 : SchedWriteRes<[SKXPort0, SKXPort23, SKXPort5, SKXPort015]> {
  let Latency = 25;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup224_3], (instrs VGATHERDPSYrm,
                                             VGATHERQPDYrm,
                                             VGATHERQPSYrm,
                                             VPGATHERDDYrm,
                                             VPGATHERDQYrm,
                                             VPGATHERQDYrm,
                                             VPGATHERQQYrm,
                                             VPGATHERDDYrm,
                                             VPGATHERQDYrm,
                                             VPGATHERDQYrm,
                                             VPGATHERQQYrm,
                                             VGATHERDPSYrm,
                                             VGATHERQPSYrm,
                                             VGATHERDPDYrm)>;

def SKXWriteResGroup225 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort015]> {
  let Latency = 22;
  let NumMicroOps = 14;
  let ResourceCycles = [5,5,4];
}
def: InstRW<[SKXWriteResGroup225], (instregex "VPCONFLICTDZ128rr(b?)(k?)(z?)",
                                              "VPCONFLICTQZ256rr(b?)(k?)(z?)")>;

def SKXWriteResGroup226 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 23;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup226], (instregex "(V?)SQRTSDm")>;

def SKXWriteResGroup227 : SchedWriteRes<[SKXPort0,SKXPort015]> {
  let Latency = 23;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup227], (instregex "VDIVPDZrr(b?)(k?)(z?)",
                                              "VDIVPSZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup228 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 23;
  let NumMicroOps = 19;
  let ResourceCycles = [2,1,4,1,1,4,6];
}
def: InstRW<[SKXWriteResGroup228], (instregex "CMPXCHG16B")>;

def SKXWriteResGroup229 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 24;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup229], (instregex "SQRTPDm",
                                              "VSQRTPDZ128m(b?)(k?)(z?)",
                                              "VSQRTPDm",
                                              "VSQRTSDZm(_Int)?(k?)(z?)")>;

def SKXWriteResGroup230 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 24;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup230], (instregex "VDIVPSZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup232 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup232], (instregex "VSQRTPDYm",
                                              "VSQRTPDZ256m(b?)(k?)(z?)")>;

def SKXWriteResGroup233 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 25;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup233], (instregex "DIV_FI16m",
                                              "DIV_FI32m")>;

def SKXWriteResGroup234 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 25;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup234], (instrs VGATHERDPDZ256rm,
                                           VGATHERQPDZ256rm,
                                           VPGATHERDQZ256rm,
                                           VPGATHERQDZrm,
                                           VPGATHERQQZ256rm)>;

def SKXWriteResGroup237 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 26;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup237], (instregex "VSQRTPSZm(b?)(k?)(z?)")>;

def SKXWriteResGroup238 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 26;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup238], (instrs VGATHERDPDZrm,
                                           VGATHERQPDZrm,
                                           VPGATHERDQZrm,
                                           VPGATHERQQZrm)>;

def SKXWriteResGroup239 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 27;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup239], (instregex "DIVR_F32m",
                                              "DIVR_F64m")>;

def SKXWriteResGroup240 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 27;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup240], (instrs VGATHERDPSZ256rm,
                                           VPGATHERDDZ256rm)>;

def SKXWriteResGroup241 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23,SKXPort0156]> {
  let Latency = 28;
  let NumMicroOps = 8;
  let ResourceCycles = [2,4,1,1];
}
def: InstRW<[SKXWriteResGroup241], (instregex "IDIV(8|16|32|64)m")>;

def SKXWriteResGroup242 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23,SKXPort015]> {
  let Latency = 29;
  let NumMicroOps = 15;
  let ResourceCycles = [5,5,1,4];
}
def: InstRW<[SKXWriteResGroup242], (instregex "VPCONFLICTQZ256rm(b?)(k?)(z?)")>;

def SKXWriteResGroup243 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 30;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup243], (instregex "DIVR_FI16m",
                                              "DIVR_FI32m")>;

def SKXWriteResGroup244 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 30;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup244], (instregex "VDIVPDZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup245 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 30;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup245], (instrs VGATHERDPSZrm,
                                           VPGATHERDDZrm)>;

def SKXWriteResGroup246 : SchedWriteRes<[SKXPort0,SKXPort015]> {
  let Latency = 31;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup246], (instregex "VSQRTPDZr(b?)(k?)(z?)")>;

def SKXWriteResGroup247 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort23,SKXPort06,SKXPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[SKXWriteResGroup247], (instregex "IN(8|16|32)ri",
                                              "IN(8|16|32)rr")>;

def SKXWriteResGroup248 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[SKXWriteResGroup248], (instregex "OUT(8|16|32)ir",
                                              "OUT(8|16|32)rr")>;

def SKXWriteResGroup249 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort015]> {
  let Latency = 37;
  let NumMicroOps = 21;
  let ResourceCycles = [9,7,5];
}
def: InstRW<[SKXWriteResGroup249], (instregex "VPCONFLICTDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup249], (instregex "VPCONFLICTQZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup250 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort23,SKXPort0156]> {
  let Latency = 37;
  let NumMicroOps = 31;
  let ResourceCycles = [1,8,1,21];
}
def: InstRW<[SKXWriteResGroup250], (instregex "XRSTOR(64)?")>;

def SKXWriteResGroup251 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 38;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup251], (instregex "VSQRTPDZm(b?)(k?)(z?)")>;

def SKXWriteResGroup252 : SchedWriteRes<[SKXPort1,SKXPort4,SKXPort5,SKXPort6,SKXPort23,SKXPort237,SKXPort15,SKXPort0156]> {
  let Latency = 40;
  let NumMicroOps = 18;
  let ResourceCycles = [1,1,2,3,1,1,1,8];
}
def: InstRW<[SKXWriteResGroup252], (instregex "VMCLEARm")>;

def SKXWriteResGroup253 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 41;
  let NumMicroOps = 39;
  let ResourceCycles = [1,10,1,1,26];
}
def: InstRW<[SKXWriteResGroup253], (instregex "XSAVE64")>;

def SKXWriteResGroup254 : SchedWriteRes<[SKXPort5,SKXPort0156]> {
  let Latency = 42;
  let NumMicroOps = 22;
  let ResourceCycles = [2,20];
}
def: InstRW<[SKXWriteResGroup254], (instrs RDTSCP)>;

def SKXWriteResGroup255 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 42;
  let NumMicroOps = 40;
  let ResourceCycles = [1,11,1,1,26];
}
def: InstRW<[SKXWriteResGroup255], (instregex "^XSAVE$", "XSAVEC", "XSAVES", "XSAVEOPT")>;

def SKXWriteResGroup256 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23,SKXPort015]> {
  let Latency = 44;
  let NumMicroOps = 22;
  let ResourceCycles = [9,7,1,5];
}
def: InstRW<[SKXWriteResGroup256], (instregex "VPCONFLICTDZ256rm(b?)(k?)(z?)",
                                              "VPCONFLICTQZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup258 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort05,SKXPort06,SKXPort0156]> {
  let Latency = 62;
  let NumMicroOps = 64;
  let ResourceCycles = [2,8,5,10,39];
}
def: InstRW<[SKXWriteResGroup258], (instregex "FLDENVm")>;

def SKXWriteResGroup259 : SchedWriteRes<[SKXPort0,SKXPort6,SKXPort23,SKXPort05,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 63;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[SKXWriteResGroup259], (instrs FXRSTOR64)>;

def SKXWriteResGroup260 : SchedWriteRes<[SKXPort0,SKXPort6,SKXPort23,SKXPort05,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 63;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[SKXWriteResGroup260], (instrs FXRSTOR)>;

def SKXWriteResGroup261 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort015]> {
  let Latency = 67;
  let NumMicroOps = 35;
  let ResourceCycles = [17,11,7];
}
def: InstRW<[SKXWriteResGroup261], (instregex "VPCONFLICTDZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup262 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23,SKXPort015]> {
  let Latency = 74;
  let NumMicroOps = 36;
  let ResourceCycles = [17,11,1,7];
}
def: InstRW<[SKXWriteResGroup262], (instregex "VPCONFLICTDZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup263 : SchedWriteRes<[SKXPort5,SKXPort05,SKXPort0156]> {
  let Latency = 75;
  let NumMicroOps = 15;
  let ResourceCycles = [6,3,6];
}
def: InstRW<[SKXWriteResGroup263], (instregex "FNINIT")>;

def SKXWriteResGroup264 : SchedWriteRes<[SKXPort0,SKXPort1,SKXPort5,SKXPort6,SKXPort05,SKXPort0156]> {
  let Latency = 76;
  let NumMicroOps = 32;
  let ResourceCycles = [7,2,8,3,1,11];
}
def: InstRW<[SKXWriteResGroup264], (instregex "DIV(16|32|64)r")>;

def SKXWriteResGroup265 : SchedWriteRes<[SKXPort0,SKXPort1,SKXPort5,SKXPort6,SKXPort06,SKXPort0156]> {
  let Latency = 102;
  let NumMicroOps = 66;
  let ResourceCycles = [4,2,4,8,14,34];
}
def: InstRW<[SKXWriteResGroup265], (instregex "IDIV(16|32|64)r")>;

def SKXWriteResGroup266 : SchedWriteRes<[SKXPort0,SKXPort1,SKXPort4,SKXPort5,SKXPort6,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 106;
  let NumMicroOps = 100;
  let ResourceCycles = [9,1,11,16,1,11,21,30];
}
def: InstRW<[SKXWriteResGroup266], (instregex "FSTENVm")>;

def SKXWriteResGroup267 : SchedWriteRes<[SKXPort6,SKXPort0156]> {
  let Latency = 140;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup267], (instregex "PAUSE")>;
} // SchedModel
