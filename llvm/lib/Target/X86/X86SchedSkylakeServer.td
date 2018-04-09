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

def SKXDivider : ProcResource<1>; // Integer division issued on port 0.
// FP division and sqrt on port 0.
def SKXFPDivider : ProcResource<1>;

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
  def : WriteRes<SchedRW.Folded, !listconcat([SKXPort23], ExePorts)> {
    let Latency = !add(Lat, LoadLat);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = !add(UOps, 1);
  }
}

// A folded store needs a cycle on port 4 for the store data, and an extra port
// 2/3/7 cycle to recompute the address.
def : WriteRes<WriteRMW, [SKXPort237,SKXPort4]>;

// Arithmetic.
defm : SKXWriteResPair<WriteALU,   [SKXPort0156], 1>; // Simple integer ALU op.
defm : SKXWriteResPair<WriteIMul,  [SKXPort1],   3>; // Integer multiplication.
defm : SKXWriteResPair<WriteIDiv, [SKXPort0, SKXDivider], 25, [1,10], 1, 4>; // Integer division.
defm : SKXWriteResPair<WriteCRC32, [SKXPort1], 3>;

def : WriteRes<WriteIMulH, []> { let Latency = 3; } // Integer multiplication, high part.
def : WriteRes<WriteLEA, [SKXPort15]>; // LEA instructions can't fold loads.

defm : SKXWriteResPair<WriteCMOV,  [SKXPort06], 1>; // Conditional move.
def  : WriteRes<WriteSETCC, [SKXPort06]>; // Setcc.
def  : WriteRes<WriteSETCCStore, [SKXPort06,SKXPort4,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

// Integer shifts and rotates.
defm : SKXWriteResPair<WriteShift, [SKXPort06],  1>;

// Bit counts.
defm : SKXWriteResPair<WriteBitScan, [SKXPort1], 3>;
defm : SKXWriteResPair<WriteLZCNT,   [SKXPort1], 3>;
defm : SKXWriteResPair<WriteTZCNT,   [SKXPort1], 3>;
defm : SKXWriteResPair<WritePOPCNT,  [SKXPort1], 3>;

// BMI1 BEXTR, BMI2 BZHI
defm : SKXWriteResPair<WriteBEXTR, [SKXPort06,SKXPort15], 2, [1,1], 2>;
defm : SKXWriteResPair<WriteBZHI, [SKXPort15], 1>;

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
defm : SKXWriteResPair<WritePMULLD,   [SKXPort015], 10, [2], 2, 6>; // Vector integer multiply.
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

// MOVMSK Instructions.
def : WriteRes<WriteFMOVMSK, [SKXPort0]> { let Latency = 2; }
def : WriteRes<WriteVecMOVMSK, [SKXPort0]> { let Latency = 2; }
def : WriteRes<WriteMMXMOVMSK, [SKXPort0]> { let Latency = 2; }

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
                                            "VPMOVB2MZ128rr",
                                            "VPMOVB2MZ256rr",
                                            "VPMOVB2MZrr",
                                            "VPMOVD2MZ128rr",
                                            "VPMOVD2MZ256rr",
                                            "VPMOVD2MZrr",
                                            "VPMOVQ2MZ128rr",
                                            "VPMOVQ2MZ256rr",
                                            "VPMOVQ2MZrr",
                                            "VPMOVW2MZ128rr",
                                            "VPMOVW2MZ256rr",
                                            "VPMOVW2MZrr")>;

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
                                            "MOVSSrr",
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
                                            "VBROADCASTI32X2Z128r",
                                            "VBROADCASTSSrr",
                                            "VINSERTPSZrr",
                                            "VINSERTPSrr",
                                            "VMOV64toPQIZrr",
                                            "VMOV64toPQIrr",
                                            "VMOVDDUPYrr",
                                            "VMOVDDUPZ128rr",
                                            "VMOVDDUPZ256rr",
                                            "VMOVDDUPZrr",
                                            "VMOVDDUPrr",
                                            "VMOVDI2PDIZrr",
                                            "VMOVDI2PDIrr",
                                            "VMOVHLPSZrr",
                                            "VMOVHLPSrr",
                                            "VMOVLHPSZrr",
                                            "VMOVLHPSrr",
                                            "VMOVSDZrr",
                                            "VMOVSDrr",
                                            "VMOVSHDUPYrr",
                                            "VMOVSHDUPZ128rr",
                                            "VMOVSHDUPZ256rr",
                                            "VMOVSHDUPZrr",
                                            "VMOVSHDUPrr",
                                            "VMOVSLDUPYrr",
                                            "VMOVSLDUPZ128rr",
                                            "VMOVSLDUPZ256rr",
                                            "VMOVSLDUPZrr",
                                            "VMOVSLDUPrr",
                                            "VMOVSSZrr",
                                            "VMOVSSrr",
                                            "VPACKSSDWYrr",
                                            "VPACKSSDWZ128rr",
                                            "VPACKSSDWZ256rr",
                                            "VPACKSSDWZrr",
                                            "VPACKSSDWrr",
                                            "VPACKSSWBYrr",
                                            "VPACKSSWBZ128rr",
                                            "VPACKSSWBZ256rr",
                                            "VPACKSSWBZrr",
                                            "VPACKSSWBrr",
                                            "VPACKUSDWYrr",
                                            "VPACKUSDWZ128rr",
                                            "VPACKUSDWZ256rr",
                                            "VPACKUSDWZrr",
                                            "VPACKUSDWrr",
                                            "VPACKUSWBYrr",
                                            "VPACKUSWBZ128rr",
                                            "VPACKUSWBZ256rr",
                                            "VPACKUSWBZrr",
                                            "VPACKUSWBrr",
                                            "VPALIGNRYrri",
                                            "VPALIGNRZ128rri",
                                            "VPALIGNRZ256rri",
                                            "VPALIGNRZrri",
                                            "VPALIGNRrri",
                                            "VPBLENDWYrri",
                                            "VPBLENDWrri",
                                            "VPBROADCASTDrr",
                                            "VPBROADCASTQrr",
                                            "VPERMILPDYri",
                                            "VPERMILPDYrr",
                                            "VPERMILPDZ128ri",
                                            "VPERMILPDZ128rr",
                                            "VPERMILPDZ256ri",
                                            "VPERMILPDZ256rr",
                                            "VPERMILPDZri",
                                            "VPERMILPDZrr",
                                            "VPERMILPDri",
                                            "VPERMILPDrr",
                                            "VPERMILPSYri",
                                            "VPERMILPSYrr",
                                            "VPERMILPSZ128ri",
                                            "VPERMILPSZ128rr",
                                            "VPERMILPSZ256ri",
                                            "VPERMILPSZ256rr",
                                            "VPERMILPSZri",
                                            "VPERMILPSZrr",
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
                                            "VPSHUFBZ128rr",
                                            "VPSHUFBZ256rr",
                                            "VPSHUFBZrr",
                                            "VPSHUFBrr",
                                            "VPSHUFDYri",
                                            "VPSHUFDZ128ri",
                                            "VPSHUFDZ256ri",
                                            "VPSHUFDZri",
                                            "VPSHUFDri",
                                            "VPSHUFHWYri",
                                            "VPSHUFHWZ128ri",
                                            "VPSHUFHWZ256ri",
                                            "VPSHUFHWZri",
                                            "VPSHUFHWri",
                                            "VPSHUFLWYri",
                                            "VPSHUFLWZ128ri",
                                            "VPSHUFLWZ256ri",
                                            "VPSHUFLWZri",
                                            "VPSHUFLWri",
                                            "VPSLLDQYri",
                                            "VPSLLDQZ128rr",
                                            "VPSLLDQZ256rr",
                                            "VPSLLDQZrr",
                                            "VPSLLDQri",
                                            "VPSRLDQYri",
                                            "VPSRLDQZ128rr",
                                            "VPSRLDQZ256rr",
                                            "VPSRLDQZrr",
                                            "VPSRLDQri",
                                            "VPUNPCKHBWYrr",
                                            "VPUNPCKHBWZ128rr",
                                            "VPUNPCKHBWZ256rr",
                                            "VPUNPCKHBWZrr",
                                            "VPUNPCKHBWrr",
                                            "VPUNPCKHDQYrr",
                                            "VPUNPCKHDQZ128rr",
                                            "VPUNPCKHDQZ256rr",
                                            "VPUNPCKHDQZrr",
                                            "VPUNPCKHDQrr",
                                            "VPUNPCKHQDQYrr",
                                            "VPUNPCKHQDQZ128rr",
                                            "VPUNPCKHQDQZ256rr",
                                            "VPUNPCKHQDQZrr",
                                            "VPUNPCKHQDQrr",
                                            "VPUNPCKHWDYrr",
                                            "VPUNPCKHWDZ128rr",
                                            "VPUNPCKHWDZ256rr",
                                            "VPUNPCKHWDZrr",
                                            "VPUNPCKHWDrr",
                                            "VPUNPCKLBWYrr",
                                            "VPUNPCKLBWZ128rr",
                                            "VPUNPCKLBWZ256rr",
                                            "VPUNPCKLBWZrr",
                                            "VPUNPCKLBWrr",
                                            "VPUNPCKLDQYrr",
                                            "VPUNPCKLDQZ128rr",
                                            "VPUNPCKLDQZ256rr",
                                            "VPUNPCKLDQZrr",
                                            "VPUNPCKLDQrr",
                                            "VPUNPCKLQDQYrr",
                                            "VPUNPCKLQDQZ128rr",
                                            "VPUNPCKLQDQZ256rr",
                                            "VPUNPCKLQDQZrr",
                                            "VPUNPCKLQDQrr",
                                            "VPUNPCKLWDYrr",
                                            "VPUNPCKLWDZ128rr",
                                            "VPUNPCKLWDZ256rr",
                                            "VPUNPCKLWDZrr",
                                            "VPUNPCKLWDrr",
                                            "VSHUFPDYrri",
                                            "VSHUFPDZ128rri",
                                            "VSHUFPDZ256rri",
                                            "VSHUFPDZrri",
                                            "VSHUFPDrri",
                                            "VSHUFPSYrri",
                                            "VSHUFPSZ128rri",
                                            "VSHUFPSZ256rri",
                                            "VSHUFPSZrri",
                                            "VSHUFPSrri",
                                            "VUNPCKHPDYrr",
                                            "VUNPCKHPDZ128rr",
                                            "VUNPCKHPDZ256rr",
                                            "VUNPCKHPDZrr",
                                            "VUNPCKHPDrr",
                                            "VUNPCKHPSYrr",
                                            "VUNPCKHPSZ128rr",
                                            "VUNPCKHPSZ256rr",
                                            "VUNPCKHPSZrr",
                                            "VUNPCKHPSrr",
                                            "VUNPCKLPDYrr",
                                            "VUNPCKLPDZ128rr",
                                            "VUNPCKLPDZ256rr",
                                            "VUNPCKLPDZrr",
                                            "VUNPCKLPDrr",
                                            "VUNPCKLPSYrr",
                                            "VUNPCKLPSZ128rr",
                                            "VUNPCKLPSZ256rr",
                                            "VUNPCKLPSZrr",
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
                                            "VPABSBZ128rr",
                                            "VPABSBZ256rr",
                                            "VPABSBZrr",
                                            "VPABSBrr",
                                            "VPABSDYrr",
                                            "VPABSDZ128rr",
                                            "VPABSDZ256rr",
                                            "VPABSDZrr",
                                            "VPABSDrr",
                                            "VPABSQZ128rr",
                                            "VPABSQZ256rr",
                                            "VPABSQZrr",
                                            "VPABSWYrr",
                                            "VPABSWZ128rr",
                                            "VPABSWZ256rr",
                                            "VPABSWZrr",
                                            "VPABSWrr",
                                            "VPADDSBYrr",
                                            "VPADDSBZ128rr",
                                            "VPADDSBZ256rr",
                                            "VPADDSBZrr",
                                            "VPADDSBrr",
                                            "VPADDSWYrr",
                                            "VPADDSWZ128rr",
                                            "VPADDSWZ256rr",
                                            "VPADDSWZrr",
                                            "VPADDSWrr",
                                            "VPADDUSBYrr",
                                            "VPADDUSBZ128rr",
                                            "VPADDUSBZ256rr",
                                            "VPADDUSBZrr",
                                            "VPADDUSBrr",
                                            "VPADDUSWYrr",
                                            "VPADDUSWZ128rr",
                                            "VPADDUSWZ256rr",
                                            "VPADDUSWZrr",
                                            "VPADDUSWrr",
                                            "VPAVGBYrr",
                                            "VPAVGBZ128rr",
                                            "VPAVGBZ256rr",
                                            "VPAVGBZrr",
                                            "VPAVGBrr",
                                            "VPAVGWYrr",
                                            "VPAVGWZ128rr",
                                            "VPAVGWZ256rr",
                                            "VPAVGWZrr",
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
                                            "VPMAXSBZ128rr",
                                            "VPMAXSBZ256rr",
                                            "VPMAXSBZrr",
                                            "VPMAXSBrr",
                                            "VPMAX(C?)SDYrr",
                                            "VPMAX(C?)SDZ128rr",
                                            "VPMAX(C?)SDZ256rr",
                                            "VPMAX(C?)SDZrr",
                                            "VPMAX(C?)SDrr",
                                            "VPMAXSWYrr",
                                            "VPMAXSWZ128rr",
                                            "VPMAXSWZ256rr",
                                            "VPMAXSWZrr",
                                            "VPMAXSWrr",
                                            "VPMAXUBYrr",
                                            "VPMAXUBZ128rr",
                                            "VPMAXUBZ256rr",
                                            "VPMAXUBZrr",
                                            "VPMAXUBrr",
                                            "VPMAXUDYrr",
                                            "VPMAXUDZ128rr",
                                            "VPMAXUDZ256rr",
                                            "VPMAXUDZrr",
                                            "VPMAXUDrr",
                                            "VPMAXUWYrr",
                                            "VPMAXUWZ128rr",
                                            "VPMAXUWZ256rr",
                                            "VPMAXUWZrr",
                                            "VPMAXUWrr",
                                            "VPMINSBYrr",
                                            "VPMINSBZ128rr",
                                            "VPMINSBZ256rr",
                                            "VPMINSBZrr",
                                            "VPMINSBrr",
                                            "VPMIN(C?)SDYrr",
                                            "VPMIN(C?)SDZ128rr",
                                            "VPMIN(C?)SDZ256rr",
                                            "VPMIN(C?)SDZrr",
                                            "VPMIN(C?)SDrr",
                                            "VPMINSWYrr",
                                            "VPMINSWZ128rr",
                                            "VPMINSWZ256rr",
                                            "VPMINSWZrr",
                                            "VPMINSWrr",
                                            "VPMINUBYrr",
                                            "VPMINUBZ128rr",
                                            "VPMINUBZ256rr",
                                            "VPMINUBZrr",
                                            "VPMINUBrr",
                                            "VPMINUDYrr",
                                            "VPMINUDZ128rr",
                                            "VPMINUDZ256rr",
                                            "VPMINUDZrr",
                                            "VPMINUDrr",
                                            "VPMINUWYrr",
                                            "VPMINUWZ128rr",
                                            "VPMINUWZ256rr",
                                            "VPMINUWZrr",
                                            "VPMINUWrr",
                                            "VPROLDZ128ri",
                                            "VPROLDZ256ri",
                                            "VPROLDZri",
                                            "VPROLQZ128ri",
                                            "VPROLQZ256ri",
                                            "VPROLQZri",
                                            "VPROLVDZ128rr",
                                            "VPROLVDZ256rr",
                                            "VPROLVDZrr",
                                            "VPROLVQZ128rr",
                                            "VPROLVQZ256rr",
                                            "VPROLVQZrr",
                                            "VPRORDZ128ri",
                                            "VPRORDZ256ri",
                                            "VPRORDZri",
                                            "VPRORQZ128ri",
                                            "VPRORQZ256ri",
                                            "VPRORQZri",
                                            "VPRORVDZ128rr",
                                            "VPRORVDZ256rr",
                                            "VPRORVDZrr",
                                            "VPRORVQZ128rr",
                                            "VPRORVQZ256rr",
                                            "VPRORVQZrr",
                                            "VPSIGNBYrr",
                                            "VPSIGNBrr",
                                            "VPSIGNDYrr",
                                            "VPSIGNDrr",
                                            "VPSIGNWYrr",
                                            "VPSIGNWrr",
                                            "VPSLLDYri",
                                            "VPSLLDZ128ri",
                                            "VPSLLDZ256ri",
                                            "VPSLLDZri",
                                            "VPSLLDri",
                                            "VPSLLQYri",
                                            "VPSLLQZ128ri",
                                            "VPSLLQZ256ri",
                                            "VPSLLQZri",
                                            "VPSLLQri",
                                            "VPSLLVDYrr",
                                            "VPSLLVDZ128rr",
                                            "VPSLLVDZ256rr",
                                            "VPSLLVDZrr",
                                            "VPSLLVDrr",
                                            "VPSLLVQYrr",
                                            "VPSLLVQZ128rr",
                                            "VPSLLVQZ256rr",
                                            "VPSLLVQZrr",
                                            "VPSLLVQrr",
                                            "VPSLLVWZ128rr",
                                            "VPSLLVWZ256rr",
                                            "VPSLLVWZrr",
                                            "VPSLLWYri",
                                            "VPSLLWZ128ri",
                                            "VPSLLWZ256ri",
                                            "VPSLLWZri",
                                            "VPSLLWri",
                                            "VPSRADYri",
                                            "VPSRADZ128ri",
                                            "VPSRADZ256ri",
                                            "VPSRADZri",
                                            "VPSRADri",
                                            "VPSRAQZ128ri",
                                            "VPSRAQZ256ri",
                                            "VPSRAQZri",
                                            "VPSRAVDYrr",
                                            "VPSRAVDZ128rr",
                                            "VPSRAVDZ256rr",
                                            "VPSRAVDZrr",
                                            "VPSRAVDrr",
                                            "VPSRAVQZ128rr",
                                            "VPSRAVQZ256rr",
                                            "VPSRAVQZrr",
                                            "VPSRAVWZ128rr",
                                            "VPSRAVWZ256rr",
                                            "VPSRAVWZrr",
                                            "VPSRAWYri",
                                            "VPSRAWZ128ri",
                                            "VPSRAWZ256ri",
                                            "VPSRAWZri",
                                            "VPSRAWri",
                                            "VPSRLDYri",
                                            "VPSRLDZ128ri",
                                            "VPSRLDZ256ri",
                                            "VPSRLDZri",
                                            "VPSRLDri",
                                            "VPSRLQYri",
                                            "VPSRLQZ128ri",
                                            "VPSRLQZ256ri",
                                            "VPSRLQZri",
                                            "VPSRLQri",
                                            "VPSRLVDYrr",
                                            "VPSRLVDZ128rr",
                                            "VPSRLVDZ256rr",
                                            "VPSRLVDZrr",
                                            "VPSRLVDrr",
                                            "VPSRLVQYrr",
                                            "VPSRLVQZ128rr",
                                            "VPSRLVQZ256rr",
                                            "VPSRLVQZrr",
                                            "VPSRLVQrr",
                                            "VPSRLVWZ128rr",
                                            "VPSRLVWZ256rr",
                                            "VPSRLVWZrr",
                                            "VPSRLWYri",
                                            "VPSRLWZ128ri",
                                            "VPSRLWZ256ri",
                                            "VPSRLWZri",
                                            "VPSRLWri",
                                            "VPSUBSBYrr",
                                            "VPSUBSBZ128rr",
                                            "VPSUBSBZ256rr",
                                            "VPSUBSBZrr",
                                            "VPSUBSBrr",
                                            "VPSUBSWYrr",
                                            "VPSUBSWZ128rr",
                                            "VPSUBSWZ256rr",
                                            "VPSUBSWZrr",
                                            "VPSUBSWrr",
                                            "VPSUBUSBYrr",
                                            "VPSUBUSBZ128rr",
                                            "VPSUBUSBZ256rr",
                                            "VPSUBUSBZrr",
                                            "VPSUBUSBrr",
                                            "VPSUBUSWYrr",
                                            "VPSUBUSWZ128rr",
                                            "VPSUBUSWZ256rr",
                                            "VPSUBUSWZrr",
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
def: InstRW<[SKXWriteResGroup7], (instrs CDQ, CQO)>;
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
                                            "CLAC",
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
                                            "MOVUPDrr",
                                            "MOVUPSrr",
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
                                            "VANDNPDZ128rr",
                                            "VANDNPDZ256rr",
                                            "VANDNPDZrr",
                                            "VANDNPDrr",
                                            "VANDNPSYrr",
                                            "VANDNPSZ128rr",
                                            "VANDNPSZ256rr",
                                            "VANDNPSZrr",
                                            "VANDNPSrr",
                                            "VANDPDYrr",
                                            "VANDPDZ128rr",
                                            "VANDPDZ256rr",
                                            "VANDPDZrr",
                                            "VANDPDrr",
                                            "VANDPSYrr",
                                            "VANDPSZ128rr",
                                            "VANDPSZ256rr",
                                            "VANDPSZrr",
                                            "VANDPSrr",
                                            "VBLENDMPDZ128rr",
                                            "VBLENDMPDZ256rr",
                                            "VBLENDMPDZrr",
                                            "VBLENDMPSZ128rr",
                                            "VBLENDMPSZ256rr",
                                            "VBLENDMPSZrr",
                                            "VBLENDPDYrri",
                                            "VBLENDPDrri",
                                            "VBLENDPSYrri",
                                            "VBLENDPSrri",
                                            "VMOVAPDYrr",
                                            "VMOVAPDZ128rr",
                                            "VMOVAPDZ256rr",
                                            "VMOVAPDZrr",
                                            "VMOVAPDrr",
                                            "VMOVAPSYrr",
                                            "VMOVAPSZ128rr",
                                            "VMOVAPSZ256rr",
                                            "VMOVAPSZrr",
                                            "VMOVAPSrr",
                                            "VMOVDQA32Z128rr",
                                            "VMOVDQA32Z256rr",
                                            "VMOVDQA32Zrr",
                                            "VMOVDQA64Z128rr",
                                            "VMOVDQA64Z256rr",
                                            "VMOVDQA64Zrr",
                                            "VMOVDQAYrr",
                                            "VMOVDQArr",
                                            "VMOVDQU16Z128rr",
                                            "VMOVDQU16Z256rr",
                                            "VMOVDQU16Zrr",
                                            "VMOVDQU32Z128rr",
                                            "VMOVDQU32Z256rr",
                                            "VMOVDQU32Zrr",
                                            "VMOVDQU64Z128rr",
                                            "VMOVDQU64Z256rr",
                                            "VMOVDQU64Zrr",
                                            "VMOVDQU8Z128rr",
                                            "VMOVDQU8Z256rr",
                                            "VMOVDQU8Zrr",
                                            "VMOVDQUYrr",
                                            "VMOVDQUrr",
                                            "VMOVPQI(2Q|Lo2PQ)IZrr",
                                            "VMOVPQI2QIrr",
                                            "VMOVUPDYrr",
                                            "VMOVUPDZ128rr",
                                            "VMOVUPDZ256rr",
                                            "VMOVUPDZrr",
                                            "VMOVUPDrr",
                                            "VMOVUPSZ128rr",
                                            "VMOVUPSZ256rr",
                                            "VMOVUPSZrr",
                                            "VMOVUPSYrr",
                                            "VMOVUPSrr",
                                            "VMOVZPQILo2PQIrr",
                                            "VORPDYrr",
                                            "VORPDZ128rr",
                                            "VORPDZ256rr",
                                            "VORPDZrr",
                                            "VORPDrr",
                                            "VORPSYrr",
                                            "VORPSZ128rr",
                                            "VORPSZ256rr",
                                            "VORPSZrr",
                                            "VORPSrr",
                                            "VPADDBYrr",
                                            "VPADDBZ128rr",
                                            "VPADDBZ256rr",
                                            "VPADDBZrr",
                                            "VPADDBrr",
                                            "VPADDDYrr",
                                            "VPADDDZ128rr",
                                            "VPADDDZ256rr",
                                            "VPADDDZrr",
                                            "VPADDDrr",
                                            "VPADDQYrr",
                                            "VPADDQZ128rr",
                                            "VPADDQZ256rr",
                                            "VPADDQZrr",
                                            "VPADDQrr",
                                            "VPADDWYrr",
                                            "VPADDWZ128rr",
                                            "VPADDWZ256rr",
                                            "VPADDWZrr",
                                            "VPADDWrr",
                                            "VPANDDZ128rr",
                                            "VPANDDZ256rr",
                                            "VPANDDZrr",
                                            "VPANDNDZ128rr",
                                            "VPANDNDZ256rr",
                                            "VPANDNDZrr",
                                            "VPANDNQZ128rr",
                                            "VPANDNQZ256rr",
                                            "VPANDNQZrr",
                                            "VPANDNYrr",
                                            "VPANDNrr",
                                            "VPANDQZ128rr",
                                            "VPANDQZ256rr",
                                            "VPANDQZrr",
                                            "VPANDYrr",
                                            "VPANDrr",
                                            "VPBLENDDYrri",
                                            "VPBLENDDrri",
                                            "VPBLENDMBZ128rr",
                                            "VPBLENDMBZ256rr",
                                            "VPBLENDMBZrr",
                                            "VPBLENDMDZ128rr",
                                            "VPBLENDMDZ256rr",
                                            "VPBLENDMDZrr",
                                            "VPBLENDMQZ128rr",
                                            "VPBLENDMQZ256rr",
                                            "VPBLENDMQZrr",
                                            "VPBLENDMWZ128rr",
                                            "VPBLENDMWZ256rr",
                                            "VPBLENDMWZrr",
                                            "VPORDZ128rr",
                                            "VPORDZ256rr",
                                            "VPORDZrr",
                                            "VPORQZ128rr",
                                            "VPORQZ256rr",
                                            "VPORQZrr",
                                            "VPORYrr",
                                            "VPORrr",
                                            "VPSUBBYrr",
                                            "VPSUBBZ128rr",
                                            "VPSUBBZ256rr",
                                            "VPSUBBZrr",
                                            "VPSUBBrr",
                                            "VPSUBDYrr",
                                            "VPSUBDZ128rr",
                                            "VPSUBDZ256rr",
                                            "VPSUBDZrr",
                                            "VPSUBDrr",
                                            "VPSUBQYrr",
                                            "VPSUBQZ128rr",
                                            "VPSUBQZ256rr",
                                            "VPSUBQZrr",
                                            "VPSUBQrr",
                                            "VPSUBWYrr",
                                            "VPSUBWZ128rr",
                                            "VPSUBWZrr",
                                            "VPSUBWrr",
                                            "VPTERNLOGDZ128rri",
                                            "VPTERNLOGDZ256rri",
                                            "VPTERNLOGDZrri",
                                            "VPTERNLOGQZ128rri",
                                            "VPTERNLOGQZ256rri",
                                            "VPTERNLOGQZrri",
                                            "VPXORDZ128rr",
                                            "VPXORDZ256rr",
                                            "VPXORDZrr",
                                            "VPXORQZ128rr",
                                            "VPXORQZ256rr",
                                            "VPXORQZrr",
                                            "VPXORYrr",
                                            "VPXORrr",
                                            "VXORPDYrr",
                                            "VXORPDZ128rr",
                                            "VXORPDZ256rr",
                                            "VXORPDZrr",
                                            "VXORPDrr",
                                            "VXORPSYrr",
                                            "VXORPSZ128rr",
                                            "VXORPSZ256rr",
                                            "VXORPSZrr",
                                            "VXORPSrr",
                                            "XORPDrr",
                                            "XORPSrr")>;

def SKXWriteResGroup10 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup10], (instrs CBW, CWDE, CDQE)>;
def: InstRW<[SKXWriteResGroup10], (instregex "CLC",
                                             "CMC",
                                             "LAHF",
                                             "NOOP",
                                             "SAHF",
                                             "SGDT64m",
                                             "SIDT64m",
                                             "SLDT64m",
                                             "SMSW16m",
                                             "STC",
                                             "STRm",
                                             "SYSCALL",
                                             "XCHG(16|32|64)rr")>;

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
                                             "VEXTRACTF32x4Z256mr(b?)",
                                             "VEXTRACTF32x4Zmr(b?)",
                                             "VEXTRACTF32x8Zmr(b?)",
                                             "VEXTRACTF64x2Z256mr(b?)",
                                             "VEXTRACTF64x2Zmr(b?)",
                                             "VEXTRACTF64x4Zmr(b?)",
                                             "VEXTRACTI128mr",
                                             "VEXTRACTI32x4Z256mr(b?)",
                                             "VEXTRACTI32x4Zmr(b?)",
                                             "VEXTRACTI32x8Zmr(b?)",
                                             "VEXTRACTI64x2Z256mr(b?)",
                                             "VEXTRACTI64x2Zmr(b?)",
                                             "VEXTRACTI64x4Zmr(b?)",
                                             "VMOVAPDYmr",
                                             "VMOVAPDZ128mr(b?)",
                                             "VMOVAPDZ256mr(b?)",
                                             "VMOVAPDZmr(b?)",
                                             "VMOVAPDmr",
                                             "VMOVAPSYmr",
                                             "VMOVAPSZ128mr(b?)",
                                             "VMOVAPSZ256mr(b?)",
                                             "VMOVAPSZmr(b?)",
                                             "VMOVAPSmr",
                                             "VMOVDQA32Z128mr(b?)",
                                             "VMOVDQA32Z256mr(b?)",
                                             "VMOVDQA32Zmr(b?)",
                                             "VMOVDQA64Z128mr(b?)",
                                             "VMOVDQA64Z256mr(b?)",
                                             "VMOVDQA64Zmr(b?)",
                                             "VMOVDQAYmr",
                                             "VMOVDQAmr",
                                             "VMOVDQU16Z128mr(b?)",
                                             "VMOVDQU16Z256mr(b?)",
                                             "VMOVDQU16Zmr(b?)",
                                             "VMOVDQU32Z128mr(b?)",
                                             "VMOVDQU32Z256mr(b?)",
                                             "VMOVDQU32Zmr(b?)",
                                             "VMOVDQU64Z128mr(b?)",
                                             "VMOVDQU64Z256mr(b?)",
                                             "VMOVDQU64Zmr(b?)",
                                             "VMOVDQU8Z128mr(b?)",
                                             "VMOVDQU8Z256mr(b?)",
                                             "VMOVDQUYmr",
                                             "VMOVDQUmr",
                                             "VMOVHPDZ128mr(b?)",
                                             "VMOVHPDmr",
                                             "VMOVHPSZ128mr(b?)",
                                             "VMOVHPSmr",
                                             "VMOVLPDZ128mr(b?)",
                                             "VMOVLPDmr",
                                             "VMOVLPSZ128mr(b?)",
                                             "VMOVLPSmr",
                                             "VMOVNTDQYmr",
                                             "VMOVNTDQZ128mr(b?)",
                                             "VMOVNTDQZ256mr(b?)",
                                             "VMOVNTDQZmr(b?)",
                                             "VMOVNTDQmr",
                                             "VMOVNTPDYmr",
                                             "VMOVNTPDZ128mr(b?)",
                                             "VMOVNTPDZ256mr(b?)",
                                             "VMOVNTPDZmr(b?)",
                                             "VMOVNTPDmr",
                                             "VMOVNTPSYmr",
                                             "VMOVNTPSZ128mr(b?)",
                                             "VMOVNTPSZ256mr(b?)",
                                             "VMOVNTPSZmr(b?)",
                                             "VMOVNTPSmr",
                                             "VMOVPDI2DIZmr(b?)",
                                             "VMOVPDI2DImr",
                                             "VMOVPQI(2QI|to64)Zmr(b?)",
                                             "VMOVPQI2QImr",
                                             "VMOVPQIto64mr",
                                             "VMOVSDZmr(b?)",
                                             "VMOVSDmr",
                                             "VMOVSSZmr(b?)",
                                             "VMOVSSmr",
                                             "VMOVUPDYmr",
                                             "VMOVUPDZ128mr(b?)",
                                             "VMOVUPDZ256mr(b?)",
                                             "VMOVUPDZmr(b?)",
                                             "VMOVUPDmr",
                                             "VMOVUPSYmr",
                                             "VMOVUPSZ128mr(b?)",
                                             "VMOVUPSZ256mr(b?)",
                                             "VMOVUPSZmr(b?)",
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
                                             "MOVPDI2DIrr",
                                             "MOVPQIto64rr",
                                             "UCOMISDrr",
                                             "UCOMISSrr",
                                             "VCOMISDZrr(b?)",
                                             "VCOMISDrr",
                                             "VCOMISSZrr(b?)",
                                             "VCOMISSrr",
                                             "VMOVPDI2DIZrr",
                                             "VMOVPDI2DIrr",
                                             "VMOVPQIto64Zrr",
                                             "VMOVPQIto64rr",
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
                                             "VPINSRBZrr",
                                             "VPINSRBrr",
                                             "VPINSRDZrr",
                                             "VPINSRDrr",
                                             "VPINSRQZrr",
                                             "VPINSRQrr",
                                             "VPINSRWZrr",
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
                                             "VPSLLDZ128rr",
                                             "VPSLLDrr",
                                             "VPSLLQZ128rr",
                                             "VPSLLQrr",
                                             "VPSLLWZ128rr",
                                             "VPSLLWrr",
                                             "VPSRADZ128rr",
                                             "VPSRADrr",
                                             "VPSRAQZ128rr",
                                             "VPSRAWZ128rr",
                                             "VPSRAWrr",
                                             "VPSRLDZ128rr",
                                             "VPSRLDrr",
                                             "VPSRLQZ128rr",
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
def: InstRW<[SKXWriteResGroup22], (instrs BSWAP64r)>;

def SKXWriteResGroup22_1 : SchedWriteRes<[SKXPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup22_1], (instrs BSWAP32r)>;

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
                                             "VEXTRACTPSZmr(b?)",
                                             "VEXTRACTPSmr",
                                             "VPEXTRBZmr(b?)",
                                             "VPEXTRBmr",
                                             "VPEXTRDZmr(b?)",
                                             "VPEXTRDmr",
                                             "VPEXTRQZmr(b?)",
                                             "VPEXTRQmr",
                                             "VPEXTRWZmr(b?)",
                                             "VPEXTRWmr",
                                             "VSTMXCSR")>;

def SKXWriteResGroup25 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup25], (instregex "FNSTCW16m")>;

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
def: InstRW<[SKXWriteResGroup29], (instregex "VMOVDQU8Zmr(b?)")>;

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
def: InstRW<[SKXWriteResGroup31], (instregex "PDEP(32|64)rr",
                                             "PEXT(32|64)rr",
                                             "SHLD(16|32|64)rri8",
                                             "SHRD(16|32|64)rri8")>;

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
                                             "VALIGNDZ128rri",
                                             "VALIGNDZ256rri",
                                             "VALIGNDZrri",
                                             "VALIGNQZ128rri",
                                             "VALIGNQZ256rri",
                                             "VALIGNQZrri",
                                             "VBROADCASTF32X2Z256r",
                                             "VBROADCASTF32X2Zr",
                                             "VBROADCASTI32X2Z256r",
                                             "VBROADCASTI32X2Zr",
                                             "VBROADCASTSDYrr",
                                             "VBROADCASTSDZ256r",
                                             "VBROADCASTSDZr",
                                             "VBROADCASTSSYrr",
                                             "VBROADCASTSSZ128r",
                                             "VBROADCASTSSZ256r",
                                             "VBROADCASTSSZr",
                                             "VCMPPDZ128rri",
                                             "VCMPPDZ256rri",
                                             "VCMPPDZrri",
                                             "VCMPPSZ128rri",
                                             "VCMPPSZ256rri",
                                             "VCMPPSZrri",
                                             "VCMPSDZrr",
                                             "VCMPSSZrr",
                                             "VDBPSADBWZ128rri",
                                             "VDBPSADBWZ256rri",
                                             "VDBPSADBWZrri",
                                             "VEXTRACTF128rr",
                                             "VEXTRACTF32x4Z256rr",
                                             "VEXTRACTF32x4Zrr",
                                             "VEXTRACTF32x8Zrr",
                                             "VEXTRACTF64x2Z256rr",
                                             "VEXTRACTF64x2Zrr",
                                             "VEXTRACTF64x4Zrr",
                                             "VEXTRACTI128rr",
                                             "VEXTRACTI32x4Z256rr",
                                             "VEXTRACTI32x4Zrr",
                                             "VEXTRACTI32x8Zrr",
                                             "VEXTRACTI64x2Z256rr",
                                             "VEXTRACTI64x2Zrr",
                                             "VEXTRACTI64x4Zrr",
                                             "VFPCLASSPDZ128rr",
                                             "VFPCLASSPDZ256rr",
                                             "VFPCLASSPDZrr",
                                             "VFPCLASSPSZ128rr",
                                             "VFPCLASSPSZ256rr",
                                             "VFPCLASSPSZrr",
                                             "VFPCLASSSDrr",
                                             "VFPCLASSSSrr",
                                             "VINSERTF128rr",
                                             "VINSERTF32x4Z256rr",
                                             "VINSERTF32x4Zrr",
                                             "VINSERTF32x8Zrr",
                                             "VINSERTF64x2Z256rr",
                                             "VINSERTF64x2Zrr",
                                             "VINSERTF64x4Zrr",
                                             "VINSERTI128rr",
                                             "VINSERTI32x4Z256rr",
                                             "VINSERTI32x4Zrr",
                                             "VINSERTI32x8Zrr",
                                             "VINSERTI64x2Z256rr",
                                             "VINSERTI64x2Zrr",
                                             "VINSERTI64x4Zrr",
                                             "VPBROADCASTBYrr",
                                             "VPBROADCASTBZ128r",
                                             "VPBROADCASTBZ256r",
                                             "VPBROADCASTBZr",
                                             "VPBROADCASTBrr",
                                             "VPBROADCASTDYrr",
                                             "VPBROADCASTDZ128r",
                                             "VPBROADCASTDZ256r",
                                             "VPBROADCASTDZr",
                                             "VPBROADCASTDrZ128r",
                                             "VPBROADCASTDrZ256r",
                                             "VPBROADCASTDrZr",
                                             "VPBROADCASTQYrr",
                                             "VPBROADCASTQZ128r",
                                             "VPBROADCASTQZ256r",
                                             "VPBROADCASTQZr",
                                             "VPBROADCASTQrZ128r",
                                             "VPBROADCASTQrZ256r",
                                             "VPBROADCASTQrZr",
                                             "VPBROADCASTWYrr",
                                             "VPBROADCASTWZ128r",
                                             "VPBROADCASTWZ256r",
                                             "VPBROADCASTWZr",
                                             "VPBROADCASTWrr",
                                             "VPCMPBZ128rri",
                                             "VPCMPBZ256rri",
                                             "VPCMPBZrri",
                                             "VPCMPDZ128rri",
                                             "VPCMPDZ256rri",
                                             "VPCMPDZrri",
                                             "VPCMPEQBZ128rr",
                                             "VPCMPEQBZ256rr",
                                             "VPCMPEQBZrr",
                                             "VPCMPEQDZ128rr",
                                             "VPCMPEQDZ256rr",
                                             "VPCMPEQDZrr",
                                             "VPCMPEQQZ128rr",
                                             "VPCMPEQQZ256rr",
                                             "VPCMPEQQZrr",
                                             "VPCMPEQWZ128rr",
                                             "VPCMPEQWZ256rr",
                                             "VPCMPEQWZrr",
                                             "VPCMPGTBZ128rr",
                                             "VPCMPGTBZ256rr",
                                             "VPCMPGTBZrr",
                                             "VPCMPGTDZ128rr",
                                             "VPCMPGTDZ256rr",
                                             "VPCMPGTDZrr",
                                             "VPCMPGTQYrr",
                                             "VPCMPGTQZ128rr",
                                             "VPCMPGTQZ256rr",
                                             "VPCMPGTQZrr",
                                             "VPCMPGTQrr",
                                             "VPCMPGTWZ128rr",
                                             "VPCMPGTWZ256rr",
                                             "VPCMPGTWZrr",
                                             "VPCMPQZ128rri",
                                             "VPCMPQZ256rri",
                                             "VPCMPQZrri",
                                             "VPCMPUBZ128rri",
                                             "VPCMPUBZ256rri",
                                             "VPCMPUBZrri",
                                             "VPCMPUDZ128rri",
                                             "VPCMPUDZ256rri",
                                             "VPCMPUDZrri",
                                             "VPCMPUQZ128rri",
                                             "VPCMPUQZ256rri",
                                             "VPCMPUQZrri",
                                             "VPCMPUWZ128rri",
                                             "VPCMPUWZ256rri",
                                             "VPCMPUWZrri",
                                             "VPCMPWZ128rri",
                                             "VPCMPWZ256rri",
                                             "VPCMPWZrri",
                                             "VPERM2F128rr",
                                             "VPERM2I128rr",
                                             "VPERMDYrr",
                                             "VPERMDZ256rr",
                                             "VPERMDZrr",
                                             "VPERMI2D128rr",
                                             "VPERMI2D256rr",
                                             "VPERMI2Drr",
                                             "VPERMI2PD128rr",
                                             "VPERMI2PD256rr",
                                             "VPERMI2PDrr",
                                             "VPERMI2PS128rr",
                                             "VPERMI2PS256rr",
                                             "VPERMI2PSrr",
                                             "VPERMI2Q128rr",
                                             "VPERMI2Q256rr",
                                             "VPERMI2Qrr",
                                             "VPERMPDYri",
                                             "VPERMPDZ256ri",
                                             "VPERMPDZ256rr",
                                             "VPERMPDZri",
                                             "VPERMPDZrr",
                                             "VPERMPSYrr",
                                             "VPERMPSZ256rr",
                                             "VPERMPSZrr",
                                             "VPERMQYri",
                                             "VPERMQZ256ri",
                                             "VPERMQZ256rr",
                                             "VPERMQZri",
                                             "VPERMQZrr",
                                             "VPERMT2D128rr",
                                             "VPERMT2D256rr",
                                             "VPERMT2Drr",
                                             "VPERMT2PD128rr",
                                             "VPERMT2PD256rr",
                                             "VPERMT2PDrr",
                                             "VPERMT2PS128rr",
                                             "VPERMT2PS256rr",
                                             "VPERMT2PSrr",
                                             "VPERMT2Q128rr",
                                             "VPERMT2Q256rr",
                                             "VPERMT2Qrr",
                                             "VPMAXSQZ128rr",
                                             "VPMAXSQZ256rr",
                                             "VPMAXSQZrr",
                                             "VPMAXUQZ128rr",
                                             "VPMAXUQZ256rr",
                                             "VPMAXUQZrr",
                                             "VPMINSQZ128rr",
                                             "VPMINSQZ256rr",
                                             "VPMINSQZrr",
                                             "VPMINUQZ128rr",
                                             "VPMINUQZ256rr",
                                             "VPMINUQZrr",
                                             "VPMOVQDZ128rr",
                                             "VPMOVQDZ256rr",
                                             "VPMOVQDZrr",
                                             "VPMOVSXBDYrr",
                                             "VPMOVSXBDZ128rr",
                                             "VPMOVSXBDZ256rr",
                                             "VPMOVSXBDZrr",
                                             "VPMOVSXBQYrr",
                                             "VPMOVSXBQZ128rr",
                                             "VPMOVSXBQZ256rr",
                                             "VPMOVSXBQZrr",
                                             "VPMOVSXBWYrr",
                                             "VPMOVSXBWZ128rr",
                                             "VPMOVSXBWZ256rr",
                                             "VPMOVSXBWZrr",
                                             "VPMOVSXDQYrr",
                                             "VPMOVSXDQZ128rr",
                                             "VPMOVSXDQZ256rr",
                                             "VPMOVSXDQZrr",
                                             "VPMOVSXWDYrr",
                                             "VPMOVSXWDZ128rr",
                                             "VPMOVSXWDZ256rr",
                                             "VPMOVSXWDZrr",
                                             "VPMOVSXWQYrr",
                                             "VPMOVSXWQZ128rr",
                                             "VPMOVSXWQZ256rr",
                                             "VPMOVSXWQZrr",
                                             "VPMOVZXBDYrr",
                                             "VPMOVZXBDZ128rr",
                                             "VPMOVZXBDZ256rr",
                                             "VPMOVZXBDZrr",
                                             "VPMOVZXBQYrr",
                                             "VPMOVZXBQZ128rr",
                                             "VPMOVZXBQZ256rr",
                                             "VPMOVZXBQZrr",
                                             "VPMOVZXBWYrr",
                                             "VPMOVZXBWZ128rr",
                                             "VPMOVZXBWZ256rr",
                                             "VPMOVZXBWZrr",
                                             "VPMOVZXDQYrr",
                                             "VPMOVZXDQZ128rr",
                                             "VPMOVZXDQZ256rr",
                                             "VPMOVZXDQZrr",
                                             "VPMOVZXWDYrr",
                                             "VPMOVZXWDZ128rr",
                                             "VPMOVZXWDZ256rr",
                                             "VPMOVZXWDZrr",
                                             "VPMOVZXWQYrr",
                                             "VPMOVZXWQZ128rr",
                                             "VPMOVZXWQZ256rr",
                                             "VPMOVZXWQZrr",
                                             "VPSADBWYrr",
                                             "VPSADBWZ128rr",
                                             "VPSADBWZ256rr",
                                             "VPSADBWZrr",
                                             "VPSADBWrr",
                                             "VPTESTMBZ128rr",
                                             "VPTESTMBZ256rr",
                                             "VPTESTMBZrr",
                                             "VPTESTMDZ128rr",
                                             "VPTESTMDZ256rr",
                                             "VPTESTMDZrr",
                                             "VPTESTMQZ128rr",
                                             "VPTESTMQZ256rr",
                                             "VPTESTMQZrr",
                                             "VPTESTMWZ128rr",
                                             "VPTESTMWZ256rr",
                                             "VPTESTMWZrr",
                                             "VPTESTNMBZ128rr",
                                             "VPTESTNMBZ256rr",
                                             "VPTESTNMBZrr",
                                             "VPTESTNMDZ128rr",
                                             "VPTESTNMDZ256rr",
                                             "VPTESTNMDZrr",
                                             "VPTESTNMQZ128rr",
                                             "VPTESTNMQZ256rr",
                                             "VPTESTNMQZrr",
                                             "VPTESTNMWZ128rr",
                                             "VPTESTNMWZ256rr",
                                             "VPTESTNMWZrr",
                                             "VSHUFF32X4Z256rri",
                                             "VSHUFF32X4Zrri",
                                             "VSHUFF64X2Z256rri",
                                             "VSHUFF64X2Zrri",
                                             "VSHUFI32X4Z256rri",
                                             "VSHUFI32X4Zrri",
                                             "VSHUFI64X2Z256rri",
                                             "VSHUFI64X2Zrri")>;

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
                                             "VEXTRACTPSZrr",
                                             "VEXTRACTPSrr",
                                             "VPEXTRBZrr",
                                             "VPEXTRBrr",
                                             "VPEXTRDZrr",
                                             "VPEXTRDrr",
                                             "VPEXTRQZrr",
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
                                             "VRCP14PDZ128r",
                                             "VRCP14PDZ256r",
                                             "VRCP14PSZ128r",
                                             "VRCP14PSZ256r",
                                             "VRCP14SDrr",
                                             "VRCP14SSrr",
                                             "VRCPPSYr",
                                             "VRCPPSr",
                                             "VRCPSSr",
                                             "VRSQRT14PDZ128r",
                                             "VRSQRT14PDZ256r",
                                             "VRSQRT14PSZ128r",
                                             "VRSQRT14PSZ256r",
                                             "VRSQRT14SDrr",
                                             "VRSQRT14SSrr",
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
                                             "VADDPDZ128rr",
                                             "VADDPDZ256rr",
                                             "VADDPDZrr",
                                             "VADDPDrr",
                                             "VADDPSYrr",
                                             "VADDPSZ128rr",
                                             "VADDPSZ256rr",
                                             "VADDPSZrr",
                                             "VADDPSrr",
                                             "VADDSDZrr",
                                             "VADDSDrr",
                                             "VADDSSZrr",
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
                                             "VCVTDQ2PSZ128rr",
                                             "VCVTDQ2PSZ256rr",
                                             "VCVTDQ2PSZrr",
                                             "VCVTDQ2PSrr",
                                             "VCVTPD2QQZ128rr",
                                             "VCVTPD2QQZ256rr",
                                             "VCVTPD2QQZrr",
                                             "VCVTPD2UQQZ128rr",
                                             "VCVTPD2UQQZ256rr",
                                             "VCVTPD2UQQZrr",
                                             "VCVTPS2DQYrr",
                                             "VCVTPS2DQZ128rr",
                                             "VCVTPS2DQZ256rr",
                                             "VCVTPS2DQZrr",
                                             "VCVTPS2DQrr",
                                             "VCVTPS2UDQZ128rr",
                                             "VCVTPS2UDQZ256rr",
                                             "VCVTPS2UDQZrr",
                                             "VCVTQQ2PDZ128rr",
                                             "VCVTQQ2PDZ256rr",
                                             "VCVTQQ2PDZrr",
                                             "VCVTTPD2QQZ128rr",
                                             "VCVTTPD2QQZ256rr",
                                             "VCVTTPD2QQZrr",
                                             "VCVTTPD2UQQZ128rr",
                                             "VCVTTPD2UQQZ256rr",
                                             "VCVTTPD2UQQZrr",
                                             "VCVTTPS2DQYrr",
                                             "VCVTTPS2DQZ128rr",
                                             "VCVTTPS2DQZ256rr",
                                             "VCVTTPS2DQZrr",
                                             "VCVTTPS2DQrr",
                                             "VCVTTPS2UDQZ128rr",
                                             "VCVTTPS2UDQZ256rr",
                                             "VCVTTPS2UDQZrr",
                                             "VCVTUDQ2PSZ128rr",
                                             "VCVTUDQ2PSZ256rr",
                                             "VCVTUDQ2PSZrr",
                                             "VCVTUQQ2PDZ128rr",
                                             "VCVTUQQ2PDZ256rr",
                                             "VCVTUQQ2PDZrr",
                                             "VFIXUPIMMPDZ128rri",
                                             "VFIXUPIMMPDZ256rri",
                                             "VFIXUPIMMPDZrri",
                                             "VFIXUPIMMPSZ128rri",
                                             "VFIXUPIMMPSZ256rri",
                                             "VFIXUPIMMPSZrri",
                                             "VFIXUPIMMSDrri",
                                             "VFIXUPIMMSSrri",
                                             "VGETEXPPDZ128r",
                                             "VGETEXPPDZ256r",
                                             "VGETEXPPDr",
                                             "VGETEXPPSZ128r",
                                             "VGETEXPPSZ256r",
                                             "VGETEXPPSr",
                                             "VGETEXPSDr",
                                             "VGETEXPSSr",
                                             "VGETMANTPDZ128rri",
                                             "VGETMANTPDZ256rri",
                                             "VGETMANTPDZrri",
                                             "VGETMANTPSZ128rri",
                                             "VGETMANTPSZ256rri",
                                             "VGETMANTPSZrri",
                                             "VGETMANTSDZ128rri",
                                             "VGETMANTSSZ128rri",
                                             "VMAX(C?)PDYrr",
                                             "VMAX(C?)PDZ128rr",
                                             "VMAX(C?)PDZ256rr",
                                             "VMAX(C?)PDZrr",
                                             "VMAX(C?)PDrr",
                                             "VMAX(C?)PSYrr",
                                             "VMAX(C?)PSZ128rr",
                                             "VMAX(C?)PSZ256rr",
                                             "VMAX(C?)PSZrr",
                                             "VMAX(C?)PSrr",
                                             "VMAX(C?)SDZrr",
                                             "VMAX(C?)SDrr",
                                             "VMAX(C?)SSZrr",
                                             "VMAX(C?)SSrr",
                                             "VMIN(C?)PDYrr",
                                             "VMIN(C?)PDZ128rr",
                                             "VMIN(C?)PDZ256rr",
                                             "VMIN(C?)PDZrr",
                                             "VMIN(C?)PDrr",
                                             "VMIN(C?)PSYrr",
                                             "VMIN(C?)PSZ128rr",
                                             "VMIN(C?)PSZ256rr",
                                             "VMIN(C?)PSZrr",
                                             "VMIN(C?)PSrr",
                                             "VMIN(C?)SDZrr",
                                             "VMIN(C?)SDrr",
                                             "VMIN(C?)SSZrr",
                                             "VMIN(C?)SSrr",
                                             "VMULPDYrr",
                                             "VMULPDZ128rr",
                                             "VMULPDZ256rr",
                                             "VMULPDZrr",
                                             "VMULPDrr",
                                             "VMULPSYrr",
                                             "VMULPSZ128rr",
                                             "VMULPSZ256rr",
                                             "VMULPSZrr",
                                             "VMULPSrr",
                                             "VMULSDZrr",
                                             "VMULSDrr",
                                             "VMULSSZrr",
                                             "VMULSSrr",
                                             "VPHMINPOSUWrr",
                                             "VPLZCNTDZ128rr",
                                             "VPLZCNTDZ256rr",
                                             "VPLZCNTDZrr",
                                             "VPLZCNTQZ128rr",
                                             "VPLZCNTQZ256rr",
                                             "VPLZCNTQZrr",
                                             "VPMADDUBSWYrr",
                                             "VPMADDUBSWZ128rr",
                                             "VPMADDUBSWZ256rr",
                                             "VPMADDUBSWZrr",
                                             "VPMADDUBSWrr",
                                             "VPMADDWDYrr",
                                             "VPMADDWDZ128rr",
                                             "VPMADDWDZ256rr",
                                             "VPMADDWDZrr",
                                             "VPMADDWDrr",
                                             "VPMULDQYrr",
                                             "VPMULDQZ128rr",
                                             "VPMULDQZ256rr",
                                             "VPMULDQZrr",
                                             "VPMULDQrr",
                                             "VPMULHRSWYrr",
                                             "VPMULHRSWZ128rr",
                                             "VPMULHRSWZ256rr",
                                             "VPMULHRSWZrr",
                                             "VPMULHRSWrr",
                                             "VPMULHUWYrr",
                                             "VPMULHUWZ128rr",
                                             "VPMULHUWZ256rr",
                                             "VPMULHUWZrr",
                                             "VPMULHUWrr",
                                             "VPMULHWYrr",
                                             "VPMULHWZ128rr",
                                             "VPMULHWZ256rr",
                                             "VPMULHWZrr",
                                             "VPMULHWrr",
                                             "VPMULLWYrr",
                                             "VPMULLWZ128rr",
                                             "VPMULLWZ256rr",
                                             "VPMULLWZrr",
                                             "VPMULLWrr",
                                             "VPMULUDQYrr",
                                             "VPMULUDQZ128rr",
                                             "VPMULUDQZ256rr",
                                             "VPMULUDQZrr",
                                             "VPMULUDQrr",
                                             "VRANGEPDZ128rri",
                                             "VRANGEPDZ256rri",
                                             "VRANGEPDZrri",
                                             "VRANGEPSZ128rri",
                                             "VRANGEPSZ256rri",
                                             "VRANGEPSZrri",
                                             "VRANGESDZ128rri",
                                             "VRANGESSZ128rri",
                                             "VREDUCEPDZ128rri",
                                             "VREDUCEPDZ256rri",
                                             "VREDUCEPDZrri",
                                             "VREDUCEPSZ128rri",
                                             "VREDUCEPSZ256rri",
                                             "VREDUCEPSZrri",
                                             "VREDUCESDZ128rri",
                                             "VREDUCESSZ128rri",
                                             "VSCALEFPDZ128rr",
                                             "VSCALEFPDZ256rr",
                                             "VSCALEFPDZrr",
                                             "VSCALEFPSZ128rr",
                                             "VSCALEFPSZ256rr",
                                             "VSCALEFPSZrr",
                                             "VSCALEFSDZ128rr",
                                             "VSCALEFSSZ128rr",
                                             "VSUBPDYrr",
                                             "VSUBPDZ128rr",
                                             "VSUBPDZ256rr",
                                             "VSUBPDZrr",
                                             "VSUBPDrr",
                                             "VSUBPSYrr",
                                             "VSUBPSZ128rr",
                                             "VSUBPSZ256rr",
                                             "VSUBPSZrr",
                                             "VSUBPSrr",
                                             "VSUBSDZrr",
                                             "VSUBSDrr",
                                             "VSUBSSZrr",
                                             "VSUBSSrr")>;
def: InstRW<[SKXWriteResGroup50],
          (instregex
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Yr",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Z128r",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Z256r",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Zr",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)r",
           "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)Zr",
           "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)r")>;

def SKXWriteResGroup51 : SchedWriteRes<[SKXPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup51], (instregex "MPSADBWrri",
                                             "VEXPANDPDZ128rr",
                                             "VEXPANDPDZ256rr",
                                             "VEXPANDPDZrr",
                                             "VEXPANDPSZ128rr",
                                             "VEXPANDPSZ256rr",
                                             "VEXPANDPSZrr",
                                             "VMPSADBWYrri",
                                             "VMPSADBWrri",
                                             "VPEXPANDDZ128rr",
                                             "VPEXPANDDZ256rr",
                                             "VPEXPANDDZrr",
                                             "VPEXPANDQZ128rr",
                                             "VPEXPANDQZ256rr",
                                             "VPEXPANDQZrr",
                                             "VPMOVDBZ128rr",
                                             "VPMOVDBZ256rr",
                                             "VPMOVDBZrr",
                                             "VPMOVDWZ128rr",
                                             "VPMOVDWZ256rr",
                                             "VPMOVDWZrr",
                                             "VPMOVQBZ128rr",
                                             "VPMOVQBZ256rr",
                                             "VPMOVQBZrr",
                                             "VPMOVQWZ128rr",
                                             "VPMOVQWZ256rr",
                                             "VPMOVQWZrr",
                                             "VPMOVSDBZ128rr",
                                             "VPMOVSDBZ256rr",
                                             "VPMOVSDBZrr",
                                             "VPMOVSDWZ128rr",
                                             "VPMOVSDWZ256rr",
                                             "VPMOVSDWZrr",
                                             "VPMOVSQBZ128rr",
                                             "VPMOVSQBZ256rr",
                                             "VPMOVSQBZrr",
                                             "VPMOVSQDZ128rr",
                                             "VPMOVSQDZ256rr",
                                             "VPMOVSQDZrr",
                                             "VPMOVSQWZ128rr",
                                             "VPMOVSQWZ256rr",
                                             "VPMOVSQWZrr",
                                             "VPMOVSWBZ128rr",
                                             "VPMOVSWBZ256rr",
                                             "VPMOVSWBZrr",
                                             "VPMOVUSDBZ128rr",
                                             "VPMOVUSDBZ256rr",
                                             "VPMOVUSDBZrr",
                                             "VPMOVUSDWZ128rr",
                                             "VPMOVUSDWZ256rr",
                                             "VPMOVUSDWZrr",
                                             "VPMOVUSQBZ128rr",
                                             "VPMOVUSQBZ256rr",
                                             "VPMOVUSQBZrr",
                                             "VPMOVUSQDZ128rr",
                                             "VPMOVUSQDZ256rr",
                                             "VPMOVUSQDZrr",
                                             "VPMOVUSQWZ128rr",
                                             "VPMOVUSQWZ256rr",
                                             "VPMOVUSQWZrr",
                                             "VPMOVUSWBZ128rr",
                                             "VPMOVUSWBZ256rr",
                                             "VPMOVUSWBZrr",
                                             "VPMOVWBZ128rr",
                                             "VPMOVWBZ256rr",
                                             "VPMOVWBZrr")>;

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
                                             "VPSLLDZ256rr",
                                             "VPSLLDZrr",
                                             "VPSLLQYrr",
                                             "VPSLLQZ256rr",
                                             "VPSLLQZrr",
                                             "VPSLLWYrr",
                                             "VPSLLWZ256rr",
                                             "VPSLLWZrr",
                                             "VPSRADYrr",
                                             "VPSRADZ256rr",
                                             "VPSRADZrr",
                                             "VPSRAQZ256rr",
                                             "VPSRAQZrr",
                                             "VPSRAWYrr",
                                             "VPSRAWZ256rr",
                                             "VPSRAWZrr",
                                             "VPSRLDYrr",
                                             "VPSRLDZ256rr",
                                             "VPSRLDZrr",
                                             "VPSRLQYrr",
                                             "VPSRLQZ256rr",
                                             "VPSRLQZrr",
                                             "VPSRLWYrr",
                                             "VPSRLWZ256rr",
                                             "VPSRLWZrr")>;

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
                                             "VPMOVQDZ128mr(b?)",
                                             "VPMOVQDZ256mr(b?)",
                                             "VPMOVQDZmr(b?)")>;

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
def: InstRW<[SKXWriteResGroup59], (instregex "VCVTSD2SSZrr")>;

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
                                             "VCVTDQ2PDZ128rr",
                                             "VCVTPD2DQZ128rr",
                                             "VCVTPD2DQrr",
                                             "VCVTPD2PSZ128rr",
                                             "VCVTPD2PSrr",
                                             "VCVTPD2UDQZ128rr",
                                             "VCVTPH2PSZ128rr",
                                             "VCVTPH2PSrr",
                                             "VCVTPS2PDZ128rr",
                                             "VCVTPS2PDrr",
                                             "VCVTPS2PHZ128rr",
                                             "VCVTPS2PHrr",
                                             "VCVTPS2QQZ128rr",
                                             "VCVTPS2UQQZ128rr",
                                             "VCVTQQ2PSZ128rr",
                                             "VCVTSD2SSrr",
                                             "VCVTSI642SDrr",
                                             "VCVTSI2SDZrr",
                                             "VCVTSI2SDrr",
                                             "VCVTSI2SSZrr",
                                             "VCVTSI2SSrr",
                                             "VCVTSI642SDZrr",
                                             "VCVTSS2SDZrr",
                                             "VCVTSS2SDrr",
                                             "VCVTTPD2DQZ128rr",
                                             "VCVTTPD2DQrr",
                                             "VCVTTPD2UDQZ128rr",
                                             "VCVTTPS2QQZ128rr",
                                             "VCVTTPS2UQQZ128rr",
                                             "VCVTUDQ2PDZ128rr",
                                             "VCVTUQQ2PSZ128rr",
                                             "VCVTUSI2SDZrr",
                                             "VCVTUSI2SSZrr",
                                             "VCVTUSI642SDZrr")>;

def SKXWriteResGroup62 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup62], (instregex "VPCONFLICTQZ128rr")>;

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
def: InstRW<[SKXWriteResGroup65], (instregex "VCVTPS2PHZ128mr(b?)",
                                             "VCVTPS2PHZ256mr(b?)",
                                             "VCVTPS2PHZmr(b?)")>;

def SKXWriteResGroup66 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVDBZ128mr(b?)",
                                             "VPMOVDBZ256mr(b?)",
                                             "VPMOVDBZmr(b?)",
                                             "VPMOVDWZ128mr(b?)",
                                             "VPMOVDWZ256mr(b?)",
                                             "VPMOVDWZmr(b?)",
                                             "VPMOVQBZ128mr(b?)",
                                             "VPMOVQBZ256mr(b?)",
                                             "VPMOVQBZmr(b?)",
                                             "VPMOVQWZ128mr(b?)",
                                             "VPMOVQWZ256mr(b?)",
                                             "VPMOVQWZmr(b?)",
                                             "VPMOVSDBZ128mr(b?)",
                                             "VPMOVSDBZ256mr(b?)",
                                             "VPMOVSDBZmr(b?)",
                                             "VPMOVSDWZ128mr(b?)",
                                             "VPMOVSDWZ256mr(b?)",
                                             "VPMOVSDWZmr(b?)",
                                             "VPMOVSQBZ128mr(b?)",
                                             "VPMOVSQBZ256mr(b?)",
                                             "VPMOVSQBZmr(b?)",
                                             "VPMOVSQDZ128mr(b?)",
                                             "VPMOVSQDZ256mr(b?)",
                                             "VPMOVSQDZmr(b?)",
                                             "VPMOVSQWZ128mr(b?)",
                                             "VPMOVSQWZ256mr(b?)",
                                             "VPMOVSQWZmr(b?)",
                                             "VPMOVSWBZ128mr(b?)",
                                             "VPMOVSWBZ256mr(b?)",
                                             "VPMOVSWBZmr(b?)",
                                             "VPMOVUSDBZ128mr(b?)",
                                             "VPMOVUSDBZ256mr(b?)",
                                             "VPMOVUSDBZmr(b?)",
                                             "VPMOVUSDWZ128mr(b?)",
                                             "VPMOVUSDWZ256mr(b?)",
                                             "VPMOVUSDWZmr(b?)",
                                             "VPMOVUSQBZ128mr(b?)",
                                             "VPMOVUSQBZ256mr(b?)",
                                             "VPMOVUSQBZmr(b?)",
                                             "VPMOVUSQDZ128mr(b?)",
                                             "VPMOVUSQDZ256mr(b?)",
                                             "VPMOVUSQDZmr(b?)",
                                             "VPMOVUSQWZ128mr(b?)",
                                             "VPMOVUSQWZ256mr(b?)",
                                             "VPMOVUSQWZmr(b?)",
                                             "VPMOVUSWBZ128mr(b?)",
                                             "VPMOVUSWBZ256mr(b?)",
                                             "VPMOVUSWBZmr(b?)",
                                             "VPMOVWBZ128mr(b?)",
                                             "VPMOVWBZ256mr(b?)",
                                             "VPMOVWBZmr(b?)")>;

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
                                             "VCOMPRESSPDZ128rr",
                                             "VCOMPRESSPDZ256rr",
                                             "VCOMPRESSPDZrr",
                                             "VCOMPRESSPSZ128rr",
                                             "VCOMPRESSPSZ256rr",
                                             "VCOMPRESSPSZrr",
                                             "VPCOMPRESSDZ128rr",
                                             "VPCOMPRESSDZ256rr",
                                             "VPCOMPRESSDZrr",
                                             "VPCOMPRESSQZ128rr",
                                             "VPCOMPRESSQZ256rr",
                                             "VPCOMPRESSQZrr",
                                             "VPERMWZ128rr",
                                             "VPERMWZ256rr",
                                             "VPERMWZrr")>;

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
                                             "VCVTSD2SI64Zrr",
                                             "VCVTSD2SI64rr",
                                             "VCVTSD2SIZrr",
                                             "VCVTSD2SIrr",
                                             "VCVTSD2USI64Zrr",
                                             "VCVTSD2USIZrr",
                                             "VCVTSS2SI64Zrr",
                                             "VCVTSS2SI64rr",
                                             "VCVTSS2SIZrr",
                                             "VCVTSS2SIrr",
                                             "VCVTSS2USIZrr",
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
                                             "VMOVHPDZ128rm(b?)",
                                             "VMOVHPDrm",
                                             "VMOVHPSZ128rm(b?)",
                                             "VMOVHPSrm",
                                             "VMOVLPDZ128rm(b?)",
                                             "VMOVLPDrm",
                                             "VMOVLPSZ128rm(b?)",
                                             "VMOVLPSrm",
                                             "VPINSRBZrm(b?)",
                                             "VPINSRBrm",
                                             "VPINSRDZrm(b?)",
                                             "VPINSRDrm",
                                             "VPINSRQZrm(b?)",
                                             "VPINSRQrm",
                                             "VPINSRWZrm(b?)",
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
def: InstRW<[SKXWriteResGroup78], (instregex "BT(16|32|64)mi8",
                                             "RORX(32|64)mi",
                                             "SARX(32|64)rm",
                                             "SHLX(32|64)rm",
                                             "SHRX(32|64)rm")>;
def: InstRW<[SKXWriteResGroup78, ReadAfterLd], (instrs ADC8rm, ADC16rm, ADC32rm, ADC64rm,
                                                       ADCX32rm, ADCX64rm,
                                                       ADOX32rm, ADOX64rm,
                                                       SBB8rm, SBB16rm, SBB32rm, SBB64rm)>;

def SKXWriteResGroup79 : SchedWriteRes<[SKXPort23,SKXPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup79], (instregex "ANDN(32|64)rm",
                                             "BLSI(32|64)rm",
                                             "BLSMSK(32|64)rm",
                                             "BLSR(32|64)rm",
                                             "MOVBE(16|32|64)rm")>;

def SKXWriteResGroup80 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup80], (instregex "VMOV(64to|QI2)PQIZrm(b?)",
                                             "VMOVDI2PDIZrm(b?)")>;

def SKXWriteResGroup81 : SchedWriteRes<[SKXPort23,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup81], (instrs POP16r, POP32r, POP64r)>;
def: InstRW<[SKXWriteResGroup81], (instregex "POP(16|32|64)rmr")>;

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
                                             "VCVTSI642SSZrr",
                                             "VCVTUSI642SSZrr",
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
def: InstRW<[SKXWriteResGroup87], (instregex "POP(16|32|64)rmm",
                                             "PUSH(16|32|64)rmm")>;

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
                                             "VMOVNTDQAZrm(b?)",
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
                                             "VCOMISDZrm(b?)",
                                             "VCOMISDrm",
                                             "VCOMISSZrm(b?)",
                                             "VCOMISSrm",
                                             "VUCOMISDZrm(b?)",
                                             "VUCOMISDrm",
                                             "VUCOMISSZrm(b?)",
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
                                             "VINSERTPSZrm(b?)",
                                             "VINSERTPSrm",
                                             "VMOVSDZrm(b?)",
                                             "VMOVSSZrm(b?)",
                                             "VPACKSSDWZ128rm(b?)",
                                             "VPACKSSDWrm",
                                             "VPACKSSWBZ128rm(b?)",
                                             "VPACKSSWBrm",
                                             "VPACKUSDWZ128rm(b?)",
                                             "VPACKUSDWrm",
                                             "VPACKUSWBZ128rm(b?)",
                                             "VPACKUSWBrm",
                                             "VPALIGNRZ128rmi(b?)",
                                             "VPALIGNRrmi",
                                             "VPBLENDWrmi",
                                             "VPBROADCASTBZ128m(b?)",
                                             "VPBROADCASTBrm",
                                             "VPBROADCASTWZ128m(b?)",
                                             "VPBROADCASTWrm",
                                             "VPERMILPDZ128m(b?)i",
                                             "VPERMILPDZ128rm(b?)",
                                             "VPERMILPDmi",
                                             "VPERMILPDrm",
                                             "VPERMILPSZ128m(b?)i",
                                             "VPERMILPSZ128rm(b?)",
                                             "VPERMILPSmi",
                                             "VPERMILPSrm",
                                             "VPSHUFBZ128rm(b?)",
                                             "VPSHUFBrm",
                                             "VPSHUFDZ128m(b?)i",
                                             "VPSHUFDmi",
                                             "VPSHUFHWZ128mi(b?)",
                                             "VPSHUFHWmi",
                                             "VPSHUFLWZ128mi(b?)",
                                             "VPSHUFLWmi",
                                             "VPSLLDQZ128rm(b?)",
                                             "VPSRLDQZ128rm(b?)",
                                             "VPUNPCKHBWZ128rm(b?)",
                                             "VPUNPCKHBWrm",
                                             "VPUNPCKHDQZ128rm(b?)",
                                             "VPUNPCKHDQrm",
                                             "VPUNPCKHQDQZ128rm(b?)",
                                             "VPUNPCKHQDQrm",
                                             "VPUNPCKHWDZ128rm(b?)",
                                             "VPUNPCKHWDrm",
                                             "VPUNPCKLBWZ128rm(b?)",
                                             "VPUNPCKLBWrm",
                                             "VPUNPCKLDQZ128rm(b?)",
                                             "VPUNPCKLDQrm",
                                             "VPUNPCKLQDQZ128rm(b?)",
                                             "VPUNPCKLQDQrm",
                                             "VPUNPCKLWDZ128rm(b?)",
                                             "VPUNPCKLWDrm",
                                             "VSHUFPDZ128rm(b?)i",
                                             "VSHUFPDrmi",
                                             "VSHUFPSZ128rm(b?)i",
                                             "VSHUFPSrmi",
                                             "VUNPCKHPDZ128rm(b?)",
                                             "VUNPCKHPDrm",
                                             "VUNPCKHPSZ128rm(b?)",
                                             "VUNPCKHPSrm",
                                             "VUNPCKLPDZ128rm(b?)",
                                             "VUNPCKLPDrm",
                                             "VUNPCKLPSZ128rm(b?)",
                                             "VUNPCKLPSrm")>;

def SKXWriteResGroup93 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTDQ2PDZ256rr",
                                             "VCVTDQ2PDZrr",
                                             "VCVTPD2DQYrr",
                                             "VCVTPD2DQZ256rr",
                                             "VCVTPD2DQZrr",
                                             "VCVTPD2PSYrr",
                                             "VCVTPD2PSZ256rr",
                                             "VCVTPD2PSZrr",
                                             "VCVTPD2UDQZ256rr",
                                             "VCVTPD2UDQZrr",
                                             "VCVTPH2PSYrr",
                                             "VCVTPH2PSZ256rr",
                                             "VCVTPH2PSZrr",
                                             "VCVTPS2PDYrr",
                                             "VCVTPS2PDZ256rr",
                                             "VCVTPS2PDZrr",
                                             "VCVTPS2PHYrr",
                                             "VCVTPS2PHZ256rr",
                                             "VCVTPS2PHZrr",
                                             "VCVTPS2QQZ256rr",
                                             "VCVTPS2QQZrr",
                                             "VCVTPS2UQQZ256rr",
                                             "VCVTPS2UQQZrr",
                                             "VCVTQQ2PSZ256rr",
                                             "VCVTQQ2PSZrr",
                                             "VCVTTPD2DQYrr",
                                             "VCVTTPD2DQZ256rr",
                                             "VCVTTPD2DQZrr",
                                             "VCVTTPD2UDQZ256rr",
                                             "VCVTTPD2UDQZrr",
                                             "VCVTTPS2QQZ256rr",
                                             "VCVTTPS2QQZrr",
                                             "VCVTTPS2UQQZ256rr",
                                             "VCVTTPS2UQQZrr",
                                             "VCVTUDQ2PDZ256rr",
                                             "VCVTUDQ2PDZrr",
                                             "VCVTUQQ2PSZ256rr",
                                             "VCVTUQQ2PSZrr")>;

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
                                             "VPABSBZ128rm(b?)",
                                             "VPABSBrm",
                                             "VPABSDZ128rm(b?)",
                                             "VPABSDrm",
                                             "VPABSQZ128rm(b?)",
                                             "VPABSWZ128rm(b?)",
                                             "VPABSWrm",
                                             "VPADDSBZ128rm(b?)",
                                             "VPADDSBrm",
                                             "VPADDSWZ128rm(b?)",
                                             "VPADDSWrm",
                                             "VPADDUSBZ128rm(b?)",
                                             "VPADDUSBrm",
                                             "VPADDUSWZ128rm(b?)",
                                             "VPADDUSWrm",
                                             "VPAVGBZ128rm(b?)",
                                             "VPAVGBrm",
                                             "VPAVGWZ128rm(b?)",
                                             "VPAVGWrm",
                                             "VPCMPEQBrm",
                                             "VPCMPEQDrm",
                                             "VPCMPEQQrm",
                                             "VPCMPEQWrm",
                                             "VPCMPGTBrm",
                                             "VPCMPGTDrm",
                                             "VPCMPGTWrm",
                                             "VPMAXSBZ128rm(b?)",
                                             "VPMAXSBrm",
                                             "VPMAX(C?)SDZ128rm(b?)",
                                             "VPMAX(C?)SDrm",
                                             "VPMAXSWZ128rm(b?)",
                                             "VPMAXSWrm",
                                             "VPMAXUBZ128rm(b?)",
                                             "VPMAXUBrm",
                                             "VPMAXUDZ128rm(b?)",
                                             "VPMAXUDrm",
                                             "VPMAXUWZ128rm(b?)",
                                             "VPMAXUWrm",
                                             "VPMINSBZ128rm(b?)",
                                             "VPMINSBrm",
                                             "VPMIN(C?)SDZ128rm(b?)",
                                             "VPMIN(C?)SDrm",
                                             "VPMINSWZ128rm(b?)",
                                             "VPMINSWrm",
                                             "VPMINUBZ128rm(b?)",
                                             "VPMINUBrm",
                                             "VPMINUDZ128rm(b?)",
                                             "VPMINUDrm",
                                             "VPMINUWZ128rm(b?)",
                                             "VPMINUWrm",
                                             "VPROLDZ128m(b?)i",
                                             "VPROLQZ128m(b?)i",
                                             "VPROLVDZ128rm(b?)",
                                             "VPROLVQZ128rm(b?)",
                                             "VPRORDZ128m(b?)i",
                                             "VPRORQZ128m(b?)i",
                                             "VPRORVDZ128rm(b?)",
                                             "VPRORVQZ128rm(b?)",
                                             "VPSIGNBrm",
                                             "VPSIGNDrm",
                                             "VPSIGNWrm",
                                             "VPSLLDZ128m(b?)i",
                                             "VPSLLDZ128rm(b?)",
                                             "VPSLLDrm",
                                             "VPSLLQZ128m(b?)i",
                                             "VPSLLQZ128rm(b?)",
                                             "VPSLLQrm",
                                             "VPSLLVDZ128rm(b?)",
                                             "VPSLLVDrm",
                                             "VPSLLVQZ128rm(b?)",
                                             "VPSLLVQrm",
                                             "VPSLLVWZ128rm(b?)",
                                             "VPSLLWZ128mi(b?)",
                                             "VPSLLWZ128rm(b?)",
                                             "VPSLLWrm",
                                             "VPSRADZ128m(b?)i",
                                             "VPSRADZ128rm(b?)",
                                             "VPSRADrm",
                                             "VPSRAQZ128m(b?)i",
                                             "VPSRAQZ128rm(b?)",
                                             "VPSRAVDZ128rm(b?)",
                                             "VPSRAVDrm",
                                             "VPSRAVQZ128rm(b?)",
                                             "VPSRAVWZ128rm(b?)",
                                             "VPSRAWZ128mi(b?)",
                                             "VPSRAWZ128rm(b?)",
                                             "VPSRAWrm",
                                             "VPSRLDZ128m(b?)i",
                                             "VPSRLDZ128rm(b?)",
                                             "VPSRLDrm",
                                             "VPSRLQZ128m(b?)i",
                                             "VPSRLQZ128rm(b?)",
                                             "VPSRLQrm",
                                             "VPSRLVDZ128rm(b?)",
                                             "VPSRLVDrm",
                                             "VPSRLVQZ128rm(b?)",
                                             "VPSRLVQrm",
                                             "VPSRLVWZ128rm(b?)",
                                             "VPSRLWZ128mi(b?)",
                                             "VPSRLWZ128rm(b?)",
                                             "VPSRLWrm",
                                             "VPSUBSBZ128rm(b?)",
                                             "VPSUBSBrm",
                                             "VPSUBSWZ128rm(b?)",
                                             "VPSUBSWrm",
                                             "VPSUBUSBZ128rm(b?)",
                                             "VPSUBUSBrm",
                                             "VPSUBUSWZ128rm(b?)",
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
                                             "VANDNPDZ128rm(b?)",
                                             "VANDNPDrm",
                                             "VANDNPSZ128rm(b?)",
                                             "VANDNPSrm",
                                             "VANDPDZ128rm(b?)",
                                             "VANDPDrm",
                                             "VANDPSZ128rm(b?)",
                                             "VANDPSrm",
                                             "VBLENDMPDZ128rm(b?)",
                                             "VBLENDMPSZ128rm(b?)",
                                             "VBLENDPDrmi",
                                             "VBLENDPSrmi",
                                             "VBROADCASTI32X2Z128m(b?)",
                                             "VBROADCASTSSZ128m(b?)",
                                             "VINSERTF128rm",
                                             "VINSERTI128rm",
                                             "VMASKMOVPDrm",
                                             "VMASKMOVPSrm",
                                             "VMOVAPDZ128rm(b?)",
                                             "VMOVAPSZ128rm(b?)",
                                             "VMOVDDUPZ128rm(b?)",
                                             "VMOVDQA32Z128rm(b?)",
                                             "VMOVDQA64Z128rm(b?)",
                                             "VMOVDQU16Z128rm(b?)",
                                             "VMOVDQU32Z128rm(b?)",
                                             "VMOVDQU64Z128rm(b?)",
                                             "VMOVDQU8Z128rm(b?)",
                                             "VMOVNTDQAZ128rm(b?)",
                                             "VMOVSHDUPZ128rm(b?)",
                                             "VMOVSLDUPZ128rm(b?)",
                                             "VMOVUPDZ128rm(b?)",
                                             "VMOVUPSZ128rm(b?)",
                                             "VORPDZ128rm(b?)",
                                             "VORPDrm",
                                             "VORPSZ128rm(b?)",
                                             "VORPSrm",
                                             "VPADDBZ128rm(b?)",
                                             "VPADDBrm",
                                             "VPADDDZ128rm(b?)",
                                             "VPADDDrm",
                                             "VPADDQZ128rm(b?)",
                                             "VPADDQrm",
                                             "VPADDWZ128rm(b?)",
                                             "VPADDWrm",
                                             "VPANDDZ128rm(b?)",
                                             "VPANDNDZ128rm(b?)",
                                             "VPANDNQZ128rm(b?)",
                                             "VPANDNrm",
                                             "VPANDQZ128rm(b?)",
                                             "VPANDrm",
                                             "VPBLENDDrmi",
                                             "VPBLENDMBZ128rm(b?)",
                                             "VPBLENDMDZ128rm(b?)",
                                             "VPBLENDMQZ128rm(b?)",
                                             "VPBLENDMWZ128rm(b?)",
                                             "VPBROADCASTDZ128m(b?)",
                                             "VPBROADCASTQZ128m(b?)",
                                             "VPMASKMOVDrm",
                                             "VPMASKMOVQrm",
                                             "VPORDZ128rm(b?)",
                                             "VPORQZ128rm(b?)",
                                             "VPORrm",
                                             "VPSUBBZ128rm(b?)",
                                             "VPSUBBrm",
                                             "VPSUBDZ128rm(b?)",
                                             "VPSUBDrm",
                                             "VPSUBQZ128rm(b?)",
                                             "VPSUBQrm",
                                             "VPSUBWZ128rm(b?)",
                                             "VPSUBWrm",
                                             "VPTERNLOGDZ128rm(b?)i",
                                             "VPTERNLOGQZ128rm(b?)i",
                                             "VPXORDZ128rm(b?)",
                                             "VPXORQZ128rm(b?)",
                                             "VPXORrm",
                                             "VXORPDZ128rm(b?)",
                                             "VXORPDrm",
                                             "VXORPSZ128rm(b?)",
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
def: InstRW<[SKXWriteResGroup97], (instregex "VPERMI2W128rr",
                                             "VPERMI2W256rr",
                                             "VPERMI2Wrr",
                                             "VPERMT2W128rr",
                                             "VPERMT2W256rr",
                                             "VPERMT2Wrr")>;

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
def: InstRW<[SKXWriteResGroup99], (instrs LEAVE, LEAVE64,
                                          SCASB, SCASL, SCASQ, SCASW)>;

def SKXWriteResGroup100 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup100], (instregex "CVTTSS2SI64rr",
                                              "CVTTSS2SIrr",
                                              "VCVTSS2USI64Zrr",
                                              "VCVTTSS2SI64Zrr",
                                              "VCVTTSS2SI64rr",
                                              "VCVTTSS2SIZrr",
                                              "VCVTTSS2SIrr",
                                              "VCVTTSS2USI64Zrr")>;

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

def SKXWriteResGroup106 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup106], (instregex "VCOMPRESSPDZ128mr(b?)",
                                              "VCOMPRESSPDZ256mr(b?)",
                                              "VCOMPRESSPDZmr(b?)",
                                              "VCOMPRESSPSZ128mr(b?)",
                                              "VCOMPRESSPSZ256mr(b?)",
                                              "VCOMPRESSPSZmr(b?)",
                                              "VPCOMPRESSDZ128mr(b?)",
                                              "VPCOMPRESSDZ256mr(b?)",
                                              "VPCOMPRESSDZmr(b?)",
                                              "VPCOMPRESSQZ128mr(b?)",
                                              "VPCOMPRESSQZ256mr(b?)",
                                              "VPCOMPRESSQZmr(b?)")>;

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
                                              "VRNDSCALEPDZ128rri",
                                              "VRNDSCALEPDZ256rri",
                                              "VRNDSCALEPDZrri",
                                              "VRNDSCALEPSZ128rri",
                                              "VRNDSCALEPSZ256rri",
                                              "VRNDSCALEPSZrri",
                                              "VRNDSCALESDr",
                                              "VRNDSCALESSr",
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
                                                "VPMULLDZ128rr",
                                                "VPMULLDZ256rr",
                                                "VPMULLDZrr",
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
def: InstRW<[SKXWriteResGroup118], (instregex "PDEP(32|64)rm",
                                              "PEXT(32|64)rm")>;

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
                                              "VFPCLASSSDrm(b?)",
                                              "VPACKSSDWYrm",
                                              "VPACKSSDWZ256rm(b?)",
                                              "VPACKSSDWZrm(b?)",
                                              "VPACKSSWBYrm",
                                              "VPACKSSWBZ256rm(b?)",
                                              "VPACKSSWBZrm(b?)",
                                              "VPACKUSDWYrm",
                                              "VPACKUSDWZ256rm(b?)",
                                              "VPACKUSDWZrm(b?)",
                                              "VPACKUSWBYrm",
                                              "VPACKUSWBZ256rm(b?)",
                                              "VPACKUSWBZrm(b?)",
                                              "VPALIGNRYrmi",
                                              "VPALIGNRZ256rmi(b?)",
                                              "VPALIGNRZrmi(b?)",
                                              "VPBLENDWYrmi",
                                              "VPBROADCASTBYrm",
                                              "VPBROADCASTBZ256m(b?)",
                                              "VPBROADCASTBZm(b?)",
                                              "VPBROADCASTWYrm",
                                              "VPBROADCASTWZ256m(b?)",
                                              "VPBROADCASTWZm(b?)",
                                              "VPERMILPDYmi",
                                              "VPERMILPDYrm",
                                              "VPERMILPDZ256m(b?)i",
                                              "VPERMILPDZ256rm(b?)",
                                              "VPERMILPDZm(b?)i",
                                              "VPERMILPDZrm(b?)",
                                              "VPERMILPSYmi",
                                              "VPERMILPSYrm",
                                              "VPERMILPSZ256m(b?)i",
                                              "VPERMILPSZ256rm(b?)",
                                              "VPERMILPSZm(b?)i",
                                              "VPERMILPSZrm(b?)",
                                              "VPMOVSXBDYrm",
                                              "VPMOVSXBQYrm",
                                              "VPMOVSXWQYrm",
                                              "VPSHUFBYrm",
                                              "VPSHUFBZ256rm(b?)",
                                              "VPSHUFBZrm(b?)",
                                              "VPSHUFDYmi",
                                              "VPSHUFDZ256m(b?)i",
                                              "VPSHUFDZm(b?)i",
                                              "VPSHUFHWYmi",
                                              "VPSHUFHWZ256mi(b?)",
                                              "VPSHUFHWZmi(b?)",
                                              "VPSHUFLWYmi",
                                              "VPSHUFLWZ256mi(b?)",
                                              "VPSHUFLWZmi(b?)",
                                              "VPSLLDQZ256rm(b?)",
                                              "VPSLLDQZrm(b?)",
                                              "VPSRLDQZ256rm(b?)",
                                              "VPSRLDQZrm(b?)",
                                              "VPUNPCKHBWYrm",
                                              "VPUNPCKHBWZ256rm(b?)",
                                              "VPUNPCKHBWZrm(b?)",
                                              "VPUNPCKHDQYrm",
                                              "VPUNPCKHDQZ256rm(b?)",
                                              "VPUNPCKHDQZrm(b?)",
                                              "VPUNPCKHQDQYrm",
                                              "VPUNPCKHQDQZ256rm(b?)",
                                              "VPUNPCKHQDQZrm(b?)",
                                              "VPUNPCKHWDYrm",
                                              "VPUNPCKHWDZ256rm(b?)",
                                              "VPUNPCKHWDZrm(b?)",
                                              "VPUNPCKLBWYrm",
                                              "VPUNPCKLBWZ256rm(b?)",
                                              "VPUNPCKLBWZrm(b?)",
                                              "VPUNPCKLDQYrm",
                                              "VPUNPCKLDQZ256rm(b?)",
                                              "VPUNPCKLDQZrm(b?)",
                                              "VPUNPCKLQDQYrm",
                                              "VPUNPCKLQDQZ256rm(b?)",
                                              "VPUNPCKLQDQZrm(b?)",
                                              "VPUNPCKLWDYrm",
                                              "VPUNPCKLWDZ256rm(b?)",
                                              "VPUNPCKLWDZrm(b?)",
                                              "VSHUFPDYrmi",
                                              "VSHUFPDZ256rm(b?)i",
                                              "VSHUFPDZrm(b?)i",
                                              "VSHUFPSYrmi",
                                              "VSHUFPSZ256rm(b?)i",
                                              "VSHUFPSZrm(b?)i",
                                              "VUNPCKHPDYrm",
                                              "VUNPCKHPDZ256rm(b?)",
                                              "VUNPCKHPDZrm(b?)",
                                              "VUNPCKHPSYrm",
                                              "VUNPCKHPSZ256rm(b?)",
                                              "VUNPCKHPSZrm(b?)",
                                              "VUNPCKLPDYrm",
                                              "VUNPCKLPDZ256rm(b?)",
                                              "VUNPCKLPDZrm(b?)",
                                              "VUNPCKLPSYrm",
                                              "VUNPCKLPSZ256rm(b?)",
                                              "VUNPCKLPSZrm(b?)")>;

def SKXWriteResGroup120 : SchedWriteRes<[SKXPort01,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSBYrm",
                                              "VPABSBZ256rm(b?)",
                                              "VPABSBZrm(b?)",
                                              "VPABSDYrm",
                                              "VPABSDZ256rm(b?)",
                                              "VPABSDZrm(b?)",
                                              "VPABSQZ256rm(b?)",
                                              "VPABSQZrm(b?)",
                                              "VPABSWYrm",
                                              "VPABSWZ256rm(b?)",
                                              "VPABSWZrm(b?)",
                                              "VPADDSBYrm",
                                              "VPADDSBZ256rm(b?)",
                                              "VPADDSBZrm(b?)",
                                              "VPADDSWYrm",
                                              "VPADDSWZ256rm(b?)",
                                              "VPADDSWZrm(b?)",
                                              "VPADDUSBYrm",
                                              "VPADDUSBZ256rm(b?)",
                                              "VPADDUSBZrm(b?)",
                                              "VPADDUSWYrm",
                                              "VPADDUSWZ256rm(b?)",
                                              "VPADDUSWZrm(b?)",
                                              "VPAVGBYrm",
                                              "VPAVGBZ256rm(b?)",
                                              "VPAVGBZrm(b?)",
                                              "VPAVGWYrm",
                                              "VPAVGWZ256rm(b?)",
                                              "VPAVGWZrm(b?)",
                                              "VPCMPEQBYrm",
                                              "VPCMPEQDYrm",
                                              "VPCMPEQQYrm",
                                              "VPCMPEQWYrm",
                                              "VPCMPGTBYrm",
                                              "VPCMPGTDYrm",
                                              "VPCMPGTWYrm",
                                              "VPMAXSBYrm",
                                              "VPMAXSBZ256rm(b?)",
                                              "VPMAXSBZrm(b?)",
                                              "VPMAX(C?)SDYrm",
                                              "VPMAX(C?)SDZ256rm(b?)",
                                              "VPMAX(C?)SDZrm(b?)",
                                              "VPMAXSWYrm",
                                              "VPMAXSWZ256rm(b?)",
                                              "VPMAXSWZrm(b?)",
                                              "VPMAXUBYrm",
                                              "VPMAXUBZ256rm(b?)",
                                              "VPMAXUBZrm(b?)",
                                              "VPMAXUDYrm",
                                              "VPMAXUDZ256rm(b?)",
                                              "VPMAXUDZrm(b?)",
                                              "VPMAXUWYrm",
                                              "VPMAXUWZ256rm(b?)",
                                              "VPMAXUWZrm(b?)",
                                              "VPMINSBYrm",
                                              "VPMINSBZ256rm(b?)",
                                              "VPMINSBZrm(b?)",
                                              "VPMIN(C?)SDYrm",
                                              "VPMIN(C?)SDZ256rm(b?)",
                                              "VPMIN(C?)SDZrm(b?)",
                                              "VPMINSWYrm",
                                              "VPMINSWZ256rm(b?)",
                                              "VPMINSWZrm(b?)",
                                              "VPMINUBYrm",
                                              "VPMINUBZ256rm(b?)",
                                              "VPMINUBZrm(b?)",
                                              "VPMINUDYrm",
                                              "VPMINUDZ256rm(b?)",
                                              "VPMINUDZrm(b?)",
                                              "VPMINUWYrm",
                                              "VPMINUWZ256rm(b?)",
                                              "VPMINUWZrm(b?)",
                                              "VPROLDZ256m(b?)i",
                                              "VPROLDZm(b?)i",
                                              "VPROLQZ256m(b?)i",
                                              "VPROLQZm(b?)i",
                                              "VPROLVDZ256rm(b?)",
                                              "VPROLVDZrm(b?)",
                                              "VPROLVQZ256rm(b?)",
                                              "VPROLVQZrm(b?)",
                                              "VPRORDZ256m(b?)i",
                                              "VPRORDZm(b?)i",
                                              "VPRORQZ256m(b?)i",
                                              "VPRORQZm(b?)i",
                                              "VPRORVDZ256rm(b?)",
                                              "VPRORVDZrm(b?)",
                                              "VPRORVQZ256rm(b?)",
                                              "VPRORVQZrm(b?)",
                                              "VPSIGNBYrm",
                                              "VPSIGNDYrm",
                                              "VPSIGNWYrm",
                                              "VPSLLDYrm",
                                              "VPSLLDZ256m(b?)i",
                                              "VPSLLDZ256rm(b?)",
                                              "VPSLLDZm(b?)i",
                                              "VPSLLDZrm(b?)",
                                              "VPSLLQYrm",
                                              "VPSLLQZ256m(b?)i",
                                              "VPSLLQZ256rm(b?)",
                                              "VPSLLQZm(b?)i",
                                              "VPSLLQZrm(b?)",
                                              "VPSLLVDYrm",
                                              "VPSLLVDZ256rm(b?)",
                                              "VPSLLVDZrm(b?)",
                                              "VPSLLVQYrm",
                                              "VPSLLVQZ256rm(b?)",
                                              "VPSLLVQZrm(b?)",
                                              "VPSLLVWZ256rm(b?)",
                                              "VPSLLVWZrm(b?)",
                                              "VPSLLWYrm",
                                              "VPSLLWZ256mi(b?)",
                                              "VPSLLWZ256rm(b?)",
                                              "VPSLLWZmi(b?)",
                                              "VPSLLWZrm(b?)",
                                              "VPSRADYrm",
                                              "VPSRADZ256m(b?)i",
                                              "VPSRADZ256rm(b?)",
                                              "VPSRADZm(b?)i",
                                              "VPSRADZrm(b?)",
                                              "VPSRAQZ256m(b?)i",
                                              "VPSRAQZ256rm(b?)",
                                              "VPSRAQZm(b?)i",
                                              "VPSRAQZrm(b?)",
                                              "VPSRAVDYrm",
                                              "VPSRAVDZ256rm(b?)",
                                              "VPSRAVDZrm(b?)",
                                              "VPSRAVQZ256rm(b?)",
                                              "VPSRAVQZrm(b?)",
                                              "VPSRAVWZ256rm(b?)",
                                              "VPSRAVWZrm(b?)",
                                              "VPSRAWYrm",
                                              "VPSRAWZ256mi(b?)",
                                              "VPSRAWZ256rm(b?)",
                                              "VPSRAWZmi(b?)",
                                              "VPSRAWZrm(b?)",
                                              "VPSRLDYrm",
                                              "VPSRLDZ256m(b?)i",
                                              "VPSRLDZ256rm(b?)",
                                              "VPSRLDZm(b?)i",
                                              "VPSRLDZrm(b?)",
                                              "VPSRLQYrm",
                                              "VPSRLQZ256m(b?)i",
                                              "VPSRLQZ256rm(b?)",
                                              "VPSRLQZm(b?)i",
                                              "VPSRLQZrm(b?)",
                                              "VPSRLVDYrm",
                                              "VPSRLVDZ256rm(b?)",
                                              "VPSRLVDZrm(b?)",
                                              "VPSRLVQYrm",
                                              "VPSRLVQZ256rm(b?)",
                                              "VPSRLVQZrm(b?)",
                                              "VPSRLVWZ256rm(b?)",
                                              "VPSRLVWZrm(b?)",
                                              "VPSRLWYrm",
                                              "VPSRLWZ256mi(b?)",
                                              "VPSRLWZ256rm(b?)",
                                              "VPSRLWZmi(b?)",
                                              "VPSRLWZrm(b?)",
                                              "VPSUBSBYrm",
                                              "VPSUBSBZ256rm(b?)",
                                              "VPSUBSBZrm(b?)",
                                              "VPSUBSWYrm",
                                              "VPSUBSWZ256rm(b?)",
                                              "VPSUBSWZrm(b?)",
                                              "VPSUBUSBYrm",
                                              "VPSUBUSBZ256rm(b?)",
                                              "VPSUBUSBZrm(b?)",
                                              "VPSUBUSWYrm",
                                              "VPSUBUSWZ256rm(b?)",
                                              "VPSUBUSWZrm(b?)")>;

def SKXWriteResGroup121 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup121], (instregex "VANDNPDYrm",
                                              "VANDNPDZ256rm(b?)",
                                              "VANDNPDZrm(b?)",
                                              "VANDNPSYrm",
                                              "VANDNPSZ256rm(b?)",
                                              "VANDNPSZrm(b?)",
                                              "VANDPDYrm",
                                              "VANDPDZ256rm(b?)",
                                              "VANDPDZrm(b?)",
                                              "VANDPSYrm",
                                              "VANDPSZ256rm(b?)",
                                              "VANDPSZrm(b?)",
                                              "VBLENDMPDZ256rm(b?)",
                                              "VBLENDMPDZrm(b?)",
                                              "VBLENDMPSZ256rm(b?)",
                                              "VBLENDMPSZrm(b?)",
                                              "VBLENDPDYrmi",
                                              "VBLENDPSYrmi",
                                              "VBROADCASTF32X2Z256m(b?)",
                                              "VBROADCASTF32X2Zm(b?)",
                                              "VBROADCASTF32X4Z256rm(b?)",
                                              "VBROADCASTF32X4rm(b?)",
                                              "VBROADCASTF32X8rm(b?)",
                                              "VBROADCASTF64X2Z128rm(b?)",
                                              "VBROADCASTF64X2rm(b?)",
                                              "VBROADCASTF64X4rm(b?)",
                                              "VBROADCASTI32X2Z256m(b?)",
                                              "VBROADCASTI32X2Zm(b?)",
                                              "VBROADCASTI32X4Z256rm(b?)",
                                              "VBROADCASTI32X4rm(b?)",
                                              "VBROADCASTI32X8rm(b?)",
                                              "VBROADCASTI64X2Z128rm(b?)",
                                              "VBROADCASTI64X2rm(b?)",
                                              "VBROADCASTI64X4rm(b?)",
                                              "VBROADCASTSDZ256m(b?)",
                                              "VBROADCASTSDZm(b?)",
                                              "VBROADCASTSSZ256m(b?)",
                                              "VBROADCASTSSZm(b?)",
                                              "VINSERTF32x4Z256rm(b?)",
                                              "VINSERTF32x4Zrm(b?)",
                                              "VINSERTF32x8Zrm(b?)",
                                              "VINSERTF64x2Z256rm(b?)",
                                              "VINSERTF64x2Zrm(b?)",
                                              "VINSERTF64x4Zrm(b?)",
                                              "VINSERTI32x4Z256rm(b?)",
                                              "VINSERTI32x4Zrm(b?)",
                                              "VINSERTI32x8Zrm(b?)",
                                              "VINSERTI64x2Z256rm(b?)",
                                              "VINSERTI64x2Zrm(b?)",
                                              "VINSERTI64x4Zrm(b?)",
                                              "VMASKMOVPDYrm",
                                              "VMASKMOVPSYrm",
                                              "VMOVAPDZ256rm(b?)",
                                              "VMOVAPDZrm(b?)",
                                              "VMOVAPSZ256rm(b?)",
                                              "VMOVAPSZrm(b?)",
                                              "VMOVDDUPZ256rm(b?)",
                                              "VMOVDDUPZrm(b?)",
                                              "VMOVDQA32Z256rm(b?)",
                                              "VMOVDQA32Zrm(b?)",
                                              "VMOVDQA64Z256rm(b?)",
                                              "VMOVDQA64Zrm(b?)",
                                              "VMOVDQU16Z256rm(b?)",
                                              "VMOVDQU16Zrm(b?)",
                                              "VMOVDQU32Z256rm(b?)",
                                              "VMOVDQU32Zrm(b?)",
                                              "VMOVDQU64Z256rm(b?)",
                                              "VMOVDQU64Zrm(b?)",
                                              "VMOVDQU8Z256rm(b?)",
                                              "VMOVDQU8Zrm(b?)",
                                              "VMOVNTDQAZ256rm(b?)",
                                              "VMOVSHDUPZ256rm(b?)",
                                              "VMOVSHDUPZrm(b?)",
                                              "VMOVSLDUPZ256rm(b?)",
                                              "VMOVSLDUPZrm(b?)",
                                              "VMOVUPDZ256rm(b?)",
                                              "VMOVUPDZrm(b?)",
                                              "VMOVUPSZ256rm(b?)",
                                              "VMOVUPSZrm(b?)",
                                              "VORPDYrm",
                                              "VORPDZ256rm(b?)",
                                              "VORPDZrm(b?)",
                                              "VORPSYrm",
                                              "VORPSZ256rm(b?)",
                                              "VORPSZrm(b?)",
                                              "VPADDBYrm",
                                              "VPADDBZ256rm(b?)",
                                              "VPADDBZrm(b?)",
                                              "VPADDDYrm",
                                              "VPADDDZ256rm(b?)",
                                              "VPADDDZrm(b?)",
                                              "VPADDQYrm",
                                              "VPADDQZ256rm(b?)",
                                              "VPADDQZrm(b?)",
                                              "VPADDWYrm",
                                              "VPADDWZ256rm(b?)",
                                              "VPADDWZrm(b?)",
                                              "VPANDDZ256rm(b?)",
                                              "VPANDDZrm(b?)",
                                              "VPANDNDZ256rm(b?)",
                                              "VPANDNDZrm(b?)",
                                              "VPANDNQZ256rm(b?)",
                                              "VPANDNQZrm(b?)",
                                              "VPANDNYrm",
                                              "VPANDQZ256rm(b?)",
                                              "VPANDQZrm(b?)",
                                              "VPANDYrm",
                                              "VPBLENDDYrmi",
                                              "VPBLENDMBZ256rm(b?)",
                                              "VPBLENDMBZrm(b?)",
                                              "VPBLENDMDZ256rm(b?)",
                                              "VPBLENDMDZrm(b?)",
                                              "VPBLENDMQZ256rm(b?)",
                                              "VPBLENDMQZrm(b?)",
                                              "VPBLENDMWZ256rm(b?)",
                                              "VPBLENDMWZrm(b?)",
                                              "VPBROADCASTDZ256m(b?)",
                                              "VPBROADCASTDZm(b?)",
                                              "VPBROADCASTQZ256m(b?)",
                                              "VPBROADCASTQZm(b?)",
                                              "VPMASKMOVDYrm",
                                              "VPMASKMOVQYrm",
                                              "VPORDZ256rm(b?)",
                                              "VPORDZrm(b?)",
                                              "VPORQZ256rm(b?)",
                                              "VPORQZrm(b?)",
                                              "VPORYrm",
                                              "VPSUBBYrm",
                                              "VPSUBBZ256rm(b?)",
                                              "VPSUBBZrm(b?)",
                                              "VPSUBDYrm",
                                              "VPSUBDZ256rm(b?)",
                                              "VPSUBDZrm(b?)",
                                              "VPSUBQYrm",
                                              "VPSUBQZ256rm(b?)",
                                              "VPSUBQZrm(b?)",
                                              "VPSUBWYrm",
                                              "VPSUBWZrm(b?)",
                                              "VPTERNLOGDZ256rm(b?)i",
                                              "VPTERNLOGDZrm(b?)i",
                                              "VPTERNLOGQZ256rm(b?)i",
                                              "VPTERNLOGQZrm(b?)i",
                                              "VPXORDZ256rm(b?)",
                                              "VPXORDZrm(b?)",
                                              "VPXORQZ256rm(b?)",
                                              "VPXORQZrm(b?)",
                                              "VPXORYrm",
                                              "VXORPDYrm",
                                              "VXORPDZ256rm(b?)",
                                              "VXORPDZrm(b?)",
                                              "VXORPSYrm",
                                              "VXORPSZ256rm(b?)",
                                              "VXORPSZrm(b?)")>;

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

def SKXWriteResGroup130 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKXWriteResGroup130], (instregex "ADC(8|16|32|64)mi",
                                              "CMPXCHG(8|16|32|64)rm",
                                              "SBB(8|16|32|64)mi")>;
def: InstRW<[SKXWriteResGroup130, ReadAfterLd], (instrs ADC8mr, ADC16mr, ADC32mr, ADC64mr,
                                                        SBB8mr, SBB16mr, SBB32mr, SBB64mr)>;

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
                                              "VALIGNDZ128rm(b?)i",
                                              "VALIGNQZ128rm(b?)i",
                                              "VCMPPDZ128rm(b?)i",
                                              "VCMPPSZ128rm(b?)i",
                                              "VCMPSDZrm",
                                              "VCMPSSZrm",
                                              "VDBPSADBWZ128rmi(b?)",
                                              "VFPCLASSSSrm(b?)",
                                              "VPCMPBZ128rmi(b?)",
                                              "VPCMPDZ128rmi(b?)",
                                              "VPCMPEQBZ128rm(b?)",
                                              "VPCMPEQDZ128rm(b?)",
                                              "VPCMPEQQZ128rm(b?)",
                                              "VPCMPEQWZ128rm(b?)",
                                              "VPCMPGTBZ128rm(b?)",
                                              "VPCMPGTDZ128rm(b?)",
                                              "VPCMPGTQZ128rm(b?)",
                                              "VPCMPGTQrm",
                                              "VPCMPGTWZ128rm(b?)",
                                              "VPCMPQZ128rmi(b?)",
                                              "VPCMPUBZ128rmi(b?)",
                                              "VPCMPUDZ128rmi(b?)",
                                              "VPCMPUQZ128rmi(b?)",
                                              "VPCMPUWZ128rmi(b?)",
                                              "VPCMPWZ128rmi(b?)",
                                              "VPERMI2D128rm(b?)",
                                              "VPERMI2PD128rm(b?)",
                                              "VPERMI2PS128rm(b?)",
                                              "VPERMI2Q128rm(b?)",
                                              "VPERMT2D128rm(b?)",
                                              "VPERMT2PD128rm(b?)",
                                              "VPERMT2PS128rm(b?)",
                                              "VPERMT2Q128rm(b?)",
                                              "VPMAXSQZ128rm(b?)",
                                              "VPMAXUQZ128rm(b?)",
                                              "VPMINSQZ128rm(b?)",
                                              "VPMINUQZ128rm(b?)",
                                              "VPMOVSXBDZ128rm(b?)",
                                              "VPMOVSXBQZ128rm(b?)",
                                              "VPMOVSXBWYrm",
                                              "VPMOVSXBWZ128rm(b?)",
                                              "VPMOVSXDQYrm",
                                              "VPMOVSXDQZ128rm(b?)",
                                              "VPMOVSXWDYrm",
                                              "VPMOVSXWDZ128rm(b?)",
                                              "VPMOVSXWQZ128rm(b?)",
                                              "VPMOVZXBDZ128rm(b?)",
                                              "VPMOVZXBQZ128rm(b?)",
                                              "VPMOVZXBWZ128rm(b?)",
                                              "VPMOVZXDQZ128rm(b?)",
                                              "VPMOVZXWDYrm",
                                              "VPMOVZXWDZ128rm(b?)",
                                              "VPMOVZXWQZ128rm(b?)",
                                              "VPSADBWZ128rm(b?)",
                                              "VPSADBWrm",
                                              "VPTESTMBZ128rm(b?)",
                                              "VPTESTMDZ128rm(b?)",
                                              "VPTESTMQZ128rm(b?)",
                                              "VPTESTMWZ128rm(b?)",
                                              "VPTESTNMBZ128rm(b?)",
                                              "VPTESTNMDZ128rm(b?)",
                                              "VPTESTNMQZ128rm(b?)",
                                              "VPTESTNMWZ128rm(b?)")>;

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
def: InstRW<[SKXWriteResGroup138], (instregex "VRCP14PDZr(b?)",
                                              "VRCP14PSZr(b?)",
                                              "VRSQRT14PDZr(b?)",
                                              "VRSQRT14PSZr(b?)")>;

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
                                              "VRCP14PDZ128m(b?)",
                                              "VRCP14PSZ128m(b?)",
                                              "VRCP14SDrm(b?)",
                                              "VRCP14SSrm(b?)",
                                              "VRCPPSm",
                                              "VRSQRT14PDZ128m(b?)",
                                              "VRSQRT14PSZ128m(b?)",
                                              "VRSQRT14SDrm(b?)",
                                              "VRSQRT14SSrm(b?)",
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
                                              "VALIGNDZ256rm(b?)i",
                                              "VALIGNDZrm(b?)i",
                                              "VALIGNQZ256rm(b?)i",
                                              "VALIGNQZrm(b?)i",
                                              "VCMPPDZ256rm(b?)i",
                                              "VCMPPDZrm(b?)i",
                                              "VCMPPSZ256rm(b?)i",
                                              "VCMPPSZrm(b?)i",
                                              "VDBPSADBWZ256rmi(b?)",
                                              "VDBPSADBWZrmi(b?)",
                                              "VPCMPBZ256rmi(b?)",
                                              "VPCMPBZrmi(b?)",
                                              "VPCMPDZ256rmi(b?)",
                                              "VPCMPDZrmi(b?)",
                                              "VPCMPEQBZ256rm(b?)",
                                              "VPCMPEQBZrm(b?)",
                                              "VPCMPEQDZ256rm(b?)",
                                              "VPCMPEQDZrm(b?)",
                                              "VPCMPEQQZ256rm(b?)",
                                              "VPCMPEQQZrm(b?)",
                                              "VPCMPEQWZ256rm(b?)",
                                              "VPCMPEQWZrm(b?)",
                                              "VPCMPGTBZ256rm(b?)",
                                              "VPCMPGTBZrm(b?)",
                                              "VPCMPGTDZ256rm(b?)",
                                              "VPCMPGTDZrm(b?)",
                                              "VPCMPGTQYrm",
                                              "VPCMPGTQZ256rm(b?)",
                                              "VPCMPGTQZrm(b?)",
                                              "VPCMPGTWZ256rm(b?)",
                                              "VPCMPGTWZrm(b?)",
                                              "VPCMPQZ256rmi(b?)",
                                              "VPCMPQZrmi(b?)",
                                              "VPCMPUBZ256rmi(b?)",
                                              "VPCMPUBZrmi(b?)",
                                              "VPCMPUDZ256rmi(b?)",
                                              "VPCMPUDZrmi(b?)",
                                              "VPCMPUQZ256rmi(b?)",
                                              "VPCMPUQZrmi(b?)",
                                              "VPCMPUWZ256rmi(b?)",
                                              "VPCMPUWZrmi(b?)",
                                              "VPCMPWZ256rmi(b?)",
                                              "VPCMPWZrmi(b?)",
                                              "VPERM2F128rm",
                                              "VPERM2I128rm",
                                              "VPERMDYrm",
                                              "VPERMDZ256rm(b?)",
                                              "VPERMDZrm(b?)",
                                              "VPERMI2D256rm(b?)",
                                              "VPERMI2Drm(b?)",
                                              "VPERMI2PD256rm(b?)",
                                              "VPERMI2PDrm(b?)",
                                              "VPERMI2PS256rm(b?)",
                                              "VPERMI2PSrm(b?)",
                                              "VPERMI2Q256rm(b?)",
                                              "VPERMI2Qrm(b?)",
                                              "VPERMPDYmi",
                                              "VPERMPDZ256m(b?)i",
                                              "VPERMPDZ256rm(b?)",
                                              "VPERMPDZm(b?)i",
                                              "VPERMPDZrm(b?)",
                                              "VPERMPSYrm",
                                              "VPERMPSZ256rm(b?)",
                                              "VPERMPSZrm(b?)",
                                              "VPERMQYmi",
                                              "VPERMQZ256m(b?)i",
                                              "VPERMQZ256rm(b?)",
                                              "VPERMQZm(b?)i",
                                              "VPERMQZrm(b?)",
                                              "VPERMT2D256rm(b?)",
                                              "VPERMT2Drm(b?)",
                                              "VPERMT2PD256rm(b?)",
                                              "VPERMT2PDrm(b?)",
                                              "VPERMT2PS256rm(b?)",
                                              "VPERMT2PSrm(b?)",
                                              "VPERMT2Q256rm(b?)",
                                              "VPERMT2Qrm(b?)",
                                              "VPMAXSQZ256rm(b?)",
                                              "VPMAXSQZrm(b?)",
                                              "VPMAXUQZ256rm(b?)",
                                              "VPMAXUQZrm(b?)",
                                              "VPMINSQZ256rm(b?)",
                                              "VPMINSQZrm(b?)",
                                              "VPMINUQZ256rm(b?)",
                                              "VPMINUQZrm(b?)",
                                              "VPMOVSXBDZ256rm(b?)",
                                              "VPMOVSXBDZrm(b?)",
                                              "VPMOVSXBQZ256rm(b?)",
                                              "VPMOVSXBQZrm(b?)",
                                              "VPMOVSXBWZ256rm(b?)",
                                              "VPMOVSXBWZrm(b?)",
                                              "VPMOVSXDQZ256rm(b?)",
                                              "VPMOVSXDQZrm(b?)",
                                              "VPMOVSXWDZ256rm(b?)",
                                              "VPMOVSXWDZrm(b?)",
                                              "VPMOVSXWQZ256rm(b?)",
                                              "VPMOVSXWQZrm(b?)",
                                              "VPMOVZXBDYrm",
                                              "VPMOVZXBDZ256rm(b?)",
                                              "VPMOVZXBDZrm(b?)",
                                              "VPMOVZXBQYrm",
                                              "VPMOVZXBQZ256rm(b?)",
                                              "VPMOVZXBQZrm(b?)",
                                              "VPMOVZXBWYrm",
                                              "VPMOVZXBWZ256rm(b?)",
                                              "VPMOVZXBWZrm(b?)",
                                              "VPMOVZXDQYrm",
                                              "VPMOVZXDQZ256rm(b?)",
                                              "VPMOVZXDQZrm(b?)",
                                              "VPMOVZXWDZ256rm(b?)",
                                              "VPMOVZXWDZrm(b?)",
                                              "VPMOVZXWQYrm",
                                              "VPMOVZXWQZ256rm(b?)",
                                              "VPMOVZXWQZrm(b?)",
                                              "VPSADBWYrm",
                                              "VPSADBWZ256rm(b?)",
                                              "VPSADBWZrm(b?)",
                                              "VPTESTMBZ256rm(b?)",
                                              "VPTESTMBZrm(b?)",
                                              "VPTESTMDZ256rm(b?)",
                                              "VPTESTMDZrm(b?)",
                                              "VPTESTMQZ256rm(b?)",
                                              "VPTESTMQZrm(b?)",
                                              "VPTESTMWZ256rm(b?)",
                                              "VPTESTMWZrm(b?)",
                                              "VPTESTNMBZ256rm(b?)",
                                              "VPTESTNMBZrm(b?)",
                                              "VPTESTNMDZ256rm(b?)",
                                              "VPTESTNMDZrm(b?)",
                                              "VPTESTNMQZ256rm(b?)",
                                              "VPTESTNMQZrm(b?)",
                                              "VPTESTNMWZ256rm(b?)",
                                              "VPTESTNMWZrm(b?)",
                                              "VSHUFF32X4Z256rm(b?)i",
                                              "VSHUFF32X4Zrm(b?)i",
                                              "VSHUFF64X2Z256rm(b?)i",
                                              "VSHUFF64X2Zrm(b?)i",
                                              "VSHUFI32X4Z256rm(b?)i",
                                              "VSHUFI32X4Zrm(b?)i",
                                              "VSHUFI64X2Z256rm(b?)i",
                                              "VSHUFI64X2Zrm(b?)i")>;

def SKXWriteResGroup149 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup149],
          (instregex
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Z128m(b?)",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)m",
           "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)Zm")>;
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
                                              "VADDPDZ128rm(b?)",
                                              "VADDPDrm",
                                              "VADDPSZ128rm(b?)",
                                              "VADDPSrm",
                                              "VADDSDZrm",
                                              "VADDSSZrm",
                                              "VADDSUBPDrm",
                                              "VADDSUBPSrm",
                                              "VCMPPDrmi",
                                              "VCMPPSrmi",
                                              "VCVTDQ2PDZ128rm(b?)",
                                              "VCVTDQ2PSZ128rm(b?)",
                                              "VCVTDQ2PSrm",
                                              "VCVTPD2QQZ128rm(b?)",
                                              "VCVTPD2UQQZ128rm(b?)",
                                              "VCVTPH2PSYrm",
                                              "VCVTPH2PSZ128rm(b?)",
                                              "VCVTPS2DQZ128rm(b?)",
                                              "VCVTPS2DQrm",
                                              "VCVTPS2PDZ128rm(b?)",
                                              "VCVTPS2QQZ128rm(b?)",
                                              "VCVTPS2UDQZ128rm(b?)",
                                              "VCVTPS2UQQZ128rm(b?)",
                                              "VCVTQQ2PDZ128rm(b?)",
                                              "VCVTQQ2PSZ128rm(b?)",
                                              "VCVTSS2SDZrm",
                                              "VCVTSS2SDrm",
                                              "VCVTTPD2QQZ128rm(b?)",
                                              "VCVTTPD2UQQZ128rm(b?)",
                                              "VCVTTPS2DQZ128rm(b?)",
                                              "VCVTTPS2DQrm",
                                              "VCVTTPS2QQZ128rm(b?)",
                                              "VCVTTPS2UDQZ128rm(b?)",
                                              "VCVTTPS2UQQZ128rm(b?)",
                                              "VCVTUDQ2PDZ128rm(b?)",
                                              "VCVTUDQ2PSZ128rm(b?)",
                                              "VCVTUQQ2PDZ128rm(b?)",
                                              "VCVTUQQ2PSZ128rm(b?)",
                                              "VFIXUPIMMPDZ128rm(b?)i",
                                              "VFIXUPIMMPSZ128rm(b?)i",
                                              "VFIXUPIMMSDrmi(b?)",
                                              "VFIXUPIMMSSrmi(b?)",
                                              "VGETEXPPDZ128m(b?)",
                                              "VGETEXPPSZ128m(b?)",
                                              "VGETEXPSDm(b?)",
                                              "VGETEXPSSm(b?)",
                                              "VGETMANTPDZ128rm(b?)i",
                                              "VGETMANTPSZ128rm(b?)i",
                                              "VGETMANTSDZ128rmi(b?)",
                                              "VGETMANTSSZ128rmi(b?)",
                                              "VMAX(C?)PDZ128rm(b?)",
                                              "VMAX(C?)PDrm",
                                              "VMAX(C?)PSZ128rm(b?)",
                                              "VMAX(C?)PSrm",
                                              "VMAX(C?)SDZrm",
                                              "VMAX(C?)SSZrm",
                                              "VMIN(C?)PDZ128rm(b?)",
                                              "VMIN(C?)PDrm",
                                              "VMIN(C?)PSZ128rm(b?)",
                                              "VMIN(C?)PSrm",
                                              "VMIN(C?)SDZrm",
                                              "VMIN(C?)SSZrm",
                                              "VMULPDZ128rm(b?)",
                                              "VMULPDrm",
                                              "VMULPSZ128rm(b?)",
                                              "VMULPSrm",
                                              "VMULSDZrm",
                                              "VMULSSZrm",
                                              "VPHMINPOSUWrm",
                                              "VPLZCNTDZ128rm(b?)",
                                              "VPLZCNTQZ128rm(b?)",
                                              "VPMADDUBSWZ128rm(b?)",
                                              "VPMADDUBSWrm",
                                              "VPMADDWDZ128rm(b?)",
                                              "VPMADDWDrm",
                                              "VPMULDQZ128rm(b?)",
                                              "VPMULDQrm",
                                              "VPMULHRSWZ128rm(b?)",
                                              "VPMULHRSWrm",
                                              "VPMULHUWZ128rm(b?)",
                                              "VPMULHUWrm",
                                              "VPMULHWZ128rm(b?)",
                                              "VPMULHWrm",
                                              "VPMULLWZ128rm(b?)",
                                              "VPMULLWrm",
                                              "VPMULUDQZ128rm(b?)",
                                              "VPMULUDQrm",
                                              "VRANGEPDZ128rm(b?)i",
                                              "VRANGEPSZ128rm(b?)i",
                                              "VRANGESDZ128rmi(b?)",
                                              "VRANGESSZ128rmi(b?)",
                                              "VREDUCEPDZ128rm(b?)i",
                                              "VREDUCEPSZ128rm(b?)i",
                                              "VREDUCESDZ128rmi(b?)",
                                              "VREDUCESSZ128rmi(b?)",
                                              "VSCALEFPDZ128rm(b?)",
                                              "VSCALEFPSZ128rm(b?)",
                                              "VSCALEFSDZ128rm(b?)",
                                              "VSCALEFSSZ128rm(b?)",
                                              "VSUBPDZ128rm(b?)",
                                              "VSUBPDrm",
                                              "VSUBPSZ128rm(b?)",
                                              "VSUBPSrm",
                                              "VSUBSDZrm",
                                              "VSUBSSZrm")>;

def SKXWriteResGroup151 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup151], (instregex "MPSADBWrmi",
                                              "VEXPANDPDZ128rm(b?)",
                                              "VEXPANDPSZ128rm(b?)",
                                              "VMPSADBWrmi",
                                              "VPEXPANDDZ128rm(b?)",
                                              "VPEXPANDQZ128rm(b?)")>;

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

def SKXWriteResGroup159 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup159], (instregex "(V?)DIVPS(Z128)?rr",
                                              "(V?)DIVSS(Z?)rr")>;

def SKXWriteResGroup159_1 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKXWriteResGroup159_1], (instregex "VDIVPS(Y|Z256)rr")>;

def SKXWriteResGroup160 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup160], (instregex "MUL_F32m",
                                              "MUL_F64m",
                                              "VRCP14PDZ256m(b?)",
                                              "VRCP14PSZ256m(b?)",
                                              "VRCPPSYm",
                                              "VRSQRT14PDZ256m(b?)",
                                              "VRSQRT14PSZ256m(b?)",
                                              "VRSQRTPSYm")>;

def SKXWriteResGroup161 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup161],
          (instregex
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Ym",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Z256m(b?)",
           "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Zm(b?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VADDPDYrm",
                                              "VADDPDZ256rm(b?)",
                                              "VADDPDZrm(b?)",
                                              "VADDPSYrm",
                                              "VADDPSZ256rm(b?)",
                                              "VADDPSZrm(b?)",
                                              "VADDSUBPDYrm",
                                              "VADDSUBPSYrm",
                                              "VCMPPDYrmi",
                                              "VCMPPSYrmi",
                                              "VCVTDQ2PDZ256rm(b?)",
                                              "VCVTDQ2PDZrm(b?)",
                                              "VCVTDQ2PSYrm",
                                              "VCVTDQ2PSZ256rm(b?)",
                                              "VCVTDQ2PSZrm(b?)",
                                              "VCVTPD2QQZ256rm(b?)",
                                              "VCVTPD2QQZrm(b?)",
                                              "VCVTPD2UQQZ256rm(b?)",
                                              "VCVTPD2UQQZrm(b?)",
                                              "VCVTPH2PSZ256rm(b?)",
                                              "VCVTPH2PSZrm(b?)",
                                              "VCVTPS2DQYrm",
                                              "VCVTPS2DQZ256rm(b?)",
                                              "VCVTPS2DQZrm(b?)",
                                              "VCVTPS2PDYrm",
                                              "VCVTPS2PDZ256rm(b?)",
                                              "VCVTPS2PDZrm(b?)",
                                              "VCVTPS2QQZ256rm(b?)",
                                              "VCVTPS2UDQZ256rm(b?)",
                                              "VCVTPS2UDQZrm(b?)",
                                              "VCVTPS2UQQZ256rm(b?)",
                                              "VCVTQQ2PDZ256rm(b?)",
                                              "VCVTQQ2PDZrm(b?)",
                                              "VCVTQQ2PSZ256rm(b?)",
                                              "VCVTTPD2QQZ256rm(b?)",
                                              "VCVTTPD2QQZrm(b?)",
                                              "VCVTTPD2UQQZ256rm(b?)",
                                              "VCVTTPD2UQQZrm(b?)",
                                              "VCVTTPS2DQYrm",
                                              "VCVTTPS2DQZ256rm(b?)",
                                              "VCVTTPS2DQZrm(b?)",
                                              "VCVTTPS2QQZ256rm(b?)",
                                              "VCVTTPS2UDQZ256rm(b?)",
                                              "VCVTTPS2UDQZrm(b?)",
                                              "VCVTTPS2UQQZ256rm(b?)",
                                              "VCVTUDQ2PDZ256rm(b?)",
                                              "VCVTUDQ2PDZrm(b?)",
                                              "VCVTUDQ2PSZ256rm(b?)",
                                              "VCVTUDQ2PSZrm(b?)",
                                              "VCVTUQQ2PDZ256rm(b?)",
                                              "VCVTUQQ2PDZrm(b?)",
                                              "VCVTUQQ2PSZ256rm(b?)",
                                              "VFIXUPIMMPDZ256rm(b?)i",
                                              "VFIXUPIMMPDZrm(b?)i",
                                              "VFIXUPIMMPSZ256rm(b?)i",
                                              "VFIXUPIMMPSZrm(b?)i",
                                              "VGETEXPPDZ256m(b?)",
                                              "VGETEXPPDm(b?)",
                                              "VGETEXPPSZ256m(b?)",
                                              "VGETEXPPSm(b?)",
                                              "VGETMANTPDZ256rm(b?)i",
                                              "VGETMANTPDZrm(b?)i",
                                              "VGETMANTPSZ256rm(b?)i",
                                              "VGETMANTPSZrm(b?)i",
                                              "VMAX(C?)PDYrm",
                                              "VMAX(C?)PDZ256rm(b?)",
                                              "VMAX(C?)PDZrm(b?)",
                                              "VMAX(C?)PSYrm",
                                              "VMAX(C?)PSZ256rm(b?)",
                                              "VMAX(C?)PSZrm(b?)",
                                              "VMIN(C?)PDYrm",
                                              "VMIN(C?)PDZ256rm(b?)",
                                              "VMIN(C?)PDZrm(b?)",
                                              "VMIN(C?)PSYrm",
                                              "VMIN(C?)PSZ256rm(b?)",
                                              "VMIN(C?)PSZrm(b?)",
                                              "VMULPDYrm",
                                              "VMULPDZ256rm(b?)",
                                              "VMULPDZrm(b?)",
                                              "VMULPSYrm",
                                              "VMULPSZ256rm(b?)",
                                              "VMULPSZrm(b?)",
                                              "VPLZCNTDZ256rm(b?)",
                                              "VPLZCNTDZrm(b?)",
                                              "VPLZCNTQZ256rm(b?)",
                                              "VPLZCNTQZrm(b?)",
                                              "VPMADDUBSWYrm",
                                              "VPMADDUBSWZ256rm(b?)",
                                              "VPMADDUBSWZrm(b?)",
                                              "VPMADDWDYrm",
                                              "VPMADDWDZ256rm(b?)",
                                              "VPMADDWDZrm(b?)",
                                              "VPMULDQYrm",
                                              "VPMULDQZ256rm(b?)",
                                              "VPMULDQZrm(b?)",
                                              "VPMULHRSWYrm",
                                              "VPMULHRSWZ256rm(b?)",
                                              "VPMULHRSWZrm(b?)",
                                              "VPMULHUWYrm",
                                              "VPMULHUWZ256rm(b?)",
                                              "VPMULHUWZrm(b?)",
                                              "VPMULHWYrm",
                                              "VPMULHWZ256rm(b?)",
                                              "VPMULHWZrm(b?)",
                                              "VPMULLWYrm",
                                              "VPMULLWZ256rm(b?)",
                                              "VPMULLWZrm(b?)",
                                              "VPMULUDQYrm",
                                              "VPMULUDQZ256rm(b?)",
                                              "VPMULUDQZrm(b?)",
                                              "VRANGEPDZ256rm(b?)i",
                                              "VRANGEPDZrm(b?)i",
                                              "VRANGEPSZ256rm(b?)i",
                                              "VRANGEPSZrm(b?)i",
                                              "VREDUCEPDZ256rm(b?)i",
                                              "VREDUCEPDZrm(b?)i",
                                              "VREDUCEPSZ256rm(b?)i",
                                              "VREDUCEPSZrm(b?)i",
                                              "VSCALEFPDZ256rm(b?)",
                                              "VSCALEFPDZrm(b?)",
                                              "VSCALEFPSZ256rm(b?)",
                                              "VSCALEFPSZrm(b?)",
                                              "VSUBPDYrm",
                                              "VSUBPDZ256rm(b?)",
                                              "VSUBPDZrm(b?)",
                                              "VSUBPSYrm",
                                              "VSUBPSZ256rm(b?)",
                                              "VSUBPSZrm(b?)")>;

def SKXWriteResGroup162 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup162], (instregex "FICOM16m",
                                              "FICOM32m",
                                              "FICOMP16m",
                                              "FICOMP32m",
                                              "VEXPANDPDZ256rm(b?)",
                                              "VEXPANDPDZrm(b?)",
                                              "VEXPANDPSZ256rm(b?)",
                                              "VEXPANDPSZrm(b?)",
                                              "VMPSADBWYrmi",
                                              "VPEXPANDDZ256rm(b?)",
                                              "VPEXPANDDZrm(b?)",
                                              "VPEXPANDQZ256rm(b?)",
                                              "VPEXPANDQZrm(b?)")>;

def SKXWriteResGroup163 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup163], (instregex "VCVTSD2SSZrm")>;

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
                                              "VCVTSD2SI64Zrm(b?)",
                                              "VCVTSD2SI64rm",
                                              "VCVTSD2SIZrm(b?)",
                                              "VCVTSD2SIrm",
                                              "VCVTSD2USI64Zrm(b?)",
                                              "VCVTSS2SI64Zrm(b?)",
                                              "VCVTSS2SI64rm",
                                              "VCVTSS2SIZrm(b?)",
                                              "VCVTSS2SIrm",
                                              "VCVTSS2USIZrm(b?)",
                                              "VCVTTSD2SI64Zrm(b?)",
                                              "VCVTTSD2SI64rm",
                                              "VCVTTSD2SIZrm(b?)",
                                              "VCVTTSD2SIrm",
                                              "VCVTTSD2USI64Zrm(b?)",
                                              "VCVTTSS2SI64Zrm(b?)",
                                              "VCVTTSS2SI64rm",
                                              "VCVTTSS2SIZrm(b?)",
                                              "VCVTTSS2SIrm",
                                              "VCVTTSS2USIZrm(b?)")>;

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
def: InstRW<[SKXWriteResGroup167], (instregex "VPCONFLICTQZ128rm(b?)")>;

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

def SKXWriteResGroup172 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup172], (instregex "(V?)SQRTPS(Z128)?r",
                                              "(V?)SQRTSS(Z?)r")>;

def SKXWriteResGroup173 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1,6];
}
def: InstRW<[SKXWriteResGroup173], (instregex "VSQRTPS(Y|Z256)r")>;

def SKXWriteResGroup174 : SchedWriteRes<[SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKXWriteResGroup174], (instregex "VPMULLQZ128rr",
                                              "VPMULLQZ256rr",
                                              "VPMULLQZrr")>;

def SKXWriteResGroup175 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup175], (instregex "VPERMWZ128rm(b?)")>;

def SKXWriteResGroup176 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup176], (instregex "VCVTSD2USIZrm(b?)",
                                              "VCVTSS2USI64Zrm(b?)",
                                              "VCVTTSD2USIZrm(b?)",
                                              "VCVTTSS2USI64Zrm(b?)")>;

def SKXWriteResGroup177 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup177], (instregex "VCVTPS2QQZrm(b?)",
                                              "VCVTPS2UQQZrm(b?)",
                                              "VCVTTPS2QQZrm(b?)",
                                              "VCVTTPS2UQQZrm(b?)")>;

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
                                              "VPERMWZ256rm(b?)",
                                              "VPERMWZrm(b?)")>;

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
                                              "VPERMI2W128rm(b?)",
                                              "VPERMT2W128rm(b?)")>;

def SKXWriteResGroup184 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup184], (instregex "(V?)DIVPDrr",
                                              "(V?)DIVSD(Z?)rr")>;

def SKXWriteResGroup184_1 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKXWriteResGroup184_1], (instregex "VDIVPD(Y|Z256)rr")>;

def SKXWriteResGroup186 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup186], (instregex "ROUNDPDm",
                                              "ROUNDPSm",
                                              "ROUNDSDm",
                                              "ROUNDSSm",
                                              "VRNDSCALEPDZ128rm(b?)i",
                                              "VRNDSCALEPSZ128rm(b?)i",
                                              "VRNDSCALESDm(b?)",
                                              "VRNDSCALESSm(b?)",
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
                                                "VPMULLDZ128rm(b?)",
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
def: InstRW<[SKXWriteResGroup188], (instregex "VCVTPD2DQZrm(b?)",
                                              "VCVTPD2PSZrm(b?)",
                                              "VCVTPD2UDQZrm(b?)",
                                              "VCVTQQ2PSZrm(b?)",
                                              "VCVTTPD2DQZrm(b?)",
                                              "VCVTTPD2UDQZrm(b?)",
                                              "VCVTUQQ2PSZrm(b?)")>;

def SKXWriteResGroup189 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup189], (instregex "VPERMI2W256rm(b?)",
                                              "VPERMI2Wrm(b?)",
                                              "VPERMT2W256rm(b?)",
                                              "VPERMT2Wrm(b?)")>;

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
def: InstRW<[SKXWriteResGroup192], (instregex "VRNDSCALEPDZ256rm(b?)i",
                                              "VRNDSCALEPDZrm(b?)i",
                                              "VRNDSCALEPSZ256rm(b?)i",
                                              "VRNDSCALEPSZrm(b?)i",
                                              "VROUNDPDYm",
                                              "VROUNDPSYm")>;

def SKXWriteResGroup192_2 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 17;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup192_2], (instregex "VPMULLDYrm",
                                                "VPMULLDZ256rm(b?)",
                                                "VPMULLDZrm(b?)")>;

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
def: InstRW<[SKXWriteResGroup194], (instregex "VPCONFLICTDZ128rm(b?)")>;

def SKXWriteResGroup195 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 15;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,5,1,1];
}
def: InstRW<[SKXWriteResGroup195], (instregex "RCL(8|16|32|64)mCL")>;

def SKXWriteResGroup196 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup196], (instregex "(V?)DIVSS(Z?)rm")>;

def SKXWriteResGroup198 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup198], (instregex "VRCP14PDZm(b?)",
                                              "VRCP14PSZm(b?)",
                                              "VRSQRT14PDZm(b?)",
                                              "VRSQRT14PSZm(b?)")>;

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

def SKXWriteResGroup201 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 17;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,5];
}
def: InstRW<[SKXWriteResGroup201], (instregex "(V?)DIVPS(Z128)?rm")>;

def SKXWriteResGroup201_1 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 17;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup201_1], (instregex "(V?)SQRTSS(Z?)m")>;

def SKXWriteResGroup202 : SchedWriteRes<[SKXPort0,SKXPort1,SKXPort5,SKXPort6,SKXPort05,SKXPort0156]> {
  let Latency = 17;
  let NumMicroOps = 15;
  let ResourceCycles = [2,1,2,4,2,4];
}
def: InstRW<[SKXWriteResGroup202], (instregex "XCH_F")>;

def SKXWriteResGroup203 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 1;
  let ResourceCycles = [1,6];
}
def: InstRW<[SKXWriteResGroup203], (instregex "(V?)SQRTPD(Z128)?r",
                                              "(V?)SQRTSD(Z?)r")>;

def SKXWriteResGroup203_1 : SchedWriteRes<[SKXPort0,SKXFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 1;
  let ResourceCycles = [1,12];
}
def: InstRW<[SKXWriteResGroup203_1], (instregex "VSQRTPD(Y|Z256)r")>;

def SKXWriteResGroup204 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,5];
}
def: InstRW<[SKXWriteResGroup204], (instregex "VDIVPS(Y|Z256)rm")>;

def SKXWriteResGroup204_1 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup204_1], (instregex "(V?)SQRTPS(Z128)?m")>;

def SKXWriteResGroup205 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 18;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup205], (instregex "VPMULLQZ128rm(b?)")>;

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

def SKXWriteResGroup209 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKXWriteResGroup209], (instregex "(V?)DIVSD(Z?)rm")>;

def SKXWriteResGroup209_1 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,6];
}
def: InstRW<[SKXWriteResGroup209_1], (instregex "VSQRTPS(Y|Z256)m")>;

def SKXWriteResGroup210 : SchedWriteRes<[SKXPort0,SKXPort5,SKXFPDivider]> {
  let Latency = 20;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,12];
}
def: InstRW<[SKXWriteResGroup210], (instregex "VSQRTPSZr")>;

def SKXWriteResGroup211 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 19;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup211], (instregex "VPMULLQZ256rm(b?)",
                                              "VPMULLQZrm(b?)")>;

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

def SKXWriteResGroup216 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKXWriteResGroup216], (instregex "(V?)DIVPD(Z128)?rm")>;

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

def SKXWriteResGroup222 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,8];
}
def: InstRW<[SKXWriteResGroup222], (instregex "VDIVPD(Y|Z256)rm")>;

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
def: InstRW<[SKXWriteResGroup225], (instregex "VPCONFLICTDZ128rr",
                                              "VPCONFLICTQZ256rr")>;

def SKXWriteResGroup226 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 23;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,6];
}
def: InstRW<[SKXWriteResGroup226], (instregex "(V?)SQRTSD(Z?)m")>;

def SKXWriteResGroup227 : SchedWriteRes<[SKXPort0,SKXPort5,SKXFPDivider]> {
  let Latency = 23;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,16];
}
def: InstRW<[SKXWriteResGroup227], (instregex "VDIVPDZrr")>;

def SKXWriteResGroup227_1 : SchedWriteRes<[SKXPort0,SKXPort5,SKXFPDivider]> {
  let Latency = 18;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,10];
}
def: InstRW<[SKXWriteResGroup227_1], (instregex "VDIVPSZrr")>;

def SKXWriteResGroup228 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 23;
  let NumMicroOps = 19;
  let ResourceCycles = [2,1,4,1,1,4,6];
}
def: InstRW<[SKXWriteResGroup228], (instregex "CMPXCHG16B")>;

def SKXWriteResGroup229 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 24;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,6];
}
def: InstRW<[SKXWriteResGroup229], (instregex "(V?)SQRTPD(Z128)?m")>;

def SKXWriteResGroup230 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort5,SKXFPDivider]> {
  let Latency = 25;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,10];
}
def: InstRW<[SKXWriteResGroup230], (instregex "VDIVPSZrm(b?)")>;

def SKXWriteResGroup232 : SchedWriteRes<[SKXPort0,SKXPort23,SKXFPDivider]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,12];
}
def: InstRW<[SKXWriteResGroup232], (instregex "VSQRTPD(Y|Z256)m")>;

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

def SKXWriteResGroup237 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort5,SKXFPDivider]> {
  let Latency = 27;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,12];
}
def: InstRW<[SKXWriteResGroup237], (instregex "VSQRTPSZm(b?)")>;

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
def: InstRW<[SKXWriteResGroup242], (instregex "VPCONFLICTQZ256rm(b?)")>;

def SKXWriteResGroup243 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 30;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup243], (instregex "DIVR_FI16m",
                                              "DIVR_FI32m")>;

def SKXWriteResGroup244 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort5,SKXFPDivider]> {
  let Latency = 30;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,16];
}
def: InstRW<[SKXWriteResGroup244], (instregex "VDIVPDZrm(b?)")>;

def SKXWriteResGroup245 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 30;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup245], (instrs VGATHERDPSZrm,
                                           VPGATHERDDZrm)>;

def SKXWriteResGroup246 : SchedWriteRes<[SKXPort0,SKXPort5,SKXFPDivider]> {
  let Latency = 32;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,24];
}
def: InstRW<[SKXWriteResGroup246], (instregex "VSQRTPDZr")>;

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
def: InstRW<[SKXWriteResGroup249], (instregex "VPCONFLICTDZ256rr",
                                              "VPCONFLICTQZrr")>;

def SKXWriteResGroup250 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort23,SKXPort0156]> {
  let Latency = 37;
  let NumMicroOps = 31;
  let ResourceCycles = [1,8,1,21];
}
def: InstRW<[SKXWriteResGroup250], (instregex "XRSTOR(64)?")>;

def SKXWriteResGroup251 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort5,SKXFPDivider]> {
  let Latency = 39;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,24];
}
def: InstRW<[SKXWriteResGroup251], (instregex "VSQRTPDZm(b?)")>;

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
def: InstRW<[SKXWriteResGroup256], (instregex "VPCONFLICTDZ256rm(b?)",
                                              "VPCONFLICTQZrm(b?)")>;

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
def: InstRW<[SKXWriteResGroup261], (instregex "VPCONFLICTDZrr")>;

def SKXWriteResGroup262 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23,SKXPort015]> {
  let Latency = 74;
  let NumMicroOps = 36;
  let ResourceCycles = [17,11,1,7];
}
def: InstRW<[SKXWriteResGroup262], (instregex "VPCONFLICTDZrm(b?)")>;

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
