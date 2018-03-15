//=- X86SchedSkylake.td - X86 Skylake Client Scheduling ------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Skylake Client to support
// instruction scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def SkylakeClientModel : SchedMachineModel {
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

let SchedModel = SkylakeClientModel in {

// Skylake Client can issue micro-ops to 8 different ports in one cycle.

// Ports 0, 1, 5, and 6 handle all computation.
// Port 4 gets the data half of stores. Store data can be available later than
// the store address, but since we don't model the latency of stores, we can
// ignore that.
// Ports 2 and 3 are identical. They handle loads and the address half of
// stores. Port 7 can handle address calculations.
def SKLPort0 : ProcResource<1>;
def SKLPort1 : ProcResource<1>;
def SKLPort2 : ProcResource<1>;
def SKLPort3 : ProcResource<1>;
def SKLPort4 : ProcResource<1>;
def SKLPort5 : ProcResource<1>;
def SKLPort6 : ProcResource<1>;
def SKLPort7 : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def SKLPort01  : ProcResGroup<[SKLPort0, SKLPort1]>;
def SKLPort23  : ProcResGroup<[SKLPort2, SKLPort3]>;
def SKLPort237 : ProcResGroup<[SKLPort2, SKLPort3, SKLPort7]>;
def SKLPort04  : ProcResGroup<[SKLPort0, SKLPort4]>;
def SKLPort05  : ProcResGroup<[SKLPort0, SKLPort5]>;
def SKLPort06  : ProcResGroup<[SKLPort0, SKLPort6]>;
def SKLPort15  : ProcResGroup<[SKLPort1, SKLPort5]>;
def SKLPort16  : ProcResGroup<[SKLPort1, SKLPort6]>;
def SKLPort56  : ProcResGroup<[SKLPort5, SKLPort6]>;
def SKLPort015 : ProcResGroup<[SKLPort0, SKLPort1, SKLPort5]>;
def SKLPort056 : ProcResGroup<[SKLPort0, SKLPort5, SKLPort6]>;
def SKLPort0156: ProcResGroup<[SKLPort0, SKLPort1, SKLPort5, SKLPort6]>;

// 60 Entry Unified Scheduler
def SKLPortAny : ProcResGroup<[SKLPort0, SKLPort1, SKLPort2, SKLPort3, SKLPort4,
                              SKLPort5, SKLPort6, SKLPort7]> {
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
multiclass SKLWriteResPair<X86FoldableSchedWrite SchedRW,
                          ProcResourceKind ExePort,
                          int Lat> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, [ExePort]> { let Latency = Lat; }

  // Memory variant also uses a cycle on port 2/3 and adds 5 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, [SKLPort23, ExePort]> {
     let Latency = !add(Lat, 5);
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [SKLPort4]>;

// Arithmetic.
defm : SKLWriteResPair<WriteALU,   SKLPort0156, 1>; // Simple integer ALU op.
defm : SKLWriteResPair<WriteIMul,  SKLPort1,   3>; // Integer multiplication.
def : WriteRes<WriteIMulH, []> { let Latency = 3; } // Integer multiplication, high part.
def SKLDivider : ProcResource<1>; // Integer division issued on port 0.     
def : WriteRes<WriteIDiv, [SKLPort0, SKLDivider]> { // Integer division.
  let Latency = 25;
  let ResourceCycles = [1, 10];
}
def : WriteRes<WriteIDivLd, [SKLPort23, SKLPort0, SKLDivider]> {
  let Latency = 29;
  let ResourceCycles = [1, 1, 10];
}

def : WriteRes<WriteLEA, [SKLPort15]>; // LEA instructions can't fold loads.

// Integer shifts and rotates.
defm : SKLWriteResPair<WriteShift, SKLPort06,  1>;

// Loads, stores, and moves, not folded with other operations.
def : WriteRes<WriteLoad,  [SKLPort23]> { let Latency = 5; }
def : WriteRes<WriteStore, [SKLPort237, SKLPort4]>;
def : WriteRes<WriteMove,  [SKLPort0156]>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
def : WriteRes<WriteZero,  []>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : SKLWriteResPair<WriteJump,  SKLPort06,   1>;

// Floating point. This covers both scalar and vector operations.
def  : WriteRes<WriteFLoad,         [SKLPort23]> { let Latency = 6; }
def  : WriteRes<WriteFStore,        [SKLPort237, SKLPort4]>;
def  : WriteRes<WriteFMove,         [SKLPort015]>;

defm : SKLWriteResPair<WriteFAdd,   SKLPort1, 3>; // Floating point add/sub/compare.
defm : SKLWriteResPair<WriteFMul,   SKLPort0, 5>; // Floating point multiplication.
defm : SKLWriteResPair<WriteFDiv,   SKLPort0, 12>; // 10-14 cycles. // Floating point division.
defm : SKLWriteResPair<WriteFSqrt,  SKLPort0, 15>; // Floating point square root.
defm : SKLWriteResPair<WriteFRcp,   SKLPort0, 5>; // Floating point reciprocal estimate.
defm : SKLWriteResPair<WriteFRsqrt, SKLPort0, 5>; // Floating point reciprocal square root estimate.
defm : SKLWriteResPair<WriteFMA,    SKLPort01, 4>; // Fused Multiply Add.
defm : SKLWriteResPair<WriteFShuffle,  SKLPort5,  1>; // Floating point vector shuffles.
defm : SKLWriteResPair<WriteFBlend,  SKLPort015,  1>; // Floating point vector blends.
def : WriteRes<WriteFVarBlend, [SKLPort5]> { // Fp vector variable blends.	       
  let Latency = 2;
  let ResourceCycles = [2];
} 
def : WriteRes<WriteFVarBlendLd, [SKLPort5, SKLPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

// FMA Scheduling helper class.
// class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Vector integer operations.
def  : WriteRes<WriteVecLoad,         [SKLPort23]> { let Latency = 6; }
def  : WriteRes<WriteVecStore,        [SKLPort237, SKLPort4]>;
def  : WriteRes<WriteVecMove,         [SKLPort015]>;

defm : SKLWriteResPair<WriteVecALU,   SKLPort15,  1>; // Vector integer ALU op, no logicals.
defm : SKLWriteResPair<WriteVecShift, SKLPort0,  1>; // Vector integer shifts.
defm : SKLWriteResPair<WriteVecIMul,  SKLPort0,   5>; // Vector integer multiply.
defm : SKLWriteResPair<WriteShuffle,  SKLPort5,  1>; // Vector shuffles.
defm : SKLWriteResPair<WriteBlend,  SKLPort15,  1>; // Vector blends.

def : WriteRes<WriteVarBlend, [SKLPort5]> { // Vector variable blends.
  let Latency = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteVarBlendLd, [SKLPort5, SKLPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

def : WriteRes<WriteMPSAD, [SKLPort0, SKLPort5]> { // Vector MPSAD.     
  let Latency = 6;
  let ResourceCycles = [1, 2];
}
def : WriteRes<WriteMPSADLd, [SKLPort23, SKLPort0, SKLPort5]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 2];
}

// Vector bitwise operations.
// These are often used on both floating point and integer vectors.
defm : SKLWriteResPair<WriteVecLogic, SKLPort015, 1>; // Vector and/or/xor.

// Conversion between integer and float.
defm : SKLWriteResPair<WriteCvtF2I, SKLPort1, 3>; // Float -> Integer.
defm : SKLWriteResPair<WriteCvtI2F, SKLPort1, 4>; // Integer -> Float.
defm : SKLWriteResPair<WriteCvtF2F, SKLPort1, 3>; // Float -> Float size conversion.

// Strings instructions.
// Packed Compare Implicit Length Strings, Return Mask
// String instructions.
def : WriteRes<WritePCmpIStrM, [SKLPort0]> {
  let Latency = 10;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [SKLPort0, SKLPort23]> {
  let Latency = 10;
  let ResourceCycles = [3, 1];
} 
// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [SKLPort0, SKLPort16, SKLPort5]> {
  let Latency = 10;
  let ResourceCycles = [3, 2, 4];
}
def : WriteRes<WritePCmpEStrMLd, [SKLPort05, SKLPort16, SKLPort23]> {
  let Latency = 10;
  let ResourceCycles = [6, 2, 1];
} 
  // Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [SKLPort0]> {
  let Latency = 11;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [SKLPort0, SKLPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 1];
}     
// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [SKLPort05, SKLPort16]> {
  let Latency = 11;
  let ResourceCycles = [6, 2];
}
def : WriteRes<WritePCmpEStrILd, [SKLPort0, SKLPort16, SKLPort5, SKLPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 2, 2, 1];
}

// AES instructions.
def : WriteRes<WriteAESDecEnc, [SKLPort5]> { // Decryption, encryption.
  let Latency = 7;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [SKLPort5, SKLPort23]> {
  let Latency = 7;
  let ResourceCycles = [1, 1];
}
def : WriteRes<WriteAESIMC, [SKLPort5]> { // InvMixColumn.
  let Latency = 14;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [SKLPort5, SKLPort23]> {
  let Latency = 14;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteAESKeyGen, [SKLPort0, SKLPort5]> { // Key Generation.
  let Latency = 10;
  let ResourceCycles = [2, 8];
}
def : WriteRes<WriteAESKeyGenLd, [SKLPort0, SKLPort5, SKLPort23]> {
  let Latency = 10;
  let ResourceCycles = [2, 7, 1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [SKLPort0, SKLPort5]> {
  let Latency = 7;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteCLMulLd, [SKLPort0, SKLPort5, SKLPort23]> {
  let Latency = 7;
  let ResourceCycles = [2, 1, 1];
}

// Catch-all for expensive system instructions.
def : WriteRes<WriteSystem,     [SKLPort0156]> { let Latency = 100; } // def WriteSystem : SchedWrite;

// AVX2.
defm : SKLWriteResPair<WriteFShuffle256,  SKLPort5,  3>; // Fp 256-bit width vector shuffles.
defm : SKLWriteResPair<WriteShuffle256,  SKLPort5,  3>;  // 256-bit width vector shuffles.
def : WriteRes<WriteVarVecShift, [SKLPort0, SKLPort5]> { // Variable vector shifts.
  let Latency = 2;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteVarVecShiftLd, [SKLPort0, SKLPort5, SKLPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1, 1];
}

// Old microcoded instructions that nobody use.
def : WriteRes<WriteMicrocoded, [SKLPort0156]> { let Latency = 100; } // def WriteMicrocoded : SchedWrite;

// Fence instructions.
def : WriteRes<WriteFence,  [SKLPort23, SKLPort4]>;

// Nop, not very useful expect it provides a model for nops!
def : WriteRes<WriteNop, []>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////
// HADD, HSUB PS/PD
// x,x / v,v,v.
def : WriteRes<WriteFHAdd, [SKLPort1]> {
  let Latency = 3;
}

// x,m / v,v,m.
def : WriteRes<WriteFHAddLd, [SKLPort1, SKLPort23]> {
  let Latency = 7;
  let ResourceCycles = [1, 1];
}

// PHADD|PHSUB (S) W/D.
// v <- v,v.
def : WriteRes<WritePHAdd, [SKLPort15]>;

// v <- v,m.
def : WriteRes<WritePHAddLd, [SKLPort15, SKLPort23]> {
  let Latency = 5;
  let ResourceCycles = [1, 1];
}

// Remaining instrs.

def SKLWriteResGroup1 : SchedWriteRes<[SKLPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PADDSBirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PADDSWirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PADDUSBirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PADDUSWirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PAVGBirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PAVGWirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PCMPEQBirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PCMPEQDirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PCMPEQWirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PCMPGTBirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PCMPGTDirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PCMPGTWirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PMAXSWirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PMAXUBirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PMINSWirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PMINUBirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSLLDri")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSLLDrr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSLLQri")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSLLQrr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSLLWri")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSLLWrr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSRADri")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSRADrr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSRAWri")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSRAWrr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSRLDri")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSRLDrr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSRLQri")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSRLQrr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSRLWri")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSRLWrr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSUBSBirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSUBSWirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSUBUSBirr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PSUBUSWirr")>;

def SKLWriteResGroup2 : SchedWriteRes<[SKLPort1]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_MASKMOVQ64")>;

def SKLWriteResGroup3 : SchedWriteRes<[SKLPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup3], (instregex "COMP_FST0r")>;
def: InstRW<[SKLWriteResGroup3], (instregex "COM_FST0r")>;
def: InstRW<[SKLWriteResGroup3], (instregex "INSERTPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_MOVD64rr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_MOVD64to64rr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_PALIGNRrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_PSHUFBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_PSHUFWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_PUNPCKHBWirr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_PUNPCKHDQirr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_PUNPCKHWDirr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_PUNPCKLBWirr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_PUNPCKLDQirr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_PUNPCKLWDirr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MOV64toPQIrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MOVDDUPrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MOVDI2PDIrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MOVHLPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MOVLHPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MOVSDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MOVSHDUPrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MOVSLDUPrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MOVUPDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "MOVUPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PACKSSDWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PACKSSWBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PACKUSDWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PACKUSWBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PALIGNRrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PBLENDWrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVSXBDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVSXBQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVSXBWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVSXDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVSXWDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVSXWQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVZXBDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVZXBQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVZXBWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVZXDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVZXWDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMOVZXWQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSHUFBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSHUFDri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSHUFHWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSHUFLWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSLLDQri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSRLDQri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PUNPCKHBWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PUNPCKHDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PUNPCKHQDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PUNPCKHWDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PUNPCKLBWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PUNPCKLDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PUNPCKLQDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PUNPCKLWDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "SHUFPDrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "SHUFPSrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "UCOM_FPr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "UCOM_Fr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "UNPCKHPDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "UNPCKHPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "UNPCKLPDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "UNPCKLPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VBROADCASTSSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VINSERTPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOV64toPQIrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVDDUPYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVDDUPrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVDI2PDIrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVHLPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVLHPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVSDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVSHDUPYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVSHDUPrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVSLDUPYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVSLDUPrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVUPDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVUPDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVUPSYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VMOVUPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPACKSSDWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPACKSSDWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPACKSSWBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPACKSSWBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPACKUSDWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPACKUSDWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPACKUSWBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPACKUSWBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPALIGNRYrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPALIGNRrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPBLENDWYrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPBLENDWrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPBROADCASTDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPBROADCASTQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPERMILPDYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPERMILPDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPERMILPDri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPERMILPDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPERMILPSYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPERMILPSYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPERMILPSri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPERMILPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVSXBDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVSXBQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVSXBWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVSXDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVSXWDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVSXWQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVZXBDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVZXBQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVZXBWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVZXDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVZXWDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMOVZXWQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSHUFBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSHUFBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSHUFDYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSHUFDri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSHUFHWYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSHUFHWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSHUFLWYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSHUFLWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLDQYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLDQri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLDQYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLDQri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKHBWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKHBWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKHDQYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKHDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKHQDQYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKHQDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKHWDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKHWDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKLBWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKLBWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKLDQYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKLDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKLQDQYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKLQDQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKLWDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPUNPCKLWDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VSHUFPDYrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VSHUFPDrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VSHUFPSYrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VSHUFPSrri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VUNPCKHPDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VUNPCKHPDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VUNPCKHPSYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VUNPCKHPSrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VUNPCKLPDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VUNPCKLPDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VUNPCKLPSYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VUNPCKLPSrr")>;

def SKLWriteResGroup4 : SchedWriteRes<[SKLPort6]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup4], (instregex "JMP(16|32|64)r")>;

def SKLWriteResGroup5 : SchedWriteRes<[SKLPort01]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup5], (instregex "PABSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PABSDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PABSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PADDSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PADDSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PADDUSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PADDUSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PAVGBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PAVGWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PCMPEQBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PCMPEQDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PCMPEQQrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PCMPEQWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PCMPGTBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PCMPGTDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PCMPGTWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMAXSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMAXSDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMAXSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMAXUBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMAXUDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMAXUWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMINSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMINSDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMINSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMINUBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMINUDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PMINUWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSIGNBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSIGNDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSIGNWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSLLDri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSLLQri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSLLWri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSRADri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSRAWri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSRLDri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSRLQri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSRLWri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSUBSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSUBSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSUBUSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "PSUBUSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPABSBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPABSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPABSDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPABSDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPABSWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPABSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPADDSBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPADDSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPADDSWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPADDSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPADDUSBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPADDUSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPADDUSWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPADDUSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPAVGBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPAVGBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPAVGWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPAVGWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPEQBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPEQBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPEQDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPEQDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPEQQYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPEQQrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPEQWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPEQWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPGTBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPGTBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPGTDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPGTDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPGTWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPCMPGTWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXSBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXSDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXSDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXSWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXUBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXUBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXUDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXUDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXUWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMAXUWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINSBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINSDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINSDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINSWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINUBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINUBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINUDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINUDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINUWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPMINUWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSIGNBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSIGNBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSIGNDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSIGNDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSIGNWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSIGNWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSLLDYri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSLLDri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSLLQYri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSLLQri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSLLVDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSLLVDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSLLVQYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSLLVQrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSLLWYri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSLLWri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRADYri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRADri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRAVDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRAVDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRAWYri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRAWri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRLDYri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRLDri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRLQYri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRLQri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRLVDYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRLVDrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRLVQYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRLVQrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRLWYri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSRLWri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSUBSBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSUBSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSUBSWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSUBSWrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSUBUSBYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSUBUSBrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSUBUSWYrr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "VPSUBUSWrr")>;

def SKLWriteResGroup6 : SchedWriteRes<[SKLPort05]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup6], (instregex "FINCSTP")>;
def: InstRW<[SKLWriteResGroup6], (instregex "FNOP")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_MOVQ64rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PABSBrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PABSDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PABSWrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PADDBirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PADDDirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PADDQirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PADDWirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PANDNirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PANDirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PORirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PSIGNBrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PSIGNDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PSIGNWrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PSUBBirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PSUBDirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PSUBQirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PSUBWirr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_PXORirr")>;

def SKLWriteResGroup7 : SchedWriteRes<[SKLPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup7], (instregex "ADC(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "ADC(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "ADC8rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "ADCX(32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "ADOX(32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "BT(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "BT(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "BTC(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "BTC(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "BTR(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "BTR(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "BTS(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "BTS(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CDQ")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CLAC")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CMOV(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CQO")>;
def: InstRW<[SKLWriteResGroup7], (instregex "J(A|AE|B|BE|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)_1")>;
def: InstRW<[SKLWriteResGroup7], (instregex "J(A|AE|B|BE|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)_4")>;
def: InstRW<[SKLWriteResGroup7], (instregex "JMP_1")>;
def: InstRW<[SKLWriteResGroup7], (instregex "JMP_4")>;
def: InstRW<[SKLWriteResGroup7], (instregex "RORX(32|64)ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SAR(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SAR(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SAR8r1")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SAR8ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SARX(32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SBB(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SBB(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SBB8rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SET(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)r")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SHL(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SHL(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SHL8r1")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SHL8ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SHLX(32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SHR(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SHR(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SHR8r1")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SHR8ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SHRX(32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "STAC")>;

def SKLWriteResGroup8 : SchedWriteRes<[SKLPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup8], (instregex "ANDN(32|64)rr")>;
def: InstRW<[SKLWriteResGroup8], (instregex "BLSI(32|64)rr")>;
def: InstRW<[SKLWriteResGroup8], (instregex "BLSMSK(32|64)rr")>;
def: InstRW<[SKLWriteResGroup8], (instregex "BLSR(32|64)rr")>;
def: InstRW<[SKLWriteResGroup8], (instregex "BZHI(32|64)rr")>;
def: InstRW<[SKLWriteResGroup8], (instregex "LEA(16|32|64)(_32)?r")>;

def SKLWriteResGroup9 : SchedWriteRes<[SKLPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup9], (instregex "ANDNPDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "ANDNPSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "ANDPDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "ANDPSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "BLENDPDrri")>;
def: InstRW<[SKLWriteResGroup9], (instregex "BLENDPSrri")>;
def: InstRW<[SKLWriteResGroup9], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "MOVAPDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "MOVAPSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "MOVDQArr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "MOVDQUrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "MOVPQI2QIrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "MOVSSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "ORPDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "ORPSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PADDBrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PADDDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PADDQrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PADDWrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PANDNrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PANDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PORrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PSUBBrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PSUBDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PSUBQrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PSUBWrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "PXORrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VANDNPDYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VANDNPDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VANDNPSYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VANDNPSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VANDPDYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VANDPDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VANDPSYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VANDPSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VBLENDPDYrri")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VBLENDPDrri")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VBLENDPSYrri")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VBLENDPSrri")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVAPDYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVAPDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVAPSYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVAPSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVDQAYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVDQArr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVDQUYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVDQUrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVPQI2QIrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVSSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VMOVZPQILo2PQIrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VORPDYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VORPDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VORPSYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VORPSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPADDBYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPADDBrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPADDDYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPADDDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPADDQYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPADDQrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPADDWYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPADDWrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPANDNYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPANDNrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPANDYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPANDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPBLENDDYrri")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPBLENDDrri")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPORYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPORrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPSUBBYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPSUBBrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPSUBDYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPSUBDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPSUBQYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPSUBQrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPSUBWYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPSUBWrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPXORYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VPXORrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VXORPDYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VXORPDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VXORPSYrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "VXORPSrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "XORPDrr")>;
def: InstRW<[SKLWriteResGroup9], (instregex "XORPSrr")>;

def SKLWriteResGroup10 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup10], (instregex "ADD(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "ADD(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "ADD8i8")>;
def: InstRW<[SKLWriteResGroup10], (instregex "ADD8ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "ADD8rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "AND(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "AND(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "AND8i8")>;
def: InstRW<[SKLWriteResGroup10], (instregex "AND8ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "AND8rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "CBW")>;
def: InstRW<[SKLWriteResGroup10], (instregex "CLC")>;
def: InstRW<[SKLWriteResGroup10], (instregex "CMC")>;
def: InstRW<[SKLWriteResGroup10], (instregex "CMP(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "CMP(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "CMP8i8")>;
def: InstRW<[SKLWriteResGroup10], (instregex "CMP8ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "CMP8rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "CWDE")>;
def: InstRW<[SKLWriteResGroup10], (instregex "DEC(16|32|64)r")>;
def: InstRW<[SKLWriteResGroup10], (instregex "DEC8r")>;
def: InstRW<[SKLWriteResGroup10], (instregex "INC(16|32|64)r")>;
def: InstRW<[SKLWriteResGroup10], (instregex "INC8r")>;
def: InstRW<[SKLWriteResGroup10], (instregex "LAHF")>;
def: InstRW<[SKLWriteResGroup10], (instregex "MOV(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "MOV8ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "MOV8rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "MOVSX(16|32|64)rr16")>;
def: InstRW<[SKLWriteResGroup10], (instregex "MOVSX(16|32|64)rr32")>;
def: InstRW<[SKLWriteResGroup10], (instregex "MOVSX(16|32|64)rr8")>;
def: InstRW<[SKLWriteResGroup10], (instregex "MOVZX(16|32|64)rr16")>;
def: InstRW<[SKLWriteResGroup10], (instregex "MOVZX(16|32|64)rr8")>;
def: InstRW<[SKLWriteResGroup10], (instregex "NEG(16|32|64)r")>;
def: InstRW<[SKLWriteResGroup10], (instregex "NEG8r")>;
def: InstRW<[SKLWriteResGroup10], (instregex "NOOP")>;
def: InstRW<[SKLWriteResGroup10], (instregex "NOT(16|32|64)r")>;
def: InstRW<[SKLWriteResGroup10], (instregex "NOT8r")>;
def: InstRW<[SKLWriteResGroup10], (instregex "OR(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "OR(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "OR8i8")>;
def: InstRW<[SKLWriteResGroup10], (instregex "OR8ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "OR8rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SAHF")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SGDT64m")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SIDT64m")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SLDT64m")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SMSW16m")>;
def: InstRW<[SKLWriteResGroup10], (instregex "STC")>;
def: InstRW<[SKLWriteResGroup10], (instregex "STRm")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SUB(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SUB(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SUB8i8")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SUB8ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SUB8rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "SYSCALL")>;
def: InstRW<[SKLWriteResGroup10], (instregex "TEST(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "TEST8i8")>;
def: InstRW<[SKLWriteResGroup10], (instregex "TEST8ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "TEST8rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "XCHG(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "XOR(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "XOR(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup10], (instregex "XOR8i8")>;
def: InstRW<[SKLWriteResGroup10], (instregex "XOR8ri")>;
def: InstRW<[SKLWriteResGroup10], (instregex "XOR8rr")>;

def SKLWriteResGroup11 : SchedWriteRes<[SKLPort4,SKLPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup11], (instregex "FBSTPm")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MMX_MOVD64mr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MMX_MOVNTQmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MMX_MOVQ64mr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOV(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOV8mi")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOV8mr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVAPDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVAPSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVDQAmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVDQUmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVHPDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVHPSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVLPDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVLPSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVNTDQmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVNTI_64mr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVNTImr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVNTPDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVNTPSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVPDI2DImr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVPQI2QImr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVPQIto64mr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVSDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVSSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVUPDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "MOVUPSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "ST_FP32m")>;
def: InstRW<[SKLWriteResGroup11], (instregex "ST_FP64m")>;
def: InstRW<[SKLWriteResGroup11], (instregex "ST_FP80m")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VEXTRACTF128mr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VEXTRACTI128mr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVAPDYmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVAPDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVAPSYmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVAPSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVDQAYmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVDQAmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVDQUYmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVDQUmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVHPDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVHPSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVLPDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVLPSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVNTDQYmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVNTDQmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVNTPDYmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVNTPDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVNTPSYmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVNTPSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVPDI2DImr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVPQI2QImr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVPQIto64mr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVSDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVSSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVUPDYmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVUPDmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVUPSYmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMOVUPSmr")>;
def: InstRW<[SKLWriteResGroup11], (instregex "VMPTRSTm")>;

def SKLWriteResGroup12 : SchedWriteRes<[SKLPort0]> {
  let Latency = 2;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup12], (instregex "COMISDrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "COMISSrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_MOVD64grr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PMOVMSKBrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MOVMSKPDrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MOVMSKPSrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MOVPDI2DIrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MOVPQIto64rr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "PMOVMSKBrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "UCOMISDrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "UCOMISSrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VCOMISDrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VCOMISSrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VMOVMSKPDYrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VMOVMSKPDrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VMOVMSKPSYrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VMOVMSKPSrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VMOVPDI2DIrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VMOVPQIto64rr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VPMOVMSKBYrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VPMOVMSKBrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VTESTPDYrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VTESTPDrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VTESTPSYrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VTESTPSrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VUCOMISDrr")>;
def: InstRW<[SKLWriteResGroup12], (instregex "VUCOMISSrr")>;

def SKLWriteResGroup13 : SchedWriteRes<[SKLPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup13], (instregex "MMX_MOVQ2DQrr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "MMX_PINSRWrr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "PINSRBrr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "PINSRDrr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "PINSRQrr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "PINSRWrr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VPINSRBrr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VPINSRDrr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VPINSRQrr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VPINSRWrr")>;

def SKLWriteResGroup14 : SchedWriteRes<[SKLPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup14], (instregex "FDECSTP")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_MOVDQ2Qrr")>;

def SKLWriteResGroup15 : SchedWriteRes<[SKLPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup15], (instregex "CMOV(A|BE)(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup15], (instregex "ROL(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup15], (instregex "ROL(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup15], (instregex "ROL8r1")>;
def: InstRW<[SKLWriteResGroup15], (instregex "ROL8ri")>;
def: InstRW<[SKLWriteResGroup15], (instregex "ROR(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup15], (instregex "ROR(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup15], (instregex "ROR8r1")>;
def: InstRW<[SKLWriteResGroup15], (instregex "ROR8ri")>;
def: InstRW<[SKLWriteResGroup15], (instregex "SET(A|BE)r")>;

def SKLWriteResGroup16 : SchedWriteRes<[SKLPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup16], (instregex "BLENDVPDrr0")>;
def: InstRW<[SKLWriteResGroup16], (instregex "BLENDVPSrr0")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PBLENDVBrr0")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VBLENDVPDYrr")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VBLENDVPDrr")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VBLENDVPSYrr")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VBLENDVPSrr")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPBLENDVBYrr")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPBLENDVBrr")>;

def SKLWriteResGroup17 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup17], (instregex "LFENCE")>;
def: InstRW<[SKLWriteResGroup17], (instregex "WAIT")>;
def: InstRW<[SKLWriteResGroup17], (instregex "XGETBV")>;

def SKLWriteResGroup18 : SchedWriteRes<[SKLPort0,SKLPort237]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup18], (instregex "MMX_MASKMOVQ64")>;
def: InstRW<[SKLWriteResGroup18], (instregex "VMASKMOVDQU")>;
def: InstRW<[SKLWriteResGroup18], (instregex "VMASKMOVPDYmr")>;
def: InstRW<[SKLWriteResGroup18], (instregex "VMASKMOVPDmr")>;
def: InstRW<[SKLWriteResGroup18], (instregex "VMASKMOVPSYmr")>;
def: InstRW<[SKLWriteResGroup18], (instregex "VMASKMOVPSmr")>;
def: InstRW<[SKLWriteResGroup18], (instregex "VPMASKMOVDYmr")>;
def: InstRW<[SKLWriteResGroup18], (instregex "VPMASKMOVDmr")>;
def: InstRW<[SKLWriteResGroup18], (instregex "VPMASKMOVQYmr")>;
def: InstRW<[SKLWriteResGroup18], (instregex "VPMASKMOVQmr")>;

def SKLWriteResGroup19 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup19], (instregex "PSLLDrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "PSLLQrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "PSLLWrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "PSRADrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "PSRAWrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "PSRLDrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "PSRLQrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "PSRLWrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "VPSLLDrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "VPSLLQrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "VPSLLWrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "VPSRADrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "VPSRAWrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "VPSRLDrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "VPSRLQrr")>;
def: InstRW<[SKLWriteResGroup19], (instregex "VPSRLWrr")>;

def SKLWriteResGroup20 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup20], (instregex "CLFLUSH")>;

def SKLWriteResGroup21 : SchedWriteRes<[SKLPort237,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup21], (instregex "SFENCE")>;

def SKLWriteResGroup22 : SchedWriteRes<[SKLPort06,SKLPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup22], (instregex "BEXTR(32|64)rr")>;
def: InstRW<[SKLWriteResGroup22], (instregex "BSWAP(16|32|64)r")>;

def SKLWriteResGroup23 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup23], (instregex "ADC8i8")>;
def: InstRW<[SKLWriteResGroup23], (instregex "ADC8ri")>;
def: InstRW<[SKLWriteResGroup23], (instregex "CWD")>;
def: InstRW<[SKLWriteResGroup23], (instregex "JRCXZ")>;
def: InstRW<[SKLWriteResGroup23], (instregex "SBB8i8")>;
def: InstRW<[SKLWriteResGroup23], (instregex "SBB8ri")>;

def SKLWriteResGroup24 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup24], (instregex "EXTRACTPSmr")>;
def: InstRW<[SKLWriteResGroup24], (instregex "PEXTRBmr")>;
def: InstRW<[SKLWriteResGroup24], (instregex "PEXTRDmr")>;
def: InstRW<[SKLWriteResGroup24], (instregex "PEXTRQmr")>;
def: InstRW<[SKLWriteResGroup24], (instregex "PEXTRWmr")>;
def: InstRW<[SKLWriteResGroup24], (instregex "STMXCSR")>;
def: InstRW<[SKLWriteResGroup24], (instregex "VEXTRACTPSmr")>;
def: InstRW<[SKLWriteResGroup24], (instregex "VPEXTRBmr")>;
def: InstRW<[SKLWriteResGroup24], (instregex "VPEXTRDmr")>;
def: InstRW<[SKLWriteResGroup24], (instregex "VPEXTRQmr")>;
def: InstRW<[SKLWriteResGroup24], (instregex "VPEXTRWmr")>;
def: InstRW<[SKLWriteResGroup24], (instregex "VSTMXCSR")>;

def SKLWriteResGroup25 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup25], (instregex "FNSTCW16m")>;

def SKLWriteResGroup26 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort06]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup26], (instregex "SET(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)m")>;

def SKLWriteResGroup27 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort15]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup27], (instregex "MOVBE(16|32|64)mr")>;

def SKLWriteResGroup28 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup28], (instregex "PUSH(16|32|64)r(mr)?")>;
def: InstRW<[SKLWriteResGroup28], (instregex "PUSH64i8")>;
def: InstRW<[SKLWriteResGroup28], (instregex "STOSB")>;
def: InstRW<[SKLWriteResGroup28], (instregex "STOSL")>;
def: InstRW<[SKLWriteResGroup28], (instregex "STOSQ")>;
def: InstRW<[SKLWriteResGroup28], (instregex "STOSW")>;

def SKLWriteResGroup29 : SchedWriteRes<[SKLPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup29], (instregex "BSF(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "BSR(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup29], (instrs IMUL16rr, IMUL32rr, IMUL32rri, IMUL32rri8, IMUL64rr, IMUL64rri32, IMUL64rri8)>;
def: InstRW<[SKLWriteResGroup29], (instrs IMUL8r)>;
def: InstRW<[SKLWriteResGroup29], (instregex "LZCNT(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup29], (instrs MUL8r)>;
def: InstRW<[SKLWriteResGroup29], (instregex "PDEP(32|64)rr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "PEXT(32|64)rr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "POPCNT(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "SHLD(16|32|64)rri8")>;
def: InstRW<[SKLWriteResGroup29], (instregex "SHRD(16|32|64)rri8")>;
def: InstRW<[SKLWriteResGroup29], (instregex "TZCNT(16|32|64)rr")>;

def SKLWriteResGroup29_16i : SchedWriteRes<[SKLPort1, SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup29_16i], (instrs IMUL16rri, IMUL16rri8)>;

def SKLWriteResGroup30 : SchedWriteRes<[SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup30], (instregex "ADD_FPrST0")>;
def: InstRW<[SKLWriteResGroup30], (instregex "ADD_FST0r")>;
def: InstRW<[SKLWriteResGroup30], (instregex "ADD_FrST0")>;
def: InstRW<[SKLWriteResGroup30], (instregex "MMX_PSADBWirr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "PCMPGTQrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "PSADBWrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "SUBR_FPrST0")>;
def: InstRW<[SKLWriteResGroup30], (instregex "SUBR_FST0r")>;
def: InstRW<[SKLWriteResGroup30], (instregex "SUBR_FrST0")>;
def: InstRW<[SKLWriteResGroup30], (instregex "SUB_FPrST0")>;
def: InstRW<[SKLWriteResGroup30], (instregex "SUB_FST0r")>;
def: InstRW<[SKLWriteResGroup30], (instregex "SUB_FrST0")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VBROADCASTSDYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VBROADCASTSSYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VEXTRACTF128rr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VEXTRACTI128rr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VINSERTF128rr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VINSERTI128rr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPBROADCASTBYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPBROADCASTBrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPBROADCASTDYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPBROADCASTQYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPBROADCASTWYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPBROADCASTWrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPCMPGTQYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPCMPGTQrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPERM2F128rr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPERM2I128rr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPERMDYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPERMPDYri")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPERMPSYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPERMQYri")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVSXBDYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVSXBQYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVSXBWYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVSXDQYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVSXWDYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVSXWQYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVZXBDYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVZXBQYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVZXBWYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVZXDQYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVZXWDYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPMOVZXWQYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPSADBWYrr")>;
def: InstRW<[SKLWriteResGroup30], (instregex "VPSADBWrr")>;

def SKLWriteResGroup31 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup31], (instregex "EXTRACTPSrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "MMX_PEXTRWrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "PEXTRBrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "PEXTRDrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "PEXTRQrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "PEXTRWrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "PTESTrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VEXTRACTPSrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VPEXTRBrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VPEXTRDrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VPEXTRQrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VPEXTRWrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VPTESTYrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VPTESTrr")>;

def SKLWriteResGroup32 : SchedWriteRes<[SKLPort0,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup32], (instregex "FNSTSW16r")>;

def SKLWriteResGroup33 : SchedWriteRes<[SKLPort06]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKLWriteResGroup33], (instregex "ROL(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup33], (instregex "ROL8rCL")>;
def: InstRW<[SKLWriteResGroup33], (instregex "ROR(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup33], (instregex "ROR8rCL")>;
def: InstRW<[SKLWriteResGroup33], (instregex "SAR(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup33], (instregex "SAR8rCL")>;
def: InstRW<[SKLWriteResGroup33], (instregex "SHL(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup33], (instregex "SHL8rCL")>;
def: InstRW<[SKLWriteResGroup33], (instregex "SHR(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup33], (instregex "SHR8rCL")>;

def SKLWriteResGroup34 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKLWriteResGroup34], (instregex "XADD(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup34], (instregex "XADD8rr")>;
def: InstRW<[SKLWriteResGroup34], (instregex "XCHG8rr")>;

def SKLWriteResGroup35 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup35], (instregex "MMX_PHADDSWrr")>;
def: InstRW<[SKLWriteResGroup35], (instregex "MMX_PHSUBSWrr")>;

def SKLWriteResGroup36 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup36], (instregex "PHADDSWrr")>;
def: InstRW<[SKLWriteResGroup36], (instregex "PHSUBSWrr")>;
def: InstRW<[SKLWriteResGroup36], (instregex "VPHADDSWrr")>;
def: InstRW<[SKLWriteResGroup36], (instregex "VPHADDSWYrr")>;
def: InstRW<[SKLWriteResGroup36], (instregex "VPHSUBSWrr")>;
def: InstRW<[SKLWriteResGroup36], (instregex "VPHSUBSWYrr")>;

def SKLWriteResGroup37 : SchedWriteRes<[SKLPort5,SKLPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup37], (instregex "MMX_PHADDDrr")>;
def: InstRW<[SKLWriteResGroup37], (instregex "MMX_PHADDWrr")>;
def: InstRW<[SKLWriteResGroup37], (instregex "MMX_PHSUBDrr")>;
def: InstRW<[SKLWriteResGroup37], (instregex "MMX_PHSUBWrr")>;

def SKLWriteResGroup38 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup38], (instregex "PHADDDrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "PHADDWrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "PHSUBDrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "PHSUBWrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPHADDDYrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPHADDDrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPHADDWYrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPHADDWrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPHSUBDYrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPHSUBDrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPHSUBWYrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPHSUBWrr")>;

def SKLWriteResGroup39 : SchedWriteRes<[SKLPort5,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup39], (instregex "MMX_PACKSSDWirr")>;
def: InstRW<[SKLWriteResGroup39], (instregex "MMX_PACKSSWBirr")>;
def: InstRW<[SKLWriteResGroup39], (instregex "MMX_PACKUSWBirr")>;

def SKLWriteResGroup40 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup40], (instregex "CLD")>;

def SKLWriteResGroup41 : SchedWriteRes<[SKLPort237,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup41], (instregex "MFENCE")>;

def SKLWriteResGroup42 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup42], (instregex "RCL(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup42], (instregex "RCL(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup42], (instregex "RCL8r1")>;
def: InstRW<[SKLWriteResGroup42], (instregex "RCL8ri")>;
def: InstRW<[SKLWriteResGroup42], (instregex "RCR(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup42], (instregex "RCR(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup42], (instregex "RCR8r1")>;
def: InstRW<[SKLWriteResGroup42], (instregex "RCR8ri")>;

def SKLWriteResGroup43 : SchedWriteRes<[SKLPort0,SKLPort4,SKLPort237]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup43], (instregex "FNSTSWm")>;

def SKLWriteResGroup44 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort06]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKLWriteResGroup44], (instregex "SET(A|BE)m")>;

def SKLWriteResGroup45 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort237,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup45], (instregex "CALL(16|32|64)r")>;

def SKLWriteResGroup46 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup46], (instregex "CALL64pcrel32")>;

def SKLWriteResGroup47 : SchedWriteRes<[SKLPort0]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup47], (instregex "AESDECLASTrr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "AESDECrr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "AESENCLASTrr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "AESENCrr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "MMX_PMADDUBSWrr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "MMX_PMADDWDirr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "MMX_PMULHRSWrr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "MMX_PMULHUWirr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "MMX_PMULHWirr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "MMX_PMULLWirr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "MMX_PMULUDQirr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "MUL_FPrST0")>;
def: InstRW<[SKLWriteResGroup47], (instregex "MUL_FST0r")>;
def: InstRW<[SKLWriteResGroup47], (instregex "MUL_FrST0")>;
def: InstRW<[SKLWriteResGroup47], (instregex "RCPPSr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "RCPSSr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "RSQRTPSr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "RSQRTSSr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "VAESDECLASTrr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "VAESDECrr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "VAESENCLASTrr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "VAESENCrr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "VRCPPSYr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "VRCPPSr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "VRCPSSr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "VRSQRTPSYr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "VRSQRTPSr")>;
def: InstRW<[SKLWriteResGroup47], (instregex "VRSQRTSSr")>;

def SKLWriteResGroup48 : SchedWriteRes<[SKLPort01]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup48], (instregex "ADDPDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "ADDPSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "ADDSDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "ADDSSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "ADDSUBPDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "ADDSUBPSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "MULPDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "MULPSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "MULSDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "MULSSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "SUBPDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "SUBPSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "SUBSDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "SUBSSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VADDPDYrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VADDPDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VADDPSYrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VADDPSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VADDSDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VADDSSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VADDSUBPDYrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VADDSUBPDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VADDSUBPSYrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VADDSUBPSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VMULPDYrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VMULPDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VMULPSYrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VMULPSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VMULSDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VMULSSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VSUBPDYrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VSUBPDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VSUBPSYrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VSUBPSrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VSUBSDrr")>;
def: InstRW<[SKLWriteResGroup48], (instregex "VSUBSSrr")>;
def: InstRW<[SKLWriteResGroup48],
            (instregex
             "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)(Y)?r",
             "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)r")>;

def SKLWriteResGroup49 : SchedWriteRes<[SKLPort015]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup49], (instregex "CMPPDrri")>;
def: InstRW<[SKLWriteResGroup49], (instregex "CMPPSrri")>;
def: InstRW<[SKLWriteResGroup49], (instregex "CMPSDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "CMPSSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "CVTDQ2PSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "CVTPS2DQrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "CVTTPS2DQrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "MAX(C?)PDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "MAX(C?)PSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "MAX(C?)SDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "MAX(C?)SSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "MIN(C?)PDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "MIN(C?)PSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "MIN(C?)SDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "MIN(C?)SSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "PHMINPOSUWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "PMADDUBSWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "PMADDWDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "PMULDQrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "PMULHRSWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "PMULHUWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "PMULHWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "PMULLWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "PMULUDQrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCMPPDYrri")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCMPPDrri")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCMPPSYrri")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCMPPSrri")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCMPSDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCMPSSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCVTDQ2PSYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCVTDQ2PSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCVTPS2DQYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCVTPS2DQrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCVTTPS2DQYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VCVTTPS2DQrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMAX(C?)PDYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMAX(C?)PDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMAX(C?)PSYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMAX(C?)PSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMAX(C?)SDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMAX(C?)SSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMIN(C?)PDYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMIN(C?)PDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMIN(C?)PSYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMIN(C?)PSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMIN(C?)SDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VMIN(C?)SSrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPHMINPOSUWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMADDUBSWYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMADDUBSWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMADDWDYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMADDWDrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULDQYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULDQrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULHRSWYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULHRSWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULHUWYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULHUWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULHWYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULHWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULLWYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULLWrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULUDQYrr")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VPMULUDQrr")>;

def SKLWriteResGroup50 : SchedWriteRes<[SKLPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup50], (instregex "MPSADBWrri")>;
def: InstRW<[SKLWriteResGroup50], (instregex "VMPSADBWYrri")>;
def: InstRW<[SKLWriteResGroup50], (instregex "VMPSADBWrri")>;

def SKLWriteResGroup51 : SchedWriteRes<[SKLPort1,SKLPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup51], (instrs IMUL64r)>;
def: InstRW<[SKLWriteResGroup51], (instrs MUL64r)>;
def: InstRW<[SKLWriteResGroup51], (instrs MULX64rr)>;

def SKLWriteResGroup51_16 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def: InstRW<[SKLWriteResGroup51_16], (instrs IMUL16r)>;
def: InstRW<[SKLWriteResGroup51_16], (instrs MUL16r)>;

def SKLWriteResGroup52 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup52], (instregex "VPSLLDYrr")>;
def: InstRW<[SKLWriteResGroup52], (instregex "VPSLLQYrr")>;
def: InstRW<[SKLWriteResGroup52], (instregex "VPSLLWYrr")>;
def: InstRW<[SKLWriteResGroup52], (instregex "VPSRADYrr")>;
def: InstRW<[SKLWriteResGroup52], (instregex "VPSRAWYrr")>;
def: InstRW<[SKLWriteResGroup52], (instregex "VPSRLDYrr")>;
def: InstRW<[SKLWriteResGroup52], (instregex "VPSRLQYrr")>;
def: InstRW<[SKLWriteResGroup52], (instregex "VPSRLWYrr")>;

def SKLWriteResGroup53 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup53], (instregex "ISTT_FP16m")>;
def: InstRW<[SKLWriteResGroup53], (instregex "ISTT_FP32m")>;
def: InstRW<[SKLWriteResGroup53], (instregex "ISTT_FP64m")>;
def: InstRW<[SKLWriteResGroup53], (instregex "IST_F16m")>;
def: InstRW<[SKLWriteResGroup53], (instregex "IST_F32m")>;
def: InstRW<[SKLWriteResGroup53], (instregex "IST_FP16m")>;
def: InstRW<[SKLWriteResGroup53], (instregex "IST_FP32m")>;
def: InstRW<[SKLWriteResGroup53], (instregex "IST_FP64m")>;

def SKLWriteResGroup54 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[SKLWriteResGroup54], (instregex "FNCLEX")>;

def SKLWriteResGroup55 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup55], (instregex "PAUSE")>;

def SKLWriteResGroup56 : SchedWriteRes<[SKLPort015,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup56], (instregex "VZEROUPPER")>;

def SKLWriteResGroup57 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKLWriteResGroup57], (instregex "LAR(16|32|64)rr")>;

def SKLWriteResGroup58 : SchedWriteRes<[SKLPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup58], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MMX_MOVD64rm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MMX_MOVD64to64rm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MMX_MOVQ64rm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOV(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOV64toPQIrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOV8rm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOVDDUPrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOVDI2PDIrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOVQI2PQIrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOVSDrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOVSSrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOVSX(16|32|64)rm16")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOVSX(16|32|64)rm32")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOVSX(16|32|64)rm8")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOVZX(16|32|64)rm16")>;
def: InstRW<[SKLWriteResGroup58], (instregex "MOVZX(16|32|64)rm8")>;
def: InstRW<[SKLWriteResGroup58], (instregex "PREFETCHNTA")>;
def: InstRW<[SKLWriteResGroup58], (instregex "PREFETCHT0")>;
def: InstRW<[SKLWriteResGroup58], (instregex "PREFETCHT1")>;
def: InstRW<[SKLWriteResGroup58], (instregex "PREFETCHT2")>;
def: InstRW<[SKLWriteResGroup58], (instregex "VMOV64toPQIrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "VMOVDDUPrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "VMOVDI2PDIrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "VMOVQI2PQIrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "VMOVSDrm")>;
def: InstRW<[SKLWriteResGroup58], (instregex "VMOVSSrm")>;

def SKLWriteResGroup59 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup59], (instregex "CVTDQ2PDrr")>;
def: InstRW<[SKLWriteResGroup59], (instregex "MMX_CVTPI2PDirr")>;
def: InstRW<[SKLWriteResGroup59], (instregex "VCVTDQ2PDrr")>;

def SKLWriteResGroup60 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup60], (instregex "CVTPD2DQrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "CVTPD2PSrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "CVTPS2PDrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "CVTSD2SSrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "CVTSI642SDrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "CVTSI2SDrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "CVTSI2SSrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "CVTSS2SDrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "CVTTPD2DQrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "MMX_CVTPD2PIirr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "MMX_CVTPS2PIirr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "MMX_CVTTPD2PIirr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "MMX_CVTTPS2PIirr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTPD2DQrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTPD2PSrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTPH2PSrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTPS2PDrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTPS2PHrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTSD2SSrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTSI642SDrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTSI2SDrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTSI2SSrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTSS2SDrr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "VCVTTPD2DQrr")>;

def SKLWriteResGroup61 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup61], (instregex "STR(16|32|64)r")>;

def SKLWriteResGroup62 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup62], (instrs IMUL32r)>;
def: InstRW<[SKLWriteResGroup62], (instrs MUL32r)>;
def: InstRW<[SKLWriteResGroup62], (instrs MULX32rr)>;

def SKLWriteResGroup63 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[SKLWriteResGroup63], (instregex "XSETBV")>;

def SKLWriteResGroup64 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[SKLWriteResGroup64], (instregex "CMPXCHG(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup64], (instregex "CMPXCHG8rr")>;

def SKLWriteResGroup65 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKLWriteResGroup65], (instregex "PUSHF16")>;
def: InstRW<[SKLWriteResGroup65], (instregex "PUSHF64")>;

def SKLWriteResGroup66 : SchedWriteRes<[SKLPort5]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup66], (instregex "PCLMULQDQrr")>;
def: InstRW<[SKLWriteResGroup66], (instregex "VPCLMULQDQrr")>;

def SKLWriteResGroup67 : SchedWriteRes<[SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup67], (instregex "LDDQUrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "MOVAPDrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "MOVAPSrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "MOVDQArm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "MOVDQUrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "MOVNTDQArm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "MOVSHDUPrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "MOVSLDUPrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "MOVUPDrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "MOVUPSrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VBROADCASTSSrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VLDDQUrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VMOVAPDrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VMOVAPSrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VMOVDQArm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VMOVDQUrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VMOVNTDQArm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VMOVSHDUPrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VMOVSLDUPrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VMOVUPDrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VMOVUPSrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VPBROADCASTDrm")>;
def: InstRW<[SKLWriteResGroup67], (instregex "VPBROADCASTQrm")>;

def SKLWriteResGroup68 : SchedWriteRes<[SKLPort0]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup68], (instregex "MMX_CVTPI2PSirr")>;

def SKLWriteResGroup69 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PADDSBirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PADDSWirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PADDUSBirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PADDUSWirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PAVGBirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PAVGWirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PCMPEQBirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PCMPEQDirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PCMPEQWirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PCMPGTBirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PCMPGTDirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PCMPGTWirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PMAXSWirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PMAXUBirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PMINSWirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PMINUBirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSLLDrm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSLLQrm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSLLWrm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSRADrm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSRAWrm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSRLDrm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSRLQrm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSRLWrm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSUBSBirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSUBSWirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSUBUSBirm")>;
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PSUBUSWirm")>;

def SKLWriteResGroup70 : SchedWriteRes<[SKLPort0,SKLPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup70], (instregex "CVTSD2SI64rr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "CVTSD2SIrr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "CVTSS2SI64rr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "CVTSS2SIrr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "CVTTSD2SI64rr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "CVTTSD2SIrr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "VCVTSD2SI64rr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "VCVTSD2SIrr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "VCVTSS2SI64rr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "VCVTSS2SIrr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "VCVTTSD2SI64rr")>;
def: InstRW<[SKLWriteResGroup70], (instregex "VCVTTSD2SIrr")>;

def SKLWriteResGroup71 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PALIGNRrmi")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PINSRWrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PSHUFBrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PSHUFWmi")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PUNPCKHBWirm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PUNPCKHDQirm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PUNPCKHWDirm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PUNPCKLBWirm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PUNPCKLDQirm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PUNPCKLWDirm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MOVHPDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MOVHPSrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MOVLPDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "MOVLPSrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PINSRBrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PINSRDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PINSRQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PINSRWrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVSXBDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVSXBQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVSXBWrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVSXDQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVSXWDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVSXWQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVZXBDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVZXBQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVZXBWrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVZXDQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVZXWDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PMOVZXWQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VMOVHPDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VMOVHPSrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VMOVLPDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VMOVLPSrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPINSRBrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPINSRDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPINSRQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPINSRWrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVSXBDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVSXBQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVSXBWrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVSXDQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVSXWDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVSXWQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVZXBDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVZXBQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVZXBWrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVZXDQrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVZXWDrm")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPMOVZXWQrm")>;

def SKLWriteResGroup72 : SchedWriteRes<[SKLPort6,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup72], (instregex "FARJMP64")>;
def: InstRW<[SKLWriteResGroup72], (instregex "JMP(16|32|64)m")>;

def SKLWriteResGroup73 : SchedWriteRes<[SKLPort23,SKLPort05]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PABSBrm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PABSDrm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PABSWrm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PADDBirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PADDDirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PADDQirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PADDWirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PANDNirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PANDirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PORirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PSIGNBrm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PSIGNDrm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PSIGNWrm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PSUBBirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PSUBDirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PSUBQirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PSUBWirm")>;
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PXORirm")>;

def SKLWriteResGroup74 : SchedWriteRes<[SKLPort23,SKLPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup74], (instregex "ADC(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "ADC8rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "ADCX(32|64)rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "ADOX(32|64)rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "BT(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup74], (instregex "CMOV(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "RORX32mi")>;
def: InstRW<[SKLWriteResGroup74], (instregex "RORX64mi")>;
def: InstRW<[SKLWriteResGroup74], (instregex "SARX32rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "SARX64rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "SBB(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "SBB8rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "SHLX32rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "SHLX64rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "SHRX32rm")>;
def: InstRW<[SKLWriteResGroup74], (instregex "SHRX64rm")>;

def SKLWriteResGroup75 : SchedWriteRes<[SKLPort23,SKLPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup75], (instregex "ANDN(32|64)rm")>;
def: InstRW<[SKLWriteResGroup75], (instregex "BLSI(32|64)rm")>;
def: InstRW<[SKLWriteResGroup75], (instregex "BLSMSK(32|64)rm")>;
def: InstRW<[SKLWriteResGroup75], (instregex "BLSR(32|64)rm")>;
def: InstRW<[SKLWriteResGroup75], (instregex "BZHI(32|64)rm")>;
def: InstRW<[SKLWriteResGroup75], (instregex "MOVBE(16|32|64)rm")>;

def SKLWriteResGroup76 : SchedWriteRes<[SKLPort23,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup76], (instregex "ADD(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "ADD8rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "AND(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "AND8rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "CMP(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup76], (instregex "CMP(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup76], (instregex "CMP(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "CMP8mi")>;
def: InstRW<[SKLWriteResGroup76], (instregex "CMP8mr")>;
def: InstRW<[SKLWriteResGroup76], (instregex "CMP8rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "OR(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "OR8rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "POP(16|32|64)r(mr)?")>;
def: InstRW<[SKLWriteResGroup76], (instregex "SUB(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "SUB8rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "TEST(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup76], (instregex "TEST8mi")>;
def: InstRW<[SKLWriteResGroup76], (instregex "TEST8mr")>;
def: InstRW<[SKLWriteResGroup76], (instregex "XOR(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup76], (instregex "XOR8rm")>;

def SKLWriteResGroup77 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup77], (instregex "HADDPDrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "HADDPSrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "HSUBPDrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "HSUBPSrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "VHADDPDYrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "VHADDPDrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "VHADDPSYrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "VHADDPSrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "VHSUBPDYrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "VHSUBPDrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "VHSUBPSYrr")>;
def: InstRW<[SKLWriteResGroup77], (instregex "VHSUBPSrr")>;

def SKLWriteResGroup78 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup78], (instregex "CVTSI642SSrr")>;
def: InstRW<[SKLWriteResGroup78], (instregex "VCVTSI642SSrr")>;

def SKLWriteResGroup79 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKLWriteResGroup79], (instregex "SHLD(16|32|64)rrCL")>;
def: InstRW<[SKLWriteResGroup79], (instregex "SHRD(16|32|64)rrCL")>;

def SKLWriteResGroup80 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup80], (instregex "SLDT(16|32|64)r")>;

def SKLWriteResGroup81 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237,SKLPort015]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup81], (instregex "VCVTPS2PHmr")>;

def SKLWriteResGroup82 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup82], (instregex "BTC(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup82], (instregex "BTR(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup82], (instregex "BTS(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SAR(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SAR(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SAR8m1")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SAR8mi")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SHL(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SHL(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SHL8m1")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SHL8mi")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SHR(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SHR(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SHR8m1")>;
def: InstRW<[SKLWriteResGroup82], (instregex "SHR8mi")>;

def SKLWriteResGroup83 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup83], (instregex "ADD(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup83], (instregex "ADD(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup83], (instregex "ADD8mi")>;
def: InstRW<[SKLWriteResGroup83], (instregex "ADD8mr")>;
def: InstRW<[SKLWriteResGroup83], (instregex "AND(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup83], (instregex "AND(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup83], (instregex "AND8mi")>;
def: InstRW<[SKLWriteResGroup83], (instregex "AND8mr")>;
def: InstRW<[SKLWriteResGroup83], (instregex "DEC(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup83], (instregex "DEC8m")>;
def: InstRW<[SKLWriteResGroup83], (instregex "INC(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup83], (instregex "INC8m")>;
def: InstRW<[SKLWriteResGroup83], (instregex "NEG(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup83], (instregex "NEG8m")>;
def: InstRW<[SKLWriteResGroup83], (instregex "NOT(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup83], (instregex "NOT8m")>;
def: InstRW<[SKLWriteResGroup83], (instregex "OR(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup83], (instregex "OR(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup83], (instregex "OR8mi")>;
def: InstRW<[SKLWriteResGroup83], (instregex "OR8mr")>;
def: InstRW<[SKLWriteResGroup83], (instregex "POP(16|32|64)rmm")>;
def: InstRW<[SKLWriteResGroup83], (instregex "PUSH(16|32|64)rmm")>;
def: InstRW<[SKLWriteResGroup83], (instregex "SUB(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup83], (instregex "SUB(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup83], (instregex "SUB8mi")>;
def: InstRW<[SKLWriteResGroup83], (instregex "SUB8mr")>;
def: InstRW<[SKLWriteResGroup83], (instregex "XOR(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup83], (instregex "XOR(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup83], (instregex "XOR8mi")>;
def: InstRW<[SKLWriteResGroup83], (instregex "XOR8mr")>;

def SKLWriteResGroup84 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKLWriteResGroup84], (instregex "STD")>;

def SKLWriteResGroup85 : SchedWriteRes<[SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup85], (instregex "LD_F32m")>;
def: InstRW<[SKLWriteResGroup85], (instregex "LD_F64m")>;
def: InstRW<[SKLWriteResGroup85], (instregex "LD_F80m")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VBROADCASTF128")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VBROADCASTI128")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VBROADCASTSDYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VBROADCASTSSYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VLDDQUYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VMOVAPDYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VMOVAPSYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VMOVDDUPYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VMOVDQAYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VMOVDQUYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VMOVNTDQAYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VMOVSHDUPYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VMOVSLDUPYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VMOVUPDYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VMOVUPSYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VPBROADCASTDYrm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "VPBROADCASTQYrm")>;

def SKLWriteResGroup86 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup86], (instregex "VCVTDQ2PDYrr")>;

def SKLWriteResGroup87 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup87], (instregex "COMISDrm")>;
def: InstRW<[SKLWriteResGroup87], (instregex "COMISSrm")>;
def: InstRW<[SKLWriteResGroup87], (instregex "UCOMISDrm")>;
def: InstRW<[SKLWriteResGroup87], (instregex "UCOMISSrm")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VCOMISDrm")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VCOMISSrm")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VUCOMISDrm")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VUCOMISSrm")>;

def SKLWriteResGroup88 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup88], (instregex "INSERTPSrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PACKSSDWrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PACKSSWBrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PACKUSDWrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PACKUSWBrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PALIGNRrmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PBLENDWrmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PSHUFBrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PSHUFDmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PSHUFHWmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PSHUFLWmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PUNPCKHBWrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PUNPCKHDQrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PUNPCKHQDQrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PUNPCKHWDrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PUNPCKLBWrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PUNPCKLDQrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PUNPCKLQDQrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "PUNPCKLWDrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "SHUFPDrmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "SHUFPSrmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "UNPCKHPDrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "UNPCKHPSrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "UNPCKLPDrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "UNPCKLPSrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VINSERTPSrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPACKSSDWrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPACKSSWBrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPACKUSDWrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPACKUSWBrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPALIGNRrmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPBLENDWrmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPBROADCASTBrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPBROADCASTWrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPERMILPDmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPERMILPDrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPERMILPSmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPERMILPSrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPSHUFBrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPSHUFDmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPSHUFHWmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPSHUFLWmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPUNPCKHBWrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPUNPCKHDQrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPUNPCKHQDQrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPUNPCKHWDrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPUNPCKLBWrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPUNPCKLDQrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPUNPCKLQDQrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VPUNPCKLWDrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VSHUFPDrmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VSHUFPSrmi")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VUNPCKHPDrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VUNPCKHPSrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VUNPCKLPDrm")>;
def: InstRW<[SKLWriteResGroup88], (instregex "VUNPCKLPSrm")>;

def SKLWriteResGroup89 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTPD2DQYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTPD2PSYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTPH2PSYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTPS2PDYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTPS2PHYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTTPD2DQYrr")>;

def SKLWriteResGroup90 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup90], (instregex "PABSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PABSDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PABSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PADDSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PADDSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PADDUSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PADDUSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PAVGBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PAVGWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PCMPEQBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PCMPEQDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PCMPEQQrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PCMPEQWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PCMPGTBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PCMPGTDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PCMPGTWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMAXSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMAXSDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMAXSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMAXUBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMAXUDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMAXUWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMINSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMINSDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMINSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMINUBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMINUDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PMINUWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSIGNBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSIGNDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSIGNWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSLLDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSLLQrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSLLWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSRADrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSRAWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSRLDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSRLQrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSRLWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSUBSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSUBSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSUBUSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "PSUBUSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPABSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPABSDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPABSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPADDSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPADDSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPADDUSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPADDUSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPAVGBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPAVGWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPCMPEQBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPCMPEQDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPCMPEQQrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPCMPEQWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPCMPGTBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPCMPGTDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPCMPGTWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMAXSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMAXSDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMAXSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMAXUBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMAXUDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMAXUWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMINSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMINSDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMINSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMINUBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMINUDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPMINUWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSIGNBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSIGNDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSIGNWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSLLDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSLLQrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSLLVDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSLLVQrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSLLWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSRADrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSRAVDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSRAWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSRLDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSRLQrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSRLVDrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSRLVQrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSRLWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSUBSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSUBSWrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSUBUSBrm")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VPSUBUSWrm")>;

def SKLWriteResGroup91 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup91], (instregex "ANDNPDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "ANDNPSrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "ANDPDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "ANDPSrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "BLENDPDrmi")>;
def: InstRW<[SKLWriteResGroup91], (instregex "BLENDPSrmi")>;
def: InstRW<[SKLWriteResGroup91], (instregex "ORPDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "ORPSrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PADDBrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PADDDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PADDQrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PADDWrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PANDNrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PANDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PORrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PSUBBrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PSUBDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PSUBQrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PSUBWrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "PXORrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VANDNPDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VANDNPSrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VANDPDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VANDPSrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VBLENDPDrmi")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VBLENDPSrmi")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VINSERTF128rm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VINSERTI128rm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VMASKMOVPDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VMASKMOVPSrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VORPDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VORPSrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPADDBrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPADDDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPADDQrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPADDWrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPANDNrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPANDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPBLENDDrmi")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPMASKMOVDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPMASKMOVQrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPORrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPSUBBrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPSUBDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPSUBQrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPSUBWrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VPXORrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VXORPDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VXORPSrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "XORPDrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "XORPSrm")>;

def SKLWriteResGroup92 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup92], (instregex "MMX_PACKSSDWirm")>;
def: InstRW<[SKLWriteResGroup92], (instregex "MMX_PACKSSWBirm")>;
def: InstRW<[SKLWriteResGroup92], (instregex "MMX_PACKUSWBirm")>;

def SKLWriteResGroup93 : SchedWriteRes<[SKLPort23,SKLPort06]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup93], (instregex "CMOV(A|BE)(16|32|64)rm")>;

def SKLWriteResGroup94 : SchedWriteRes<[SKLPort23,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup94], (instregex "LEAVE64")>;
def: InstRW<[SKLWriteResGroup94], (instregex "SCASB")>;
def: InstRW<[SKLWriteResGroup94], (instregex "SCASL")>;
def: InstRW<[SKLWriteResGroup94], (instregex "SCASQ")>;
def: InstRW<[SKLWriteResGroup94], (instregex "SCASW")>;

def SKLWriteResGroup95 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup95], (instregex "CVTTSS2SI64rr")>;
def: InstRW<[SKLWriteResGroup95], (instregex "CVTTSS2SIrr")>;
def: InstRW<[SKLWriteResGroup95], (instregex "VCVTTSS2SI64rr")>;
def: InstRW<[SKLWriteResGroup95], (instregex "VCVTTSS2SIrr")>;

def SKLWriteResGroup96 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort05]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup96], (instregex "FLDCW16m")>;

def SKLWriteResGroup97 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup97], (instregex "LDMXCSR")>;
def: InstRW<[SKLWriteResGroup97], (instregex "VLDMXCSR")>;

def SKLWriteResGroup98 : SchedWriteRes<[SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup98], (instregex "LRETQ")>;
def: InstRW<[SKLWriteResGroup98], (instregex "RETQ")>;

def SKLWriteResGroup99 : SchedWriteRes<[SKLPort23,SKLPort06,SKLPort15]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup99], (instregex "BEXTR(32|64)rm")>;

def SKLWriteResGroup100 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup100], (instregex "ROL(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup100], (instregex "ROL(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup100], (instregex "ROL8m1")>;
def: InstRW<[SKLWriteResGroup100], (instregex "ROL8mi")>;
def: InstRW<[SKLWriteResGroup100], (instregex "ROR(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup100], (instregex "ROR(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup100], (instregex "ROR8m1")>;
def: InstRW<[SKLWriteResGroup100], (instregex "ROR8mi")>;

def SKLWriteResGroup101 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup101], (instregex "XADD(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup101], (instregex "XADD8rm")>;

def SKLWriteResGroup102 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[SKLWriteResGroup102], (instregex "CALL(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup102], (instregex "FARCALL64")>;

def SKLWriteResGroup103 : SchedWriteRes<[SKLPort6,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,1,2];
}
def: InstRW<[SKLWriteResGroup103], (instregex "LOOP")>;

def SKLWriteResGroup104 : SchedWriteRes<[SKLPort0]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup104], (instregex "AESIMCrr")>;
def: InstRW<[SKLWriteResGroup104], (instregex "VAESIMCrr")>;

def SKLWriteResGroup105 : SchedWriteRes<[SKLPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup105], (instregex "PMULLDrr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "ROUNDPDr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "ROUNDPSr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "ROUNDSDr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "ROUNDSSr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "VPMULLDYrr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "VPMULLDrr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "VROUNDPDr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "VROUNDPSr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "VROUNDSDr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "VROUNDSSr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "VROUNDYPDr")>;
def: InstRW<[SKLWriteResGroup105], (instregex "VROUNDYPSr")>;

def SKLWriteResGroup106 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup106], (instregex "VTESTPDrm")>;
def: InstRW<[SKLWriteResGroup106], (instregex "VTESTPSrm")>;

def SKLWriteResGroup107 : SchedWriteRes<[SKLPort1,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup107], (instregex "BSF(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup107], (instregex "BSR(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup107], (instrs IMUL64m)>;
def: InstRW<[SKLWriteResGroup107], (instrs IMUL32rm, IMUL32rmi, IMUL32rmi8, IMUL64rm, IMUL64rmi32, IMUL64rmi8)>;
def: InstRW<[SKLWriteResGroup107], (instrs IMUL8m)>;
def: InstRW<[SKLWriteResGroup107], (instregex "LZCNT(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup107], (instregex "MUL(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup107], (instregex "MUL8m")>;
def: InstRW<[SKLWriteResGroup107], (instregex "PDEP(32|64)rm")>;
def: InstRW<[SKLWriteResGroup107], (instregex "PEXT(32|64)rm")>;
def: InstRW<[SKLWriteResGroup107], (instregex "POPCNT(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup107], (instregex "TZCNT(16|32|64)rm")>;

def SKLWriteResGroup107_16 : SchedWriteRes<[SKLPort1, SKLPort0156, SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1]; 
}
def: InstRW<[SKLWriteResGroup107_16], (instrs IMUL16rm, IMUL16rmi, IMUL16rmi8)>;

def SKLWriteResGroup107_16_2 : SchedWriteRes<[SKLPort1, SKLPort0156, SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 5;
}
def: InstRW<[SKLWriteResGroup107_16_2], (instrs IMUL16m)>;
def: InstRW<[SKLWriteResGroup107_16_2], (instrs MUL16m)>;

def SKLWriteResGroup107_32 : SchedWriteRes<[SKLPort1, SKLPort0156, SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup107_32], (instrs IMUL32m)>;
def: InstRW<[SKLWriteResGroup107_32], (instrs MUL32m)>;

def SKLWriteResGroup108 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup108], (instregex "FCOM32m")>;
def: InstRW<[SKLWriteResGroup108], (instregex "FCOM64m")>;
def: InstRW<[SKLWriteResGroup108], (instregex "FCOMP32m")>;
def: InstRW<[SKLWriteResGroup108], (instregex "FCOMP64m")>;
def: InstRW<[SKLWriteResGroup108], (instregex "MMX_PSADBWirm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPACKSSDWYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPACKSSWBYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPACKUSDWYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPACKUSWBYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPALIGNRYrmi")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPBLENDWYrmi")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPBROADCASTBYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPBROADCASTWYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPERMILPDYmi")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPERMILPDYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPERMILPSYmi")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPERMILPSYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPMOVSXBDYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPMOVSXBQYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPMOVSXWQYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPSHUFBYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPSHUFDYmi")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPSHUFHWYmi")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPSHUFLWYmi")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPUNPCKHBWYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPUNPCKHDQYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPUNPCKHQDQYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPUNPCKHWDYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPUNPCKLBWYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPUNPCKLDQYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPUNPCKLQDQYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VPUNPCKLWDYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VSHUFPDYrmi")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VSHUFPSYrmi")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VUNPCKHPDYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VUNPCKHPSYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VUNPCKLPDYrm")>;
def: InstRW<[SKLWriteResGroup108], (instregex "VUNPCKLPSYrm")>;

def SKLWriteResGroup109 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup109], (instregex "VPABSBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPABSDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPABSWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPADDSBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPADDSWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPADDUSBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPADDUSWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPAVGBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPAVGWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPCMPEQBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPCMPEQDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPCMPEQQYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPCMPEQWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPCMPGTBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPCMPGTDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPCMPGTWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMAXSBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMAXSDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMAXSWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMAXUBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMAXUDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMAXUWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMINSBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMINSDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMINSWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMINUBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMINUDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPMINUWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSIGNBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSIGNDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSIGNWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSLLDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSLLQYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSLLVDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSLLVQYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSLLWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSRADYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSRAVDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSRAWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSRLDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSRLQYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSRLVDYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSRLVQYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSRLWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSUBSBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSUBSWYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSUBUSBYrm")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VPSUBUSWYrm")>;

def SKLWriteResGroup110 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup110], (instregex "VANDNPDYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VANDNPSYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VANDPDYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VANDPSYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VBLENDPDYrmi")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VBLENDPSYrmi")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VMASKMOVPDYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VMASKMOVPSYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VORPDYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VORPSYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPADDBYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPADDDYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPADDQYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPADDWYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPANDNYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPANDYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPBLENDDYrmi")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPMASKMOVDYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPMASKMOVQYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPORYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPSUBBYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPSUBDYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPSUBQYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPSUBWYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VPXORYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VXORPDYrm")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VXORPSYrm")>;

def SKLWriteResGroup111 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup111], (instregex "BLENDVPDrm0")>;
def: InstRW<[SKLWriteResGroup111], (instregex "BLENDVPSrm0")>;
def: InstRW<[SKLWriteResGroup111], (instregex "PBLENDVBrm0")>;
def: InstRW<[SKLWriteResGroup111], (instregex "VBLENDVPDrm")>;
def: InstRW<[SKLWriteResGroup111], (instregex "VBLENDVPSrm")>;
def: InstRW<[SKLWriteResGroup111], (instregex "VPBLENDVBYrm")>;
def: InstRW<[SKLWriteResGroup111], (instregex "VPBLENDVBrm")>;

def SKLWriteResGroup112 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKLWriteResGroup112], (instregex "MMX_PHADDSWrm")>;
def: InstRW<[SKLWriteResGroup112], (instregex "MMX_PHSUBSWrm")>;

def SKLWriteResGroup113 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort05]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup113], (instregex "MMX_PHADDDrm")>;
def: InstRW<[SKLWriteResGroup113], (instregex "MMX_PHADDWrm")>;
def: InstRW<[SKLWriteResGroup113], (instregex "MMX_PHSUBDrm")>;
def: InstRW<[SKLWriteResGroup113], (instregex "MMX_PHSUBWrm")>;

def SKLWriteResGroup114 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237,SKLPort015]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup114], (instregex "VCVTPS2PHYmr")>;

def SKLWriteResGroup115 : SchedWriteRes<[SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKLWriteResGroup115], (instregex "ROR(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup115], (instregex "ROR8mCL")>;

def SKLWriteResGroup116 : SchedWriteRes<[SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup116], (instregex "RCL(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup116], (instregex "RCL(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup116], (instregex "RCL8m1")>;
def: InstRW<[SKLWriteResGroup116], (instregex "RCL8mi")>;
def: InstRW<[SKLWriteResGroup116], (instregex "RCR(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup116], (instregex "RCR(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup116], (instregex "RCR8m1")>;
def: InstRW<[SKLWriteResGroup116], (instregex "RCR8mi")>;

def SKLWriteResGroup117 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKLWriteResGroup117], (instregex "ROL(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup117], (instregex "ROL8mCL")>;
def: InstRW<[SKLWriteResGroup117], (instregex "SAR(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup117], (instregex "SAR8mCL")>;
def: InstRW<[SKLWriteResGroup117], (instregex "SHL(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup117], (instregex "SHL8mCL")>;
def: InstRW<[SKLWriteResGroup117], (instregex "SHR(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup117], (instregex "SHR8mCL")>;

def SKLWriteResGroup118 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKLWriteResGroup118], (instregex "ADC(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup118], (instregex "ADC8mi")>;

def SKLWriteResGroup119 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKLWriteResGroup119], (instregex "ADC(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup119], (instregex "ADC8mr")>;
def: InstRW<[SKLWriteResGroup119], (instregex "CMPXCHG(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup119], (instregex "CMPXCHG8rm")>;
def: InstRW<[SKLWriteResGroup119], (instregex "SBB(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup119], (instregex "SBB(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup119], (instregex "SBB8mi")>;
def: InstRW<[SKLWriteResGroup119], (instregex "SBB8mr")>;

def SKLWriteResGroup120 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup120], (instregex "MMX_CVTPI2PSirm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "MMX_PMADDUBSWrm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "MMX_PMADDWDirm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "MMX_PMULHRSWrm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "MMX_PMULHUWirm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "MMX_PMULHWirm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "MMX_PMULLWirm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "MMX_PMULUDQirm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "RCPSSm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "RSQRTSSm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "VRCPSSm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "VRSQRTSSm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "VTESTPDYrm")>;
def: InstRW<[SKLWriteResGroup120], (instregex "VTESTPSYrm")>;

def SKLWriteResGroup121 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup121], (instregex "PCMPGTQrm")>;
def: InstRW<[SKLWriteResGroup121], (instregex "PSADBWrm")>;
def: InstRW<[SKLWriteResGroup121], (instregex "VPCMPGTQrm")>;
def: InstRW<[SKLWriteResGroup121], (instregex "VPMOVSXBWYrm")>;
def: InstRW<[SKLWriteResGroup121], (instregex "VPMOVSXDQYrm")>;
def: InstRW<[SKLWriteResGroup121], (instregex "VPMOVSXWDYrm")>;
def: InstRW<[SKLWriteResGroup121], (instregex "VPMOVZXWDYrm")>;
def: InstRW<[SKLWriteResGroup121], (instregex "VPSADBWrm")>;

def SKLWriteResGroup122 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup122], (instregex "ADDSDrm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "ADDSSrm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "MULSDrm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "MULSSrm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "SUBSDrm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "SUBSSrm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "VADDSDrm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "VADDSSrm")>;
def: InstRW<[SKLWriteResGroup122],
            (instregex "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)m")>;
def: InstRW<[SKLWriteResGroup122], (instregex "VMULSDrm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "VMULSSrm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "VSUBSDrm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "VSUBSSrm")>;

def SKLWriteResGroup123 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup123], (instregex "CMPSDrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "CMPSSrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "CVTPS2PDrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "MAX(C?)SDrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "MAX(C?)SSrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "MIN(C?)SDrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "MIN(C?)SSrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "MMX_CVTPS2PIirm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "MMX_CVTTPS2PIirm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "VCMPSDrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "VCMPSSrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "VCVTPH2PSrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "VCVTPS2PDrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "VMAX(C?)SDrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "VMAX(C?)SSrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "VMIN(C?)SDrm")>;
def: InstRW<[SKLWriteResGroup123], (instregex "VMIN(C?)SSrm")>;

def SKLWriteResGroup124 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup124], (instregex "DPPDrri")>;
def: InstRW<[SKLWriteResGroup124], (instregex "VDPPDrri")>;

def SKLWriteResGroup125 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup125], (instregex "VBLENDVPDYrm")>;
def: InstRW<[SKLWriteResGroup125], (instregex "VBLENDVPSYrm")>;

def SKLWriteResGroup126 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup126], (instregex "PTESTrm")>;
def: InstRW<[SKLWriteResGroup126], (instregex "VPTESTrm")>;

def SKLWriteResGroup127 : SchedWriteRes<[SKLPort1,SKLPort5,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup127], (instrs MULX64rm)>;

def SKLWriteResGroup128 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup128], (instregex "PHADDSWrm")>;
def: InstRW<[SKLWriteResGroup128], (instregex "PHSUBSWrm")>;
def: InstRW<[SKLWriteResGroup128], (instregex "VPHADDSWrm")>;
def: InstRW<[SKLWriteResGroup128], (instregex "VPHSUBSWrm")>;

def SKLWriteResGroup129 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup129], (instregex "PHADDDrm")>;
def: InstRW<[SKLWriteResGroup129], (instregex "PHADDWrm")>;
def: InstRW<[SKLWriteResGroup129], (instregex "PHSUBDrm")>;
def: InstRW<[SKLWriteResGroup129], (instregex "PHSUBWrm")>;
def: InstRW<[SKLWriteResGroup129], (instregex "VPHADDDrm")>;
def: InstRW<[SKLWriteResGroup129], (instregex "VPHADDWrm")>;
def: InstRW<[SKLWriteResGroup129], (instregex "VPHSUBDrm")>;
def: InstRW<[SKLWriteResGroup129], (instregex "VPHSUBWrm")>;

def SKLWriteResGroup130 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup130], (instregex "SHLD(16|32|64)mri8")>;
def: InstRW<[SKLWriteResGroup130], (instregex "SHRD(16|32|64)mri8")>;

def SKLWriteResGroup131 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKLWriteResGroup131], (instregex "LAR(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup131], (instregex "LSL(16|32|64)rm")>;

def SKLWriteResGroup132 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup132], (instregex "AESDECLASTrm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "AESDECrm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "AESENCLASTrm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "AESENCrm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "RCPPSm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "RSQRTPSm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "VAESDECLASTrm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "VAESDECrm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "VAESENCLASTrm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "VAESENCrm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "VRCPPSm")>;
def: InstRW<[SKLWriteResGroup132], (instregex "VRSQRTPSm")>;

def SKLWriteResGroup133 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup133], (instregex "ADD_F32m")>;
def: InstRW<[SKLWriteResGroup133], (instregex "ADD_F64m")>;
def: InstRW<[SKLWriteResGroup133], (instregex "ILD_F16m")>;
def: InstRW<[SKLWriteResGroup133], (instregex "ILD_F32m")>;
def: InstRW<[SKLWriteResGroup133], (instregex "ILD_F64m")>;
def: InstRW<[SKLWriteResGroup133], (instregex "SUBR_F32m")>;
def: InstRW<[SKLWriteResGroup133], (instregex "SUBR_F64m")>;
def: InstRW<[SKLWriteResGroup133], (instregex "SUB_F32m")>;
def: InstRW<[SKLWriteResGroup133], (instregex "SUB_F64m")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPCMPGTQYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPERM2F128rm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPERM2I128rm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPERMDYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPERMPDYmi")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPERMPSYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPERMQYmi")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPMOVZXBDYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPMOVZXBQYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPMOVZXBWYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPMOVZXDQYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPMOVZXWQYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VPSADBWYrm")>;

def SKLWriteResGroup134 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup134], (instregex "ADDPDrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "ADDPSrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "ADDSUBPDrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "ADDSUBPSrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "MULPDrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "MULPSrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "SUBPDrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "SUBPSrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "VADDPDrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "VADDPSrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "VADDSUBPDrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "VADDSUBPSrm")>;
def: InstRW<[SKLWriteResGroup134],
            (instregex "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)m")>;
def: InstRW<[SKLWriteResGroup134], (instregex "VMULPDrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "VMULPSrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "VSUBPDrm")>;
def: InstRW<[SKLWriteResGroup134], (instregex "VSUBPSrm")>;

def SKLWriteResGroup135 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup135], (instregex "CMPPDrmi")>;
def: InstRW<[SKLWriteResGroup135], (instregex "CMPPSrmi")>;
def: InstRW<[SKLWriteResGroup135], (instregex "CVTDQ2PSrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "CVTPS2DQrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "CVTSS2SDrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "CVTTPS2DQrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "MAX(C?)PDrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "MAX(C?)PSrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "MIN(C?)PDrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "MIN(C?)PSrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "PHMINPOSUWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "PMADDUBSWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "PMADDWDrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "PMULDQrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "PMULHRSWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "PMULHUWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "PMULHWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "PMULLWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "PMULUDQrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VCMPPDrmi")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VCMPPSrmi")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VCVTDQ2PSrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VCVTPH2PSYrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VCVTPS2DQrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VCVTSS2SDrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VCVTTPS2DQrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VMAX(C?)PDrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VMAX(C?)PSrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VMIN(C?)PDrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VMIN(C?)PSrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VPHMINPOSUWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VPMADDUBSWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VPMADDWDrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VPMULDQrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VPMULHRSWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VPMULHUWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VPMULHWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VPMULLWrm")>;
def: InstRW<[SKLWriteResGroup135], (instregex "VPMULUDQrm")>;

def SKLWriteResGroup136 : SchedWriteRes<[SKLPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKLWriteResGroup136], (instregex "PCMPISTRIrr")>;
def: InstRW<[SKLWriteResGroup136], (instregex "PCMPISTRM128rr")>;
def: InstRW<[SKLWriteResGroup136], (instregex "VPCMPISTRIrr")>;
def: InstRW<[SKLWriteResGroup136], (instregex "VPCMPISTRM128rr")>;

def SKLWriteResGroup137 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup137], (instregex "MPSADBWrmi")>;
def: InstRW<[SKLWriteResGroup137], (instregex "VMPSADBWrmi")>;

def SKLWriteResGroup138 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup138], (instregex "MMX_CVTPI2PDirm")>;
def: InstRW<[SKLWriteResGroup138], (instregex "VPTESTYrm")>;

def SKLWriteResGroup139 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup139], (instregex "CVTSD2SSrm")>;
def: InstRW<[SKLWriteResGroup139], (instregex "VCVTSD2SSrm")>;

def SKLWriteResGroup140 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup140], (instregex "VPHADDSWYrm")>;
def: InstRW<[SKLWriteResGroup140], (instregex "VPHSUBSWYrm")>;

def SKLWriteResGroup141 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup141], (instregex "VPHADDDYrm")>;
def: InstRW<[SKLWriteResGroup141], (instregex "VPHADDWYrm")>;
def: InstRW<[SKLWriteResGroup141], (instregex "VPHSUBDYrm")>;
def: InstRW<[SKLWriteResGroup141], (instregex "VPHSUBWYrm")>;

def SKLWriteResGroup142 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort06,SKLPort0156]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup142], (instrs MULX32rm)>;

def SKLWriteResGroup143 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 10;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,3];
}
def: InstRW<[SKLWriteResGroup143], (instregex "ADD8mi")>;
def: InstRW<[SKLWriteResGroup143], (instregex "AND8mi")>;
def: InstRW<[SKLWriteResGroup143], (instregex "OR8mi")>;
def: InstRW<[SKLWriteResGroup143], (instregex "SUB8mi")>;
def: InstRW<[SKLWriteResGroup143], (instregex "XCHG(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup143], (instregex "XCHG8rm")>;
def: InstRW<[SKLWriteResGroup143], (instregex "XOR8mi")>;

def SKLWriteResGroup144 : SchedWriteRes<[SKLPort05,SKLPort0156]> {
  let Latency = 10;
  let NumMicroOps = 10;
  let ResourceCycles = [9,1];
}
def: InstRW<[SKLWriteResGroup144], (instregex "MMX_EMMS")>;

def SKLWriteResGroup145 : SchedWriteRes<[SKLPort0]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup145], (instregex "DIVPSrr")>;
def: InstRW<[SKLWriteResGroup145], (instregex "DIVSSrr")>;
def: InstRW<[SKLWriteResGroup145], (instregex "VDIVPSYrr")>;
def: InstRW<[SKLWriteResGroup145], (instregex "VDIVPSrr")>;
def: InstRW<[SKLWriteResGroup145], (instregex "VDIVSSrr")>;

def SKLWriteResGroup146 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup146], (instregex "MUL_F32m")>;
def: InstRW<[SKLWriteResGroup146], (instregex "MUL_F64m")>;
def: InstRW<[SKLWriteResGroup146], (instregex "VRCPPSYm")>;
def: InstRW<[SKLWriteResGroup146], (instregex "VRSQRTPSYm")>;

def SKLWriteResGroup147 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup147], (instregex "VADDPDYrm")>;
def: InstRW<[SKLWriteResGroup147], (instregex "VADDPSYrm")>;
def: InstRW<[SKLWriteResGroup147], (instregex "VADDSUBPDYrm")>;
def: InstRW<[SKLWriteResGroup147], (instregex "VADDSUBPSYrm")>;
def: InstRW<[SKLWriteResGroup147],
            (instregex "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Ym")>;
def: InstRW<[SKLWriteResGroup147], (instregex "VMULPDYrm")>;
def: InstRW<[SKLWriteResGroup147], (instregex "VMULPSYrm")>;
def: InstRW<[SKLWriteResGroup147], (instregex "VSUBPDYrm")>;
def: InstRW<[SKLWriteResGroup147], (instregex "VSUBPSYrm")>;

def SKLWriteResGroup148 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup148], (instregex "VCMPPDYrmi")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VCMPPSYrmi")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VCVTDQ2PSYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VCVTPS2DQYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VCVTPS2PDYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VCVTTPS2DQYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VMAX(C?)PDYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VMAX(C?)PSYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VMIN(C?)PDYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VMIN(C?)PSYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VPMADDUBSWYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VPMADDWDYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VPMULDQYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VPMULHRSWYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VPMULHUWYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VPMULHWYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VPMULLWYrm")>;
def: InstRW<[SKLWriteResGroup148], (instregex "VPMULUDQYrm")>;

def SKLWriteResGroup149 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup149], (instregex "FICOM16m")>;
def: InstRW<[SKLWriteResGroup149], (instregex "FICOM32m")>;
def: InstRW<[SKLWriteResGroup149], (instregex "FICOMP16m")>;
def: InstRW<[SKLWriteResGroup149], (instregex "FICOMP32m")>;
def: InstRW<[SKLWriteResGroup149], (instregex "VMPSADBWYrmi")>;

def SKLWriteResGroup150 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup150], (instregex "CVTDQ2PDrm")>;
def: InstRW<[SKLWriteResGroup150], (instregex "VCVTDQ2PDrm")>;

def SKLWriteResGroup151 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup151], (instregex "CVTSD2SI64rm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "CVTSD2SIrm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "CVTSS2SI64rm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "CVTSS2SIrm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "CVTTSD2SI64rm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "CVTTSD2SIrm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "CVTTSS2SIrm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "VCVTSD2SI64rm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "VCVTSD2SIrm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "VCVTSS2SI64rm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "VCVTSS2SIrm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "VCVTTSD2SI64rm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "VCVTTSD2SIrm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "VCVTTSS2SI64rm")>;
def: InstRW<[SKLWriteResGroup151], (instregex "VCVTTSS2SIrm")>;

def SKLWriteResGroup152 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup152], (instregex "CVTPD2DQrm")>;
def: InstRW<[SKLWriteResGroup152], (instregex "CVTPD2PSrm")>;
def: InstRW<[SKLWriteResGroup152], (instregex "CVTTPD2DQrm")>;
def: InstRW<[SKLWriteResGroup152], (instregex "MMX_CVTPD2PIirm")>;
def: InstRW<[SKLWriteResGroup152], (instregex "MMX_CVTTPD2PIirm")>;

def SKLWriteResGroup153 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKLWriteResGroup153], (instregex "SHLD(16|32|64)mrCL")>;
def: InstRW<[SKLWriteResGroup153], (instregex "SHRD(16|32|64)mrCL")>;

def SKLWriteResGroup154 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,3,2];
}
def: InstRW<[SKLWriteResGroup154], (instregex "RCL(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup154], (instregex "RCR(16|32|64)rCL")>;

def SKLWriteResGroup155 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,1,2];
}
def: InstRW<[SKLWriteResGroup155], (instregex "RCL8rCL")>;

def SKLWriteResGroup156 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 11;
  let ResourceCycles = [2,9];
}
def: InstRW<[SKLWriteResGroup156], (instregex "LOOPE")>;
def: InstRW<[SKLWriteResGroup156], (instregex "LOOPNE")>;

def SKLWriteResGroup157 : SchedWriteRes<[SKLPort0]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup157], (instregex "VSQRTPSYr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "VSQRTPSr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "VSQRTSSr")>;

def SKLWriteResGroup158 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup158], (instregex "PCLMULQDQrm")>;
def: InstRW<[SKLWriteResGroup158], (instregex "VPCLMULQDQrm")>;

def SKLWriteResGroup159 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup159], (instregex "HADDPDrm")>;
def: InstRW<[SKLWriteResGroup159], (instregex "HADDPSrm")>;
def: InstRW<[SKLWriteResGroup159], (instregex "HSUBPDrm")>;
def: InstRW<[SKLWriteResGroup159], (instregex "HSUBPSrm")>;
def: InstRW<[SKLWriteResGroup159], (instregex "VHADDPDrm")>;
def: InstRW<[SKLWriteResGroup159], (instregex "VHADDPSrm")>;
def: InstRW<[SKLWriteResGroup159], (instregex "VHSUBPDrm")>;
def: InstRW<[SKLWriteResGroup159], (instregex "VHSUBPSrm")>;

def SKLWriteResGroup160 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup160], (instregex "CVTTSS2SI64rm")>;

def SKLWriteResGroup161 : SchedWriteRes<[SKLPort0]> {
  let Latency = 13;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup161], (instregex "SQRTPSr")>;
def: InstRW<[SKLWriteResGroup161], (instregex "SQRTSSr")>;

def SKLWriteResGroup162 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup162], (instregex "ADD_FI16m")>;
def: InstRW<[SKLWriteResGroup162], (instregex "ADD_FI32m")>;
def: InstRW<[SKLWriteResGroup162], (instregex "SUBR_FI16m")>;
def: InstRW<[SKLWriteResGroup162], (instregex "SUBR_FI32m")>;
def: InstRW<[SKLWriteResGroup162], (instregex "SUB_FI16m")>;
def: InstRW<[SKLWriteResGroup162], (instregex "SUB_FI32m")>;

def SKLWriteResGroup163 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup163], (instregex "VCVTDQ2PDYrm")>;

def SKLWriteResGroup164 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup164], (instregex "DPPSrri")>;
def: InstRW<[SKLWriteResGroup164], (instregex "VDPPSYrri")>;
def: InstRW<[SKLWriteResGroup164], (instregex "VDPPSrri")>;

def SKLWriteResGroup165 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup165], (instregex "VHADDPDYrm")>;
def: InstRW<[SKLWriteResGroup165], (instregex "VHADDPSYrm")>;
def: InstRW<[SKLWriteResGroup165], (instregex "VHSUBPDYrm")>;
def: InstRW<[SKLWriteResGroup165], (instregex "VHSUBPSYrm")>;

def SKLWriteResGroup166 : SchedWriteRes<[SKLPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup166], (instregex "DIVPDrr")>;
def: InstRW<[SKLWriteResGroup166], (instregex "DIVSDrr")>;
def: InstRW<[SKLWriteResGroup166], (instregex "VDIVPDYrr")>;
def: InstRW<[SKLWriteResGroup166], (instregex "VDIVPDrr")>;
def: InstRW<[SKLWriteResGroup166], (instregex "VDIVSDrr")>;

def SKLWriteResGroup167 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup167], (instregex "AESIMCrm")>;
def: InstRW<[SKLWriteResGroup167], (instregex "VAESIMCrm")>;

def SKLWriteResGroup168 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup168], (instregex "PMULLDrm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "ROUNDPDm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "ROUNDPSm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "ROUNDSDm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "ROUNDSSm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "VPMULLDrm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "VROUNDPDm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "VROUNDPSm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "VROUNDSDm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "VROUNDSSm")>;

def SKLWriteResGroup169 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup169], (instregex "MUL_FI16m")>;
def: InstRW<[SKLWriteResGroup169], (instregex "MUL_FI32m")>;

def SKLWriteResGroup170 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 14;
  let NumMicroOps = 10;
  let ResourceCycles = [2,4,1,3];
}
def: InstRW<[SKLWriteResGroup170], (instregex "RCR8rCL")>;

def SKLWriteResGroup171 : SchedWriteRes<[SKLPort0]> {
  let Latency = 15;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup171], (instregex "DIVR_FPrST0")>;
def: InstRW<[SKLWriteResGroup171], (instregex "DIVR_FST0r")>;
def: InstRW<[SKLWriteResGroup171], (instregex "DIVR_FrST0")>;

def SKLWriteResGroup172 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 15;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup172], (instregex "VPMULLDYrm")>;
def: InstRW<[SKLWriteResGroup172], (instregex "VROUNDYPDm")>;
def: InstRW<[SKLWriteResGroup172], (instregex "VROUNDYPSm")>;

def SKLWriteResGroup173 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 15;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKLWriteResGroup173], (instregex "DPPDrmi")>;
def: InstRW<[SKLWriteResGroup173], (instregex "VDPPDrmi")>;

def SKLWriteResGroup174 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 15;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,5,1,1];
}
def: InstRW<[SKLWriteResGroup174], (instregex "RCL(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup174], (instregex "RCL8mCL")>;

def SKLWriteResGroup175 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup175], (instregex "DIVSSrm")>;
def: InstRW<[SKLWriteResGroup175], (instregex "VDIVSSrm")>;

def SKLWriteResGroup176 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}
def: InstRW<[SKLWriteResGroup176], (instregex "PCMPISTRIrm")>;
def: InstRW<[SKLWriteResGroup176], (instregex "PCMPISTRM128rm")>;
def: InstRW<[SKLWriteResGroup176], (instregex "VPCMPISTRIrm")>;
def: InstRW<[SKLWriteResGroup176], (instregex "VPCMPISTRM128rm")>;

def SKLWriteResGroup177 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 16;
  let NumMicroOps = 14;
  let ResourceCycles = [1,1,1,4,2,5];
}
def: InstRW<[SKLWriteResGroup177], (instregex "CMPXCHG8B")>;

def SKLWriteResGroup178 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 16;
  let NumMicroOps = 16;
  let ResourceCycles = [16];
}
def: InstRW<[SKLWriteResGroup178], (instregex "VZEROALL")>;

def SKLWriteResGroup179 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 17;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup179], (instregex "DIVPSrm")>;
def: InstRW<[SKLWriteResGroup179], (instregex "VDIVPSrm")>;
def: InstRW<[SKLWriteResGroup179], (instregex "VSQRTSSm")>;

def SKLWriteResGroup180 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort5,SKLPort6,SKLPort05,SKLPort0156]> {
  let Latency = 17;
  let NumMicroOps = 15;
  let ResourceCycles = [2,1,2,4,2,4];
}
def: InstRW<[SKLWriteResGroup180], (instregex "XCH_F")>;

def SKLWriteResGroup181 : SchedWriteRes<[SKLPort0]> {
  let Latency = 18;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup181], (instregex "VSQRTPDYr")>;
def: InstRW<[SKLWriteResGroup181], (instregex "VSQRTPDr")>;
def: InstRW<[SKLWriteResGroup181], (instregex "VSQRTSDr")>;

def SKLWriteResGroup182 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup182], (instregex "SQRTSSm")>;
def: InstRW<[SKLWriteResGroup182], (instregex "VDIVPSYrm")>;
def: InstRW<[SKLWriteResGroup182], (instregex "VSQRTPSm")>;

def SKLWriteResGroup183 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def: InstRW<[SKLWriteResGroup183], (instregex "PCMPESTRIrr")>;
def: InstRW<[SKLWriteResGroup183], (instregex "VPCMPESTRIrr")>;

def SKLWriteResGroup184 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[SKLWriteResGroup184], (instregex "CPUID")>;
def: InstRW<[SKLWriteResGroup184], (instregex "RDTSC")>;

def SKLWriteResGroup185 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,4,1,2];
}
def: InstRW<[SKLWriteResGroup185], (instregex "RCR(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup185], (instregex "RCR8mCL")>;

def SKLWriteResGroup186 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup186], (instregex "DIVSDrm")>;
def: InstRW<[SKLWriteResGroup186], (instregex "SQRTPSm")>;
def: InstRW<[SKLWriteResGroup186], (instregex "VDIVSDrm")>;
def: InstRW<[SKLWriteResGroup186], (instregex "VSQRTPSYm")>;

def SKLWriteResGroup187 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 19;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKLWriteResGroup187], (instregex "DPPSrmi")>;
def: InstRW<[SKLWriteResGroup187], (instregex "VDPPSrmi")>;

def SKLWriteResGroup188 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort015,SKLPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[SKLWriteResGroup188], (instregex "PCMPESTRM128rr")>;
def: InstRW<[SKLWriteResGroup188], (instregex "VPCMPESTRM128rr")>;

def SKLWriteResGroup189 : SchedWriteRes<[SKLPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup189], (instregex "DIV_FPrST0")>;
def: InstRW<[SKLWriteResGroup189], (instregex "DIV_FST0r")>;
def: InstRW<[SKLWriteResGroup189], (instregex "DIV_FrST0")>;
def: InstRW<[SKLWriteResGroup189], (instregex "SQRTPDr")>;
def: InstRW<[SKLWriteResGroup189], (instregex "SQRTSDr")>;

def SKLWriteResGroup190 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup190], (instregex "DIVPDrm")>;
def: InstRW<[SKLWriteResGroup190], (instregex "VDIVPDrm")>;

def SKLWriteResGroup191 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 20;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKLWriteResGroup191], (instregex "VDPPSYrmi")>;

def SKLWriteResGroup192 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 20;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,1,2];
}
def: InstRW<[SKLWriteResGroup192], (instregex "INSB")>;
def: InstRW<[SKLWriteResGroup192], (instregex "INSL")>;
def: InstRW<[SKLWriteResGroup192], (instregex "INSW")>;

def SKLWriteResGroup193 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort0156]> {
  let Latency = 20;
  let NumMicroOps = 10;
  let ResourceCycles = [1,2,7];
}
def: InstRW<[SKLWriteResGroup193], (instregex "MWAITrr")>;

def SKLWriteResGroup194 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort015]> {
  let Latency = 20;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,2];
}
def: InstRW<[SKLWriteResGroup194], (instregex "AESKEYGENASSIST128rr")>;
def: InstRW<[SKLWriteResGroup194], (instregex "VAESKEYGENASSIST128rr")>;

def SKLWriteResGroup195 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup195], (instregex "VDIVPDYrm")>;

def SKLWriteResGroup196 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 22;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup196], (instregex "DIV_F32m")>;
def: InstRW<[SKLWriteResGroup196], (instregex "DIV_F64m")>;

def SKLWriteResGroup196_1 : SchedWriteRes<[SKLPort0, SKLPort23, SKLPort5, SKLPort015]> {
  let Latency = 22;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKLWriteResGroup196_1], (instrs VGATHERDPSrm,
                                             VGATHERDPDrm,
                                             VGATHERQPDrm,
                                             VGATHERQPSrm,
                                             VPGATHERDDrm,
                                             VPGATHERDQrm,
                                             VPGATHERQDrm,
                                             VPGATHERQQrm)>;

def SKLWriteResGroup196_2 : SchedWriteRes<[SKLPort0, SKLPort23, SKLPort5, SKLPort015]> {
  let Latency = 25;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKLWriteResGroup196_2], (instrs VGATHERDPSYrm,
                                             VGATHERQPDYrm,
                                             VGATHERQPSYrm,
                                             VPGATHERDDYrm,
                                             VPGATHERDQYrm,
                                             VPGATHERQDYrm,
                                             VPGATHERQQYrm,
                                             VGATHERDPDYrm)>;

def SKLWriteResGroup197 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 23;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup197], (instregex "VSQRTSDm")>;

def SKLWriteResGroup198 : SchedWriteRes<[SKLPort0,SKLPort4,SKLPort5,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 23;
  let NumMicroOps = 19;
  let ResourceCycles = [2,1,4,1,1,4,6];
}
def: InstRW<[SKLWriteResGroup198], (instregex "CMPXCHG16B")>;

def SKLWriteResGroup199 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 24;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup199], (instregex "VSQRTPDm")>;

def SKLWriteResGroup200 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[SKLWriteResGroup200], (instregex "PCMPESTRIrm")>;
def: InstRW<[SKLWriteResGroup200], (instregex "VPCMPESTRIrm")>;

def SKLWriteResGroup201 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup201], (instregex "SQRTSDm")>;
def: InstRW<[SKLWriteResGroup201], (instregex "VSQRTPDYm")>;

def SKLWriteResGroup202 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 25;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup202], (instregex "DIV_FI16m")>;
def: InstRW<[SKLWriteResGroup202], (instregex "DIV_FI32m")>;

def SKLWriteResGroup203 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort015,SKLPort0156]> {
  let Latency = 25;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}
def: InstRW<[SKLWriteResGroup203], (instregex "PCMPESTRM128rm")>;
def: InstRW<[SKLWriteResGroup203], (instregex "VPCMPESTRM128rm")>;

def SKLWriteResGroup204 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 25;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,1,1];
}
def: InstRW<[SKLWriteResGroup204], (instregex "AESKEYGENASSIST128rm")>;
def: InstRW<[SKLWriteResGroup204], (instregex "VAESKEYGENASSIST128rm")>;

def SKLWriteResGroup205 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 26;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup205], (instregex "SQRTPDm")>;

def SKLWriteResGroup206 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 27;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup206], (instregex "DIVR_F32m")>;
def: InstRW<[SKLWriteResGroup206], (instregex "DIVR_F64m")>;

def SKLWriteResGroup207 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort0156]> {
  let Latency = 28;
  let NumMicroOps = 8;
  let ResourceCycles = [2,4,1,1];
}
def: InstRW<[SKLWriteResGroup207], (instregex "IDIV(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup207], (instregex "IDIV8m")>;

def SKLWriteResGroup208 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 30;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup208], (instregex "DIVR_FI16m")>;
def: InstRW<[SKLWriteResGroup208], (instregex "DIVR_FI32m")>;

def SKLWriteResGroup209 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort23,SKLPort06,SKLPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[SKLWriteResGroup209], (instregex "IN(16|32)ri")>;
def: InstRW<[SKLWriteResGroup209], (instregex "IN(16|32)rr")>;
def: InstRW<[SKLWriteResGroup209], (instregex "IN8ri")>;
def: InstRW<[SKLWriteResGroup209], (instregex "IN8rr")>;

def SKLWriteResGroup210 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[SKLWriteResGroup210], (instregex "OUT(16|32)ir")>;
def: InstRW<[SKLWriteResGroup210], (instregex "OUT(16|32)rr")>;
def: InstRW<[SKLWriteResGroup210], (instregex "OUT8ir")>;
def: InstRW<[SKLWriteResGroup210], (instregex "OUT8rr")>;

def SKLWriteResGroup211 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 37;
  let NumMicroOps = 31;
  let ResourceCycles = [1,8,1,21];
}
def: InstRW<[SKLWriteResGroup211], (instregex "XRSTOR(64)?")>;

def SKLWriteResGroup212 : SchedWriteRes<[SKLPort1,SKLPort4,SKLPort5,SKLPort6,SKLPort23,SKLPort237,SKLPort15,SKLPort0156]> {
  let Latency = 40;
  let NumMicroOps = 18;
  let ResourceCycles = [1,1,2,3,1,1,1,8];
}
def: InstRW<[SKLWriteResGroup212], (instregex "VMCLEARm")>;

def SKLWriteResGroup213 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 41;
  let NumMicroOps = 39;
  let ResourceCycles = [1,10,1,1,26];
}
def: InstRW<[SKLWriteResGroup213], (instregex "XSAVE64")>;

def SKLWriteResGroup214 : SchedWriteRes<[SKLPort5,SKLPort0156]> {
  let Latency = 42;
  let NumMicroOps = 22;
  let ResourceCycles = [2,20];
}
def: InstRW<[SKLWriteResGroup214], (instregex "RDTSCP")>;

def SKLWriteResGroup215 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 42;
  let NumMicroOps = 40;
  let ResourceCycles = [1,11,1,1,26];
}
def: InstRW<[SKLWriteResGroup215], (instregex "^XSAVE$", "XSAVEC", "XSAVES")>;

def SKLWriteResGroup216 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 46;
  let NumMicroOps = 44;
  let ResourceCycles = [1,11,1,1,30];
}
def: InstRW<[SKLWriteResGroup216], (instregex "XSAVEOPT")>;

def SKLWriteResGroup217 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort05,SKLPort06,SKLPort0156]> {
  let Latency = 62;
  let NumMicroOps = 64;
  let ResourceCycles = [2,8,5,10,39];
}
def: InstRW<[SKLWriteResGroup217], (instregex "FLDENVm")>;

def SKLWriteResGroup218 : SchedWriteRes<[SKLPort0,SKLPort6,SKLPort23,SKLPort05,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 63;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[SKLWriteResGroup218], (instregex "FXRSTOR64")>;

def SKLWriteResGroup219 : SchedWriteRes<[SKLPort0,SKLPort6,SKLPort23,SKLPort05,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 63;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[SKLWriteResGroup219], (instregex "FXRSTOR")>;

def SKLWriteResGroup220 : SchedWriteRes<[SKLPort5,SKLPort05,SKLPort0156]> {
  let Latency = 75;
  let NumMicroOps = 15;
  let ResourceCycles = [6,3,6];
}
def: InstRW<[SKLWriteResGroup220], (instregex "FNINIT")>;

def SKLWriteResGroup221 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort5,SKLPort6,SKLPort05,SKLPort0156]> {
  let Latency = 76;
  let NumMicroOps = 32;
  let ResourceCycles = [7,2,8,3,1,11];
}
def: InstRW<[SKLWriteResGroup221], (instregex "DIV(16|32|64)r")>;

def SKLWriteResGroup222 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort5,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 102;
  let NumMicroOps = 66;
  let ResourceCycles = [4,2,4,8,14,34];
}
def: InstRW<[SKLWriteResGroup222], (instregex "IDIV(16|32|64)r")>;

def SKLWriteResGroup223 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort4,SKLPort5,SKLPort6,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 106;
  let NumMicroOps = 100;
  let ResourceCycles = [9,1,11,16,1,11,21,30];
}
def: InstRW<[SKLWriteResGroup223], (instregex "FSTENVm")>;

} // SchedModel
