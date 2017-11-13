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
                          ProcResourceKind ExePort,
                          int Lat> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, [ExePort]> { let Latency = Lat; }

  // Memory variant also uses a cycle on port 2/3 and adds 5 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, [SKXPort23, ExePort]> {
     let Latency = !add(Lat, 5);
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [SKXPort4]>;

// Arithmetic.
defm : SKXWriteResPair<WriteALU,   SKXPort0156, 1>; // Simple integer ALU op.
defm : SKXWriteResPair<WriteIMul,  SKXPort1,   3>; // Integer multiplication.
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
defm : SKXWriteResPair<WriteShift, SKXPort06,  1>;

// Loads, stores, and moves, not folded with other operations.
def : WriteRes<WriteLoad,  [SKXPort23]> { let Latency = 5; }
def : WriteRes<WriteStore, [SKXPort237, SKXPort4]>;
def : WriteRes<WriteMove,  [SKXPort0156]>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
def : WriteRes<WriteZero,  []>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : SKXWriteResPair<WriteJump,  SKXPort06,   1>;

// Floating point. This covers both scalar and vector operations.
defm : SKXWriteResPair<WriteFAdd,   SKXPort1, 3>; // Floating point add/sub/compare.
defm : SKXWriteResPair<WriteFMul,   SKXPort0, 5>; // Floating point multiplication.
defm : SKXWriteResPair<WriteFDiv,   SKXPort0, 12>; // 10-14 cycles. // Floating point division.
defm : SKXWriteResPair<WriteFSqrt,  SKXPort0, 15>; // Floating point square root.
defm : SKXWriteResPair<WriteFRcp,   SKXPort0, 5>; // Floating point reciprocal estimate.
defm : SKXWriteResPair<WriteFRsqrt, SKXPort0, 5>; // Floating point reciprocal square root estimate.
// defm WriteFMA    : X86SchedWritePair; // Fused Multiply Add.
defm : SKXWriteResPair<WriteFShuffle,  SKXPort5,  1>; // Floating point vector shuffles.
defm : SKXWriteResPair<WriteFBlend,  SKXPort015,  1>; // Floating point vector blends.
def : WriteRes<WriteFVarBlend, [SKXPort5]> { // Fp vector variable blends.	       
  let Latency = 2;
  let ResourceCycles = [2];
} 
def : WriteRes<WriteFVarBlendLd, [SKXPort5, SKXPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

// FMA Scheduling helper class.
// class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Vector integer operations.
defm : SKXWriteResPair<WriteVecALU,   SKXPort15,  1>; // Vector integer ALU op, no logicals.
defm : SKXWriteResPair<WriteVecShift, SKXPort0,  1>; // Vector integer shifts.
defm : SKXWriteResPair<WriteVecIMul,  SKXPort0,   5>; // Vector integer multiply.
defm : SKXWriteResPair<WriteShuffle,  SKXPort5,  1>; // Vector shuffles.
defm : SKXWriteResPair<WriteBlend,  SKXPort15,  1>; // Vector blends.

def : WriteRes<WriteVarBlend, [SKXPort5]> { // Vector variable blends.
  let Latency = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteVarBlendLd, [SKXPort5, SKXPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

def : WriteRes<WriteMPSAD, [SKXPort0, SKXPort5]> { // Vector MPSAD.     
  let Latency = 6;
  let ResourceCycles = [1, 2];
}
def : WriteRes<WriteMPSADLd, [SKXPort23, SKXPort0, SKXPort5]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 2];
}

// Vector bitwise operations.
// These are often used on both floating point and integer vectors.
defm : SKXWriteResPair<WriteVecLogic, SKXPort015, 1>; // Vector and/or/xor.

// Conversion between integer and float.
defm : SKXWriteResPair<WriteCvtF2I, SKXPort1, 3>; // Float -> Integer.
defm : SKXWriteResPair<WriteCvtI2F, SKXPort1, 4>; // Integer -> Float.
defm : SKXWriteResPair<WriteCvtF2F, SKXPort1, 3>; // Float -> Float size conversion.

// Strings instructions.
// Packed Compare Implicit Length Strings, Return Mask
// String instructions.
def : WriteRes<WritePCmpIStrM, [SKXPort0]> {
  let Latency = 10;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [SKXPort0, SKXPort23]> {
  let Latency = 10;
  let ResourceCycles = [3, 1];
} 
// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [SKXPort0, SKXPort16, SKXPort5]> {
  let Latency = 10;
  let ResourceCycles = [3, 2, 4];
}
def : WriteRes<WritePCmpEStrMLd, [SKXPort05, SKXPort16, SKXPort23]> {
  let Latency = 10;
  let ResourceCycles = [6, 2, 1];
} 
  // Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [SKXPort0]> {
  let Latency = 11;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [SKXPort0, SKXPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 1];
}     
// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [SKXPort05, SKXPort16]> {
  let Latency = 11;
  let ResourceCycles = [6, 2];
}
def : WriteRes<WritePCmpEStrILd, [SKXPort0, SKXPort16, SKXPort5, SKXPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 2, 2, 1];
}

// AES instructions.
def : WriteRes<WriteAESDecEnc, [SKXPort5]> { // Decryption, encryption.
  let Latency = 7;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [SKXPort5, SKXPort23]> {
  let Latency = 7;
  let ResourceCycles = [1, 1];
}
def : WriteRes<WriteAESIMC, [SKXPort5]> { // InvMixColumn.
  let Latency = 14;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [SKXPort5, SKXPort23]> {
  let Latency = 14;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteAESKeyGen, [SKXPort0, SKXPort5]> { // Key Generation.
  let Latency = 10;
  let ResourceCycles = [2, 8];
}
def : WriteRes<WriteAESKeyGenLd, [SKXPort0, SKXPort5, SKXPort23]> {
  let Latency = 10;
  let ResourceCycles = [2, 7, 1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [SKXPort0, SKXPort5]> {
  let Latency = 7;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteCLMulLd, [SKXPort0, SKXPort5, SKXPort23]> {
  let Latency = 7;
  let ResourceCycles = [2, 1, 1];
}

// Catch-all for expensive system instructions.
def : WriteRes<WriteSystem,     [SKXPort0156]> { let Latency = 100; } // def WriteSystem : SchedWrite;

// AVX2.
defm : SKXWriteResPair<WriteFShuffle256,  SKXPort5,  3>; // Fp 256-bit width vector shuffles.
defm : SKXWriteResPair<WriteShuffle256,  SKXPort5,  3>;  // 256-bit width vector shuffles.
def : WriteRes<WriteVarVecShift, [SKXPort0, SKXPort5]> { // Variable vector shifts.
  let Latency = 2;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteVarVecShiftLd, [SKXPort0, SKXPort5, SKXPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1, 1];
}

// Old microcoded instructions that nobody use.
def : WriteRes<WriteMicrocoded, [SKXPort0156]> { let Latency = 100; } // def WriteMicrocoded : SchedWrite;

// Fence instructions.
def : WriteRes<WriteFence,  [SKXPort23, SKXPort4]>;

// Nop, not very useful expect it provides a model for nops!
def : WriteRes<WriteNop, []>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////
// HADD, HSUB PS/PD
// x,x / v,v,v.
def : WriteRes<WriteFHAdd, [SKXPort1]> {
  let Latency = 3;
}

// x,m / v,v,m.
def : WriteRes<WriteFHAddLd, [SKXPort1, SKXPort23]> {
  let Latency = 7;
  let ResourceCycles = [1, 1];
}

// PHADD|PHSUB (S) W/D.
// v <- v,v.
def : WriteRes<WritePHAdd, [SKXPort15]>;

// v <- v,m.
def : WriteRes<WritePHAddLd, [SKXPort15, SKXPort23]> {
  let Latency = 5;
  let ResourceCycles = [1, 1];
}

// Remaining instrs.

def SKXWriteResGroup1 : SchedWriteRes<[SKXPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup1], (instregex "KANDBrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KANDDrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KANDNBrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KANDNDrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KANDNQrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KANDNWrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KANDQrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KANDWrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KMOVBkk")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KMOVDkk")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KMOVQkk")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KMOVWkk")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KNOTBrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KNOTDrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KNOTQrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KNOTWrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KORBrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KORDrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KORQrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KORWrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KXNORBrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KXNORDrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KXNORQrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KXNORWrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KXORBrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KXORDrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KXORQrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "KXORWrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PADDSBirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PADDSWirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PADDUSBirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PADDUSWirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PAVGBirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PAVGWirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PCMPEQBirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PCMPEQDirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PCMPEQWirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PCMPGTBirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PCMPGTDirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PCMPGTWirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PMAXSWirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PMAXUBirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PMINSWirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PMINUBirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSLLDri")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSLLDrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSLLQri")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSLLQrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSLLWri")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSLLWrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSRADri")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSRADrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSRAWri")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSRAWrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSRLDri")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSRLDrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSRLQri")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSRLQrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSRLWri")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSRLWrr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSUBSBirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSUBSWirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSUBUSBirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "MMX_PSUBUSWirr")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVB2MZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVB2MZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVB2MZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVD2MZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVD2MZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVD2MZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVQ2MZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVQ2MZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVQ2MZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVW2MZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVW2MZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup1], (instregex "VPMOVW2MZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup2 : SchedWriteRes<[SKXPort1]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup2], (instregex "MMX_MASKMOVQ64")>;

def SKXWriteResGroup3 : SchedWriteRes<[SKXPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup3], (instregex "COMP_FST0r")>;
def: InstRW<[SKXWriteResGroup3], (instregex "COM_FST0r")>;
def: InstRW<[SKXWriteResGroup3], (instregex "INSERTPSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "KMOVBkr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "KMOVDkr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "KMOVQkr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "KMOVWkr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_MOVD64rr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_MOVD64to64rr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_PALIGNR64irr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_PSHUFBrr64")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_PSHUFWri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_PUNPCKHBWirr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_PUNPCKHDQirr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_PUNPCKHWDirr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_PUNPCKLBWirr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_PUNPCKLDQirr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MMX_PUNPCKLWDirr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MOV64toPQIrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MOVDDUPrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MOVDI2PDIrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MOVHLPSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MOVLHPSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MOVSDrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MOVSHDUPrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MOVSLDUPrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MOVUPDrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "MOVUPSrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PACKSSDWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PACKSSWBrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PACKUSDWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PACKUSWBrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PALIGNRrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PBLENDWrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVSXBDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVSXBQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVSXBWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVSXDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVSXWDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVSXWQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVZXBDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVZXBQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVZXBWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVZXDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVZXWDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PMOVZXWQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PSHUFBrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PSHUFDri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PSHUFHWri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PSHUFLWri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PSLLDQri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PSRLDQri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PUNPCKHBWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PUNPCKHDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PUNPCKHQDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PUNPCKHWDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PUNPCKLBWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PUNPCKLDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PUNPCKLQDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "PUNPCKLWDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "SHUFPDrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "SHUFPSrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "UCOM_FPr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "UCOM_Fr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "UNPCKHPDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "UNPCKHPSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "UNPCKLPDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "UNPCKLPSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VBROADCASTI32X2Z128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VBROADCASTSSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VINSERTPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VINSERTPSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOV64toPQIZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOV64toPQIrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVDDUPYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVDDUPZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVDDUPZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVDDUPZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVDDUPrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVDI2PDIZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVDI2PDIrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVHLPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVHLPSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVLHPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVLHPSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSDrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSHDUPYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSHDUPZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSHDUPZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSHDUPZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSHDUPrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSLDUPYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSLDUPZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSLDUPZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSLDUPZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSLDUPrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVSSZrr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVUPDYrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVUPDrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVUPSYrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VMOVUPSrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKSSDWYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKSSDWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKSSDWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKSSDWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKSSDWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKSSWBYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKSSWBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKSSWBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKSSWBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKSSWBrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKUSDWYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKUSDWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKUSDWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKUSDWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKUSDWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKUSWBYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKUSWBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKUSWBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKUSWBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPACKUSWBrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPALIGNRYrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPALIGNRZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPALIGNRZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPALIGNRZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPALIGNRrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPBLENDWYrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPBLENDWrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPBROADCASTDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPBROADCASTQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPDYri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPDYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPDZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPDZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPDZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPDri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPSYri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPSYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPSZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPSZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPSZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPSri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPERMILPSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVSXBDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVSXBQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVSXBWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVSXDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVSXWDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVSXWQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVZXBDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVZXBQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVZXBWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVZXDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVZXWDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPMOVZXWQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFBYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFBrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFDYri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFDZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFDZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFDZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFDri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFHWYri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFHWZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFHWZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFHWZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFHWri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFLWYri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFLWZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFLWZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFLWZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSHUFLWri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSLLDQYri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSLLDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSLLDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSLLDQZ512rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSLLDQri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSRLDQYri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSRLDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSRLDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSRLDQZ512rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPSRLDQri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHBWYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHBWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHBWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHBWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHBWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHDQYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHQDQYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHQDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHQDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHQDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHQDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHWDYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHWDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHWDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHWDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKHWDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLBWYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLBWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLBWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLBWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLBWrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLDQYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLQDQYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLQDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLQDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLQDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLQDQrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLWDYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLWDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLWDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VPUNPCKLWDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VSHUFPDYrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VSHUFPDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VSHUFPDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VSHUFPDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VSHUFPDrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VSHUFPSYrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VSHUFPSZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VSHUFPSZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VSHUFPSZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VSHUFPSrri")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKHPDYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKHPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKHPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKHPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKHPDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKHPSYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKHPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKHPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKHPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKHPSrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKLPDYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKLPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKLPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKLPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKLPDrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKLPSYrr")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKLPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKLPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKLPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup3], (instregex "VUNPCKLPSrr")>;

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
def: InstRW<[SKXWriteResGroup5], (instregex "PABSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PABSDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PABSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PADDSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PADDSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PADDUSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PADDUSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PAVGBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PAVGWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PCMPEQBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PCMPEQDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PCMPEQQrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PCMPEQWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PCMPGTBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PCMPGTDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PCMPGTWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMAXSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMAX(C?)SDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMAXSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMAXUBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMAXUDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMAXUWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMINSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMIN(C?)SDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMINSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMINUBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMINUDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PMINUWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSIGNBrr128")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSIGNDrr128")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSIGNWrr128")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSLLDri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSLLQri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSLLWri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSRADri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSRAWri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSRLDri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSRLQri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSRLWri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSUBSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSUBSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSUBUSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "PSUBUSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSDYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPABSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDSBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDSBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDSBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDSBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDSWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDSWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDSWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDSWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDUSBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDUSBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDUSBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDUSBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDUSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDUSWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDUSWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDUSWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDUSWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPADDUSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPAVGBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPAVGBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPAVGBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPAVGBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPAVGBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPAVGWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPAVGWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPAVGWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPAVGWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPAVGWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPEQBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPEQBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPEQDYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPEQDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPEQQYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPEQQrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPEQWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPEQWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPGTBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPGTBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPGTDYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPGTDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPGTWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPCMPGTWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXSBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXSBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXSBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXSBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAX(C?)SDYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAX(C?)SDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAX(C?)SDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAX(C?)SDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAX(C?)SDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXSWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXSWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXSWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXSWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUDYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMAXUWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINSBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINSBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINSBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINSBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMIN(C?)SDYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMIN(C?)SDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMIN(C?)SDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMIN(C?)SDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMIN(C?)SDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINSWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINSWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINSWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINSWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUDYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPMINUWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLDZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLDZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLDZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLQZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLQZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLQZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLVDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLVDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLVDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLVQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLVQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPROLVQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORDZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORDZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORDZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORQZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORQZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORQZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORVDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORVDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORVDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORVQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORVQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPRORVQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSIGNBYrr256")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSIGNBrr128")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSIGNDYrr256")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSIGNDrr128")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSIGNWYrr256")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSIGNWrr128")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLDYri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLDZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLDZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLDZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLDri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLQYri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLQZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLQZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLQZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLQri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVDYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVQYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVQrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLVWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLWYri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLWZ128ri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLWZ256ri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLWZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSLLWri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRADYri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRADZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRADZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRADZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRADri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAQZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAQZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAQZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVDYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAVWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAWYri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAWZ128ri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAWZ256ri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAWZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRAWri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLDYri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLDZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLDZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLDZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLDri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLQYri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLQZ128r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLQZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLQZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLQri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVDYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVDrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVQYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVQrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLVWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLWYri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLWZ128ri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLWZ256ri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLWZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSRLWri")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBSBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBSBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBSBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBSBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBSWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBSWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBSWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBSWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBSWrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBUSBYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBUSBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBUSBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBUSBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBUSBrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBUSWYrr")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBUSWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBUSWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBUSWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup5], (instregex "VPSUBUSWrr")>;

def SKXWriteResGroup6 : SchedWriteRes<[SKXPort05]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup6], (instregex "FINCSTP")>;
def: InstRW<[SKXWriteResGroup6], (instregex "FNOP")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_MOVQ64rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PABSBrr64")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PABSDrr64")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PABSWrr64")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PADDBirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PADDDirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PADDQirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PADDWirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PANDNirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PANDirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PORirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PSIGNBrr64")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PSIGNDrr64")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PSIGNWrr64")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PSUBBirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PSUBDirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PSUBQirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PSUBWirr")>;
def: InstRW<[SKXWriteResGroup6], (instregex "MMX_PXORirr")>;

def SKXWriteResGroup7 : SchedWriteRes<[SKXPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup7], (instregex "ADC(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup7], (instregex "ADC(16|32|64)rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup7], (instregex "ADC8rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup7], (instregex "ADCX32rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "ADCX64rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "ADOX32rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "ADOX64rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "BT(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup7], (instregex "BT(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "BTC(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup7], (instregex "BTC(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "BTR(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup7], (instregex "BTR(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "BTS(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup7], (instregex "BTS(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CDQ")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CLAC")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVAE(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVB(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVE(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVG(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVGE(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVL(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVLE(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVNE(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVNO(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVNP(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVNS(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVO(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVP(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CMOVS(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "CQO")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JAE_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JAE_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JA_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JA_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JBE_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JBE_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JB_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JB_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JE_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JE_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JGE_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JGE_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JG_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JG_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JLE_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JLE_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JL_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JL_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JMP_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JMP_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JNE_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JNE_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JNO_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JNO_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JNP_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JNP_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JNS_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JNS_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JO_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JO_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JP_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JP_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JS_1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "JS_4")>;
def: InstRW<[SKXWriteResGroup7], (instregex "RORX32ri")>;
def: InstRW<[SKXWriteResGroup7], (instregex "RORX64ri")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SAR(16|32|64)r1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SAR(16|32|64)ri")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SAR8r1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SAR8ri")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SARX32rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SARX64rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SBB(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SBB(16|32|64)rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SBB8rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETAEr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETBr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETEr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETGEr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETGr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETLEr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETLr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETNEr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETNOr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETNPr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETNSr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETOr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETPr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SETSr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHL(16|32|64)r1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHL(16|32|64)ri")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHL8r1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHL8ri")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHLX32rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHLX64rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHR(16|32|64)r1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHR(16|32|64)ri")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHR8r1")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHR8ri")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHRX32rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "SHRX64rr")>;
def: InstRW<[SKXWriteResGroup7], (instregex "STAC")>;

def SKXWriteResGroup8 : SchedWriteRes<[SKXPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup8], (instregex "ANDN32rr")>;
def: InstRW<[SKXWriteResGroup8], (instregex "ANDN64rr")>;
def: InstRW<[SKXWriteResGroup8], (instregex "BLSI32rr")>;
def: InstRW<[SKXWriteResGroup8], (instregex "BLSI64rr")>;
def: InstRW<[SKXWriteResGroup8], (instregex "BLSMSK32rr")>;
def: InstRW<[SKXWriteResGroup8], (instregex "BLSMSK64rr")>;
def: InstRW<[SKXWriteResGroup8], (instregex "BLSR32rr")>;
def: InstRW<[SKXWriteResGroup8], (instregex "BLSR64rr")>;
def: InstRW<[SKXWriteResGroup8], (instregex "BZHI32rr")>;
def: InstRW<[SKXWriteResGroup8], (instregex "BZHI64rr")>;
def: InstRW<[SKXWriteResGroup8], (instregex "LEA(16|32|64)r")>;

def SKXWriteResGroup9 : SchedWriteRes<[SKXPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup9], (instregex "ANDNPDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "ANDNPSrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "ANDPDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "ANDPSrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "BLENDPDrri")>;
def: InstRW<[SKXWriteResGroup9], (instregex "BLENDPSrri")>;
def: InstRW<[SKXWriteResGroup9], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "MOVAPDrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "MOVAPSrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "MOVDQArr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "MOVDQUrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "MOVPQI2QIrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "MOVSSrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "ORPDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "ORPSrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PADDBrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PADDDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PADDQrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PADDWrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PANDNrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PANDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PORrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PSUBBrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PSUBDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PSUBQrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PSUBWrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "PXORrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDNPDYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDNPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDNPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDNPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDNPDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDNPSYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDNPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDNPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDNPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDNPSrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDPDYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDPDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDPSYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VANDPSrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDMPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDMPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDMPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDMPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDMPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDMPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDPDYrri")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDPDrri")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDPSYrri")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VBLENDPSrri")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVAPDYrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVAPDZ128rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVAPDZ256rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVAPDZrr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVAPDrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVAPSYrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVAPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVAPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVAPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVAPSrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQA32Z128rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQA32Z256rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQA32Zrr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQA64Z128rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQA64Z256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQA64Zrr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQAYrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQArr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU16Z128rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU16Z256rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU16Zrr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU32Z128rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU32Z256rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU32Zrr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU64Z128rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU64Z256rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU64Zrr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU8Z128rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU8Z256rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQU8Zrr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQUYrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVDQUrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVPQI(2Q|Lo2PQ)IZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVPQI2QIrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVSSrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVUPDZ128rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVUPDZ256rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVUPDZrr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVUPSZ128rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVUPSZ256rr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVUPSZrr(b?)(k?)(z?)(_REV?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VMOVZPQILo2PQIrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VORPDYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VORPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VORPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VORPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VORPDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VORPSYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VORPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VORPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VORPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VORPSrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDBYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDBrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDDYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDQYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDQrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDWYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPADDWrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDNDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDNDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDNDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDNQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDNQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDNQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDNYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDNrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPANDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDDYrri")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDDrri")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPBLENDMWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPORDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPORDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPORDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPORQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPORQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPORQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPORYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPORrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBBYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBBrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBDYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBQYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBQrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBWYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPSUBWrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPTERNLOGDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPTERNLOGDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPTERNLOGDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPTERNLOGQZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPTERNLOGQZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPTERNLOGQZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPXORDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPXORDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPXORDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPXORQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPXORQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPXORQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPXORYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VPXORrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VXORPDYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VXORPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VXORPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VXORPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VXORPDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VXORPSYrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VXORPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VXORPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VXORPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup9], (instregex "VXORPSrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "XORPDrr")>;
def: InstRW<[SKXWriteResGroup9], (instregex "XORPSrr")>;

def SKXWriteResGroup10 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup10], (instregex "ADD(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "ADD(16|32|64)rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "ADD8i8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "ADD8ri")>;
def: InstRW<[SKXWriteResGroup10], (instregex "ADD8rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "AND(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "AND(16|32|64)rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "AND8i8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "AND8ri")>;
def: InstRW<[SKXWriteResGroup10], (instregex "AND8rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "CBW")>;
def: InstRW<[SKXWriteResGroup10], (instregex "CLC")>;
def: InstRW<[SKXWriteResGroup10], (instregex "CMC")>;
def: InstRW<[SKXWriteResGroup10], (instregex "CMP(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "CMP(16|32|64)rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "CMP8i8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "CMP8ri")>;
def: InstRW<[SKXWriteResGroup10], (instregex "CMP8rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "CWDE")>;
def: InstRW<[SKXWriteResGroup10], (instregex "DEC(16|32|64)r")>;
def: InstRW<[SKXWriteResGroup10], (instregex "DEC8r")>;
def: InstRW<[SKXWriteResGroup10], (instregex "INC(16|32|64)r")>;
def: InstRW<[SKXWriteResGroup10], (instregex "INC8r")>;
def: InstRW<[SKXWriteResGroup10], (instregex "LAHF")>;
def: InstRW<[SKXWriteResGroup10], (instregex "MOV(16|32|64)rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "MOV8ri")>;
def: InstRW<[SKXWriteResGroup10], (instregex "MOV8ri_alt")>;
def: InstRW<[SKXWriteResGroup10], (instregex "MOV8rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "MOVSX(16|32|64)rr16")>;
def: InstRW<[SKXWriteResGroup10], (instregex "MOVSX(16|32|64)rr32")>;
def: InstRW<[SKXWriteResGroup10], (instregex "MOVSX(16|32|64)rr8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "MOVZX(16|32|64)rr16")>;
def: InstRW<[SKXWriteResGroup10], (instregex "MOVZX(16|32|64)rr8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "NEG(16|32|64)r")>;
def: InstRW<[SKXWriteResGroup10], (instregex "NEG8r")>;
def: InstRW<[SKXWriteResGroup10], (instregex "NOOP")>;
def: InstRW<[SKXWriteResGroup10], (instregex "NOT(16|32|64)r")>;
def: InstRW<[SKXWriteResGroup10], (instregex "NOT8r")>;
def: InstRW<[SKXWriteResGroup10], (instregex "OR(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "OR(16|32|64)rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "OR8i8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "OR8ri")>;
def: InstRW<[SKXWriteResGroup10], (instregex "OR8rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SAHF")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SGDT64m")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SIDT64m")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SLDT64m")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SMSW16m")>;
def: InstRW<[SKXWriteResGroup10], (instregex "STC")>;
def: InstRW<[SKXWriteResGroup10], (instregex "STRm")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SUB(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SUB(16|32|64)rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SUB8i8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SUB8ri")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SUB8rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "SYSCALL")>;
def: InstRW<[SKXWriteResGroup10], (instregex "TEST(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup10], (instregex "TEST8i8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "TEST8ri")>;
def: InstRW<[SKXWriteResGroup10], (instregex "TEST8rr")>;
def: InstRW<[SKXWriteResGroup10], (instregex "XCHG(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup10], (instregex "XOR(16|32|64)ri8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "XOR(16|32|64)rr(_REV?)")>;
def: InstRW<[SKXWriteResGroup10], (instregex "XOR8i8")>;
def: InstRW<[SKXWriteResGroup10], (instregex "XOR8ri")>;
def: InstRW<[SKXWriteResGroup10], (instregex "XOR8rr(_REV?)")>;

def SKXWriteResGroup11 : SchedWriteRes<[SKXPort4,SKXPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup11], (instregex "FBSTPm")>;
def: InstRW<[SKXWriteResGroup11], (instregex "KMOVBmk")>;
def: InstRW<[SKXWriteResGroup11], (instregex "KMOVDmk")>;
def: InstRW<[SKXWriteResGroup11], (instregex "KMOVQmk")>;
def: InstRW<[SKXWriteResGroup11], (instregex "KMOVWmk")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MMX_MOVD64mr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MMX_MOVNTQmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MMX_MOVQ64mr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOV(16|32|64)mr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOV8mi")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOV8mr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVAPDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVAPSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVDQAmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVDQUmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVHPDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVHPSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVLPDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVLPSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVNTDQmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVNTI_64mr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVNTImr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVNTPDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVNTPSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVPDI2DImr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVPQI2QImr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVPQIto64mr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVSSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVUPDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "MOVUPSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "ST_FP32m")>;
def: InstRW<[SKXWriteResGroup11], (instregex "ST_FP64m")>;
def: InstRW<[SKXWriteResGroup11], (instregex "ST_FP80m")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTF128mr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTF32x4Z256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTF32x4Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTF32x8Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTF64x2Z256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTF64x2Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTF64x4Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTI128mr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTI32x4Z256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTI32x4Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTI32x8Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTI64x2Z256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTI64x2Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VEXTRACTI64x4Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVAPDYmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVAPDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVAPDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVAPDZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVAPDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVAPSYmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVAPSZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVAPSZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVAPSZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVAPSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQA32Z128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQA32Z256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQA32Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQA64Z128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQA64Z256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQA64Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQAYmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQAmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU16Z128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU16Z256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU16Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU32Z128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU32Z256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU32Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU64Z128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU64Z256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU64Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU8Z128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQU8Z256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQUYmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVDQUmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVHPDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVHPDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVHPSZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVHPSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVLPDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVLPDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVLPSZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVLPSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTDQYmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTDQZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTDQZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTDQZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTDQmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTPDYmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTPDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTPDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTPDZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTPDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTPSYmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTPSZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTPSZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTPSZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVNTPSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVPDI2DIZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVPDI2DImr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVPQI(2QI|to64)Zmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVPQI2QImr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVPQIto64mr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVSDZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVSDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVSSZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVSSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVUPDYmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVUPDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVUPDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVUPDZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVUPDmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVUPSYmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVUPSZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVUPSZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVUPSZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMOVUPSmr")>;
def: InstRW<[SKXWriteResGroup11], (instregex "VMPTRSTm")>;

def SKXWriteResGroup12 : SchedWriteRes<[SKXPort0]> {
  let Latency = 2;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup12], (instregex "COMISDrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "COMISSrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "MMX_MOVD64grr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "MMX_PMOVMSKBrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "MOVMSKPDrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "MOVMSKPSrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "MOVPDI2DIrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "MOVPQIto64rr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "PMOVMSKBrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "UCOMISDrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "UCOMISSrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VCOMISDZrb")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VCOMISDrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VCOMISSZrb")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VCOMISSrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VMOVMSKPDYrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VMOVMSKPDrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VMOVMSKPSYrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VMOVMSKPSrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VMOVPDI2DIZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VMOVPDI2DIrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VMOVPQIto64Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VMOVPQIto64rr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VPMOVMSKBYrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VPMOVMSKBrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VTESTPDYrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VTESTPDrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VTESTPSYrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VTESTPSrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VUCOMISDZrb")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VUCOMISDrr")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VUCOMISSZrb")>;
def: InstRW<[SKXWriteResGroup12], (instregex "VUCOMISSrr")>;

def SKXWriteResGroup13 : SchedWriteRes<[SKXPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup13], (instregex "MMX_MOVQ2DQrr")>;
def: InstRW<[SKXWriteResGroup13], (instregex "MMX_PINSRWirri")>;
def: InstRW<[SKXWriteResGroup13], (instregex "PINSRBrr")>;
def: InstRW<[SKXWriteResGroup13], (instregex "PINSRDrr")>;
def: InstRW<[SKXWriteResGroup13], (instregex "PINSRQrr")>;
def: InstRW<[SKXWriteResGroup13], (instregex "PINSRWrri")>;
def: InstRW<[SKXWriteResGroup13], (instregex "VPINSRBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup13], (instregex "VPINSRBrr")>;
def: InstRW<[SKXWriteResGroup13], (instregex "VPINSRDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup13], (instregex "VPINSRDrr")>;
def: InstRW<[SKXWriteResGroup13], (instregex "VPINSRQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup13], (instregex "VPINSRQrr")>;
def: InstRW<[SKXWriteResGroup13], (instregex "VPINSRWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup13], (instregex "VPINSRWrri")>;

def SKXWriteResGroup14 : SchedWriteRes<[SKXPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup14], (instregex "FDECSTP")>;
def: InstRW<[SKXWriteResGroup14], (instregex "MMX_MOVDQ2Qrr")>;

def SKXWriteResGroup15 : SchedWriteRes<[SKXPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup15], (instregex "CMOVA(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup15], (instregex "CMOVBE(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup15], (instregex "ROL(16|32|64)r1")>;
def: InstRW<[SKXWriteResGroup15], (instregex "ROL(16|32|64)ri")>;
def: InstRW<[SKXWriteResGroup15], (instregex "ROL8r1")>;
def: InstRW<[SKXWriteResGroup15], (instregex "ROL8ri")>;
def: InstRW<[SKXWriteResGroup15], (instregex "ROR(16|32|64)r1")>;
def: InstRW<[SKXWriteResGroup15], (instregex "ROR(16|32|64)ri")>;
def: InstRW<[SKXWriteResGroup15], (instregex "ROR8r1")>;
def: InstRW<[SKXWriteResGroup15], (instregex "ROR8ri")>;
def: InstRW<[SKXWriteResGroup15], (instregex "SETAr")>;
def: InstRW<[SKXWriteResGroup15], (instregex "SETBEr")>;

def SKXWriteResGroup16 : SchedWriteRes<[SKXPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup16], (instregex "BLENDVPDrr0")>;
def: InstRW<[SKXWriteResGroup16], (instregex "BLENDVPSrr0")>;
def: InstRW<[SKXWriteResGroup16], (instregex "PBLENDVBrr0")>;
def: InstRW<[SKXWriteResGroup16], (instregex "VBLENDVPDYrr")>;
def: InstRW<[SKXWriteResGroup16], (instregex "VBLENDVPDrr")>;
def: InstRW<[SKXWriteResGroup16], (instregex "VBLENDVPSYrr")>;
def: InstRW<[SKXWriteResGroup16], (instregex "VBLENDVPSrr")>;
def: InstRW<[SKXWriteResGroup16], (instregex "VPBLENDVBYrr")>;
def: InstRW<[SKXWriteResGroup16], (instregex "VPBLENDVBrr")>;

def SKXWriteResGroup17 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup17], (instregex "LFENCE")>;
def: InstRW<[SKXWriteResGroup17], (instregex "WAIT")>;
def: InstRW<[SKXWriteResGroup17], (instregex "XGETBV")>;

def SKXWriteResGroup18 : SchedWriteRes<[SKXPort0,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup18], (instregex "MMX_MASKMOVQ64")>;
def: InstRW<[SKXWriteResGroup18], (instregex "VMASKMOVDQU")>;
def: InstRW<[SKXWriteResGroup18], (instregex "VMASKMOVPDYmr")>;
def: InstRW<[SKXWriteResGroup18], (instregex "VMASKMOVPDmr")>;
def: InstRW<[SKXWriteResGroup18], (instregex "VMASKMOVPSYmr")>;
def: InstRW<[SKXWriteResGroup18], (instregex "VMASKMOVPSmr")>;
def: InstRW<[SKXWriteResGroup18], (instregex "VPMASKMOVDYmr")>;
def: InstRW<[SKXWriteResGroup18], (instregex "VPMASKMOVDmr")>;
def: InstRW<[SKXWriteResGroup18], (instregex "VPMASKMOVQYmr")>;
def: InstRW<[SKXWriteResGroup18], (instregex "VPMASKMOVQmr")>;

def SKXWriteResGroup19 : SchedWriteRes<[SKXPort5,SKXPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup19], (instregex "PSLLDrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "PSLLQrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "PSLLWrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "PSRADrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "PSRAWrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "PSRLDrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "PSRLQrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "PSRLWrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSLLDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSLLDrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSLLQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSLLQrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSLLWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSLLWrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSRADZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSRADrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSRAQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSRAWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSRAWrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSRLDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSRLDrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSRLQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSRLQrr")>;
def: InstRW<[SKXWriteResGroup19], (instregex "VPSRLWrr")>;

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
def: InstRW<[SKXWriteResGroup22], (instregex "BEXTR32rr")>;
def: InstRW<[SKXWriteResGroup22], (instregex "BEXTR64rr")>;
def: InstRW<[SKXWriteResGroup22], (instregex "BSWAP(16|32|64)r")>;

def SKXWriteResGroup23 : SchedWriteRes<[SKXPort06,SKXPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup23], (instregex "ADC8i8")>;
def: InstRW<[SKXWriteResGroup23], (instregex "ADC8ri")>;
def: InstRW<[SKXWriteResGroup23], (instregex "CWD")>;
def: InstRW<[SKXWriteResGroup23], (instregex "JRCXZ")>;
def: InstRW<[SKXWriteResGroup23], (instregex "SBB8i8")>;
def: InstRW<[SKXWriteResGroup23], (instregex "SBB8ri")>;

def SKXWriteResGroup24 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup24], (instregex "EXTRACTPSmr")>;
def: InstRW<[SKXWriteResGroup24], (instregex "PEXTRBmr")>;
def: InstRW<[SKXWriteResGroup24], (instregex "PEXTRDmr")>;
def: InstRW<[SKXWriteResGroup24], (instregex "PEXTRQmr")>;
def: InstRW<[SKXWriteResGroup24], (instregex "PEXTRWmr")>;
def: InstRW<[SKXWriteResGroup24], (instregex "STMXCSR")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VEXTRACTPSZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VEXTRACTPSmr")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VPEXTRBZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VPEXTRBmr")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VPEXTRDZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VPEXTRDmr")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VPEXTRQZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VPEXTRQmr")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VPEXTRWZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VPEXTRWmr")>;
def: InstRW<[SKXWriteResGroup24], (instregex "VSTMXCSR")>;

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
def: InstRW<[SKXWriteResGroup26], (instregex "SETAEm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETBm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETEm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETGEm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETGm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETLEm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETLm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETNEm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETNOm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETNPm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETNSm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETOm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETPm")>;
def: InstRW<[SKXWriteResGroup26], (instregex "SETSm")>;

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
def: InstRW<[SKXWriteResGroup28], (instregex "PUSH(16|32|64)r")>;
def: InstRW<[SKXWriteResGroup28], (instregex "PUSH(16|32|64)rmr")>;
def: InstRW<[SKXWriteResGroup28], (instregex "PUSH64i8")>;
def: InstRW<[SKXWriteResGroup28], (instregex "STOSB")>;
def: InstRW<[SKXWriteResGroup28], (instregex "STOSL")>;
def: InstRW<[SKXWriteResGroup28], (instregex "STOSQ")>;
def: InstRW<[SKXWriteResGroup28], (instregex "STOSW")>;

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
def: InstRW<[SKXWriteResGroup30], (instregex "KADDBrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KADDDrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KADDQrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KADDWrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KMOVBrk")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KMOVDrk")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KMOVQrk")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KMOVWrk")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KORTESTBrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KORTESTDrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KORTESTQrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KORTESTWrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KTESTBrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KTESTDrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KTESTQrr")>;
def: InstRW<[SKXWriteResGroup30], (instregex "KTESTWrr")>;

def SKXWriteResGroup31 : SchedWriteRes<[SKXPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup31], (instregex "BSF(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup31], (instregex "BSR(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup31], (instregex "IMUL64rr(i8?)")>;
def: InstRW<[SKXWriteResGroup31], (instregex "IMUL8r")>;
def: InstRW<[SKXWriteResGroup31], (instregex "LZCNT(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup31], (instregex "MUL8r")>;
def: InstRW<[SKXWriteResGroup31], (instregex "PDEP32rr")>;
def: InstRW<[SKXWriteResGroup31], (instregex "PDEP64rr")>;
def: InstRW<[SKXWriteResGroup31], (instregex "PEXT32rr")>;
def: InstRW<[SKXWriteResGroup31], (instregex "PEXT64rr")>;
def: InstRW<[SKXWriteResGroup31], (instregex "POPCNT(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup31], (instregex "SHLD(16|32|64)rri8")>;
def: InstRW<[SKXWriteResGroup31], (instregex "SHRD(16|32|64)rri8")>;
def: InstRW<[SKXWriteResGroup31], (instregex "TZCNT(16|32|64)rr")>;

def SKXWriteResGroup31_16 : SchedWriteRes<[SKXPort1, SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup31_16], (instregex "IMUL16rr(i8?)")>;

def SKXWriteResGroup31_32 : SchedWriteRes<[SKXPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
}
def: InstRW<[SKXWriteResGroup31_32], (instregex "IMUL32rr(i8?)")>;

def SKXWriteResGroup32 : SchedWriteRes<[SKXPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup32], (instregex "ADD_FPrST0")>;
def: InstRW<[SKXWriteResGroup32], (instregex "ADD_FST0r")>;
def: InstRW<[SKXWriteResGroup32], (instregex "ADD_FrST0")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KSHIFTLBri")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KSHIFTLDri")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KSHIFTLQri")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KSHIFTLWri")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KSHIFTRBri")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KSHIFTRDri")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KSHIFTRQri")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KSHIFTRWri")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KUNPCKBWrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KUNPCKDQrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "KUNPCKWDrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "MMX_PSADBWirr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "PCMPGTQrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "PSADBWrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "SUBR_FPrST0")>;
def: InstRW<[SKXWriteResGroup32], (instregex "SUBR_FST0r")>;
def: InstRW<[SKXWriteResGroup32], (instregex "SUBR_FrST0")>;
def: InstRW<[SKXWriteResGroup32], (instregex "SUB_FPrST0")>;
def: InstRW<[SKXWriteResGroup32], (instregex "SUB_FST0r")>;
def: InstRW<[SKXWriteResGroup32], (instregex "SUB_FrST0")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VALIGNDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VALIGNDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VALIGNDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VALIGNQZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VALIGNQZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VALIGNQZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTF32X2Z256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTF32X2Zr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTI32X2Z256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTI32X2Zr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTSDYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTSDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTSDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTSSYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTSSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTSSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VBROADCASTSSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VCMPPDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VCMPPDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VCMPPDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VCMPPSZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VCMPPSZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VCMPPSZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VCMPSDZrr(_Int?)(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VCMPSSZrr(_Int?)(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VDBPSADBWZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VDBPSADBWZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VDBPSADBWZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTF128rr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTF32x4Z256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTF32x4Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTF32x8Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTF64x2Z256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTF64x2Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTF64x4Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTI128rr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTI32x4Z256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTI32x4Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTI32x8Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTI64x2Z256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTI64x2Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VEXTRACTI64x4Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VFPCLASSPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VFPCLASSPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VFPCLASSPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VFPCLASSPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VFPCLASSPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VFPCLASSPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VFPCLASSSDrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VFPCLASSSSrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTF128rr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTF32x4Z256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTF32x4Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTF32x8Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTF64x2Z256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTF64x2Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTF64x4Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTI128rr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTI32x4Z256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTI32x4Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTI32x8Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTI64x2Z256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTI64x2Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VINSERTI64x4Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTBYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTBZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTBZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTBZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTBrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTDYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTDrZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTDrZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTDrZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTQYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTQZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTQZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTQZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTQrZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTQrZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTQrZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTWYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTWZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTWZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTWZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPBROADCASTWrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPBZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPBZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPBZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPEQWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTQYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTQrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPGTWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPQZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPQZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPQZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUBZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUBZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUBZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUQZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUQZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUQZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUWZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUWZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPUWZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPWZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPWZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPCMPWZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERM2F128rr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERM2I128rr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMDYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2D128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2D256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2Drr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2PD128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2PD256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2PDrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2PS128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2PS256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2PSrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2Q128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2Q256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMI2Qrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMPDYri")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMPDZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMPDZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMPSYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMQYri")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMQZ256r(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMQZri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2D128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2D256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2Drr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2PD128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2PD256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2PDrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2PS128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2PS256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2PSrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2Q128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2Q256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPERMT2Qrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMAXSQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMAXSQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMAXSQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMAXUQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMAXUQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMAXUQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMINSQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMINSQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMINSQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMINUQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMINUQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMINUQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVQDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVQDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVQDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBDYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBQYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBWYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXBWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXDQYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXWDYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXWDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXWDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXWDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXWQYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXWQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXWQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVSXWQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBDYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBQYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBWYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXBWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXDQYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXWDYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXWDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXWDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXWDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXWQYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXWQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXWQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPMOVZXWQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPSADBWYrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPSADBWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPSADBWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPSADBWrr")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTMWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VPTESTNMWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VSHUFF32X4Z256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VSHUFF32X4Zrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VSHUFF64X2Z256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VSHUFF64X2Zrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VSHUFI32X4Z256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VSHUFI32X4Zrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VSHUFI64X2Z256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup32], (instregex "VSHUFI64X2Zrri(b?)(k?)(z?)")>;

def SKXWriteResGroup33 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup33], (instregex "EXTRACTPSrr")>;
def: InstRW<[SKXWriteResGroup33], (instregex "MMX_PEXTRWirri")>;
def: InstRW<[SKXWriteResGroup33], (instregex "PEXTRBrr")>;
def: InstRW<[SKXWriteResGroup33], (instregex "PEXTRDrr")>;
def: InstRW<[SKXWriteResGroup33], (instregex "PEXTRQrr")>;
def: InstRW<[SKXWriteResGroup33], (instregex "PEXTRWri")>;
def: InstRW<[SKXWriteResGroup33], (instregex "PEXTRWrr_REV")>;
def: InstRW<[SKXWriteResGroup33], (instregex "PTESTrr")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VEXTRACTPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VEXTRACTPSrr")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPEXTRBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPEXTRBrr")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPEXTRDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPEXTRDrr")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPEXTRQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPEXTRQrr")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPEXTRWZrr(_REV?)")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPEXTRWri")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPEXTRWrr_REV")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPTESTYrr")>;
def: InstRW<[SKXWriteResGroup33], (instregex "VPTESTrr")>;

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
def: InstRW<[SKXWriteResGroup35], (instregex "ROL(16|32|64)rCL")>;
def: InstRW<[SKXWriteResGroup35], (instregex "ROL8rCL")>;
def: InstRW<[SKXWriteResGroup35], (instregex "ROR(16|32|64)rCL")>;
def: InstRW<[SKXWriteResGroup35], (instregex "ROR8rCL")>;
def: InstRW<[SKXWriteResGroup35], (instregex "SAR(16|32|64)rCL")>;
def: InstRW<[SKXWriteResGroup35], (instregex "SAR8rCL")>;
def: InstRW<[SKXWriteResGroup35], (instregex "SHL(16|32|64)rCL")>;
def: InstRW<[SKXWriteResGroup35], (instregex "SHL8rCL")>;
def: InstRW<[SKXWriteResGroup35], (instregex "SHR(16|32|64)rCL")>;
def: InstRW<[SKXWriteResGroup35], (instregex "SHR8rCL")>;

def SKXWriteResGroup36 : SchedWriteRes<[SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKXWriteResGroup36], (instregex "XADD(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup36], (instregex "XADD8rr")>;
def: InstRW<[SKXWriteResGroup36], (instregex "XCHG8rr")>;

def SKXWriteResGroup37 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup37], (instregex "MMX_PHADDSWrr64")>;
def: InstRW<[SKXWriteResGroup37], (instregex "MMX_PHSUBSWrr64")>;

def SKXWriteResGroup38 : SchedWriteRes<[SKXPort5,SKXPort01]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup38], (instregex "PHADDSWrr128")>;
def: InstRW<[SKXWriteResGroup38], (instregex "PHSUBSWrr128")>;
def: InstRW<[SKXWriteResGroup38], (instregex "VPHADDSWrr128")>;
def: InstRW<[SKXWriteResGroup38], (instregex "VPHADDSWrr256")>;
def: InstRW<[SKXWriteResGroup38], (instregex "VPHSUBSWrr128")>;
def: InstRW<[SKXWriteResGroup38], (instregex "VPHSUBSWrr256")>;

def SKXWriteResGroup39 : SchedWriteRes<[SKXPort5,SKXPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup39], (instregex "MMX_PHADDWrr64")>;
def: InstRW<[SKXWriteResGroup39], (instregex "MMX_PHADDrr64")>;
def: InstRW<[SKXWriteResGroup39], (instregex "MMX_PHSUBDrr64")>;
def: InstRW<[SKXWriteResGroup39], (instregex "MMX_PHSUBWrr64")>;

def SKXWriteResGroup40 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup40], (instregex "PHADDDrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "PHADDWrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "PHSUBDrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "PHSUBWrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "VPHADDDYrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "VPHADDDrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "VPHADDWYrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "VPHADDWrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "VPHSUBDYrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "VPHSUBDrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "VPHSUBWYrr")>;
def: InstRW<[SKXWriteResGroup40], (instregex "VPHSUBWrr")>;

def SKXWriteResGroup41 : SchedWriteRes<[SKXPort5,SKXPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup41], (instregex "MMX_PACKSSDWirr")>;
def: InstRW<[SKXWriteResGroup41], (instregex "MMX_PACKSSWBirr")>;
def: InstRW<[SKXWriteResGroup41], (instregex "MMX_PACKUSWBirr")>;

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
def: InstRW<[SKXWriteResGroup44], (instregex "RCL(16|32|64)r1")>;
def: InstRW<[SKXWriteResGroup44], (instregex "RCL(16|32|64)ri")>;
def: InstRW<[SKXWriteResGroup44], (instregex "RCL8r1")>;
def: InstRW<[SKXWriteResGroup44], (instregex "RCL8ri")>;
def: InstRW<[SKXWriteResGroup44], (instregex "RCR(16|32|64)r1")>;
def: InstRW<[SKXWriteResGroup44], (instregex "RCR(16|32|64)ri")>;
def: InstRW<[SKXWriteResGroup44], (instregex "RCR8r1")>;
def: InstRW<[SKXWriteResGroup44], (instregex "RCR8ri")>;

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
def: InstRW<[SKXWriteResGroup46], (instregex "SETAm")>;
def: InstRW<[SKXWriteResGroup46], (instregex "SETBEm")>;

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
def: InstRW<[SKXWriteResGroup49], (instregex "AESDECLASTrr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "AESDECrr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "AESENCLASTrr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "AESENCrr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "MMX_PMADDUBSWrr64")>;
def: InstRW<[SKXWriteResGroup49], (instregex "MMX_PMADDWDirr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "MMX_PMULHRSWrr64")>;
def: InstRW<[SKXWriteResGroup49], (instregex "MMX_PMULHUWirr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "MMX_PMULHWirr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "MMX_PMULLWirr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "MMX_PMULUDQirr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "MUL_FPrST0")>;
def: InstRW<[SKXWriteResGroup49], (instregex "MUL_FST0r")>;
def: InstRW<[SKXWriteResGroup49], (instregex "MUL_FrST0")>;
def: InstRW<[SKXWriteResGroup49], (instregex "RCPPSr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "RCPSSr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "RSQRTPSr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "RSQRTSSr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VAESDECLASTrr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VAESDECrr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VAESENCLASTrr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VAESENCrr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRCP14PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRCP14PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRCP14PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRCP14PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRCP14SDrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRCP14SSrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRCPPSYr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRCPPSr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRCPSSr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRSQRT14PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRSQRT14PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRSQRT14PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRSQRT14PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRSQRT14SDrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRSQRT14SSrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRSQRTPSYr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRSQRTPSr")>;
def: InstRW<[SKXWriteResGroup49], (instregex "VRSQRTSSr")>;

def SKXWriteResGroup50 : SchedWriteRes<[SKXPort015]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup50], (instregex "ADDPDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "ADDPSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "ADDSDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "ADDSSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "ADDSUBPDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "ADDSUBPSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "CMPPDrri")>;
def: InstRW<[SKXWriteResGroup50], (instregex "CMPPSrri")>;
def: InstRW<[SKXWriteResGroup50], (instregex "CMPSSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "CVTDQ2PSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "CVTPS2DQrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "CVTTPS2DQrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MAX(C?)PDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MAX(C?)PSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MAX(C?)SDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MAX(C?)SSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MIN(C?)PDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MIN(C?)PSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MIN(C?)SDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MIN(C?)SSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MULPDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MULPSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MULSDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "MULSSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "PHMINPOSUWrr128")>;
def: InstRW<[SKXWriteResGroup50], (instregex "PMADDUBSWrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "PMADDWDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "PMULDQrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "PMULHRSWrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "PMULHUWrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "PMULHWrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "PMULLWrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "PMULUDQrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "SUBPDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "SUBPSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "SUBSDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "SUBSSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDPDYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDPDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDPSYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDPSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDSDZrr(_Int?)(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDSDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDSSZrr(_Int?)(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDSSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDSUBPDYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDSUBPDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDSUBPSYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VADDSUBPSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCMPPDYrri")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCMPPDrri")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCMPPSYrri")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCMPPSrri")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCMPSDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCMPSSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTDQ2PSYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTDQ2PSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTDQ2PSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTDQ2PSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTDQ2PSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPD2QQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPD2QQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPD2QQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPD2UQQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPD2UQQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPD2UQQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPS2DQYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPS2DQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPS2DQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPS2DQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPS2DQrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPS2UDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPS2UDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTPS2UDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTQQ2PDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTQQ2PDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTQQ2PDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPD2QQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPD2QQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPD2QQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPD2UQQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPD2UQQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPD2UQQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPS2DQYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPS2DQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPS2DQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPS2DQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPS2DQrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPS2UDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPS2UDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTTPS2UDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTUDQ2PSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTUDQ2PSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTUDQ2PSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTUQQ2PDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTUQQ2PDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VCVTUQQ2PDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFIXUPIMMPDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFIXUPIMMPDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFIXUPIMMPDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFIXUPIMMPSZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFIXUPIMMPSZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFIXUPIMMPSZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFIXUPIMMSDrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFIXUPIMMSSrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132SDZr(_Int?)(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132SSZr(_Int?)(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD132SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD213SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADD231SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB132PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB132PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB132PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB132PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB132PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB132PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB132PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB132PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB132PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB132PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB213PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB213PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB213PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB213PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB213PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB213PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB213PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB213PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB213PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB213PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB231PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB231PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB231PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB231PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB231PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB231PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB231PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB231PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB231PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMADDSUB231PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB132SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB213SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUB231SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD132PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD132PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD132PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD132PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD132PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD132PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD132PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD132PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD132PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD132PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD213PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD213PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD213PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD213PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD213PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD213PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD213PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD213PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD213PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD213PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD231PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD231PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD231PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD231PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD231PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD231PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD231PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD231PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD231PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFMSUBADD231PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD132SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD213SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMADD231SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB132SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB213SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231PDYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231PDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231PDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231PDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231PSYr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231PSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231PSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231PSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231SDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231SDr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231SSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VFNMSUB231SSr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETEXPPDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETEXPPDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETEXPPDr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETEXPPSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETEXPPSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETEXPPSr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETEXPSDr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETEXPSSr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETMANTPDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETMANTPDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETMANTPDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETMANTPSZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETMANTPSZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETMANTPSZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETMANTSDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VGETMANTSSZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)PDYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)PDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)PDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)PDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)PDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)PSYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)PSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)PSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)PSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)PSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)SDZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)SDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)SSZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMAX(C?)SSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)PDYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)PDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)PDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)PDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)PDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)PSYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)PSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)PSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)PSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)PSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)SDZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)SDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)SSZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMIN(C?)SSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULPDYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULPDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULPSYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULPSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULSDZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULSDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULSSZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VMULSSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPHMINPOSUWrr128")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPLZCNTDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPLZCNTDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPLZCNTDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPLZCNTQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPLZCNTQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPLZCNTQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMADDUBSWYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMADDUBSWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMADDUBSWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMADDUBSWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMADDUBSWrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMADDWDYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMADDWDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMADDWDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMADDWDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMADDWDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULDQYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULDQrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHRSWYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHRSWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHRSWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHRSWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHRSWrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHUWYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHUWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHUWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHUWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHUWrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHWYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULHWrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULLWYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULLWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULLWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULLWrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULUDQYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULUDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULUDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULUDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VPMULUDQrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VRANGEPDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VRANGEPDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VRANGEPDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VRANGEPSZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VRANGEPSZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VRANGEPSZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VRANGESDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VRANGESSZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VREDUCEPDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VREDUCEPDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VREDUCEPDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VREDUCEPSZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VREDUCEPSZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VREDUCEPSZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VREDUCESDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VREDUCESSZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSCALEFPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSCALEFPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSCALEFPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSCALEFPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSCALEFPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSCALEFPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSCALEFSDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSCALEFSSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBPDYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBPDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBPSYrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBPSrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBSDZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBSDrr")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBSSZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup50], (instregex "VSUBSSrr")>;

def SKXWriteResGroup51 : SchedWriteRes<[SKXPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup51], (instregex "MPSADBWrri")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VEXPANDPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VEXPANDPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VEXPANDPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VEXPANDPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VEXPANDPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VEXPANDPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VMPSADBWYrri")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VMPSADBWrri")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPEXPANDDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPEXPANDDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPEXPANDDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPEXPANDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPEXPANDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPEXPANDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVDBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVDBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVDBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVDWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVDWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVDWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVQBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVQBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVQBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVQWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVQWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVQWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSDBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSDBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSDBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSDWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSDWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSDWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSQBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSQBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSQBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSQDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSQDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSQDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSQWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSQWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSQWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSWBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSWBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVSWBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSDBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSDBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSDBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSDWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSDWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSDWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSQBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSQBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSQBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSQDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSQDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSQDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSQWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSQWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSQWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSWBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSWBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVUSWBZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVWBZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVWBZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup51], (instregex "VPMOVWBZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup52 : SchedWriteRes<[SKXPort1,SKXPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup52], (instregex "IMUL(32|64)r")>;
def: InstRW<[SKXWriteResGroup52], (instregex "MUL(32|64)r")>;
def: InstRW<[SKXWriteResGroup52], (instregex "MULX64rr")>;

def SKXWriteResGroup52_16 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def: InstRW<[SKXWriteResGroup52_16], (instregex "IMUL16r")>;
def: InstRW<[SKXWriteResGroup52_16], (instregex "MUL16r")>;

def SKXWriteResGroup53 : SchedWriteRes<[SKXPort5,SKXPort01]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup53], (instregex "VPSLLDYrr")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSLLDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSLLDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSLLQYrr")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSLLQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSLLQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSLLWYrr")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSLLWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSLLWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRADYrr")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRADZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRADZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRAQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRAQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRAWYrr")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRAWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRAWZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRLDYrr")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRLDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRLDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRLQYrr")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRLQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRLQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRLWYrr")>;
def: InstRW<[SKXWriteResGroup53], (instregex "VPSRLWZ256rr(b?)(k?)(z?)")>;

def SKXWriteResGroup54 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup54], (instregex "ISTT_FP16m")>;
def: InstRW<[SKXWriteResGroup54], (instregex "ISTT_FP32m")>;
def: InstRW<[SKXWriteResGroup54], (instregex "ISTT_FP64m")>;
def: InstRW<[SKXWriteResGroup54], (instregex "IST_F16m")>;
def: InstRW<[SKXWriteResGroup54], (instregex "IST_F32m")>;
def: InstRW<[SKXWriteResGroup54], (instregex "IST_FP16m")>;
def: InstRW<[SKXWriteResGroup54], (instregex "IST_FP32m")>;
def: InstRW<[SKXWriteResGroup54], (instregex "IST_FP64m")>;
def: InstRW<[SKXWriteResGroup54], (instregex "VPMOVQDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup54], (instregex "VPMOVQDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup54], (instregex "VPMOVQDZmr(b?)(k?)(z?)")>;

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
def: InstRW<[SKXWriteResGroup58], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MMX_MOVD64rm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MMX_MOVD64to64rm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MMX_MOVQ64rm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOV(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOV64toPQIrm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOV8rm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOVDDUPrm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOVDI2PDIrm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOVSSrm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOVSX(16|32|64)rm16")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOVSX(16|32|64)rm32")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOVSX(16|32|64)rm8")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOVZX(16|32|64)rm16")>;
def: InstRW<[SKXWriteResGroup58], (instregex "MOVZX(16|32|64)rm8")>;
def: InstRW<[SKXWriteResGroup58], (instregex "PREFETCHNTA")>;
def: InstRW<[SKXWriteResGroup58], (instregex "PREFETCHT0")>;
def: InstRW<[SKXWriteResGroup58], (instregex "PREFETCHT1")>;
def: InstRW<[SKXWriteResGroup58], (instregex "PREFETCHT2")>;
def: InstRW<[SKXWriteResGroup58], (instregex "VMOV64toPQIrm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "VMOVDDUPrm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "VMOVDI2PDIrm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "VMOVQI2PQIrm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "VMOVSDrm")>;
def: InstRW<[SKXWriteResGroup58], (instregex "VMOVSSrm")>;

def SKXWriteResGroup59 : SchedWriteRes<[SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup59], (instregex "VCVTSD2SSZrr_Int(b?)(k?)(z?)")>;

def SKXWriteResGroup60 : SchedWriteRes<[SKXPort0,SKXPort5]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup60], (instregex "CVTDQ2PDrr")>;
def: InstRW<[SKXWriteResGroup60], (instregex "MMX_CVTPI2PDirr")>;
def: InstRW<[SKXWriteResGroup60], (instregex "VCVTDQ2PDrr")>;

def SKXWriteResGroup61 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup61], (instregex "CVTPD2DQrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "CVTPD2PSrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "CVTPS2PDrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "CVTSD2SSrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "CVTSI2SD64rr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "CVTSI2SDrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "CVTSI2SSrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "CVTSS2SDrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "CVTTPD2DQrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "MMX_CVTPD2PIirr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "MMX_CVTPS2PIirr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "MMX_CVTTPD2PIirr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "MMX_CVTTPS2PIirr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTDQ2PDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPD2DQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPD2DQrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPD2PSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPD2PSrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPD2UDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPH2PSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPH2PSrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPS2PDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPS2PDrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPS2PHZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPS2PHrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPS2QQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTPS2UQQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTQQ2PSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTSD2SSrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTSI2SD64rr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTSI2SDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTSI2SDrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTSI2SSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTSI2SSrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTSI642SDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTSS2SDZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTSS2SDrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTTPD2DQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTTPD2DQrr")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTTPD2UDQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTTPS2QQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTTPS2UQQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTUDQ2PDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTUQQ2PSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTUSI2SDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTUSI2SSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup61], (instregex "VCVTUSI642SDZrr(b?)(k?)(z?)")>;

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
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup64], (instregex "MULX32rr")>;

def SKXWriteResGroup65 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort015]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup65], (instregex "VCVTPS2PHZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup65], (instregex "VCVTPS2PHZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup65], (instregex "VCVTPS2PHZmr(b?)(k?)(z?)")>;

def SKXWriteResGroup66 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVDBZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVDBZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVDBZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVDWZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVDWZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVDWZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVQBZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVQBZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVQBZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVQWZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVQWZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVQWZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSDBZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSDBZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSDBZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSDWZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSDWZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSDWZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSQBZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSQBZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSQBZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSQDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSQDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSQDZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSQWZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSQWZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSQWZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSWBZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSWBZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVSWBZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSDBZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSDBZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSDBZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSDWZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSDWZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSDWZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSQBZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSQBZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSQBZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSQDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSQDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSQDZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSQWZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSQWZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSQWZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSWBZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSWBZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVUSWBZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVWBZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVWBZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup66], (instregex "VPMOVWBZmr(b?)(k?)(z?)")>;

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
def: InstRW<[SKXWriteResGroup68], (instregex "CMPXCHG(16|32|64)rr")>;
def: InstRW<[SKXWriteResGroup68], (instregex "CMPXCHG8rr")>;

def SKXWriteResGroup69 : SchedWriteRes<[SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKXWriteResGroup69], (instregex "PUSHF16")>;
def: InstRW<[SKXWriteResGroup69], (instregex "PUSHF64")>;

def SKXWriteResGroup70 : SchedWriteRes<[SKXPort5]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup70], (instregex "PCLMULQDQrr")>;
def: InstRW<[SKXWriteResGroup70], (instregex "VPCLMULQDQrr")>;

def SKXWriteResGroup71 : SchedWriteRes<[SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup71], (instregex "LDDQUrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "MOVAPDrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "MOVAPSrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "MOVDQArm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "MOVDQUrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "MOVNTDQArm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "MOVSHDUPrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "MOVSLDUPrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "MOVUPDrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "MOVUPSrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VBROADCASTSSrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VLDDQUrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VMOVAPDrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VMOVAPSrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VMOVDQArm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VMOVDQUrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VMOVNTDQArm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VMOVSHDUPrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VMOVSLDUPrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VMOVUPDrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VMOVUPSrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VPBROADCASTDrm")>;
def: InstRW<[SKXWriteResGroup71], (instregex "VPBROADCASTQrm")>;

def SKXWriteResGroup72 : SchedWriteRes<[SKXPort0]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup72], (instregex "MMX_CVTPI2PSirr")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VCOMPRESSPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VCOMPRESSPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VCOMPRESSPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VCOMPRESSPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VCOMPRESSPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VCOMPRESSPSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VPCOMPRESSDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VPCOMPRESSDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VPCOMPRESSDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VPCOMPRESSQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VPCOMPRESSQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VPCOMPRESSQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VPERMWZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VPERMWZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup72], (instregex "VPERMWZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup73 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PADDSBirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PADDSWirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PADDUSBirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PADDUSWirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PAVGBirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PAVGWirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PCMPEQBirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PCMPEQDirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PCMPEQWirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PCMPGTBirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PCMPGTDirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PCMPGTWirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PMAXSWirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PMAXUBirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PMINSWirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PMINUBirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSLLDrm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSLLQrm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSLLWrm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSRADrm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSRAWrm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSRLDrm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSRLQrm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSRLWrm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSUBSBirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSUBSWirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSUBUSBirm")>;
def: InstRW<[SKXWriteResGroup73], (instregex "MMX_PSUBUSWirm")>;

def SKXWriteResGroup74 : SchedWriteRes<[SKXPort0,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup74], (instregex "CVTSD2SI64rr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "CVTSD2SIrr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "CVTSS2SI64rr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "CVTSS2SIrr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "CVTTSD2SI64rr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "CVTTSD2SIrr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSD2SI64Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSD2SI64rr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSD2SIZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSD2SIrr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSD2USI64Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSD2USIZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSS2SI64Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSS2SI64rr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSS2SIZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSS2SIrr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTSS2USIZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTTSD2SI64Zrb")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTTSD2SI64rr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTTSD2SIZrb")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTTSD2SIrr")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTTSD2USI64Zrb")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTTSD2USIZrb")>;
def: InstRW<[SKXWriteResGroup74], (instregex "VCVTTSS2USIZrb")>;

def SKXWriteResGroup75 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PALIGNR64irm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PINSRWirmi")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PSHUFBrm64")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PSHUFWmi")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PUNPCKHBWirm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PUNPCKHDQirm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PUNPCKHWDirm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PUNPCKLBWirm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PUNPCKLDQirm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MMX_PUNPCKLWDirm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MOVHPDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MOVHPSrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MOVLPDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "MOVLPSrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PINSRBrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PINSRDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PINSRQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PINSRWrmi")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVSXBDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVSXBQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVSXBWrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVSXDQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVSXWDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVSXWQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVZXBDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVZXBQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVZXBWrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVZXDQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVZXWDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "PMOVZXWQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VMOVHPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VMOVHPDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VMOVHPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VMOVHPSrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VMOVLPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VMOVLPDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VMOVLPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VMOVLPSrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPINSRBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPINSRBrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPINSRDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPINSRDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPINSRQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPINSRQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPINSRWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPINSRWrmi")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVSXBDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVSXBQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVSXBWrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVSXDQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVSXWDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVSXWQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVZXBDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVZXBQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVZXBWrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVZXDQrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVZXWDrm")>;
def: InstRW<[SKXWriteResGroup75], (instregex "VPMOVZXWQrm")>;

def SKXWriteResGroup76 : SchedWriteRes<[SKXPort6,SKXPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup76], (instregex "FARJMP64")>;
def: InstRW<[SKXWriteResGroup76], (instregex "JMP(16|32|64)m")>;

def SKXWriteResGroup77 : SchedWriteRes<[SKXPort23,SKXPort05]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PABSBrm64")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PABSDrm64")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PABSWrm64")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PADDBirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PADDDirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PADDQirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PADDWirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PANDNirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PANDirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PORirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PSIGNBrm64")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PSIGNDrm64")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PSIGNWrm64")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PSUBBirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PSUBDirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PSUBQirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PSUBWirm")>;
def: InstRW<[SKXWriteResGroup77], (instregex "MMX_PXORirm")>;

def SKXWriteResGroup78 : SchedWriteRes<[SKXPort23,SKXPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup78], (instregex "ADC(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "ADC8rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "ADCX32rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "ADCX64rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "ADOX32rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "ADOX64rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "BT(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVAE(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVB(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVE(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVG(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVGE(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVL(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVLE(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVNE(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVNO(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVNP(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVNS(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVO(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVP(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "CMOVS(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "RORX32mi")>;
def: InstRW<[SKXWriteResGroup78], (instregex "RORX64mi")>;
def: InstRW<[SKXWriteResGroup78], (instregex "SARX32rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "SARX64rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "SBB(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "SBB8rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "SHLX32rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "SHLX64rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "SHRX32rm")>;
def: InstRW<[SKXWriteResGroup78], (instregex "SHRX64rm")>;

def SKXWriteResGroup79 : SchedWriteRes<[SKXPort23,SKXPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup79], (instregex "ANDN32rm")>;
def: InstRW<[SKXWriteResGroup79], (instregex "ANDN64rm")>;
def: InstRW<[SKXWriteResGroup79], (instregex "BLSI32rm")>;
def: InstRW<[SKXWriteResGroup79], (instregex "BLSI64rm")>;
def: InstRW<[SKXWriteResGroup79], (instregex "BLSMSK32rm")>;
def: InstRW<[SKXWriteResGroup79], (instregex "BLSMSK64rm")>;
def: InstRW<[SKXWriteResGroup79], (instregex "BLSR32rm")>;
def: InstRW<[SKXWriteResGroup79], (instregex "BLSR64rm")>;
def: InstRW<[SKXWriteResGroup79], (instregex "BZHI32rm")>;
def: InstRW<[SKXWriteResGroup79], (instregex "BZHI64rm")>;
def: InstRW<[SKXWriteResGroup79], (instregex "MOVBE(16|32|64)rm")>;

def SKXWriteResGroup80 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup80], (instregex "VMOV(64to|QI2)PQIZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup80], (instregex "VMOVDI2PDIZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup81 : SchedWriteRes<[SKXPort23,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup81], (instregex "ADD(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "ADD8rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "AND(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "AND8rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "CMP(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup81], (instregex "CMP(16|32|64)mr")>;
def: InstRW<[SKXWriteResGroup81], (instregex "CMP(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "CMP8mi")>;
def: InstRW<[SKXWriteResGroup81], (instregex "CMP8mr")>;
def: InstRW<[SKXWriteResGroup81], (instregex "CMP8rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "OR(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "OR8rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "POP(16|32|64)r")>;
def: InstRW<[SKXWriteResGroup81], (instregex "POP(16|32|64)rmr")>;
def: InstRW<[SKXWriteResGroup81], (instregex "SUB(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "SUB8rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "TEST(16|32|64)mr")>;
def: InstRW<[SKXWriteResGroup81], (instregex "TEST8mi")>;
def: InstRW<[SKXWriteResGroup81], (instregex "TEST8mr")>;
def: InstRW<[SKXWriteResGroup81], (instregex "XOR(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup81], (instregex "XOR8rm")>;

def SKXWriteResGroup82 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup82], (instregex "CVTSI2SS64rr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "HADDPDrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "HADDPSrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "HSUBPDrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "HSUBPSrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VCVTSI2SS64rr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VCVTSI642SSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VCVTUSI642SSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VHADDPDYrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VHADDPDrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VHADDPSYrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VHADDPSrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VHSUBPDYrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VHSUBPDrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VHSUBPSYrr")>;
def: InstRW<[SKXWriteResGroup82], (instregex "VHSUBPSrr")>;

def SKXWriteResGroup83 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup83], (instregex "SHLD(16|32|64)rrCL")>;
def: InstRW<[SKXWriteResGroup83], (instregex "SHRD(16|32|64)rrCL")>;

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
def: InstRW<[SKXWriteResGroup86], (instregex "BTC(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup86], (instregex "BTR(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup86], (instregex "BTS(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SAR(16|32|64)m1")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SAR(16|32|64)mi")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SAR8m1")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SAR8mi")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SHL(16|32|64)m1")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SHL(16|32|64)mi")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SHL8m1")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SHL8mi")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SHR(16|32|64)m1")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SHR(16|32|64)mi")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SHR8m1")>;
def: InstRW<[SKXWriteResGroup86], (instregex "SHR8mi")>;

def SKXWriteResGroup87 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup87], (instregex "ADD(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup87], (instregex "ADD(16|32|64)mr")>;
def: InstRW<[SKXWriteResGroup87], (instregex "ADD8mi")>;
def: InstRW<[SKXWriteResGroup87], (instregex "ADD8mr")>;
def: InstRW<[SKXWriteResGroup87], (instregex "AND(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup87], (instregex "AND(16|32|64)mr")>;
def: InstRW<[SKXWriteResGroup87], (instregex "AND8mi")>;
def: InstRW<[SKXWriteResGroup87], (instregex "AND8mr")>;
def: InstRW<[SKXWriteResGroup87], (instregex "DEC(16|32|64)m")>;
def: InstRW<[SKXWriteResGroup87], (instregex "DEC8m")>;
def: InstRW<[SKXWriteResGroup87], (instregex "INC(16|32|64)m")>;
def: InstRW<[SKXWriteResGroup87], (instregex "INC8m")>;
def: InstRW<[SKXWriteResGroup87], (instregex "NEG(16|32|64)m")>;
def: InstRW<[SKXWriteResGroup87], (instregex "NEG8m")>;
def: InstRW<[SKXWriteResGroup87], (instregex "NOT(16|32|64)m")>;
def: InstRW<[SKXWriteResGroup87], (instregex "NOT8m")>;
def: InstRW<[SKXWriteResGroup87], (instregex "OR(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup87], (instregex "OR(16|32|64)mr")>;
def: InstRW<[SKXWriteResGroup87], (instregex "OR8mi")>;
def: InstRW<[SKXWriteResGroup87], (instregex "OR8mr")>;
def: InstRW<[SKXWriteResGroup87], (instregex "POP(16|32|64)rmm")>;
def: InstRW<[SKXWriteResGroup87], (instregex "PUSH(16|32|64)rmm")>;
def: InstRW<[SKXWriteResGroup87], (instregex "SUB(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup87], (instregex "SUB(16|32|64)mr")>;
def: InstRW<[SKXWriteResGroup87], (instregex "SUB8mi")>;
def: InstRW<[SKXWriteResGroup87], (instregex "SUB8mr")>;
def: InstRW<[SKXWriteResGroup87], (instregex "XOR(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup87], (instregex "XOR(16|32|64)mr")>;
def: InstRW<[SKXWriteResGroup87], (instregex "XOR8mi")>;
def: InstRW<[SKXWriteResGroup87], (instregex "XOR8mr")>;

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
def: InstRW<[SKXWriteResGroup89], (instregex "LD_F32m")>;
def: InstRW<[SKXWriteResGroup89], (instregex "LD_F64m")>;
def: InstRW<[SKXWriteResGroup89], (instregex "LD_F80m")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VBROADCASTF128")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VBROADCASTI128")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VBROADCASTSDYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VBROADCASTSSYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VLDDQUYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVAPDYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVAPSYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVDDUPYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVDQAYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVDQUYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVNTDQAYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVNTDQAZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVSHDUPYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVSLDUPYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVUPDYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VMOVUPSYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VPBROADCASTDYrm")>;
def: InstRW<[SKXWriteResGroup89], (instregex "VPBROADCASTQYrm")>;

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
def: InstRW<[SKXWriteResGroup91], (instregex "COMISDrm")>;
def: InstRW<[SKXWriteResGroup91], (instregex "COMISSrm")>;
def: InstRW<[SKXWriteResGroup91], (instregex "UCOMISDrm")>;
def: InstRW<[SKXWriteResGroup91], (instregex "UCOMISSrm")>;
def: InstRW<[SKXWriteResGroup91], (instregex "VCOMISDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup91], (instregex "VCOMISDrm")>;
def: InstRW<[SKXWriteResGroup91], (instregex "VCOMISSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup91], (instregex "VCOMISSrm")>;
def: InstRW<[SKXWriteResGroup91], (instregex "VUCOMISDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup91], (instregex "VUCOMISDrm")>;
def: InstRW<[SKXWriteResGroup91], (instregex "VUCOMISSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup91], (instregex "VUCOMISSrm")>;

def SKXWriteResGroup92 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup92], (instregex "INSERTPSrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PACKSSDWrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PACKSSWBrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PACKUSDWrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PACKUSWBrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PALIGNRrmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PBLENDWrmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PSHUFBrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PSHUFDmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PSHUFHWmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PSHUFLWmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PUNPCKHBWrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PUNPCKHDQrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PUNPCKHQDQrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PUNPCKHWDrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PUNPCKLBWrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PUNPCKLDQrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PUNPCKLQDQrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "PUNPCKLWDrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "SHUFPDrmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "SHUFPSrmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "UNPCKHPDrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "UNPCKHPSrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "UNPCKLPDrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "UNPCKLPSrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VINSERTPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VINSERTPSrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VMOVSDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VMOVSSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPACKSSDWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPACKSSDWrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPACKSSWBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPACKSSWBrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPACKUSDWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPACKUSDWrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPACKUSWBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPACKUSWBrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPALIGNRZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPALIGNRrmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPBLENDWrmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPBROADCASTBZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPBROADCASTBrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPBROADCASTWZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPBROADCASTWrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPERMILPDZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPERMILPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPERMILPDmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPERMILPDrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPERMILPSZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPERMILPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPERMILPSmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPERMILPSrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPSHUFBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPSHUFBrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPSHUFDZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPSHUFDmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPSHUFHWZ128mi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPSHUFHWmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPSHUFLWZ128mi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPSHUFLWmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPSLLDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPSRLDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKHBWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKHBWrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKHDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKHDQrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKHQDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKHQDQrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKHWDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKHWDrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKLBWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKLBWrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKLDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKLDQrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKLQDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKLQDQrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKLWDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VPUNPCKLWDrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VSHUFPDZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VSHUFPDrmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VSHUFPSZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VSHUFPSrmi")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VUNPCKHPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VUNPCKHPDrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VUNPCKHPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VUNPCKHPSrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VUNPCKLPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VUNPCKLPDrm")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VUNPCKLPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup92], (instregex "VUNPCKLPSrm")>;

def SKXWriteResGroup93 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTDQ2PDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTDQ2PDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPD2DQYrr")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPD2DQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPD2DQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPD2PSYrr")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPD2PSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPD2PSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPD2UDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPD2UDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPH2PSYrr")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPH2PSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPH2PSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPS2PDYrr")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPS2PDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPS2PDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPS2PHYrr")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPS2PHZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPS2PHZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPS2QQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPS2QQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPS2UQQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTPS2UQQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTQQ2PSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTQQ2PSZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTTPD2DQYrr")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTTPD2DQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTTPD2DQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTTPD2UDQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTTPD2UDQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTTPS2QQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTTPS2QQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTTPS2UQQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTTPS2UQQZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTUDQ2PDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTUDQ2PDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTUQQ2PSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup93], (instregex "VCVTUQQ2PSZrr(b?)(k?)(z?)")>;

def SKXWriteResGroup94 : SchedWriteRes<[SKXPort01,SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup94], (instregex "PABSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PABSDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PABSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PADDSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PADDSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PADDUSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PADDUSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PAVGBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PAVGWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PCMPEQBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PCMPEQDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PCMPEQQrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PCMPEQWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PCMPGTBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PCMPGTDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PCMPGTWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMAXSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMAX(C?)SDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMAXSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMAXUBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMAXUDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMAXUWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMINSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMIN(C?)SDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMINSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMINUBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMINUDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PMINUWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSIGNBrm128")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSIGNDrm128")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSIGNWrm128")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSLLDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSLLQrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSLLWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSRADrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSRAWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSRLDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSRLQrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSRLWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSUBSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSUBSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSUBUSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "PSUBUSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPABSBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPABSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPABSDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPABSDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPABSQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPABSWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPABSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPADDSBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPADDSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPADDSWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPADDSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPADDUSBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPADDUSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPADDUSWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPADDUSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPAVGBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPAVGBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPAVGWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPAVGWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPCMPEQBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPCMPEQDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPCMPEQQrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPCMPEQWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPCMPGTBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPCMPGTDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPCMPGTWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAXSBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAXSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAX(C?)SDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAX(C?)SDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAXSWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAXSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAXUBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAXUBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAXUDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAXUDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAXUWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMAXUWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMINSBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMINSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMIN(C?)SDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMIN(C?)SDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMINSWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMINSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMINUBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMINUBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMINUDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMINUDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMINUWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPMINUWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPROLDZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPROLQZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPROLVDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPROLVQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPRORDZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPRORQZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPRORVDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPRORVQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSIGNBrm128")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSIGNDrm128")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSIGNWrm128")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLDZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLQZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLQrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLVDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLVDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLVQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLVQrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLVWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLWZ128mi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSLLWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRADZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRADZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRADrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRAQZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRAQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRAVDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRAVDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRAVQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRAVWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRAWZ128mi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRAWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRAWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLDZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLQZ128m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLQrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLVDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLVDrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLVQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLVQrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLVWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLWZ128mi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSRLWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSUBSBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSUBSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSUBSWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSUBSWrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSUBUSBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSUBUSBrm")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSUBUSWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup94], (instregex "VPSUBUSWrm")>;

def SKXWriteResGroup95 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup95], (instregex "ANDNPDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "ANDNPSrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "ANDPDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "ANDPSrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "BLENDPDrmi")>;
def: InstRW<[SKXWriteResGroup95], (instregex "BLENDPSrmi")>;
def: InstRW<[SKXWriteResGroup95], (instregex "ORPDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "ORPSrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PADDBrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PADDDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PADDQrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PADDWrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PANDNrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PANDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PORrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PSUBBrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PSUBDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PSUBQrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PSUBWrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "PXORrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VANDNPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VANDNPDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VANDNPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VANDNPSrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VANDPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VANDPDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VANDPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VANDPSrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VBLENDMPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VBLENDMPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VBLENDPDrmi")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VBLENDPSrmi")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VBROADCASTI32X2Z128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VBROADCASTSSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VINSERTF128rm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VINSERTI128rm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMASKMOVPDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMASKMOVPSrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVAPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVAPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVDDUPZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVDQA32Z128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVDQA64Z128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVDQU16Z128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVDQU32Z128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVDQU64Z128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVDQU8Z128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVNTDQAZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVSHDUPZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVSLDUPZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVUPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VMOVUPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VORPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VORPDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VORPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VORPSrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPADDBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPADDBrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPADDDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPADDDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPADDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPADDQrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPADDWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPADDWrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPANDDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPANDNDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPANDNQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPANDNrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPANDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPANDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPBLENDDrmi")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPBLENDMBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPBLENDMDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPBLENDMQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPBLENDMWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPBROADCASTDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPBROADCASTQZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPMASKMOVDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPMASKMOVQrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPORDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPORQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPORrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPSUBBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPSUBBrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPSUBDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPSUBDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPSUBQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPSUBQrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPSUBWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPSUBWrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPTERNLOGDZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPTERNLOGQZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPXORDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPXORQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VPXORrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VXORPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VXORPDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VXORPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup95], (instregex "VXORPSrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "XORPDrm")>;
def: InstRW<[SKXWriteResGroup95], (instregex "XORPSrm")>;

def SKXWriteResGroup96 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup96], (instregex "MMX_PACKSSDWirm")>;
def: InstRW<[SKXWriteResGroup96], (instregex "MMX_PACKSSWBirm")>;
def: InstRW<[SKXWriteResGroup96], (instregex "MMX_PACKUSWBirm")>;

def SKXWriteResGroup97 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup97], (instregex "VPERMI2W128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup97], (instregex "VPERMI2W256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup97], (instregex "VPERMI2Wrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup97], (instregex "VPERMT2W128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup97], (instregex "VPERMT2W256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup97], (instregex "VPERMT2Wrr(b?)(k?)(z?)")>;

def SKXWriteResGroup98 : SchedWriteRes<[SKXPort23,SKXPort06]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup98], (instregex "CMOVA(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup98], (instregex "CMOVBE(16|32|64)rm")>;

def SKXWriteResGroup99 : SchedWriteRes<[SKXPort23,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup99], (instregex "LEAVE64")>;
def: InstRW<[SKXWriteResGroup99], (instregex "SCASB")>;
def: InstRW<[SKXWriteResGroup99], (instregex "SCASL")>;
def: InstRW<[SKXWriteResGroup99], (instregex "SCASQ")>;
def: InstRW<[SKXWriteResGroup99], (instregex "SCASW")>;

def SKXWriteResGroup100 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup100], (instregex "CVTTSS2SI64rr")>;
def: InstRW<[SKXWriteResGroup100], (instregex "CVTTSS2SIrr")>;
def: InstRW<[SKXWriteResGroup100], (instregex "VCVTSS2USI64Zrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup100], (instregex "VCVTTSS2SI64Zrb")>;
def: InstRW<[SKXWriteResGroup100], (instregex "VCVTTSS2SI64rr")>;
def: InstRW<[SKXWriteResGroup100], (instregex "VCVTTSS2SIZrb")>;
def: InstRW<[SKXWriteResGroup100], (instregex "VCVTTSS2SIrr")>;
def: InstRW<[SKXWriteResGroup100], (instregex "VCVTTSS2USI64Zrb")>;

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
def: InstRW<[SKXWriteResGroup102], (instregex "LDMXCSR")>;
def: InstRW<[SKXWriteResGroup102], (instregex "VLDMXCSR")>;

def SKXWriteResGroup103 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup103], (instregex "KMOVBkm")>;
def: InstRW<[SKXWriteResGroup103], (instregex "KMOVDkm")>;
def: InstRW<[SKXWriteResGroup103], (instregex "KMOVQkm")>;
def: InstRW<[SKXWriteResGroup103], (instregex "KMOVWkm")>;

def SKXWriteResGroup104 : SchedWriteRes<[SKXPort6,SKXPort23,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup104], (instregex "LRETQ")>;
def: InstRW<[SKXWriteResGroup104], (instregex "RETQ")>;

def SKXWriteResGroup105 : SchedWriteRes<[SKXPort23,SKXPort06,SKXPort15]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup105], (instregex "BEXTR32rm")>;
def: InstRW<[SKXWriteResGroup105], (instregex "BEXTR64rm")>;

def SKXWriteResGroup106 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort237]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup106], (instregex "VCOMPRESSPDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VCOMPRESSPDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VCOMPRESSPDZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VCOMPRESSPSZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VCOMPRESSPSZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VCOMPRESSPSZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VPCOMPRESSDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VPCOMPRESSDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VPCOMPRESSDZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VPCOMPRESSQZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VPCOMPRESSQZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup106], (instregex "VPCOMPRESSQZmr(b?)(k?)(z?)")>;

def SKXWriteResGroup107 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKXWriteResGroup107], (instregex "ROL(16|32|64)m1")>;
def: InstRW<[SKXWriteResGroup107], (instregex "ROL(16|32|64)mi")>;
def: InstRW<[SKXWriteResGroup107], (instregex "ROL8m1")>;
def: InstRW<[SKXWriteResGroup107], (instregex "ROL8mi")>;
def: InstRW<[SKXWriteResGroup107], (instregex "ROR(16|32|64)m1")>;
def: InstRW<[SKXWriteResGroup107], (instregex "ROR(16|32|64)mi")>;
def: InstRW<[SKXWriteResGroup107], (instregex "ROR8m1")>;
def: InstRW<[SKXWriteResGroup107], (instregex "ROR8mi")>;

def SKXWriteResGroup108 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKXWriteResGroup108], (instregex "XADD(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup108], (instregex "XADD8rm")>;

def SKXWriteResGroup109 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[SKXWriteResGroup109], (instregex "CALL(16|32|64)m")>;
def: InstRW<[SKXWriteResGroup109], (instregex "FARCALL64")>;

def SKXWriteResGroup110 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [1,2,2,2];
}
def: InstRW<[SKXWriteResGroup110], (instregex "VPSCATTERDQZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup110], (instregex "VPSCATTERQQZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup110], (instregex "VSCATTERDPDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup110], (instregex "VSCATTERQPDZ128mr(b?)(k?)(z?)")>;

def SKXWriteResGroup111 : SchedWriteRes<[SKXPort6,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,1,2];
}
def: InstRW<[SKXWriteResGroup111], (instregex "LOOP")>;

def SKXWriteResGroup112 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 11;
  let ResourceCycles = [1,4,4,2];
}
def: InstRW<[SKXWriteResGroup112], (instregex "VPSCATTERDQZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup112], (instregex "VPSCATTERQQZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup112], (instregex "VSCATTERDPDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup112], (instregex "VSCATTERQPDZ256mr(b?)(k?)(z?)")>;

def SKXWriteResGroup113 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 19;
  let ResourceCycles = [1,8,8,2];
}
def: InstRW<[SKXWriteResGroup113], (instregex "VPSCATTERDQZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup113], (instregex "VPSCATTERQQZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup113], (instregex "VSCATTERDPDZmr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup113], (instregex "VSCATTERQPDZmr(b?)(k?)(z?)")>;

def SKXWriteResGroup114 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 7;
  let NumMicroOps = 36;
  let ResourceCycles = [1,16,1,16,2];
}
def: InstRW<[SKXWriteResGroup114], (instregex "VSCATTERDPSZmr(b?)(k?)(z?)")>;

def SKXWriteResGroup115 : SchedWriteRes<[SKXPort0]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup115], (instregex "AESIMCrr")>;
def: InstRW<[SKXWriteResGroup115], (instregex "VAESIMCrr")>;

def SKXWriteResGroup116 : SchedWriteRes<[SKXPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKXWriteResGroup116], (instregex "PMULLDrr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "ROUNDPDr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "ROUNDPSr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "ROUNDSDr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "ROUNDSSr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VPMULLDYrr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VPMULLDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VPMULLDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VPMULLDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VPMULLDrr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VRNDSCALEPDZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VRNDSCALEPDZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VRNDSCALEPDZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VRNDSCALEPSZ128rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VRNDSCALEPSZ256rri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VRNDSCALEPSZrri(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VRNDSCALESDr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VRNDSCALESSr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VROUNDPDr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VROUNDPSr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VROUNDSDr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VROUNDSSr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VROUNDYPDr")>;
def: InstRW<[SKXWriteResGroup116], (instregex "VROUNDYPSr")>;

def SKXWriteResGroup117 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup117], (instregex "VTESTPDrm")>;
def: InstRW<[SKXWriteResGroup117], (instregex "VTESTPSrm")>;

def SKXWriteResGroup118 : SchedWriteRes<[SKXPort1,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup118], (instregex "BSF(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup118], (instregex "BSR(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup118], (instregex "IMUL64m")>;
def: InstRW<[SKXWriteResGroup118], (instregex "IMUL(32|64)rm(i8?)")>;
def: InstRW<[SKXWriteResGroup118], (instregex "IMUL8m")>;
def: InstRW<[SKXWriteResGroup118], (instregex "LZCNT(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup118], (instregex "MUL(16|32|64)m")>;
def: InstRW<[SKXWriteResGroup118], (instregex "MUL8m")>;
def: InstRW<[SKXWriteResGroup118], (instregex "PDEP32rm")>;
def: InstRW<[SKXWriteResGroup118], (instregex "PDEP64rm")>;
def: InstRW<[SKXWriteResGroup118], (instregex "PEXT32rm")>;
def: InstRW<[SKXWriteResGroup118], (instregex "PEXT64rm")>;
def: InstRW<[SKXWriteResGroup118], (instregex "POPCNT(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup118], (instregex "TZCNT(16|32|64)rm")>;

def SKXWriteResGroup118_16_1 : SchedWriteRes<[SKXPort1, SKXPort0156, SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1]; 
}
def: InstRW<[SKXWriteResGroup118_16_1], (instregex "IMUL16rm(i8?)")>;

def SKXWriteResGroup118_16_2 : SchedWriteRes<[SKXPort1, SKXPort0156, SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 5;
}
def: InstRW<[SKXWriteResGroup118_16_2], (instregex "IMUL16m")>;
def: InstRW<[SKXWriteResGroup118_16_2], (instregex "MUL16m")>;

def SKXWriteResGroup118_32 : SchedWriteRes<[SKXPort1, SKXPort0156, SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup118_32], (instregex "IMUL32m")>;
def: InstRW<[SKXWriteResGroup118_32], (instregex "MUL32m")>;

def SKXWriteResGroup119 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup119], (instregex "FCOM32m")>;
def: InstRW<[SKXWriteResGroup119], (instregex "FCOM64m")>;
def: InstRW<[SKXWriteResGroup119], (instregex "FCOMP32m")>;
def: InstRW<[SKXWriteResGroup119], (instregex "FCOMP64m")>;
def: InstRW<[SKXWriteResGroup119], (instregex "MMX_PSADBWirm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VFPCLASSSDrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKSSDWYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKSSDWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKSSDWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKSSWBYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKSSWBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKSSWBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKUSDWYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKUSDWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKUSDWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKUSWBYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKUSWBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPACKUSWBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPALIGNRYrmi")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPALIGNRZ256rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPALIGNRZrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPBLENDWYrmi")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPBROADCASTBYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPBROADCASTBZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPBROADCASTBZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPBROADCASTWYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPBROADCASTWZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPBROADCASTWZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPDYmi")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPDYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPDZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPDZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPSYmi")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPSYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPSZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPSZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPERMILPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPMOVSXBDYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPMOVSXBQYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPMOVSXWQYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFBYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFDYmi")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFDZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFDZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFHWYmi")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFHWZ128mi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFHWZmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFLWYmi")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFLWZ128mi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSHUFLWZmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSLLDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSLLDQZ512rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSRLDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPSRLDQZ512rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHBWYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHBWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHBWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHDQYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHQDQYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHQDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHQDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHWDYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHWDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKHWDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLBWYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLBWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLBWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLDQYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLQDQYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLQDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLQDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLWDYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLWDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VPUNPCKLWDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VSHUFPDYrmi")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VSHUFPDZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VSHUFPDZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VSHUFPSYrmi")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VSHUFPSZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VSHUFPSZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKHPDYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKHPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKHPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKHPSYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKHPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKHPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKLPDYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKLPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKLPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKLPSYrm")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKLPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup119], (instregex "VUNPCKLPSZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup120 : SchedWriteRes<[SKXPort01,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPABSWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDSBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDSBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDSBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDSWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDSWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDSWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDUSBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDUSBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDUSBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDUSWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDUSWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPADDUSWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPAVGBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPAVGBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPAVGBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPAVGWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPAVGWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPAVGWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPCMPEQBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPCMPEQDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPCMPEQQYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPCMPEQWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPCMPGTBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPCMPGTDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPCMPGTWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXSBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXSBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXSBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAX(C?)SDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAX(C?)SDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAX(C?)SDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXSWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXSWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXSWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXUBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXUBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXUBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXUDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXUDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXUDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXUWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXUWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMAXUWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINSBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINSBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINSBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMIN(C?)SDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMIN(C?)SDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMIN(C?)SDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINSWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINSWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINSWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINUBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINUBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINUBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINUDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINUDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINUDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINUWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINUWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPMINUWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPROLDZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPROLDZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPROLQZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPROLQZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPROLVDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPROLVDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPROLVQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPROLVQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPRORDZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPRORDZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPRORQZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPRORQZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPRORVDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPRORVDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPRORVQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPRORVQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSIGNBYrm256")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSIGNDYrm256")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSIGNWYrm256")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLDZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLDZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLQYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLQZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLQZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLVDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLVDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLVDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLVQYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLVQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLVQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLVWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLVWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLWZ256mi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLWZmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSLLWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRADYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRADZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRADZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRADZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRADZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAQZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAQZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAVDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAVDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAVDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAVQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAVQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAVWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAVWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAWZ256mi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAWZmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRAWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLDZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLDZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLQYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLQZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLQZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLVDYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLVDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLVDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLVQYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLVQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLVQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLVWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLVWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLWZ256mi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLWZmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSRLWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBSBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBSBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBSBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBSWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBSWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBSWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBUSBYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBUSBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBUSBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBUSWYrm")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBUSWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup120], (instregex "VPSUBUSWZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup121 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup121], (instregex "VANDNPDYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDNPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDNPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDNPSYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDNPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDNPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDPDYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDPSYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VANDPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBLENDMPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBLENDMPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBLENDMPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBLENDMPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBLENDPDYrmi")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBLENDPSYrmi")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTF32X2Z256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTF32X2Zm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTF32X4Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTF32X4rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTF32X8rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTF64X2Z128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTF64X2rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTF64X4rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTI32X2Z256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTI32X2Zm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTI32X4Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTI32X4rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTI32X8rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTI64X2Z128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTI64X2rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTI64X4rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTSDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTSDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTSSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VBROADCASTSSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTF32x4Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTF32x4Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTF32x8Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTF64x2Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTF64x2Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTF64x4Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTI32x4Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTI32x4Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTI32x8Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTI64x2Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTI64x2Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VINSERTI64x4Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMASKMOVPDYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMASKMOVPSYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVAPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVAPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVAPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVAPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDDUPZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDDUPZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQA32Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQA32Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQA64Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQA64Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQU16Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQU16Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQU32Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQU32Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQU64Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQU64Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQU8Z256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVDQU8Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVNTDQAZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVSHDUPZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVSHDUPZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVSLDUPZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVSLDUPZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVUPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVUPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVUPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VMOVUPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VORPDYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VORPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VORPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VORPSYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VORPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VORPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDBYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDDYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDQYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDWYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPADDWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPANDDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPANDDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPANDNDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPANDNDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPANDNQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPANDNQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPANDNYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPANDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPANDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPANDYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBLENDDYrmi")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBLENDMBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBLENDMBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBLENDMDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBLENDMDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBLENDMQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBLENDMQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBLENDMWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBLENDMWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBROADCASTDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBROADCASTDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBROADCASTQZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPBROADCASTQZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPMASKMOVDYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPMASKMOVQYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPORDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPORDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPORQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPORQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPORYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBBYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBDYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBQYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBWYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPSUBWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPTERNLOGDZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPTERNLOGDZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPTERNLOGQZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPTERNLOGQZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPXORDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPXORDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPXORQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPXORQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VPXORYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VXORPDYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VXORPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VXORPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VXORPSYrm")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VXORPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup121], (instregex "VXORPSZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup122 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup122], (instregex "BLENDVPDrm0")>;
def: InstRW<[SKXWriteResGroup122], (instregex "BLENDVPSrm0")>;
def: InstRW<[SKXWriteResGroup122], (instregex "PBLENDVBrm0")>;
def: InstRW<[SKXWriteResGroup122], (instregex "VBLENDVPDrm")>;
def: InstRW<[SKXWriteResGroup122], (instregex "VBLENDVPSrm")>;
def: InstRW<[SKXWriteResGroup122], (instregex "VPBLENDVBYrm")>;
def: InstRW<[SKXWriteResGroup122], (instregex "VPBLENDVBrm")>;

def SKXWriteResGroup123 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKXWriteResGroup123], (instregex "MMX_PHADDSWrm64")>;
def: InstRW<[SKXWriteResGroup123], (instregex "MMX_PHSUBSWrm64")>;

def SKXWriteResGroup124 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort05]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup124], (instregex "MMX_PHADDWrm64")>;
def: InstRW<[SKXWriteResGroup124], (instregex "MMX_PHADDrm64")>;
def: InstRW<[SKXWriteResGroup124], (instregex "MMX_PHSUBDrm64")>;
def: InstRW<[SKXWriteResGroup124], (instregex "MMX_PHSUBWrm64")>;

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
def: InstRW<[SKXWriteResGroup126], (instregex "ROR(16|32|64)mCL")>;
def: InstRW<[SKXWriteResGroup126], (instregex "ROR8mCL")>;

def SKXWriteResGroup127 : SchedWriteRes<[SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKXWriteResGroup127], (instregex "RCL(16|32|64)m1")>;
def: InstRW<[SKXWriteResGroup127], (instregex "RCL(16|32|64)mi")>;
def: InstRW<[SKXWriteResGroup127], (instregex "RCL8m1")>;
def: InstRW<[SKXWriteResGroup127], (instregex "RCL8mi")>;
def: InstRW<[SKXWriteResGroup127], (instregex "RCR(16|32|64)m1")>;
def: InstRW<[SKXWriteResGroup127], (instregex "RCR(16|32|64)mi")>;
def: InstRW<[SKXWriteResGroup127], (instregex "RCR8m1")>;
def: InstRW<[SKXWriteResGroup127], (instregex "RCR8mi")>;

def SKXWriteResGroup128 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKXWriteResGroup128], (instregex "ROL(16|32|64)mCL")>;
def: InstRW<[SKXWriteResGroup128], (instregex "ROL8mCL")>;
def: InstRW<[SKXWriteResGroup128], (instregex "SAR(16|32|64)mCL")>;
def: InstRW<[SKXWriteResGroup128], (instregex "SAR8mCL")>;
def: InstRW<[SKXWriteResGroup128], (instregex "SHL(16|32|64)mCL")>;
def: InstRW<[SKXWriteResGroup128], (instregex "SHL8mCL")>;
def: InstRW<[SKXWriteResGroup128], (instregex "SHR(16|32|64)mCL")>;
def: InstRW<[SKXWriteResGroup128], (instregex "SHR8mCL")>;

def SKXWriteResGroup129 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKXWriteResGroup129], (instregex "ADC(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup129], (instregex "ADC8mi")>;

def SKXWriteResGroup130 : SchedWriteRes<[SKXPort4,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKXWriteResGroup130], (instregex "ADC(16|32|64)mr")>;
def: InstRW<[SKXWriteResGroup130], (instregex "ADC8mr")>;
def: InstRW<[SKXWriteResGroup130], (instregex "CMPXCHG(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup130], (instregex "CMPXCHG8rm")>;
def: InstRW<[SKXWriteResGroup130], (instregex "SBB(16|32|64)mi8")>;
def: InstRW<[SKXWriteResGroup130], (instregex "SBB(16|32|64)mr")>;
def: InstRW<[SKXWriteResGroup130], (instregex "SBB8mi")>;
def: InstRW<[SKXWriteResGroup130], (instregex "SBB8mr")>;

def SKXWriteResGroup131 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 8;
  let ResourceCycles = [1,2,1,2,2];
}
def: InstRW<[SKXWriteResGroup131], (instregex "VPSCATTERQDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup131], (instregex "VPSCATTERQDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup131], (instregex "VSCATTERQPSZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup131], (instregex "VSCATTERQPSZ256mr(b?)(k?)(z?)")>;

def SKXWriteResGroup132 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 12;
  let ResourceCycles = [1,4,1,4,2];
}
def: InstRW<[SKXWriteResGroup132], (instregex "VPSCATTERDDZ128mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup132], (instregex "VSCATTERDPSZ128mr(b?)(k?)(z?)")>;

def SKXWriteResGroup133 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 20;
  let ResourceCycles = [1,8,1,8,2];
}
def: InstRW<[SKXWriteResGroup133], (instregex "VPSCATTERDDZ256mr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup133], (instregex "VSCATTERDPSZ256mr(b?)(k?)(z?)")>;

def SKXWriteResGroup134 : SchedWriteRes<[SKXPort0,SKXPort4,SKXPort5,SKXPort237,SKXPort0156]> {
  let Latency = 8;
  let NumMicroOps = 36;
  let ResourceCycles = [1,16,1,16,2];
}
def: InstRW<[SKXWriteResGroup134], (instregex "VPSCATTERDDZmr(b?)(k?)(z?)")>;

def SKXWriteResGroup135 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup135], (instregex "MMX_CVTPI2PSirm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "MMX_PMADDUBSWrm64")>;
def: InstRW<[SKXWriteResGroup135], (instregex "MMX_PMADDWDirm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "MMX_PMULHRSWrm64")>;
def: InstRW<[SKXWriteResGroup135], (instregex "MMX_PMULHUWirm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "MMX_PMULHWirm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "MMX_PMULLWirm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "MMX_PMULUDQirm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "RCPSSm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "RSQRTSSm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "VRCPSSm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "VRSQRTSSm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "VTESTPDYrm")>;
def: InstRW<[SKXWriteResGroup135], (instregex "VTESTPSYrm")>;

def SKXWriteResGroup136 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup136], (instregex "PCMPGTQrm")>;
def: InstRW<[SKXWriteResGroup136], (instregex "PSADBWrm")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VALIGNDZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VALIGNQZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VCMPPDZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VCMPPSZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VCMPSDZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VCMPSSZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VDBPSADBWZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VFPCLASSSSrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPBZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPDZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPEQBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPEQDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPEQQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPEQWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPGTBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPGTDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPGTQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPGTQrm")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPGTWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPQZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPUBZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPUDZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPUQZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPUWZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPCMPWZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPERMI2D128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPERMI2PD128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPERMI2PS128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPERMI2Q128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPERMT2D128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPERMT2PD128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPERMT2PS128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPERMT2Q128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMAXSQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMAXUQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMINSQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMINUQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVSXBDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVSXBQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVSXBWYrm")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVSXBWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVSXDQYrm")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVSXDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVSXWDYrm")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVSXWDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVSXWQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVZXBDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVZXBQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVZXBWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVZXDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVZXWDYrm")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVZXWDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPMOVZXWQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPSADBWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPSADBWrm")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPTESTMBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPTESTMDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPTESTMQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPTESTMWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPTESTNMBZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPTESTNMDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPTESTNMQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup136], (instregex "VPTESTNMWZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup137 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup137], (instregex "ADDSDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "ADDSSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "CMPSSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "CVTPS2PDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "MAX(C?)SDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "MAX(C?)SSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "MIN(C?)SDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "MIN(C?)SSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "MMX_CVTPS2PIirm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "MMX_CVTTPS2PIirm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "MULSDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "MULSSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "SUBSDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "SUBSSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VADDSDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VADDSSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VCMPSDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VCMPSSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VCVTPH2PSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VCVTPS2PDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMADD132SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMADD132SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMADD213SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMADD213SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMADD231SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMADD231SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMSUB132SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMSUB132SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMSUB213SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMSUB213SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMSUB231SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFMSUB231SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMADD132SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMADD132SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMADD213SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMADD213SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMADD231SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMADD231SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMSUB132SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMSUB132SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMSUB213SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMSUB213SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMSUB231SDm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VFNMSUB231SSm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VMAX(C?)SDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VMAX(C?)SSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VMIN(C?)SDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VMIN(C?)SSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VMULSDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VMULSSrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VSUBSDrm")>;
def: InstRW<[SKXWriteResGroup137], (instregex "VSUBSSrm")>;

def SKXWriteResGroup138 : SchedWriteRes<[SKXPort0,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup138], (instregex "VRCP14PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup138], (instregex "VRCP14PSZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup138], (instregex "VRSQRT14PDZr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup138], (instregex "VRSQRT14PSZr(b?)(k?)(z?)")>;

def SKXWriteResGroup139 : SchedWriteRes<[SKXPort5,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup139], (instregex "DPPDrri")>;
def: InstRW<[SKXWriteResGroup139], (instregex "VDPPDrri")>;

def SKXWriteResGroup140 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup140], (instregex "VBLENDVPDYrm")>;
def: InstRW<[SKXWriteResGroup140], (instregex "VBLENDVPSYrm")>;

def SKXWriteResGroup141 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup141], (instregex "PTESTrm")>;
def: InstRW<[SKXWriteResGroup141], (instregex "VPTESTrm")>;

def SKXWriteResGroup142 : SchedWriteRes<[SKXPort1,SKXPort5,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup142], (instregex "MULX64rm")>;

def SKXWriteResGroup143 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup143], (instregex "PHADDSWrm128")>;
def: InstRW<[SKXWriteResGroup143], (instregex "PHSUBSWrm128")>;
def: InstRW<[SKXWriteResGroup143], (instregex "VPHADDSWrm128")>;
def: InstRW<[SKXWriteResGroup143], (instregex "VPHSUBSWrm128")>;

def SKXWriteResGroup144 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup144], (instregex "PHADDDrm")>;
def: InstRW<[SKXWriteResGroup144], (instregex "PHADDWrm")>;
def: InstRW<[SKXWriteResGroup144], (instregex "PHSUBDrm")>;
def: InstRW<[SKXWriteResGroup144], (instregex "PHSUBWrm")>;
def: InstRW<[SKXWriteResGroup144], (instregex "VPHADDDrm")>;
def: InstRW<[SKXWriteResGroup144], (instregex "VPHADDWrm")>;
def: InstRW<[SKXWriteResGroup144], (instregex "VPHSUBDrm")>;
def: InstRW<[SKXWriteResGroup144], (instregex "VPHSUBWrm")>;

def SKXWriteResGroup145 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup145], (instregex "SHLD(16|32|64)mri8")>;
def: InstRW<[SKXWriteResGroup145], (instregex "SHRD(16|32|64)mri8")>;

def SKXWriteResGroup146 : SchedWriteRes<[SKXPort1,SKXPort6,SKXPort23,SKXPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup146], (instregex "LAR(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup146], (instregex "LSL(16|32|64)rm")>;

def SKXWriteResGroup147 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup147], (instregex "AESDECLASTrm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "AESDECrm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "AESENCLASTrm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "AESENCrm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "RCPPSm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "RSQRTPSm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VAESDECLASTrm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VAESDECrm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VAESENCLASTrm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VAESENCrm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VRCP14PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VRCP14PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VRCP14SDrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VRCP14SSrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VRCPPSm")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VRSQRT14PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VRSQRT14PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VRSQRT14SDrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VRSQRT14SSrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup147], (instregex "VRSQRTPSm")>;

def SKXWriteResGroup148 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup148], (instregex "ADD_F32m")>;
def: InstRW<[SKXWriteResGroup148], (instregex "ADD_F64m")>;
def: InstRW<[SKXWriteResGroup148], (instregex "ILD_F16m")>;
def: InstRW<[SKXWriteResGroup148], (instregex "ILD_F32m")>;
def: InstRW<[SKXWriteResGroup148], (instregex "ILD_F64m")>;
def: InstRW<[SKXWriteResGroup148], (instregex "SUBR_F32m")>;
def: InstRW<[SKXWriteResGroup148], (instregex "SUBR_F64m")>;
def: InstRW<[SKXWriteResGroup148], (instregex "SUB_F32m")>;
def: InstRW<[SKXWriteResGroup148], (instregex "SUB_F64m")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VALIGNDZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VALIGNDZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VALIGNQZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VALIGNQZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VCMPPDZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VCMPPDZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VCMPPSZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VCMPPSZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VDBPSADBWZ256rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VDBPSADBWZrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPBZ256rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPBZrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPDZ256rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPDZrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPEQBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPEQBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPEQDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPEQDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPEQQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPEQQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPEQWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPEQWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPGTBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPGTBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPGTDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPGTDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPGTQYrm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPGTQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPGTQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPGTWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPGTWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPQZ256rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPQZrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPUBZ256rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPUBZrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPUDZ256rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPUDZrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPUQZ256rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPUQZrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPUWZ256rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPUWZrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPWZ256rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPCMPWZrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERM2F128rm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERM2I128rm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMDYrm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMI2D256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMI2Drm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMI2PD256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMI2PDrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMI2PS256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMI2PSrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMI2Q256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMI2Qrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMPDYmi")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMPDZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMPDZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMPSYrm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMQYmi")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMQZ256m(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMQZm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMT2D256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMT2Drm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMT2PD256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMT2PDrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMT2PS256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMT2PSrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMT2Q256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPERMT2Qrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMAXSQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMAXSQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMAXUQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMAXUQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMINSQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMINSQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMINUQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMINUQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXBDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXBDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXBQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXBQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXBWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXBWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXWDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXWDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXWQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVSXWQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXBDYrm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXBDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXBDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXBQYrm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXBQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXBQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXBWYrm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXBWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXBWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXDQYrm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXWDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXWDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXWQYrm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXWQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPMOVZXWQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPSADBWYrm")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPSADBWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTMBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTMBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTMDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTMDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTMQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTMQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTMWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTMWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTNMBZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTNMBZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTNMDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTNMDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTNMQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTNMQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTNMWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VPTESTNMWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VSHUFF32X4Z256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VSHUFF32X4Zrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VSHUFF64X2Z256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VSHUFF64X2Zrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VSHUFI32X4Z256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VSHUFI32X4Zrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VSHUFI64X2Z256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup148], (instregex "VSHUFI64X2Zrm(b?)i(k?)(z?)")>;

def SKXWriteResGroup149 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup149], (instregex "ADDPDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "ADDPSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "ADDSUBPDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "ADDSUBPSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "CMPPDrmi")>;
def: InstRW<[SKXWriteResGroup149], (instregex "CMPPSrmi")>;
def: InstRW<[SKXWriteResGroup149], (instregex "CVTDQ2PSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "CVTPS2DQrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "CVTSS2SDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "CVTTPS2DQrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "MAX(C?)PDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "MAX(C?)PSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "MIN(C?)PDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "MIN(C?)PSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "MULPDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "MULPSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "PHMINPOSUWrm128")>;
def: InstRW<[SKXWriteResGroup149], (instregex "PMADDUBSWrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "PMADDWDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "PMULDQrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "PMULHRSWrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "PMULHUWrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "PMULHWrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "PMULLWrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "PMULUDQrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "SUBPDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "SUBPSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VADDPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VADDPDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VADDPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VADDPSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VADDSDZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VADDSSZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VADDSUBPDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VADDSUBPSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCMPPDrmi")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCMPPSrmi")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTDQ2PDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTDQ2PSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTDQ2PSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTPD2QQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTPD2UQQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTPH2PSYrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTPH2PSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTPS2DQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTPS2DQrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTPS2PDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTPS2QQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTPS2UDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTPS2UQQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTQQ2PDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTSS2SDZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTSS2SDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTTPD2QQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTTPD2UQQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTTPS2DQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTTPS2DQrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTTPS2QQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTTPS2UDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTTPS2UQQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTUDQ2PDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTUDQ2PSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VCVTUQQ2PDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFIXUPIMMPDZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFIXUPIMMPSZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFIXUPIMMSDrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFIXUPIMMSSrmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD132PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD132PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD132PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD132PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD132SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD132SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD213PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD213PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD213PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD213PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD213SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD213SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD231PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD231PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD231PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD231PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD231SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADD231SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB132PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB132PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB132PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB132PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB213PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB213PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB213PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB213PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB231PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB231PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB231PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMADDSUB231PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB132PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB132PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB132PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB132PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB132SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB132SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB213PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB213PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB213PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB213PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB213SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB213SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB231PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB231PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB231PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB231PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB231SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUB231SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD132PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD132PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD132PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD132PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD213PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD213PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD213PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD213PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD231PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD231PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD231PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFMSUBADD231PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD132PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD132PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD132PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD132PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD132SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD132SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD213PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD213PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD213PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD213PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD213SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD213SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD231PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD231PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD231PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD231PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD231SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMADD231SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB132PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB132PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB132PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB132PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB132SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB132SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB213PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB213PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB213PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB213PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB213SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB213SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB231PDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB231PDm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB231PSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB231PSm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB231SDZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VFNMSUB231SSZm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VGETEXPPDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VGETEXPPSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VGETEXPSDm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VGETEXPSSm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VGETMANTPDZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VGETMANTPSZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VGETMANTSDZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VGETMANTSSZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMAX(C?)PDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMAX(C?)PDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMAX(C?)PSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMAX(C?)PSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMAX(C?)SDZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMAX(C?)SSZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMIN(C?)PDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMIN(C?)PDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMIN(C?)PSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMIN(C?)PSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMIN(C?)SDZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMIN(C?)SSZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMULPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMULPDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMULPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMULPSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMULSDZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VMULSSZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPHMINPOSUWrm128")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPLZCNTDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPLZCNTQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMADDUBSWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMADDUBSWrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMADDWDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMADDWDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULDQrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULHRSWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULHRSWrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULHUWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULHUWrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULHWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULHWrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULLWZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULLWrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULUDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VPMULUDQrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VRANGEPDZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VRANGEPSZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VRANGESDZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VRANGESSZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VREDUCEPDZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VREDUCEPSZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VREDUCESDZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VREDUCESSZ128rmi(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VSCALEFPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VSCALEFPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VSCALEFSDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VSCALEFSSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VSUBPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VSUBPDrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VSUBPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VSUBPSrm")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VSUBSDZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup149], (instregex "VSUBSSZrm_Int(b?)(k?)(z?)")>;

def SKXWriteResGroup150 : SchedWriteRes<[SKXPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKXWriteResGroup150], (instregex "PCMPISTRIrr")>;
def: InstRW<[SKXWriteResGroup150], (instregex "PCMPISTRM128rr")>;
def: InstRW<[SKXWriteResGroup150], (instregex "VPCMPISTRIrr")>;
def: InstRW<[SKXWriteResGroup150], (instregex "VPCMPISTRM128rr")>;

def SKXWriteResGroup151 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup151], (instregex "MPSADBWrmi")>;
def: InstRW<[SKXWriteResGroup151], (instregex "VEXPANDPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup151], (instregex "VEXPANDPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup151], (instregex "VMPSADBWrmi")>;
def: InstRW<[SKXWriteResGroup151], (instregex "VPEXPANDDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup151], (instregex "VPEXPANDQZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup152 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup152], (instregex "MMX_CVTPI2PDirm")>;
def: InstRW<[SKXWriteResGroup152], (instregex "VPTESTYrm")>;

def SKXWriteResGroup153 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup153], (instregex "CVTSD2SSrm")>;
def: InstRW<[SKXWriteResGroup153], (instregex "VCVTSD2SSrm")>;

def SKXWriteResGroup154 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup154], (instregex "VPHADDSWrm256")>;
def: InstRW<[SKXWriteResGroup154], (instregex "VPHSUBSWrm256")>;

def SKXWriteResGroup155 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup155], (instregex "VPHADDDYrm")>;
def: InstRW<[SKXWriteResGroup155], (instregex "VPHADDWYrm")>;
def: InstRW<[SKXWriteResGroup155], (instregex "VPHSUBDYrm")>;
def: InstRW<[SKXWriteResGroup155], (instregex "VPHSUBWYrm")>;

def SKXWriteResGroup156 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort06,SKXPort0156]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKXWriteResGroup156], (instregex "MULX32rm")>;

def SKXWriteResGroup157 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 10;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,3];
}
def: InstRW<[SKXWriteResGroup157], (instregex "ADD8mi")>;
def: InstRW<[SKXWriteResGroup157], (instregex "AND8mi")>;
def: InstRW<[SKXWriteResGroup157], (instregex "OR8mi")>;
def: InstRW<[SKXWriteResGroup157], (instregex "SUB8mi")>;
def: InstRW<[SKXWriteResGroup157], (instregex "XCHG(16|32|64)rm")>;
def: InstRW<[SKXWriteResGroup157], (instregex "XCHG8rm")>;
def: InstRW<[SKXWriteResGroup157], (instregex "XOR8mi")>;

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
def: InstRW<[SKXWriteResGroup159], (instregex "DIVPSrr")>;
def: InstRW<[SKXWriteResGroup159], (instregex "DIVSSrr")>;
def: InstRW<[SKXWriteResGroup159], (instregex "VDIVPSYrr")>;
def: InstRW<[SKXWriteResGroup159], (instregex "VDIVPSZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup159], (instregex "VDIVPSZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup159], (instregex "VDIVPSrr")>;
def: InstRW<[SKXWriteResGroup159], (instregex "VDIVSSZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup159], (instregex "VDIVSSrr")>;

def SKXWriteResGroup160 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup160], (instregex "MUL_F32m")>;
def: InstRW<[SKXWriteResGroup160], (instregex "MUL_F64m")>;
def: InstRW<[SKXWriteResGroup160], (instregex "VRCP14PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup160], (instregex "VRCP14PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup160], (instregex "VRCPPSYm")>;
def: InstRW<[SKXWriteResGroup160], (instregex "VRSQRT14PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup160], (instregex "VRSQRT14PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup160], (instregex "VRSQRTPSYm")>;

def SKXWriteResGroup161 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup161], (instregex "VADDPDYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VADDPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VADDPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VADDPSYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VADDPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VADDPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VADDSUBPDYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VADDSUBPSYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCMPPDYrmi")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCMPPSYrmi")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTDQ2PDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTDQ2PDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTDQ2PSYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTDQ2PSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTDQ2PSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPD2QQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPD2QQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPD2UQQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPD2UQQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPH2PSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPH2PSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPS2DQYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPS2DQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPS2DQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPS2PDYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPS2PDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPS2PDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPS2QQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPS2UDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPS2UDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTPS2UQQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTQQ2PDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTQQ2PDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPD2QQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPD2QQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPD2UQQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPD2UQQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPS2DQYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPS2DQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPS2DQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPS2QQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPS2UDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPS2UDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTTPS2UQQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTUDQ2PDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTUDQ2PDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTUDQ2PSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTUDQ2PSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTUQQ2PDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VCVTUQQ2PDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFIXUPIMMPDZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFIXUPIMMPDZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFIXUPIMMPSZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFIXUPIMMPSZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD132PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD132PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD132PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD132PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD132PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD132PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD213PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD213PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD213PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD213PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD213PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD213PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD231PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD231PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD231PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD231PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD231PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADD231PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB132PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB132PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB132PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB132PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB132PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB132PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB213PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB213PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB213PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB213PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB213PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB213PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB231PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB231PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB231PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB231PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB231PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMADDSUB231PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB132PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB132PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB132PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB132PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB132PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB132PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB213PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB213PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB213PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB213PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB213PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB213PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB231PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB231PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB231PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB231PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB231PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUB231PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD132PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD132PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD132PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD132PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD132PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD132PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD213PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD213PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD213PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD213PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD213PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD213PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD231PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD231PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD231PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD231PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD231PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFMSUBADD231PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD132PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD132PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD132PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD132PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD132PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD132PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD213PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD213PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD213PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD213PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD213PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD213PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD231PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD231PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD231PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD231PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD231PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMADD231PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB132PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB132PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB132PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB132PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB132PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB132PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB213PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB213PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB213PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB213PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB213PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB213PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB231PDYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB231PDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB231PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB231PSYm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB231PSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VFNMSUB231PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VGETEXPPDZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VGETEXPPDm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VGETEXPPSZ256m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VGETEXPPSm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VGETMANTPDZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VGETMANTPDZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VGETMANTPSZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VGETMANTPSZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMAX(C?)PDYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMAX(C?)PDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMAX(C?)PDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMAX(C?)PSYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMAX(C?)PSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMAX(C?)PSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMIN(C?)PDYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMIN(C?)PDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMIN(C?)PDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMIN(C?)PSYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMIN(C?)PSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMIN(C?)PSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMULPDYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMULPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMULPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMULPSYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMULPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VMULPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPLZCNTDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPLZCNTDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPLZCNTQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPLZCNTQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMADDUBSWYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMADDUBSWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMADDUBSWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMADDWDYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMADDWDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMADDWDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULDQYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULHRSWYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULHRSWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULHRSWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULHUWYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULHUWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULHUWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULHWYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULHWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULHWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULLWYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULLWZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULUDQYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULUDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VPMULUDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VRANGEPDZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VRANGEPDZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VRANGEPSZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VRANGEPSZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VREDUCEPDZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VREDUCEPDZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VREDUCEPSZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VREDUCEPSZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VSCALEFPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VSCALEFPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VSCALEFPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VSCALEFPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VSUBPDYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VSUBPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VSUBPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VSUBPSYrm")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VSUBPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup161], (instregex "VSUBPSZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup162 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup162], (instregex "FICOM16m")>;
def: InstRW<[SKXWriteResGroup162], (instregex "FICOM32m")>;
def: InstRW<[SKXWriteResGroup162], (instregex "FICOMP16m")>;
def: InstRW<[SKXWriteResGroup162], (instregex "FICOMP32m")>;
def: InstRW<[SKXWriteResGroup162], (instregex "VEXPANDPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup162], (instregex "VEXPANDPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup162], (instregex "VEXPANDPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup162], (instregex "VEXPANDPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup162], (instregex "VMPSADBWYrmi")>;
def: InstRW<[SKXWriteResGroup162], (instregex "VPEXPANDDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup162], (instregex "VPEXPANDDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup162], (instregex "VPEXPANDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup162], (instregex "VPEXPANDQZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup163 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup163], (instregex "VCVTSD2SSZrm_Int(b?)(k?)(z?)")>;

def SKXWriteResGroup164 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup164], (instregex "CVTDQ2PDrm")>;
def: InstRW<[SKXWriteResGroup164], (instregex "VCVTDQ2PDrm")>;

def SKXWriteResGroup165 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup165], (instregex "CVTSD2SI64rm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "CVTSD2SIrm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "CVTSS2SI64rm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "CVTSS2SIrm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "CVTTSD2SI64rm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "CVTTSD2SIrm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "CVTTSS2SIrm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTSD2SI64Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTSD2SI64rm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTSD2SIZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTSD2SIrm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTSD2USI64Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTSS2SI64Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTSS2SI64rm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTSS2SIZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTSS2SIrm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTSS2USIZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTTSD2SI64Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTTSD2SI64rm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTTSD2SIZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTTSD2SIrm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTTSD2USI64Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTTSS2SI64Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTTSS2SI64rm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTTSS2SIZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTTSS2SIrm")>;
def: InstRW<[SKXWriteResGroup165], (instregex "VCVTTSS2USIZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup166 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup166], (instregex "CVTPD2DQrm")>;
def: InstRW<[SKXWriteResGroup166], (instregex "CVTPD2PSrm")>;
def: InstRW<[SKXWriteResGroup166], (instregex "CVTTPD2DQrm")>;
def: InstRW<[SKXWriteResGroup166], (instregex "MMX_CVTPD2PIirm")>;
def: InstRW<[SKXWriteResGroup166], (instregex "MMX_CVTTPD2PIirm")>;

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
def: InstRW<[SKXWriteResGroup168], (instregex "SHLD(16|32|64)mrCL")>;
def: InstRW<[SKXWriteResGroup168], (instregex "SHRD(16|32|64)mrCL")>;

def SKXWriteResGroup169 : SchedWriteRes<[SKXPort1,SKXPort06,SKXPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,3,2];
}
def: InstRW<[SKXWriteResGroup169], (instregex "RCL(16|32|64)rCL")>;
def: InstRW<[SKXWriteResGroup169], (instregex "RCR(16|32|64)rCL")>;

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
def: InstRW<[SKXWriteResGroup171], (instregex "LOOPE")>;
def: InstRW<[SKXWriteResGroup171], (instregex "LOOPNE")>;

def SKXWriteResGroup172 : SchedWriteRes<[SKXPort0]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup172], (instregex "SQRTPSr")>;
def: InstRW<[SKXWriteResGroup172], (instregex "SQRTSSr")>;
def: InstRW<[SKXWriteResGroup172], (instregex "VSQRTPSYr")>;
def: InstRW<[SKXWriteResGroup172], (instregex "VSQRTPSZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup172], (instregex "VSQRTPSZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup172], (instregex "VSQRTPSr")>;
def: InstRW<[SKXWriteResGroup172], (instregex "VSQRTSSZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup172], (instregex "VSQRTSSr")>;

def SKXWriteResGroup173 : SchedWriteRes<[SKXPort5,SKXPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup173], (instregex "PCLMULQDQrm")>;
def: InstRW<[SKXWriteResGroup173], (instregex "VPCLMULQDQrm")>;

def SKXWriteResGroup174 : SchedWriteRes<[SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKXWriteResGroup174], (instregex "VPMULLQZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup174], (instregex "VPMULLQZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup174], (instregex "VPMULLQZrr(b?)(k?)(z?)")>;

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
def: InstRW<[SKXWriteResGroup176], (instregex "VCVTSD2USIZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup176], (instregex "VCVTSS2USI64Zrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup176], (instregex "VCVTTSD2USIZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup176], (instregex "VCVTTSS2USI64Zrm(b?)(k?)(z?)")>;

def SKXWriteResGroup177 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup177], (instregex "VCVTPS2QQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup177], (instregex "VCVTPS2UQQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup177], (instregex "VCVTTPS2QQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup177], (instregex "VCVTTPS2UQQZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup178 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup178], (instregex "HADDPDrm")>;
def: InstRW<[SKXWriteResGroup178], (instregex "HADDPSrm")>;
def: InstRW<[SKXWriteResGroup178], (instregex "HSUBPDrm")>;
def: InstRW<[SKXWriteResGroup178], (instregex "HSUBPSrm")>;
def: InstRW<[SKXWriteResGroup178], (instregex "VHADDPDrm")>;
def: InstRW<[SKXWriteResGroup178], (instregex "VHADDPSrm")>;
def: InstRW<[SKXWriteResGroup178], (instregex "VHSUBPDrm")>;
def: InstRW<[SKXWriteResGroup178], (instregex "VHSUBPSrm")>;

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
def: InstRW<[SKXWriteResGroup180], (instregex "ADD_FI16m")>;
def: InstRW<[SKXWriteResGroup180], (instregex "ADD_FI32m")>;
def: InstRW<[SKXWriteResGroup180], (instregex "SUBR_FI16m")>;
def: InstRW<[SKXWriteResGroup180], (instregex "SUBR_FI32m")>;
def: InstRW<[SKXWriteResGroup180], (instregex "SUB_FI16m")>;
def: InstRW<[SKXWriteResGroup180], (instregex "SUB_FI32m")>;
def: InstRW<[SKXWriteResGroup180], (instregex "VPERMWZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup180], (instregex "VPERMWZrm(b?)(k?)(z?)")>;

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
def: InstRW<[SKXWriteResGroup182], (instregex "DPPSrri")>;
def: InstRW<[SKXWriteResGroup182], (instregex "VDPPSYrri")>;
def: InstRW<[SKXWriteResGroup182], (instregex "VDPPSrri")>;

def SKXWriteResGroup183 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup183], (instregex "VHADDPDYrm")>;
def: InstRW<[SKXWriteResGroup183], (instregex "VHADDPSYrm")>;
def: InstRW<[SKXWriteResGroup183], (instregex "VHSUBPDYrm")>;
def: InstRW<[SKXWriteResGroup183], (instregex "VHSUBPSYrm")>;
def: InstRW<[SKXWriteResGroup183], (instregex "VPERMI2W128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup183], (instregex "VPERMT2W128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup184 : SchedWriteRes<[SKXPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup184], (instregex "DIVPDrr")>;
def: InstRW<[SKXWriteResGroup184], (instregex "DIVSDrr")>;
def: InstRW<[SKXWriteResGroup184], (instregex "VDIVPDYrr")>;
def: InstRW<[SKXWriteResGroup184], (instregex "VDIVPDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup184], (instregex "VDIVPDZ256rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup184], (instregex "VDIVPDrr")>;
def: InstRW<[SKXWriteResGroup184], (instregex "VDIVSDZrr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup184], (instregex "VDIVSDrr")>;

def SKXWriteResGroup185 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup185], (instregex "AESIMCrm")>;
def: InstRW<[SKXWriteResGroup185], (instregex "VAESIMCrm")>;

def SKXWriteResGroup186 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup186], (instregex "PMULLDrm")>;
def: InstRW<[SKXWriteResGroup186], (instregex "ROUNDPDm")>;
def: InstRW<[SKXWriteResGroup186], (instregex "ROUNDPSm")>;
def: InstRW<[SKXWriteResGroup186], (instregex "ROUNDSDm")>;
def: InstRW<[SKXWriteResGroup186], (instregex "ROUNDSSm")>;
def: InstRW<[SKXWriteResGroup186], (instregex "VPMULLDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup186], (instregex "VPMULLDrm")>;
def: InstRW<[SKXWriteResGroup186], (instregex "VRNDSCALEPDZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup186], (instregex "VRNDSCALEPSZ128rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup186], (instregex "VRNDSCALESDm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup186], (instregex "VRNDSCALESSm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup186], (instregex "VROUNDPDm")>;
def: InstRW<[SKXWriteResGroup186], (instregex "VROUNDPSm")>;
def: InstRW<[SKXWriteResGroup186], (instregex "VROUNDSDm")>;
def: InstRW<[SKXWriteResGroup186], (instregex "VROUNDSSm")>;

def SKXWriteResGroup187 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup187], (instregex "MUL_FI16m")>;
def: InstRW<[SKXWriteResGroup187], (instregex "MUL_FI32m")>;

def SKXWriteResGroup188 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup188], (instregex "VCVTPD2DQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup188], (instregex "VCVTPD2PSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup188], (instregex "VCVTPD2UDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup188], (instregex "VCVTQQ2PSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup188], (instregex "VCVTTPD2DQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup188], (instregex "VCVTTPD2UDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup188], (instregex "VCVTUQQ2PSZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup189 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup189], (instregex "VPERMI2W256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup189], (instregex "VPERMI2Wrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup189], (instregex "VPERMT2W256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup189], (instregex "VPERMT2Wrm(b?)(k?)(z?)")>;

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
def: InstRW<[SKXWriteResGroup191], (instregex "DIVR_FPrST0")>;
def: InstRW<[SKXWriteResGroup191], (instregex "DIVR_FST0r")>;
def: InstRW<[SKXWriteResGroup191], (instregex "DIVR_FrST0")>;

def SKXWriteResGroup192 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 15;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKXWriteResGroup192], (instregex "VPMULLDYrm")>;
def: InstRW<[SKXWriteResGroup192], (instregex "VPMULLDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup192], (instregex "VPMULLDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup192], (instregex "VRNDSCALEPDZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup192], (instregex "VRNDSCALEPDZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup192], (instregex "VRNDSCALEPSZ256rm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup192], (instregex "VRNDSCALEPSZrm(b?)i(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup192], (instregex "VROUNDYPDm")>;
def: InstRW<[SKXWriteResGroup192], (instregex "VROUNDYPSm")>;

def SKXWriteResGroup193 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 15;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKXWriteResGroup193], (instregex "DPPDrmi")>;
def: InstRW<[SKXWriteResGroup193], (instregex "VDPPDrmi")>;

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
def: InstRW<[SKXWriteResGroup195], (instregex "RCL(16|32|64)mCL")>;
def: InstRW<[SKXWriteResGroup195], (instregex "RCL8mCL")>;

def SKXWriteResGroup196 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup196], (instregex "DIVSSrm")>;
def: InstRW<[SKXWriteResGroup196], (instregex "VDIVSSrm")>;

def SKXWriteResGroup197 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}
def: InstRW<[SKXWriteResGroup197], (instregex "PCMPISTRIrm")>;
def: InstRW<[SKXWriteResGroup197], (instregex "PCMPISTRM128rm")>;
def: InstRW<[SKXWriteResGroup197], (instregex "VPCMPISTRIrm")>;
def: InstRW<[SKXWriteResGroup197], (instregex "VPCMPISTRM128rm")>;

def SKXWriteResGroup198 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup198], (instregex "VRCP14PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup198], (instregex "VRCP14PSZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup198], (instregex "VRSQRT14PDZm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup198], (instregex "VRSQRT14PSZm(b?)(k?)(z?)")>;

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
def: InstRW<[SKXWriteResGroup201], (instregex "DIVPSrm")>;
def: InstRW<[SKXWriteResGroup201], (instregex "SQRTSSm")>;
def: InstRW<[SKXWriteResGroup201], (instregex "VDIVPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup201], (instregex "VDIVPSrm")>;
def: InstRW<[SKXWriteResGroup201], (instregex "VDIVSSZrm_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup201], (instregex "VSQRTSSm")>;

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
def: InstRW<[SKXWriteResGroup203], (instregex "SQRTPDr")>;
def: InstRW<[SKXWriteResGroup203], (instregex "SQRTSDr")>;
def: InstRW<[SKXWriteResGroup203], (instregex "VSQRTPDYr")>;
def: InstRW<[SKXWriteResGroup203], (instregex "VSQRTPDZ128r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup203], (instregex "VSQRTPDZ256r(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup203], (instregex "VSQRTPDr")>;
def: InstRW<[SKXWriteResGroup203], (instregex "VSQRTSDZr_Int(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup203], (instregex "VSQRTSDr")>;

def SKXWriteResGroup204 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup204], (instregex "SQRTPSm")>;
def: InstRW<[SKXWriteResGroup204], (instregex "VDIVPSYrm")>;
def: InstRW<[SKXWriteResGroup204], (instregex "VDIVPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup204], (instregex "VSQRTPSZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup204], (instregex "VSQRTPSm")>;
def: InstRW<[SKXWriteResGroup204], (instregex "VSQRTSSZm_Int(b?)(k?)(z?)")>;

def SKXWriteResGroup205 : SchedWriteRes<[SKXPort23,SKXPort015]> {
  let Latency = 18;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup205], (instregex "VPMULLQZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup206 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def: InstRW<[SKXWriteResGroup206], (instregex "PCMPESTRIrr")>;
def: InstRW<[SKXWriteResGroup206], (instregex "VPCMPESTRIrr")>;

def SKXWriteResGroup207 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort06,SKXPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[SKXWriteResGroup207], (instregex "CPUID")>;
def: InstRW<[SKXWriteResGroup207], (instregex "RDTSC")>;

def SKXWriteResGroup208 : SchedWriteRes<[SKXPort1,SKXPort23,SKXPort237,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 18;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,4,1,2];
}
def: InstRW<[SKXWriteResGroup208], (instregex "RCR(16|32|64)mCL")>;
def: InstRW<[SKXWriteResGroup208], (instregex "RCR8mCL")>;

def SKXWriteResGroup209 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup209], (instregex "DIVSDrm")>;
def: InstRW<[SKXWriteResGroup209], (instregex "VDIVSDrm")>;
def: InstRW<[SKXWriteResGroup209], (instregex "VSQRTPSYm")>;
def: InstRW<[SKXWriteResGroup209], (instregex "VSQRTPSZ256m(b?)(k?)(z?)")>;

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
def: InstRW<[SKXWriteResGroup211], (instregex "VPMULLQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup211], (instregex "VPMULLQZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup212 : SchedWriteRes<[SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 19;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKXWriteResGroup212], (instregex "DPPSrmi")>;
def: InstRW<[SKXWriteResGroup212], (instregex "VDPPSrmi")>;

def SKXWriteResGroup213 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort015,SKXPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[SKXWriteResGroup213], (instregex "PCMPESTRM128rr")>;
def: InstRW<[SKXWriteResGroup213], (instregex "VPCMPESTRM128rr")>;

def SKXWriteResGroup214 : SchedWriteRes<[]> {
  let Latency = 20;
  let NumMicroOps = 0;
}
def: InstRW<[SKXWriteResGroup214], (instregex "VGATHERDPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup214], (instregex "VGATHERQPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup214], (instregex "VPGATHERDDZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup215 : SchedWriteRes<[SKXPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKXWriteResGroup215], (instregex "DIV_FPrST0")>;
def: InstRW<[SKXWriteResGroup215], (instregex "DIV_FST0r")>;
def: InstRW<[SKXWriteResGroup215], (instregex "DIV_FrST0")>;

def SKXWriteResGroup216 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup216], (instregex "DIVPDrm")>;
def: InstRW<[SKXWriteResGroup216], (instregex "VDIVPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup216], (instregex "VDIVPDrm")>;
def: InstRW<[SKXWriteResGroup216], (instregex "VDIVSDZrm_Int(b?)(k?)(z?)")>;

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
def: InstRW<[SKXWriteResGroup218], (instregex "VGATHERQPSZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup218], (instregex "VGATHERQPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup218], (instregex "VPGATHERQDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup218], (instregex "VPGATHERQDZ256rm(b?)(k?)(z?)")>;

def SKXWriteResGroup219 : SchedWriteRes<[SKXPort4,SKXPort5,SKXPort6,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 20;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,1,2];
}
def: InstRW<[SKXWriteResGroup219], (instregex "INSB")>;
def: InstRW<[SKXWriteResGroup219], (instregex "INSL")>;
def: InstRW<[SKXWriteResGroup219], (instregex "INSW")>;

def SKXWriteResGroup220 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort0156]> {
  let Latency = 20;
  let NumMicroOps = 10;
  let ResourceCycles = [1,2,7];
}
def: InstRW<[SKXWriteResGroup220], (instregex "MWAITrr")>;

def SKXWriteResGroup221 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort015]> {
  let Latency = 20;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,2];
}
def: InstRW<[SKXWriteResGroup221], (instregex "AESKEYGENASSIST128rr")>;
def: InstRW<[SKXWriteResGroup221], (instregex "VAESKEYGENASSIST128rr")>;

def SKXWriteResGroup222 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup222], (instregex "VDIVPDYrm")>;
def: InstRW<[SKXWriteResGroup222], (instregex "VDIVPDZ256rm(b?)(k?)(z?)")>;

def SKXWriteResGroup223 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 22;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup223], (instregex "DIV_F32m")>;
def: InstRW<[SKXWriteResGroup223], (instregex "DIV_F64m")>;

def SKXWriteResGroup224 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 22;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup224], (instregex "VGATHERDPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup224], (instregex "VGATHERQPDZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup224], (instregex "VPGATHERDQZ128rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup224], (instregex "VPGATHERQQZ128rm(b?)(k?)(z?)")>;

def SKXWriteResGroup224_2 : SchedWriteRes<[SKXPort0, SKXPort23, SKXPort5, SKXPort015]> {
  let Latency = 22;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup224_2], (instregex "VGATHERDPSrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VGATHERDPDrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VGATHERQPDrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VGATHERQPSrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VPGATHERDDrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VPGATHERDQrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VPGATHERQDrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VPGATHERQQrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VPGATHERDDrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VPGATHERQDrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VPGATHERDQrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VPGATHERQQrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VGATHERDPSrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VGATHERQPSrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VGATHERDPDrm")>;
def: InstRW<[SKXWriteResGroup224_2], (instregex "VGATHERQPDrm")>;

def SKXWriteResGroup224_3 : SchedWriteRes<[SKXPort0, SKXPort23, SKXPort5, SKXPort015]> {
  let Latency = 25;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup224_3], (instregex "VGATHERDPSYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VGATHERQPDYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VGATHERQPSYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VPGATHERDDYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VPGATHERDQYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VPGATHERQDYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VPGATHERQQYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VPGATHERDDYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VPGATHERQDYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VPGATHERDQYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VPGATHERQQYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VGATHERDPSYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VGATHERQPSYrm")>;
def: InstRW<[SKXWriteResGroup224_3], (instregex "VGATHERDPDYrm")>;

def SKXWriteResGroup225 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort015]> {
  let Latency = 22;
  let NumMicroOps = 14;
  let ResourceCycles = [5,5,4];
}
def: InstRW<[SKXWriteResGroup225], (instregex "VPCONFLICTDZ128rr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup225], (instregex "VPCONFLICTQZ256rr(b?)(k?)(z?)")>;

def SKXWriteResGroup226 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 23;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup226], (instregex "SQRTSDm")>;
def: InstRW<[SKXWriteResGroup226], (instregex "VSQRTSDm")>;

def SKXWriteResGroup227 : SchedWriteRes<[SKXPort0,SKXPort015]> {
  let Latency = 23;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKXWriteResGroup227], (instregex "VDIVPDZrr(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup227], (instregex "VDIVPSZrr(b?)(k?)(z?)")>;

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
def: InstRW<[SKXWriteResGroup229], (instregex "SQRTPDm")>;
def: InstRW<[SKXWriteResGroup229], (instregex "VSQRTPDZ128m(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup229], (instregex "VSQRTPDm")>;
def: InstRW<[SKXWriteResGroup229], (instregex "VSQRTSDZm_Int(b?)(k?)(z?)")>;

def SKXWriteResGroup230 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015]> {
  let Latency = 24;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKXWriteResGroup230], (instregex "VDIVPSZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup231 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23,SKXPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[SKXWriteResGroup231], (instregex "PCMPESTRIrm")>;
def: InstRW<[SKXWriteResGroup231], (instregex "VPCMPESTRIrm")>;

def SKXWriteResGroup232 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup232], (instregex "VSQRTPDYm")>;
def: InstRW<[SKXWriteResGroup232], (instregex "VSQRTPDZ256m(b?)(k?)(z?)")>;

def SKXWriteResGroup233 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23]> {
  let Latency = 25;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKXWriteResGroup233], (instregex "DIV_FI16m")>;
def: InstRW<[SKXWriteResGroup233], (instregex "DIV_FI32m")>;

def SKXWriteResGroup234 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 25;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup234], (instregex "VGATHERDPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup234], (instregex "VGATHERQPDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup234], (instregex "VPGATHERDQZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup234], (instregex "VPGATHERQDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup234], (instregex "VPGATHERQQZ256rm(b?)(k?)(z?)")>;

def SKXWriteResGroup235 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 25;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}
def: InstRW<[SKXWriteResGroup235], (instregex "PCMPESTRM128rm")>;
def: InstRW<[SKXWriteResGroup235], (instregex "VPCMPESTRM128rm")>;

def SKXWriteResGroup236 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23,SKXPort015]> {
  let Latency = 25;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,1,1];
}
def: InstRW<[SKXWriteResGroup236], (instregex "AESKEYGENASSIST128rm")>;
def: InstRW<[SKXWriteResGroup236], (instregex "VAESKEYGENASSIST128rm")>;

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
def: InstRW<[SKXWriteResGroup238], (instregex "VGATHERDPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup238], (instregex "VGATHERQPDZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup238], (instregex "VPGATHERDQZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup238], (instregex "VPGATHERQQZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup239 : SchedWriteRes<[SKXPort0,SKXPort23]> {
  let Latency = 27;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKXWriteResGroup239], (instregex "DIVR_F32m")>;
def: InstRW<[SKXWriteResGroup239], (instregex "DIVR_F64m")>;

def SKXWriteResGroup240 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort015,SKXPort0156]> {
  let Latency = 27;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKXWriteResGroup240], (instregex "VGATHERDPSZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup240], (instregex "VPGATHERDDZ256rm(b?)(k?)(z?)")>;

def SKXWriteResGroup241 : SchedWriteRes<[SKXPort0,SKXPort5,SKXPort23,SKXPort0156]> {
  let Latency = 28;
  let NumMicroOps = 8;
  let ResourceCycles = [2,4,1,1];
}
def: InstRW<[SKXWriteResGroup241], (instregex "IDIV(16|32|64)m")>;
def: InstRW<[SKXWriteResGroup241], (instregex "IDIV8m")>;

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
def: InstRW<[SKXWriteResGroup243], (instregex "DIVR_FI16m")>;
def: InstRW<[SKXWriteResGroup243], (instregex "DIVR_FI32m")>;

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
def: InstRW<[SKXWriteResGroup245], (instregex "VGATHERDPSZrm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup245], (instregex "VPGATHERDDZrm(b?)(k?)(z?)")>;

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
def: InstRW<[SKXWriteResGroup247], (instregex "IN32ri")>;
def: InstRW<[SKXWriteResGroup247], (instregex "IN32rr")>;
def: InstRW<[SKXWriteResGroup247], (instregex "IN8ri")>;
def: InstRW<[SKXWriteResGroup247], (instregex "IN8rr")>;

def SKXWriteResGroup248 : SchedWriteRes<[SKXPort5,SKXPort6,SKXPort23,SKXPort237,SKXPort06,SKXPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[SKXWriteResGroup248], (instregex "OUT32ir")>;
def: InstRW<[SKXWriteResGroup248], (instregex "OUT32rr")>;
def: InstRW<[SKXWriteResGroup248], (instregex "OUT8ir")>;
def: InstRW<[SKXWriteResGroup248], (instregex "OUT8rr")>;

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
def: InstRW<[SKXWriteResGroup250], (instregex "XRSTOR(64?)")>;

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
def: InstRW<[SKXWriteResGroup254], (instregex "RDTSCP")>;

def SKXWriteResGroup255 : SchedWriteRes<[SKXPort4,SKXPort6,SKXPort23,SKXPort237,SKXPort0156]> {
  let Latency = 42;
  let NumMicroOps = 40;
  let ResourceCycles = [1,11,1,1,26];
}
def: InstRW<[SKXWriteResGroup255], (instregex "XSAVE")>;

def SKXWriteResGroup256 : SchedWriteRes<[SKXPort5,SKXPort01,SKXPort23,SKXPort015]> {
  let Latency = 44;
  let NumMicroOps = 22;
  let ResourceCycles = [9,7,1,5];
}
def: InstRW<[SKXWriteResGroup256], (instregex "VPCONFLICTDZ256rm(b?)(k?)(z?)")>;
def: InstRW<[SKXWriteResGroup256], (instregex "VPCONFLICTQZrm(b?)(k?)(z?)")>;

def SKXWriteResGroup258 : SchedWriteRes<[SKXPort0,SKXPort23,SKXPort05,SKXPort06,SKXPort0156]> {
  let Latency = 62;
  let NumMicroOps = 64;
  let ResourceCycles = [2,8,5,10,39];
}
def: InstRW<[SKXWriteResGroup258], (instregex "FLDENVm")>;
def: InstRW<[SKXWriteResGroup258], (instregex "FLDENVm")>;

def SKXWriteResGroup259 : SchedWriteRes<[SKXPort0,SKXPort6,SKXPort23,SKXPort05,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 63;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[SKXWriteResGroup259], (instregex "FXRSTOR64")>;

def SKXWriteResGroup260 : SchedWriteRes<[SKXPort0,SKXPort6,SKXPort23,SKXPort05,SKXPort06,SKXPort15,SKXPort0156]> {
  let Latency = 63;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[SKXWriteResGroup260], (instregex "FXRSTOR")>;

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
def: InstRW<[SKXWriteResGroup266], (instregex "FSTENVm")>;

def SKXWriteResGroup267 : SchedWriteRes<[SKXPort6,SKXPort0156]> {
  let Latency = 140;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKXWriteResGroup267], (instregex "PAUSE")>;
} // SchedModel
