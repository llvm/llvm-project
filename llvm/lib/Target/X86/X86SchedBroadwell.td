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

// Loads are 5 cycles, so ReadAfterLd registers needn't be available until 5
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 5>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass BWWriteResPair<X86FoldableSchedWrite SchedRW,
                          ProcResourceKind ExePort,
                          int Lat> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, [ExePort]> { let Latency = Lat; }

  // Memory variant also uses a cycle on port 2/3 and adds 5 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, [BWPort23, ExePort]> {
     let Latency = !add(Lat, 5);
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [BWPort4]>;

// Arithmetic.
defm : BWWriteResPair<WriteALU,   BWPort0156, 1>; // Simple integer ALU op.
defm : BWWriteResPair<WriteIMul,  BWPort1,   3>; // Integer multiplication.
def : WriteRes<WriteIMulH, []> { let Latency = 3; } // Integer multiplication, high part.
def BWDivider : ProcResource<1>; // Integer division issued on port 0.     
def : WriteRes<WriteIDiv, [BWPort0, BWDivider]> { // Integer division.
  let Latency = 25;
  let ResourceCycles = [1, 10];
}
def : WriteRes<WriteIDivLd, [BWPort23, BWPort0, BWDivider]> {
  let Latency = 29;
  let ResourceCycles = [1, 1, 10];
}

def : WriteRes<WriteLEA, [BWPort15]>; // LEA instructions can't fold loads.

// Integer shifts and rotates.
defm : BWWriteResPair<WriteShift, BWPort06,  1>;

// Loads, stores, and moves, not folded with other operations.
def : WriteRes<WriteLoad,  [BWPort23]> { let Latency = 5; }
def : WriteRes<WriteStore, [BWPort237, BWPort4]>;
def : WriteRes<WriteMove,  [BWPort0156]>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
def : WriteRes<WriteZero,  []>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : BWWriteResPair<WriteJump,  BWPort06,   1>;

// Floating point. This covers both scalar and vector operations.
defm : BWWriteResPair<WriteFAdd,   BWPort1, 3>; // Floating point add/sub/compare.
defm : BWWriteResPair<WriteFMul,   BWPort0, 5>; // Floating point multiplication.
defm : BWWriteResPair<WriteFDiv,   BWPort0, 12>; // 10-14 cycles. // Floating point division.
defm : BWWriteResPair<WriteFSqrt,  BWPort0, 15>; // Floating point square root.
defm : BWWriteResPair<WriteFRcp,   BWPort0, 5>; // Floating point reciprocal estimate.
defm : BWWriteResPair<WriteFRsqrt, BWPort0, 5>; // Floating point reciprocal square root estimate.
defm : BWWriteResPair<WriteFMA,    BWPort01, 5>; // Fused Multiply Add.
defm : BWWriteResPair<WriteFShuffle,  BWPort5,  1>; // Floating point vector shuffles.
defm : BWWriteResPair<WriteFBlend,  BWPort015,  1>; // Floating point vector blends.
def : WriteRes<WriteFVarBlend, [BWPort5]> { // Fp vector variable blends.	       
  let Latency = 2;
  let ResourceCycles = [2];
} 
def : WriteRes<WriteFVarBlendLd, [BWPort5, BWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

// FMA Scheduling helper class.
// class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Vector integer operations.
defm : BWWriteResPair<WriteVecALU,   BWPort15,  1>; // Vector integer ALU op, no logicals.
defm : BWWriteResPair<WriteVecShift, BWPort0,  1>; // Vector integer shifts.
defm : BWWriteResPair<WriteVecIMul,  BWPort0,   5>; // Vector integer multiply.
defm : BWWriteResPair<WriteShuffle,  BWPort5,  1>; // Vector shuffles.
defm : BWWriteResPair<WriteBlend,  BWPort15,  1>; // Vector blends.

def : WriteRes<WriteVarBlend, [BWPort5]> { // Vector variable blends.
  let Latency = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteVarBlendLd, [BWPort5, BWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

def : WriteRes<WriteMPSAD, [BWPort0, BWPort5]> { // Vector MPSAD.     
  let Latency = 6;
  let ResourceCycles = [1, 2];
}
def : WriteRes<WriteMPSADLd, [BWPort23, BWPort0, BWPort5]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 2];
}

// Vector bitwise operations.
// These are often used on both floating point and integer vectors.
defm : BWWriteResPair<WriteVecLogic, BWPort015, 1>; // Vector and/or/xor.

// Conversion between integer and float.
defm : BWWriteResPair<WriteCvtF2I, BWPort1, 3>; // Float -> Integer.
defm : BWWriteResPair<WriteCvtI2F, BWPort1, 4>; // Integer -> Float.
defm : BWWriteResPair<WriteCvtF2F, BWPort1, 3>; // Float -> Float size conversion.

// Strings instructions.
// Packed Compare Implicit Length Strings, Return Mask
// String instructions.
def : WriteRes<WritePCmpIStrM, [BWPort0]> {
  let Latency = 10;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [BWPort0, BWPort23]> {
  let Latency = 10;
  let ResourceCycles = [3, 1];
} 
// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [BWPort0, BWPort16, BWPort5]> {
  let Latency = 10;
  let ResourceCycles = [3, 2, 4];
}
def : WriteRes<WritePCmpEStrMLd, [BWPort05, BWPort16, BWPort23]> {
  let Latency = 10;
  let ResourceCycles = [6, 2, 1];
} 
  // Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [BWPort0]> {
  let Latency = 11;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [BWPort0, BWPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 1];
}     
// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [BWPort05, BWPort16]> {
  let Latency = 11;
  let ResourceCycles = [6, 2];
}
def : WriteRes<WritePCmpEStrILd, [BWPort0, BWPort16, BWPort5, BWPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 2, 2, 1];
}

// AES instructions.
def : WriteRes<WriteAESDecEnc, [BWPort5]> { // Decryption, encryption.
  let Latency = 7;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [BWPort5, BWPort23]> {
  let Latency = 7;
  let ResourceCycles = [1, 1];
}
def : WriteRes<WriteAESIMC, [BWPort5]> { // InvMixColumn.
  let Latency = 14;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [BWPort5, BWPort23]> {
  let Latency = 14;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteAESKeyGen, [BWPort0, BWPort5]> { // Key Generation.
  let Latency = 10;
  let ResourceCycles = [2, 8];
}
def : WriteRes<WriteAESKeyGenLd, [BWPort0, BWPort5, BWPort23]> {
  let Latency = 10;
  let ResourceCycles = [2, 7, 1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [BWPort0, BWPort5]> {
  let Latency = 7;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteCLMulLd, [BWPort0, BWPort5, BWPort23]> {
  let Latency = 7;
  let ResourceCycles = [2, 1, 1];
}

// Catch-all for expensive system instructions.
def : WriteRes<WriteSystem,     [BWPort0156]> { let Latency = 100; } // def WriteSystem : SchedWrite;

// AVX2.
defm : BWWriteResPair<WriteFShuffle256,  BWPort5,  3>; // Fp 256-bit width vector shuffles.
defm : BWWriteResPair<WriteShuffle256,  BWPort5,  3>;  // 256-bit width vector shuffles.
def : WriteRes<WriteVarVecShift, [BWPort0, BWPort5]> { // Variable vector shifts.
  let Latency = 2;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteVarVecShiftLd, [BWPort0, BWPort5, BWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1, 1];
}

// Old microcoded instructions that nobody use.
def : WriteRes<WriteMicrocoded, [BWPort0156]> { let Latency = 100; } // def WriteMicrocoded : SchedWrite;

// Fence instructions.
def : WriteRes<WriteFence,  [BWPort23, BWPort4]>;

// Nop, not very useful expect it provides a model for nops!
def : WriteRes<WriteNop, []>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////
// HADD, HSUB PS/PD
// x,x / v,v,v.
def : WriteRes<WriteFHAdd, [BWPort1]> {
  let Latency = 3;
}

// x,m / v,v,m.
def : WriteRes<WriteFHAddLd, [BWPort1, BWPort23]> {
  let Latency = 7;
  let ResourceCycles = [1, 1];
}

// PHADD|PHSUB (S) W/D.
// v <- v,v.
def : WriteRes<WritePHAdd, [BWPort15]>;

// v <- v,m.
def : WriteRes<WritePHAddLd, [BWPort15, BWPort23]> {
  let Latency = 5;
  let ResourceCycles = [1, 1];
}

// Remaining instrs.

def BWWriteResGroup1 : SchedWriteRes<[BWPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup1], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_MOVD64grr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PMOVMSKBrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSLLDri")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSLLDrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSLLQri")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSLLQrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSLLWri")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSLLWrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSRADri")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSRADrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSRAWri")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSRAWrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSRLDri")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSRLDrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSRLQri")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSRLQrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSRLWri")>;
def: InstRW<[BWWriteResGroup1], (instregex "MMX_PSRLWrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MOVPDI2DIrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "MOVPQIto64rr")>;
def: InstRW<[BWWriteResGroup1], (instregex "PSLLDri")>;
def: InstRW<[BWWriteResGroup1], (instregex "PSLLQri")>;
def: InstRW<[BWWriteResGroup1], (instregex "PSLLWri")>;
def: InstRW<[BWWriteResGroup1], (instregex "PSRADri")>;
def: InstRW<[BWWriteResGroup1], (instregex "PSRAWri")>;
def: InstRW<[BWWriteResGroup1], (instregex "PSRLDri")>;
def: InstRW<[BWWriteResGroup1], (instregex "PSRLQri")>;
def: InstRW<[BWWriteResGroup1], (instregex "PSRLWri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VMOVPDI2DIrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "VMOVPQIto64rr")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSLLDYri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSLLDri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSLLQYri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSLLQri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSLLVQYrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSLLVQrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSLLWYri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSLLWri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRADYri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRADri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRAWYri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRAWri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRLDYri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRLDri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRLQYri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRLQri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRLVQYrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRLVQrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRLWYri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VPSRLWri")>;
def: InstRW<[BWWriteResGroup1], (instregex "VTESTPDYrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "VTESTPDrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "VTESTPSYrr")>;
def: InstRW<[BWWriteResGroup1], (instregex "VTESTPSrr")>;

def BWWriteResGroup2 : SchedWriteRes<[BWPort1]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup2], (instregex "COMP_FST0r")>;
def: InstRW<[BWWriteResGroup2], (instregex "COM_FST0r")>;
def: InstRW<[BWWriteResGroup2], (instregex "MMX_MASKMOVQ64")>;
def: InstRW<[BWWriteResGroup2], (instregex "MMX_MASKMOVQ64")>;
def: InstRW<[BWWriteResGroup2], (instregex "UCOM_FPr")>;
def: InstRW<[BWWriteResGroup2], (instregex "UCOM_Fr")>;
def: InstRW<[BWWriteResGroup2], (instregex "VMASKMOVDQU")>;

def BWWriteResGroup3 : SchedWriteRes<[BWPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup3], (instregex "ANDNPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "ANDNPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "ANDPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "ANDPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "INSERTPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_MOVD64rr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_MOVD64to64rr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_MOVQ2DQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_PALIGNR64irr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_PSHUFBrr64")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_PSHUFWri")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_PUNPCKHBWirr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_PUNPCKHDQirr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_PUNPCKHWDirr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_PUNPCKLBWirr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_PUNPCKLDQirr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MMX_PUNPCKLWDirr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOV64toPQIrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVAPDrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVAPSrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVDDUPrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVDI2PDIrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVHLPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVLHPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVSDrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVSHDUPrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVSLDUPrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVSSrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVUPDrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "MOVUPSrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "ORPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "ORPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PACKSSDWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PACKSSWBrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PACKUSDWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PACKUSWBrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PALIGNRrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "PBLENDWrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVSXBDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVSXBQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVSXBWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVSXDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVSXWDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVSXWQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVZXBDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVZXBQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVZXBWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVZXDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVZXWDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PMOVZXWQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PSHUFBrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PSHUFDri")>;
def: InstRW<[BWWriteResGroup3], (instregex "PSHUFHWri")>;
def: InstRW<[BWWriteResGroup3], (instregex "PSHUFLWri")>;
def: InstRW<[BWWriteResGroup3], (instregex "PSLLDQri")>;
def: InstRW<[BWWriteResGroup3], (instregex "PSRLDQri")>;
def: InstRW<[BWWriteResGroup3], (instregex "PUNPCKHBWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PUNPCKHDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PUNPCKHQDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PUNPCKHWDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PUNPCKLBWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PUNPCKLDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PUNPCKLQDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "PUNPCKLWDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "SHUFPDrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "SHUFPSrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "UNPCKHPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "UNPCKHPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "UNPCKLPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "UNPCKLPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VANDNPDYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VANDNPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VANDNPSYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VANDNPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VANDPDYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VANDPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VANDPSYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VANDPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VBROADCASTSSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VINSERTPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOV64toPQIrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVAPDYrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVAPDrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVAPSYrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVAPSrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVDDUPYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVDDUPrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVDI2PDIrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVHLPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVLHPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVSDrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVSHDUPYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVSHDUPrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVSLDUPYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVSLDUPrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVSSrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVUPDYrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVUPDrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVUPSYrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "VMOVUPSrr(_REV?)")>;
def: InstRW<[BWWriteResGroup3], (instregex "VORPDYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VORPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VORPSYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VORPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPACKSSDWYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPACKSSDWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPACKSSWBYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPACKSSWBrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPACKUSDWYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPACKUSDWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPACKUSWBYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPACKUSWBrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPALIGNRYrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPALIGNRrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPBLENDWYrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPBLENDWrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPBROADCASTDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPBROADCASTQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPERMILPDYri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPERMILPDYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPERMILPDri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPERMILPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPERMILPSYri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPERMILPSYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPERMILPSri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPERMILPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVSXBDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVSXBQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVSXBWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVSXDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVSXWDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVSXWQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVZXBDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVZXBQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVZXBWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVZXDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVZXWDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPMOVZXWQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSHUFBYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSHUFBrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSHUFDYri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSHUFDri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSHUFHWYri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSHUFHWri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSHUFLWYri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSHUFLWri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSLLDQYri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSLLDQri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSRLDQYri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPSRLDQri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKHBWYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKHBWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKHDQYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKHDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKHQDQYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKHQDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKHWDYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKHWDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKLBWYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKLBWrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKLDQYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKLDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKLQDQYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKLQDQrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKLWDYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VPUNPCKLWDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VSHUFPDYrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VSHUFPDrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VSHUFPSYrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VSHUFPSrri")>;
def: InstRW<[BWWriteResGroup3], (instregex "VUNPCKHPDYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VUNPCKHPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VUNPCKHPSYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VUNPCKHPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VUNPCKLPDYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VUNPCKLPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VUNPCKLPSYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VUNPCKLPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VXORPDYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VXORPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VXORPSYrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "VXORPSrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "XORPDrr")>;
def: InstRW<[BWWriteResGroup3], (instregex "XORPSrr")>;

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
def: InstRW<[BWWriteResGroup5], (instregex "FINCSTP")>;
def: InstRW<[BWWriteResGroup5], (instregex "FNOP")>;

def BWWriteResGroup6 : SchedWriteRes<[BWPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup6], (instregex "ADC(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup6], (instregex "ADC(16|32|64)rr(_REV?)")>;
def: InstRW<[BWWriteResGroup6], (instregex "ADC8rr(_REV?)")>;
def: InstRW<[BWWriteResGroup6], (instregex "ADCX32rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "ADCX64rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "ADOX32rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "ADOX64rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "BT(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup6], (instregex "BT(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "BTC(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup6], (instregex "BTC(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "BTR(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup6], (instregex "BTR(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "BTS(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup6], (instregex "BTS(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CDQ")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVAE(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVB(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVE(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVG(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVGE(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVL(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVLE(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVNE(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVNO(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVNP(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVNS(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVO(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVP(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CMOVS(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "CQO")>;
def: InstRW<[BWWriteResGroup6], (instregex "JAE_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JAE_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JA_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JA_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JBE_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JBE_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JB_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JB_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JE_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JE_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JGE_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JGE_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JG_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JG_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JLE_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JLE_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JL_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JL_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JMP_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JMP_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JNE_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JNE_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JNO_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JNO_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JNP_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JNP_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JNS_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JNS_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JO_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JO_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JP_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JP_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "JS_1")>;
def: InstRW<[BWWriteResGroup6], (instregex "JS_4")>;
def: InstRW<[BWWriteResGroup6], (instregex "RORX32ri")>;
def: InstRW<[BWWriteResGroup6], (instregex "RORX64ri")>;
def: InstRW<[BWWriteResGroup6], (instregex "SAR(16|32|64)r1")>;
def: InstRW<[BWWriteResGroup6], (instregex "SAR(16|32|64)ri")>;
def: InstRW<[BWWriteResGroup6], (instregex "SAR8r1")>;
def: InstRW<[BWWriteResGroup6], (instregex "SAR8ri")>;
def: InstRW<[BWWriteResGroup6], (instregex "SARX32rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SARX64rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SBB(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup6], (instregex "SBB(16|32|64)rr(_REV?)")>;
def: InstRW<[BWWriteResGroup6], (instregex "SBB8rr(_REV?)")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETAEr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETBr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETEr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETGEr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETGr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETLEr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETLr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETNEr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETNOr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETNPr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETNSr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETOr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETPr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SETSr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHL(16|32|64)r1")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHL(16|32|64)ri")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHL8r1")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHL8ri")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHLX32rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHLX64rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHR(16|32|64)r1")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHR(16|32|64)ri")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHR8r1")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHR8ri")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHRX32rr")>;
def: InstRW<[BWWriteResGroup6], (instregex "SHRX64rr")>;

def BWWriteResGroup7 : SchedWriteRes<[BWPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup7], (instregex "ANDN32rr")>;
def: InstRW<[BWWriteResGroup7], (instregex "ANDN64rr")>;
def: InstRW<[BWWriteResGroup7], (instregex "BLSI32rr")>;
def: InstRW<[BWWriteResGroup7], (instregex "BLSI64rr")>;
def: InstRW<[BWWriteResGroup7], (instregex "BLSMSK32rr")>;
def: InstRW<[BWWriteResGroup7], (instregex "BLSMSK64rr")>;
def: InstRW<[BWWriteResGroup7], (instregex "BLSR32rr")>;
def: InstRW<[BWWriteResGroup7], (instregex "BLSR64rr")>;
def: InstRW<[BWWriteResGroup7], (instregex "BZHI32rr")>;
def: InstRW<[BWWriteResGroup7], (instregex "BZHI64rr")>;
def: InstRW<[BWWriteResGroup7], (instregex "LEA(16|32|64)r")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PABSBrr64")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PABSDrr64")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PABSWrr64")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PADDBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PADDDirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PADDQirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PADDSBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PADDSWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PADDUSBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PADDUSWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PADDWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PAVGBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PAVGWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PCMPEQBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PCMPEQDirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PCMPEQWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PCMPGTBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PCMPGTDirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PCMPGTWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PMAXSWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PMAXUBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PMINSWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PMINUBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSIGNBrr64")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSIGNDrr64")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSIGNWrr64")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSUBBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSUBDirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSUBQirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSUBSBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSUBSWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSUBUSBirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSUBUSWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "MMX_PSUBWirr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PABSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PABSDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PABSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PADDBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PADDDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PADDQrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PADDSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PADDSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PADDUSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PADDUSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PADDWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PAVGBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PAVGWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PCMPEQBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PCMPEQDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PCMPEQQrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PCMPEQWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PCMPGTBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PCMPGTDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PCMPGTWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMAXSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMAXSDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMAXSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMAXUBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMAXUDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMAXUWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMINSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMINSDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMINSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMINUBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMINUDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PMINUWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSIGNBrr128")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSIGNDrr128")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSIGNWrr128")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSUBBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSUBDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSUBQrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSUBSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSUBSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSUBUSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSUBUSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "PSUBWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPABSBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPABSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPABSDYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPABSDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPABSWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPABSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDDYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDQYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDQrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDSBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDSWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDUSBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDUSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDUSWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDUSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPADDWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPAVGBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPAVGBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPAVGWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPAVGWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPEQBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPEQBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPEQDYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPEQDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPEQQYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPEQQrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPEQWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPEQWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPGTBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPGTBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPGTDYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPGTDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPGTWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPCMPGTWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXSBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXSDYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXSDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXSWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXUBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXUBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXUDYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXUDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXUWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMAXUWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINSBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINSDYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINSDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINSWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINUBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINUBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINUDYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINUDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINUWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPMINUWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSIGNBYrr256")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSIGNBrr128")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSIGNDYrr256")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSIGNDrr128")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSIGNWYrr256")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSIGNWrr128")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBDYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBDrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBQYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBQrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBSBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBSWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBUSBYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBUSBrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBUSWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBUSWrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBWYrr")>;
def: InstRW<[BWWriteResGroup7], (instregex "VPSUBWrr")>;

def BWWriteResGroup8 : SchedWriteRes<[BWPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup8], (instregex "BLENDPDrri")>;
def: InstRW<[BWWriteResGroup8], (instregex "BLENDPSrri")>;
def: InstRW<[BWWriteResGroup8], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[BWWriteResGroup8], (instregex "MMX_MOVQ64rr(_REV?)")>;
def: InstRW<[BWWriteResGroup8], (instregex "MMX_PANDNirr")>;
def: InstRW<[BWWriteResGroup8], (instregex "MMX_PANDirr")>;
def: InstRW<[BWWriteResGroup8], (instregex "MMX_PORirr")>;
def: InstRW<[BWWriteResGroup8], (instregex "MMX_PXORirr")>;
def: InstRW<[BWWriteResGroup8], (instregex "MOVDQArr(_REV?)")>;
def: InstRW<[BWWriteResGroup8], (instregex "MOVDQUrr(_REV?)")>;
def: InstRW<[BWWriteResGroup8], (instregex "MOVPQI2QIrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "PANDNrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "PANDrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "PORrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "PXORrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "VBLENDPDYrri")>;
def: InstRW<[BWWriteResGroup8], (instregex "VBLENDPDrri")>;
def: InstRW<[BWWriteResGroup8], (instregex "VBLENDPSYrri")>;
def: InstRW<[BWWriteResGroup8], (instregex "VBLENDPSrri")>;
def: InstRW<[BWWriteResGroup8], (instregex "VMOVDQAYrr(_REV?)")>;
def: InstRW<[BWWriteResGroup8], (instregex "VMOVDQArr(_REV?)")>;
def: InstRW<[BWWriteResGroup8], (instregex "VMOVDQUYrr(_REV?)")>;
def: InstRW<[BWWriteResGroup8], (instregex "VMOVDQUrr(_REV?)")>;
def: InstRW<[BWWriteResGroup8], (instregex "VMOVPQI2QIrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "VMOVZPQILo2PQIrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "VPANDNYrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "VPANDNrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "VPANDYrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "VPANDrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "VPBLENDDYrri")>;
def: InstRW<[BWWriteResGroup8], (instregex "VPBLENDDrri")>;
def: InstRW<[BWWriteResGroup8], (instregex "VPORYrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "VPORrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "VPXORYrr")>;
def: InstRW<[BWWriteResGroup8], (instregex "VPXORrr")>;

def BWWriteResGroup9 : SchedWriteRes<[BWPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup9], (instregex "ADD(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup9], (instregex "ADD(16|32|64)rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "ADD8i8")>;
def: InstRW<[BWWriteResGroup9], (instregex "ADD8ri")>;
def: InstRW<[BWWriteResGroup9], (instregex "ADD8rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "AND(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup9], (instregex "AND(16|32|64)rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "AND8i8")>;
def: InstRW<[BWWriteResGroup9], (instregex "AND8ri")>;
def: InstRW<[BWWriteResGroup9], (instregex "AND8rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "CBW")>;
def: InstRW<[BWWriteResGroup9], (instregex "CLC")>;
def: InstRW<[BWWriteResGroup9], (instregex "CMC")>;
def: InstRW<[BWWriteResGroup9], (instregex "CMP(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup9], (instregex "CMP(16|32|64)rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "CMP8i8")>;
def: InstRW<[BWWriteResGroup9], (instregex "CMP8ri")>;
def: InstRW<[BWWriteResGroup9], (instregex "CMP8rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "CWDE")>;
def: InstRW<[BWWriteResGroup9], (instregex "DEC(16|32|64)r")>;
def: InstRW<[BWWriteResGroup9], (instregex "DEC8r")>;
def: InstRW<[BWWriteResGroup9], (instregex "INC(16|32|64)r")>;
def: InstRW<[BWWriteResGroup9], (instregex "INC8r")>;
def: InstRW<[BWWriteResGroup9], (instregex "LAHF")>;
def: InstRW<[BWWriteResGroup9], (instregex "MOV(16|32|64)rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "MOV8ri")>;
def: InstRW<[BWWriteResGroup9], (instregex "MOV8ri_alt")>;
def: InstRW<[BWWriteResGroup9], (instregex "MOV8rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "MOVSX(16|32|64)rr16")>;
def: InstRW<[BWWriteResGroup9], (instregex "MOVSX(16|32|64)rr32")>;
def: InstRW<[BWWriteResGroup9], (instregex "MOVSX(16|32|64)rr8")>;
def: InstRW<[BWWriteResGroup9], (instregex "MOVZX(16|32|64)rr16")>;
def: InstRW<[BWWriteResGroup9], (instregex "MOVZX(16|32|64)rr8")>;
def: InstRW<[BWWriteResGroup9], (instregex "NEG(16|32|64)r")>;
def: InstRW<[BWWriteResGroup9], (instregex "NEG8r")>;
def: InstRW<[BWWriteResGroup9], (instregex "NOOP")>;
def: InstRW<[BWWriteResGroup9], (instregex "NOT(16|32|64)r")>;
def: InstRW<[BWWriteResGroup9], (instregex "NOT8r")>;
def: InstRW<[BWWriteResGroup9], (instregex "OR(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup9], (instregex "OR(16|32|64)rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "OR8i8")>;
def: InstRW<[BWWriteResGroup9], (instregex "OR8ri")>;
def: InstRW<[BWWriteResGroup9], (instregex "OR8rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "SAHF")>;
def: InstRW<[BWWriteResGroup9], (instregex "SGDT64m")>;
def: InstRW<[BWWriteResGroup9], (instregex "SIDT64m")>;
def: InstRW<[BWWriteResGroup9], (instregex "SLDT64m")>;
def: InstRW<[BWWriteResGroup9], (instregex "SMSW16m")>;
def: InstRW<[BWWriteResGroup9], (instregex "STC")>;
def: InstRW<[BWWriteResGroup9], (instregex "STRm")>;
def: InstRW<[BWWriteResGroup9], (instregex "SUB(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup9], (instregex "SUB(16|32|64)rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "SUB8i8")>;
def: InstRW<[BWWriteResGroup9], (instregex "SUB8ri")>;
def: InstRW<[BWWriteResGroup9], (instregex "SUB8rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "SYSCALL")>;
def: InstRW<[BWWriteResGroup9], (instregex "TEST(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup9], (instregex "TEST8i8")>;
def: InstRW<[BWWriteResGroup9], (instregex "TEST8ri")>;
def: InstRW<[BWWriteResGroup9], (instregex "TEST8rr")>;
def: InstRW<[BWWriteResGroup9], (instregex "XCHG(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup9], (instregex "XOR(16|32|64)ri8")>;
def: InstRW<[BWWriteResGroup9], (instregex "XOR(16|32|64)rr(_REV?)")>;
def: InstRW<[BWWriteResGroup9], (instregex "XOR8i8")>;
def: InstRW<[BWWriteResGroup9], (instregex "XOR8ri")>;
def: InstRW<[BWWriteResGroup9], (instregex "XOR8rr(_REV?)")>;

def BWWriteResGroup10 : SchedWriteRes<[BWPort4,BWPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup10], (instregex "FBSTPm")>;
def: InstRW<[BWWriteResGroup10], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[BWWriteResGroup10], (instregex "MMX_MOVD64mr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MMX_MOVNTQmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MMX_MOVQ64mr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOV(16|32|64)mr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOV8mi")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOV8mr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVAPDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVAPSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVDQAmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVDQUmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVHPDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVHPSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVLPDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVLPSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVNTDQmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVNTI_64mr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVNTImr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVNTPDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVNTPSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVPDI2DImr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVPQI2QImr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVPQIto64mr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVSSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVUPDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "MOVUPSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "ST_FP32m")>;
def: InstRW<[BWWriteResGroup10], (instregex "ST_FP64m")>;
def: InstRW<[BWWriteResGroup10], (instregex "ST_FP80m")>;
def: InstRW<[BWWriteResGroup10], (instregex "VEXTRACTF128mr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VEXTRACTI128mr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVAPDYmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVAPDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVAPSYmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVAPSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVDQAYmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVDQAmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVDQUYmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVDQUmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVHPDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVHPSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVLPDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVLPSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVNTDQYmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVNTDQmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVNTPDYmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVNTPDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVNTPSYmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVNTPSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVPDI2DImr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVPQI2QImr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVPQIto64mr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVSDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVSSmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVUPDYmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVUPDmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVUPSYmr")>;
def: InstRW<[BWWriteResGroup10], (instregex "VMOVUPSmr")>;

def BWWriteResGroup11 : SchedWriteRes<[BWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup11], (instregex "BLENDVPDrr0")>;
def: InstRW<[BWWriteResGroup11], (instregex "BLENDVPSrr0")>;
def: InstRW<[BWWriteResGroup11], (instregex "MMX_PINSRWirri")>;
def: InstRW<[BWWriteResGroup11], (instregex "PBLENDVBrr0")>;
def: InstRW<[BWWriteResGroup11], (instregex "PINSRBrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "PINSRDrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "PINSRQrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "PINSRWrri")>;
def: InstRW<[BWWriteResGroup11], (instregex "VBLENDVPDYrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "VBLENDVPDrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "VBLENDVPSYrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "VBLENDVPSrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "VPBLENDVBYrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "VPBLENDVBrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "VPINSRBrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "VPINSRDrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "VPINSRQrr")>;
def: InstRW<[BWWriteResGroup11], (instregex "VPINSRWrri")>;

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
def: InstRW<[BWWriteResGroup13], (instregex "ROL(16|32|64)r1")>;
def: InstRW<[BWWriteResGroup13], (instregex "ROL(16|32|64)ri")>;
def: InstRW<[BWWriteResGroup13], (instregex "ROL8r1")>;
def: InstRW<[BWWriteResGroup13], (instregex "ROL8ri")>;
def: InstRW<[BWWriteResGroup13], (instregex "ROR(16|32|64)r1")>;
def: InstRW<[BWWriteResGroup13], (instregex "ROR(16|32|64)ri")>;
def: InstRW<[BWWriteResGroup13], (instregex "ROR8r1")>;
def: InstRW<[BWWriteResGroup13], (instregex "ROR8ri")>;

def BWWriteResGroup14 : SchedWriteRes<[BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup14], (instregex "LFENCE")>;
def: InstRW<[BWWriteResGroup14], (instregex "MFENCE")>;
def: InstRW<[BWWriteResGroup14], (instregex "WAIT")>;
def: InstRW<[BWWriteResGroup14], (instregex "XGETBV")>;

def BWWriteResGroup15 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup15], (instregex "CVTPS2PDrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "CVTSS2SDrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "EXTRACTPSrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "MMX_PEXTRWirri")>;
def: InstRW<[BWWriteResGroup15], (instregex "PEXTRBrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PEXTRDrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PEXTRQrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PEXTRWri")>;
def: InstRW<[BWWriteResGroup15], (instregex "PEXTRWrr_REV")>;
def: InstRW<[BWWriteResGroup15], (instregex "PSLLDrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PSLLQrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PSLLWrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PSRADrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PSRAWrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PSRLDrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PSRLQrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PSRLWrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "PTESTrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VCVTPH2PSYrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VCVTPH2PSrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VCVTPS2PDrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VCVTSS2SDrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VEXTRACTPSrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPEXTRBrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPEXTRDrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPEXTRQrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPEXTRWri")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPEXTRWrr_REV")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPSLLDrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPSLLQrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPSLLWrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPSRADrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPSRAWrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPSRLDrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPSRLQrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPSRLWrr")>;
def: InstRW<[BWWriteResGroup15], (instregex "VPTESTrr")>;

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
def: InstRW<[BWWriteResGroup19], (instregex "BEXTR32rr")>;
def: InstRW<[BWWriteResGroup19], (instregex "BEXTR64rr")>;
def: InstRW<[BWWriteResGroup19], (instregex "BSWAP(16|32|64)r")>;

def BWWriteResGroup20 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup20], (instregex "ADC8i8")>;
def: InstRW<[BWWriteResGroup20], (instregex "ADC8ri")>;
def: InstRW<[BWWriteResGroup20], (instregex "CMOVA(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup20], (instregex "CMOVBE(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup20], (instregex "CWD")>;
def: InstRW<[BWWriteResGroup20], (instregex "JRCXZ")>;
def: InstRW<[BWWriteResGroup20], (instregex "SBB8i8")>;
def: InstRW<[BWWriteResGroup20], (instregex "SBB8ri")>;
def: InstRW<[BWWriteResGroup20], (instregex "SETAr")>;
def: InstRW<[BWWriteResGroup20], (instregex "SETBEr")>;

def BWWriteResGroup21 : SchedWriteRes<[BWPort4,BWPort5,BWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup21], (instregex "EXTRACTPSmr")>;
def: InstRW<[BWWriteResGroup21], (instregex "PEXTRBmr")>;
def: InstRW<[BWWriteResGroup21], (instregex "PEXTRDmr")>;
def: InstRW<[BWWriteResGroup21], (instregex "PEXTRQmr")>;
def: InstRW<[BWWriteResGroup21], (instregex "PEXTRWmr")>;
def: InstRW<[BWWriteResGroup21], (instregex "STMXCSR")>;
def: InstRW<[BWWriteResGroup21], (instregex "VEXTRACTPSmr")>;
def: InstRW<[BWWriteResGroup21], (instregex "VPEXTRBmr")>;
def: InstRW<[BWWriteResGroup21], (instregex "VPEXTRDmr")>;
def: InstRW<[BWWriteResGroup21], (instregex "VPEXTRQmr")>;
def: InstRW<[BWWriteResGroup21], (instregex "VPEXTRWmr")>;
def: InstRW<[BWWriteResGroup21], (instregex "VSTMXCSR")>;

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
def: InstRW<[BWWriteResGroup23], (instregex "SETAEm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETBm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETEm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETGEm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETGm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETLEm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETLm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETNEm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETNOm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETNPm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETNSm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETOm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETPm")>;
def: InstRW<[BWWriteResGroup23], (instregex "SETSm")>;

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
def: InstRW<[BWWriteResGroup25], (instregex "PUSH(16|32|64)r")>;
def: InstRW<[BWWriteResGroup25], (instregex "PUSH(16|32|64)rmr")>;
def: InstRW<[BWWriteResGroup25], (instregex "PUSH64i8")>;
def: InstRW<[BWWriteResGroup25], (instregex "STOSB")>;
def: InstRW<[BWWriteResGroup25], (instregex "STOSL")>;
def: InstRW<[BWWriteResGroup25], (instregex "STOSQ")>;
def: InstRW<[BWWriteResGroup25], (instregex "STOSW")>;

def BWWriteResGroup26 : SchedWriteRes<[BWPort0]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup26], (instregex "MOVMSKPDrr")>;
def: InstRW<[BWWriteResGroup26], (instregex "MOVMSKPSrr")>;
def: InstRW<[BWWriteResGroup26], (instregex "PMOVMSKBrr")>;
def: InstRW<[BWWriteResGroup26], (instregex "VMOVMSKPDYrr")>;
def: InstRW<[BWWriteResGroup26], (instregex "VMOVMSKPDrr")>;
def: InstRW<[BWWriteResGroup26], (instregex "VMOVMSKPSYrr")>;
def: InstRW<[BWWriteResGroup26], (instregex "VMOVMSKPSrr")>;
def: InstRW<[BWWriteResGroup26], (instregex "VPMOVMSKBYrr")>;
def: InstRW<[BWWriteResGroup26], (instregex "VPMOVMSKBrr")>;

def BWWriteResGroup27 : SchedWriteRes<[BWPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup27], (instregex "ADDPDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "ADDPSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "ADDSDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "ADDSSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "ADDSUBPDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "ADDSUBPSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "ADD_FPrST0")>;
def: InstRW<[BWWriteResGroup27], (instregex "ADD_FST0r")>;
def: InstRW<[BWWriteResGroup27], (instregex "ADD_FrST0")>;
def: InstRW<[BWWriteResGroup27], (instregex "BSF(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup27], (instregex "BSR(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup27], (instregex "CMPPDrri")>;
def: InstRW<[BWWriteResGroup27], (instregex "CMPPSrri")>;
def: InstRW<[BWWriteResGroup27], (instregex "CMPSSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "COMISDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "COMISSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "CVTDQ2PSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "CVTPS2DQrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "CVTTPS2DQrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "IMUL(32|64)rr(i8?)")>;
def: InstRW<[BWWriteResGroup27], (instregex "IMUL8r")>;
def: InstRW<[BWWriteResGroup27], (instregex "LZCNT(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup27], (instregex "MAXPDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "MAXPSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "MAXSDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "MAXSSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "MINPDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "MINPSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "MINSDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "MINSSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "MMX_CVTPI2PSirr")>;
def: InstRW<[BWWriteResGroup27], (instregex "MUL8r")>;
def: InstRW<[BWWriteResGroup27], (instregex "PDEP32rr")>;
def: InstRW<[BWWriteResGroup27], (instregex "PDEP64rr")>;
def: InstRW<[BWWriteResGroup27], (instregex "PEXT32rr")>;
def: InstRW<[BWWriteResGroup27], (instregex "PEXT64rr")>;
def: InstRW<[BWWriteResGroup27], (instregex "POPCNT(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup27], (instregex "SHLD(16|32|64)rri8")>;
def: InstRW<[BWWriteResGroup27], (instregex "SHRD(16|32|64)rri8")>;
def: InstRW<[BWWriteResGroup27], (instregex "SUBPDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "SUBPSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "SUBR_FPrST0")>;
def: InstRW<[BWWriteResGroup27], (instregex "SUBR_FST0r")>;
def: InstRW<[BWWriteResGroup27], (instregex "SUBR_FrST0")>;
def: InstRW<[BWWriteResGroup27], (instregex "SUBSDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "SUBSSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "SUB_FPrST0")>;
def: InstRW<[BWWriteResGroup27], (instregex "SUB_FST0r")>;
def: InstRW<[BWWriteResGroup27], (instregex "SUB_FrST0")>;
def: InstRW<[BWWriteResGroup27], (instregex "TZCNT(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup27], (instregex "UCOMISDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "UCOMISSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VADDPDYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VADDPDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VADDPSYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VADDPSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VADDSDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VADDSSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VADDSUBPDYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VADDSUBPDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VADDSUBPSYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VADDSUBPSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCMPPDYrri")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCMPPDrri")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCMPPSYrri")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCMPPSrri")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCMPSDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCMPSSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCOMISDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCOMISSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCVTDQ2PSYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCVTDQ2PSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCVTPS2DQYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCVTPS2DQrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCVTTPS2DQYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VCVTTPS2DQrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMAXPDYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMAXPDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMAXPSYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMAXPSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMAXSDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMAXSSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMINPDYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMINPDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMINPSYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMINPSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMINSDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VMINSSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VSUBPDYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VSUBPDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VSUBPSYrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VSUBPSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VSUBSDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VSUBSSrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VUCOMISDrr")>;
def: InstRW<[BWWriteResGroup27], (instregex "VUCOMISSrr")>;

def BWWriteResGroup27_16 : SchedWriteRes<[BWPort1, BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup27_16], (instregex "IMUL16rr(i8?)")>;

def BWWriteResGroup28 : SchedWriteRes<[BWPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup28], (instregex "VBROADCASTSDYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VBROADCASTSSYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VEXTRACTF128rr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VEXTRACTI128rr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VINSERTF128rr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VINSERTI128rr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPBROADCASTBYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPBROADCASTBrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPBROADCASTDYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPBROADCASTQYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPBROADCASTWYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPBROADCASTWrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPERM2F128rr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPERM2I128rr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPERMDYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPERMPDYri")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPERMPSYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPERMQYri")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVSXBDYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVSXBQYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVSXBWYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVSXDQYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVSXWDYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVSXWQYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVZXBDYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVZXBQYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVZXBWYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVZXDQYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVZXWDYrr")>;
def: InstRW<[BWWriteResGroup28], (instregex "VPMOVZXWQYrr")>;

def BWWriteResGroup29 : SchedWriteRes<[BWPort01]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup29], (instregex "MULPDrr")>;
def: InstRW<[BWWriteResGroup29], (instregex "MULPSrr")>;
def: InstRW<[BWWriteResGroup29], (instregex "MULSDrr")>;
def: InstRW<[BWWriteResGroup29], (instregex "MULSSrr")>;
def: InstRW<[BWWriteResGroup29], (instregex "VMULPDYrr")>;
def: InstRW<[BWWriteResGroup29], (instregex "VMULPDrr")>;
def: InstRW<[BWWriteResGroup29], (instregex "VMULPSYrr")>;
def: InstRW<[BWWriteResGroup29], (instregex "VMULPSrr")>;
def: InstRW<[BWWriteResGroup29], (instregex "VMULSDrr")>;
def: InstRW<[BWWriteResGroup29], (instregex "VMULSSrr")>;

def BWWriteResGroup30 : SchedWriteRes<[BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[BWWriteResGroup30], (instregex "XADD(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup30], (instregex "XADD8rr")>;
def: InstRW<[BWWriteResGroup30], (instregex "XCHG8rr")>;

def BWWriteResGroup31 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup31], (instregex "VPSLLVDYrr")>;
def: InstRW<[BWWriteResGroup31], (instregex "VPSLLVDrr")>;
def: InstRW<[BWWriteResGroup31], (instregex "VPSRAVDYrr")>;
def: InstRW<[BWWriteResGroup31], (instregex "VPSRAVDrr")>;
def: InstRW<[BWWriteResGroup31], (instregex "VPSRLVDYrr")>;
def: InstRW<[BWWriteResGroup31], (instregex "VPSRLVDrr")>;

def BWWriteResGroup32 : SchedWriteRes<[BWPort5,BWPort15]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup32], (instregex "MMX_PHADDSWrr64")>;
def: InstRW<[BWWriteResGroup32], (instregex "MMX_PHADDWrr64")>;
def: InstRW<[BWWriteResGroup32], (instregex "MMX_PHADDrr64")>;
def: InstRW<[BWWriteResGroup32], (instregex "MMX_PHSUBDrr64")>;
def: InstRW<[BWWriteResGroup32], (instregex "MMX_PHSUBSWrr64")>;
def: InstRW<[BWWriteResGroup32], (instregex "MMX_PHSUBWrr64")>;
def: InstRW<[BWWriteResGroup32], (instregex "PHADDDrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "PHADDSWrr128")>;
def: InstRW<[BWWriteResGroup32], (instregex "PHADDWrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "PHSUBDrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "PHSUBSWrr128")>;
def: InstRW<[BWWriteResGroup32], (instregex "PHSUBWrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHADDDYrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHADDDrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHADDSWrr128")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHADDSWrr256")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHADDWYrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHADDWrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHSUBDYrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHSUBDrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHSUBSWrr128")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHSUBSWrr256")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHSUBWYrr")>;
def: InstRW<[BWWriteResGroup32], (instregex "VPHSUBWrr")>;

def BWWriteResGroup33 : SchedWriteRes<[BWPort5,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup33], (instregex "MMX_PACKSSDWirr")>;
def: InstRW<[BWWriteResGroup33], (instregex "MMX_PACKSSWBirr")>;
def: InstRW<[BWWriteResGroup33], (instregex "MMX_PACKUSWBirr")>;

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
def: InstRW<[BWWriteResGroup35], (instregex "RCL(16|32|64)r1")>;
def: InstRW<[BWWriteResGroup35], (instregex "RCL(16|32|64)ri")>;
def: InstRW<[BWWriteResGroup35], (instregex "RCL8r1")>;
def: InstRW<[BWWriteResGroup35], (instregex "RCL8ri")>;
def: InstRW<[BWWriteResGroup35], (instregex "RCR(16|32|64)r1")>;
def: InstRW<[BWWriteResGroup35], (instregex "RCR(16|32|64)ri")>;
def: InstRW<[BWWriteResGroup35], (instregex "RCR8r1")>;
def: InstRW<[BWWriteResGroup35], (instregex "RCR8ri")>;

def BWWriteResGroup36 : SchedWriteRes<[BWPort06,BWPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup36], (instregex "ROL(16|32|64)rCL")>;
def: InstRW<[BWWriteResGroup36], (instregex "ROL8rCL")>;
def: InstRW<[BWWriteResGroup36], (instregex "ROR(16|32|64)rCL")>;
def: InstRW<[BWWriteResGroup36], (instregex "ROR8rCL")>;
def: InstRW<[BWWriteResGroup36], (instregex "SAR(16|32|64)rCL")>;
def: InstRW<[BWWriteResGroup36], (instregex "SAR8rCL")>;
def: InstRW<[BWWriteResGroup36], (instregex "SHL(16|32|64)rCL")>;
def: InstRW<[BWWriteResGroup36], (instregex "SHL8rCL")>;
def: InstRW<[BWWriteResGroup36], (instregex "SHR(16|32|64)rCL")>;
def: InstRW<[BWWriteResGroup36], (instregex "SHR8rCL")>;

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
def: InstRW<[BWWriteResGroup38], (instregex "CALL64pcrel32")>;
def: InstRW<[BWWriteResGroup38], (instregex "SETAm")>;
def: InstRW<[BWWriteResGroup38], (instregex "SETBEm")>;

def BWWriteResGroup39 : SchedWriteRes<[BWPort0,BWPort1]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup39], (instregex "CVTSD2SI64rr")>;
def: InstRW<[BWWriteResGroup39], (instregex "CVTSD2SIrr")>;
def: InstRW<[BWWriteResGroup39], (instregex "CVTSS2SI64rr")>;
def: InstRW<[BWWriteResGroup39], (instregex "CVTSS2SIrr")>;
def: InstRW<[BWWriteResGroup39], (instregex "CVTTSD2SI64rr")>;
def: InstRW<[BWWriteResGroup39], (instregex "CVTTSD2SIrr")>;
def: InstRW<[BWWriteResGroup39], (instregex "CVTTSS2SI64rr")>;
def: InstRW<[BWWriteResGroup39], (instregex "CVTTSS2SIrr")>;
def: InstRW<[BWWriteResGroup39], (instregex "VCVTSD2SI64rr")>;
def: InstRW<[BWWriteResGroup39], (instregex "VCVTSD2SIrr")>;
def: InstRW<[BWWriteResGroup39], (instregex "VCVTSS2SI64rr")>;
def: InstRW<[BWWriteResGroup39], (instregex "VCVTSS2SIrr")>;
def: InstRW<[BWWriteResGroup39], (instregex "VCVTTSD2SI64rr")>;
def: InstRW<[BWWriteResGroup39], (instregex "VCVTTSD2SIrr")>;
def: InstRW<[BWWriteResGroup39], (instregex "VCVTTSS2SI64rr")>;
def: InstRW<[BWWriteResGroup39], (instregex "VCVTTSS2SIrr")>;

def BWWriteResGroup40 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup40], (instregex "VCVTPS2PDYrr")>;
def: InstRW<[BWWriteResGroup40], (instregex "VPSLLDYrr")>;
def: InstRW<[BWWriteResGroup40], (instregex "VPSLLQYrr")>;
def: InstRW<[BWWriteResGroup40], (instregex "VPSLLWYrr")>;
def: InstRW<[BWWriteResGroup40], (instregex "VPSRADYrr")>;
def: InstRW<[BWWriteResGroup40], (instregex "VPSRAWYrr")>;
def: InstRW<[BWWriteResGroup40], (instregex "VPSRLDYrr")>;
def: InstRW<[BWWriteResGroup40], (instregex "VPSRLQYrr")>;
def: InstRW<[BWWriteResGroup40], (instregex "VPSRLWYrr")>;
def: InstRW<[BWWriteResGroup40], (instregex "VPTESTYrr")>;

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
def: InstRW<[BWWriteResGroup42], (instregex "CVTDQ2PDrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "CVTPD2DQrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "CVTPD2PSrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "CVTSD2SSrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "CVTSI2SD64rr")>;
def: InstRW<[BWWriteResGroup42], (instregex "CVTSI2SDrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "CVTSI2SSrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "CVTTPD2DQrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "IMUL(32|64)r")>;
def: InstRW<[BWWriteResGroup42], (instregex "MMX_CVTPD2PIirr")>;
def: InstRW<[BWWriteResGroup42], (instregex "MMX_CVTPI2PDirr")>;
def: InstRW<[BWWriteResGroup42], (instregex "MMX_CVTPS2PIirr")>;
def: InstRW<[BWWriteResGroup42], (instregex "MMX_CVTTPD2PIirr")>;
def: InstRW<[BWWriteResGroup42], (instregex "MMX_CVTTPS2PIirr")>;
def: InstRW<[BWWriteResGroup42], (instregex "MUL(32|64)r")>;
def: InstRW<[BWWriteResGroup42], (instregex "MULX64rr")>;
def: InstRW<[BWWriteResGroup42], (instregex "VCVTDQ2PDrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "VCVTPD2DQrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "VCVTPD2PSrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "VCVTPS2PHrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "VCVTSD2SSrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "VCVTSI2SD64rr")>;
def: InstRW<[BWWriteResGroup42], (instregex "VCVTSI2SDrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "VCVTSI2SSrr")>;
def: InstRW<[BWWriteResGroup42], (instregex "VCVTTPD2DQrr")>;

def BWWriteResGroup42_16 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def: InstRW<[BWWriteResGroup42_16], (instregex "IMUL16r")>;
def: InstRW<[BWWriteResGroup42_16], (instregex "MUL16r")>;

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
def: InstRW<[BWWriteResGroup44], (instregex "ISTT_FP16m")>;
def: InstRW<[BWWriteResGroup44], (instregex "ISTT_FP32m")>;
def: InstRW<[BWWriteResGroup44], (instregex "ISTT_FP64m")>;
def: InstRW<[BWWriteResGroup44], (instregex "IST_F16m")>;
def: InstRW<[BWWriteResGroup44], (instregex "IST_F32m")>;
def: InstRW<[BWWriteResGroup44], (instregex "IST_FP16m")>;
def: InstRW<[BWWriteResGroup44], (instregex "IST_FP32m")>;
def: InstRW<[BWWriteResGroup44], (instregex "IST_FP64m")>;
def: InstRW<[BWWriteResGroup44], (instregex "VCVTPS2PHYmr")>;
def: InstRW<[BWWriteResGroup44], (instregex "VCVTPS2PHmr")>;

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
def: InstRW<[BWWriteResGroup47], (instregex "MMX_PMADDUBSWrr64")>;
def: InstRW<[BWWriteResGroup47], (instregex "MMX_PMADDWDirr")>;
def: InstRW<[BWWriteResGroup47], (instregex "MMX_PMULHRSWrr64")>;
def: InstRW<[BWWriteResGroup47], (instregex "MMX_PMULHUWirr")>;
def: InstRW<[BWWriteResGroup47], (instregex "MMX_PMULHWirr")>;
def: InstRW<[BWWriteResGroup47], (instregex "MMX_PMULLWirr")>;
def: InstRW<[BWWriteResGroup47], (instregex "MMX_PMULUDQirr")>;
def: InstRW<[BWWriteResGroup47], (instregex "MMX_PSADBWirr")>;
def: InstRW<[BWWriteResGroup47], (instregex "MUL_FPrST0")>;
def: InstRW<[BWWriteResGroup47], (instregex "MUL_FST0r")>;
def: InstRW<[BWWriteResGroup47], (instregex "MUL_FrST0")>;
def: InstRW<[BWWriteResGroup47], (instregex "PCLMULQDQrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "PCMPGTQrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "PHMINPOSUWrr128")>;
def: InstRW<[BWWriteResGroup47], (instregex "PMADDUBSWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "PMADDWDrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "PMULDQrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "PMULHRSWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "PMULHUWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "PMULHWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "PMULLWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "PMULUDQrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "PSADBWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "RCPPSr")>;
def: InstRW<[BWWriteResGroup47], (instregex "RCPSSr")>;
def: InstRW<[BWWriteResGroup47], (instregex "RSQRTPSr")>;
def: InstRW<[BWWriteResGroup47], (instregex "RSQRTSSr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPCLMULQDQrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPCMPGTQYrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPCMPGTQrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPHMINPOSUWrr128")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMADDUBSWYrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMADDUBSWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMADDWDYrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMADDWDrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULDQYrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULDQrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULHRSWYrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULHRSWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULHUWYrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULHUWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULHWYrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULHWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULLWYrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULLWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULUDQYrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPMULUDQrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPSADBWYrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VPSADBWrr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VRCPPSr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VRCPSSr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VRSQRTPSr")>;
def: InstRW<[BWWriteResGroup47], (instregex "VRSQRTSSr")>;

def BWWriteResGroup48 : SchedWriteRes<[BWPort01]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD132PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD132PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD132PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD132PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD132SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD132SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD213PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD213PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD213PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD213PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD213SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD213SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD231PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD231PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD231PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD231PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD231SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADD231SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB132PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB132PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB132PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB132PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB213PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB213PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB213PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB213PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB231PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB231PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB231PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMADDSUB231PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB132PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB132PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB132PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB132PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB132SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB132SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB213PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB213PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB213PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB213PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB213SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB213SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB231PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB231PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB231PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB231PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB231SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUB231SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD132PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD132PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD132PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD132PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD213PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD213PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD213PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD213PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD231PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD231PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD231PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFMSUBADD231PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD132PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD132PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD132PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD132PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD132SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD132SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD213PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD213PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD213PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD213PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD213SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD213SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD231PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD231PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD231PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD231PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD231SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMADD231SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB132PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB132PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB132PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB132PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB132SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB132SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB213PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB213PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB213PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB213PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB213SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB213SSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB231PDYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB231PDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB231PSYr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB231PSr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB231SDr")>;
def: InstRW<[BWWriteResGroup48], (instregex "VFNMSUB231SSr")>;

def BWWriteResGroup49 : SchedWriteRes<[BWPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup49], (instregex "LDDQUrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MMX_MOVD64rm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MMX_MOVD64to64rm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MMX_MOVQ64rm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOV(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOV64toPQIrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOV8rm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVAPDrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVAPSrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVDDUPrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVDI2PDIrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVDQArm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVDQUrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVNTDQArm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVSHDUPrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVSLDUPrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVSSrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVSX(16|32|64)rm16")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVSX(16|32|64)rm32")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVSX(16|32|64)rm8")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVUPDrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVUPSrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVZX(16|32|64)rm16")>;
def: InstRW<[BWWriteResGroup49], (instregex "MOVZX(16|32|64)rm8")>;
def: InstRW<[BWWriteResGroup49], (instregex "PREFETCHNTA")>;
def: InstRW<[BWWriteResGroup49], (instregex "PREFETCHT0")>;
def: InstRW<[BWWriteResGroup49], (instregex "PREFETCHT1")>;
def: InstRW<[BWWriteResGroup49], (instregex "PREFETCHT2")>;
def: InstRW<[BWWriteResGroup49], (instregex "VBROADCASTSSrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VLDDQUrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOV64toPQIrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVAPDrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVAPSrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVDDUPrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVDI2PDIrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVDQArm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVDQUrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVNTDQArm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVQI2PQIrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVSDrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVSHDUPrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVSLDUPrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVSSrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVUPDrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VMOVUPSrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VPBROADCASTDrm")>;
def: InstRW<[BWWriteResGroup49], (instregex "VPBROADCASTQrm")>;

def BWWriteResGroup50 : SchedWriteRes<[BWPort1,BWPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup50], (instregex "CVTSI2SS64rr")>;
def: InstRW<[BWWriteResGroup50], (instregex "HADDPDrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "HADDPSrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "HSUBPDrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "HSUBPSrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "VCVTSI2SS64rr")>;
def: InstRW<[BWWriteResGroup50], (instregex "VHADDPDYrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "VHADDPDrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "VHADDPSYrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "VHADDPSrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "VHSUBPDYrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "VHSUBPDrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "VHSUBPSYrr")>;
def: InstRW<[BWWriteResGroup50], (instregex "VHSUBPSrr")>;

def BWWriteResGroup51 : SchedWriteRes<[BWPort1,BWPort6,BWPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup51], (instregex "STR(16|32|64)r")>;

def BWWriteResGroup52 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup52], (instregex "MULX32rr")>;

def BWWriteResGroup53 : SchedWriteRes<[BWPort0,BWPort4,BWPort237,BWPort15]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup53], (instregex "VMASKMOVPDYmr")>;
def: InstRW<[BWWriteResGroup53], (instregex "VMASKMOVPDmr")>;
def: InstRW<[BWWriteResGroup53], (instregex "VMASKMOVPSYmr")>;
def: InstRW<[BWWriteResGroup53], (instregex "VMASKMOVPSmr")>;
def: InstRW<[BWWriteResGroup53], (instregex "VPMASKMOVDYmr")>;
def: InstRW<[BWWriteResGroup53], (instregex "VPMASKMOVDmr")>;
def: InstRW<[BWWriteResGroup53], (instregex "VPMASKMOVQYmr")>;
def: InstRW<[BWWriteResGroup53], (instregex "VPMASKMOVQmr")>;

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
def: InstRW<[BWWriteResGroup56], (instregex "CMPXCHG(16|32|64)rr")>;
def: InstRW<[BWWriteResGroup56], (instregex "CMPXCHG8rr")>;

def BWWriteResGroup57 : SchedWriteRes<[BWPort4,BWPort237,BWPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[BWWriteResGroup57], (instregex "PUSHF16")>;
def: InstRW<[BWWriteResGroup57], (instregex "PUSHF64")>;

def BWWriteResGroup58 : SchedWriteRes<[BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup58], (instregex "LD_F32m")>;
def: InstRW<[BWWriteResGroup58], (instregex "LD_F64m")>;
def: InstRW<[BWWriteResGroup58], (instregex "LD_F80m")>;
def: InstRW<[BWWriteResGroup58], (instregex "VBROADCASTF128")>;
def: InstRW<[BWWriteResGroup58], (instregex "VBROADCASTI128")>;
def: InstRW<[BWWriteResGroup58], (instregex "VBROADCASTSDYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VBROADCASTSSYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VLDDQUYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VMOVAPDYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VMOVAPSYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VMOVDDUPYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VMOVDQAYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VMOVDQUYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VMOVNTDQAYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VMOVSHDUPYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VMOVSLDUPYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VMOVUPDYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VMOVUPSYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VPBROADCASTDYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "VPBROADCASTQYrm")>;
def: InstRW<[BWWriteResGroup58], (instregex "ROUNDPDr")>;
def: InstRW<[BWWriteResGroup58], (instregex "ROUNDPSr")>;
def: InstRW<[BWWriteResGroup58], (instregex "ROUNDSDr")>;
def: InstRW<[BWWriteResGroup58], (instregex "ROUNDSSr")>;
def: InstRW<[BWWriteResGroup58], (instregex "VROUNDPDr")>;
def: InstRW<[BWWriteResGroup58], (instregex "VROUNDPSr")>;
def: InstRW<[BWWriteResGroup58], (instregex "VROUNDSDr")>;
def: InstRW<[BWWriteResGroup58], (instregex "VROUNDSSr")>;
def: InstRW<[BWWriteResGroup58], (instregex "VROUNDYPDr")>;
def: InstRW<[BWWriteResGroup58], (instregex "VROUNDYPSr")>;

def BWWriteResGroup59 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup59], (instregex "CVTPS2PDrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "CVTSS2SDrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "MMX_PSLLDrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "MMX_PSLLQrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "MMX_PSLLWrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "MMX_PSRADrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "MMX_PSRAWrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "MMX_PSRLDrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "MMX_PSRLQrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "MMX_PSRLWrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "VCVTPH2PSYrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "VCVTPH2PSrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "VCVTPS2PDrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "VCVTSS2SDrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "VPSLLVQrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "VPSRLVQrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "VTESTPDrm")>;
def: InstRW<[BWWriteResGroup59], (instregex "VTESTPSrm")>;

def BWWriteResGroup60 : SchedWriteRes<[BWPort1,BWPort5]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup60], (instregex "VCVTDQ2PDYrr")>;
def: InstRW<[BWWriteResGroup60], (instregex "VCVTPD2DQYrr")>;
def: InstRW<[BWWriteResGroup60], (instregex "VCVTPD2PSYrr")>;
def: InstRW<[BWWriteResGroup60], (instregex "VCVTPS2PHYrr")>;
def: InstRW<[BWWriteResGroup60], (instregex "VCVTTPD2DQYrr")>;

def BWWriteResGroup61 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup61], (instregex "ANDNPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "ANDNPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "ANDPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "ANDPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "INSERTPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MMX_PALIGNR64irm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MMX_PINSRWirmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "MMX_PSHUFBrm64")>;
def: InstRW<[BWWriteResGroup61], (instregex "MMX_PSHUFWmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "MMX_PUNPCKHBWirm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MMX_PUNPCKHDQirm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MMX_PUNPCKHWDirm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MMX_PUNPCKLBWirm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MMX_PUNPCKLDQirm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MMX_PUNPCKLWDirm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MOVHPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MOVHPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MOVLPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "MOVLPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "ORPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "ORPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PACKSSDWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PACKSSWBrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PACKUSDWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PACKUSWBrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PALIGNRrmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "PBLENDWrmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "PINSRBrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PINSRDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PINSRQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PINSRWrmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVSXBDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVSXBQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVSXBWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVSXDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVSXWDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVSXWQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVZXBDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVZXBQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVZXBWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVZXDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVZXWDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PMOVZXWQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PSHUFBrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PSHUFDmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "PSHUFHWmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "PSHUFLWmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "PUNPCKHBWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PUNPCKHDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PUNPCKHQDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PUNPCKHWDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PUNPCKLBWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PUNPCKLDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PUNPCKLQDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "PUNPCKLWDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "SHUFPDrmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "SHUFPSrmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "UNPCKHPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "UNPCKHPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "UNPCKLPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "UNPCKLPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VANDNPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VANDNPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VANDPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VANDPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VINSERTPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VMOVHPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VMOVHPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VMOVLPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VMOVLPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VORPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VORPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPACKSSDWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPACKSSWBrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPACKUSDWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPACKUSWBrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPALIGNRrmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPBLENDWrmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPERMILPDmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPERMILPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPERMILPSmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPERMILPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPINSRBrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPINSRDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPINSRQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPINSRWrmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVSXBDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVSXBQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVSXBWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVSXDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVSXWDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVSXWQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVZXBDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVZXBQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVZXBWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVZXDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVZXWDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPMOVZXWQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPSHUFBrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPSHUFDmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPSHUFHWmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPSHUFLWmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPUNPCKHBWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPUNPCKHDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPUNPCKHQDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPUNPCKHWDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPUNPCKLBWrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPUNPCKLDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPUNPCKLQDQrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VPUNPCKLWDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VSHUFPDrmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "VSHUFPSrmi")>;
def: InstRW<[BWWriteResGroup61], (instregex "VUNPCKHPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VUNPCKHPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VUNPCKLPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VUNPCKLPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VXORPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "VXORPSrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "XORPDrm")>;
def: InstRW<[BWWriteResGroup61], (instregex "XORPSrm")>;

def BWWriteResGroup62 : SchedWriteRes<[BWPort6,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup62], (instregex "FARJMP64")>;
def: InstRW<[BWWriteResGroup62], (instregex "JMP(16|32|64)m")>;

def BWWriteResGroup63 : SchedWriteRes<[BWPort23,BWPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup63], (instregex "ADC(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "ADC8rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "ADCX32rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "ADCX64rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "ADOX32rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "ADOX64rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "BT(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVAE(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVB(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVE(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVG(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVGE(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVL(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVLE(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVNE(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVNO(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVNP(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVNS(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVO(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVP(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "CMOVS(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "RORX32mi")>;
def: InstRW<[BWWriteResGroup63], (instregex "RORX64mi")>;
def: InstRW<[BWWriteResGroup63], (instregex "SARX32rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "SARX64rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "SBB(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "SBB8rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "SHLX32rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "SHLX64rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "SHRX32rm")>;
def: InstRW<[BWWriteResGroup63], (instregex "SHRX64rm")>;

def BWWriteResGroup64 : SchedWriteRes<[BWPort23,BWPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup64], (instregex "ANDN32rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "ANDN64rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "BLSI32rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "BLSI64rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "BLSMSK32rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "BLSMSK64rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "BLSR32rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "BLSR64rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "BZHI32rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "BZHI64rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PABSBrm64")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PABSDrm64")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PABSWrm64")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PADDBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PADDDirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PADDQirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PADDSBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PADDSWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PADDUSBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PADDUSWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PADDWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PAVGBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PAVGWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PCMPEQBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PCMPEQDirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PCMPEQWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PCMPGTBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PCMPGTDirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PCMPGTWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PMAXSWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PMAXUBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PMINSWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PMINUBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSIGNBrm64")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSIGNDrm64")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSIGNWrm64")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSUBBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSUBDirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSUBQirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSUBSBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSUBSWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSUBUSBirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSUBUSWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MMX_PSUBWirm")>;
def: InstRW<[BWWriteResGroup64], (instregex "MOVBE(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PABSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PABSDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PABSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PADDBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PADDDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PADDQrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PADDSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PADDSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PADDUSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PADDUSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PADDWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PAVGBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PAVGWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PCMPEQBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PCMPEQDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PCMPEQQrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PCMPEQWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PCMPGTBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PCMPGTDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PCMPGTWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMAXSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMAXSDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMAXSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMAXUBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMAXUDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMAXUWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMINSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMINSDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMINSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMINUBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMINUDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PMINUWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSIGNBrm128")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSIGNDrm128")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSIGNWrm128")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSUBBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSUBDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSUBQrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSUBSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSUBSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSUBUSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSUBUSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "PSUBWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPABSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPABSDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPABSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPADDBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPADDDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPADDQrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPADDSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPADDSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPADDUSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPADDUSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPADDWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPAVGBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPAVGWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPCMPEQBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPCMPEQDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPCMPEQQrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPCMPEQWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPCMPGTBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPCMPGTDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPCMPGTWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMAXSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMAXSDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMAXSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMAXUBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMAXUDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMAXUWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMINSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMINSDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMINSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMINUBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMINUDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPMINUWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSIGNBrm128")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSIGNDrm128")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSIGNWrm128")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSUBBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSUBDrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSUBQrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSUBSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSUBSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSUBUSBrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSUBUSWrm")>;
def: InstRW<[BWWriteResGroup64], (instregex "VPSUBWrm")>;

def BWWriteResGroup65 : SchedWriteRes<[BWPort23,BWPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup65], (instregex "BLENDPDrmi")>;
def: InstRW<[BWWriteResGroup65], (instregex "BLENDPSrmi")>;
def: InstRW<[BWWriteResGroup65], (instregex "MMX_PANDNirm")>;
def: InstRW<[BWWriteResGroup65], (instregex "MMX_PANDirm")>;
def: InstRW<[BWWriteResGroup65], (instregex "MMX_PORirm")>;
def: InstRW<[BWWriteResGroup65], (instregex "MMX_PXORirm")>;
def: InstRW<[BWWriteResGroup65], (instregex "PANDNrm")>;
def: InstRW<[BWWriteResGroup65], (instregex "PANDrm")>;
def: InstRW<[BWWriteResGroup65], (instregex "PORrm")>;
def: InstRW<[BWWriteResGroup65], (instregex "PXORrm")>;
def: InstRW<[BWWriteResGroup65], (instregex "VBLENDPDrmi")>;
def: InstRW<[BWWriteResGroup65], (instregex "VBLENDPSrmi")>;
def: InstRW<[BWWriteResGroup65], (instregex "VINSERTF128rm")>;
def: InstRW<[BWWriteResGroup65], (instregex "VINSERTI128rm")>;
def: InstRW<[BWWriteResGroup65], (instregex "VPANDNrm")>;
def: InstRW<[BWWriteResGroup65], (instregex "VPANDrm")>;
def: InstRW<[BWWriteResGroup65], (instregex "VPBLENDDrmi")>;
def: InstRW<[BWWriteResGroup65], (instregex "VPORrm")>;
def: InstRW<[BWWriteResGroup65], (instregex "VPXORrm")>;

def BWWriteResGroup66 : SchedWriteRes<[BWPort23,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup66], (instregex "ADD(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "ADD8rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "AND(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "AND8rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "CMP(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup66], (instregex "CMP(16|32|64)mr")>;
def: InstRW<[BWWriteResGroup66], (instregex "CMP(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "CMP8mi")>;
def: InstRW<[BWWriteResGroup66], (instregex "CMP8mr")>;
def: InstRW<[BWWriteResGroup66], (instregex "CMP8rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "OR(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "OR8rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "POP(16|32|64)r")>;
def: InstRW<[BWWriteResGroup66], (instregex "POP(16|32|64)rmr")>;
def: InstRW<[BWWriteResGroup66], (instregex "SUB(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "SUB8rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "TEST(16|32|64)mr")>;
def: InstRW<[BWWriteResGroup66], (instregex "TEST8mi")>;
def: InstRW<[BWWriteResGroup66], (instregex "TEST8mr")>;
def: InstRW<[BWWriteResGroup66], (instregex "XOR(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup66], (instregex "XOR8rm")>;

def BWWriteResGroup67 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[BWWriteResGroup67], (instregex "SHLD(16|32|64)rrCL")>;
def: InstRW<[BWWriteResGroup67], (instregex "SHRD(16|32|64)rrCL")>;

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
def: InstRW<[BWWriteResGroup69], (instregex "BTC(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup69], (instregex "BTR(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup69], (instregex "BTS(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup69], (instregex "SAR(16|32|64)m1")>;
def: InstRW<[BWWriteResGroup69], (instregex "SAR(16|32|64)mi")>;
def: InstRW<[BWWriteResGroup69], (instregex "SAR8m1")>;
def: InstRW<[BWWriteResGroup69], (instregex "SAR8mi")>;
def: InstRW<[BWWriteResGroup69], (instregex "SHL(16|32|64)m1")>;
def: InstRW<[BWWriteResGroup69], (instregex "SHL(16|32|64)mi")>;
def: InstRW<[BWWriteResGroup69], (instregex "SHL8m1")>;
def: InstRW<[BWWriteResGroup69], (instregex "SHL8mi")>;
def: InstRW<[BWWriteResGroup69], (instregex "SHR(16|32|64)m1")>;
def: InstRW<[BWWriteResGroup69], (instregex "SHR(16|32|64)mi")>;
def: InstRW<[BWWriteResGroup69], (instregex "SHR8m1")>;
def: InstRW<[BWWriteResGroup69], (instregex "SHR8mi")>;

def BWWriteResGroup70 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup70], (instregex "ADD(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup70], (instregex "ADD(16|32|64)mr")>;
def: InstRW<[BWWriteResGroup70], (instregex "ADD8mi")>;
def: InstRW<[BWWriteResGroup70], (instregex "ADD8mr")>;
def: InstRW<[BWWriteResGroup70], (instregex "AND(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup70], (instregex "AND(16|32|64)mr")>;
def: InstRW<[BWWriteResGroup70], (instregex "AND8mi")>;
def: InstRW<[BWWriteResGroup70], (instregex "AND8mr")>;
def: InstRW<[BWWriteResGroup70], (instregex "DEC(16|32|64)m")>;
def: InstRW<[BWWriteResGroup70], (instregex "DEC8m")>;
def: InstRW<[BWWriteResGroup70], (instregex "INC(16|32|64)m")>;
def: InstRW<[BWWriteResGroup70], (instregex "INC8m")>;
def: InstRW<[BWWriteResGroup70], (instregex "NEG(16|32|64)m")>;
def: InstRW<[BWWriteResGroup70], (instregex "NEG8m")>;
def: InstRW<[BWWriteResGroup70], (instregex "NOT(16|32|64)m")>;
def: InstRW<[BWWriteResGroup70], (instregex "NOT8m")>;
def: InstRW<[BWWriteResGroup70], (instregex "OR(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup70], (instregex "OR(16|32|64)mr")>;
def: InstRW<[BWWriteResGroup70], (instregex "OR8mi")>;
def: InstRW<[BWWriteResGroup70], (instregex "OR8mr")>;
def: InstRW<[BWWriteResGroup70], (instregex "POP(16|32|64)rmm")>;
def: InstRW<[BWWriteResGroup70], (instregex "PUSH(16|32|64)rmm")>;
def: InstRW<[BWWriteResGroup70], (instregex "SUB(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup70], (instregex "SUB(16|32|64)mr")>;
def: InstRW<[BWWriteResGroup70], (instregex "SUB8mi")>;
def: InstRW<[BWWriteResGroup70], (instregex "SUB8mr")>;
def: InstRW<[BWWriteResGroup70], (instregex "XOR(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup70], (instregex "XOR(16|32|64)mr")>;
def: InstRW<[BWWriteResGroup70], (instregex "XOR8mi")>;
def: InstRW<[BWWriteResGroup70], (instregex "XOR8mr")>;

def BWWriteResGroup71 : SchedWriteRes<[BWPort6,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,5];
}
def: InstRW<[BWWriteResGroup71], (instregex "STD")>;

def BWWriteResGroup72 : SchedWriteRes<[BWPort5]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup72], (instregex "AESDECLASTrr")>;
def: InstRW<[BWWriteResGroup72], (instregex "AESDECrr")>;
def: InstRW<[BWWriteResGroup72], (instregex "AESENCLASTrr")>;
def: InstRW<[BWWriteResGroup72], (instregex "AESENCrr")>;
def: InstRW<[BWWriteResGroup72], (instregex "VAESDECLASTrr")>;
def: InstRW<[BWWriteResGroup72], (instregex "VAESDECrr")>;
def: InstRW<[BWWriteResGroup72], (instregex "VAESENCLASTrr")>;
def: InstRW<[BWWriteResGroup72], (instregex "VAESENCrr")>;

def BWWriteResGroup73 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup73], (instregex "VPSLLDYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VPSLLQYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VPSLLVQYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VPSLLWYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VPSRADYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VPSRAWYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VPSRLDYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VPSRLQYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VPSRLVQYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VPSRLWYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VTESTPDYrm")>;
def: InstRW<[BWWriteResGroup73], (instregex "VTESTPSYrm")>;

def BWWriteResGroup74 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup74], (instregex "FCOM32m")>;
def: InstRW<[BWWriteResGroup74], (instregex "FCOM64m")>;
def: InstRW<[BWWriteResGroup74], (instregex "FCOMP32m")>;
def: InstRW<[BWWriteResGroup74], (instregex "FCOMP64m")>;

def BWWriteResGroup75 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup75], (instregex "VANDNPDYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VANDNPSYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VANDPDYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VANDPSYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VORPDYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VORPSYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPACKSSDWYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPACKSSWBYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPACKUSDWYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPACKUSWBYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPALIGNRYrmi")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPBLENDWYrmi")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPERMILPDYmi")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPERMILPDYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPERMILPSYmi")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPERMILPSYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPSHUFBYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPSHUFDYmi")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPSHUFHWYmi")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPSHUFLWYmi")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPUNPCKHBWYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPUNPCKHDQYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPUNPCKHQDQYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPUNPCKHWDYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPUNPCKLBWYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPUNPCKLDQYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPUNPCKLQDQYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VPUNPCKLWDYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VSHUFPDYrmi")>;
def: InstRW<[BWWriteResGroup75], (instregex "VSHUFPSYrmi")>;
def: InstRW<[BWWriteResGroup75], (instregex "VUNPCKHPDYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VUNPCKHPSYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VUNPCKLPDYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VUNPCKLPSYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VXORPDYrm")>;
def: InstRW<[BWWriteResGroup75], (instregex "VXORPSYrm")>;

def BWWriteResGroup76 : SchedWriteRes<[BWPort23,BWPort15]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup76], (instregex "VPABSBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPABSDYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPABSWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPADDBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPADDDYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPADDQYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPADDSBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPADDSWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPADDUSBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPADDUSWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPADDWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPAVGBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPAVGWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPCMPEQBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPCMPEQDYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPCMPEQQYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPCMPEQWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPCMPGTBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPCMPGTDYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPCMPGTWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMAXSBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMAXSDYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMAXSWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMAXUBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMAXUDYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMAXUWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMINSBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMINSDYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMINSWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMINUBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMINUDYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPMINUWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSIGNBYrm256")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSIGNDYrm256")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSIGNWYrm256")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSUBBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSUBDYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSUBQYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSUBSBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSUBSWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSUBUSBYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSUBUSWYrm")>;
def: InstRW<[BWWriteResGroup76], (instregex "VPSUBWYrm")>;

def BWWriteResGroup77 : SchedWriteRes<[BWPort23,BWPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup77], (instregex "VBLENDPDYrmi")>;
def: InstRW<[BWWriteResGroup77], (instregex "VBLENDPSYrmi")>;
def: InstRW<[BWWriteResGroup77], (instregex "VPANDNYrm")>;
def: InstRW<[BWWriteResGroup77], (instregex "VPANDYrm")>;
def: InstRW<[BWWriteResGroup77], (instregex "VPBLENDDYrmi")>;
def: InstRW<[BWWriteResGroup77], (instregex "VPORYrm")>;
def: InstRW<[BWWriteResGroup77], (instregex "VPXORYrm")>;

def BWWriteResGroup78 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup78], (instregex "MPSADBWrri")>;
def: InstRW<[BWWriteResGroup78], (instregex "VMPSADBWYrri")>;
def: InstRW<[BWWriteResGroup78], (instregex "VMPSADBWrri")>;

def BWWriteResGroup79 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup79], (instregex "BLENDVPDrm0")>;
def: InstRW<[BWWriteResGroup79], (instregex "BLENDVPSrm0")>;
def: InstRW<[BWWriteResGroup79], (instregex "MMX_PACKSSDWirm")>;
def: InstRW<[BWWriteResGroup79], (instregex "MMX_PACKSSWBirm")>;
def: InstRW<[BWWriteResGroup79], (instregex "MMX_PACKUSWBirm")>;
def: InstRW<[BWWriteResGroup79], (instregex "PBLENDVBrm0")>;
def: InstRW<[BWWriteResGroup79], (instregex "VBLENDVPDrm")>;
def: InstRW<[BWWriteResGroup79], (instregex "VBLENDVPSrm")>;
def: InstRW<[BWWriteResGroup79], (instregex "VMASKMOVPDrm")>;
def: InstRW<[BWWriteResGroup79], (instregex "VMASKMOVPSrm")>;
def: InstRW<[BWWriteResGroup79], (instregex "VPBLENDVBrm")>;
def: InstRW<[BWWriteResGroup79], (instregex "VPMASKMOVDrm")>;
def: InstRW<[BWWriteResGroup79], (instregex "VPMASKMOVQrm")>;

def BWWriteResGroup80 : SchedWriteRes<[BWPort23,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup80], (instregex "LEAVE64")>;
def: InstRW<[BWWriteResGroup80], (instregex "SCASB")>;
def: InstRW<[BWWriteResGroup80], (instregex "SCASL")>;
def: InstRW<[BWWriteResGroup80], (instregex "SCASQ")>;
def: InstRW<[BWWriteResGroup80], (instregex "SCASW")>;

def BWWriteResGroup81 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup81], (instregex "PSLLDrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "PSLLQrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "PSLLWrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "PSRADrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "PSRAWrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "PSRLDrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "PSRLQrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "PSRLWrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "PTESTrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "VPSLLDrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "VPSLLQrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "VPSLLWrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "VPSRADrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "VPSRAWrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "VPSRLDrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "VPSRLQrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "VPSRLWrm")>;
def: InstRW<[BWWriteResGroup81], (instregex "VPTESTrm")>;

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
def: InstRW<[BWWriteResGroup83], (instregex "LDMXCSR")>;
def: InstRW<[BWWriteResGroup83], (instregex "VLDMXCSR")>;

def BWWriteResGroup84 : SchedWriteRes<[BWPort6,BWPort23,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup84], (instregex "LRETQ")>;
def: InstRW<[BWWriteResGroup84], (instregex "RETQ")>;

def BWWriteResGroup85 : SchedWriteRes<[BWPort23,BWPort06,BWPort15]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup85], (instregex "BEXTR32rm")>;
def: InstRW<[BWWriteResGroup85], (instregex "BEXTR64rm")>;

def BWWriteResGroup86 : SchedWriteRes<[BWPort23,BWPort06,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup86], (instregex "CMOVA(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup86], (instregex "CMOVBE(16|32|64)rm")>;

def BWWriteResGroup87 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[BWWriteResGroup87], (instregex "ROL(16|32|64)m1")>;
def: InstRW<[BWWriteResGroup87], (instregex "ROL(16|32|64)mi")>;
def: InstRW<[BWWriteResGroup87], (instregex "ROL8m1")>;
def: InstRW<[BWWriteResGroup87], (instregex "ROL8mi")>;
def: InstRW<[BWWriteResGroup87], (instregex "ROR(16|32|64)m1")>;
def: InstRW<[BWWriteResGroup87], (instregex "ROR(16|32|64)mi")>;
def: InstRW<[BWWriteResGroup87], (instregex "ROR8m1")>;
def: InstRW<[BWWriteResGroup87], (instregex "ROR8mi")>;

def BWWriteResGroup88 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[BWWriteResGroup88], (instregex "XADD(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup88], (instregex "XADD8rm")>;

def BWWriteResGroup89 : SchedWriteRes<[BWPort4,BWPort6,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[BWWriteResGroup89], (instregex "CALL(16|32|64)m")>;
def: InstRW<[BWWriteResGroup89], (instregex "FARCALL64")>;

def BWWriteResGroup90 : SchedWriteRes<[BWPort6,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [2,2,1,2];
}
def: InstRW<[BWWriteResGroup90], (instregex "LOOP")>;

def BWWriteResGroup91 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup91], (instregex "ADDPDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "ADDPSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "ADDSDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "ADDSSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "ADDSUBPDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "ADDSUBPSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "BSF(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup91], (instregex "BSR(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup91], (instregex "CMPPDrmi")>;
def: InstRW<[BWWriteResGroup91], (instregex "CMPPSrmi")>;
def: InstRW<[BWWriteResGroup91], (instregex "CMPSSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "COMISDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "COMISSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "CVTDQ2PSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "CVTPS2DQrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "CVTTPS2DQrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "IMUL64m")>;
def: InstRW<[BWWriteResGroup91], (instregex "IMUL(32|64)rm(i8?)")>;
def: InstRW<[BWWriteResGroup91], (instregex "IMUL8m")>;
def: InstRW<[BWWriteResGroup91], (instregex "LZCNT(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MAXPDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MAXPSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MAXSDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MAXSSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MINPDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MINPSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MINSDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MINSSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MMX_CVTPI2PSirm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MMX_CVTPS2PIirm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MMX_CVTTPS2PIirm")>;
def: InstRW<[BWWriteResGroup91], (instregex "MUL64m")>;
def: InstRW<[BWWriteResGroup91], (instregex "MUL8m")>;
def: InstRW<[BWWriteResGroup91], (instregex "PDEP32rm")>;
def: InstRW<[BWWriteResGroup91], (instregex "PDEP64rm")>;
def: InstRW<[BWWriteResGroup91], (instregex "PEXT32rm")>;
def: InstRW<[BWWriteResGroup91], (instregex "PEXT64rm")>;
def: InstRW<[BWWriteResGroup91], (instregex "POPCNT(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup91], (instregex "SUBPDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "SUBPSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "SUBSDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "SUBSSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "TZCNT(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup91], (instregex "UCOMISDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "UCOMISSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VADDPDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VADDPSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VADDSDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VADDSSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VADDSUBPDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VADDSUBPSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VCMPPDrmi")>;
def: InstRW<[BWWriteResGroup91], (instregex "VCMPPSrmi")>;
def: InstRW<[BWWriteResGroup91], (instregex "VCMPSDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VCMPSSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VCOMISDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VCOMISSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VCVTDQ2PSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VCVTPS2DQrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VCVTTPS2DQrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VMAXPDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VMAXPSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VMAXSDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VMAXSSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VMINPDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VMINPSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VMINSDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VMINSSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VSUBPDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VSUBPSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VSUBSDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VSUBSSrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VUCOMISDrm")>;
def: InstRW<[BWWriteResGroup91], (instregex "VUCOMISSrm")>;

def BWWriteResGroup91_16 : SchedWriteRes<[BWPort1, BWPort0156, BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1]; 
}
def: InstRW<[BWWriteResGroup91_16], (instregex "IMUL16rm(i8?)")>;

def BWWriteResGroup91_16_2 : SchedWriteRes<[BWPort1, BWPort0156, BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 5;
}
def: InstRW<[BWWriteResGroup91_16_2], (instregex "IMUL16m")>;
def: InstRW<[BWWriteResGroup91_16_2], (instregex "MUL16m")>;

def BWWriteResGroup91_32 : SchedWriteRes<[BWPort1, BWPort0156, BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup91_32], (instregex "IMUL32m")>;
def: InstRW<[BWWriteResGroup91_32], (instregex "MUL32m")>;

def BWWriteResGroup92 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup92], (instregex "VPMOVSXBDYrm")>;
def: InstRW<[BWWriteResGroup92], (instregex "VPMOVSXBQYrm")>;
def: InstRW<[BWWriteResGroup92], (instregex "VPMOVSXBWYrm")>;
def: InstRW<[BWWriteResGroup92], (instregex "VPMOVSXDQYrm")>;
def: InstRW<[BWWriteResGroup92], (instregex "VPMOVSXWDYrm")>;
def: InstRW<[BWWriteResGroup92], (instregex "VPMOVSXWQYrm")>;
def: InstRW<[BWWriteResGroup92], (instregex "VPMOVZXWDYrm")>;

def BWWriteResGroup93 : SchedWriteRes<[BWPort01,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup93], (instregex "MULPDrm")>;
def: InstRW<[BWWriteResGroup93], (instregex "MULPSrm")>;
def: InstRW<[BWWriteResGroup93], (instregex "MULSDrm")>;
def: InstRW<[BWWriteResGroup93], (instregex "MULSSrm")>;
def: InstRW<[BWWriteResGroup93], (instregex "VMULPDrm")>;
def: InstRW<[BWWriteResGroup93], (instregex "VMULPSrm")>;
def: InstRW<[BWWriteResGroup93], (instregex "VMULSDrm")>;
def: InstRW<[BWWriteResGroup93], (instregex "VMULSSrm")>;

def BWWriteResGroup94 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup94], (instregex "VBLENDVPDYrm")>;
def: InstRW<[BWWriteResGroup94], (instregex "VBLENDVPSYrm")>;
def: InstRW<[BWWriteResGroup94], (instregex "VMASKMOVPDYrm")>;
def: InstRW<[BWWriteResGroup94], (instregex "VMASKMOVPSYrm")>;
def: InstRW<[BWWriteResGroup94], (instregex "VPBLENDVBYrm")>;
def: InstRW<[BWWriteResGroup94], (instregex "VPMASKMOVDYrm")>;
def: InstRW<[BWWriteResGroup94], (instregex "VPMASKMOVQYrm")>;

def BWWriteResGroup95 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup95], (instregex "VPSLLVDrm")>;
def: InstRW<[BWWriteResGroup95], (instregex "VPSRAVDrm")>;
def: InstRW<[BWWriteResGroup95], (instregex "VPSRLVDrm")>;

def BWWriteResGroup96 : SchedWriteRes<[BWPort5,BWPort23,BWPort15]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup96], (instregex "MMX_PHADDSWrm64")>;
def: InstRW<[BWWriteResGroup96], (instregex "MMX_PHADDWrm64")>;
def: InstRW<[BWWriteResGroup96], (instregex "MMX_PHADDrm64")>;
def: InstRW<[BWWriteResGroup96], (instregex "MMX_PHSUBDrm64")>;
def: InstRW<[BWWriteResGroup96], (instregex "MMX_PHSUBSWrm64")>;
def: InstRW<[BWWriteResGroup96], (instregex "MMX_PHSUBWrm64")>;
def: InstRW<[BWWriteResGroup96], (instregex "PHADDDrm")>;
def: InstRW<[BWWriteResGroup96], (instregex "PHADDSWrm128")>;
def: InstRW<[BWWriteResGroup96], (instregex "PHADDWrm")>;
def: InstRW<[BWWriteResGroup96], (instregex "PHSUBDrm")>;
def: InstRW<[BWWriteResGroup96], (instregex "PHSUBSWrm128")>;
def: InstRW<[BWWriteResGroup96], (instregex "PHSUBWrm")>;
def: InstRW<[BWWriteResGroup96], (instregex "VPHADDDrm")>;
def: InstRW<[BWWriteResGroup96], (instregex "VPHADDSWrm128")>;
def: InstRW<[BWWriteResGroup96], (instregex "VPHADDWrm")>;
def: InstRW<[BWWriteResGroup96], (instregex "VPHSUBDrm")>;
def: InstRW<[BWWriteResGroup96], (instregex "VPHSUBSWrm128")>;
def: InstRW<[BWWriteResGroup96], (instregex "VPHSUBWrm")>;

def BWWriteResGroup97 : SchedWriteRes<[BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[BWWriteResGroup97], (instregex "RCL(16|32|64)m1")>;
def: InstRW<[BWWriteResGroup97], (instregex "RCL(16|32|64)mi")>;
def: InstRW<[BWWriteResGroup97], (instregex "RCL8m1")>;
def: InstRW<[BWWriteResGroup97], (instregex "RCL8mi")>;
def: InstRW<[BWWriteResGroup97], (instregex "RCR(16|32|64)m1")>;
def: InstRW<[BWWriteResGroup97], (instregex "RCR(16|32|64)mi")>;
def: InstRW<[BWWriteResGroup97], (instregex "RCR8m1")>;
def: InstRW<[BWWriteResGroup97], (instregex "RCR8mi")>;

def BWWriteResGroup98 : SchedWriteRes<[BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,2,1];
}
def: InstRW<[BWWriteResGroup98], (instregex "ROR(16|32|64)mCL")>;
def: InstRW<[BWWriteResGroup98], (instregex "ROR8mCL")>;

def BWWriteResGroup99 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[BWWriteResGroup99], (instregex "ADC(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup99], (instregex "ADC8mi")>;
def: InstRW<[BWWriteResGroup99], (instregex "ADD8mi")>;
def: InstRW<[BWWriteResGroup99], (instregex "AND8mi")>;
def: InstRW<[BWWriteResGroup99], (instregex "OR8mi")>;
def: InstRW<[BWWriteResGroup99], (instregex "SUB8mi")>;
def: InstRW<[BWWriteResGroup99], (instregex "XCHG(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup99], (instregex "XCHG8rm")>;
def: InstRW<[BWWriteResGroup99], (instregex "XOR8mi")>;

def BWWriteResGroup100 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[BWWriteResGroup100], (instregex "ADC(16|32|64)mr")>;
def: InstRW<[BWWriteResGroup100], (instregex "ADC8mr")>;
def: InstRW<[BWWriteResGroup100], (instregex "CMPXCHG(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup100], (instregex "CMPXCHG8rm")>;
def: InstRW<[BWWriteResGroup100], (instregex "ROL(16|32|64)mCL")>;
def: InstRW<[BWWriteResGroup100], (instregex "ROL8mCL")>;
def: InstRW<[BWWriteResGroup100], (instregex "SAR(16|32|64)mCL")>;
def: InstRW<[BWWriteResGroup100], (instregex "SAR8mCL")>;
def: InstRW<[BWWriteResGroup100], (instregex "SBB(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup100], (instregex "SBB(16|32|64)mr")>;
def: InstRW<[BWWriteResGroup100], (instregex "SBB8mi")>;
def: InstRW<[BWWriteResGroup100], (instregex "SBB8mr")>;
def: InstRW<[BWWriteResGroup100], (instregex "SHL(16|32|64)mCL")>;
def: InstRW<[BWWriteResGroup100], (instregex "SHL8mCL")>;
def: InstRW<[BWWriteResGroup100], (instregex "SHR(16|32|64)mCL")>;
def: InstRW<[BWWriteResGroup100], (instregex "SHR8mCL")>;

def BWWriteResGroup101 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup101], (instregex "ADD_F32m")>;
def: InstRW<[BWWriteResGroup101], (instregex "ADD_F64m")>;
def: InstRW<[BWWriteResGroup101], (instregex "ILD_F16m")>;
def: InstRW<[BWWriteResGroup101], (instregex "ILD_F32m")>;
def: InstRW<[BWWriteResGroup101], (instregex "ILD_F64m")>;
def: InstRW<[BWWriteResGroup101], (instregex "SUBR_F32m")>;
def: InstRW<[BWWriteResGroup101], (instregex "SUBR_F64m")>;
def: InstRW<[BWWriteResGroup101], (instregex "SUB_F32m")>;
def: InstRW<[BWWriteResGroup101], (instregex "SUB_F64m")>;
def: InstRW<[BWWriteResGroup101], (instregex "VADDPDYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VADDPSYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VADDSUBPDYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VADDSUBPSYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VCMPPDYrmi")>;
def: InstRW<[BWWriteResGroup101], (instregex "VCMPPSYrmi")>;
def: InstRW<[BWWriteResGroup101], (instregex "VCVTDQ2PSYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VCVTPS2DQYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VCVTTPS2DQYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VMAXPDYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VMAXPSYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VMINPDYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VMINPSYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VSUBPDYrm")>;
def: InstRW<[BWWriteResGroup101], (instregex "VSUBPSYrm")>;

def BWWriteResGroup102 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup102], (instregex "VPERM2F128rm")>;
def: InstRW<[BWWriteResGroup102], (instregex "VPERM2I128rm")>;
def: InstRW<[BWWriteResGroup102], (instregex "VPERMDYrm")>;
def: InstRW<[BWWriteResGroup102], (instregex "VPERMPDYmi")>;
def: InstRW<[BWWriteResGroup102], (instregex "VPERMPSYrm")>;
def: InstRW<[BWWriteResGroup102], (instregex "VPERMQYmi")>;
def: InstRW<[BWWriteResGroup102], (instregex "VPMOVZXBDYrm")>;
def: InstRW<[BWWriteResGroup102], (instregex "VPMOVZXBQYrm")>;
def: InstRW<[BWWriteResGroup102], (instregex "VPMOVZXBWYrm")>;
def: InstRW<[BWWriteResGroup102], (instregex "VPMOVZXDQYrm")>;
def: InstRW<[BWWriteResGroup102], (instregex "VPMOVZXWQYrm")>;

def BWWriteResGroup103 : SchedWriteRes<[BWPort01,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup103], (instregex "VMULPDYrm")>;
def: InstRW<[BWWriteResGroup103], (instregex "VMULPSYrm")>;

def BWWriteResGroup104 : SchedWriteRes<[BWPort0,BWPort1,BWPort5]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup104], (instregex "DPPDrri")>;
def: InstRW<[BWWriteResGroup104], (instregex "VDPPDrri")>;

def BWWriteResGroup105 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup105], (instregex "CVTSD2SI64rm")>;
def: InstRW<[BWWriteResGroup105], (instregex "CVTSD2SIrm")>;
def: InstRW<[BWWriteResGroup105], (instregex "CVTSS2SI64rm")>;
def: InstRW<[BWWriteResGroup105], (instregex "CVTSS2SIrm")>;
def: InstRW<[BWWriteResGroup105], (instregex "CVTTSD2SI64rm")>;
def: InstRW<[BWWriteResGroup105], (instregex "CVTTSD2SIrm")>;
def: InstRW<[BWWriteResGroup105], (instregex "CVTTSS2SIrm")>;
def: InstRW<[BWWriteResGroup105], (instregex "VCVTSD2SI64rm")>;
def: InstRW<[BWWriteResGroup105], (instregex "VCVTSD2SIrm")>;
def: InstRW<[BWWriteResGroup105], (instregex "VCVTSS2SI64rm")>;
def: InstRW<[BWWriteResGroup105], (instregex "VCVTSS2SIrm")>;
def: InstRW<[BWWriteResGroup105], (instregex "VCVTTSD2SI64rm")>;
def: InstRW<[BWWriteResGroup105], (instregex "VCVTTSD2SIrm")>;
def: InstRW<[BWWriteResGroup105], (instregex "VCVTTSS2SI64rm")>;
def: InstRW<[BWWriteResGroup105], (instregex "VCVTTSS2SIrm")>;

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
def: InstRW<[BWWriteResGroup107], (instregex "CVTDQ2PDrm")>;
def: InstRW<[BWWriteResGroup107], (instregex "CVTPD2DQrm")>;
def: InstRW<[BWWriteResGroup107], (instregex "CVTPD2PSrm")>;
def: InstRW<[BWWriteResGroup107], (instregex "CVTSD2SSrm")>;
def: InstRW<[BWWriteResGroup107], (instregex "CVTTPD2DQrm")>;
def: InstRW<[BWWriteResGroup107], (instregex "MMX_CVTPD2PIirm")>;
def: InstRW<[BWWriteResGroup107], (instregex "MMX_CVTPI2PDirm")>;
def: InstRW<[BWWriteResGroup107], (instregex "MMX_CVTTPD2PIirm")>;
def: InstRW<[BWWriteResGroup107], (instregex "MULX64rm")>;
def: InstRW<[BWWriteResGroup107], (instregex "VCVTDQ2PDrm")>;
def: InstRW<[BWWriteResGroup107], (instregex "VCVTSD2SSrm")>;

def BWWriteResGroup108 : SchedWriteRes<[BWPort5,BWPort23,BWPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup108], (instregex "VPBROADCASTBYrm")>;
def: InstRW<[BWWriteResGroup108], (instregex "VPBROADCASTBrm")>;
def: InstRW<[BWWriteResGroup108], (instregex "VPBROADCASTWYrm")>;
def: InstRW<[BWWriteResGroup108], (instregex "VPBROADCASTWrm")>;

def BWWriteResGroup109 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup109], (instregex "VPSLLVDYrm")>;
def: InstRW<[BWWriteResGroup109], (instregex "VPSRAVDYrm")>;
def: InstRW<[BWWriteResGroup109], (instregex "VPSRLVDYrm")>;

def BWWriteResGroup110 : SchedWriteRes<[BWPort5,BWPort23,BWPort15]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup110], (instregex "VPHADDDYrm")>;
def: InstRW<[BWWriteResGroup110], (instregex "VPHADDSWrm256")>;
def: InstRW<[BWWriteResGroup110], (instregex "VPHADDWYrm")>;
def: InstRW<[BWWriteResGroup110], (instregex "VPHSUBDYrm")>;
def: InstRW<[BWWriteResGroup110], (instregex "VPHSUBSWrm256")>;
def: InstRW<[BWWriteResGroup110], (instregex "VPHSUBWYrm")>;

def BWWriteResGroup111 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup111], (instregex "SHLD(16|32|64)mri8")>;
def: InstRW<[BWWriteResGroup111], (instregex "SHRD(16|32|64)mri8")>;

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
def: InstRW<[BWWriteResGroup113], (instregex "LAR(16|32|64)rm")>;
def: InstRW<[BWWriteResGroup113], (instregex "LSL(16|32|64)rm")>;

def BWWriteResGroup114 : SchedWriteRes<[BWPort0]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup114], (instregex "PMULLDrr")>;
def: InstRW<[BWWriteResGroup114], (instregex "VPMULLDYrr")>;
def: InstRW<[BWWriteResGroup114], (instregex "VPMULLDrr")>;

def BWWriteResGroup115 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup115], (instregex "MMX_PMADDUBSWrm64")>;
def: InstRW<[BWWriteResGroup115], (instregex "MMX_PMADDWDirm")>;
def: InstRW<[BWWriteResGroup115], (instregex "MMX_PMULHRSWrm64")>;
def: InstRW<[BWWriteResGroup115], (instregex "MMX_PMULHUWirm")>;
def: InstRW<[BWWriteResGroup115], (instregex "MMX_PMULHWirm")>;
def: InstRW<[BWWriteResGroup115], (instregex "MMX_PMULLWirm")>;
def: InstRW<[BWWriteResGroup115], (instregex "MMX_PMULUDQirm")>;
def: InstRW<[BWWriteResGroup115], (instregex "MMX_PSADBWirm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PCLMULQDQrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PCMPGTQrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PHMINPOSUWrm128")>;
def: InstRW<[BWWriteResGroup115], (instregex "PMADDUBSWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PMADDWDrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PMULDQrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PMULHRSWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PMULHUWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PMULHWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PMULLWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PMULUDQrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "PSADBWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "RCPPSm")>;
def: InstRW<[BWWriteResGroup115], (instregex "RCPSSm")>;
def: InstRW<[BWWriteResGroup115], (instregex "RSQRTPSm")>;
def: InstRW<[BWWriteResGroup115], (instregex "RSQRTSSm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPCLMULQDQrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPCMPGTQrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPHMINPOSUWrm128")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPMADDUBSWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPMADDWDrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPMULDQrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPMULHRSWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPMULHUWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPMULHWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPMULLWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPMULUDQrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VPSADBWrm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VRCPPSm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VRCPSSm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VRSQRTPSm")>;
def: InstRW<[BWWriteResGroup115], (instregex "VRSQRTSSm")>;

def BWWriteResGroup116 : SchedWriteRes<[BWPort01,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD132PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD132PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD132SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD132SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD213PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD213PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD213SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD213SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD231PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD231PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD231SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADD231SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADDSUB132PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADDSUB132PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADDSUB213PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADDSUB213PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADDSUB231PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMADDSUB231PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB132PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB132PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB132SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB132SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB213PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB213PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB213SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB213SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB231PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB231PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB231SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUB231SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUBADD132PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUBADD132PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUBADD213PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUBADD213PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUBADD231PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFMSUBADD231PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD132PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD132PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD132SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD132SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD213PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD213PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD213SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD213SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD231PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD231PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD231SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMADD231SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB132PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB132PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB132SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB132SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB213PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB213PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB213SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB213SSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB231PDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB231PSm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB231SDm")>;
def: InstRW<[BWWriteResGroup116], (instregex "VFNMSUB231SSm")>;

def BWWriteResGroup117 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup117], (instregex "FICOM16m")>;
def: InstRW<[BWWriteResGroup117], (instregex "FICOM32m")>;
def: InstRW<[BWWriteResGroup117], (instregex "FICOMP16m")>;
def: InstRW<[BWWriteResGroup117], (instregex "FICOMP32m")>;

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
def: InstRW<[BWWriteResGroup119], (instregex "HADDPDrm")>;
def: InstRW<[BWWriteResGroup119], (instregex "HADDPSrm")>;
def: InstRW<[BWWriteResGroup119], (instregex "HSUBPDrm")>;
def: InstRW<[BWWriteResGroup119], (instregex "HSUBPSrm")>;
def: InstRW<[BWWriteResGroup119], (instregex "VHADDPDrm")>;
def: InstRW<[BWWriteResGroup119], (instregex "VHADDPSrm")>;
def: InstRW<[BWWriteResGroup119], (instregex "VHSUBPDrm")>;
def: InstRW<[BWWriteResGroup119], (instregex "VHSUBPSrm")>;

def BWWriteResGroup120 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup120], (instregex "CVTTSS2SI64rm")>;

def BWWriteResGroup121 : SchedWriteRes<[BWPort1,BWPort23,BWPort06,BWPort0156]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup121], (instregex "MULX32rm")>;

def BWWriteResGroup122 : SchedWriteRes<[BWPort0]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup122], (instregex "DIVPSrr")>;
def: InstRW<[BWWriteResGroup122], (instregex "DIVSSrr")>;
def: InstRW<[BWWriteResGroup122], (instregex "VDIVPSrr")>;
def: InstRW<[BWWriteResGroup122], (instregex "VDIVSSrr")>;

def BWWriteResGroup123 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup123], (instregex "MUL_F32m")>;
def: InstRW<[BWWriteResGroup123], (instregex "MUL_F64m")>;
def: InstRW<[BWWriteResGroup123], (instregex "VPCMPGTQYrm")>;
def: InstRW<[BWWriteResGroup123], (instregex "VPMADDUBSWYrm")>;
def: InstRW<[BWWriteResGroup123], (instregex "VPMADDWDYrm")>;
def: InstRW<[BWWriteResGroup123], (instregex "VPMULDQYrm")>;
def: InstRW<[BWWriteResGroup123], (instregex "VPMULHRSWYrm")>;
def: InstRW<[BWWriteResGroup123], (instregex "VPMULHUWYrm")>;
def: InstRW<[BWWriteResGroup123], (instregex "VPMULHWYrm")>;
def: InstRW<[BWWriteResGroup123], (instregex "VPMULLWYrm")>;
def: InstRW<[BWWriteResGroup123], (instregex "VPMULUDQYrm")>;
def: InstRW<[BWWriteResGroup123], (instregex "VPSADBWYrm")>;

def BWWriteResGroup124 : SchedWriteRes<[BWPort01,BWPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup124], (instregex "VFMADD132PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADD132PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADD213PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADD213PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADD231PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADD231PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADDSUB132PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADDSUB132PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADDSUB213PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADDSUB213PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADDSUB231PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMADDSUB231PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUB132PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUB132PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUB213PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUB213PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUB231PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUB231PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUBADD132PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUBADD132PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUBADD213PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUBADD213PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUBADD231PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFMSUBADD231PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMADD132PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMADD132PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMADD213PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMADD213PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMADD231PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMADD231PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMSUB132PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMSUB132PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMSUB213PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMSUB213PSYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMSUB231PDYm")>;
def: InstRW<[BWWriteResGroup124], (instregex "VFNMSUB231PSYm")>;

def BWWriteResGroup125 : SchedWriteRes<[BWPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[BWWriteResGroup125], (instregex "PCMPISTRIrr")>;
def: InstRW<[BWWriteResGroup125], (instregex "PCMPISTRM128rr")>;
def: InstRW<[BWWriteResGroup125], (instregex "VPCMPISTRIrr")>;
def: InstRW<[BWWriteResGroup125], (instregex "VPCMPISTRM128rr")>;

def BWWriteResGroup126 : SchedWriteRes<[BWPort0,BWPort015]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup126], (instregex "VRCPPSYr")>;
def: InstRW<[BWWriteResGroup126], (instregex "VRSQRTPSYr")>;

def BWWriteResGroup127 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup127], (instregex "ROUNDPDm")>;
def: InstRW<[BWWriteResGroup127], (instregex "ROUNDPSm")>;
def: InstRW<[BWWriteResGroup127], (instregex "ROUNDSDm")>;
def: InstRW<[BWWriteResGroup127], (instregex "ROUNDSSm")>;
def: InstRW<[BWWriteResGroup127], (instregex "VROUNDPDm")>;
def: InstRW<[BWWriteResGroup127], (instregex "VROUNDPSm")>;
def: InstRW<[BWWriteResGroup127], (instregex "VROUNDSDm")>;
def: InstRW<[BWWriteResGroup127], (instregex "VROUNDSSm")>;

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
def: InstRW<[BWWriteResGroup129], (instregex "VHADDPDYrm")>;
def: InstRW<[BWWriteResGroup129], (instregex "VHADDPSYrm")>;
def: InstRW<[BWWriteResGroup129], (instregex "VHSUBPDYrm")>;
def: InstRW<[BWWriteResGroup129], (instregex "VHSUBPSYrm")>;

def BWWriteResGroup130 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,1,2];
}
def: InstRW<[BWWriteResGroup130], (instregex "SHLD(16|32|64)mrCL")>;
def: InstRW<[BWWriteResGroup130], (instregex "SHRD(16|32|64)mrCL")>;

def BWWriteResGroup131 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,2,3];
}
def: InstRW<[BWWriteResGroup131], (instregex "RCL(16|32|64)rCL")>;
def: InstRW<[BWWriteResGroup131], (instregex "RCR(16|32|64)rCL")>;

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
def: InstRW<[BWWriteResGroup133], (instregex "LOOPE")>;
def: InstRW<[BWWriteResGroup133], (instregex "LOOPNE")>;

def BWWriteResGroup134 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup134], (instregex "AESDECLASTrm")>;
def: InstRW<[BWWriteResGroup134], (instregex "AESDECrm")>;
def: InstRW<[BWWriteResGroup134], (instregex "AESENCLASTrm")>;
def: InstRW<[BWWriteResGroup134], (instregex "AESENCrm")>;
def: InstRW<[BWWriteResGroup134], (instregex "VAESDECLASTrm")>;
def: InstRW<[BWWriteResGroup134], (instregex "VAESDECrm")>;
def: InstRW<[BWWriteResGroup134], (instregex "VAESENCLASTrm")>;
def: InstRW<[BWWriteResGroup134], (instregex "VAESENCrm")>;

def BWWriteResGroup135 : SchedWriteRes<[BWPort1,BWPort23]> {
  let Latency = 12;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup135], (instregex "ADD_FI16m")>;
def: InstRW<[BWWriteResGroup135], (instregex "ADD_FI32m")>;
def: InstRW<[BWWriteResGroup135], (instregex "SUBR_FI16m")>;
def: InstRW<[BWWriteResGroup135], (instregex "SUBR_FI32m")>;
def: InstRW<[BWWriteResGroup135], (instregex "SUB_FI16m")>;
def: InstRW<[BWWriteResGroup135], (instregex "SUB_FI32m")>;
def: InstRW<[BWWriteResGroup135], (instregex "VROUNDYPDm")>;
def: InstRW<[BWWriteResGroup135], (instregex "VROUNDYPSm")>;

def BWWriteResGroup136 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[BWWriteResGroup136], (instregex "MPSADBWrmi")>;
def: InstRW<[BWWriteResGroup136], (instregex "VMPSADBWrmi")>;

def BWWriteResGroup137 : SchedWriteRes<[BWPort0]> {
  let Latency = 13;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup137], (instregex "SQRTPSr")>;
def: InstRW<[BWWriteResGroup137], (instregex "SQRTSSr")>;

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
def: InstRW<[BWWriteResGroup139], (instregex "DIVPDrr")>;
def: InstRW<[BWWriteResGroup139], (instregex "DIVSDrr")>;
def: InstRW<[BWWriteResGroup139], (instregex "VDIVPDrr")>;
def: InstRW<[BWWriteResGroup139], (instregex "VDIVSDrr")>;
def: InstRW<[BWWriteResGroup139], (instregex "VSQRTPSr")>;
def: InstRW<[BWWriteResGroup139], (instregex "VSQRTSSr")>;

def BWWriteResGroup140 : SchedWriteRes<[BWPort5]> {
  let Latency = 14;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup140], (instregex "AESIMCrr")>;
def: InstRW<[BWWriteResGroup140], (instregex "VAESIMCrr")>;

def BWWriteResGroup141 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup141], (instregex "MUL_FI16m")>;
def: InstRW<[BWWriteResGroup141], (instregex "MUL_FI32m")>;

def BWWriteResGroup142 : SchedWriteRes<[BWPort0,BWPort1,BWPort5]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup142], (instregex "DPPSrri")>;
def: InstRW<[BWWriteResGroup142], (instregex "VDPPSYrri")>;
def: InstRW<[BWWriteResGroup142], (instregex "VDPPSrri")>;

def BWWriteResGroup143 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23]> {
  let Latency = 14;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[BWWriteResGroup143], (instregex "DPPDrmi")>;
def: InstRW<[BWWriteResGroup143], (instregex "VDPPDrmi")>;

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
def: InstRW<[BWWriteResGroup147], (instregex "DIVR_FPrST0")>;
def: InstRW<[BWWriteResGroup147], (instregex "DIVR_FST0r")>;
def: InstRW<[BWWriteResGroup147], (instregex "DIVR_FrST0")>;

def BWWriteResGroup148 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 15;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup148], (instregex "PMULLDrm")>;
def: InstRW<[BWWriteResGroup148], (instregex "VPMULLDrm")>;

def BWWriteResGroup149 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 15;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,4,1,2];
}
def: InstRW<[BWWriteResGroup149], (instregex "RCL(16|32|64)mCL")>;
def: InstRW<[BWWriteResGroup149], (instregex "RCL8mCL")>;

def BWWriteResGroup150 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup150], (instregex "DIVPSrm")>;
def: InstRW<[BWWriteResGroup150], (instregex "DIVSSrm")>;
def: InstRW<[BWWriteResGroup150], (instregex "VDIVPSrm")>;
def: InstRW<[BWWriteResGroup150], (instregex "VDIVSSrm")>;

def BWWriteResGroup151 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 16;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup151], (instregex "VPMULLDYrm")>;

def BWWriteResGroup152 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}
def: InstRW<[BWWriteResGroup152], (instregex "PCMPISTRIrm")>;
def: InstRW<[BWWriteResGroup152], (instregex "PCMPISTRM128rm")>;
def: InstRW<[BWWriteResGroup152], (instregex "VPCMPISTRIrm")>;
def: InstRW<[BWWriteResGroup152], (instregex "VPCMPISTRM128rm")>;

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
def: InstRW<[BWWriteResGroup154], (instregex "VZEROALL")>;

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
def: InstRW<[BWWriteResGroup156], (instregex "VRCPPSYm")>;
def: InstRW<[BWWriteResGroup156], (instregex "VRSQRTPSYm")>;

def BWWriteResGroup157 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup157], (instregex "SQRTPSm")>;
def: InstRW<[BWWriteResGroup157], (instregex "SQRTSSm")>;

def BWWriteResGroup158 : SchedWriteRes<[BWPort0,BWPort5,BWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def: InstRW<[BWWriteResGroup158], (instregex "PCMPESTRIrr")>;
def: InstRW<[BWWriteResGroup158], (instregex "VPCMPESTRIrr")>;

def BWWriteResGroup159 : SchedWriteRes<[BWPort5,BWPort6,BWPort06,BWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[BWWriteResGroup159], (instregex "CPUID")>;
def: InstRW<[BWWriteResGroup159], (instregex "RDTSC")>;

def BWWriteResGroup160 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 18;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,3,1,3];
}
def: InstRW<[BWWriteResGroup160], (instregex "RCR(16|32|64)mCL")>;
def: InstRW<[BWWriteResGroup160], (instregex "RCR8mCL")>;

def BWWriteResGroup161 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup161], (instregex "DIVPDrm")>;
def: InstRW<[BWWriteResGroup161], (instregex "DIVSDrm")>;
def: InstRW<[BWWriteResGroup161], (instregex "VDIVPDrm")>;
def: InstRW<[BWWriteResGroup161], (instregex "VDIVSDrm")>;
def: InstRW<[BWWriteResGroup161], (instregex "VSQRTPSm")>;
def: InstRW<[BWWriteResGroup161], (instregex "VSQRTSSm")>;

def BWWriteResGroup162 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 19;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup162], (instregex "AESIMCrm")>;
def: InstRW<[BWWriteResGroup162], (instregex "VAESIMCrm")>;

def BWWriteResGroup163 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23]> {
  let Latency = 19;
  let NumMicroOps = 5;
  let ResourceCycles = [2,1,1,1];
}
def: InstRW<[BWWriteResGroup163], (instregex "DPPSrmi")>;
def: InstRW<[BWWriteResGroup163], (instregex "VDPPSrmi")>;

def BWWriteResGroup164 : SchedWriteRes<[BWPort0,BWPort5,BWPort015,BWPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[BWWriteResGroup164], (instregex "PCMPESTRM128rr")>;
def: InstRW<[BWWriteResGroup164], (instregex "VPCMPESTRM128rr")>;

def BWWriteResGroup165 : SchedWriteRes<[BWPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup165], (instregex "DIV_FPrST0")>;
def: InstRW<[BWWriteResGroup165], (instregex "DIV_FST0r")>;
def: InstRW<[BWWriteResGroup165], (instregex "DIV_FrST0")>;
def: InstRW<[BWWriteResGroup165], (instregex "SQRTPDr")>;
def: InstRW<[BWWriteResGroup165], (instregex "SQRTSDr")>;

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
def: InstRW<[BWWriteResGroup167], (instregex "INSB")>;
def: InstRW<[BWWriteResGroup167], (instregex "INSL")>;
def: InstRW<[BWWriteResGroup167], (instregex "INSW")>;

def BWWriteResGroup168 : SchedWriteRes<[BWPort0]> {
  let Latency = 21;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup168], (instregex "VSQRTPDr")>;
def: InstRW<[BWWriteResGroup168], (instregex "VSQRTSDr")>;

def BWWriteResGroup169 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup169], (instregex "DIV_F32m")>;
def: InstRW<[BWWriteResGroup169], (instregex "DIV_F64m")>;

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

def BWWriteResGroup175 : SchedWriteRes<[BWPort0,BWPort5,BWPort23,BWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[BWWriteResGroup175], (instregex "PCMPESTRIrm")>;
def: InstRW<[BWWriteResGroup175], (instregex "VPCMPESTRIrm")>;

def BWWriteResGroup176 : SchedWriteRes<[BWPort6,BWPort23,BWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 19;
  let ResourceCycles = [3,1,15];
}
def: InstRW<[BWWriteResGroup176], (instregex "XRSTOR(64?)")>;

def BWWriteResGroup177 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 24;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup177], (instregex "DIV_FI16m")>;
def: InstRW<[BWWriteResGroup177], (instregex "DIV_FI32m")>;

def BWWriteResGroup178 : SchedWriteRes<[BWPort0,BWPort5,BWPort23,BWPort015,BWPort0156]> {
  let Latency = 24;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}
def: InstRW<[BWWriteResGroup178], (instregex "PCMPESTRM128rm")>;
def: InstRW<[BWWriteResGroup178], (instregex "VPCMPESTRM128rm")>;

def BWWriteResGroup179 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup179], (instregex "SQRTPDm")>;
def: InstRW<[BWWriteResGroup179], (instregex "SQRTSDm")>;

def BWWriteResGroup180 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 26;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup180], (instregex "DIVR_F32m")>;
def: InstRW<[BWWriteResGroup180], (instregex "DIVR_F64m")>;
def: InstRW<[BWWriteResGroup180], (instregex "VSQRTPDm")>;
def: InstRW<[BWWriteResGroup180], (instregex "VSQRTSDm")>;

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
def: InstRW<[BWWriteResGroup182], (instregex "DIVR_FI16m")>;
def: InstRW<[BWWriteResGroup182], (instregex "DIVR_FI32m")>;

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
def: InstRW<[BWWriteResGroup183_1], (instregex "VGATHERQPDrm")>;

def BWWriteResGroup183_2 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 23;
  let NumMicroOps = 9;
  let ResourceCycles = [1,3,4,1];
}
def: InstRW<[BWWriteResGroup183_2], (instregex "VGATHERQPDYrm")>;

def BWWriteResGroup183_3 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,2,1];
}
def: InstRW<[BWWriteResGroup183_3], (instregex "VGATHERQPSYrm")>;

def BWWriteResGroup183_4 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 25;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,2,1];
}
def: InstRW<[BWWriteResGroup183_4], (instregex "VGATHERDPDrm")>;
def: InstRW<[BWWriteResGroup183_4], (instregex "VGATHERDPSrm")>;

def BWWriteResGroup183_5 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 26;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,2,1];
}
def: InstRW<[BWWriteResGroup183_5], (instregex "VGATHERDPDYrm")>;

def BWWriteResGroup183_6 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 26;
  let NumMicroOps = 14;
  let ResourceCycles = [1,4,8,1];  
}
def: InstRW<[BWWriteResGroup183_6], (instregex "VGATHERDPSYrm")>;

def BWWriteResGroup183_7 : SchedWriteRes<[BWPort4, BWPort5, BWPort23, BWPort0156]> {
  let Latency = 27;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,2,1];
}
def: InstRW<[BWWriteResGroup183_7], (instregex "VGATHERQPSrm")>;

def BWWriteResGroup184 : SchedWriteRes<[BWPort0,BWPort5,BWPort015]> {
  let Latency = 29;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,2];
}
def: InstRW<[BWWriteResGroup184], (instregex "AESKEYGENASSIST128rr")>;
def: InstRW<[BWWriteResGroup184], (instregex "VAESKEYGENASSIST128rr")>;

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
def: InstRW<[BWWriteResGroup186], (instregex "XSAVE(OPT?)")>;

def BWWriteResGroup187 : SchedWriteRes<[BWPort01,BWPort15,BWPort015,BWPort0156]> {
  let Latency = 31;
  let NumMicroOps = 31;
  let ResourceCycles = [8,1,21,1];
}
def: InstRW<[BWWriteResGroup187], (instregex "MMX_EMMS")>;

def BWWriteResGroup188 : SchedWriteRes<[BWPort0,BWPort5,BWPort23,BWPort015]> {
  let Latency = 33;
  let NumMicroOps = 11;
  let ResourceCycles = [2,7,1,1];
}
def: InstRW<[BWWriteResGroup188], (instregex "AESKEYGENASSIST128rm")>;
def: InstRW<[BWWriteResGroup188], (instregex "VAESKEYGENASSIST128rm")>;

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
def: InstRW<[BWWriteResGroup190], (instregex "DIV(16|32|64)m")>;
def: InstRW<[BWWriteResGroup190], (instregex "DIV8m")>;

def BWWriteResGroup191 : SchedWriteRes<[BWPort5,BWPort6,BWPort23,BWPort06,BWPort0156]> {
  let Latency = 34;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[BWWriteResGroup191], (instregex "IN32ri")>;
def: InstRW<[BWWriteResGroup191], (instregex "IN32rr")>;
def: InstRW<[BWWriteResGroup191], (instregex "IN8ri")>;
def: InstRW<[BWWriteResGroup191], (instregex "IN8rr")>;

def BWWriteResGroup193 : SchedWriteRes<[BWPort0,BWPort1,BWPort5,BWPort23,BWPort0156]> {
  let Latency = 35;
  let NumMicroOps = 8;
  let ResourceCycles = [2,2,2,1,1];
}
def: InstRW<[BWWriteResGroup193], (instregex "IDIV(16|32|64)m")>;
def: InstRW<[BWWriteResGroup193], (instregex "IDIV8m")>;

def BWWriteResGroup194 : SchedWriteRes<[BWPort5,BWPort6,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[BWWriteResGroup194], (instregex "OUT32ir")>;
def: InstRW<[BWWriteResGroup194], (instregex "OUT32rr")>;
def: InstRW<[BWWriteResGroup194], (instregex "OUT8ir")>;
def: InstRW<[BWWriteResGroup194], (instregex "OUT8rr")>;

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
def: InstRW<[BWWriteResGroup196], (instregex "RDTSCP")>;

def BWWriteResGroup197 : SchedWriteRes<[BWPort0,BWPort01,BWPort23,BWPort05,BWPort06,BWPort015,BWPort0156]> {
  let Latency = 60;
  let NumMicroOps = 64;
  let ResourceCycles = [2,2,8,1,10,2,39];
}
def: InstRW<[BWWriteResGroup197], (instregex "FLDENVm")>;
def: InstRW<[BWWriteResGroup197], (instregex "FLDENVm")>;

def BWWriteResGroup198 : SchedWriteRes<[BWPort0,BWPort6,BWPort23,BWPort05,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 63;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[BWWriteResGroup198], (instregex "FXRSTOR64")>;

def BWWriteResGroup199 : SchedWriteRes<[BWPort0,BWPort6,BWPort23,BWPort05,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 63;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[BWWriteResGroup199], (instregex "FXRSTOR")>;

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
def: InstRW<[BWWriteResGroup202], (instregex "FSTENVm")>;

} // SchedModel

