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
defm : SKLWriteResPair<WriteFAdd,   SKLPort1, 3>; // Floating point add/sub/compare.
defm : SKLWriteResPair<WriteFMul,   SKLPort0, 5>; // Floating point multiplication.
defm : SKLWriteResPair<WriteFDiv,   SKLPort0, 12>; // 10-14 cycles. // Floating point division.
defm : SKLWriteResPair<WriteFSqrt,  SKLPort0, 15>; // Floating point square root.
defm : SKLWriteResPair<WriteFRcp,   SKLPort0, 5>; // Floating point reciprocal estimate.
defm : SKLWriteResPair<WriteFRsqrt, SKLPort0, 5>; // Floating point reciprocal square root estimate.
// defm WriteFMA    : X86SchedWritePair; // Fused Multiply Add.
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

def SKLWriteResGroup0 : SchedWriteRes<[SKLPort23]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup0], (instregex "LDDQUrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "LD_F32m")>;
def: InstRW<[SKLWriteResGroup0], (instregex "LD_F64m")>;
def: InstRW<[SKLWriteResGroup0], (instregex "LD_F80m")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MMX_MOVD64rm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MMX_MOVD64to64rm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MMX_MOVQ64rm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOV(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOV64toPQIrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOV8rm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVAPDrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVAPSrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVDDUPrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVDI2PDIrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVDQArm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVDQUrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVNTDQArm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVSHDUPrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVSLDUPrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVSSrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVSX(16|32|64)rm16")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVSX(16|32|64)rm32")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVSX(16|32|64)rm8")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVUPDrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVUPSrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVZX(16|32|64)rm16")>;
def: InstRW<[SKLWriteResGroup0], (instregex "MOVZX(16|32|64)rm8")>;
def: InstRW<[SKLWriteResGroup0], (instregex "PREFETCHNTA")>;
def: InstRW<[SKLWriteResGroup0], (instregex "PREFETCHT0")>;
def: InstRW<[SKLWriteResGroup0], (instregex "PREFETCHT1")>;
def: InstRW<[SKLWriteResGroup0], (instregex "PREFETCHT2")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VBROADCASTF128")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VBROADCASTI128")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VBROADCASTSDYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VBROADCASTSSYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VBROADCASTSSrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VLDDQUYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VLDDQUrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOV64toPQIrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVAPDYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVAPDrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVAPSYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVAPSrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVDDUPYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVDDUPrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVDI2PDIrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVDQAYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVDQArm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVDQUYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVDQUrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVNTDQAYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVNTDQArm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVQI2PQIrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVSDrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVSHDUPYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVSHDUPrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVSLDUPYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVSLDUPrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVSSrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVUPDYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVUPDrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVUPSYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VMOVUPSrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VPBROADCASTDYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VPBROADCASTDrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VPBROADCASTQYrm")>;
def: InstRW<[SKLWriteResGroup0], (instregex "VPBROADCASTQrm")>;

def SKLWriteResGroup1 : SchedWriteRes<[SKLPort4,SKLPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup1], (instregex "FBSTPm")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_MOVD64from64rm")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_MOVD64mr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_MOVNTQmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_MOVQ64mr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOV(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOV8mi")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOV8mr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVAPDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVAPSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVDQAmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVDQUmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVHPDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVHPSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVLPDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVLPSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVNTDQmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVNTI_64mr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVNTImr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVNTPDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVNTPSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVPDI2DImr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVPQI2QImr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVPQIto64mr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVSSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVUPDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "MOVUPSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "ST_FP32m")>;
def: InstRW<[SKLWriteResGroup1], (instregex "ST_FP64m")>;
def: InstRW<[SKLWriteResGroup1], (instregex "ST_FP80m")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VEXTRACTF128mr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VEXTRACTI128mr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVAPDYmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVAPDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVAPSYmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVAPSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVDQAYmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVDQAmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVDQUYmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVDQUmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVHPDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVHPSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVLPDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVLPSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVNTDQYmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVNTDQmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVNTPDYmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVNTPDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVNTPSYmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVNTPSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVPDI2DImr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVPQI2QImr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVPQIto64mr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVSDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVSSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVUPDYmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVUPDmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVUPSYmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMOVUPSmr")>;
def: InstRW<[SKLWriteResGroup1], (instregex "VMPTRSTm")>;

def SKLWriteResGroup2 : SchedWriteRes<[SKLPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PADDSBirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PADDSWirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PADDUSBirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PADDUSWirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PAVGBirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PAVGWirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PCMPEQBirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PCMPEQDirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PCMPEQWirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PCMPGTBirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PCMPGTDirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PCMPGTWirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PMAXSWirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PMAXUBirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PMINSWirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PMINUBirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSLLDri")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSLLDrr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSLLQri")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSLLQrr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSLLWri")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSLLWrr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSRADri")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSRADrr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSRAWri")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSRAWrr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSRLDri")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSRLDrr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSRLQri")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSRLQrr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSRLWri")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSRLWrr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSUBSBirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSUBSWirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSUBUSBirr")>;
def: InstRW<[SKLWriteResGroup2], (instregex "MMX_PSUBUSWirr")>;

def SKLWriteResGroup3 : SchedWriteRes<[SKLPort1]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup3], (instregex "MMX_MASKMOVQ64")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PABSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PABSDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PABSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PADDSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PADDSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PADDUSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PADDUSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PAVGBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PAVGWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PCMPEQBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PCMPEQDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PCMPEQQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PCMPEQWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PCMPGTBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PCMPGTDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PCMPGTWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMAXSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMAXSDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMAXSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMAXUBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMAXUDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMAXUWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMINSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMINSDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMINSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMINUBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMINUDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PMINUWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSIGNBrr128")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSIGNDrr128")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSIGNWrr128")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSLLDri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSLLQri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSLLWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSRADri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSRAWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSRLDri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSRLQri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSRLWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSUBSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSUBSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSUBUSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "PSUBUSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPABSBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPABSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPABSDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPABSDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPABSWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPABSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPADDSBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPADDSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPADDSWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPADDSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPADDUSBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPADDUSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPADDUSWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPADDUSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPAVGBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPAVGBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPAVGWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPAVGWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPEQBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPEQBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPEQDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPEQDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPEQQYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPEQQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPEQWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPEQWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPGTBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPGTBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPGTDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPGTDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPGTWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPCMPGTWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXSBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXSDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXSDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXSWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXUBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXUBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXUDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXUDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXUWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMAXUWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINSBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINSDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINSDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINSWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINUBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINUBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINUDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINUDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINUWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPMINUWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSIGNBYrr256")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSIGNBrr128")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSIGNDYrr256")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSIGNDrr128")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSIGNWYrr256")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSIGNWrr128")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLDYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLDri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLQYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLQri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLVDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLVDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLVQYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLVQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLWYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSLLWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRADYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRADri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRAVDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRAVDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRAWYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRAWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLDYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLDri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLQYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLQri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLVDYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLVDrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLVQYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLVQrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLWYri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSRLWri")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSUBSBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSUBSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSUBSWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSUBSWrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSUBUSBYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSUBUSBrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSUBUSWYrr")>;
def: InstRW<[SKLWriteResGroup3], (instregex "VPSUBUSWrr")>;

def SKLWriteResGroup4 : SchedWriteRes<[SKLPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup4], (instregex "COMP_FST0r")>;
def: InstRW<[SKLWriteResGroup4], (instregex "COM_FST0r")>;
def: InstRW<[SKLWriteResGroup4], (instregex "FINCSTP")>;
def: InstRW<[SKLWriteResGroup4], (instregex "FNOP")>;
def: InstRW<[SKLWriteResGroup4], (instregex "INSERTPSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_MOVD64rr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_MOVD64to64rr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_MOVQ64rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PABSBrr64")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PABSDrr64")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PABSWrr64")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PADDBirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PADDDirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PADDQirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PADDWirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PALIGNR64irr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PANDNirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PANDirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PORirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PSHUFBrr64")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PSHUFWri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PSIGNBrr64")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PSIGNDrr64")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PSIGNWrr64")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PSUBBirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PSUBDirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PSUBQirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PSUBWirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PUNPCKHBWirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PUNPCKHDQirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PUNPCKHWDirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PUNPCKLBWirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PUNPCKLDQirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PUNPCKLWDirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MMX_PXORirr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MOV64toPQIrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MOVDDUPrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MOVDI2PDIrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MOVHLPSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MOVLHPSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MOVSDrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MOVSHDUPrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MOVSLDUPrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MOVUPDrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup4], (instregex "MOVUPSrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PACKSSDWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PACKSSWBrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PACKUSDWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PACKUSWBrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PALIGNRrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PBLENDWrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVSXBDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVSXBQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVSXBWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVSXDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVSXWDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVSXWQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVZXBDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVZXBQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVZXBWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVZXDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVZXWDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PMOVZXWQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PSHUFBrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PSHUFDri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PSHUFHWri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PSHUFLWri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PSLLDQri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PSRLDQri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PUNPCKHBWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PUNPCKHDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PUNPCKHQDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PUNPCKHWDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PUNPCKLBWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PUNPCKLDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PUNPCKLQDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "PUNPCKLWDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "SHUFPDrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "SHUFPSrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "UCOM_FPr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "UCOM_Fr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "UNPCKHPDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "UNPCKHPSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "UNPCKLPDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "UNPCKLPSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VBROADCASTSSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VINSERTPSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOV64toPQIrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVDDUPYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVDDUPrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVDI2PDIrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVHLPSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVLHPSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVSDrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVSHDUPYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVSHDUPrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVSLDUPYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVSLDUPrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVUPDYrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVUPDrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVUPSYrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VMOVUPSrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPACKSSDWYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPACKSSDWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPACKSSWBYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPACKSSWBrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPACKUSDWYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPACKUSDWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPACKUSWBYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPACKUSWBrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPALIGNRYrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPALIGNRrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPBLENDWYrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPBLENDWrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPBROADCASTDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPBROADCASTQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPERMILPDYri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPERMILPDYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPERMILPDri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPERMILPDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPERMILPSYri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPERMILPSYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPERMILPSri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPERMILPSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVSXBDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVSXBQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVSXBWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVSXDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVSXWDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVSXWQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVZXBDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVZXBQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVZXBWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVZXDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVZXWDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPMOVZXWQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSHUFBYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSHUFBrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSHUFDYri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSHUFDri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSHUFHWYri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSHUFHWri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSHUFLWYri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSHUFLWri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSLLDQYri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSLLDQri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSRLDQYri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPSRLDQri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKHBWYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKHBWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKHDQYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKHDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKHQDQYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKHQDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKHWDYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKHWDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKLBWYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKLBWrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKLDQYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKLDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKLQDQYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKLQDQrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKLWDYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VPUNPCKLWDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VSHUFPDYrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VSHUFPDrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VSHUFPSYrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VSHUFPSrri")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VUNPCKHPDYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VUNPCKHPDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VUNPCKHPSYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VUNPCKHPSrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VUNPCKLPDYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VUNPCKLPDrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VUNPCKLPSYrr")>;
def: InstRW<[SKLWriteResGroup4], (instregex "VUNPCKLPSrr")>;

def SKLWriteResGroup5 : SchedWriteRes<[SKLPort6]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup5], (instregex "ADC(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup5], (instregex "ADC(16|32|64)rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup5], (instregex "ADC8rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup5], (instregex "ADCX32rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "ADCX64rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "ADOX32rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "ADOX64rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "BT(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup5], (instregex "BT(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "BTC(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup5], (instregex "BTC(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "BTR(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup5], (instregex "BTR(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "BTS(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup5], (instregex "BTS(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CDQ")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CLAC")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVAE(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVB(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVE(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVG(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVGE(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVL(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVLE(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVNE(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVNO(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVNP(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVNS(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVO(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVP(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CMOVS(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "CQO")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JAE_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JAE_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JA_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JA_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JBE_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JBE_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JB_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JB_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JE_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JE_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JGE_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JGE_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JG_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JG_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JLE_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JLE_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JL_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JL_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JMP(16|32|64)r")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JMP_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JMP_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JNE_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JNE_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JNO_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JNO_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JNP_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JNP_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JNS_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JNS_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JO_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JO_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JP_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JP_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JS_1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "JS_4")>;
def: InstRW<[SKLWriteResGroup5], (instregex "RORX32ri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "RORX64ri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SAR(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SAR(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SAR8r1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SAR8ri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SARX32rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SARX64rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SBB(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SBB(16|32|64)rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SBB8rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETAEr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETBr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETEr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETGEr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETGr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETLEr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETLr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETNEr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETNOr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETNPr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETNSr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETOr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETPr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SETSr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHL(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHL(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHL8r1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHL8ri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHLX32rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHLX64rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHR(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHR(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHR8r1")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHR8ri")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHRX32rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "SHRX64rr")>;
def: InstRW<[SKLWriteResGroup5], (instregex "STAC")>;

def SKLWriteResGroup6 : SchedWriteRes<[SKLPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup6], (instregex "ANDN32rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "ANDN64rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "ANDNPDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "ANDNPSrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "ANDPDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "ANDPSrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "BLENDPDrri")>;
def: InstRW<[SKLWriteResGroup6], (instregex "BLENDPSrri")>;
def: InstRW<[SKLWriteResGroup6], (instregex "BLSI32rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "BLSI64rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "BLSMSK32rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "BLSMSK64rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "BLSR32rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "BLSR64rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "BZHI32rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "BZHI64rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "LEA(16|32|64)r")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MOVAPDrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MOVAPSrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MOVDQArr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MOVDQUrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MOVPQI2QIrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "MOVSSrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "ORPDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "ORPSrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PADDBrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PADDDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PADDQrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PADDWrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PANDNrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PANDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PORrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PSUBBrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PSUBDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PSUBQrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PSUBWrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "PXORrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VANDNPDYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VANDNPDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VANDNPSYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VANDNPSrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VANDPDYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VANDPDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VANDPSYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VANDPSrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VBLENDPDYrri")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VBLENDPDrri")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VBLENDPSYrri")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VBLENDPSrri")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVAPDYrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVAPDrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVAPSYrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVAPSrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVDQAYrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVDQArr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVDQUYrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVDQUrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVPQI2QIrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVSSrr(_REV?)")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VMOVZPQILo2PQIrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VORPDYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VORPDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VORPSYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VORPSrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPADDBYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPADDBrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPADDDYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPADDDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPADDQYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPADDQrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPADDWYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPADDWrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPANDNYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPANDNrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPANDYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPANDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPBLENDDYrri")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPBLENDDrri")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPORYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPORrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPSUBBYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPSUBBrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPSUBDYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPSUBDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPSUBQYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPSUBQrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPSUBWYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPSUBWrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPXORYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VPXORrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VXORPDYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VXORPDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VXORPSYrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "VXORPSrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "XORPDrr")>;
def: InstRW<[SKLWriteResGroup6], (instregex "XORPSrr")>;

def SKLWriteResGroup7 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup7], (instregex "ADD(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "ADD(16|32|64)rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "ADD8i8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "ADD8ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "ADD8rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "AND(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "AND(16|32|64)rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "AND8i8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "AND8ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "AND8rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CBW")>;
//def: InstRW<[SKLWriteResGroup7], (instregex "CDQE")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CLC")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CMC")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CMP(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CMP(16|32|64)rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CMP8i8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CMP8ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CMP8rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "CWDE")>;
def: InstRW<[SKLWriteResGroup7], (instregex "DEC(16|32|64)r")>;
def: InstRW<[SKLWriteResGroup7], (instregex "DEC8r")>;
def: InstRW<[SKLWriteResGroup7], (instregex "INC(16|32|64)r")>;
def: InstRW<[SKLWriteResGroup7], (instregex "INC8r")>;
def: InstRW<[SKLWriteResGroup7], (instregex "LAHF")>;
def: InstRW<[SKLWriteResGroup7], (instregex "MOV(16|32|64)rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "MOV8ri(_alt?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "MOV8rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "MOVSX(16|32|64)rr16")>;
def: InstRW<[SKLWriteResGroup7], (instregex "MOVSX(16|32|64)rr32")>;
def: InstRW<[SKLWriteResGroup7], (instregex "MOVSX(16|32|64)rr8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "MOVZX(16|32|64)rr16")>;
def: InstRW<[SKLWriteResGroup7], (instregex "MOVZX(16|32|64)rr8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "NEG(16|32|64)r")>;
def: InstRW<[SKLWriteResGroup7], (instregex "NEG8r")>;
def: InstRW<[SKLWriteResGroup7], (instregex "NOOP")>;
def: InstRW<[SKLWriteResGroup7], (instregex "NOT(16|32|64)r")>;
def: InstRW<[SKLWriteResGroup7], (instregex "NOT8r")>;
def: InstRW<[SKLWriteResGroup7], (instregex "OR(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "OR(16|32|64)rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "OR8i8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "OR8ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "OR8rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SAHF")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SGDT64m")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SIDT64m")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SLDT64m")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SMSW16m")>;
def: InstRW<[SKLWriteResGroup7], (instregex "STC")>;
def: InstRW<[SKLWriteResGroup7], (instregex "STRm")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SUB(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SUB(16|32|64)rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SUB8i8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SUB8ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SUB8rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "SYSCALL")>;
def: InstRW<[SKLWriteResGroup7], (instregex "TEST(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "TEST8i8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "TEST8ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "TEST8rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "XCHG(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup7], (instregex "XOR(16|32|64)ri8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "XOR(16|32|64)rr(_REV?)")>;
def: InstRW<[SKLWriteResGroup7], (instregex "XOR8i8")>;
def: InstRW<[SKLWriteResGroup7], (instregex "XOR8ri")>;
def: InstRW<[SKLWriteResGroup7], (instregex "XOR8rr(_REV?)")>;

def SKLWriteResGroup12 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PADDSBirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PADDSWirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PADDUSBirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PADDUSWirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PAVGBirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PAVGWirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PCMPEQBirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PCMPEQDirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PCMPEQWirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PCMPGTBirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PCMPGTDirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PCMPGTWirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PMAXSWirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PMAXUBirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PMINSWirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PMINUBirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSLLDrm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSLLQrm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSLLWrm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSRADrm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSRAWrm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSRLDrm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSRLQrm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSRLWrm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSUBSBirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSUBSWirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSUBUSBirm")>;
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_PSUBUSWirm")>;

def SKLWriteResGroup13 : SchedWriteRes<[SKLPort0,SKLPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup13], (instregex "MMX_MASKMOVQ64")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VMASKMOVDQU")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VMASKMOVPDYmr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VMASKMOVPDmr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VMASKMOVPSYmr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VMASKMOVPSmr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VPMASKMOVDYmr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VPMASKMOVDmr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VPMASKMOVQYmr")>;
def: InstRW<[SKLWriteResGroup13], (instregex "VPMASKMOVQmr")>;

def SKLWriteResGroup14 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup14], (instregex "FCOM32m")>;
def: InstRW<[SKLWriteResGroup14], (instregex "FCOM64m")>;
def: InstRW<[SKLWriteResGroup14], (instregex "FCOMP32m")>;
def: InstRW<[SKLWriteResGroup14], (instregex "FCOMP64m")>;
def: InstRW<[SKLWriteResGroup14], (instregex "INSERTPSrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_PALIGNR64irm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_PINSRWirmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_PSHUFBrm64")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_PSHUFWmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_PUNPCKHBWirm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_PUNPCKHDQirm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_PUNPCKHWDirm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_PUNPCKLBWirm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_PUNPCKLDQirm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MMX_PUNPCKLWDirm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MOVHPDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MOVHPSrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MOVLPDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "MOVLPSrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PACKSSDWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PACKSSWBrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PACKUSDWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PACKUSWBrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PALIGNRrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PBLENDWrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PINSRBrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PINSRDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PINSRQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PINSRWrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVSXBDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVSXBQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVSXBWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVSXDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVSXWDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVSXWQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVZXBDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVZXBQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVZXBWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVZXDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVZXWDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PMOVZXWQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PSHUFBrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PSHUFDmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PSHUFHWmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PSHUFLWmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PUNPCKHBWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PUNPCKHDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PUNPCKHQDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PUNPCKHWDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PUNPCKLBWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PUNPCKLDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PUNPCKLQDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "PUNPCKLWDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "SHUFPDrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "SHUFPSrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "UNPCKHPDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "UNPCKHPSrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "UNPCKLPDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "UNPCKLPSrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VINSERTPSrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VMOVHPDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VMOVHPSrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VMOVLPDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VMOVLPSrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPACKSSDWYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPACKSSDWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPACKSSWBYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPACKSSWBrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPACKUSDWYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPACKUSDWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPACKUSWBYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPACKUSWBrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPALIGNRYrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPALIGNRrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPBLENDWYrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPBLENDWrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPBROADCASTBYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPBROADCASTBrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPBROADCASTWYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPBROADCASTWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPERMILPDYmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPERMILPDYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPERMILPDmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPERMILPDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPERMILPSYmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPERMILPSYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPERMILPSmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPERMILPSrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPINSRBrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPINSRDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPINSRQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPINSRWrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVSXBDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVSXBQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVSXBWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVSXDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVSXWDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVSXWQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVZXBDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVZXBQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVZXBWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVZXDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVZXWDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPMOVZXWQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPSHUFBYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPSHUFBrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPSHUFDYmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPSHUFDmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPSHUFHWYmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPSHUFHWmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPSHUFLWYmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPSHUFLWmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKHBWYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKHBWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKHDQYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKHDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKHQDQYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKHQDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKHWDYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKHWDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKLBWYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKLBWrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKLDQYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKLDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKLQDQYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKLQDQrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKLWDYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VPUNPCKLWDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VSHUFPDYrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VSHUFPDrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VSHUFPSYrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VSHUFPSrmi")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VUNPCKHPDYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VUNPCKHPDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VUNPCKHPSYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VUNPCKHPSrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VUNPCKLPDYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VUNPCKLPDrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VUNPCKLPSYrm")>;
def: InstRW<[SKLWriteResGroup14], (instregex "VUNPCKLPSrm")>;

def SKLWriteResGroup15 : SchedWriteRes<[SKLPort6,SKLPort23]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup15], (instregex "FARJMP64")>;
def: InstRW<[SKLWriteResGroup15], (instregex "JMP(16|32|64)m")>;

def SKLWriteResGroup16 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup16], (instregex "PABSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PABSDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PABSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PADDSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PADDSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PADDUSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PADDUSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PAVGBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PAVGWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PCMPEQBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PCMPEQDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PCMPEQQrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PCMPEQWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PCMPGTBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PCMPGTDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PCMPGTWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMAXSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMAXSDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMAXSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMAXUBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMAXUDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMAXUWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMINSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMINSDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMINSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMINUBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMINUDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PMINUWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSIGNBrm128")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSIGNDrm128")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSIGNWrm128")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSLLDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSLLQrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSLLWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSRADrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSRAWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSRLDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSRLQrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSRLWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSUBSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSUBSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSUBUSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "PSUBUSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPABSBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPABSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPABSDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPABSDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPABSWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPABSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPADDSBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPADDSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPADDSWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPADDSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPADDUSBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPADDUSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPADDUSWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPADDUSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPAVGBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPAVGBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPAVGWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPAVGWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPEQBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPEQBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPEQDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPEQDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPEQQYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPEQQrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPEQWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPEQWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPGTBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPGTBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPGTDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPGTDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPGTWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPCMPGTWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXSBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXSDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXSDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXSWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXUBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXUBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXUDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXUDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXUWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMAXUWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINSBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINSDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINSDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINSWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINUBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINUBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINUDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINUDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINUWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPMINUWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSIGNBYrm256")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSIGNBrm128")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSIGNDYrm256")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSIGNDrm128")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSIGNWYrm256")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSIGNWrm128")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSLLDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSLLDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSLLQYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSLLQrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSLLVDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSLLVDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSLLVQYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSLLVQrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSLLWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSLLWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRADYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRADrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRAVDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRAVDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRAWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRAWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRLDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRLDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRLQYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRLQrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRLVDYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRLVDrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRLVQYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRLVQrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRLWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSRLWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSUBSBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSUBSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSUBSWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSUBSWrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSUBUSBYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSUBUSBrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSUBUSWYrm")>;
def: InstRW<[SKLWriteResGroup16], (instregex "VPSUBUSWrm")>;

def SKLWriteResGroup17 : SchedWriteRes<[SKLPort23,SKLPort05]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PABSBrm64")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PABSDrm64")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PABSWrm64")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PADDBirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PADDDirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PADDQirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PADDWirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PANDNirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PANDirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PORirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PSIGNBrm64")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PSIGNDrm64")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PSIGNWrm64")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PSUBBirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PSUBDirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PSUBQirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PSUBWirm")>;
def: InstRW<[SKLWriteResGroup17], (instregex "MMX_PXORirm")>;

def SKLWriteResGroup18 : SchedWriteRes<[SKLPort23,SKLPort06]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup18], (instregex "ADC(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "ADC8rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "ADCX32rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "ADCX64rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "ADOX32rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "ADOX64rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "BT(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVAE(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVB(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVE(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVG(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVGE(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVL(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVLE(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVNE(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVNO(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVNP(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVNS(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVO(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVP(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "CMOVS(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "RORX32mi")>;
def: InstRW<[SKLWriteResGroup18], (instregex "RORX64mi")>;
def: InstRW<[SKLWriteResGroup18], (instregex "SARX32rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "SARX64rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "SBB(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "SBB8rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "SHLX32rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "SHLX64rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "SHRX32rm")>;
def: InstRW<[SKLWriteResGroup18], (instregex "SHRX64rm")>;

def SKLWriteResGroup19 : SchedWriteRes<[SKLPort23,SKLPort15]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup19], (instregex "ANDN32rm")>;
def: InstRW<[SKLWriteResGroup19], (instregex "ANDN64rm")>;
def: InstRW<[SKLWriteResGroup19], (instregex "BLSI32rm")>;
def: InstRW<[SKLWriteResGroup19], (instregex "BLSI64rm")>;
def: InstRW<[SKLWriteResGroup19], (instregex "BLSMSK32rm")>;
def: InstRW<[SKLWriteResGroup19], (instregex "BLSMSK64rm")>;
def: InstRW<[SKLWriteResGroup19], (instregex "BLSR32rm")>;
def: InstRW<[SKLWriteResGroup19], (instregex "BLSR64rm")>;
def: InstRW<[SKLWriteResGroup19], (instregex "BZHI32rm")>;
def: InstRW<[SKLWriteResGroup19], (instregex "BZHI64rm")>;
def: InstRW<[SKLWriteResGroup19], (instregex "MOVBE(16|32|64)rm")>;

def SKLWriteResGroup20 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup20], (instregex "ANDNPDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "ANDNPSrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "ANDPDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "ANDPSrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "BLENDPDrmi")>;
def: InstRW<[SKLWriteResGroup20], (instregex "BLENDPSrmi")>;
def: InstRW<[SKLWriteResGroup20], (instregex "ORPDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "ORPSrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PADDBrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PADDDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PADDQrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PADDWrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PANDNrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PANDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PORrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PSUBBrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PSUBDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PSUBQrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PSUBWrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "PXORrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VANDNPDYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VANDNPDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VANDNPSYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VANDNPSrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VANDPDYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VANDPDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VANDPSYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VANDPSrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VBLENDPDYrmi")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VBLENDPDrmi")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VBLENDPSYrmi")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VBLENDPSrmi")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VINSERTF128rm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VINSERTI128rm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VMASKMOVPDYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VMASKMOVPDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VMASKMOVPSYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VMASKMOVPSrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VORPDYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VORPDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VORPSYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VORPSrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPADDBYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPADDBrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPADDDYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPADDDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPADDQYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPADDQrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPADDWYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPADDWrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPANDNYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPANDNrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPANDYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPANDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPBLENDDYrmi")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPBLENDDrmi")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPMASKMOVDYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPMASKMOVDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPMASKMOVQYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPMASKMOVQrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPORYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPORrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPSUBBYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPSUBBrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPSUBDYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPSUBDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPSUBQYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPSUBQrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPSUBWYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPSUBWrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPXORYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VPXORrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VXORPDYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VXORPDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VXORPSYrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "VXORPSrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "XORPDrm")>;
def: InstRW<[SKLWriteResGroup20], (instregex "XORPSrm")>;

def SKLWriteResGroup21 : SchedWriteRes<[SKLPort23,SKLPort0156]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup21], (instregex "ADD(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "ADD8rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "AND(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "AND8rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "CMP(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup21], (instregex "CMP(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup21], (instregex "CMP(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "CMP8mi")>;
def: InstRW<[SKLWriteResGroup21], (instregex "CMP8mr")>;
def: InstRW<[SKLWriteResGroup21], (instregex "CMP8rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "OR(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "OR8rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "POP(16|32|64)r(mr?)")>;
def: InstRW<[SKLWriteResGroup21], (instregex "SUB(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "SUB8rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "TEST(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "TEST8mi")>;
def: InstRW<[SKLWriteResGroup21], (instregex "TEST8rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "XOR(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup21], (instregex "XOR8rm")>;

def SKLWriteResGroup22 : SchedWriteRes<[SKLPort237,SKLPort0156]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup22], (instregex "SFENCE")>;

def SKLWriteResGroup23 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237]> {
  let Latency = 1;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup23], (instregex "EXTRACTPSmr")>;
def: InstRW<[SKLWriteResGroup23], (instregex "PEXTRBmr")>;
def: InstRW<[SKLWriteResGroup23], (instregex "PEXTRDmr")>;
def: InstRW<[SKLWriteResGroup23], (instregex "PEXTRQmr")>;
def: InstRW<[SKLWriteResGroup23], (instregex "PEXTRWmr")>;
def: InstRW<[SKLWriteResGroup23], (instregex "STMXCSR")>;
def: InstRW<[SKLWriteResGroup23], (instregex "VEXTRACTPSmr")>;
def: InstRW<[SKLWriteResGroup23], (instregex "VPEXTRBmr")>;
def: InstRW<[SKLWriteResGroup23], (instregex "VPEXTRDmr")>;
def: InstRW<[SKLWriteResGroup23], (instregex "VPEXTRQmr")>;
def: InstRW<[SKLWriteResGroup23], (instregex "VPEXTRWmr")>;
def: InstRW<[SKLWriteResGroup23], (instregex "VSTMXCSR")>;

def SKLWriteResGroup24 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort237]> {
  let Latency = 1;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup24], (instregex "FNSTCW16m")>;

def SKLWriteResGroup25 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort06]> {
  let Latency = 1;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup25], (instregex "SETAEm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETBm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETEm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETGEm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETGm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETLEm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETLm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETNEm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETNOm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETNPm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETNSm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETOm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETPm")>;
def: InstRW<[SKLWriteResGroup25], (instregex "SETSm")>;

def SKLWriteResGroup26 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort15]> {
  let Latency = 1;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup26], (instregex "MOVBE(16|32|64)mr")>;

def SKLWriteResGroup27 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort0156]> {
  let Latency = 1;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup27], (instregex "PUSH(16|32|64)r(mr?)")>;
def: InstRW<[SKLWriteResGroup27], (instregex "PUSH64i8")>;
def: InstRW<[SKLWriteResGroup27], (instregex "STOSB")>;
def: InstRW<[SKLWriteResGroup27], (instregex "STOSL")>;
def: InstRW<[SKLWriteResGroup27], (instregex "STOSQ")>;
def: InstRW<[SKLWriteResGroup27], (instregex "STOSW")>;

def SKLWriteResGroup28 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 1;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup28], (instregex "BTC(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup28], (instregex "BTR(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup28], (instregex "BTS(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SAR(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SAR(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SAR8m1")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SAR8mi")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SHL(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SHL(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SHL8m1")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SHL8mi")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SHR(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SHR(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SHR8m1")>;
def: InstRW<[SKLWriteResGroup28], (instregex "SHR8mi")>;

def SKLWriteResGroup29 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 1;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup29], (instregex "ADD(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup29], (instregex "ADD(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "ADD8mi")>;
def: InstRW<[SKLWriteResGroup29], (instregex "ADD8mr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "AND(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup29], (instregex "AND(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "AND8mi")>;
def: InstRW<[SKLWriteResGroup29], (instregex "AND8mr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "DEC(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup29], (instregex "DEC8m")>;
def: InstRW<[SKLWriteResGroup29], (instregex "INC(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup29], (instregex "INC8m")>;
def: InstRW<[SKLWriteResGroup29], (instregex "NEG(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup29], (instregex "NEG8m")>;
def: InstRW<[SKLWriteResGroup29], (instregex "NOT(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup29], (instregex "NOT8m")>;
def: InstRW<[SKLWriteResGroup29], (instregex "OR(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup29], (instregex "OR(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "OR8mi")>;
def: InstRW<[SKLWriteResGroup29], (instregex "OR8mr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "POP(16|32|64)rmm")>;
def: InstRW<[SKLWriteResGroup29], (instregex "PUSH(16|32|64)rmm")>;
def: InstRW<[SKLWriteResGroup29], (instregex "SUB(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup29], (instregex "SUB(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "SUB8mi")>;
def: InstRW<[SKLWriteResGroup29], (instregex "SUB8mr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "XOR(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup29], (instregex "XOR(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup29], (instregex "XOR8mi")>;
def: InstRW<[SKLWriteResGroup29], (instregex "XOR8mr")>;

def SKLWriteResGroup31 : SchedWriteRes<[SKLPort0]> {
  let Latency = 2;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup31], (instregex "COMISDrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "COMISSrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "MMX_MOVD64from64rr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "MMX_MOVD64grr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "MMX_PMOVMSKBrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "MOVMSKPDrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "MOVMSKPSrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "MOVPDI2DIrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "MOVPQIto64rr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "PMOVMSKBrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "UCOMISDrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "UCOMISSrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VCOMISDrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VCOMISSrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VMOVMSKPDYrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VMOVMSKPDrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VMOVMSKPSYrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VMOVMSKPSrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VMOVPDI2DIrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VMOVPQIto64rr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VPMOVMSKBYrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VPMOVMSKBrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VTESTPDYrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VTESTPDrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VTESTPSYrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VTESTPSrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VUCOMISDrr")>;
def: InstRW<[SKLWriteResGroup31], (instregex "VUCOMISSrr")>;

def SKLWriteResGroup32 : SchedWriteRes<[SKLPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup32], (instregex "MMX_MOVQ2DQrr")>;
def: InstRW<[SKLWriteResGroup32], (instregex "MMX_PINSRWirri")>;
def: InstRW<[SKLWriteResGroup32], (instregex "PINSRBrr")>;
def: InstRW<[SKLWriteResGroup32], (instregex "PINSRDrr")>;
def: InstRW<[SKLWriteResGroup32], (instregex "PINSRQrr")>;
def: InstRW<[SKLWriteResGroup32], (instregex "PINSRWrri")>;
def: InstRW<[SKLWriteResGroup32], (instregex "VPINSRBrr")>;
def: InstRW<[SKLWriteResGroup32], (instregex "VPINSRDrr")>;
def: InstRW<[SKLWriteResGroup32], (instregex "VPINSRQrr")>;
def: InstRW<[SKLWriteResGroup32], (instregex "VPINSRWrri")>;

def SKLWriteResGroup33 : SchedWriteRes<[SKLPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup33], (instregex "FDECSTP")>;
def: InstRW<[SKLWriteResGroup33], (instregex "MMX_MOVDQ2Qrr")>;

def SKLWriteResGroup34 : SchedWriteRes<[SKLPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup34], (instregex "CMOVA(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup34], (instregex "CMOVBE(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup34], (instregex "ROL(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup34], (instregex "ROL(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup34], (instregex "ROL8r1")>;
def: InstRW<[SKLWriteResGroup34], (instregex "ROL8ri")>;
def: InstRW<[SKLWriteResGroup34], (instregex "ROR(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup34], (instregex "ROR(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup34], (instregex "ROR8r1")>;
def: InstRW<[SKLWriteResGroup34], (instregex "ROR8ri")>;
def: InstRW<[SKLWriteResGroup34], (instregex "SETAr")>;
def: InstRW<[SKLWriteResGroup34], (instregex "SETBEr")>;

def SKLWriteResGroup35 : SchedWriteRes<[SKLPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup35], (instregex "BLENDVPDrr0")>;
def: InstRW<[SKLWriteResGroup35], (instregex "BLENDVPSrr0")>;
def: InstRW<[SKLWriteResGroup35], (instregex "PBLENDVBrr0")>;
def: InstRW<[SKLWriteResGroup35], (instregex "VBLENDVPDYrr")>;
def: InstRW<[SKLWriteResGroup35], (instregex "VBLENDVPDrr")>;
def: InstRW<[SKLWriteResGroup35], (instregex "VBLENDVPSYrr")>;
def: InstRW<[SKLWriteResGroup35], (instregex "VBLENDVPSrr")>;
def: InstRW<[SKLWriteResGroup35], (instregex "VPBLENDVBYrr")>;
def: InstRW<[SKLWriteResGroup35], (instregex "VPBLENDVBrr")>;

def SKLWriteResGroup36 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup36], (instregex "LFENCE")>;
def: InstRW<[SKLWriteResGroup36], (instregex "WAIT")>;
def: InstRW<[SKLWriteResGroup36], (instregex "XGETBV")>;

def SKLWriteResGroup37 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup37], (instregex "COMISDrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "COMISSrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "UCOMISDrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "UCOMISSrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "VCOMISDrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "VCOMISSrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "VTESTPDYrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "VTESTPDrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "VTESTPSYrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "VTESTPSrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "VUCOMISDrm")>;
def: InstRW<[SKLWriteResGroup37], (instregex "VUCOMISSrm")>;

def SKLWriteResGroup38 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup38], (instregex "PSLLDrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "PSLLQrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "PSLLWrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "PSRADrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "PSRAWrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "PSRLDrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "PSRLQrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "PSRLWrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPSLLDrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPSLLQrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPSLLWrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPSRADrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPSRAWrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPSRLDrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPSRLQrr")>;
def: InstRW<[SKLWriteResGroup38], (instregex "VPSRLWrr")>;

def SKLWriteResGroup39 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup39], (instregex "CLFLUSH")>;

def SKLWriteResGroup40 : SchedWriteRes<[SKLPort06,SKLPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup40], (instregex "BEXTR32rr")>;
def: InstRW<[SKLWriteResGroup40], (instregex "BEXTR64rr")>;
def: InstRW<[SKLWriteResGroup40], (instregex "BSWAP(16|32|64)r")>;

def SKLWriteResGroup41 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup41], (instregex "ADC8i8")>;
def: InstRW<[SKLWriteResGroup41], (instregex "ADC8ri")>;
def: InstRW<[SKLWriteResGroup41], (instregex "CWD")>;
def: InstRW<[SKLWriteResGroup41], (instregex "JRCXZ")>;
def: InstRW<[SKLWriteResGroup41], (instregex "SBB8i8")>;
def: InstRW<[SKLWriteResGroup41], (instregex "SBB8ri")>;

def SKLWriteResGroup42 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup42], (instregex "MMX_PACKSSDWirm")>;
def: InstRW<[SKLWriteResGroup42], (instregex "MMX_PACKSSWBirm")>;
def: InstRW<[SKLWriteResGroup42], (instregex "MMX_PACKUSWBirm")>;

def SKLWriteResGroup43 : SchedWriteRes<[SKLPort23,SKLPort06]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup43], (instregex "CMOVA(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup43], (instregex "CMOVBE(16|32|64)rm")>;

def SKLWriteResGroup44 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup44], (instregex "BLENDVPDrm0")>;
def: InstRW<[SKLWriteResGroup44], (instregex "BLENDVPSrm0")>;
def: InstRW<[SKLWriteResGroup44], (instregex "PBLENDVBrm0")>;
def: InstRW<[SKLWriteResGroup44], (instregex "VBLENDVPDYrm")>;
def: InstRW<[SKLWriteResGroup44], (instregex "VBLENDVPDrm")>;
def: InstRW<[SKLWriteResGroup44], (instregex "VBLENDVPSYrm")>;
def: InstRW<[SKLWriteResGroup44], (instregex "VBLENDVPSrm")>;
def: InstRW<[SKLWriteResGroup44], (instregex "VPBLENDVBYrm")>;
def: InstRW<[SKLWriteResGroup44], (instregex "VPBLENDVBrm")>;

def SKLWriteResGroup45 : SchedWriteRes<[SKLPort23,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup45], (instregex "LEAVE64")>;
def: InstRW<[SKLWriteResGroup45], (instregex "SCASB")>;
def: InstRW<[SKLWriteResGroup45], (instregex "SCASL")>;
def: InstRW<[SKLWriteResGroup45], (instregex "SCASQ")>;
def: InstRW<[SKLWriteResGroup45], (instregex "SCASW")>;

def SKLWriteResGroup46 : SchedWriteRes<[SKLPort237,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup46], (instregex "MFENCE")>;

def SKLWriteResGroup47 : SchedWriteRes<[SKLPort0,SKLPort4,SKLPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup47], (instregex "FNSTSWm")>;

def SKLWriteResGroup48 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort05]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup48], (instregex "FLDCW16m")>;

def SKLWriteResGroup49 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup49], (instregex "LDMXCSR")>;
def: InstRW<[SKLWriteResGroup49], (instregex "VLDMXCSR")>;

def SKLWriteResGroup51 : SchedWriteRes<[SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup51], (instregex "LRETQ")>;
def: InstRW<[SKLWriteResGroup51], (instregex "RETQ")>;

def SKLWriteResGroup52 : SchedWriteRes<[SKLPort23,SKLPort06,SKLPort15]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup52], (instregex "BEXTR32rm")>;
def: InstRW<[SKLWriteResGroup52], (instregex "BEXTR64rm")>;

def SKLWriteResGroup53 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort06]> {
  let Latency = 2;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKLWriteResGroup53], (instregex "SETAm")>;
def: InstRW<[SKLWriteResGroup53], (instregex "SETBEm")>;

def SKLWriteResGroup54 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort237,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup54], (instregex "CALL(16|32|64)r")>;

def SKLWriteResGroup55 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup55], (instregex "CALL64pcrel32")>;

def SKLWriteResGroup56 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 2;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup56], (instregex "ROL(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup56], (instregex "ROL(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup56], (instregex "ROL8m1")>;
def: InstRW<[SKLWriteResGroup56], (instregex "ROL8mi")>;
def: InstRW<[SKLWriteResGroup56], (instregex "ROR(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup56], (instregex "ROR(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup56], (instregex "ROR8m1")>;
def: InstRW<[SKLWriteResGroup56], (instregex "ROR8mi")>;

def SKLWriteResGroup57 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup57], (instregex "XADD(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup57], (instregex "XADD8rm")>;

def SKLWriteResGroup58 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[SKLWriteResGroup58], (instregex "CALL(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup58], (instregex "FARCALL64")>;

def SKLWriteResGroup60 : SchedWriteRes<[SKLPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup60], (instregex "BSF(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "BSR(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "IMUL64rr(i8?)")>;
def: InstRW<[SKLWriteResGroup60], (instregex "IMUL8r")>;
def: InstRW<[SKLWriteResGroup60], (instregex "LZCNT(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "MUL8r")>;
def: InstRW<[SKLWriteResGroup60], (instregex "PDEP32rr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "PDEP64rr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "PEXT32rr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "PEXT64rr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "POPCNT(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup60], (instregex "SHLD(16|32|64)rri8")>;
def: InstRW<[SKLWriteResGroup60], (instregex "SHRD(16|32|64)rri8")>;
def: InstRW<[SKLWriteResGroup60], (instregex "TZCNT(16|32|64)rr")>;

def SKLWriteResGroup60_16 : SchedWriteRes<[SKLPort1, SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup60_16], (instregex "IMUL16rr(i8?)")>;

def SKLWriteResGroup60_32 : SchedWriteRes<[SKLPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
}
def: InstRW<[SKLWriteResGroup60_32], (instregex "IMUL32rr(i8?)")>;

def SKLWriteResGroup61 : SchedWriteRes<[SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup61], (instregex "ADD_FPrST0")>;
def: InstRW<[SKLWriteResGroup61], (instregex "ADD_FST0r")>;
def: InstRW<[SKLWriteResGroup61], (instregex "ADD_FrST0")>;
def: InstRW<[SKLWriteResGroup61], (instregex "MMX_PSADBWirr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "PCMPGTQrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "PSADBWrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "SUBR_FPrST0")>;
def: InstRW<[SKLWriteResGroup61], (instregex "SUBR_FST0r")>;
def: InstRW<[SKLWriteResGroup61], (instregex "SUBR_FrST0")>;
def: InstRW<[SKLWriteResGroup61], (instregex "SUB_FPrST0")>;
def: InstRW<[SKLWriteResGroup61], (instregex "SUB_FST0r")>;
def: InstRW<[SKLWriteResGroup61], (instregex "SUB_FrST0")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VBROADCASTSDYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VBROADCASTSSYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VEXTRACTF128rr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VEXTRACTI128rr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VINSERTF128rr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VINSERTI128rr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPBROADCASTBYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPBROADCASTBrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPBROADCASTDYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPBROADCASTQYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPBROADCASTWYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPBROADCASTWrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPCMPGTQYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPCMPGTQrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPERM2F128rr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPERM2I128rr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPERMDYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPERMPDYri")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPERMPSYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPERMQYri")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVSXBDYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVSXBQYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVSXBWYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVSXDQYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVSXWDYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVSXWQYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVZXBDYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVZXBQYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVZXBWYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVZXDQYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVZXWDYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPMOVZXWQYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPSADBWYrr")>;
def: InstRW<[SKLWriteResGroup61], (instregex "VPSADBWrr")>;

def SKLWriteResGroup62 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup62], (instregex "EXTRACTPSrr")>;
def: InstRW<[SKLWriteResGroup62], (instregex "MMX_PEXTRWirri")>;
def: InstRW<[SKLWriteResGroup62], (instregex "PEXTRBrr")>;
def: InstRW<[SKLWriteResGroup62], (instregex "PEXTRDrr")>;
def: InstRW<[SKLWriteResGroup62], (instregex "PEXTRQrr")>;
def: InstRW<[SKLWriteResGroup62], (instregex "PEXTRWri")>;
def: InstRW<[SKLWriteResGroup62], (instregex "PEXTRWrr_REV")>;
def: InstRW<[SKLWriteResGroup62], (instregex "PTESTrr")>;
def: InstRW<[SKLWriteResGroup62], (instregex "VEXTRACTPSrr")>;
def: InstRW<[SKLWriteResGroup62], (instregex "VPEXTRBrr")>;
def: InstRW<[SKLWriteResGroup62], (instregex "VPEXTRDrr")>;
def: InstRW<[SKLWriteResGroup62], (instregex "VPEXTRQrr")>;
def: InstRW<[SKLWriteResGroup62], (instregex "VPEXTRWri")>;
def: InstRW<[SKLWriteResGroup62], (instregex "VPEXTRWrr_REV")>;
def: InstRW<[SKLWriteResGroup62], (instregex "VPTESTYrr")>;
def: InstRW<[SKLWriteResGroup62], (instregex "VPTESTrr")>;

def SKLWriteResGroup63 : SchedWriteRes<[SKLPort0,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup63], (instregex "FNSTSW16r")>;

def SKLWriteResGroup64 : SchedWriteRes<[SKLPort1,SKLPort23]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup64], (instregex "BSF(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup64], (instregex "BSR(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup64], (instregex "IMUL64m")>;
def: InstRW<[SKLWriteResGroup64], (instregex "IMUL(32|64)rm(i8?)")>;
def: InstRW<[SKLWriteResGroup64], (instregex "IMUL8m")>;
def: InstRW<[SKLWriteResGroup64], (instregex "LZCNT(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup64], (instregex "MUL64m")>;
def: InstRW<[SKLWriteResGroup64], (instregex "MUL8m")>;
def: InstRW<[SKLWriteResGroup64], (instregex "PDEP32rm")>;
def: InstRW<[SKLWriteResGroup64], (instregex "PDEP64rm")>;
def: InstRW<[SKLWriteResGroup64], (instregex "PEXT32rm")>;
def: InstRW<[SKLWriteResGroup64], (instregex "PEXT64rm")>;
def: InstRW<[SKLWriteResGroup64], (instregex "POPCNT(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup64], (instregex "TZCNT(16|32|64)rm")>;

def SKLWriteResGroup64_16 : SchedWriteRes<[SKLPort1, SKLPort0156, SKLPort23]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1]; 
}
def: InstRW<[SKLWriteResGroup64_16], (instregex "IMUL16rm(i8?)")>;

def SKLWriteResGroup64_16_2 : SchedWriteRes<[SKLPort1, SKLPort0156, SKLPort23]> {
  let Latency = 3;
  let NumMicroOps = 5;
}
def: InstRW<[SKLWriteResGroup64_16_2], (instregex "IMUL16m")>;
def: InstRW<[SKLWriteResGroup64_16_2], (instregex "MUL16m")>;

def SKLWriteResGroup64_32 : SchedWriteRes<[SKLPort1, SKLPort0156, SKLPort23]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup64_32], (instregex "IMUL32m")>;
def: InstRW<[SKLWriteResGroup64_32], (instregex "MUL32m")>;

def SKLWriteResGroup65 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup65], (instregex "ADD_F32m")>;
def: InstRW<[SKLWriteResGroup65], (instregex "ADD_F64m")>;
def: InstRW<[SKLWriteResGroup65], (instregex "ILD_F16m")>;
def: InstRW<[SKLWriteResGroup65], (instregex "ILD_F32m")>;
def: InstRW<[SKLWriteResGroup65], (instregex "ILD_F64m")>;
def: InstRW<[SKLWriteResGroup65], (instregex "MMX_PSADBWirm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "PCMPGTQrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "PSADBWrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "SUBR_F32m")>;
def: InstRW<[SKLWriteResGroup65], (instregex "SUBR_F64m")>;
def: InstRW<[SKLWriteResGroup65], (instregex "SUB_F32m")>;
def: InstRW<[SKLWriteResGroup65], (instregex "SUB_F64m")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPCMPGTQYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPCMPGTQrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPERM2F128rm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPERM2I128rm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPERMDYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPERMPDYmi")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPERMPSYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPERMQYmi")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVSXBDYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVSXBQYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVSXBWYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVSXDQYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVSXWDYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVSXWQYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVZXBDYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVZXBQYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVZXBWYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVZXDQYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVZXWDYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPMOVZXWQYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPSADBWYrm")>;
def: InstRW<[SKLWriteResGroup65], (instregex "VPSADBWrm")>;

def SKLWriteResGroup66 : SchedWriteRes<[SKLPort06]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKLWriteResGroup66], (instregex "ROL(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup66], (instregex "ROL8rCL")>;
def: InstRW<[SKLWriteResGroup66], (instregex "ROR(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup66], (instregex "ROR8rCL")>;
def: InstRW<[SKLWriteResGroup66], (instregex "SAR(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup66], (instregex "SAR8rCL")>;
def: InstRW<[SKLWriteResGroup66], (instregex "SHL(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup66], (instregex "SHL8rCL")>;
def: InstRW<[SKLWriteResGroup66], (instregex "SHR(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup66], (instregex "SHR8rCL")>;

def SKLWriteResGroup67 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKLWriteResGroup67], (instregex "XADD(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup67], (instregex "XADD8rr")>;
def: InstRW<[SKLWriteResGroup67], (instregex "XCHG8rr")>;

def SKLWriteResGroup68 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup68], (instregex "MMX_PHADDSWrr64")>;
def: InstRW<[SKLWriteResGroup68], (instregex "MMX_PHSUBSWrr64")>;

def SKLWriteResGroup69 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup69], (instregex "PHADDSWrr128")>;
def: InstRW<[SKLWriteResGroup69], (instregex "PHSUBSWrr128")>;
def: InstRW<[SKLWriteResGroup69], (instregex "VPHADDSWrr128")>;
def: InstRW<[SKLWriteResGroup69], (instregex "VPHADDSWrr256")>;
def: InstRW<[SKLWriteResGroup69], (instregex "VPHSUBSWrr128")>;
def: InstRW<[SKLWriteResGroup69], (instregex "VPHSUBSWrr256")>;

def SKLWriteResGroup70 : SchedWriteRes<[SKLPort5,SKLPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup70], (instregex "MMX_PHADDWrr64")>;
def: InstRW<[SKLWriteResGroup70], (instregex "MMX_PHADDrr64")>;
def: InstRW<[SKLWriteResGroup70], (instregex "MMX_PHSUBDrr64")>;
def: InstRW<[SKLWriteResGroup70], (instregex "MMX_PHSUBWrr64")>;

def SKLWriteResGroup71 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup71], (instregex "PHADDDrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PHADDWrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PHSUBDrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "PHSUBWrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPHADDDYrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPHADDDrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPHADDWYrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPHADDWrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPHSUBDYrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPHSUBDrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPHSUBWYrr")>;
def: InstRW<[SKLWriteResGroup71], (instregex "VPHSUBWrr")>;

def SKLWriteResGroup72 : SchedWriteRes<[SKLPort5,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup72], (instregex "MMX_PACKSSDWirr")>;
def: InstRW<[SKLWriteResGroup72], (instregex "MMX_PACKSSWBirr")>;
def: InstRW<[SKLWriteResGroup72], (instregex "MMX_PACKUSWBirr")>;

def SKLWriteResGroup73 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup73], (instregex "CLD")>;

def SKLWriteResGroup74 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup74], (instregex "RCL(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup74], (instregex "RCL(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup74], (instregex "RCL8r1")>;
def: InstRW<[SKLWriteResGroup74], (instregex "RCL8ri")>;
def: InstRW<[SKLWriteResGroup74], (instregex "RCR(16|32|64)r1")>;
def: InstRW<[SKLWriteResGroup74], (instregex "RCR(16|32|64)ri")>;
def: InstRW<[SKLWriteResGroup74], (instregex "RCR8r1")>;
def: InstRW<[SKLWriteResGroup74], (instregex "RCR8ri")>;

def SKLWriteResGroup75 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup75], (instregex "PTESTrm")>;
def: InstRW<[SKLWriteResGroup75], (instregex "VPTESTYrm")>;
def: InstRW<[SKLWriteResGroup75], (instregex "VPTESTrm")>;

def SKLWriteResGroup76 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup76], (instregex "ISTT_FP16m")>;
def: InstRW<[SKLWriteResGroup76], (instregex "ISTT_FP32m")>;
def: InstRW<[SKLWriteResGroup76], (instregex "ISTT_FP64m")>;
def: InstRW<[SKLWriteResGroup76], (instregex "IST_F16m")>;
def: InstRW<[SKLWriteResGroup76], (instregex "IST_F32m")>;
def: InstRW<[SKLWriteResGroup76], (instregex "IST_FP16m")>;
def: InstRW<[SKLWriteResGroup76], (instregex "IST_FP32m")>;
def: InstRW<[SKLWriteResGroup76], (instregex "IST_FP64m")>;

def SKLWriteResGroup77 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKLWriteResGroup77], (instregex "MMX_PHADDSWrm64")>;
def: InstRW<[SKLWriteResGroup77], (instregex "MMX_PHSUBSWrm64")>;

def SKLWriteResGroup78 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup78], (instregex "PHADDSWrm128")>;
def: InstRW<[SKLWriteResGroup78], (instregex "PHSUBSWrm128")>;
def: InstRW<[SKLWriteResGroup78], (instregex "VPHADDSWrm128")>;
def: InstRW<[SKLWriteResGroup78], (instregex "VPHADDSWrm256")>;
def: InstRW<[SKLWriteResGroup78], (instregex "VPHSUBSWrm128")>;
def: InstRW<[SKLWriteResGroup78], (instregex "VPHSUBSWrm256")>;

def SKLWriteResGroup79 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort05]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup79], (instregex "MMX_PHADDWrm64")>;
def: InstRW<[SKLWriteResGroup79], (instregex "MMX_PHADDrm64")>;
def: InstRW<[SKLWriteResGroup79], (instregex "MMX_PHSUBDrm64")>;
def: InstRW<[SKLWriteResGroup79], (instregex "MMX_PHSUBWrm64")>;

def SKLWriteResGroup80 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 3;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup80], (instregex "PHADDDrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "PHADDWrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "PHSUBDrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "PHSUBWrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "VPHADDDYrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "VPHADDDrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "VPHADDWYrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "VPHADDWrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "VPHSUBDYrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "VPHSUBDrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "VPHSUBWYrm")>;
def: InstRW<[SKLWriteResGroup80], (instregex "VPHSUBWrm")>;

def SKLWriteResGroup81 : SchedWriteRes<[SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 3;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKLWriteResGroup81], (instregex "ROR(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup81], (instregex "ROR8mCL")>;

def SKLWriteResGroup82 : SchedWriteRes<[SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup82], (instregex "RCL(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup82], (instregex "RCL(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup82], (instregex "RCL8m1")>;
def: InstRW<[SKLWriteResGroup82], (instregex "RCL8mi")>;
def: InstRW<[SKLWriteResGroup82], (instregex "RCR(16|32|64)m1")>;
def: InstRW<[SKLWriteResGroup82], (instregex "RCR(16|32|64)mi")>;
def: InstRW<[SKLWriteResGroup82], (instregex "RCR8m1")>;
def: InstRW<[SKLWriteResGroup82], (instregex "RCR8mi")>;

def SKLWriteResGroup83 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 3;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKLWriteResGroup83], (instregex "ROL(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup83], (instregex "ROL8mCL")>;
def: InstRW<[SKLWriteResGroup83], (instregex "SAR(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup83], (instregex "SAR8mCL")>;
def: InstRW<[SKLWriteResGroup83], (instregex "SHL(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup83], (instregex "SHL8mCL")>;
def: InstRW<[SKLWriteResGroup83], (instregex "SHR(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup83], (instregex "SHR8mCL")>;

def SKLWriteResGroup84 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKLWriteResGroup84], (instregex "ADC(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup84], (instregex "ADC8mi")>;

def SKLWriteResGroup85 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKLWriteResGroup85], (instregex "ADC(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup85], (instregex "ADC8mr")>;
def: InstRW<[SKLWriteResGroup85], (instregex "CMPXCHG(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "CMPXCHG8rm")>;
def: InstRW<[SKLWriteResGroup85], (instregex "SBB(16|32|64)mi8")>;
def: InstRW<[SKLWriteResGroup85], (instregex "SBB(16|32|64)mr")>;
def: InstRW<[SKLWriteResGroup85], (instregex "SBB8mi")>;
def: InstRW<[SKLWriteResGroup85], (instregex "SBB8mr")>;

def SKLWriteResGroup86 : SchedWriteRes<[SKLPort0]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup86], (instregex "AESDECLASTrr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "AESDECrr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "AESENCLASTrr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "AESENCrr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "MMX_PMADDUBSWrr64")>;
def: InstRW<[SKLWriteResGroup86], (instregex "MMX_PMADDWDirr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "MMX_PMULHRSWrr64")>;
def: InstRW<[SKLWriteResGroup86], (instregex "MMX_PMULHUWirr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "MMX_PMULHWirr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "MMX_PMULLWirr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "MMX_PMULUDQirr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "MUL_FPrST0")>;
def: InstRW<[SKLWriteResGroup86], (instregex "MUL_FST0r")>;
def: InstRW<[SKLWriteResGroup86], (instregex "MUL_FrST0")>;
def: InstRW<[SKLWriteResGroup86], (instregex "RCPPSr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "RCPSSr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "RSQRTPSr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "RSQRTSSr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "VAESDECLASTrr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "VAESDECrr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "VAESENCLASTrr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "VAESENCrr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "VRCPPSYr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "VRCPPSr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "VRCPSSr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "VRSQRTPSYr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "VRSQRTPSr")>;
def: InstRW<[SKLWriteResGroup86], (instregex "VRSQRTSSr")>;

def SKLWriteResGroup87 : SchedWriteRes<[SKLPort01]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup87], (instregex "ADDPDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "ADDPSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "ADDSDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "ADDSSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "ADDSUBPDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "ADDSUBPSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "MULPDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "MULPSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "MULSDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "MULSSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "SUBPDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "SUBPSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "SUBSDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "SUBSSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VADDPDYrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VADDPDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VADDPSYrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VADDPSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VADDSDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VADDSSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VADDSUBPDYrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VADDSUBPDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VADDSUBPSYrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VADDSUBPSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD132PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD132PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD132PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD132PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD132SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD132SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD213PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD213PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD213PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD213PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD213SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD213SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD231PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD231PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD231PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD231PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD231SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADD231SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB132PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB132PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB132PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB132PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB213PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB213PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB213PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB213PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB231PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB231PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB231PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMADDSUB231PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB132PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB132PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB132PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB132PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB132SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB132SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB213PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB213PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB213PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB213PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB213SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB213SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB231PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB231PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB231PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB231PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB231SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUB231SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD132PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD132PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD132PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD132PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD213PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD213PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD213PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD213PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD231PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD231PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD231PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFMSUBADD231PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD132PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD132PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD132PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD132PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD132SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD132SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD213PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD213PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD213PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD213PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD213SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD213SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD231PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD231PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD231PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD231PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD231SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMADD231SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB132PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB132PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB132PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB132PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB132SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB132SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB213PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB213PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB213PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB213PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB213SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB213SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB231PDYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB231PDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB231PSYr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB231PSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB231SDr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VFNMSUB231SSr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VMULPDYrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VMULPDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VMULPSYrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VMULPSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VMULSDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VMULSSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VSUBPDYrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VSUBPDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VSUBPSYrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VSUBPSrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VSUBSDrr")>;
def: InstRW<[SKLWriteResGroup87], (instregex "VSUBSSrr")>;

def SKLWriteResGroup89 : SchedWriteRes<[SKLPort015]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup89], (instregex "CMPPDrri")>;
def: InstRW<[SKLWriteResGroup89], (instregex "CMPPSrri")>;
def: InstRW<[SKLWriteResGroup89], (instregex "CMPSSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "CVTDQ2PSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "CVTPS2DQrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "CVTTPS2DQrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "MAXPDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "MAXPSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "MAXSDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "MAXSSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "MINPDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "MINPSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "MINSDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "MINSSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "PHMINPOSUWrr128")>;
def: InstRW<[SKLWriteResGroup89], (instregex "PMADDUBSWrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "PMADDWDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "PMULDQrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "PMULHRSWrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "PMULHUWrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "PMULHWrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "PMULLWrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "PMULUDQrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCMPPDYrri")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCMPPDrri")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCMPPSYrri")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCMPPSrri")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCMPSDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCMPSSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTDQ2PSYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTDQ2PSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTPS2DQYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTPS2DQrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTTPS2DQYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTTPS2DQrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMAXPDYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMAXPDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMAXPSYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMAXPSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMAXSDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMAXSSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMINPDYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMINPDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMINPSYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMINPSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMINSDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VMINSSrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPHMINPOSUWrr128")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMADDUBSWYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMADDUBSWrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMADDWDYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMADDWDrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULDQYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULDQrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULHRSWYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULHRSWrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULHUWYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULHUWrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULHWYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULHWrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULLWYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULLWrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULUDQYrr")>;
def: InstRW<[SKLWriteResGroup89], (instregex "VPMULUDQrr")>;

def SKLWriteResGroup90 : SchedWriteRes<[SKLPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup90], (instregex "MPSADBWrri")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VMPSADBWYrri")>;
def: InstRW<[SKLWriteResGroup90], (instregex "VMPSADBWrri")>;

def SKLWriteResGroup91 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup91], (instregex "AESDECLASTrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "AESDECrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "AESENCLASTrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "AESENCrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "MMX_CVTPI2PSirm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "MMX_PMADDUBSWrm64")>;
def: InstRW<[SKLWriteResGroup91], (instregex "MMX_PMADDWDirm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "MMX_PMULHRSWrm64")>;
def: InstRW<[SKLWriteResGroup91], (instregex "MMX_PMULHUWirm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "MMX_PMULHWirm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "MMX_PMULLWirm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "MMX_PMULUDQirm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "MUL_F32m")>;
def: InstRW<[SKLWriteResGroup91], (instregex "MUL_F64m")>;
def: InstRW<[SKLWriteResGroup91], (instregex "RCPPSm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "RCPSSm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "RSQRTPSm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "RSQRTSSm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VAESDECLASTrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VAESDECrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VAESENCLASTrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VAESENCrm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VRCPPSYm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VRCPPSm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VRCPSSm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VRSQRTPSYm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VRSQRTPSm")>;
def: InstRW<[SKLWriteResGroup91], (instregex "VRSQRTSSm")>;

def SKLWriteResGroup92 : SchedWriteRes<[SKLPort1,SKLPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup92], (instregex "IMUL64r")>;
def: InstRW<[SKLWriteResGroup92], (instregex "MUL64r")>;
def: InstRW<[SKLWriteResGroup92], (instregex "MULX64rr")>;

def SKLWriteResGroup92_16 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;  
}
def: InstRW<[SKLWriteResGroup92_16], (instregex "IMUL16r")>;
def: InstRW<[SKLWriteResGroup92_16], (instregex "MUL16r")>;

def SKLWriteResGroup93 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup93], (instregex "VPSLLDYrr")>;
def: InstRW<[SKLWriteResGroup93], (instregex "VPSLLQYrr")>;
def: InstRW<[SKLWriteResGroup93], (instregex "VPSLLWYrr")>;
def: InstRW<[SKLWriteResGroup93], (instregex "VPSRADYrr")>;
def: InstRW<[SKLWriteResGroup93], (instregex "VPSRAWYrr")>;
def: InstRW<[SKLWriteResGroup93], (instregex "VPSRLDYrr")>;
def: InstRW<[SKLWriteResGroup93], (instregex "VPSRLQYrr")>;
def: InstRW<[SKLWriteResGroup93], (instregex "VPSRLWYrr")>;

def SKLWriteResGroup94 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup94], (instregex "ADDPDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "ADDPSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "ADDSDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "ADDSSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "ADDSUBPDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "ADDSUBPSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "MULPDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "MULPSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "MULSDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "MULSSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "SUBPDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "SUBPSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "SUBSDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "SUBSSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VADDPDYrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VADDPDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VADDPSYrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VADDPSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VADDSDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VADDSSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VADDSUBPDYrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VADDSUBPDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VADDSUBPSYrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VADDSUBPSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD132PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD132PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD132PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD132PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD132SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD132SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD213PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD213PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD213PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD213PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD213SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD213SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD231PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD231PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD231PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD231PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD231SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADD231SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB132PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB132PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB132PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB132PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB213PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB213PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB213PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB213PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB231PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB231PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB231PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMADDSUB231PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB132PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB132PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB132PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB132PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB132SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB132SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB213PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB213PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB213PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB213PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB213SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB213SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB231PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB231PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB231PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB231PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB231SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUB231SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD132PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD132PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD132PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD132PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD213PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD213PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD213PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD213PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD231PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD231PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD231PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFMSUBADD231PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD132PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD132PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD132PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD132PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD132SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD132SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD213PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD213PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD213PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD213PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD213SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD213SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD231PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD231PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD231PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD231PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD231SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMADD231SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB132PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB132PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB132PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB132PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB132SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB132SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB213PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB213PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB213PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB213PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB213SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB213SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB231PDYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB231PDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB231PSYm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB231PSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB231SDm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VFNMSUB231SSm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VMULPDYrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VMULPDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VMULPSYrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VMULPSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VMULSDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VMULSSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VSUBPDYrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VSUBPDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VSUBPSYrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VSUBPSrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VSUBSDrm")>;
def: InstRW<[SKLWriteResGroup94], (instregex "VSUBSSrm")>;

def SKLWriteResGroup96 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup96], (instregex "CMPPDrmi")>;
def: InstRW<[SKLWriteResGroup96], (instregex "CMPPSrmi")>;
def: InstRW<[SKLWriteResGroup96], (instregex "CMPSSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "CVTDQ2PSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "CVTPS2DQrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "CVTPS2PDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "CVTSS2SDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "CVTTPS2DQrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "MAXPDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "MAXPSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "MAXSDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "MAXSSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "MINPDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "MINPSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "MINSDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "MINSSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "MMX_CVTPS2PIirm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "MMX_CVTTPS2PIirm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "PHMINPOSUWrm128")>;
def: InstRW<[SKLWriteResGroup96], (instregex "PMADDUBSWrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "PMADDWDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "PMULDQrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "PMULHRSWrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "PMULHUWrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "PMULHWrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "PMULLWrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "PMULUDQrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCMPPDYrmi")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCMPPDrmi")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCMPPSYrmi")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCMPPSrmi")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCMPSDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCMPSSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTDQ2PSYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTDQ2PSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTPH2PSYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTPH2PSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTPS2DQYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTPS2DQrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTPS2PDYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTPS2PDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTSS2SDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTTPS2DQYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VCVTTPS2DQrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMAXPDYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMAXPDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMAXPSYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMAXPSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMAXSDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMAXSSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMINPDYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMINPDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMINPSYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMINPSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMINSDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VMINSSrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPHMINPOSUWrm128")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMADDUBSWYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMADDUBSWrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMADDWDYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMADDWDrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULDQYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULDQrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULHRSWYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULHRSWrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULHUWYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULHUWrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULHWYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULHWrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULLWYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULLWrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULUDQYrm")>;
def: InstRW<[SKLWriteResGroup96], (instregex "VPMULUDQrm")>;

def SKLWriteResGroup97 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup97], (instregex "FICOM16m")>;
def: InstRW<[SKLWriteResGroup97], (instregex "FICOM32m")>;
def: InstRW<[SKLWriteResGroup97], (instregex "FICOMP16m")>;
def: InstRW<[SKLWriteResGroup97], (instregex "FICOMP32m")>;
def: InstRW<[SKLWriteResGroup97], (instregex "MPSADBWrmi")>;
def: InstRW<[SKLWriteResGroup97], (instregex "VMPSADBWYrmi")>;
def: InstRW<[SKLWriteResGroup97], (instregex "VMPSADBWrmi")>;

def SKLWriteResGroup98 : SchedWriteRes<[SKLPort1,SKLPort5,SKLPort23]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup98], (instregex "MULX64rm")>;

def SKLWriteResGroup100 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[SKLWriteResGroup100], (instregex "FNCLEX")>;

def SKLWriteResGroup101 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup101], (instregex "PAUSE")>;

def SKLWriteResGroup102 : SchedWriteRes<[SKLPort015,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup102], (instregex "VZEROUPPER")>;

def SKLWriteResGroup103 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKLWriteResGroup103], (instregex "LAR(16|32|64)rr")>;

def SKLWriteResGroup105 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup105], (instregex "SHLD(16|32|64)mri8")>;
def: InstRW<[SKLWriteResGroup105], (instregex "SHRD(16|32|64)mri8")>;

def SKLWriteResGroup106 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKLWriteResGroup106], (instregex "LAR(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup106], (instregex "LSL(16|32|64)rm")>;

def SKLWriteResGroup107 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKLWriteResGroup107], (instregex "PUSHF16")>;
def: InstRW<[SKLWriteResGroup107], (instregex "PUSHF64")>;

def SKLWriteResGroup109 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup109], (instregex "CVTDQ2PDrr")>;
def: InstRW<[SKLWriteResGroup109], (instregex "MMX_CVTPI2PDirr")>;
def: InstRW<[SKLWriteResGroup109], (instregex "VCVTDQ2PDrr")>;

def SKLWriteResGroup110 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup110], (instregex "CVTPD2DQrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "CVTPD2PSrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "CVTPS2PDrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "CVTSD2SSrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "CVTSI2SD64rr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "CVTSI2SDrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "CVTSI2SSrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "CVTSS2SDrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "CVTTPD2DQrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "MMX_CVTPD2PIirr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "MMX_CVTPS2PIirr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "MMX_CVTTPD2PIirr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "MMX_CVTTPS2PIirr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTPD2DQrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTPD2PSrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTPH2PSrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTPS2PDrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTPS2PHrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTSD2SSrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTSI2SD64rr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTSI2SDrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTSI2SSrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTSS2SDrr")>;
def: InstRW<[SKLWriteResGroup110], (instregex "VCVTTPD2DQrr")>;

def SKLWriteResGroup113 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup113], (instregex "CVTDQ2PDrm")>;
def: InstRW<[SKLWriteResGroup113], (instregex "MMX_CVTPI2PDirm")>;
def: InstRW<[SKLWriteResGroup113], (instregex "VCVTDQ2PDrm")>;

def SKLWriteResGroup114 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup114], (instregex "STR(16|32|64)r")>;

def SKLWriteResGroup115 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup115], (instregex "IMUL32r")>;
def: InstRW<[SKLWriteResGroup115], (instregex "MUL32r")>;
def: InstRW<[SKLWriteResGroup115], (instregex "MULX32rr")>;

def SKLWriteResGroup116 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup116], (instregex "CVTPD2DQrm")>;
def: InstRW<[SKLWriteResGroup116], (instregex "CVTPD2PSrm")>;
def: InstRW<[SKLWriteResGroup116], (instregex "CVTSD2SSrm")>;
def: InstRW<[SKLWriteResGroup116], (instregex "CVTTPD2DQrm")>;
def: InstRW<[SKLWriteResGroup116], (instregex "MMX_CVTPD2PIirm")>;
def: InstRW<[SKLWriteResGroup116], (instregex "MMX_CVTTPD2PIirm")>;
def: InstRW<[SKLWriteResGroup116], (instregex "VCVTSD2SSrm")>;

def SKLWriteResGroup118 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort06,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup118], (instregex "MULX32rm")>;

def SKLWriteResGroup119 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237,SKLPort015]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup119], (instregex "VCVTPS2PHmr")>;

def SKLWriteResGroup120 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [1,4];
}
def: InstRW<[SKLWriteResGroup120], (instregex "XSETBV")>;

def SKLWriteResGroup121 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[SKLWriteResGroup121], (instregex "CMPXCHG(16|32|64)rr")>;
def: InstRW<[SKLWriteResGroup121], (instregex "CMPXCHG8rr")>;

def SKLWriteResGroup122 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,3];
}
def: InstRW<[SKLWriteResGroup122], (instregex "ADD8mi")>;
def: InstRW<[SKLWriteResGroup122], (instregex "AND8mi")>;
def: InstRW<[SKLWriteResGroup122], (instregex "OR8mi")>;
def: InstRW<[SKLWriteResGroup122], (instregex "SUB8mi")>;
def: InstRW<[SKLWriteResGroup122], (instregex "XCHG(16|32|64)rm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "XCHG8rm")>;
def: InstRW<[SKLWriteResGroup122], (instregex "XOR8mi")>;

def SKLWriteResGroup123 : SchedWriteRes<[SKLPort5]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup123], (instregex "PCLMULQDQrr")>;
def: InstRW<[SKLWriteResGroup123], (instregex "VPCLMULQDQrr")>;

def SKLWriteResGroup124 : SchedWriteRes<[SKLPort0]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup124], (instregex "MMX_CVTPI2PSirr")>;

def SKLWriteResGroup125 : SchedWriteRes<[SKLPort0,SKLPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup125], (instregex "CVTSD2SI64rr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "CVTSD2SIrr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "CVTSS2SI64rr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "CVTSS2SIrr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "CVTTSD2SI64rr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "CVTTSD2SIrr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "VCVTSD2SI64rr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "VCVTSD2SIrr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "VCVTSS2SI64rr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "VCVTSS2SIrr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "VCVTTSD2SI64rr")>;
def: InstRW<[SKLWriteResGroup125], (instregex "VCVTTSD2SIrr")>;

def SKLWriteResGroup126 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup126], (instregex "PCLMULQDQrm")>;
def: InstRW<[SKLWriteResGroup126], (instregex "VPCLMULQDQrm")>;

def SKLWriteResGroup127 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup127], (instregex "HADDPDrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "HADDPSrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "HSUBPDrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "HSUBPSrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "VHADDPDYrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "VHADDPDrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "VHADDPSYrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "VHADDPSrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "VHSUBPDYrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "VHSUBPDrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "VHSUBPSYrr")>;
def: InstRW<[SKLWriteResGroup127], (instregex "VHSUBPSrr")>;

def SKLWriteResGroup128 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup128], (instregex "ADD_FI16m")>;
def: InstRW<[SKLWriteResGroup128], (instregex "ADD_FI32m")>;
def: InstRW<[SKLWriteResGroup128], (instregex "SUBR_FI16m")>;
def: InstRW<[SKLWriteResGroup128], (instregex "SUBR_FI32m")>;
def: InstRW<[SKLWriteResGroup128], (instregex "SUB_FI16m")>;
def: InstRW<[SKLWriteResGroup128], (instregex "SUB_FI32m")>;

def SKLWriteResGroup129 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup129], (instregex "CVTSI2SS64rr")>;
def: InstRW<[SKLWriteResGroup129], (instregex "VCVTSI2SS64rr")>;

def SKLWriteResGroup130 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort015]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup130], (instregex "CVTSD2SI64rm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "CVTSD2SIrm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "CVTSS2SI64rm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "CVTSS2SIrm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "CVTTSD2SI64rm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "CVTTSD2SIrm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "CVTTSS2SIrm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "VCVTSD2SI64rm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "VCVTSD2SIrm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "VCVTSS2SI64rm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "VCVTSS2SIrm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "VCVTTSD2SI64rm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "VCVTTSD2SIrm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "VCVTTSS2SI64rm")>;
def: InstRW<[SKLWriteResGroup130], (instregex "VCVTTSS2SIrm")>;

def SKLWriteResGroup131 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKLWriteResGroup131], (instregex "SHLD(16|32|64)rrCL")>;
def: InstRW<[SKLWriteResGroup131], (instregex "SHRD(16|32|64)rrCL")>;

def SKLWriteResGroup133 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup133], (instregex "HADDPDrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "HADDPSrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "HSUBPDrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "HSUBPSrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VHADDPDYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VHADDPDrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VHADDPSYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VHADDPSrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VHSUBPDYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VHSUBPDrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VHSUBPSYrm")>;
def: InstRW<[SKLWriteResGroup133], (instregex "VHSUBPSrm")>;

def SKLWriteResGroup134 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup134], (instregex "SLDT(16|32|64)r")>;

def SKLWriteResGroup136 : SchedWriteRes<[SKLPort6,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,5];
}
def: InstRW<[SKLWriteResGroup136], (instregex "STD")>;

def SKLWriteResGroup137 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKLWriteResGroup137], (instregex "SHLD(16|32|64)mrCL")>;
def: InstRW<[SKLWriteResGroup137], (instregex "SHRD(16|32|64)mrCL")>;

def SKLWriteResGroup142 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup142], (instregex "VCVTDQ2PDYrr")>;

def SKLWriteResGroup143 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup143], (instregex "VCVTPD2DQYrr")>;
def: InstRW<[SKLWriteResGroup143], (instregex "VCVTPD2PSYrr")>;
def: InstRW<[SKLWriteResGroup143], (instregex "VCVTPH2PSYrr")>;
def: InstRW<[SKLWriteResGroup143], (instregex "VCVTPS2PDYrr")>;
def: InstRW<[SKLWriteResGroup143], (instregex "VCVTPS2PHYrr")>;
def: InstRW<[SKLWriteResGroup143], (instregex "VCVTTPD2DQYrr")>;

def SKLWriteResGroup145 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup145], (instregex "MUL_FI16m")>;
def: InstRW<[SKLWriteResGroup145], (instregex "MUL_FI32m")>;
def: InstRW<[SKLWriteResGroup145], (instregex "VCVTDQ2PDYrm")>;

def SKLWriteResGroup146 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup146], (instregex "CVTTSS2SI64rr")>;
def: InstRW<[SKLWriteResGroup146], (instregex "CVTTSS2SIrr")>;
def: InstRW<[SKLWriteResGroup146], (instregex "VCVTTSS2SI64rr")>;
def: InstRW<[SKLWriteResGroup146], (instregex "VCVTTSS2SIrr")>;

def SKLWriteResGroup149 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup149], (instregex "CVTTSS2SI64rm")>;

def SKLWriteResGroup150 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237,SKLPort015]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup150], (instregex "VCVTPS2PHYmr")>;

def SKLWriteResGroup151 : SchedWriteRes<[SKLPort6,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,1,2];
}
def: InstRW<[SKLWriteResGroup151], (instregex "LOOP")>;

def SKLWriteResGroup156 : SchedWriteRes<[SKLPort0]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup156], (instregex "AESIMCrr")>;
def: InstRW<[SKLWriteResGroup156], (instregex "VAESIMCrr")>;

def SKLWriteResGroup157 : SchedWriteRes<[SKLPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup157], (instregex "PMULLDrr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "ROUNDPDr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "ROUNDPSr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "ROUNDSDr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "ROUNDSSr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "VPMULLDYrr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "VPMULLDrr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "VROUNDPDr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "VROUNDPSr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "VROUNDSDr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "VROUNDSSr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "VROUNDYPDr")>;
def: InstRW<[SKLWriteResGroup157], (instregex "VROUNDYPSr")>;

def SKLWriteResGroup160 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup160], (instregex "AESIMCrm")>;
def: InstRW<[SKLWriteResGroup160], (instregex "VAESIMCrm")>;

def SKLWriteResGroup161 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup161], (instregex "PMULLDrm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "ROUNDPDm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "ROUNDPSm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "ROUNDSDm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "ROUNDSSm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "VPMULLDYrm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "VPMULLDrm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "VROUNDPDm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "VROUNDPSm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "VROUNDSDm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "VROUNDSSm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "VROUNDYPDm")>;
def: InstRW<[SKLWriteResGroup161], (instregex "VROUNDYPSm")>;

def SKLWriteResGroup165 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup165], (instregex "DPPDrri")>;
def: InstRW<[SKLWriteResGroup165], (instregex "VDPPDrri")>;

def SKLWriteResGroup167 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKLWriteResGroup167], (instregex "DPPDrmi")>;
def: InstRW<[SKLWriteResGroup167], (instregex "VDPPDrmi")>;

def SKLWriteResGroup169 : SchedWriteRes<[SKLPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKLWriteResGroup169], (instregex "PCMPISTRIrr")>;
def: InstRW<[SKLWriteResGroup169], (instregex "PCMPISTRM128rr")>;
def: InstRW<[SKLWriteResGroup169], (instregex "VPCMPISTRIrr")>;
def: InstRW<[SKLWriteResGroup169], (instregex "VPCMPISTRM128rr")>;

def SKLWriteResGroup170 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}
def: InstRW<[SKLWriteResGroup170], (instregex "PCMPISTRIrm")>;
def: InstRW<[SKLWriteResGroup170], (instregex "PCMPISTRM128rm")>;
def: InstRW<[SKLWriteResGroup170], (instregex "VPCMPISTRIrm")>;
def: InstRW<[SKLWriteResGroup170], (instregex "VPCMPISTRM128rm")>;

def SKLWriteResGroup171 : SchedWriteRes<[SKLPort05,SKLPort0156]> {
  let Latency = 10;
  let NumMicroOps = 10;
  let ResourceCycles = [9,1];
}
def: InstRW<[SKLWriteResGroup171], (instregex "MMX_EMMS")>;

def SKLWriteResGroup172 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 10;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,5,1,1];
}
def: InstRW<[SKLWriteResGroup172], (instregex "RCL(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup172], (instregex "RCL8mCL")>;

def SKLWriteResGroup173 : SchedWriteRes<[SKLPort0]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup173], (instregex "DIVPSrr")>;
def: InstRW<[SKLWriteResGroup173], (instregex "DIVSSrr")>;
def: InstRW<[SKLWriteResGroup173], (instregex "VDIVPSYrr")>;
def: InstRW<[SKLWriteResGroup173], (instregex "VDIVPSrr")>;
def: InstRW<[SKLWriteResGroup173], (instregex "VDIVSSrr")>;

def SKLWriteResGroup174 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup174], (instregex "DIVPSrm")>;
def: InstRW<[SKLWriteResGroup174], (instregex "DIVSSrm")>;
def: InstRW<[SKLWriteResGroup174], (instregex "VDIVPSYrm")>;
def: InstRW<[SKLWriteResGroup174], (instregex "VDIVPSrm")>;
def: InstRW<[SKLWriteResGroup174], (instregex "VDIVSSrm")>;

def SKLWriteResGroup175 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,3,2];
}
def: InstRW<[SKLWriteResGroup175], (instregex "RCL(16|32|64)rCL")>;
def: InstRW<[SKLWriteResGroup175], (instregex "RCR(16|32|64)rCL")>;

def SKLWriteResGroup176 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 9;
  let ResourceCycles = [1,5,1,2];
}
def: InstRW<[SKLWriteResGroup176], (instregex "RCL8rCL")>;

def SKLWriteResGroup177 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 11;
  let ResourceCycles = [2,9];
}
def: InstRW<[SKLWriteResGroup177], (instregex "LOOPE")>;
def: InstRW<[SKLWriteResGroup177], (instregex "LOOPNE")>;

def SKLWriteResGroup178 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 14;
  let ResourceCycles = [1,1,1,4,2,5];
}
def: InstRW<[SKLWriteResGroup178], (instregex "CMPXCHG8B")>;

def SKLWriteResGroup179 : SchedWriteRes<[SKLPort0]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup179], (instregex "VSQRTPSYr")>;
def: InstRW<[SKLWriteResGroup179], (instregex "VSQRTPSr")>;
def: InstRW<[SKLWriteResGroup179], (instregex "VSQRTSSr")>;

def SKLWriteResGroup180 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup180], (instregex "VSQRTPSYm")>;
def: InstRW<[SKLWriteResGroup180], (instregex "VSQRTPSm")>;
def: InstRW<[SKLWriteResGroup180], (instregex "VSQRTSSm")>;

def SKLWriteResGroup181 : SchedWriteRes<[SKLPort0]> {
  let Latency = 13;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup181], (instregex "SQRTPSr")>;
def: InstRW<[SKLWriteResGroup181], (instregex "SQRTSSr")>;

def SKLWriteResGroup182 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 13;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup182], (instregex "SQRTPSm")>;
def: InstRW<[SKLWriteResGroup182], (instregex "SQRTSSm")>;

def SKLWriteResGroup187 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup187], (instregex "DPPSrri")>;
def: InstRW<[SKLWriteResGroup187], (instregex "VDPPSYrri")>;
def: InstRW<[SKLWriteResGroup187], (instregex "VDPPSrri")>;

def SKLWriteResGroup188 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 13;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKLWriteResGroup188], (instregex "DPPSrmi")>;
def: InstRW<[SKLWriteResGroup188], (instregex "VDPPSYrmi")>;
def: InstRW<[SKLWriteResGroup188], (instregex "VDPPSrmi")>;

def SKLWriteResGroup189 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 13;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,4,1,2];
}
def: InstRW<[SKLWriteResGroup189], (instregex "RCR(16|32|64)mCL")>;
def: InstRW<[SKLWriteResGroup189], (instregex "RCR8mCL")>;

def SKLWriteResGroup190 : SchedWriteRes<[SKLPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup190], (instregex "DIVPDrr")>;
def: InstRW<[SKLWriteResGroup190], (instregex "DIVSDrr")>;
def: InstRW<[SKLWriteResGroup190], (instregex "VDIVPDYrr")>;
def: InstRW<[SKLWriteResGroup190], (instregex "VDIVPDrr")>;
def: InstRW<[SKLWriteResGroup190], (instregex "VDIVSDrr")>;

def SKLWriteResGroup191 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 14;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup191], (instregex "DIVPDrm")>;
def: InstRW<[SKLWriteResGroup191], (instregex "DIVSDrm")>;
def: InstRW<[SKLWriteResGroup191], (instregex "VDIVPDYrm")>;
def: InstRW<[SKLWriteResGroup191], (instregex "VDIVPDrm")>;
def: InstRW<[SKLWriteResGroup191], (instregex "VDIVSDrm")>;

def SKLWriteResGroup192 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 14;
  let NumMicroOps = 10;
  let ResourceCycles = [2,4,1,3];
}
def: InstRW<[SKLWriteResGroup192], (instregex "RCR8rCL")>;

def SKLWriteResGroup193 : SchedWriteRes<[SKLPort0]> {
  let Latency = 15;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup193], (instregex "DIVR_FPrST0")>;
def: InstRW<[SKLWriteResGroup193], (instregex "DIVR_FST0r")>;
def: InstRW<[SKLWriteResGroup193], (instregex "DIVR_FrST0")>;

def SKLWriteResGroup194 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 15;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup194], (instregex "DIV_F32m")>;
def: InstRW<[SKLWriteResGroup194], (instregex "DIV_F64m")>;

def SKLWriteResGroup195 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 15;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,1,2];
}
def: InstRW<[SKLWriteResGroup195], (instregex "INSB")>;
def: InstRW<[SKLWriteResGroup195], (instregex "INSL")>;
def: InstRW<[SKLWriteResGroup195], (instregex "INSW")>;

def SKLWriteResGroup196 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 16;
  let NumMicroOps = 16;
  let ResourceCycles = [16];
}
def: InstRW<[SKLWriteResGroup196], (instregex "VZEROALL")>;

def SKLWriteResGroup197 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort5,SKLPort6,SKLPort05,SKLPort0156]> {
  let Latency = 17;
  let NumMicroOps = 15;
  let ResourceCycles = [2,1,2,4,2,4];
}
def: InstRW<[SKLWriteResGroup197], (instregex "XCH_F")>;

def SKLWriteResGroup198 : SchedWriteRes<[SKLPort0]> {
  let Latency = 18;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup198], (instregex "VSQRTPDYr")>;
def: InstRW<[SKLWriteResGroup198], (instregex "VSQRTPDr")>;
def: InstRW<[SKLWriteResGroup198], (instregex "VSQRTSDr")>;

def SKLWriteResGroup199 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup199], (instregex "VSQRTPDYm")>;
def: InstRW<[SKLWriteResGroup199], (instregex "VSQRTPDm")>;
def: InstRW<[SKLWriteResGroup199], (instregex "VSQRTSDm")>;

def SKLWriteResGroup200 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 18;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup200], (instregex "DIV_FI16m")>;
def: InstRW<[SKLWriteResGroup200], (instregex "DIV_FI32m")>;

def SKLWriteResGroup201 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def: InstRW<[SKLWriteResGroup201], (instregex "PCMPESTRIrr")>;
def: InstRW<[SKLWriteResGroup201], (instregex "VPCMPESTRIrr")>;

def SKLWriteResGroup202 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[SKLWriteResGroup202], (instregex "CPUID")>;
def: InstRW<[SKLWriteResGroup202], (instregex "RDTSC")>;

def SKLWriteResGroup203 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[SKLWriteResGroup203], (instregex "PCMPESTRIrm")>;
def: InstRW<[SKLWriteResGroup203], (instregex "VPCMPESTRIrm")>;

def SKLWriteResGroup204 : SchedWriteRes<[SKLPort0,SKLPort4,SKLPort5,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 19;
  let ResourceCycles = [2,1,4,1,1,4,6];
}
def: InstRW<[SKLWriteResGroup204], (instregex "CMPXCHG16B")>;

def SKLWriteResGroup205 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort015,SKLPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def: InstRW<[SKLWriteResGroup205], (instregex "PCMPESTRM128rr")>;
def: InstRW<[SKLWriteResGroup205], (instregex "VPCMPESTRM128rr")>;

def SKLWriteResGroup206 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort015,SKLPort0156]> {
  let Latency = 19;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}
def: InstRW<[SKLWriteResGroup206], (instregex "PCMPESTRM128rm")>;
def: InstRW<[SKLWriteResGroup206], (instregex "VPCMPESTRM128rm")>;

def SKLWriteResGroup207 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 19;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,1,1];
}
def: InstRW<[SKLWriteResGroup207], (instregex "AESKEYGENASSIST128rm")>;
def: InstRW<[SKLWriteResGroup207], (instregex "VAESKEYGENASSIST128rm")>;

def SKLWriteResGroup208 : SchedWriteRes<[SKLPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup208], (instregex "DIV_FPrST0")>;
def: InstRW<[SKLWriteResGroup208], (instregex "DIV_FST0r")>;
def: InstRW<[SKLWriteResGroup208], (instregex "DIV_FrST0")>;
def: InstRW<[SKLWriteResGroup208], (instregex "SQRTPDr")>;
def: InstRW<[SKLWriteResGroup208], (instregex "SQRTSDr")>;

def SKLWriteResGroup209 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup209], (instregex "DIVR_F32m")>;
def: InstRW<[SKLWriteResGroup209], (instregex "DIVR_F64m")>;
def: InstRW<[SKLWriteResGroup209], (instregex "SQRTPDm")>;
def: InstRW<[SKLWriteResGroup209], (instregex "SQRTSDm")>;

def SKLWriteResGroup210 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort0156]> {
  let Latency = 20;
  let NumMicroOps = 10;
  let ResourceCycles = [1,2,7];
}
def: InstRW<[SKLWriteResGroup210], (instregex "MWAITrr")>;

def SKLWriteResGroup211 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort015]> {
  let Latency = 20;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,2];
}
def: InstRW<[SKLWriteResGroup211], (instregex "AESKEYGENASSIST128rr")>;
def: InstRW<[SKLWriteResGroup211], (instregex "VAESKEYGENASSIST128rr")>;

def SKLWriteResGroup212 : SchedWriteRes<[SKLPort0, SKLPort23, SKLPort5, SKLPort015]> {
  let Latency = 17;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKLWriteResGroup212], (instregex "VGATHERDPSrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VGATHERDPDrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VGATHERQPDrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VGATHERQPSrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VPGATHERDDrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VPGATHERDQrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VPGATHERQDrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VPGATHERQQrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VPGATHERDDrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VPGATHERQDrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VPGATHERDQrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VPGATHERQQrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VGATHERDPSrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VGATHERQPSrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VGATHERDPDrm")>;
def: InstRW<[SKLWriteResGroup212], (instregex "VGATHERQPDrm")>;

def SKLWriteResGroup213 : SchedWriteRes<[SKLPort0, SKLPort23, SKLPort5, SKLPort015]> {
  let Latency = 20;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKLWriteResGroup213], (instregex "VGATHERDPSYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VGATHERQPDYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VGATHERQPSYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VPGATHERDDYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VPGATHERDQYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VPGATHERQDYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VPGATHERQQYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VPGATHERDDYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VPGATHERQDYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VPGATHERDQYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VPGATHERQQYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VGATHERDPSYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VGATHERQPSYrm")>;
def: InstRW<[SKLWriteResGroup213], (instregex "VGATHERDPDYrm")>;

def SKLWriteResGroup215 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 23;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup215], (instregex "DIVR_FI16m")>;
def: InstRW<[SKLWriteResGroup215], (instregex "DIVR_FI32m")>;

def SKLWriteResGroup217 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort0156]> {
  let Latency = 23;
  let NumMicroOps = 8;
  let ResourceCycles = [2,4,1,1];
}
def: InstRW<[SKLWriteResGroup217], (instregex "IDIV(16|32|64)m")>;
def: InstRW<[SKLWriteResGroup217], (instregex "IDIV8m")>;

def SKLWriteResGroup222 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort23,SKLPort06,SKLPort0156]> {
  let Latency = 30;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[SKLWriteResGroup222], (instregex "IN32ri")>;
def: InstRW<[SKLWriteResGroup222], (instregex "IN32rr")>;
def: InstRW<[SKLWriteResGroup222], (instregex "IN8ri")>;
def: InstRW<[SKLWriteResGroup222], (instregex "IN8rr")>;

def SKLWriteResGroup223 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 30;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[SKLWriteResGroup223], (instregex "OUT32ir")>;
def: InstRW<[SKLWriteResGroup223], (instregex "OUT32rr")>;
def: InstRW<[SKLWriteResGroup223], (instregex "OUT8ir")>;
def: InstRW<[SKLWriteResGroup223], (instregex "OUT8rr")>;

def SKLWriteResGroup224 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 32;
  let NumMicroOps = 31;
  let ResourceCycles = [1,8,1,21];
}
def: InstRW<[SKLWriteResGroup224], (instregex "XRSTOR(64?)")>;

def SKLWriteResGroup225 : SchedWriteRes<[SKLPort1,SKLPort4,SKLPort5,SKLPort6,SKLPort23,SKLPort237,SKLPort15,SKLPort0156]> {
  let Latency = 35;
  let NumMicroOps = 18;
  let ResourceCycles = [1,1,2,3,1,1,1,8];
}
def: InstRW<[SKLWriteResGroup225], (instregex "VMCLEARm")>;

def SKLWriteResGroup226 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 36;
  let NumMicroOps = 39;
  let ResourceCycles = [1,10,1,1,26];
}
def: InstRW<[SKLWriteResGroup226], (instregex "XSAVE64")>;

def SKLWriteResGroup231 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 37;
  let NumMicroOps = 40;
  let ResourceCycles = [1,11,1,1,26];
}
def: InstRW<[SKLWriteResGroup231], (instregex "XSAVE")>;

def SKLWriteResGroup232 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 41;
  let NumMicroOps = 44;
  let ResourceCycles = [1,11,1,1,30];
}
def: InstRW<[SKLWriteResGroup232], (instregex "XSAVEOPT")>;

def SKLWriteResGroup233 : SchedWriteRes<[SKLPort5,SKLPort0156]> {
  let Latency = 42;
  let NumMicroOps = 22;
  let ResourceCycles = [2,20];
}
def: InstRW<[SKLWriteResGroup233], (instregex "RDTSCP")>;

def SKLWriteResGroup234 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort05,SKLPort06,SKLPort0156]> {
  let Latency = 57;
  let NumMicroOps = 64;
  let ResourceCycles = [2,8,5,10,39];
}
def: InstRW<[SKLWriteResGroup234], (instregex "FLDENVm")>;
def: InstRW<[SKLWriteResGroup234], (instregex "FLDENVm")>;

def SKLWriteResGroup235 : SchedWriteRes<[SKLPort0,SKLPort6,SKLPort23,SKLPort05,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 58;
  let NumMicroOps = 88;
  let ResourceCycles = [4,4,31,1,2,1,45];
}
def: InstRW<[SKLWriteResGroup235], (instregex "FXRSTOR64")>;

def SKLWriteResGroup236 : SchedWriteRes<[SKLPort0,SKLPort6,SKLPort23,SKLPort05,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 58;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[SKLWriteResGroup236], (instregex "FXRSTOR")>;

def SKLWriteResGroup239 : SchedWriteRes<[SKLPort5,SKLPort05,SKLPort0156]> {
  let Latency = 75;
  let NumMicroOps = 15;
  let ResourceCycles = [6,3,6];
}
def: InstRW<[SKLWriteResGroup239], (instregex "FNINIT")>;

def SKLWriteResGroup240 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort5,SKLPort6,SKLPort05,SKLPort0156]> {
  let Latency = 76;
  let NumMicroOps = 32;
  let ResourceCycles = [7,2,8,3,1,11];
}
def: InstRW<[SKLWriteResGroup240], (instregex "DIV(16|32|64)r")>;

def SKLWriteResGroup241 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort5,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 102;
  let NumMicroOps = 66;
  let ResourceCycles = [4,2,4,8,14,34];
}
def: InstRW<[SKLWriteResGroup241], (instregex "IDIV(16|32|64)r")>;

def SKLWriteResGroup242 : SchedWriteRes<[SKLPort0,SKLPort1,SKLPort4,SKLPort5,SKLPort6,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 105;
  let NumMicroOps = 100;
  let ResourceCycles = [9,1,11,16,1,11,21,30];
}
def: InstRW<[SKLWriteResGroup242], (instregex "FSTENVm")>;
def: InstRW<[SKLWriteResGroup242], (instregex "FSTENVm")>;

} // SchedModel
