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

def SKLDivider : ProcResource<1>; // Integer division issued on port 0.

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
  def : WriteRes<SchedRW.Folded, !listconcat([SKLPort23], ExePorts)> {
    let Latency = !add(Lat, LoadLat);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = !add(UOps, 1);
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [SKLPort4]>;

// Arithmetic.
defm : SKLWriteResPair<WriteALU,   [SKLPort0156], 1>; // Simple integer ALU op.
defm : SKLWriteResPair<WriteIMul,  [SKLPort1],   3>; // Integer multiplication.
defm : SKLWriteResPair<WriteIDiv, [SKLPort0, SKLDivider], 25, [1,10], 1, 4>; // Integer division.
defm : SKLWriteResPair<WriteCRC32, [SKLPort1], 3>;

def : WriteRes<WriteIMulH, []> { let Latency = 3; } // Integer multiplication, high part.
def : WriteRes<WriteLEA, [SKLPort15]>; // LEA instructions can't fold loads.

// Bit counts.
defm : SKLWriteResPair<WriteBitScan, [SKLPort1], 3>;
defm : SKLWriteResPair<WriteLZCNT,   [SKLPort1], 3>;
defm : SKLWriteResPair<WriteTZCNT,   [SKLPort1], 3>;
defm : SKLWriteResPair<WritePOPCNT,  [SKLPort1], 3>;

// Integer shifts and rotates.
defm : SKLWriteResPair<WriteShift, [SKLPort06],  1>;

// BMI1 BEXTR, BMI2 BZHI
defm : SKLWriteResPair<WriteBEXTR, [SKLPort06,SKLPort15], 2, [1,1], 2>;
defm : SKLWriteResPair<WriteBZHI, [SKLPort15], 1>;

// Loads, stores, and moves, not folded with other operations.
def : WriteRes<WriteLoad,  [SKLPort23]> { let Latency = 5; }
def : WriteRes<WriteStore, [SKLPort237, SKLPort4]>;
def : WriteRes<WriteMove,  [SKLPort0156]>;

// Idioms that clear a register, like xorps %xmm0, %xmm0.
// These can often bypass execution ports completely.
def : WriteRes<WriteZero,  []>;

// Branches don't produce values, so they have no latency, but they still
// consume resources. Indirect branches can fold loads.
defm : SKLWriteResPair<WriteJump,  [SKLPort06],   1>;

// Floating point. This covers both scalar and vector operations.
def  : WriteRes<WriteFLoad,         [SKLPort23]> { let Latency = 6; }
def  : WriteRes<WriteFStore,        [SKLPort237, SKLPort4]>;
def  : WriteRes<WriteFMove,         [SKLPort015]>;

defm : SKLWriteResPair<WriteFAdd,   [SKLPort1], 3>; // Floating point add/sub/compare.
defm : SKLWriteResPair<WriteFMul,   [SKLPort0], 5>; // Floating point multiplication.
defm : SKLWriteResPair<WriteFDiv,   [SKLPort0], 12>; // 10-14 cycles. // Floating point division.
defm : SKLWriteResPair<WriteFSqrt,  [SKLPort0], 15>; // Floating point square root.
defm : SKLWriteResPair<WriteFRcp,   [SKLPort0], 5>; // Floating point reciprocal estimate.
defm : SKLWriteResPair<WriteFRsqrt, [SKLPort0], 5>; // Floating point reciprocal square root estimate.
defm : SKLWriteResPair<WriteFMA,    [SKLPort01], 4>; // Fused Multiply Add.
defm : SKLWriteResPair<WriteFShuffle,  [SKLPort5],  1>; // Floating point vector shuffles.
defm : SKLWriteResPair<WriteFBlend,  [SKLPort015],  1>; // Floating point vector blends.
defm : SKLWriteResPair<WriteFVarBlend,  [SKLPort5], 2, [2]>; // Fp vector variable blends.

// FMA Scheduling helper class.
// class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Vector integer operations.
def  : WriteRes<WriteVecLoad,         [SKLPort23]> { let Latency = 6; }
def  : WriteRes<WriteVecStore,        [SKLPort237, SKLPort4]>;
def  : WriteRes<WriteVecMove,         [SKLPort015]>;

defm : SKLWriteResPair<WriteVecALU,   [SKLPort15],  1>; // Vector integer ALU op, no logicals.
defm : SKLWriteResPair<WriteVecShift, [SKLPort0],  1>; // Vector integer shifts.
defm : SKLWriteResPair<WriteVecIMul,  [SKLPort0],   5>; // Vector integer multiply.
defm : SKLWriteResPair<WritePMULLD,   [SKLPort01], 10, [2], 2, 6>;
defm : SKLWriteResPair<WriteShuffle,  [SKLPort5],  1>; // Vector shuffles.
defm : SKLWriteResPair<WriteBlend,  [SKLPort15],  1>; // Vector blends.
defm : SKLWriteResPair<WriteVarBlend,  [SKLPort5], 2, [2]>; // Vector variable blends.
defm : SKLWriteResPair<WriteMPSAD,  [SKLPort0, SKLPort5], 6, [1, 2]>; // Vector MPSAD.

// Vector bitwise operations.
// These are often used on both floating point and integer vectors.
defm : SKLWriteResPair<WriteVecLogic, [SKLPort015], 1>; // Vector and/or/xor.

// Conversion between integer and float.
defm : SKLWriteResPair<WriteCvtF2I, [SKLPort1], 3>; // Float -> Integer.
defm : SKLWriteResPair<WriteCvtI2F, [SKLPort1], 4>; // Integer -> Float.
defm : SKLWriteResPair<WriteCvtF2F, [SKLPort1], 3>; // Float -> Float size conversion.

// Strings instructions.

// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [SKLPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [SKLPort0, SKLPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [SKLPort0, SKLPort5, SKLPort015, SKLPort0156]> {
  let Latency = 19;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}
def : WriteRes<WritePCmpEStrMLd, [SKLPort0, SKLPort5,SKLPort23, SKLPort015, SKLPort0156]> {
  let Latency = 25;
  let NumMicroOps = 10;
  let ResourceCycles = [4,3,1,1,1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [SKLPort0]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [SKLPort0, SKLPort23]> {
  let Latency = 16;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [SKLPort0, SKLPort5, SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [4,3,1];
}
def : WriteRes<WritePCmpEStrILd, [SKLPort0, SKLPort5, SKLPort23, SKLPort0156]> {
  let Latency = 24;
  let NumMicroOps = 9;
  let ResourceCycles = [4,3,1,1];
}

// MOVMSK Instructions.
def : WriteRes<WriteFMOVMSK, [SKLPort0]> { let Latency = 2; }
def : WriteRes<WriteVecMOVMSK, [SKLPort0]> { let Latency = 2; }
def : WriteRes<WriteMMXMOVMSK, [SKLPort0]> { let Latency = 2; }

// AES instructions.
def : WriteRes<WriteAESDecEnc, [SKLPort0]> { // Decryption, encryption.
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [SKLPort0, SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

def : WriteRes<WriteAESIMC, [SKLPort0]> { // InvMixColumn.
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [SKLPort0, SKLPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}

def : WriteRes<WriteAESKeyGen, [SKLPort0, SKLPort5, SKLPort015]> { // Key Generation.
  let Latency = 20;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,2];
}
def : WriteRes<WriteAESKeyGenLd, [SKLPort0, SKLPort5, SKLPort23, SKLPort015]> {
  let Latency = 25;
  let NumMicroOps = 11;
  let ResourceCycles = [3,6,1,1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [SKLPort5]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def : WriteRes<WriteCLMulLd, [SKLPort5, SKLPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}

// Catch-all for expensive system instructions.
def : WriteRes<WriteSystem,     [SKLPort0156]> { let Latency = 100; } // def WriteSystem : SchedWrite;

// AVX2.
defm : SKLWriteResPair<WriteFShuffle256,  [SKLPort5],  3>; // Fp 256-bit width vector shuffles.
defm : SKLWriteResPair<WriteShuffle256,  [SKLPort5],  3>;  // 256-bit width vector shuffles.
defm : SKLWriteResPair<WriteVarVecShift,  [SKLPort0, SKLPort5], 2, [2, 1]>;  // Variable vector shifts.

// Old microcoded instructions that nobody use.
def : WriteRes<WriteMicrocoded, [SKLPort0156]> { let Latency = 100; } // def WriteMicrocoded : SchedWrite;

// Fence instructions.
def : WriteRes<WriteFence,  [SKLPort23, SKLPort4]>;

// Nop, not very useful expect it provides a model for nops!
def : WriteRes<WriteNop, []>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : SKLWriteResPair<WriteFHAdd,   [SKLPort1], 3>;
defm : SKLWriteResPair<WritePHAdd,  [SKLPort15], 1>;

// Remaining instrs.

def SKLWriteResGroup1 : SchedWriteRes<[SKLPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup1], (instregex "MMX_PADDSBirr",
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
                                            "MMX_PSUBUSWirr")>;

def SKLWriteResGroup3 : SchedWriteRes<[SKLPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup3], (instregex "COMP_FST0r",
                                            "COM_FST0r",
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
                                            "UCOM_FPr",
                                            "UCOM_Fr",
                                            "VBROADCASTSSrr",
                                            "(V?)INSERTPSrr",
                                            "(V?)MOV64toPQIrr",
                                            "(V?)MOVDDUP(Y?)rr",
                                            "(V?)MOVDI2PDIrr",
                                            "(V?)MOVHLPSrr",
                                            "(V?)MOVLHPSrr",
                                            "(V?)MOVSDrr",
                                            "(V?)MOVSHDUP(Y?)rr",
                                            "(V?)MOVSLDUP(Y?)rr",
                                            "(V?)MOVSSrr",
                                            "(V?)PACKSSDW(Y?)rr",
                                            "(V?)PACKSSWB(Y?)rr",
                                            "(V?)PACKUSDW(Y?)rr",
                                            "(V?)PACKUSWB(Y?)rr",
                                            "(V?)PALIGNR(Y?)rri",
                                            "(V?)PBLENDW(Y?)rri",
                                            "VPBROADCASTDrr",
                                            "VPBROADCASTQrr",
                                            "VPERMILPD(Y?)ri",
                                            "VPERMILPD(Y?)rr",
                                            "VPERMILPS(Y?)ri",
                                            "VPERMILPS(Y?)rr",
                                            "(V?)PMOVSXBDrr",
                                            "(V?)PMOVSXBQrr",
                                            "(V?)PMOVSXBWrr",
                                            "(V?)PMOVSXDQrr",
                                            "(V?)PMOVSXWDrr",
                                            "(V?)PMOVSXWQrr",
                                            "(V?)PMOVZXBDrr",
                                            "(V?)PMOVZXBQrr",
                                            "(V?)PMOVZXBWrr",
                                            "(V?)PMOVZXDQrr",
                                            "(V?)PMOVZXWDrr",
                                            "(V?)PMOVZXWQrr",
                                            "(V?)PSHUFB(Y?)rr",
                                            "(V?)PSHUFD(Y?)ri",
                                            "(V?)PSHUFHW(Y?)ri",
                                            "(V?)PSHUFLW(Y?)ri",
                                            "(V?)PSLLDQ(Y?)ri",
                                            "(V?)PSRLDQ(Y?)ri",
                                            "(V?)PUNPCKHBW(Y?)rr",
                                            "(V?)PUNPCKHDQ(Y?)rr",
                                            "(V?)PUNPCKHQDQ(Y?)rr",
                                            "(V?)PUNPCKHWD(Y?)rr",
                                            "(V?)PUNPCKLBW(Y?)rr",
                                            "(V?)PUNPCKLDQ(Y?)rr",
                                            "(V?)PUNPCKLQDQ(Y?)rr",
                                            "(V?)PUNPCKLWD(Y?)rr",
                                            "(V?)SHUFPD(Y?)rri",
                                            "(V?)SHUFPS(Y?)rri",
                                            "(V?)UNPCKHPD(Y?)rr",
                                            "(V?)UNPCKHPS(Y?)rr",
                                            "(V?)UNPCKLPD(Y?)rr",
                                            "(V?)UNPCKLPS(Y?)rr")>;

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
def: InstRW<[SKLWriteResGroup5], (instregex "(V?)PABSB(Y?)rr",
                                            "(V?)PABSD(Y?)rr",
                                            "(V?)PABSW(Y?)rr",
                                            "(V?)PADDSB(Y?)rr",
                                            "(V?)PADDSW(Y?)rr",
                                            "(V?)PADDUSB(Y?)rr",
                                            "(V?)PADDUSW(Y?)rr",
                                            "(V?)PAVGB(Y?)rr",
                                            "(V?)PAVGW(Y?)rr",
                                            "(V?)PCMPEQB(Y?)rr",
                                            "(V?)PCMPEQD(Y?)rr",
                                            "(V?)PCMPEQQ(Y?)rr",
                                            "(V?)PCMPEQW(Y?)rr",
                                            "(V?)PCMPGTB(Y?)rr",
                                            "(V?)PCMPGTD(Y?)rr",
                                            "(V?)PCMPGTW(Y?)rr",
                                            "(V?)PMAXSB(Y?)rr",
                                            "(V?)PMAXSD(Y?)rr",
                                            "(V?)PMAXSW(Y?)rr",
                                            "(V?)PMAXUB(Y?)rr",
                                            "(V?)PMAXUD(Y?)rr",
                                            "(V?)PMAXUW(Y?)rr",
                                            "(V?)PMINSB(Y?)rr",
                                            "(V?)PMINSD(Y?)rr",
                                            "(V?)PMINSW(Y?)rr",
                                            "(V?)PMINUB(Y?)rr",
                                            "(V?)PMINUD(Y?)rr",
                                            "(V?)PMINUW(Y?)rr",
                                            "(V?)PSIGNB(Y?)rr",
                                            "(V?)PSIGND(Y?)rr",
                                            "(V?)PSIGNW(Y?)rr",
                                            "(V?)PSLLD(Y?)ri",
                                            "(V?)PSLLQ(Y?)ri",
                                            "VPSLLVD(Y?)rr",
                                            "VPSLLVQ(Y?)rr",
                                            "(V?)PSLLW(Y?)ri",
                                            "(V?)PSRAD(Y?)ri",
                                            "VPSRAVD(Y?)rr",
                                            "(V?)PSRAW(Y?)ri",
                                            "(V?)PSRLD(Y?)ri",
                                            "(V?)PSRLQ(Y?)ri",
                                            "VPSRLVD(Y?)rr",
                                            "VPSRLVQ(Y?)rr",
                                            "(V?)PSRLW(Y?)ri",
                                            "(V?)PSUBSB(Y?)rr",
                                            "(V?)PSUBSW(Y?)rr",
                                            "(V?)PSUBUSB(Y?)rr",
                                            "(V?)PSUBUSW(Y?)rr")>;

def SKLWriteResGroup6 : SchedWriteRes<[SKLPort05]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup6], (instregex "FINCSTP",
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

def SKLWriteResGroup7 : SchedWriteRes<[SKLPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup7], (instregex "ADC(16|32|64)ri",
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

def SKLWriteResGroup8 : SchedWriteRes<[SKLPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup8], (instregex "ANDN(32|64)rr",
                                            "BLSI(32|64)rr",
                                            "BLSMSK(32|64)rr",
                                            "BLSR(32|64)rr",
                                            "LEA(16|32|64)(_32)?r")>;

def SKLWriteResGroup9 : SchedWriteRes<[SKLPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup9], (instregex "(V?)ANDNPD(Y?)rr",
                                            "(V?)ANDNPS(Y?)rr",
                                            "(V?)ANDPD(Y?)rr",
                                            "(V?)ANDPS(Y?)rr",
                                            "(V?)BLENDPD(Y?)rri",
                                            "(V?)BLENDPS(Y?)rri",
                                            "(V?)MOVAPD(Y?)rr",
                                            "(V?)MOVAPS(Y?)rr",
                                            "(V?)MOVDQA(Y?)rr",
                                            "(V?)MOVDQU(Y?)rr",
                                            "(V?)MOVPQI2QIrr",
                                            "(V?)MOVUPD(Y?)rr",
                                            "(V?)MOVUPS(Y?)rr",
                                            "(V?)MOVZPQILo2PQIrr",
                                            "(V?)ORPD(Y?)rr",
                                            "(V?)ORPS(Y?)rr",
                                            "(V?)PADDB(Y?)rr",
                                            "(V?)PADDD(Y?)rr",
                                            "(V?)PADDQ(Y?)rr",
                                            "(V?)PADDW(Y?)rr",
                                            "(V?)PANDN(Y?)rr",
                                            "(V?)PAND(Y?)rr",
                                            "VPBLENDD(Y?)rri",
                                            "(V?)POR(Y?)rr",
                                            "(V?)PSUBB(Y?)rr",
                                            "(V?)PSUBD(Y?)rr",
                                            "(V?)PSUBQ(Y?)rr",
                                            "(V?)PSUBW(Y?)rr",
                                            "(V?)PXOR(Y?)rr",
                                            "(V?)XORPD(Y?)rr",
                                            "(V?)XORPS(Y?)rr")>;

def SKLWriteResGroup10 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup10], (instrs CWDE)>;
def: InstRW<[SKLWriteResGroup10], (instregex "ADD(8|16|32|64)ri",
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

def SKLWriteResGroup11 : SchedWriteRes<[SKLPort4,SKLPort237]> {
  let Latency = 1;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup11], (instregex "FBSTPm",
                                             "MMX_MOVD64from64rm",
                                             "MMX_MOVD64mr",
                                             "MMX_MOVNTQmr",
                                             "MMX_MOVQ64mr",
                                             "MOV(8|16|32|64)mr",
                                             "MOV8mi",
                                             "MOVNTI_64mr",
                                             "MOVNTImr",
                                             "ST_FP32m",
                                             "ST_FP64m",
                                             "ST_FP80m",
                                             "VEXTRACTF128mr",
                                             "VEXTRACTI128mr",
                                             "(V?)MOVAPDYmr",
                                             "(V?)MOVAPS(Y?)mr",
                                             "(V?)MOVDQA(Y?)mr",
                                             "(V?)MOVDQU(Y?)mr",
                                             "(V?)MOVHPDmr",
                                             "(V?)MOVHPSmr",
                                             "(V?)MOVLPDmr",
                                             "(V?)MOVLPSmr",
                                             "(V?)MOVNTDQ(Y?)mr",
                                             "(V?)MOVNTPD(Y?)mr",
                                             "(V?)MOVNTPS(Y?)mr",
                                             "(V?)MOVPDI2DImr",
                                             "(V?)MOVPQI2QImr",
                                             "(V?)MOVPQIto64mr",
                                             "(V?)MOVSDmr",
                                             "(V?)MOVSSmr",
                                             "(V?)MOVUPD(Y?)mr",
                                             "(V?)MOVUPS(Y?)mr",
                                             "VMPTRSTm")>;

def SKLWriteResGroup12 : SchedWriteRes<[SKLPort0]> {
  let Latency = 2;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup12], (instregex "MMX_MOVD64from64rr",
                                             "MMX_MOVD64grr",
                                             "(V?)COMISDrr",
                                             "(V?)COMISSrr",
                                             "(V?)MOVPDI2DIrr",
                                             "(V?)MOVPQIto64rr",
                                             "VTESTPD(Y?)rr",
                                             "VTESTPS(Y?)rr",
                                             "(V?)UCOMISDrr",
                                             "(V?)UCOMISSrr")>;

def SKLWriteResGroup13 : SchedWriteRes<[SKLPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup13], (instregex "MMX_MOVQ2DQrr",
                                             "MMX_PINSRWrr",
                                             "(V?)PINSRBrr",
                                             "(V?)PINSRDrr",
                                             "(V?)PINSRQrr",
                                             "(V?)PINSRWrr")>;

def SKLWriteResGroup14 : SchedWriteRes<[SKLPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup14], (instregex "FDECSTP",
                                             "MMX_MOVDQ2Qrr")>;

def SKLWriteResGroup15 : SchedWriteRes<[SKLPort06]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup15], (instregex "CMOV(A|BE)(16|32|64)rr",
                                             "ROL(8|16|32|64)r1",
                                             "ROL(8|16|32|64)ri",
                                             "ROR(8|16|32|64)r1",
                                             "ROR(8|16|32|64)ri",
                                             "SET(A|BE)r")>;

def SKLWriteResGroup16 : SchedWriteRes<[SKLPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup16], (instregex "BLENDVPDrr0",
                                             "BLENDVPSrr0",
                                             "PBLENDVBrr0",
                                             "VBLENDVPD(Y?)rr",
                                             "VBLENDVPS(Y?)rr",
                                             "VPBLENDVB(Y?)rr")>;

def SKLWriteResGroup17 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup17], (instregex "LFENCE",
                                             "WAIT",
                                             "XGETBV")>;

def SKLWriteResGroup18 : SchedWriteRes<[SKLPort0,SKLPort237]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup18], (instregex "VMASKMOVPD(Y?)mr",
                                             "VMASKMOVPS(Y?)mr",
                                             "VPMASKMOVD(Y?)mr",
                                             "VPMASKMOVQ(Y?)mr")>;

def SKLWriteResGroup19 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup19], (instregex "(V?)PSLLDrr",
                                             "(V?)PSLLQrr",
                                             "(V?)PSLLWrr",
                                             "(V?)PSRADrr",
                                             "(V?)PSRAWrr",
                                             "(V?)PSRLDrr",
                                             "(V?)PSRLQrr",
                                             "(V?)PSRLWrr")>;

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
def: InstRW<[SKLWriteResGroup22], (instregex "BSWAP(16|32|64)r")>;

def SKLWriteResGroup23 : SchedWriteRes<[SKLPort06,SKLPort0156]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup23], (instrs CWD)>;
def: InstRW<[SKLWriteResGroup23], (instrs JCXZ, JECXZ, JRCXZ)>;
def: InstRW<[SKLWriteResGroup23], (instregex "ADC8i8",
                                             "ADC8ri",
                                             "SBB8i8",
                                             "SBB8ri")>;

def SKLWriteResGroup24 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup24], (instregex "(V?)EXTRACTPSmr",
                                             "(V?)PEXTRBmr",
                                             "(V?)PEXTRDmr",
                                             "(V?)PEXTRQmr",
                                             "(V?)PEXTRWmr",
                                             "(V?)STMXCSR")>;

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
def: InstRW<[SKLWriteResGroup28], (instrs PUSH16r, PUSH32r, PUSH64r)>;
def: InstRW<[SKLWriteResGroup28], (instregex "PUSH(16|32|64)rmr",
                                             "PUSH64i8",
                                             "STOSB",
                                             "STOSL",
                                             "STOSQ",
                                             "STOSW")>;

def SKLWriteResGroup29 : SchedWriteRes<[SKLPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup29], (instrs IMUL16rr, IMUL32rr, IMUL32rri, IMUL32rri8, IMUL64rr, IMUL64rri32, IMUL64rri8)>;
def: InstRW<[SKLWriteResGroup29], (instrs IMUL8r, MUL8r)>;
def: InstRW<[SKLWriteResGroup29], (instregex "PDEP(32|64)rr",
                                             "PEXT(32|64)rr",
                                             "SHLD(16|32|64)rri8",
                                             "SHRD(16|32|64)rri8")>;

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
def: InstRW<[SKLWriteResGroup30], (instregex "ADD_FPrST0",
                                             "ADD_FST0r",
                                             "ADD_FrST0",
                                             "MMX_PSADBWirr",
                                             "SUBR_FPrST0",
                                             "SUBR_FST0r",
                                             "SUBR_FrST0",
                                             "SUB_FPrST0",
                                             "SUB_FST0r",
                                             "SUB_FrST0",
                                             "VBROADCASTSDYrr",
                                             "VBROADCASTSSYrr",
                                             "VEXTRACTF128rr",
                                             "VEXTRACTI128rr",
                                             "VINSERTF128rr",
                                             "VINSERTI128rr",
                                             "VPBROADCASTB(Y?)rr",
                                             "VPBROADCASTDYrr",
                                             "VPBROADCASTQYrr",
                                             "VPBROADCASTW(Y?)rr",
                                             "(V?)PCMPGTQ(Y?)rr",
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
                                             "VPMOVZXWQYrr",
                                             "(V?)PSADBW(Y?)rr")>;

def SKLWriteResGroup31 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup31], (instregex "MMX_PEXTRWrr",
                                             "(V?)EXTRACTPSrr",
                                             "(V?)PEXTRBrr",
                                             "(V?)PEXTRDrr",
                                             "(V?)PEXTRQrr",
                                             "(V?)PEXTRWrr",
                                             "(V?)PTEST(Y?)rr")>;

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
def: InstRW<[SKLWriteResGroup33], (instregex "ROL(8|16|32|64)rCL",
                                             "ROR(8|16|32|64)rCL",
                                             "SAR(8|16|32|64)rCL",
                                             "SHL(8|16|32|64)rCL",
                                             "SHR(8|16|32|64)rCL")>;

def SKLWriteResGroup34 : SchedWriteRes<[SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SKLWriteResGroup34], (instregex "XADD(8|16|32|64)rr",
                                             "XCHG8rr")>;

def SKLWriteResGroup35 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup35], (instregex "MMX_PHADDSWrr",
                                             "MMX_PHSUBSWrr")>;

def SKLWriteResGroup36 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup36], (instregex "(V?)PHADDSW(Y?)rr",
                                             "(V?)PHSUBSW(Y?)rr")>;

def SKLWriteResGroup37 : SchedWriteRes<[SKLPort5,SKLPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup37], (instregex "MMX_PHADDDrr",
                                             "MMX_PHADDWrr",
                                             "MMX_PHSUBDrr",
                                             "MMX_PHSUBWrr")>;

def SKLWriteResGroup38 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup38], (instregex "(V?)PHADDD(Y?)rr",
                                             "(V?)PHADDW(Y?)rr",
                                             "(V?)PHSUBD(Y?)rr",
                                             "(V?)PHSUBW(Y?)rr")>;

def SKLWriteResGroup39 : SchedWriteRes<[SKLPort5,SKLPort0156]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup39], (instregex "MMX_PACKSSDWirr",
                                             "MMX_PACKSSWBirr",
                                             "MMX_PACKUSWBirr")>;

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
def: InstRW<[SKLWriteResGroup42], (instregex "RCL(8|16|32|64)r1",
                                             "RCL(8|16|32|64)ri",
                                             "RCR(8|16|32|64)r1",
                                             "RCR(8|16|32|64)ri")>;

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
def: InstRW<[SKLWriteResGroup47], (instregex "MMX_PMADDUBSWrr",
                                             "MMX_PMADDWDirr",
                                             "MMX_PMULHRSWrr",
                                             "MMX_PMULHUWirr",
                                             "MMX_PMULHWirr",
                                             "MMX_PMULLWirr",
                                             "MMX_PMULUDQirr",
                                             "MUL_FPrST0",
                                             "MUL_FST0r",
                                             "MUL_FrST0",
                                             "(V?)RCPPS(Y?)r",
                                             "(V?)RCPSSr",
                                             "(V?)RSQRTPS(Y?)r",
                                             "(V?)RSQRTSSr")>;

def SKLWriteResGroup48 : SchedWriteRes<[SKLPort01]> {
  let Latency = 4;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup48], (instregex "(V?)ADDPD(Y?)rr",
                                             "(V?)ADDPS(Y?)rr",
                                             "(V?)ADDSDrr",
                                             "(V?)ADDSSrr",
                                             "(V?)ADDSUBPD(Y?)rr",
                                             "(V?)ADDSUBPS(Y?)rr",
                                             "(V?)CMPPD(Y?)rri",
                                             "(V?)CMPPS(Y?)rri",
                                             "(V?)CMPSDrr",
                                             "(V?)CMPSSrr",
                                             "(V?)CVTDQ2PS(Y?)rr",
                                             "(V?)CVTPS2DQ(Y?)rr",
                                             "(V?)CVTTPS2DQ(Y?)rr",
                                             "(V?)MAX(C?)PD(Y?)rr",
                                             "(V?)MAX(C?)PS(Y?)rr",
                                             "(V?)MAX(C?)SDrr",
                                             "(V?)MAX(C?)SSrr",
                                             "(V?)MIN(C?)PD(Y?)rr",
                                             "(V?)MIN(C?)PS(Y?)rr",
                                             "(V?)MIN(C?)SDrr",
                                             "(V?)MIN(C?)SSrr",
                                             "(V?)MULPD(Y?)rr",
                                             "(V?)MULPS(Y?)rr",
                                             "(V?)MULSDrr",
                                             "(V?)MULSSrr",
                                             "(V?)PHMINPOSUWrr",
                                             "(V?)PMADDUBSW(Y?)rr",
                                             "(V?)PMADDWD(Y?)rr",
                                             "(V?)PMULDQ(Y?)rr",
                                             "(V?)PMULHRSW(Y?)rr",
                                             "(V?)PMULHUW(Y?)rr",
                                             "(V?)PMULHW(Y?)rr",
                                             "(V?)PMULLW(Y?)rr",
                                             "(V?)PMULUDQ(Y?)rr",
                                             "(V?)SUBPD(Y?)rr",
                                             "(V?)SUBPS(Y?)rr",
                                             "(V?)SUBSDrr",
                                             "(V?)SUBSSrr")>;
def: InstRW<[SKLWriteResGroup48],
            (instregex
             "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)(Y)?r",
             "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)r")>;

def SKLWriteResGroup50 : SchedWriteRes<[SKLPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup50], (instregex "(V?)MPSADBW(Y?)rri")>;

def SKLWriteResGroup51 : SchedWriteRes<[SKLPort1,SKLPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup51], (instrs IMUL64r, MUL64r,
                                          MULX64rr)>;

def SKLWriteResGroup51_16 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
}
def: InstRW<[SKLWriteResGroup51_16], (instrs IMUL16r, MUL16r)>;

def SKLWriteResGroup52 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup52], (instregex "VPSLLDYrr",
                                             "VPSLLQYrr",
                                             "VPSLLWYrr",
                                             "VPSRADYrr",
                                             "VPSRAWYrr",
                                             "VPSRLDYrr",
                                             "VPSRLQYrr",
                                             "VPSRLWYrr")>;

def SKLWriteResGroup53 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup53], (instregex "ISTT_FP16m",
                                             "ISTT_FP32m",
                                             "ISTT_FP64m",
                                             "IST_F16m",
                                             "IST_F32m",
                                             "IST_FP16m",
                                             "IST_FP32m",
                                             "IST_FP64m")>;

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
def: InstRW<[SKLWriteResGroup58], (instregex "MMX_MOVD64rm",
                                             "MMX_MOVD64to64rm",
                                             "MMX_MOVQ64rm",
                                             "MOV(8|16|32|64)rm",
                                             "MOVSX(16|32|64)rm16",
                                             "MOVSX(16|32|64)rm32",
                                             "MOVSX(16|32|64)rm8",
                                             "MOVZX(16|32|64)rm16",
                                             "MOVZX(16|32|64)rm8",
                                             "PREFETCHNTA",
                                             "PREFETCHT0",
                                             "PREFETCHT1",
                                             "PREFETCHT2",
                                             "(V?)MOV64toPQIrm",
                                             "(V?)MOVDDUPrm",
                                             "(V?)MOVDI2PDIrm",
                                             "(V?)MOVQI2PQIrm",
                                             "(V?)MOVSDrm",
                                             "(V?)MOVSSrm")>;

def SKLWriteResGroup59 : SchedWriteRes<[SKLPort0,SKLPort5]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup59], (instregex "MMX_CVTPI2PDirr",
                                             "(V?)CVTDQ2PDrr")>;

def SKLWriteResGroup60 : SchedWriteRes<[SKLPort5,SKLPort015]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup60], (instregex "MMX_CVTPD2PIirr",
                                             "MMX_CVTPS2PIirr",
                                             "MMX_CVTTPD2PIirr",
                                             "MMX_CVTTPS2PIirr",
                                             "(V?)CVTPD2DQrr",
                                             "(V?)CVTPD2PSrr",
                                             "VCVTPH2PSrr",
                                             "(V?)CVTPS2PDrr",
                                             "VCVTPS2PHrr",
                                             "(V?)CVTSD2SSrr",
                                             "(V?)CVTSI642SDrr",
                                             "(V?)CVTSI2SDrr",
                                             "(V?)CVTSI2SSrr",
                                             "(V?)CVTSS2SDrr",
                                             "(V?)CVTTPD2DQrr")>;

def SKLWriteResGroup61 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort06]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup61], (instregex "STR(16|32|64)r")>;

def SKLWriteResGroup62 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup62], (instrs IMUL32r, MUL32r, MULX32rr)>;

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
def: InstRW<[SKLWriteResGroup64], (instregex "CMPXCHG(8|16|32|64)rr")>;

def SKLWriteResGroup65 : SchedWriteRes<[SKLPort4,SKLPort237,SKLPort0156]> {
  let Latency = 5;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,4];
}
def: InstRW<[SKLWriteResGroup65], (instregex "PUSHF16",
                                             "PUSHF64")>;

def SKLWriteResGroup67 : SchedWriteRes<[SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup67], (instregex "VBROADCASTSSrm",
                                             "(V?)LDDQUrm",
                                             "(V?)MOVAPDrm",
                                             "(V?)MOVAPSrm",
                                             "(V?)MOVDQArm",
                                             "(V?)MOVDQUrm",
                                             "(V?)MOVNTDQArm",
                                             "(V?)MOVSHDUPrm",
                                             "(V?)MOVSLDUPrm",
                                             "(V?)MOVUPDrm",
                                             "(V?)MOVUPSrm",
                                             "VPBROADCASTDrm",
                                             "VPBROADCASTQrm")>;

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
def: InstRW<[SKLWriteResGroup69], (instregex "MMX_PADDSBirm",
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

def SKLWriteResGroup70 : SchedWriteRes<[SKLPort0,SKLPort01]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup70], (instregex "(V?)CVTSD2SI64rr",
                                             "(V?)CVTSD2SIrr",
                                             "(V?)CVTSS2SI64rr",
                                             "(V?)CVTSS2SIrr",
                                             "(V?)CVTTSD2SI64rr",
                                             "(V?)CVTTSD2SIrr")>;

def SKLWriteResGroup71 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup71], (instregex "MMX_PALIGNRrmi",
                                             "MMX_PINSRWrm",
                                             "MMX_PSHUFBrm",
                                             "MMX_PSHUFWmi",
                                             "MMX_PUNPCKHBWirm",
                                             "MMX_PUNPCKHDQirm",
                                             "MMX_PUNPCKHWDirm",
                                             "MMX_PUNPCKLBWirm",
                                             "MMX_PUNPCKLDQirm",
                                             "MMX_PUNPCKLWDirm",
                                             "(V?)MOVHPDrm",
                                             "(V?)MOVHPSrm",
                                             "(V?)MOVLPDrm",
                                             "(V?)MOVLPSrm",
                                             "(V?)PINSRBrm",
                                             "(V?)PINSRDrm",
                                             "(V?)PINSRQrm",
                                             "(V?)PINSRWrm",
                                             "(V?)PMOVSXBDrm",
                                             "(V?)PMOVSXBQrm",
                                             "(V?)PMOVSXBWrm",
                                             "(V?)PMOVSXDQrm",
                                             "(V?)PMOVSXWDrm",
                                             "(V?)PMOVSXWQrm",
                                             "(V?)PMOVZXBDrm",
                                             "(V?)PMOVZXBQrm",
                                             "(V?)PMOVZXBWrm",
                                             "(V?)PMOVZXDQrm",
                                             "(V?)PMOVZXWDrm",
                                             "(V?)PMOVZXWQrm")>;

def SKLWriteResGroup72 : SchedWriteRes<[SKLPort6,SKLPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup72], (instregex "FARJMP64",
                                             "JMP(16|32|64)m")>;

def SKLWriteResGroup73 : SchedWriteRes<[SKLPort23,SKLPort05]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup73], (instregex "MMX_PABSBrm",
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

def SKLWriteResGroup74 : SchedWriteRes<[SKLPort23,SKLPort06]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup74], (instregex "ADC(8|16|32|64)rm",
                                             "ADCX(32|64)rm",
                                             "ADOX(32|64)rm",
                                             "BT(16|32|64)mi8",
                                             "CMOV(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)(16|32|64)rm",
                                             "RORX(32|64)mi",
                                             "SARX(32|64)rm",
                                             "SBB(8|16|32|64)rm",
                                             "SHLX(32|64)rm",
                                             "SHRX(32|64)rm")>;

def SKLWriteResGroup75 : SchedWriteRes<[SKLPort23,SKLPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup75], (instregex "ANDN(32|64)rm",
                                             "BLSI(32|64)rm",
                                             "BLSMSK(32|64)rm",
                                             "BLSR(32|64)rm",
                                             "MOVBE(16|32|64)rm")>;

def SKLWriteResGroup76 : SchedWriteRes<[SKLPort23,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup76], (instrs POP16r, POP32r, POP64r)>;
def: InstRW<[SKLWriteResGroup76], (instregex "ADD(8|16|32|64)rm",
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

def SKLWriteResGroup77 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup77], (instregex "(V?)HADDPD(Y?)rr",
                                             "(V?)HADDPS(Y?)rr",
                                             "(V?)HSUBPD(Y?)rr",
                                             "(V?)HSUBPS(Y?)rr")>;

def SKLWriteResGroup78 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup78], (instregex "(V?)CVTSI642SSrr")>;

def SKLWriteResGroup79 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKLWriteResGroup79], (instregex "SHLD(16|32|64)rrCL",
                                             "SHRD(16|32|64)rrCL")>;

def SKLWriteResGroup80 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup80], (instregex "SLDT(16|32|64)r")>;

def SKLWriteResGroup81 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237,SKLPort01]> {
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
def: InstRW<[SKLWriteResGroup82], (instregex "BTC(16|32|64)mi8",
                                             "BTR(16|32|64)mi8",
                                             "BTS(16|32|64)mi8",
                                             "SAR(8|16|32|64)m1",
                                             "SAR(8|16|32|64)mi",
                                             "SHL(8|16|32|64)m1",
                                             "SHL(8|16|32|64)mi",
                                             "SHR(8|16|32|64)m1",
                                             "SHR(8|16|32|64)mi")>;

def SKLWriteResGroup83 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 6;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup83], (instregex "ADD(8|16|32|64)mi",
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
def: InstRW<[SKLWriteResGroup85], (instregex "LD_F32m",
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
                                             "VPBROADCASTQYrm")>;

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
def: InstRW<[SKLWriteResGroup87], (instregex "(V?)COMISDrm",
                                             "(V?)COMISSrm",
                                             "(V?)UCOMISDrm",
                                             "(V?)UCOMISSrm")>;

def SKLWriteResGroup88 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup88], (instregex "(V?)INSERTPSrm",
                                             "(V?)PACKSSDWrm",
                                             "(V?)PACKSSWBrm",
                                             "(V?)PACKUSDWrm",
                                             "(V?)PACKUSWBrm",
                                             "(V?)PALIGNRrmi",
                                             "(V?)PBLENDWrmi",
                                             "VPBROADCASTBrm",
                                             "VPBROADCASTWrm",
                                             "VPERMILPDmi",
                                             "VPERMILPDrm",
                                             "VPERMILPSmi",
                                             "VPERMILPSrm",
                                             "(V?)PSHUFBrm",
                                             "(V?)PSHUFDmi",
                                             "(V?)PSHUFHWmi",
                                             "(V?)PSHUFLWmi",
                                             "(V?)PUNPCKHBWrm",
                                             "(V?)PUNPCKHDQrm",
                                             "(V?)PUNPCKHQDQrm",
                                             "(V?)PUNPCKHWDrm",
                                             "(V?)PUNPCKLBWrm",
                                             "(V?)PUNPCKLDQrm",
                                             "(V?)PUNPCKLQDQrm",
                                             "(V?)PUNPCKLWDrm",
                                             "(V?)SHUFPDrmi",
                                             "(V?)SHUFPSrmi",
                                             "(V?)UNPCKHPDrm",
                                             "(V?)UNPCKHPSrm",
                                             "(V?)UNPCKLPDrm",
                                             "(V?)UNPCKLPSrm")>;

def SKLWriteResGroup89 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup89], (instregex "VCVTPD2DQYrr",
                                             "VCVTPD2PSYrr",
                                             "VCVTPH2PSYrr",
                                             "VCVTPS2PDYrr",
                                             "VCVTPS2PHYrr",
                                             "VCVTTPD2DQYrr")>;

def SKLWriteResGroup90 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup90], (instregex "(V?)PABSBrm",
                                             "(V?)PABSDrm",
                                             "(V?)PABSWrm",
                                             "(V?)PADDSBrm",
                                             "(V?)PADDSWrm",
                                             "(V?)PADDUSBrm",
                                             "(V?)PADDUSWrm",
                                             "(V?)PAVGBrm",
                                             "(V?)PAVGWrm",
                                             "(V?)PCMPEQBrm",
                                             "(V?)PCMPEQDrm",
                                             "(V?)PCMPEQQrm",
                                             "(V?)PCMPEQWrm",
                                             "(V?)PCMPGTBrm",
                                             "(V?)PCMPGTDrm",
                                             "(V?)PCMPGTWrm",
                                             "(V?)PMAXSBrm",
                                             "(V?)PMAXSDrm",
                                             "(V?)PMAXSWrm",
                                             "(V?)PMAXUBrm",
                                             "(V?)PMAXUDrm",
                                             "(V?)PMAXUWrm",
                                             "(V?)PMINSBrm",
                                             "(V?)PMINSDrm",
                                             "(V?)PMINSWrm",
                                             "(V?)PMINUBrm",
                                             "(V?)PMINUDrm",
                                             "(V?)PMINUWrm",
                                             "(V?)PSIGNBrm",
                                             "(V?)PSIGNDrm",
                                             "(V?)PSIGNWrm",
                                             "(V?)PSLLDrm",
                                             "(V?)PSLLQrm",
                                             "VPSLLVDrm",
                                             "VPSLLVQrm",
                                             "(V?)PSLLWrm",
                                             "(V?)PSRADrm",
                                             "VPSRAVDrm",
                                             "(V?)PSRAWrm",
                                             "(V?)PSRLDrm",
                                             "(V?)PSRLQrm",
                                             "(V?)PSRLVDrm",
                                             "VPSRLVQrm",
                                             "(V?)PSRLWrm",
                                             "(V?)PSUBSBrm",
                                             "(V?)PSUBSWrm",
                                             "(V?)PSUBUSBrm",
                                             "(V?)PSUBUSWrm")>;

def SKLWriteResGroup91 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup91], (instregex "(V?)ANDNPDrm",
                                             "(V?)ANDNPSrm",
                                             "(V?)ANDPDrm",
                                             "(V?)ANDPSrm",
                                             "(V?)BLENDPDrmi",
                                             "(V?)BLENDPSrmi",
                                             "(V?)INSERTF128rm",
                                             "(V?)INSERTI128rm",
                                             "(V?)MASKMOVPDrm",
                                             "(V?)MASKMOVPSrm",
                                             "(V?)ORPDrm",
                                             "(V?)ORPSrm",
                                             "(V?)PADDBrm",
                                             "(V?)PADDDrm",
                                             "(V?)PADDQrm",
                                             "(V?)PADDWrm",
                                             "(V?)PANDNrm",
                                             "(V?)PANDrm",
                                             "(V?)PBLENDDrmi",
                                             "(V?)PMASKMOVDrm",
                                             "(V?)PMASKMOVQrm",
                                             "(V?)PORrm",
                                             "(V?)PSUBBrm",
                                             "(V?)PSUBDrm",
                                             "(V?)PSUBQrm",
                                             "(V?)PSUBWrm",
                                             "(V?)PXORrm",
                                             "(V?)XORPDrm",
                                             "(V?)XORPSrm")>;

def SKLWriteResGroup92 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup92], (instregex "MMX_PACKSSDWirm",
                                             "MMX_PACKSSWBirm",
                                             "MMX_PACKUSWBirm")>;

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
def: InstRW<[SKLWriteResGroup94], (instregex "LEAVE64",
                                             "SCASB",
                                             "SCASL",
                                             "SCASQ",
                                             "SCASW")>;

def SKLWriteResGroup95 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort01]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup95], (instregex "(V?)CVTTSS2SI64rr",
                                             "(V?)CVTTSS2SIrr")>;

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
def: InstRW<[SKLWriteResGroup97], (instregex "(V?)LDMXCSR")>;

def SKLWriteResGroup98 : SchedWriteRes<[SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup98], (instregex "LRETQ",
                                             "RETQ")>;

def SKLWriteResGroup100 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup100], (instregex "ROL(8|16|32|64)m1",
                                              "ROL(8|16|32|64)mi",
                                              "ROR(8|16|32|64)m1",
                                              "ROR(8|16|32|64)mi")>;

def SKLWriteResGroup101 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup101], (instregex "XADD(8|16|32|64)rm")>;

def SKLWriteResGroup102 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,1,1];
}
def: InstRW<[SKLWriteResGroup102], (instregex "CALL(16|32|64)m",
                                              "FARCALL64")>;

def SKLWriteResGroup103 : SchedWriteRes<[SKLPort6,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 7;
  let NumMicroOps = 7;
  let ResourceCycles = [1,3,1,2];
}
def: InstRW<[SKLWriteResGroup103], (instrs LOOP)>;

def SKLWriteResGroup105 : SchedWriteRes<[SKLPort01]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SKLWriteResGroup105], (instregex "(V?)ROUNDPD(Y?)r",
                                              "(V?)ROUNDPS(Y?)r",
                                              "(V?)ROUNDSDr",
                                              "(V?)ROUNDSSr")>;

def SKLWriteResGroup106 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup106], (instregex "VTESTPDrm",
                                              "VTESTPSrm")>;

def SKLWriteResGroup107 : SchedWriteRes<[SKLPort1,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup107], (instrs IMUL32rmi, IMUL32rmi8, IMUL64rm, IMUL64rmi32, IMUL64rmi8)>;
def: InstRW<[SKLWriteResGroup107], (instrs IMUL8m, MUL8m)>;
def: InstRW<[SKLWriteResGroup107], (instregex "PDEP(32|64)rm",
                                              "PEXT(32|64)rm")>;

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
def: InstRW<[SKLWriteResGroup107_16_2], (instrs IMUL16m, MUL16m)>;

def SKLWriteResGroup108 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup108], (instregex "FCOM32m",
                                              "FCOM64m",
                                              "FCOMP32m",
                                              "FCOMP64m",
                                              "MMX_PSADBWirm",
                                              "VPACKSSDWYrm",
                                              "VPACKSSWBYrm",
                                              "VPACKUSDWYrm",
                                              "VPACKUSWBYrm",
                                              "VPALIGNRYrmi",
                                              "VPBLENDWYrmi",
                                              "VPBROADCASTBYrm",
                                              "VPBROADCASTWYrm",
                                              "VPERMILPDYmi",
                                              "VPERMILPDYrm",
                                              "VPERMILPSYmi",
                                              "VPERMILPSYrm",
                                              "VPMOVSXBDYrm",
                                              "VPMOVSXBQYrm",
                                              "VPMOVSXWQYrm",
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
                                              "VUNPCKLPSYrm")>;

def SKLWriteResGroup109 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup109], (instregex "VPABSBYrm",
                                              "VPABSDYrm",
                                              "VPABSWYrm",
                                              "VPADDSBYrm",
                                              "VPADDSWYrm",
                                              "VPADDUSBYrm",
                                              "VPADDUSWYrm",
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
                                              "VPSLLDYrm",
                                              "VPSLLQYrm",
                                              "VPSLLVDYrm",
                                              "VPSLLVQYrm",
                                              "VPSLLWYrm",
                                              "VPSRADYrm",
                                              "VPSRAVDYrm",
                                              "VPSRAWYrm",
                                              "VPSRLDYrm",
                                              "VPSRLQYrm",
                                              "VPSRLVDYrm",
                                              "VPSRLVQYrm",
                                              "VPSRLWYrm",
                                              "VPSUBSBYrm",
                                              "VPSUBSWYrm",
                                              "VPSUBUSBYrm",
                                              "VPSUBUSWYrm")>;

def SKLWriteResGroup110 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup110], (instregex "VANDNPDYrm",
                                              "VANDNPSYrm",
                                              "VANDPDYrm",
                                              "VANDPSYrm",
                                              "VBLENDPDYrmi",
                                              "VBLENDPSYrmi",
                                              "VMASKMOVPDYrm",
                                              "VMASKMOVPSYrm",
                                              "VORPDYrm",
                                              "VORPSYrm",
                                              "VPADDBYrm",
                                              "VPADDDYrm",
                                              "VPADDQYrm",
                                              "VPADDWYrm",
                                              "VPANDNYrm",
                                              "VPANDYrm",
                                              "VPBLENDDYrmi",
                                              "VPMASKMOVDYrm",
                                              "VPMASKMOVQYrm",
                                              "VPORYrm",
                                              "VPSUBBYrm",
                                              "VPSUBDYrm",
                                              "VPSUBQYrm",
                                              "VPSUBWYrm",
                                              "VPXORYrm",
                                              "VXORPDYrm",
                                              "VXORPSYrm")>;

def SKLWriteResGroup111 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup111], (instregex "BLENDVPDrm0",
                                              "BLENDVPSrm0",
                                              "PBLENDVBrm0",
                                              "VBLENDVPDrm",
                                              "VBLENDVPSrm",
                                              "VPBLENDVB(Y?)rm")>;

def SKLWriteResGroup112 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SKLWriteResGroup112], (instregex "MMX_PHADDSWrm",
                                              "MMX_PHSUBSWrm")>;

def SKLWriteResGroup113 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort05]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup113], (instregex "MMX_PHADDDrm",
                                              "MMX_PHADDWrm",
                                              "MMX_PHSUBDrm",
                                              "MMX_PHSUBWrm")>;

def SKLWriteResGroup114 : SchedWriteRes<[SKLPort4,SKLPort5,SKLPort237,SKLPort01]> {
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
def: InstRW<[SKLWriteResGroup115], (instregex "ROR(8|16|32|64)mCL")>;

def SKLWriteResGroup116 : SchedWriteRes<[SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,1,2];
}
def: InstRW<[SKLWriteResGroup116], (instregex "RCL(8|16|32|64)m1",
                                              "RCL(8|16|32|64)mi",
                                              "RCR(8|16|32|64)m1",
                                              "RCR(8|16|32|64)mi")>;

def SKLWriteResGroup117 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKLWriteResGroup117], (instregex "ROL(8|16|32|64)mCL",
                                              "SAR(8|16|32|64)mCL",
                                              "SHL(8|16|32|64)mCL",
                                              "SHR(8|16|32|64)mCL")>;

def SKLWriteResGroup118 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,3];
}
def: InstRW<[SKLWriteResGroup118], (instregex "ADC(8|16|32|64)mi")>;

def SKLWriteResGroup119 : SchedWriteRes<[SKLPort4,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKLWriteResGroup119], (instregex "ADC(8|16|32|64)mr",
                                              "CMPXCHG(8|16|32|64)rm",
                                              "SBB(8|16|32|64)mi",
                                              "SBB(8|16|32|64)mr")>;

def SKLWriteResGroup120 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup120], (instregex "MMX_CVTPI2PSirm",
                                              "MMX_PMADDUBSWrm",
                                              "MMX_PMADDWDirm",
                                              "MMX_PMULHRSWrm",
                                              "MMX_PMULHUWirm",
                                              "MMX_PMULHWirm",
                                              "MMX_PMULLWirm",
                                              "MMX_PMULUDQirm",
                                              "(V?)RCPSSm",
                                              "(V?)RSQRTSSm",
                                              "VTESTPDYrm",
                                              "VTESTPSYrm")>;

def SKLWriteResGroup121 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup121], (instregex "(V?)PCMPGTQrm",
                                              "VPMOVSXBWYrm",
                                              "VPMOVSXDQYrm",
                                              "VPMOVSXWDYrm",
                                              "VPMOVZXWDYrm",
                                              "(V?)PSADBWrm")>;

def SKLWriteResGroup122 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup122], (instregex "(V?)ADDSDrm",
                                              "(V?)ADDSSrm",
                                              "(V?)CMPSDrm",
                                              "(V?)CMPSSrm",
                                              "(V?)MAX(C?)SDrm",
                                              "(V?)MAX(C?)SSrm",
                                              "(V?)MIN(C?)SDrm",
                                              "(V?)MIN(C?)SSrm",
                                              "(V?)MULSDrm",
                                              "(V?)MULSSrm",
                                              "(V?)SUBSDrm",
                                              "(V?)SUBSSrm")>;
def: InstRW<[SKLWriteResGroup122],
            (instregex "VF(N)?M(ADD|SUB)(132|213|231)S(D|S)m")>;

def SKLWriteResGroup123 : SchedWriteRes<[SKLPort23,SKLPort01]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup123], (instregex "MMX_CVTPS2PIirm",
                                              "MMX_CVTTPS2PIirm",
                                              "VCVTPH2PSrm",
                                              "(V?)CVTPS2PDrm")>;

def SKLWriteResGroup124 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup124], (instregex "(V?)DPPDrri")>;

def SKLWriteResGroup125 : SchedWriteRes<[SKLPort23,SKLPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup125], (instregex "VBLENDVPDYrm",
                                              "VBLENDVPSYrm")>;

def SKLWriteResGroup126 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup126], (instregex "(V?)PTESTrm")>;

def SKLWriteResGroup127 : SchedWriteRes<[SKLPort1,SKLPort5,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup127], (instrs IMUL64m, MUL64m, MULX64rm)>;

def SKLWriteResGroup128 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup128], (instregex "(V?)PHADDSWrm",
                                              "(V?)PHSUBSWrm")>;

def SKLWriteResGroup129 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup129], (instregex "(V?)PHADDDrm",
                                              "(V?)PHADDWrm",
                                              "(V?)PHSUBDrm",
                                              "(V?)PHSUBWrm")>;

def SKLWriteResGroup130 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup130], (instregex "SHLD(16|32|64)mri8",
                                              "SHRD(16|32|64)mri8")>;

def SKLWriteResGroup131 : SchedWriteRes<[SKLPort1,SKLPort6,SKLPort23,SKLPort0156]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,1,1];
}
def: InstRW<[SKLWriteResGroup131], (instregex "LAR(16|32|64)rm",
                                              "LSL(16|32|64)rm")>;

def SKLWriteResGroup132 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup132], (instregex "(V?)RCPPSm",
                                              "(V?)RSQRTPSm")>;

def SKLWriteResGroup133 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup133], (instregex "ADD_F32m",
                                              "ADD_F64m",
                                              "ILD_F16m",
                                              "ILD_F32m",
                                              "ILD_F64m",
                                              "SUBR_F32m",
                                              "SUBR_F64m",
                                              "SUB_F32m",
                                              "SUB_F64m",
                                              "VPCMPGTQYrm",
                                              "VPERM2F128rm",
                                              "VPERM2I128rm",
                                              "VPERMDYrm",
                                              "VPERMPDYmi",
                                              "VPERMPSYrm",
                                              "VPERMQYmi",
                                              "VPMOVZXBDYrm",
                                              "VPMOVZXBQYrm",
                                              "VPMOVZXBWYrm",
                                              "VPMOVZXDQYrm",
                                              "VPMOVZXWQYrm",
                                              "VPSADBWYrm")>;

def SKLWriteResGroup134 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup134], (instregex "(V?)ADDPDrm",
                                              "(V?)ADDPSrm",
                                              "(V?)ADDSUBPDrm",
                                              "(V?)ADDSUBPSrm",
                                              "(V?)CMPPDrmi",
                                              "(V?)CMPPSrmi",
                                              "(V?)CVTDQ2PSrm",
                                              "(V?)CVTPH2PSYrm",
                                              "(V?)CVTPS2DQrm",
                                              "(V?)CVTSS2SDrm",
                                              "(V?)CVTTPS2DQrm",
                                              "(V?)MAX(C?)PDrm",
                                              "(V?)MAX(C?)PSrm",
                                              "(V?)MIN(C?)PDrm",
                                              "(V?)MIN(C?)PSrm",
                                              "(V?)MULPDrm",
                                              "(V?)MULPSrm",
                                              "(V?)PHMINPOSUWrm",
                                              "(V?)PMADDUBSWrm",
                                              "(V?)PMADDWDrm",
                                              "(V?)PMULDQrm",
                                              "(V?)PMULHRSWrm",
                                              "(V?)PMULHUWrm",
                                              "(V?)PMULHWrm",
                                              "(V?)PMULLWrm",
                                              "(V?)PMULUDQrm",
                                              "(V?)SUBPDrm",
                                              "(V?)SUBPSrm")>;
def: InstRW<[SKLWriteResGroup134],
            (instregex "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)m")>;

def SKLWriteResGroup137 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup137], (instregex "(V?)MPSADBWrmi")>;

def SKLWriteResGroup138 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup138], (instregex "MMX_CVTPI2PDirm",
                                              "VPTESTYrm")>;

def SKLWriteResGroup139 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort01]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup139], (instregex "(V?)CVTSD2SSrm")>;

def SKLWriteResGroup140 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup140], (instregex "VPHADDSWYrm",
                                              "VPHSUBSWYrm")>;

def SKLWriteResGroup141 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort015]> {
  let Latency = 10;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup141], (instregex "VPHADDDYrm",
                                              "VPHADDWYrm",
                                              "VPHSUBDYrm",
                                              "VPHSUBWYrm")>;

def SKLWriteResGroup142 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort06,SKLPort0156]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup142], (instrs IMUL32rm, MUL32m, MULX32rm)>;

def SKLWriteResGroup143 : SchedWriteRes<[SKLPort4,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 10;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,1,1,3];
}
def: InstRW<[SKLWriteResGroup143], (instregex "XCHG(8|16|32|64)rm")>;

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
def: InstRW<[SKLWriteResGroup145], (instregex "(V?)DIVPS(Y?)rr",
                                              "(V?)DIVSSrr")>;

def SKLWriteResGroup146 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup146], (instregex "MUL_F32m",
                                              "MUL_F64m",
                                              "VRCPPSYm",
                                              "VRSQRTPSYm")>;

def SKLWriteResGroup147 : SchedWriteRes<[SKLPort01,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup147], (instregex "VADDPDYrm",
                                              "VADDPSYrm",
                                              "VADDSUBPDYrm",
                                              "VADDSUBPSYrm",
                                              "VCMPPDYrmi",
                                              "VCMPPSYrmi",
                                              "VCVTDQ2PSYrm",
                                              "VCVTPS2DQYrm",
                                              "VCVTPS2PDYrm",
                                              "VCVTTPS2DQYrm",
                                              "VMAX(C?)PDYrm",
                                              "VMAX(C?)PSYrm",
                                              "VMIN(C?)PDYrm",
                                              "VMIN(C?)PSYrm",
                                              "VMULPDYrm",
                                              "VMULPSYrm",
                                              "VPMADDUBSWYrm",
                                              "VPMADDWDYrm",
                                              "VPMULDQYrm",
                                              "VPMULHRSWYrm",
                                              "VPMULHUWYrm",
                                              "VPMULHWYrm",
                                              "VPMULLWYrm",
                                              "VPMULUDQYrm",
                                              "VSUBPDYrm",
                                              "VSUBPSYrm")>;
def: InstRW<[SKLWriteResGroup147],
            (instregex "VF(N)?M(ADD|SUB|ADDSUB|SUBADD)(132|213|231)P(D|S)Ym")>;

def SKLWriteResGroup149 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup149], (instregex "FICOM16m",
                                              "FICOM32m",
                                              "FICOMP16m",
                                              "FICOMP32m",
                                              "VMPSADBWYrmi")>;

def SKLWriteResGroup150 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup150], (instregex "(V?)CVTDQ2PDrm")>;

def SKLWriteResGroup151 : SchedWriteRes<[SKLPort0,SKLPort23,SKLPort01]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup151], (instregex "(V?)CVTSD2SI64rm",
                                              "(V?)CVTSD2SIrm",
                                              "(V?)CVTSS2SI64rm",
                                              "(V?)CVTSS2SIrm",
                                              "(V?)CVTTSD2SI64rm",
                                              "(V?)CVTTSD2SIrm",
                                              "VCVTTSS2SI64rm",
                                              "(V?)CVTTSS2SIrm")>;

def SKLWriteResGroup152 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort01]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup152], (instregex "CVTPD2DQrm",
                                              "CVTPD2PSrm",
                                              "CVTTPD2DQrm",
                                              "MMX_CVTPD2PIirm",
                                              "MMX_CVTTPD2PIirm")>;

def SKLWriteResGroup153 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[SKLWriteResGroup153], (instregex "SHLD(16|32|64)mrCL",
                                              "SHRD(16|32|64)mrCL")>;

def SKLWriteResGroup154 : SchedWriteRes<[SKLPort1,SKLPort06,SKLPort0156]> {
  let Latency = 11;
  let NumMicroOps = 7;
  let ResourceCycles = [2,3,2];
}
def: InstRW<[SKLWriteResGroup154], (instregex "RCL(16|32|64)rCL",
                                              "RCR(16|32|64)rCL")>;

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
def: InstRW<[SKLWriteResGroup156], (instrs LOOPE, LOOPNE)>;

def SKLWriteResGroup157 : SchedWriteRes<[SKLPort0]> {
  let Latency = 12;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup157], (instregex "(V?)SQRTPS(Y?)r",
                                              "(V?)SQRTSSr")>;

def SKLWriteResGroup159 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup159], (instregex "(V?)HADDPDrm",
                                              "(V?)HADDPSrm",
                                              "(V?)HSUBPDrm",
                                              "(V?)HSUBPSrm")>;

def SKLWriteResGroup160 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort01]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SKLWriteResGroup160], (instregex "CVTTSS2SI64rm")>;

def SKLWriteResGroup162 : SchedWriteRes<[SKLPort5,SKLPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SKLWriteResGroup162], (instregex "ADD_FI16m",
                                              "ADD_FI32m",
                                              "SUBR_FI16m",
                                              "SUBR_FI32m",
                                              "SUB_FI16m",
                                              "SUB_FI32m")>;

def SKLWriteResGroup163 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup163], (instregex "VCVTDQ2PDYrm")>;

def SKLWriteResGroup164 : SchedWriteRes<[SKLPort5,SKLPort01]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SKLWriteResGroup164], (instregex "(V?)DPPS(Y?)rri")>;

def SKLWriteResGroup165 : SchedWriteRes<[SKLPort5,SKLPort01,SKLPort23]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SKLWriteResGroup165], (instregex "VHADDPDYrm",
                                              "VHADDPSYrm",
                                              "VHSUBPDYrm",
                                              "VHSUBPSYrm")>;

def SKLWriteResGroup166 : SchedWriteRes<[SKLPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup166], (instregex "(V?)DIVPD(Y?)rr",
                                              "(V?)DIVSDrr")>;

def SKLWriteResGroup168 : SchedWriteRes<[SKLPort23,SKLPort01]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup168], (instregex "(V?)ROUNDPDm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "(V?)ROUNDPSm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "(V?)ROUNDSDm")>;
def: InstRW<[SKLWriteResGroup168], (instregex "(V?)ROUNDSSm")>;

def SKLWriteResGroup169 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 14;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup169], (instregex "MUL_FI16m",
                                              "MUL_FI32m")>;

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
def: InstRW<[SKLWriteResGroup171], (instregex "DIVR_FPrST0",
                                              "DIVR_FST0r",
                                              "DIVR_FrST0")>;

def SKLWriteResGroup172 : SchedWriteRes<[SKLPort23,SKLPort01]> {
  let Latency = 15;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup172], (instregex "VROUNDPDYm",
                                              "VROUNDPSYm")>;

def SKLWriteResGroup172_2 : SchedWriteRes<[SKLPort23,SKLPort01]> {
  let Latency = 17;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SKLWriteResGroup172_2], (instregex "VPMULLDYrm")>;

def SKLWriteResGroup173 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort01]> {
  let Latency = 15;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SKLWriteResGroup173], (instregex "(V?)DPPDrmi")>;

def SKLWriteResGroup174 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 15;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,5,1,1];
}
def: InstRW<[SKLWriteResGroup174], (instregex "RCL(8|16|32|64)mCL")>;

def SKLWriteResGroup175 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup175], (instregex "(V?)DIVSSrm")>;

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
def: InstRW<[SKLWriteResGroup179], (instregex "(V?)DIVPSrm",
                                              "(V?)SQRTSSm")>;

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
def: InstRW<[SKLWriteResGroup181], (instregex "(V?)SQRTPD(Y?)r",
                                              "(V?)SQRTSDr")>;

def SKLWriteResGroup182 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 18;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup182], (instregex "VDIVPSYrm",
                                              "(V?)SQRTPSm")>;

def SKLWriteResGroup184 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort06,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 8;
  let ResourceCycles = [1,1,1,5];
}
def: InstRW<[SKLWriteResGroup184], (instrs CPUID, RDTSC)>;

def SKLWriteResGroup185 : SchedWriteRes<[SKLPort1,SKLPort23,SKLPort237,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 18;
  let NumMicroOps = 11;
  let ResourceCycles = [2,1,1,4,1,2];
}
def: InstRW<[SKLWriteResGroup185], (instregex "RCR(8|16|32|64)mCL")>;

def SKLWriteResGroup186 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup186], (instregex "(V?)DIVSDrm",
                                              "VSQRTPSYm")>;

def SKLWriteResGroup187 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort01]> {
  let Latency = 19;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,3];
}
def: InstRW<[SKLWriteResGroup187], (instregex "(V?)DPPSrmi")>;

def SKLWriteResGroup189 : SchedWriteRes<[SKLPort0]> {
  let Latency = 20;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SKLWriteResGroup189], (instregex "DIV_FPrST0",
                                              "DIV_FST0r",
                                              "DIV_FrST0")>;

def SKLWriteResGroup190 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup190], (instregex "(V?)DIVPDrm")>;

def SKLWriteResGroup191 : SchedWriteRes<[SKLPort5,SKLPort23,SKLPort01]> {
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
def: InstRW<[SKLWriteResGroup192], (instregex "INSB",
                                              "INSL",
                                              "INSW")>;

def SKLWriteResGroup193 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort0156]> {
  let Latency = 20;
  let NumMicroOps = 10;
  let ResourceCycles = [1,2,7];
}
def: InstRW<[SKLWriteResGroup193], (instregex "MWAITrr")>;

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
def: InstRW<[SKLWriteResGroup196], (instregex "DIV_F32m",
                                              "DIV_F64m")>;

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
def: InstRW<[SKLWriteResGroup197], (instregex "(V?)SQRTSDm")>;

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
def: InstRW<[SKLWriteResGroup199], (instregex "(V?)SQRTPDm")>;

def SKLWriteResGroup201 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 25;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup201], (instregex "VSQRTPDYm")>;

def SKLWriteResGroup202 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 25;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup202], (instregex "DIV_FI16m",
                                              "DIV_FI32m")>;

def SKLWriteResGroup206 : SchedWriteRes<[SKLPort0,SKLPort23]> {
  let Latency = 27;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SKLWriteResGroup206], (instregex "DIVR_F32m",
                                              "DIVR_F64m")>;

def SKLWriteResGroup207 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23,SKLPort0156]> {
  let Latency = 28;
  let NumMicroOps = 8;
  let ResourceCycles = [2,4,1,1];
}
def: InstRW<[SKLWriteResGroup207], (instregex "IDIV(8|16|32|64)m")>;

def SKLWriteResGroup208 : SchedWriteRes<[SKLPort0,SKLPort5,SKLPort23]> {
  let Latency = 30;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SKLWriteResGroup208], (instregex "DIVR_FI16m",
                                              "DIVR_FI32m")>;

def SKLWriteResGroup209 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort23,SKLPort06,SKLPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,3,4,10];
}
def: InstRW<[SKLWriteResGroup209], (instregex "IN(8|16|32)ri",
                                              "IN(8|16|32)rr")>;

def SKLWriteResGroup210 : SchedWriteRes<[SKLPort5,SKLPort6,SKLPort23,SKLPort237,SKLPort06,SKLPort0156]> {
  let Latency = 35;
  let NumMicroOps = 23;
  let ResourceCycles = [1,5,2,1,4,10];
}
def: InstRW<[SKLWriteResGroup210], (instregex "OUT(8|16|32)ir",
                                              "OUT(8|16|32)rr")>;

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
def: InstRW<[SKLWriteResGroup214], (instrs RDTSCP)>;

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
def: InstRW<[SKLWriteResGroup218], (instrs FXRSTOR64)>;

def SKLWriteResGroup219 : SchedWriteRes<[SKLPort0,SKLPort6,SKLPort23,SKLPort05,SKLPort06,SKLPort15,SKLPort0156]> {
  let Latency = 63;
  let NumMicroOps = 90;
  let ResourceCycles = [4,2,33,1,2,1,47];
}
def: InstRW<[SKLWriteResGroup219], (instrs FXRSTOR)>;

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
