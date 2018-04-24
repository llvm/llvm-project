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
  // All x86 instructions are modeled as a single micro-op, and BW can decode 4
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
def BWDivider : ProcResource<1>;
// FP division and sqrt on port 0.
def BWFPDivider : ProcResource<1>;

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
  def : WriteRes<SchedRW.Folded, !listconcat([BWPort23], ExePorts)> {
    let Latency = !add(Lat, LoadLat);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = !add(UOps, 1);
  }
}

// A folded store needs a cycle on port 4 for the store data, and an extra port
// 2/3/7 cycle to recompute the address.
def : WriteRes<WriteRMW, [BWPort237,BWPort4]>;

// Arithmetic.
defm : BWWriteResPair<WriteALU,   [BWPort0156], 1>; // Simple integer ALU op.
defm : BWWriteResPair<WriteIMul,  [BWPort1],   3>; // Integer multiplication.
defm : BWWriteResPair<WriteIDiv, [BWPort0, BWDivider], 25, [1, 10]>;
defm : BWWriteResPair<WriteCRC32, [BWPort1],   3>;
def : WriteRes<WriteIMulH, []> { let Latency = 3; } // Integer multiplication, high part.

def : WriteRes<WriteLEA, [BWPort15]>; // LEA instructions can't fold loads.

defm : BWWriteResPair<WriteCMOV,  [BWPort06], 1>; // Conditional move.
def  : WriteRes<WriteSETCC, [BWPort06]>; // Setcc.
def  : WriteRes<WriteSETCCStore, [BWPort06,BWPort4,BWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

// Bit counts.
defm : BWWriteResPair<WriteBitScan, [BWPort1], 3>;
defm : BWWriteResPair<WriteLZCNT,   [BWPort1], 3>;
defm : BWWriteResPair<WriteTZCNT,   [BWPort1], 3>;
defm : BWWriteResPair<WritePOPCNT,  [BWPort1], 3>;

// Integer shifts and rotates.
defm : BWWriteResPair<WriteShift, [BWPort06],  1>;

// BMI1 BEXTR, BMI2 BZHI
defm : BWWriteResPair<WriteBEXTR, [BWPort06,BWPort15], 2, [1,1], 2>;
defm : BWWriteResPair<WriteBZHI, [BWPort15], 1>;

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

defm : BWWriteResPair<WriteFAdd,   [BWPort1], 3>; // Floating point add/sub.
defm : BWWriteResPair<WriteFCmp,   [BWPort1], 3>; // Floating point compare.
defm : BWWriteResPair<WriteFCom,   [BWPort1], 3>; // Floating point compare to flags.
defm : BWWriteResPair<WriteFMul,   [BWPort0], 5>; // Floating point multiplication.
defm : BWWriteResPair<WriteFDiv,   [BWPort0], 12>; // 10-14 cycles. // Floating point division.
defm : BWWriteResPair<WriteFSqrt,  [BWPort0], 15>; // Floating point square root.
defm : BWWriteResPair<WriteFRcp,   [BWPort0], 5>; // Floating point reciprocal estimate.
defm : BWWriteResPair<WriteFRsqrt, [BWPort0], 5>; // Floating point reciprocal square root estimate.
defm : BWWriteResPair<WriteFMA,    [BWPort01], 5>; // Fused Multiply Add.
defm : BWWriteResPair<WriteFSign,     [BWPort5],  1>; // Floating point fabs/fchs.
defm : BWWriteResPair<WriteFLogic,    [BWPort5],  1>; // Floating point and/or/xor logicals.
defm : BWWriteResPair<WriteFShuffle,  [BWPort5],  1>; // Floating point vector shuffles.
defm : BWWriteResPair<WriteFVarShuffle,  [BWPort5],  1>; // Floating point vector variable shuffles.
defm : BWWriteResPair<WriteFBlend,  [BWPort015],  1>; // Floating point vector blends.
defm : BWWriteResPair<WriteFVarBlend,  [BWPort5], 2, [2], 2, 5>; // Fp vector variable blends.

def  : WriteRes<WriteCvtF2FSt, [BWPort1,BWPort4,BWPort237]> {
  let Latency = 4;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}

// FMA Scheduling helper class.
// class FMASC { X86FoldableSchedWrite Sched = WriteFAdd; }

// Vector integer operations.
def  : WriteRes<WriteVecLoad,        [BWPort23]> { let Latency = 5; }
def  : WriteRes<WriteVecStore,       [BWPort237, BWPort4]>;
def  : WriteRes<WriteVecMove,        [BWPort015]>;

defm : BWWriteResPair<WriteVecALU,   [BWPort15],  1>; // Vector integer ALU op, no logicals.
defm : BWWriteResPair<WriteVecLogic, [BWPort015], 1>; // Vector integer and/or/xor.
defm : BWWriteResPair<WriteVecShift, [BWPort0],  1>; // Vector integer shifts.
defm : BWWriteResPair<WriteVecIMul,  [BWPort0],   5>; // Vector integer multiply.
defm : BWWriteResPair<WritePMULLD,   [BWPort0], 10, [2], 2, 5>; // PMULLD
defm : BWWriteResPair<WriteShuffle,  [BWPort5],  1>; // Vector shuffles.
defm : BWWriteResPair<WriteVarShuffle, [BWPort5],  1>; // Vector variable shuffles.
defm : BWWriteResPair<WriteBlend,  [BWPort5],  1>; // Vector blends.
defm : BWWriteResPair<WriteVarBlend,  [BWPort5], 2, [2], 2, 5>; // Vector variable blends.
defm : BWWriteResPair<WriteMPSAD,  [BWPort0, BWPort5], 7, [1, 2], 3, 5>; // Vector MPSAD.
defm : BWWriteResPair<WritePSADBW,   [BWPort0],   5>; // Vector PSADBW.
defm : BWWriteResPair<WritePHMINPOS, [BWPort0],   5>; // Vector PHMINPOS.

// Vector insert/extract operations.
def : WriteRes<WriteVecInsert, [BWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteVecInsertLd, [BWPort5,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
}

def : WriteRes<WriteVecExtract, [BWPort0,BWPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
}
def : WriteRes<WriteVecExtractSt, [BWPort4,BWPort5,BWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
}

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

// MOVMSK Instructions.
def : WriteRes<WriteFMOVMSK, [BWPort0]> { let Latency = 3; }
def : WriteRes<WriteVecMOVMSK, [BWPort0]> { let Latency = 3; }
def : WriteRes<WriteMMXMOVMSK, [BWPort0]> { let Latency = 1; }

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
defm : BWWriteResPair<WriteFVarShuffle256,  [BWPort5],  3>; // Fp 256-bit width vector variable shuffles.
defm : BWWriteResPair<WriteShuffle256,  [BWPort5],  3>;  // 256-bit width vector shuffles.
defm : BWWriteResPair<WriteVarShuffle256,  [BWPort5],  3>;  // 256-bit width vector variable shuffles.
defm : BWWriteResPair<WriteVarVecShift,  [BWPort0, BWPort5], 2, [2, 1]>; // Variable vector shifts.

// Old microcoded instructions that nobody use.
def : WriteRes<WriteMicrocoded, [BWPort0156]> { let Latency = 100; } // def WriteMicrocoded : SchedWrite;

// Fence instructions.
def : WriteRes<WriteFence,  [BWPort23, BWPort4]>;

// Load/store MXCSR.
def : WriteRes<WriteLDMXCSR, [BWPort0,BWPort23,BWPort0156]> { let Latency = 7; let NumMicroOps = 3; let ResourceCycles = [1,1,1]; }
def : WriteRes<WriteSTMXCSR, [BWPort4,BWPort5,BWPort237]> { let Latency = 2; let NumMicroOps = 3; let ResourceCycles = [1,1,1]; }

// Nop, not very useful expect it provides a model for nops!
def : WriteRes<WriteNop, []>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : BWWriteResPair<WriteFHAdd,   [BWPort1,BWPort5], 5, [1,2], 3>;
defm : BWWriteResPair<WritePHAdd,  [BWPort5,BWPort15], 3, [2,1], 3>;

// Remaining instrs.

def BWWriteResGroup1 : SchedWriteRes<[BWPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup1], (instregex "MMX_MOVD64from64rr",
                                           "MMX_MOVD64grr",
                                           "(V?)MOVPDI2DIrr",
                                           "(V?)MOVPQIto64rr",
                                           "VPSLLVQ(Y?)rr",
                                           "VPSRLVQ(Y?)rr",
                                           "VTESTPD(Y?)rr",
                                           "VTESTPS(Y?)rr")>;

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
def: InstRW<[BWWriteResGroup3], (instregex "MMX_MOVD64rr",
                                           "MMX_MOVD64to64rr",
                                           "MMX_MOVQ2DQrr",
                                           "(V?)MOV64toPQIrr",
                                           "(V?)MOVDI2PDIrr",
                                           "(V?)PSLLDQ(Y?)ri",
                                           "(V?)PSRLDQ(Y?)ri")>;

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
def: InstRW<[BWWriteResGroup5], (instrs FINCSTP, FNOP)>;

def BWWriteResGroup6 : SchedWriteRes<[BWPort06]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup6], (instrs CDQ, CQO)>;
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
                                           "SBB(16|32|64)ri",
                                           "SBB(16|32|64)i",
                                           "SBB(8|16|32|64)rr")>;

def BWWriteResGroup7 : SchedWriteRes<[BWPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup7], (instregex "ANDN(32|64)rr",
                                           "BLSI(32|64)rr",
                                           "BLSMSK(32|64)rr",
                                           "BLSR(32|64)rr")>;

def BWWriteResGroup8 : SchedWriteRes<[BWPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup8], (instregex "MMX_MOVQ64rr",
                                           "VPBLENDD(Y?)rri")>;

def BWWriteResGroup9 : SchedWriteRes<[BWPort0156]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup9], (instregex "LAHF", // TODO: This doesnt match Agner's data
                                           "NOOP",
                                           "SAHF", // TODO: This doesn't match Agner's data
                                           "SGDT64m",
                                           "SIDT64m",
                                           "SLDT64m",
                                           "SMSW16m",
                                           "STRm",
                                           "SYSCALL")>;

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
                                            "MOVNTI_64mr",
                                            "MOVNTImr",
                                            "ST_FP32m",
                                            "ST_FP64m",
                                            "ST_FP80m",
                                            "VEXTRACTF128mr",
                                            "VEXTRACTI128mr",
                                            "(V?)MOVAPD(Y?)mr",
                                            "(V?)MOVAPS(Y?)mr",
                                            "(V?)MOVDQA(Y?)mr",
                                            "(V?)MOVDQU(Y?)mr",
                                            "(V?)MOVHPDmr",
                                            "(V?)MOVHPSmr",
                                            "(V?)MOVLPDmr",
                                            "(V?)MOVLPSmr",
                                            "(V?)MOVNTDQ(V?)mr",
                                            "(V?)MOVNTPD(V?)mr",
                                            "(V?)MOVNTPS(V?)mr",
                                            "(V?)MOVPDI2DImr",
                                            "(V?)MOVPQI2QImr",
                                            "(V?)MOVPQIto64mr",
                                            "(V?)MOVSDmr",
                                            "(V?)MOVSSmr",
                                            "(V?)MOVUPD(Y?)mr",
                                            "(V?)MOVUPS(Y?)mr")>;

def BWWriteResGroup12 : SchedWriteRes<[BWPort01]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[BWWriteResGroup12], (instrs FDECSTP)>;

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
def: InstRW<[BWWriteResGroup15], (instregex "VCVTPH2PS(Y?)rr",
                                            "(V?)CVTPS2PDrr",
                                            "(V?)CVTSS2SDrr",
                                            "(V?)PSLLDrr",
                                            "(V?)PSLLQrr",
                                            "(V?)PSLLWrr",
                                            "(V?)PSRADrr",
                                            "(V?)PSRAWrr",
                                            "(V?)PSRLDrr",
                                            "(V?)PSRLQrr",
                                            "(V?)PSRLWrr",
                                            "(V?)PTESTrr")>;

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
def: InstRW<[BWWriteResGroup19], (instrs BSWAP64r)>;

def BWWriteResGroup19_1 : SchedWriteRes<[BWPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup19_1], (instrs BSWAP32r)>;

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

def BWWriteResGroup22 : SchedWriteRes<[BWPort4,BWPort6,BWPort237]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup22], (instregex "FNSTCW16m")>;

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

def BWWriteResGroup27 : SchedWriteRes<[BWPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup27], (instregex "MMX_CVTPI2PSirr",
                                            "PDEP(32|64)rr",
                                            "PEXT(32|64)rr",
                                            "SHLD(16|32|64)rri8",
                                            "SHRD(16|32|64)rri8",
                                            "(V?)CVTDQ2PS(Y?)rr")>;

def BWWriteResGroup27_16 : SchedWriteRes<[BWPort1, BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup27_16], (instrs IMUL16rri, IMUL16rri8)>;

def BWWriteResGroup28 : SchedWriteRes<[BWPort5]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup28], (instregex "VPBROADCASTBrr",
                                            "VPBROADCASTWrr",
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
def: InstRW<[BWWriteResGroup29], (instregex "(V?)MULPD(Y?)rr",
                                            "(V?)MULPS(Y?)rr",
                                            "(V?)MULSDrr",
                                            "(V?)MULSSrr")>;

def BWWriteResGroup30 : SchedWriteRes<[BWPort0156]> {
  let Latency = 2;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[BWWriteResGroup30], (instrs XADD8rr, XADD16rr, XADD32rr, XADD64rr,
                                         XCHG8rr, XCHG16rr, XCHG32rr, XCHG64rr,
                                         XCHG16ar, XCHG32ar, XCHG64ar)>;

def BWWriteResGroup31 : SchedWriteRes<[BWPort0,BWPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup31], (instregex "VPSLLVD(Y?)rr",
                                            "VPSRAVD(Y?)rr",
                                            "VPSRLVD(Y?)rr")>;

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
def: InstRW<[BWWriteResGroup39], (instregex "(V?)CVTSD2SI64rr",
                                            "(V?)CVTSD2SIrr",
                                            "(V?)CVTSS2SI64rr",
                                            "(V?)CVTSS2SIrr",
                                            "(V?)CVTTSD2SI64rr",
                                            "(V?)CVTTSD2SIrr",
                                            "(V?)CVTTSS2SI64rr",
                                            "(V?)CVTTSS2SIrr")>;

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
def: InstRW<[BWWriteResGroup42], (instregex "MMX_CVTPD2PIirr",
                                            "MMX_CVTPI2PDirr",
                                            "MMX_CVTPS2PIirr",
                                            "MMX_CVTTPD2PIirr",
                                            "MMX_CVTTPS2PIirr",
                                            "(V?)CVTDQ2PDrr",
                                            "(V?)CVTPD2DQrr",
                                            "(V?)CVTPD2PSrr",
                                            "VCVTPS2PHrr",
                                            "(V?)CVTSD2SSrr",
                                            "(V?)CVTSI642SDrr",
                                            "(V?)CVTSI2SDrr",
                                            "(V?)CVTSI2SSrr",
                                            "(V?)CVTTPD2DQrr")>;

def BWWriteResGroup42_16 : SchedWriteRes<[BWPort1,BWPort06,BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
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
                                            "IST_FP64m")>;

def BWWriteResGroup45 : SchedWriteRes<[BWPort0156]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [4];
}
def: InstRW<[BWWriteResGroup45], (instrs FNCLEX)>;

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
def: InstRW<[BWWriteResGroup47], (instregex "(V?)PCMPGTQ(Y?)rr")>;

def BWWriteResGroup49 : SchedWriteRes<[BWPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[BWWriteResGroup49], (instregex "MOVSX(16|32|64)rm16",
                                            "MOVSX(16|32|64)rm32",
                                            "MOVSX(16|32|64)rm8",
                                            "MOVZX(16|32|64)rm16",
                                            "MOVZX(16|32|64)rm8",
                                            "VBROADCASTSSrm",
                                            "(V?)MOVDDUPrm",
                                            "(V?)MOVSHDUPrm",
                                            "(V?)MOVSLDUPrm",
                                            "VPBROADCASTDrm",
                                            "VPBROADCASTQrm")>;

def BWWriteResGroup50 : SchedWriteRes<[BWPort1,BWPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup50], (instregex "(V?)CVTSI642SSrr")>;

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
def: InstRW<[BWWriteResGroup53], (instregex "VMASKMOVPD(Y?)mr",
                                            "VMASKMOVPS(Y?)mr",
                                            "VPMASKMOVD(Y?)mr",
                                            "VPMASKMOVQ(Y?)mr")>;

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
                                            "(V?)ROUNDPD(Y?)r",
                                            "(V?)ROUNDPS(Y?)r",
                                            "(V?)ROUNDSDr",
                                            "(V?)ROUNDSSr")>;

def BWWriteResGroup59 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup59], (instregex "VCVTPH2PS(Y?)rm",
                                            "(V?)CVTPS2PDrm",
                                            "(V?)CVTSS2SDrm",
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
def: InstRW<[BWWriteResGroup63], (instregex "BT(16|32|64)mi8")>;
def: InstRW<[BWWriteResGroup63, ReadAfterLd], (instrs ADC8rm, ADC16rm, ADC32rm, ADC64rm,
                                                      ADCX32rm, ADCX64rm,
                                                      ADOX32rm, ADOX64rm,
                                                      SBB8rm, SBB16rm, SBB32rm, SBB64rm)>;

def BWWriteResGroup64 : SchedWriteRes<[BWPort23,BWPort15]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup64], (instregex "ANDN(32|64)rm",
                                            "BLSI(32|64)rm",
                                            "BLSMSK(32|64)rm",
                                            "BLSR(32|64)rm",
                                            "MOVBE(16|32|64)rm")>;

def BWWriteResGroup65 : SchedWriteRes<[BWPort23,BWPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup65], (instregex "VINSERTF128rm",
                                            "VINSERTI128rm",
                                            "VPBLENDDrmi")>;

def BWWriteResGroup66 : SchedWriteRes<[BWPort23,BWPort0156]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup66], (instrs POP16r, POP32r, POP64r)>;
def: InstRW<[BWWriteResGroup66], (instregex "POP(16|32|64)rmr")>;

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
def: InstRW<[BWWriteResGroup70], (instregex "POP(16|32|64)rmm",
                                            "PUSH(16|32|64)rmm")>;

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

def BWWriteResGroup79 : SchedWriteRes<[BWPort5,BWPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[BWWriteResGroup79], (instregex "MMX_PACKSSDWirm",
                                            "MMX_PACKSSWBirm",
                                            "MMX_PACKUSWBirm",
                                            "VMASKMOVPDrm",
                                            "VMASKMOVPSrm",
                                            "VPMASKMOVDrm",
                                            "VPMASKMOVQrm")>;

def BWWriteResGroup80 : SchedWriteRes<[BWPort23,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[BWWriteResGroup80], (instrs LEAVE, LEAVE64,
                                         SCASB, SCASL, SCASQ, SCASW)>;

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

def BWWriteResGroup84 : SchedWriteRes<[BWPort6,BWPort23,BWPort0156]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup84], (instregex "LRETQ",
                                            "RETQ")>;

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
def: InstRW<[BWWriteResGroup91], (instregex "MMX_CVTPI2PSirm",
                                            "PDEP(32|64)rm",
                                            "PEXT(32|64)rm",
                                            "(V?)CVTDQ2PSrm")>;

def BWWriteResGroup91_16 : SchedWriteRes<[BWPort1, BWPort0156, BWPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup91_16], (instrs IMUL16rmi, IMUL16rmi8)>;

def BWWriteResGroup91_16_2 : SchedWriteRes<[BWPort1, BWPort06, BWPort0156, BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 5;
  let ResourceCycles = [1,1,2,1];
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
def: InstRW<[BWWriteResGroup93], (instregex "(V?)MULPDrm",
                                            "(V?)MULPSrm",
                                            "(V?)MULSDrm",
                                            "(V?)MULSSrm")>;

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
def: InstRW<[BWWriteResGroup99], (instregex "XCHG(8|16|32|64)rm")>;

def BWWriteResGroup100 : SchedWriteRes<[BWPort4,BWPort23,BWPort237,BWPort06,BWPort0156]> {
  let Latency = 8;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,1,2,1];
}
def: InstRW<[BWWriteResGroup100], (instregex "ADC(8|16|32|64)mi",
                                             "CMPXCHG(8|16|32|64)rm",
                                             "ROL(8|16|32|64)mCL",
                                             "SAR(8|16|32|64)mCL",
                                             "SBB(8|16|32|64)mi",
                                             "SHL(8|16|32|64)mCL",
                                             "SHR(8|16|32|64)mCL")>;
def: InstRW<[BWWriteResGroup100, ReadAfterLd], (instrs ADC8mr, ADC16mr, ADC32mr, ADC64mr,
                                                       SBB8mr, SBB16mr, SBB32mr, SBB64mr)>;

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
def: InstRW<[BWWriteResGroup104], (instregex "(V?)DPPDrri")>;

def BWWriteResGroup105 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup105], (instregex "(V?)CVTSD2SI64rm",
                                             "(V?)CVTSD2SIrm",
                                             "(V?)CVTSS2SI64rm",
                                             "(V?)CVTSS2SIrm",
                                             "(V?)CVTTSD2SI64rm",
                                             "(V?)CVTTSD2SIrm",
                                             "VCVTTSS2SI64rm",
                                             "(V?)CVTTSS2SIrm")>;

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
def: InstRW<[BWWriteResGroup107], (instregex "CVTPD2DQrm",
                                             "CVTPD2PSrm",
                                             "CVTTPD2DQrm",
                                             "MMX_CVTPD2PIirm",
                                             "MMX_CVTPI2PDirm",
                                             "MMX_CVTTPD2PIirm",
                                             "(V?)CVTDQ2PDrm",
                                             "(V?)CVTSD2SSrm")>;

def BWWriteResGroup108 : SchedWriteRes<[BWPort5,BWPort23,BWPort015]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup108], (instregex "VPBROADCASTB(Y?)rm",
                                             "VPBROADCASTW(Y?)rm")>;

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

def BWWriteResGroup115 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup115], (instregex "(V?)PCMPGTQrm")>;

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

def BWWriteResGroup122 : SchedWriteRes<[BWPort0,BWFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,5];
}
def: InstRW<[BWWriteResGroup122], (instregex "(V?)DIVPSrr")>;

def BWWriteResGroup122_1 : SchedWriteRes<[BWPort0,BWFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,3]; // Really 2.5 cycle throughput
}
def: InstRW<[BWWriteResGroup122_1], (instregex "(V?)DIVSSrr")>;

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
def: InstRW<[BWWriteResGroup127], (instregex "(V?)ROUNDPDm",
                                             "(V?)ROUNDPSm",
                                             "(V?)ROUNDSDm",
                                             "(V?)ROUNDSSm")>;

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

def BWWriteResGroup137 : SchedWriteRes<[BWPort0,BWFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,7];
}
def: InstRW<[BWWriteResGroup137], (instregex "(V?)SQRTPSr")>;

def BWWriteResGroup137_1 : SchedWriteRes<[BWPort0,BWFPDivider]> {
  let Latency = 11;
  let NumMicroOps = 1;
  let ResourceCycles = [1,4];
}
def: InstRW<[BWWriteResGroup137_1], (instregex "(V?)SQRTSSr")>;

def BWWriteResGroup138 : SchedWriteRes<[BWPort0,BWPort5,BWPort23]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1,7];
}
def: InstRW<[BWWriteResGroup138], (instregex "VMPSADBWYrmi")>;

def BWWriteResGroup139 : SchedWriteRes<[BWPort0,BWFPDivider]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1,8];
}
def: InstRW<[BWWriteResGroup139], (instregex "(V?)DIVPDrr")>;

def BWWriteResGroup139_1 : SchedWriteRes<[BWPort0,BWFPDivider]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1,4];
}
def: InstRW<[BWWriteResGroup139_1], (instregex "(V?)DIVSDrr")>;

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
def: InstRW<[BWWriteResGroup142], (instregex "(V?)DPPS(Y?)rri")>;

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

def BWWriteResGroup149 : SchedWriteRes<[BWPort1,BWPort23,BWPort237,BWPort06,BWPort15,BWPort0156]> {
  let Latency = 15;
  let NumMicroOps = 10;
  let ResourceCycles = [1,1,1,4,1,2];
}
def: InstRW<[BWWriteResGroup149], (instregex "RCL(8|16|32|64)mCL")>;

def BWWriteResGroup150 : SchedWriteRes<[BWPort0,BWPort23,BWFPDivider]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,5];
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

def BWWriteResGroup155 : SchedWriteRes<[BWPort0,BWPort015,BWFPDivider]> {
  let Latency = 17;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,10];
}
def: InstRW<[BWWriteResGroup155], (instregex "VDIVPSYrr")>;

def BWWriteResGroup156 : SchedWriteRes<[BWPort0,BWPort23,BWPort015]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[BWWriteResGroup156], (instregex "VRCPPSYm",
                                             "VRSQRTPSYm")>;

def BWWriteResGroup157 : SchedWriteRes<[BWPort0,BWPort23,BWFPDivider]> {
  let Latency = 16;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,7];
}
def: InstRW<[BWWriteResGroup157], (instregex "(V?)SQRTPSm",
                                             "(V?)SQRTSSm")>;

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

def BWWriteResGroup161 : SchedWriteRes<[BWPort0,BWPort23,BWFPDivider]> {
  let Latency = 19;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,8];
}
def: InstRW<[BWWriteResGroup161], (instregex "(V?)DIVPDrm",
                                             "(V?)DIVSDrm")>;

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
                                             "DIV_FrST0")>;

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

def BWWriteResGroup168 : SchedWriteRes<[BWPort0,BWFPDivider]> {
  let Latency = 16;
  let NumMicroOps = 1;
  let ResourceCycles = [1,14];
}
def: InstRW<[BWWriteResGroup168], (instregex "(V?)SQRTPDr")>;

def BWWriteResGroup168_1 : SchedWriteRes<[BWPort0,BWFPDivider]> {
  let Latency = 16;
  let NumMicroOps = 1;
  let ResourceCycles = [1,8];
}
def: InstRW<[BWWriteResGroup168_1], (instregex "(V?)SQRTSDr")>;

def BWWriteResGroup169 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup169], (instregex "DIV_F32m",
                                             "DIV_F64m")>;

def BWWriteResGroup170 : SchedWriteRes<[BWPort0,BWPort015,BWFPDivider]> {
  let Latency = 21;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,14];
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

def BWWriteResGroup173 : SchedWriteRes<[BWPort0,BWPort015,BWFPDivider]> {
  let Latency = 23;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,16];
}
def: InstRW<[BWWriteResGroup173], (instregex "VDIVPDYrr")>;

def BWWriteResGroup174 : SchedWriteRes<[BWPort0,BWPort23,BWPort015,BWFPDivider]> {
  let Latency = 23;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,10];
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

def BWWriteResGroup179 : SchedWriteRes<[BWPort0,BWPort23,BWFPDivider]> {
  let Latency = 21;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1,14];
}
def: InstRW<[BWWriteResGroup179], (instregex "(V?)SQRTPDm",
                                             "(V?)SQRTSDm")>;

def BWWriteResGroup180 : SchedWriteRes<[BWPort0,BWPort23]> {
  let Latency = 26;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[BWWriteResGroup180], (instregex "DIVR_F32m",
                                             "DIVR_F64m")>;

def BWWriteResGroup181 : SchedWriteRes<[BWPort0,BWPort23,BWPort015,BWFPDivider]> {
  let Latency = 27;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,14];
}
def: InstRW<[BWWriteResGroup181], (instregex "VSQRTPSYm")>;

def BWWriteResGroup182 : SchedWriteRes<[BWPort0,BWPort1,BWPort23]> {
  let Latency = 29;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[BWWriteResGroup182], (instregex "DIVR_FI16m",
                                             "DIVR_FI32m")>;

def BWWriteResGroup183 : SchedWriteRes<[BWPort0,BWPort23,BWPort015,BWFPDivider]> {
  let Latency = 29;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,16];
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

def BWWriteResGroup189 : SchedWriteRes<[BWPort0,BWPort015,BWFPDivider]> {
  let Latency = 29;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1,28];
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

def BWWriteResGroup195 : SchedWriteRes<[BWPort0,BWPort23,BWPort015,BWFPDivider]> {
  let Latency = 35;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1,28];
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
def: InstRW<[BWWriteResGroup200], (instrs FNINIT)>;

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

