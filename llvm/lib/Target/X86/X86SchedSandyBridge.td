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
                          list<ProcResourceKind> ExePorts,
                          int Lat, list<int> Res = [1], int UOps = 1> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, ExePorts> {
    let Latency = Lat;
    let ResourceCycles = Res;
    let NumMicroOps = UOps;
  }

  // Memory variant also uses a cycle on port 2/3 and adds 4 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, !listconcat([SBPort23], ExePorts)> {
    let Latency = !add(Lat, 4);
    let ResourceCycles = !listconcat([1], Res);
    let NumMicroOps = UOps;
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [SBPort4]>;

def : WriteRes<WriteStore, [SBPort23, SBPort4]>;
def : WriteRes<WriteLoad,  [SBPort23]> { let Latency = 4; }
def : WriteRes<WriteMove,  [SBPort015]>;
def : WriteRes<WriteZero,  []>;

defm : SBWriteResPair<WriteALU,   [SBPort015], 1>;
defm : SBWriteResPair<WriteIMul,  [SBPort1],   3>;
defm : SBWriteResPair<WriteIDiv, [SBPort0, SBDivider], 25, [1, 10]>;
def  : WriteRes<WriteIMulH, []> { let Latency = 3; }

defm : SBWriteResPair<WriteShift, [SBPort05],  1>;
defm : SBWriteResPair<WriteJump,  [SBPort5],   1>;

// This is for simple LEAs with one or two input operands.
// The complex ones can only execute on port 1, and they require two cycles on
// the port to read all inputs. We don't model that.
def : WriteRes<WriteLEA, [SBPort15]>;

// Scalar and vector floating point.
def  : WriteRes<WriteFStore,       [SBPort23, SBPort4]>;
def  : WriteRes<WriteFLoad,        [SBPort23]> { let Latency = 6; }
def  : WriteRes<WriteFMove,        [SBPort5]>;

defm : SBWriteResPair<WriteFAdd,   [SBPort1], 3>;
defm : SBWriteResPair<WriteFMul,   [SBPort0], 5>;
defm : SBWriteResPair<WriteFDiv,   [SBPort0], 24>;
defm : SBWriteResPair<WriteFRcp,   [SBPort0], 5>;
defm : SBWriteResPair<WriteFRsqrt, [SBPort0], 5>;
defm : SBWriteResPair<WriteFSqrt,  [SBPort0], 14>;
defm : SBWriteResPair<WriteCvtF2I, [SBPort1], 3>;
defm : SBWriteResPair<WriteCvtI2F, [SBPort1], 4>;
defm : SBWriteResPair<WriteCvtF2F, [SBPort1], 3>;
defm : SBWriteResPair<WriteFShuffle, [SBPort5],  1>;
defm : SBWriteResPair<WriteFBlend,  [SBPort05],  1>;
defm : SBWriteResPair<WriteFVarBlend,  [SBPort0, SBPort5], 2>;

// Vector integer operations.
def  : WriteRes<WriteVecStore,       [SBPort23, SBPort4]>;
def  : WriteRes<WriteVecLoad,        [SBPort23]> { let Latency = 6; }
def  : WriteRes<WriteVecMove,        [SBPort05]>;

defm : SBWriteResPair<WriteVecShift, [SBPort5], 1>;
defm : SBWriteResPair<WriteVecLogic, [SBPort5], 1>;
defm : SBWriteResPair<WriteVecALU,   [SBPort1], 3>;
defm : SBWriteResPair<WriteVecIMul,  [SBPort0], 5>;
defm : SBWriteResPair<WriteShuffle,  [SBPort5], 1>;
defm : SBWriteResPair<WriteBlend,   [SBPort15], 1>;
defm : SBWriteResPair<WriteVarBlend, [SBPort1, SBPort5], 2>;
defm : SBWriteResPair<WriteMPSAD, [SBPort0, SBPort15], 5, [1,2], 3>;

////////////////////////////////////////////////////////////////////////////////
// Horizontal add/sub  instructions.
////////////////////////////////////////////////////////////////////////////////

defm : SBWriteResPair<WriteFHAdd,   [SBPort1], 3>;
defm : SBWriteResPair<WritePHAdd,   [SBPort15], 1>;

// String instructions.

// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [SBPort0]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [SBPort0, SBPort23]> {
  let Latency = 17;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
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
defm : SBWriteResPair<WriteFShuffle256, [SBPort0],  1>;
defm : SBWriteResPair<WriteShuffle256, [SBPort0],  1>;
defm : SBWriteResPair<WriteVarVecShift, [SBPort0],  1>;
defm : SBWriteResPair<WriteFMA, [SBPort01],  5>;

// Remaining SNB instrs.

def SBWriteResGroup0 : SchedWriteRes<[SBPort0]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup0], (instregex "(V?)CVTSS2SDrr",
                                           "(V?)PSLLDri",
                                           "(V?)PSLLQri",
                                           "(V?)PSLLWri",
                                           "(V?)PSRADri",
                                           "(V?)PSRAWri",
                                           "(V?)PSRLDri",
                                           "(V?)PSRLQri",
                                           "(V?)PSRLWri",
                                           "VTESTPDYrr",
                                           "VTESTPDrr",
                                           "VTESTPSYrr",
                                           "VTESTPSrr")>;

def SBWriteResGroup1 : SchedWriteRes<[SBPort1]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup1], (instregex "COMP_FST0r",
                                           "COM_FST0r",
                                           "UCOM_FPr",
                                           "UCOM_Fr")>;

def SBWriteResGroup2 : SchedWriteRes<[SBPort5]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup2], (instrs LOOP, LOOPE, LOOPNE)>;
def: InstRW<[SBWriteResGroup2], (instregex "FDECSTP",
                                           "FFREE",
                                           "FINCSTP",
                                           "FNOP",
                                           "INSERTPSrr",
                                           "J(A|AE|B|BE|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)_1",
                                           "J(A|AE|B|BE|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)_4",
                                           "JMP(16|32|64)r",
                                           "JMP_1",
                                           "JMP_4",
                                           "LD_Frr",
                                           "RETQ",
                                           "ST_FPrr",
                                           "ST_Frr",
                                           "VANDNPDYrr",
                                           "(V?)ANDNPDrr",
                                           "VANDNPSYrr",
                                           "(V?)ANDNPSrr",
                                           "VANDPDYrr",
                                           "(V?)ANDPDrr",
                                           "VANDPSYrr",
                                           "(V?)ANDPSrr",
                                           "VEXTRACTF128rr",
                                           "VINSERTF128rr",
                                           "VINSERTPSrr",
                                           "(V?)MOV64toPQIrr",
                                           "VMOVAPDYrr",
                                           "(V?)MOVAPDrr",
                                           "VMOVAPSYrr",
                                           "(V?)MOVAPSrr",
                                           "VMOVDDUPYrr",
                                           "(V?)MOVDDUPrr",
                                           "(V?)MOVDI2PDIrr",
                                           "(V?)MOVHLPSrr",
                                           "(V?)MOVLHPSrr",
                                           "(V?)MOVSDrr",
                                           "VMOVSHDUPYrr",
                                           "(V?)MOVSHDUPrr",
                                           "VMOVSLDUPYrr",
                                           "(V?)MOVSLDUPrr",
                                           "(V?)MOVSSrr",
                                           "VMOVUPDYrr",
                                           "(V?)MOVUPDrr",
                                           "VMOVUPSYrr",
                                           "(V?)MOVUPSrr",
                                           "VORPDYrr",
                                           "(V?)ORPDrr",
                                           "VORPSYrr",
                                           "(V?)ORPSrr",
                                           "VPERM2F128rr",
                                           "VPERMILPDYri",
                                           "VPERMILPDYrr",
                                           "VPERMILPDri",
                                           "VPERMILPDrr",
                                           "VPERMILPSYri",
                                           "VPERMILPSYrr",
                                           "VPERMILPSri",
                                           "VPERMILPSrr",
                                           "VSHUFPDYrri",
                                           "(V?)SHUFPDrri",
                                           "VSHUFPSYrri",
                                           "(V?)SHUFPSrri",
                                           "VUNPCKHPDYrr",
                                           "(V?)UNPCKHPDrr",
                                           "VUNPCKHPSYrr",
                                           "(V?)UNPCKHPSrr",
                                           "VUNPCKLPDYrr",
                                           "(V?)UNPCKLPDrr",
                                           "VUNPCKLPSYrr",
                                           "(V?)UNPCKLPSrr",
                                           "VXORPDYrr",
                                           "(V?)XORPDrr",
                                           "VXORPSYrr",
                                           "(V?)XORPSrr")>;

def SBWriteResGroup3 : SchedWriteRes<[SBPort01]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup3], (instregex "LEA(16|32|64)(_32)?r")>;

def SBWriteResGroup4 : SchedWriteRes<[SBPort05]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup4], (instregex "BT(16|32|64)ri8",
                                           "BT(16|32|64)rr",
                                           "BTC(16|32|64)ri8",
                                           "BTC(16|32|64)rr",
                                           "BTR(16|32|64)ri8",
                                           "BTR(16|32|64)rr",
                                           "BTS(16|32|64)ri8",
                                           "BTS(16|32|64)rr",
                                           "CDQ",
                                           "CQO",
                                           "LAHF",
                                           "SAHF",
                                           "SAR(8|16|32|64)ri",
                                           "SAR(8|16|32|64)r1",
                                           "SET(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)r",
                                           "SHL(8|16|32|64)ri",
                                           "SHL(8|16|32|64)r1",
                                           "SHR(8|16|32|64)ri",
                                           "SHR(8|16|32|64)r1",
                                           "VBLENDPDYrri",
                                           "(V?)BLENDPDrri",
                                           "VBLENDPSYrri",
                                           "(V?)BLENDPSrri",
                                           "VMOVDQAYrr",
                                           "VMOVDQArr",
                                           "VMOVDQUYrr",
                                           "VMOVDQUrr")>;

def SBWriteResGroup5 : SchedWriteRes<[SBPort15]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup5], (instregex "MMX_PABSBrr",
                                           "MMX_PABSDrr",
                                           "MMX_PABSWrr",
                                           "MMX_PADDQirr",
                                           "MMX_PALIGNRrri",
                                           "MMX_PSHUFBrr",
                                           "MMX_PSIGNBrr",
                                           "MMX_PSIGNDrr",
                                           "MMX_PSIGNWrr",
                                           "(V?)PABSBrr",
                                           "(V?)PABSDrr",
                                           "(V?)PABSWrr",
                                           "(V?)PACKSSDWrr",
                                           "(V?)PACKSSWBrr",
                                           "(V?)PACKUSDWrr",
                                           "(V?)PACKUSWBrr",
                                           "(V?)PADDBrr",
                                           "(V?)PADDDrr",
                                           "(V?)PADDQrr",
                                           "(V?)PADDSBrr",
                                           "(V?)PADDSWrr",
                                           "(V?)PADDUSBrr",
                                           "(V?)PADDUSWrr",
                                           "(V?)PADDWrr",
                                           "(V?)PALIGNRrri",
                                           "(V?)PAVGBrr",
                                           "(V?)PAVGWrr",
                                           "(V?)PBLENDWrri",
                                           "(V?)PCMPEQBrr",
                                           "(V?)PCMPEQDrr",
                                           "(V?)PCMPEQQrr",
                                           "(V?)PCMPEQWrr",
                                           "(V?)PCMPGTBrr",
                                           "(V?)PCMPGTDrr",
                                           "(V?)PCMPGTWrr",
                                           "(V?)PMAXSBrr",
                                           "(V?)PMAXSDrr",
                                           "(V?)PMAXSWrr",
                                           "(V?)PMAXUBrr",
                                           "(V?)PMAXUDrr",
                                           "(V?)PMAXUWrr",
                                           "(V?)PMINSBrr",
                                           "(V?)PMINSDrr",
                                           "(V?)PMINSWrr",
                                           "(V?)PMINUBrr",
                                           "(V?)PMINUDrr",
                                           "(V?)PMINUWrr",
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
                                           "(V?)PSHUFBrr",
                                           "(V?)PSHUFDri",
                                           "(V?)PSHUFHWri",
                                           "(V?)PSHUFLWri",
                                           "(V?)PSIGNBrr",
                                           "(V?)PSIGNDrr",
                                           "(V?)PSIGNWrr",
                                           "(V?)PSLLDQri",
                                           "(V?)PSRLDQri",
                                           "(V?)PSUBBrr",
                                           "(V?)PSUBDrr",
                                           "(V?)PSUBQrr",
                                           "(V?)PSUBSBrr",
                                           "(V?)PSUBSWrr",
                                           "(V?)PSUBUSBrr",
                                           "(V?)PSUBUSWrr",
                                           "(V?)PSUBWrr",
                                           "(V?)PUNPCKHBWrr",
                                           "(V?)PUNPCKHDQrr",
                                           "(V?)PUNPCKHQDQrr",
                                           "(V?)PUNPCKHWDrr",
                                           "(V?)PUNPCKLBWrr",
                                           "(V?)PUNPCKLDQrr",
                                           "(V?)PUNPCKLQDQrr",
                                           "(V?)PUNPCKLWDrr")>;

def SBWriteResGroup6 : SchedWriteRes<[SBPort015]> {
  let Latency = 1;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup6], (instrs CWDE)>;
def: InstRW<[SBWriteResGroup6], (instregex "ADD(8|16|32|64)ri",
                                           "ADD(8|16|32|64)rr",
                                           "ADD(8|16|32|64)i",
                                           "AND(8|16|32|64)ri",
                                           "AND(8|16|32|64)rr",
                                           "AND(8|16|32|64)i",
                                           "CBW",
                                           "CMC",
                                           "CMP(8|16|32|64)ri",
                                           "CMP(8|16|32|64)rr",
                                           "CMP(8|16|32|64)i",
                                           "DEC(8|16|32|64)r",
                                           "INC(8|16|32|64)r",
                                           "MMX_MOVD64from64rr",
                                           "MMX_MOVQ2DQrr",
                                           "MOV(8|16|32|64)rr",
                                           "MOV(8|16|32|64)ri",
                                           "MOVDQArr",
                                           "MOVDQUrr",
                                           "MOVSX(16|32|64)rr16",
                                           "MOVSX(16|32|64)rr32",
                                           "MOVSX(16|32|64)rr8",
                                           "MOVZX(16|32|64)rr16",
                                           "MOVZX(16|32|64)rr8",
                                           "NEG(8|16|32|64)r",
                                           "NOT(8|16|32|64)r",
                                           "OR(8|16|32|64)ri",
                                           "OR(8|16|32|64)rr",
                                           "OR(8|16|32|64)i",
                                           "STC",
                                           "SUB(8|16|32|64)ri",
                                           "SUB(8|16|32|64)rr",
                                           "SUB(8|16|32|64)i",
                                           "TEST(8|16|32|64)rr",
                                           "TEST(8|16|32|64)i",
                                           "TEST(8|16|32|64)ri",
                                           "(V?)MOVPQI2QIrr",
                                           "VMOVZPQILo2PQIrr",
                                           "(V?)PANDNrr",
                                           "(V?)PANDrr",
                                           "(V?)PORrr",
                                           "(V?)PXORrr",
                                           "XOR(8|16|32|64)ri",
                                           "XOR(8|16|32|64)rr",
                                           "XOR(8|16|32|64)i")>;

def SBWriteResGroup7 : SchedWriteRes<[SBPort0]> {
  let Latency = 2;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup7], (instregex "(V?)PMOVMSKBrr",
                                           "VMOVMSKPDYrr",
                                           "(V?)MOVMSKPDrr",
                                           "VMOVMSKPSYrr",
                                           "(V?)MOVMSKPSrr",
                                           "(V?)MOVPDI2DIrr",
                                           "(V?)MOVPQIto64rr")>;

def SBWriteResGroup9 : SchedWriteRes<[SBPort05]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup9], (instregex "BLENDVPDrr0",
                                           "BLENDVPSrr0",
                                           "ROL(8|16|32|64)r1",
                                           "ROL(8|16|32|64)ri",
                                           "ROR(8|16|32|64)r1",
                                           "ROR(8|16|32|64)ri",
                                           "SET(A|BE)r",
                                           "VBLENDVPDYrr",
                                           "VBLENDVPDrr",
                                           "VBLENDVPSYrr",
                                           "VBLENDVPSrr")>;

def SBWriteResGroup10 : SchedWriteRes<[SBPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup10], (instregex "PBLENDVBrr0",
                                            "VPBLENDVBrr")>;

def SBWriteResGroup11 : SchedWriteRes<[SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [2];
}
def: InstRW<[SBWriteResGroup11], (instregex "SCASB",
                                            "SCASL",
                                            "SCASQ",
                                            "SCASW")>;

def SBWriteResGroup12 : SchedWriteRes<[SBPort0,SBPort1]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup12], (instregex "(V?)COMISDrr",
                                            "(V?)COMISSrr",
                                            "(V?)UCOMISDrr",
                                            "(V?)UCOMISSrr")>;

def SBWriteResGroup13 : SchedWriteRes<[SBPort0,SBPort5]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup13], (instregex "VCVTPS2PDYrr",
                                            "(V?)CVTPS2PDrr",
                                            "VPTESTYrr",
                                            "(V?)PTESTrr")>;

def SBWriteResGroup14 : SchedWriteRes<[SBPort0,SBPort15]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup14], (instregex "(V?)PSLLDrr",
                                            "(V?)PSLLQrr",
                                            "(V?)PSLLWrr",
                                            "(V?)PSRADrr",
                                            "(V?)PSRAWrr",
                                            "(V?)PSRLDrr",
                                            "(V?)PSRLQrr",
                                            "(V?)PSRLWrr")>;

def SBWriteResGroup15 : SchedWriteRes<[SBPort0,SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup15], (instrs CWD)>;
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
def: InstRW<[SBWriteResGroup17], (instregex "(V?)PINSRBrr",
                                            "(V?)PINSRDrr",
                                            "(V?)PINSRQrr",
                                            "(V?)PINSRWrr")>;

def SBWriteResGroup18 : SchedWriteRes<[SBPort5,SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup18], (instrs JCXZ, JECXZ, JRCXZ)>;
def: InstRW<[SBWriteResGroup18], (instregex "MMX_MOVDQ2Qrr")>;

def SBWriteResGroup19 : SchedWriteRes<[SBPort05,SBPort015]> {
  let Latency = 2;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup19], (instregex "ADC(8|16|32|64)ri",
                                            "ADC(8|16|32|64)rr",
                                            "ADC(8|16|32|64)i",
                                            "CMOV(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)(16|32|64)rr",
                                            "SBB(8|16|32|64)ri",
                                            "SBB(8|16|32|64)rr",
                                            "SBB(8|16|32|64)i",
                                            "SHLD(16|32|64)rri8",
                                            "SHRD(16|32|64)rri8")>;

def SBWriteResGroup20 : SchedWriteRes<[SBPort0]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup20], (instregex "MMX_PMADDUBSWrr",
                                            "MMX_PMADDWDirr",
                                            "MMX_PMULHRSWrr",
                                            "MMX_PMULHUWirr",
                                            "MMX_PMULHWirr",
                                            "MMX_PMULLWirr",
                                            "MMX_PMULUDQirr",
                                            "MMX_PSADBWirr",
                                            "(V?)PMADDUBSWrr",
                                            "(V?)PMADDWDrr",
                                            "(V?)PMULDQrr",
                                            "(V?)PMULHRSWrr",
                                            "(V?)PMULHUWrr",
                                            "(V?)PMULHWrr",
                                            "(V?)PMULLDrr",
                                            "(V?)PMULLWrr",
                                            "(V?)PMULUDQrr",
                                            "(V?)PSADBWrr")>;

def SBWriteResGroup21 : SchedWriteRes<[SBPort1]> {
  let Latency = 3;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup21], (instrs MUL8r, IMUL16rr, IMUL32rr, IMUL32rri, IMUL32rri8, IMUL64rr, IMUL64rri32, IMUL64rri8)>;
def: InstRW<[SBWriteResGroup21], (instregex "ADD_FPrST0",
                                            "ADD_FST0r",
                                            "ADD_FrST0",
                                            "BSF(16|32|64)rr",
                                            "BSR(16|32|64)rr",
                                            "CRC32r(16|32|64)r8",
                                            "CRC32r(16|32|64)r64",
                                            "MMX_CVTPI2PSirr",
                                            "MMX_CVTPS2PIirr",
                                            "MMX_CVTTPS2PIirr",
                                            "POPCNT(16|32|64)rr",
                                            "PUSHFS64",
                                            "SUBR_FPrST0",
                                            "SUBR_FST0r",
                                            "SUBR_FrST0",
                                            "SUB_FPrST0",
                                            "SUB_FST0r",
                                            "SUB_FrST0",
                                            "VADDPDYrr",
                                            "(V?)ADDPDrr",
                                            "VADDPSYrr",
                                            "(V?)ADDPSrr",
                                            "(V?)ADDSDrr",
                                            "(V?)ADDSSrr",
                                            "VADDSUBPDYrr",
                                            "(V?)ADDSUBPDrr",
                                            "VADDSUBPSYrr",
                                            "(V?)ADDSUBPSrr",
                                            "VCMPPDYrri",
                                            "(V?)CMPPDrri",
                                            "VCMPPSYrri",
                                            "(V?)CMPPSrri",
                                            "(V?)CMPSDrr",
                                            "(V?)CMPSSrr",
                                            "VCVTDQ2PSYrr",
                                            "(V?)CVTDQ2PSrr",
                                            "VCVTPS2DQYrr",
                                            "(V?)CVTPS2DQrr",
                                            "VCVTTPS2DQYrr",
                                            "(V?)CVTTPS2DQrr",
                                            "VMAX(C?)PDYrr",
                                            "(V?)MAX(C?)PDrr",
                                            "VMAX(C?)PSYrr",
                                            "(V?)MAX(C?)PSrr",
                                            "(V?)MAX(C?)SDrr",
                                            "(V?)MAX(C?)SSrr",
                                            "VMIN(C?)PDYrr",
                                            "(V?)MIN(C?)PDrr",
                                            "VMIN(C?)PSYrr",
                                            "(V?)MIN(C?)PSrr",
                                            "(V?)MIN(C?)SDrr",
                                            "(V?)MIN(C?)SSrr",
                                            "(V?)ROUNDPDr",
                                            "(V?)ROUNDPSr",
                                            "(V?)ROUNDSDr",
                                            "(V?)ROUNDSSr",
                                            "VROUNDPDYr",
                                            "VROUNDPSYr",
                                            "VSUBPDYrr",
                                            "(V?)SUBPDrr",
                                            "VSUBPSYrr",
                                            "(V?)SUBPSrr",
                                            "(V?)SUBSDrr",
                                            "(V?)SUBSSrr")>;

def SBWriteResGroup21_16i : SchedWriteRes<[SBPort1, SBPort015]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup21_16i], (instrs IMUL16rri, IMUL16rri8)>;

def SBWriteResGroup22 : SchedWriteRes<[SBPort0,SBPort5]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup22], (instregex "(V?)EXTRACTPSrr")>;

def SBWriteResGroup23 : SchedWriteRes<[SBPort0,SBPort15]> {
  let Latency = 3;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup23], (instregex "(V?)PEXTRBrr",
                                            "(V?)PEXTRDrr",
                                            "(V?)PEXTRQrr",
                                            "(V?)PEXTRWrr")>;

def SBWriteResGroup23_2 : SchedWriteRes<[SBPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SBWriteResGroup23_2], (instregex "ROL(8|16|32|64)rCL",
                                              "ROR(8|16|32|64)rCL",
                                              "SAR(8|16|32|64)rCL",
                                              "SHL(8|16|32|64)rCL",
                                              "SHR(8|16|32|64)rCL")>;

def SBWriteResGroup24 : SchedWriteRes<[SBPort15]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SBWriteResGroup24], (instregex "MMX_PHADDDrr",
                                            "MMX_PHADDSWrr",
                                            "MMX_PHADDWrr",
                                            "MMX_PHSUBDrr",
                                            "MMX_PHSUBSWrr",
                                            "MMX_PHSUBWrr",
                                            "(V?)PHADDDrr",
                                            "(V?)PHADDSWrr",
                                            "(V?)PHADDWrr",
                                            "(V?)PHSUBDrr",
                                            "(V?)PHSUBSWrr",
                                            "(V?)PHSUBWrr")>;

def SBWriteResGroup25 : SchedWriteRes<[SBPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [3];
}
def: InstRW<[SBWriteResGroup25], (instregex "LEAVE64",
                                            "XADD(8|16|32|64)rr")>;

def SBWriteResGroup25_2 : SchedWriteRes<[SBPort5,SBPort05]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup25_2], (instregex "CMOVBE_F",
                                              "CMOVB_F",
                                              "CMOVE_F",
                                              "CMOVNBE_F",
                                              "CMOVNB_F",
                                              "CMOVNE_F",
                                              "CMOVNP_F",
                                              "CMOVP_F")>;

def SBWriteResGroup26 : SchedWriteRes<[SBPort05,SBPort015]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup26], (instregex "CMOV(A|BE)(16|32|64)rr")>;

def SBWriteResGroup26_2 : SchedWriteRes<[SBPort0,SBPort1,SBPort5]> {
  let Latency = 3;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup26_2], (instregex "COM_FIPr",
                                              "COM_FIr",
                                              "UCOM_FIPr",
                                              "UCOM_FIr")>;

// FIXME: this is probably incorrect.
def SBWriteResGroup27 : SchedWriteRes<[SBPort0,SBPort1]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup27], (instrs MUL16r, MUL32r, MUL64r)>;

def SBWriteResGroup28 : SchedWriteRes<[SBPort1,SBPort5]> {
  let Latency = 4;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup28], (instregex "MMX_CVTPD2PIirr",
                                            "MMX_CVTPI2PDirr",
                                            "MMX_CVTTPD2PIirr",
                                            "VCVTDQ2PDYrr",
                                            "(V?)CVTDQ2PDrr",
                                            "VCVTPD2DQYrr",
                                            "(V?)CVTPD2DQrr",
                                            "VCVTPD2PSYrr",
                                            "(V?)CVTPD2PSrr",
                                            "(V?)CVTSD2SSrr",
                                            "(V?)CVTSI642SDrr",
                                            "(V?)CVTSI2SDrr",
                                            "VCVTTPD2DQYrr",
                                            "(V?)CVTTPD2DQrr")>;

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
def: InstRW<[SBWriteResGroup29_2], (instregex "PAUSE")>;

def SBWriteResGroup29_3 : SchedWriteRes<[SBPort05,SBPort015]> {
  let Latency = 4;
  let NumMicroOps = 4;
  let ResourceCycles = [3,1];
}
def: InstRW<[SBWriteResGroup29_3], (instregex "SHLD(16|32|64)rrCL",
                                              "SHRD(16|32|64)rrCL")>;

def SBWriteResGroup30 : SchedWriteRes<[SBPort0]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup30], (instregex "MUL_FPrST0",
                                            "MUL_FST0r",
                                            "MUL_FrST0",
                                            "VMULPDYrr",
                                            "(V?)MULPDrr",
                                            "VMULPSYrr",
                                            "(V?)MULPSrr",
                                            "(V?)MULSDrr",
                                            "(V?)MULSSrr",
                                            "(V?)PCMPGTQrr",
                                            "(V?)PHMINPOSUWrr",
                                            "(V?)RCPPSr",
                                            "(V?)RCPSSr",
                                            "(V?)RSQRTPSr",
                                            "(V?)RSQRTSSr")>;

def SBWriteResGroup31 : SchedWriteRes<[SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup31], (instregex "MOV(8|16|32|64)rm",
                                            "MOVSX(16|32|64)rm16",
                                            "MOVSX(16|32|64)rm32",
                                            "MOVSX(16|32|64)rm8",
                                            "MOVZX(16|32|64)rm16",
                                            "MOVZX(16|32|64)rm8",
                                            "PREFETCH")>;

def SBWriteResGroup32 : SchedWriteRes<[SBPort0,SBPort1]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup32], (instregex "(V?)CVTSD2SI64rr",
                                            "(V?)CVTSD2SIrr",
                                            "(V?)CVTSS2SI64rr",
                                            "(V?)CVTSS2SIrr",
                                            "(V?)CVTTSD2SI64rr",
                                            "(V?)CVTTSD2SIrr",
                                            "(V?)CVTTSS2SI64rr",
                                            "(V?)CVTTSS2SIrr")>;

def SBWriteResGroup33 : SchedWriteRes<[SBPort4,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup33], (instregex "MOV(8|16|32|64)mr",
                                            "MOVNTI_64mr",
                                            "MOVNTImr",
                                            "PUSH64i8",
                                            "PUSH(16|32|64)r",
                                            "VEXTRACTF128mr",
                                            "VMOVAPDYmr",
                                            "(V?)MOVAPDmr",
                                            "VMOVAPSYmr",
                                            "(V?)MOVAPSmr",
                                            "VMOVDQAYmr",
                                            "(V?)MOVDQAmr",
                                            "VMOVDQUYmr",
                                            "(V?)MOVDQUmr",
                                            "(V?)MOVHPDmr",
                                            "(V?)MOVHPSmr",
                                            "(V?)MOVLPDmr",
                                            "(V?)MOVLPSmr",
                                            "VMOVNTDQYmr",
                                            "(V?)MOVNTDQmr",
                                            "VMOVNTPDYmr",
                                            "(V?)MOVNTPDmr",
                                            "VMOVNTPSYmr",
                                            "(V?)MOVNTPSmr",
                                            "(V?)MOVPDI2DImr",
                                            "(V?)MOVPQI2QImr",
                                            "(V?)MOVPQIto64mr",
                                            "(V?)MOVSDmr",
                                            "(V?)MOVSSmr",
                                            "VMOVUPDYmr",
                                            "(V?)MOVUPDmr",
                                            "VMOVUPSYmr",
                                            "(V?)MOVUPSmr")>;

def SBWriteResGroup34 : SchedWriteRes<[SBPort0,SBPort15]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup34], (instregex "(V?)MPSADBWrri")>;

def SBWriteResGroup35 : SchedWriteRes<[SBPort1,SBPort5]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup35], (instregex "CLI")>;
def: InstRW<[SBWriteResGroup35], (instregex "(V?)CVTSI642SSrr",
                                            "(V?)CVTSI2SSrr",
                                            "VHADDPDYrr",
                                            "(V?)HADDPDrr",
                                            "VHADDPSYrr",
                                            "(V?)HADDPSrr",
                                            "VHSUBPDYrr",
                                            "(V?)HSUBPDrr",
                                            "VHSUBPSYrr",
                                            "(V?)HSUBPSrr")>;

def SBWriteResGroup35_2 : SchedWriteRes<[SBPort1,SBPort4,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup35_2], (instregex "ISTT_FP16m",
                                              "ISTT_FP32m",
                                              "ISTT_FP64m",
                                              "PUSHGS64")>;

def SBWriteResGroup36 : SchedWriteRes<[SBPort4,SBPort5,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup36], (instregex "CALL64pcrel32",
                                            "CALL(16|32|64)r",
                                            "(V?)EXTRACTPSmr")>;

def SBWriteResGroup37 : SchedWriteRes<[SBPort4,SBPort01,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup37], (instregex "VMASKMOVPDYmr",
                                            "VMASKMOVPDmr",
                                            "VMASKMOVPSYmr",
                                            "VMASKMOVPSmr")>;

def SBWriteResGroup38 : SchedWriteRes<[SBPort4,SBPort23,SBPort05]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup38], (instregex "SET(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)m")>;

def SBWriteResGroup39 : SchedWriteRes<[SBPort4,SBPort23,SBPort15]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup39], (instregex "(V?)PEXTRBmr",
                                            "VPEXTRDmr",
                                            "VPEXTRWmr")>;

def SBWriteResGroup40 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 5;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup40], (instregex "MOV8mi",
                                            "STOSB",
                                            "STOSL",
                                            "STOSQ",
                                            "STOSW")>;

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
def: InstRW<[SBWriteResGroup42], (instregex "CMPXCHG(8|16|32|64)rr")>;

def SBWriteResGroup43 : SchedWriteRes<[SBPort4,SBPort23,SBPort05]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup43], (instregex "SET(A|BE)m")>;

def SBWriteResGroup44 : SchedWriteRes<[SBPort0,SBPort4,SBPort5,SBPort23]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup44], (instregex "(V?)LDMXCSR",
                                            "(V?)STMXCSR")>;

def SBWriteResGroup45 : SchedWriteRes<[SBPort0,SBPort4,SBPort23,SBPort15]> {
  let Latency = 5;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup45], (instregex "PEXTRDmr",
                                            "(V?)PEXTRQmr",
                                            "PUSHF16",
                                            "PUSHF64")>;

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
def: InstRW<[SBWriteResGroup48], (instregex "MMX_MOVD64from64rm",
                                            "POP(16|32|64)r",
                                            "VBROADCASTSSrm",
                                            "VLDDQUYrm",
                                            "(V?)LDDQUrm",
                                            "(V?)MOV64toPQIrm",
                                            "(V?)MOVAPDrm",
                                            "(V?)MOVAPSrm",
                                            "(V?)MOVDDUPrm",
                                            "(V?)MOVDI2PDIrm",
                                            "(V?)MOVDQArm",
                                            "(V?)MOVDQUrm",
                                            "(V?)MOVNTDQArm",
                                            "(V?)MOVQI2PQIrm",
                                            "(V?)MOVSDrm",
                                            "(V?)MOVSHDUPrm",
                                            "(V?)MOVSLDUPrm",
                                            "(V?)MOVSSrm",
                                            "(V?)MOVUPDrm",
                                            "(V?)MOVUPSrm")>;

def SBWriteResGroup49 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup49], (instregex "JMP(16|32|64)m",
                                            "MOV16sm")>;

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
def: InstRW<[SBWriteResGroup51], (instregex "MMX_PABSBrm",
                                            "MMX_PABSDrm",
                                            "MMX_PABSWrm",
                                            "MMX_PALIGNRrmi",
                                            "MMX_PSHUFBrm",
                                            "MMX_PSIGNBrm",
                                            "MMX_PSIGNDrm",
                                            "MMX_PSIGNWrm")>;

def SBWriteResGroup52 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 6;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup52], (instregex "ADD(8|16|32|64)rm",
                                            "AND(8|16|32|64)rm",
                                            "CMP(8|16|32|64)mi",
                                            "CMP(8|16|32|64)mr",
                                            "CMP(8|16|32|64)rm",
                                            "LODSL",
                                            "LODSQ",
                                            "OR(8|16|32|64)rm",
                                            "SUB(8|16|32|64)rm",
                                            "TEST(8|16|32|64)mr",
                                            "TEST(8|16|32|64)mi",
                                            "XOR(8|16|32|64)rm")>;

def SBWriteResGroup53 : SchedWriteRes<[SBPort4,SBPort23]> {
  let Latency = 6;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup53], (instregex "ST_F32m",
                                            "ST_F64m",
                                            "ST_FP32m",
                                            "ST_FP64m",
                                            "ST_FP80m")>;

def SBWriteResGroup54 : SchedWriteRes<[SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup54], (instregex "VBROADCASTSDYrm",
                                            "VBROADCASTSSYrm",
                                            "VMOVAPDYrm",
                                            "VMOVAPSYrm",
                                            "VMOVDDUPYrm",
                                            "VMOVDQAYrm",
                                            "VMOVDQUYrm",
                                            "VMOVSHDUPYrm",
                                            "VMOVSLDUPYrm",
                                            "VMOVUPDYrm",
                                            "VMOVUPSYrm")>;

def SBWriteResGroup55 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup55], (instregex "VCVTPS2PDYrm",
                                            "(V?)CVTPS2PDrm",
                                            "(V?)CVTSS2SDrm",
                                            "VTESTPDrm",
                                            "VTESTPSrm")>;

def SBWriteResGroup56 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup56], (instregex "(V?)ANDNPDrm",
                                            "(V?)ANDNPSrm",
                                            "(V?)ANDPDrm",
                                            "(V?)ANDPSrm",
                                            "VBROADCASTF128",
                                            "(V?)INSERTPSrm",
                                            "(V?)MOVHPDrm",
                                            "(V?)MOVHPSrm",
                                            "(V?)MOVLPDrm",
                                            "(V?)MOVLPSrm",
                                            "(V?)ORPDrm",
                                            "(V?)ORPSrm",
                                            "VPERMILPDmi",
                                            "VPERMILPDrm",
                                            "VPERMILPSmi",
                                            "VPERMILPSrm",
                                            "(V?)SHUFPDrmi",
                                            "(V?)SHUFPSrmi",
                                            "(V?)UNPCKHPDrm",
                                            "(V?)UNPCKHPSrm",
                                            "(V?)UNPCKLPDrm",
                                            "(V?)UNPCKLPSrm",
                                            "(V?)XORPDrm",
                                            "(V?)XORPSrm")>;

def SBWriteResGroup58 : SchedWriteRes<[SBPort23,SBPort05]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup58], (instregex "(V?)BLENDPDrmi",
                                            "(V?)BLENDPSrmi",
                                            "VINSERTF128rm")>;

def SBWriteResGroup59 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup59], (instregex "MMX_PADDQirm",
                                            "(V?)PABSBrm",
                                            "(V?)PABSDrm",
                                            "(V?)PABSWrm",
                                            "(V?)PACKSSDWrm",
                                            "(V?)PACKSSWBrm",
                                            "(V?)PACKUSDWrm",
                                            "(V?)PACKUSWBrm",
                                            "(V?)PADDBrm",
                                            "(V?)PADDDrm",
                                            "(V?)PADDQrm",
                                            "(V?)PADDSBrm",
                                            "(V?)PADDSWrm",
                                            "(V?)PADDUSBrm",
                                            "(V?)PADDUSWrm",
                                            "(V?)PADDWrm",
                                            "(V?)PALIGNRrmi",
                                            "(V?)PAVGBrm",
                                            "(V?)PAVGWrm",
                                            "(V?)PBLENDWrmi",
                                            "(V?)PCMPEQBrm",
                                            "(V?)PCMPEQDrm",
                                            "(V?)PCMPEQQrm",
                                            "(V?)PCMPEQWrm",
                                            "(V?)PCMPGTBrm",
                                            "(V?)PCMPGTDrm",
                                            "(V?)PCMPGTWrm",
                                            "(V?)PINSRBrm",
                                            "(V?)PINSRDrm",
                                            "(V?)PINSRQrm",
                                            "(V?)PINSRWrm",
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
                                            "(V?)PMOVZXWQrm",
                                            "(V?)PSHUFBrm",
                                            "(V?)PSHUFDmi",
                                            "(V?)PSHUFHWmi",
                                            "(V?)PSHUFLWmi",
                                            "(V?)PSIGNBrm",
                                            "(V?)PSIGNDrm",
                                            "(V?)PSIGNWrm",
                                            "(V?)PSUBBrm",
                                            "(V?)PSUBDrm",
                                            "(V?)PSUBQrm",
                                            "(V?)PSUBSBrm",
                                            "(V?)PSUBSWrm",
                                            "(V?)PSUBUSBrm",
                                            "(V?)PSUBUSWrm",
                                            "(V?)PSUBWrm",
                                            "(V?)PUNPCKHBWrm",
                                            "(V?)PUNPCKHDQrm",
                                            "(V?)PUNPCKHQDQrm",
                                            "(V?)PUNPCKHWDrm",
                                            "(V?)PUNPCKLBWrm",
                                            "(V?)PUNPCKLDQrm",
                                            "(V?)PUNPCKLQDQrm",
                                            "(V?)PUNPCKLWDrm")>;

def SBWriteResGroup60 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup60], (instregex "(V?)PANDNrm",
                                            "(V?)PANDrm",
                                            "(V?)PORrm",
                                            "(V?)PXORrm")>;

def SBWriteResGroup61 : SchedWriteRes<[SBPort0,SBPort05]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup61], (instregex "VRCPPSYr",
                                            "VRSQRTPSYr")>;

def SBWriteResGroup62 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup62], (instregex "VERRm",
                                            "VERWm")>;

def SBWriteResGroup63 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup63], (instregex "LODSB",
                                            "LODSW")>;

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
def: InstRW<[SBWriteResGroup65], (instregex "ADC(8|16|32|64)rm",
                                            "CMOV(AE|B|E|G|GE|L|LE|NE|NO|NP|NS|O|P|S)(16|32|64)rm",
                                            "SBB(8|16|32|64)rm")>;

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
def: InstRW<[SBWriteResGroup67], (instregex "SLDT(16|32|64)r",
                                            "STR(16|32|64)r")>;

def SBWriteResGroup68 : SchedWriteRes<[SBPort4,SBPort5,SBPort23]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup68], (instregex "CALL(16|32|64)m",
                                            "FNSTCW16m")>;

def SBWriteResGroup69 : SchedWriteRes<[SBPort4,SBPort23,SBPort05]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup69], (instregex "BTC(16|32|64)mi8",
                                            "BTR(16|32|64)mi8",
                                            "BTS(16|32|64)mi8",
                                            "SAR(8|16|32|64)m1",
                                            "SAR(8|16|32|64)mi",
                                            "SHL(8|16|32|64)m1",
                                            "SHL(8|16|32|64)mi",
                                            "SHR(8|16|32|64)m1",
                                            "SHR(8|16|32|64)mi")>;

def SBWriteResGroup70 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 7;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup70], (instregex "ADD(8|16|32|64)mi",
                                            "ADD(8|16|32|64)mr",
                                            "AND(8|16|32|64)mi",
                                            "AND(8|16|32|64)mr",
                                            "DEC(8|16|32|64)m",
                                            "INC(8|16|32|64)m",
                                            "NEG(8|16|32|64)m",
                                            "NOT(8|16|32|64)m",
                                            "OR(8|16|32|64)mi",
                                            "OR(8|16|32|64)mr",
                                            "SUB(8|16|32|64)mi",
                                            "SUB(8|16|32|64)mr",
                                            "XOR(8|16|32|64)mi",
                                            "XOR(8|16|32|64)mr")>;

def SBWriteResGroup71 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup71], (instregex "VTESTPDYrm",
                                            "VTESTPSYrm")>;

def SBWriteResGroup72 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup72], (instregex "BSF(16|32|64)rm",
                                            "BSR(16|32|64)rm",
                                            "CRC32r(16|32|64)m64",
                                            "CRC32r(16|32|64)m8",
                                            "FCOM32m",
                                            "FCOM64m",
                                            "FCOMP32m",
                                            "FCOMP64m")>;
def: InstRW<[SBWriteResGroup72], (instrs MUL8m)>;

def SBWriteResGroup73 : SchedWriteRes<[SBPort5,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup73], (instregex "VANDNPDYrm",
                                            "VANDNPSYrm",
                                            "VANDPDYrm",
                                            "VANDPSYrm",
                                            "VORPDYrm",
                                            "VORPSYrm",
                                            "VPERM2F128rm",
                                            "VPERMILPDYmi",
                                            "VPERMILPDYrm",
                                            "VPERMILPSYmi",
                                            "VPERMILPSYrm",
                                            "VSHUFPDYrmi",
                                            "VSHUFPSYrmi",
                                            "VUNPCKHPDYrm",
                                            "VUNPCKHPSYrm",
                                            "VUNPCKLPDYrm",
                                            "VUNPCKLPSYrm",
                                            "VXORPDYrm",
                                            "VXORPSYrm")>;

def SBWriteResGroup74 : SchedWriteRes<[SBPort23,SBPort05]> {
  let Latency = 8;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup74], (instregex "VBLENDPDYrmi",
                                            "VBLENDPSYrmi")>;

def SBWriteResGroup75 : SchedWriteRes<[SBPort23,SBPort05]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup75], (instregex "BLENDVPDrm0",
                                            "BLENDVPSrm0",
                                            "VBLENDVPDrm",
                                            "VBLENDVPSrm",
                                            "VMASKMOVPDrm",
                                            "VMASKMOVPSrm")>;

def SBWriteResGroup76 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup76], (instregex "PBLENDVBrm0",
                                            "VPBLENDVBrm")>;

def SBWriteResGroup77 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup77], (instregex "(V?)COMISDrm",
                                            "(V?)COMISSrm",
                                            "(V?)UCOMISDrm",
                                            "(V?)UCOMISSrm")>;

def SBWriteResGroup78 : SchedWriteRes<[SBPort0,SBPort5,SBPort23]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup78], (instregex "(V?)PTESTrm")>;

def SBWriteResGroup79 : SchedWriteRes<[SBPort0,SBPort23,SBPort15]> {
  let Latency = 8;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup79], (instregex "(V?)PSLLDrm",
                                            "(V?)PSLLQrm",
                                            "(V?)PSLLWrm",
                                            "(V?)PSRADrm",
                                            "(V?)PSRAWrm",
                                            "(V?)PSRLDrm",
                                            "(V?)PSRLQrm",
                                            "(V?)PSRLWrm")>;

def SBWriteResGroup80 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup80], (instregex "MMX_PHADDDrm",
                                            "MMX_PHADDSWrm",
                                            "MMX_PHADDWrm",
                                            "MMX_PHSUBDrm",
                                            "MMX_PHSUBSWrm",
                                            "MMX_PHSUBWrm")>;

def SBWriteResGroup81 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup81], (instregex "CMPXCHG(8|16|32|64)rm")>;

def SBWriteResGroup82 : SchedWriteRes<[SBPort23,SBPort05,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup82], (instregex "CMOV(A|BE)(16|32|64)rm")>;

def SBWriteResGroup83 : SchedWriteRes<[SBPort23,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [2,3];
}
def: InstRW<[SBWriteResGroup83], (instregex "CMPSB",
                                            "CMPSL",
                                            "CMPSQ",
                                            "CMPSW")>;

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
def: InstRW<[SBWriteResGroup85], (instregex "ROL(8|16|32|64)m1",
                                            "ROL(8|16|32|64)mi",
                                            "ROR(8|16|32|64)m1",
                                            "ROR(8|16|32|64)mi")>;

def SBWriteResGroup86 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 8;
  let NumMicroOps = 5;
  let ResourceCycles = [1,2,2];
}
def: InstRW<[SBWriteResGroup86], (instrs MOVSL)>;
def: InstRW<[SBWriteResGroup86], (instregex "MOVSB",
                                            "MOVSQ",
                                            "MOVSW",
                                            "XADD(8|16|32|64)rm")>;

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
def: InstRW<[SBWriteResGroup88], (instregex "SHLD(16|32|64)mri8",
                                            "SHRD(16|32|64)mri8")>;

def SBWriteResGroup89 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup89], (instregex "(V?)PMADDUBSWrm",
                                            "(V?)PMADDWDrm",
                                            "(V?)PMULDQrm",
                                            "(V?)PMULHRSWrm",
                                            "(V?)PMULHUWrm",
                                            "(V?)PMULHWrm",
                                            "(V?)PMULLDrm",
                                            "(V?)PMULLWrm",
                                            "(V?)PMULUDQrm",
                                            "(V?)PSADBWrm")>;

def SBWriteResGroup89_2 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup89], (instregex "MMX_PMADDUBSWrm",
                                            "MMX_PMADDWDirm",
                                            "MMX_PMULHRSWrm",
                                            "MMX_PMULHUWirm",
                                            "MMX_PMULHWirm",
                                            "MMX_PMULLWirm",
                                            "MMX_PMULUDQirm",
                                            "MMX_PSADBWirm")>;

def SBWriteResGroup90 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup90], (instregex "MMX_CVTPI2PSirm",
                                            "MMX_CVTPS2PIirm",
                                            "MMX_CVTTPS2PIirm",
                                            "POPCNT(16|32|64)rm",
                                            "(V?)ADDPDrm",
                                            "(V?)ADDPSrm",
                                            "(V?)ADDSDrm",
                                            "(V?)ADDSSrm",
                                            "(V?)ADDSUBPDrm",
                                            "(V?)ADDSUBPSrm",
                                            "(V?)CMPPDrmi",
                                            "(V?)CMPPSrmi",
                                            "(V?)CMPSDrm",
                                            "(V?)CMPSSrm",
                                            "(V?)CVTDQ2PSrm",
                                            "(V?)CVTPS2DQrm",
                                            "(V?)CVTSI642SDrm",
                                            "(V?)CVTSI2SDrm",
                                            "(V?)CVTTPS2DQrm",
                                            "(V?)MAX(C?)PDrm",
                                            "(V?)MAX(C?)PSrm",
                                            "(V?)MAX(C?)SDrm",
                                            "(V?)MAX(C?)SSrm",
                                            "(V?)MIN(C?)PDrm",
                                            "(V?)MIN(C?)PSrm",
                                            "(V?)MIN(C?)SDrm",
                                            "(V?)MIN(C?)SSrm",
                                            "(V?)ROUNDPDm",
                                            "(V?)ROUNDPSm",
                                            "(V?)ROUNDSDm",
                                            "(V?)ROUNDSSm",
                                            "(V?)SUBPDrm",
                                            "(V?)SUBPSrm",
                                            "(V?)SUBSDrm",
                                            "(V?)SUBSSrm")>;

def SBWriteResGroup91 : SchedWriteRes<[SBPort23,SBPort05]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,2];
}
def: InstRW<[SBWriteResGroup91], (instregex "VBLENDVPDYrm",
                                            "VBLENDVPSYrm",
                                            "VMASKMOVPDYrm",
                                            "VMASKMOVPSYrm")>;

def SBWriteResGroup92 : SchedWriteRes<[SBPort0,SBPort1,SBPort5]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup92], (instregex "(V?)DPPDrri")>;

def SBWriteResGroup93 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup93], (instregex "CVTSD2SI64rm",
                                            "CVTSD2SIrm",
                                            "CVTSS2SI64rm",
                                            "CVTSS2SIrm",
                                            "CVTTSD2SI64rm",
                                            "CVTTSD2SIrm",
                                            "CVTTSS2SI64rm",
                                            "CVTTSS2SIrm")>;
// FIXME this is probably incorrect.
def: InstRW<[SBWriteResGroup93], (instrs MUL16m, MUL32m, MUL64m)>;

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
def: InstRW<[SBWriteResGroup95], (instregex "LD_F32m",
                                            "LD_F64m",
                                            "LD_F80m")>;

def SBWriteResGroup96 : SchedWriteRes<[SBPort23,SBPort15]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,3];
}
def: InstRW<[SBWriteResGroup96], (instregex "(V?)PHADDDrm",
                                            "(V?)PHADDSWrm",
                                            "(V?)PHADDWrm",
                                            "(V?)PHSUBDrm",
                                            "(V?)PHSUBSWrm",
                                            "(V?)PHSUBWrm")>;

def SBWriteResGroup97 : SchedWriteRes<[SBPort1,SBPort4,SBPort23]> {
  let Latency = 9;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup97], (instregex "IST_F16m",
                                            "IST_F32m",
                                            "IST_FP16m",
                                            "IST_FP32m",
                                            "IST_FP64m")>;

def SBWriteResGroup97_2 : SchedWriteRes<[SBPort4,SBPort23,SBPort05]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,2,3];
}
def: InstRW<[SBWriteResGroup97_2], (instregex "ROL(8|16|32|64)mCL",
                                              "ROR(8|16|32|64)mCL",
                                              "SAR(8|16|32|64)mCL",
                                              "SHL(8|16|32|64)mCL",
                                              "SHR(8|16|32|64)mCL")>;

def SBWriteResGroup98 : SchedWriteRes<[SBPort4,SBPort23,SBPort015]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,2,3];
}
def: InstRW<[SBWriteResGroup98], (instregex "ADC(8|16|32|64)mi",
                                            "SBB(8|16|32|64)mi")>;

def SBWriteResGroup99 : SchedWriteRes<[SBPort4,SBPort23,SBPort05,SBPort015]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,2,2,1];
}
def: InstRW<[SBWriteResGroup99], (instregex "ADC(8|16|32|64)mr",
                                            "SBB(8|16|32|64)mr")>;

def SBWriteResGroup100 : SchedWriteRes<[SBPort4,SBPort5,SBPort23,SBPort05,SBPort015]> {
  let Latency = 9;
  let NumMicroOps = 6;
  let ResourceCycles = [1,1,2,1,1];
}
def: InstRW<[SBWriteResGroup100], (instregex "BT(16|32|64)mr",
                                             "BTC(16|32|64)mr",
                                             "BTR(16|32|64)mr",
                                             "BTS(16|32|64)mr")>;

def SBWriteResGroup101 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 10;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup101], (instregex "ADD_F32m",
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
                                             "VROUNDPDYm",
                                             "VROUNDPSYm",
                                             "VSUBPDYrm",
                                             "VSUBPSYrm")>;

def SBWriteResGroup102 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup102], (instregex "VCVTSD2SI64rm",
                                             "VCVTSD2SIrm",
                                             "VCVTSS2SI64rm",
                                             "VCVTSS2SIrm",
                                             "VCVTTSD2SI64rm",
                                             "VCVTTSD2SIrm",
                                             "VCVTTSS2SI64rm",
                                             "VCVTTSS2SIrm")>;

def SBWriteResGroup103 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 10;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup103], (instregex "MMX_CVTPD2PIirm",
                                             "MMX_CVTPI2PDirm",
                                             "MMX_CVTTPD2PIirm",
                                             "VCVTDQ2PDYrm",
                                             "(V?)CVTDQ2PDrm",
                                             "(V?)CVTPD2DQrm",
                                             "(V?)CVTPD2PSrm",
                                             "(V?)CVTSD2SSrm",
                                             "(V?)CVTSI642SSrm",
                                             "(V?)CVTSI2SSrm",
                                             "(V?)CVTTPD2DQrm")>;

def SBWriteResGroup103_2 : SchedWriteRes<[SBPort4,SBPort23,SBPort05,SBPort015]> {
  let Latency = 10;
  let NumMicroOps = 7;
  let ResourceCycles = [1,2,3,1];
}
def: InstRW<[SBWriteResGroup103_2], (instregex "SHLD(16|32|64)mrCL",
                                               "SHRD(16|32|64)mrCL")>;

def SBWriteResGroup104 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup104], (instregex "(V?)MULPDrm",
                                             "(V?)MULPSrm",
                                             "(V?)MULSDrm",
                                             "(V?)MULSSrm",
                                             "(V?)PCMPGTQrm",
                                             "(V?)PHMINPOSUWrm",
                                             "(V?)RCPPSm",
                                             "(V?)RCPSSm",
                                             "(V?)RSQRTPSm",
                                             "(V?)RSQRTSSm")>;

def SBWriteResGroup106 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup106], (instregex "FICOM16m",
                                             "FICOM32m",
                                             "FICOMP16m",
                                             "FICOMP32m")>;

def SBWriteResGroup107 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup107], (instregex "VCVTPD2DQYrm",
                                             "VCVTPD2PSYrm",
                                             "VCVTTPD2DQYrm")>;

def SBWriteResGroup108 : SchedWriteRes<[SBPort0,SBPort23,SBPort15]> {
  let Latency = 13;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,2];
}
def: InstRW<[SBWriteResGroup108], (instregex "(V?)MPSADBWrmi")>;

def SBWriteResGroup109 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 11;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup109], (instregex "(V?)HADDPDrm",
                                             "(V?)HADDPSrm",
                                             "(V?)HSUBPDrm",
                                             "(V?)HSUBPSrm")>;

def SBWriteResGroup111 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 12;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup111], (instregex "MUL_F32m",
                                             "MUL_F64m",
                                             "VMULPDYrm",
                                             "VMULPSYrm")>;

def SBWriteResGroup112 : SchedWriteRes<[SBPort0,SBPort1,SBPort5]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup112], (instregex "VDPPSYrri",
                                             "(V?)DPPSrri")>;

def SBWriteResGroup113 : SchedWriteRes<[SBPort1,SBPort5,SBPort23]> {
  let Latency = 12;
  let NumMicroOps = 4;
  let ResourceCycles = [1,2,1];
}
def: InstRW<[SBWriteResGroup113], (instregex "VHADDPDYrm",
                                             "VHADDPSYrm",
                                             "VHSUBPDYrm",
                                             "VHSUBPSYrm")>;

def SBWriteResGroup114 : SchedWriteRes<[SBPort1,SBPort23]> {
  let Latency = 13;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup114], (instregex "ADD_FI16m",
                                             "ADD_FI32m",
                                             "SUBR_FI16m",
                                             "SUBR_FI32m",
                                             "SUB_FI16m",
                                             "SUB_FI32m")>;

def SBWriteResGroup116 : SchedWriteRes<[SBPort0]> {
  let Latency = 14;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup116], (instregex "SQRTSSr",
                                             "(V?)DIVPSrr",
                                             "(V?)DIVSSrr",
                                             "(V?)SQRTPSr")>;

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
def: InstRW<[SBWriteResGroup118], (instregex "VRCPPSYm",
                                             "VRSQRTPSYm")>;

def SBWriteResGroup119 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 15;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup119], (instregex "MUL_FI16m",
                                             "MUL_FI32m")>;

def SBWriteResGroup120 : SchedWriteRes<[SBPort0,SBPort1,SBPort5,SBPort23]> {
  let Latency = 15;
  let NumMicroOps = 4;
  let ResourceCycles = [1,1,1,1];
}
def: InstRW<[SBWriteResGroup120], (instregex "(V?)DPPDrmi")>;

def SBWriteResGroup123 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 20;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup123], (instregex "SQRTSSm",
                                             "(V?)DIVPSrm",
                                             "(V?)DIVSSrm",
                                             "(V?)SQRTPSm")>;

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
def: InstRW<[SBWriteResGroup126], (instregex "SQRTSDr",
                                             "(V?)DIVPDrr",
                                             "(V?)DIVSDrr",
                                             "(V?)SQRTPDr")>;

def SBWriteResGroup127 : SchedWriteRes<[SBPort0]> {
  let Latency = 24;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup127], (instregex "DIVR_FPrST0",
                                             "DIVR_FST0r",
                                             "DIVR_FrST0",
                                             "DIV_FPrST0",
                                             "DIV_FST0r",
                                             "DIV_FrST0")>;

def SBWriteResGroup128 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 28;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup128], (instregex "SQRTSDm",
                                             "(V?)DIVPDrm",
                                             "(V?)DIVSDrm",
                                             "(V?)SQRTPDm")>;

def SBWriteResGroup129 : SchedWriteRes<[SBPort0,SBPort05]> {
  let Latency = 29;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup129], (instregex "VDIVPSYrr",
                                             "VSQRTPSYr")>;

def SBWriteResGroup130 : SchedWriteRes<[SBPort0,SBPort23]> {
  let Latency = 31;
  let NumMicroOps = 2;
  let ResourceCycles = [1,1];
}
def: InstRW<[SBWriteResGroup130], (instregex "DIVR_F32m",
                                             "DIVR_F64m",
                                             "DIV_F32m",
                                             "DIV_F64m")>;

def SBWriteResGroup131 : SchedWriteRes<[SBPort0,SBPort1,SBPort23]> {
  let Latency = 34;
  let NumMicroOps = 3;
  let ResourceCycles = [1,1,1];
}
def: InstRW<[SBWriteResGroup131], (instregex "DIVR_FI16m",
                                             "DIVR_FI32m",
                                             "DIV_FI16m",
                                             "DIV_FI32m")>;

def SBWriteResGroup132 : SchedWriteRes<[SBPort0,SBPort23,SBPort05]> {
  let Latency = 36;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SBWriteResGroup132], (instregex "VDIVPSYrm",
                                             "VSQRTPSYm")>;

def SBWriteResGroup133 : SchedWriteRes<[SBPort0,SBPort05]> {
  let Latency = 45;
  let NumMicroOps = 3;
  let ResourceCycles = [2,1];
}
def: InstRW<[SBWriteResGroup133], (instregex "VDIVPDYrr",
                                             "VSQRTPDYr")>;

def SBWriteResGroup134 : SchedWriteRes<[SBPort0,SBPort23,SBPort05]> {
  let Latency = 52;
  let NumMicroOps = 4;
  let ResourceCycles = [2,1,1];
}
def: InstRW<[SBWriteResGroup134], (instregex "VDIVPDYrm",
                                             "VSQRTPDYm")>;

def SBWriteResGroup135 : SchedWriteRes<[SBPort0]> {
  let Latency = 114;
  let NumMicroOps = 1;
  let ResourceCycles = [1];
}
def: InstRW<[SBWriteResGroup135], (instregex "VSQRTSSr")>;

} // SchedModel
