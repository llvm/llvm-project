//==- RISCVSchedXiangShanKunMingHu.td - XiangShanKunMingHu Scheduling Defs -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The XiangShan is a high-performance open-source RISC-V processor project 
// initiated by the Institute of Computing Technology(ICT), Chinese Academy of Sciences(CAS). 
// The KunMingHu architecture is its third-generation derivative, 
// developed by the Institute of Computing Technology, Chinese Academy of Sciences  
// and the Beijing Institute of Open Source Chip (BOSC), 
// with a focus on achieving higher performance.
// Source: https://github.com/OpenXiangShan/XiangShan
// Documentation: https://github.com/OpenXiangShan/XiangShan-doc

//===----------------------------------------------------------------------===//
// KunMingHu core supports "RV64IMAFDCV_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh
// _zksed_zksh_svinval_zicbom_zicboz_zicsr_zifencei"
// then floating-point SEW can only be 64 and 32, not 16 and 8.
//===----------------------------------------------------------------------===//

def XiangShanKunMingHuModel : SchedMachineModel {
  let IssueWidth = 6;   // 6-way decode and dispatch
  let MicroOpBufferSize = 256;
  let LoopMicroOpBufferSize = 48;  // Instruction queue size
  let LoadLatency = 6;
  let MispredictPenalty = 13; // Based on estimate of pipeline depth.
  let CompleteModel = 0;
  let UnsupportedFeatures = [HasStdExtZcmt, HasStdExtZkr, HasVInstructions,
                             HasVInstructionsI64];
}

let SchedModel = XiangShanKunMingHuModel in {
// Define each kind of processor resource and number available.
/// Pipline
let BufferSize = 12 in {
  // Integer
  def XSPipeALU0 : ProcResource<1>; // ALU, MUL, BKU
  def XSPipeALU1 : ProcResource<1>; // ALU, MUL, BKU
  def XSPipeALU2 : ProcResource<1>; // ALU
  def XSPipeALU3 : ProcResource<1>; // ALU

  def XSPipeBJU0 : ProcResource<1>; // BRU, JMP
  def XSPipeBJU1 : ProcResource<1>; // BRU, JMP
  def XSPipeBJU2 : ProcResource<1>; // BRU, JMP, I2F, I2V, CSR
  def XSPipeDIV  : ProcResource<1>; // DIV

  // Floating-point
  def XSPipeFEX0 : ProcResource<1>; // FALU, FMA
  def XSPipeFEX1 : ProcResource<1>; // FCVT
  def XSPipeFEX2 : ProcResource<1>; // FALU, FMA
  def XSPipeFEX3 : ProcResource<1>; // FDIV
}

let BufferSize = 24 in {
  // Load and store
  def XSPipeLDU : ProcResource<3>; // LDU

  def XSPipeSTA : ProcResource<2>; // STA
  def XSPipeSTD : ProcResource<2>; // STD
}

def XSPipeGroupALU : ProcResGroup<[XSPipeALU0, XSPipeALU1, XSPipeALU2, XSPipeALU3]>;
def XSPipeGroupMUL : ProcResGroup<[XSPipeALU0, XSPipeALU1]>;
def XSPipeGroupBKU : ProcResGroup<[XSPipeALU0, XSPipeALU1]>;
def XSPipeGroupBRU : ProcResGroup<[XSPipeBJU0, XSPipeBJU1, XSPipeBJU2]>;
def XSPipeGroupJMP : ProcResGroup<[XSPipeBJU0, XSPipeBJU1, XSPipeBJU2]>;

def XSPipeGroupFALU : ProcResGroup<[XSPipeFEX0, XSPipeFEX2]>;
def XSPipeGroupFMA  : ProcResGroup<[XSPipeFEX0, XSPipeFEX2]>;

def XSPipeGroupSTU : ProcResGroup<[XSPipeSTA, XSPipeSTD]>;

//===----------------------------------------------------------------------===//


// Jump
def : WriteRes<WriteJmp, [XSPipeGroupBRU]>;
def : WriteRes<WriteJal, [XSPipeGroupJMP]>;
def : WriteRes<WriteJalr, [XSPipeGroupJMP]>;

// Integer arithmetic and logic
def : WriteRes<WriteIALU32, [XSPipeGroupALU]>;
def : WriteRes<WriteIALU, [XSPipeGroupALU]>;
def : WriteRes<WriteShiftImm32, [XSPipeGroupALU]>;
def : WriteRes<WriteShiftImm, [XSPipeGroupALU]>;
def : WriteRes<WriteShiftReg32, [XSPipeGroupALU]>;
def : WriteRes<WriteShiftReg, [XSPipeGroupALU]>;

// Integer multiplication
let Latency = 2 in {
  def : WriteRes<WriteIMul, [XSPipeGroupMUL]>;
  def : WriteRes<WriteIMul32, [XSPipeGroupMUL]>;
}

// Integer division
// Worst case latency is used.
// The latency of integer division ranges from 4 to 20.
let Latency = 20, ReleaseAtCycles = [20] in {
  def : WriteRes<WriteIDiv32, [XSPipeDIV]>;
  def : WriteRes<WriteIDiv, [XSPipeDIV]>;
  def : WriteRes<WriteIRem32, [XSPipeDIV]>;
  def : WriteRes<WriteIRem, [XSPipeDIV]>;
}

// Memory
let Latency = 4 in {
  def : WriteRes<WriteSTB, [XSPipeGroupSTU]>;
  def : WriteRes<WriteSTH, [XSPipeGroupSTU]>;
  def : WriteRes<WriteSTW, [XSPipeGroupSTU]>;
  def : WriteRes<WriteSTD, [XSPipeGroupSTU]>;
  def : WriteRes<WriteFST32, [XSPipeGroupSTU]>;
  def : WriteRes<WriteFST64, [XSPipeGroupSTU]>;
  def : WriteRes<WriteAtomicSTW, [XSPipeGroupSTU]>;
  def : WriteRes<WriteAtomicSTD, [XSPipeGroupSTU]>;
}

let Latency = 6 in {
  def : WriteRes<WriteLDB, [XSPipeLDU]>;
  def : WriteRes<WriteLDH, [XSPipeLDU]>;
  def : WriteRes<WriteLDW, [XSPipeLDU]>;
  def : WriteRes<WriteLDD, [XSPipeLDU]>;

  def : WriteRes<WriteFLD32, [XSPipeLDU]>;
  def : WriteRes<WriteFLD64, [XSPipeLDU]>;

  def : WriteRes<WriteAtomicW, [XSPipeLDU]>;
  def : WriteRes<WriteAtomicD, [XSPipeLDU]>;
  def : WriteRes<WriteAtomicLDW, [XSPipeLDU]>;
  def : WriteRes<WriteAtomicLDD, [XSPipeLDU]>;
}

let Latency = 2 in {
  def : WriteRes<WriteFAdd32, [XSPipeGroupFALU]>;
  def : WriteRes<WriteFAdd64, [XSPipeGroupFALU]>;
  def : WriteRes<WriteFCmp32, [XSPipeGroupFALU]>;
  def : WriteRes<WriteFCmp64, [XSPipeGroupFALU]>;
  def : WriteRes<WriteFMinMax32, [XSPipeGroupFALU]>;
  def : WriteRes<WriteFMinMax64, [XSPipeGroupFALU]>;
  def : WriteRes<WriteFClass32, [XSPipeGroupFALU]>;
  def : WriteRes<WriteFClass64, [XSPipeGroupFALU]>;
  def : WriteRes<WriteFSGNJ32, [XSPipeGroupFALU]>;
  def : WriteRes<WriteFSGNJ64, [XSPipeGroupFALU]>;
}

let Latency = 2 in {
  def : WriteRes<WriteFMul32, [XSPipeGroupFMA]>;
  def : WriteRes<WriteFMul64, [XSPipeGroupFMA]>;
  def : WriteRes<WriteFMA32, [XSPipeGroupFMA]>;
  def : WriteRes<WriteFMA64, [XSPipeGroupFMA]>;
}

// FDIV
let Latency = 10 in {
  def : WriteRes<WriteFDiv32, [XSPipeFEX3]>;
  def : WriteRes<WriteFSqrt32, [XSPipeFEX3]>;
}
let Latency = 15 in {
  def : WriteRes<WriteFDiv64, [XSPipeFEX3]>;
  def : WriteRes<WriteFSqrt64, [XSPipeFEX3]>;
}

// FCVT
let Latency = 3 in {
  def : WriteRes<WriteFCvtF32ToI32, [XSPipeFEX1]>;
  def : WriteRes<WriteFCvtF32ToI64, [XSPipeFEX1]>;
  def : WriteRes<WriteFCvtF64ToI32, [XSPipeFEX1]>;
  def : WriteRes<WriteFCvtF64ToI64, [XSPipeFEX1]>;
  def : WriteRes<WriteFCvtF64ToF32, [XSPipeFEX1]>;
  def : WriteRes<WriteFCvtF32ToF64, [XSPipeFEX1]>;
  def : WriteRes<WriteFMovF64ToI64, [XSPipeFEX1]>;
  def : WriteRes<WriteFMovF32ToI32, [XSPipeFEX1]>;
}

// I2V
let Latency = 1 in {
  def : WriteRes<WriteFMovI64ToF64, [XSPipeBJU2]>;
  def : WriteRes<WriteFMovI32ToF32, [XSPipeBJU2]>;
}

// I2F
let Latency = 3 in {
  def : WriteRes<WriteFCvtI32ToF32, [XSPipeBJU2]>;
  def : WriteRes<WriteFCvtI64ToF32, [XSPipeBJU2]>;
  def : WriteRes<WriteFCvtI32ToF64, [XSPipeBJU2]>;
  def : WriteRes<WriteFCvtI64ToF64, [XSPipeBJU2]>;
}

/// Zb*
let Latency = 1 in {
  // Zba
  def : WriteRes<WriteSHXADD, [XSPipeGroupALU]>;
  def : WriteRes<WriteSHXADD32, [XSPipeGroupALU]>;
  
  // Zbb
  def : WriteRes<WriteRotateImm, [XSPipeGroupALU]>;
  def : WriteRes<WriteRotateImm32, [XSPipeGroupALU]>;
  def : WriteRes<WriteRotateReg, [XSPipeGroupALU]>;
  def : WriteRes<WriteRotateReg32, [XSPipeGroupALU]>;
  def : WriteRes<WriteREV8, [XSPipeGroupALU]>;
  def : WriteRes<WriteORCB, [XSPipeGroupALU]>;
  def : WriteRes<WriteIMinMax, [XSPipeGroupALU]>;

  // Zbs
  def : WriteRes<WriteSingleBit, [XSPipeGroupALU]>;
  def : WriteRes<WriteSingleBitImm, [XSPipeGroupALU]>;
  def : WriteRes<WriteBEXT, [XSPipeGroupALU]>;
  def : WriteRes<WriteBEXTI, [XSPipeGroupALU]>;

  // Zbkb
  def : WriteRes<WriteBREV8, [XSPipeGroupALU]>;
  def : WriteRes<WritePACK, [XSPipeGroupALU]>;
  def : WriteRes<WritePACK32, [XSPipeGroupALU]>;
  def : WriteRes<WriteZIP, [XSPipeGroupALU]>;
}

let Latency = 3 in {
  // Zbb
  def : WriteRes<WriteCLZ, [XSPipeGroupBKU]>;
  def : WriteRes<WriteCLZ32, [XSPipeGroupBKU]>;
  def : WriteRes<WriteCTZ, [XSPipeGroupBKU]>;
  def : WriteRes<WriteCTZ32, [XSPipeGroupBKU]>;
  def : WriteRes<WriteCPOP, [XSPipeGroupBKU]>;
  def : WriteRes<WriteCPOP32, [XSPipeGroupBKU]>;

  // Zbc
  def : WriteRes<WriteCLMUL, [XSPipeGroupBKU]>;

  // Zbkx
  def : WriteRes<WriteXPERM, [XSPipeGroupBKU]>;
}

// Others
def : WriteRes<WriteCSR, [XSPipeBJU2]>;
def : WriteRes<WriteNop, []>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//

// Bypass and advance

class XSLDUtoAnyBypass<SchedRead read, int cycles = 2>
    : ReadAdvance<read, cycles, [WriteLDB, WriteLDH,
                                 WriteLDW, WriteLDD,
                                 WriteFLD32, WriteFLD64,
                                 WriteAtomicW, WriteAtomicD,
                                 WriteAtomicLDW, WriteAtomicLDD]>;

def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : XSLDUtoAnyBypass<ReadIALU>;
def : XSLDUtoAnyBypass<ReadIALU32>;
def : XSLDUtoAnyBypass<ReadShiftImm>;
def : XSLDUtoAnyBypass<ReadShiftImm32>;
def : XSLDUtoAnyBypass<ReadShiftReg>;
def : XSLDUtoAnyBypass<ReadShiftReg32>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : XSLDUtoAnyBypass<ReadAtomicWA>;
def : XSLDUtoAnyBypass<ReadAtomicWD>;
def : XSLDUtoAnyBypass<ReadAtomicDA>;
def : XSLDUtoAnyBypass<ReadAtomicDD>;
def : XSLDUtoAnyBypass<ReadAtomicLDW>;
def : XSLDUtoAnyBypass<ReadAtomicLDD>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

/// B extension
// Zba
def : ReadAdvance<ReadSHXADD, 0>;
def : ReadAdvance<ReadSHXADD32, 0>;
// Zbb
def : ReadAdvance<ReadRotateImm, 0>;
def : ReadAdvance<ReadRotateImm32, 0>;
def : ReadAdvance<ReadRotateReg, 0>;
def : ReadAdvance<ReadRotateReg32, 0>;
def : ReadAdvance<ReadCLZ, 0>;
def : ReadAdvance<ReadCLZ32, 0>;
def : ReadAdvance<ReadCTZ, 0>;
def : ReadAdvance<ReadCTZ32, 0>;
def : ReadAdvance<ReadCPOP, 0>;
def : ReadAdvance<ReadCPOP32, 0>;
def : ReadAdvance<ReadREV8, 0>;
def : ReadAdvance<ReadORCB, 0>;
def : ReadAdvance<ReadIMinMax, 0>;
// Zbc
def : ReadAdvance<ReadCLMUL, 0>;
// Zbs
def : ReadAdvance<ReadSingleBit, 0>;
def : ReadAdvance<ReadSingleBitImm, 0>;
// Zbkb
def : ReadAdvance<ReadBREV8, 0>;
def : ReadAdvance<ReadPACK, 0>;
def : ReadAdvance<ReadPACK32, 0>;
def : ReadAdvance<ReadZIP, 0>;
// Zbkx
def : ReadAdvance<ReadXPERM, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedQ;
defm : UnsupportedSchedV;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZfhmin;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedZabha;
defm : UnsupportedSchedXsf;
defm : UnsupportedSchedZvk;

} // SchedModel
