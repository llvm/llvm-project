//===-- PPCScheduleP9.td - PPC P9 Scheduling Definitions ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the POWER9 processor.
//
//===----------------------------------------------------------------------===//
include "PPCInstrInfo.td"

def P9Model : SchedMachineModel {
  let IssueWidth = 8;

  let LoadLatency = 5;

  let MispredictPenalty = 16;

  // Try to make sure we have at least 10 dispatch groups in a loop.
  let LoopMicroOpBufferSize = 60;

  let CompleteModel = 0;

}

let SchedModel = P9Model in {

  // ***************** Processor Resources *****************

  //Dispatcher:
  def DISPATCHER : ProcResource<12>;

  // Issue Ports
  def IP_AGEN : ProcResource<4>;
  def IP_EXEC : ProcResource<4>;
  def IP_EXECE : ProcResource<2> {
    //Even Exec Ports
    let Super = IP_EXEC;
  }
  def IP_EXECO : ProcResource<2> {
    //Odd Exec Ports
    let Super = IP_EXEC;
  }

  // Pipeline Groups
  def ALU : ProcResource<4>;
  def ALUE : ProcResource<2> {
    //Even ALU pipelines
    let Super = ALU;
  }
  def ALUO : ProcResource<2> {
    //Odd ALU pipelines
    let Super = ALU;
  }
  def DIV : ProcResource<2>;
  def DP : ProcResource<4>;
  def DPE : ProcResource<2> {
    //Even DP pipelines
    let Super = DP;
  }
  def DPO : ProcResource<2> {
    //Odd DP pipelines
    let Super = DP;
  }
  def LS : ProcResource<4>;
  def PM : ProcResource<2>;
  def DFU : ProcResource<1>;

  def TestGroup : ProcResGroup<[ALU, DP]>;

  // ***************** SchedWriteRes Definitions *****************

  //Dispatcher
  def DISP_1C : SchedWriteRes<[DISPATCHER]> {
    let NumMicroOps = 0;
    let Latency = 1;
  }

  // Issue Ports
  def IP_AGEN_1C : SchedWriteRes<[IP_AGEN]> {
    let NumMicroOps = 0;
    let Latency = 1;
  }

  def IP_EXEC_1C : SchedWriteRes<[IP_EXEC]> {
    let NumMicroOps = 0;
    let Latency = 1;
  }

  def IP_EXECE_1C : SchedWriteRes<[IP_EXECE]> {
    let NumMicroOps = 0;
    let Latency = 1;
  }

  def IP_EXECO_1C : SchedWriteRes<[IP_EXECO]> {
    let NumMicroOps = 0;
    let Latency = 1;
  }

  //Pipeline Groups
  def P9_ALU_2C : SchedWriteRes<[ALU]> {
    let Latency = 2;
  }

  def P9_ALUE_2C : SchedWriteRes<[ALUE]> {
    let Latency = 2;
  }

  def P9_ALUO_2C : SchedWriteRes<[ALUO]> {
    let Latency = 2;
  }

  def P9_ALU_3C : SchedWriteRes<[ALU]> {
    let Latency = 3;
  }

  def P9_ALUE_3C : SchedWriteRes<[ALUE]> {
    let Latency = 3;
  }

  def P9_ALUO_3C : SchedWriteRes<[ALUO]> {
    let Latency = 3;
  }

  def P9_ALU_4C : SchedWriteRes<[ALU]> {
    let Latency = 4;
  }

  def P9_ALUE_4C : SchedWriteRes<[ALUE]> {
    let Latency = 4;
  }

  def P9_ALUO_4C : SchedWriteRes<[ALUO]> {
    let Latency = 4;
  }

  def P9_ALU_5C : SchedWriteRes<[ALU]> {
    let Latency = 5;
  }

  def P9_ALU_6C : SchedWriteRes<[ALU]> {
    let Latency = 6;
  }

  def P9_DIV_16C_8 : SchedWriteRes<[DIV]> {
    let ResourceCycles = [8];
    let Latency = 16;
  }

  def P9_DIV_24C_8 : SchedWriteRes<[DIV]> {
    let ResourceCycles = [8];
    let Latency = 24;
  }

  def P9_DIV_40C_8 : SchedWriteRes<[DIV]> {
    let ResourceCycles = [8];
    let Latency = 40;
  }

  def P9_DP_2C : SchedWriteRes<[DP]> {
    let Latency = 2;
  }

  def P9_DP_5C : SchedWriteRes<[DP]> {
    let Latency = 5;
  }

  def P9_DP_7C : SchedWriteRes<[DP]> {
    let Latency = 7;
  }

  def P9_DPE_7C : SchedWriteRes<[DPE]> {
    let Latency = 7;
  }

  def P9_DPO_7C : SchedWriteRes<[DPO]> {
    let Latency = 7;
  }

  def P9_DP_22C_5 : SchedWriteRes<[DP]> {
    let ResourceCycles = [5];
    let Latency = 22;
  }

  def P9_DP_24C_8 : SchedWriteRes<[DP]> {
    let ResourceCycles = [8];
    let Latency = 24;
  }

  def P9_DP_26C_5 : SchedWriteRes<[DP]> {
    let ResourceCycles = [5];
    let Latency = 22;
  }

  def P9_DP_27C_7 : SchedWriteRes<[DP]> {
    let ResourceCycles = [7];
    let Latency = 27;
  }

  def P9_DP_33C_8 : SchedWriteRes<[DP]> {
    let ResourceCycles = [8];
    let Latency = 33;
  }

  def P9_DP_36C_10 : SchedWriteRes<[DP]> {
    let ResourceCycles = [10];
    let Latency = 36;
  }

  def P9_PM_3C : SchedWriteRes<[PM]> {
    let Latency = 3;
  }

  def P9_PM_7C : SchedWriteRes<[PM]> {
    let Latency = 3;
  }

  def P9_LS_1C : SchedWriteRes<[LS]> {
    let Latency = 1;
  }

  def P9_LS_4C : SchedWriteRes<[LS]> {
    let Latency = 4;
  }

  def P9_LS_5C : SchedWriteRes<[LS]> {
    let Latency = 5;
  }

  def P9_DFU_12C : SchedWriteRes<[DFU]> {
    let Latency = 12;
  }

  def P9_DFU_24C : SchedWriteRes<[DFU]> {
    let Latency = 24;
    let ResourceCycles = [12];
  }

  def P9_DFU_58C : SchedWriteRes<[DFU]> {
    let Latency = 58;
    let ResourceCycles = [44];
  }

  def P9_DFU_76C : SchedWriteRes<[TestGroup, DFU]> {
    let Latency = 76;
    let ResourceCycles = [62];
  }
  // ***************** WriteSeq Definitions *****************

  def P9_LoadAndALUOp_6C : WriteSequence<[P9_LS_4C, P9_ALU_2C]>;
  def P9_LoadAndALUOp_7C : WriteSequence<[P9_LS_5C, P9_ALU_2C]>;
  def P9_LoadAndPMOp_8C : WriteSequence<[P9_LS_5C, P9_PM_3C]>;
  def P9_IntDivAndALUOp_26C_8 : WriteSequence<[P9_DIV_24C_8, P9_ALU_2C]>;
  def P9_IntDivAndALUOp_42C_8 : WriteSequence<[P9_DIV_40C_8, P9_ALU_2C]>;
  def P9_StoreAndALUOp_4C : WriteSequence<[P9_LS_1C, P9_ALU_3C]>;
  def P9_ALUOpAndALUOp_4C : WriteSequence<[P9_ALU_2C, P9_ALU_2C]>;

  // ***************** Defining Itinerary Class Resources *****************

  def : ItinRW<[P9_DFU_76C, IP_EXEC_1C, DISP_1C, DISP_1C], [IIC_IntSimple,
                                         IIC_IntGeneral]>;

  def : ItinRW<[P9_ALU_2C, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_IntISEL, IIC_IntRotate, IIC_IntShift]>;

  def : ItinRW<[P9_ALU_2C, IP_EXEC_1C, DISP_1C, DISP_1C], [IIC_IntCompare]>;

  def : ItinRW<[P9_DP_5C, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_IntMulHW, IIC_IntMulHWU, IIC_IntMulLI]>;

  def : ItinRW<[P9_LS_5C, IP_EXEC_1C, DISP_1C, DISP_1C],
               [IIC_LdStLoad, IIC_LdStLD]>;

  def : ItinRW<[P9_LS_4C, P9_ALU_2C, IP_EXEC_1C, IP_EXEC_1C,
                DISP_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_LdStLoadUpd, IIC_LdStLDU]>;

  def : ItinRW<[P9_LS_4C, P9_ALU_2C, IP_EXECE_1C, IP_EXECO_1C,
                DISP_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_LdStLoadUpdX, IIC_LdStLDUX]>;

  def : ItinRW<[P9_LS_1C, P9_ALU_2C, IP_EXEC_1C, IP_EXEC_1C, IP_AGEN_1C,
                DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_LdStSTFDU]>;

  def : ItinRW<[P9_LoadAndALUOp_6C,
                IP_AGEN_1C, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_LdStLHA, IIC_LdStLWA]>;

  def : ItinRW<[P9_LoadAndALUOp_6C, P9_ALU_2C,
                IP_AGEN_1C, IP_EXEC_1C, IP_EXEC_1C,
                DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_LdStLHAU, IIC_LdStLHAUX]>;

  // IIC_LdStLMW contains two microcoded insns. This is not accurate, but
  // those insns are not used that much, if at all.
  def : ItinRW<[P9_LS_4C, IP_EXEC_1C, DISP_1C, DISP_1C],
               [IIC_LdStLWARX, IIC_LdStLDARX, IIC_LdStLMW]>;

  def : ItinRW<[P9_LS_1C, IP_EXEC_1C, IP_AGEN_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_LdStSTFD, IIC_LdStSTD, IIC_LdStStore]>;

  def : ItinRW<[P9_LS_1C, P9_ALU_2C, IP_EXEC_1C, IP_EXEC_1C, IP_AGEN_1C,
                DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_LdStSTDU, IIC_LdStSTDUX]>;

  def : ItinRW<[P9_StoreAndALUOp_4C, IP_EXEC_1C, IP_EXEC_1C, IP_AGEN_1C,
                DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_LdStSTDCX, IIC_LdStSTWCX]>;

  def : ItinRW<[P9_ALU_5C, IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C],
               [IIC_BrCR, IIC_IntMTFSB0]>;

  def : ItinRW<[P9_ALUOpAndALUOp_4C, P9_ALU_2C, IP_EXEC_1C, IP_EXEC_1C,
                IP_EXEC_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C, DISP_1C,
                DISP_1C, DISP_1C, DISP_1C], [IIC_SprMFCR, IIC_SprMFCRF]>;

  // This class should be broken down to instruction level, once some missing
  // info is obtained.
  def : ItinRW<[P9_LoadAndALUOp_6C, IP_EXEC_1C, IP_AGEN_1C,
                DISP_1C, DISP_1C, DISP_1C], [IIC_SprMTSPR]>;

  def : ItinRW<[P9_DP_7C, IP_EXEC_1C,
                DISP_1C, DISP_1C, DISP_1C], [IIC_FPGeneral, IIC_FPAddSub]>;

  def : ItinRW<[P9_DP_36C_10, IP_EXEC_1C], [IIC_FPSqrtD]>;
  def : ItinRW<[P9_DP_26C_5, P9_DP_26C_5, IP_EXEC_1C, IP_EXEC_1C], [IIC_FPSqrtS]>;

  include "P9InstrResources.td"

}

