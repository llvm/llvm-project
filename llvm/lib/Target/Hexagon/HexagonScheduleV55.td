//=-HexagonScheduleV4.td - HexagonV4 Scheduling Definitions --*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// There are four SLOTS (four parallel pipelines) in Hexagon V4 machine.
// This file describes that machine information.

//
//    |===========|==================================================|
//    | PIPELINE  |              Instruction Classes                 |
//    |===========|==================================================|
//    | SLOT0     |  LD       ST    ALU32     MEMOP     NV    SYSTEM |
//    |-----------|--------------------------------------------------|
//    | SLOT1     |  LD       ST    ALU32                            |
//    |-----------|--------------------------------------------------|
//    | SLOT2     |  XTYPE          ALU32     J         JR           |
//    |-----------|--------------------------------------------------|
//    | SLOT3     |  XTYPE          ALU32     J         CR           |
//    |===========|==================================================|

def CJ_tc_1_SLOT23              : InstrItinClass;
def CJ_tc_2early_SLOT23         : InstrItinClass;
def COPROC_VMEM_vtc_long_SLOT01 : InstrItinClass;
def COPROC_VX_vtc_long_SLOT23   : InstrItinClass;
def COPROC_VX_vtc_SLOT23        : InstrItinClass;
def J_tc_3stall_SLOT2           : InstrItinClass;
def MAPPING_tc_1_SLOT0123       : InstrItinClass;
def M_tc_3stall_SLOT23          : InstrItinClass;

def HexagonItinerariesV55 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3, SLOT_ENDLOOP], [], [
        // ALU32
        InstrItinData<ALU32_2op_tc_1_SLOT0123     ,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 1, 1]>,
        InstrItinData<ALU32_2op_tc_2early_SLOT0123,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 1, 1]>,
        InstrItinData<ALU32_3op_tc_1_SLOT0123     ,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 1, 1]>,
        InstrItinData<ALU32_3op_tc_2_SLOT0123     ,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 1, 1]>,
        InstrItinData<ALU32_3op_tc_2early_SLOT0123,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 1, 1]>,
        InstrItinData<ALU32_ADDI_tc_1_SLOT0123    ,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 1, 1]>,

        // ALU64
        InstrItinData<ALU64_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                              [1, 1, 1]>,
        InstrItinData<ALU64_tc_2_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                              [2, 1, 1]>,
        InstrItinData<ALU64_tc_2early_SLOT23, [InstrStage<1, [SLOT2, SLOT3]>],
                                              [2, 1, 1]>,
        InstrItinData<ALU64_tc_3x_SLOT23    , [InstrStage<1, [SLOT2, SLOT3]>],
                                              [3, 1, 1]>,

        // CR -> System
        InstrItinData<CR_tc_2_SLOT3      , [InstrStage<1, [SLOT3]>], [2, 1, 1]>,
        InstrItinData<CR_tc_2early_SLOT3 , [InstrStage<1, [SLOT3]>], [2, 1, 1]>,
        InstrItinData<CR_tc_3x_SLOT3     , [InstrStage<1, [SLOT3]>], [3, 1, 1]>,

        // Jump (conditional/unconditional/return etc)
        InstrItinData<CR_tc_2early_SLOT23, [InstrStage<1, [SLOT2, SLOT3]>],
                                           [2, 1, 1, 1]>,
        InstrItinData<CR_tc_3x_SLOT23    , [InstrStage<1, [SLOT2, SLOT3]>],
                                           [3, 1, 1, 1]>,
        InstrItinData<CJ_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                           [1, 1, 1, 1]>,
        InstrItinData<CJ_tc_2early_SLOT23, [InstrStage<1, [SLOT2, SLOT3]>],
                                           [2, 1, 1, 1]>,
        InstrItinData<J_tc_2early_SLOT23 , [InstrStage<1, [SLOT2, SLOT3]>],
                                           [2, 1, 1, 1]>,
        InstrItinData<J_tc_2early_CJUMP_UCJUMP_ARCHDEPSLOT,
                                 [InstrStage<1, [SLOT2, SLOT3]>], [2, 1, 1, 1]>,

        // JR
        InstrItinData<J_tc_2early_SLOT2  , [InstrStage<1, [SLOT2]>], [2, 1, 1]>,
        InstrItinData<J_tc_3stall_SLOT2  , [InstrStage<1, [SLOT2]>], [3, 1, 1]>,

        // Extender
        InstrItinData<EXTENDER_tc_1_SLOT0123,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 1, 1]>,

        // Load
        InstrItinData<LD_tc_ld_SLOT01      , [InstrStage<1, [SLOT0, SLOT1]>],
                                             [2, 1]>,
        InstrItinData<LD_tc_ld_pi_SLOT01   , [InstrStage<1, [SLOT0, SLOT1]>],
                                             [2, 1]>,
        InstrItinData<LD_tc_3or4stall_SLOT0, [InstrStage<1, [SLOT0]>], [2, 1]>,
        InstrItinData<LD_tc_ld_SLOT0       , [InstrStage<1, [SLOT0]>], [2, 1]>,

        // M
        InstrItinData<M_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                          [1, 1, 1]>,
        InstrItinData<M_tc_2_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                          [2, 1, 1]>,
        InstrItinData<M_tc_2_acc_SLOT23 , [InstrStage<1, [SLOT2, SLOT3]>],
                                          [2, 1, 1]>,
        InstrItinData<M_tc_3_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                          [1, 1, 1]>,
        InstrItinData<M_tc_3x_SLOT23    , [InstrStage<1, [SLOT2, SLOT3]>],
                                          [3, 1, 1]>,
        InstrItinData<M_tc_3x_acc_SLOT23, [InstrStage<1, [SLOT2, SLOT3]>],
                                          [3, 1, 1, 1]>,
        InstrItinData<M_tc_3or4x_SLOT23 , [InstrStage<1, [SLOT2, SLOT3]>],
                                          [3, 1, 1]>,
        InstrItinData<M_tc_3or4x_acc_SLOT23 , [InstrStage<1, [SLOT2, SLOT3]>],
                                          [3, 1, 1]>,
        InstrItinData<M_tc_3stall_SLOT23, [InstrStage<1, [SLOT2, SLOT3]>],
                                          [3, 1, 1]>,

        // Store
        InstrItinData<ST_tc_st_SLOT01   , [InstrStage<1, [SLOT0, SLOT1]>],
                                          [1, 1, 1]>,
        InstrItinData<ST_tc_st_pi_SLOT01, [InstrStage<1, [SLOT0, SLOT1]>],
                                          [1, 1, 1]>,
        InstrItinData<ST_tc_3stall_SLOT0, [InstrStage<1, [SLOT0]>], [2, 1, 1]>,
        InstrItinData<ST_tc_ld_SLOT0    , [InstrStage<1, [SLOT0]>], [2, 1, 1]>,
        InstrItinData<ST_tc_st_SLOT0    , [InstrStage<1, [SLOT0]>], [1, 1, 1]>,
        InstrItinData<ST_tc_st_pi_SLOT0 , [InstrStage<1, [SLOT0]>], [1, 1, 1]>,

        // S
        InstrItinData<S_2op_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                              [1, 1, 1]>,
        InstrItinData<S_2op_tc_2_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                              [2, 1, 1]>,
        InstrItinData<S_2op_tc_2early_SLOT23, [InstrStage<1, [SLOT2, SLOT3]>],
                                              [2, 1, 1]>,
        InstrItinData<S_2op_tc_3or4x_SLOT23 , [InstrStage<1, [SLOT2, SLOT3]>],
                                              [3, 1, 1]>,
        InstrItinData<S_3op_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                              [1, 1, 1]>,
        InstrItinData<S_3op_tc_2_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                              [2, 1, 1]>,
        InstrItinData<S_3op_tc_2early_SLOT23, [InstrStage<1, [SLOT2, SLOT3]>],
                                              [2, 1, 1]>,
        InstrItinData<S_3op_tc_3_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>],
                                              [3, 1, 1]>,
        InstrItinData<S_3op_tc_3stall_SLOT23, [InstrStage<1, [SLOT2, SLOT3]>],
                                              [3, 1, 1]>,
        InstrItinData<S_3op_tc_3x_SLOT23    , [InstrStage<1, [SLOT2, SLOT3]>],
                                              [3, 1, 1]>,

        // New Value Compare Jump
        InstrItinData<NCJ_tc_3or4stall_SLOT0, [InstrStage<1, [SLOT0]>],
                                              [3, 1, 1, 1]>,

        // Mem ops
        InstrItinData<V2LDST_tc_st_SLOT0  , [InstrStage<1, [SLOT0]>],
                                            [1, 1, 1, 1]>,
        InstrItinData<V2LDST_tc_ld_SLOT01 , [InstrStage<1, [SLOT0, SLOT1]>],
                                            [2, 1, 1, 1]>,
        InstrItinData<V2LDST_tc_st_SLOT01 , [InstrStage<1, [SLOT0, SLOT1]>],
                                            [1, 1, 1, 1]>,
        InstrItinData<V4LDST_tc_st_SLOT0  , [InstrStage<1, [SLOT0]>],
                                            [1, 1, 1, 1]>,
        InstrItinData<V4LDST_tc_ld_SLOT01 , [InstrStage<1, [SLOT0, SLOT1]>],
                                            [3, 1, 1, 1]>,
        InstrItinData<V4LDST_tc_st_SLOT01 , [InstrStage<1, [SLOT0, SLOT1]>],
                                            [1, 1, 1, 1]>,

        // Endloop
        InstrItinData<J_tc_2early_SLOT0123, [InstrStage<1, [SLOT_ENDLOOP]>],
                                            [2]>,

        // Vector
        InstrItinData<COPROC_VMEM_vtc_long_SLOT01,
                      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 1]>,
        InstrItinData<COPROC_VX_vtc_long_SLOT23  ,
                      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1, 1]>,
        InstrItinData<COPROC_VX_vtc_SLOT23 ,
                      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1, 1]>,
        InstrItinData<MAPPING_tc_1_SLOT0123      ,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>],
                      [1, 1, 1, 1]>,

        // Misc
        InstrItinData<COMPOUND_CJ_ARCHDEPSLOT , [InstrStage<1, [SLOT2, SLOT3]>],
                                                [1, 1, 1]>,
        InstrItinData<COMPOUND , [InstrStage<1, [SLOT2, SLOT3]>],
                                 [1, 1, 1]>,
        InstrItinData<DUPLEX , [InstrStage<1, [SLOT0]>], [1, 1, 1]>,
        InstrItinData<PREFIX , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>],
                               [1, 1, 1]>,
        InstrItinData<PSEUDO , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>],
                               [1, 1, 1]>,
        InstrItinData<PSEUDOM, [InstrStage<1, [SLOT2, SLOT3], 0>,
                                InstrStage<1, [SLOT2, SLOT3]>], [1, 1, 1]>
      ]>;

def HexagonModelV55 : SchedMachineModel {
  // Max issue per cycle == bundle width.
  let IssueWidth = 4;
  let Itineraries = HexagonItinerariesV55;
  let LoadLatency = 1;
  let CompleteModel = 0;
}

//===----------------------------------------------------------------------===//
// Hexagon V4 Resource Definitions -
//===----------------------------------------------------------------------===//
