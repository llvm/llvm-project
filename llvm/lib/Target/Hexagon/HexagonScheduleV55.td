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
def SUBINSN_tc_1_SLOT01         : InstrItinClass;
def SUBINSN_tc_2early_SLOT0     : InstrItinClass;
def SUBINSN_tc_2early_SLOT01    : InstrItinClass;
def SUBINSN_tc_3stall_SLOT0     : InstrItinClass;
def SUBINSN_tc_ld_SLOT0         : InstrItinClass;
def SUBINSN_tc_ld_SLOT01        : InstrItinClass;
def SUBINSN_tc_st_SLOT01        : InstrItinClass;

def HexagonItinerariesV55 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3, SLOT_ENDLOOP], [], [
        // ALU32
        InstrItinData<ALU32_2op_tc_1_SLOT0123     ,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU32_2op_tc_2early_SLOT0123,
                      [InstrStage<2, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU32_3op_tc_1_SLOT0123     ,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU32_3op_tc_2_SLOT0123     ,
                      [InstrStage<2, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU32_3op_tc_2early_SLOT0123,
                      [InstrStage<2, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU32_ADDI_tc_1_SLOT0123    ,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,

        // ALU64
        InstrItinData<ALU64_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<ALU64_tc_2_SLOT23     , [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<ALU64_tc_2early_SLOT23, [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<ALU64_tc_3x_SLOT23    , [InstrStage<3, [SLOT2, SLOT3]>]>,

        // CR -> System
        InstrItinData<CR_tc_2_SLOT3      , [InstrStage<2, [SLOT3]>]>,
        InstrItinData<CR_tc_2early_SLOT3 , [InstrStage<2, [SLOT3]>]>,
        InstrItinData<CR_tc_3x_SLOT3     , [InstrStage<3, [SLOT3]>]>,

        // Jump (conditional/unconditional/return etc)
        InstrItinData<CR_tc_2early_SLOT23, [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<CR_tc_3x_SLOT23    , [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<CJ_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<CJ_tc_2early_SLOT23, [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<J_tc_2early_SLOT23 , [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<J_tc_2early_CJUMP_UCJUMP_ARCHDEPSLOT     , [InstrStage<1, [SLOT2, SLOT3]>]>,

        // JR
        InstrItinData<J_tc_2early_SLOT2  , [InstrStage<2, [SLOT2]>]>,
        InstrItinData<J_tc_3stall_SLOT2  , [InstrStage<3, [SLOT2]>]>,

        // Extender
        InstrItinData<EXTENDER_tc_1_SLOT0123,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,

        // Load
        InstrItinData<LD_tc_ld_SLOT01      , [InstrStage<3, [SLOT0, SLOT1]>]>,
        InstrItinData<LD_tc_3or4stall_SLOT0, [InstrStage<3, [SLOT0]>]>,
        InstrItinData<LD_tc_ld_SLOT0       , [InstrStage<3, [SLOT0]>]>,

        // M
        InstrItinData<M_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_2_SLOT23     , [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3_SLOT23     , [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3x_SLOT23    , [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3or4x_SLOT23 , [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3stall_SLOT23, [InstrStage<3, [SLOT2, SLOT3]>]>,

        // Store
        InstrItinData<ST_tc_st_SLOT01   , [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<ST_tc_3stall_SLOT0, [InstrStage<3, [SLOT0]>]>,
        InstrItinData<ST_tc_ld_SLOT0    , [InstrStage<3, [SLOT0]>]>,
        InstrItinData<ST_tc_st_SLOT0    , [InstrStage<1, [SLOT0]>]>,

        // Subinsn
        InstrItinData<SUBINSN_tc_2early_SLOT0, [InstrStage<2, [SLOT0]>]>,
        InstrItinData<SUBINSN_tc_3stall_SLOT0, [InstrStage<3, [SLOT0]>]>,
        InstrItinData<SUBINSN_tc_ld_SLOT0    , [InstrStage<3, [SLOT0]>]>,
        InstrItinData<SUBINSN_tc_1_SLOT01    , [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<SUBINSN_tc_2early_SLOT01,
                              [InstrStage<2, [SLOT0, SLOT1]>]>,
        InstrItinData<SUBINSN_tc_ld_SLOT01   , [InstrStage<3, [SLOT0, SLOT1]>]>,
        InstrItinData<SUBINSN_tc_st_SLOT01   , [InstrStage<1, [SLOT0, SLOT1]>]>,

        // S
        InstrItinData<S_2op_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_2op_tc_2_SLOT23     , [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<S_2op_tc_2early_SLOT23, [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<S_2op_tc_3or4x_SLOT23 , [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_2_SLOT23     , [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_2early_SLOT23, [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_3_SLOT23     , [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_3stall_SLOT23, [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_3x_SLOT23    , [InstrStage<3, [SLOT2, SLOT3]>]>,

        // New Value Compare Jump
        InstrItinData<NCJ_tc_3or4stall_SLOT0, [InstrStage<3, [SLOT0]>]>,

        // Mem ops
        InstrItinData<V2LDST_tc_st_SLOT0  , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<V2LDST_tc_ld_SLOT01 , [InstrStage<2, [SLOT0, SLOT1]>]>,
        InstrItinData<V2LDST_tc_st_SLOT01 , [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<V4LDST_tc_st_SLOT0  , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<V4LDST_tc_ld_SLOT01 , [InstrStage<3, [SLOT0, SLOT1]>]>,
        InstrItinData<V4LDST_tc_st_SLOT01 , [InstrStage<1, [SLOT0, SLOT1]>]>,

        // Endloop
        InstrItinData<J_tc_2early_SLOT0123, [InstrStage<2, [SLOT_ENDLOOP]>]>,

        // Vector
        InstrItinData<COPROC_VMEM_vtc_long_SLOT01,
                      [InstrStage<3, [SLOT0, SLOT1]>]>,
        InstrItinData<COPROC_VX_vtc_long_SLOT23  ,
                      [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<COPROC_VX_vtc_SLOT23 ,
                      [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<MAPPING_tc_1_SLOT0123      ,
                      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,

        // Misc
        InstrItinData<COMPOUND_CJ_ARCHDEPSLOT , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<COMPOUND , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<DUPLEX , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<PREFIX , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<PSEUDO , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<PSEUDOM, [InstrStage<1, [SLOT2, SLOT3], 0>,
                                InstrStage<1, [SLOT2, SLOT3]>]>

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
