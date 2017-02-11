//=-HexagonScheduleV60.td - HexagonV60 Scheduling Definitions *- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// CVI pipes from the "Hexagon Multimedia Co-Processor Extensions Arch Spec".
def CVI_ST     : FuncUnit;
def CVI_XLANE  : FuncUnit;
def CVI_SHIFT  : FuncUnit;
def CVI_MPY0   : FuncUnit;
def CVI_MPY1   : FuncUnit;
def CVI_LD     : FuncUnit;

// Combined functional units.
def CVI_XLSHF  : FuncUnit;
def CVI_MPY01  : FuncUnit;
def CVI_ALL    : FuncUnit;
def CVI_XLMPY0 : FuncUnit;
def CVI_SHFMPY1: FuncUnit;

// Combined functional unit data.
def HexagonComboFuncsV60 :
    ComboFuncUnits<[
      ComboFuncData<CVI_XLSHF    , [CVI_XLANE, CVI_SHIFT]>,
      ComboFuncData<CVI_MPY01    , [CVI_MPY0, CVI_MPY1]>,
      ComboFuncData<CVI_ALL      , [CVI_ST, CVI_XLANE, CVI_SHIFT,
                                    CVI_MPY0, CVI_MPY1, CVI_LD]>,
      ComboFuncData<CVI_XLMPY0   , [CVI_XLANE, CVI_MPY0]>,
      ComboFuncData<CVI_SHFMPY1  , [CVI_SHIFT, CVI_MPY1]>
    ]>;

// Note: When adding additional vector scheduling classes, add the
// corresponding methods to the class HexagonInstrInfo.
def CVI_VA           : InstrItinClass;
def CVI_VA_DV        : InstrItinClass;
def CVI_VX_LONG      : InstrItinClass;
def CVI_VX_LATE      : InstrItinClass;
def CVI_VX           : InstrItinClass;
def CVI_VX_DV_LONG   : InstrItinClass;
def CVI_VX_DV        : InstrItinClass;
def CVI_VX_DV_SLOT2  : InstrItinClass;
def CVI_VX_DV_SLOT2_LONG_EARLY : InstrItinClass;
def CVI_VP           : InstrItinClass;
def CVI_VP_LONG      : InstrItinClass;
def CVI_VP_VS_EARLY  : InstrItinClass;
def CVI_VP_VS_LONG_EARLY   : InstrItinClass;
def CVI_VP_VS_LONG   : InstrItinClass;
def CVI_VP_VS   : InstrItinClass;
def CVI_VP_DV        : InstrItinClass;
def CVI_VS           : InstrItinClass;
def CVI_VINLANESAT   : InstrItinClass;
def CVI_VM_LD        : InstrItinClass;
def CVI_VM_TMP_LD    : InstrItinClass;
def CVI_VM_CUR_LD    : InstrItinClass;
def CVI_VM_VP_LDU    : InstrItinClass;
def CVI_VM_ST        : InstrItinClass;
def CVI_VM_NEW_ST    : InstrItinClass;
def CVI_VM_STU       : InstrItinClass;
def CVI_HIST         : InstrItinClass;
def CVI_VA_EXT       : InstrItinClass;

// There are four SLOTS (four parallel pipelines) in Hexagon V60 machine.
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
//
//
// In addition to using the above SLOTS, there are also six vector pipelines
// in the CVI co-processor in the Hexagon V60 machine.
//
//      |=========| |=========| |=========| |=========| |=========| |=========|
// SLOT | CVI_LD  | |CVI_MPY3 | |CVI_MPY2 | |CVI_SHIFT| |CVI_XLANE| | CVI_ST  |
// ==== |=========| |=========| |=========| |=========| |=========| |=========|
// S0-3 |         | | CVI_VA  | | CVI_VA  | | CVI_VA  | | CVI_VA  | |         |
// S2-3 |         | | CVI_VX  | | CVI_VX  | |         | |         | |         |
// S0-3 |         | |         | |         | |         | | CVI_VP  | |         |
// S0-3 |         | |         | |         | | CVI_VS  | |         | |         |
// S0-1 |(CVI_LD) | | CVI_LD  | | CVI_LD  | | CVI_LD  | | CVI_LD  | |         |
// S0-1 |(C*TMP_LD) |         | |         | |         | |         | |         |
// S01  |(C*_LDU) | |         | |         | |         | | C*_LDU  | |         |
// S0   |         | | CVI_ST  | | CVI_ST  | | CVI_ST  | | CVI_ST  | |(CVI_ST) |
// S0   |         | |         | |         | |         | |         | |(C*TMP_ST)
// S01  |         | |         | |         | |         | | VSTU    | |(C*_STU) |
//      |=========| |=========| |=========| |=========| |=========| |=========|
//                  |=====================| |=====================|
//                  | CVI_MPY2 & CVI_MPY3 | |CVI_XLANE & CVI_SHIFT|
//                  |=====================| |=====================|
// S0-3             | CVI_VA_DV           | | CVI_VA_DV           |
// S0-3             |                     | | CVI_VP_DV           |
// S2-3             | CVI_VX_DV           | |                     |
//                  |=====================| |=====================|
//      |=====================================================================|
// S0-3 | CVI_HIST   Histogram                                                |
// S0123| CVI_VA_EXT Extract                                                  |
//      |=====================================================================|

def HexagonItinerariesV60 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3, SLOT_ENDLOOP,
                            CVI_ST, CVI_XLANE, CVI_SHIFT, CVI_MPY0, CVI_MPY1,
                            CVI_LD, CVI_XLSHF, CVI_MPY01, CVI_ALL], [], [
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
        InstrItinData<EXTENDER_tc_1_SLOT0123, [InstrStage<1,
                              [SLOT0, SLOT1, SLOT2, SLOT3]>]>,

        // Load
        InstrItinData<LD_tc_ld_SLOT01      , [InstrStage<3, [SLOT0, SLOT1]>]>,
        InstrItinData<LD_tc_ld_pi_SLOT01   , [InstrStage<3, [SLOT0, SLOT1]>]>,
        InstrItinData<LD_tc_3or4stall_SLOT0, [InstrStage<4, [SLOT0]>]>,
        InstrItinData<LD_tc_ld_SLOT0       , [InstrStage<3, [SLOT0]>]>,

        // M
        InstrItinData<M_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_2_SLOT23     , [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_2_acc_SLOT23 , [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3_SLOT23     , [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3x_SLOT23    , [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3x_acc_SLOT23, [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3or4x_SLOT23 , [InstrStage<4, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3or4x_acc_SLOT23 , [InstrStage<4, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3stall_SLOT23, [InstrStage<3, [SLOT2, SLOT3]>]>,

        // Store
        InstrItinData<ST_tc_st_SLOT01   , [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<ST_tc_st_pi_SLOT01, [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<ST_tc_3stall_SLOT0, [InstrStage<3, [SLOT0]>]>,
        InstrItinData<ST_tc_ld_SLOT0    , [InstrStage<3, [SLOT0]>]>,
        InstrItinData<ST_tc_st_SLOT0    , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<ST_tc_st_pi_SLOT0 , [InstrStage<1, [SLOT0]>]>,

        // S
        InstrItinData<S_2op_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_2op_tc_2_SLOT23     , [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<S_2op_tc_2early_SLOT23, [InstrStage<2, [SLOT2, SLOT3]>]>,
        // The S_2op_tc_3x_SLOT23 slots are 4 cycles on v60.
        InstrItinData<S_2op_tc_3or4x_SLOT23 , [InstrStage<4, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_1_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_2_SLOT23     , [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_2early_SLOT23, [InstrStage<2, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_3_SLOT23     , [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_3stall_SLOT23, [InstrStage<3, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_3x_SLOT23    , [InstrStage<3, [SLOT2, SLOT3]>]>,

        // New Value Compare Jump
        InstrItinData<NCJ_tc_3or4stall_SLOT0, [InstrStage<4, [SLOT0]>]>,

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

        // Duplex and Compound
        InstrItinData<DUPLEX     , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<COMPOUND_CJ_ARCHDEPSLOT   , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<COMPOUND , [InstrStage<1, [SLOT2, SLOT3]>]>,
        // Misc
        InstrItinData<PREFIX , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<PSEUDO , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<PSEUDOM    , [InstrStage<1, [SLOT2, SLOT3], 0>,
                                    InstrStage<1, [SLOT2, SLOT3]>]>,

        // Latest CVI spec definitions.
        InstrItinData<CVI_VA,[InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_XLANE,CVI_SHIFT,
                                                   CVI_MPY0, CVI_MPY1]>]>,
        InstrItinData<CVI_VA_DV,
                                   [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_XLSHF, CVI_MPY01]>]>,
        InstrItinData<CVI_VX_LONG, [InstrStage<1, [SLOT2, SLOT3], 0>,
                                    InstrStage<1, [CVI_MPY0, CVI_MPY1]>]>,
        InstrItinData<CVI_VX_LATE, [InstrStage<1, [SLOT2, SLOT3], 0>,
                                    InstrStage<1, [CVI_MPY0, CVI_MPY1]>]>,
        InstrItinData<CVI_VX,[InstrStage<1, [SLOT2, SLOT3], 0>,
                                    InstrStage<1, [CVI_MPY0, CVI_MPY1]>]>,
        InstrItinData<CVI_VX_DV_LONG,
                                   [InstrStage<1, [SLOT2, SLOT3], 0>,
                                    InstrStage<1, [CVI_MPY01]>]>,
        InstrItinData<CVI_VX_DV,
                                   [InstrStage<1, [SLOT2, SLOT3], 0>,
                                    InstrStage<1, [CVI_MPY01]>]>,
        InstrItinData<CVI_VX_DV_SLOT2,
                                   [InstrStage<1, [SLOT2], 0>,
                                    InstrStage<1, [CVI_MPY01]>]>,
        InstrItinData<CVI_VP,      [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_XLANE]>]>,
        InstrItinData<CVI_VP_LONG, [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_XLANE]>]>,
        InstrItinData<CVI_VP_VS_EARLY,
                                   [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_XLSHF]>]>,
        InstrItinData<CVI_VP_VS_LONG,
                                   [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_XLSHF]>]>,
        InstrItinData<CVI_VP_VS,
                                   [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_XLSHF]>]>,
        InstrItinData<CVI_VP_VS_LONG_EARLY,
                                   [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_XLSHF]>]>,
        InstrItinData<CVI_VP_DV  , [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_XLSHF]>]>,
        InstrItinData<CVI_VS,
                                   [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_SHIFT]>]>,
        InstrItinData<CVI_VINLANESAT,
                                   [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_SHIFT]>]>,
        InstrItinData<CVI_VM_LD  , [InstrStage<1, [SLOT0, SLOT1], 0>,
                                    InstrStage<1, [CVI_LD], 0>,
                                    InstrStage<1, [CVI_XLANE, CVI_SHIFT,
                                                   CVI_MPY0, CVI_MPY1]>]>,
        InstrItinData<CVI_VM_TMP_LD,[InstrStage<1,[SLOT0, SLOT1], 0>,
                                    InstrStage<1, [CVI_LD]>]>,
        InstrItinData<CVI_VM_CUR_LD,[InstrStage<1,[SLOT0, SLOT1], 0>,
                                    InstrStage<1, [CVI_LD], 0>,
                                    InstrStage<1, [CVI_XLANE, CVI_SHIFT,
                                                   CVI_MPY0, CVI_MPY1]>]>,
        InstrItinData<CVI_VM_VP_LDU,[InstrStage<1,[SLOT0], 0>,
                                    InstrStage<1, [SLOT1], 0>,
                                    InstrStage<1, [CVI_LD], 0>,
                                    InstrStage<1, [CVI_XLANE]>]>,
        InstrItinData<CVI_VM_ST  , [InstrStage<1, [SLOT0], 0>,
                                    InstrStage<1, [CVI_ST], 0>,
                                    InstrStage<1, [CVI_XLANE, CVI_SHIFT,
                                                   CVI_MPY0, CVI_MPY1]>]>,
        InstrItinData<CVI_VM_NEW_ST,[InstrStage<1,[SLOT0], 0>,
                                    InstrStage<1, [CVI_ST]>]>,
        InstrItinData<CVI_VM_STU , [InstrStage<1, [SLOT0], 0>,
                                    InstrStage<1, [SLOT1], 0>,
                                    InstrStage<1, [CVI_ST], 0>,
                                    InstrStage<1, [CVI_XLANE]>]>,
        InstrItinData<CVI_HIST   , [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
                                    InstrStage<1, [CVI_ALL]>]>
      ]>;

def HexagonModelV60 : SchedMachineModel {
  // Max issue per cycle == bundle width.
  let IssueWidth = 4;
  let Itineraries = HexagonItinerariesV60;
  let LoadLatency = 1;
  let CompleteModel = 0;
}

//===----------------------------------------------------------------------===//
// Hexagon V60 Resource Definitions -
//===----------------------------------------------------------------------===//
