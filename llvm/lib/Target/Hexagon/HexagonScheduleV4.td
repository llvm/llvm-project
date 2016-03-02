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

// Functional Units.
def SLOT0       : FuncUnit;
def SLOT1       : FuncUnit;
def SLOT2       : FuncUnit;
def SLOT3       : FuncUnit;
// Endloop is a pseudo instruction that is encoded with 2 bits in a packet
// rather than taking an execution slot. This special unit is needed
// to schedule an ENDLOOP with 4 other instructions.
def SLOT_ENDLOOP: FuncUnit;

// Itinerary classes.
def PSEUDO      : InstrItinClass;
def PSEUDOM     : InstrItinClass;
// ALU64/M/S Instruction classes of V2 are collectively knownn as XTYPE in V4.
def DUPLEX      : InstrItinClass;
def PREFIX      : InstrItinClass;
def COMPOUND_CJ_ARCHDEPSLOT    : InstrItinClass;
def COMPOUND    : InstrItinClass;

def ALU32_2op_tc_1_SLOT0123  : InstrItinClass;
def ALU32_2op_tc_2early_SLOT0123  : InstrItinClass;
def ALU32_3op_tc_2early_SLOT0123  : InstrItinClass;
def ALU32_3op_tc_1_SLOT0123  : InstrItinClass;
def ALU32_3op_tc_2_SLOT0123  : InstrItinClass;
def ALU32_ADDI_tc_1_SLOT0123 : InstrItinClass;
def ALU64_tc_1_SLOT23        : InstrItinClass;
def ALU64_tc_1or2_SLOT23     : InstrItinClass;
def ALU64_tc_2_SLOT23        : InstrItinClass;
def ALU64_tc_2early_SLOT23   : InstrItinClass;
def ALU64_tc_3x_SLOT23       : InstrItinClass;
def CR_tc_2_SLOT3            : InstrItinClass;
def CR_tc_2early_SLOT23      : InstrItinClass;
def CR_tc_2early_SLOT3       : InstrItinClass;
def CR_tc_3x_SLOT23          : InstrItinClass;
def CR_tc_3x_SLOT3           : InstrItinClass;
def J_tc_2early_SLOT23       : InstrItinClass;
def J_tc_2early_CJUMP_UCJUMP_ARCHDEPSLOT       : InstrItinClass;
def J_tc_2early_SLOT2        : InstrItinClass;
def LD_tc_ld_SLOT01          : InstrItinClass;
def LD_tc_ld_SLOT0           : InstrItinClass;
def LD_tc_3or4stall_SLOT0    : InstrItinClass;
def M_tc_1_SLOT23            : InstrItinClass;
def M_tc_1or2_SLOT23         : InstrItinClass;
def M_tc_2_SLOT23            : InstrItinClass;
def M_tc_3_SLOT23            : InstrItinClass;
def M_tc_3x_SLOT23           : InstrItinClass;
def M_tc_3or4x_SLOT23        : InstrItinClass;
def ST_tc_st_SLOT01          : InstrItinClass;
def ST_tc_st_SLOT0           : InstrItinClass;
def ST_tc_ld_SLOT0           : InstrItinClass;
def ST_tc_3stall_SLOT0       : InstrItinClass;
def S_2op_tc_1_SLOT23        : InstrItinClass;
def S_2op_tc_2_SLOT23        : InstrItinClass;
def S_2op_tc_2early_SLOT23   : InstrItinClass;
def S_2op_tc_3or4x_SLOT23    : InstrItinClass;
def S_3op_tc_1_SLOT23        : InstrItinClass;
def S_3op_tc_1or2_SLOT23     : InstrItinClass;
def S_3op_tc_2_SLOT23        : InstrItinClass;
def S_3op_tc_2early_SLOT23   : InstrItinClass;
def S_3op_tc_3_SLOT23        : InstrItinClass;
def S_3op_tc_3x_SLOT23       : InstrItinClass;
def NCJ_tc_3or4stall_SLOT0   : InstrItinClass;
def V2LDST_tc_ld_SLOT01      : InstrItinClass;
def V2LDST_tc_st_SLOT0       : InstrItinClass;
def V2LDST_tc_st_SLOT01      : InstrItinClass;
def V4LDST_tc_ld_SLOT01      : InstrItinClass;
def V4LDST_tc_st_SLOT0       : InstrItinClass;
def V4LDST_tc_st_SLOT01      : InstrItinClass;
def J_tc_2early_SLOT0123     : InstrItinClass;
def EXTENDER_tc_1_SLOT0123   : InstrItinClass;
def S_3op_tc_3stall_SLOT23   : InstrItinClass;


def HexagonItinerariesV4 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3, SLOT_ENDLOOP], [], [
        // ALU32
        InstrItinData<ALU32_2op_tc_1_SLOT0123  ,
                     [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU32_2op_tc_2early_SLOT0123,
                     [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU32_3op_tc_1_SLOT0123   ,
                     [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU32_3op_tc_2early_SLOT0123,
                     [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU32_3op_tc_2_SLOT0123   ,
                     [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU32_ADDI_tc_1_SLOT0123  ,
                     [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,

        // ALU64
        InstrItinData<ALU64_tc_1_SLOT23      , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<ALU64_tc_1or2_SLOT23   , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<ALU64_tc_2_SLOT23      , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<ALU64_tc_2early_SLOT23 , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<ALU64_tc_3x_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,

        // CR -> System
        InstrItinData<CR_tc_2_SLOT3          , [InstrStage<1, [SLOT3]>]>,
        InstrItinData<CR_tc_2early_SLOT3     , [InstrStage<1, [SLOT3]>]>,
        InstrItinData<CR_tc_3x_SLOT3         , [InstrStage<1, [SLOT3]>]>,

        // Jump (conditional/unconditional/return etc)
        // CR
        InstrItinData<CR_tc_2early_SLOT23    , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<CR_tc_3x_SLOT23        , [InstrStage<1, [SLOT2, SLOT3]>]>,
        // J
        InstrItinData<J_tc_2early_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,
        // JR
        InstrItinData<J_tc_2early_SLOT2      , [InstrStage<1, [SLOT2]>]>,

        //Load
        InstrItinData<LD_tc_ld_SLOT01        , [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<LD_tc_ld_SLOT0         , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<LD_tc_3or4stall_SLOT0  , [InstrStage<1, [SLOT0]>]>,

        // M
        InstrItinData<M_tc_1_SLOT23          , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_1or2_SLOT23       , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_2_SLOT23          , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3_SLOT23          , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3x_SLOT23         , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<M_tc_3or4x_SLOT23      , [InstrStage<1, [SLOT2, SLOT3]>]>,

        // Store
        // ST
        InstrItinData<ST_tc_st_SLOT01        , [InstrStage<1, [SLOT0, SLOT1]>]>,
        // ST0
        InstrItinData<ST_tc_st_SLOT0         , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<ST_tc_ld_SLOT0         , [InstrStage<1, [SLOT0]>]>,

        // S
        InstrItinData<S_2op_tc_1_SLOT23      , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_2op_tc_2_SLOT23      , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_2op_tc_2early_SLOT23 , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_2op_tc_3or4x_SLOT23  , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_1_SLOT23      , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_1or2_SLOT23   , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_2early_SLOT23 , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_2_SLOT23      , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_3_SLOT23      , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<S_3op_tc_3x_SLOT23     , [InstrStage<1, [SLOT2, SLOT3]>]>,

        // SYS
        InstrItinData<ST_tc_3stall_SLOT0     , [InstrStage<1, [SLOT0]>]>,

        // New Value Compare Jump
        InstrItinData<NCJ_tc_3or4stall_SLOT0 , [InstrStage<1, [SLOT0]>]>,

        // Mem ops - MEM_V4
        InstrItinData<V2LDST_tc_st_SLOT0     , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<V2LDST_tc_ld_SLOT01    , [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<V2LDST_tc_st_SLOT01    , [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<V4LDST_tc_st_SLOT0     , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<V4LDST_tc_ld_SLOT01    , [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<V4LDST_tc_st_SLOT01    , [InstrStage<1, [SLOT0, SLOT1]>]>,

        InstrItinData<DUPLEX , [InstrStage<1, [SLOT0]>]>,

        // ENDLOOP
        InstrItinData<J_tc_2early_SLOT0123   , [InstrStage<1, [SLOT_ENDLOOP]>]>,

        // Extender/PREFIX
        InstrItinData<EXTENDER_tc_1_SLOT0123,
                     [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,

        InstrItinData<COMPOUND , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<PSEUDO , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<PSEUDOM, [InstrStage<1, [SLOT2, SLOT3], 0>,
                                InstrStage<1, [SLOT2, SLOT3]>]>
      ]>;

def HexagonModelV4 : SchedMachineModel {
  // Max issue per cycle == bundle width.
  let IssueWidth = 4;
  let Itineraries = HexagonItinerariesV4;
  let LoadLatency = 1;
  let CompleteModel = 0;
}

//===----------------------------------------------------------------------===//
// Hexagon V4 Resource Definitions -
//===----------------------------------------------------------------------===//
