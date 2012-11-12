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

// Itinerary classes.
def NV_V4       : InstrItinClass;
def MEM_V4      : InstrItinClass;
// ALU64/M/S Instruction classes of V2 are collectively knownn as XTYPE in V4.
def PREFIX      : InstrItinClass;

def HexagonItinerariesV4 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3], [], [
        InstrItinData<ALU32  , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<ALU64  , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<CR     , [InstrStage<1, [SLOT3]>]>,
        InstrItinData<J      , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<JR     , [InstrStage<1, [SLOT2]>]>,
        InstrItinData<LD     , [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<M      , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<ST     , [InstrStage<1, [SLOT0, SLOT1]>]>,
        InstrItinData<S      , [InstrStage<1, [SLOT2, SLOT3]>]>,
        InstrItinData<SYS    , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<NV_V4  , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<MEM_V4 , [InstrStage<1, [SLOT0]>]>,
        InstrItinData<MARKER , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<PREFIX , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
        InstrItinData<PSEUDO , [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>
      ]>;

def HexagonModelV4 : SchedMachineModel {
  // Max issue per cycle == bundle width.
  let IssueWidth = 4;
  let Itineraries = HexagonItinerariesV4;
  let LoadLatency = 1;
}

//===----------------------------------------------------------------------===//
// Hexagon V4 Resource Definitions -
//===----------------------------------------------------------------------===//
