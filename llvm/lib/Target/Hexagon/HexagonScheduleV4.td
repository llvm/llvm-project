//=-HexagonScheduleV4.td - HexagonV4 Scheduling Definitions --*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def LD_tc_ld_SLOT01 : InstrItinClass;
def ST_tc_st_SLOT01 : InstrItinClass;

class HexagonV4PseudoItin {
  list<InstrItinData> V4PseudoItin_list = [
    InstrItinData<PSEUDO,     [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData<PSEUDOM,    [InstrStage<1, [SLOT2, SLOT3], 0>,
                               InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData<DUPLEX,     [InstrStage<1, [SLOT0]>]>,
    InstrItinData<tc_ENDLOOP, [InstrStage<1, [SLOT_ENDLOOP]>]>
  ];
}

def HexagonV4ItinList : DepScalarItinV4, HexagonV4PseudoItin {
  list<InstrItinData> V4Itin_list = [
    InstrItinData<LD_tc_ld_SLOT01, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData<ST_tc_st_SLOT01, [InstrStage<1, [SLOT0, SLOT1]>]>
  ];
  list<InstrItinData> ItinList =
    !listconcat(V4Itin_list, DepScalarItinV4_list, V4PseudoItin_list);
}

def HexagonItinerariesV4 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3, SLOT_ENDLOOP],
                           [Hex_FWD], HexagonV4ItinList.ItinList>;

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
