//=-HexagonScheduleV75.td - HexagonV75 Scheduling Definitions *- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def HexagonV75ItinList : DepScalarItinV75, ScalarItin,
                         DepHVXItinV75, HVXItin, PseudoItin {
  list<InstrItinData> ItinList =
    !listconcat(DepScalarItinV75_list, ScalarItin_list,
                DepHVXItinV75_list, HVXItin_list, PseudoItin_list);
}

def HexagonItinerariesV75 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3, SLOT_ENDLOOP,
                            CVI_ST, CVI_XLANE, CVI_SHIFT, CVI_MPY0, CVI_MPY1,
                            CVI_LD, CVI_XLSHF, CVI_MPY01, CVI_ALL,
                            CVI_ALL_NOMEM, CVI_ZW],
                            [Hex_FWD, HVX_FWD],
                            HexagonV75ItinList.ItinList>;

def HexagonModelV75 : SchedMachineModel {
  // Max issue per cycle == bundle width.
  let IssueWidth = 4;
  let Itineraries = HexagonItinerariesV75;
  let LoadLatency = 1;
  let CompleteModel = 0;
}

//===----------------------------------------------------------------------===//
// Hexagon V75 Resource Definitions -
//===----------------------------------------------------------------------===//
