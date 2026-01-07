//=-HexagonScheduleV81.td - HexagonV81 Scheduling Definitions *- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def HexagonV81ItinList : DepScalarItinV81, ScalarItin,
                         DepHVXItinV81, HVXItin, PseudoItin {
  list<InstrItinData> ItinList =
    !listconcat(DepScalarItinV81_list, ScalarItin_list,
                DepHVXItinV81_list, HVXItin_list, PseudoItin_list);
}

def HexagonItinerariesV81 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3, SLOT_ENDLOOP,
                            CVI_ST, CVI_XLANE, CVI_SHIFT, CVI_MPY0, CVI_MPY1,
                            CVI_LD, CVI_XLSHF, CVI_MPY01, CVI_ALL,
                            CVI_ALL_NOMEM, CVI_ZW],
                            [Hex_FWD, HVX_FWD],
                            HexagonV81ItinList.ItinList>;

def HexagonModelV81 : SchedMachineModel {
  // Max issue per cycle == bundle width.
  let IssueWidth = 4;
  let Itineraries = HexagonItinerariesV81;
  let LoadLatency = 1;
  let CompleteModel = 0;
}
