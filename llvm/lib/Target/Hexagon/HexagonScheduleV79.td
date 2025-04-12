//=-HexagonScheduleV79.td - HexagonV79 Scheduling Definitions *- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def HexagonV79ItinList : DepScalarItinV79, ScalarItin,
                         DepHVXItinV79, HVXItin, PseudoItin {
  list<InstrItinData> ItinList =
    !listconcat(DepScalarItinV79_list, ScalarItin_list,
                DepHVXItinV79_list, HVXItin_list, PseudoItin_list);
}

def HexagonItinerariesV79 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3, SLOT_ENDLOOP,
                            CVI_ST, CVI_XLANE, CVI_SHIFT, CVI_MPY0, CVI_MPY1,
                            CVI_LD, CVI_XLSHF, CVI_MPY01, CVI_ALL,
                            CVI_ALL_NOMEM, CVI_ZW],
                            [Hex_FWD, HVX_FWD],
                            HexagonV79ItinList.ItinList>;

def HexagonModelV79 : SchedMachineModel {
  // Max issue per cycle == bundle width.
  let IssueWidth = 4;
  let Itineraries = HexagonItinerariesV79;
  let LoadLatency = 1;
  let CompleteModel = 0;
}
