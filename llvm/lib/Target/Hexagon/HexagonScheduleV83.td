//=-HexagonScheduleV83.td - HexagonV83 Scheduling Definitions *- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def HexagonV83ItinList : DepScalarItinV83, ScalarItin,
                         DepHVXItinV83, HVXItin, PseudoItin {
  list<InstrItinData> ItinList =
    !listconcat(DepScalarItinV83_list, ScalarItin_list,
                DepHVXItinV83_list, HVXItin_list, PseudoItin_list);
}

def HexagonItinerariesV83 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3, SLOT_ENDLOOP,
                            CVI_ST, CVI_XLANE, CVI_SHIFT, CVI_MPY0, CVI_MPY1,
                            CVI_LD, CVI_XLSHF, CVI_MPY01, CVI_ALL,
                            CVI_ALL_NOMEM, CVI_ZW],
                            [Hex_FWD, HVX_FWD],
                            HexagonV83ItinList.ItinList>;

def HexagonModelV83 : SchedMachineModel {
  // Max issue per cycle == bundle width.
  let IssueWidth = 4;
  let Itineraries = HexagonItinerariesV83;
  let LoadLatency = 1;
  let CompleteModel = 0;
}
