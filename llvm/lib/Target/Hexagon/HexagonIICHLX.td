//===--- HexagonIICHLX.td -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Define itineraries for HLX load and store pseudo instructions
def HLX_LD_tc_ld_SLOT1 : InstrItinClass;
def HLX_ST_tc_st_SLOT0 : InstrItinClass;

class HLXItin {
  // Load pseudo will be expanded to
  //   LX_L16b_tmp [SLOT1]
  //   LX_assign [SLOT2 or SLOT3]
  // Store pseudo will be expanded to
  //   LX_S16b_new [SLOT0]
  //   LX_assign [SLOT2 or SLOT3]
  // The slot contraints are specified by adding multiple InstrStages.
  //
  // Both stages can start at the same time,
  // set NextCycles_ for the first InstrStage to 0.
  list<InstrItinData> HLXItin_list = [
    InstrItinData<HLX_LD_tc_ld_SLOT1, [InstrStage<1, [SLOT1], 0>,
                                       InstrStage<1, [SLOT2, SLOT3]>],
                                   [4, 1], [Hex_FWD, Hex_FWD], 2>,
    InstrItinData<HLX_ST_tc_st_SLOT0, [InstrStage<1, [SLOT0], 0>,
                                       InstrStage<1, [SLOT2, SLOT3]>],
                                   [1, 1], [Hex_FWD, Hex_FWD], 2>
  ];
}
