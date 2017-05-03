//===--- HexagonIICHVX.td -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def CVI_VA            : InstrItinClass;

class HVXItin {
  list<InstrItinData> HVXItin_list = [
    InstrItinData<CVI_VA,
      [InstrStage<1, [SLOT0,SLOT1,SLOT2,SLOT3], 0>,
       InstrStage<1, [CVI_XLANE,CVI_SHIFT, CVI_MPY0, CVI_MPY1]>],
      [9, 7, 7, 7], [HVX_FWD, HVX_FWD, HVX_FWD]>];
}
