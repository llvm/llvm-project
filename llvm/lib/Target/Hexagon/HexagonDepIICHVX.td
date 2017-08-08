//===--- HexagonDepIICHVX.td ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def tc_0317c6ca : InstrItinClass;
def tc_1b93bdc6 : InstrItinClass;
def tc_2171ebae : InstrItinClass;
def tc_28978789 : InstrItinClass;
def tc_316c637c : InstrItinClass;
def tc_354299ad : InstrItinClass;
def tc_35e92f8e : InstrItinClass;
def tc_38208312 : InstrItinClass;
def tc_4105d6b5 : InstrItinClass;
def tc_41f4b64e : InstrItinClass;
def tc_41f99e1c : InstrItinClass;
def tc_45453b98 : InstrItinClass;
def tc_4e2a5159 : InstrItinClass;
def tc_4fd8566e : InstrItinClass;
def tc_51cd3aab : InstrItinClass;
def tc_5a9fc4ec : InstrItinClass;
def tc_5c120602 : InstrItinClass;
def tc_5cbf490b : InstrItinClass;
def tc_644584f8 : InstrItinClass;
def tc_69b6dd20 : InstrItinClass;
def tc_6b78cf13 : InstrItinClass;
def tc_6fd9ad30 : InstrItinClass;
def tc_71337255 : InstrItinClass;
def tc_72ad7b54 : InstrItinClass;
def tc_77a4c701 : InstrItinClass;
def tc_7c3f55c4 : InstrItinClass;
def tc_7e9f581b : InstrItinClass;
def tc_7fa82b08 : InstrItinClass;
def tc_7fa8b40f : InstrItinClass;
def tc_85d237e3 : InstrItinClass;
def tc_8b6a873f : InstrItinClass;
def tc_908a4c8c : InstrItinClass;
def tc_9311da3f : InstrItinClass;
def tc_9777e6bf : InstrItinClass;
def tc_97c165b9 : InstrItinClass;
def tc_99093773 : InstrItinClass;
def tc_9b9642a1 : InstrItinClass;
def tc_9c267309 : InstrItinClass;
def tc_a3127e12 : InstrItinClass;
def tc_a4c9df3b : InstrItinClass;
def tc_aedb9f9e : InstrItinClass;
def tc_b06ab583 : InstrItinClass;
def tc_b712833a : InstrItinClass;
def tc_b77635b4 : InstrItinClass;
def tc_bbaf280e : InstrItinClass;
def tc_bf142ae2 : InstrItinClass;
def tc_c00bf9c9 : InstrItinClass;
def tc_c4b515c5 : InstrItinClass;
def tc_cbf6d1dc : InstrItinClass;
def tc_cedf314b : InstrItinClass;
def tc_d2cb81ea : InstrItinClass;
def tc_d5090f3e : InstrItinClass;
def tc_d642eff3 : InstrItinClass;
def tc_d725e5b0 : InstrItinClass;
def tc_d7bea0ec : InstrItinClass;
def tc_d98f4d63 : InstrItinClass;
def tc_da979fb3 : InstrItinClass;
def tc_db5b9e2f : InstrItinClass;
def tc_e172d86a : InstrItinClass;
def tc_e231aa4f : InstrItinClass;
def tc_e3748cdf : InstrItinClass;
def tc_e5053c8f : InstrItinClass;
def tc_e6299d16 : InstrItinClass;
def tc_eb669007 : InstrItinClass;
def tc_eda67dcd : InstrItinClass;
def tc_f3fc3f83 : InstrItinClass;

class DepHVXItinV55 {
  list<InstrItinData> DepHVXItinV55_list = [
    InstrItinData <tc_0317c6ca, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_1b93bdc6, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_2171ebae, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_28978789, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_316c637c, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_354299ad, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_35e92f8e, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_38208312, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4105d6b5, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_41f4b64e, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_41f99e1c, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_45453b98, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_4e2a5159, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_4fd8566e, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_51cd3aab, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5a9fc4ec, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_5c120602, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_5cbf490b, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_644584f8, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_69b6dd20, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_6b78cf13, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_6fd9ad30, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_71337255, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_72ad7b54, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_77a4c701, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7c3f55c4, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_7e9f581b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7fa82b08, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_7fa8b40f, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_85d237e3, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_8b6a873f, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_908a4c8c, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_9311da3f, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_9777e6bf, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_97c165b9, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_99093773, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_9b9642a1, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_9c267309, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a3127e12, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_a4c9df3b, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_aedb9f9e, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_b06ab583, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_b712833a, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b77635b4, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_bbaf280e, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_bf142ae2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_c00bf9c9, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_c4b515c5, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_cbf6d1dc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_cedf314b, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_d2cb81ea, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_d5090f3e, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_d642eff3, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_d725e5b0, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_d7bea0ec, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_d98f4d63, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_da979fb3, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_db5b9e2f, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_e172d86a, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_e231aa4f, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_e3748cdf, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_e5053c8f, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_e6299d16, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_eb669007, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_eda67dcd, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_f3fc3f83, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>
  ];
}

class DepHVXItinV60 {
  list<InstrItinData> DepHVXItinV60_list = [
    InstrItinData <tc_0317c6ca, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_1b93bdc6, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_2171ebae, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_28978789, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_316c637c, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_354299ad, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_35e92f8e, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_38208312, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4105d6b5, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_41f4b64e, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_41f99e1c, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_45453b98, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_4e2a5159, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_4fd8566e, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_51cd3aab, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5a9fc4ec, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_5c120602, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_5cbf490b, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_644584f8, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_69b6dd20, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_6b78cf13, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_6fd9ad30, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_71337255, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_72ad7b54, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_77a4c701, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7c3f55c4, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_7e9f581b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7fa82b08, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_7fa8b40f, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_85d237e3, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_8b6a873f, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_908a4c8c, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_9311da3f, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_9777e6bf, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_97c165b9, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_99093773, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_9b9642a1, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_9c267309, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a3127e12, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_a4c9df3b, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_aedb9f9e, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_b06ab583, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_b712833a, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b77635b4, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_bbaf280e, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_bf142ae2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_c00bf9c9, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_c4b515c5, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_cbf6d1dc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_cedf314b, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_d2cb81ea, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_d5090f3e, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_d642eff3, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_d725e5b0, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_d7bea0ec, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_d98f4d63, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_da979fb3, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_db5b9e2f, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_e172d86a, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_e231aa4f, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_e3748cdf, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_e5053c8f, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_e6299d16, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_eb669007, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_eda67dcd, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_f3fc3f83, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>
  ];
}

class DepHVXItinV62 {
  list<InstrItinData> DepHVXItinV62_list = [
    InstrItinData <tc_0317c6ca, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_1b93bdc6, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_2171ebae, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_28978789, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_316c637c, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_354299ad, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_35e92f8e, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_38208312, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4105d6b5, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_41f4b64e, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_41f99e1c, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_45453b98, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_4e2a5159, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_4fd8566e, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_51cd3aab, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5a9fc4ec, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_5c120602, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_5cbf490b, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_644584f8, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_69b6dd20, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_6b78cf13, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_6fd9ad30, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_71337255, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_72ad7b54, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_77a4c701, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7c3f55c4, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_7e9f581b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7fa82b08, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_7fa8b40f, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_85d237e3, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_8b6a873f, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_908a4c8c, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_9311da3f, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_9777e6bf, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_97c165b9, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_99093773, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_9b9642a1, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_9c267309, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a3127e12, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_a4c9df3b, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_aedb9f9e, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_b06ab583, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_b712833a, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b77635b4, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_bbaf280e, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_bf142ae2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_c00bf9c9, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_c4b515c5, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_cbf6d1dc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_cedf314b, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_d2cb81ea, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_d5090f3e, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_d642eff3, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_d725e5b0, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_d7bea0ec, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_d98f4d63, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_da979fb3, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_db5b9e2f, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_e172d86a, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_e231aa4f, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_e3748cdf, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_e5053c8f, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_e6299d16, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_eb669007, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_eda67dcd, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_f3fc3f83, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>
  ];
}
