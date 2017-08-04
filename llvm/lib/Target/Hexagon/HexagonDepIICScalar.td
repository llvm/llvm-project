//===--- HexagonDepIICScalar.td -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def tc_049dfb74 : InstrItinClass;
def tc_0767081f : InstrItinClass;
def tc_07ac815d : InstrItinClass;
def tc_090485bb : InstrItinClass;
def tc_09c86199 : InstrItinClass;
def tc_09faec3b : InstrItinClass;
def tc_0cb867f2 : InstrItinClass;
def tc_1000eb10 : InstrItinClass;
def tc_128719e8 : InstrItinClass;
def tc_136c4786 : InstrItinClass;
def tc_14da557c : InstrItinClass;
def tc_1b6011fb : InstrItinClass;
def tc_1b834fe7 : InstrItinClass;
def tc_1e062b18 : InstrItinClass;
def tc_1e69aa99 : InstrItinClass;
def tc_1f9668cc : InstrItinClass;
def tc_1fe8323c : InstrItinClass;
def tc_20a8e109 : InstrItinClass;
def tc_210b2456 : InstrItinClass;
def tc_251c87b2 : InstrItinClass;
def tc_261d9b78 : InstrItinClass;
def tc_28d296df : InstrItinClass;
def tc_29c14515 : InstrItinClass;
def tc_2aaab1e0 : InstrItinClass;
def tc_2c8fe5ae : InstrItinClass;
def tc_2d1e6f5c : InstrItinClass;
def tc_2e55aa16 : InstrItinClass;
def tc_30665cb0 : InstrItinClass;
def tc_336e698c : InstrItinClass;
def tc_34e882a4 : InstrItinClass;
def tc_35fb9d13 : InstrItinClass;
def tc_37326008 : InstrItinClass;
def tc_3993c58b : InstrItinClass;
def tc_3b4892c6 : InstrItinClass;
def tc_3bea1824 : InstrItinClass;
def tc_3c10f809 : InstrItinClass;
def tc_3d905451 : InstrItinClass;
def tc_3e61d314 : InstrItinClass;
def tc_3eab77bd : InstrItinClass;
def tc_43068634 : InstrItinClass;
def tc_45631a8d : InstrItinClass;
def tc_47ab9233 : InstrItinClass;
def tc_47f0b7ad : InstrItinClass;
def tc_485bb57c : InstrItinClass;
def tc_4997da4a : InstrItinClass;
def tc_511f28f6 : InstrItinClass;
def tc_537e2013 : InstrItinClass;
def tc_53ee6546 : InstrItinClass;
def tc_548f402d : InstrItinClass;
def tc_5625c6c1 : InstrItinClass;
def tc_580a779c : InstrItinClass;
def tc_583510c7 : InstrItinClass;
def tc_5d806107 : InstrItinClass;
def tc_5fa2857c : InstrItinClass;
def tc_5fe9fcd0 : InstrItinClass;
def tc_6264c5e0 : InstrItinClass;
def tc_639d93ee : InstrItinClass;
def tc_63cd9d2d : InstrItinClass;
def tc_65dc7cc4 : InstrItinClass;
def tc_69bb508b : InstrItinClass;
def tc_6c52d277 : InstrItinClass;
def tc_6c576d46 : InstrItinClass;
def tc_70cabf66 : InstrItinClass;
def tc_7639d4b0 : InstrItinClass;
def tc_7675c0e9 : InstrItinClass;
def tc_76c4c5ef : InstrItinClass;
def tc_77781686 : InstrItinClass;
def tc_78b3c689 : InstrItinClass;
def tc_7986ba30 : InstrItinClass;
def tc_7bc567a7 : InstrItinClass;
def tc_7c2dcd4d : InstrItinClass;
def tc_7ca2ea10 : InstrItinClass;
def tc_7d01cbdc : InstrItinClass;
def tc_7d9a56cd : InstrItinClass;
def tc_81a23d44 : InstrItinClass;
def tc_821c4233 : InstrItinClass;
def tc_82f0f122 : InstrItinClass;
def tc_84630363 : InstrItinClass;
def tc_86442910 : InstrItinClass;
def tc_87601822 : InstrItinClass;
def tc_88fa2da6 : InstrItinClass;
def tc_8c8041e6 : InstrItinClass;
def tc_8cb685d9 : InstrItinClass;
def tc_8def9c57 : InstrItinClass;
def tc_8f0a6bad : InstrItinClass;
def tc_8fab9ac3 : InstrItinClass;
def tc_92d1833c : InstrItinClass;
def tc_94e6ffd9 : InstrItinClass;
def tc_95c54f8b : InstrItinClass;
def tc_9a13af9d : InstrItinClass;
def tc_9b73d261 : InstrItinClass;
def tc_9c18c9a5 : InstrItinClass;
def tc_9c68db63 : InstrItinClass;
def tc_9ce7a5ab : InstrItinClass;
def tc_9da3628f : InstrItinClass;
def tc_9dafb7d3 : InstrItinClass;
def tc_9df8b0dc : InstrItinClass;
def tc_9e86015f : InstrItinClass;
def tc_9f518242 : InstrItinClass;
def tc_a12a5971 : InstrItinClass;
def tc_a1fb80e1 : InstrItinClass;
def tc_a333d2a9 : InstrItinClass;
def tc_a4567c39 : InstrItinClass;
def tc_a87879e8 : InstrItinClass;
def tc_a9c993d9 : InstrItinClass;
def tc_aad55963 : InstrItinClass;
def tc_ab1b5e74 : InstrItinClass;
def tc_ae0722f7 : InstrItinClass;
def tc_ae2c2dc2 : InstrItinClass;
def tc_ae762521 : InstrItinClass;
def tc_b08b653e : InstrItinClass;
def tc_b08be45e : InstrItinClass;
def tc_b0f50e3c : InstrItinClass;
def tc_b189ad4c : InstrItinClass;
def tc_b324366f : InstrItinClass;
def tc_b5bfaa60 : InstrItinClass;
def tc_b5f5a094 : InstrItinClass;
def tc_b86c7e8b : InstrItinClass;
def tc_baccf077 : InstrItinClass;
def tc_bc5561d8 : InstrItinClass;
def tc_bcf0e36e : InstrItinClass;
def tc_bd16579e : InstrItinClass;
def tc_be995eaf : InstrItinClass;
def tc_bf6fa601 : InstrItinClass;
def tc_c0cd91a8 : InstrItinClass;
def tc_c14739d5 : InstrItinClass;
def tc_c1dbc916 : InstrItinClass;
def tc_c58f771a : InstrItinClass;
def tc_c85212ca : InstrItinClass;
def tc_c8f9a6f6 : InstrItinClass;
def tc_ca280e8b : InstrItinClass;
def tc_cbe45117 : InstrItinClass;
def tc_cd321066 : InstrItinClass;
def tc_d108a090 : InstrItinClass;
def tc_d1b5a4b6 : InstrItinClass;
def tc_d2609065 : InstrItinClass;
def tc_d267fa19 : InstrItinClass;
def tc_d2a33af5 : InstrItinClass;
def tc_d63b71d1 : InstrItinClass;
def tc_d6a805a8 : InstrItinClass;
def tc_d95f4e98 : InstrItinClass;
def tc_da79106e : InstrItinClass;
def tc_dbe218dd : InstrItinClass;
def tc_dcfee7ae : InstrItinClass;
def tc_e17ce9ad : InstrItinClass;
def tc_e2480a7f : InstrItinClass;
def tc_e2c08bb4 : InstrItinClass;
def tc_e2c31426 : InstrItinClass;
def tc_e578178f : InstrItinClass;
def tc_e836c161 : InstrItinClass;
def tc_e8c7a357 : InstrItinClass;
def tc_eb07ef6f : InstrItinClass;
def tc_ecfaae86 : InstrItinClass;
def tc_ef0ebaaa : InstrItinClass;
def tc_ef2676fd : InstrItinClass;
def tc_f027ebe9 : InstrItinClass;
def tc_f055fbb6 : InstrItinClass;
def tc_f1240c08 : InstrItinClass;
def tc_f16d5b17 : InstrItinClass;
def tc_f1aa2cdb : InstrItinClass;
def tc_f26aa619 : InstrItinClass;
def tc_f4608adc : InstrItinClass;
def tc_faab1248 : InstrItinClass;
def tc_fcee8723 : InstrItinClass;
def tc_feb4974b : InstrItinClass;

class DepScalarItinV4 {
  list<InstrItinData> DepScalarItinV4_list = [
    InstrItinData <tc_049dfb74, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_0767081f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_07ac815d, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_090485bb, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_09c86199, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_09faec3b, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_0cb867f2, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_1000eb10, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_128719e8, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_136c4786, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_14da557c, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_1b6011fb, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_1b834fe7, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_1e062b18, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_1e69aa99, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_1f9668cc, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_1fe8323c, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_20a8e109, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_210b2456, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_251c87b2, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_261d9b78, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_28d296df, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_29c14515, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_2aaab1e0, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_2c8fe5ae, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_2d1e6f5c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_2e55aa16, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_30665cb0, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_336e698c, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_34e882a4, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_35fb9d13, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_37326008, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_3993c58b, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3b4892c6, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_3bea1824, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_3c10f809, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_3d905451, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_3e61d314, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3eab77bd, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_43068634, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_45631a8d, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_47ab9233, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_47f0b7ad, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_485bb57c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_4997da4a, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_511f28f6, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_537e2013, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_53ee6546, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_548f402d, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_5625c6c1, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_580a779c, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_583510c7, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_5d806107, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_5fa2857c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_5fe9fcd0, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_6264c5e0, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_639d93ee, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_63cd9d2d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_65dc7cc4, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_69bb508b, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_6c52d277, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_6c576d46, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_70cabf66, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_7639d4b0, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_7675c0e9, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_76c4c5ef, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_77781686, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_78b3c689, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_7986ba30, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_7bc567a7, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_7c2dcd4d, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_7ca2ea10, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_7d01cbdc, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_7d9a56cd, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_81a23d44, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_821c4233, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_82f0f122, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_84630363, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_86442910, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_87601822, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_88fa2da6, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_8c8041e6, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_8cb685d9, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_8def9c57, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_8f0a6bad, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_8fab9ac3, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_92d1833c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_94e6ffd9, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_95c54f8b, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9a13af9d, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_9b73d261, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_9c18c9a5, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_9c68db63, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9ce7a5ab, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9da3628f, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9dafb7d3, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_9df8b0dc, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_9e86015f, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9f518242, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a12a5971, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a1fb80e1, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_a333d2a9, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a4567c39, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_a87879e8, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a9c993d9, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_aad55963, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_ab1b5e74, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_ae0722f7, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_ae2c2dc2, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_ae762521, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_b08b653e, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_b08be45e, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_b0f50e3c, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_b189ad4c, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_b324366f, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_b5bfaa60, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_b5f5a094, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_b86c7e8b, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_baccf077, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_bc5561d8, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_bcf0e36e, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_bd16579e, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_be995eaf, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_bf6fa601, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_c0cd91a8, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c14739d5, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_c1dbc916, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_c58f771a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c85212ca, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_c8f9a6f6, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_ca280e8b, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_cbe45117, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_cd321066, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d108a090, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d1b5a4b6, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d2609065, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_d267fa19, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_d2a33af5, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_d63b71d1, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d6a805a8, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_d95f4e98, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_da79106e, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_dbe218dd, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_dcfee7ae, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e17ce9ad, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e2480a7f, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e2c08bb4, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e2c31426, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_e578178f, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_e836c161, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e8c7a357, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_eb07ef6f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_ecfaae86, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_ef0ebaaa, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_ef2676fd, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_f027ebe9, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_f055fbb6, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_f1240c08, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f16d5b17, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_f1aa2cdb, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f26aa619, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_f4608adc, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_faab1248, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_fcee8723, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_feb4974b, [InstrStage<1, [SLOT3]>]>  ];
}

class DepScalarItinV5 {
  list<InstrItinData> DepScalarItinV5_list = [
    InstrItinData <tc_049dfb74, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_0767081f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_07ac815d, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_090485bb, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_09c86199, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_09faec3b, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_0cb867f2, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_1000eb10, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_128719e8, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_136c4786, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_14da557c, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_1b6011fb, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_1b834fe7, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_1e062b18, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_1e69aa99, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_1f9668cc, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_1fe8323c, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_20a8e109, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_210b2456, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_251c87b2, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_261d9b78, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_28d296df, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_29c14515, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_2aaab1e0, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_2c8fe5ae, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_2d1e6f5c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_2e55aa16, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_30665cb0, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_336e698c, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_34e882a4, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_35fb9d13, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_37326008, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_3993c58b, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3b4892c6, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_3bea1824, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_3c10f809, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_3d905451, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_3e61d314, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3eab77bd, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_43068634, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_45631a8d, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_47ab9233, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_47f0b7ad, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_485bb57c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_4997da4a, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_511f28f6, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_537e2013, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_53ee6546, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_548f402d, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_5625c6c1, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_580a779c, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_583510c7, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_5d806107, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_5fa2857c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_5fe9fcd0, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_6264c5e0, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_639d93ee, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_63cd9d2d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_65dc7cc4, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_69bb508b, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_6c52d277, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_6c576d46, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_70cabf66, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_7639d4b0, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_7675c0e9, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_76c4c5ef, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_77781686, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_78b3c689, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_7986ba30, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_7bc567a7, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_7c2dcd4d, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_7ca2ea10, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_7d01cbdc, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_7d9a56cd, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_81a23d44, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_821c4233, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_82f0f122, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_84630363, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_86442910, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_87601822, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_88fa2da6, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_8c8041e6, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_8cb685d9, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_8def9c57, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_8f0a6bad, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_8fab9ac3, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_92d1833c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_94e6ffd9, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_95c54f8b, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9a13af9d, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_9b73d261, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_9c18c9a5, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_9c68db63, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9ce7a5ab, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9da3628f, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9dafb7d3, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_9df8b0dc, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_9e86015f, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9f518242, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a12a5971, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a1fb80e1, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_a333d2a9, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a4567c39, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_a87879e8, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a9c993d9, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_aad55963, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_ab1b5e74, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_ae0722f7, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_ae2c2dc2, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_ae762521, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_b08b653e, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_b08be45e, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_b0f50e3c, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_b189ad4c, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_b324366f, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_b5bfaa60, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_b5f5a094, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_b86c7e8b, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_baccf077, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_bc5561d8, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_bcf0e36e, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_bd16579e, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_be995eaf, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_bf6fa601, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_c0cd91a8, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c14739d5, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_c1dbc916, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_c58f771a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c85212ca, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_c8f9a6f6, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_ca280e8b, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_cbe45117, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_cd321066, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d108a090, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d1b5a4b6, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d2609065, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_d267fa19, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_d2a33af5, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_d63b71d1, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d6a805a8, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_d95f4e98, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_da79106e, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_dbe218dd, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_dcfee7ae, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e17ce9ad, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e2480a7f, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e2c08bb4, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e2c31426, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_e578178f, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_e836c161, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e8c7a357, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_eb07ef6f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_ecfaae86, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_ef0ebaaa, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_ef2676fd, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_f027ebe9, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_f055fbb6, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_f1240c08, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f16d5b17, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_f1aa2cdb, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f26aa619, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_f4608adc, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_faab1248, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_fcee8723, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_feb4974b, [InstrStage<1, [SLOT3]>]>  ];
}

class DepScalarItinV55 {
  list<InstrItinData> DepScalarItinV55_list = [
    InstrItinData <tc_049dfb74, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_0767081f, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_07ac815d, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_090485bb, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_09c86199, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_09faec3b, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0cb867f2, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1000eb10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_128719e8, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_136c4786, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_14da557c, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b6011fb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b834fe7, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e062b18, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e69aa99, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1f9668cc, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1fe8323c, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_20a8e109, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_210b2456, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_251c87b2, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_261d9b78, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_28d296df, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29c14515, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2aaab1e0, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2c8fe5ae, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2d1e6f5c, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2e55aa16, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_30665cb0, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_336e698c, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_34e882a4, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_35fb9d13, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_37326008, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3993c58b, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3b4892c6, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3bea1824, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3c10f809, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3d905451, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3e61d314, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3eab77bd, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_43068634, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_45631a8d, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_47ab9233, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_47f0b7ad, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_485bb57c, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4997da4a, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_511f28f6, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_537e2013, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53ee6546, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_548f402d, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5625c6c1, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_580a779c, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_583510c7, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5d806107, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5fa2857c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5fe9fcd0, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6264c5e0, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_639d93ee, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_63cd9d2d, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_65dc7cc4, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_69bb508b, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6c52d277, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6c576d46, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_70cabf66, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7639d4b0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7675c0e9, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_76c4c5ef, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_77781686, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_78b3c689, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7986ba30, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7bc567a7, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7c2dcd4d, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_7ca2ea10, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7d01cbdc, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7d9a56cd, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_81a23d44, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_821c4233, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_82f0f122, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_84630363, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_86442910, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_87601822, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_88fa2da6, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8c8041e6, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8cb685d9, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8def9c57, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8f0a6bad, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8fab9ac3, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_92d1833c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_94e6ffd9, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_95c54f8b, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_9a13af9d, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_9b73d261, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c18c9a5, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c68db63, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9ce7a5ab, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9da3628f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9dafb7d3, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9df8b0dc, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9e86015f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9f518242, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a12a5971, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a1fb80e1, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a333d2a9, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_a4567c39, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a87879e8, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a9c993d9, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_aad55963, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_ab1b5e74, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ae0722f7, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ae2c2dc2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ae762521, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b08b653e, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_b08be45e, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b0f50e3c, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b189ad4c, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_b324366f, /*tc_2early*/
      [InstrStage<1, [SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5bfaa60, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5f5a094, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b86c7e8b, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_baccf077, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bc5561d8, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bcf0e36e, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_bd16579e, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_be995eaf, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bf6fa601, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c0cd91a8, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c14739d5, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c1dbc916, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c58f771a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c85212ca, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c8f9a6f6, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ca280e8b, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cbe45117, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_cd321066, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d108a090, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d1b5a4b6, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d2609065, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d267fa19, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_d2a33af5, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d63b71d1, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d6a805a8, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d95f4e98, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_da79106e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dbe218dd, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dcfee7ae, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e17ce9ad, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e2480a7f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e2c08bb4, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e2c31426, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_e578178f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e836c161, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e8c7a357, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_eb07ef6f, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ecfaae86, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_ef0ebaaa, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef2676fd, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_f027ebe9, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_f055fbb6, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f1240c08, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f16d5b17, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f1aa2cdb, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f26aa619, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_f4608adc, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_faab1248, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_fcee8723, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_feb4974b, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV60 {
  list<InstrItinData> DepScalarItinV60_list = [
    InstrItinData <tc_049dfb74, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_0767081f, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_07ac815d, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_090485bb, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_09c86199, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_09faec3b, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0cb867f2, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1000eb10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_128719e8, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_136c4786, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_14da557c, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b6011fb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b834fe7, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e062b18, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e69aa99, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1f9668cc, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1fe8323c, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_20a8e109, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_210b2456, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_251c87b2, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_261d9b78, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_28d296df, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29c14515, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2aaab1e0, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2c8fe5ae, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2d1e6f5c, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2e55aa16, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_30665cb0, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_336e698c, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_34e882a4, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_35fb9d13, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_37326008, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3993c58b, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3b4892c6, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3bea1824, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3c10f809, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3d905451, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3e61d314, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3eab77bd, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_43068634, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_45631a8d, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_47ab9233, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_47f0b7ad, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_485bb57c, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4997da4a, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_511f28f6, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_537e2013, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53ee6546, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_548f402d, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5625c6c1, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_580a779c, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_583510c7, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5d806107, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5fa2857c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5fe9fcd0, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6264c5e0, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_639d93ee, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_63cd9d2d, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_65dc7cc4, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_69bb508b, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6c52d277, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6c576d46, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_70cabf66, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7639d4b0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7675c0e9, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_76c4c5ef, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_77781686, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_78b3c689, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7986ba30, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7bc567a7, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7c2dcd4d, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_7ca2ea10, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7d01cbdc, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7d9a56cd, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_81a23d44, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_821c4233, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_82f0f122, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_84630363, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_86442910, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_87601822, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_88fa2da6, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8c8041e6, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8cb685d9, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8def9c57, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8f0a6bad, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8fab9ac3, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_92d1833c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_94e6ffd9, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_95c54f8b, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_9a13af9d, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_9b73d261, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c18c9a5, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c68db63, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9ce7a5ab, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9da3628f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9dafb7d3, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9df8b0dc, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9e86015f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9f518242, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a12a5971, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a1fb80e1, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a333d2a9, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_a4567c39, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a87879e8, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a9c993d9, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_aad55963, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_ab1b5e74, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ae0722f7, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ae2c2dc2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ae762521, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b08b653e, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_b08be45e, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b0f50e3c, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b189ad4c, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_b324366f, /*tc_2early*/
      [InstrStage<1, [SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5bfaa60, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5f5a094, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b86c7e8b, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_baccf077, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bc5561d8, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bcf0e36e, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_bd16579e, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_be995eaf, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bf6fa601, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c0cd91a8, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c14739d5, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c1dbc916, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c58f771a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c85212ca, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c8f9a6f6, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ca280e8b, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cbe45117, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_cd321066, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d108a090, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d1b5a4b6, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d2609065, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d267fa19, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_d2a33af5, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d63b71d1, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d6a805a8, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d95f4e98, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_da79106e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dbe218dd, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dcfee7ae, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e17ce9ad, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e2480a7f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e2c08bb4, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e2c31426, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_e578178f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e836c161, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e8c7a357, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_eb07ef6f, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ecfaae86, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_ef0ebaaa, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef2676fd, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_f027ebe9, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_f055fbb6, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f1240c08, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f16d5b17, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f1aa2cdb, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f26aa619, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_f4608adc, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_faab1248, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_fcee8723, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_feb4974b, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV62 {
  list<InstrItinData> DepScalarItinV62_list = [
    InstrItinData <tc_049dfb74, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_0767081f, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_07ac815d, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_090485bb, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_09c86199, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_09faec3b, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0cb867f2, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1000eb10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_128719e8, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_136c4786, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_14da557c, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b6011fb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b834fe7, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e062b18, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e69aa99, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1f9668cc, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1fe8323c, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_20a8e109, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_210b2456, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_251c87b2, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_261d9b78, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_28d296df, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29c14515, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2aaab1e0, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2c8fe5ae, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2d1e6f5c, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2e55aa16, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_30665cb0, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_336e698c, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_34e882a4, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_35fb9d13, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_37326008, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3993c58b, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3b4892c6, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3bea1824, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3c10f809, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3d905451, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3e61d314, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3eab77bd, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_43068634, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_45631a8d, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_47ab9233, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_47f0b7ad, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_485bb57c, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4997da4a, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_511f28f6, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_537e2013, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53ee6546, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_548f402d, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5625c6c1, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_580a779c, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_583510c7, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5d806107, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5fa2857c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5fe9fcd0, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6264c5e0, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_639d93ee, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_63cd9d2d, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_65dc7cc4, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_69bb508b, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6c52d277, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6c576d46, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_70cabf66, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7639d4b0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7675c0e9, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_76c4c5ef, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_77781686, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_78b3c689, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7986ba30, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7bc567a7, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7c2dcd4d, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_7ca2ea10, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7d01cbdc, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7d9a56cd, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_81a23d44, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_821c4233, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_82f0f122, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_84630363, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_86442910, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_87601822, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_88fa2da6, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8c8041e6, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8cb685d9, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8def9c57, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8f0a6bad, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8fab9ac3, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_92d1833c, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_94e6ffd9, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_95c54f8b, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_9a13af9d, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_9b73d261, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c18c9a5, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c68db63, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9ce7a5ab, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9da3628f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9dafb7d3, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9df8b0dc, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9e86015f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9f518242, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a12a5971, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a1fb80e1, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a333d2a9, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_a4567c39, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a87879e8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a9c993d9, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_aad55963, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_ab1b5e74, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ae0722f7, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ae2c2dc2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ae762521, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b08b653e, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_b08be45e, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b0f50e3c, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b189ad4c, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_b324366f, /*tc_2early*/
      [InstrStage<1, [SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5bfaa60, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5f5a094, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b86c7e8b, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_baccf077, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bc5561d8, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bcf0e36e, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_bd16579e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_be995eaf, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bf6fa601, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c0cd91a8, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c14739d5, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c1dbc916, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c58f771a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c85212ca, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c8f9a6f6, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ca280e8b, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cbe45117, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_cd321066, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d108a090, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d1b5a4b6, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d2609065, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d267fa19, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_d2a33af5, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d63b71d1, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d6a805a8, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d95f4e98, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_da79106e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dbe218dd, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dcfee7ae, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e17ce9ad, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e2480a7f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e2c08bb4, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e2c31426, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_e578178f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e836c161, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e8c7a357, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_eb07ef6f, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ecfaae86, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_ef0ebaaa, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef2676fd, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_f027ebe9, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_f055fbb6, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f1240c08, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f16d5b17, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f1aa2cdb, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f26aa619, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_f4608adc, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_faab1248, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_fcee8723, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_feb4974b, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>
  ];
}
