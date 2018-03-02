//===- HexagonDepIICScalar.td ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Automatically generated file, please consult code owner before editing.
//===----------------------------------------------------------------------===//


def tc_0077f68c : InstrItinClass;
def tc_00afc57e : InstrItinClass;
def tc_00e7c26e : InstrItinClass;
def tc_03220ffa : InstrItinClass;
def tc_038a1342 : InstrItinClass;
def tc_04c9decc : InstrItinClass;
def tc_05b6c987 : InstrItinClass;
def tc_0a2b8c7c : InstrItinClass;
def tc_0cd51c76 : InstrItinClass;
def tc_0dc560de : InstrItinClass;
def tc_0fc1ae07 : InstrItinClass;
def tc_10b97e27 : InstrItinClass;
def tc_128f96e3 : InstrItinClass;
def tc_1372bca1 : InstrItinClass;
def tc_1432937d : InstrItinClass;
def tc_14cd4cfa : InstrItinClass;
def tc_15411484 : InstrItinClass;
def tc_16d0d8d5 : InstrItinClass;
def tc_181af5d0 : InstrItinClass;
def tc_1853ea6d : InstrItinClass;
def tc_1b82a277 : InstrItinClass;
def tc_1b9c9ee5 : InstrItinClass;
def tc_1c0005f9 : InstrItinClass;
def tc_1d5a38a8 : InstrItinClass;
def tc_1e856f58 : InstrItinClass;
def tc_20280784 : InstrItinClass;
def tc_234a11a5 : InstrItinClass;
def tc_238d91d2 : InstrItinClass;
def tc_29175780 : InstrItinClass;
def tc_29641329 : InstrItinClass;
def tc_2a160009 : InstrItinClass;
def tc_2b2f4060 : InstrItinClass;
def tc_2b6f77c6 : InstrItinClass;
def tc_2e00db30 : InstrItinClass;
def tc_2f185f5c : InstrItinClass;
def tc_2fc0c436 : InstrItinClass;
def tc_351fed2d : InstrItinClass;
def tc_3669266a : InstrItinClass;
def tc_367f7f3d : InstrItinClass;
def tc_36c68ad1 : InstrItinClass;
def tc_395dc00f : InstrItinClass;
def tc_3bc2c5d3 : InstrItinClass;
def tc_3cb8ea06 : InstrItinClass;
def tc_3d04548d : InstrItinClass;
def tc_3da80ba5 : InstrItinClass;
def tc_3e07fb90 : InstrItinClass;
def tc_41d5298e : InstrItinClass;
def tc_4403ca65 : InstrItinClass;
def tc_44126683 : InstrItinClass;
def tc_452f85af : InstrItinClass;
def tc_481e5e5c : InstrItinClass;
def tc_49eb22c8 : InstrItinClass;
def tc_4ca572d4 : InstrItinClass;
def tc_4d9914c9 : InstrItinClass;
def tc_4d99bca9 : InstrItinClass;
def tc_4f7cd700 : InstrItinClass;
def tc_513bef45 : InstrItinClass;
def tc_51b866be : InstrItinClass;
def tc_523fcf30 : InstrItinClass;
def tc_5274e61a : InstrItinClass;
def tc_52d7bbea : InstrItinClass;
def tc_53173427 : InstrItinClass;
def tc_53bc8a6a : InstrItinClass;
def tc_53bdb2f6 : InstrItinClass;
def tc_540fdfbc : InstrItinClass;
def tc_55050d58 : InstrItinClass;
def tc_56d25411 : InstrItinClass;
def tc_57288781 : InstrItinClass;
def tc_594ab548 : InstrItinClass;
def tc_59a01ead : InstrItinClass;
def tc_5acef64a : InstrItinClass;
def tc_5ba5997d : InstrItinClass;
def tc_5eb851fc : InstrItinClass;
def tc_5f6847a1 : InstrItinClass;
def tc_60571023 : InstrItinClass;
def tc_609d2efe : InstrItinClass;
def tc_60d76817 : InstrItinClass;
def tc_60f5738d : InstrItinClass;
def tc_63fe3df7 : InstrItinClass;
def tc_66888ded : InstrItinClass;
def tc_6792d5ff : InstrItinClass;
def tc_681a2300 : InstrItinClass;
def tc_68cb12ce : InstrItinClass;
def tc_6aa5711a : InstrItinClass;
def tc_6ac37025 : InstrItinClass;
def tc_6ebb4a12 : InstrItinClass;
def tc_6efc556e : InstrItinClass;
def tc_6fa4db47 : InstrItinClass;
def tc_73043bf4 : InstrItinClass;
def tc_746baa8e : InstrItinClass;
def tc_74e47fd9 : InstrItinClass;
def tc_7934b9df : InstrItinClass;
def tc_7a830544 : InstrItinClass;
def tc_7f881c76 : InstrItinClass;
def tc_84df2cd3 : InstrItinClass;
def tc_85523bcb : InstrItinClass;
def tc_855b0b61 : InstrItinClass;
def tc_87735c3b : InstrItinClass;
def tc_88fa1a78 : InstrItinClass;
def tc_897d1a9d : InstrItinClass;
def tc_8b15472a : InstrItinClass;
def tc_8bb285ec : InstrItinClass;
def tc_8fd5f294 : InstrItinClass;
def tc_8fe6b782 : InstrItinClass;
def tc_90f3e30c : InstrItinClass;
def tc_976ddc4f : InstrItinClass;
def tc_97743097 : InstrItinClass;
def tc_994333cd : InstrItinClass;
def tc_999d32db : InstrItinClass;
def tc_99be14ca : InstrItinClass;
def tc_9c00ce8d : InstrItinClass;
def tc_9c98e8af : InstrItinClass;
def tc_9d5941c7 : InstrItinClass;
def tc_9ef61e5c : InstrItinClass;
def tc_9faf76ae : InstrItinClass;
def tc_9fdb5406 : InstrItinClass;
def tc_a21dc435 : InstrItinClass;
def tc_a27582fa : InstrItinClass;
def tc_a46f0df5 : InstrItinClass;
def tc_a788683e : InstrItinClass;
def tc_a8acdac0 : InstrItinClass;
def tc_a904d137 : InstrItinClass;
def tc_adb14c66 : InstrItinClass;
def tc_b13761ae : InstrItinClass;
def tc_b166348b : InstrItinClass;
def tc_b44c6e2a : InstrItinClass;
def tc_b5a33b22 : InstrItinClass;
def tc_b77c481f : InstrItinClass;
def tc_b7dd427e : InstrItinClass;
def tc_b9488031 : InstrItinClass;
def tc_b9c0b731 : InstrItinClass;
def tc_b9c4623f : InstrItinClass;
def tc_bad2bcaf : InstrItinClass;
def tc_bcc96cee : InstrItinClass;
def tc_bd90564c : InstrItinClass;
def tc_bde7aaf4 : InstrItinClass;
def tc_be706f30 : InstrItinClass;
def tc_c2f7d806 : InstrItinClass;
def tc_c5e2426d : InstrItinClass;
def tc_c6aa82f7 : InstrItinClass;
def tc_c6ce9b3f : InstrItinClass;
def tc_c6ebf8dd : InstrItinClass;
def tc_c74f796f : InstrItinClass;
def tc_c82dc1ff : InstrItinClass;
def tc_caaebcba : InstrItinClass;
def tc_cd7374a0 : InstrItinClass;
def tc_cde8b071 : InstrItinClass;
def tc_cf47a43f : InstrItinClass;
def tc_cf59f215 : InstrItinClass;
def tc_d088982c : InstrItinClass;
def tc_d1090e34 : InstrItinClass;
def tc_d24b2d85 : InstrItinClass;
def tc_d580173f : InstrItinClass;
def tc_d6bf0472 : InstrItinClass;
def tc_d9709180 : InstrItinClass;
def tc_d9f95eef : InstrItinClass;
def tc_daa058fa : InstrItinClass;
def tc_dbdffe3d : InstrItinClass;
def tc_e0739b8c : InstrItinClass;
def tc_e1e0a2dc : InstrItinClass;
def tc_e1e99bfa : InstrItinClass;
def tc_e216a5db : InstrItinClass;
def tc_e421e012 : InstrItinClass;
def tc_e6b38e01 : InstrItinClass;
def tc_e7624c08 : InstrItinClass;
def tc_e7d02c66 : InstrItinClass;
def tc_e913dc32 : InstrItinClass;
def tc_e9c822f7 : InstrItinClass;
def tc_e9fae2d6 : InstrItinClass;
def tc_ef20db1c : InstrItinClass;
def tc_ef52ed71 : InstrItinClass;
def tc_ef84f62f : InstrItinClass;
def tc_f2704b9a : InstrItinClass;
def tc_f3eaa14b : InstrItinClass;
def tc_f47d212f : InstrItinClass;
def tc_f49e76f4 : InstrItinClass;
def tc_f4f43fb5 : InstrItinClass;
def tc_f7dd9c9f : InstrItinClass;
def tc_f86c328a : InstrItinClass;
def tc_f8eeed7a : InstrItinClass;
def tc_fcab4871 : InstrItinClass;
def tc_ff9ee76e : InstrItinClass;

class DepScalarItinV4 {
  list<InstrItinData> DepScalarItinV4_list = [
    InstrItinData <tc_0077f68c, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_00afc57e, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_00e7c26e, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_03220ffa, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_038a1342, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_04c9decc, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_05b6c987, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_0a2b8c7c, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_0cd51c76, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_0dc560de, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_0fc1ae07, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_10b97e27, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_128f96e3, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_1372bca1, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_1432937d, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_14cd4cfa, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_15411484, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_16d0d8d5, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_181af5d0, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_1853ea6d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_1b82a277, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_1b9c9ee5, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_1c0005f9, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_1d5a38a8, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_1e856f58, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_20280784, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_234a11a5, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_238d91d2, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_29175780, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_29641329, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_2a160009, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_2b2f4060, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_2b6f77c6, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_2e00db30, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_2f185f5c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_2fc0c436, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_351fed2d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_3669266a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_367f7f3d, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_36c68ad1, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_395dc00f, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3bc2c5d3, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3cb8ea06, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_3d04548d, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3da80ba5, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3e07fb90, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_41d5298e, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_4403ca65, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_44126683, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_452f85af, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_481e5e5c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_49eb22c8, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_4ca572d4, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_4d9914c9, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_4d99bca9, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_4f7cd700, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_513bef45, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_51b866be, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_523fcf30, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_5274e61a, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_52d7bbea, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_53173427, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_53bc8a6a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_53bdb2f6, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_540fdfbc, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_55050d58, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_56d25411, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_57288781, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_594ab548, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_59a01ead, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_5acef64a, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_5ba5997d, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_5eb851fc, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_5f6847a1, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_60571023, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_609d2efe, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_60d76817, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_60f5738d, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_63fe3df7, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_66888ded, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_6792d5ff, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_681a2300, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_68cb12ce, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_6aa5711a, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_6ac37025, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_6ebb4a12, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_6efc556e, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_6fa4db47, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_73043bf4, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_746baa8e, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_74e47fd9, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_7934b9df, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_7a830544, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_7f881c76, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_84df2cd3, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_85523bcb, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_855b0b61, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_87735c3b, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_88fa1a78, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_897d1a9d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_8b15472a, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_8bb285ec, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_8fd5f294, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_8fe6b782, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_90f3e30c, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_976ddc4f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_97743097, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_994333cd, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_999d32db, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_99be14ca, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_9c00ce8d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_9c98e8af, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_9d5941c7, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9ef61e5c, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_9faf76ae, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_9fdb5406, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_a21dc435, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_a27582fa, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a46f0df5, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a788683e, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_a8acdac0, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_a904d137, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_adb14c66, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_b13761ae, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_b166348b, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_b44c6e2a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_b5a33b22, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_b77c481f, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_b7dd427e, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_b9488031, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_b9c0b731, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_b9c4623f, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_bad2bcaf, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_bcc96cee, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_bd90564c, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_bde7aaf4, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_be706f30, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c2f7d806, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c5e2426d, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_c6aa82f7, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_c6ce9b3f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c6ebf8dd, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c74f796f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c82dc1ff, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_caaebcba, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_cd7374a0, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_cde8b071, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_cf47a43f, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_cf59f215, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_d088982c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d1090e34, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_d24b2d85, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_d580173f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d6bf0472, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_d9709180, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_d9f95eef, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_daa058fa, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_dbdffe3d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e0739b8c, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_e1e0a2dc, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_e1e99bfa, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e216a5db, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e421e012, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e6b38e01, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_e7624c08, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e7d02c66, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e913dc32, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e9c822f7, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_e9fae2d6, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_ef20db1c, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_ef52ed71, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_ef84f62f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f2704b9a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f3eaa14b, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f47d212f, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_f49e76f4, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f4f43fb5, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_f7dd9c9f, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_f86c328a, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_f8eeed7a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_fcab4871, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_ff9ee76e, [InstrStage<1, [SLOT0]>]>  ];
}

class DepScalarItinV5 {
  list<InstrItinData> DepScalarItinV5_list = [
    InstrItinData <tc_0077f68c, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_00afc57e, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_00e7c26e, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_03220ffa, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_038a1342, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_04c9decc, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_05b6c987, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_0a2b8c7c, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_0cd51c76, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_0dc560de, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_0fc1ae07, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_10b97e27, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_128f96e3, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_1372bca1, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_1432937d, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_14cd4cfa, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_15411484, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_16d0d8d5, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_181af5d0, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_1853ea6d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_1b82a277, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_1b9c9ee5, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_1c0005f9, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_1d5a38a8, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_1e856f58, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_20280784, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_234a11a5, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_238d91d2, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_29175780, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_29641329, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_2a160009, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_2b2f4060, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_2b6f77c6, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_2e00db30, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_2f185f5c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_2fc0c436, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_351fed2d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_3669266a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_367f7f3d, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_36c68ad1, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_395dc00f, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3bc2c5d3, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3cb8ea06, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_3d04548d, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3da80ba5, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_3e07fb90, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_41d5298e, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_4403ca65, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_44126683, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_452f85af, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_481e5e5c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_49eb22c8, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_4ca572d4, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_4d9914c9, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_4d99bca9, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_4f7cd700, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_513bef45, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_51b866be, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_523fcf30, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_5274e61a, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_52d7bbea, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_53173427, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_53bc8a6a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_53bdb2f6, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_540fdfbc, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_55050d58, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_56d25411, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_57288781, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_594ab548, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_59a01ead, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_5acef64a, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_5ba5997d, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_5eb851fc, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_5f6847a1, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_60571023, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_609d2efe, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_60d76817, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_60f5738d, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_63fe3df7, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_66888ded, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_6792d5ff, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_681a2300, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_68cb12ce, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_6aa5711a, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_6ac37025, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_6ebb4a12, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_6efc556e, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_6fa4db47, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_73043bf4, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_746baa8e, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_74e47fd9, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_7934b9df, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_7a830544, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_7f881c76, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_84df2cd3, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_85523bcb, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_855b0b61, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_87735c3b, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_88fa1a78, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_897d1a9d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_8b15472a, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_8bb285ec, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_8fd5f294, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_8fe6b782, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_90f3e30c, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_976ddc4f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_97743097, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_994333cd, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_999d32db, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_99be14ca, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_9c00ce8d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_9c98e8af, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_9d5941c7, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_9ef61e5c, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_9faf76ae, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_9fdb5406, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_a21dc435, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_a27582fa, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a46f0df5, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_a788683e, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_a8acdac0, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_a904d137, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_adb14c66, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_b13761ae, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_b166348b, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_b44c6e2a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_b5a33b22, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_b77c481f, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_b7dd427e, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_b9488031, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_b9c0b731, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_b9c4623f, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_bad2bcaf, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_bcc96cee, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_bd90564c, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_bde7aaf4, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_be706f30, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c2f7d806, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c5e2426d, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_c6aa82f7, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_c6ce9b3f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c6ebf8dd, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c74f796f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_c82dc1ff, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_caaebcba, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_cd7374a0, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_cde8b071, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_cf47a43f, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_cf59f215, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_d088982c, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d1090e34, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_d24b2d85, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_d580173f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_d6bf0472, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
    InstrItinData <tc_d9709180, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_d9f95eef, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_daa058fa, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_dbdffe3d, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e0739b8c, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_e1e0a2dc, [InstrStage<1, [SLOT2]>]>,
    InstrItinData <tc_e1e99bfa, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e216a5db, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e421e012, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e6b38e01, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_e7624c08, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e7d02c66, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_e913dc32, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_e9c822f7, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_e9fae2d6, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_ef20db1c, [InstrStage<1, [SLOT3]>]>,
    InstrItinData <tc_ef52ed71, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_ef84f62f, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f2704b9a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f3eaa14b, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f47d212f, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_f49e76f4, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_f4f43fb5, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_f7dd9c9f, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_f86c328a, [InstrStage<1, [SLOT0, SLOT1]>]>,
    InstrItinData <tc_f8eeed7a, [InstrStage<1, [SLOT2, SLOT3]>]>,
    InstrItinData <tc_fcab4871, [InstrStage<1, [SLOT0]>]>,
    InstrItinData <tc_ff9ee76e, [InstrStage<1, [SLOT0]>]>  ];
}

class DepScalarItinV55 {
  list<InstrItinData> DepScalarItinV55_list = [
    InstrItinData <tc_0077f68c, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_00afc57e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_00e7c26e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_03220ffa, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_038a1342, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_04c9decc, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_05b6c987, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0a2b8c7c, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0cd51c76, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0dc560de, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0fc1ae07, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_10b97e27, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_128f96e3, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1372bca1, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1432937d, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_14cd4cfa, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_15411484, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_16d0d8d5, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_181af5d0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1853ea6d, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b82a277, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_1b9c9ee5, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1c0005f9, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1d5a38a8, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e856f58, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_20280784, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_234a11a5, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_238d91d2, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29175780, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29641329, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2a160009, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_2b2f4060, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2b6f77c6, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2e00db30, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_2f185f5c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2fc0c436, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_351fed2d, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3669266a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_367f7f3d, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_36c68ad1, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_395dc00f, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3bc2c5d3, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_3cb8ea06, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3d04548d, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3da80ba5, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_3e07fb90, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_41d5298e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4403ca65, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_44126683, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_452f85af, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_481e5e5c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_49eb22c8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4ca572d4, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_4d9914c9, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4d99bca9, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4f7cd700, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_513bef45, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_51b866be, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_523fcf30, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5274e61a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_52d7bbea, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_53173427, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53bc8a6a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53bdb2f6, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_540fdfbc, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_55050d58, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_56d25411, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_57288781, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_594ab548, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_59a01ead, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5acef64a, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5ba5997d, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5eb851fc, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5f6847a1, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_60571023, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_609d2efe, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_60d76817, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_60f5738d, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_63fe3df7, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_66888ded, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6792d5ff, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_681a2300, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_68cb12ce, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6aa5711a, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6ac37025, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6ebb4a12, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6efc556e, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_6fa4db47, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_73043bf4, /*tc_2early*/
      [InstrStage<1, [SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_746baa8e, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_74e47fd9, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7934b9df, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7a830544, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7f881c76, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_84df2cd3, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_85523bcb, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_855b0b61, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_87735c3b, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_88fa1a78, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_897d1a9d, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8b15472a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8bb285ec, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_8fd5f294, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8fe6b782, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_90f3e30c, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_976ddc4f, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_97743097, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_994333cd, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_999d32db, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_99be14ca, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c00ce8d, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c98e8af, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9d5941c7, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9ef61e5c, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9faf76ae, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_9fdb5406, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a21dc435, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a27582fa, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_a46f0df5, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a788683e, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a8acdac0, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a904d137, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_adb14c66, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b13761ae, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_b166348b, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b44c6e2a, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5a33b22, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b77c481f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b7dd427e, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9488031, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9c0b731, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9c4623f, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bad2bcaf, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bcc96cee, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bd90564c, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bde7aaf4, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_be706f30, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c2f7d806, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c5e2426d, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6aa82f7, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6ce9b3f, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6ebf8dd, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c74f796f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c82dc1ff, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_caaebcba, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cd7374a0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cde8b071, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cf47a43f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cf59f215, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d088982c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d1090e34, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d24b2d85, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d580173f, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d6bf0472, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d9709180, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d9f95eef, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_daa058fa, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dbdffe3d, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e0739b8c, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e1e0a2dc, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_e1e99bfa, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e216a5db, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e421e012, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e6b38e01, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e7624c08, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_e7d02c66, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e913dc32, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e9c822f7, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_e9fae2d6, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef20db1c, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef52ed71, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef84f62f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f2704b9a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f3eaa14b, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f47d212f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f49e76f4, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f4f43fb5, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f7dd9c9f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f86c328a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f8eeed7a, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_fcab4871, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_ff9ee76e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV60 {
  list<InstrItinData> DepScalarItinV60_list = [
    InstrItinData <tc_0077f68c, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_00afc57e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_00e7c26e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_03220ffa, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_038a1342, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_04c9decc, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_05b6c987, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0a2b8c7c, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0cd51c76, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0dc560de, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0fc1ae07, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_10b97e27, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_128f96e3, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1372bca1, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1432937d, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_14cd4cfa, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_15411484, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_16d0d8d5, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_181af5d0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1853ea6d, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b82a277, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_1b9c9ee5, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1c0005f9, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1d5a38a8, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e856f58, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_20280784, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_234a11a5, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_238d91d2, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29175780, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29641329, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2a160009, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_2b2f4060, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2b6f77c6, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2e00db30, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_2f185f5c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2fc0c436, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_351fed2d, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3669266a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_367f7f3d, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_36c68ad1, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_395dc00f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3bc2c5d3, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_3cb8ea06, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3d04548d, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3da80ba5, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_3e07fb90, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_41d5298e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4403ca65, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_44126683, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_452f85af, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_481e5e5c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_49eb22c8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4ca572d4, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_4d9914c9, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4d99bca9, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4f7cd700, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_513bef45, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_51b866be, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_523fcf30, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5274e61a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_52d7bbea, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_53173427, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53bc8a6a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53bdb2f6, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_540fdfbc, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_55050d58, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_56d25411, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_57288781, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_594ab548, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_59a01ead, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5acef64a, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5ba5997d, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5eb851fc, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5f6847a1, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_60571023, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_609d2efe, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_60d76817, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_60f5738d, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_63fe3df7, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_66888ded, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6792d5ff, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_681a2300, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_68cb12ce, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6aa5711a, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6ac37025, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6ebb4a12, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6efc556e, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_6fa4db47, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_73043bf4, /*tc_2early*/
      [InstrStage<1, [SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_746baa8e, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_74e47fd9, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7934b9df, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7a830544, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7f881c76, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_84df2cd3, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_85523bcb, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_855b0b61, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_87735c3b, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_88fa1a78, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_897d1a9d, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8b15472a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8bb285ec, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_8fd5f294, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8fe6b782, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_90f3e30c, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_976ddc4f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_97743097, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_994333cd, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_999d32db, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_99be14ca, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c00ce8d, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c98e8af, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9d5941c7, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9ef61e5c, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9faf76ae, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_9fdb5406, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a21dc435, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a27582fa, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_a46f0df5, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a788683e, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a8acdac0, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a904d137, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_adb14c66, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b13761ae, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_b166348b, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b44c6e2a, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5a33b22, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b77c481f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b7dd427e, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9488031, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9c0b731, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9c4623f, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bad2bcaf, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bcc96cee, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bd90564c, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bde7aaf4, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_be706f30, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c2f7d806, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c5e2426d, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6aa82f7, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6ce9b3f, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6ebf8dd, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c74f796f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c82dc1ff, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_caaebcba, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cd7374a0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cde8b071, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cf47a43f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cf59f215, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d088982c, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d1090e34, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d24b2d85, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d580173f, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d6bf0472, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d9709180, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d9f95eef, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_daa058fa, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dbdffe3d, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e0739b8c, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e1e0a2dc, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_e1e99bfa, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e216a5db, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e421e012, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e6b38e01, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e7624c08, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_e7d02c66, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e913dc32, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e9c822f7, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_e9fae2d6, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef20db1c, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef52ed71, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef84f62f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f2704b9a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f3eaa14b, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f47d212f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f49e76f4, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f4f43fb5, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f7dd9c9f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f86c328a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f8eeed7a, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_fcab4871, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_ff9ee76e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV60se {
  list<InstrItinData> DepScalarItinV60se_list = [
    InstrItinData <tc_0077f68c, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_00afc57e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_00e7c26e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_03220ffa, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_038a1342, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_04c9decc, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_05b6c987, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0a2b8c7c, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0cd51c76, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0dc560de, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0fc1ae07, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_10b97e27, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_128f96e3, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1372bca1, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1432937d, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_14cd4cfa, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_15411484, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_16d0d8d5, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_181af5d0, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1853ea6d, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b82a277, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_1b9c9ee5, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1c0005f9, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1d5a38a8, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e856f58, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_20280784, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_234a11a5, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_238d91d2, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29175780, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29641329, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2a160009, /*tc_2early*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [],
      []>,

    InstrItinData <tc_2b2f4060, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2b6f77c6, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2e00db30, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_2f185f5c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2fc0c436, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_351fed2d, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3669266a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_367f7f3d, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_36c68ad1, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_395dc00f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3bc2c5d3, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_3cb8ea06, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3d04548d, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3da80ba5, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_3e07fb90, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_41d5298e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4403ca65, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_44126683, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_452f85af, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_481e5e5c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_49eb22c8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4ca572d4, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_4d9914c9, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4d99bca9, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4f7cd700, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_513bef45, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_51b866be, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_523fcf30, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5274e61a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_52d7bbea, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [],
      []>,

    InstrItinData <tc_53173427, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53bc8a6a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53bdb2f6, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_540fdfbc, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_55050d58, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_56d25411, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_57288781, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_594ab548, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_59a01ead, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5acef64a, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5ba5997d, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5eb851fc, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5f6847a1, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_60571023, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_609d2efe, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_60d76817, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_60f5738d, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_63fe3df7, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_66888ded, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6792d5ff, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_681a2300, /*tc_3stall*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_68cb12ce, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6aa5711a, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6ac37025, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6ebb4a12, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6efc556e, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_6fa4db47, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_73043bf4, /*tc_2early*/
      [InstrStage<1, [SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_746baa8e, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_74e47fd9, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7934b9df, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7a830544, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7f881c76, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_84df2cd3, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_85523bcb, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_855b0b61, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_87735c3b, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_88fa1a78, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_897d1a9d, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8b15472a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8bb285ec, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_8fd5f294, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8fe6b782, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_90f3e30c, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_976ddc4f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_97743097, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_994333cd, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_999d32db, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_99be14ca, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c00ce8d, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c98e8af, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9d5941c7, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9ef61e5c, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9faf76ae, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_9fdb5406, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a21dc435, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a27582fa, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_a46f0df5, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a788683e, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a8acdac0, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a904d137, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_adb14c66, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b13761ae, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_b166348b, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b44c6e2a, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5a33b22, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b77c481f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b7dd427e, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9488031, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9c0b731, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9c4623f, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bad2bcaf, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bcc96cee, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bd90564c, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bde7aaf4, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_be706f30, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c2f7d806, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c5e2426d, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6aa82f7, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6ce9b3f, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6ebf8dd, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c74f796f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c82dc1ff, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_caaebcba, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cd7374a0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cde8b071, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cf47a43f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cf59f215, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d088982c, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d1090e34, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d24b2d85, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d580173f, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d6bf0472, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d9709180, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d9f95eef, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_daa058fa, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dbdffe3d, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e0739b8c, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e1e0a2dc, /*tc_3stall*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [],
      []>,

    InstrItinData <tc_e1e99bfa, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e216a5db, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e421e012, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e6b38e01, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e7624c08, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_e7d02c66, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e913dc32, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e9c822f7, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_e9fae2d6, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef20db1c, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef52ed71, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef84f62f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f2704b9a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f3eaa14b, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f47d212f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f49e76f4, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f4f43fb5, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f7dd9c9f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f86c328a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f8eeed7a, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_fcab4871, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [],
      []>,

    InstrItinData <tc_ff9ee76e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV62 {
  list<InstrItinData> DepScalarItinV62_list = [
    InstrItinData <tc_0077f68c, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_00afc57e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_00e7c26e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_03220ffa, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_038a1342, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_04c9decc, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_05b6c987, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0a2b8c7c, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0cd51c76, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0dc560de, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0fc1ae07, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_10b97e27, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_128f96e3, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1372bca1, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1432937d, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_14cd4cfa, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_15411484, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_16d0d8d5, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_181af5d0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1853ea6d, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b82a277, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_1b9c9ee5, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1c0005f9, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1d5a38a8, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e856f58, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_20280784, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_234a11a5, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_238d91d2, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29175780, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29641329, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2a160009, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_2b2f4060, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2b6f77c6, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2e00db30, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_2f185f5c, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2fc0c436, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_351fed2d, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3669266a, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_367f7f3d, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_36c68ad1, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_395dc00f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3bc2c5d3, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_3cb8ea06, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3d04548d, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3da80ba5, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_3e07fb90, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_41d5298e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4403ca65, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_44126683, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_452f85af, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_481e5e5c, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_49eb22c8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4ca572d4, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_4d9914c9, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4d99bca9, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4f7cd700, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_513bef45, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_51b866be, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_523fcf30, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5274e61a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_52d7bbea, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_53173427, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53bc8a6a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53bdb2f6, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_540fdfbc, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_55050d58, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_56d25411, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_57288781, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_594ab548, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_59a01ead, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5acef64a, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5ba5997d, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5eb851fc, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5f6847a1, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_60571023, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_609d2efe, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_60d76817, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_60f5738d, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_63fe3df7, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_66888ded, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6792d5ff, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_681a2300, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_68cb12ce, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6aa5711a, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6ac37025, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6ebb4a12, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6efc556e, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_6fa4db47, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_73043bf4, /*tc_2early*/
      [InstrStage<1, [SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_746baa8e, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_74e47fd9, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7934b9df, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7a830544, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7f881c76, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_84df2cd3, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_85523bcb, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_855b0b61, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_87735c3b, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_88fa1a78, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_897d1a9d, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8b15472a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8bb285ec, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_8fd5f294, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8fe6b782, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_90f3e30c, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_976ddc4f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_994333cd, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_97743097, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_999d32db, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_99be14ca, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c00ce8d, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c98e8af, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9d5941c7, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9ef61e5c, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9faf76ae, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_9fdb5406, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a21dc435, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a27582fa, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_a46f0df5, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a788683e, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a8acdac0, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a904d137, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_adb14c66, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b13761ae, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_b166348b, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b44c6e2a, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5a33b22, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b77c481f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b7dd427e, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9488031, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9c0b731, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9c4623f, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bad2bcaf, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bcc96cee, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bd90564c, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bde7aaf4, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_be706f30, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c2f7d806, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c5e2426d, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6aa82f7, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6ce9b3f, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6ebf8dd, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c74f796f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c82dc1ff, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_caaebcba, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cd7374a0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cde8b071, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cf47a43f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cf59f215, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d088982c, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d1090e34, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d24b2d85, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d580173f, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d6bf0472, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d9709180, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d9f95eef, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_daa058fa, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dbdffe3d, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e0739b8c, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e1e0a2dc, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_e1e99bfa, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e216a5db, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e421e012, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e6b38e01, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e7624c08, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_e7d02c66, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e913dc32, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e9c822f7, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_e9fae2d6, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef20db1c, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef52ed71, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef84f62f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f2704b9a, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f3eaa14b, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f47d212f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f49e76f4, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f4f43fb5, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f7dd9c9f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f86c328a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f8eeed7a, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_fcab4871, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_ff9ee76e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV65 {
  list<InstrItinData> DepScalarItinV65_list = [
    InstrItinData <tc_0077f68c, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_00afc57e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_00e7c26e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_03220ffa, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_038a1342, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_04c9decc, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_05b6c987, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0a2b8c7c, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0cd51c76, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0dc560de, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_0fc1ae07, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_10b97e27, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_128f96e3, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1372bca1, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1432937d, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_14cd4cfa, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_15411484, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_16d0d8d5, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_181af5d0, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1853ea6d, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1b82a277, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_1b9c9ee5, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1c0005f9, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1d5a38a8, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_1e856f58, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_20280784, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_234a11a5, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_238d91d2, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29175780, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_29641329, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2a160009, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_2b2f4060, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2b6f77c6, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2e00db30, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_2f185f5c, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_2fc0c436, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_351fed2d, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3669266a, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_367f7f3d, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_36c68ad1, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_395dc00f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3bc2c5d3, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_3cb8ea06, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3d04548d, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_3da80ba5, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_3e07fb90, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_41d5298e, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4403ca65, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_44126683, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_452f85af, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_481e5e5c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_49eb22c8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4ca572d4, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_4d9914c9, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4d99bca9, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_4f7cd700, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_513bef45, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_51b866be, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_523fcf30, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5274e61a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_52d7bbea, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_53173427, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53bc8a6a, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_53bdb2f6, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_540fdfbc, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_55050d58, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_56d25411, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_57288781, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_594ab548, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_59a01ead, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5acef64a, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5ba5997d, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5eb851fc, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_5f6847a1, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_60571023, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_609d2efe, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_60d76817, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_60f5738d, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_63fe3df7, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_66888ded, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6792d5ff, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_681a2300, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_68cb12ce, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6aa5711a, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6ac37025, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6ebb4a12, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_6efc556e, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_6fa4db47, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_73043bf4, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_746baa8e, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_74e47fd9, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7934b9df, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7a830544, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_7f881c76, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_84df2cd3, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_85523bcb, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_855b0b61, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_87735c3b, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_88fa1a78, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_897d1a9d, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8b15472a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8bb285ec, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_8fd5f294, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_8fe6b782, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_90f3e30c, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_976ddc4f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_97743097, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_994333cd, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_999d32db, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_99be14ca, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c00ce8d, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9c98e8af, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9d5941c7, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9ef61e5c, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_9faf76ae, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_9fdb5406, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a21dc435, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a27582fa, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_a46f0df5, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a788683e, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a8acdac0, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_a904d137, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_adb14c66, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b13761ae, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_b166348b, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b44c6e2a, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b5a33b22, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b77c481f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b7dd427e, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9488031, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9c0b731, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_b9c4623f, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bad2bcaf, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bcc96cee, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bd90564c, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_bde7aaf4, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_be706f30, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c2f7d806, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c5e2426d, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6aa82f7, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6ce9b3f, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c6ebf8dd, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c74f796f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_c82dc1ff, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_caaebcba, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cd7374a0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cde8b071, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cf47a43f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_cf59f215, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d088982c, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d1090e34, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d24b2d85, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d580173f, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d6bf0472, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d9709180, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_d9f95eef, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_daa058fa, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dbdffe3d, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e0739b8c, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e1e0a2dc, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_e1e99bfa, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e216a5db, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e421e012, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e6b38e01, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e7624c08, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_e7d02c66, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e913dc32, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_e9c822f7, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_e9fae2d6, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef20db1c, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef52ed71, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_ef84f62f, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f2704b9a, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f3eaa14b, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f47d212f, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f49e76f4, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f4f43fb5, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f7dd9c9f, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f86c328a, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_f8eeed7a, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_fcab4871, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_ff9ee76e, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>
  ];
}
