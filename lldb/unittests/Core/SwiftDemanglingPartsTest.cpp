//===-- SwiftMangledTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "TestingSupport/TestUtilities.h"

#include "lldb/Core/DemangledNameInfo.h"
#include "lldb/Core/Mangled.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

struct SwiftDemanglingPartsTestCase {
  const char *mangled;
  DemangledNameInfo expected_info;
  std::string_view basename;
  std::string_view arguments;
};

SwiftDemanglingPartsTestCase g_swift_demangling_parts_test_cases[] = {
    // clang-format off
  { "_TFC3foo3bar3basfT3zimCS_3zim_T_",
    { /*.BasenameRange=*/{8, 11}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{11, 25},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 8}, /*.SuffixRange=*/{25, 31}
    },
    /*.basename=*/"bas",
    /*.arguments=*/"(zim: foo.zim)"
  },
  { "_TToFC3foo3bar3basfT3zimCS_3zim_T_",
    { /*.BasenameRange=*/{14, 17}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{17, 31},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 14}, /*.SuffixRange=*/{31, 37}
    },
    /*.basename=*/"bas",
    /*.arguments=*/"(zim: foo.zim)"
  },
  { "_TTOFSC3fooFTSdSd_Sd",
    { /*.BasenameRange=*/{25, 28}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{28, 56},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 25}, /*.SuffixRange=*/{56, 72}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(Swift.Double, Swift.Double)"
  },
  { "_T03foo3barC3basyAA3zimCAE_tFTo",
    { /*.BasenameRange=*/{14, 17}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{17, 31},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 14}, /*.SuffixRange=*/{31, 37}
    },
    /*.basename=*/"bas",
    /*.arguments=*/"(zim: foo.zim)"
  },
  { "_T0SC3fooS2d_SdtFTO",
    { /*.BasenameRange=*/{25, 28}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{28, 56},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 25}, /*.SuffixRange=*/{56, 72}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(Swift.Double, Swift.Double)"
  },
  { "_$s3foo3barC3bas3zimyAaEC_tFTo",
    { /*.BasenameRange=*/{14, 17}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{17, 31},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 14}, /*.SuffixRange=*/{31, 37}
    },
    /*.basename=*/"bas",
    /*.arguments=*/"(zim: foo.zim)"
  },
  { "_$sSC3fooyS2d_SdtFTO",
    { /*.BasenameRange=*/{25, 28}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{28, 56},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 25}, /*.SuffixRange=*/{56, 72}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(Swift.Double, Swift.Double)"
  },
  { "$s4main3fooyySiFyyXEfU_TA.1",
    { /*.BasenameRange=*/{28, 38}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{39, 41},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 28}, /*.SuffixRange=*/{41, 103}
    },
    /*.basename=*/"closure #1",
    /*.arguments=*/"()"
  },
  { "$s4main8MyStructV3fooyyFAA1XV_Tg5.foo",
    { /*.BasenameRange=*/{49, 52}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{52, 54},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 49}, /*.SuffixRange=*/{54, 89}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"()"
  },
  { "_TTDFC3foo3bar3basfT3zimCS_3zim_T_",
    { /*.BasenameRange=*/{16, 19}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{19, 33},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 16}, /*.SuffixRange=*/{33, 39}
    },
    /*.basename=*/"bas",
    /*.arguments=*/"(zim: foo.zim)"
  },
  { "_TF3foooi1pFTCS_3barVS_3bas_OS_3zim",
    { /*.BasenameRange=*/{4, 11}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{11, 29},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 4}, /*.SuffixRange=*/{29, 40}
    },
    /*.basename=*/"+ infix",
    /*.arguments=*/"(foo.bar, foo.bas)"
  },
  { "_TF3foooP1xFTCS_3barVS_3bas_OS_3zim",
    { /*.BasenameRange=*/{4, 13}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{13, 31},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 4}, /*.SuffixRange=*/{31, 42}
    },
    /*.basename=*/"^ postfix",
    /*.arguments=*/"(foo.bar, foo.bas)"
  },
  { "_TFC3foo3barCfT_S0_",
    { /*.BasenameRange=*/{8, 25}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{25, 27},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 8}, /*.SuffixRange=*/{27, 38}
    },
    /*.basename=*/"__allocating_init",
    /*.arguments=*/"()"
  },
  { "_TFC3foo3barcfT_S0_",
    { /*.BasenameRange=*/{8, 12}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{12, 14},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 8}, /*.SuffixRange=*/{14, 25}
    },
    /*.basename=*/"init",
    /*.arguments=*/"()"
  },
  { "_TIF1t1fFT1iSi1sSS_T_A_",
    { /*.BasenameRange=*/{24, 25}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{25, 56},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 24}, /*.SuffixRange=*/{56, 62}
    },
    /*.basename=*/"f",
    /*.arguments=*/"(i: Swift.Int, s: Swift.String)"
  },
  { "_TIF1t1fFT1iSi1sSS_T_A0_",
    { /*.BasenameRange=*/{24, 25}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{25, 56},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 24}, /*.SuffixRange=*/{56, 62}
    },
    /*.basename=*/"f",
    /*.arguments=*/"(i: Swift.Int, s: Swift.String)"
  },
  { "_TFSqcfT_GSqx_",
    { /*.BasenameRange=*/{15, 19}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{19, 21},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 15}, /*.SuffixRange=*/{21, 42}
    },
    /*.basename=*/"init",
    /*.arguments=*/"()"
  },
  { "_TF21class_bound_protocols32class_bound_protocol_compositionFT1xPS_10ClassBoundS_13NotClassBound__PS0_S1__",
    { /*.BasenameRange=*/{22, 54}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{54, 129},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 22}, /*.SuffixRange=*/{129, 203}
    },
    /*.basename=*/"class_bound_protocol_composition",
    /*.arguments=*/"(x: class_bound_protocols.ClassBound & class_bound_protocols.NotClassBound)"
  },
  { "_TFVCC6nested6AClass12AnotherClass7AStruct9aFunctionfT1aSi_S2_",
    { /*.BasenameRange=*/{35, 44}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{44, 58},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 35}, /*.SuffixRange=*/{58, 96}
    },
    /*.basename=*/"aFunction",
    /*.arguments=*/"(a: Swift.Int)"
  },
  { "_TFCF5types1gFT1bSb_T_L0_10Collection3zimfT_T_",
    { /*.BasenameRange=*/{0, 3}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{3, 5},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{5, 60}
    },
    /*.basename=*/"zim",
    /*.arguments=*/"()"
  },
  { "_TFF17capture_promotion22test_capture_promotionFT_FT_SiU_FT_Si_promote0",
    { /*.BasenameRange=*/{0, 10}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{11, 13},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{13, 125}
    },
    /*.basename=*/"closure #1",
    /*.arguments=*/"()"
  },
  { "_TFIVs8_Processi10_argumentsGSaSS_U_FT_GSaSS_",
    { /*.BasenameRange=*/{0, 10}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{0, 0},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{10, 130}
    },
    /*.basename=*/"_arguments",
    /*.arguments=*/""
  },
  { "_TFIvVs8_Process10_argumentsGSaSS_iU_FT_GSaSS_",
    { /*.BasenameRange=*/{0, 10}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{11, 13},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{13, 137}
    },
    /*.basename=*/"closure #1",
    /*.arguments=*/"()"
  },
  { "_TTWC13call_protocol1CS_1PS_FS1_3foofT_Si",
    { /*.BasenameRange=*/{37, 40}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{40, 42},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 37}, /*.SuffixRange=*/{42, 121}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"()"
  },
  { "_T013call_protocol1CCAA1PA2aDP3fooSiyFTW",
    { /*.BasenameRange=*/{37, 40}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{40, 42},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 37}, /*.SuffixRange=*/{42, 121}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"()"
  },
  { "_TFC12dynamic_self1X1ffT_DS0_",
    { /*.BasenameRange=*/{15, 16}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{16, 18},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 15}, /*.SuffixRange=*/{18, 26}
    },
    /*.basename=*/"f",
    /*.arguments=*/"()"
  },
  { "_TTSg5Si___TFSqcfT_GSqx_",
    { /*.BasenameRange=*/{53, 57}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{57, 59},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 53}, /*.SuffixRange=*/{59, 80}
    },
    /*.basename=*/"init",
    /*.arguments=*/"()"
  },
  { "_TTSgq5Si___TFSqcfT_GSqx_",
    { /*.BasenameRange=*/{65, 69}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{69, 71},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 65}, /*.SuffixRange=*/{71, 92}
    },
    /*.basename=*/"init",
    /*.arguments=*/"()"
  },
  { "_TTSg5SiSis3Foos_Sf___TFSqcfT_GSqx_",
    { /*.BasenameRange=*/{102, 106}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{106, 108},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 102}, /*.SuffixRange=*/{108, 129}
    },
    /*.basename=*/"init",
    /*.arguments=*/"()"
  },
  { "_TTSg5Si_Sf___TFSqcfT_GSqx_",
    { /*.BasenameRange=*/{66, 70}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{70, 72},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 66}, /*.SuffixRange=*/{72, 93}
    },
    /*.basename=*/"init",
    /*.arguments=*/"()"
  },
  { "_TTSr5Si___TF4test7genericurFxx",
    { /*.BasenameRange=*/{61, 68}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{71, 74},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 61}, /*.SuffixRange=*/{74, 79}
    },
    /*.basename=*/"generic",
    /*.arguments=*/"(A)"
  },
  { "_TTSrq5Si___TF4test7genericurFxx",
    { /*.BasenameRange=*/{73, 80}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{83, 86},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 73}, /*.SuffixRange=*/{86, 91}
    },
    /*.basename=*/"generic",
    /*.arguments=*/"(A)"
  },
  { "_TF8manglingX22egbpdajGbuEbxfgehfvwxnFT_T_",
    { /*.BasenameRange=*/{9, 43}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{43, 45},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{45, 51}
    },
    /*.basename=*/"ليهمابتكلموشعربي؟",
    /*.arguments=*/"()"
  },
  { "_TF8manglingX24ihqwcrbEcvIaIdqgAFGpqjyeFT_T_",
    { /*.BasenameRange=*/{9, 36}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{36, 38},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{38, 44}
    },
    /*.basename=*/"他们为什么不说中文",
    /*.arguments=*/"()"
  },
  { "_TF8manglingX27ihqwctvzcJBfGFJdrssDxIboAybFT_T_",
    { /*.BasenameRange=*/{9, 36}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{36, 38},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{38, 44}
    },
    /*.basename=*/"他們爲什麽不說中文",
    /*.arguments=*/"()"
  },
  { "_TF8manglingX30Proprostnemluvesky_uybCEdmaEBaFT_T_",
    { /*.BasenameRange=*/{9, 35}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{35, 37},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{37, 43}
    },
    /*.basename=*/"Pročprostěnemluvíčesky",
    /*.arguments=*/"()"
  },
  { "_TF8manglingXoi7p_qcaDcFTSiSi_Si",
    { /*.BasenameRange=*/{9, 20}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{20, 42},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{42, 55}
    },
    /*.basename=*/"«+» infix",
    /*.arguments=*/"(Swift.Int, Swift.Int)"
  },
  { "_TF8manglingoi2qqFTSiSi_T_",
    { /*.BasenameRange=*/{9, 17}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{17, 39},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{39, 45}
    },
    /*.basename=*/"?? infix",
    /*.arguments=*/"(Swift.Int, Swift.Int)"
  },
  { "_TFE11ext_structAV11def_structA1A4testfT_T_",
    { /*.BasenameRange=*/{41, 45}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{45, 47},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 41}, /*.SuffixRange=*/{47, 53}
    },
    /*.basename=*/"test",
    /*.arguments=*/"()"
  },
  { "_TF13devirt_accessP5_DISC15getPrivateClassFT_CS_P5_DISC12PrivateClass",
    { /*.BasenameRange=*/{14, 40}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{40, 42},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 14}, /*.SuffixRange=*/{42, 83}
    },
    /*.basename=*/"(getPrivateClass in _DISC)",
    /*.arguments=*/"()"
  },
  { "_TF4mainP5_mainX3wxaFT_T_",
    { /*.BasenameRange=*/{5, 18}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{18, 20},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{20, 26}
    },
    /*.basename=*/"(λ in _main)",
    /*.arguments=*/"()"
  },
  { "_TF4mainP5_main3abcFT_aS_P5_DISC3xyz",
    { /*.BasenameRange=*/{5, 19}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{19, 21},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{21, 44}
    },
    /*.basename=*/"(abc in _main)",
    /*.arguments=*/"()"
  },
  { "_TFCs13_NSSwiftArray29canStoreElementsOfDynamicTypefPMP_Sb",
    { /*.BasenameRange=*/{20, 49}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{49, 59},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 20}, /*.SuffixRange=*/{59, 73}
    },
    /*.basename=*/"canStoreElementsOfDynamicType",
    /*.arguments=*/"(Any.Type)"
  },
  { "_TTSf1cl35_TFF7specgen6callerFSiT_U_FTSiSi_T_Si___TF7specgen12take_closureFFTSiSi_T_T_",
    { /*.BasenameRange=*/{183, 195}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{195, 225},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 183}, /*.SuffixRange=*/{225, 231}
    },
    /*.basename=*/"take_closure",
    /*.arguments=*/"((Swift.Int, Swift.Int) -> ())"
  },
  { "_TTSfq1cl35_TFF7specgen6callerFSiT_U_FTSiSi_T_Si___TF7specgen12take_closureFFTSiSi_T_T_",
    { /*.BasenameRange=*/{195, 207}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{207, 237},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 195}, /*.SuffixRange=*/{237, 243}
    },
    /*.basename=*/"take_closure",
    /*.arguments=*/"((Swift.Int, Swift.Int) -> ())"
  },
  { "_TTSf1cl35_TFF7specgen6callerFSiT_U_FTSiSi_T_Si___TTSg5Si___TF7specgen12take_closureFFTSiSi_T_T_",
    { /*.BasenameRange=*/{221, 233}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{233, 263},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 221}, /*.SuffixRange=*/{263, 269}
    },
    /*.basename=*/"take_closure",
    /*.arguments=*/"((Swift.Int, Swift.Int) -> ())"
  },
  { "_TTSg5Si___TTSf1cl35_TFF7specgen6callerFSiT_U_FTSiSi_T_Si___TF7specgen12take_closureFFTSiSi_T_T_",
    { /*.BasenameRange=*/{221, 233}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{233, 263},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 221}, /*.SuffixRange=*/{263, 269}
    },
    /*.basename=*/"take_closure",
    /*.arguments=*/"((Swift.Int, Swift.Int) -> ())"
  },
  { "_TTSf1cpi0_cpfl0_cpse0v4u123_cpg53globalinit_33_06E7F1D906492AE070936A9B58CBAE1C_token8_cpfr36_TFtest_capture_propagation2_closure___TF7specgen12take_closureFFTSiSi_T_T_",
    { /*.BasenameRange=*/{357, 369}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{369, 399},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 357}, /*.SuffixRange=*/{399, 405}
    },
    /*.basename=*/"take_closure",
    /*.arguments=*/"((Swift.Int, Swift.Int) -> ())"
  },
  { "_TTSf0gs___TFVs17_LegacyStringCore15_invariantCheckfT_T_",
    { /*.BasenameRange=*/{105, 120}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{120, 122},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 105}, /*.SuffixRange=*/{122, 128}
    },
    /*.basename=*/"_invariantCheck",
    /*.arguments=*/"()"
  },
  { "_TTSf2g___TTSf2s_d___TFVs17_LegacyStringCoreCfVs13_StringBufferS_",
    { /*.BasenameRange=*/{164, 168}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{168, 189},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 164}, /*.SuffixRange=*/{189, 216}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(Swift._StringBuffer)"
  },
  { "_TTSf2dg___TTSf2s_d___TFVs17_LegacyStringCoreCfVs13_StringBufferS_",
    { /*.BasenameRange=*/{173, 177}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{177, 198},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 173}, /*.SuffixRange=*/{198, 225}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(Swift._StringBuffer)"
  },
  { "_TTSf2dgs___TTSf2s_d___TFVs17_LegacyStringCoreCfVs13_StringBufferS_",
    { /*.BasenameRange=*/{186, 190}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{190, 211},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 186}, /*.SuffixRange=*/{211, 238}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(Swift._StringBuffer)"
  },
  { "_TTSf3d_i_d_i_d_i___TFVs17_LegacyStringCoreCfVs13_StringBufferS_",
    { /*.BasenameRange=*/{209, 213}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{213, 234},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 209}, /*.SuffixRange=*/{234, 261}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(Swift._StringBuffer)"
  },
  { "_TTSf3d_i_n_i_d_i___TFVs17_LegacyStringCoreCfVs13_StringBufferS_",
    { /*.BasenameRange=*/{194, 198}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{198, 219},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 194}, /*.SuffixRange=*/{219, 246}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(Swift._StringBuffer)"
  },
  { "_TFFV23interface_type_mangling18GenericTypeContext23closureInGenericContexturFqd__T_L_3fooFTqd__x_T_",
    { /*.BasenameRange=*/{0, 6}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{7, 14},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{14, 103}
    },
    /*.basename=*/"foo #1",
    /*.arguments=*/"(A1, A)"
  },
  { "_TFFV23interface_type_mangling18GenericTypeContextg31closureInGenericPropertyContextxL_3fooFT_x",
    { /*.BasenameRange=*/{0, 6}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{7, 9},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{9, 103}
    },
    /*.basename=*/"foo #1",
    /*.arguments=*/"()"
  },
  { "_TTWurGV23interface_type_mangling18GenericTypeContextx_S_18GenericWitnessTestS_FS1_23closureInGenericContextuRxS1_rfqd__T_",
    { /*.BasenameRange=*/{64, 87}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{142, 146},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 64}, /*.SuffixRange=*/{146, 289}
    },
    /*.basename=*/"closureInGenericContext",
    /*.arguments=*/"(A1)"
  },
  { "_TTWurGV23interface_type_mangling18GenericTypeContextx_S_18GenericWitnessTestS_FS1_16twoParamsAtDepthu0_RxS1_rfTqd__1yqd_0__T_",
    { /*.BasenameRange=*/{64, 80}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{138, 149},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 64}, /*.SuffixRange=*/{149, 292}
    },
    /*.basename=*/"twoParamsAtDepth",
    /*.arguments=*/"(A1, y: B1)"
  },
  { "_TFC3red11BaseClassEHcfzT1aSi_S0_",
    { /*.BasenameRange=*/{16, 20}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{20, 34},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 16}, /*.SuffixRange=*/{34, 60}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(a: Swift.Int)"
  },
  { "_TFC4testP33_83378C430F65473055F1BD53F3ADCDB71C5doFoofT_T_",
    { /*.BasenameRange=*/{46, 51}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{51, 53},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 46}, /*.SuffixRange=*/{53, 59}
    },
    /*.basename=*/"doFoo",
    /*.arguments=*/"()"
  },
  { "_TFVV15nested_generics5Lunch6DinnerCfT11firstCoursex12secondCourseGSqqd___9leftoversx14transformationFxqd___GS1_x_qd___",
    { /*.BasenameRange=*/{29, 33}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{33, 124},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 29}, /*.SuffixRange=*/{124, 163}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(firstCourse: A, secondCourse: Swift.Optional<A1>, leftovers: A, transformation: (A) -> A1)"
  },
  { "_TFVFC15nested_generics7HotDogs11applyRelishFT_T_L_6RelishCfT8materialx_GS1_x_",
    { /*.BasenameRange=*/{0, 4}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{4, 17},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{17, 140}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(material: A)"
  },
  { "_TFVFE15nested_genericsSS3fooFT_T_L_6CheeseCfT8materialx_GS0_x_",
    { /*.BasenameRange=*/{0, 4}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{4, 17},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{17, 164}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(material: A)"
  },
  { "_T0s17MutableCollectionP1asAARzs012RandomAccessB0RzsAA11SubSequences013BidirectionalB0PRpzsAdHRQlE06rotatecD05Indexs01_A9IndexablePQzAM15shiftingToStart_tFAJs01_J4BasePQzAQcfU_",
    { /*.BasenameRange=*/{0, 10}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{11, 41},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{41, 435}
    },
    /*.basename=*/"closure #1",
    /*.arguments=*/"(A.Swift._IndexableBase.Index)"
  },
  { "_$Ss17MutableCollectionP1asAARzs012RandomAccessB0RzsAA11SubSequences013BidirectionalB0PRpzsAdHRQlE06rotatecD015shiftingToStart5Indexs01_A9IndexablePQzAN_tFAKs01_M4BasePQzAQcfU_",
    { /*.BasenameRange=*/{0, 10}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{11, 41},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 0}, /*.SuffixRange=*/{41, 435}
    },
    /*.basename=*/"closure #1",
    /*.arguments=*/"(A.Swift._IndexableBase.Index)"
  },
  { "_T04main5innerys5Int32Vz_yADctF25closure_with_box_argumentxz_Bi32__lXXTf1nc_n",
    { /*.BasenameRange=*/{151, 156}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{156, 196},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 151}, /*.SuffixRange=*/{196, 202}
    },
    /*.basename=*/"inner",
    /*.arguments=*/"(inout Swift.Int32, (Swift.Int32) -> ())"
  },
  { "_$S4main5inneryys5Int32Vz_yADctF25closure_with_box_argumentxz_Bi32__lXXTf1nc_n",
    { /*.BasenameRange=*/{151, 156}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{156, 196},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 151}, /*.SuffixRange=*/{196, 202}
    },
    /*.basename=*/"inner",
    /*.arguments=*/"(inout Swift.Int32, (Swift.Int32) -> ())"
  },
  { "_T03foo6testityyyc_yyctF1a1bTf3pfpf_n",
    { /*.BasenameRange=*/{132, 138}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{138, 158},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 132}, /*.SuffixRange=*/{158, 164}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"(() -> (), () -> ())"
  },
  { "_$S3foo6testityyyyc_yyctF1a1bTf3pfpf_n",
    { /*.BasenameRange=*/{132, 138}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{138, 158},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 132}, /*.SuffixRange=*/{158, 164}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"(() -> (), () -> ())"
  },
  { "_T0s10DictionaryV3t17E6Index2V1loiSbAEyxq__G_AGtFZ",
    { /*.BasenameRange=*/{50, 57}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{57, 157},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 50}, /*.SuffixRange=*/{157, 171}
    },
    /*.basename=*/"< infix",
    /*.arguments=*/"((extension in t17):Swift.Dictionary<A, B>.Index2, (extension in t17):Swift.Dictionary<A, B>.Index2)"
  },
  { "_T08mangling14varargsVsArrayySi3arrd_SS1ntF",
    { /*.BasenameRange=*/{9, 23}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{23, 59},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{59, 65}
    },
    /*.basename=*/"varargsVsArray",
    /*.arguments=*/"(arr: Swift.Int..., n: Swift.String)"
  },
  { "_T08mangling14varargsVsArrayySaySiG3arr_SS1ntF",
    { /*.BasenameRange=*/{9, 23}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{23, 69},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{69, 75}
    },
    /*.basename=*/"varargsVsArray",
    /*.arguments=*/"(arr: Swift.Array<Swift.Int>, n: Swift.String)"
  },
  { "_T08mangling14varargsVsArrayySaySiG3arrd_SS1ntF",
    { /*.BasenameRange=*/{9, 23}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{23, 72},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{72, 78}
    },
    /*.basename=*/"varargsVsArray",
    /*.arguments=*/"(arr: Swift.Array<Swift.Int>..., n: Swift.String)"
  },
  { "_T08mangling14varargsVsArrayySi3arrd_tF",
    { /*.BasenameRange=*/{9, 23}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{23, 42},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{42, 48}
    },
    /*.basename=*/"varargsVsArray",
    /*.arguments=*/"(arr: Swift.Int...)"
  },
  { "_T08mangling14varargsVsArrayySaySiG3arrd_tF",
    { /*.BasenameRange=*/{9, 23}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{23, 55},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{55, 61}
    },
    /*.basename=*/"varargsVsArray",
    /*.arguments=*/"(arr: Swift.Array<Swift.Int>...)"
  },
  { "_$Ss10DictionaryV3t17E6Index2V1loiySbAEyxq__G_AGtFZ",
    { /*.BasenameRange=*/{50, 57}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{57, 157},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 50}, /*.SuffixRange=*/{157, 171}
    },
    /*.basename=*/"< infix",
    /*.arguments=*/"((extension in t17):Swift.Dictionary<A, B>.Index2, (extension in t17):Swift.Dictionary<A, B>.Index2)"
  },
  { "_$S8mangling14varargsVsArray3arr1nySid_SStF",
    { /*.BasenameRange=*/{9, 23}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{23, 59},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{59, 65}
    },
    /*.basename=*/"varargsVsArray",
    /*.arguments=*/"(arr: Swift.Int..., n: Swift.String)"
  },
  { "_$S8mangling14varargsVsArray3arr1nySaySiG_SStF",
    { /*.BasenameRange=*/{9, 23}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{23, 69},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{69, 75}
    },
    /*.basename=*/"varargsVsArray",
    /*.arguments=*/"(arr: Swift.Array<Swift.Int>, n: Swift.String)"
  },
  { "_$S8mangling14varargsVsArray3arr1nySaySiGd_SStF",
    { /*.BasenameRange=*/{9, 23}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{23, 72},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{72, 78}
    },
    /*.basename=*/"varargsVsArray",
    /*.arguments=*/"(arr: Swift.Array<Swift.Int>..., n: Swift.String)"
  },
  { "_$S8mangling14varargsVsArray3arrySid_tF",
    { /*.BasenameRange=*/{9, 23}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{23, 42},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{42, 48}
    },
    /*.basename=*/"varargsVsArray",
    /*.arguments=*/"(arr: Swift.Int...)"
  },
  { "_$S8mangling14varargsVsArray3arrySaySiGd_tF",
    { /*.BasenameRange=*/{9, 23}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{23, 55},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{55, 61}
    },
    /*.basename=*/"varargsVsArray",
    /*.arguments=*/"(arr: Swift.Array<Swift.Int>...)"
  },
  { "_T010Foundation11MeasurementV12SimulatorKitSo9UnitAngleCRszlE11OrientationO2eeoiSbAcDEAGOyAF_G_AKtFZ",
    { /*.BasenameRange=*/{98, 106}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{106, 264},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 98}, /*.SuffixRange=*/{264, 278}
    },
    /*.basename=*/"== infix",
    /*.arguments=*/"((extension in SimulatorKit):Foundation.Measurement<__C.UnitAngle>.Orientation, (extension in SimulatorKit):Foundation.Measurement<__C.UnitAngle>.Orientation)"
  },
  { "_$S10Foundation11MeasurementV12SimulatorKitSo9UnitAngleCRszlE11OrientationO2eeoiySbAcDEAGOyAF_G_AKtFZ",
    { /*.BasenameRange=*/{98, 106}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{106, 264},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 98}, /*.SuffixRange=*/{264, 278}
    },
    /*.basename=*/"== infix",
    /*.arguments=*/"((extension in SimulatorKit):Foundation.Measurement<__C.UnitAngle>.Orientation, (extension in SimulatorKit):Foundation.Measurement<__C.UnitAngle>.Orientation)"
  },
  { "_T04main1_yyF",
    { /*.BasenameRange=*/{5, 6}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{6, 8},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{8, 14}
    },
    /*.basename=*/"_",
    /*.arguments=*/"()"
  },
  { "_T04test6testitSiyt_tF",
    { /*.BasenameRange=*/{5, 11}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{11, 15},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{15, 28}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"(())"
  },
  { "_$S4test6testitySiyt_tF",
    { /*.BasenameRange=*/{5, 11}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{11, 15},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{15, 28}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"(())"
  },
  { "_T03abc6testitySiFTm",
    { /*.BasenameRange=*/{11, 17}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{17, 28},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 11}, /*.SuffixRange=*/{28, 34}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"(Swift.Int)"
  },
  { "_T04main4TestCACSi1x_tc6_PRIV_Llfc",
    { /*.BasenameRange=*/{10, 26}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{26, 40},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 10}, /*.SuffixRange=*/{40, 53}
    },
    /*.basename=*/"(in _PRIV_).init",
    /*.arguments=*/"(x: Swift.Int)"
  },
  { "_$S3abc6testityySiFTm",
    { /*.BasenameRange=*/{11, 17}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{17, 28},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 11}, /*.SuffixRange=*/{28, 34}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"(Swift.Int)"
  },
  { "_$S4main4TestC1xACSi_tc6_PRIV_Llfc",
    { /*.BasenameRange=*/{10, 26}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{26, 40},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 10}, /*.SuffixRange=*/{40, 53}
    },
    /*.basename=*/"(in _PRIV_).init",
    /*.arguments=*/"(x: Swift.Int)"
  },
  { "_T03nix6testitSaySiGyFTv_",
    { /*.BasenameRange=*/{28, 34}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{34, 36},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 28}, /*.SuffixRange=*/{36, 62}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"()"
  },
  { "_T03nix6testitSaySiGyFTv_r",
    { /*.BasenameRange=*/{36, 42}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{42, 44},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 36}, /*.SuffixRange=*/{44, 70}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"()"
  },
  { "_T03nix6testitSaySiGyFTv0_",
    { /*.BasenameRange=*/{28, 34}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{34, 36},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 28}, /*.SuffixRange=*/{36, 62}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"()"
  },
  { "$sSo5GizmoC11doSomethingyypSgSaySSGSgFToTembgnn_",
    { /*.BasenameRange=*/{51, 62}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{62, 105},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 51}, /*.SuffixRange=*/{105, 128}
    },
    /*.basename=*/"doSomething",
    /*.arguments=*/"(Swift.Optional<Swift.Array<Swift.String>>)"
  },
  { "_T0s24_UnicodeScalarExceptions33_0E4228093681F6920F0AB2E48B4F1C69LLVACycfC",
    { /*.BasenameRange=*/{70, 74}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{74, 76},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 70}, /*.SuffixRange=*/{76, 149}
    },
    /*.basename=*/"init",
    /*.arguments=*/"()"
  },
  { "_T0s18EnumeratedIteratorVyxGs8Sequencess0B8ProtocolRzlsADP5splitSay03SubC0QzGSi9maxSplits_Sb25omittingEmptySubsequencesSb7ElementQzKc14whereSeparatortKFTW",
    { /*.BasenameRange=*/{36, 41}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{41, 152},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 36}, /*.SuffixRange=*/{152, 294}
    },
    /*.basename=*/"split",
    /*.arguments=*/"(maxSplits: Swift.Int, omittingEmptySubsequences: Swift.Bool, whereSeparator: (A.Element) throws -> Swift.Bool)"
  },
  { "$s18opaque_return_type3fooQryFQOHo",
    { /*.BasenameRange=*/{85, 88}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{88, 90},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 85}, /*.SuffixRange=*/{90, 100}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"()"
  },
  { "$s20mangling_retroactive5test0yyAA1ZVy12RetroactiveB1XVSiAE1YVAG0D1A1PAAyHCg_AiJ1QAAyHCg1_GF",
    { /*.BasenameRange=*/{21, 26}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{26, 93},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 21}, /*.SuffixRange=*/{93, 99}
    },
    /*.basename=*/"test0",
    /*.arguments=*/"(mangling_retroactive.Z<RetroactiveB.X, Swift.Int, RetroactiveB.Y>)"
  },
  { "$s20mangling_retroactive5test0yyAA1ZVy12RetroactiveB1XVSiAE1YVAG0D1A1PHPyHCg_AiJ1QHPyHCg1_GF",
    { /*.BasenameRange=*/{21, 26}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{26, 93},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 21}, /*.SuffixRange=*/{93, 99}
    },
    /*.basename=*/"test0",
    /*.arguments=*/"(mangling_retroactive.Z<RetroactiveB.X, Swift.Int, RetroactiveB.Y>)"
  },
  { "$s20mangling_retroactive5test0yyAA1ZVy12RetroactiveB1XVSiAE1YVAG0D1A1PHpyHCg_AiJ1QHpyHCg1_GF",
    { /*.BasenameRange=*/{21, 26}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{26, 93},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 21}, /*.SuffixRange=*/{93, 99}
    },
    /*.basename=*/"test0",
    /*.arguments=*/"(mangling_retroactive.Z<RetroactiveB.X, Swift.Int, RetroactiveB.Y>)"
  },
  { "_TTSf0os___TFVs17_LegacyStringCore15_invariantCheckfT_T_",
    { /*.BasenameRange=*/{105, 120}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{120, 122},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 105}, /*.SuffixRange=*/{122, 128}
    },
    /*.basename=*/"_invariantCheck",
    /*.arguments=*/"()"
  },
  { "_TTSf2o___TTSf2s_d___TFVs17_LegacyStringCoreCfVs13_StringBufferS_",
    { /*.BasenameRange=*/{164, 168}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{168, 189},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 164}, /*.SuffixRange=*/{189, 216}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(Swift._StringBuffer)"
  },
  { "_TTSf2do___TTSf2s_d___TFVs17_LegacyStringCoreCfVs13_StringBufferS_",
    { /*.BasenameRange=*/{173, 177}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{177, 198},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 173}, /*.SuffixRange=*/{198, 225}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(Swift._StringBuffer)"
  },
  { "_TTSf2dos___TTSf2s_d___TFVs17_LegacyStringCoreCfVs13_StringBufferS_",
    { /*.BasenameRange=*/{186, 190}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{190, 211},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 186}, /*.SuffixRange=*/{211, 238}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(Swift._StringBuffer)"
  },
  { "_TtCF4test11doNotCrash1FT_QuL_8MyClass1",
    { /*.BasenameRange=*/{20, 31}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{31, 33},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 20}, /*.SuffixRange=*/{33, 41}
    },
    /*.basename=*/"doNotCrash1",
    /*.arguments=*/"()"
  },
  { "$s4Test5ProtoP8IteratorV10collectionAEy_qd__Gqd___tcfc",
    { /*.BasenameRange=*/{20, 24}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{24, 40},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 20}, /*.SuffixRange=*/{40, 67}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(collection: A1)"
  },
  { "$s4test3fooV4blahyAA1SV1fQryFQOy_Qo_AHF",
    { /*.BasenameRange=*/{9, 13}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{13, 61},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{61, 111}
    },
    /*.basename=*/"blah",
    /*.arguments=*/"(<<opaque return type of test.S.f() -> some>>.0)"
  },
  { "$S3nix8MystructV1xACyxGx_tcfc7MyaliasL_ayx__GD",
    { /*.BasenameRange=*/{30, 34}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{34, 40},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 30}, /*.SuffixRange=*/{40, 59}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(x: A)"
  },
  { "$S3nix8MystructV6testit1xyx_tF7MyaliasL_ayx__GD",
    { /*.BasenameRange=*/{30, 36}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{36, 42},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 30}, /*.SuffixRange=*/{42, 48}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"(x: A)"
  },
  { "$S3nix8MystructV6testit1x1u1vyx_qd__qd_0_tr0_lF7MyaliasL_ayx_qd__qd_0__GD",
    { /*.BasenameRange=*/{30, 36}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{44, 64},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 30}, /*.SuffixRange=*/{64, 70}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"(x: A, u: A1, v: B1)"
  },
  { "$s1A1gyyxlFx_qd__t_Ti5",
    { /*.BasenameRange=*/{40, 41}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{44, 47},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 40}, /*.SuffixRange=*/{47, 53}
    },
    /*.basename=*/"g",
    /*.arguments=*/"(A)"
  },
  { "$s4Test6testityyxlFAA8MystructV_TB5",
    { /*.BasenameRange=*/{47, 53}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{56, 59},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 47}, /*.SuffixRange=*/{59, 65}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"(A)"
  },
  { "$sSUss17FixedWidthIntegerRzrlEyxqd__cSzRd__lufCSu_SiTg5",
    { /*.BasenameRange=*/{128, 132}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{165, 169},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 128}, /*.SuffixRange=*/{169, 174}
    },
    /*.basename=*/"init",
    /*.arguments=*/"(A1)"
  },
  { "$s4test7genFuncyyx_q_tr0_lFSi_SbTtt1g5",
    { /*.BasenameRange=*/{55, 62}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{68, 74},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 55}, /*.SuffixRange=*/{74, 80}
    },
    /*.basename=*/"genFunc",
    /*.arguments=*/"(A, B)"
  },
  { "$s4test3StrCACycfC",
    { /*.BasenameRange=*/{9, 26}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{26, 28},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{28, 40}
    },
    /*.basename=*/"__allocating_init",
    /*.arguments=*/"()"
  },
  { "$s3red4testyAA3ResOyxSayq_GAEs5ErrorAAq_sAFHD1__HCg_GADyxq_GsAFR_r0_lF",
    { /*.BasenameRange=*/{4, 8}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{35, 50},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 4}, /*.SuffixRange=*/{50, 80}
    },
    /*.basename=*/"test",
    /*.arguments=*/"(red.Res<A, B>)"
  },
  { "$s3red4testyAA7OurTypeOy4them05TheirD0Vy5AssocQzGAjE0F8ProtocolAAxAA0c7DerivedH0HD1_AA0c4BaseH0HI1_AieKHA2__HCg_GxmAaLRzlF",
    { /*.BasenameRange=*/{4, 8}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{43, 51},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 4}, /*.SuffixRange=*/{51, 91}
    },
    /*.basename=*/"test",
    /*.arguments=*/"(A.Type)"
  },
  { "$sSo17OS_dispatch_queueC4sync7executeyyyXE_tFTOTA",
    { /*.BasenameRange=*/{59, 63}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{63, 82},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 59}, /*.SuffixRange=*/{82, 88}
    },
    /*.basename=*/"sync",
    /*.arguments=*/"(execute: () -> ())"
  },
  { "$s7example1fyyYaF",
    { /*.BasenameRange=*/{8, 9}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{9, 11},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 8}, /*.SuffixRange=*/{11, 23}
    },
    /*.basename=*/"f",
    /*.arguments=*/"()"
  },
  { "$s7example1fyyYaKF",
    { /*.BasenameRange=*/{8, 9}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{9, 11},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 8}, /*.SuffixRange=*/{11, 30}
    },
    /*.basename=*/"f",
    /*.arguments=*/"()"
  },
  { "$s4main20receiveInstantiationyySo34__CxxTemplateInst12MagicWrapperIiEVzF",
    { /*.BasenameRange=*/{5, 25}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{25, 71},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{71, 77}
    },
    /*.basename=*/"receiveInstantiation",
    /*.arguments=*/"(inout __C.__CxxTemplateInst12MagicWrapperIiE)"
  },
  { "$s4main19returnInstantiationSo34__CxxTemplateInst12MagicWrapperIiEVyF",
    { /*.BasenameRange=*/{5, 24}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{24, 26},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{26, 68}
    },
    /*.basename=*/"returnInstantiation",
    /*.arguments=*/"()"
  },
  { "$s4main6testityyYaFTu",
    { /*.BasenameRange=*/{31, 37}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{37, 39},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 31}, /*.SuffixRange=*/{39, 51}
    },
    /*.basename=*/"testit",
    /*.arguments=*/"()"
  },
  { "$s13test_mangling3fooyS2f_S2ftFTJfUSSpSr",
    { /*.BasenameRange=*/{41, 44}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{44, 83},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 41}, /*.SuffixRange=*/{83, 148}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(Swift.Float, Swift.Float, Swift.Float)"
  },
  { "$s13test_mangling4foo21xq_x_t16_Differentiation14DifferentiableR_AA1P13TangentVectorRp_r0_lFAdERzAdER_AafGRpzAafHRQr0_lTJrSpSr",
    { /*.BasenameRange=*/{41, 45}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{126, 132},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 41}, /*.SuffixRange=*/{132, 341}
    },
    /*.basename=*/"foo2",
    /*.arguments=*/"(x: A)"
  },
  { "$s13test_mangling4foo21xq_x_t16_Differentiation14DifferentiableR_AA1P13TangentVectorRp_r0_lFAdERzAdER_AafGRpzAafHRQr0_lTJVrSpSr",
    { /*.BasenameRange=*/{58, 62}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{143, 149},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 58}, /*.SuffixRange=*/{149, 358}
    },
    /*.basename=*/"foo2",
    /*.arguments=*/"(x: A)"
  },
  { "$s13test_mangling3fooyS2f_xq_t16_Differentiation14DifferentiableR_r0_lFAcDRzAcDR_r0_lTJpUSSpSr",
    { /*.BasenameRange=*/{26, 29}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{76, 95},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 26}, /*.SuffixRange=*/{95, 249}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(Swift.Float, A, B)"
  },
  { "$s13test_mangling4foo21xq_x_t16_Differentiation14DifferentiableR_AA1P13TangentVectorRp_r0_lFTSAdERzAdER_AafGRpzAafHRQr0_lTJrSpSr",
    { /*.BasenameRange=*/{79, 83}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{164, 170},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 79}, /*.SuffixRange=*/{170, 379}
    },
    /*.basename=*/"foo2",
    /*.arguments=*/"(x: A)"
  },
  { "$s13test_mangling3fooyS2f_xq_t16_Differentiation14DifferentiableR_r0_lFAcDRzAcDR_r0_lTJpUSSpSrTj",
    { /*.BasenameRange=*/{44, 47}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{94, 113},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 44}, /*.SuffixRange=*/{113, 267}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(Swift.Float, A, B)"
  },
  { "$s13test_mangling3fooyS2f_xq_t16_Differentiation14DifferentiableR_r0_lFAcDRzAcDR_r0_lTJpUSSpSrTq",
    { /*.BasenameRange=*/{48, 51}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{98, 117},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 48}, /*.SuffixRange=*/{117, 271}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(Swift.Float, A, B)"
  },
  { "$s39differentiation_subset_parameters_thunk19inoutIndirectCalleryq_x_q_q0_t16_Differentiation14DifferentiableRzAcDR_AcDR0_r1_lFxq_Sdq_xq_Sdr0_ly13TangentVectorAcDPQy_AeFQzIsegnrr_Iegnnnro_TJSrSSSpSrSUSP",
    { /*.BasenameRange=*/{106, 125}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{247, 256},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 106}, /*.SuffixRange=*/{256, 658}
    },
    /*.basename=*/"inoutIndirectCaller",
    /*.arguments=*/"(A, B, C)"
  },
  { "$s13test_mangling3fooyS2f_S2ftFWJrSpSr",
    { /*.BasenameRange=*/{57, 60}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{60, 99},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 57}, /*.SuffixRange=*/{99, 161}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(Swift.Float, Swift.Float, Swift.Float)"
  },
  { "$s13test_mangling3fooyS2f_xq_t16_Differentiation14DifferentiableR_r0_lFAcDRzAcDR_r0_lWJrUSSpSr",
    { /*.BasenameRange=*/{57, 60}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{107, 126},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 57}, /*.SuffixRange=*/{126, 280}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(Swift.Float, A, B)"
  },
  { "$s5async1hyyS2iYbXEF",
    { /*.BasenameRange=*/{6, 7}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{7, 43},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 6}, /*.SuffixRange=*/{43, 49}
    },
    /*.basename=*/"h",
    /*.arguments=*/"(@Sendable (Swift.Int) -> Swift.Int)"
  },
  { "$s5Actor02MyA0C17testAsyncFunctionyyYaKFTY0_",
    { /*.BasenameRange=*/{54, 71}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{71, 73},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 54}, /*.SuffixRange=*/{73, 92}
    },
    /*.basename=*/"testAsyncFunction",
    /*.arguments=*/"()"
  },
  { "$s5Actor02MyA0C17testAsyncFunctionyyYaKFTQ1_",
    { /*.BasenameRange=*/{52, 69}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{69, 71},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 52}, /*.SuffixRange=*/{71, 90}
    },
    /*.basename=*/"testAsyncFunction",
    /*.arguments=*/"()"
  },
  { "$s4diff1hyyS2iYjfXEF",
    { /*.BasenameRange=*/{5, 6}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{6, 58},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{58, 64}
    },
    /*.basename=*/"h",
    /*.arguments=*/"(@differentiable(_forward) (Swift.Int) -> Swift.Int)"
  },
  { "$s4diff1hyyS2iYjrXEF",
    { /*.BasenameRange=*/{5, 6}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{6, 57},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{57, 63}
    },
    /*.basename=*/"h",
    /*.arguments=*/"(@differentiable(reverse) (Swift.Int) -> Swift.Int)"
  },
  { "$s4diff1hyyS2iYjdXEF",
    { /*.BasenameRange=*/{5, 6}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{6, 48},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{48, 54}
    },
    /*.basename=*/"h",
    /*.arguments=*/"(@differentiable (Swift.Int) -> Swift.Int)"
  },
  { "$s4diff1hyyS2iYjlXEF",
    { /*.BasenameRange=*/{5, 6}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{6, 57},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{57, 63}
    },
    /*.basename=*/"h",
    /*.arguments=*/"(@differentiable(_linear) (Swift.Int) -> Swift.Int)"
  },
  { "$s4test3fooyyS2f_SfYkztYjrXEF",
    { /*.BasenameRange=*/{5, 8}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{8, 96},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{96, 102}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(@differentiable(reverse) (Swift.Float, inout @noDerivative Swift.Float) -> Swift.Float)"
  },
  { "$s4test3fooyyS2f_SfYkntYjrXEF",
    { /*.BasenameRange=*/{5, 8}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{8, 98},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{98, 104}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(@differentiable(reverse) (Swift.Float, __owned @noDerivative Swift.Float) -> Swift.Float)"
  },
  { "$s4test3fooyyS2f_SfYktYjrXEF",
    { /*.BasenameRange=*/{5, 8}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{8, 90},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{90, 96}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(@differentiable(reverse) (Swift.Float, @noDerivative Swift.Float) -> Swift.Float)"
  },
  { "$s4test3fooyyS2f_SfYktYaYbYjrXEF",
    { /*.BasenameRange=*/{5, 8}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{8, 106},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{106, 112}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(@differentiable(reverse) @Sendable (Swift.Float, @noDerivative Swift.Float) async -> Swift.Float)"
  },
  { "$s1t10globalFuncyyAA7MyActorCYiF",
    { /*.BasenameRange=*/{2, 12}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{12, 32},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 2}, /*.SuffixRange=*/{32, 38}
    },
    /*.basename=*/"globalFunc",
    /*.arguments=*/"(isolated t.MyActor)"
  },
  { "$s6Foobar7Vector2VAASdRszlE10simdMatrix5scale6rotate9translateSo0C10_double3x3aACySdG_SdAJtFZ0D4TypeL_aySd__GD",
    { /*.BasenameRange=*/{102, 112}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{112, 212},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 102}, /*.SuffixRange=*/{212, 234}
    },
    /*.basename=*/"simdMatrix",
    /*.arguments=*/"(scale: Foobar.Vector2<Swift.Double>, rotate: Swift.Double, translate: Foobar.Vector2<Swift.Double>)"
  },
  { "$s17distributed_thunk2DAC1fyyFTE",
    { /*.BasenameRange=*/{39, 40}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{40, 42},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 39}, /*.SuffixRange=*/{42, 48}
    },
    /*.basename=*/"f",
    /*.arguments=*/"()"
  },
  { "$s16distributed_test1XC7computeyS2iFTF",
    { /*.BasenameRange=*/{44, 51}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{51, 62},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 44}, /*.SuffixRange=*/{62, 75}
    },
    /*.basename=*/"compute",
    /*.arguments=*/"(Swift.Int)"
  },
  { "$s27distributed_actor_accessors7MyActorC7simple2ySSSiFTETFHF",
    { /*.BasenameRange=*/{118, 125}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{125, 136},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 118}, /*.SuffixRange=*/{136, 152}
    },
    /*.basename=*/"simple2",
    /*.arguments=*/"(Swift.Int)"
  },
  { "$s1A3bar1aySSYt_tF",
    { /*.BasenameRange=*/{2, 5}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{5, 29},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 2}, /*.SuffixRange=*/{29, 35}
    },
    /*.basename=*/"bar",
    /*.arguments=*/"(a: _const Swift.String)"
  },
  { "$s1t1fyyFSiAA3StrVcs7KeyPathCyADSiGcfu_SiADcfu0_33_556644b740b1b333fecb81e55a7cce98ADSiTf3npk_n",
    { /*.BasenameRange=*/{258, 259}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{259, 261},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 258}, /*.SuffixRange=*/{261, 267}
    },
    /*.basename=*/"f",
    /*.arguments=*/"()"
  },
  { "$s21back_deploy_attribute0A12DeployedFuncyyFTwb",
    { /*.BasenameRange=*/{48, 64}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{64, 66},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 48}, /*.SuffixRange=*/{66, 72}
    },
    /*.basename=*/"backDeployedFunc",
    /*.arguments=*/"()"
  },
  { "$s21back_deploy_attribute0A12DeployedFuncyyFTwB",
    { /*.BasenameRange=*/{51, 67}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{67, 69},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 51}, /*.SuffixRange=*/{69, 75}
    },
    /*.basename=*/"backDeployedFunc",
    /*.arguments=*/"()"
  },
  { "$s4test3fooyyAA1P_px1TRts_XPlF",
    { /*.BasenameRange=*/{5, 8}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{11, 36},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{36, 42}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(any test.P<Self.T == A>)"
  },
  { "$s4test3fooyyAA1P_pSS1TAaCPRts_Si1UAERtsXPF",
    { /*.BasenameRange=*/{5, 8}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{8, 79},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{79, 85}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"(any test.P<Self.test.P.T == Swift.String, Self.test.P.U == Swift.Int>)"
  },
  { "$s4test3FooVAAyyAA1P_pF",
    { /*.BasenameRange=*/{9, 13}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{13, 21},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 9}, /*.SuffixRange=*/{21, 27}
    },
    /*.basename=*/"test",
    /*.arguments=*/"(test.P)"
  },
  { "$s7Library3fooyyFTwS",
    { /*.BasenameRange=*/{30, 33}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{33, 35},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 30}, /*.SuffixRange=*/{35, 41}
    },
    /*.basename=*/"foo",
    /*.arguments=*/"()"
  },
  { "$s9MacroUser13testStringify1a1bySi_SitF9stringifyfMf1_",
    { /*.BasenameRange=*/{58, 71}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{71, 99},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 58}, /*.SuffixRange=*/{99, 105}
    },
    /*.basename=*/"testStringify",
    /*.arguments=*/"(a: Swift.Int, b: Swift.Int)"
  },
  { "$s9MacroUser016testFreestandingA9ExpansionyyF4Foo3L_V23bitwidthNumberedStructsfMf_6methodfMu0_",
    { /*.BasenameRange=*/{111, 141}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{141, 143},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 111}, /*.SuffixRange=*/{143, 149}
    },
    /*.basename=*/"testFreestandingMacroExpansion",
    /*.arguments=*/"()"
  },
  { "@__swiftmacro_1a13testStringifyAA1bySi_SitF9stringifyfMf_",
    { /*.BasenameRange=*/{50, 63}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{63, 91},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 50}, /*.SuffixRange=*/{91, 97}
    },
    /*.basename=*/"testStringify",
    /*.arguments=*/"(a: Swift.Int, b: Swift.Int)"
  },
  { "$s12typed_throws15rethrowConcreteyyAA7MyErrorOYKF",
    { /*.BasenameRange=*/{13, 28}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{28, 30},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 13}, /*.SuffixRange=*/{30, 65}
    },
    /*.basename=*/"rethrowConcrete",
    /*.arguments=*/"()"
  },
  { "$s3red3use2fnySiyYAXE_tF",
    { /*.BasenameRange=*/{4, 7}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{7, 43},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 4}, /*.SuffixRange=*/{43, 49}
    },
    /*.basename=*/"use",
    /*.arguments=*/"(fn: @isolated(any) () -> Swift.Int)"
  },
  { "$s4testAAyAA5KlassC_ACtACnYTF",
    { /*.BasenameRange=*/{5, 9}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{9, 29},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{29, 65}
    },
    /*.basename=*/"test",
    /*.arguments=*/"(__owned test.Klass)"
  },
  { "$s5test24testyyAA5KlassCnYuF",
    { /*.BasenameRange=*/{6, 10}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{10, 39},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 6}, /*.SuffixRange=*/{39, 45}
    },
    /*.basename=*/"test",
    /*.arguments=*/"(sending __owned test2.Klass)"
  },
  { "$s4testA2A5KlassCyYTF",
    { /*.BasenameRange=*/{5, 9}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{9, 11},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{11, 33}
    },
    /*.basename=*/"test",
    /*.arguments=*/"()"
  },
  { "$s4null19transferAsyncResultAA16NonSendableKlassCyYaYTF",
    { /*.BasenameRange=*/{5, 24}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{24, 26},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 5}, /*.SuffixRange=*/{26, 65}
    },
    /*.basename=*/"transferAsyncResult",
    /*.arguments=*/"()"
  },
  { "$s3red7MyActorC3runyxxyYaKACYcYTXEYaKlFZ",
    { /*.BasenameRange=*/{19, 22}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{25, 68},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 19}, /*.SuffixRange=*/{68, 86}
    },
    /*.basename=*/"run",
    /*.arguments=*/"(@red.MyActor () async throws -> sending A)"
  },
  { "$s3red7MyActorC3runyxxyYaKYAYTXEYaKlFZ",
    { /*.BasenameRange=*/{19, 22}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{25, 70},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 19}, /*.SuffixRange=*/{70, 88}
    },
    /*.basename=*/"run",
    /*.arguments=*/"(@isolated(any) () async throws -> sending A)"
  },
  { "$s3red7MyActorC3runyxxyYaKYCXEYaKlFZ",
    { /*.BasenameRange=*/{19, 22}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{25, 71},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 19}, /*.SuffixRange=*/{71, 89}
    },
    /*.basename=*/"run",
    /*.arguments=*/"(nonisolated(nonsending) () async throws -> A)"
  },
  { "_$s15raw_identifiers0020pathfoo_yuEHaaCiJskayyF",
    { /*.BasenameRange=*/{16, 28}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{28, 30},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 16}, /*.SuffixRange=*/{30, 36}
    },
    /*.basename=*/"`path://foo`",
    /*.arguments=*/"()"
  },
  { "_$s15raw_identifiers10FontWeightO009_100_FpEpdyyFZ",
    { /*.BasenameRange=*/{34, 39}, /*.ScopeRange=*/{0, 0}, /*.ArgumentsRange=*/{39, 41},
    /*.QualifiersRange=*/{0, 0}, /*.PrefixRange=*/{0, 34}, /*.SuffixRange=*/{41, 47}
    },
    /*.basename=*/"`100`",
    /*.arguments=*/"()"
  },
    // clang-format on
};

struct SwiftDemanglingPartsTestFixture
    : public ::testing::TestWithParam<SwiftDemanglingPartsTestCase> {};

TEST_P(SwiftDemanglingPartsTestFixture, SwiftDemanglingParts) {
  const auto &[mangled, info, basename, arguments] = GetParam();

  TrackingNodePrinter printer =
      TrackingNodePrinter(swift::Demangle::DemangleOptions());
  swift::Demangle::demangleSymbolAsString(std::string(mangled), printer);
  std::string demangled = printer.takeString();
  DemangledNameInfo nameInfo = printer.takeInfo();
  nameInfo.PrefixRange.second =
      std::min(info.BasenameRange.first, info.ArgumentsRange.first);
  nameInfo.SuffixRange.first =
      std::max(info.BasenameRange.second, info.ArgumentsRange.second);
  nameInfo.SuffixRange.second = demangled.length();

  EXPECT_EQ(nameInfo.BasenameRange, info.BasenameRange);
  EXPECT_EQ(nameInfo.ScopeRange, info.ScopeRange);
  EXPECT_EQ(nameInfo.ArgumentsRange, info.ArgumentsRange);
  EXPECT_EQ(nameInfo.QualifiersRange, info.QualifiersRange);
  EXPECT_EQ(nameInfo.PrefixRange, info.PrefixRange);
  EXPECT_EQ(nameInfo.SuffixRange, info.SuffixRange);

  auto get_part = [&](const std::pair<size_t, size_t> &loc) {
    return demangled.substr(loc.first, loc.second - loc.first);
  };

  EXPECT_EQ(get_part(nameInfo.BasenameRange), basename);
  EXPECT_EQ(get_part(nameInfo.ArgumentsRange), arguments);
}

INSTANTIATE_TEST_SUITE_P(
    SwiftDemanglingPartsTests, SwiftDemanglingPartsTestFixture,
    ::testing::ValuesIn(g_swift_demangling_parts_test_cases));