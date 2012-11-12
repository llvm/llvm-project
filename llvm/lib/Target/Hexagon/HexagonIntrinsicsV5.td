class sf_SInst_sf<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1))]>;

class si_SInst_sf<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1))]>;

class sf_SInst_si<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins IntRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1))]>;

class sf_SInst_di<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1))]>;

class sf_SInst_df<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1))]>;

class si_SInst_df<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set IntRegs:$dst, (IntID DoubleRegs:$src1))]>;

class df_SInst_sf<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1))]>;

class di_SInst_sf<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1))]>;

class df_SInst_si<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins IntRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set DoubleRegs:$dst, (IntID IntRegs:$src1))]>;

class df_SInst_df<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1))]>;

class di_SInst_df<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1))]>;


class df_SInst_di<string opc, Intrinsic IntID>
  : SInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1),
             !strconcat("$dst = ", !strconcat(opc , "($src1)")),
             [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1))]>;

class sf_MInst_sfsf<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set IntRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class df_MInst_dfdf<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
           !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
           [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class qi_ALU64_dfdf<string opc, Intrinsic IntID>
  : ALU64_rr<(outs PredRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2),
           !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
           [(set PredRegs:$dst, (IntID DoubleRegs:$src1, DoubleRegs:$src2))]>;

class qi_ALU64_dfu5<string opc, Intrinsic IntID>
  : ALU64_ri<(outs PredRegs:$dst), (ins DoubleRegs:$src1, u5Imm:$src2),
           !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
           [(set PredRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2))]>;


class sf_MInst_sfsfsf_acc<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                        IntRegs:$dst2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1, $src2)")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1,
                                          IntRegs:$src2, IntRegs:$dst2))],
               "$dst2 = $dst">;

class sf_MInst_sfsfsf_nac<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                        IntRegs:$dst2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1, $src2)")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1,
                                          IntRegs:$src2, IntRegs:$dst2))],
               "$dst2 = $dst">;


class sf_MInst_sfsfsfsi_sc<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2, IntRegs:$src3),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1, $src2, $src3):scale")),
               [(set IntRegs:$dst, (IntID IntRegs:$dst2, IntRegs:$src1,
                                        IntRegs:$src2, IntRegs:$src3))],
               "$dst2 = $dst">;

class sf_MInst_sfsfsf_acc_lib<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                        IntRegs:$dst2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1, $src2):lib")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1,
                                          IntRegs:$src2, IntRegs:$dst2))],
               "$dst2 = $dst">;

class sf_MInst_sfsfsf_nac_lib<string opc, Intrinsic IntID>
  : MInst_acc<(outs IntRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2,
                                        IntRegs:$dst2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1, $src2):lib")),
               [(set IntRegs:$dst, (IntID IntRegs:$src1,
                                          IntRegs:$src2, IntRegs:$dst2))],
               "$dst2 = $dst">;

class df_MInst_dfdfdf_acc<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2,
                                        DoubleRegs:$dst2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1, $src2)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                          DoubleRegs:$src2, DoubleRegs:$dst2))],
               "$dst2 = $dst">;

class df_MInst_dfdfdf_nac<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2,
                                        DoubleRegs:$dst2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1, $src2)")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                          DoubleRegs:$src2, DoubleRegs:$dst2))],
               "$dst2 = $dst">;


class df_MInst_dfdfdfsi_sc<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$dst2, DoubleRegs:$src1,
                                        DoubleRegs:$src2, IntRegs:$src3),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1, $src2, $src3):scale")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$dst2, DoubleRegs:$src1,
                                        DoubleRegs:$src2, IntRegs:$src3))],
               "$dst2 = $dst">;

class df_MInst_dfdfdf_acc_lib<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2,
                                        DoubleRegs:$dst2),
               !strconcat("$dst += ", !strconcat(opc ,
                                                 "($src1, $src2):lib")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                          DoubleRegs:$src2, DoubleRegs:$dst2))],
               "$dst2 = $dst">;

class df_MInst_dfdfdf_nac_lib<string opc, Intrinsic IntID>
  : MInst_acc<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, DoubleRegs:$src2,
                                        DoubleRegs:$dst2),
               !strconcat("$dst -= ", !strconcat(opc ,
                                                 "($src1, $src2):lib")),
               [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1,
                                          DoubleRegs:$src2, DoubleRegs:$dst2))],
               "$dst2 = $dst">;

class qi_SInst_sfsf<string opc, Intrinsic IntID>
  : SInst<(outs PredRegs:$dst), (ins IntRegs:$src1, IntRegs:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, $src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, IntRegs:$src2))]>;

class qi_SInst_sfu5<string opc, Intrinsic IntID>
  : MInst<(outs PredRegs:$dst), (ins IntRegs:$src1, u5Imm:$src2),
             !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
             [(set PredRegs:$dst, (IntID IntRegs:$src1, imm:$src2))]>;

class sf_ALU64_u10_pos<string opc, Intrinsic IntID>
  : ALU64_ri<(outs IntRegs:$dst), (ins u10Imm:$src1),
             !strconcat("$dst = ", !strconcat(opc , "#$src1):pos")),
             [(set IntRegs:$dst, (IntID imm:$src1))]>;

class sf_ALU64_u10_neg<string opc, Intrinsic IntID>
  : ALU64_ri<(outs IntRegs:$dst), (ins u10Imm:$src1),
             !strconcat("$dst = ", !strconcat(opc , "#$src1):neg")),
             [(set IntRegs:$dst, (IntID imm:$src1))]>;

class df_ALU64_u10_pos<string opc, Intrinsic IntID>
  : ALU64_ri<(outs DoubleRegs:$dst), (ins u10Imm:$src1),
             !strconcat("$dst = ", !strconcat(opc , "#$src1):pos")),
             [(set DoubleRegs:$dst, (IntID imm:$src1))]>;

class df_ALU64_u10_neg<string opc, Intrinsic IntID>
  : ALU64_ri<(outs DoubleRegs:$dst), (ins u10Imm:$src1),
             !strconcat("$dst = ", !strconcat(opc , "#$src1):neg")),
             [(set DoubleRegs:$dst, (IntID imm:$src1))]>;

class di_MInst_diu6<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, u6Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2)")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2))]>;

class di_MInst_diu4_rnd<string opc, Intrinsic IntID>
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, u4Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2):rnd")),
          [(set DoubleRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2))]>;

class si_MInst_diu4_rnd_sat<string opc, Intrinsic IntID>
  : MInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, u4Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2):rnd:sat")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2))]>;

class si_SInst_diu4_sat<string opc, Intrinsic IntID>
  : SInst<(outs IntRegs:$dst), (ins DoubleRegs:$src1, u4Imm:$src2),
          !strconcat("$dst = ", !strconcat(opc , "($src1, #$src2):sat")),
          [(set IntRegs:$dst, (IntID DoubleRegs:$src1, imm:$src2))]>;


def HEXAGON_C4_fastcorner9:
    qi_SInst_qiqi       <"fastcorner9", int_hexagon_C4_fastcorner9>;
def HEXAGON_C4_fastcorner9_not:
    qi_SInst_qiqi <"!fastcorner9", int_hexagon_C4_fastcorner9_not>;
def HEXAGON_M5_vrmpybuu:
    di_MInst_didi <"vrmpybu", int_hexagon_M5_vrmpybuu>;
def HEXAGON_M5_vrmacbuu:
    di_MInst_dididi_acc <"vrmpybu", int_hexagon_M5_vrmacbuu>;
def HEXAGON_M5_vrmpybsu:
    di_MInst_didi <"vrmpybsu", int_hexagon_M5_vrmpybsu>;
def HEXAGON_M5_vrmacbsu:
    di_MInst_dididi_acc <"vrmpybsu", int_hexagon_M5_vrmacbsu>;
def HEXAGON_M5_vmpybuu:
    di_MInst_sisi <"vmpybu", int_hexagon_M5_vmpybuu>;
def HEXAGON_M5_vmpybsu:
    di_MInst_sisi <"vmpybsu", int_hexagon_M5_vmpybsu>;
def HEXAGON_M5_vmacbuu:
    di_MInst_disisi_acc <"vmpybu", int_hexagon_M5_vmacbuu>;
def HEXAGON_M5_vmacbsu:
    di_MInst_disisi_acc <"vmpybsu", int_hexagon_M5_vmacbsu>;
def HEXAGON_M5_vdmpybsu:
    di_MInst_didi_sat <"vdmpybsu", int_hexagon_M5_vdmpybsu>;
def HEXAGON_M5_vdmacbsu:
    di_MInst_dididi_acc_sat <"vdmpybsu", int_hexagon_M5_vdmacbsu>;
def HEXAGON_A5_vaddhubs:
    si_SInst_didi_sat <"vaddhub", int_hexagon_A5_vaddhubs>;
def HEXAGON_S5_popcountp:
    si_SInst_di <"popcount", int_hexagon_S5_popcountp>;
def HEXAGON_S5_asrhub_rnd_sat_goodsyntax:
    si_MInst_diu4_rnd_sat <"vasrhub", int_hexagon_S5_asrhub_rnd_sat_goodsyntax>;
def HEXAGON_S5_asrhub_sat:
    si_SInst_diu4_sat <"vasrhub", int_hexagon_S5_asrhub_sat>;
def HEXAGON_S5_vasrhrnd_goodsyntax:
    di_MInst_diu4_rnd <"vasrh", int_hexagon_S5_vasrhrnd_goodsyntax>;
def HEXAGON_S2_asr_i_p_rnd:
    di_SInst_diu6 <"asr", int_hexagon_S2_asr_i_p_rnd>;
def HEXAGON_S2_asr_i_p_rnd_goodsyntax:
    di_MInst_diu6 <"asrrnd", int_hexagon_S2_asr_i_p_rnd_goodsyntax>;
def HEXAGON_F2_sfadd:
    sf_MInst_sfsf <"sfadd", int_hexagon_F2_sfadd>;
def HEXAGON_F2_sfsub:
    sf_MInst_sfsf <"sfsub", int_hexagon_F2_sfsub>;
def HEXAGON_F2_sfmpy:
    sf_MInst_sfsf <"sfmpy", int_hexagon_F2_sfmpy>;
def HEXAGON_F2_sffma:
    sf_MInst_sfsfsf_acc <"sfmpy", int_hexagon_F2_sffma>;
def HEXAGON_F2_sffma_sc:
    sf_MInst_sfsfsfsi_sc <"sfmpy", int_hexagon_F2_sffma_sc>;
def HEXAGON_F2_sffms:
    sf_MInst_sfsfsf_nac <"sfmpy", int_hexagon_F2_sffms>;
def HEXAGON_F2_sffma_lib:
    sf_MInst_sfsfsf_acc_lib <"sfmpy", int_hexagon_F2_sffma_lib>;
def HEXAGON_F2_sffms_lib:
    sf_MInst_sfsfsf_nac_lib <"sfmpy", int_hexagon_F2_sffms_lib>;
def HEXAGON_F2_sfcmpeq:
    qi_SInst_sfsf <"sfcmp.eq", int_hexagon_F2_sfcmpeq>;
def HEXAGON_F2_sfcmpgt:
    qi_SInst_sfsf <"sfcmp.gt", int_hexagon_F2_sfcmpgt>;
def HEXAGON_F2_sfcmpge:
    qi_SInst_sfsf <"sfcmp.ge", int_hexagon_F2_sfcmpge>;
def HEXAGON_F2_sfcmpuo:
    qi_SInst_sfsf <"sfcmp.uo", int_hexagon_F2_sfcmpuo>;
def HEXAGON_F2_sfmax:
    sf_MInst_sfsf <"sfmax", int_hexagon_F2_sfmax>;
def HEXAGON_F2_sfmin:
    sf_MInst_sfsf <"sfmin", int_hexagon_F2_sfmin>;
def HEXAGON_F2_sfclass:
    qi_SInst_sfu5 <"sfclass", int_hexagon_F2_sfclass>;
def HEXAGON_F2_sfimm_p:
    sf_ALU64_u10_pos <"sfmake", int_hexagon_F2_sfimm_p>;
def HEXAGON_F2_sfimm_n:
    sf_ALU64_u10_neg <"sfmake", int_hexagon_F2_sfimm_n>;
def HEXAGON_F2_sffixupn:
    sf_MInst_sfsf <"sffixupn", int_hexagon_F2_sffixupn>;
def HEXAGON_F2_sffixupd:
    sf_MInst_sfsf <"sffixupd", int_hexagon_F2_sffixupd>;
def HEXAGON_F2_sffixupr:
    sf_SInst_sf <"sffixupr", int_hexagon_F2_sffixupr>;
def HEXAGON_F2_dfadd:
    df_MInst_dfdf <"dfadd", int_hexagon_F2_dfadd>;
def HEXAGON_F2_dfsub:
    df_MInst_dfdf <"dfsub", int_hexagon_F2_dfsub>;
def HEXAGON_F2_dfmpy:
    df_MInst_dfdf <"dfmpy", int_hexagon_F2_dfmpy>;
def HEXAGON_F2_dffma:
    df_MInst_dfdfdf_acc <"dfmpy", int_hexagon_F2_dffma>;
def HEXAGON_F2_dffms:
    df_MInst_dfdfdf_nac <"dfmpy", int_hexagon_F2_dffms>;
def HEXAGON_F2_dffma_lib:
    df_MInst_dfdfdf_acc_lib <"dfmpy", int_hexagon_F2_dffma_lib>;
def HEXAGON_F2_dffms_lib:
    df_MInst_dfdfdf_nac_lib <"dfmpy", int_hexagon_F2_dffms_lib>;
def HEXAGON_F2_dffma_sc:
    df_MInst_dfdfdfsi_sc <"dfmpy", int_hexagon_F2_dffma_sc>;
def HEXAGON_F2_dfmax:
    df_MInst_dfdf <"dfmax", int_hexagon_F2_dfmax>;
def HEXAGON_F2_dfmin:
    df_MInst_dfdf <"dfmin", int_hexagon_F2_dfmin>;
def HEXAGON_F2_dfcmpeq:
    qi_ALU64_dfdf <"dfcmp.eq", int_hexagon_F2_dfcmpeq>;
def HEXAGON_F2_dfcmpgt:
    qi_ALU64_dfdf <"dfcmp.gt", int_hexagon_F2_dfcmpgt>;
def HEXAGON_F2_dfcmpge:
    qi_ALU64_dfdf <"dfcmp.ge", int_hexagon_F2_dfcmpge>;
def HEXAGON_F2_dfcmpuo:
    qi_ALU64_dfdf <"dfcmp.uo", int_hexagon_F2_dfcmpuo>;
def HEXAGON_F2_dfclass:
    qi_ALU64_dfu5 <"dfclass", int_hexagon_F2_dfclass>;
def HEXAGON_F2_dfimm_p:
    df_ALU64_u10_pos <"dfmake", int_hexagon_F2_dfimm_p>;
def HEXAGON_F2_dfimm_n:
    df_ALU64_u10_neg <"dfmake", int_hexagon_F2_dfimm_n>;
def HEXAGON_F2_dffixupn:
    df_MInst_dfdf <"dffixupn", int_hexagon_F2_dffixupn>;
def HEXAGON_F2_dffixupd:
    df_MInst_dfdf <"dffixupd", int_hexagon_F2_dffixupd>;
def HEXAGON_F2_dffixupr:
    df_SInst_df <"dffixupr", int_hexagon_F2_dffixupr>;
def HEXAGON_F2_conv_sf2df:
    df_SInst_sf <"convert_sf2df", int_hexagon_F2_conv_sf2df>;
def HEXAGON_F2_conv_df2sf:
    sf_SInst_df <"convert_df2sf", int_hexagon_F2_conv_df2sf>;
def HEXAGON_F2_conv_uw2sf:
    sf_SInst_si <"convert_uw2sf", int_hexagon_F2_conv_uw2sf>;
def HEXAGON_F2_conv_uw2df:
    df_SInst_si <"convert_uw2df", int_hexagon_F2_conv_uw2df>;
def HEXAGON_F2_conv_w2sf:
    sf_SInst_si <"convert_w2sf", int_hexagon_F2_conv_w2sf>;
def HEXAGON_F2_conv_w2df:
    df_SInst_si <"convert_w2df", int_hexagon_F2_conv_w2df>;
def HEXAGON_F2_conv_ud2sf:
    sf_SInst_di <"convert_ud2sf", int_hexagon_F2_conv_ud2sf>;
def HEXAGON_F2_conv_ud2df:
    df_SInst_di <"convert_ud2df", int_hexagon_F2_conv_ud2df>;
def HEXAGON_F2_conv_d2sf:
    sf_SInst_di <"convert_d2sf", int_hexagon_F2_conv_d2sf>;
def HEXAGON_F2_conv_d2df:
    df_SInst_di <"convert_d2df", int_hexagon_F2_conv_d2df>;
def HEXAGON_F2_conv_sf2uw:
    si_SInst_sf <"convert_sf2uw", int_hexagon_F2_conv_sf2uw>;
def HEXAGON_F2_conv_sf2w:
    si_SInst_sf <"convert_sf2w", int_hexagon_F2_conv_sf2w>;
def HEXAGON_F2_conv_sf2ud:
    di_SInst_sf <"convert_sf2ud", int_hexagon_F2_conv_sf2ud>;
def HEXAGON_F2_conv_sf2d:
    di_SInst_sf <"convert_sf2d", int_hexagon_F2_conv_sf2d>;
def HEXAGON_F2_conv_df2uw:
    si_SInst_df <"convert_df2uw", int_hexagon_F2_conv_df2uw>;
def HEXAGON_F2_conv_df2w:
    si_SInst_df <"convert_df2w", int_hexagon_F2_conv_df2w>;
def HEXAGON_F2_conv_df2ud:
    di_SInst_df <"convert_df2ud", int_hexagon_F2_conv_df2ud>;
def HEXAGON_F2_conv_df2d:
    di_SInst_df <"convert_df2d", int_hexagon_F2_conv_df2d>;
def HEXAGON_F2_conv_sf2uw_chop:
    si_SInst_sf <"convert_sf2uw", int_hexagon_F2_conv_sf2uw_chop>;
def HEXAGON_F2_conv_sf2w_chop:
    si_SInst_sf <"convert_sf2w", int_hexagon_F2_conv_sf2w_chop>;
def HEXAGON_F2_conv_sf2ud_chop:
    di_SInst_sf <"convert_sf2ud", int_hexagon_F2_conv_sf2ud_chop>;
def HEXAGON_F2_conv_sf2d_chop:
    di_SInst_sf <"convert_sf2d", int_hexagon_F2_conv_sf2d_chop>;
def HEXAGON_F2_conv_df2uw_chop:
    si_SInst_df <"convert_df2uw", int_hexagon_F2_conv_df2uw_chop>;
def HEXAGON_F2_conv_df2w_chop:
    si_SInst_df <"convert_df2w", int_hexagon_F2_conv_df2w_chop>;
def HEXAGON_F2_conv_df2ud_chop:
    di_SInst_df <"convert_df2ud", int_hexagon_F2_conv_df2ud_chop>;
def HEXAGON_F2_conv_df2d_chop:
    di_SInst_df <"convert_df2d", int_hexagon_F2_conv_df2d_chop>;
