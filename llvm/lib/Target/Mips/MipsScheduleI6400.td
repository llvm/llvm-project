//==- MipsScheduleI6400.td - I6400 Scheduling Definitions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def MipsI6400Model : SchedMachineModel {
  int IssueWidth = 2;         // 2x dispatched per cycle
  int MicroOpBufferSize = 0;
  int LoadLatency = 3;
  int MispredictPenalty = 8;

  let CompleteModel = 0;
  let FullInstRWOverlapCheck = 1;

  list<Predicate> UnsupportedFeatures = [HasMips64r5, HasMips32r5,
                                         InMicroMips, InMips16Mode, HasCnMips,
					 HasCnMipsP,  HasDSP, HasDSPR2,
					 HasMips3D, HasMT, HasCRC,
					 NotMips32r6, NotMips64r6, HasEVA];
}

let SchedModel = MipsI6400Model in {

  // I6400 Resources
  // ===============
  let BufferSize = 0 in {

    def I6400AGEN : ProcResource<1>;
    def I6400IssueLSU : ProcResource<1> { let Super = I6400AGEN; }
    def I6400IssueALU1 : ProcResource<1> { let Super = I6400AGEN; }
    def I6400CTRL : ProcResource<1>;
    def I6400IssueCTU : ProcResource<1> { let Super = I6400CTRL; }
    def I6400IssueALU0 : ProcResource<1> { let Super = I6400CTRL; }
    def I6400MDU : ProcResource<1>;
    def I6400FPU : ProcResource<3>;
    def I6400FPUShort : ProcResource<1> { let Super = I6400FPU; }
    def I6400FPULong : ProcResource<1> { let Super = I6400FPU; }
    def I6400FPUApu : ProcResource<1>;
    def I6400FPUFloatL : ProcResource<1>;
    def I6400Atomic : ProcResource<1>;
  }

  // AGEN Pipelines
  // ==============
  def I6400WriteLSUStore : SchedWriteRes<[I6400IssueLSU]>;
  def I6400WriteLSUStore2 : SchedWriteRes<[I6400IssueLSU]> {
    let Latency = 8;
    let ReleaseAtCycles = [5];
  }
  def I6400WriteLSULoad : SchedWriteRes<[I6400IssueLSU]> { let Latency = 3; }
  def I6400WriteLSUPref : SchedWriteRes<[I6400IssueLSU]>;
  def I6400WriteLSUOther : SchedWriteRes<[I6400IssueLSU]> {
    let Latency = 6;
    let ReleaseAtCycles = [5];
  }

  // LSU pipelines
  // =============
  def : InstRW<[I6400WriteLSUStore], (instrs SB, SD, SH, SW, SDC1, SDC164, SWC1, SWC2_R6,
				      SDC2_R6, SDC3)>;
  def : InstRW<[I6400WriteLSUStore2], (instrs SC_R6, SCD_R6, SYNCI, TLBP, TLBR,
                                          TLBWI, TLBWR, TLBINV, TLBINVF,
                                          CACHE_R6, SC64_R6)>;
  def : InstRW<[I6400WriteLSULoad],
               (instrs LB, LBu, LBu64, LD, LH, LHu, LHu64, LW, LWu, LDC1,
		   LDC164, LWC1, LD_F16, ST_F16, LDC2_R6, LDC3, LWC2_R6,
                   LLD_R6, LL_R6, LWPC, LWUPC, LDPC, ST_B, ST_H, ST_W, ST_D,
		   LB64, LH64, LW64, LWL64, LWR64, SB64, SH64, SW64, SWL64,
		   SWR64, LL64_R6)>;
  def : InstRW<[I6400WriteLSUPref], (instrs PREF, PREF_R6, PAUSE)>;
  def : InstRW<[I6400WriteLSUOther], (instrs SYNC)>;

  // CONTROL Pipelines
  // =================

  def I6400WriteALU0 : SchedWriteRes<[I6400IssueALU0]>;
  def I6400WriteALU1 : SchedWriteRes<[I6400IssueALU1]>;
  def I6400WriteCTU : SchedWriteRes<[I6400IssueCTU]>;

  // CTU pipelines
  // =============
  def : InstRW<[I6400WriteCTU],
               (instrs J, JAL, JALR, B, BEQ, BNE, BGEZ, BGTZ, BLEZ, BLTZ,
                   JALR64, JALR64Pseudo, JIALC, JIALC64, JIC, JIC64, JR64,
		   JR_HB_R6,
                   JR_HB64_R6, NAL, SDBBP_R6, SYSCALL, BEQC64, BEQZC64, BGEC64,
                   BGEUC64, BGTZC64, BLEZC64, BLTC64, BLTUC64, BNEC64, BNEZC64,
                   PseudoIndirectBranchR6, BC, BALC, BEQZC, BNEZC, BLEZC, BGEZC,
                   BGTZC, BLTZC, BEQZALC, BNEZALC, BLEZALC, BGEZALC, BGTZALC,
                   BLTZALC, BEQC, BNEC, BGEC, BLTC, BGEUC, BLTUC, BAL, BOVC,
                   BNVC, BC1EQZ, BC1NEZ, BREAK, ERET, ERETNC, BAL_BR, DERET,
		   JALRHBPseudo, JALRPseudo, JALR_HB, JALR_HB64, TAILCALL,
		   PseudoIndirectBranch64R6, TAILCALL64R6REG,
		   TAILCALLR6REG, PseudoIndrectHazardBranch64R6,
		   PseudoIndrectHazardBranchR6, TAILCALLHB64R6REG,
		   TAILCALLHBR6REG, PseudoReturn, PseudoReturn64, ERet, RetRA,
		   BC2EQZ, BC2NEZ, TLT, TLTU, TNE, WAIT, DI, TRAP, EI,
		   BEQ64, BGEZ64, BGEZC64, BGTZ64, BGTZ64, BLEZ64, BLTZ64,
		   BLTZC64, BNE64, JALRHB64Pseudo)>;

  // Either ALU0 or ALU1 pipelines
  // =============================
  def I6400IssueEitherALU : ProcResGroup<[I6400IssueALU0, I6400IssueALU1]>;
  def I6400WriteEitherALU : SchedWriteRes<[I6400IssueEitherALU]>;

  def : InstRW<[I6400WriteEitherALU],
               (instrs ADD, ADDiu, ADDIUPC, ADDu, ALIGN, ALUIPC, AND, ANDi, AUI,
                   AUIPC, BITSWAP, CFC1, CLO_R6, CLZ_R6, CTC1, DADD, DADDiu,
                   DADDu, DAHI, DALIGN, DATI, DAUI, DBITSWAP, DCLO_R6, DCLZ_R6,
                   DEXT, DEXT64_32, DEXTM, DEXTU, DINS, DINSM, DINSU, DLSA_R6,
                   DMFC1, DMTC1, DROTR, DROTR32, DROTRV, DSBH, DSHD, DSLL,
                   DSLL32, DSLLV, DSRA, DSRA32, DSRAV, DSRL, DSRL32, DSRLV,
                   DSUB, DSUBu, EXT, INS, LSA, LSA_R6, LUi, MFC1, MFC1_D64,
		   MFC0, MFC2, MTC0, MTC2,
                   MFHC1_D32, MFHC1_D64, MTC1, MTC1_D64, MTHC1_D32, MTHC1_D64,
                   NOP, NOR, OR, ORi, ROTR, ROTRV, SEB, SEB64, SEH, SEH64,
                   SELEQZ, SELEQZ64, SELNEZ, SELNEZ64, SLL, SLLV, SLT, SLTi, SLTiu, SLTu, SRA, SRAV,
                   SRL, SRLV, SSNOP, SUB, SUBu, WSBH, XOR, XORi, SLT64, SLTu64,
                   AND64, OR64, XOR64, NOR64, SLTi64, SLTiu64, ANDi64, ORi64,
                   XORi64, LUi64, DSLL64_32, SLL64_32, SLL64_64,
                   LONG_BRANCH_LUi2Op_64, LONG_BRANCH_DADDiu2Op,
                   LONG_BRANCH_DADDiu, DLSA,
		   TEQ, TGE, TGEU, COPY,
		   BuildPairF64, BuildPairF64_64,
		   ExtractElementF64, ExtractElementF64_64,
		   SELNEZ_D, SELNEZ_S, SELEQZ_D, SELEQZ_S,
		   SEL_D, SEL_S, EHB, RDHWR, RDHWR64, EVP, DVP,
		   DMFC0, DMFC2, DMTC0, DMTC2)>;

  // MDU pipelines
  // =============
  def I6400GPMUL : SchedWriteRes<[I6400MDU]> { let Latency = 4; }
  def : InstRW<[I6400GPMUL], (instrs MUL_R6, MULU, MUH, MUHU, DMUL_R6, DMUH,
                                 DMULU, DMUHU)>;

  def I6400GPDIV : SchedWriteRes<[I6400MDU]> { let Latency = 32; }
  def : InstRW<[I6400GPDIV], (instrs DIV, DIVU, MOD, MODU, DDIV, DMOD, DDIVU,
                                 DMODU)>;
  def : InstRW<[I6400GPDIV], (instregex "^MOD_(S|U)_[BHWD]$")>;

  // FPU pipelines
  // =============

  def I6400FPUFabs : SchedWriteRes<[I6400FPUShort]>;
  def : InstRW<[I6400FPUFabs], (instrs FABS_S, FNEG_S, FMOV_S,
				FMOV_D32, FMOV_D64,
				FNEG_D32, FNEG_D64, CLASS_S, CLASS_D)>;

  def I6400FPUFadd : SchedWriteRes<[I6400FPULong]> {
    let Latency = 4;
    let ReleaseAtCycles = [1];
  }
  def : InstRW<[I6400FPUFadd], (instrs FADD_S, FSUB_S, FSUB_D32, FSUB_D64)>;

  def I6400FPUFmul : SchedWriteRes<[I6400FPULong, I6400FPUApu]> {
    let Latency = 5;
    let ReleaseAtCycles = [1, 4];
  }
  def : InstRW<[I6400FPUFmul], (instrs FMUL_S, FMUL_D32, FMUL_D64)>;

  def I6400FPUFdivS : SchedWriteRes<[I6400FPULong, I6400FPUApu]> {
    let Latency = 22;
    let ReleaseAtCycles = [1, 21];
  }
  def : InstRW<[I6400FPUFdivS], (instrs FDIV_S, FSQRT_S, FDIV_D32, FDIV_D64,
				 FSQRT_D, FSQRT_D32, FSQRT_D64,
				 FSQRT_W, RECIP_S, RECIP_D32, RECIP_D64,
				 FRSQRT_D, FRSQRT_W, RSQRT_S, RSQRT_D32,
				 RSQRT_D64, FRCP_D, FRCP_W)>;
  def : InstRW<[I6400FPUFdivS], (instregex "^DIV_S_(B|D|H|W)$")>;
  def : InstRW<[I6400FPUFdivS], (instregex "^DIV_U_(B|D|H|W)$")>;

  // MSA pipelines
  // =============
  // Short pipe
  def I6400MSAAdd : SchedWriteRes<[I6400FPUShort]> {
    let Latency = 2;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSAAdd], (instrs ADDV_B, ADDV_H, ADDV_W, ADDV_D, ADDVI_B,
                                  ADDVI_H, ADDVI_W, ADDVI_D, SUBV_B, SUBV_H,
                                  SUBV_W, SUBV_D, SUBVI_B, SUBVI_H, SUBVI_W,
                                  SUBVI_D)>;
  def : InstRW<[I6400MSAAdd], (instregex "^ASUB_S_(B|D|H|W)$")>;
  def : InstRW<[I6400MSAAdd], (instregex "^ASUB_U_(B|D|H|W)$")>;
  def : InstRW<[I6400MSAAdd], (instregex "^SUBSUS_U_(B|D|H|W)$")>;

  def I6400MSAMaxMin : SchedWriteRes<[I6400FPUShort]> {
    let Latency = 2;
    let ReleaseAtCycles = [1];
  }
  def : InstRW<[I6400MSAMaxMin], (instrs MAX_D, MAX_S, MAXA_D, MAXA_S,
				  MIN_D, MIN_S, MINA_D, MINA_S)>;

  def I6400MSAIntAdd : SchedWriteRes<[I6400FPUShort]> {
    let Latency = 2;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSAIntAdd],
               (instrs ADD_A_B, ADD_A_H, ADD_A_W, ADD_A_D, ADDS_A_B, ADDS_A_H,
                   ADDS_A_W, ADDS_A_D, ADDS_S_B, ADDS_S_H, ADDS_S_W, ADDS_S_D,
                   ADDS_U_B, ADDS_U_H, ADDS_U_W, ADDS_U_D, HADD_S_H, HADD_S_W,
                   HADD_S_D, HADD_U_H, HADD_U_W, HADD_U_D, SUBS_S_B, SUBS_S_H,
                   SUBS_S_W, SUBS_S_D, SUBS_U_B, SUBS_U_H, SUBS_U_W, SUBS_U_D,
                   SUBSUU_S_B, SUBSUU_S_H, SUBSUU_S_W, SUBSUU_S_D, HSUB_S_H,
                   HSUB_S_W, HSUB_S_D, HSUB_U_H, HSUB_U_W, HSUB_U_D, AVE_S_B,
                   AVE_S_H, AVE_S_W, AVE_S_D, AVE_U_B, AVE_U_H, AVE_U_W,
                   AVE_U_D, AVER_S_B, AVER_S_H, AVER_S_W, AVER_S_D, AVER_U_B,
                   AVER_U_H, AVER_U_W, AVER_U_D, MIN_A_B, MIN_A_H, MIN_A_W,
                   MIN_A_D, MIN_S_B, MIN_S_H, MIN_S_W, MIN_S_D, MIN_U_B,
                   MIN_U_H, MIN_U_W, MIN_U_D, MINI_S_B, MINI_S_H, MINI_S_W,
                   MINI_S_D, MINI_U_B, MINI_U_H, MINI_U_W, MINI_U_D, MAX_A_B,
                   MAX_A_H, MAX_A_W, MAX_A_D, MAX_S_B, MAX_S_H, MAX_S_W,
                   MAX_S_D, MAX_U_B, MAX_U_H, MAX_U_W, MAX_U_D, MAXI_S_B,
                   MAXI_S_H, MAXI_S_W, MAXI_S_D, MAXI_U_B, MAXI_U_H, MAXI_U_W,
                   MAXI_U_D, CEQ_B, CEQ_H, CEQ_W, CEQ_D, CEQI_B, CEQI_H, CEQI_W,
                   CEQI_D, CLE_S_B, CLE_S_H, CLE_S_W, CLE_S_D, CLE_U_B, CLE_U_H,
                   CLE_U_W, CLE_U_D, CLEI_S_B, CLEI_S_H, CLEI_S_W, CLEI_S_D,
                   CLEI_U_B, CLEI_U_H, CLEI_U_W, CLEI_U_D, CLT_S_B, CLT_S_H,
                   CLT_S_W, CLT_S_D, CLT_U_B, CLT_U_H, CLT_U_W, CLT_U_D,
                   CLTI_S_B, CLTI_S_H, CLTI_S_W, CLTI_S_D, CLTI_U_B, CLTI_U_H,
                   CLTI_U_W, CLTI_U_D)>;

  def I6400MSAShortLogic3 : SchedWriteRes<[I6400FPUShort]> {
    let Latency = 3;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSAShortLogic3],
               (instrs SAT_S_B, SAT_S_H, SAT_S_W, SAT_S_D, SAT_U_B, SAT_U_H,
                   SAT_U_W, SAT_U_D, PCNT_B, PCNT_H, PCNT_W, PCNT_D)>;

  def I6400MSAShortLogic2 : SchedWriteRes<[I6400FPUShort]> {
    let Latency = 2;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSAShortLogic2],
               (instrs SLL_B, SLL_H, SLL_W, SLL_D, SLLI_B, SLLI_H, SLLI_W,
                   SLLI_D, SRA_B, SRA_H, SRA_W, SRA_D, SRAI_B, SRAI_H, SRAI_W,
                   SRAI_D, SRAR_B, SRAR_H, SRAR_W, SRAR_D, SRARI_B, SRARI_H,
                   SRARI_W, SRARI_D, SRL_B, SRL_H, SRL_W, SRL_D, SRLI_B, SRLI_H,
                   SRLI_W, SRLI_D, SRLR_B, SRLR_H, SRLR_W, SRLR_D, SRLRI_B,
                   SRLRI_H, SRLRI_W, SRLRI_D, NLOC_B, NLOC_H, NLOC_W, NLOC_D,
                   NLZC_B, NLZC_H, NLZC_W, NLZC_D, BNEG_B, BNEG_H, BNEG_W,
                   BNEG_D, BNEGI_B, BNEGI_H, BNEGI_W, BNEGI_D, BCLR_B, BCLR_H,
                   BCLR_W, BCLR_D, BCLRI_B, BCLRI_H, BCLRI_W, BCLRI_D, SHF_B,
                   SHF_H, SHF_W)>;

  def I6400MSAShortLogic : SchedWriteRes<[I6400FPUShort]> {
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSAShortLogic],
               (instrs AND_V, ANDI_B, OR_V, ORI_B, XOR_V, XORI_B,
		NOR_V, NORI_B)>;

  def : InstRW<[I6400MSAShortLogic], (instregex "^NOR_V_(H|W|D)_PSEUDO$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^OR_V_(H|W|D)_PSEUDO$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^XOR_V_(H|W|D)_PSEUDO$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^AND_V_(H|W|D)_PSEUDO$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^ILVEV_(B|H|W|D)$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^ILVL_(B|H|W|D)$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^ILVOD_(B|H|W|D)$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^ILVR_(B|H|W|D)$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^PCKEV_(B|H|W|D)$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^PCKOD_(B|H|W|D)$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^FILL_(B|H|W|D)$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^FILL_F(D|W)_PSEUDO$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^INSERT_F(D|W)_PSEUDO$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^SPLAT_(B|H|W|D)$")>;
  def : InstRW<[I6400MSAShortLogic], (instregex "^SPLATI_(B|H|W|D)$")>;

  def I6400MSAShortLogic4 : SchedWriteRes<[I6400FPUShort]>;
  def : InstRW<[I6400MSAShortLogic4],
               (instrs CTCMSA, CFCMSA, COPY_S_B, COPY_S_H, COPY_S_W, COPY_S_D,
                   COPY_U_B, COPY_U_H, COPY_U_W, BNZ_B, BNZ_H, BNZ_W, BNZ_D,
                   BNZ_V, BZ_B, BZ_H, BZ_W, BZ_D, BZ_V)>;

  def I6400MSAMove : SchedWriteRes<[I6400FPUShort]> {
    let Latency = 3;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSAMove], (instrs LD_B, LD_H, LD_W, LD_D)>;

  def I6400MSAMove2 : SchedWriteRes<[I6400FPUShort]> {
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSAMove2], (instrs LDI_B, LDI_H, LDI_W, LDI_D, MOVE_V)>;

  def I6400MSACmp : SchedWriteRes<[I6400FPUShort]> {
    let Latency = 2;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSACmp],
               (instrs FCAF_W, FCAF_D, FCUN_W, FCUN_D, FCOR_W, FCOR_D, FCEQ_W,
                   FCEQ_D, FCUNE_W, FCUNE_D, FCUEQ_W, FCUEQ_D, FCNE_W, FCNE_D,
                   FCLT_W, FCLT_D, FCULT_W, FCULT_D, FCLE_W, FCLE_D, FCULE_W,
                   FCULE_D, FSAF_W, FSAF_D, FSUN_W, FSUN_D, FSOR_W, FSOR_D,
                   FSEQ_W, FSEQ_D, FSUNE_W, FSUNE_D, FSUEQ_W, FSUEQ_D, FSNE_W,
                   FSNE_D, FSLT_W, FSLT_D, FSULT_W, FSULT_D, FSLE_W, FSLE_D,
                   FSULE_W, FSULE_D)>;

  def I6400MSAShortFloat : SchedWriteRes<[I6400FPUShort]> {
    let Latency = 2;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSAShortFloat], (instrs FMAX_W, FMAX_D, FMAX_A_W, FMAX_A_D,
                                         FMIN_W, FMIN_D, FMIN_A_W, FMIN_A_D,
                                         FCLASS_W, FCLASS_D, FABS_D, FABS_W,
					 FABS_D32, FABS_D64)>;
  def I6400MSAFloatCompare : SchedWriteRes<[I6400FPUShort]> {
    let Latency = 2;
    let ReleaseAtCycles = [1];
  }
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_UN_(S|D)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_UEQ_(S|D)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_EQ_(S|D)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_LT_(S|D)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_ULT_(S|D)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_LE_(S|D)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_ULE_(S|D)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_F_(D|S)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_SAF_(D|S)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_SEQ_(D|S)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_SLE_(D|S)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_SLT_(D|S)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_SUEQ_(D|S)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_SULE_(D|S)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_SULT_(D|S)$")>;
  def : InstRW<[I6400MSAFloatCompare], (instregex "^CMP_SUN_(D|S)$")>;

  def I6400FPULongFloat : SchedWriteRes<[I6400FPULong]> { let Latency = 4; }
  def : InstRW<[I6400FPULongFloat],
               (instrs TRUNC_W_S, TRUNC_L_S, TRUNC_L_D64,
		   TRUNC_W_D32, TRUNC_W_D64, PseudoTRUNC_W_D,
		   PseudoTRUNC_W_D32, PseudoTRUNC_W_S,
		   ROUND_W_S, ROUND_L_S,
                   ROUND_L_D64, FLOOR_W_S, FLOOR_L_S, FLOOR_L_D64, CVT_D32_S,
                   CVT_D32_W, CVT_D64_W, CVT_D64_S, CVT_D64_L, CVT_L_S,
                   CVT_L_D64, CVT_S_W, CVT_S_D32, CVT_S_PU64, CVT_S_PL64,
                   CVT_S_L, CVT_S_D64, CVT_W_S, CEIL_W_S, CEIL_L_S,
                   CEIL_L_D64, FLOOR_W_D32, FLOOR_W_D64, CVT_W_D64, CVT_W_D32,
		   RINT_D, RINT_S, ROUND_W_D32, ROUND_W_D64,
		   CEIL_W_D32, CEIL_W_D64)>;

  def I6400MSALongLogic1 : SchedWriteRes<[I6400FPULong]> {
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSALongLogic1], (instrs BMZ_V, BMZI_B, BMNZ_V, BMNZI_B,
                                         INSERT_B, INSERT_H,
                                         INSERT_W, INSERT_D, INSVE_B, INSVE_H,
                                         INSVE_W, INSVE_D)>;
  def : InstRW<[I6400MSALongLogic1], (instregex "^(BSEL_V|BSELI_B)$")>;
  def : InstRW<[I6400MSALongLogic1],
             (instregex "^BSEL_(H|W|D|FW|FD)_PSEUDO$")>;

  def I6400MSALongLogic2 : SchedWriteRes<[I6400FPULong]> {
    let Latency = 2;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSALongLogic2],
               (instrs BINSL_B, BINSL_H, BINSL_W, BINSL_D, BINSLI_B, BINSLI_H,
                   BINSLI_W, BINSLI_D, BINSR_B, BINSR_H, BINSR_W, BINSR_D,
                   BINSRI_B, BINSRI_H, BINSRI_W, BINSRI_D, VSHF_B, VSHF_H,
                   VSHF_W, VSHF_D, SLD_B, SLD_H, SLD_W, SLD_D, SLDI_B, SLDI_H,
                   SLDI_W, SLDI_D, BSET_B, BSET_H, BSET_W, BSET_D, BSETI_B,
                   BSETI_H, BSETI_W, BSETI_D)>;

  def I6400MSAMult : SchedWriteRes<[I6400FPULong]> {
    let Latency = 5;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSAMult],
               (instrs MADDV_B, MADDV_H, MADDV_W, MADDV_D, MSUBV_B, MSUBV_H,
                   MSUBV_W, MSUBV_D, MULV_B, MULV_H, MULV_W, MULV_D, DOTP_S_H,
                   DOTP_S_W, DOTP_S_D, DOTP_U_H, DOTP_U_W, DOTP_U_D, MUL_Q_H,
                   MUL_Q_W, MULR_Q_H, MULR_Q_W, MSUB_Q_H, MSUB_Q_W, MSUBR_Q_H,
                   MSUBR_Q_W, MADD_Q_H, MADD_Q_W, MADDR_Q_H, MADDR_Q_W)>;

  def I6400MSALongFloat2 : SchedWriteRes<[I6400FPULong]> {
    let Latency = 2;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSALongFloat2], (instrs FLOG2_W, FLOG2_D)>;

  def I6400MSALongFloat4 : SchedWriteRes<[I6400FPULong]> {
    let Latency = 4;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSALongFloat4],
               (instrs FADD_W, FADD_D, FSUB_W, FSUB_D, FEXDO_H, FEXDO_W,
                   FEXUPL_W, FEXUPL_D, FEXUPR_W, FEXUPR_D, FFINT_S_W, FFINT_S_D,
                   FFINT_U_W, FFINT_U_D, FFQL_W, FFQL_D, FFQR_W, FFQR_D,
                   FTINT_S_W, FTINT_S_D, FTINT_U_W, FTINT_U_D, FTRUNC_S_W,
                   FTRUNC_S_D, FTRUNC_U_W, FTRUNC_U_D, FTQ_H, FTQ_W, FRINT_W,
                   FRINT_D, FADD_D32, FADD_D64)>;

  def : InstRW<[I6400MSALongFloat4], (instregex "^PseudoCVT_(S|D32|D64)_(L|W)$")>;

  def I6400MSALongFloat5 : SchedWriteRes<[I6400FPULong]> {
    let Latency = 5;
    let ReleaseAtCycles = [2];
  }

  def : InstRW<[I6400MSALongFloat5],
               (instrs FMUL_W, FMUL_D, FEXP2_W, FEXP2_D, DPADD_S_H, DPADD_S_W,
                   DPADD_S_D, DPADD_U_H, DPADD_U_W, DPADD_U_D, DPSUB_S_H,
                   DPSUB_S_W, DPSUB_S_D, DPSUB_U_H, DPSUB_U_W, DPSUB_U_D)>;

  def I6400MSALongFloat8 : SchedWriteRes<[I6400FPULong]> {
    let Latency = 8;
    let ReleaseAtCycles = [2];
  }
  def : InstRW<[I6400MSALongFloat8], (instrs FMADD_W, FMADD_D, FMSUB_W,
                                         FMSUB_D)>;

  def I6400MSALongFloatFuse : SchedWriteRes<[I6400FPULong]> {
    let Latency = 8;
    let ReleaseAtCycles = [1];
  }
  def : InstRW<[I6400MSALongFloatFuse], (instrs MSUBF_D, MSUBF_S,
					 MADDF_D, MADDF_S)>;

  def I6400MSADivD : SchedWriteRes<[I6400FPULong, I6400FPUFloatL]> {
    let Latency = 30;
    let ReleaseAtCycles = [1, 29];
  }
  def : InstRW<[I6400MSADivD], (instrs FDIV_D)>;

  def I6400MSADivW : SchedWriteRes<[I6400FPULong, I6400FPUFloatL]> {
    let Latency = 22;
    let ReleaseAtCycles = [1, 21];
  }
  def : InstRW<[I6400MSADivW], (instrs FDIV_W)>;

  // Atomic instructions

  // FIXME: Define `WriteAtomic` in the MipsSchedule.td and
  // attach it to the Atomic2OpsPostRA, AtomicCmpSwapPostRA, ...
  // classes. Then just define resources for the `WriteAtomic` in each
  // machine models.
  def I6400WriteAtomic : SchedWriteRes<[I6400Atomic]> { let Latency = 2; }

  def : InstRW<[I6400WriteAtomic],
      (instregex "^ATOMIC_SWAP_I(8|16|32|64)_POSTRA$")>;
  def : InstRW<[I6400WriteAtomic],
      (instregex "^ATOMIC_CMP_SWAP_I(8|16|32|64)_POSTRA$")>;
  def : InstRW<[I6400WriteAtomic],
      (instregex "^ATOMIC_LOAD_(ADD|SUB|AND|OR|XOR|NAND|MIN|MAX|UMIN|UMAX)"
               "_I(8|16|32|64)_POSTRA$")>;
}
