//=-HexagonScheduleV62.td - HexagonV62 Scheduling Definitions *- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// V62 follows the same schedule as V60 with following exceptions:
// Following instructions are permissible on any slot on V62:
// V4_J4_cmpeq_fp0_jump_nt
// V4_J4_cmpeq_fp0_jump_t
// V4_J4_cmpeq_fp1_jump_nt
// V4_J4_cmpeq_fp1_jump_t
// V4_J4_cmpeq_tp0_jump_nt
// V4_J4_cmpeq_tp0_jump_t
// V4_J4_cmpeq_tp1_jump_nt
// V4_J4_cmpeq_tp1_jump_t
// V4_J4_cmpeqi_fp0_jump_nt
// V4_J4_cmpeqi_fp0_jump_t
// V4_J4_cmpeqi_fp1_jump_nt
// V4_J4_cmpeqi_fp1_jump_t
// V4_J4_cmpeqi_tp0_jump_nt
// V4_J4_cmpeqi_tp0_jump_t
// V4_J4_cmpeqi_tp1_jump_nt
// V4_J4_cmpeqi_tp1_jump_t
// V4_J4_cmpeqn1_fp0_jump_nt
// V4_J4_cmpeqn1_fp0_jump_t
// V4_J4_cmpeqn1_fp1_jump_nt
// V4_J4_cmpeqn1_fp1_jump_t
// V4_J4_cmpeqn1_tp0_jump_nt
// V4_J4_cmpeqn1_tp0_jump_t
// V4_J4_cmpeqn1_tp1_jump_nt
// V4_J4_cmpeqn1_tp1_jump_t
// V4_J4_cmpgt_fp0_jump_nt
// V4_J4_cmpgt_fp0_jump_t
// V4_J4_cmpgt_fp1_jump_nt
// V4_J4_cmpgt_fp1_jump_t
// V4_J4_cmpgt_tp0_jump_nt
// V4_J4_cmpgt_tp0_jump_t
// V4_J4_cmpgt_tp1_jump_nt
// V4_J4_cmpgt_tp1_jump_t
// V4_J4_cmpgti_fp0_jump_nt
// V4_J4_cmpgti_fp0_jump_t
// V4_J4_cmpgti_fp1_jump_nt
// V4_J4_cmpgti_fp1_jump_t
// V4_J4_cmpgti_tp0_jump_nt
// V4_J4_cmpgti_tp0_jump_t
// V4_J4_cmpgti_tp1_jump_nt
// V4_J4_cmpgti_tp1_jump_t
// V4_J4_cmpgtn1_fp0_jump_nt
// V4_J4_cmpgtn1_fp0_jump_t
// V4_J4_cmpgtn1_fp1_jump_nt
// V4_J4_cmpgtn1_fp1_jump_t
// V4_J4_cmpgtn1_tp0_jump_nt
// V4_J4_cmpgtn1_tp0_jump_t
// V4_J4_cmpgtn1_tp1_jump_nt
// V4_J4_cmpgtn1_tp1_jump_t
// V4_J4_cmpgtu_fp0_jump_nt
// V4_J4_cmpgtu_fp0_jump_t
// V4_J4_cmpgtu_fp1_jump_nt
// V4_J4_cmpgtu_fp1_jump_t
// V4_J4_cmpgtu_tp0_jump_nt
// V4_J4_cmpgtu_tp0_jump_t
// V4_J4_cmpgtu_tp1_jump_nt
// V4_J4_cmpgtu_tp1_jump_t
// V4_J4_cmpgtui_fp0_jump_nt
// V4_J4_cmpgtui_fp0_jump_t
// V4_J4_cmpgtui_fp1_jump_nt
// V4_J4_cmpgtui_fp1_jump_t
// V4_J4_cmpgtui_tp0_jump_nt
// V4_J4_cmpgtui_tp0_jump_t
// V4_J4_cmpgtui_tp1_jump_nt
// V4_J4_cmpgtui_tp1_jump_t
// V4_J4_tstbit0_fp0_jump_nt
// V4_J4_tstbit0_fp0_jump_t
// V4_J4_tstbit0_fp1_jump_nt
// V4_J4_tstbit0_fp1_jump_t
// V4_J4_tstbit0_tp0_jump_nt
// V4_J4_tstbit0_tp0_jump_t
// V4_J4_tstbit0_tp1_jump_nt
// V4_J4_tstbit0_tp1_jump_t
// JMP
// JMPEXT
// JMPEXT_f
// JMPEXT_fnew_nt
// JMPEXT_fnew_t
// JMPEXT_t
// JMPEXT_tnew_nt
// JMPEXT_tnew_t
// JMPNOTEXT
// JMPNOTEXT_f
// JMPNOTEXT_fnew_nt
// JMPNOTEXT_fnew_t
// JMPNOTEXT_t
// JMPNOTEXT_tnew_nt
// JMPNOTEXT_tnew_t
// JMP_f
// JMP_fnew_nt
// JMP_fnew_t
// JMP_t
// JMP_tnew_nt
// JMP_tnew_t
// RESTORE_DEALLOC_RET_JMP_V4
// RESTORE_DEALLOC_RET_JMP_V4_EXT

def HexagonV62ItinList : ScalarItin, HVXV62Itin {
  list<InstrItinData> ItinList =
    !listconcat(ScalarItin_list, HVXV62Itin_list);
}

def HexagonItinerariesV62 :
      ProcessorItineraries<[SLOT0, SLOT1, SLOT2, SLOT3, SLOT_ENDLOOP,
                            CVI_ST, CVI_XLANE, CVI_SHIFT, CVI_MPY0, CVI_MPY1,
                            CVI_LD, CVI_XLSHF, CVI_MPY01, CVI_ALL],
                           [], HexagonV62ItinList.ItinList>;

def HexagonModelV62 : SchedMachineModel {
  // Max issue per cycle == bundle width.
  let IssueWidth = 4;
  let Itineraries = HexagonItinerariesV62;
  let LoadLatency = 1;
  let CompleteModel = 0;
}

//===----------------------------------------------------------------------===//
// Hexagon V62 Resource Definitions -
//===----------------------------------------------------------------------===//
