// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa %s -filetype=null 2>&1 | FileCheck %s -implicit-check-not=warning: -check-prefix=GFX1250

// GFX1250: :[[@LINE+1]]:1: warning: kernel 'test_wrong_before' does not begin with the required prologue sequence: S_MOV_B64 followed by V_NOP and GLOBAL_PREFETCH_B8
test_wrong_before:
  s_nop 1

.amdhsa_kernel test_wrong_before
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
.end_amdhsa_kernel

.amdhsa_kernel test_wrong_after
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
.end_amdhsa_kernel

// GFX1250: :[[@LINE+1]]:1: warning: kernel 'test_wrong_after' does not begin with the required prologue sequence: S_MOV_B64 followed by V_NOP and GLOBAL_PREFETCH_B8
test_wrong_after:
  s_nop 2

.amdhsa_kernel test_old_sequence
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 1
.end_amdhsa_kernel

// GFX1250: :[[@LINE+1]]:1: warning: kernel 'test_old_sequence' does not begin with the required prologue sequence: S_MOV_B64 followed by V_NOP and GLOBAL_PREFETCH_B8
test_old_sequence:
  global_prefetch_b8 v0, null scope:SCOPE_SE
  v_nop

.amdhsa_kernel test_correct
  .amdhsa_next_free_sgpr 66
  .amdhsa_next_free_vgpr 1
.end_amdhsa_kernel

test_correct:
  s_mov_b64 s[64:65], 0
  v_nop
  global_prefetch_b8 v0, s[64:65] scope:SCOPE_SE th:TH_LOAD_RT
