// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 %s -filetype=null 2>&1 | FileCheck %s -implicit-check-not=warning: -check-prefix=GFX1250

// GFX1250: :[[@LINE+1]]:1: warning: kernel 'test_wrong_before' does not begin with the required prologue sequence: GLOBAL_WB followed by V_NOP
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

// GFX1250: :[[@LINE+1]]:1: warning: kernel 'test_wrong_after' does not begin with the required prologue sequence: GLOBAL_WB followed by V_NOP
test_wrong_after:
  s_nop 2

.amdhsa_kernel test_correct
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
.end_amdhsa_kernel

test_correct:
  global_wb
  v_nop
