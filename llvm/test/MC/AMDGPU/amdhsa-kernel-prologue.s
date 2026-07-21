// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 %s -filetype=null 2>&1 | FileCheck %s -implicit-check-not=warning: -check-prefix=GFX1250

// GFX1250: :[[@LINE+1]]:1: warning: kernel 'test_wrong_before' does not begin with the required prologue sequence: GLOBAL_PREFETCH_B8 v0, s[0:1] scope:SCOPE_SE followed by V_NOP
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

// GFX1250: :[[@LINE+1]]:1: warning: kernel 'test_wrong_after' does not begin with the required prologue sequence: GLOBAL_PREFETCH_B8 v0, s[0:1] scope:SCOPE_SE followed by V_NOP
test_wrong_after:
  s_nop 2

// Test wrong registers - should warn
.amdhsa_kernel test_wrong_regs
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
.end_amdhsa_kernel

// GFX1250: :[[@LINE+1]]:1: warning: kernel 'test_wrong_regs' does not begin with the required prologue sequence: GLOBAL_PREFETCH_B8 v0, s[0:1] scope:SCOPE_SE followed by V_NOP
test_wrong_regs:
  global_prefetch_b8 v1, s[2:3] scope:SCOPE_SE
  v_nop

// Test wrong scope - should warn
.amdhsa_kernel test_wrong_scope
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
.end_amdhsa_kernel

// GFX1250: :[[@LINE+1]]:1: warning: kernel 'test_wrong_scope' does not begin with the required prologue sequence: GLOBAL_PREFETCH_B8 v0, s[0:1] scope:SCOPE_SE followed by V_NOP
test_wrong_scope:
  global_prefetch_b8 v0, s[0:1] scope:SCOPE_DEV
  v_nop

// Test correct sequence - no warning
.amdhsa_kernel test_correct
  .amdhsa_next_free_sgpr 0
  .amdhsa_next_free_vgpr 0
.end_amdhsa_kernel

test_correct:
  global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
  v_nop
