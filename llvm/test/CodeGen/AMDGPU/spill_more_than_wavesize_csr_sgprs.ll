; RUN: llc -mtriple amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -enable-var-scope %s

; CHECK-LABEL: {{^}}spill_more_than_wavesize_csr_sgprs:
; CHECK:        v_writelane_b32 v0, s98, 63
; CHECK-NEXT:   v_writelane_b32 v1, s99, 0
; CHECK:        v_readlane_b32 s99, v1, 0
; CHECK-NEXT:   v_readlane_b32 s98, v0, 63

define void @spill_more_than_wavesize_csr_sgprs() {
  call void asm sideeffect "",
   "~{s35},~{s36},~{s37},~{s38},~{s39},~{s40},~{s41},~{s42}
   ,~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49},~{s50}
   ,~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58}
   ,~{s59},~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66}
   ,~{s67},~{s68},~{s69},~{s70},~{s71},~{s72},~{s73},~{s74}
   ,~{s75},~{s76},~{s77},~{s78},~{s79},~{s80},~{s81},~{s82}
   ,~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89},~{s90}
   ,~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98}
   ,~{s99},~{s100},~{s101},~{s102}"()
  ret void
}

; CHECK-LABEL: {{^}}spill_more_than_wavesize_csr_sgprs_with_stack_object:
; CHECK:        v_writelane_b32 v1, s98, 63
; CHECK-NEXT:   v_writelane_b32 v2, s99, 0
; CHECK:        v_readlane_b32 s99, v2, 0
; CHECK-NEXT:   v_readlane_b32 s98, v1, 63

define void @spill_more_than_wavesize_csr_sgprs_with_stack_object() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  call void asm sideeffect "",
   "~{s35},~{s36},~{s37},~{s38},~{s39},~{s40},~{s41},~{s42}
   ,~{s43},~{s44},~{s45},~{s46},~{s47},~{s48},~{s49},~{s50}
   ,~{s51},~{s52},~{s53},~{s54},~{s55},~{s56},~{s57},~{s58}
   ,~{s59},~{s60},~{s61},~{s62},~{s63},~{s64},~{s65},~{s66}
   ,~{s67},~{s68},~{s69},~{s70},~{s71},~{s72},~{s73},~{s74}
   ,~{s75},~{s76},~{s77},~{s78},~{s79},~{s80},~{s81},~{s82}
   ,~{s83},~{s84},~{s85},~{s86},~{s87},~{s88},~{s89},~{s90}
   ,~{s91},~{s92},~{s93},~{s94},~{s95},~{s96},~{s97},~{s98}
   ,~{s99},~{s100},~{s101},~{s102}"()
  ret void
}
