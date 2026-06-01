; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -stop-after=finalize-isel -o %t.mir %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -run-pass=none %t.mir -o - | FileCheck %s

; minNumAGPRs is derived from the amdgpu-agpr-alloc attribute (only relevant on
; gfx90a+) and must survive the MIR round-trip.

; CHECK-LABEL: {{^}}name: min_num_agprs
; CHECK: machineFunctionInfo:
; CHECK: minNumAGPRs:     3
define void @min_num_agprs() #0 {
  ret void
}

attributes #0 = { "amdgpu-agpr-alloc"="3,5" }
