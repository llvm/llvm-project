; Verify that the AMDGPU summary entries round-trip through bitcode
; as FS_AMDGPU_ENTRY records inside the GLOBALVAL_SUMMARY_BLOCK.
; Only kernels and functions with explicit occupancy attributes get entries.
; RUN: opt -mtriple=amdgcn-amd-amdhsa -module-summary < %s -o %t.bc
; RUN: llvm-bcanalyzer -dump %t.bc | FileCheck %s --check-prefix=PERMODULE
; RUN: llvm-lto2 run %t.bc -o %t.o -thinlto-distributed-indexes \
; RUN:   -r=%t.bc,kernel,px -r=%t.bc,device_func,px -r=%t.bc,plain,px
; RUN: llvm-bcanalyzer -dump %t.bc.thinlto.bc \
; RUN:   | FileCheck %s --check-prefix=COMBINED

; PERMODULE:     <GLOBALVAL_SUMMARY_BLOCK
; PERMODULE-NOT:   <AMDGPU_ENTRY
; PERMODULE:       <AMDGPU_ENTRY op0=91 op1=15 op2=64 op3=256 op4=2 op5=8 op6=16 op7=16 op8=1/>
; PERMODULE-NOT:   <AMDGPU_ENTRY
; PERMODULE:       <AMDGPU_ENTRY op0=0 op1=2 op2=0 op3=0 op4=4 op5=0 op6=0 op7=0 op8=0/>
; PERMODULE-NOT:   <AMDGPU_ENTRY
; PERMODULE:     </GLOBALVAL_SUMMARY_BLOCK>

; COMBINED:     <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NOT:   <AMDGPU_ENTRY
; COMBINED:       <AMDGPU_ENTRY op0=0 op1=2 op2=0 op3=0 op4=4 op5=0 op6=0 op7=0 op8=0/>
; COMBINED-NOT:   <AMDGPU_ENTRY
; COMBINED:       <AMDGPU_ENTRY op0=91 op1=15 op2=64 op3=256 op4=2 op5=8 op6=16 op7=16 op8=1/>
; COMBINED-NOT:   <AMDGPU_ENTRY
; COMBINED:     </GLOBALVAL_SUMMARY_BLOCK>

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @kernel(ptr %p) #0 {
  call void @device_func(ptr %p)
  ret void
}

define void @device_func(ptr %p) #1 {
  store i32 42, ptr %p
  ret void
}

define void @plain() {
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" "amdgpu-waves-per-eu"="2,8" "amdgpu-max-num-workgroups"="16,16,1" }
attributes #1 = { "amdgpu-waves-per-eu"="4" }
