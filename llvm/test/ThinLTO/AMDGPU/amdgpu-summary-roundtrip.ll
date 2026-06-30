; Verify that the AMDGPU_SUMMARY block round-trips through bitcode.
; RUN: opt -mtriple=amdgcn-amd-amdhsa -module-summary %s -o %t.bc
; RUN: llvm-bcanalyzer -dump %t.bc | FileCheck %s --check-prefix=BLOCK

; All attributes present.
; BLOCK: <AMDGPU_SUMMARY_BLOCK
; BLOCK-NEXT: <AMDGPU_SUMMARY_VERSION op0=1/>
; BLOCK-NEXT: <AMDGPU_SUMMARY_ENTRY {{.*}} op1=1 op2=64 op3=256 op4=2 op5=8 op6=16 op7=16 op8=1/>

; Only flat-work-group-size — waves and max-workgroups use defaults.
; BLOCK-NEXT: <AMDGPU_SUMMARY_ENTRY {{.*}} op1=1 op2=128 op3=512 op4=1 op5=10 op6=4294967295 op7=4294967295 op8=4294967295/>

; Only waves-per-eu — flat-work-group-size and max-workgroups use defaults.
; BLOCK-NEXT: <AMDGPU_SUMMARY_ENTRY {{.*}} op1=1 op2=1 op3=1024 op4=4 op5=6 op6=4294967295 op7=4294967295 op8=4294967295/>

; Bare kernel with no attributes — all defaults.
; BLOCK-NEXT: <AMDGPU_SUMMARY_ENTRY {{.*}} op1=1 op2=1 op3=1024 op4=1 op5=10 op6=4294967295 op7=4294967295 op8=4294967295/>
; BLOCK-NEXT: </AMDGPU_SUMMARY_BLOCK>

define amdgpu_kernel void @kernel_all(ptr %p) #0 {
  call void @device_func(ptr %p)
  ret void
}

define amdgpu_kernel void @kernel_wg_only(ptr %p) #1 {
  call void @device_func(ptr %p)
  ret void
}

define amdgpu_kernel void @kernel_waves_only(ptr %p) #2 {
  call void @device_func(ptr %p)
  ret void
}

define amdgpu_kernel void @kernel_bare(ptr %p) {
  call void @device_func(ptr %p)
  ret void
}

define void @device_func(ptr %p) {
  store i32 42, ptr %p
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" "amdgpu-waves-per-eu"="2,8" "amdgpu-max-num-workgroups"="16,16,1" }
attributes #1 = { "amdgpu-flat-work-group-size"="128,512" }
attributes #2 = { "amdgpu-waves-per-eu"="4,6" }
