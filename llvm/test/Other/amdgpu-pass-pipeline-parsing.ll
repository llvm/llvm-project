; REQUIRES: amdgpu-registered-target

; RUN: not opt -S -mtriple=amdgcn-amd-amdhsa -passes='amdgpu-attributor<random>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: amdgpu-attributor: invalid AMDGPUAttributor pass parameter 'random'

define void @f() {
entry:
 br label %loop
loop:
 br label %loop
}
