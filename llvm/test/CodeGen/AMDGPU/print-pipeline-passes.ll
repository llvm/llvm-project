; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O0>" -print-pipeline-passes %s -o - | FileCheck --check-prefix=O0 %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O1>" -print-pipeline-passes %s -o - | FileCheck %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O2>" -print-pipeline-passes %s -o - | FileCheck %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O3>" -print-pipeline-passes %s -o - | FileCheck %s

; CHECK: amdgpu-attributor
; O0-NOT: amdgpu-attributor

define amdgpu_kernel void @kernel() {
entry:
  ret void
}
