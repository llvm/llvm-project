; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O1>" -print-pipeline-passes %s -o - | FileCheck %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O2>" -print-pipeline-passes %s -o - | FileCheck %s
; RUN: opt -mtriple=amdgcn--amdhsa -S -passes="lto<O3>" -print-pipeline-passes %s -o - | FileCheck %s

; CHECK: amdgpu-attributor

define amdgpu_kernel void @kernel() {
entry:
  ret void
}
