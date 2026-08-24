; RUN: %if x86-registered-target %{ opt -mtriple=x86_64 -passes=pre-isel-intrinsic-lowering -S < %s | FileCheck %s --check-prefix=UNSUPPORTED %}
; RUN: %if nvptx-registered-target %{ opt -mtriple=nvptx64-nvidia-cuda -passes=pre-isel-intrinsic-lowering -S < %s | FileCheck %s --check-prefix=UNSUPPORTED %}
; RUN: %if amdgpu-registered-target %{ opt -mtriple=r600 -passes=pre-isel-intrinsic-lowering -S < %s | FileCheck %s --check-prefix=UNSUPPORTED %}
; RUN: %if amdgpu-registered-target %{ opt -mtriple=amdgcn-amd-amdhsa -passes=pre-isel-intrinsic-lowering -S < %s | FileCheck %s --check-prefix=GCN %}

define i1 @query() {
; UNSUPPORTED-LABEL: define i1 @query() {
; UNSUPPORTED-NEXT:    ret i1 false
;
; GCN-LABEL: define i1 @query() {
; GCN-NEXT:    [[ENABLED:%.*]] = call i1 @llvm.is.debugging.enabled()
; GCN-NEXT:    ret i1 [[ENABLED]]
  %enabled = call i1 @llvm.is.debugging.enabled()
  ret i1 %enabled
}

declare i1 @llvm.is.debugging.enabled()
