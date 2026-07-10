; RUN: opt < %s -passes=amdgpu-sw-lower-lds -S -amdgpu-asan-instrument-lds=false -mtriple=amdgcn-amd-amdhsa | FileCheck %s

; Test that the sw-lower-lds pass moves constant-size allocas from the original
; entry block to the new entry block (WId), so they remain static allocas.

@lds = internal addrspace(3) global [64 x i32] poison, align 4

; Allocas clustered at the top of the entry block (common case).
define amdgpu_kernel void @kernel_allocas_at_top(i32 %n) sanitize_address {
; CHECK-LABEL: define amdgpu_kernel void @kernel_allocas_at_top(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  [[WID:.*]]:
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4, addrspace(5)
; CHECK-NEXT:    [[B:%.*]] = alloca i32, align 4, addrspace(5)
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.amdgcn.workitem.id.z()
; CHECK-NEXT:    [[TMP3:%.*]] = or i32 [[TMP0]], [[TMP1]]
; CHECK-NEXT:    [[TMP4:%.*]] = or i32 [[TMP3]], [[TMP2]]
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i32 [[TMP4]], 0
; CHECK-NEXT:    br i1 [[TMP5]], label %[[MALLOC:.*]], label %[[BB18:.*]]
; CHECK:       [[BB18]]:
; CHECK:         store i32 [[N]], ptr addrspace(5) [[A]], align 4
; CHECK-NEXT:    store i32 [[N]], ptr addrspace(5) [[B]], align 4
;
  %a = alloca i32, align 4, addrspace(5)
  %b = alloca i32, align 4, addrspace(5)
  store i32 %n, ptr addrspace(5) %a, align 4
  store i32 %n, ptr addrspace(5) %b, align 4
  store i32 %n, ptr addrspace(3) @lds, align 4
  ret void
}

; Allocas interleaved with non-alloca instructions.
define amdgpu_kernel void @kernel_allocas_scattered(i32 %n) sanitize_address {
; CHECK-LABEL: define amdgpu_kernel void @kernel_allocas_scattered(
; CHECK-SAME: i32 [[N:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  [[WID:.*]]:
; CHECK-NEXT:    [[A:%.*]] = alloca i32, align 4, addrspace(5)
; CHECK-NEXT:    [[B:%.*]] = alloca i32, align 4, addrspace(5)
; CHECK-NEXT:    [[TMP0:%.*]] = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK-NEXT:    [[TMP1:%.*]] = call i32 @llvm.amdgcn.workitem.id.y()
; CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.amdgcn.workitem.id.z()
; CHECK-NEXT:    [[TMP3:%.*]] = or i32 [[TMP0]], [[TMP1]]
; CHECK-NEXT:    [[TMP4:%.*]] = or i32 [[TMP3]], [[TMP2]]
; CHECK-NEXT:    [[TMP5:%.*]] = icmp eq i32 [[TMP4]], 0
; CHECK-NEXT:    br i1 [[TMP5]], label %[[MALLOC:.*]], label %[[BB18:.*]]
; CHECK:       [[BB18]]:
; CHECK:         store i32 [[N]], ptr addrspace(5) [[A]], align 4
; CHECK-NEXT:    store i32 [[N]], ptr addrspace(5) [[B]], align 4
;
  %a = alloca i32, align 4, addrspace(5)
  store i32 %n, ptr addrspace(5) %a, align 4
  %b = alloca i32, align 4, addrspace(5)
  store i32 %n, ptr addrspace(5) %b, align 4
  store i32 %n, ptr addrspace(3) @lds, align 4
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"nosanitize_address", i32 1}
