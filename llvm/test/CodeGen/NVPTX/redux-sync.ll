; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck %s
; RUN: %if ptxas-sm_80 && ptxas-isa-7.0 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | %ptxas-verify -arch=sm_80 %}

declare i32 @llvm.nvvm.redux.sync.umin(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_min_u32
define i32 @redux_sync_min_u32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.min.u32
  %val = call i32 @llvm.nvvm.redux.sync.umin(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.umax(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_max_u32
define i32 @redux_sync_max_u32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.max.u32
  %val = call i32 @llvm.nvvm.redux.sync.umax(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.add(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_add_s32
define i32 @redux_sync_add_s32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.add.s32
  %val = call i32 @llvm.nvvm.redux.sync.add(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.min(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_min_s32
define i32 @redux_sync_min_s32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.min.s32
  %val = call i32 @llvm.nvvm.redux.sync.min(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.max(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_max_s32
define i32 @redux_sync_max_s32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.max.s32
  %val = call i32 @llvm.nvvm.redux.sync.max(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.and(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_and_b32
define i32 @redux_sync_and_b32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.and.b32
  %val = call i32 @llvm.nvvm.redux.sync.and(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.xor(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_xor_b32
define i32 @redux_sync_xor_b32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.xor.b32
  %val = call i32 @llvm.nvvm.redux.sync.xor(i32 %src, i32 %mask)
  ret i32 %val
}

declare i32 @llvm.nvvm.redux.sync.or(i32, i32)
; CHECK-LABEL: .func{{.*}}redux_sync_or_b32
define i32 @redux_sync_or_b32(i32 %src, i32 %mask) {
  ; CHECK: redux.sync.or.b32
  %val = call i32 @llvm.nvvm.redux.sync.or(i32 %src, i32 %mask)
  ret i32 %val
}

; A full-warp butterfly add reduction is folded into redux.sync.add.
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)
; CHECK-LABEL: .func{{.*}}butterfly_reduce_add
define i32 @butterfly_reduce_add(i32 %x) {
  ; CHECK-NOT: shfl.sync
  ; CHECK: redux.sync.add.s32
  ; CHECK-NOT: shfl.sync
  %s1 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %x, i32 1, i32 31)
  %a1 = add i32 %x, %s1
  %s2 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a1, i32 2, i32 31)
  %a2 = add i32 %a1, %s2
  %s4 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a2, i32 4, i32 31)
  %a4 = add i32 %a2, %s4
  %s8 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a4, i32 8, i32 31)
  %a8 = add i32 %a4, %s8
  %s16 = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %a8, i32 16, i32 31)
  %sum = add i32 %a8, %s16
  ret i32 %sum
}
