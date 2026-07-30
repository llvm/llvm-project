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
