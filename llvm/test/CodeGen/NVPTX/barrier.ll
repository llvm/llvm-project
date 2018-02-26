; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 -mattr=+ptx60 | FileCheck %s

declare void @llvm.nvvm.bar.warp.sync(i32)
declare void @llvm.nvvm.barrier.sync(i32)
declare void @llvm.nvvm.barrier.sync.cnt(i32, i32)

; CHECK-LABEL: .func{{.*}}barrier.sync
define void @barrier.sync(i32 %id, i32 %cnt) {
  ; CHECK: ld.param.u32 	[[ID:%r[0-9]+]], [barrier.sync_param_0];
  ; CHECK: ld.param.u32 	[[CNT:%r[0-9]+]], [barrier.sync_param_1];

  ; CHECK:  barrier.sync [[ID]], [[CNT]];
  call void @llvm.nvvm.barrier.sync.cnt(i32 %id, i32 %cnt)
  ; CHECK:  barrier.sync [[ID]], 2;
  call void @llvm.nvvm.barrier.sync.cnt(i32 %id, i32 2)
  ; CHECK:  barrier.sync 3, [[CNT]];
  call void @llvm.nvvm.barrier.sync.cnt(i32 3, i32 %cnt)
  ; CHECK:  barrier.sync 4, 5;
  call void @llvm.nvvm.barrier.sync.cnt(i32 4, i32 5)

  ; CHECK: barrier.sync [[ID]];
  call void @llvm.nvvm.barrier.sync(i32 %id)
  ; CHECK: barrier.sync 1;
  call void @llvm.nvvm.barrier.sync(i32 1)

  ; CHECK: bar.warp.sync [[ID]];
  call void @llvm.nvvm.bar.warp.sync(i32 %id)
  ; CHECK: bar.warp.sync 6;
  call void @llvm.nvvm.bar.warp.sync(i32 6)
  ret void;
}

