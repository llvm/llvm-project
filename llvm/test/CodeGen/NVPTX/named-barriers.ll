; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Use bar.sync to arrive at a pre-computed barrier number and
; wait for all threads in CTA to also arrive:
define ptx_device void @test_barrier_named_cta() {
; CHECK: mov.b32  %r[[REG0:[0-9]+]], 0;
; CHECK: bar.sync %r[[REG0]];
; CHECK: mov.b32  %r[[REG1:[0-9]+]], 10;
; CHECK: bar.sync %r[[REG1]];
; CHECK: mov.b32  %r[[REG2:[0-9]+]], 15;
; CHECK: bar.sync %r[[REG2]];
; CHECK: ret;
  call void @llvm.nvvm.barrier.n(i32 0)
  call void @llvm.nvvm.barrier.n(i32 10)
  call void @llvm.nvvm.barrier.n(i32 15)
  ret void
}

; Use bar.sync to arrive at a pre-computed barrier number and
; wait for fixed number of cooperating threads to arrive:
define ptx_device void @test_barrier_named() {
; CHECK: mov.b32  %r[[REG0A:[0-9]+]], 32;
; CHECK: mov.b32  %r[[REG0B:[0-9]+]], 0;
; CHECK: bar.sync %r[[REG0B]], %r[[REG0A]];
; CHECK: mov.b32  %r[[REG1A:[0-9]+]], 352;
; CHECK: mov.b32  %r[[REG1B:[0-9]+]], 10;
; CHECK: bar.sync %r[[REG1B]], %r[[REG1A]];
; CHECK: mov.b32  %r[[REG2A:[0-9]+]], 992;
; CHECK: mov.b32  %r[[REG2B:[0-9]+]], 15;
; CHECK: bar.sync %r[[REG2B]], %r[[REG2A]];
; CHECK: ret;
  call void @llvm.nvvm.barrier(i32 0, i32 32)
  call void @llvm.nvvm.barrier(i32 10, i32 352)
  call void @llvm.nvvm.barrier(i32 15, i32 992)
  ret void
}

declare void @llvm.nvvm.barrier(i32, i32)
declare void @llvm.nvvm.barrier.n(i32)
