; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

declare void @llvm.nvvm.pm.event.mask(i16 %mask)

; CHECK-LABEL: test_pm_event
define void @test_pm_event() {
  ; CHECK: pmevent.mask 255;
  call void @llvm.nvvm.pm.event.mask(i16 u0xff)

  ; CHECK: pmevent.mask 4096;
  call void @llvm.nvvm.pm.event.mask(i16 u0x1000)

  ret void
}
