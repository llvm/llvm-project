; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

declare void @llvm.nvvm.pm.event.mask(i16 %mask)

; CHECK-LABEL: test_pm_event
define void @test_pm_event() {
  ; CHECK: pmevent.mask 0xffU;
  call void @llvm.nvvm.pm.event.mask(i16 u0xff)

  ; CHECK: pmevent.mask 0x1000U;
  call void @llvm.nvvm.pm.event.mask(i16 u0x1000)

  ; CHECK: pmevent.mask 0x8000U;
  call void @llvm.nvvm.pm.event.mask(i16 u0x8000)

  ; CHECK: pmevent.mask 0xffffU;
  call void @llvm.nvvm.pm.event.mask(i16 u0xFFFF)

  ;; LLVM IR doesn't distinguish signed and unsigned integers. So, NVVM calls
  ;; with negative integers are functionally correct here and processed
  ;; correctly in NVPTX backend

  ; CHECK: pmevent.mask 0x8000U;
  call void @llvm.nvvm.pm.event.mask(i16 -32768)

  ; CHECK: pmevent.mask 0xffffU;
  call void @llvm.nvvm.pm.event.mask(i16 -1)

  ret void
}
