; RUN: llc < %s -march=nvptx64 -mcpu=sm_100a -mattr=+ptx86 | FileCheck --check-prefixes=CHECK %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_100f -mattr=+ptx88 | FileCheck --check-prefixes=CHECK %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_110f -mattr=+ptx90 | FileCheck --check-prefixes=CHECK %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_120f -mattr=+ptx88 | FileCheck --check-prefixes=CHECK %s
; RUN: %if ptxas-sm_100a && ptxas-isa-8.6 %{ llc < %s -march=nvptx64 -mcpu=sm_100a -mattr=+ptx86 | %ptxas-verify -arch=sm_100a %}
; RUN: %if ptxas-sm_100f && ptxas-isa-8.8 %{ llc < %s -march=nvptx64 -mcpu=sm_100f -mattr=+ptx88 | %ptxas-verify -arch=sm_100f %}
; RUN: %if ptxas-sm_110f && ptxas-isa-9.0 %{ llc < %s -march=nvptx64 -mcpu=sm_110f -mattr=+ptx90 | %ptxas-verify -arch=sm_110f %}
; RUN: %if ptxas-sm_120f && ptxas-isa-8.8 %{ llc < %s -march=nvptx64 -mcpu=sm_120f -mattr=+ptx88 | %ptxas-verify -arch=sm_120f %}

; CHECK-LABEL: test_set_maxn_reg_sm100a
define void @test_set_maxn_reg_sm100a() {
  ; CHECK: setmaxnreg.inc.sync.aligned.u32 96;
  call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 96)

  ; CHECK: setmaxnreg.dec.sync.aligned.u32 64;
  call void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 64)

  ret void
}
