; RUN: llc < %s -march=nvptx64 -mcpu=sm_100a -mattr=+ptx86 | FileCheck --check-prefixes=CHECK %s
; RUN: %if ptxas-sm_100a && ptxas-isa-8.6 %{ llc < %s -march=nvptx64 -mcpu=sm_100a -mattr=+ptx86 | %ptxas-verify -arch=sm_100a %}

; CHECK-LABEL: test_set_maxn_reg_sm100a
define void @test_set_maxn_reg_sm100a() {
  ; CHECK: setmaxnreg.inc.sync.aligned.u32 96;
  call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 96)

  ; CHECK: setmaxnreg.dec.sync.aligned.u32 64;
  call void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 64)

  ret void
}
