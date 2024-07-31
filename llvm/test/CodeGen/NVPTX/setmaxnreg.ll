; RUN: llc < %s -march=nvptx64 -mcpu=sm_90a -mattr=+ptx80| FileCheck --check-prefixes=CHECK %s
; RUN: %if ptxas-12.0 %{ llc < %s -march=nvptx64 -mcpu=sm_90a -mattr=+ptx80| %ptxas-verify -arch=sm_90a %}

declare void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 %reg_count)
declare void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 %reg_count)

; CHECK-LABEL: test_set_maxn_reg
define void @test_set_maxn_reg() {
  ; CHECK: setmaxnreg.inc.sync.aligned.u32 96;
  call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 96)

  ; CHECK: setmaxnreg.dec.sync.aligned.u32 64;
  call void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 64)

  ret void
}
