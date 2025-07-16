; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 %reg_count)
declare void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 %reg_count)

define void @test_set_maxn_reg() {
  ; CHECK: reg_count argument to nvvm.setmaxnreg must be in multiples of 8
  call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 95)

  ; CHECK: immarg value 16 out of range [24, 257)
  call void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 16)

  ; CHECK: immarg value 264 out of range [24, 257)
  call void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 264)
  ret void
}
