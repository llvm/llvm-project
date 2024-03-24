; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s  

declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

define void @kernel_func() {

  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()

  switch i32 %1, label %unreachabledefault [
    i32 0, label %bb0
    i32 1, label %bb1
    i32 2, label %bb1
    i32 3, label %bb2
  ]

  bb0:
    ret void

  bb1:
    ret void

  bb2:
    ret void

  unreachabledefault:
    unreachable

; CHECK:  @kernel_func
; CHECK-NOT: unreachabledefault
; CHECK:  -- End function
}
