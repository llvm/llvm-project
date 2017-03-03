; RUN: llc < %s -O0 -mtriple=armv7-eabi -mcpu=cortex-a8 | FileCheck %s
; RUN: llc < %s -O3 -mtriple=armv7-eabi -mcpu=cortex-a8 | FileCheck %s

; Function Attrs: nounwind
define void @fn1(i32* nocapture %p) local_unnamed_addr {
entry:
  ; CHECK: vmrs r{{[0-9]+}}, fpscr
  %0 = tail call i32 @llvm.arm.get.fpscr()
  store i32 %0, i32* %p, align 4
  ; CHECK: vmsr fpscr, r{{[0-9]+}}
  tail call void @llvm.arm.set.fpscr(i32 1)
  ; CHECK: vmrs r{{[0-9]+}}, fpscr
  %1 = tail call i32 @llvm.arm.get.fpscr()
  %arrayidx1 = getelementptr inbounds i32, i32* %p, i32 1
  store i32 %1, i32* %arrayidx1, align 4
  ret void
}

; Function Attrs: nounwind readonly
declare i32 @llvm.arm.get.fpscr()

; Function Attrs: nounwind writeonly
declare void @llvm.arm.set.fpscr(i32)
