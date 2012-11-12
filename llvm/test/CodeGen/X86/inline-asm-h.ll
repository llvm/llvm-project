; RUN: llc -mtriple=x86_64-pc-linux-gnu < %s | FileCheck %s

@foobar = common global i32 0, align 4

define void @zed() nounwind {
entry:
  call void asm "movq %mm2,${0:H}", "=*m,~{dirflag},~{fpsr},~{flags}"(i32* @foobar) nounwind
  ret void
}

; CHECK: zed
; CHECK: movq %mm2,foobar+8(%rip)
