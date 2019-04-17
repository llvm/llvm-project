; RUN: opt -reassociate -S < %s | FileCheck %s

declare void @use(float)

define void @test1(float %x, float %y) {
; CHECK-LABEL: test1
; CHECK: fmul float %x, %y
; CHECK: fmul float %x, %y
; CHECK: fsub float %1, %2
; CHECK: call void @use(float %{{.*}})
; CHECK: call void @use(float %{{.*}})

  %1 = fmul float %x, %y
  %2 = fmul float %y, %x
  %3 = fsub float %1, %2
  call void @use(float %1)
  call void @use(float %3)
  ret void
}
