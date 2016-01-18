; RUN: opt -reassociate -S < %s | FileCheck %s

declare void @use(float)

define void @test1(float %x, float %y) {
; CHECK-LABEL: test1
; CHECK: fmul fast float %y, %x
; CHECK: fmul fast float %y, %x
; CHECK: fsub fast float %1, %2
; CHECK: call void @use(float %{{.*}})
; CHECK: call void @use(float %{{.*}})

  %1 = fmul fast float %x, %y
  %2 = fmul fast float %y, %x
  %3 = fsub fast float %1, %2
  call void @use(float %1)
  call void @use(float %3)
  ret void
}

define float @test2(float %x, float %y) {
; CHECK-LABEL: test2
; CHECK-NEXT: fmul fast float %y, %x
; CHECK-NEXT: fmul fast float %y, %x
; CHECK-NEXT: fsub fast float %1, %2
; CHECK-NEXT: ret float %3

  %1 = fmul fast float %x, %y
  %2 = fmul fast float %y, %x
  %3 = fsub fast float %1, %2
  ret float %3
}

define float @test3(float %x, float %y) {
; CHECK-LABEL: test3
; CHECK-NEXT: %factor = fmul fast float %y, 2.000000e+00
; CHECK-NEXT: %tmp1 = fmul fast float %factor, %x
; CHECK-NEXT: ret float %tmp1

  %1 = fmul fast float %x, %y
  %2 = fmul fast float %y, %x
  %3 = fadd fast float %1, %2
  ret float %3
}
