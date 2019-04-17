; RUN: opt -reassociate -S < %s | FileCheck %s

declare void @use(float)

define void @test1(float %x, float %y) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[TMP1:%.*]] = fmul fast float %y, %x
; CHECK-NEXT:    [[TMP2:%.*]] = fmul fast float %y, %x
; CHECK-NEXT:    [[TMP3:%.*]] = fsub fast float [[TMP1]], [[TMP2]]
; CHECK-NEXT:    call void @use(float [[TMP1]])
; CHECK-NEXT:    call void @use(float [[TMP3]])
; CHECK-NEXT:    ret void
;
  %1 = fmul fast float %x, %y
  %2 = fmul fast float %y, %x
  %3 = fsub fast float %1, %2
  call void @use(float %1)
  call void @use(float %3)
  ret void
}

define float @test2(float %x, float %y) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    [[TMP1:%.*]] = fmul fast float %y, %x
; CHECK-NEXT:    [[TMP2:%.*]] = fmul fast float %y, %x
; CHECK-NEXT:    [[TMP3:%.*]] = fsub fast float [[TMP1]], [[TMP2]]
; CHECK-NEXT:    ret float [[TMP3]]
;
  %1 = fmul fast float %x, %y
  %2 = fmul fast float %y, %x
  %3 = fsub fast float %1, %2
  ret float %3
}

define float @test3(float %x, float %y) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    [[FACTOR:%.*]] = fmul fast float %y, %x
; CHECK-NEXT:    [[REASS_MUL:%.*]] = fmul fast float [[FACTOR]], 2.000000e+00
; CHECK-NEXT:    ret float [[REASS_MUL]]
;
  %1 = fmul fast float %x, %y
  %2 = fmul fast float %y, %x
  %3 = fadd fast float %1, %2
  ret float %3
}

