; RUN: opt -S -reassociate < %s | FileCheck %s

define void @test1() {
; CHECK-LABEL: @test1
; CHECK: call
; CHECK: fsub
; CHECK: fadd
  %tmp = tail call <4 x float> @blam()
  %tmp23 = fsub fast <4 x float> undef, %tmp
  %tmp24 = fadd fast <4 x float> %tmp23, undef
  tail call void @wombat(<4 x float> %tmp24)
  ret void
}

define half @test2() {
; CHECK-LABEL: @test2
; CHECK: fsub
; CHECK: fsub
; CHECK: fadd
  %tmp15 = fsub fast half undef, undef
  %tmp17 = fsub fast half undef, %tmp15
  %tmp18 = fadd fast half undef, %tmp17
  ret half %tmp18
}



; Function Attrs: optsize
declare <4 x float> @blam()

; Function Attrs: optsize
declare void @wombat(<4 x float>)

