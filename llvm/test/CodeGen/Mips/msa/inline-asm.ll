; A basic inline assembly test

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck %s

@v4i32_r  = global <4 x i32> zeroinitializer, align 16

define void @test1() nounwind {
entry:
  ; CHECK-LABEL: test1:
  %0 = call <4 x i32> asm "ldi.w ${0:w}, 1", "=f"()
  ; CHECK: ldi.w $w{{[1-3]?[0-9]}}, 1
  store <4 x i32> %0, ptr @v4i32_r
  ret void
}

define void @test2() nounwind {
entry:
  ; CHECK-LABEL: test2:
  %0 = load <4 x i32>, ptr @v4i32_r
  %1 = call <4 x i32> asm "addvi.w ${0:w}, ${1:w}, 1", "=f,f"(<4 x i32> %0)
  ; CHECK: addvi.w $w{{[1-3]?[0-9]}}, $w{{[1-3]?[0-9]}}, 1
  store <4 x i32> %1, ptr @v4i32_r
  ret void
}

define void @test3() nounwind {
entry:
  ; CHECK-LABEL: test3:
  %0 = load <4 x i32>, ptr @v4i32_r
  %1 = call <4 x i32> asm sideeffect "addvi.w ${0:w}, ${1:w}, 1", "=f,f,~{$w0}"(<4 x i32> %0)
  ; CHECK: addvi.w $w{{([1-9]|[1-3][0-9])}}, $w{{([1-9]|[1-3][0-9])}}, 1
  store <4 x i32> %1, ptr @v4i32_r
  ret void
}
