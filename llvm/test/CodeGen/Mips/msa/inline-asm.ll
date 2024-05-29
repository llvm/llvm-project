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

define dso_local double @test4(double noundef %a, double noundef %b, double noundef %c) {
entry:
  ; CHECK-LABEL: test4:
  %0 = tail call double asm sideeffect "fmadd.d ${0:w}, ${1:w}, ${2:w}", "=f,f,f,0,~{$1}"(double %b, double %c, double %a)
  ; CHECK: fmadd.d $w{{([0-9]|[1-3][0-9])}}, $w{{([0-9]|[1-3][0-9])}}, $w{{([0-9]|[1-3][0-9])}}
  ret double %0
}

define dso_local float @test5(float noundef %a, float noundef %b, float noundef %c) {
entry:
  ; CHECK-LABEL: test5:
  %0 = tail call float asm sideeffect "fmadd.w ${0:w}, ${1:w}, ${2:w}", "=f,f,f,0,~{$1}"(float %b, float %c, float %a)
  ; CHECK: fmadd.w $w{{([0-9]|[1-3][0-9])}}, $w{{([0-9]|[1-3][0-9])}}, $w{{([0-9]|[1-3][0-9])}}
  ret float %0
}
