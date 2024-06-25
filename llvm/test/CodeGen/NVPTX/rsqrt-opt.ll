; RUN: llc < %s -march=nvptx64 | FileCheck %s --check-prefixes CHECK,CHECK-APPROX-OPT,CHECK-SQRT-NOOPT
; RUN: llc < %s -march=nvptx64 -nvptx-prec-sqrtf32=0 | FileCheck %s --check-prefixes CHECK,CHECK-APPROX-OPT,CHECK-SQRT-OPT
; RUN: llc < %s -march=nvptx64 -nvptx-rsqrt-approx-opt=0 | FileCheck %s --check-prefixes CHECK,CHECK-APPROX-NOOPT,CHECK-SQRT-NOOPT
;
; RUN: %if ptxas %{ llc < %s -march=nvptx64 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -nvptx-prec-sqrtf32=0 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -nvptx-rsqrt-approx-opt=0 | %ptxas-verify %}


; CHECK-LABEL: .func{{.*}}test1
define float @test1(float %in) local_unnamed_addr {
; CHECK-APPROX-OPT: rsqrt.approx.f32
; CHECK-APPROX-NOOPT: sqrt.approx.f32
; CHECK-APPROX-NOOPT-NEXT: rcp.rn.f32
  %sqrt = tail call float @llvm.nvvm.sqrt.approx.f(float %in)
  %rsqrt = fdiv float 1.0, %sqrt
  ret float %rsqrt
}
; CHECK-LABEL: .func{{.*}}test2
define float @test2(float %in) local_unnamed_addr {
; CHECK-APPROX-OPT: rsqrt.approx.ftz.f32
; CHECK-APPROX-NOOPT: sqrt.approx.ftz.f32
; CHECK-APPROX-NOOPT-NEXT: rcp.rn.f32
  %sqrt = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %in)
  %rsqrt = fdiv float 1.0, %sqrt
  ret float %rsqrt
}

; CHECK-LABEL: .func{{.*}}test3
define float @test3(float %in) local_unnamed_addr {
; CHECK-SQRT-OPT: rsqrt.approx.f32
; CHECK-SQRT-NOOPT: sqrt.rn.f32
; CHECK-SQRT-NOOPT-NEXT: rcp.rn.f32
  %sqrt = tail call float @llvm.nvvm.sqrt.f(float %in)
  %rsqrt = fdiv float 1.0, %sqrt
  ret float %rsqrt
}

; CHECK-LABEL: .func{{.*}}test4
define float @test4(float %in) local_unnamed_addr #0 {
; CHECK-SQRT-OPT: rsqrt.approx.ftz.f32
; CHECK-SQRT-NOOPT: sqrt.rn.ftz.f32
; CHECK-SQRT-NOOPT-NEXT: rcp.rn.ftz.f32
  %sqrt = tail call float @llvm.nvvm.sqrt.f(float %in)
  %rsqrt = fdiv float 1.0, %sqrt
  ret float %rsqrt
}

; CHECK-LABEL: .func{{.*}}test5
define float @test5(float %in) local_unnamed_addr {
; CHECK-SQRT-OPT: rsqrt.approx.f32
; CHECK-SQRT-NOOPT: sqrt.rn.f32
; CHECK-SQRT-NOOPT-NEXT: rcp.rn.f32
  %sqrt = tail call float @llvm.sqrt.f32(float %in)
  %rsqrt = fdiv float 1.0, %sqrt
  ret float %rsqrt
}

; CHECK-LABEL: .func{{.*}}test6
define float @test6(float %in) local_unnamed_addr #0 {
; CHECK-SQRT-OPT: rsqrt.approx.ftz.f32
; CHECK-SQRT-NOOPT: sqrt.rn.ftz.f32
; CHECK-SQRT-NOOPT-NEXT: rcp.rn.ftz.f32
  %sqrt = tail call float @llvm.sqrt.f32(float %in)
  %rsqrt = fdiv float 1.0, %sqrt
  ret float %rsqrt
}


declare float @llvm.nvvm.sqrt.f(float)
declare float @llvm.nvvm.sqrt.approx.f(float)
declare float @llvm.nvvm.sqrt.approx.ftz.f(float)
declare float @llvm.sqrt.f32(float)

attributes #0 = { "denormal-fp-math-f32" = "preserve-sign" }
