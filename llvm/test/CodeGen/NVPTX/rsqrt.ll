; RUN: llc < %s -march=nvptx64 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 | %ptxas-verify %}

; CHECK-LABEL: .func{{.*}}test1
define float @test1(float %in) local_unnamed_addr {
; CHECK: rsqrt.approx.f32
  %call = call float @llvm.nvvm.rsqrt.approx.f(float %in)
  ret float %call
}

; CHECK-LABEL: .func{{.*}}test2
define double @test2(double %in) local_unnamed_addr {
; CHECK: rsqrt.approx.f64
  %call = call double @llvm.nvvm.rsqrt.approx.d(double %in)
  ret double %call
}

; CHECK-LABEL: .func{{.*}}test3
define float @test3(float %in) local_unnamed_addr {
; CHECK: rsqrt.approx.ftz.f32
  %call = tail call float @llvm.nvvm.rsqrt.approx.ftz.f(float %in)
  ret float %call
}

; CHECK-LABEL: .func{{.*}}test4
define double @test4(double %in) local_unnamed_addr {
; CHECK: rsqrt.approx.ftz.f64
  %call = tail call double @llvm.nvvm.rsqrt.approx.ftz.d(double %in)
  ret double %call
}

declare float @llvm.nvvm.rsqrt.approx.ftz.f(float)
declare double @llvm.nvvm.rsqrt.approx.ftz.d(double)
declare float @llvm.nvvm.rsqrt.approx.f(float)
declare double @llvm.nvvm.rsqrt.approx.d(double)
