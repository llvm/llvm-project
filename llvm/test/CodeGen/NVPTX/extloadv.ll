; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

define void @foo(ptr nocapture readonly %x_value, ptr nocapture %output) #0 {
  %1 = load <4 x float>, ptr %x_value, align 16
  %2 = fpext <4 x float> %1 to <4 x double>
; CHECK-NOT: ld.v2.f32 {%fd{{[0-9]+}}, %fd{{[0-9]+}}}, [%rd{{[0-9]+}}];
; CHECK:  cvt.f64.f32
; CHECK:  cvt.f64.f32
; CHECK:  cvt.f64.f32
; CHECK:  cvt.f64.f32
  store <4 x double> %2, ptr %output
  ret void
}
