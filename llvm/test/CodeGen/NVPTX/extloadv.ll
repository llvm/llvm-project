; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_35 | %ptxas-verify -arch=sm_35 %}

define void @foo(float* nocapture readonly %x_value, double* nocapture %output) #0 {
  %1 = bitcast float* %x_value to <4 x float>*
  %2 = load <4 x float>, <4 x float>* %1, align 16
  %3 = fpext <4 x float> %2 to <4 x double>
; CHECK-NOT: ld.v2.f32 {%fd{{[0-9]+}}, %fd{{[0-9]+}}}, [%rd{{[0-9]+}}];
; CHECK:  cvt.f64.f32
; CHECK:  cvt.f64.f32
; CHECK:  cvt.f64.f32
; CHECK:  cvt.f64.f32
  %4 = bitcast double* %output to <4 x double>*
  store <4 x double> %3, <4 x double>* %4
  ret void
}
