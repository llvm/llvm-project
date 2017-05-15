; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s



; CHECK: .func ({{.*}}) device_func
define float @device_func(float %a) noinline {
  %ret = fmul float %a, %a
  ret float %ret
}

; CHECK: .entry kernel_func
define void @kernel_func(float* %a) {
  %val = load float, float* %a
; CHECK: call.uni (retval0),
; CHECK: device_func,
  %mul = call float @device_func(float %val)
  store float %mul, float* %a
  ret void
}



!nvvm.annotations = !{!1}

!1 = !{void (float*)* @kernel_func, !"kernel", i32 1}
