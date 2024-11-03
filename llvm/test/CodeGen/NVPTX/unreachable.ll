; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

; CHECK: .extern .func throw
declare void @throw() #0

; CHECK: .entry kernel_func
define void @kernel_func() {
; CHECK: call.uni
; CHECK: throw,
  call void @throw()
; CHECK: exit
  unreachable
}

attributes #0 = { noreturn }


!nvvm.annotations = !{!1}

!1 = !{ptr @kernel_func, !"kernel", i32 1}
