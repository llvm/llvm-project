; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs \
; RUN:     | FileCheck %s  --check-prefix=CHECK --check-prefix=CHECK-NOTRAP
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs \
; RUN:     | FileCheck %s  --check-prefix=CHECK --check-prefix=CHECK-NOTRAP
; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs -trap-unreachable \
; RUN:     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-TRAP
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs -trap-unreachable \
; RUN:     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-TRAP
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

; CHECK: .extern .func throw
declare void @throw() #0

; CHECK: .entry kernel_func
define void @kernel_func() {
; CHECK: call.uni
; CHECK: throw,
  call void @throw()
; CHECK-TRAP-NOT: exit;
; CHECK-TRAP: trap;
; CHECK-NOTRAP-NOT: trap;
; CHECK: exit;
  unreachable
}

attributes #0 = { noreturn }


!nvvm.annotations = !{!1}

!1 = !{ptr @kernel_func, !"kernel", i32 1}
