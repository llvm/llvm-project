; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs -trap-unreachable=false \
; RUN:     | FileCheck %s  --check-prefix=CHECK --check-prefix=CHECK-NOTRAP
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs -trap-unreachable=false \
; RUN:     | FileCheck %s  --check-prefix=CHECK --check-prefix=CHECK-NOTRAP
; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs -trap-unreachable -no-trap-after-noreturn \
; RUN:     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOTRAP
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs -trap-unreachable -no-trap-after-noreturn \
; RUN:     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOTRAP
; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs -trap-unreachable -no-trap-after-noreturn=false \
; RUN:     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-TRAP
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs -trap-unreachable -no-trap-after-noreturn=false \
; RUN:     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-TRAP
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

; CHECK: .extern .func throw
declare void @throw() #0
declare void @llvm.trap() #0

; CHECK-LABEL: .entry kernel_func
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

; CHECK-LABEL: kernel_func_2
define void @kernel_func_2() {
; CHECK: trap; exit;
  call void @llvm.trap()

;; Make sure we avoid emitting two trap instructions.
; CHECK-NOT: trap;
; CHECK-NOT: exit;
  unreachable
}

attributes #0 = { noreturn }


!nvvm.annotations = !{!1}

!1 = !{ptr @kernel_func, !"kernel", i32 1}
