; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s

; Check that, if a Win64 ABI function calls a SysV ABI function, all the
; Win64 nonvolatile registers get saved.

; CHECK-LABEL: bar:
define win64cc void @bar(i32 %a, i32 %b) {
; CHECK-DAG: pushq %rdi
; CHECK-DAG: pushq %rsi
; CHECK-DAG: movaps %xmm6,
; CHECK-DAG: movaps %xmm7,
; CHECK-DAG: movaps %xmm8,
; CHECK-DAG: movaps %xmm9,
; CHECK-DAG: movaps %xmm10,
; CHECK-DAG: movaps %xmm11,
; CHECK-DAG: movaps %xmm12,
; CHECK-DAG: movaps %xmm13,
; CHECK-DAG: movaps %xmm14,
; CHECK-DAG: movaps %xmm15,
; CHECK: callq foo
; CHECK: ret
  call x86_64_sysvcc void @foo(i32 %a, i32 %b)
  ret void
}

declare x86_64_sysvcc void @foo(i32 %a, i32 %b)

