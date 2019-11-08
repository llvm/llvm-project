; RUN: llc -O0 --march=aarch64 -verify-machineinstrs --filetype=asm %s -o - 2>&1 | FileCheck %s
; RUN: llc -O1 --march=aarch64 -verify-machineinstrs --filetype=asm %s -o - 2>&1 | FileCheck %s
; RUN: llc -O2 --march=aarch64 -verify-machineinstrs --filetype=asm %s -o - 2>&1 | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare void @normal_cc()

; Caller: preserve_mostcc; callee: normalcc. Normally callee saved registers
; x9~x15 need to be spilled. Since most of them will be spilled in pairs in
; reverse order, we only check the odd number ones due to FileCheck not
; matching the same line of assembly twice.
; CHECK-LABEL: preserve_most
; CHECK-DAG: {{st[rp]}} {{(x[0-9]+, )?x9(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(x[0-9]+, )?x11(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(x[0-9]+, )?x13(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(x[0-9]+, )?x15(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
define preserve_mostcc void @preserve_most() {
  call void @normal_cc()
  ret void
}

; Caller: normalcc; callee: preserve_mostcc. x9 does not need to be spilled.
; The same holds for x10 through x15, but we only check x9.
; CHECK-LABEL: normal_cc_caller
; CHECK-NOT: stp {{x[0-9]+}}, x9, [sp, #{{[-0-9]+}}]
; CHECK-NOT: stp x9, {{x[0-9]+}}, [sp, #{{[-0-9]+}}]
; CHECK-NOT: str x9, [sp, {{#[-0-9]+}}]
define dso_local void @normal_cc_caller() {
entry:
  %v = alloca i32, align 4
  call void asm sideeffect "mov x9, $0", "N,~{x9}"(i32 48879) #2
  call preserve_mostcc void @preserve_most()
  %0 = load i32, i32* %v, align 4
  %1 = call i32 asm sideeffect "mov ${0:w}, w9", "=r,r"(i32 %0) #2
  store i32 %1, i32* %v, align 4
  ret void
}
