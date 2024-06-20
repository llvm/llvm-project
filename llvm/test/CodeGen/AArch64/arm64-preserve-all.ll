; RUN: llc -O0 --mtriple=aarch64 -verify-machineinstrs --filetype=asm %s -o - 2>&1 | FileCheck %s
; RUN: llc -O1 --mtriple=aarch64 -verify-machineinstrs --filetype=asm %s -o - 2>&1 | FileCheck %s
; RUN: llc -O2 --mtriple=aarch64 -verify-machineinstrs --filetype=asm %s -o - 2>&1 | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare void @normal_cc()

; Caller: preserve_allcc; callee: normalcc. Normally callee saved registers
; x9~x15 need to be spilled. Since most of them will be spilled in pairs in
; reverse order, we only check the odd number ones due to FileCheck not
; matching the same line of assembly twice.
; CHECK-LABEL: preserve_all
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q8(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q10(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q12(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q14(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q16(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q18(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q22(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q24(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q26(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q28(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(q[0-9]+, )?q30(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(x[0-9]+, )?x9(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(x[0-9]+, )?x11(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(x[0-9]+, )?x13(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
; CHECK-DAG: {{st[rp]}} {{(x[0-9]+, )?x15(, x[0-9]+)?}}, [sp, #{{[-0-9]+}}]
define preserve_allcc void @preserve_all() {
  call void @normal_cc()
  ret void
}

; Caller: normalcc; callee: preserve_allcc. x9/q9 does not need to be spilled.
; The same holds for other x and q registers, but we only check x9 and q9.
; CHECK-LABEL: normal_cc_caller
; CHECK-NOT: stp {{x[0-9]+}}, x9, [sp, #{{[-0-9]+}}]
; CHECK-NOT: stp x9, {{x[0-9]+}}, [sp, #{{[-0-9]+}}]
; CHECK-NOT: str x9, [sp, {{#[-0-9]+}}]
; CHECK-NOT: stp {{q[0-9]+}}, q9, [sp, #{{[-0-9]+}}]
; CHECK-NOT: stp q9, {{q[0-9]+}}, [sp, #{{[-0-9]+}}]
; CHECK-NOT: str q9, [sp, {{#[-0-9]+}}]
define dso_local void @normal_cc_caller() {
entry:
  %v = alloca i32, align 4
  call void asm sideeffect "mov x9, $0", "N,~{x9}"(i32 48879) #2
  call void asm sideeffect "movi v9.2d, #0","~{v9}" () #2


  call preserve_allcc void @preserve_all()
  %0 = load i32, ptr %v, align 4
  %1 = call i32 asm sideeffect "mov ${0:w}, w9", "=r,r"(i32 %0) #2
  %2 = call i32 asm sideeffect "fneg v9.4s, v9.4s", "=r,~{v9}"() #2
  store i32 %1, ptr %v, align 4
  ret void
}
