; RUN: llc -mtriple=x86_64-apple-darwin -mcpu=corei7                             < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-apple-darwin -mcpu=corei7 -fast-isel -fast-isel-abort=1 < %s | FileCheck %s --check-prefix=FAST

; Test the webkit_jscc calling convention.
; One argument will be passed in register, the other will be pushed on the stack.
; Return value in $rax.
define void @jscall_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: jscall_patchpoint_codegen:
; CHECK:      movq %r{{.+}}, (%rsp)
; CHECK:      movq %r{{.+}}, %rax
; CHECK:      Ltmp
; CHECK-NEXT: movabsq $-559038736, %r11
; CHECK-NEXT: callq *%r11
; CHECK:      movq %rax, (%rsp)
; CHECK:      callq
; FAST-LABEL: jscall_patchpoint_codegen:
; FAST:       movq %r{{.+}}, (%rsp)
; FAST:       movq %r{{.+}}, %rax
; FAST:       Ltmp
; FAST-NEXT:  movabsq $-559038736, %r11
; FAST-NEXT:  callq *%r11
; FAST:       movq %rax, (%rsp)
; FAST:       callq
  %resolveCall2 = inttoptr i64 -559038736 to ptr
  %result = tail call webkit_jscc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 15, ptr %resolveCall2, i32 2, i64 %p4, i64 %p2)
  %resolveCall3 = inttoptr i64 -559038737 to ptr
  tail call webkit_jscc void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 6, i32 15, ptr %resolveCall3, i32 2, i64 %p4, i64 %result)
  ret void
}

; Test if the arguments are properly aligned and that we don't store undef arguments.
define i64 @jscall_patchpoint_codegen2(i64 %callee) {
entry:
; CHECK-LABEL: jscall_patchpoint_codegen2:
; CHECK:      movq $6, 24(%rsp)
; CHECK-NEXT: movl $4, 16(%rsp)
; CHECK-NEXT: movq $2, (%rsp)
; CHECK:      Ltmp
; CHECK-NEXT: movabsq $-559038736, %r11
; CHECK-NEXT: callq *%r11
; FAST-LABEL: jscall_patchpoint_codegen2:
; FAST:       movq $2, (%rsp)
; FAST-NEXT:  movl $4, 16(%rsp)
; FAST-NEXT:  movq $6, 24(%rsp)
; FAST:       Ltmp
; FAST-NEXT:  movabsq $-559038736, %r11
; FAST-NEXT:  callq *%r11
  %call = inttoptr i64 -559038736 to ptr
  %result = call webkit_jscc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 7, i32 15, ptr %call, i32 6, i64 %callee, i64 2, i64 undef, i32 4, i32 undef, i64 6)
  ret i64 %result
}

; Test if the arguments are properly aligned and that we don't store undef arguments.
define i64 @jscall_patchpoint_codegen3(i64 %callee) {
entry:
; CHECK-LABEL: jscall_patchpoint_codegen3:
; CHECK:      movq $10, 48(%rsp)
; CHECK-NEXT: movl  $8, 36(%rsp)
; CHECK-NEXT: movq  $6, 24(%rsp)
; CHECK-NEXT: movl  $4, 16(%rsp)
; CHECK-NEXT: movq  $2, (%rsp)
; CHECK:      Ltmp
; CHECK-NEXT: movabsq $-559038736, %r11
; CHECK-NEXT: callq *%r11
; FAST-LABEL: jscall_patchpoint_codegen3:
; FAST:       movq  $2, (%rsp)
; FAST-NEXT:  movl  $4, 16(%rsp)
; FAST-NEXT:  movq  $6, 24(%rsp)
; FAST-NEXT:  movl  $8, 36(%rsp)
; FAST-NEXT:  movq $10, 48(%rsp)
; FAST:       Ltmp
; FAST-NEXT:  movabsq $-559038736, %r11
; FAST-NEXT:  callq *%r11
  %call = inttoptr i64 -559038736 to ptr
  %result = call webkit_jscc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 7, i32 15, ptr %call, i32 10, i64 %callee, i64 2, i64 undef, i32 4, i32 undef, i64 6, i32 undef, i32 8, i32 undef, i64 10)
  ret i64 %result
}

declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)

