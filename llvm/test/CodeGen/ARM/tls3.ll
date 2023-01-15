; RUN: llc -mtriple arm-linux-gnueabi -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix NOEMU
; RUN: llc -mtriple arm-linux-gnueabi -emulated-tls -filetype asm -o - %s | FileCheck %s -check-prefix CHECK -check-prefix EMU

%struct.anon = type { i32, i32 }
@teste = internal thread_local global %struct.anon zeroinitializer

define i32 @main() {
entry:
  %tmp2 = load i32, ptr @teste, align 8
  ret i32 %tmp2
}

; CHECK-LABEL: main:
; NOEMU-NOT:   __emutls_get_address

; NOEMU:       .section .tbss
; NOEMU-LABEL: teste:
; NOEMU-NEXT:  .zero 8

; CHECK-NOT: __emutls_t.teste

; EMU:       .p2align 2
; EMU-LABEL: __emutls_v.teste:
; EMU-NEXT:  .long 8
; EMU-NEXT:  .long 4
; EMU-NEXT:  .long 0
; EMU-NEXT:  .long 0

; CHECK-NOT: teste:
; CHECK-NOT: __emutls_t.teste

