; RUN: llc < %s -march=arm -mtriple=arm-linux-gnueabi | \
; RUN:     grep "tbss"
; RUN: llc < %s -march=arm -mtriple=arm-linux-gnueabi | \
; RUN:     FileCheck %s -check-prefix=CHECK -check-prefix=NOEMU
; RUN: llc < %s -emulated-tls -march=arm -mtriple=arm-linux-gnueabi | \
; RUN:     FileCheck %s -check-prefix=CHECK -check-prefix=EMU

%struct.anon = type { i32, i32 }
@teste = internal thread_local global %struct.anon zeroinitializer ; <%struct.anon*> [#uses=1]

define i32 @main() {
entry:
  %tmp2 = load i32, i32* getelementptr (%struct.anon, %struct.anon* @teste, i32 0, i32 0), align 8 ; <i32> [#uses=1]
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
