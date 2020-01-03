; RUN: llc -mtriple=x86_64-unknown-linux-gnu -x86-asm-syntax=intel -relocation-model=static < %s | FileCheck %s

; Test we are emitting the 'offset' operator upon an immediate reference of a label:
; The emitted 'att-equivalent' of this one is "movl $.L.str, %eax"

@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

define i8* @test_offset_operator() {
; CHECK-LABEL: test_offset_operator:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mov eax, offset .L.str
; CHECK-NEXT:    ret
entry:
  ret i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0)
}
