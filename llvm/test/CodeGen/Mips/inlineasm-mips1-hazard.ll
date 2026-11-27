; RUN: llc -mtriple=mips -mcpu=mips1 -filetype=obj < %s \
; RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

; Inline assembly is assembled in .set reorder mode, so hazards inside it get
; nops, and so does a load at its end before the code that follows.

define void @load_at_end(ptr %p) {
; CHECK-LABEL: <load_at_end>:
; CHECK:         lw $2, 0x0($4)
; CHECK-NEXT:    nop
; CHECK-NEXT:    addiu $1, $2, 0x1
; CHECK-NEXT:    jr $ra
; CHECK-NEXT:    sw $1, 0x0($4)
  %v = call i32 asm sideeffect "lw $0, 0($1)", "={$2},r"(ptr %p)
  %r = add i32 %v, 1
  store i32 %r, ptr %p
  ret void
}

define i32 @load_use(ptr %p) {
; CHECK-LABEL: <load_use>:
; CHECK:         lw $2, 0x0($4)
; CHECK-NEXT:    nop
; CHECK-NEXT:    addu $2, $2, $2
  %v = call i32 asm sideeffect "lw $0, 0($1)\0Aaddu $0, $0, $0", "={$2},r"(ptr %p)
  ret i32 %v
}
