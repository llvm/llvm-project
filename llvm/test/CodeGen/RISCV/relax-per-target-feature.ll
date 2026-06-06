; RUN: llc -filetype=obj -mtriple=riscv64 %s -o %t
; RUN: llvm-objdump -dr -M no-aliases --mattr=+c %t | FileCheck %s

;; Functions may have more features than the base triple; code generation and
;; instruction selection may be performed based on this information. This test
;; makes sure that the MC layer uses the target-features of the function.

declare dso_local i32 @ext(i32)

; CHECK-LABEL: <f>:
; CHECK-NEXT:    c.li a0, 0x1f
; CHECK-NEXT:    auipc t1, 0x0
; CHECK-NEXT:    R_RISCV_CALL_PLT     ext
; CHECK-NEXT:    R_RISCV_RELAX *ABS*
; CHECK-NEXT:    jalr zero, 0x0(t1)
define dso_local i32 @f() #0 {
entry:
  %r = tail call i32 @ext(i32 31)
  ret i32 %r
}

; CHECK-LABEL: <g>:
; CHECK-NEXT:    addi a0, zero, 0x1f
; CHECK-NEXT:    auipc t1, 0x0
; CHECK-NEXT:    R_RISCV_CALL_PLT     ext
; CHECK-NEXT:    jalr zero, 0x0(t1)
define dso_local i32 @g() #1 {
entry:
  %r = tail call i32 @ext(i32 31)
  ret i32 %r
}

attributes #0 = { nounwind "target-features"="+c,+relax" }
attributes #1 = { nounwind "target-features"="-c,-relax" }
