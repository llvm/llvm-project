; RUN: llc -mtriple=riscv32 -verify-machineinstrs -riscv-no-aliases < %s \
; RUN:      | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple=riscv64 -verify-machineinstrs -riscv-no-aliases < %s \
; RUN:      | FileCheck %s --check-prefixes=CHECK,RV64

; CHECK:       .word 12345678
define void @f1(ptr noundef %x) !kcfi_type !1 {
; CHECK-LABEL: f1:
; CHECK:       # %bb.0:
; CHECK:         lw t1, -4(a0)
; CHECK-NEXT:    lui t2, 3014
; RV32-NEXT:     addi t2, t2, 334
; RV64-NEXT:     addiw t2, t2, 334
; CHECK-NEXT:    beq t1, t2, .Ltmp0
; CHECK-NEXT:  .Ltmp1:
; CHECK-NEXT:    ebreak
; CHECK-NEXT:    .section .kcfi_traps,"ao",@progbits,.text
; CHECK-NEXT:  .Ltmp2:
; CHECK-NEXT:    .word .Ltmp1-.Ltmp2
; CHECK-NEXT:    .text
; CHECK-NEXT:  .Ltmp0:
; CHECK-NEXT:    jalr ra, 0(a0)
  call void %x() [ "kcfi"(i32 12345678) ]
; CHECK:         lw t1, -4(s0)
; CHECK-NEXT:    addi t2, t2, 1234
; CHECK-NEXT:    beq t1, t2, .Ltmp3
; CHECK-NEXT:  .Ltmp4:
; CHECK-NEXT:    ebreak
; CHECK-NEXT:    .section .kcfi_traps,"ao",@progbits,.text
; CHECK-NEXT:  .Ltmp5:
; CHECK-NEXT:    .word .Ltmp4-.Ltmp5
; CHECK-NEXT:    .text
; CHECK-NEXT:  .Ltmp3:
; CHECK-NEXT:    jalr ra, 0(s0)
  call void %x() [ "kcfi"(i32 1234) ]
  ret void
}

; CHECK-NOT:   .word:
define void @f2(ptr noundef %x) #0 {
; CHECK-LABEL: f2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi zero, zero, 0
; CHECK-NEXT:    addi zero, zero, 0
; CHECK-NEXT:    lw t1, -4(a0)
; CHECK-NEXT:    lui t2, 3014
; RV32-NEXT:     addi t2, t2, 334
; RV64-NEXT:     addiw t2, t2, 334
; CHECK-NEXT:    beq t1, t2, .Ltmp6
; CHECK-NEXT:  .Ltmp7:
; CHECK-NEXT:    ebreak
; CHECK-NEXT:    .section .kcfi_traps,"ao",@progbits,.text
; CHECK-NEXT:  .Ltmp8:
; CHECK-NEXT:    .word .Ltmp7-.Ltmp8
; CHECK-NEXT:    .text
; CHECK-NEXT:  .Ltmp6:
; CHECK-NEXT:    jalr zero, 0(a0)
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { "patchable-function-entry"="2" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
