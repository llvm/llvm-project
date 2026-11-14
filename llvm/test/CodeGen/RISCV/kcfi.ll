; RUN: llc -mtriple=riscv32 -verify-machineinstrs -M no-aliases < %s \
; RUN:      | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple=riscv64 -verify-machineinstrs -M no-aliases < %s \
; RUN:      | FileCheck %s --check-prefixes=CHECK,RV64
; RUN: llc -mtriple=riscv32 -global-isel -global-isel-abort=1 \
; RUN:      -verify-machineinstrs -M no-aliases < %s \
; RUN:      | FileCheck %s --check-prefix=RV32-GISEL
; RUN: llc -mtriple=riscv64 -global-isel -global-isel-abort=1 \
; RUN:      -verify-machineinstrs -M no-aliases < %s \
; RUN:      | FileCheck %s --check-prefix=RV64-GISEL

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
;
; RV32-GISEL-LABEL: f1:
; RV32-GISEL:  # %bb.0:
; RV32-GISEL:    lw t1, -4(a0)
; RV32-GISEL:    lui t2, 3014
; RV32-GISEL:    addi t2, t2, 334
; RV32-GISEL:    beq t1, t2, .Ltmp0
; RV32-GISEL:  .Ltmp1:
; RV32-GISEL:    ebreak
; RV32-GISEL:    .section .kcfi_traps,"ao",@progbits,.text
; RV32-GISEL:  .Ltmp2:
; RV32-GISEL:    .word .Ltmp1-.Ltmp2
; RV32-GISEL:    .text
; RV32-GISEL:  .Ltmp0:
; RV32-GISEL:    jalr ra, 0(a0)
; RV32-GISEL:    lw t1, -4(s0)
; RV32-GISEL:    addi t2, t2, 1234
; RV32-GISEL:    beq t1, t2, .Ltmp3
; RV32-GISEL:  .Ltmp4:
; RV32-GISEL:    ebreak
; RV32-GISEL:    .section .kcfi_traps,"ao",@progbits,.text
; RV32-GISEL:  .Ltmp5:
; RV32-GISEL:    .word .Ltmp4-.Ltmp5
; RV32-GISEL:    .text
; RV32-GISEL:  .Ltmp3:
; RV32-GISEL:    jalr ra, 0(s0)
;
; RV64-GISEL-LABEL: f1:
; RV64-GISEL:  # %bb.0:
; RV64-GISEL:    lw t1, -4(a0)
; RV64-GISEL:    lui t2, 3014
; RV64-GISEL:    addiw t2, t2, 334
; RV64-GISEL:    beq t1, t2, .Ltmp0
; RV64-GISEL:  .Ltmp1:
; RV64-GISEL:    ebreak
; RV64-GISEL:    .section .kcfi_traps,"ao",@progbits,.text
; RV64-GISEL:  .Ltmp2:
; RV64-GISEL:    .word .Ltmp1-.Ltmp2
; RV64-GISEL:    .text
; RV64-GISEL:  .Ltmp0:
; RV64-GISEL:    jalr ra, 0(a0)
; RV64-GISEL:    lw t1, -4(s0)
; RV64-GISEL:    addi t2, t2, 1234
; RV64-GISEL:    beq t1, t2, .Ltmp3
; RV64-GISEL:  .Ltmp4:
; RV64-GISEL:    ebreak
; RV64-GISEL:    .section .kcfi_traps,"ao",@progbits,.text
; RV64-GISEL:  .Ltmp5:
; RV64-GISEL:    .word .Ltmp4-.Ltmp5
; RV64-GISEL:    .text
; RV64-GISEL:  .Ltmp3:
; RV64-GISEL:    jalr ra, 0(s0)
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
;
; RV32-GISEL-LABEL: f2:
; RV32-GISEL:  # %bb.0:
; RV32-GISEL:    addi zero, zero, 0
; RV32-GISEL:    addi zero, zero, 0
; RV32-GISEL:    lw t1, -4(a0)
; RV32-GISEL:    lui t2, 3014
; RV32-GISEL:    addi t2, t2, 334
; RV32-GISEL:    beq t1, t2, .Ltmp6
; RV32-GISEL:  .Ltmp7:
; RV32-GISEL:    ebreak
; RV32-GISEL:    .section .kcfi_traps,"ao",@progbits,.text
; RV32-GISEL:  .Ltmp8:
; RV32-GISEL:    .word .Ltmp7-.Ltmp8
; RV32-GISEL:    .text
; RV32-GISEL:  .Ltmp6:
; RV32-GISEL:    jalr ra, 0(a0)
;
; RV64-GISEL-LABEL: f2:
; RV64-GISEL:  # %bb.0:
; RV64-GISEL:    addi zero, zero, 0
; RV64-GISEL:    addi zero, zero, 0
; RV64-GISEL:    lw t1, -4(a0)
; RV64-GISEL:    lui t2, 3014
; RV64-GISEL:    addiw t2, t2, 334
; RV64-GISEL:    beq t1, t2, .Ltmp6
; RV64-GISEL:  .Ltmp7:
; RV64-GISEL:    ebreak
; RV64-GISEL:    .section .kcfi_traps,"ao",@progbits,.text
; RV64-GISEL:  .Ltmp8:
; RV64-GISEL:    .word .Ltmp7-.Ltmp8
; RV64-GISEL:    .text
; RV64-GISEL:  .Ltmp6:
; RV64-GISEL:    jalr ra, 0(a0)
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @f3(ptr noundef %x) {
; CHECK-LABEL: f3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi sp, sp, -16
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; RV32-NEXT:     sw ra, 12(sp) # 4-byte Folded Spill
; RV64-NEXT:     sd ra, 8(sp) # 8-byte Folded Spill
; RV32-NEXT:     .cfi_offset ra, -4
; RV64-NEXT:     .cfi_offset ra, -8
; CHECK-NEXT:    jalr ra, 0(a0)
; RV32-NEXT:     lw ra, 12(sp) # 4-byte Folded Reload
; RV64-NEXT:     ld ra, 8(sp) # 8-byte Folded Reload
; CHECK-NEXT:    .cfi_restore ra
; CHECK-NEXT:    addi sp, sp, 16
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    jalr zero, 0(ra)
;
; RV32-GISEL-LABEL: f3:
; RV32-GISEL:       # %bb.0:
; RV32-GISEL-NEXT:    addi sp, sp, -16
; RV32-GISEL-NEXT:    .cfi_def_cfa_offset 16
; RV32-GISEL-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32-GISEL-NEXT:    .cfi_offset ra, -4
; RV32-GISEL-NEXT:    jalr ra, 0(a0)
; RV32-GISEL-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32-GISEL-NEXT:    .cfi_restore ra
; RV32-GISEL-NEXT:    addi sp, sp, 16
; RV32-GISEL-NEXT:    .cfi_def_cfa_offset 0
; RV32-GISEL-NEXT:    jalr zero, 0(ra)
;
; RV64-GISEL-LABEL: f3:
; RV64-GISEL:       # %bb.0:
; RV64-GISEL-NEXT:    addi sp, sp, -16
; RV64-GISEL-NEXT:    .cfi_def_cfa_offset 16
; RV64-GISEL-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64-GISEL-NEXT:    .cfi_offset ra, -8
; RV64-GISEL-NEXT:    jalr ra, 0(a0)
; RV64-GISEL-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64-GISEL-NEXT:    .cfi_restore ra
; RV64-GISEL-NEXT:    addi sp, sp, 16
; RV64-GISEL-NEXT:    .cfi_def_cfa_offset 0
; RV64-GISEL-NEXT:    jalr zero, 0(ra)
  call void %x()
  ret void
}

declare void @g()

define void @f4() {
; CHECK-LABEL: f4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi sp, sp, -16
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; RV32-NEXT:     sw ra, 12(sp) # 4-byte Folded Spill
; RV64-NEXT:     sd ra, 8(sp) # 8-byte Folded Spill
; RV32-NEXT:     .cfi_offset ra, -4
; RV64-NEXT:     .cfi_offset ra, -8
; CHECK-NEXT:    call g
; RV32-NEXT:     lw ra, 12(sp) # 4-byte Folded Reload
; RV64-NEXT:     ld ra, 8(sp) # 8-byte Folded Reload
; CHECK-NEXT:    .cfi_restore ra
; CHECK-NEXT:    addi sp, sp, 16
; CHECK-NEXT:    .cfi_def_cfa_offset 0
; CHECK-NEXT:    jalr zero, 0(ra)
;
; RV32-GISEL-LABEL: f4:
; RV32-GISEL:       # %bb.0:
; RV32-GISEL-NEXT:    addi sp, sp, -16
; RV32-GISEL-NEXT:    .cfi_def_cfa_offset 16
; RV32-GISEL-NEXT:    sw ra, 12(sp) # 4-byte Folded Spill
; RV32-GISEL-NEXT:    .cfi_offset ra, -4
; RV32-GISEL-NEXT:    call g
; RV32-GISEL-NEXT:    lw ra, 12(sp) # 4-byte Folded Reload
; RV32-GISEL-NEXT:    .cfi_restore ra
; RV32-GISEL-NEXT:    addi sp, sp, 16
; RV32-GISEL-NEXT:    .cfi_def_cfa_offset 0
; RV32-GISEL-NEXT:    jalr zero, 0(ra)
;
; RV64-GISEL-LABEL: f4:
; RV64-GISEL:       # %bb.0:
; RV64-GISEL-NEXT:    addi sp, sp, -16
; RV64-GISEL-NEXT:    .cfi_def_cfa_offset 16
; RV64-GISEL-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64-GISEL-NEXT:    .cfi_offset ra, -8
; RV64-GISEL-NEXT:    call g
; RV64-GISEL-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64-GISEL-NEXT:    .cfi_restore ra
; RV64-GISEL-NEXT:    addi sp, sp, 16
; RV64-GISEL-NEXT:    .cfi_def_cfa_offset 0
; RV64-GISEL-NEXT:    jalr zero, 0(ra)
  call void @g()
  ret void
}

define void @f5(ptr noundef %x) {
; CHECK-LABEL: f5:
; CHECK:       # %bb.0:
; CHECK:         lw t1, -4(a0)
; CHECK-NEXT:    lui t2, 912092
; RV32-NEXT:     addi t2, t2, -273
; RV64-NEXT:     addiw t2, t2, -273
; CHECK-NEXT:    beq t1, t2, .Ltmp9
; CHECK-NEXT:  .Ltmp10:
; CHECK-NEXT:    ebreak
; CHECK-NEXT:    .section .kcfi_traps,"ao",@progbits,.text
; CHECK-NEXT:  .Ltmp11:
; CHECK-NEXT:    .word .Ltmp10-.Ltmp11
; CHECK-NEXT:    .text
; CHECK-NEXT:  .Ltmp9:
; CHECK-NEXT:    jalr ra, 0(a0)
;
; RV32-GISEL-LABEL: f5:
; RV32-GISEL:  # %bb.0:
; RV32-GISEL:    lw t1, -4(a0)
; RV32-GISEL:    lui t2, 912092
; RV32-GISEL:    addi t2, t2, -273
; RV32-GISEL:    beq t1, t2, .Ltmp9
; RV32-GISEL:  .Ltmp10:
; RV32-GISEL:    ebreak
; RV32-GISEL:    .section .kcfi_traps,"ao",@progbits,.text
; RV32-GISEL:  .Ltmp11:
; RV32-GISEL:    .word .Ltmp10-.Ltmp11
; RV32-GISEL:    .text
; RV32-GISEL:  .Ltmp9:
; RV32-GISEL:    jalr ra, 0(a0)
;
; RV64-GISEL-LABEL: f5:
; RV64-GISEL:  # %bb.0:
; RV64-GISEL:    lw t1, -4(a0)
; RV64-GISEL:    lui t2, 912092
; RV64-GISEL:    addiw t2, t2, -273
; RV64-GISEL:    beq t1, t2, .Ltmp9
; RV64-GISEL:  .Ltmp10:
; RV64-GISEL:    ebreak
; RV64-GISEL:    .section .kcfi_traps,"ao",@progbits,.text
; RV64-GISEL:  .Ltmp11:
; RV64-GISEL:    .word .Ltmp10-.Ltmp11
; RV64-GISEL:    .text
; RV64-GISEL:  .Ltmp9:
; RV64-GISEL:    jalr ra, 0(a0)
  call void %x() [ "kcfi"(i32 -559038737) ]
  ret void
}

attributes #0 = { "patchable-function-entry"="2" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
