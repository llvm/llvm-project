;; Test the function attribute "patchable-function-entry".
; RUN: llc -mtriple=riscv32 --riscv-no-aliases < %s | FileCheck %s --check-prefixes=CHECK,RV32,NORVC
; RUN: llc -mtriple=riscv64 --riscv-no-aliases < %s | FileCheck %s --check-prefixes=CHECK,RV64,NORVC
; RUN: llc -mtriple=riscv32 -mattr=+c --riscv-no-aliases < %s | FileCheck %s --check-prefixes=CHECK,RV32,RVC
; RUN: llc -mtriple=riscv64 -mattr=+c --riscv-no-aliases < %s | FileCheck %s --check-prefixes=CHECK,RV64,RVC

define void @f0() "patchable-function-entry"="0" {
; CHECK-LABEL: f0:
; CHECK-NEXT:  .Lfunc_begin0:
; CHECK-NOT:     {{addi|c.nop}}
; NORVC:         jalr zero, 0(ra)
; RVC:           c.jr ra
; CHECK-NOT:   .section __patchable_function_entries
  ret void
}

define void @f1() "patchable-function-entry"="1" {
; CHECK-LABEL: f1:
; CHECK-NEXT: .Lfunc_begin1:
; NORVC:         addi zero, zero, 0
; NORVC-NEXT:    jalr zero, 0(ra)
; RVC:           c.nop
; RVC-NEXT:      c.jr ra
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f1{{$}}
; 32:          .p2align 2
; 32-NEXT:     .word .Lfunc_begin1
; 64:          .p2align 3
; 64-NEXT:     .quad .Lfunc_begin1
  ret void
}

$f5 = comdat any
define void @f5() "patchable-function-entry"="5" comdat {
; CHECK-LABEL: f5:
; CHECK-NEXT:  .Lfunc_begin2:
; NORVC-COUNT-5: addi zero, zero, 0
; NORVC-NEXT:    jalr zero, 0(ra)
; RVC-COUNT-5:   c.nop
; RVC-NEXT:      c.jr ra
; CHECK:       .section __patchable_function_entries,"awoG",@progbits,f5,f5,comdat{{$}}
; RV32:        .p2align 2
; RV32-NEXT:   .word .Lfunc_begin2
; RV64:        .p2align 3
; RV64-NEXT:   .quad .Lfunc_begin2
  ret void
}

;; -fpatchable-function-entry=3,2
;; "patchable-function-prefix" emits data before the function entry label.
define void @f3_2() "patchable-function-entry"="1" "patchable-function-prefix"="2" {
; CHECK-LABEL: .type f3_2,@function
; CHECK-NEXT:  .Ltmp0:
; NORVC-COUNT-2: addi zero, zero, 0
; RVC-COUNT-2:   c.nop
; CHECK-NEXT:  f3_2:  # @f3_2
; CHECK:       # %bb.0:
; NORVC-NEXT:    addi zero, zero, 0
; NORVC-NEXT:    addi sp, sp, -16
; RVC-NEXT:      c.nop
; RVC-NEXT:      c.addi sp, -16
;; .size does not include the prefix.
; CHECK:      .Lfunc_end3:
; CHECK-NEXT: .size f3_2, .Lfunc_end3-f3_2
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f3_2{{$}}
; RV32:       .p2align 2
; RV32-NEXT:  .word .Ltmp0
; RV64:       .p2align 3
; RV64-NEXT:  .quad .Ltmp0
  %frame = alloca i8, i32 16
  ret void
}
