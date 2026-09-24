; Check that the RISC-V machine outliner preserves an explicit input section
; when it runs in the full LTO backend.
;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-lto2 run -enable-machine-outliner=always -filetype=asm -o %t.s %t.bc \
; RUN:   -r=%t.bc,shared_0,px -r=%t.bc,shared_1,px -r=%t.bc,shared_2,px \
; RUN:   -r=%t.bc,plain_0,px -r=%t.bc,plain_1,px -r=%t.bc,plain_2,px
; RUN: FileCheck %s --input-file %t.s.0 --check-prefixes=CHECK,NOFS
;
; RUN: llvm-lto2 run -enable-machine-outliner=always --function-sections -filetype=asm -o %t.s %t.bc \
; RUN:   -r=%t.bc,shared_0,px -r=%t.bc,shared_1,px -r=%t.bc,shared_2,px \
; RUN:   -r=%t.bc,plain_0,px -r=%t.bc,plain_1,px -r=%t.bc,plain_2,px
; RUN: FileCheck %s --input-file %t.s.0 --check-prefixes=CHECK,FS

target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-linux-gnu"

; CHECK: .section .sec_shared,"ax",@progbits

define i32 @shared_0(i32 %a, i32 %b) noinline nounwind section ".sec_shared" {
; CHECK-LABEL: shared_0:
; CHECK: call t0, OUTLINED_FUNCTION_0
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 1
  ret i32 %result
}

define i32 @shared_1(i32 %a, i32 %b) noinline nounwind section ".sec_shared" {
; CHECK-LABEL: shared_1:
; CHECK: call t0, OUTLINED_FUNCTION_0
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 2
  ret i32 %result
}

define i32 @shared_2(i32 %a, i32 %b) noinline nounwind section ".sec_shared" {
; CHECK-LABEL: shared_2:
; CHECK: call t0, OUTLINED_FUNCTION_0
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 3
  ret i32 %result
}

define i32 @plain_0(i32 %a, i32 %b) noinline nounwind {
; NOFS: .text{{$}}
; FS: .section .text.plain_0,"ax",@progbits
; CHECK-LABEL: plain_0:
; CHECK: call t0, OUTLINED_FUNCTION_1
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 4
  ret i32 %result
}

define i32 @plain_1(i32 %a, i32 %b) noinline nounwind {
; FS: .section .text.plain_1,"ax",@progbits
; CHECK-LABEL: plain_1:
; CHECK: call t0, OUTLINED_FUNCTION_1
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 5
  ret i32 %result
}

define i32 @plain_2(i32 %a, i32 %b) noinline nounwind {
; FS: .section .text.plain_2,"ax",@progbits
; CHECK-LABEL: plain_2:
; CHECK: call t0, OUTLINED_FUNCTION_1
  %a0 = xor i32 %a, %b
  %b0 = add i32 %b, %a0
  %a1 = shl i32 %a0, 3
  %a2 = lshr i32 %a0, 29
  %a3 = or i32 %a1, %a2
  %a4 = xor i32 %a3, %b0
  %result = add i32 %a4, 6
  ret i32 %result
}

; The outlined function for the explicitly sectioned parents inherits their
; input section, while the one for the unsectioned parents is placed in .text,
; or in its own synthesized section when --function-sections is enabled.
; CHECK: .section .sec_shared,"ax",@progbits
; CHECK: OUTLINED_FUNCTION_0:
; CHECK: xor a0, a0, a1
; CHECK-NEXT: srli a2, a0, 29
; CHECK-NEXT: slli a3, a0, 3
; CHECK-NEXT: add a0, a1, a0
; CHECK-NEXT: or a2, a3, a2
; CHECK-NEXT: xor a0, a2, a0
; CHECK-NEXT: jr t0
; NOFS: .text{{$}}
; NOFS: OUTLINED_FUNCTION_1:
; FS: .section .text.OUTLINED_FUNCTION_1,"ax",@progbits
; FS: OUTLINED_FUNCTION_1:
