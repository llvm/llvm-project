; RUN: llc -mtriple=aarch64-unknown-linux-gnu -relocation-model=pic -verify-machineinstrs %s -o - | FileCheck %s

; TLSDESC resolver calling convention does not retain the flags register.
; Check that a TLS descriptor call cannot be lowered in between a cmp and the use of flags.

@var = thread_local global i32 zeroinitializer
@test = global i32 zeroinitializer

define i32 @test_thread_local() {
; CHECK-LABEL: test_thread_local:
; CHECK:       // %bb.0:
; CHECK-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset w30, -16
; CHECK-NEXT:    adrp x8, :got:test
; CHECK-NEXT:    ldr x8, [x8, :got_lo12:test]
; CHECK-NEXT:    adrp x0, :tlsdesc:var
; CHECK-NEXT:    ldr x1, [x0, :tlsdesc_lo12:var]
; CHECK-NEXT:    add x0, x0, :tlsdesc_lo12:var
; CHECK-NEXT:    .tlsdesccall var
; CHECK-NEXT:    blr x1
; CHECK-NEXT:    mrs x9, TPIDR_EL0
; CHECK-NEXT:    ldr w9, [x9, x0]
; CHECK-NEXT:    cmp x8, #0
; CHECK-NEXT:    cinc w0, w9, eq
; CHECK-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; CHECK-NEXT:    ret

  %testval = load i32, ptr @test
  %test = icmp eq ptr @test, null
  %val = load i32, ptr @var
  %result = zext i1 %test to i32
  %result2 = add i32 %val, %result
  ret i32 %result2

}
