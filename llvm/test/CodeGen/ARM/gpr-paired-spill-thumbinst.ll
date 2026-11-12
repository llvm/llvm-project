; REQUIRES: asserts
; RUN: llc -mtriple=thumbv7-none-linux-gnueabi -debug -o /dev/null < %s 2>&1 | FileCheck %s

; This test makes sure spills of 64-bit pairs in Thumb mode actually
; generate thumb instructions. Previously we were inserting an ARM
; STMIA which happened to have the same encoding.

define void @foo(ptr %addr) {
  %val1 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(ptr %addr)
  %val2 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(ptr %addr)
  %val3 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(ptr %addr)
  %val4 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(ptr %addr)
  %val5 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(ptr %addr)
  %val6 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(ptr %addr)
  %val7 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [r0]", "=&r,r"(ptr %addr)

  ; Make sure we are actually creating the Thumb versions of the spill
  ; instructions.
; CHECK: t2STRDi8
; CHECK: t2LDRDi8

  store volatile i64 %val1, ptr %addr
  store volatile i64 %val2, ptr %addr
  store volatile i64 %val3, ptr %addr
  store volatile i64 %val4, ptr %addr
  store volatile i64 %val5, ptr %addr
  store volatile i64 %val6, ptr %addr
  store volatile i64 %val7, ptr %addr
  ret void
}
