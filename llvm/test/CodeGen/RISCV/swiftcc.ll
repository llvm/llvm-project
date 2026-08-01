; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s | FileCheck %s

; swiftcc is lowered like the C convention on RISC-V. Check that it is
; accepted at all: LowerFormalArguments used to reject it with
; "Unsupported calling convention".

define swiftcc i32 @swiftcc_param(i32 %a, i32 %b) {
; CHECK-LABEL: swiftcc_param:
; CHECK: ret
  %r = add i32 %a, %b
  ret i32 %r
}

define swiftcc i32 @call_swiftcc(i32 %a, i32 %b) {
; CHECK-LABEL: call_swiftcc:
; CHECK: call swiftcc_param
  %r = call swiftcc i32 @swiftcc_param(i32 %a, i32 %b)
  ret i32 %r
}

define swiftcc ptr @swiftself_param(ptr swiftself %addr) {
; CHECK-LABEL: swiftself_param:
; CHECK: ret
  ret ptr %addr
}
