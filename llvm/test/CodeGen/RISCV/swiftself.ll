; RUN: llc -mtriple=riscv32 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SDAG
; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SDAG
; RUN: llc -mtriple=riscv64 -global-isel -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; Parameter with swiftself should be allocated to x20 (s4).
; CHECK-LABEL: swiftself_param:
; CHECK: mv a0, s4
; CHECK-NEXT: ret
define ptr @swiftself_param(ptr swiftself %addr0) {
  ret ptr %addr0
}

; Check that x20 is used to pass a swiftself argument.
; CHECK-LABEL: call_swiftself:
; CHECK: mv s4, a0
; CHECK: call swiftself_param
; CHECK: ret
define ptr @call_swiftself(ptr %arg) {
  %res = call ptr @swiftself_param(ptr swiftself %arg)
  ret ptr %res
}

; x20 should be saved by the callee even if used for swiftself.
; CHECK-LABEL: swiftself_clobber:
; CHECK: {{sw|sd}} s4, {{[0-9]+}}(sp)
; ...
; CHECK: {{lw|ld}} s4, {{[0-9]+}}(sp)
; CHECK: ret
define ptr @swiftself_clobber(ptr swiftself %addr0) {
  call void asm sideeffect "", "~{x20}"()
  ret ptr %addr0
}

; Demonstrate that we do not need any moves when calling multiple functions
; with the same swiftself argument.
; CHECK-LABEL: swiftself_passthrough:
; CHECK-NOT: mv s4,
; CHECK: call swiftself_param
; CHECK-NOT: mv s4,
; CHECK-NEXT: call swiftself_param
; CHECK: ret
define void @swiftself_passthrough(ptr swiftself %addr0) {
  call ptr @swiftself_param(ptr swiftself %addr0)
  call ptr @swiftself_param(ptr swiftself %addr0)
  ret void
}

; We can use a tail call if the callee swiftself is the same as the caller
; one. GlobalISel does not implement tail calls on RISC-V yet and emits a
; normal call.
; CHECK-LABEL: swiftself_tail:
; SDAG: tail swiftself_param
; SDAG-NOT: ret
define ptr @swiftself_tail(ptr swiftself %addr0) {
  call void asm sideeffect "", "~{x20}"()
  %res = tail call ptr @swiftself_param(ptr swiftself %addr0)
  ret ptr %res
}

; We can not use a tail call if the callee swiftself is not the same as the
; caller one: the epilogue restores s4 before the tail-call jump would
; clobber the outgoing value.
; CHECK-LABEL: swiftself_notail:
; CHECK: mv s4, a0
; CHECK: call swiftself_param
; CHECK: ret
define ptr @swiftself_notail(ptr swiftself %addr0, ptr %addr1) nounwind {
  %res = tail call ptr @swiftself_param(ptr swiftself %addr1)
  ret ptr %res
}

; swiftself does not steal any of the normal argument registers a0-a7.
; CHECK-LABEL: swiftself_all_argregs:
; CHECK: mv a0, s4
; CHECK: ret
define ptr @swiftself_all_argregs(i32 %a0, i32 %a1, i32 %a2, i32 %a3,
                                  i32 %a4, i32 %a5, i32 %a6, i32 %a7,
                                  ptr swiftself %addr0) {
  ret ptr %addr0
}
