; RUN: llc -mtriple=thumbv7 -trap-unreachable < %s | FileCheck %s --check-prefixes CHECK,TRAP_UNREACHABLE
; RUN: llc -mtriple=thumbv7 -trap-unreachable -no-trap-after-noreturn < %s | FileCheck %s --check-prefixes CHECK,NTANR

define void @test_trap_unreachable() #0 {
; CHECK-LABEL: test_trap_unreachable:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .inst.n 0xdefe
  unreachable
}

attributes #0 = { nounwind }

declare void @no_return() noreturn
declare void @could_return()

define void @test_ntanr_noreturn() {
; CHECK-LABEL:           test_ntanr_noreturn:
; CHECK:                 @ %bb.0:
; CHECK-NEXT:              push {r7, lr}
; CHECK-NEXT:              bl no_return
; TRAP_UNREACHABLE-NEXT:   .inst.n 0xdefe
; NTANR-NOT:               .inst.n 0xdefe
;
  call void @no_return()
  unreachable
}

define void @test_ntanr_could_return() {
; CHECK-LABEL: test_ntanr_could_return:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    bl could_return
; CHECK-NEXT:    .inst.n 0xdefe
  call void @could_return()
  unreachable
}
