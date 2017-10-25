; RUN: llc -mtriple arm-unknown -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefix=CHECK

define arm_aapcscc void @test_indirect_call(void() *%fptr) {
; CHECK-LABEL: name: test_indirect_call
; CHECK: registers:
; CHECK-NEXT: id: [[FPTR:[0-9]+]], class: gpr
; CHECK: %[[FPTR]]:gpr(p0) = COPY %r0
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: BLX %[[FPTR]](p0), csr_aapcs, implicit-def %lr, implicit %sp
; CHECK: ADJCALLSTACKUP 0, 0, 14, _, implicit-def %sp, implicit %sp
entry:
  notail call arm_aapcscc void %fptr()
  ret void
}

declare arm_aapcscc void @call_target()

define arm_aapcscc void @test_direct_call() {
; CHECK-LABEL: name: test_direct_call
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, _, implicit-def %sp, implicit %sp
; CHECK: BLX @call_target, csr_aapcs, implicit-def %lr, implicit %sp
; CHECK: ADJCALLSTACKUP 0, 0, 14, _, implicit-def %sp, implicit %sp
entry:
  notail call arm_aapcscc void @call_target()
  ret void
}
