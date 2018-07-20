; RUN: llc -mtriple arm-unknown -mattr=-v4t -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefixes=CHECK,NOV4T
; RUN: llc -mtriple arm-unknown -mattr=+v4t -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefixes=CHECK,V4T
; RUN: llc -mtriple arm-unknown -mattr=+v5t -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefixes=CHECK,V5T

define arm_aapcscc void @test_indirect_call(void() *%fptr) {
; CHECK-LABEL: name: test_indirect_call
; V5T: %[[FPTR:[0-9]+]]:gpr(p0) = COPY $r0
; V4T: %[[FPTR:[0-9]+]]:tgpr(p0) = COPY $r0
; NOV4T: %[[FPTR:[0-9]+]]:tgpr(p0) = COPY $r0
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, $noreg, implicit-def $sp, implicit $sp
; V5T: BLX %[[FPTR]](p0), csr_aapcs, implicit-def $lr, implicit $sp
; V4T: BX_CALL %[[FPTR]](p0), csr_aapcs, implicit-def $lr, implicit $sp
; NOV4T: BMOVPCRX_CALL %[[FPTR]](p0), csr_aapcs, implicit-def $lr, implicit $sp
; CHECK: ADJCALLSTACKUP 0, 0, 14, $noreg, implicit-def $sp, implicit $sp
entry:
  notail call arm_aapcscc void %fptr()
  ret void
}

declare arm_aapcscc void @call_target()

define arm_aapcscc void @test_direct_call() {
; CHECK-LABEL: name: test_direct_call
; CHECK: ADJCALLSTACKDOWN 0, 0, 14, $noreg, implicit-def $sp, implicit $sp
; CHECK: BL @call_target, csr_aapcs, implicit-def $lr, implicit $sp
; CHECK: ADJCALLSTACKUP 0, 0, 14, $noreg, implicit-def $sp, implicit $sp
entry:
  notail call arm_aapcscc void @call_target()
  ret void
}
