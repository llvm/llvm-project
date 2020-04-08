; RUN: llc -simplify-mir -verify-machineinstrs -stop-after=finalize-isel \
; RUN:   -mtriple=powerpc64le-unknown-unknown -mattr=-vsx < %s | FileCheck %s

define float @test(float %a) {
; CHECK:        stack:
; CHECK-NEXT:   - { id: 0, size: 4, alignment: 4 }
; CHECK:        %2:f8rc = FCTIWZ killed %1, implicit $rm
; CHECK:        STFIWX killed %2, $zero8, %3
; CHECK-NEXT:   %4:f8rc = LFIWAX $zero8, %3 :: (load 4 from %stack.0, align 1)
entry:
  %b = fptosi float %a to i32
  %c = sitofp i32 %b to float
  ret float %c
}
