; RUN: llc -simplify-mir -verify-machineinstrs -stop-after=finalize-isel \
; RUN:   -mtriple=powerpc64le-unknown-unknown -mattr=-vsx < %s | FileCheck %s
; RUN: llc -simplify-mir -verify-machineinstrs -stop-after=finalize-isel \
; RUN:   -mtriple=powerpc-unknown-unknown -mcpu=pwr6 -mattr=-vsx < %s | \
; RUN:   FileCheck --check-prefix=CHECK-P6 %s
; RUN: llc -simplify-mir -verify-machineinstrs -stop-after=finalize-isel \
; RUN:   -mtriple=powerpc64-unknown-unknown -mcpu=pwr6 -mattr=-vsx < %s | \
; RUN:   FileCheck --check-prefix=CHECK-P6-64 %s

define float @test(float %a) {
; CHECK:        stack:
; CHECK-NEXT:   - { id: 0, size: 4, alignment: 4 }
; CHECK:        %2:f8rc = nofpexcept FCTIWZ killed %1, implicit $rm
; CHECK:        STFIWX killed %2, $zero8, %3
; CHECK-NEXT:   %4:f8rc = LFIWAX $zero8, %3 :: (load 4 from %stack.0)
; CHECK-NEXT:   %5:f4rc = nofpexcept FCFIDS killed %4, implicit $rm
; CHECK-NEXT:   $f1 = COPY %5
; CHECK-NEXT:   BLR8 implicit $lr8, implicit $rm, implicit $f1

; CHECK-P6:        stack:
; CHECK-P6-NEXT:   - { id: 0, size: 4, alignment: 4 }
; CHECK-P6:        %2:f8rc = nofpexcept FCTIWZ killed %1, implicit $rm
; CHECK-P6:        STFIWX killed %2, $zero, %3
; CHECK-P6-NEXT:   %4:f8rc = LFIWAX $zero, %3 :: (load 4 from %stack.0)
; CHECK-P6-NEXT:   %5:f8rc = nofpexcept FCFID killed %4, implicit $rm
; CHECK-P6-NEXT:   %6:f4rc = nofpexcept FRSP killed %5, implicit $rm
; CHECK-P6-NEXT:   $f1 = COPY %6
; CHECK-P6-NEXT:   BLR implicit $lr, implicit $rm, implicit $f1

; CHECK-P6-64:        stack:
; CHECK-P6-64-NEXT:   - { id: 0, size: 4, alignment: 4 }
; CHECK-P6-64:        %2:f8rc = nofpexcept FCTIWZ killed %1, implicit $rm
; CHECK-P6-64:        STFIWX killed %2, $zero8, %3
; CHECK-P6-64-NEXT:   %4:f8rc = LFIWAX $zero8, %3 :: (load 4 from %stack.0)
; CHECK-P6-64-NEXT:   %5:f8rc = nofpexcept FCFID killed %4, implicit $rm
; CHECK-P6-64-NEXT:   %6:f4rc = nofpexcept FRSP killed %5, implicit $rm
; CHECK-P6-64-NEXT:   $f1 = COPY %6
; CHECK-P6-64-NEXT:   BLR8 implicit $lr8, implicit $rm, implicit $f1

entry:
  %b = fptosi float %a to i32
  %c = sitofp i32 %b to float
  ret float %c
}
