; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 -debug-only=machine-scheduler \
; RUN:   2>&1 | FileCheck %s --check-prefix=DBG
; REQUIRES: asserts

; The many stores should not all end up at the bottom, or spilling would result.
; DBG:      ********** MI Scheduling **********
; DBG-NEXT: f1:%bb.0
; DBG:      *** Final schedule for %bb.0 ***
; DBG-NEXT: SU(0):   %1:addr64bit = COPY $r3d
; DBG-NEXT: SU(1):   %0:addr64bit = COPY $r2d
; DBG-NEXT: SU(2):   %33:fp32bit = LE %0:addr64bit, 60, $noreg
; DBG-NEXT: SU(33):  %33:fp32bit = nofpexcept AEBR %33:fp32bit(tied-def 0), %33:fp32bit,
; DBG-NEXT: SU(34):  STE %33:fp32bit, %1:addr64bit, 60, $noreg
; DBG-NEXT: SU(3):   %32:fp32bit = LE %0:addr64bit, 56, $noreg
; DBG-NEXT: SU(32):  %32:fp32bit = nofpexcept AEBR %32:fp32bit(tied-def 0), %32:fp32bit
; DBG-NEXT: SU(35):  STE %32:fp32bit, %1:addr64bit, 56, $noreg
; ...
; DBG:      SU(17):  %18:fp32bit = LE %0:addr64bit, 0, $noreg
; DBG-NEXT: SU(18):  %18:fp32bit = nofpexcept AEBR %18:fp32bit(tied-def 0), %18:fp32bit,
; DBG-NEXT: SU(49):  STE %18:fp32bit, %1:addr64bit, 0, $noreg

define void @f1(ptr noalias %src1, ptr noalias %dest) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r15
; CHECK: br %r14
  %val = load <16 x float>, ptr %src1
  %add = fadd <16 x float> %val, %val
  store <16 x float> %add, ptr %dest
  ret void
}
