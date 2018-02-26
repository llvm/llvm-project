; RUN: llc < %s -O0 -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s

define i64 @bitcasti64tof64() {
; CHECK-LABEL: bitcasti64tof64:
; CHECK:       # %bb.0:
; CHECK:         vmovsd {{.*#+}} xmm0 = mem[0],zero
; CHECK-NEXT:    vmovq %xmm0, %rax
; CHECK-NEXT:    retq
  %a = load double, double* undef
  %b = bitcast double %a to i64
  ret i64 %b
}

