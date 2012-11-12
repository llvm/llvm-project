; RUN: llc < %s -O0 -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; CHECK: vmovsd (%
; CHECK-NEXT: vmovd %xmm
define i64 @bitcasti64tof64() {
  %a = load double* undef
  %b = bitcast double %a to i64
  ret i64 %b
}

