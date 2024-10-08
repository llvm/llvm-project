; RUN: llc -mtriple=aarch64-apple-darwin -stop-after finalize-isel <%s | FileCheck %s

; Check that the callee doesn't have calleeSavedRegisters.
define preserve_nonecc i64 @callee1(i64 %a0, i64 %b0, i64 %c0, i64 %d0, i64 %e0) nounwind {
  %a1 = mul i64 %a0, %b0
  %a2 = mul i64 %a1, %c0
  %a3 = mul i64 %a2, %d0
  %a4 = mul i64 %a3, %e0
  ret i64 %a4
}
; CHECK:     name: callee1
; CHECK-NOT: calleeSavedRegisters:
; CHECK:     RET_ReallyLR implicit $x0

; Check that RegMask is csr_aarch64_noneregs.
define i64 @caller1(i64 %a0) nounwind {
  %b1 = call preserve_nonecc i64 @callee1(i64 %a0, i64 %a0, i64 %a0, i64 %a0, i64 %a0)
  %b2 = add i64 %b1, %a0
  ret i64 %b2
}
; CHECK:    name: caller1
; CHECK:    BL @callee1, csr_aarch64_noneregs
; CHECK:    RET_ReallyLR implicit $x0


; Check that the callee doesn't have calleeSavedRegisters.
define preserve_nonecc {i64, i64} @callee2(i64 %a0, i64 %b0, i64 %c0, i64 %d0, i64 %e0) nounwind {
  %a1 = mul i64 %a0, %b0
  %a2 = mul i64 %a1, %c0
  %a3 = mul i64 %a2, %d0
  %a4 = mul i64 %a3, %e0
  %b4 = insertvalue {i64, i64} undef, i64 %a3, 0
  %b5 = insertvalue {i64, i64} %b4, i64 %a4, 1
  ret {i64, i64} %b5
}
; CHECK:     name: callee2
; CHECK-NOT: calleeSavedRegisters:
; CHECK:     RET_ReallyLR implicit $x0


; Check that RegMask is csr_aarch64_noneregs.
define {i64, i64} @caller2(i64 %a0) nounwind {
  %b1 = call preserve_nonecc {i64, i64} @callee2(i64 %a0, i64 %a0, i64 %a0, i64 %a0, i64 %a0)
  ret {i64, i64} %b1
}
; CHECK:    name: caller2
; CHECK:    BL @callee2, csr_aarch64_noneregs
; CHECK:    RET_ReallyLR implicit $x0


%struct.Large = type { i64, double, double }

; Declare the callee with a sret parameter.
declare preserve_nonecc void @callee3(ptr noalias nocapture writeonly sret(%struct.Large) align 4 %a0, i64 %b0) nounwind;

; Check that RegMask is csr_aarch64_noneregs.
define void @caller3(i64 %a0) nounwind {
  %a1 = alloca %struct.Large, align 8
  call preserve_nonecc void @callee3(ptr nonnull sret(%struct.Large) align 8 %a1, i64 %a0)
  ret void
}
; CHECK:    name: caller3
; CHECK:    BL @callee3, csr_aarch64_noneregs
; CHECK:    RET_ReallyLR


; Check that the callee doesn't have calleeSavedRegisters.
define preserve_nonecc {i64, double} @callee4(i64 %a0, i64 %b0, i64 %c0, i64 %d0, i64 %e0) nounwind {
  %a1 = mul i64 %a0, %b0
  %a2 = mul i64 %a1, %c0
  %a3 = mul i64 %a2, %d0
  %a4 = mul i64 %a3, %e0
  %b4 = insertvalue {i64, double} undef, i64 %a3, 0
  %b5 = insertvalue {i64, double} %b4, double 1.2, 1
  ret {i64, double} %b5
}
; CHECK:     name: callee4
; CHECK-NOT: calleeSavedRegisters:
; CHECK:     RET_ReallyLR implicit $x0, implicit $d0

; Check that RegMask is csr_aarch64_noneregs.
define {i64, double} @caller4(i64 %a0) nounwind {
  %b1 = call preserve_nonecc {i64, double} @callee4(i64 %a0, i64 %a0, i64 %a0, i64 %a0, i64 %a0)
  ret {i64, double} %b1
}
; CHECK:    name: caller4
; CHECK:    BL @callee4, csr_aarch64_noneregs
; CHECK:    RET_ReallyLR implicit $x0, implicit $d0
