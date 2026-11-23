; RUN: llc -mtriple=mips64 -target-abi n32 -mattr=+reserve-gpr8,+reserve-gpr26,+reserve-gpr30 \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mips64 -target-abi n64 -mattr=+reserve-gpr8,+reserve-gpr26,+reserve-gpr30 \
; RUN:   -verify-machineinstrs < %s | FileCheck %s

; ABI names and aliases in named-register lowering.
define i64 @get_a4() {
; CHECK-LABEL: get_a4:
; CHECK: move $2, $8
  %value = call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %value
}

define i64 @get_kt0() {
; CHECK-LABEL: get_kt0:
; CHECK: move $2, $26
  %value = call i64 @llvm.read_register.i64(metadata !1)
  ret i64 %value
}

define i64 @get_s8() {
; CHECK-LABEL: get_s8:
; CHECK: move $2, $fp
  %value = call i64 @llvm.read_register.i64(metadata !2)
  ret i64 %value
}

declare i64 @llvm.read_register.i64(metadata)

!llvm.named.register = !{!0, !1, !2}
!0 = !{!"$a4"}
!1 = !{!"$kt0"}
!2 = !{!"$s8"}
