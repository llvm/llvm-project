; RUN: llc -mtriple=riscv32-unknown-linux-gnu -mattr=+d,+c < %s | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=riscv64-unknown-linux-gnu -mattr=+d,+c < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-RISCV64 %s

define i32 @foo() nounwind "function-instrument"="xray-always" {
; CHECK:                        .p2align 2
; CHECK-LABEL:                  .Lxray_sled_0:
; CHECK-NEXT:                   j .Ltmp0
; CHECK-COUNT-21:               nop
; CHECK-RISCV64-COUNT-12:       nop
; CHECK-LABEL:                  .Ltmp0:
  ret i32 0
; CHECK:                        .p2align 2
; CHECK-LABEL:                  .Lxray_sled_1:
; CHECK-NEXT:                   j .Ltmp1
; CHECK-COUNT-21:               nop
; CHECK-RISCV64-COUNT-12:       nop
; CHECK-LABEL:                  .Ltmp1:
; CHECK-NEXT:                   ret
}
; CHECK:                        .section xray_instr_map,"ao",@progbits,foo
; CHECK-LABEL:                  .Lxray_sleds_start0:
; CHECK:                        .Lxray_sled_0-[[TMP:.Ltmp[0-9]+]]
; CHECK:                        .Lxray_sled_1-[[TMP:.Ltmp[0-9]+]]
; CHECK-LABEL:                  .Lxray_sleds_end0:
