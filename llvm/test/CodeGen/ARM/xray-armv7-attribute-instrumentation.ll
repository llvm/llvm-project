; RUN: llc -filetype=asm -o - -mtriple=armv7-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -filetype=asm -o - -mtriple=armv7-apple-ios6.0.0  < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b  #20
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp0:
  ret i32 0
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b  #20
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: Ltmp1:
; CHECK-NEXT:  bx lr
}
; CHECK:       .p2align 4
; CHECK-NEXT:  .long {{.*}}Lxray_synthetic_0
; CHECK-NEXT:  .long {{.*}}Lxray_fn_idx_synth_0
; CHECK-NEXT:  .section {{.*}}xray_instr_map{{.*}}
; CHECK-LABEL: Lxray_synthetic_0:
; CHECK:       .long {{.*}}Lxray_sled_0
; CHECK:       .long {{.*}}Lxray_sled_1
; CHECK-LABEL: Lxray_synthetic_end0:
; CHECK:       .section {{.*}}xray_fn_idx{{.*}}
; CHECK-LABEL: Lxray_fn_idx_synth_0:
; CHECK:       .long {{.*}}xray_synthetic_0
; CHECK-NEXT:  .long {{.*}}xray_synthetic_end0

