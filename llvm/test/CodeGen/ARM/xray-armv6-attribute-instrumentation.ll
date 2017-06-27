; RUN: llc -filetype=asm -o - -mtriple=armv6-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -filetype=asm -o - -mtriple=armv6-apple-ios6.0.0  < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  b  #20
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-LABEL: Ltmp0:
  ret i32 0
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  b  #20
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-NEXT:  mov	r0, r0
; CHECK-LABEL: Ltmp1:
; CHECK-NEXT:  bx	lr
}
; CHECK:       .p2align 4
; CHECK-NEXT:  .long {{.*}}Lxray_fn_idx_synth_0
; CHECK-NEXT:  .section {{.*}}xray_instr_map{{.*}}
; CHECK-LABEL: Lxray_sleds_start0:
; CHECK:       .long {{.*}}Lxray_sled_0
; CHECK:       .long {{.*}}Lxray_sled_1
; CHECK-LABEL: Lxray_sleds_end0:
; CHECK:       .section {{.*}}xray_fn_idx{{.*}}
; CHECK-LABEL: Lxray_fn_idx_synth_0:
; CHECK:       .long {{.*}}Lxray_sleds_start0
; CHECK-NEXT:  .long {{.*}}Lxray_sleds_end0
