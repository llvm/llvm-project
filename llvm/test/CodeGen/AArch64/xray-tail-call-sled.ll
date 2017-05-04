; RUN: llc -filetype=asm -o - -mtriple=aarch64-linux-gnu < %s | FileCheck %s

define i32 @callee() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align	2
; CHECK-LABEL: .Lxray_sled_0:
; CHECK-NEXT:  b	#32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: .Ltmp0:
  ret i32 0
; CHECK-NEXT:  mov	w0, wzr
; CHECK-NEXT:  .p2align	2
; CHECK-LABEL: .Lxray_sled_1:
; CHECK-NEXT:  b	#32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: .Ltmp1:
; CHECK-NEXT:  ret
}
; CHECK:       .p2align 4
; CHECK-NEXT:  .xword .Lxray_synthetic_0
; CHECK-NEXT:  .xword .Lxray_fn_idx_synth_0
; CHECK-NEXT:  .section xray_instr_map,{{.*}}
; CHECK-LABEL: Lxray_synthetic_0:
; CHECK:       .xword .Lxray_sled_0
; CHECK:       .xword .Lxray_sled_1
; CHECK-LABEL: Lxray_synthetic_end0:
; CHECK:       .section xray_fn_idx,{{.*}}
; CHECK-LABEL: Lxray_fn_idx_synth_0:
; CHECK:       .xword .Lxray_synthetic_0
; CHECK-NEXT:  .xword .Lxray_synthetic_end0

define i32 @caller() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align	2
; CHECK-LABEL: .Lxray_sled_2:
; CHECK-NEXT:  b	#32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: .Ltmp2:
; CHECK:       .p2align	2
; CHECK-LABEL: .Lxray_sled_3:
; CHECK-NEXT:  b	#32
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-LABEL: .Ltmp3:
  %retval = tail call i32 @callee()
; CHECK:       b	callee
  ret i32 %retval
}
; CHECK:       .p2align 4
; CHECK-NEXT:  .xword .Lxray_synthetic_1
; CHECK-NEXT:  .xword .Lxray_fn_idx_synth_1
; CHECK-NEXT:  .section xray_instr_map,{{.*}}
; CHECK-LABEL: Lxray_synthetic_1:
; CHECK:       .xword .Lxray_sled_2
; CHECK:       .xword .Lxray_sled_3
; CHECK-LABEL: Lxray_synthetic_end1:
; CHECK:       .section xray_fn_idx,{{.*}}
; CHECK-LABEL: Lxray_fn_idx_synth_1:
; CHECK:       .xword .Lxray_synthetic_1
; CHECK-NEXT:  .xword .Lxray_synthetic_end1
