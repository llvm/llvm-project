; RUN: llc -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -filetype=asm -o - -mtriple=x86_64-darwin-unknown    < %s | FileCheck %s

define i32 @callee() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_0:
; CHECK-NEXT:  .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
; CHECK-LABEL: Ltmp0:
  ret i32 0
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_1:
; CHECK-NEXT:  retq
; CHECK-NEXT:  nopw %cs:512(%rax,%rax)
}
; CHECK:       .p2align 4, 0x90
; CHECK-NEXT:  .quad {{.*}}xray_fn_idx_synth_0{{.*}}
; CHECK-NEXT:  .section {{.*}}xray_instr_map
; CHECK-LABEL: Lxray_sleds_start0:
; CHECK:       .quad {{.*}}xray_sled_0
; CHECK:       .quad {{.*}}xray_sled_1
; CHECK-LABEL: Lxray_sleds_end0:
; CHECK-NEXT:  .section {{.*}}xray_fn_idx
; CHECK-LABEL: Lxray_fn_idx_synth_0:
; CHECK:       .quad {{.*}}xray_sleds_start0
; CHECK-NEXT:  .quad {{.*}}xray_sleds_end0

define i32 @caller() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_2:
; CHECK-NEXT:  .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
; CHECK-LABEL: Ltmp1:
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_3:
; CHECK-NEXT:  .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
; CHECK-LABEL: Ltmp2:
  %retval = tail call i32 @callee()
; CHECK:       jmp {{.*}}callee {{.*}}# TAILCALL
  ret i32 %retval
}
; CHECK:       .p2align 4, 0x90
; CHECK-NEXT:  .quad {{.*}}xray_fn_idx_synth_1{{.*}}
; CHECK-LABEL: Lxray_sleds_start1:
; CHECK:       .quad {{.*}}xray_sled_2
; CHECK:       .quad {{.*}}xray_sled_3
; CHECK-LABEL: Lxray_sleds_end1:
; CHECK:       .section {{.*}}xray_fn_idx
; CHECK-LABEL: Lxray_fn_idx_synth_1:
; CHECK:       .quad {{.*}}xray_sleds_start1
; CHECK:       .quad {{.*}}xray_sleds_end1
