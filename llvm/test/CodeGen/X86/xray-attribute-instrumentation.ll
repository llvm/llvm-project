; RUN: llc -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -filetype=asm -o - -mtriple=x86_64-darwin-unknown    < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
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
; CHECK-NEXT:  .quad {{.*}}xray_synthetic_0
; CHECK-NEXT:  .section {{.*}}xray_instr_map
; CHECK-LABEL: Lxray_synthetic_0:
; CHECK:       .quad {{.*}}xray_sled_0
; CHECK:       .quad {{.*}}xray_sled_1

; We test multiple returns in a single function to make sure we're getting all
; of them with XRay instrumentation.
define i32 @bar(i32 %i) nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_2:
; CHECK-NEXT:  .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
; CHECK-LABEL: Ltmp1:
Test:
  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %IsEqual, label %NotEqual
IsEqual:
  ret i32 0
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_3:
; CHECK-NEXT:  retq
; CHECK-NEXT:  nopw %cs:512(%rax,%rax)
NotEqual:
  ret i32 1
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_4:
; CHECK-NEXT:  retq
; CHECK-NEXT:  nopw %cs:512(%rax,%rax)
}
; CHECK:       .p2align 4, 0x90
; CHECK-NEXT:  .quad {{.*}}xray_synthetic_1
; CHECK-NEXT:  .section {{.*}}xray_instr_map
; CHECK-LABEL: Lxray_synthetic_1:
; CHECK:       .quad {{.*}}xray_sled_2
; CHECK:       .quad {{.*}}xray_sled_3
; CHECK:       .quad {{.*}}xray_sled_4
