; RUN: llc -verify-machineinstrs -filetype=asm -o - -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -filetype=asm -o - \
; RUN:     -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -filetype=asm -o - -mtriple=x86_64-darwin-unknown    < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" "xray-skip-exit" {
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_0:
; CHECK:       .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
  ret i32 0
; CHECK-NOT: Lxray_sled_1:
; CHECK:       retq
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start0:
; CHECK:       .quad {{.*}}xray_sled_0
; CHECK-LABEL: Lxray_sleds_end0:
; CHECK-LABEL: xray_fn_idx
; CHECK:       .quad {{.*}}xray_sleds_start0
; CHECK-NEXT:  .quad {{.*}}xray_sleds_end0


; We test multiple returns in a single function to make sure we're skipping all
; of them with XRay instrumentation.
define i32 @bar(i32 %i) nounwind noinline uwtable "function-instrument"="xray-always" "xray-skip-exit" {
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_1:
; CHECK:       .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
Test:
  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %IsEqual, label %NotEqual
IsEqual:
  ret i32 0
; CHECK-NOT: Lxray_sled_{{.*}}:
; CHECK:       retq
NotEqual:
  ret i32 1
; CHECK-NOT: Lxray_sled_{{.*}}:
; CHECK:       retq
}
; CHECK-LABEL: xray_instr_map
; CHECK-LABEL: Lxray_sleds_start1:
; CHECK:       .quad {{.*}}xray_sled_1
; CHECK-LABEL: Lxray_sleds_end1:
; CHECK-LABEL: xray_fn_idx
; CHECK:       .quad {{.*}}xray_sleds_start1
; CHECK-NEXT:  .quad {{.*}}xray_sleds_end1
