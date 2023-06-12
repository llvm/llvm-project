; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=x86_64-darwin-unknown    < %s | FileCheck %s --check-prefixes=CHECK,CHECK-MACOS

define dso_local i32 @callee() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_0:
; CHECK:       .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
  ret i32 0
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_1:
; CHECK:       retq
; CHECK-NEXT:  nopw %cs:512(%rax,%rax)
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,callee{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start0:
; CHECK-LINUX:         .quad .Lxray_sled_0
; CHECK-LINUX:         .quad .Lxray_sled_1
; CHECK-LINUX-LABEL: .Lxray_sleds_end0:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"awo",@progbits,callee{{$}}
; CHECK-LINUX:         .quad .Lxray_sleds_start0
; CHECK-LINUX-NEXT:    .quad .Lxray_sleds_end0

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map{{$}}
; CHECK-MACOS-LABEL: Lxray_sleds_start0:
; CHECK-MACOS:         .quad Lxray_sled_0
; CHECK-MACOS:         .quad Lxray_sled_1
; CHECK-MACOS-LABEL: Lxray_sleds_end0:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx{{$}}
; CHECK-MACOS:         .quad Lxray_sleds_start0
; CHECK-MACOS-NEXT:    .quad Lxray_sleds_end0

define dso_local i32 @caller() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_2:
; CHECK:       .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_3:
; CHECK-NEXT:  .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
  %retval = tail call i32 @callee()
; CHECK:       jmp {{.*}}callee {{.*}}# TAILCALL
  ret i32 %retval
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,caller{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start1:
; CHECK-LINUX:         .quad .Lxray_sled_2
; CHECK-LINUX:         .quad .Lxray_sled_3
; CHECK-LINUX-LABEL: .Lxray_sleds_end1:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"awo",@progbits,caller{{$}}
; CHECK-LINUX:         .quad .Lxray_sleds_start1
; CHECK-LINUX:         .quad .Lxray_sleds_end1

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map{{$}}
; CHECK-MACOS-LABEL: Lxray_sleds_start1:
; CHECK-MACOS:         .quad Lxray_sled_2
; CHECK-MACOS:         .quad Lxray_sled_3
; CHECK-MACOS-LABEL: Lxray_sleds_end1:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx{{$}}
; CHECK-MACOS:         .quad Lxray_sleds_start1
; CHECK-MACOS:         .quad Lxray_sleds_end1
