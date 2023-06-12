; RUN: llc -mtriple=x86_64-unknown-linux-gnu                       < %s | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=x86_64-darwin-unknown                          < %s | FileCheck %s --check-prefixes=CHECK,CHECK-MACOS

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
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

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,foo{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start0:
; CHECK-LINUX:         .quad .Lxray_sled_0
; CHECK-LINUX:         .quad .Lxray_sled_1
; CHECK-LINUX-LABEL: .Lxray_sleds_end0:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"awo",@progbits,foo{{$}}
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


; We test multiple returns in a single function to make sure we're getting all
; of them with XRay instrumentation.
define i32 @bar(i32 %i) nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_2:
; CHECK:       .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
Test:
  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %IsEqual, label %NotEqual
IsEqual:
  ret i32 0
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_3:
; CHECK:       retq
; CHECK-NEXT:  nopw %cs:512(%rax,%rax)
NotEqual:
  ret i32 1
; CHECK:       .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_4:
; CHECK:       retq
; CHECK-NEXT:  nopw %cs:512(%rax,%rax)
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,bar{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start1:
; CHECK-LINUX:       .Ltmp2:
; CHECK-LINUX-NEXT:    .quad .Lxray_sled_2-.Ltmp2
; CHECK-LINUX:       .Ltmp3:
; CHECK-LINUX-NEXT:    .quad .Lxray_sled_3-.Ltmp3
; CHECK-LINUX:       .Ltmp4:
; CHECK-LINUX-NEXT:    .quad .Lxray_sled_4-.Ltmp4
; CHECK-LINUX-LABEL: .Lxray_sleds_end1:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"awo",@progbits,bar{{$}}
; CHECK-LINUX:         .quad .Lxray_sleds_start1
; CHECK-LINUX-NEXT:    .quad .Lxray_sleds_end1

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map{{$}}
; CHECK-MACOS-LABEL: Lxray_sleds_start1:
; CHECK-MACOS:       Ltmp2:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_2-Ltmp2
; CHECK-MACOS:       Ltmp3:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_3-Ltmp3
; CHECK-MACOS:       Ltmp4:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_4-Ltmp4
; CHECK-MACOS-LABEL: Lxray_sleds_end1:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx{{$}}
; CHECK-MACOS:         .quad Lxray_sleds_start1
; CHECK-MACOS-NEXT:    .quad Lxray_sleds_end1
