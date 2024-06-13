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
; CHECK-LINUX-LABEL: .section xray_fn_idx,"ao",@progbits,callee{{$}}
; CHECK-LINUX:       [[IDX:\.Lxray_fn_idx[0-9]+]]:
; CHECK-LINUX-NEXT:    .quad .Lxray_sleds_start0-[[IDX]]
; CHECK-LINUX-NEXT:    .quad 2

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start0:
; CHECK-MACOS:         .quad Lxray_sled_0
; CHECK-MACOS:         .quad Lxray_sled_1
; CHECK-MACOS-LABEL: Lxray_sleds_end0:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx,regular,live_support{{$}}
; CHECK-MACOS:       [[IDX:lxray_fn_idx[0-9]+]]:
; CHECK-MACOS-NEXT:    .quad lxray_sleds_start0-[[IDX]]
; CHECK-MACOS-NEXT:    .quad 2

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
; CHECK-LINUX-LABEL: .section xray_fn_idx,"ao",@progbits,caller{{$}}
; CHECK-LINUX:       [[IDX:\.Lxray_fn_idx[0-9]+]]:
; CHECK-LINUX-NEXT:    .quad .Lxray_sleds_start1-[[IDX]]
; CHECK-LINUX-NEXT:    .quad 2

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start1:
; CHECK-MACOS:         .quad Lxray_sled_2
; CHECK-MACOS:         .quad Lxray_sled_3
; CHECK-MACOS-LABEL: Lxray_sleds_end1:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx,regular,live_support{{$}}
; CHECK-MACOS:       [[IDX:lxray_fn_idx[0-9]+]]:
; CHECK-MACOS-NEXT:    .quad lxray_sleds_start1-[[IDX]]
; CHECK-MACOS-NEXT:    .quad 2

define dso_local i32 @conditional_tail_call(i32 %cond) nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK-LABEL: conditional_tail_call:
; CHECK:         .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_4:
; CHECK:         .ascii "\353\t"
; CHECK-NEXT:    nopw 512(%rax,%rax)
; CHECK-NEXT:    testl %edi, %edi
; CHECK-NEXT:    je {{\.?Ltmp5}}
; CHECK:         .p2align 1, 0x90
; CHECK-LABEL: Lxray_sled_5:
; CHECK-NEXT:    .ascii "\353\t"
; CHECK-NEXT:    nopw 512(%rax,%rax)
; CHECK-LABEL: Ltmp6:
; CHECK-NEXT:    jmp {{.*}}callee {{.*}}# TAILCALL
; CHECK-LABEL: Ltmp5:
; CHECK:         xorl %eax, %eax
; CHECK-NEXT:   .p2align  1, 0x90
; CHECK-LABEL: Lxray_sled_6:
; CHECK-NEXT:    retq
; CHECK-NEXT:    nopw %cs:512(%rax,%rax)
  %cmp = icmp ne i32 %cond, 0
  br i1 %cmp, label %docall, label %ret
docall:
  %retval = tail call i32 @callee()
  ret i32 %retval
ret:
  ret i32 0
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,conditional_tail_call{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start2:
; CHECK-LINUX:         .quad .Lxray_sled_4
; CHECK-LINUX:         .quad .Lxray_sled_5
; CHECK-LINUX:         .quad .Lxray_sled_6
; CHECK-LINUX-LABEL: .Lxray_sleds_end2:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"ao",@progbits,conditional_tail_call{{$}}
; CHECK-LINUX:       [[IDX:\.Lxray_fn_idx[0-9]+]]:
; CHECK-LINUX-NEXT:    .quad .Lxray_sleds_start2-[[IDX]]
; CHECK-LINUX-NEXT:    .quad 3

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start2:
; CHECK-MACOS:         .quad Lxray_sled_4
; CHECK-MACOS:         .quad Lxray_sled_5
; CHECK-MACOS:         .quad Lxray_sled_6
; CHECK-MACOS-LABEL: Lxray_sleds_end2:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx,regular,live_support{{$}}
; CHECK-MACOS:       [[IDX:lxray_fn_idx[0-9]+]]:
; CHECK-MACOS-NEXT:    .quad lxray_sleds_start2-[[IDX]]
; CHECK-MACOS-NEXT:    .quad 3
