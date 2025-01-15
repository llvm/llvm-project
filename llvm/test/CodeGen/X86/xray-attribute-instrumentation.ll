; RUN: llc -mtriple=x86_64-unknown-linux-gnu                       < %s | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s --check-prefixes=CHECK,CHECK-LINUX
; RUN: llc -mtriple=x86_64-darwin-unknown                          < %s | FileCheck %s --check-prefixes=CHECK,CHECK-MACOS

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 1
; CHECK-LABEL: Lxray_sled_0:
; CHECK:       .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
  ret i32 0
; CHECK:       .p2align 1
; CHECK-LABEL: Lxray_sled_1:
; CHECK:       retq
; CHECK-NEXT:  nopw %cs:512(%rax,%rax)
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,foo{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start0:
; CHECK-LINUX:         .quad .Lxray_sled_0
; CHECK-LINUX:         .quad .Lxray_sled_1
; CHECK-LINUX-LABEL: .Lxray_sleds_end0:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"ao",@progbits,foo{{$}}
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


; We test multiple returns in a single function to make sure we're getting all
; of them with XRay instrumentation.
define i32 @bar(i32 %i) nounwind noinline uwtable "function-instrument"="xray-always" {
; CHECK:       .p2align 1
; CHECK-LABEL: Lxray_sled_2:
; CHECK:       .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
Test:
  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %IsEqual, label %NotEqual
IsEqual:
  ret i32 0
; CHECK:       .p2align 1
; CHECK-LABEL: Lxray_sled_3:
; CHECK:       retq
; CHECK-NEXT:  nopw %cs:512(%rax,%rax)
NotEqual:
  ret i32 1
; CHECK:       .p2align 1
; CHECK-LABEL: Lxray_sled_4:
; CHECK:       retq
; CHECK-NEXT:  nopw %cs:512(%rax,%rax)
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,bar{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start1:
; CHECK-LINUX:       [[TMP:.Ltmp[0-9]+]]:
; CHECK-LINUX-NEXT:    .quad .Lxray_sled_2-[[TMP]]
; CHECK-LINUX:       [[TMP:.Ltmp[0-9]+]]:
; CHECK-LINUX-NEXT:    .quad .Lxray_sled_3-[[TMP]]
; CHECK-LINUX:       [[TMP:.Ltmp[0-9]+]]:
; CHECK-LINUX-NEXT:    .quad .Lxray_sled_4-[[TMP]]
; CHECK-LINUX-LABEL: .Lxray_sleds_end1:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"ao",@progbits,bar{{$}}
; CHECK-LINUX:       [[IDX:\.Lxray_fn_idx[0-9]+]]:
; CHECK-LINUX-NEXT:    .quad .Lxray_sleds_start1-[[IDX]]
; CHECK-LINUX-NEXT:    .quad 3

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start1:
; CHECK-MACOS:       [[TMP:Ltmp[0-9]+]]:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_2-[[TMP]]
; CHECK-MACOS:       [[TMP:Ltmp[0-9]+]]:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_3-[[TMP]]
; CHECK-MACOS:       [[TMP:Ltmp[0-9]+]]:
; CHECK-MACOS-NEXT:    .quad Lxray_sled_4-[[TMP]]
; CHECK-MACOS-LABEL: Lxray_sleds_end1:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx,regular,live_support{{$}}
; CHECK-MACOS:       [[IDX:lxray_fn_idx[0-9]+]]:
; CHECK-MACOS-NEXT:    .quad lxray_sleds_start1-[[IDX]]
; CHECK-MACOS-NEXT:    .quad 3
