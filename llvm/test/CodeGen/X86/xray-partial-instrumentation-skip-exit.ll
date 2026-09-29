; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc \
; RUN:     -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-darwin-unknown    < %s | FileCheck %s

define i32 @foo() nounwind noinline uwtable "function-instrument"="xray-always" "xray-skip-exit" {
; CHECK:       .p2align 1
; CHECK-LABEL: Lxray_sled_0:
; CHECK:       .ascii "\353\t"
; CHECK-NEXT:  nopw 512(%rax,%rax)
  ret i32 0
; CHECK-NOT: Lxray_sled_1:
; CHECK:       retq
}

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,foo{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start0:
; CHECK-LINUX:         .quad .Lxray_sled_0
; CHECK-LINUX-LABEL: .Lxray_sleds_end0:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"awo",@progbits,foo{{$}}
; CHECK-LINUX:       [[IDX:.lxray_fn_idx[0-9]+]]:
; CHECK-LINUX:         .quad .Lxray_sleds_start0-[[IDX]]
; CHECK-LINUX-NEXT:    .quad .Lxray_sleds_end0

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start0:
; CHECK-MACOS:         .quad Lxray_sled_0
; CHECK-MACOS-LABEL: Lxray_sleds_end0:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx,regular,live_support{{$}}
; CHECK-MACOS:       [[IDX:lxray_fn_idx[0-9]+]]:
; CHECK-MACOS:         .quad lxray_sleds_start0-[[IDX]]
; CHECK-MACOS-NEXT:    .quad Lxray_sleds_end0


; We test multiple returns in a single function to make sure we're skipping all
; of them with XRay instrumentation.
define i32 @bar(i32 %i) nounwind noinline uwtable "function-instrument"="xray-always" "xray-skip-exit" {
; CHECK:       .p2align 1
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

; CHECK-LINUX-LABEL: .section xray_instr_map,"ao",@progbits,bar{{$}}
; CHECK-LINUX-LABEL: .Lxray_sleds_start1:
; CHECK-LINUX:         .quad .Lxray_sled_1
; CHECK-LINUX-LABEL: .Lxray_sleds_end1:
; CHECK-LINUX-LABEL: .section xray_fn_idx,"awo",@progbits,bar{[$}}
; CHECK-LINUX:       .Lxray_fn_idx0:
; CHECK-LINUX-NEXT:    .quad .Lxray_sleds_start1-.Lxray_fn_idx0
; CHECK-LINUX-NEXT:    .quad 2

; CHECK-MACOS-LABEL: .section __DATA,xray_instr_map,regular,live_support{{$}}
; CHECK-MACOS-LABEL: lxray_sleds_start1:
; CHECK-MACOS:         .quad Lxray_sled_1
; CHECK-MACOS-LABEL: Lxray_sleds_end1:
; CHECK-MACOS-LABEL: .section __DATA,xray_fn_idx,regular,live_support{{$}}
; CHECK-MACOS:       Lxray_fn_idx0:
; CHECK-MACOS-NEXT:    .quad Lxray_sleds_start1-Lxray_fn_idx0
; CHECK-MACOS-NEXT:    .quad 2
