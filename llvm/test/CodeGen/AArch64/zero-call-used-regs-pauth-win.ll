; RUN: llc -verify-machineinstrs -mtriple aarch64-windows %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple aarch64-windows -filetype=obj %s -o %t.obj

define i32 @leaf(i32 %x) "sign-return-address"="all" "sign-return-address-key"="b_key" "zero-call-used-regs"="all" uwtable {
; CHECK-LABEL: leaf:
  %add = add i32 %x, 1
; CHECK:       .seh_startepilogue
; CHECK-NEXT:  mov x1, #0
; CHECK-NEXT:  .seh_nop
; CHECK:       mov x17, #0
; CHECK-NEXT:  .seh_nop
; CHECK:       movi v31.2d, #0000000000000000
; CHECK-NEXT:  .seh_nop
; CHECK-NEXT:  hint #31
; CHECK-NEXT:  .seh_pac_sign_lr
; CHECK-NEXT:  .seh_endepilogue
; CHECK-NEXT:  ret
  ret i32 %add
}

declare void @ext()

define void @nonleaf() "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" "zero-call-used-regs"="all-gpr" uwtable {
; CHECK-LABEL: nonleaf:
  call void @ext()
; CHECK:       bl ext
; CHECK:       .seh_startepilogue
; CHECK:       ldr x30, [sp], #16
; CHECK:       mov x0, #0
; CHECK-NEXT:  .seh_nop
; CHECK:       mov x17, #0
; CHECK-NEXT:  .seh_nop
; CHECK-NEXT:  hint #31
; CHECK-NEXT:  .seh_pac_sign_lr
; CHECK-NEXT:  .seh_endepilogue
; CHECK-NEXT:  ret
  ret void
}

define i32 @leaf_harden(i32 %x) "sign-return-address"="all" "sign-return-address-key"="b_key" "sign-return-address-harden"="load-return-address" "zero-call-used-regs"="all" uwtable {
; CHECK-LABEL: leaf_harden:
  %add = add i32 %x, 1
; CHECK:       .seh_startepilogue
; CHECK-NEXT:  mov x1, #0
; CHECK-NEXT:  .seh_nop
; CHECK:       mov x17, #0
; CHECK-NEXT:  .seh_nop
; CHECK:       movi v31.2d, #0000000000000000
; CHECK-NEXT:  .seh_nop
; CHECK-NEXT:  hint #31
; CHECK-NEXT:  .seh_pac_sign_lr
; CHECK-NEXT:  mov x8, x30
; CHECK-NEXT:  .seh_nop
; CHECK-NEXT:  hint #7
; CHECK-NEXT:  .seh_nop
; CHECK-NEXT:  ldr w30, [x30]
; CHECK-NEXT:  .seh_nop
; CHECK-NEXT:  .seh_endepilogue
; CHECK-NEXT:  ret x8
  ret i32 %add
}

define void @nonleaf_harden() "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" "sign-return-address-harden"="load-return-address" "zero-call-used-regs"="all-gpr" uwtable {
; CHECK-LABEL: nonleaf_harden:
  call void @ext()
; CHECK:       bl ext
; CHECK:       .seh_startepilogue
; CHECK:       ldr x30, [sp], #16
; CHECK:       mov x0, #0
; CHECK-NEXT:  .seh_nop
; CHECK:       mov x17, #0
; CHECK-NEXT:  .seh_nop
; CHECK-NEXT:  hint #31
; CHECK-NEXT:  .seh_pac_sign_lr
; CHECK-NEXT:  mov x8, x30
; CHECK-NEXT:  .seh_nop
; CHECK-NEXT:  hint #7
; CHECK-NEXT:  .seh_nop
; CHECK-NEXT:  ldr w30, [x30]
; CHECK-NEXT:  .seh_nop
; CHECK-NEXT:  .seh_endepilogue
; CHECK-NEXT:  ret x8
  ret void
}
