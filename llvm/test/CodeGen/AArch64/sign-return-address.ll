; RUN: llc -mtriple=aarch64-none-eabi < %s | FileCheck %s

; CHECK-LABEL: @leaf
; CHECK-NOT: paci{{[a,b]}}sp
; CHECK-NOT: auti{{[a,b]}}sp
define i32 @leaf(i32 %x) {
  ret i32 %x
}

; CHECK-LABEL: @leaf_sign_none
; CHECK-NOT: paci{{[a,b]}}sp
; CHECK-NOT: auti{{[a,b]}}sp
define i32 @leaf_sign_none(i32 %x) "sign-return-address"="none"  {
  ret i32 %x
}

; CHECK-LABEL: @leaf_sign_non_leaf
; CHECK-NOT: paci{{[a,b]}}sp
; CHECK-NOT: auti{{[a,b]}}sp
define i32 @leaf_sign_non_leaf(i32 %x) "sign-return-address"="non-leaf"  {
  ret i32 %x
}

; CHECK-LABEL: @leaf_sign_all
; CHECK: paciasp
; CHECK: autiasp
; CHECK: ret
define i32 @leaf_sign_all(i32 %x) "sign-return-address"="all" {
  ret i32 %x
}

; CHECK: @leaf_clobbers_lr
; CHECK: paciasp
; CHECK: str x30, [sp, #-16]!
; CHECK: ldr  x30, [sp], #16
; CHECK-NEXT: autiasp
; CHECK: ret
define i64 @leaf_clobbers_lr(i64 %x) "sign-return-address"="non-leaf"  {
  call void asm sideeffect "mov x30, $0", "r,~{lr}"(i64 %x) #1
  ret i64 %x
}

declare i32 @foo(i32)

; CHECK: @non_leaf_sign_all
; CHECK: paciasp
; CHECK: autiasp
; CHECK: ret
define i32 @non_leaf_sign_all(i32 %x) "sign-return-address"="all" {
  %call = call i32 @foo(i32 %x)
  ret i32 %call
}

; CHECK: @non_leaf_sign_non_leaf
; CHECK: paciasp
; CHECK: str x30, [sp, #-16]!
; CHECK: ldr  x30, [sp], #16
; CHECK: autiasp
; CHECK: ret
define i32 @non_leaf_sign_non_leaf(i32 %x) "sign-return-address"="non-leaf"  {
  %call = call i32 @foo(i32 %x)
  ret i32 %call
}

; CHECK-LABEL: @leaf_sign_all_v83
; CHECK: paciasp
; CHECK-NOT: ret
; CHECK: retaa
; CHECK-NOT: ret
define i32 @leaf_sign_all_v83(i32 %x) "sign-return-address"="all" "target-features"="+v8.3a" {
  ret i32 %x
}

declare fastcc i64 @bar(i64)

; CHECK-LABEL: @spill_lr_and_tail_call
; CHECK: paciasp
; CHECK: str x30, [sp, #-16]!
; CHECK: ldr  x30, [sp], #16
; CHECK: autiasp
; CHECK: b  bar
define fastcc void @spill_lr_and_tail_call(i64 %x) "sign-return-address"="all" {
  call void asm sideeffect "mov x30, $0", "r,~{lr}"(i64 %x) #1
  tail call fastcc i64 @bar(i64 %x)
  ret void
}

; CHECK-LABEL: @leaf_sign_all_a_key
; CHECK: paciasp
; CHECK: autiasp
define i32 @leaf_sign_all_a_key(i32 %x) "sign-return-address"="all" "sign-return-address-key"="a_key" {
  ret i32 %x
}

; CHECK-LABEL: @leaf_sign_all_b_key
; CHECK: pacibsp
; CHECK: autibsp
define i32 @leaf_sign_all_b_key(i32 %x) "sign-return-address"="all" "sign-return-address-key"="b_key" {
  ret i32 %x
}

; CHECK-LABEL: @leaf_sign_all_v83_b_key
; CHECK: pacibsp
; CHECK-NOT: ret
; CHECK: retab
; CHECK-NOT: ret
define i32 @leaf_sign_all_v83_b_key(i32 %x) "sign-return-address"="all" "target-features"="+v8.3a" "sign-return-address-key"="b_key" {
  ret i32 %x
}
