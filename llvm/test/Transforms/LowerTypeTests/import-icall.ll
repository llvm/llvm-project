; RUN: opt -S -lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%S/Inputs/import-icall.yaml < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8 @local_a() {
  call void @external()
  call void @external_weak()
  ret i8 1
}

define internal i8 @local_b() {
  %x = call i8 @local_a()
  ret i8 %x
}

define i8 @use_b() {
  %x = call i8 @local_b()
  ret i8 %x
}

define void @local_decl() {
  call void @local_decl()
  ret void
}

declare void @external()
declare extern_weak void @external_weak()

; CHECK:      define hidden i8 @local_a.cfi() {
; CHECK-NEXT:   call void @external.cfi_jt()
; CHECK-NEXT:   call void select (i1 icmp ne (void ()* @external_weak, void ()* null), void ()* @external_weak.cfi_jt, void ()* null)()
; CHECK-NEXT:   ret i8 1
; CHECK-NEXT: }

; internal @local_b is not the same function as "local_b" in the summary.
; CHECK:      define internal i8 @local_b() {
; CHECK-NEXT:   call i8 @local_a()

; CHECK:      define void @local_decl()
; CHECK-NEXT:   call void @local_decl()

; CHECK: declare void @external()
; CHECK: declare extern_weak void @external_weak()
; CHECK: declare i8 @local_a()
; CHECK: declare hidden void @external.cfi_jt()
; CHECK: declare hidden void @external_weak.cfi_jt()
