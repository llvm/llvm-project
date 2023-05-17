; RUN: opt -S -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%S/Inputs/import-icall.yaml %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.used = appending global [1 x ptr] [ptr @local_decl], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr @local_decl], section "llvm.metadata"

@local_decl_alias = alias ptr (), ptr @local_decl

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

define ptr @local_decl() {
  call ptr @local_decl()
  ret ptr @local_decl
}

declare void @external()
declare extern_weak void @external_weak()

; CHECK: @local_decl_alias = alias ptr (), ptr @local_decl

; CHECK:      define hidden i8 @local_a.cfi() {
; CHECK-NEXT:   call void @external()
; CHECK-NEXT:   call void @external_weak()
; CHECK-NEXT:   ret i8 1
; CHECK-NEXT: }

; internal @local_b is not the same function as "local_b" in the summary.
; CHECK:      define internal i8 @local_b() {
; CHECK-NEXT:   call i8 @local_a()

; CHECK:      define ptr @local_decl()
; CHECK-NEXT:   call ptr @local_decl()
; CHECK-NEXT:   ret ptr @local_decl.cfi_jt

; CHECK: declare void @external()
; CHECK: declare extern_weak void @external_weak()
; CHECK: declare i8 @local_a()
; CHECK: declare hidden void @external.cfi_jt()
; CHECK: declare hidden void @external_weak.cfi_jt()
