; RUN: opt -S -lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%S/Inputs/import-icall.yaml < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.used = appending global [1 x i8*] [i8* bitcast (i8* ()* @local_decl to i8*)], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (i8* ()* @local_decl to i8*)], section "llvm.metadata"

@local_decl_alias = alias i8* (), i8* ()* @local_decl

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

define i8* @local_decl() {
  call i8* @local_decl()
  ret i8* bitcast (i8* ()* @local_decl to i8*)
}

declare void @external()
declare extern_weak void @external_weak()

; CHECK: @local_decl_alias = alias i8* (), i8* ()* @local_decl

; CHECK:      define hidden i8 @local_a.cfi() {
; CHECK-NEXT:   call void @external()
; CHECK-NEXT:   call void @external_weak()
; CHECK-NEXT:   ret i8 1
; CHECK-NEXT: }

; internal @local_b is not the same function as "local_b" in the summary.
; CHECK:      define internal i8 @local_b() {
; CHECK-NEXT:   call i8 @local_a()

; CHECK:      define i8* @local_decl()
; CHECK-NEXT:   call i8* @local_decl()
; CHECK-NEXT:   ret i8* bitcast (i8* ()* @local_decl.cfi_jt to i8*)

; CHECK: declare void @external()
; CHECK: declare extern_weak void @external_weak()
; CHECK: declare i8 @local_a()
; CHECK: declare hidden void @external.cfi_jt()
; CHECK: declare hidden void @external_weak.cfi_jt()
