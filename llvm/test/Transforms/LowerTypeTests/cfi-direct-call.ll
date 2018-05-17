; RUN: opt -lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%p/Inputs/cfi-direct-call.yaml %s -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

declare void @external_decl()
declare void @external_nodecl()
declare void @internal_def()

define i8 @local_a() {
  call void @external_decl()
  call void @external_nodecl()
  call void @internal_def()
  call void @local_b()
  ret i8 1
}

define void @local_b() {
  ret void
}

; CHECK: define i8 @local_a() {

; Even though a jump table entry is generated, the call goes directly
; to the function
; CHECK-NEXT:   call void @external_decl()

; External call with no CFI decl - no action
; CHECK-NEXT:   call void @external_nodecl()

; Internal function defined outside the module generates a jump table
; entry and is renamed to *.cfi: route direct call to actual function,
; not jump table
; CHECK-NEXT:   call void @internal_def.cfi()

; Local call - no action
; CHECK-NEXT:   call void @local_b

; CHECK-NEXT:   ret i8 1

; CHECK: declare void @internal_def.cfi()
; CHECK: declare hidden void @external_decl.cfi_jt()
