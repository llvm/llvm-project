; RUN: opt -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%p/Inputs/cfi-direct-call.yaml %s -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

declare void @external_decl()
declare void @external_nodecl()
;declare void @internal_default_def()
declare hidden void @internal_hidden_def()

define i8 @local_a() {
  call void @external_decl()
  call void @external_nodecl()
  call void @internal_default_def()
  call void @internal_hidden_def()
  call void @dsolocal_default_def()
  call void @local_b()
  ret i8 1
}

define dso_local void @dsolocal_default_def() {
  ret void
}

define void @internal_default_def() {
  ret void
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

; Internal function with default visibility gets routed through the jump table
; as it may be overriden at run time.
; CHECK-NEXT:   call void @internal_default_def()

; Internal function with hidden visibility defined outside the module
; generates a jump table entry and is renamed to *.cfi: route direct call
; to the actual function, not jump table
; CHECK-NEXT:   call void @internal_hidden_def.cfi()

; dso_local function with defailt visibility can be short-circuited
; CHECK-NEXT:   call void @dsolocal_default_def.cfi()

; Local call - no action
; CHECK-NEXT:   call void @local_b

; CHECK-NEXT:   ret i8 1

; CHECK: declare hidden void @internal_hidden_def.cfi()
; CHECK: declare hidden void @external_decl.cfi_jt()
