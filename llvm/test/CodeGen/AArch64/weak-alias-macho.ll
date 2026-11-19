; RUN: llc -mtriple=arm64-apple-macosx13.0.0 < %s | FileCheck %s --implicit-check-not=.weak_reference

; Aliases define their own linkage even when the aliasee is private.
@storage = private global [2 x i32] [i32 42, i32 43]

; CHECK: .globl _weak_alias
; CHECK-NEXT: .weak_definition _weak_alias
; CHECK-NEXT: _weak_alias = l_storage
@weak_alias = weak alias i32, ptr @storage

; CHECK: .globl _weak_odr_alias
; CHECK-NEXT: .weak_definition _weak_odr_alias
; CHECK-NEXT: _weak_odr_alias = l_storage
@weak_odr_alias = weak_odr alias i32, ptr @storage

; CHECK: .globl _linkonce_alias
; CHECK-NEXT: .weak_definition _linkonce_alias
; CHECK-NEXT: _linkonce_alias = l_storage
@linkonce_alias = linkonce alias i32, ptr @storage

; CHECK: .globl _linkonce_odr_alias
; CHECK-NEXT: .weak_definition _linkonce_odr_alias
; CHECK-NEXT: _linkonce_odr_alias = l_storage
@linkonce_odr_alias = linkonce_odr alias i32, ptr @storage

; CHECK: .globl _unnamed_alias
; CHECK-NEXT: .weak_def_can_be_hidden _unnamed_alias
; CHECK-NEXT: _unnamed_alias = l_storage
@unnamed_alias = linkonce_odr unnamed_addr alias i32, ptr @storage

; CHECK: .globl _hidden_alias
; CHECK-NEXT: .weak_definition _hidden_alias
; CHECK-NEXT: .private_extern _hidden_alias
; CHECK-NEXT: _hidden_alias = l_storage
@hidden_alias = weak_odr hidden alias i32, ptr @storage

; CHECK: .globl _offset_alias
; CHECK-NEXT: .weak_definition _offset_alias
; CHECK-NEXT: .alt_entry _offset_alias
; CHECK-NEXT: _offset_alias = l_storage+4
@offset_alias = weak_odr alias i32, getelementptr ([2 x i32], ptr @storage, i64 0, i64 1)

; CHECK: .globl _external_alias
; CHECK-NEXT: _external_alias = l_storage
@external_alias = alias i32, ptr @storage

; CHECK-NEXT: _internal_alias = l_storage
@internal_alias = internal alias i32, ptr @storage

; CHECK-NEXT: l_private_alias = l_storage
@private_alias = private alias i32, ptr @storage

define private void @func() {
  ret void
}

; CHECK: .globl _func_alias
; CHECK-NEXT: .weak_definition _func_alias
; CHECK-NEXT: _func_alias = l_func
@func_alias = weak alias void (), ptr @func
