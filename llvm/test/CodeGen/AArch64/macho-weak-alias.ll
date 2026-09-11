; RUN: llc -mtriple=aarch64-apple-macosx13.0.0 %s -o - | FileCheck %s

@foo = internal global i32 0
@external_alias = alias i32, ptr @foo
@internal_alias = internal alias i32, ptr @foo
@weak_alias = weak alias i32, ptr @foo
@hidden_weak_alias = weak hidden alias i32, ptr @foo
@linkonce_alias = linkonce alias i32, ptr @foo
@auto_hide_alias = linkonce_odr unnamed_addr alias i32, ptr @foo

; CHECK:      .globl _external_alias
; CHECK-NEXT: _external_alias = _foo
; CHECK:      _internal_alias = _foo
; CHECK:      .globl _weak_alias
; CHECK-NEXT: .weak_definition _weak_alias
; CHECK-NEXT: _weak_alias = _foo
; CHECK:      .globl _hidden_weak_alias
; CHECK-NEXT: .weak_definition _hidden_weak_alias
; CHECK-NEXT: .private_extern _hidden_weak_alias
; CHECK-NEXT: _hidden_weak_alias = _foo
; CHECK:      .globl _linkonce_alias
; CHECK-NEXT: .weak_definition _linkonce_alias
; CHECK-NEXT: _linkonce_alias = _foo
; CHECK:      .globl _auto_hide_alias
; CHECK-NEXT: .weak_def_can_be_hidden _auto_hide_alias
; CHECK-NEXT: _auto_hide_alias = _foo
