; RUN: opt -S -passes=spirv-prepare-globals -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

; The pass folds non-interposable GlobalAlias instances away from their uses.
; - For private (discardable) aliases, the alias declaration is also erased.
; - For externally-linked aliases, the declaration is preserved but uses are
;   replaced.
; Interposable aliases (or aliases of interposable globals) must be left
; untouched, since their target may change at link time.

; CHECK-NOT: @alias_private
; CHECK-DAG: @aliasee = global i32 42
; CHECK-DAG: @alias_external = alias i32, ptr @aliasee
; CHECK-DAG: @weak_alias = weak alias i32, ptr @weak_aliasee
; CHECK-DAG: @alias_of_interposable = alias i32, ptr @interposable_aliasee
; CHECK-DAG: @alias_of_noipa = alias void (), ptr @noipa_aliasee

@aliasee = global i32 42
@alias_private = private alias i32, ptr @aliasee
@alias_external = alias i32, ptr @aliasee

@weak_aliasee = global i32 7
@weak_alias = weak alias i32, ptr @weak_aliasee

@interposable_aliasee = weak global i32 99
@alias_of_interposable = alias i32, ptr @interposable_aliasee

@alias_of_noipa = alias void (), ptr @noipa_aliasee

; Private alias is discardable: use is RAUW'd to the aliasee.
define ptr @use_private_alias() {
; CHECK-LABEL: define ptr @use_private_alias()
; CHECK-NEXT:    ret ptr @aliasee
  ret ptr @alias_private
}

; External alias: use is RAUW'd to the aliasee, but @alias_external itself
; remains in the module (checked above with CHECK-DAG).
define ptr @use_external_alias() {
; CHECK-LABEL: define ptr @use_external_alias()
; CHECK-NEXT:    ret ptr @aliasee
  ret ptr @alias_external
}

; Weak (interposable) alias is left alone.
define ptr @use_weak_alias() {
; CHECK-LABEL: define ptr @use_weak_alias()
; CHECK-NEXT:    ret ptr @weak_alias
  ret ptr @weak_alias
}

; Alias of an interposable global is left alone.
define ptr @use_alias_of_interposable() {
; CHECK-LABEL: define ptr @use_alias_of_interposable()
; CHECK-NEXT:    ret ptr @alias_of_interposable
  ret ptr @alias_of_interposable
}

; noipa doesn't make a function interposable at link time, so an alias of a
; noipa function is still folded.
define void @noipa_aliasee() noipa {
  ret void
}

define ptr @use_alias_of_noipa() {
; CHECK-LABEL: define ptr @use_alias_of_noipa()
; CHECK-NEXT:    ret ptr @noipa_aliasee
  ret ptr @alias_of_noipa
}
