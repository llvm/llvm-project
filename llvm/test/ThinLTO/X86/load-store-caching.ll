; Test that instruction operands from loads are not cached when
; processing stores. Reference from @foo to @obj should not be
; readonly or writeonly

; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { ptr }
%struct.Derived = type { i32 }
%struct.Base = type { i32 }

@obj = dso_local local_unnamed_addr global %struct.S zeroinitializer, align 8

define dso_local ptr @foo() local_unnamed_addr {
entry:
  %0 = load ptr, ptr @obj, align 8
  store ptr null, ptr @obj, align 8
  ret ptr %0
}

; CHECK:       ^0 = module:
; CHECK-NEXT:  ^1 = gv: (name: "obj", summaries: (variable: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), varFlags: (readonly: 1, writeonly: 1, constant: 0)))) ; guid =
; CHECK-NEXT:  ^2 = gv: (name: "foo", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 3, refs: (^1)))) ; guid =
