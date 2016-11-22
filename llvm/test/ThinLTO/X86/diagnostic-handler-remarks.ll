; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/diagnostic-handler-remarks.ll -o %t2.bc

; Optimization records are collected regardless of the diagnostic handler
; RUN: llvm-lto -thinlto-action=run \
; RUN:          -lto-pass-remarks-output=%t.yaml \
; RUN:          -exported-symbol _func2 \
; RUN:          -exported-symbol _main %t1.bc %t2.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty
; CHECK-NOT: remark:
; CHECK-NOT: llvm-lto:


; Verify that bar is imported and inlined into foo
; RUN: cat %t.yaml.thin.0.yaml | FileCheck %s -check-prefix=YAML1
; YAML1: --- !Passed
; YAML1: Pass:            inline
; YAML1: Name:            Inlined
; YAML1: Function:        main
; YAML1: Args:
; YAML1:   - Callee:          foo
; YAML1:   - String:          ' inlined into '
; YAML1:   - Caller:          main
; YAML1: ...


; Verify that bar is imported and inlined into foo
; RUN: cat %t.yaml.thin.1.yaml | FileCheck %s -check-prefix=YAML2
; YAML2: --- !Passed
; YAML2: Pass:            inline
; YAML2: Name:            Inlined
; YAML2: Function:        foo
; YAML2: Args:
; YAML2:   - Callee:          bar
; YAML2:   - String:          ' inlined into '
; YAML2:   - Caller:          foo
; YAML2: ...


target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define i32 @bar() {
	ret i32 42
}
declare i32 @foo()
define i32 @main() {
  %i = call i32 @foo()
  ret i32 %i
}

