; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/diagnostic-handler-remarks.ll -o %t2.bc

; Optimization records are collected regardless of the diagnostic handler
; RUN: rm -f %t.yaml.thin.0.yaml %t.yaml.thin.1.yaml
; RUN: llvm-lto -thinlto-action=run \
; RUN:          -lto-pass-remarks-output=%t.yaml \
; RUN:          -lto-pass-remarks-filter=inline \
; RUN:          -lto-pass-remarks-format=yaml \
; RUN:          -exported-symbol _func2 \
; RUN:          -exported-symbol _main %t1.bc %t2.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty
; CHECK-NOT: remark:
; CHECK-NOT: llvm-lto:


; Verify that bar is imported 'and' inlined into 'foo'
; RUN: cat %t.yaml.thin.0.yaml | FileCheck %s -check-prefixes=YAML1,YAML1-NO-ANNOTATE

; Verify that bar is imported 'and' inlined into 'foo'
; RUN: cat %t.yaml.thin.1.yaml | FileCheck %s -check-prefixes=YAML2,YAML2-NO-ANNOTATE

; Try again with -annotate-inline-phase
; RUN: rm -f %t.yaml.thin.0.yaml %t.yaml.thin.1.yaml
; RUN: llvm-lto -thinlto-action=run \
; RUN:          -annotate-inline-phase \
; RUN:          -lto-pass-remarks-output=%t.yaml \
; RUN:          -lto-pass-remarks-filter=inline \
; RUN:          -lto-pass-remarks-format=yaml \
; RUN:          -exported-symbol _func2 \
; RUN:          -exported-symbol _main %t1.bc %t2.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty
; CHECK-NOT: remark:
; CHECK-NOT: llvm-lto:

; Verify thati inline happens the same way and pass name are annotated with LTO phase information.
; RUN: cat %t.yaml.thin.0.yaml | FileCheck %s -check-prefixes=YAML1,YAML1-ANNOTATE

; Verify that bar is imported 'and' inlined into 'foo'
; RUN: cat %t.yaml.thin.1.yaml | FileCheck %s -check-prefixes=YAML2,YAML2-ANNOTATE

; YAML1:      --- !Passed
; YAML1:      --- !Passed
; YAML1-NO-ANNOTATE: Pass: inline
; YAML1-ANNOTATE-NEXT: Pass:   postlink-cgscc-inline
; YAML1-NEXT: Name:            Inlined
; YAML1-NEXT: Function:        main
; YAML1-NEXT: Args:
; YAML1-NEXT:   - String:          ''''
; YAML1-NEXT:   - Callee:          foo
; YAML1-NEXT:   - String:          ''' inlined into '
; YAML1-NEXT:   - Caller:          main
; YAML1-NEXT:   - String:          ''''
; YAML1-NEXT:   - String:          ' with '
; YAML1-NEXT:   - String:          '(cost='
; YAML1-NEXT:   - Cost:            '-30'
; YAML1-NEXT:   - String:          ', threshold='
; YAML1-NEXT:   - Threshold:       '375'
; YAML1-NEXT:   - String:          ')'
; YAML1-NEXT: ...

; YAML2: --- !Passed
; YAML2-NO-ANNOTATE: Pass: inline
; YAML2-ANNOTATE-NEXT: Pass:   postlink-cgscc-inline
; YAML2-NEXT: Name:            Inlined
; YAML2-NEXT: Function:        foo
; YAML2-NEXT: Args:
; YAML2-NEXT:   - String:          ''''
; YAML2-NEXT:   - Callee:          bar
; YAML2-NEXT:   - String:          ''' inlined into '
; YAML2-NEXT:   - Caller:          foo
; YAML2-NEXT:   - String:          ''''
; YAML2-NEXT:   - String:          ' with '
; YAML2-NEXT:   - String:          '(cost='
; YAML2-NEXT:   - Cost:            '-30'
; YAML2-NEXT:   - String:          ', threshold='
; YAML2-NEXT:   - Threshold:       '375'
; YAML2-NEXT:   - String:          ')'
; YAML2-NEXT: ...

; The file extension depends on the format of the remarks
; RUN: rm -f %t.bitstream.thin.0.bitstream %t.bitstream.thin.1.bitstream
; RUN: llvm-lto -thinlto-action=run \
; RUN:          -lto-pass-remarks-output=%t.bitstream \
; RUN:          -lto-pass-remarks-filter=inline \
; RUN:          -lto-pass-remarks-format=bitstream \
; RUN:          -exported-symbol _func2 \
; RUN:          -exported-symbol _main %t1.bc %t2.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty
; RUN: llvm-bcanalyzer %t.bitstream.thin.0.bitstream
; RUN: llvm-bcanalyzer %t.bitstream.thin.1.bitstream
; CHECK-NOT: remark:
; CHECK-NOT: llvm-lto:

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define i32 @bar() {
	ret i32 42
}
declare i32 @foo()
define i32 @main() {
  %i = call i32 @foo()
  ret i32 %i
}

