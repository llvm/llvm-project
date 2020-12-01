; Check that the hotness attribute is included in the optimization record file
; with -lto-pass-remarks-with-hotness.

; RUN: llvm-as < %s >%t.bc
; RUN: rm -f %t.yaml %t.t300.yaml %t.t301.yaml
; RUN: llvm-lto -lto-pass-remarks-output=%t.yaml \
; RUN:          -lto-pass-remarks-with-hotness \
; RUN:          -exported-symbol _main -o %t.o %t.bc
; RUN: cat %t.yaml | FileCheck -check-prefix=YAML %s

; RUN: llvm-lto -lto-pass-remarks-output=%t.t300.yaml \
; RUN:          -lto-pass-remarks-with-hotness \
; RUN:          -lto-pass-remarks-hotness-threshold=300 \
; RUN:          -exported-symbol _main -o %t.o %t.bc
; RUN: FileCheck -check-prefix=YAML %s < %t.t300.yaml

; RUN: llvm-lto -lto-pass-remarks-output=%t.t301.yaml \
; RUN:          -lto-pass-remarks-with-hotness \
; RUN:          -lto-pass-remarks-hotness-threshold=301 \
; RUN:          -exported-symbol _main -o %t.o %t.bc
; RUN: not FileCheck -check-prefix=YAML %s < %t.t301.yaml

; YAML:      --- !Passed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            Inlined
; YAML-NEXT: Function:        main
; YAML-NEXT: Hotness:         300
; YAML-NEXT: Args:
; YAML-NEXT:   - Callee:          foo
; YAML-NEXT:   - String:          ' inlined into '
; YAML-NEXT:   - Caller:          main
; YAML-NEXT:   - String:          ' with '
; YAML-NEXT:   - String:          '(cost='
; YAML-NEXT:   - Cost:            '-15000'
; YAML-NEXT:   - String:          ', threshold='
; YAML-NEXT:   - Threshold:       '337'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT: ...

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

declare i32 @bar()

define i32 @foo() {
  %a = call i32 @bar()
  ret i32 %a
}

define i32 @main() !prof !0 {
  %i = call i32 @foo()
  ret i32 %i
}

!0 = !{!"function_entry_count", i64 300}
