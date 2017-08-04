; Check that the hotness attribute is included in the optimization record file
; with -lto-pass-remarks-with-hotness.

; RUN: llvm-as < %s >%t.bc
; RUN: rm -f %t.yaml
; RUN: llvm-lto -lto-pass-remarks-output=%t.yaml \
; RUN:          -lto-pass-remarks-with-hotness \
; RUN:          -exported-symbol _main -o %t.o %t.bc
; RUN: cat %t.yaml | FileCheck -check-prefix=YAML %s

; YAML:      --- !Passed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            Inlined
; YAML-NEXT: Function:        main
; YAML-NEXT: Hotness:         300
; YAML-NEXT: Args:
; YAML-NEXT:   - Callee:          foo
; YAML-NEXT:   - String:          ' inlined into '
; YAML-NEXT:   - Caller:          main
; YAML-NEXT: ...

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
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
