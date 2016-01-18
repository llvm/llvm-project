; RUN: llvm-as < %s >%t.bc
; PR21108: Diagnostic handlers get pass remarks, even if they're not enabled.

; Confirm that there are -pass-remarks.
; RUN: llvm-lto -pass-remarks=inline \
; RUN:          -exported-symbol _main -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=REMARKS
; RUN: llvm-nm %t.o | FileCheck %s -check-prefix NM

; RUN: llvm-lto -pass-remarks=inline -use-diagnostic-handler \
; RUN:         -exported-symbol _main -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty -check-prefix=REMARKS
; RUN: llvm-nm %t.o | FileCheck %s -check-prefix NM

; Confirm that -pass-remarks are not printed by default.
; RUN: llvm-lto \
; RUN:         -exported-symbol _main -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty
; RUN: llvm-nm %t.o | FileCheck %s -check-prefix NM

; RUN: llvm-lto -use-diagnostic-handler \
; RUN:         -exported-symbol _main -o %t.o %t.bc 2>&1 | \
; RUN:     FileCheck %s -allow-empty
; RUN: llvm-nm %t.o | FileCheck %s -check-prefix NM

; REMARKS: remark:
; CHECK-NOT: remark:
; NM-NOT: foo
; NM: main

target triple = "x86_64-apple-darwin"

define i32 @foo() {
  ret i32 7
}

define i32 @main() {
  %i = call i32 @foo()
  ret i32 %i
}
