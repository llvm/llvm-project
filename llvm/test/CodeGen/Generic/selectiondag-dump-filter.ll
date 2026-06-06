; RUN: llc -debug-only=isel-dump -filter-print-funcs=foo < %s 2>&1 | FileCheck %s --check-prefix=FOO
; RUN: llc -debug-only=isel-dump -filter-print-funcs=bar < %s 2>&1 | FileCheck %s --check-prefix=BAR
; RUN: llc -debug-only=isel-dump -filter-print-funcs=foo,zap < %s 2>&1 | FileCheck %s --check-prefixes=FOO,ZAP
; Make sure the original -debug-only=isel still works.
; RUN: llc -debug-only=isel < %s 2>&1 | FileCheck %s  --check-prefixes=FOO,BAR,ZAP
; REQUIRES: asserts

; FOO:     === foo
; BAR-NOT: === foo
; ZAP-NOT: === foo
; FOO: # Machine code for function foo
define i32 @foo(i32 %a, i32 %b) {
  %r = add i32 %a, %b
  ret i32 %r
}

; BAR:     === bar
; FOO-NOT: === bar
; ZAP-NOT: === bar
; BAR: # Machine code for function bar
define i32 @bar(i32 %a, i32 %b) {
  %r = mul i32 %a, %b
  ret i32 %r
}

; ZAP:     === zap
; FOO-NOT: === zap
; BAR-NOT: === zap
; ZAP: # Machine code for function zap
define i32 @zap(i32 %a, i32 %b) {
  %r = sub i32 %a, %b
  ret i32 %r
}
