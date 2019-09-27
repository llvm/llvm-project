; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECK,32BIT %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefixes=CHECK,64BIT %s


define i32 @foo() {
entry:
  ret i32 3
}

define i32 @main() {
entry:
  %0 = call i32 @foo()
  %1 = call i32 bitcast (i32 (...)* @extern_foo to i32 ()*)()
  %2 = call i32 @static_foo()
  %3 = add nsw i32 %0, %1
  %4 = add nsw i32 %3, %2
  ret i32 %4
}

declare i32 @extern_foo(...)

define internal i32 @static_foo() {
entry:
  ret i32 3
}

; CHECK: .globl foo
; CHECK: .globl .foo
; CHECK: .csect foo[DS]
; CHECK-NEXT: foo:
; 32BIT: .long .foo
; 32BIT-NEXT: .long TOC[TC0]
; 32BIT-NEXT: .long 0
; 64BIT: .llong .foo
; 64BIT-NEXT: .llong TOC[TC0]
; 64BIT-NEXT: .llong 0
; CHECK-NEXT: .csect .text[PR]
; CHECK-LABEL: .foo:

; CHECK: .globl main
; CHECK: .globl .main
; CHECK: .csect main[DS]
; CHECK-NEXT: main:
; 32BIT: .long .main
; 32BIT-NEXT: .long TOC[TC0]
; 32BIT-NEXT: .long 0
; 64BIT: .llong .main
; 64BIT-NEXT: .llong TOC[TC0]
; 64BIT-NEXT: .llong 0
; CHECK-NEXT: .csect .text[PR]
; CHECK-LABEL: .main:
; CHECK: bl .foo
; CHECK: bl .extern_foo
; CHECK: bl .static_foo

; CHECK: .lglobl .static_foo
; CHECK: .csect static_foo[DS]
; CHECK-NEXT: static_foo:
; 32BIT: .long .static_foo
; 32BIT-NEXT: .long TOC[TC0]
; 32BIT-NEXT: .long 0
; 64BIT: .llong .static_foo
; 64BIT-NEXT: .llong TOC[TC0]
; 64BIT-NEXT: .llong 0
; CHECK-NEXT: .csect .text[PR]
; CHECK-LABEL: .static_foo:

; CHECK-NOT: .csect extern_foo

; CHECK: .toc
; CHECK-NOT: .tc
