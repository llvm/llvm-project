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
  %1 = call i32 @extern_foo()
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

; CHECK: .globl foo[DS]
; CHECK: .globl .foo
; 32BIT: .csect foo[DS],2
; 32BIT-NEXT: .vbyte	4, .foo
; 32BIT-NEXT: .vbyte	4, TOC[TC0]
; 32BIT-NEXT: .vbyte	4, 0
; 64BIT: .csect foo[DS],3
; 64BIT-NEXT: .vbyte	8, .foo
; 64BIT-NEXT: .vbyte	8, TOC[TC0]
; 64BIT-NEXT: .vbyte	8, 0
; CHECK-NEXT: .csect .text[PR],5
; CHECK-LABEL: .foo:

; CHECK: .globl main[DS]
; CHECK: .globl .main
; 32BIT: .csect main[DS],2
; 32BIT-NEXT: .vbyte	4, .main
; 32BIT-NEXT: .vbyte	4, TOC[TC0]
; 32BIT-NEXT: .vbyte	4, 0
; 64BIT: .csect main[DS],3
; 64BIT-NEXT: .vbyte	8, .main
; 64BIT-NEXT: .vbyte	8, TOC[TC0]
; 64BIT-NEXT: .vbyte	8, 0
; CHECK-NEXT: .csect .text[PR],5
; CHECK-LABEL: .main:
; CHECK: bl .foo
; CHECK: bl .extern_foo
; CHECK: bl .static_foo

; CHECK: .lglobl static_foo[DS]
; CHECK: .lglobl .static_foo
; 32BIT: .csect static_foo[DS],2
; 32BIT-NEXT: .vbyte	4, .static_foo
; 32BIT-NEXT: .vbyte	4, TOC[TC0]
; 32BIT-NEXT: .vbyte	4, 0
; 64BIT: .csect static_foo[DS],3
; 64BIT-NEXT: .vbyte	8, .static_foo
; 64BIT-NEXT: .vbyte	8, TOC[TC0]
; 64BIT-NEXT: .vbyte	8, 0
; CHECK-NEXT: .csect .text[PR],5
; CHECK-LABEL: .static_foo:

; CHECK-NOT: .csect extern_foo

; CHECK: .toc
; CHECK-NOT: .tc
