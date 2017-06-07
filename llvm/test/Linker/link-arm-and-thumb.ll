; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/Inputs/thumb.ll -o %t2.bc
; RUN: llvm-link %t1.bc %t2.bc -S 2> %t3.out | llc | FileCheck %s
; RUN: FileCheck --allow-empty --input-file %t3.out --check-prefix STDERR %s

target triple = "armv7-linux-gnueabihf"

declare i32 @foo(i32 %a, i32 %b);

define i32 @main() {
entry:
  %add = call i32 @foo(i32 10, i32 20)
  ret i32 %add
}

; CHECK: .code  32 @ @main
; CHECK-NEXT: main

; CHECK: .code  32 @ @foo
; CHECK-NEXT: foo

; CHECK: .code  16 @ @bar
; CHECK-NEXT: .thumb_func
; CHECK-NEXT: bar

; STDERR-NOT: warning: Linking two modules of different target triples:
