; Testcase to check that functions from a Thumb module can be inlined in an
; ARM function.
;
; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/Inputs/thumb.ll -o %t2.bc
; RUN: llvm-lto -exported-symbol main \
; RUN:          -exported-symbol bar \
; RUN:          -filetype=asm \
; RUN:          -o - \
; RUN:          %t1.bc %t2.bc 2> %t3.out| FileCheck %s
; RUN: FileCheck --allow-empty --input-file %t3.out --check-prefix STDERR %s

target triple = "armv7-linux-gnueabihf"

; CHECK: .code  32
; CHECK-NEXT: main
; CHECK-NEXT: .fnstart
; CHECK-NEXT: mov r0, #30

; CHECK: .code  16
; CHECK-NEXT: .thumb_func
; CHECK-NEXT: bar

declare i32 @foo(i32 %a, i32 %b);

define i32 @main() {
entry:
  %add = call i32 @foo(i32 10, i32 20)
  ret i32 %add
}

; STDERR-NOT: warning: Linking two modules of different target triples:
