; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/Inputs/thumb.ll -o %t2.bc
; RUN: llvm-link %t1.bc %t2.bc -S 2> %t3.out | FileCheck %s
; RUN: FileCheck --allow-empty --input-file %t3.out --check-prefix STDERR %s

target triple = "armv7-linux-gnueabihf"

declare i32 @foo(i32 %a, i32 %b);

define i32 @main() {
entry:
  %add = call i32 @foo(i32 10, i32 20)
  ret i32 %add
}

; CHECK: define i32 @main() [[MAIN_ATTRS:#[0-9]+]]
; CHECK: define i32 @foo(i32 %a, i32 %b) [[ARM_ATTRS:#[0-9]+]]
; CHECK: define i32 @bar(i32 %a, i32 %b) [[THUMB_ATTRS:#[0-9]+]]

; CHECK: attributes [[MAIN_ATTRS]] = { {{.*}} }
; CHECK: attributes [[ARM_ATTRS]] = { {{.*}} "target-features"="-thumb-mode" }
; CHECK: attributes [[THUMB_ATTRS]] = { {{.*}} "target-features"="+thumb-mode" }

; STDERR-NOT: warning: Linking two modules of different target triples:
