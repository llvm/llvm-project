; RUN: llvm-as -o %t.bc %s
; RUN: rm -f %t.opt.bc0 %t.opt.bc1 %t.o0 %t.o1
; RUN: env LD_PRELOAD=%llvmshlibdir/LLVMgold.so %gold -plugin %llvmshlibdir/LLVMgold.so -u foo -u bar -plugin-opt jobs=2 -plugin-opt save-temps -m elf_x86_64 -o %t %t.bc
; RUN: llvm-dis %t.5.precodegen.bc -o - | FileCheck --check-prefix=CHECK-BC0 %s
; RUN: llvm-dis %t.1.5.precodegen.bc -o - | FileCheck --check-prefix=CHECK-BC1 %s
; RUN: llvm-nm %t.o0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-nm %t.o1 | FileCheck --check-prefix=CHECK1 %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK-BC0: define void @foo
; CHECK-BC0: declare void @bar
; CHECK0-NOT: bar
; CHECK0: T foo
; CHECK0-NOT: bar
define void @foo() {
  call void @bar()
  ret void
}

; CHECK-BC1: declare void @foo
; CHECK-BC1: define void @bar
; CHECK1-NOT: foo
; CHECK1: T bar
; CHECK1-NOT: foo
define void @bar() {
  call void @foo()
  ret void
}
