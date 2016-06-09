; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto_alias.ll -o %t2.o

; Ensure that a preempted weak symbol that is linked in as a local
; copy is handled properly. Specifically, the local copy will be promoted,
; and internalization should be able to use the original non-promoted
; name to locate the summary (otherwise internalization will abort because
; it expects to locate summaries for all definitions).
; Note that gold picks the first copy of weakfunc() as the prevailing one,
; so listing %t2.o first is sufficient to ensure that this copy is
; preempted.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=save-temps \
; RUN:     -o %t3.o %t2.o %t.o
; RUN: llvm-nm %t3.o | FileCheck %s
; RUN: llvm-dis %t.o.opt.bc -o - | FileCheck --check-prefix=OPT %s
; RUN: llvm-dis %t2.o.opt.bc -o - | FileCheck --check-prefix=OPT2 %s

; CHECK-NOT: U f
; OPT: define hidden void @weakfunc.llvm.0()
; OPT2: define weak void @weakfunc()

target triple = "x86_64-unknown-linux-gnu"

@weakfuncAlias = alias void (...), bitcast (void ()* @weakfunc to void (...)*)
define weak void @weakfunc() {
entry:
  ret void
}
