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
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     --plugin-opt=thinlto \
; RUN:     --plugin-opt=save-temps \
; RUN:     -o %t3.o %t2.o %t.o
; RUN: llvm-nm %t3.o | FileCheck %s
; RUN: llvm-dis %t.o.4.opt.bc -o - | FileCheck --check-prefix=OPT %s
; RUN: llvm-dis %t2.o.4.opt.bc -o - | FileCheck --check-prefix=OPT2 %s

; This does not currently pass because the gold plugin now uses the
; combined summary rather than the IRMover to change the module's linkage
; during the ThinLTO backend. The internalization step implemented by IRMover
; for preempted symbols has not yet been implemented for the combined summary.
; XFAIL: *

; CHECK-NOT: U f
; OPT: define hidden void @weakfunc.llvm.0()
; OPT2: define weak void @weakfunc()

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@weakfuncAlias = alias void (...), @weakfunc
define weak void @weakfunc() {
entry:
  ret void
}
