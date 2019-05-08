; REQUIRES: x86-registered-target
; Compile with thinlto indices, to enable thinlto.
; RUN: opt -module-summary %s -o %t1.bc

; Test old lto interface with thinlto (currently known to be broken, so
; the FileCheck line is commented out).
; FIXME:  The new LTO implementation has been fixed to not internalize
; (and later dead-code-eliminate) builtin functions.  However the old LTO
; implementation still internalizes them, and when used with ThinLTO they
; get dead-code eliminated before the optimizations run that insert calls
; to them (thus breaking these inserted calls).  This needs to be fixed.
; RUN: llvm-lto -exported-symbol=main -thinlto-action=run %t1.bc
;;; RUN llvm-nm %t1.bc.thinlto.o | FileCheck %s --check-prefix=CHECK-NM

; Test new lto interface with thinlto.
; RUN: llvm-lto2 run %t1.bc -o %t.out -save-temps \
; RUN:   -r %t1.bc,bar,pl \
; RUN:   -r %t1.bc,__stack_chk_fail,pl
; RUN: llvm-nm %t.out.1 | FileCheck %s --check-prefix=CHECK-NM
;

; Re-compile, this time without the thinlto indices.
; RUN: opt %s -o %t4.bc

; Test the new lto interface without thinlto.
; RUN: llvm-lto2 run %t4.bc -o %t5.out -save-temps \
; RUN:   -r %t4.bc,bar,pl \
; RUN:   -r %t4.bc,__stack_chk_fail,pl 
; RUN: llvm-nm %t5.out.0 | FileCheck %s --check-prefix=CHECK-NM

; Test the old lto interface without thinlto.  For now we need to
; use a different nm check, because currently the old lto interface
; internalizes these symbols.  Once the old lto interface gets
; fixed, we should be able to use the same CHECK-NM tests as the
; other FileChecks.
; RUN: llvm-lto -exported-symbol=main %t4.bc -o %t6
; RUN: llvm-nm %t6 | FileCheck %s --check-prefix=CHECK-NM2

; The final binary should not contain any of the dead functions;
; make sure memmove and memcpy are there.
; CHECK-NM-NOT: bar
; CHECK-NM-DAG: T __stack_chk_fail
; CHECK-NM-NOT: bar

; Test case for old lto without thinlto.  Hopefully these can be
; eliminated once the old lto interface is fixed.
; CHECK-NM2-DAG: t __stack_chk_fail

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @bar() {
    ret void
}


define void @__stack_chk_fail() {
    ret void
}
