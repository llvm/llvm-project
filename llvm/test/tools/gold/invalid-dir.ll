; RUN: rm -rf %t.output
; RUN: mkdir %t.output
; RUN: llvm-as %s -o %t.o
; RUN: not %gold -plugin %llvmshlibdir/LLVMgold%shlibext  -shared \
; RUN:    %t.o -o %t.output 2>&1 | FileCheck %s -check-prefix=OUTDIR

; OUTDIR: fatal error:
