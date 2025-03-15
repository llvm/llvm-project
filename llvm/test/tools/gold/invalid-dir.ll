; RUN: rm -rf %t.output
; RUN: mkdir %t.output
; RUN: llvm-as %s -o %t.o
; RUN: not %ld_bfd -plugin %llvmshlibdir/LLVMgold%shlibext  -shared \
; RUN:    %t.o -o %t.output 2>&1 | FileCheck %s -check-prefix=OUTDIR

; OUTDIR: cannot open output file
