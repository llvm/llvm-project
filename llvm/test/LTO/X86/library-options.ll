;; llvm-lto sets library options in the LLVMContexts, where LoopVectorize reads
;; -vectorize-loops.
; RUN: rm -rf %t && mkdir %t
; RUN: opt -module-summary %s -o %t/a.bc

; RUN: llvm-lto -exported-symbol=f -save-merged-module -o %t/vec.o %t/a.bc
; RUN: llvm-dis %t/vec.o.merged.bc -o - | FileCheck %s --check-prefix=VEC
; RUN: llvm-lto -vectorize-loops=0 -exported-symbol=f -save-merged-module -o %t/novec.o %t/a.bc
; RUN: llvm-dis %t/novec.o.merged.bc -o - | FileCheck %s --check-prefix=NOVEC

;; ThinLTO with the LLVMContexts that ThinLTOCodeGenerator creates.
; RUN: llvm-lto -thinlto-action=run -vectorize-loops=0 -thinlto-save-temps=%t/ %t/a.bc
; RUN: llvm-dis %t/0.4.opt.bc -o - | FileCheck %s --check-prefix=NOVEC

;; Target library options are registered when targets are initialized.
; RUN: llvm-lto -x86-asm-syntax=intel -exported-symbol=f -filetype=asm -o %t/intel.s %t/a.bc
; RUN: FileCheck %s --check-prefix=INTEL --input-file=%t/intel.s

; VEC:       !"llvm.loop.isvectorized"
; NOVEC-NOT: isvectorized
; INTEL:     .intel_syntax noprefix

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr noalias %a, ptr noalias %b) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pa = getelementptr inbounds i32, ptr %a, i64 %i
  %pb = getelementptr inbounds i32, ptr %b, i64 %i
  %v = load i32, ptr %pb
  %w = add i32 %v, 1
  store i32 %w, ptr %pa
  %i.next = add nuw nsw i64 %i, 1
  %c = icmp eq i64 %i.next, 1024
  br i1 %c, label %exit, label %loop

exit:
  ret void
}
