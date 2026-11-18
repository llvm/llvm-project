;; -mllvm library options reach the LLVMContexts of libLTO's regular LTO and
;; ThinLTO code generators.
; RUN: rm -rf %t && mkdir %t
; RUN: llvm-as %s -o %t/a.o
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -dylib -arch x86_64 -macos_version_min 10.10.0 -o %t/vec.dylib %t/a.o -save-temps -lSystem
; RUN: llvm-dis %t/vec.dylib.lto.opt.bc -o - | FileCheck %s --check-prefix=VEC
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -dylib -arch x86_64 -macos_version_min 10.10.0 -o %t/novec.dylib %t/a.o -save-temps -lSystem -mllvm -vectorize-loops=false
; RUN: llvm-dis %t/novec.dylib.lto.opt.bc -o - | FileCheck %s --check-prefix=NOVEC

; RUN: opt -module-summary %s -o %t/thin.o
; RUN: %ld64 -lto_library %llvmshlibdir/libLTO.dylib -dylib -arch x86_64 -macos_version_min 10.10.0 -o %t/thin.dylib %t/thin.o -save-temps -lSystem -mllvm -vectorize-loops=false
; RUN: llvm-dis %t/thin.dylib.thinlto.bcs/0.4.opt.bc -o - | FileCheck %s --check-prefix=NOVEC

;; SLP may vectorize the unrolled loop, so check LoopVectorize's metadata.
; VEC:       !"llvm.loop.isvectorized"
; NOVEC-NOT: isvectorized

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

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
