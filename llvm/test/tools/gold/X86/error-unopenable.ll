; RUN: llvm-as -o %t %s
; RUN: not %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=obj-path=%t/nonexistent-dir/foo.o \
; RUN:    %t -o %t2 2>&1 | FileCheck %s

; CHECK: Could not open file {{.*}}nonexistent-dir

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
