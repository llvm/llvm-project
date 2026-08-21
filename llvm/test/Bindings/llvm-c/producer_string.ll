; RUN: llvm-as < %s | llvm-c-test --module-get-producer-string | FileCheck %s
; CHECK: LLVM{{[0-9]+.*}}
