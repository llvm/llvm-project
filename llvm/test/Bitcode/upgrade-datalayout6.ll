; Test that an explicit i128:64 entry is added to SystemZ data layouts, which
; predate the change of the default i128 alignment to 128 bits.
;
; RUN: llvm-as %s -o - | llvm-dis - | FileCheck %s

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64-S64"
target triple = "s390x-unknown-linux-gnu"

; CHECK: target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64-S64-i128:64"
