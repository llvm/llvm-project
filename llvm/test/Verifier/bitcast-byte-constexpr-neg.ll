; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s
;
; The same size requirement applies to bitcast constant expressions.

target datalayout = "e-p:64:64:64"

; CHECK: Invalid bitcast
; CHECK: b1 bitcast (ptr @g to b1)
@g = global i8 0
@x = global b1 bitcast (ptr @g to b1)
