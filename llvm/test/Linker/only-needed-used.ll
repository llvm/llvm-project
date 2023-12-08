; RUN: llvm-link -S                           %s %p/Inputs/only-needed-used.ll | FileCheck %s
; RUN: llvm-link -S              -internalize %s %p/Inputs/only-needed-used.ll | FileCheck %s
; RUN: llvm-link -S -only-needed              %s %p/Inputs/only-needed-used.ll | FileCheck %s
; RUN: llvm-link -S -only-needed -internalize %s %p/Inputs/only-needed-used.ll | FileCheck %s

; Empty destination module!


; CHECK-DAG:          @llvm.used = appending global [2 x ptr] [ptr @used1, ptr @used2], section "llvm.metadata"
; CHECK-DAG: @used1 = global i8 4
; CHECK-DAG: @used2 = global i32 123
