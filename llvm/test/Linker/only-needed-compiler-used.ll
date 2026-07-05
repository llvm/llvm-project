; RUN: llvm-link -S                           %s %p/Inputs/only-needed-compiler-used.ll | FileCheck %s --check-prefix=CHECK --check-prefix=NO-INTERNALIZE
; RUN: llvm-link -S              -internalize %s %p/Inputs/only-needed-compiler-used.ll | FileCheck %s --check-prefix=CHECK --check-prefix=INTERNALIZE
; RUN: llvm-link -S -only-needed              %s %p/Inputs/only-needed-compiler-used.ll | FileCheck %s --check-prefix=CHECK --check-prefix=NO-INTERNALIZE
; RUN: llvm-link -S -only-needed -internalize %s %p/Inputs/only-needed-compiler-used.ll | FileCheck %s --check-prefix=CHECK --check-prefix=INTERNALIZE

; Empty destination module!


; CHECK-DAG:          @llvm.compiler.used = appending global [2 x ptr] [ptr @used1, ptr @used2], section "llvm.metadata"
; NO-INTERNALIZE-DAG: @used1 = global i8 4
; INTERNALIZE-DAG:    @used1 = internal global i8 4
; NO-INTERNALIZE-DAG: @used2 = global i32 123
; INTERNALIZE-DAG:    @used2 = internal global i32 123
