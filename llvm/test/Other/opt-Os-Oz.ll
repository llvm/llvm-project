; RUN: not opt -Os < %s -S 2>&1 | FileCheck %s --check-prefix=Os
; RUN: not opt -Oz < %s -S 2>&1 | FileCheck %s --check-prefix=Oz

; Os: The optimization level "Os" is no longer supported. Use O2 in conjunction with the optsize attribute instead.
; Oz: The optimization level "Oz" is no longer supported. Use O2 in conjunction with the minsize attribute instead.
