; RUN: llc -mtriple=aarch64-linux %s               -o - | \
; RUN:   FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-linux %s -filetype=obj -o - |  \
; RUN:   llvm-readelf --notes | FileCheck %s --check-prefix=OBJ
@x = common dso_local global i32 0, align 4

attributes #0 = { "branch-target-enforcement" }

; Both attributes present in a file with no functions.
; ASM:	    .word	3221225472
; ASM-NEXT:	.word	4
; ASM-NEXT	.word	3

; OBJ: Properties: aarch64 feature: BTI, PAC
