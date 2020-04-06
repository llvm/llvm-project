; RUN: llc -mtriple=aarch64-linux %s               -o - | \
; RUN:   FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-linux %s -filetype=obj -o - |  \
; RUN:   llvm-readelf --notes | FileCheck %s --check-prefix=OBJ

define dso_local i32 @f() #0 {
entry:
  ret i32 0
}

attributes #0 = { "branch-target-enforcement" "sign-return-address"="non-leaf" }

; Both attribute present
; ASM:	    .word	3221225472
; ASM-NEXT:	.word	4
; ASM-NEXT:	.word	3

; OBJ: Properties: aarch64 feature: BTI, PAC
