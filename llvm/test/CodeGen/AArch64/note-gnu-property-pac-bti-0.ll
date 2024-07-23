; RUN: llc -mtriple=aarch64-linux %s               -o - | \
; RUN:   FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-linux %s -filetype=obj -o - |  \
; RUN:   llvm-readelf --notes - | FileCheck %s --check-prefix=OBJ
@x = common dso_local global i32 0, align 4

attributes #0 = { "branch-target-enforcement" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 0}

; Both attributes present in a file with no functions.
; ASM:	    .word	3221225472
; ASM-NEXT:	.word	4
; ASM-NEXT:	.word	3

; OBJ: Properties: aarch64 feature: BTI, PAC
