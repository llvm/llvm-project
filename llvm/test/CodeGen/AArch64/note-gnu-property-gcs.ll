; RUN: llc -mtriple=aarch64-linux %s               -o - | \
; RUN:   FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-linux %s -filetype=obj -o - | \
; RUN:   llvm-readelf --notes - | FileCheck %s --check-prefix=OBJ

define dso_local i32 @f() {
entry:
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"guarded-control-stack", i32 1}

; GCS attribute present
; ASM:	    .word	3221225472
; ASM-NEXT:	.word	4
; ASM-NEXT:	.word	4

; OBJ: Properties: aarch64 feature: GCS
