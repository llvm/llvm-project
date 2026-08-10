; RUN: llvm-ml -filetype=s %s /Fo - 2>&1 | FileCheck %s

.data

; CHECK: :[[# @LINE + 1]]:25: warning: MASM-style hex floats ignore explicit sign
negative_hexfloat REAL4 -3fa66666r
; CHECK-LABEL: negative_hexfloat:
; CHECK-NEXT: .long 1067869798

.code

END
