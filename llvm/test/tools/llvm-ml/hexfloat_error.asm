; RUN: not llvm-ml -filetype=s %s /WX /Fo /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

.data

; CHECK: :[[# @LINE + 1]]:25: error: MASM-style hex floats ignore explicit sign
negative_hexfloat REAL4 -3fa66666r

.code

END
