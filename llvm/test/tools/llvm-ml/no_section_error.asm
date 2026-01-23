; RUN: not llvm-ml -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s

; CHECK: :[[# @LINE + 1]]:6: error: expected section directive before assembly directive in 'BYTE' directive
BYTE 2, 3, 4
