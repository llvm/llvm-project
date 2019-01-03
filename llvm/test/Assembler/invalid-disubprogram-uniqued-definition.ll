; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: <stdin>:[[@LINE+1]]:6: error: missing 'distinct', required for !DISubprogram that is a Definition
!0 = !DISubprogram(isDefinition: true)
