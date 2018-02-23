; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck -check-prefix VERIFY %s
; RUN: llvm-as -disable-verify < %s | llvm-dis | FileCheck -check-prefix NOVERIFY %s

; NOVERIFY: !named = !{!0}
!named = !{!0}

; NOVERIFY: !0 = !DIExpression(0, 1, 9, 7, 2)
; VERIFY: assembly parsed, but does not verify
!0 = !DIExpression(0, 1, 9, 7, 2)
