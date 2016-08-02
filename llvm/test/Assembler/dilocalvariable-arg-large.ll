; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1}
!named = !{!0, !1}

!0 = distinct !DISubprogram()

; CHECK: !1 = !DILocalVariable(name: "foo", arg: 65535, scope: !0)
!1 = !DILocalVariable(name: "foo", arg: 65535, scope: !0)
