; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !1}
!named = !{!0, !1, !2, !3}

!0 = distinct !{}

; CHECK: !1 = !DIModule(scope: !0, name: "Module")
!1 = !DIModule(scope: !0, name: "Module")

; CHECK: !2 = !DIModule(scope: !0, name: "Module", configMacros: "-DNDEBUG", includePath: "/usr/include", sysroot: "/")
!2 = !DIModule(scope: !0, name: "Module", configMacros: "-DNDEBUG", includePath: "/usr/include", sysroot: "/")

!3 = !DIModule(scope: !0, name: "Module", configMacros: "")
