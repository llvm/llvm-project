; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !1, !3, !4}
!named = !{!0, !1, !2, !3, !4, !5}

!0 = distinct !{}

; CHECK: !1 = !DIModule(scope: !0, name: "Module")
!1 = !DIModule(scope: !0, name: "Module")

; CHECK: !2 = !DIModule(scope: !0, name: "Module", configMacros: "-DNDEBUG", includePath: "/usr/include")
!2 = !DIModule(scope: !0, name: "Module", configMacros: "-DNDEBUG", includePath: "/usr/include")

!3 = !DIModule(scope: !0, name: "Module", configMacros: "")

; CHECK: !3 = !DIModule(scope: !0, name: "Module", configMacros: "-DNDEBUG", includePath: "/usr/include", apinotes: "/tmp/m.apinotes", file: !0, line: 1)
!4 = !DIModule(scope: !0, name: "Module", configMacros: "-DNDEBUG", includePath: "/usr/include", apinotes: "/tmp/m.apinotes", file: !0, line: 1)

; CHECK: !4 = !DIModule(scope: !0, name: "Module", isDecl: true)
!5 = !DIModule(scope: !0, name: "Module", isDecl: true)
