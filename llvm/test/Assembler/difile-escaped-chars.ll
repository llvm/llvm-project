; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Spot-check that metadata string fields are correctly escaped.

!named = !{!0}

; CHECK: !0 = !DIFile(filename: "\00\01\02\80\81\82\FD\FE\FF", directory: "/dir")
!0 = !DIFile(filename: "\00\01\02\80\81\82\FD\FE\FF", directory: "/dir")
