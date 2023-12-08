; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s
; Make sure arbitrary metadata strings don't get mutated.  These may be
; (strange) filenames that are part of debug info.

; CHECK: !named = !{!0}
!named = !{!0}

; CHECK: !0 = !{!"llvm.vectorizer.unroll"}
!0 = !{!"llvm.vectorizer.unroll"}
