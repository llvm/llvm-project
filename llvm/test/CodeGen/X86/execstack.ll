;; Check that .note.GNU-stack sections are emitted on Linux, but not on Solaris.

; RUN: llc < %s -mtriple=i686-linux | FileCheck %s -check-prefix=CHECK-GNUSTACK
; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s -check-prefix=CHECK-GNUSTACK
; RUN: llc < %s -mtriple=i386-solaris | FileCheck %s -check-prefix=CHECK-NOGNUSTACK
; RUN: llc < %s -mtriple=amd64-solaris | FileCheck %s -check-prefix=CHECK-NOGNUSTACK

; CHECK-GNUSTACK: .section	".note.GNU-stack","",@progbits
; CHECK-NOGNUSTACK-NOT: .section	".note.GNU-stack","",@progbits
