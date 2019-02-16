; REQUIRES: shell
; XFAIL: windows-gnu

; RUN: umask 000
; RUN: rm -f %t.000
; RUN: llvm-as %s -o %t.000
; RUN: ls -l %t.000 | FileCheck --check-prefix=CHECK000 %s
; CHECK000: rw-rw-rw

; RUN: umask 002
; RUN: rm -f %t.002
; RUN: llvm-as %s -o %t.002
; RUN: ls -l %t.002 | FileCheck --check-prefix=CHECK002 %s
; CHECK002: rw-rw-r-
