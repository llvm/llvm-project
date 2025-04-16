; REQUIRES: x86-registered-target
; RUN: not llc -mtriple=x86_64 < %s -M invalid 2>&1 | FileCheck %s --implicit-check-not=error:

; CHECK: error: invalid InstPrinter option 'invalid'
