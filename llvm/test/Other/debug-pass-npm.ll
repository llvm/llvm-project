; RUN: opt -enable-new-pm=0 -O1 %s -debug-pass=Structure
; RUN: not opt -enable-new-pm=1 -O1 %s -debug-pass=Structure 2>&1 | FileCheck %s

; CHECK: does not work
