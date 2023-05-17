; RUN: llc --mtriple=loongarch64 --mattr=help 2>&1 | FileCheck %s
; RUN: llc --mtriple=loongarch32 --mattr=help 2>&1 | FileCheck %s

; CHECK: Available features for this target:
; CHECK: 32bit - LA32 Basic Integer and Privilege Instruction Set.
