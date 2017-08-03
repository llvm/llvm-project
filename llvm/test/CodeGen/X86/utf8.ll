; RUN: llc < %s -march=x86 | FileCheck %s

; CHECK: iΔ
@"i\CE\94" = common global i32 0, align 4
