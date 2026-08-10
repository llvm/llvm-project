; RUN: llc < %s -mtriple=i686-- | FileCheck %s

; CHECK: iΔ
@"i\CE\94" = common global i32 0, align 4
