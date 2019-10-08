; RUN: llc < %s -mtriple=i686-- | FileCheck %s
define tailcc i32 @bar(i32 %X, i32(double, i32) *%FP) {
     %Y = tail call tailcc i32 %FP(double 0.0, i32 %X)
     ret i32 %Y
; CHECK: jmpl
}
