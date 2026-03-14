; RUN: llc < %s -mtriple=i686-- | FileCheck %s
define tailcc i32 @bar(i32 %X, ptr%FP) {
     %Y = tail call tailcc i32 %FP(double 0.0, i32 %X)
     ret i32 %Y
; CHECK: jmpl
}
