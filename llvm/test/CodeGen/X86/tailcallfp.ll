; RUN: llc < %s -mtriple=i686-- -tailcallopt | FileCheck %s
define fastcc i32 @bar(i32 %X, ptr%FP) {
     %Y = tail call fastcc i32 %FP(double 0.0, i32 %X)
     ret i32 %Y
; CHECK: jmpl
}
