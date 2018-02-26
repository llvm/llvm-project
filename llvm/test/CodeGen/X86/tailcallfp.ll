; RUN: llc < %s -mtriple=i686-- -tailcallopt | FileCheck %s
define fastcc i32 @bar(i32 %X, i32(double, i32) *%FP) {
     %Y = tail call fastcc i32 %FP(double 0.0, i32 %X)
     ret i32 %Y
; CHECK: jmpl
}
