; RUN: llc  < %s -march=mipsel | FileCheck %s 
; RUN: llc  < %s -march=mips   | FileCheck %s

define void @f(i64 %l, i64* nocapture %p) nounwind {
entry:
; CHECK: lui  
; CHECK: ori
; CHECK: addu  
  %add = add i64 %l, 1311768467294899695
  store i64 %add, i64* %p, align 4 
  ret void
}

