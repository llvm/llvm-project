; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s

define i64 @t1(i64 %X) nounwind readnone {
entry:
; CHECK: dclz
  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %X, i1 true)
  ret i64 %tmp1
}

declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone

define i64 @t3(i64 %X) nounwind readnone {
entry:
; CHECK: dclo 
  %neg = xor i64 %X, -1
  %tmp1 = tail call i64 @llvm.ctlz.i64(i64 %neg, i1 true)
  ret i64 %tmp1
}

