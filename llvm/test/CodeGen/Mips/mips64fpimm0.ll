; RUN: llc  < %s -mtriple=mips64el -mcpu=mips4 -target-abi=n64 | FileCheck %s
; RUN: llc  < %s -mtriple=mips64el -mcpu=mips64 -target-abi=n64 | FileCheck %s

define double @foo1() nounwind readnone {
entry:
; CHECK: dmtc1 $zero
  ret double 0.000000e+00
}
