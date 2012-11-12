; RUN: llc < %s -mtriple=x86_64-linux-pc -mcpu=core-avx2 | FileCheck %s

; FIXME: vpmovsxwd should be generated instead of vpmovzxwd followed by
; SLL/SRA.

define <8 x i32> @foo(<8 x i1> %bar) nounwind readnone {
entry:
  %s = sext <8 x i1> %bar to <8 x i32>
  ret <8 x i32> %s
; CHECK: foo
; CHECK: vpmovzxwd
; CHECK: vpslld
; CHECK: vpsrad
; CHECK: ret
}
