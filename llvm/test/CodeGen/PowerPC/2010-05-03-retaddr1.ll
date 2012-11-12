; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin -mcpu=g5 | FileCheck %s
; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin -mcpu=g5 -regalloc=basic | FileCheck %s

declare i8* @llvm.frameaddress(i32) nounwind readnone

define i8* @g2() nounwind readnone {
entry:
; CHECK: _g2:
; CHECK: lwz r3, 0(r1)
  %0 = tail call i8* @llvm.frameaddress(i32 1)    ; <i8*> [#uses=1]
  ret i8* %0
}

declare i8* @llvm.returnaddress(i32) nounwind readnone

define i8* @g() nounwind readnone {
entry:
; CHECK: _g:
; CHECK:  mflr r0
; CHECK:  stw r0, 8(r1)
; CHECK:  lwz r3, 0(r1)
; CHECK:  lwz r3, 8(r3)
  %0 = tail call i8* @llvm.returnaddress(i32 1)   ; <i8*> [#uses=1]
  ret i8* %0
}
