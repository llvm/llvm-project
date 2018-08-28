; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu  | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu  -regalloc=basic | FileCheck %s

declare i8* @llvm.frameaddress(i32) nounwind readnone

define i8* @g2() nounwind readnone {
entry:
; CHECK: g2:
; CHECK: lwz 3, 0(1)
  %0 = tail call i8* @llvm.frameaddress(i32 1)    ; <i8*> [#uses=1]
  ret i8* %0
}

declare i8* @llvm.returnaddress(i32) nounwind readnone

define i8* @g() nounwind readnone {
entry:
; CHECK:  g:
; CHECK:  mflr 0
; CHECK:  stw 0, 4(1)
; CHECK:  lwz 3, 4(3)
; CHECK:  lwz 0, 20(1)
  %0 = tail call i8* @llvm.returnaddress(i32 1)   ; <i8*> [#uses=1]
  ret i8* %0
}
