; RUN: llc --mtriple=s390x-ibm-zos --show-mc-encoding -emit-gnuas-syntax-on-zos=1 < %s | FileCheck %s

define internal signext i32 @caller() {
entry:
  ret i32 0
}


define hidden signext i32 @caller2() {
entry:
; CHECK-LABEL:   caller2:
; CHECK:         brasl 7,caller       * encoding: [0xc0,0x75,A,A,A,A]
; CHECK-NEXT:    * fixup A - offset: 2, value: caller+2, kind: FK_390_PC32DBL
; CHECK-NEXT:    bcr     0,3          * encoding: [0x07,0x03]
  %call = call signext i32 @caller()
  ret i32 %call
}
