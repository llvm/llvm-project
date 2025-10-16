; Test the setup of the environment area, or associated data area (ADA)
;
; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z10 | FileCheck %s

; CHECK-LABEL: caller:
; CHECK:         stmg 6,8,1872(4)
; CHECK-NEXT:    aghi 4,-192
; CHECK-NEXT:    lgr 8,5
; CHECK-NEXT:    brasl 7,callee_internal@PLT
; CHECK-NEXT:    bcr 0,3
; CHECK-NEXT:    lg 6,8(8)
; CHECK-NEXT:    lg 5,0(8)
; CHECK-NEXT:    lgr 8,3
; CHECK-NEXT:    basr 7,6
; CHECK-NEXT:    bcr 0,0
; CHECK-NEXT:    la 3,0(3,8)
; CHECK-NEXT:    lmg 7,8,2072(4)
; CHECK-NEXT:    aghi 4,192
; CHECK-NEXT:    b 2(7)
define i64 @caller() {
  %r1 = call i64 () @callee_internal()
  %r2 = call i64 () @callee_external()
  %r3 = add i64 %r1, %r2
  ret i64 %r3
}

define internal i64 @callee_internal() {
  ret i64 10
}

declare i64 @callee_external()
