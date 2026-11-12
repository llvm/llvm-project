; RUN: llc < %s -mtriple=i686-unknown-linux-gnu | FileCheck %s
; rdar://6699246

define signext i8 @t1(ptr %A) nounwind readnone ssp {
entry:
        %0 = icmp ne ptr %A, null
        %1 = zext i1 %0 to i8
        ret i8 %1

; CHECK-LABEL: t1:
; CHECK: cmpl
; CHECK-NEXT: setne
; CHECK-NEXT: retl
}

define i8 @t2(ptr %A) nounwind readnone ssp {
entry:
        %0 = icmp ne ptr %A, null
        %1 = zext i1 %0 to i8
        ret i8 %1

; CHECK-LABEL: t2:
; CHECK: cmpl
; CHECK-NEXT: setne
; CHECK-NEXT: retl
}
