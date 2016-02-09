; RUN: llc < %s -march=x86 | FileCheck %s
; rdar://6699246

define signext i8 @t1(i8* %A) nounwind readnone ssp {
entry:
        %0 = icmp ne i8* %A, null
        %1 = zext i1 %0 to i8
        ret i8 %1

; CHECK-LABEL: t1:
; CHECK: cmpl
; CHECK-NEXT: setne
; CHECK-NEXT: retl
}

define i8 @t2(i8* %A) nounwind readnone ssp {
entry:
        %0 = icmp ne i8* %A, null
        %1 = zext i1 %0 to i8
        ret i8 %1

; CHECK-LABEL: t2:
; CHECK: cmpl
; CHECK-NEXT: setne
; CHECK-NEXT: retl
}
