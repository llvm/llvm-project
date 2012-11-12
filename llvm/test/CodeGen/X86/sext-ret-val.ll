; RUN: llc < %s -march=x86 | grep movzbl | count 1
; rdar://6699246

define signext i8 @t1(i8* %A) nounwind readnone ssp {
entry:
        %0 = icmp ne i8* %A, null
        %1 = zext i1 %0 to i8
        ret i8 %1
}

define i8 @t2(i8* %A) nounwind readnone ssp {
entry:
        %0 = icmp ne i8* %A, null
        %1 = zext i1 %0 to i8
        ret i8 %1
}
