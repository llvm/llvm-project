; RUN: llc < %s -mtriple=i686-unknown-linux | FileCheck %s
%struct.s = type {i32, i32, i32, i32, i32, i32, i32, i32,
                  i32, i32, i32, i32, i32, i32, i32, i32,
                  i32, i32, i32, i32, i32, i32, i32, i32 }

define  tailcc i32 @tailcallee(ptr byval(%struct.s) %a) nounwind {
entry:
        %tmp3 = load i32, ptr %a
        ret i32 %tmp3
; CHECK: tailcallee
; CHECK: movl 4(%esp), %eax
}

define  tailcc i32 @tailcaller(ptr byval(%struct.s) %a) nounwind {
entry:
        %tmp4 = tail call tailcc i32 @tailcallee(ptr byval(%struct.s) %a )
        ret i32 %tmp4
; CHECK: tailcaller
; CHECK: jmp tailcallee
}
