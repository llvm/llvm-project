; RUN: llc < %s -mtriple=i686-unknown-linux -tailcallopt | FileCheck %s
%struct.s = type {i32, i32, i32, i32, i32, i32, i32, i32,
                  i32, i32, i32, i32, i32, i32, i32, i32,
                  i32, i32, i32, i32, i32, i32, i32, i32 }

define  fastcc i32 @tailcallee(ptr byval(%struct.s) %a) nounwind {
entry:
        %tmp3 = load i32, ptr %a
        ret i32 %tmp3
; CHECK: tailcallee
; CHECK: movl 4(%esp), %eax
}

define  fastcc i32 @tailcaller(ptr byval(%struct.s) %a) nounwind {
entry:
        %tmp4 = tail call fastcc i32 @tailcallee(ptr byval(%struct.s) %a )
        ret i32 %tmp4
; CHECK: tailcaller
; CHECK: jmp tailcallee
}
