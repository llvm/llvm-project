; RUN: llc < %s -mtriple=i686-unknown-linux -tailcallopt | FileCheck %s
%struct.s = type {i32, i32, i32, i32, i32, i32, i32, i32,
                  i32, i32, i32, i32, i32, i32, i32, i32,
                  i32, i32, i32, i32, i32, i32, i32, i32 }

define  fastcc i32 @tailcallee(%struct.s* byval(%struct.s) %a) nounwind {
entry:
        %tmp2 = getelementptr %struct.s, %struct.s* %a, i32 0, i32 0
        %tmp3 = load i32, i32* %tmp2
        ret i32 %tmp3
; CHECK: tailcallee
; CHECK: movl 4(%esp), %eax
}

define  fastcc i32 @tailcaller(%struct.s* byval(%struct.s) %a) nounwind {
entry:
        %tmp4 = tail call fastcc i32 @tailcallee(%struct.s* byval(%struct.s) %a )
        ret i32 %tmp4
; CHECK: tailcaller
; CHECK: jmp tailcallee
}
