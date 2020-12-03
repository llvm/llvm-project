; RUN: llc < %s -mtriple=i686-unknown-linux | FileCheck %s
%struct.s = type {i32, i32, i32, i32, i32, i32, i32, i32,
                  i32, i32, i32, i32, i32, i32, i32, i32,
                  i32, i32, i32, i32, i32, i32, i32, i32 }

define  tailcc i32 @tailcallee(%struct.s* byval(%struct.s) %a) nounwind {
entry:
        %tmp2 = getelementptr %struct.s, %struct.s* %a, i32 0, i32 0
        %tmp3 = load i32, i32* %tmp2
        ret i32 %tmp3
; CHECK: tailcallee
; CHECK: movl 4(%esp), %eax
}

define  tailcc i32 @tailcaller(%struct.s* byval(%struct.s) %a) nounwind {
entry:
        %tmp4 = tail call tailcc i32 @tailcallee(%struct.s* byval(%struct.s) %a )
        ret i32 %tmp4
; CHECK: tailcaller
; CHECK: jmp tailcallee
}
