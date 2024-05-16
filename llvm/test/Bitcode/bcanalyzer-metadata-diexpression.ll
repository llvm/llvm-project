; RUN: llvm-as < %s | llvm-bcanalyzer -dump | FileCheck %s

!named = !{!0, !1}

%t = type { i32, i32 }

; CHECK: <EXPRESSION op0=32 op1=1 op2=2/>
!0 = !DIExpression(DIOpReferrer(%t))
; CHECK: <EXPRESSION op0=6/>
!1 = !DIExpression()
