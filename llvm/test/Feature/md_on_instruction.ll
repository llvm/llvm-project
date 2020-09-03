; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

define i32 @foo() nounwind ssp {
entry:
  ; CHECK: %retval = alloca i32
  ; CHECK: store i32 42, i32* %retval, align 4, !md !0
  ; CHECK: br label %0, !md !1
  %retval = alloca i32
  store i32 42, i32* %retval, !md !0
  br label %0, !md !1

; <label:0>
  ; CHECK: %1 = load i32, i32* %retval, align 4, !md !2
  ; CHECK: ret i32 %1, !md !3
  %1 = load i32, i32* %retval, !md !2
  ret i32 %1, !md !3
}

; CHECK: !0 = !{}
; CHECK: !1 = distinct !{}
; CHECK: !2 = !{!0}
; CHECK: !3 = !{!4}
; CHECK: !4 = !{!0, !2}
!0 = !{}
!1 = distinct !{}
!2 = !{!0}
!3 = !{!4}
!4 = !{!0, !2}
