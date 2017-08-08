; RUN: llc -enable-machine-outliner -mtriple=x86_64-apple-darwin < %s | FileCheck %s

@x = common local_unnamed_addr global i32 0, align 4

define i32 @foo0(i32) local_unnamed_addr #0 {
; CHECK-LABEL: _foo0:
; CHECK: jmp l_OUTLINED_FUNCTION_0
; CHECK-NEXT: .cfi_endproc
  store i32 0, i32* @x, align 4, !tbaa !2
  %2 = tail call i32 @ext(i32 1) #2
  ret i32 undef
}

declare i32 @ext(i32) local_unnamed_addr #1

define i32 @foo1(i32) local_unnamed_addr #0 {
; CHECK-LABEL: _foo1:
; CHECK: jmp l_OUTLINED_FUNCTION_0
; CHECK-NEXT: .cfi_endproc
  store i32 0, i32* @x, align 4, !tbaa !2
  %2 = tail call i32 @ext(i32 1) #2
  ret i32 undef
}

attributes #0 = { noredzone nounwind ssp uwtable "no-frame-pointer-elim"="false" }

!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

; CHECK-LABEL: l_OUTLINED_FUNCTION_0:
; CHECK: movl  $0, (%rax)
; CHECK-NEXT: movl  $1, %edi
; CHECK-NEXT: jmp _ext 