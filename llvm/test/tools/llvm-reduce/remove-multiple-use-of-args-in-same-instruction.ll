; Test that llvm-reduce can remove uninteresting function arguments from function definitions as well as their calls.
;
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefix=CHECK-ALL --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-ALL %s < %t

; CHECK-ALL: declare void @use(i32, i32, i32)
declare void @use(i32, i32, i32)

; CHECK-ALL: @interesting(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3
define void @interesting(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3) {
entry:
  ; CHECK-ALL: call void @use(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3)
  call void @use(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3)
  call void @use(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3)
  call void @use(i32 %uninteresting1, i32 %uninteresting2, i32 %uninteresting3)
  ret void
}
