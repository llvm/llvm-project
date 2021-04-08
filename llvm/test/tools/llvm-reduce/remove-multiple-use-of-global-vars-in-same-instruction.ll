; Test that llvm-reduce can remove uninteresting function arguments from function definitions as well as their calls.
;
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefix=CHECK-ALL --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-ALL %s < %t

; CHECK-ALL: @uninteresting1 = global
; CHECK-ALL: @uninteresting2 = global
; CHECK-ALL: @uninteresting3 = global
@uninteresting1 = global i32 0, align 4
@uninteresting2 = global i32 0, align 4
@uninteresting3 = global i32 0, align 4

declare void @use(i32*, i32*, i32*)

; CHECK-LABEL: @interesting()
define void @interesting() {
entry:
  ; CHECK-ALL: call void @use(i32* @uninteresting1, i32* @uninteresting2, i32* @uninteresting3)
  call void @use(i32* @uninteresting1, i32* @uninteresting2, i32* @uninteresting3)
  call void @use(i32* @uninteresting1, i32* @uninteresting2, i32* @uninteresting3)
  call void @use(i32* @uninteresting1, i32* @uninteresting2, i32* @uninteresting3)
  ret void
}
