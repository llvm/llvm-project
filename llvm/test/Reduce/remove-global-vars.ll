; Test that llvm-reduce can remove uninteresting Global Variables as well as
; their direct uses (which in turn are replaced with 'undef').

; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL --implicit-check-not=uninteresting %s

; CHECK-INTERESTINGNESS: @interesting = {{.*}}global i32{{.*}}, align 4
; CHECK-INTERESTINGNESS: @interesting2 = global i32 0, align 4
; CHECK-INTERESTINGNESS: @interesting3 = {{.*}}global i32{{.*}}, align 4

; CHECK-FINAL: @interesting = external global i32, align 4
; CHECK-FINAL: @interesting2 = global i32 0, align 4
; CHECK-FINAL: @interesting3 = external global i32, align 4
@interesting = global i32 0, align 4
@interesting2 = global i32 0, align 4
@interesting3 = external global i32, align 4
@uninteresting = global i32 1, align 4
@uninteresting2 = external global i32, align 4

define i32 @main() {
entry:
  %0 = load i32, i32* @uninteresting, align 4

  ; CHECK-INTERESTINGNESS: store i32 {{.*}}, i32* @interesting, align 4
  ; CHECK-FINAL: store i32 undef, i32* @interesting, align 4
  store i32 %0, i32* @interesting, align 4

  ; CHECK-INTERESTINGNESS: store i32 {{.*}}, i32* @interesting3, align 4
  ; CHECK-FINAL: store i32 undef, i32* @interesting3, align 4
  store i32 %0, i32* @interesting3, align 4

  ; CHECK-ALL: load i32, i32* @interesting, align 4
  %1 = load i32, i32* @interesting, align 4
  store i32 %1, i32* @uninteresting, align 4

  ; CHECK-ALL: load i32, i32* @interesting3, align 4
  %2 = load i32, i32* @interesting3, align 4
  store i32 %2, i32* @uninteresting2, align 4

  ; CHECK-ALL: store i32 5, i32* @interesting, align 4
  store i32 5, i32* @interesting, align 4

  ; CHECK-ALL: store i32 5, i32* @interesting3, align 4
  store i32 5, i32* @interesting3, align 4

  ret i32 0
}
