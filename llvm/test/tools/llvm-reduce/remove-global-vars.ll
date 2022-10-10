; Test that llvm-reduce can remove uninteresting Global Variables as well as
; their direct uses (which in turn are replaced with '0').

; RUN: llvm-reduce --delta-passes=global-variables,global-initializers --test FileCheck --test-arg --check-prefixes=CHECK-ALL,CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck --check-prefixes=CHECK-ALL,CHECK-FINAL --implicit-check-not=uninteresting %s

$interesting5 = comdat any

; CHECK-INTERESTINGNESS: @interesting = {{.*}}global i32{{.*}}, align 4
; CHECK-INTERESTINGNESS: @interesting2 = global i32 0, align 4
; CHECK-INTERESTINGNESS: @interesting3 = {{.*}}global i32{{.*}}, align 4
; CHECK-INTERESTINGNESS: @interesting4 = {{.*}}constant i32{{.*}}, align 4
; CHECK-INTERESTINGNESS: @interesting5 = {{.*}}global i32{{.*}}, align 4

; CHECK-FINAL: @interesting = external global i32, align 4
; CHECK-FINAL: @interesting2 = global i32 0, align 4
; CHECK-FINAL: @interesting3 = external global i32, align 4
; CHECK-FINAL: @interesting4 = external dso_local constant i32, align 4
; CHECK-FINAL: @interesting5 = external global i32, align 4
@interesting = global i32 0, align 4
@interesting2 = global i32 0, align 4
@interesting3 = external global i32, align 4
@interesting4 = private constant i32 2, align 4
@interesting5 = global i32 2, align 4, comdat

@uninteresting = global i32 1, align 4
@uninteresting2 = external global i32, align 4

define i32 @main() {
entry:
  %0 = load i32, ptr @uninteresting, align 4

  ; CHECK-INTERESTINGNESS: store i32 {{.*}}, ptr @interesting, align 4
  ; CHECK-FINAL: store i32 0, ptr @interesting, align 4
  store i32 %0, ptr @interesting, align 4

  ; CHECK-INTERESTINGNESS: store i32 {{.*}}, ptr @interesting3, align 4
  ; CHECK-FINAL: store i32 0, ptr @interesting3, align 4
  store i32 %0, ptr @interesting3, align 4

  ; CHECK-ALL: load i32, ptr @interesting, align 4
  %1 = load i32, ptr @interesting, align 4
  store i32 %1, ptr @uninteresting, align 4

  ; CHECK-ALL: load i32, ptr @interesting3, align 4
  %2 = load i32, ptr @interesting3, align 4
  store i32 %2, ptr @uninteresting2, align 4

  ; CHECK-ALL: store i32 5, ptr @interesting, align 4
  store i32 5, ptr @interesting, align 4

  ; CHECK-ALL: store i32 5, ptr @interesting3, align 4
  store i32 5, ptr @interesting3, align 4

  ret i32 0
}
