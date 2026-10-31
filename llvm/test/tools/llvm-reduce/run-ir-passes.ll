; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=ir-passes --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-FINAL --input-file=%t %s
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=ir-passes --ir-passes='function(sroa)' -sroa-skip-mem2reg --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t.skip
; RUN: FileCheck --check-prefixes=SKIP-MEM2REG --input-file=%t.skip %s

; CHECK-INTERESTINGNESS-LABEL: @f1
; CHECK-INTERESTINGNESS: add
; CHECK-INTERESTINGNESS-LABEL: @f2
; CHECK-INTERESTINGNESS: add
; CHECK-INTERESTINGNESS: add

; CHECK-FINAL-LABEL: @f1
; CHECK-FINAL: add i32 %a, 10
; CHECK-FINAL-LABEL: @f2
; CHECK-FINAL: add i32 %a, 5
; CHECK-FINAL: add i32 %b, 5
; CHECK-FINAL-LABEL: @f4
; CHECK-FINAL-NOT: alloca

; SKIP-MEM2REG-LABEL: @f4
; SKIP-MEM2REG: alloca

define i32 @f1(i32 %a) {
  %b = add i32 %a, 5
  %c = add i32 %b, 5
  ret i32 %c
}

define i32 @f2(i32 %a) {
  %b = add i32 %a, 5
  %c = add i32 %b, 5
  ret i32 %c
}

declare void @f3()

define i32 @f4(i32 %a) {
  %p = alloca i32
  store i32 %a, ptr %p
  %v = load i32, ptr %p
  ret i32 %v
}
