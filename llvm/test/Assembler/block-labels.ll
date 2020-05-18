; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s --match-full-lines
; RUN: verify-uselistorder %s

define i32 @test1(i32 %X) {
; Implicit entry label. Not printed on output.
  %1 = alloca i32
  br label %2
; Implicit label ids still allowed.
  br label %3
3: ; Explicit numeric label
  br label %"2"
"2": ; string label, quoted number
  br label %-3
-3: ; numeric-looking, but actually string, label
  br label %-N-
-N-:
  br label %$N
$N:
  %4 = add i32 1, 1
  ret i32 %4
}

; CHECK-LABEL: define i32 @test1(i32 %X) {
; CHECK-NEXT:   %1 = alloca i32, align 4
; CHECK-NEXT:   br label %2
; CHECK:      2:       ; preds = %0
; CHECK-NEXT:   br label %3
; CHECK:      3:       ; preds = %2
; CHECK-NEXT:   br label %"2"
; CHECK:      "2":     ; preds = %3
; CHECK-NEXT:   br label %-3
; CHECK:      -3:      ; preds = %"2"
; CHECK-NEXT:   br label %-N-
; CHECK:      -N-:     ; preds = %-3
; CHECK-NEXT:   br label %"$N"
; CHECK:      "$N":    ; preds = %-N-
; CHECK-NEXT:   %4 = add i32 1, 1
; CHECK-NEXT:   ret i32 %4
; CHECK-NEXT: }

define void @test2(i32 %0, i32 %1) {
; entry label id still not printed on output
2:
  ret void
}

; CHECK-LABEL: define void @test2(i32 %0, i32 %1) {
; CHECK-NEXT:    ret void
