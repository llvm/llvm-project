; RUN: opt < %s -newgvn -S | FileCheck %s

define i32 @main() {
block1:
	%z1 = bitcast i32 0 to i32
	br label %block2
block2:
  %z2 = bitcast i32 0 to i32
  ret i32 %z2
}

; CHECK: define i32 @main() {
; CHECK-NEXT: block1:
; CHECK-NEXT:   br label %block2
; CHECK: block2:
; CHECK-NEXT:   ret i32 0
; CHECK-NEXT: }
