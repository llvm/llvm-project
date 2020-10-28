; RUN: opt -sink -S < %s | FileCheck %s

; Verify that IR sinking does not move convergent operations to
; blocks that are not control equivalent.

; CHECK: define i32 @foo
; CHECK: entry
; CHECK-NEXT: call i32 @bar
; CHECK-NEXT: br i1 %arg

define i32 @foo(i1 %arg) {
entry:
  %c = call i32 @bar() nounwind readonly convergent
  br i1 %arg, label %then, label %end

then:
  ret i32 %c

end:
  ret i32 0
}

declare i32 @bar() nounwind readonly convergent
