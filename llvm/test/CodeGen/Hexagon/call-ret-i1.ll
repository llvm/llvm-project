; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Test that the compiler does not assert because the DAG is not correct.
; CHECK: call foo

%returntype = type { i1, i32 }

define i32 @test(i32* %a0, i32* %a1, i32* %a2) #0 {
b3:
  br i1 undef, label %b6, label %b4

b4:                                               ; preds = %b3
  %v5 = call %returntype @foo(i32* nonnull undef, i32* %a2, i32* %a0) #0
  ret i32 1

b6:                                               ; preds = %b3
  unreachable
}

declare %returntype @foo(i32*, i32*, i32*) #0

attributes #0 = { nounwind }
