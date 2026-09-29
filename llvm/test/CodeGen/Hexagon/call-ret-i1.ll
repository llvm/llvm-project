; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

; Test that the compiler does not assert because the DAG is not correct.
; CHECK: call foo

%returntype = type { i1, i32 }

define i32 @test(ptr %a0, ptr %a1, ptr %a2) #0 {
b3:
  br i1 undef, label %b6, label %b4

b4:                                               ; preds = %b3
  %v5 = call %returntype @foo(ptr nonnull undef, ptr %a2, ptr %a0) #0
  ret i32 1

b6:                                               ; preds = %b3
  unreachable
}

declare %returntype @foo(ptr, ptr, ptr) #0

attributes #0 = { nounwind }
