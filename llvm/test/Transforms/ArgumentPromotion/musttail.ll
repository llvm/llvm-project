; RUN: opt < %s -argpromotion -S | FileCheck %s
; PR36543

; Don't promote arguments of musttail callee

%T = type { i32, i32, i32, i32 }

; CHECK-LABEL: define internal i32 @test(%T* %p)
define internal i32 @test(%T* %p) {
  %a.gep = getelementptr %T, %T* %p, i64 0, i32 3
  %b.gep = getelementptr %T, %T* %p, i64 0, i32 2
  %a = load i32, i32* %a.gep
  %b = load i32, i32* %b.gep
  %v = add i32 %a, %b
  ret i32 %v
}

; CHECK-LABEL: define i32 @caller(%T* %p)
define i32 @caller(%T* %p) {
  %v = musttail call i32 @test(%T* %p)
  ret i32 %v
}

; Don't promote arguments of musttail caller

define i32 @foo(%T* %p, i32 %v) {
  ret i32 0
}

; CHECK-LABEL: define internal i32 @test2(%T* %p, i32 %p2)
define internal i32 @test2(%T* %p, i32 %p2) {
  %a.gep = getelementptr %T, %T* %p, i64 0, i32 3
  %b.gep = getelementptr %T, %T* %p, i64 0, i32 2
  %a = load i32, i32* %a.gep
  %b = load i32, i32* %b.gep
  %v = add i32 %a, %b
  %ca = musttail call i32 @foo(%T* undef, i32 %v)
  ret i32 %ca
}

; CHECK-LABEL: define i32 @caller2(%T* %g)
define i32 @caller2(%T* %g) {
  %v = call i32 @test2(%T* %g, i32 0)
  ret i32 %v
}
