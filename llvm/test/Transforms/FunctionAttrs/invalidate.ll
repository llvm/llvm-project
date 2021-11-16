; RUN: opt -passes='function(require<no-op-function>),cgscc(function-attrs)' -disable-output < %s -debug-pass-manager 2>&1 | FileCheck %s

; CHECK: Running pass: PostOrderFunctionAttrsPass on (f)
; CHECK: Invalidating analysis: NoOpFunctionAnalysis on f
; CHECK-NOT: Invalidating analysis: NoOpFunctionAnalysis on h
; CHECK: Invalidating analysis: NoOpFunctionAnalysis on g
; CHECK-NOT: Invalidating analysis: NoOpFunctionAnalysis on h
; CHECK: Running pass: PostOrderFunctionAttrsPass on (g)
; CHECK: Running pass: PostOrderFunctionAttrsPass on (h)

declare i32 @e(i32(i32)*)

define i32 @f(i32 %a) {
  ret i32 %a
}

define i32 @g(i32 %b) {
  %c = call i32 @f(i32 %b)
  ret i32 %c
}

define i32 @h(i32 %b) {
  %c = call i32 @e(i32(i32)* @f)
  ret i32 %c
}
