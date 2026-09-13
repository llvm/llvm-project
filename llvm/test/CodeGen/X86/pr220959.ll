; RUN: not llc %s -mtriple=x86_64-unknown-linux-gnu -filetype=null 2>&1 | FileCheck %s
; RUN: not llc %s -mtriple=x86_64-unknown-linux-gnu -O0 -filetype=null 2>&1 | FileCheck %s

; A landingpad whose result type is not (exception pointer, selector) used to
; trip an assertion in SelectionDAGBuilder::visitLandingPad. Expect a clean
; diagnostic instead.

; CHECK: error: {{.*}}in function main{{.*}}landingpad result type must consist of exactly two values, the exception pointer and the selector
define i32 @main() personality ptr @__gxx_personality_v0 {
  invoke void @g()
          to label %cont unwind label %cleanup
cont:
  ret i32 0
cleanup:
  %lp = landingpad {}
          cleanup
  ret i32 1
}

; CHECK: error: {{.*}}in function scalar{{.*}}landingpad result type must consist of exactly two values, the exception pointer and the selector
define i32 @scalar() personality ptr @__gxx_personality_v0 {
  invoke void @g()
          to label %cont unwind label %cleanup
cont:
  ret i32 0
cleanup:
  %lp = landingpad i32
          cleanup
  ret i32 %lp
}

; CHECK: error: {{.*}}in function three_values{{.*}}landingpad result type must consist of exactly two values, the exception pointer and the selector
define i32 @three_values() personality ptr @__gxx_personality_v0 {
  invoke void @g()
          to label %cont unwind label %cleanup
cont:
  ret i32 0
cleanup:
  %lp = landingpad { ptr, i32, i32 }
          cleanup
  %sel = extractvalue { ptr, i32, i32 } %lp, 2
  ret i32 %sel
}

declare void @g()
declare i32 @__gxx_personality_v0(...)
