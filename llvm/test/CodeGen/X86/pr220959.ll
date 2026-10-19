; RUN: not llc %s -mtriple=x86_64-unknown-linux-gnu -filetype=null 2>&1 | FileCheck %s
; RUN: not llc %s -mtriple=x86_64-unknown-linux-gnu -O0 -filetype=null 2>&1 | FileCheck %s

; Landingpad result types other than (exception pointer, integer selector) used
; to trip an assertion in SelectionDAGBuilder::visitLandingPad.

; CHECK: error: {{.*}}in function main{{.*}}landingpad result type must be a struct of an exception pointer and an integer selector
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

; CHECK: error: {{.*}}in function scalar{{.*}}landingpad result type must be a struct of an exception pointer and an integer selector
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

; CHECK: error: {{.*}}in function three_elements{{.*}}landingpad result type must be a struct of an exception pointer and an integer selector
define i32 @three_elements() personality ptr @__gxx_personality_v0 {
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

; Right element count, wrong element type.
; CHECK: error: {{.*}}in function float_exception{{.*}}landingpad result type must be a struct of an exception pointer and an integer selector
define float @float_exception() personality ptr @__gxx_personality_v0 {
  invoke void @g()
          to label %cont unwind label %cleanup
cont:
  ret float 0.0
cleanup:
  %lp = landingpad { float, i32 }
          cleanup
  %exn = extractvalue { float, i32 } %lp, 0
  ret float %exn
}

declare void @g()
declare i32 @__gxx_personality_v0(...)
