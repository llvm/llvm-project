; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT: %divergentval
; CHECK-NOT: DIVERGENT: %uniformval
; CHECK: %uniformval
define void @test() {
  %divergentval = call i32 @normalfunc()
  %uniformval = call i32 @nodivergencesourcefunc()
  ret void
}

declare i32 @normalfunc() #0
declare i32 @nodivergencesourcefunc() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind nodivergencesource }
