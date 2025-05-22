; RUN: opt  -passes=inline,early-cse -earlycse-debug-hash < %s
; This test used to crash (PR35469).

define void @func1() {
  tail call void @func2()
  ret void
}

define void @func2() {
  tail call void @func3()
  ret void
}

define void @func3() {
  tail call void @func4()
  ret void
}

define void @func4() {
  br i1 undef, label %left, label %right

left:
  tail call void @func5()
  ret void

right:
  ret void
}

define void @func5() {
  tail call void @func6()
  ret void
}

define void @func6() {
  tail call void @func2()
  ret void
}

define void @func7() {
  tail call void @func8(ptr @func3)
  ret void
}

define void @func8(ptr %f) {
  tail call void %f()
  ret void
}
