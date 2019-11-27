; RUN: opt -inline -early-cse < %s
; This test used to crash (PR35469).

define void @func1() {
  %t = bitcast void ()* @func2 to void ()*
  tail call void %t()
  ret void
}

define void @func2() {
  %t = bitcast void ()* @func3 to void ()*
  tail call void %t()
  ret void
}

define void @func3() {
  %t = bitcast void ()* @func4 to void ()*
  tail call void %t()
  ret void
}

define void @func4() {
  br i1 undef, label %left, label %right

left:
  %t = bitcast void ()* @func5 to void ()*
  tail call void %t()
  ret void

right:
  ret void
}

define void @func5() {
  %t = bitcast void ()* @func6 to void ()*
  tail call void %t()
  ret void
}

define void @func6() {
  %t = bitcast void ()* @func2 to void ()*
  tail call void %t()
  ret void
}

define void @func7() {
  %t = bitcast void ()* @func3 to void ()*
  tail call void @func8(void()* %t)
  ret void
}

define void @func8(void()* %f) {
  tail call void %f()
  ret void
}
