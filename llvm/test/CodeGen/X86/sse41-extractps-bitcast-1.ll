; RUN: llvm-as < %s | llc -march=x86 -mattr=sse41 | not grep extractps

; The non-store form of extractps puts its result into a GPR.
; This makes it suitable for an extract from a <4 x float> that
; is bitcasted to i32, but unsuitable for much of anything else.

define float @bar(<4 x float> %v) {
  %s = extractelement <4 x float> %v, i32 3
  %t = add float %s, 1.0
  ret float %t
}
define float @baz(<4 x float> %v) {
  %s = extractelement <4 x float> %v, i32 3
  ret float %s
}
define i32 @qux(<4 x i32> %v) {
  %i = extractelement <4 x i32> %v, i32 3
  ret i32 %i
}
