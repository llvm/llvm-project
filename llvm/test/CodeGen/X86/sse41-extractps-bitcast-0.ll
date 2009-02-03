; RUN: llvm-as < %s | llc -march=x86 -mattr=sse41 | grep extractps | count 2

define i32 @foo(<4 x float> %v) {
  %s = extractelement <4 x float> %v, i32 3
  %i = bitcast float %s to i32
  ret i32 %i
}
define i32 @boo(<4 x float> %v) {
  %t = bitcast <4 x float> %v to <4 x i32>
  %s = extractelement <4 x i32> %t, i32 3
  ret i32 %s
}
