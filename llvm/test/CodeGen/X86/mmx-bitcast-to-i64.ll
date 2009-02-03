; RUN: llvm-as < %s | llc -march=x86-64 | grep movd | count 4

define i64 @foo(<1 x i64>* %p) {
  %t = load <1 x i64>* %p
  %u = add <1 x i64> %t, %t
  %s = bitcast <1 x i64> %u to i64
  ret i64 %s
}
define i64 @goo(<2 x i32>* %p) {
  %t = load <2 x i32>* %p
  %u = add <2 x i32> %t, %t
  %s = bitcast <2 x i32> %u to i64
  ret i64 %s
}
define i64 @hoo(<4 x i16>* %p) {
  %t = load <4 x i16>* %p
  %u = add <4 x i16> %t, %t
  %s = bitcast <4 x i16> %u to i64
  ret i64 %s
}
define i64 @ioo(<8 x i8>* %p) {
  %t = load <8 x i8>* %p
  %u = add <8 x i8> %t, %t
  %s = bitcast <8 x i8> %u to i64
  ret i64 %s
}
