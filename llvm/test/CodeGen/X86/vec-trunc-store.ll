; RUN: llc < %s -march=x86-64

define void @foo(<8 x i32>* %p) nounwind {
  %t = load <8 x i32>* %p
  %cti69 = trunc <8 x i32> %t to <8 x i16>     ; <<8 x i16>> [#uses=1]
  store <8 x i16> %cti69, <8 x i16>* undef
  ret void
}

define void @bar(<4 x i32>* %p) nounwind {
  %t = load <4 x i32>* %p
  %cti44 = trunc <4 x i32> %t to <4 x i16>     ; <<4 x i16>> [#uses=1]
  store <4 x i16> %cti44, <4 x i16>* undef
  ret void
}
