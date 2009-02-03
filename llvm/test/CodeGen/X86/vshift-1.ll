; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -disable-mmx -o %t -f
; RUN: grep psllq  %t | count 2
; RUN: grep pslld %t | count 2
; RUN: grep psllw %t | count 2

; test vector shifts converted to proper SSE2 vector shifts when the shift
; amounts are the same.

define void @shift1a(<2 x i64> %val, <2 x i64>* %dst) nounwind {
entry:
  %shl = shl <2 x i64> %val, < i64 32, i64 32 >
  store <2 x i64> %shl, <2 x i64>* %dst
  ret void
}

define void @shift1b(<2 x i64> %val, <2 x i64>* %dst, i64 %amt) nounwind {
entry:
  %0 = insertelement <2 x i64> undef, i64 %amt, i32 0
  %1 = insertelement <2 x i64> %0, i64 %amt, i32 1
  %shl = shl <2 x i64> %val, %1
  store <2 x i64> %shl, <2 x i64>* %dst
  ret void
}


define void @shift2a(<4 x i32> %val, <4 x i32>* %dst) nounwind {
entry:
  %shl = shl <4 x i32> %val, < i32 5, i32 5, i32 5, i32 5 >
  store <4 x i32> %shl, <4 x i32>* %dst
  ret void
}

define void @shift2b(<4 x i32> %val, <4 x i32>* %dst, i32 %amt) nounwind {
entry:
  %0 = insertelement <4 x i32> undef, i32 %amt, i32 0
  %1 = insertelement <4 x i32> %0, i32 %amt, i32 1
  %2 = insertelement <4 x i32> %1, i32 %amt, i32 2
  %3 = insertelement <4 x i32> %2, i32 %amt, i32 3
  %shl = shl <4 x i32> %val, %3
  store <4 x i32> %shl, <4 x i32>* %dst
  ret void
}

define void @shift3a(<8 x i16> %val, <8 x i16>* %dst) nounwind {
entry:
  %shl = shl <8 x i16> %val, < i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5 >
  store <8 x i16> %shl, <8 x i16>* %dst
  ret void
}

define void @shift3b(<8 x i16> %val, <8 x i16>* %dst, i16 %amt) nounwind {
entry:
  %0 = insertelement <8 x i16> undef, i16 %amt, i32 0
  %1 = insertelement <8 x i16> %0, i16 %amt, i32 1
  %2 = insertelement <8 x i16> %0, i16 %amt, i32 2
  %3 = insertelement <8 x i16> %0, i16 %amt, i32 3
  %4 = insertelement <8 x i16> %0, i16 %amt, i32 4
  %5 = insertelement <8 x i16> %0, i16 %amt, i32 5
  %6 = insertelement <8 x i16> %0, i16 %amt, i32 6
  %7 = insertelement <8 x i16> %0, i16 %amt, i32 7
  %shl = shl <8 x i16> %val, %7
  store <8 x i16> %shl, <8 x i16>* %dst
  ret void
}

