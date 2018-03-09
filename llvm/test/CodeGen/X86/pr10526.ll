; RUN: llc < %s -mtriple=x86_64-- -mattr=+sse2,+sse4.1

; No check in a crash test

define void @autogen_142660_5000() {
BB:
  %Shuff49 = shufflevector <8 x i32> zeroinitializer, <8 x i32> undef, <8 x i32> <i32 2, i32 4, i32 undef, i32 8, i32 10, i32 12, i32 14, i32 0>
  %B85 = sub <8 x i32> %Shuff49, zeroinitializer
  %S242 = icmp eq <8 x i32> zeroinitializer, %B85
  %FC284 = uitofp <8 x i1> %S242 to <8 x float>
  store <8 x float> %FC284, <8 x float>* undef
  ret void
}
