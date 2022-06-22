; RUN: llc < %s -mtriple=x86_64-- -mattr=+sse2,+sse4.1

; No check in a crash test

define void @autogen_178513_5000() {
BB:
  %Shuff22 = shufflevector <2 x i32> undef, <2 x i32> zeroinitializer, <2 x i32> <i32 3, i32 1>
  %B26 = sub <2 x i32> %Shuff22, zeroinitializer
  %S79 = icmp eq <2 x i32> %B26, zeroinitializer
  %B269 = urem <2 x i1> zeroinitializer, %S79
  %Se335 = sext <2 x i1> %B269 to <2 x i8>
  store <2 x i8> %Se335, ptr undef
  ret void
}
