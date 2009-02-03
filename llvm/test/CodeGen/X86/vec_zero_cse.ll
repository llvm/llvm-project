; RUN: llvm-as < %s | llc -relocation-model=static -march=x86 -mcpu=yonah | grep pxor | count 1
; RUN: llvm-as < %s | llc -relocation-model=static -march=x86 -mcpu=yonah | grep xorps | count 1
; RUN: llvm-as < %s | llc -relocation-model=static -march=x86 -mcpu=yonah | grep pcmpeqd | count 2

@M1 = external global <1 x i64>
@M2 = external global <2 x i32>

@S1 = external global <2 x i64>
@S2 = external global <4 x i32>

define void @test() {
  store <1 x i64> zeroinitializer, <1 x i64>* @M1
  store <2 x i32> zeroinitializer, <2 x i32>* @M2
  ret void
}

define void @test2() {
  store <1 x i64> < i64 -1 >, <1 x i64>* @M1
  store <2 x i32> < i32 -1, i32 -1 >, <2 x i32>* @M2
  ret void
}

define void @test3() {
  store <2 x i64> zeroinitializer, <2 x i64>* @S1
  store <4 x i32> zeroinitializer, <4 x i32>* @S2
  ret void
}

define void @test4() {
  store <2 x i64> < i64 -1, i64 -1>, <2 x i64>* @S1
  store <4 x i32> < i32 -1, i32 -1, i32 -1, i32 -1 >, <4 x i32>* @S2
  ret void
}


