; RUN: llvm-as < %s | opt -indmemrem | llvm-dis | grep bounce | grep noalias

declare i8* @malloc(i32)

@g = external global i8*

define void @test() {
  %A = bitcast i8* (i32) * @malloc to i8*
  store i8* %A, i8** @g
  ret void
}
