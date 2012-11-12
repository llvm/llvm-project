;RUN: opt -instsimplify %s -disable-output
declare void @helper(<2 x i8*>)
define void @test(<2 x i8*> %a) {
  %A = getelementptr <2 x i8*> %a, <2 x i32> <i32 0, i32 0>
  call void @helper(<2 x i8*> %A)
  ret void
}

